//! Exact ScanJson lowering for the closed JSON-log producers.
use std::sync::Arc;

use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{DataFusionError, Result};
use datafusion::datasource::provider_as_source;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder, TableType};
use datafusion::physical_plan::ExecutionPlan;
use datafusion_datasource::file_groups::FileGroup;
use datafusion_datasource::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
use datafusion_datasource::source::DataSourceExec;
use datafusion_datasource::{PartitionedFile, TableSchema};
use datafusion_datasource_json::source::JsonSource;
use delta_kernel::engine::arrow_conversion::TryIntoArrow;
use delta_kernel::plans::ir::nodes::ScanJson;
use delta_kernel::tasks::LogIdentityManifest;
use object_store::ObjectStore;

/// Lowering is reachable only with a task's immutable discovery manifest.
pub(crate) fn lower_scan_json(
    scan: &ScanJson,
    manifest: &LogIdentityManifest,
    store: Arc<dyn ObjectStore>,
) -> Result<LogicalPlan> {
    if scan.file_constant_columns != ["version"]
        || scan.schema.fields().any(|f| f.is_metadata_column())
    {
        return Err(DataFusionError::NotImplemented(
            "JSON task metadata specification".into(),
        ));
    }
    let declared: Schema = scan.schema.as_ref().try_into_arrow()?;
    let version = declared.field_with_name("version")?;
    if version.data_type() != &datafusion::arrow::datatypes::DataType::Int64 {
        return Err(DataFusionError::Plan(
            "JSON task version must be Int64".into(),
        ));
    }
    let file_schema = Arc::new(Schema::new(
        declared
            .fields()
            .iter()
            .filter(|f| f.name() != "version")
            .cloned()
            .collect::<Vec<_>>(),
    ));
    let table_schema = TableSchema::builder(file_schema)
        .with_table_partition_cols(vec![Arc::new(version.clone())])
        .build();
    let source = Arc::new(JsonSource::new(table_schema).with_object_store(store));
    if scan.files.len() != manifest.files().len() || scan.files.is_empty() {
        return Err(DataFusionError::Plan(
            "JSON commit cover length mismatch".into(),
        ));
    }
    let mut files = Vec::with_capacity(scan.files.len());
    for (position, file) in scan.files.iter().enumerate() {
        let mismatch =
            || DataFusionError::Plan("JSON file descriptor/version/cover mismatch".into());
        let [delta_kernel::expressions::Scalar::Long(expected)] = file.file_constants.as_slice()
        else {
            return Err(mismatch());
        };
        let index = usize::try_from(*expected).map_err(|_| mismatch())?;
        // Both sealed producers call commit_cover_version_tagged_scan_files:
        // the contiguous ascending manifest is selected in descending version order.
        if Some(index) != scan.files.len().checked_sub(position + 1) {
            return Err(mismatch());
        }
        let admitted = manifest.files().get(index).ok_or_else(mismatch)?;
        #[cfg(test)]
        crate::log_store::tests::note_path_comparison();
        if file.meta.location.as_str() != admitted.path
            || file.meta.size != admitted.size
            || file.meta.last_modified != admitted.modification_time
        {
            return Err(mismatch());
        }
        // Preserve the already decoded ObjectStore path. Feeding its display
        // through Path::from would encode '%' again for escaped table roots.
        let mut partitioned = PartitionedFile::new_from_meta(object_store::ObjectMeta {
            location: object_store::path::Path::from_url_path(file.meta.location.path())?,
            size: file.meta.size,
            last_modified: chrono::DateTime::from_timestamp_millis(admitted.modification_time)
                .ok_or_else(|| DataFusionError::Plan("JSON file timestamp range".into()))?,
            e_tag: None,
            version: None,
        });
        partitioned.partition_values =
            vec![datafusion::common::ScalarValue::Int64(Some(*expected))];
        files.push(partitioned);
    }
    // One ordinary file group avoids parallel decoder ownership. No byte-range subdivision.
    // Provenance already admitted these canonical Url values. Clone one and
    // replace only its path; do not run host/IDNA parsing again in lowering.
    let mut origin = scan
        .files
        .first()
        .ok_or_else(|| DataFusionError::Plan("JSON task has no admitted files".into()))?
        .meta
        .location
        .clone();
    origin.set_path("/");
    let config = FileScanConfigBuilder::new(ObjectStoreUrl::try_from_url(origin)?, source)
        .with_file_group(FileGroup::new(files))
        .with_batch_size(Some(1))
        .build();
    let schema = Arc::clone(config.file_source.table_schema().table_schema());
    let provider = Arc::new(JsonPlanTable { schema, config });
    LogicalPlanBuilder::scan("kernel_json", provider_as_source(provider), None)?
        .project(
            declared
                .fields()
                .iter()
                .map(|f| Expr::Column(datafusion::common::Column::new_unqualified(f.name())))
                .collect::<Vec<_>>(),
        )?
        .build()
}

#[derive(Debug)]
struct JsonPlanTable {
    schema: SchemaRef,
    config: FileScanConfig,
}
impl TableProvider for JsonPlanTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
    fn table_type(&self) -> TableType {
        TableType::Base
    }
    fn scan<'life0, 'life1, 'life2, 'life3, 'async_trait>(
        &'life0 self,
        _state: &'life1 dyn Session,
        projection: Option<&'life2 Vec<usize>>,
        _filters: &'life3 [Expr],
        _limit: Option<usize>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Arc<dyn ExecutionPlan>>> + Send + 'async_trait>,
    >
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        'life2: 'async_trait,
        'life3: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(self.scan_unboxed(projection))
    }
}

impl JsonPlanTable {
    async fn scan_unboxed(
        &self,
        projection: Option<&Vec<usize>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let config = FileScanConfigBuilder::from(self.config.clone())
            .with_projection_indices(projection.cloned())?
            .build();
        Ok(DataSourceExec::from_data_source(config))
    }
}

pub(crate) fn provider_future_layout() -> std::alloc::Layout {
    fn layout<A, F: std::future::Future>(_factory: impl FnOnce(A) -> F) -> std::alloc::Layout {
        std::alloc::Layout::new::<F>()
    }
    layout(
        |(provider, projection): (&JsonPlanTable, Option<&Vec<usize>>)| {
            provider.scan_unboxed(projection)
        },
    )
}

/// Concrete provider header, excluding its separately admitted file/schema owners.
pub(crate) fn provider_layout() -> std::alloc::Layout {
    std::alloc::Layout::new::<JsonPlanTable>()
}

#[cfg(test)]
mod many_file_tests {
    use delta_kernel::expressions::Scalar;
    use delta_kernel::plans::ir::nodes::Operator;
    use delta_kernel::tasks::{AdmittedPlan, FileDescriptor, ObjectIdentity, Resource, TaskLimits};

    use super::*;
    use crate::closed_plan_facts::ClosedPlanFacts;

    fn admitted(count: usize) -> AdmittedPlan {
        crate::log_store::tests::admitted_plan(
            (0..count)
                .map(|version| FileDescriptor {
                    path: format!("memory:///table/_delta_log/{version:020}.json"),
                    size: 1,
                    modification_time: 0,
                    identity: ObjectIdentity::new([version as u8; 32]),
                })
                .collect(),
        )
    }

    fn scan(plan: &AdmittedPlan) -> &ScanJson {
        plan.plan()
            .nodes
            .iter()
            .find_map(|node| match &node.op {
                Operator::ScanJson(scan) => Some(scan),
                _ => None,
            })
            .unwrap()
    }

    #[test]
    fn long_common_prefix_has_linear_prepaid_work_and_one_comparison_per_file() {
        let root =
            url::Url::parse(&format!("memory:///table/{}", "shared-prefix/".repeat(12))).unwrap();
        let make = |count| {
            crate::log_store::tests::admitted_plan_at(
                &root,
                (0..count)
                    .map(|version| FileDescriptor {
                        path: format!("{root}_delta_log/{version:020}.json"),
                        size: 1,
                        modification_time: 0,
                        identity: ObjectIdentity::new([version as u8; 32]),
                    })
                    .collect(),
            )
        };
        let one = make(1);
        let many = make(64);
        let limits = TaskLimits::qualification();
        let one_facts = ClosedPlanFacts::inspect(&one, limits).unwrap();
        let many_facts = ClosedPlanFacts::inspect(&many, limits).unwrap();
        assert_eq!(many_facts.file_url_bytes, one_facts.file_url_bytes * 64);
        assert_eq!(
            crate::evaluation_work::planning_files(&many_facts).unwrap(),
            crate::evaluation_work::planning_files(&one_facts).unwrap() * 64
        );
        assert_eq!(
            crate::evaluation_work::execution_files(&many_facts).unwrap(),
            crate::evaluation_work::execution_files(&one_facts).unwrap() * 64
        );
        crate::log_store::tests::take_path_comparisons();
        lower_scan_json(
            scan(&many),
            many.log_identity_manifest().unwrap(),
            Arc::new(object_store::memory::InMemory::new()),
        )
        .unwrap();
        assert_eq!(crate::log_store::tests::take_path_comparisons(), 64);
    }

    #[test]
    fn planning_prepays_many_file_path_work() {
        let limits = TaskLimits::qualification();
        let one = admitted(1);
        let many = admitted(64);
        let one = ClosedPlanFacts::inspect(&one, limits).unwrap();
        let many = ClosedPlanFacts::inspect(&many, limits).unwrap();
        assert_eq!((one.files, many.files), (1, 64));
        let (work, allocations) = crate::json_arrays::tests::observe_allocations(|| {
            (
                crate::evaluation_work::planning(&one, limits).unwrap(),
                crate::evaluation_work::planning(&many, limits).unwrap(),
            )
        });
        assert_eq!(allocations, 0);
        // Even one exact path-validation walk must be paid before lowering.
        // This lower bound is independent of the implementation's final named
        // URL decode/copy and file-group visit counts.
        let minimum_delta = many.file_url_bytes - one.file_url_bytes;
        assert!(
            work.1 >= work.0 + minimum_delta,
            "64-file lowering omitted path work: one={}, many={}, extra URL bytes={}",
            work.0,
            work.1,
            minimum_delta
        );
    }

    #[test]
    fn execution_prepays_many_file_cached_store_path_work() {
        let limits = TaskLimits::qualification();
        let one = admitted(1);
        let many = admitted(64);
        let one = ClosedPlanFacts::inspect(&one, limits).unwrap();
        let many = ClosedPlanFacts::inspect(&many, limits).unwrap();
        // Equal row/byte envelopes isolate file lookup/open work from decoding.
        // Empty records still require every selected JsonSource file open.
        let one_work = crate::evaluation_work::execution(&one, 0, 64, limits).unwrap();
        let many_work = crate::evaluation_work::execution(&many, 0, 64, limits).unwrap();
        assert!(
            many_work >= one_work + many.file_url_bytes - one.file_url_bytes,
            "cached-store path work omitted: one={one_work}, many={many_work}"
        );
    }

    #[test]
    fn planning_rejects_file_work_arithmetic_overflow_without_allocating() {
        let limits = TaskLimits::qualification();
        for facts in [
            ClosedPlanFacts {
                files: usize::MAX,
                ..Default::default()
            },
            ClosedPlanFacts {
                file_url_bytes: usize::MAX,
                ..Default::default()
            },
        ] {
            let (result, allocations) = crate::json_arrays::tests::observe_allocations(|| {
                crate::evaluation_work::planning(&facts, limits)
            });
            assert_eq!(allocations, 0);
            let error = result.expect_err("file/path work overflow must not disappear");
            assert!(
                matches!(error.kind(), delta_kernel::tasks::FailureKind::ResourceExhausted(e)
                if e.resource == Resource::WorkUnits && e.observed == usize::MAX)
            );
        }
    }

    #[test]
    fn execution_file_work_overflow_is_typed_without_allocation() {
        for facts in [
            ClosedPlanFacts {
                files: usize::MAX,
                ..Default::default()
            },
            ClosedPlanFacts {
                file_url_bytes: usize::MAX,
                ..Default::default()
            },
        ] {
            let (result, allocations) = crate::json_arrays::tests::observe_allocations(|| {
                crate::evaluation_work::execution(&facts, 0, 0, TaskLimits::qualification())
            });
            assert_eq!(allocations, 0);
            assert!(matches!(result.unwrap_err().kind(),
                delta_kernel::tasks::FailureKind::ResourceExhausted(e)
                if e.resource == Resource::WorkUnits && e.observed == usize::MAX));
        }
    }

    #[test]
    fn many_file_work_exact_and_one_less_are_charged_before_lowering() {
        use delta_kernel::tasks::TaskAccounting;
        let limits = TaskLimits::qualification();
        let plan = admitted(64);
        let facts = ClosedPlanFacts::inspect(&plan, limits).unwrap();
        for units in [
            crate::evaluation_work::planning(&facts, limits).unwrap(),
            crate::evaluation_work::execution(&facts, 0, 64, limits).unwrap(),
        ] {
            // This is a minimum bound, not the final implementation formula:
            // one complete path comparison for every selected file is required.
            assert!(
                units >= facts.file_url_bytes,
                "many-file work omits path walks"
            );
            let exact = TaskAccounting::new(limits.with_limit(Resource::WorkUnits, units));
            exact.charge(Resource::WorkUnits, units).unwrap();
            assert_eq!(exact.usage(Resource::WorkUnits).consumed(), units);
            let short = TaskAccounting::new(limits.with_limit(Resource::WorkUnits, units - 1));
            let mut reached_lowering = false;
            let result = short.charge(Resource::WorkUnits, units).map(|()| {
                reached_lowering = true;
            });
            let error = result.unwrap_err();
            assert_eq!(error.resource, Resource::WorkUnits);
            assert_eq!((error.limit, error.observed), (units - 1, units));
            assert!(!reached_lowering);
            assert_eq!(short.usage(Resource::WorkUnits).consumed(), 0);
        }
    }

    #[tokio::test]
    async fn many_file_cached_reads_validate_canonical_path_and_descending_indices() {
        use delta_kernel::tasks::AdmittedRead;
        let plan = admitted(64);
        let manifest = plan.log_identity_manifest().unwrap().clone();
        let reads: Vec<_> = manifest
            .files()
            .iter()
            .map(|file| AdmittedRead {
                identity: file.identity,
                bytes: vec![b'\n'],
                offset: 0,
                eof: true,
            })
            .collect();
        let retained_bytes = manifest.retained_bytes()
            + plan.retained_bytes()
            + reads.capacity() * std::mem::size_of::<AdmittedRead>()
            + reads
                .iter()
                .map(|read| read.bytes.capacity())
                .sum::<usize>();
        let input = crate::log_input::LogInput {
            manifest,
            reads,
            retained_bytes,
            framing: crate::json_framing::JsonFraming::default(),
        };
        let store =
            crate::log_store::AdmittedLogStore::try_new(input, 64, TaskLimits::qualification())
                .unwrap();
        for invalid in [
            "other/_delta_log/00000000000000000000.json",
            "table/_delta_log/00000000000000000064.json",
            "table/_delta_log/0000000000000000000x.json",
            "table/_delta_log/18446744073709551616.json",
            "table/_delta_log/0000000000000000000.json",
            "table/_delta_log/00000000000000000000.crc",
        ] {
            assert!(
                store
                    .get_opts(
                        &object_store::path::Path::from(invalid),
                        object_store::GetOptions::default()
                    )
                    .await
                    .is_err(),
                "{invalid}"
            );
        }
        store.with_trace(|trace| assert!(trace.is_empty())).unwrap();
        crate::log_store::tests::take_path_comparisons();
        for version in (0..64).rev() {
            let path =
                object_store::path::Path::from(format!("table/_delta_log/{version:020}.json"));
            let result = store
                .get_opts(&path, object_store::GetOptions::default())
                .await
                .unwrap();
            assert_eq!(result.meta.location, path);
            assert_eq!(result.bytes().await.unwrap().as_ref(), b"\n");
        }
        assert_eq!(
            crate::log_store::tests::take_path_comparisons(),
            64,
            "one exact full-path comparison per indexed cached read"
        );
        store
            .with_trace(|trace| {
                assert_eq!(trace.len(), 64);
                for (position, read) in trace.iter().enumerate() {
                    assert_eq!(read.index, 63 - position);
                }
            })
            .unwrap();
        let path = object_store::path::Path::from("table/_delta_log/00000000000000000000.json");
        assert!(
            store
                .get_opts(&path, object_store::GetOptions::default())
                .await
                .is_err(),
            "duplicate pull must retain request limit"
        );
    }

    #[test]
    fn many_file_lowering_preserves_descending_cover_and_version_values() {
        let plan = admitted(64);
        let manifest = plan.log_identity_manifest().unwrap();
        let scan = scan(&plan);
        for (position, file) in scan.files.iter().enumerate() {
            let version = 63 - position;
            assert_eq!(file.file_constants, [Scalar::Long(version as i64)]);
            assert_eq!(file.meta.location.as_str(), manifest.files()[version].path);
        }
        crate::log_store::tests::take_path_comparisons();
        let logical = lower_scan_json(
            scan,
            manifest,
            Arc::new(object_store::memory::InMemory::new()),
        )
        .unwrap();
        assert_eq!(
            crate::log_store::tests::take_path_comparisons(),
            64,
            "one exact descriptor path comparison per indexed lowering"
        );
        let LogicalPlan::Projection(project) = logical else {
            panic!("declared-order projection")
        };
        let LogicalPlan::TableScan(table) = project.input.as_ref() else {
            panic!("ordinary table scan")
        };
        let provider = datafusion::datasource::source_as_provider(&table.source).unwrap();
        let provider = provider.downcast_ref::<JsonPlanTable>().unwrap();
        assert_eq!(provider.config.file_groups.len(), 1);
        let files = provider.config.file_groups[0].files();
        assert_eq!(files.len(), 64);
        for (position, file) in files.iter().enumerate() {
            let version = 63 - position;
            assert_eq!(
                file.partition_values,
                [datafusion::common::ScalarValue::Int64(Some(version as i64))]
            );
            assert_eq!(
                file.object_meta.location.as_ref(),
                format!("table/_delta_log/{version:020}.json")
            );
        }
    }

    #[test]
    fn many_file_lowering_rejects_descriptor_and_cover_mismatches() {
        let plan = admitted(64);
        let manifest = plan.log_identity_manifest().unwrap();
        // Mutate only a private lowering input cloned from the real sealed
        // producer; never construct or admit an arbitrary public plan.
        for case in 0..8 {
            let mut scan = scan(&plan).clone();
            match case {
                0 => scan.files[0].file_constants = vec![Scalar::Long(-1)],
                1 => scan.files[0].file_constants = vec![Scalar::Long(i64::MAX)],
                2 => scan.files[0].meta.size += 1,
                3 => scan.files[0].meta.last_modified += 1,
                4 => scan.files[0]
                    .meta
                    .location
                    .set_path("/table/_delta_log/outside.json"),
                5 => scan.files[0] = scan.files[1].clone(),
                6 => scan.files.swap(0, 1),
                7 => {
                    scan.files.pop();
                }
                _ => unreachable!(),
            }
            let result = lower_scan_json(
                &scan,
                manifest,
                Arc::new(object_store::memory::InMemory::new()),
            );
            assert!(matches!(result, Err(DataFusionError::Plan(_))),
                "descriptor/cover mismatch case {case} accepted or lost typed plan error: {result:?}");
        }
    }
}
