//! Test-only task/lowering parity over the frozen finite fixture. This intentionally uses ordinary
//! DataFusion collection to expose semantic defects while the production host envelope is being
//! completed. Physical size diagnostics below validate fixture-page transfer only; they are not
//! a decoder allocation proof, an AdmittedAsyncHost implementation, or final qualification.

use std::mem::size_of;
use std::sync::Arc;

use datafusion::arrow::array::{RecordBatch, StructArray};
use datafusion::prelude::SessionContext;
use delta_kernel::engine::arrow_conversion::TryIntoArrow;
use delta_kernel::engine::arrow_data::{fix_nested_null_masks, ArrowEngineData};
use delta_kernel::engine_data::RowVisitor;
use delta_kernel::expressions::{ArrayData, ColumnName};
use delta_kernel::schema::SchemaRef;
use delta_kernel::tasks::*;
use delta_kernel::{DeltaResult, EngineData};
use object_store::ObjectStore;

use crate::json_framing::{self, JsonFraming};
use crate::log_input::LogInput;
use crate::log_store::AdmittedLogStore;

const V0: &[u8] = include_bytes!("../tests/data/phase_d/00000000000000000000.json");
const V1: &[u8] = include_bytes!("../tests/data/phase_d/00000000000000000001.json");

struct FixturePageData {
    data: ArrowEngineData,
    bytes: usize,
}
impl FixturePageData {
    fn new(batch: RecordBatch) -> Self {
        let batch: RecordBatch = fix_nested_null_masks(StructArray::from(batch)).into();
        let schema = batch.schema();
        assert!(schema.metadata().is_empty());
        let fields = schema.flattened_fields().len();
        let schema_bytes = size_of_val(schema.as_ref())
            + schema.fields().size()
            + (2 * fields + 2) * 2 * size_of::<usize>();
        let bytes = batch.get_array_memory_size() + schema_bytes + size_of::<Self>();
        Self {
            data: ArrowEngineData::new(batch),
            bytes,
        }
    }
}
impl AccountedEngineData for FixturePageData {
    fn accounted_bytes(&self) -> Result<usize, ResourceExhausted> {
        Ok(self.bytes)
    }
}
impl EngineData for FixturePageData {
    fn len(&self) -> usize {
        self.data.len()
    }
    fn visit_rows(&self, columns: &[ColumnName], visitor: &mut dyn RowVisitor) -> DeltaResult<()> {
        self.data.visit_rows(columns, visitor)
    }
    fn append_columns(
        &self,
        schema: SchemaRef,
        columns: Vec<ArrayData>,
    ) -> DeltaResult<Box<dyn EngineData>> {
        self.data.append_columns(schema, columns)
    }
    fn apply_selection_vector(
        self: Box<Self>,
        selection: Vec<bool>,
    ) -> DeltaResult<Box<dyn EngineData>> {
        Box::new(self.data).apply_selection_vector(selection)
    }
    fn has_field(&self, name: &ColumnName) -> bool {
        self.data.has_field(name)
    }
}

fn descriptor(index: usize, bytes: &[u8]) -> FileDescriptor {
    FileDescriptor {
        path: format!("memory:///table/_delta_log/{index:020}.json"),
        size: bytes.len() as u64,
        modification_time: 0,
        identity: ObjectIdentity::new([index as u8; 32]),
    }
}

async fn evaluate(plan: &AdmittedPlan, session: &SessionContext) -> Vec<RecordBatch> {
    evaluate_result(plan, session).await.unwrap()
}

async fn evaluate_result(
    plan: &AdmittedPlan,
    session: &SessionContext,
) -> datafusion::common::Result<Vec<RecordBatch>> {
    let limits = TaskLimits::qualification();
    let (facts, allocations) = crate::json_arrays::tests::observe_allocations(|| {
        crate::closed_plan_facts::ClosedPlanFacts::inspect(plan, limits)
    });
    assert_eq!(allocations, 0);
    let facts = facts.unwrap();
    println!("closed producer resource inventory: {facts:?}");
    assert_eq!(facts.nodes, plan.plan().nodes.len());
    assert!(facts.fields > 0 && facts.aggregates > 0);
    assert_eq!(facts.case_depth, if facts.nodes == 2 { 0 } else { 2 });
    let (rejected, allocations) = crate::json_arrays::tests::observe_allocations(|| {
        crate::closed_plan_facts::ClosedPlanFacts::inspect(
            plan,
            limits.with_limit(delta_kernel::tasks::Resource::WorkUnits, facts.work - 1),
        )
    });
    assert_eq!(allocations, 0);
    assert!(rejected.is_err());
    let manifest = plan.log_identity_manifest().unwrap().clone();
    let mut framing = JsonFraming::default();
    let reads: Vec<_> = manifest
        .files()
        .iter()
        .zip([V0, V1])
        .map(|(file, bytes)| {
            let observed = json_framing::preflight(bytes, limits).unwrap();
            framing.records += observed.records;
            framing.tokens += observed.tokens;
            framing.max_depth = framing.max_depth.max(observed.max_depth);
            framing.max_record_bytes = framing.max_record_bytes.max(observed.max_record_bytes);
            AdmittedRead {
                identity: file.identity,
                bytes: bytes.to_vec(),
                offset: 0,
                eof: true,
            }
        })
        .collect();
    let input = LogInput {
        retained_bytes: manifest.retained_bytes()
            + plan.retained_bytes()
            + reads.iter().map(|r| r.bytes.capacity()).sum::<usize>()
            + reads.capacity() * size_of::<AdmittedRead>(),
        reads,
        manifest: manifest.clone(),
        framing,
    };
    report_component_budget(plan, &input, &facts, limits);
    let store = Arc::new(AdmittedLogStore::try_new(input, manifest.files().len(), limits).unwrap());
    let injected: Arc<dyn ObjectStore> = store.clone();
    let logical = crate::plan::lower_plan(plan.plan(), Some((&manifest, &injected))).unwrap();
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    let mut scalar_owners = Vec::new();
    let mut aggregate_owners = Vec::new();
    logical
        .apply(|node| {
            for expression in node.expressions() {
                expression.apply(|expression| {
                    match expression {
                        datafusion::logical_expr::Expr::ScalarFunction(function) => {
                            scalar_owners.push(Arc::downgrade(function.func.inner()))
                        }
                        datafusion::logical_expr::Expr::AggregateFunction(function) => {
                            aggregate_owners.push(Arc::downgrade(function.func.inner()))
                        }
                        _ => {}
                    }
                    Ok(TreeNodeRecursion::Continue)
                })?;
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
    assert!(!aggregate_owners.is_empty());
    let state = crate::metadata_session::new(session).unwrap();
    // Closed lowering has already supplied declared schemas, projection casts,
    // typed predicates, ordered FIRST_VALUE and the builtin Coalesce rewrite.
    // Test the concrete physical planner directly, without a second logical
    // analyzer/optimizer traversal over this producer-owned typed plan.
    let planner = datafusion::physical_planner::DefaultPhysicalPlanner::default();
    let future = planner.create_physical_plan_unboxed(&logical, &state);
    let _outer_frame = size_of_val(&future); // unpolled, unboxed frame only
    let physical = future.await.unwrap();
    fn check_case_format_domain(
        expr: &Arc<dyn datafusion::physical_expr::PhysicalExpr>,
        inside_case: bool,
    ) {
        use datafusion::physical_expr::expressions::{CaseExpr, CastExpr};
        // function_allocation's return-field formatter envelope relies on
        // CASE descendants using the fixed accessor/literal/boolean domain,
        // not formatting arbitrary cast target schemas.
        assert!(
            !(inside_case && expr.downcast_ref::<CastExpr>().is_some()),
            "cast beneath CASE is outside the proved formatter domain: {expr}"
        );
        let nested = inside_case || expr.downcast_ref::<CaseExpr>().is_some();
        for child in expr.children() {
            check_case_format_domain(child, nested);
        }
    }
    fn check_closed_requirements(plan: &dyn datafusion::physical_plan::ExecutionPlan) -> usize {
        use datafusion::physical_plan::Distribution;
        assert_eq!(plan.properties().output_partitioning().partition_count(), 1);
        assert!(plan.required_input_ordering().iter().all(Option::is_none));
        assert!(plan.properties().output_ordering().is_none());
        if let Some(project) =
            plan.downcast_ref::<datafusion::physical_plan::projection::ProjectionExec>()
        {
            for expression in project.expr() {
                check_case_format_domain(&expression.expr, false);
            }
        }
        if let Some(filter) = plan.downcast_ref::<datafusion::physical_plan::filter::FilterExec>() {
            check_case_format_domain(filter.predicate(), false);
            // NOT/IS NULL are outside interval check_support. The fixed
            // filters therefore never allocate interval-analysis arrays.
            assert!(!datafusion::physical_expr::intervals::utils::check_support(
                filter.predicate(),
                &filter.input().schema()
            ));
        }
        assert!(plan
            .input_distribution_requirements()
            .per_child_distributions()
            .all(|d| matches!(
                d,
                Distribution::SinglePartition | Distribution::UnspecifiedDistribution
            )));
        assert!(matches!(
            plan.name(),
            "DataSourceExec" | "ProjectionExec" | "FilterExec" | "AggregateExec"
        ));
        let children = plan.children();
        assert!(children.len() <= 1);
        1 + children
            .iter()
            .map(|child| check_closed_requirements(child.as_ref()))
            .sum::<usize>()
    }
    let physical_nodes = check_closed_requirements(physical.as_ref());
    assert_eq!(
        physical_nodes,
        if plan.plan().nodes.len() == 2 { 4 } else { 9 }
    );
    println!(
        "{}",
        datafusion::physical_plan::displayable(physical.as_ref()).indent(true)
    );
    let batches = datafusion::physical_plan::collect(physical, state.task_ctx()).await?;
    drop(logical);
    assert!(
        scalar_owners.iter().all(|owner| owner.upgrade().is_none()),
        "task scalar builtin survived plan/stream drop"
    );
    assert!(
        aggregate_owners
            .iter()
            .all(|owner| owner.upgrade().is_none()),
        "task aggregate builtin survived plan/stream drop"
    );
    assert_eq!(
        store.with_trace(|trace| trace.len()).unwrap(),
        manifest.files().len()
    );
    let expected = match &plan.plan().nodes.last().unwrap().op {
        delta_kernel::plans::ir::nodes::Operator::Project(project) => project.schema.as_ref(),
        delta_kernel::plans::ir::nodes::Operator::Aggregate(aggregate) => aggregate.schema.as_ref(),
        _ => panic!("unexpected closed producer terminal"),
    };
    let expected: datafusion::arrow::datatypes::SchemaRef =
        Arc::new(expected.try_into_arrow().unwrap());
    Ok(batches
        .into_iter()
        .map(|batch| {
            let batch = crate::result_schema::rebind(batch, expected.clone()).unwrap();
            assert_eq!(batch.schema(), expected);
            batch
        })
        .collect())
}

/// Coordinator-requested source-component table. This deliberately does NOT
/// turn an incomplete subtotal into host admission, and does not measure heap
/// growth to derive any component. Missing owner groups remain listed in P.
fn report_component_budget(
    plan: &AdmittedPlan,
    input: &LogInput,
    facts: &crate::closed_plan_facts::ClosedPlanFacts,
    limits: TaskLimits,
) {
    use datafusion::arrow::datatypes::Schema;
    use delta_kernel::plans::ir::nodes::Operator;
    let bytes = input.reads.iter().map(|r| r.bytes.len()).sum::<usize>();
    let scan = plan
        .plan()
        .nodes
        .iter()
        .find_map(|n| match &n.op {
            Operator::ScanJson(scan) => Some(scan),
            _ => None,
        })
        .unwrap();
    let agg = plan
        .plan()
        .nodes
        .iter()
        .find_map(|n| match &n.op {
            Operator::Aggregate(agg) => Some(agg),
            _ => None,
        })
        .unwrap();
    let mut conversions = 0usize;
    let mut max_copy = 0usize;
    let mut max_tree = 0usize;
    let mut max_headers = 0usize;
    let mut max_schema = 0usize;
    for node in &plan.plan().nodes {
        let schema = match &node.op {
            Operator::ScanJson(op) => Some(&op.schema),
            Operator::Project(op) => Some(&op.schema),
            Operator::Aggregate(op) => Some(&op.schema),
            _ => None,
        };
        if let Some(schema) = schema {
            let schema_owner = crate::schema_conversion_allocation::peak(schema, limits).unwrap();
            conversions += schema_owner;
            max_schema = max_schema.max(schema_owner);
            let converted: Schema = schema.as_ref().try_into_arrow().unwrap();
            max_copy = max_copy
                .max(crate::json_arrays::copy_owner_peak(&converted, 1, bytes, 1, limits).unwrap());
            max_tree = max_tree.max(
                crate::json_arrays::output_owner_peak(
                    &converted,
                    input.framing.records,
                    bytes,
                    limits,
                )
                .unwrap(),
            );
            max_headers = max_headers
                .max(crate::json_arrays::output_owner_peak(&converted, 1, 0, limits).unwrap());
        }
    }
    let input_schema: Schema = scan.schema.as_ref().try_into_arrow().unwrap();
    let output_schema: Schema = agg.schema.as_ref().try_into_arrow().unwrap();
    let decoder =
        crate::json_arrays::DecoderEnvelope::preflight(&input_schema, input.framing, limits)
            .unwrap();
    let (case_buffers, allocations) = crate::json_arrays::tests::observe_allocations(|| {
        crate::case_allocation::peak(facts, max_headers, max_schema, bytes, limits)
    });
    assert_eq!(allocations, 0);
    let case_buffers = case_buffers.unwrap();
    if case_buffers != 0 {
        for (limit, allowed) in [(case_buffers, true), (case_buffers - 1, false)] {
            let (result, allocations) = crate::json_arrays::tests::observe_allocations(|| {
                crate::case_allocation::peak(
                    facts,
                    max_headers,
                    max_schema,
                    bytes,
                    limits.with_limit(Resource::MetadataAllocatedBytes, limit),
                )
            });
            assert_eq!(allocations, 0);
            assert_eq!(result.is_ok(), allowed);
        }
    }
    let (planning, allocations) = crate::json_arrays::tests::observe_allocations(|| {
        crate::planning_allocation::preflight(facts, conversions, limits)
    });
    assert_eq!(allocations, 0);
    let planning = planning.unwrap();
    for (limit, allowed) in [
        (planning.construction, true),
        (planning.construction - 1, false),
    ] {
        let (result, allocations) = crate::json_arrays::tests::observe_allocations(|| {
            crate::planning_allocation::preflight(
                facts,
                conversions,
                limits.with_limit(Resource::MetadataAllocatedBytes, limit),
            )
        });
        assert_eq!(allocations, 0);
        assert_eq!(result.is_ok(), allowed);
    }
    println!(
        "PLANNING_COMPONENT kind={} version={} retained={} construction={}",
        if agg.group_by.is_empty() {
            "pm"
        } else {
            "live_add"
        },
        input.manifest.version(),
        planning.retained,
        planning.construction
    );
    let (functions, allocations) = crate::json_arrays::tests::observe_allocations(|| {
        crate::function_allocation::peak(facts, max_headers, limits)
    });
    assert_eq!(allocations, 0);
    let functions = functions.unwrap();
    for (limit, allowed) in [(functions, true), (functions - 1, false)] {
        let (result, allocations) = crate::json_arrays::tests::observe_allocations(|| {
            crate::function_allocation::peak(
                facts,
                max_headers,
                limits.with_limit(Resource::MetadataAllocatedBytes, limit),
            )
        });
        assert_eq!(allocations, 0);
        assert_eq!(result.is_ok(), allowed);
    }
    println!(
        "FUNCTION_COMPONENT kind={} version={} bytes={}",
        if agg.group_by.is_empty() {
            "pm"
        } else {
            "live_add"
        },
        input.manifest.version(),
        functions
    );
    let auxiliary = crate::execution_aux::peak(limits).unwrap();
    println!("NEW_COMPONENTS kind={} version={} case_buffers={} execution_aux={} coalesce_utf8={} coalesce_int32={}",
        if agg.group_by.is_empty() { "pm" } else { "live_add" }, input.manifest.version(),
        case_buffers, auxiliary, facts.coalesce_utf8, facts.coalesce_int32);
    let frames = crate::evaluation_frames::peak(limits).unwrap();
    let bootstrap = crate::metadata_session::bootstrap_peak(limits).unwrap();
    // Each FIRST_VALUE owns only its own value subtree. Global PM has one
    // winner/output row; grouped live-add retains all admitted groups. The
    // borrowed DataType API allocates no one-field Schema wrapper.
    let state_rows = if agg.group_by.is_empty() {
        1
    } else {
        input.framing.records
    };
    let mut both_states = 0usize;
    for field in output_schema.fields().iter().skip(agg.group_by.len()) {
        let (bound, allocations) = crate::json_arrays::tests::observe_allocations(|| {
            crate::first_value_allocation::state_type_peak(
                field.data_type(),
                state_rows,
                bytes,
                limits,
            )
        });
        assert_eq!(allocations, 0);
        both_states += bound.unwrap() * 2;
    }
    let keys = if agg.group_by.is_empty() {
        0
    } else {
        let key_schema = Schema::new(vec![output_schema.fields()[0].clone()]);
        crate::row_key_allocation::grouping_owner_peak(
            &key_schema,
            input.framing.records,
            bytes,
            limits,
        )
        .unwrap()
            * 2
    };
    let page_tree =
        crate::json_arrays::output_owner_peak(&output_schema, input.framing.records, bytes, limits)
            .unwrap();
    let terminal = match &plan.plan().nodes.last().unwrap().op {
        Operator::Aggregate(op) => op.schema.as_ref(),
        Operator::Project(op) => op.schema.as_ref(),
        _ => unreachable!(),
    };
    let terminal_schema: datafusion::arrow::datatypes::SchemaRef =
        Arc::new(terminal.try_into_arrow().unwrap());
    let terminal_schema_owners =
        crate::schema_conversion_allocation::peak(terminal, limits).unwrap();
    let page = crate::evaluation_page::PageEnvelope::preflight(
        &terminal_schema,
        state_rows,
        bytes,
        terminal_schema_owners,
        limits,
    )
    .unwrap();
    println!(
        "PAGE_COMPONENTS kind={} version={} retained={} rebind_peak={} max_pages={}",
        if agg.group_by.is_empty() {
            "pm"
        } else {
            "live_add"
        },
        input.manifest.version(),
        page.retained,
        page.rebind_peak,
        state_rows
    );

    let subtotal =
        input.retained_bytes + bootstrap + frames + conversions + decoder.peak + both_states + keys;
    println!("UNCOMPOSED_BUFFER_INPUTS kind={} version={} max_one_row_system_copy={} max_full_backing_system_tree={} cases={} case_branches={}",
        if agg.group_by.is_empty() { "pm" } else { "scan" }, input.manifest.files().len()-1,
        max_copy, max_tree, facts.cases, facts.case_branches);
    println!("COMPONENT_BUDGET kind={} version={} log_bytes={} records={} tokens={} max_record={} retained_input_plan_manifest={} plan_retained={} bootstrap={} frames={} initial_schema_conversions={} decoder={} value_states_both_stages={} key_states_both_stages={} INCOMPLETE_live_subtotal={} full_backing_aggregate_array_tree={} kernel_plan_work={} inventory_work={} task_state_limit={} metadata_limit={} page_limit={} work_limit={} cases={}",
        if agg.group_by.is_empty() { "pm" } else { "scan" }, input.manifest.files().len()-1,
        bytes, input.framing.records, input.framing.tokens, input.framing.max_record_bytes,
        input.retained_bytes, plan.retained_bytes(), bootstrap, frames, conversions, decoder.peak,
        both_states, keys, subtotal, page_tree, plan.shape().work_units(), facts.work,
        limits.limit(Resource::TaskStateBytes), limits.limit(Resource::MetadataAllocatedBytes),
        limits.limit(Resource::EvaluationPageBytes), limits.limit(Resource::WorkUnits), facts.cases);
}

fn page(next: Option<RecordBatch>, limits: EvaluationPageLimits) -> Option<EvaluationPage> {
    next.map(|batch| {
        EvaluationPage::try_new(vec![Box::new(FixturePageData::new(batch))], limits).unwrap()
    })
}

async fn drive<T: OperationTask>(task: &mut T, session: &SessionContext) -> T::Output {
    let cpu = CpuSlice::new(1024, 1 << 20, 256).unwrap();
    let mut step = task.start(cpu).unwrap();
    let mut batches = Vec::new().into_iter();
    for _ in 0..100 {
        step = match step {
            TaskStep::Yield => task.progress(cpu).unwrap(),
            TaskStep::Execute(request) => {
                let response = match request.operation {
                    TaskRequestV1::List {
                        continuation: None, ..
                    } => TaskResponseV1::Listing {
                        files: vec![descriptor(0, V0), descriptor(1, V1)],
                        continuation: None,
                    },
                    TaskRequestV1::EvaluationStart {
                        evaluation,
                        plan,
                        limits,
                    } => {
                        batches = evaluate(&plan, session).await.into_iter();
                        TaskResponseV1::Evaluation {
                            evaluation,
                            page: page(batches.next(), limits.page()),
                        }
                    }
                    TaskRequestV1::Evaluation { evaluation, limits } => {
                        TaskResponseV1::Evaluation {
                            evaluation,
                            page: page(batches.next(), limits),
                        }
                    }
                    _ => panic!("unexpected fixture task request"),
                };
                task.resume(request.key, Ok(response), cpu).unwrap()
            }
            TaskStep::Complete(output) => return output,
            TaskStep::Failed(error) => panic!("fixture task failed: {error:?}"),
            TaskStep::Cancelled => panic!("unexpected task cancellation"),
        };
    }
    panic!("fixture task did not terminate");
}

#[tokio::test]
async fn fixed_metadata_rules_preserve_snapshot_and_live_add_semantics() {
    let session = SessionContext::new();
    session.runtime_env().register_object_store(
        &url::Url::parse("memory:///").unwrap(),
        Arc::new(object_store::memory::InMemory::new()),
    );
    let limits = TaskLimits::qualification();
    for (version, expected_version, expected_paths) in [
        (None, 1, vec!["part-a.parquet"]),
        (Some(0), 0, vec!["part-a.parquet", "part-b.parquet"]),
    ] {
        let id = TaskId::allocate().unwrap();
        let evaluation = EvaluationKey::allocate(id.get()).unwrap();
        let mut load = SnapshotLoadTask::try_new(
            id,
            evaluation,
            &url::Url::parse("memory:///table/").unwrap(),
            version,
            limits,
        )
        .unwrap();
        let snapshot = drive(&mut load, &session).await;
        assert_eq!(snapshot.version(), expected_version);
        assert_eq!(snapshot.schema().num_fields(), 9);
        let id = TaskId::allocate().unwrap();
        let evaluation = EvaluationKey::allocate(id.get()).unwrap();
        let mut scan = ScanMetadataTask::try_new(id, evaluation, snapshot, limits).unwrap();
        let metadata = drive(&mut scan, &session).await;
        let mut paths = Vec::new();
        for batch in &metadata {
            paths = batch
                .visit_scan_files(paths, |paths, file| {
                    assert_eq!(file.size, 3029);
                    assert!(!file.dv_info.has_vector());
                    assert!(file.partition_values.is_empty());
                    assert!(file.transform.is_none());
                    paths.push(file.path);
                })
                .unwrap();
        }
        paths.sort();
        assert_eq!(paths, expected_paths);
    }
}

/// Source-narrowing probe, not an evaluation allocation bound or raised task
/// limit. The selected no-disk implementation cannot create a spill file.
#[tokio::test]
async fn reached_live_add_tiny_pool_spill_probe() {
    use datafusion::execution::config::SessionConfig;
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    let caller = SessionContext::new();
    caller.runtime_env().register_object_store(
        &url::Url::parse("memory:///").unwrap(),
        Arc::new(object_store::memory::InMemory::new()),
    );
    let limits = TaskLimits::qualification();
    let id = TaskId::allocate().unwrap();
    let evaluation = EvaluationKey::allocate(id.get()).unwrap();
    let mut load = SnapshotLoadTask::try_new(
        id,
        evaluation,
        &url::Url::parse("memory:///table/").unwrap(),
        None,
        limits,
    )
    .unwrap();
    let snapshot = drive(&mut load, &caller).await;
    let id = TaskId::allocate().unwrap();
    let evaluation = EvaluationKey::allocate(id.get()).unwrap();
    let mut scan = ScanMetadataTask::try_new(id, evaluation, snapshot, limits).unwrap();
    let TaskStep::Execute(request) = scan
        .start(CpuSlice::new(1024, 1 << 20, 256).unwrap())
        .unwrap()
    else {
        panic!("scan evaluation expected")
    };
    let TaskRequestV1::EvaluationStart { plan, .. } = request.operation else {
        panic!("closed plan expected")
    };
    let mut refused = 0;
    for bytes in [1, 1024, 4096, 8192, 16384, 32768, 65536, 131072] {
        let pool = Arc::new(GreedyMemoryPool::new(bytes));
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(pool.clone())
            .build_arc()
            .unwrap();
        assert!(!runtime.disk_manager.tmp_files_enabled());
        assert!(runtime
            .disk_manager
            .create_tmp_file("closed task probe")
            .is_err());
        runtime.register_object_store(
            &url::Url::parse("memory:///").unwrap(),
            Arc::new(object_store::memory::InMemory::new()),
        );
        let session = SessionContext::new_with_config_rt(SessionConfig::new(), runtime);
        let result = evaluate_result(&plan, &session).await;
        println!(
            "live-add tiny-pool bytes={bytes} disk_capability=false result={}",
            match &result {
                Ok(rows) => format!("ok:{} batches", rows.len()),
                Err(error) => error.to_string(),
            }
        );
        if result.is_err() {
            refused += 1;
        }
        assert_eq!(
            pool.reserved(),
            0,
            "execution owners must release after success/error"
        );
    }
    assert!(refused > 0, "the probe must reach a memory refusal");
}
