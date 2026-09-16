//! Fixed planning environment for Kernel's two closed metadata producers.
//!
//! This state shares the caller's pool, cache and storage resources through Arc ownership. It never
//! clones the caller's user registries, invokes custom optimizer/planner callbacks, or alters
//! registrations. The host must admit this state's construction, lowering and execution before
//! calling `new`.

use std::sync::Arc;

use datafusion::catalog::MemoryCatalogProviderList;
use datafusion::common::Result;
use datafusion::execution::config::SessionConfig;
use datafusion::execution::context::SessionContext;
use datafusion::execution::disk_manager::{DiskManagerBuilder, DiskManagerMode};
use datafusion::execution::session_state::{SessionState, SessionStateBuilder};
use datafusion::physical_optimizer::sanity_checker::SanityCheckPlan;

/// Construction owners of this module's fixed session (excluding logical/
/// physical plans, streams and the surrounding host/future). No caller maps,
/// registered functions or cache objects are cloned by this calculation.
pub(crate) fn bootstrap_peak(
    limits: delta_kernel::tasks::TaskLimits,
) -> std::result::Result<usize, delta_kernel::tasks::OperationFailure> {
    use std::mem::size_of;
    use std::sync::atomic::{AtomicU64, AtomicUsize};
    use std::sync::RwLock;

    use datafusion::common::alias::AliasGenerator;
    use datafusion::common::config::{ConfigOptions, TableOptions};
    use datafusion::common::HashMap;
    use datafusion::execution::disk_manager::DiskManager;
    use datafusion::execution::runtime_env::RuntimeEnv;
    use datafusion::logical_expr::registry::{
        ExtensionTypeRegistrationRef, MemoryExtensionTypeRegistry,
    };
    use datafusion::optimizer::analyzer::resolve_grouping_function::ResolveGroupingFunction;
    use datafusion::optimizer::analyzer::type_coercion::TypeCoercion;
    use delta_kernel::tasks::{Resource, ResourceExhausted};
    let arc = 2 * size_of::<usize>();
    let overflow = || ResourceExhausted {
        resource: Resource::MetadataAllocatedBytes,
        limit: limits.limit(Resource::MetadataAllocatedBytes),
        observed: usize::MAX,
    };
    // ConfigOptions' nonempty default String sites: catalog/schema, SQL null
    // ordering, Parquet compression/statistics/created_by and display formats.
    // Empty strings and None options allocate no backing. These selected
    // defaults are independent of caller session options and registries.
    let parquet_strings = "zstd(3)".len()
        + "page".len()
        + "datafusion version ".len()
        + datafusion::DATAFUSION_VERSION.len();
    let config_strings = "datafusion".len()
        + "public".len()
        + "nulls_max".len()
        + parquet_strings
        + "%Y-%m-%d".len()
        + 2 * "%Y-%m-%dT%H:%M:%S%.f".len()
        + "%H:%M:%S%.f".len()
        + "pretty".len();
    // TableOptions default, its clone, then replacement Parquet options cloned
    // from ConfigOptions; old/new strings coexist at that assignment.
    let table_strings = parquet_strings.checked_mul(3).ok_or_else(overflow)?;
    let headers = [
        size_of::<SessionState>(),
        size_of::<SessionStateBuilder>(),
        size_of::<RuntimeEnv>() + arc,
        size_of::<DiskManager>() + arc,
        size_of::<DiskManagerBuilder>(),
        size_of::<ConfigOptions>() + arc,
        2 * size_of::<TableOptions>(),
        // Disabled DiskManager still owns these two independent counter Arcs.
        size_of::<AtomicU64>() + arc,
        size_of::<AtomicUsize>() + arc,
        // Default execution props plus its replacement when execution starts.
        2 * (size_of::<AliasGenerator>() + arc),
        size_of::<MemoryCatalogProviderList>() + arc,
        size_of::<MemoryExtensionTypeRegistry>() + arc,
        size_of::<RwLock<HashMap<String, ExtensionTypeRegistrationRef>>>() + arc,
        size_of::<ResolveGroupingFunction>() + arc,
        size_of::<TypeCoercion>() + arc,
        size_of::<SanityCheckPlan>() + arc,
        // DefaultQueryPlanner and EmptySerializerRegistry are zero-sized types.
        2 * arc,
    ];
    let headers = headers
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(overflow)?;
    // Two nonempty rule Vec sites (2 unused analyzer,1 physical sanity rule).
    let rules = [2, 1]
        .into_iter()
        .try_fold(0usize, |n, count| {
            n.checked_add(crate::json_arrays::vec_peak(count, 2 * size_of::<usize>())?)
        })
        .ok_or_else(overflow)?;
    // DashMap6.2.1 uses two empty RawTable shards, each inside its one-word
    // RawRwLock and CachePadded. On selected x86_64 and wasm32 the enclosing
    // alignment is at most128, and the empty shard header fits one such unit.
    // Vec->Box shrink/relocation is covered by the old/new capacity helper.
    let shards = crate::json_arrays::vec_peak(2, 128).ok_or_else(overflow)?;
    let peak = [
        headers,
        rules,
        shards,
        config_strings,
        table_strings,
        "kernel_json_task".len(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .ok_or_else(overflow)?;
    if peak > limits.limit(Resource::MetadataAllocatedBytes) {
        return Err(ResourceExhausted {
            resource: Resource::MetadataAllocatedBytes,
            limit: limits.limit(Resource::MetadataAllocatedBytes),
            observed: peak,
        }
        .into());
    }
    Ok(peak)
}

/// Task-local built-in planning rules; user SQL still executes on the caller SessionContext.
/// Lowered Kernel expressions carry the selected builtin function Arcs directly, so this state
/// needs neither name-based UDF registration nor a catalog/table lookup.
pub(crate) fn new(caller: &SessionContext) -> Result<SessionState> {
    // Clone the resource Arc handles directly. RuntimeEnvBuilder::from_runtime_env
    // would rebuild CacheManager and update caller cache policies as a side effect.
    // Only metadata execution receives the local Disabled disk manager; caller SQL
    // and its Parquet execution retain the original runtime and registrations.
    let mut runtime = caller.runtime_env().as_ref().clone();
    runtime.disk_manager = Arc::new(
        DiskManagerBuilder::default()
            .with_mode(DiskManagerMode::Disabled)
            .build()?,
    );
    let runtime = Arc::new(runtime);
    let mut config = SessionConfig::new()
        .with_create_default_catalog_and_schema(false)
        .with_target_partitions(1)
        .with_batch_size(1)
        .with_repartition_joins(false)
        .with_repartition_aggregations(false)
        .with_repartition_file_scans(false)
        .with_repartition_windows(false)
        .with_repartition_sorts(false);
    // The metadata converter applies the actual Coalesce rewrite once.
    // No general logical optimizer passes are needed for these closed plans.
    // Planning concurrency and default catalog sharding otherwise depend on
    // available_parallelism, independently of target_partitions/batch size.
    config.options_mut().execution.planning_concurrency = 1;
    // One admitted file group has no sibling work to steal. This also avoids
    // allocating a second shared queue of cloned file descriptors.
    config
        .options_mut()
        .execution
        .enable_file_stream_work_stealing = false;
    let catalog = Arc::new(MemoryCatalogProviderList {
        catalogs: dashmap::DashMap::with_shard_amount(2),
    });
    config.options_mut().optimizer.max_passes = 1;
    config.options_mut().optimizer.skip_failed_rules = false;
    config.options_mut().execution.enable_migration_aggregate = true;
    // Closed producers contain no scalar subqueries. Avoid the otherwise
    // unconditional collection walk; ordinary caller query options are intact.
    config
        .options_mut()
        .optimizer
        .enable_physical_uncorrelated_scalar_subquery = false;
    // Selected PartialHashAggregateStream treats1.0 as the explicit disabled
    // setting. Keep the single admitted partial state machine: never construct
    // an additional partial-skip table or change into row-to-state conversion.
    // This is an optimizer choice, not a task resource allowance.
    config
        .options_mut()
        .execution
        .skip_partial_aggregation_probe_ratio_threshold = 1.0;
    let mut state = SessionStateBuilder::new()
        .with_session_id("kernel_json_task".into())
        .with_config(config)
        .with_runtime_env(runtime)
        .with_catalog_list(catalog)
        // Analyzer's two builtins (ResolveGroupingFunction, TypeCoercion) remain enabled.
        // Coalesce has already been rewritten by its builtin implementation.
        .with_optimizer_rules(vec![])
        // One file group and all unary operators already satisfy Final's
        // SinglePartition requirement. FIRST_VALUE's ordering is Beneficial,
        // not required: its accumulator compares the admitted version key.
        // Keep sanity plus DefaultPhysicalPlanner's invariant checks; there
        // is no need to run broad distribution/order enforcement passes.
        .with_physical_optimizer_rules(vec![Arc::new(SanityCheckPlan::new())])
        .build();
    // Direct physical planning must bind the existing config Arc. Otherwise
    // create_physical_expr allocates default ConfigOptions per scalar UDF.
    // bootstrap_peak includes mark_start_execution's alias-generator replacement.
    state.mark_start_execution();
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parsed_store_origin_reuses_canonical_host_and_same_validation() {
        use datafusion::execution::object_store::ObjectStoreUrl;
        for text in ["memory://", "file://", "https://xn--r8jz45g.xn--zckzah"] {
            let parsed = url::Url::parse(text).unwrap();
            let from_parsed = ObjectStoreUrl::try_from_url(parsed).unwrap();
            assert_eq!(from_parsed, ObjectStoreUrl::parse(text).unwrap());
        }
        for text in [
            "memory:///table",
            "https://example.com/?query",
            "https://example.com/#fragment",
        ] {
            assert!(ObjectStoreUrl::try_from_url(url::Url::parse(text).unwrap()).is_err());
            assert!(ObjectStoreUrl::parse(text).is_err());
        }
    }

    #[test]
    fn projected_struct_accessors_share_the_private_config() {
        use datafusion::arrow::datatypes::{DataType, Field, Schema};
        use datafusion::common::ScalarValue;
        use datafusion::logical_expr::ScalarUDF;
        use datafusion::physical_expr::expressions::{Column, Literal};
        use datafusion::physical_expr::projection::ProjectionMapping;
        use datafusion::physical_expr::{PhysicalExpr, ScalarFunctionExpr};
        let state = new(&SessionContext::new()).unwrap();
        let options = state.config().options();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "source",
            DataType::Int64,
            false,
        )]));
        let udf = Arc::new(ScalarUDF::from(
            datafusion::functions::core::named_struct::NamedStructFunc::new(),
        ));
        let args: Vec<Arc<dyn PhysicalExpr>> = vec![
            Arc::new(Literal::new(ScalarValue::Utf8(Some("child".into())))),
            Arc::new(Column::new("source", 0)),
        ];
        let expr: Arc<dyn PhysicalExpr> =
            Arc::new(ScalarFunctionExpr::try_new(udf, args, &schema, options.clone()).unwrap());
        let mapping = ProjectionMapping::try_new(vec![(expr, "packed".into())], &schema).unwrap();
        let mut accessors = 0;
        for targets in mapping.values() {
            for (target, _) in targets.iter() {
                if let Some(function) = target.downcast_ref::<ScalarFunctionExpr>() {
                    assert_eq!(function.name(), "get_field");
                    assert!(Arc::ptr_eq(function.config_options_arc(), options));
                    accessors += 1;
                }
            }
        }
        assert_eq!(accessors, 1);
    }

    #[test]
    fn fixed_bootstrap_admission_is_allocation_free_at_the_boundary() {
        use delta_kernel::tasks::{FailureKind, Resource, TaskLimits};
        let limits = TaskLimits::qualification();
        let (peak, allocations) =
            crate::json_arrays::tests::observe_allocations(|| bootstrap_peak(limits));
        assert_eq!(allocations, 0);
        let peak = peak.unwrap();
        assert!(bootstrap_peak(limits.with_limit(Resource::MetadataAllocatedBytes, peak)).is_ok());
        let (error, allocations) = crate::json_arrays::tests::observe_allocations(|| {
            bootstrap_peak(limits.with_limit(Resource::MetadataAllocatedBytes, peak - 1))
        });
        assert_eq!(allocations, 0);
        assert!(
            matches!(error.unwrap_err().kind(), FailureKind::ResourceExhausted(e)
            if e.resource == Resource::MetadataAllocatedBytes && e.observed == peak)
        );
    }

    #[test]
    fn metadata_state_keeps_caller_resources_and_fixed_rules() {
        let caller = SessionContext::new();
        let state = new(&caller).unwrap();
        let original = caller.runtime_env();
        assert!(!Arc::ptr_eq(state.runtime_env(), &original));
        assert!(Arc::ptr_eq(
            &state.runtime_env().object_store_registry,
            &original.object_store_registry
        ));
        assert!(Arc::ptr_eq(
            &state.runtime_env().cache_manager,
            &original.cache_manager
        ));
        assert!(!Arc::ptr_eq(
            &state.runtime_env().disk_manager,
            &original.disk_manager
        ));
        assert!(!state.runtime_env().disk_manager.tmp_files_enabled());
        assert!(Arc::ptr_eq(
            &state.runtime_env().memory_pool,
            &caller.runtime_env().memory_pool
        ));
        assert_eq!(state.config().target_partitions(), 1);
        assert_eq!(state.config().batch_size(), 1);
        assert_eq!(state.config().options().execution.planning_concurrency, 1);
        assert!(
            !state
                .config()
                .options()
                .execution
                .enable_file_stream_work_stealing
        );
        assert!(
            !state
                .config()
                .options()
                .optimizer
                .enable_physical_uncorrelated_scalar_subquery
        );
        assert_eq!(
            state
                .config()
                .options()
                .execution
                .skip_partial_aggregation_probe_ratio_threshold,
            1.0
        );
        assert!(state.catalog_list().catalog_names().is_empty());
        assert_eq!(state.config().options().optimizer.max_passes, 1);
        assert!(!state.config().options().optimizer.skip_failed_rules);
        assert!(
            state
                .config()
                .options()
                .execution
                .enable_migration_aggregate
        );
        assert!(Arc::ptr_eq(
            state.execution_props().config_options.as_ref().unwrap(),
            state.config().options()
        ));
        assert!(state.optimizers().is_empty());
        assert_eq!(state.analyzer().rules.len(), 2);
        assert_eq!(state.physical_optimizers().len(), 1);
        assert_eq!(state.physical_optimizers()[0].name(), "SanityCheckPlan");
        assert!(state.scalar_functions().is_empty());
        assert!(state.aggregate_functions().is_empty());
    }
}
