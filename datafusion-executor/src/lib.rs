//! A DataFusion-based [`PlanExecutor`](delta_kernel::PlanExecutor) for delta_kernel declarative
//! plans.
//!
//! Kernel emits executor-independent logical [`Plan`](delta_kernel::plans::ir::plan::Plan)s; this
//! crate executes them by lowering each plan to a DataFusion `LogicalPlan`, optimizing it, and
//! running the resulting `ExecutionPlan`.

// TODO: remove once `session_ctx` and `storage_handler` are consumed by the query-execution path.
#![allow(dead_code)]

use std::sync::Arc;

use datafusion::execution::context::SessionContext;
use delta_kernel::StorageHandler;

mod closed_plan_facts;
mod expression;
mod json_allocation;
mod json_arrays;
mod json_framing;
mod json_scan;
#[cfg(test)]
mod json_task_semantics;
mod log_input;
pub mod log_storage;
mod log_store;
mod metadata_session;
mod operator;
mod plan;
mod predicate;
mod result_schema;
mod row_key_allocation;
mod scalar;
mod utils;

pub use expression::to_df_expr;
pub use predicate::to_df_predicate_expr;
pub use scalar::to_df_scalar;

/// Executes kernel declarative plans on DataFusion.
///
/// Holds two handles, each owning a distinct part of the work:
/// - `session_ctx` -- *plan it, then run it*: DataFusion's `SessionContext` is the front door to
///   the query engine. It holds the session-scoped state needed to turn a query into something
///   runnable: configuration, registered tables/catalogs and functions, the logical/physical
///   optimizer rules, and a handle to the shared runtime environment (memory pool, object-store
///   registry). We use it to compile and optimize a kernel plan into a DataFusion `LogicalPlan`,
///   then lower it to a physical `ExecutionPlan`. It is heavyweight and meant to be long-lived and
///   shared. At execution time we derive a fresh per-run `TaskContext` from it via
///   `session_ctx.task_ctx()` and pass that to `ExecutionPlan::execute`.
/// - `storage_handler` -- *fetch the bytes the query engine can't*: a kernel [`StorageHandler`] for
///   the storage I/O DataFusion cannot do itself (deletion-vector resolution, footer reads,
///   listing). This is the file-system subset of a kernel [`Engine`](delta_kernel::Engine) -- the
///   executor needs nothing else from the engine, so it holds only this.
pub struct DataFusionExecutor {
    session_ctx: SessionContext,
    storage_handler: Arc<dyn StorageHandler>,
}

impl DataFusionExecutor {
    pub fn new(storage_handler: Arc<dyn StorageHandler>) -> Self {
        Self {
            session_ctx: SessionContext::new(),
            storage_handler,
        }
    }
}

mod evaluation_frames;

mod first_value_allocation;

mod schema_conversion_allocation;

mod execution_aux;

mod evaluation_page;

mod case_allocation;

mod planning_allocation;

mod function_allocation;

mod file_pipeline_allocation;

mod evaluation_admission;
mod evaluation_work;
mod json_host;
pub use json_host::{JsonHostAdmissionError, JsonTaskHost};

#[cfg(any(test, feature = "qualification-fixture"))]
pub mod qualification_fixture;

#[cfg(test)]
mod json_host_tests;

/// Allocation-free bound for the existing Kernel-to-Arrow schema conversion.
/// The caller must reserve this complete owner peak before invoking conversion.
pub fn arrow_schema_owner_bytes(
    schema: &delta_kernel::schema::StructType,
    limits: delta_kernel::tasks::TaskLimits,
) -> Result<usize, delta_kernel::tasks::OperationFailure> {
    schema_conversion_allocation::peak(schema, limits)
}
