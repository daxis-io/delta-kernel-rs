//! Staged admission for the concrete task host. All arrays remain unconstructed
//! until the complete execution envelope has been checked and reserved.
use std::mem::size_of;
use std::sync::Arc;

use datafusion::arrow::datatypes::{FieldRef, Schema, SchemaRef};
use delta_kernel::engine::arrow_conversion::TryIntoArrow;
use delta_kernel::plans::ir::nodes::Operator;
use delta_kernel::tasks::{
    AccountedEngineData, AdmittedPlan, OperationFailure, Resource, ResourceExhausted, TaskLimits,
};

use crate::closed_plan_facts::ClosedPlanFacts;
use crate::evaluation_page::PageEnvelope;
use crate::log_input::LogInput;

pub(crate) struct PreparedEvaluation {
    pub expected: SchemaRef,
    pub page: PageEnvelope,
    pub peak: usize,
    pub input_rows: usize,
}

/// The initial stage needs only borrowed Kernel types. It covers conversion
/// containers before any Arrow Schema/Field/Fields allocation is attempted.
pub(crate) fn schema_peak(
    plan: &AdmittedPlan,
    limits: TaskLimits,
) -> Result<usize, OperationFailure> {
    let mut bytes = size_of::<Schema>() + size_of::<FieldRef>();
    for node in &plan.plan().nodes {
        let schema = match &node.op {
            Operator::ScanJson(op) => Some(&op.schema),
            Operator::Project(op) => Some(&op.schema),
            Operator::Aggregate(op) => Some(&op.schema),
            _ => None,
        };
        if let Some(schema) = schema {
            bytes = add(
                bytes,
                crate::schema_conversion_allocation::peak(schema, limits)?,
                limits,
            )?;
        }
    }
    check(bytes, limits)?;
    Ok(bytes)
}

/// Called only under the initial schema/input reservation. Schema conversion
/// is source-preflighted above; this stage constructs no JSON decoder, physical
/// plan, stream, row converter, accumulator or result array.
pub(crate) fn prepare(
    plan: &AdmittedPlan,
    input: &LogInput,
    facts: &ClosedPlanFacts,
    conversions: usize,
    limits: TaskLimits,
) -> Result<PreparedEvaluation, OperationFailure> {
    let bytes = input
        .reads
        .iter()
        .try_fold(0usize, |n, r| n.checked_add(r.bytes.len()))
        .ok_or_else(|| overflow(limits))?;
    let rows = input.framing.records;
    let mut decoder = 0;
    let mut states = 0;
    let mut keys = 0;
    let mut max_headers = 0;
    let mut max_schema = 0;
    let mut terminal_owners = 0;
    let mut expected = None;
    let state_rows = if facts.grouping_keys == 0 { 1 } else { rows };
    for (index, node) in plan.plan().nodes.iter().enumerate() {
        let schema = match &node.op {
            Operator::ScanJson(op) => Some(&op.schema),
            Operator::Project(op) => Some(&op.schema),
            Operator::Aggregate(op) => Some(&op.schema),
            _ => None,
        };
        let Some(schema) = schema else {
            continue;
        };
        let owners = crate::schema_conversion_allocation::peak(schema, limits)?;
        max_schema = max_schema.max(owners);
        let arrow: Schema = schema
            .as_ref()
            .try_into_arrow()
            .map_err(super::json_host::engine_error)?;
        max_headers = max_headers.max(crate::json_arrays::output_owner_peak(&arrow, 1, 0, limits)?);
        match &node.op {
            Operator::ScanJson(_) => {
                decoder =
                    crate::json_arrays::DecoderEnvelope::preflight(&arrow, input.framing, limits)?
                        .peak
            }
            Operator::Aggregate(aggregate) => {
                for field in arrow.fields().iter().skip(aggregate.group_by.len()) {
                    let value = crate::first_value_allocation::state_type_peak(
                        field.data_type(),
                        state_rows,
                        bytes,
                        limits,
                    )?;
                    states = add(
                        states,
                        value.checked_mul(2).ok_or_else(|| overflow(limits))?,
                        limits,
                    )?;
                }
                if !aggregate.group_by.is_empty() {
                    let key = Schema::new(vec![Arc::clone(&arrow.fields()[0])]);
                    keys =
                        crate::row_key_allocation::grouping_owner_peak(&key, rows, bytes, limits)?
                            .checked_mul(2)
                            .ok_or_else(|| overflow(limits))?;
                }
            }
            _ => {}
        }
        if index + 1 == plan.plan().nodes.len() {
            terminal_owners = owners;
            expected = Some(Arc::new(arrow));
        }
    }
    let expected = expected.ok_or_else(OperationFailure::malformed_response)?;
    let page = PageEnvelope::preflight(&expected, state_rows, bytes, terminal_owners, limits)?;
    let planning = crate::planning_allocation::preflight(facts, conversions, limits)?;
    let cases = crate::case_allocation::peak(facts, max_headers, max_schema, bytes, limits)?;
    let functions = crate::function_allocation::peak(facts, max_headers, limits)?;
    let store = crate::log_store::AdmittedLogStore::owner_peak(input, facts.files, limits)?;
    let common = sum(
        &[
            store,
            crate::metadata_session::bootstrap_peak(limits)?,
            crate::evaluation_frames::peak(limits)?,
            crate::execution_aux::peak(limits)?,
            crate::file_pipeline_allocation::peak(facts, limits)?,
        ],
        limits,
    )?;
    // Kernel can retain every emitted one-row scan batch. Charge each page's
    // full aggregate backing, including shared buffers and the lease, until
    // the task transfers/drops its output. There are no more groups than rows.
    let page_owners = page
        .retained
        .checked_add(size_of::<Box<dyn AccountedEngineData>>())
        .and_then(|n| n.checked_mul(state_rows))
        .ok_or_else(|| overflow(limits))?;
    let execution = sum(
        &[
            common,
            planning.retained,
            decoder,
            states,
            keys,
            cases,
            functions,
            page.rebind_peak,
            page_owners,
        ],
        limits,
    )?;
    // Lowered logical plans/conversion scratch are dropped before execution.
    // They coexist with the physical plan during construction, not with its
    // decoder/aggregate buffers. The driver still owns the admitted request.
    let construction = sum(&[common, conversions, planning.construction], limits)?;
    let peak = execution.max(construction);
    check(peak, limits)?;
    Ok(PreparedEvaluation {
        expected,
        page,
        peak,
        input_rows: rows,
    })
}

pub(crate) fn check(bytes: usize, limits: TaskLimits) -> Result<(), OperationFailure> {
    for resource in [Resource::TaskStateBytes, Resource::MetadataAllocatedBytes] {
        if bytes > limits.limit(resource) {
            return Err(ResourceExhausted {
                resource,
                limit: limits.limit(resource),
                observed: bytes,
            }
            .into());
        }
    }
    Ok(())
}
pub(crate) fn add(a: usize, b: usize, limits: TaskLimits) -> Result<usize, OperationFailure> {
    a.checked_add(b).ok_or_else(|| overflow(limits))
}
fn sum(values: &[usize], limits: TaskLimits) -> Result<usize, OperationFailure> {
    values.iter().try_fold(0, |n, v| add(n, *v, limits))
}
fn overflow(limits: TaskLimits) -> OperationFailure {
    ResourceExhausted {
        resource: Resource::MetadataAllocatedBytes,
        limit: limits.limit(Resource::MetadataAllocatedBytes),
        observed: usize::MAX,
    }
    .into()
}
