//! CASE buffer owners reached by the typed closed metadata expressions.
//! Function argument/return-field and physical-plan owners are separate.

use crate::closed_plan_facts::ClosedPlanFacts;
use crate::json_arrays::{merge_type_owner_peak, output_type_owner_peak};
use datafusion::arrow::datatypes::DataType;
use delta_kernel::tasks::{OperationFailure, Resource, ResourceExhausted, TaskLimits};

pub(crate) fn peak(
    facts: &ClosedPlanFacts,
    one_row_header_tree: usize,
    schema_owners: usize,
    log_bytes: usize,
    limits: TaskLimits,
) -> Result<usize, OperationFailure> {
    let overflow = || ResourceExhausted {
        resource: Resource::MetadataAllocatedBytes,
        limit: limits.limit(Resource::MetadataAllocatedBytes),
        observed: usize::MAX,
    };
    let coalesces = facts
        .coalesce_utf8
        .checked_add(facts.coalesce_int32)
        .ok_or_else(overflow)?;
    if coalesces > facts.cases || facts.case_branches != facts.cases {
        return Err(OperationFailure::malformed_response());
    }
    if facts.cases == 0 {
        return Ok(0);
    }
    // ClosedPlanFacts resolves both Coalesce operands from borrowed Kernel
    // input schemas. All are Utf8 or Int32 columns; there is no struct/map
    // merge. The existing builtin rewrite yields exactly one WHEN and ELSE.
    let strings = merge_type_owner_peak(&DataType::Utf8, 1, log_bytes, limits)?;
    let integers = merge_type_owner_peak(&DataType::Int32, 1, 0, limits)?;
    let merges = strings
        .checked_mul(facts.coalesce_utf8)
        .and_then(|n| n.checked_add(integers.checked_mul(facts.coalesce_int32)?))
        .ok_or_else(overflow)?;
    // The remaining guards have no ELSE. scatter(mask, then_array) at len<=1
    // selects its slice/all-null fast path; no generic scatter_fallback runs.
    // Guards over structs may allocate a complete all-null header/bitmap tree.
    let guards = one_row_header_tree
        .checked_mul(facts.cases - coalesces)
        .ok_or_else(overflow)?;
    // At most one projected input and two filtered branch batches per CASE.
    // Arrow FilterPredicate uses All/None at len<=1, constructing only sliced
    // or empty array headers. RecordBatch::project constructs a subset Schema;
    // include a full admitted system schema for each of these three sites.
    // Sum all CASE sites to cover nested evaluation and retained child results.
    let branches = one_row_header_tree
        .checked_add(schema_owners)
        .and_then(|n| n.checked_mul(3))
        .and_then(|n| n.checked_mul(facts.cases))
        .ok_or_else(overflow)?;
    // WHEN result, null-normalized predicate, and NOT/ELSE predicate are three
    // simultaneous one-bit BooleanArray trees. Generic mixed-row selection is
    // unreachable for the admitted batch-size1 pipeline, including empty input.
    let boolean = output_type_owner_peak(&DataType::Boolean, 1, 0, limits)?;
    let masks = boolean
        .checked_mul(3)
        .and_then(|n| n.checked_mul(facts.cases))
        .ok_or_else(overflow)?;
    let bytes = merges
        .checked_add(guards)
        .and_then(|n| n.checked_add(branches))
        .and_then(|n| n.checked_add(masks))
        .ok_or_else(overflow)?;
    if bytes > limits.limit(Resource::MetadataAllocatedBytes) {
        return Err(ResourceExhausted {
            resource: Resource::MetadataAllocatedBytes,
            limit: limits.limit(Resource::MetadataAllocatedBytes),
            observed: bytes,
        }
        .into());
    }
    Ok(bytes)
}
