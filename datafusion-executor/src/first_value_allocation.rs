//! Value-state component for the closed producers' ordered FIRST_VALUE.
//! Keys, operator/plan headers, evaluated arguments, and retained execution
//! pages are separate owners; this component must be charged for BOTH stages.

use std::alloc::Layout;
use std::mem::size_of;

use datafusion::arrow::array::{ArrayRef, BooleanArray, BooleanBufferBuilder, Int64Array};
use datafusion::arrow::buffer::{NullBuffer, ScalarBuffer};
use datafusion::arrow::compute::{SortColumn, SortOptions};
use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::common::ScalarValue;
use datafusion::physical_expr_common::sort_expr::{LexOrdering, PhysicalSortExpr};
use delta_kernel::tasks::{OperationFailure, Resource, ResourceExhausted, TaskLimits};

use crate::json_arrays::{copy_type_owner_peak, vec_peak};

/// One ordered aggregate's state, bounded by all source rows/bytes, including
/// grouped and global implementations. `schema` contains its value subtree;
/// using the whole closed input schema is also conservative. Ordering is one
/// Int64 version and null flags are Boolean: no variable-size ordering scalar.
pub(crate) fn state_peak(
    schema: &Schema,
    rows: usize,
    bytes: usize,
    limits: TaskLimits,
) -> Result<usize, OperationFailure> {
    state_type_peak(
        &DataType::Struct(schema.fields().clone()),
        rows,
        bytes,
        limits,
    )
}

/// One borrowed value subtree. For the global PM accumulator pass1 for rows:
/// it stores one winner and emits one scalar, regardless of input row count.
/// For grouped scan pass the full admitted row bound (groups cannot exceed it).
pub(crate) fn state_type_peak(
    value_type: &DataType,
    rows: usize,
    bytes: usize,
    limits: TaskLimits,
) -> Result<usize, OperationFailure> {
    let overflow = || ResourceExhausted {
        resource: Resource::MetadataAllocatedBytes,
        limit: limits.limit(Resource::MetadataAllocatedBytes),
        observed: usize::MAX,
    };
    let rows = rows.max(1); // global empty-input aggregate still has a null state
    let scalar = size_of::<ScalarValue>();
    let word = size_of::<usize>();
    let one = copy_type_owner_peak(value_type, 1, bytes, 1, limits)?;
    let emitted = copy_type_owner_peak(value_type, rows, bytes, rows, limits)?;
    // GenericValueState winner per group; the old winner remains while a new
    // extracted candidate is compacted. Global state()/evaluate() may clone a
    // string scalar. Null default and its returned clone are independent too.
    let values = one
        .checked_mul(rows.checked_add(3).ok_or_else(overflow)?)
        .and_then(|n| n.checked_add(emitted))
        .ok_or_else(overflow)?;
    let backing = (|| -> Option<usize> {
        let mut total = 0usize;
        // GenericValueState vals and EmitTo::First taken/remaining Vec owners.
        total =
            total.checked_add(vec_peak(rows, size_of::<Option<ScalarValue>>())?.checked_mul(2)?)?;
        // take() collects values into a new ScalarValue Vec for iter_to_array.
        total = total.checked_add(vec_peak(rows, scalar)?)?;
        // orderings outer Vec plus taken split and transposed ordering_cols.
        total =
            total.checked_add(vec_peak(rows, size_of::<Vec<ScalarValue>>())?.checked_mul(2)?)?;
        // Each group's resize clone owns a one-element ordering Vec. Also
        // default_orderings, resize's temporary clone, update ordering_buf.
        total = total.checked_add(vec_peak(1, scalar)?.checked_mul(rows.checked_add(3)?)?)?;
        total = total.checked_add(vec_peak(1, size_of::<Vec<ScalarValue>>())?)?;
        total = total.checked_add(vec_peak(rows, scalar)?)?;
        // Global get_row_at_idx candidate and retained ordering Vec capacity2;
        // state() creates a three-value result and may grow from capacity1.
        total = total.checked_add(vec_peak(2, scalar)?.checked_mul(2)?)?;
        total = total.checked_add(vec_peak(3, scalar)?)?;
        // Group result capacity is remaining_groups+2, not ordering_arity+2.
        total = total.checked_add(vec_peak(rows.checked_add(2)?, size_of::<ArrayRef>())?)?;
        // Extreme indices and split.
        total = total.checked_add(vec_peak(rows, word)?.checked_mul(2)?)?;
        total = total.checked_add(vec_peak(rows, size_of::<(usize, usize)>())?)?;
        // is_sets, extreme validity, emitted flag backing and rebuilt remaining
        // flags can coexist. BooleanBufferBuilder uses aligned MutableBuffer.
        let flags = rows.checked_add(7)?.checked_div(8)?.max(64);
        total = total.checked_add(vec_peak(flags, 1)?.checked_mul(4)?)?;
        // One Int64 ordering column materialized by state().
        total = total.checked_add(vec_peak(rows, size_of::<i64>())?)?;
        total =
            total.checked_add(size_of::<BooleanArray>() + size_of::<Int64Array>() + 4 * word)?;
        total = total.checked_add(vec_peak(1, size_of::<PhysicalSortExpr>())?)?;
        total = total.checked_add(vec_peak(1, size_of::<SortOptions>())?)?;
        total = total.checked_add(vec_peak(1, size_of::<DataType>())?)?;
        // Comparator: SortColumn Vec, Vec<DynComparator>, one boxed closure
        // capturing two Int64 ScalarBuffers and (nullable branch) NullBuffers
        // plus two Ordering values. No string comparator or nested comparator.
        total = total.checked_add(vec_peak(1, size_of::<SortColumn>())?)?;
        total = total.checked_add(vec_peak(1, 2 * word)?)?;
        total = total.checked_add(
            2 * size_of::<ScalarBuffer<i64>>() + 2 * size_of::<NullBuffer>() + 2 * word,
        )?;
        Some(total)
    })()
    .ok_or_else(overflow)?;
    // Source fields of FirstLastGroupsAccumulator<GenericValueState>.
    // Use each field's compiler alignment, including nested state fields.
    let fields = [
        Layout::new::<Vec<Option<ScalarValue>>>(),
        Layout::new::<DataType>(),
        Layout::new::<usize>(),
        Layout::new::<Vec<Vec<ScalarValue>>>(),
        Layout::new::<BooleanBufferBuilder>(),
        Layout::new::<usize>(),
        Layout::new::<Vec<usize>>(),
        Layout::new::<BooleanBufferBuilder>(),
        Layout::new::<LexOrdering>(),
        Layout::new::<bool>(),
        Layout::new::<Vec<SortOptions>>(),
        Layout::new::<bool>(),
        Layout::new::<Vec<ScalarValue>>(),
    ];
    let grouped_header = crate::evaluation_frames::aggregate(&fields).ok_or_else(overflow)?;
    let header = grouped_header.max(size_of::<
        datafusion::functions_aggregate::first_last::FirstValueAccumulator,
    >());
    let peak = values
        .checked_add(backing)
        .and_then(|n| n.checked_add(header))
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

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::Field;
    use delta_kernel::tasks::FailureKind;

    use super::*;
    use crate::json_arrays::tests::observe_allocations;
    #[test]
    fn value_state_bounds_are_checked_before_allocating() {
        let schema = Schema::new(vec![Field::new("value", DataType::Utf8, true)]);
        let limits = TaskLimits::qualification();
        let (peak, count) = observe_allocations(|| state_peak(&schema, 6, 2048, limits));
        assert_eq!(count, 0);
        let peak = peak.unwrap();
        for (limit, success) in [(peak, true), (peak - 1, false)] {
            let (result, count) = observe_allocations(|| {
                state_peak(
                    &schema,
                    6,
                    2048,
                    limits.with_limit(Resource::MetadataAllocatedBytes, limit),
                )
            });
            assert_eq!(count, 0);
            if success {
                assert_eq!(result.unwrap(), peak);
            } else {
                assert!(
                    matches!(result.unwrap_err().kind(), FailureKind::ResourceExhausted(e)
                if e.resource == Resource::MetadataAllocatedBytes)
                );
            }
        }
        for (rows, bytes) in [(usize::MAX, 1), (1, usize::MAX)] {
            let (result, count) = observe_allocations(|| state_peak(&schema, rows, bytes, limits));
            assert_eq!(count, 0);
            assert!(result.is_err());
        }
    }
}
