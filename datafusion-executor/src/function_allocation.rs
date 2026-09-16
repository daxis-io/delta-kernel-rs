//! Per-pull owners of the fixed named_struct/get_field/CASE expression trees.
use std::mem::size_of;

use datafusion::arrow::array::ArrayRef;
use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::logical_expr::ColumnarValue;
use delta_kernel::tasks::{OperationFailure, Resource, ResourceExhausted, TaskLimits};

use crate::closed_plan_facts::ClosedPlanFacts;
use crate::json_arrays::{output_type_owner_peak, vec_peak};

pub(crate) fn peak(
    f: &ClosedPlanFacts,
    one_row_headers: usize,
    limits: TaskLimits,
) -> Result<usize, OperationFailure> {
    let overflow = || ResourceExhausted {
        resource: Resource::MetadataAllocatedBytes,
        limit: limits.limit(Resource::MetadataAllocatedBytes),
        observed: usize::MAX,
    };
    let bytes = (|| -> Option<usize> {
        // Two arguments per named field (name/value), two per get_field,
        // and four generated aggregate operands (value/filter/order clones).
        let args = f
            .fields
            .checked_mul(2)?
            .checked_add(f.path_components.checked_mul(2)?)?
            .checked_add(f.aggregates.checked_mul(4)?)?;
        let functions = f.fields.checked_add(f.path_components)?;
        let partitioned_vec =
            |width| vec_peak(args, width)?.checked_add(functions.checked_mul(vec_peak(0, width)?)?);
        // ScalarFunctionExpr::evaluate: argument values + return fields;
        // named_struct::invoke: cloned value Vec + values_to_arrays output.
        let containers = partitioned_vec(size_of::<ColumnarValue>())?
            .checked_mul(2)?
            .checked_add(partitioned_vec(size_of::<FieldRef>())?)?
            .checked_add(partitioned_vec(size_of::<ArrayRef>())?)?;
        // Literal::evaluate clones its String; get_field then clones the
        // ScalarValue and extract_single_field makes a third String copy.
        // Access paths in the closed producers traverse Structs, never Map
        // lookup/dictionary reconstruction. Named-struct names are scalar-only.
        let strings = f.name_bytes.checked_mul(3)?;
        // The only uncached return_field inside a reached function's args is
        // CASE. Its descendants are columns/get_field/named_struct/boolean
        // predicates; outer projection casts are not descendants of CASE.
        // Enumerate formatter syntax and duplicated field/path names. Physical
        // column @index uses at most twenty decimal digits on either target.
        let display = f
            .name_bytes
            .checked_mul(4)?
            .checked_add(
                f.path_components
                    .checked_mul("get_field(, )Utf8(\"\")@18446744073709551615".len())?,
            )?
            .checked_add(f.fields.checked_mul("named_struct()Utf8(\"\"), ".len())?)?
            .checked_add(f.cases.checked_mul("CASE WHEN  THEN  ELSE  END".len())?)?
            .checked_add(
                f.expressions
                    .checked_mul("NOT  IS NOT NULL AND OR ".len())?,
            )?;
        let fields = f.cases.checked_mul(
            crate::json_arrays::shared_owner_bytes::<Field>() + vec_peak(display, 1)?,
        )?;
        // One current batch at four projections, two filters, the source
        // schema adapter and BatchSplitStream. Their payloads share admitted
        // decoder/aggregate backing; only sliced/cast/struct/null-mask trees
        // are additional. CASE's own branch batches have a separate envelope.
        let batches = one_row_headers.checked_mul(8)?;
        containers
            .checked_add(strings)?
            .checked_add(fields)?
            .checked_add(batches)
    })()
    .ok_or_else(overflow)?;
    // Binary/null predicate temporaries, including generated FIRST_VALUE
    // AND(IS NOT NULL(sentinel), IS NOT NULL(version)) per aggregate.
    let booleans = f
        .expressions
        .checked_add(f.aggregates.checked_mul(3).ok_or_else(overflow)?)
        .and_then(|n| n.checked_add(f.boolean_junctions))
        .ok_or_else(overflow)?;
    let boolean = output_type_owner_peak(&DataType::Boolean, 1, 0, limits)?;
    let bytes = boolean
        .checked_mul(booleans)
        .and_then(|n| n.checked_add(bytes))
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
