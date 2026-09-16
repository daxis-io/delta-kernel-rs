//! Validate the original Kernel result contract before rebinding the nullable
//! internal DataFusion representation. Validation borrows arrays and allocates
//! nothing. Rebinding requires the host's already-admitted array/container envelope.

use datafusion::arrow::array::{Array, RecordBatch, StructArray};
use datafusion::arrow::datatypes::{DataType, Field, SchemaRef};
use delta_kernel::tasks::OperationFailure;

/// No fingerprint/schema claim may precede this check. Parent validity controls
/// required-child validity; an inferred nullable field alone is not a mismatch.
pub(crate) fn validate(batch: &RecordBatch, expected: &SchemaRef) -> Result<(), OperationFailure> {
    let actual = batch.schema();
    if actual.metadata() != expected.metadata() || actual.fields().len() != expected.fields().len()
    {
        return Err(OperationFailure::malformed_response());
    }
    for ((actual, expected), array) in actual
        .fields()
        .iter()
        .zip(expected.fields())
        .zip(batch.columns())
    {
        validate_shape(actual, expected, array.as_ref(), 0)?;
        for row in 0..batch.num_rows() {
            validate_row(array.as_ref(), expected, row)?;
        }
    }
    Ok(())
}

fn validate_shape(
    actual: &Field,
    expected: &Field,
    array: &dyn Array,
    depth: usize,
) -> Result<(), OperationFailure> {
    if depth >= 64 || actual.name() != expected.name() || actual.metadata() != expected.metadata() {
        return Err(OperationFailure::malformed_response());
    }
    match (actual.data_type(), expected.data_type()) {
        (DataType::Struct(actual), DataType::Struct(expected)) => {
            let array = array
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(OperationFailure::malformed_response)?;
            if actual.len() != expected.len() || array.num_columns() != expected.len() {
                return Err(OperationFailure::malformed_response());
            }
            for ((actual, expected), child) in actual.iter().zip(expected).zip(array.columns()) {
                validate_shape(actual, expected, child.as_ref(), depth + 1)?;
            }
        }
        (actual, expected) if actual == expected => {}
        _ => return Err(OperationFailure::malformed_response()),
    }
    Ok(())
}

fn validate_row(array: &dyn Array, field: &Field, row: usize) -> Result<(), OperationFailure> {
    if array.is_null(row) {
        return if field.is_nullable() {
            Ok(())
        } else {
            Err(OperationFailure::malformed_response())
        };
    }
    if let DataType::Struct(fields) = field.data_type() {
        let array = array
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(OperationFailure::malformed_response)?;
        for (child, field) in array.columns().iter().zip(fields) {
            validate_row(child.as_ref(), field, row)?;
        }
    }
    Ok(())
}

/// Existing Kernel null-mask normalization followed by Arrow's checked struct
/// cast preserves values, names/order and the original nested field nullability.
/// The host must admit normalization/cast owners before calling this function.
pub(crate) fn rebind(
    batch: RecordBatch,
    expected: SchemaRef,
) -> Result<RecordBatch, OperationFailure> {
    validate(&batch, &expected)?;
    let normalized =
        delta_kernel::engine::arrow_data::fix_nested_null_masks(StructArray::from(batch));
    let cast =
        datafusion::arrow::compute::cast(&normalized, &DataType::Struct(expected.fields().clone()))
            .map_err(|_| OperationFailure::malformed_response())?;
    let cast = cast
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(OperationFailure::malformed_response)?;
    RecordBatch::try_new(expected, cast.columns().to_vec())
        .map_err(|_| OperationFailure::malformed_response())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::array::StringArray;
    use datafusion::arrow::buffer::NullBuffer;
    use datafusion::arrow::datatypes::Schema;

    use super::*;

    fn fixture(parent_valid: bool) -> (RecordBatch, SchemaRef) {
        let child = Arc::new(Field::new("required", DataType::Utf8, true));
        let nested = StructArray::new(
            vec![child].into(),
            vec![Arc::new(StringArray::from(vec![None::<&str>]))],
            Some(NullBuffer::from(vec![parent_valid])),
        );
        let actual = Arc::new(Schema::new(vec![Field::new(
            "parent",
            nested.data_type().clone(),
            true,
        )]));
        let expected = Arc::new(Schema::new(vec![Field::new(
            "parent",
            DataType::Struct(vec![Arc::new(Field::new("required", DataType::Utf8, false))].into()),
            true,
        )]));
        (
            RecordBatch::try_new(actual, vec![Arc::new(nested)]).unwrap(),
            expected,
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn original_schema_validation_allocates_nothing_on_success_and_failure() {
        use crate::json_arrays::tests::observe_allocations;
        for parent_valid in [false, true] {
            let (batch, expected) = fixture(parent_valid);
            let (result, allocations) = observe_allocations(|| validate(&batch, &expected));
            assert_eq!(allocations, 0);
            assert_eq!(result.is_ok(), !parent_valid);
            let wrong = Arc::new(Schema::new(vec![Field::new("wrong", DataType::Utf8, true)]));
            let (result, allocations) = observe_allocations(|| validate(&batch, &wrong));
            assert_eq!(allocations, 0);
            assert!(result.is_err());
        }
    }

    #[test]
    fn null_parent_allows_null_child_and_rebinds_original_schema() {
        let (batch, expected) = fixture(false);
        validate(&batch, &expected).unwrap();
        let batch = rebind(batch, expected.clone()).unwrap();
        assert_eq!(batch.schema(), expected);
        assert!(batch.column(0).is_null(0));
    }

    #[test]
    fn valid_parent_rejects_null_required_child_before_rebinding() {
        let (batch, expected) = fixture(true);
        assert!(validate(&batch, &expected).is_err());
        assert!(rebind(batch, expected).is_err());
    }

    #[test]
    fn exact_names_order_and_types_are_checked_even_under_null_parent() {
        let (batch, _) = fixture(false);
        for field in [
            Field::new("different", DataType::Utf8, true),
            Field::new("parent", DataType::Int64, true),
        ] {
            assert!(validate(&batch, &Arc::new(Schema::new(vec![field]))).is_err());
        }
    }
}
