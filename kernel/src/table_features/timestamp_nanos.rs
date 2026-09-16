//! Validation for TIMESTAMP_NANOS feature support

use std::borrow::Cow;

use super::TableFeature;
use crate::schema::{PrimitiveType, Schema};
use crate::table_configuration::TableConfiguration;
use crate::transforms::{transform_output_type, SchemaTransform};
use crate::utils::require;
use crate::{DeltaResult, Error};

/// Validates that if a table schema contains TIMESTAMP_NANOS columns, the table must have the
/// TimestampNanos feature in both reader and writer features.
pub(crate) fn validate_timestamp_nanos_feature_support(tc: &TableConfiguration) -> DeltaResult<()> {
    let protocol = tc.protocol();
    if !protocol.has_table_feature(&TableFeature::TimestampNanos) {
        require!(
            !schema_contains_timestamp_nanos(&tc.logical_schema()),
            Error::unsupported(
                "Table contains TIMESTAMP_NANOS columns but does not have the required 'timestampNanos' feature in reader and writer features"
            )
        );
    }
    Ok(())
}

#[cfg(feature = "nanosecond-timestamps")]
/// Checks if any column in the schema (including nested structs, arrays, maps) uses
/// the TIMESTAMP_NANOS primitive type.
pub(crate) fn schema_contains_timestamp_nanos(schema: &Schema) -> bool {
    let mut uses_timestamp_nanos = UsesTimestampNanos(false);
    let _ = uses_timestamp_nanos.transform_struct(schema);
    uses_timestamp_nanos.0
}

struct UsesTimestampNanos(bool);

impl<'a> SchemaTransform<'a> for UsesTimestampNanos {
    transform_output_type!(|'a, T| Option<Cow<'a, T>>);

    fn transform_primitive(&mut self, ptype: &'a PrimitiveType) -> Option<Cow<'a, PrimitiveType>> {
        if *ptype == PrimitiveType::TimestampNanos {
            self.0 = true;
        }
        None
    }
}
