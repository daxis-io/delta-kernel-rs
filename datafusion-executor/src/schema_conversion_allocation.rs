//! Initial conversion of borrowed closed Kernel system schemas into Arrow.
//! All field metadata must be empty, before the converter's serde/format paths.
use crate::json_arrays::vec_peak;
use datafusion::arrow::datatypes::{Field, FieldRef, Schema};
use delta_kernel::schema::{DataType, PrimitiveType, StructType};
use delta_kernel::tasks::{OperationFailure, Resource, ResourceExhausted, TaskLimits};
use std::mem::size_of;

pub(crate) fn peak(schema: &StructType, limits: TaskLimits) -> Result<usize, OperationFailure> {
    let mut bound = Bound {
        bytes: crate::json_arrays::shared_owner_bytes::<Schema>(),
        limits,
    };
    bound.schema(schema, 0)?;
    if bound.bytes > limits.limit(Resource::MetadataAllocatedBytes) {
        return Err(ResourceExhausted {
            resource: Resource::MetadataAllocatedBytes,
            limit: limits.limit(Resource::MetadataAllocatedBytes),
            observed: bound.bytes,
        }
        .into());
    }
    Ok(bound.bytes)
}
struct Bound {
    bytes: usize,
    limits: TaskLimits,
}
impl Bound {
    fn add(&mut self, amount: Option<usize>) -> Result<(), OperationFailure> {
        self.bytes = amount
            .and_then(|n| self.bytes.checked_add(n))
            .ok_or_else(|| ResourceExhausted {
                resource: Resource::MetadataAllocatedBytes,
                limit: self.limits.limit(Resource::MetadataAllocatedBytes),
                observed: usize::MAX,
            })?;
        Ok(())
    }
    fn depth(&self, depth: usize) -> Result<(), OperationFailure> {
        if depth > self.limits.limit(Resource::SchemaDepth) {
            return Err(ResourceExhausted {
                resource: Resource::SchemaDepth,
                limit: self.limits.limit(Resource::SchemaDepth),
                observed: depth,
            }
            .into());
        }
        Ok(())
    }
    fn fields(&mut self, count: usize) -> Result<(), OperationFailure> {
        // try_kernel_struct_to_arrow_fields Vec<Field>, Fields::from Vec<Arc>
        // and Arc<[FieldRef]> relocation. Field::new name backing below;
        // empty HashMaps have no buckets. Each field obtains its own Arc.
        self.add(vec_peak(count, size_of::<Field>()))?;
        self.add(vec_peak(count, size_of::<FieldRef>()))?;
        self.add(count.checked_mul(crate::json_arrays::shared_owner_bytes::<Field>()))?;
        self.add(Some(2 * size_of::<usize>()))
    }
    fn schema(&mut self, schema: &StructType, depth: usize) -> Result<(), OperationFailure> {
        self.depth(depth)?;
        self.fields(schema.num_fields())?;
        for field in schema.fields() {
            if !field.metadata.is_empty() {
                return Err(OperationFailure::malformed_response());
            }
            self.add(Some(field.name.len()))?;
            self.ty(&field.data_type, field.name.len(), depth + 1)?;
        }
        Ok(())
    }
    fn ty(&mut self, ty: &DataType, path: usize, depth: usize) -> Result<(), OperationFailure> {
        self.depth(depth)?;
        match ty {
            DataType::Primitive(
                PrimitiveType::String
                | PrimitiveType::Long
                | PrimitiveType::Integer
                | PrimitiveType::Boolean
                | PrimitiveType::Void,
            ) => Ok(()),
            DataType::Struct(s) => self.schema(s, depth),
            DataType::Array(a) => {
                let next = path.checked_add(".element".len());
                self.add(next.and_then(|n| vec_peak(n, 1)))?;
                self.fields(1)?;
                self.add(Some("element".len()))?;
                self.ty(a.element_type(), next.unwrap(), depth + 1)
            }
            DataType::Map(m) => {
                let key = path.checked_add(".key".len());
                let value = path.checked_add(".value".len());
                self.add(key.and_then(|n| vec_peak(n, 1)))?;
                self.add(value.and_then(|n| vec_peak(n, 1)))?;
                // Synthesized key/value Fields and their parent entries Field.
                self.fields(2)?;
                self.fields(1)?;
                self.add(Some("key".len() + "value".len() + "key_value".len()))?;
                self.ty(m.key_type(), key.unwrap(), depth + 1)?;
                self.ty(m.value_type(), value.unwrap(), depth + 1)
            }
            _ => Err(OperationFailure::malformed_response()),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::json_arrays::tests::observe_allocations;
    use delta_kernel::engine::arrow_conversion::TryIntoArrow;
    use delta_kernel::schema::{ArrayType, MapType, StructField};
    #[test]
    fn system_conversion_is_admitted_before_names_paths_or_field_arcs() {
        let schema = StructType::new_unchecked([StructField::nullable(
            "config",
            MapType::new(DataType::STRING, ArrayType::new(DataType::LONG, true), true),
        )]);
        let limits = TaskLimits::qualification();
        let (result, count) = observe_allocations(|| peak(&schema, limits));
        assert_eq!(count, 0);
        let bytes = result.unwrap();
        for (limit, pass) in [(bytes, true), (bytes - 1, false)] {
            let (result, count) = observe_allocations(|| {
                peak(
                    &schema,
                    limits.with_limit(Resource::MetadataAllocatedBytes, limit),
                )
            });
            assert_eq!(count, 0);
            assert_eq!(result.is_ok(), pass);
        }
        let converted: Schema = (&schema).try_into_arrow().unwrap();
        assert_eq!(converted.fields().len(), 1);
        assert_eq!(converted.field(0).name(), "config");
        assert!(converted.field(0).metadata().is_empty());
    }
}
