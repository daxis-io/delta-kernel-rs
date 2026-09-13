//! Primitive schema shape admission before Kernel's ordinary schema deserializer.
//!
//! This pass retains no strings, fields, metadata maps or serde_json::Value tree. It rejects
//! non-primitive field types and nonempty field metadata before the ordinary materializer can
//! enter their recursive allocation paths. Unknown schema properties retain serde's ignored-value
//! behavior. Kernel's existing deserializer still owns required-field and schema semantics.

use super::json_materialization::{check, engine, exhausted, preflight_schema_json};
use super::{OperationFailure, Resource, TaskLimits};
use serde::de::Error as _;
use serde::de::{self, DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::Deserializer;
use std::fmt;
use std::mem::size_of;

#[derive(Default, Debug)]
pub(super) struct PrimitiveSchemaShape {
    pub fields: usize,
    pub name_bytes: usize,
    pub type_bytes: usize,
    unsupported: bool,
    failure: Option<OperationFailure>,
}

impl PrimitiveSchemaShape {
    /// `admit` includes the simultaneous task owner before serde's scratch/error allocation.
    pub(super) fn inspect(
        json: &str,
        limits: &TaskLimits,
        admit: impl FnOnce(usize) -> Result<(), OperationFailure>,
    ) -> Result<Self, OperationFailure> {
        preflight_schema_json(json, limits)?;
        let scratch = Self::inspection_bytes(json.len())
            .ok_or_else(|| exhausted(Resource::MetadataAllocatedBytes, limits))?;
        check(Resource::MetadataAllocatedBytes, scratch, limits)?;
        admit(scratch)?;
        let mut shape = Self::default();
        let mut decoder = serde_json::Deserializer::from_str(json);
        let result = Root {
            shape: &mut shape,
            limits,
        }
        .deserialize(&mut decoder)
        .and_then(|()| decoder.end());
        if let Some(failure) = shape.failure {
            return Err(failure);
        }
        if shape.unsupported {
            return Err(engine(crate::Error::unsupported(
                "JSON operation tasks require primitive fields with empty metadata",
            )));
        }
        // Do not add an unrelated backtrace capture to this bounded shape inspection error.
        result.map_err(|error| engine(crate::Error::MalformedJson(error)))?;
        Ok(shape)
    }

    #[cfg(test)]
    fn admit_materialization(
        &self,
        json_bytes: usize,
        limits: &TaskLimits,
        admit: impl FnOnce(usize) -> Result<(), OperationFailure>,
    ) -> Result<(), OperationFailure> {
        let bytes = self
            .materialization_bytes(json_bytes)
            .ok_or_else(|| exhausted(Resource::MetadataAllocatedBytes, limits))?;
        check(Resource::MetadataAllocatedBytes, bytes, limits)?;
        admit(bytes)
    }

    pub(super) fn inspection_bytes(bytes: usize) -> Option<usize> {
        // serde_json 1.0.149/1.0.150 SliceRead owns one reusable unescape Vec;
        // ignore_value also uses it as a container stack for unknown properties.
        // UTF-8 unescaping and the ignored-value stack cannot grow
        // beyond encoded input bytes. RawVec capacity <= max(8, 2B), plus relocation overlap.
        let scratch = bytes.checked_mul(2)?.max(8).checked_mul(2)?;
        // serde invalid_type can Debug-format a supplied string (<=10 bytes per input byte).
        // Unexpected::Float can print a subnormal as fixed decimal: IEEE-754 exponent -324,
        // at most 17 significant digits, sign and decimal prefix. Line/column use <=20 digits.
        let diagnostic = bytes.checked_mul("\\u{10ffff}".len())?
            .checked_add(324 + 17 + 3)?
            .checked_add("invalid type: string , expected an object with primitive fields and empty field metadata at line  column ".len())?
            .checked_add(2 * 20)?;
        let error_string = diagnostic.checked_mul(2)?.max(8).checked_mul(2)?;
        // Private serde ErrorImpl: ErrorCode's largest payload is String, line and column,
        // plus enum discriminant/alignment. Include both serde and Kernel error owners.
        let errors = size_of::<String>()
            .checked_add(4 * size_of::<usize>())?
            .checked_add(size_of::<crate::Error>())?
            .checked_add(size_of::<OperationFailure>())?;
        scratch
            .checked_add(error_string)?
            .checked_add(errors)?
            .checked_add(size_of::<
                serde_json::Deserializer<serde_json::de::StrRead<'static>>,
            >())?
            .checked_add(size_of::<Self>())
    }

    /// Peak of the ordinary `StructType` serde materializer for the inspected
    /// primitive/empty-metadata class. This is not the later TableConfiguration
    /// or visitor envelope: their independently admitted owners remain live.
    pub(super) fn materialization_bytes(&self, json_bytes: usize) -> Option<usize> {
        use crate::schema::{StructField, StructType};
        let fields = self.fields;
        // StructTypeSerDeHelper's input Vec lives while try_new builds IndexMap.
        let input = vector_peak::<StructField>(fields)?;
        // indexmap 2.x Bucket<K,V> contains HashValue(usize), K, V; entries are
        // a Vec and the separate hashbrown index contains usize values.
        let entries = vector_peak::<(usize, String, StructField)>(fields)?;
        let indices = hash_peak::<usize>(fields)?;
        // Both the field and IndexMap key own the decoded name. Input field
        // Strings move into entries, so counting all input slots is conservative.
        let names = self.name_bytes.checked_mul(2)?;
        // Rust Unicode lowercase iterates at most three chars per scalar, each
        // encoded in at most four bytes. A nonempty scalar uses >=1 input byte.
        // Each String may grow from MIN_NON_ZERO_CAP(8); count every old/new
        // allocation, including one independent minimum per field.
        let lowercase_strings = self
            .name_bytes
            .checked_mul(3 * 4)?
            .checked_mul(4)?
            .checked_add(fields.checked_mul(2 * 8)?)?;
        let lowercase_set = hash_peak::<String>(fields)?;
        // DataType first owns Value::String, then clones that Value into
        // PrimitiveType::deserialize. Empty field metadata owns no heap entries.
        let primitive_strings = self.type_bytes.checked_mul(2)?;
        // Helper root type_name plus the final canonical "struct" String.
        let root_strings = json_bytes.checked_add("struct".len())?;
        // serde scratch and syntax/invalid-type errors share the inspected
        // bounded input. DataType's primitive error path has four simultaneous
        // formatted owners: primitive custom error, to_string, outer custom
        // error, and StructType custom conversion. Each includes a decoded
        // name/type (Debug expansion <=10B), a fixed message and line/column.
        let error_text = json_bytes.checked_mul("\\u{10ffff}".len())?
            .checked_add("Duplicate field name (case-insensitive): Unsupported Delta table type: Invalid decimal format (expected 2 parts):  at line  column ".len())?
            .checked_add(2 * 20)?;
        let errors = vector_peak::<u8>(error_text)?
            .checked_add(size_of::<String>() + 4 * size_of::<usize>())?
            .checked_mul(4)?;
        [
            input,
            entries,
            indices,
            names,
            lowercase_strings,
            lowercase_set,
            primitive_strings,
            root_strings,
            errors,
            Self::inspection_bytes(json_bytes)?,
            size_of::<StructType>(),
            size_of::<serde_json::Value>().checked_mul(2)?,
            size_of::<crate::Error>(),
            size_of::<super::OperationFailure>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    /// Additional configuration/snapshot construction envelope, including the
    /// initial ordinary schema decoder. Incoming Metadata/Protocol/log owners
    /// are accounted by the task and are not replaced by this reservation.
    pub(super) fn configuration_bytes(
        &self,
        metadata: &crate::actions::Metadata,
        root_bytes: usize,
    ) -> Option<usize> {
        use crate::schema::{StructField, StructType};
        use std::borrow::Cow;
        use std::collections::HashMap;
        let schema_owner = vector_peak::<(usize, String, StructField)>(self.fields)?
            .checked_add(hash_peak::<usize>(self.fields)?)?
            .checked_add(self.name_bytes.checked_mul(2)?)?
            .checked_add(size_of::<StructType>() + "struct".len() + 2 * size_of::<usize>())?;
        // Four retained schemas: initial logical plus physical and both
        // unpartitioned projections. The initial decoder is charged separately;
        // the other three use new_unchecked (no lowercase validation set).
        let clones = schema_owner.checked_mul(3)?;
        let initial = self
            .materialization_bytes(metadata.schema_string().len())?
            .checked_add(2 * size_of::<usize>())?; // logical Schema Arc
                                                   // MakePhysical has one root sibling frame and primitive path depth one.
                                                   // Mapping mode is None for the admitted reader protocol1, so its ID and
                                                   // sibling maps stay empty. map_owned_children_or_else owns Cow and owned
                                                   // field Vecs before new_unchecked moves fields into the physical index.
        let transform = vector_peak::<Cow<'static, StructField>>(self.fields)?
            .checked_add(vector_peak::<StructField>(self.fields)?)?
            .checked_add(vector_peak::<HashMap<&str, &str>>(1)?)?
            .checked_add(vector_peak::<&str>(1)?)?
            .checked_add(self.name_bytes)?;
        // ColumnDefaultCollector has no results with empty field metadata, but
        // still clones each field name into a path Vec (primitive depth one).
        let defaults = vector_peak::<String>(1)?.checked_add(self.name_bytes)?;
        let properties = property_owners(metadata.configuration())?;
        // Snapshot moves Metadata/Protocol and TableConfiguration in place.
        // The table-root String is cloned once; Snapshot's Arc has two atomics.
        let feature_error = vector_peak::<u8>(
            "Table contains TIMESTAMP_NTZ columns but does not have the required 'timestampNtz' feature in reader and writer features".len()
            + "Table contains TIMESTAMP_NANOS columns but does not have the required 'timestampNanos' feature in reader and writer features".len())?;
        [
            initial,
            clones,
            transform,
            defaults,
            properties,
            feature_error,
            root_bytes,
            size_of::<crate::snapshot::Snapshot>(),
            2 * size_of::<usize>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    fn add<E: de::Error>(
        &mut self,
        kind: StringKind,
        bytes: usize,
        limits: &TaskLimits,
    ) -> Result<(), E> {
        let target = match kind {
            StringKind::Name => &mut self.name_bytes,
            _ => &mut self.type_bytes,
        };
        match target.checked_add(bytes) {
            Some(value) => *target = value,
            None => {
                self.failure = Some(exhausted(Resource::MetadataAllocatedBytes, limits));
                return Err(E::custom("schema size overflow"));
            }
        }
        Ok(())
    }
}

fn property_owners(properties: &std::collections::HashMap<String, String>) -> Option<usize> {
    use crate::expressions::ColumnName;
    let mut bytes = hash_peak::<(String, String)>(properties.len())?;
    for (key, value) in properties {
        // Unknown properties retain exact key/value clones. The three known
        // owned String properties retain their value instead; counting both is
        // a safe union across try_parse's fixed match arms.
        bytes = bytes
            .checked_add(key.len())?
            .checked_add(value.len().checked_mul(2)?)?;
        if key == crate::table_properties::DATA_SKIPPING_STATS_COLUMNS {
            // The column-name parser can emit <=B+1 columns and <=B+1 field
            // segments. Each independent path Vec/String has its own minimum
            // capacity; include both parser path and ColumnName collection.
            let parts = value.len().checked_add(1)?;
            let paths = vector_peak::<String>(1)?
                .checked_mul(parts)?
                .checked_mul(2)?;
            let names = value
                .len()
                .checked_mul(4)?
                .checked_add(parts.checked_mul(2 * 8)?)?;
            bytes = bytes
                .checked_add(vector_peak::<ColumnName>(parts)?)?
                .checked_add(paths)?
                .checked_add(names)?;
        }
        // ParseIntervalError owns one input substring. Column-name diagnostics
        // Debug-format a bounded name; rejected property values then enter the
        // unknown map. The source has no recursive value deserializer here.
        let diagnostic = value.len().checked_mul("\\u{10ffff}".len())?
            .checked_add("Unescaped field name cannot start with a digit Invalid character after field No closing after field couldn't parse DataSkippingNumIndexedCols to positive integer Interval overflows seconds".len())?;
        bytes = bytes
            .checked_add(vector_peak::<u8>(diagnostic)?)?
            .checked_add(size_of::<String>() + size_of::<usize>())?;
    }
    Some(bytes)
}

// Rust 1.97 RawVec growth and hashbrown's 7/8 load factor. Both count
// simultaneous old/new backing. Checked arithmetic runs before allocation.
pub(super) fn vector_peak<T>(items: usize) -> Option<usize> {
    if items == 0 || size_of::<T>() == 0 {
        return Some(0);
    }
    let minimum = if size_of::<T>() == 1 {
        8
    } else if size_of::<T>() <= 1024 {
        4
    } else {
        1
    };
    items
        .checked_mul(2)?
        .max(minimum)
        .checked_mul(2)?
        .checked_mul(size_of::<T>())
}
pub(super) fn hash_peak<T>(items: usize) -> Option<usize> {
    if items == 0 {
        return Some(0);
    }
    let buckets = items
        .checked_mul(8)?
        .checked_add(6)?
        .checked_div(7)?
        .checked_next_power_of_two()?
        .max(4);
    // One control byte/bucket plus trailing group and alignment (max group16).
    buckets
        .checked_mul(size_of::<T>().checked_add(1)?)?
        .checked_add(2 * 16 - 1)?
        .checked_mul(2)
}

#[derive(Clone, Copy)]
enum Key {
    Type,
    Fields,
    Name,
    Nullable,
    Metadata,
    Other,
}
impl<'de> de::Deserialize<'de> for Key {
    fn deserialize<D: Deserializer<'de>>(decoder: D) -> Result<Self, D::Error> {
        struct KeyVisitor;
        impl Visitor<'_> for KeyVisitor {
            type Value = Key;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a schema property name")
            }
            fn visit_str<E: de::Error>(self, key: &str) -> Result<Key, E> {
                Ok(match key {
                    "type" => Key::Type,
                    "fields" => Key::Fields,
                    "name" => Key::Name,
                    "nullable" => Key::Nullable,
                    "metadata" => Key::Metadata,
                    _ => Key::Other,
                })
            }
        }
        decoder.deserialize_identifier(KeyVisitor)
    }
}

#[derive(Clone, Copy)]
enum StringKind {
    Name,
    Type,
    RootType,
}
struct CountString<'a> {
    shape: &'a mut PrimitiveSchemaShape,
    limits: &'a TaskLimits,
    kind: StringKind,
}
impl<'de> DeserializeSeed<'de> for CountString<'_> {
    type Value = ();
    fn deserialize<D: Deserializer<'de>>(self, decoder: D) -> Result<(), D::Error> {
        decoder.deserialize_any(self)
    }
}
impl<'de> Visitor<'de> for CountString<'_> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a primitive schema string")
    }
    fn visit_str<E: de::Error>(self, value: &str) -> Result<(), E> {
        if matches!(self.kind, StringKind::Type) && value == "variant" {
            self.shape.unsupported = true;
            return Err(E::custom("non-primitive schema type"));
        }
        self.shape.add(self.kind, value.len(), self.limits)
    }
    fn visit_map<A: MapAccess<'de>>(self, _: A) -> Result<(), A::Error> {
        self.shape.unsupported = true;
        Err(de::Error::custom("non-primitive schema type"))
    }
}

struct Root<'a> {
    shape: &'a mut PrimitiveSchemaShape,
    limits: &'a TaskLimits,
}
impl<'de> DeserializeSeed<'de> for Root<'_> {
    type Value = ();
    fn deserialize<D: Deserializer<'de>>(self, decoder: D) -> Result<(), D::Error> {
        decoder.deserialize_map(self)
    }
}
impl<'de> Visitor<'de> for Root<'_> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an object with primitive fields and empty field metadata")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        while let Some(key) = map.next_key::<Key>()? {
            match key {
                Key::Type => map.next_value_seed(CountString {
                    shape: self.shape,
                    limits: self.limits,
                    kind: StringKind::RootType,
                })?,
                Key::Fields => map.next_value_seed(FieldList {
                    shape: self.shape,
                    limits: self.limits,
                })?,
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        Ok(())
    }
}
struct FieldList<'a> {
    shape: &'a mut PrimitiveSchemaShape,
    limits: &'a TaskLimits,
}
impl<'de> DeserializeSeed<'de> for FieldList<'_> {
    type Value = ();
    fn deserialize<D: Deserializer<'de>>(self, decoder: D) -> Result<(), D::Error> {
        decoder.deserialize_seq(self)
    }
}
impl<'de> Visitor<'de> for FieldList<'_> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a list of primitive fields")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        while seq
            .next_element_seed(Field {
                shape: self.shape,
                limits: self.limits,
            })?
            .is_some()
        {}
        Ok(())
    }
}
struct Field<'a> {
    shape: &'a mut PrimitiveSchemaShape,
    limits: &'a TaskLimits,
}
impl<'de> DeserializeSeed<'de> for Field<'_> {
    type Value = ();
    fn deserialize<D: Deserializer<'de>>(self, decoder: D) -> Result<(), D::Error> {
        decoder.deserialize_map(self)
    }
}
impl<'de> Visitor<'de> for Field<'_> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a primitive field object")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        self.shape.fields = self
            .shape
            .fields
            .checked_add(1)
            .ok_or_else(|| A::Error::custom("schema field count overflow"))?;
        if let Err(failure) = check(Resource::SchemaNodes, self.shape.fields, self.limits) {
            self.shape.failure = Some(failure);
            return Err(de::Error::custom("schema field count exceeded"));
        }
        while let Some(key) = map.next_key::<Key>()? {
            match key {
                Key::Name => map.next_value_seed(CountString {
                    shape: self.shape,
                    limits: self.limits,
                    kind: StringKind::Name,
                })?,
                Key::Type => map.next_value_seed(CountString {
                    shape: self.shape,
                    limits: self.limits,
                    kind: StringKind::Type,
                })?,
                Key::Nullable => {
                    map.next_value::<bool>()?;
                }
                Key::Metadata => map.next_value_seed(EmptyMetadata(self.shape))?,
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        Ok(())
    }
}
struct EmptyMetadata<'a>(&'a mut PrimitiveSchemaShape);
impl<'de> DeserializeSeed<'de> for EmptyMetadata<'_> {
    type Value = ();
    fn deserialize<D: Deserializer<'de>>(self, decoder: D) -> Result<(), D::Error> {
        decoder.deserialize_map(self)
    }
}
impl<'de> Visitor<'de> for EmptyMetadata<'_> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("empty field metadata")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        if map.next_key::<IgnoredAny>()?.is_some() {
            self.0.unsupported = true;
            return Err(de::Error::custom("nonempty field metadata"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    const PRIMITIVE: &str = r#"{"type":"struct","fields":[{"name":"A\u0062","type":"long","nullable":true,"metadata":{}},{"name":"s","type":"string","nullable":false,"metadata":{}}]}"#;

    #[test]
    fn task_schema_errors_do_not_capture_optional_backtraces() {
        // Also executed in a separately recorded RUST_BACKTRACE=1 process. Do
        // not mutate the process environment while other unit tests run.
        for (schema, unsupported) in [
            (
                r#"{"type":"struct","fields":[{"name":"x","type":"unknown_primitive","nullable":true,"metadata":{}}]}"#,
                true,
            ),
            (
                r#"{"type":"struct","fields":[{"type":"long","nullable":true,"metadata":{}}]}"#,
                false,
            ),
        ] {
            let metadata: crate::actions::Metadata = serde_json::from_value(serde_json::json!({
                "id": "bounded-schema-error", "format": {"provider":"parquet", "options":{}},
                "schemaString": schema, "partitionColumns": [], "configuration": {}
            }))
            .unwrap();
            let shape =
                PrimitiveSchemaShape::inspect(schema, &TaskLimits::qualification(), |_| Ok(()))
                    .unwrap();
            shape
                .admit_materialization(schema.len(), &TaskLimits::qualification(), |_| Ok(()))
                .unwrap();
            let error = metadata.parse_schema_for_task().unwrap_err();
            if unsupported {
                assert!(matches!(error, crate::Error::Schema(_)));
            } else {
                assert!(matches!(error, crate::Error::MalformedJson(_)));
            }
        }
    }

    #[test]
    fn primitive_shape_borrows_strings_and_preserves_semantic_decoder() {
        let limits = TaskLimits::qualification();
        let admitted = Cell::new(0);
        let shape = PrimitiveSchemaShape::inspect(PRIMITIVE, &limits, |bytes| {
            admitted.set(bytes);
            Ok(())
        })
        .unwrap();
        assert_eq!(shape.fields, 2);
        assert_eq!(shape.name_bytes, 3);
        assert_eq!(shape.type_bytes, "structlongstring".len());
        assert_eq!(
            admitted.get(),
            PrimitiveSchemaShape::inspection_bytes(PRIMITIVE.len()).unwrap()
        );
        let schema: crate::schema::StructType = serde_json::from_str(PRIMITIVE).unwrap();
        assert_eq!(schema.num_fields(), shape.fields);
        assert_eq!(schema.fields().next().unwrap().name(), "Ab");
    }

    #[test]
    fn inspection_requires_admission_before_parser_and_checks_exact_boundary() {
        let limits = TaskLimits::qualification();
        let bytes = PrimitiveSchemaShape::inspection_bytes(PRIMITIVE.len()).unwrap();
        PrimitiveSchemaShape::inspect(
            PRIMITIVE,
            &limits.with_limit(Resource::MetadataAllocatedBytes, bytes),
            |_| Ok(()),
        )
        .unwrap();
        let called = Cell::new(false);
        let error = PrimitiveSchemaShape::inspect(
            PRIMITIVE,
            &limits.with_limit(Resource::MetadataAllocatedBytes, bytes - 1),
            |_| {
                called.set(true);
                Ok(())
            },
        )
        .unwrap_err();
        assert!(!called.get());
        assert!(
            matches!(error.kind(), super::super::FailureKind::ResourceExhausted(e)
            if e.resource == Resource::MetadataAllocatedBytes && e.observed == bytes)
        );
        assert!(PrimitiveSchemaShape::inspection_bytes(usize::MAX).is_none());
    }

    #[test]
    fn ordinary_materialization_requires_exact_peak_before_serde() {
        let limits = TaskLimits::qualification();
        let shape = PrimitiveSchemaShape::inspect(PRIMITIVE, &limits, |_| Ok(())).unwrap();
        let bytes = shape.materialization_bytes(PRIMITIVE.len()).unwrap();
        let called = Cell::new(false);
        shape
            .admit_materialization(
                PRIMITIVE.len(),
                &limits.with_limit(Resource::MetadataAllocatedBytes, bytes),
                |observed| {
                    assert_eq!(observed, bytes);
                    called.set(true);
                    Ok(())
                },
            )
            .unwrap();
        assert!(called.replace(false));
        let error = shape
            .admit_materialization(
                PRIMITIVE.len(),
                &limits.with_limit(Resource::MetadataAllocatedBytes, bytes - 1),
                |_| {
                    called.set(true);
                    Ok(())
                },
            )
            .unwrap_err();
        assert!(!called.get());
        assert!(
            matches!(error.kind(), super::super::FailureKind::ResourceExhausted(e)
            if e.observed == bytes)
        );
        assert!(shape.materialization_bytes(usize::MAX).is_none());
        assert!(PrimitiveSchemaShape {
            fields: usize::MAX,
            ..Default::default()
        }
        .materialization_bytes(0)
        .is_none());
        assert_eq!(vector_peak::<u8>(0), Some(0));
        assert_eq!(hash_peak::<usize>(0), Some(0));
    }

    #[test]
    fn unicode_expansion_and_duplicate_name_error_keep_ordinary_semantics() {
        let limits = TaskLimits::qualification();
        let json = r#"{"type":"struct","fields":[{"name":"İ","type":"long","nullable":true,"metadata":{}},{"name":"i\u0307","type":"long","nullable":true,"metadata":{}}]}"#;
        let shape = PrimitiveSchemaShape::inspect(json, &limits, |_| Ok(())).unwrap();
        shape
            .admit_materialization(json.len(), &limits, |_| Ok(()))
            .unwrap();
        let error = serde_json::from_str::<crate::schema::StructType>(json).unwrap_err();
        assert!(error.to_string().contains("Duplicate field name"));
    }

    #[test]
    fn recursive_materializers_refuse_before_decoding_their_payload() {
        let limits = TaskLimits::qualification();
        for schema in [
            r#"{"type":"struct","fields":[{"name":"x","type":"long","nullable":true,"metadata":{"comment":"ordinary comment"}}]}"#,
            r#"{"type":"struct","fields":[{"name":"x","type":{"type":"array","elementType":"long"},"nullable":true}]}"#,
            r#"{"type":"struct","fields":[{"name":"x","type":"variant","nullable":true}]}"#,
            r#"{"type":"struct","fields":[{"name":"x","type":"long","metadata":{"unsupported":{"large":[1,2,3]}}}]}"#,
        ] {
            let error = PrimitiveSchemaShape::inspect(schema, &limits, |_| Ok(())).unwrap_err();
            assert!(matches!(error.into_error(), crate::Error::Unsupported(_)));
        }
        // Unknown properties follow serde's existing ignored-value behavior.
        let unknown = r#"{"type":"struct","ignored":{"nested":[1,2,3]},"fields":[]}"#;
        assert_eq!(
            PrimitiveSchemaShape::inspect(unknown, &limits, |_| Ok(()))
                .unwrap()
                .fields,
            0
        );
    }
}
