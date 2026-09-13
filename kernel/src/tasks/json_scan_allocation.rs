//! Source-derived scratch of ScanBuilder with primitive empty-metadata fields,
//! no predicate, StatsOptions::none(), no partitions and column mapping None.
//! The ordinary builder is retained, including its stats-column membership set.

use super::json_schema_shape::{hash_peak, vector_peak};
use crate::schema::{ColumnName, StructField, StructType};
use crate::table_configuration::TableConfiguration;
use std::mem::size_of;

pub(super) fn scan_builder_peak(schema: &StructType, config: &TableConfiguration) -> Option<usize> {
    let fields = schema.num_fields();
    let names = schema
        .fields()
        .try_fold(0usize, |n, f| n.checked_add(f.name().len()))?;
    // StateInfo read_fields is consumed by StructType::try_new. Its IndexMap
    // owns both field names and keys, plus a case-insensitive validation set.
    let physical = vector_peak::<StructField>(fields)?
        .checked_add(vector_peak::<(usize, String, StructField)>(fields)?)?
        .checked_add(hash_peak::<usize>(fields)?)?
        .checked_add(names.checked_mul(2)?)?
        .checked_add(hash_peak::<String>(fields)?)?
        // Rust char::to_lowercase emits <=3 chars, <=4 UTF-8 bytes each;
        // RawVec old/new growth plus one independent minimum per name.
        .checked_add(names.checked_mul(3 * 4)?.checked_mul(4)?)?
        .checked_add(fields.checked_mul(2 * 8)?)?
        .checked_add(size_of::<StructType>() + "struct".len() + 2 * size_of::<usize>())?;
    // make_physical(field) has a depth-one borrowed logical_path; its ID and
    // sibling maps stay empty. Cow::with_name can clone the old name before
    // replacing it; physical_name and last_physical_field overlap assignment.
    let mapping = vector_peak::<&str>(1)?
        .checked_add(names.checked_mul(2)?)?
        .checked_add(vector_peak::<
            crate::scan::transform_spec::FieldTransformSpec,
        >(fields)?)?;
    // StatsColumnFilter walks each primitive field once. Its path holds one
    // String; accepted paths each own a one-element String Vec. The result Vec
    // remains live while physical_stats_columns_set moves paths into HashSet.
    let stats = vector_peak::<ColumnName>(fields)?
        .checked_add(hash_peak::<ColumnName>(fields)?)?
        .checked_add(vector_peak::<String>(1)?.checked_mul(fields)?)?
        .checked_add(vector_peak::<String>(1)?)?
        .checked_add(names.checked_mul(2)?)?;
    let mut properties = 0usize;
    if let Some(columns) = &config.table_properties().data_skipping_stats_columns {
        // required_physical_stats_columns retains one result per property
        // entry, including duplicates. Only a one-component path can resolve
        // in this primitive schema. Invalid paths still reserve fields_of_path
        // for ALL supplied components before producing a bounded diagnostic.
        properties =
            vector_peak::<ColumnName>(columns.len())?
                .checked_add(hash_peak::<(&str, crate::column_trie::ColumnTrie<'_>)>(
                    columns.len(),
                )?)?;
        for column in columns {
            let components = column.path().len();
            let bytes = column
                .iter()
                .try_fold(0usize, |n, p| n.checked_add(p.len()))?;
            properties = properties
                .checked_add(vector_peak::<&StructField>(components)?)?
                .checked_add(vector_peak::<String>(1)?)?
                .checked_add(bytes)?;
            // ColumnName Display doubles backticks and adds at most two quotes
            // plus a separator per component. fields_of_path_by also prints
            // one raw component. Error::generic(ToString) clones the formatted
            // String, so both owners coexist. No backtrace wrapper is called.
            let diagnostic = bytes.checked_mul(3)?.checked_add(components.checked_mul(3)?)?
                .checked_add("Could not resolve column '': field '' not found in schema Cannot resolve column '': intermediate field '' is not a struct type Column path cannot be empty".len())?;
            properties = properties.checked_add(vector_peak::<u8>(diagnostic)?.checked_mul(2)?)?;
        }
    }
    // The ordinary empty-schema rejection and duplicate-field diagnostic are
    // included even though valid task-built snapshots rule out duplicate names.
    let diagnostics = vector_peak::<u8>(names.checked_mul(2)?.checked_add(
        "Cannot scan Delta table with empty schema; use ALTER TABLE ADD COLUMN to add at least one column before scanning Duplicate field name (case-insensitive): ".len())?)?
        .checked_mul(2)?;
    [
        physical,
        mapping,
        stats,
        properties,
        diagnostics,
        size_of::<crate::scan::state_info::StateInfo>() + 2 * size_of::<usize>(),
        size_of::<crate::scan::Scan>(),
        size_of::<crate::scan::ScanBuilder>(),
        size_of::<crate::Error>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    #[test]
    fn scan_builder_properties_count_duplicate_and_invalid_paths() {
        let schema =
            StructType::try_new([StructField::nullable("a", crate::schema::DataType::LONG)])
                .unwrap();
        let config = |columns: Option<&str>| {
            let mut properties = std::collections::HashMap::new();
            if let Some(columns) = columns {
                properties.insert("delta.dataSkippingStatsColumns".into(), columns.into());
            }
            let metadata = crate::actions::Metadata::try_new(
                None,
                None,
                Arc::new(schema.clone()),
                vec![],
                0,
                properties,
            )
            .unwrap();
            let protocol =
                serde_json::from_str(r#"{"minReaderVersion":1,"minWriterVersion":2}"#).unwrap();
            TableConfiguration::try_new(
                metadata,
                protocol,
                url::Url::parse("memory:///table/").unwrap(),
                0,
            )
            .unwrap()
        };
        let plain = config(None);
        let repeated = config(Some("a,a,missing.nested"));
        assert_eq!(
            repeated
                .table_properties()
                .data_skipping_stats_columns
                .as_ref()
                .unwrap()
                .len(),
            3
        );
        assert!(
            scan_builder_peak(&schema, &repeated).unwrap()
                > scan_builder_peak(&schema, &plain).unwrap()
        );
        // The ordinary stats path still preserves its existing semantics: invalid
        // paths are omitted and repeated valid paths deduplicate in the final set.
        assert_eq!(
            repeated.physical_stats_columns_set(None),
            plain.physical_stats_columns_set(None)
        );
    }
}
