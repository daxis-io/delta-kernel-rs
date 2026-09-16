//! Fixed PM visitor construction and traversal owners from ToSchema/GetSchemaLeaves
//! and ArrowEngineData::visit_rows. These resource descriptors are checked against
//! the authoritative selected action schemas in tests; they do not interpret logs.

use std::mem::size_of;

use super::json_schema_shape::{hash_peak, vector_peak};
use crate::schema::{ArrayType, ColumnName, DataType, MapType, StructField, StructType};

pub(super) const PM_PATHS: &[&[&str]] = &[
    &["protocol", "minReaderVersion"],
    &["protocol", "minWriterVersion"],
    &["protocol", "readerFeatures"],
    &["protocol", "writerFeatures"],
    &["metaData", "id"],
    &["metaData", "name"],
    &["metaData", "description"],
    &["metaData", "format", "provider"],
    &["metaData", "format", "options"],
    &["metaData", "schemaString"],
    &["metaData", "partitionColumns"],
    &["metaData", "createdTime"],
    &["metaData", "configuration"],
    &["protocol_version"],
    &["metadata_version"],
];

/// Fixed schema initialization plus the largest simultaneous visitor scratch.
/// Metadata has ten StructFields in two StructTypes, two MapTypes and one
/// ArrayType; Protocol has four fields in one StructType and two ArrayTypes.
/// PmVersions has two scalar columns. ToSchema uses stack field arrays then
/// new_unchecked's IndexMap; leaves clones only map/list leaf types.
pub(super) fn pm_fixed_peak() -> Option<usize> {
    let field_names = [
        "id",
        "name",
        "description",
        "format",
        "provider",
        "options",
        "schemaString",
        "partitionColumns",
        "createdTime",
        "configuration",
        "minReaderVersion",
        "minWriterVersion",
        "readerFeatures",
        "writerFeatures",
    ];
    let names = field_names.iter().map(|name| name.len()).sum::<usize>();
    let fields = field_names.len();
    // Each of the three struct indexes contains at most all fourteen fields.
    // Sum their individual growth/relocation peaks; no field metadata entries.
    let schema_indexes = vector_peak::<(usize, String, StructField)>(fields)?
        .checked_add(hash_peak::<usize>(fields)?)?
        .checked_mul(3)?;
    let schema_headers = (size_of::<StructType>() + "struct".len()).checked_mul(3)?;
    let schema_names = names.checked_mul(2)?; // field name and IndexMap key
    let leaf_type_owners = (size_of::<MapType>() + "map".len())
        .checked_mul(2)?
        .checked_add((size_of::<ArrayType>() + "array".len()).checked_mul(3)?)?
        .checked_mul(2)?; // source schema and leaves' cloned DataTypes
    let paths = path_owners(PM_PATHS)?;
    let leaf_vectors =
        vector_peak::<ColumnName>(PM_PATHS.len())?
            .checked_add(vector_peak::<DataType>(PM_PATHS.len())?)?;
    let leaf_walker = vector_peak::<String>(3)?.checked_add(names)?;
    let visits = visitor_peak(PM_PATHS)?;
    let actions = size_of::<crate::log_segment::protocol_metadata_replay::PmCandidate>()
        .checked_add(size_of::<crate::actions::visitors::MetadataVisitor>())?
        .checked_add(size_of::<crate::actions::visitors::ProtocolVisitor>())?;
    // Required-column diagnostics contain only fixed paths and bounded counts.
    // Also include the original schema type's fixed primitive/collection names.
    let diagnostic = vector_peak::<u8>(names.checked_add(
        "Wrong number of MetadataVisitor getters: Missing required value for column protocol_version metadata_version".len())?
        .checked_add(2 * 20)?)?;
    [
        schema_indexes,
        schema_headers,
        schema_names,
        leaf_type_owners,
        paths,
        leaf_vectors,
        leaf_walker,
        visits,
        actions,
        diagnostic,
        size_of::<crate::Error>(),
        size_of::<super::OperationFailure>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

pub(super) const SCAN_PATHS: &[&[&str]] = &[
    &["path"],
    &["size"],
    &["modificationTime"],
    &["stats"],
    &["deletionVector", "storageType"],
    &["deletionVector", "pathOrInlineDv"],
    &["deletionVector", "offset"],
    &["deletionVector", "sizeInBytes"],
    &["deletionVector", "cardinality"],
    &["fileConstantValues", "partitionValues"],
    &["fileConstantValues", "baseRowId"],
    &["fileConstantValues", "defaultRowCommitVersion"],
    &["fileConstantValues", "tags"],
    &["fileConstantValues", "clusteringProvider"],
];

/// SCAN_ROW_SCHEMA contains six root fields, five DV fields and five constant
/// fields, in three StructTypes, and two MapTypes. No dynamic table schema is
/// part of this fixed scan-row representation. Count initialization even when
/// another task initialized the LazyLock, so admission never depends on warmth.
pub(super) fn scan_fixed_peak() -> Option<usize> {
    let names = [
        "path",
        "size",
        "modificationTime",
        "stats",
        "deletionVector",
        "fileConstantValues",
        "storageType",
        "pathOrInlineDv",
        "offset",
        "sizeInBytes",
        "cardinality",
        "partitionValues",
        "baseRowId",
        "defaultRowCommitVersion",
        "tags",
        "clusteringProvider",
    ];
    let name_bytes = names.iter().map(|name| name.len()).sum::<usize>();
    let indexes = vector_peak::<(usize, String, StructField)>(names.len())?
        .checked_add(hash_peak::<usize>(names.len())?)?
        .checked_mul(3)?;
    let structs =
        (size_of::<StructType>() + "struct".len() + 2 * size_of::<usize>()).checked_mul(3)?;
    // Two source maps and the two DataType clones in GetSchemaLeaves.
    let maps = (size_of::<MapType>() + "map".len())
        .checked_mul(2)?
        .checked_mul(2)?;
    let leaves = vector_peak::<ColumnName>(SCAN_PATHS.len())?
        .checked_add(vector_peak::<DataType>(SCAN_PATHS.len())?)?
        .checked_add(path_owners(SCAN_PATHS)?)?;
    let walker = vector_peak::<String>(2)?.checked_add(name_bytes)?;
    let diagnostic = vector_peak::<u8>(name_bytes.checked_add(
        "Wrong number of ScanFileVisitor getters: Missing required value for column add.modificationTime".len())?
        .checked_add(2 * 20)?)?;
    [
        indexes,
        structs,
        maps,
        name_bytes.checked_mul(2)?,
        leaves,
        walker,
        visitor_peak(SCAN_PATHS)?,
        diagnostic,
        size_of::<crate::scan::state::ScanFile>(),
        size_of::<crate::Error>(),
        size_of::<super::OperationFailure>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

/// Semantic work for a scan page: prepaid selected byte operations plus
/// both the borrowed preflight and ordinary visitor's path/getter visits.
/// Also charge each selection bit and the six ScanFile callback checks
/// (DV, transform, partitions, path emptiness, size and URL confinement).
/// URL construction/storage was admitted separately by the borrowed preflight.
pub(super) fn scan_work(selected_bytes: usize, rows: usize) -> Option<usize> {
    let path_steps = SCAN_PATHS
        .iter()
        .try_fold(0usize, |n, path| n.checked_add(path.len()))?;
    let visitor_steps = path_steps.checked_add(SCAN_PATHS.len())?.checked_mul(2)?;
    let row_steps = visitor_steps.checked_add(1 + 6)?;
    // Two column-map/getter setup walks also occur for an empty page.
    visitor_steps
        .checked_add(selected_bytes)?
        .checked_add(rows.checked_mul(row_steps)?)
}

fn path_owners(paths: &[&[&str]]) -> Option<usize> {
    paths.iter().try_fold(0usize, |total, path| {
        total
            .checked_add(vector_peak::<String>(path.len())?)?
            .checked_add(path.iter().map(|part| part.len()).sum::<usize>())
    })
}

fn visitor_peak(paths: &[&[&str]]) -> Option<usize> {
    let entries = paths
        .iter()
        .try_fold(0usize, |n, path| n.checked_add(path.len()))?
        .max(paths.len().checked_mul(2)?);
    // ColumnState is enum { Parent, AwaitingGetter(&DataType),
    // HasGetter(&dyn GetData) }: fat pointer plus discriminant, usize alignment.
    let map = hash_peak::<(ColumnName, [usize; 3])>(entries)?;
    let mut keys = 0usize;
    for path in paths {
        for length in 1..=path.len() {
            let prefix = &path[..length];
            let bytes = vector_peak::<String>(length)?
                .checked_add(prefix.iter().map(|name| name.len()).sum::<usize>())?;
            // Stored key, parent() temporary, entry(parent.clone()) temporary.
            keys = keys.checked_add(bytes.checked_mul(3)?)?;
        }
    }
    let depth = paths.iter().map(|path| path.len()).max().unwrap_or(0);
    let traversal = vector_peak::<String>(depth)?.checked_add(path_owners(paths)?)?;
    map.checked_add(keys)?
        .checked_add(traversal)?
        .checked_add(vector_peak::<&dyn crate::engine_data::GetData<'static>>(
            paths.len(),
        )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_resource_paths_match_authoritative_scan_schema() {
        let actual = &crate::scan::state::SCAN_ROW_LEAVES.as_ref().0;
        assert_eq!(actual.len(), SCAN_PATHS.len());
        for (actual, expected) in actual.iter().zip(SCAN_PATHS) {
            assert_eq!(actual.as_ref(), *expected);
        }
        assert!(scan_fixed_peak().unwrap() > size_of::<StructType>());
    }
    #[test]
    fn fixed_resource_paths_match_authoritative_pm_schemas() {
        use crate::actions::visitors::{METADATA_LEAVES, PROTOCOL_LEAVES};
        for (actual, expected) in PROTOCOL_LEAVES
            .as_ref()
            .0
            .iter()
            .chain(METADATA_LEAVES.as_ref().0.iter())
            .zip(&PM_PATHS[..13])
        {
            assert_eq!(actual.as_ref(), *expected);
        }
        assert_eq!(
            PROTOCOL_LEAVES.as_ref().0.len() + METADATA_LEAVES.as_ref().0.len(),
            13
        );
        assert!(pm_fixed_peak().unwrap() > size_of::<StructType>());
    }
}
