//! Fixed descriptors of the two ordinary no-checkpoint JSON producers.
//! Counts are expanded from their source declarations, before either builder
//! runs. Tests compare the resulting ordinary wire encoding as a drift check.

// Add and Remove each have eleven fields and one five-field DV child.
const ADD: usize = 11 + 5;
const REMOVE: usize = 11 + 5;
// file_action_key has path and DV, whose three fields are storageType,
// pathOrInlineDv and offset. Include the enclosing key field when appended.
const KEY: usize = 2 + 3;
const READ: usize = 3 + ADD + REMOVE;
const PATCH: usize = READ + 1 + 1 + KEY;
const AGGREGATE: usize = 1 + KEY + 1 + ADD;
const OUTPUT: usize = 1 + ADD - 1; // metadata_output_projection drops stats
const SCAN_ROW: usize = 6 + 5 + 5;
// Filter nodes have no separately serialized schema. PM's read/output schemas
// have17 and18 fields and are dominated by this five-schema live-add total.
pub(super) const SCHEMA_FIELDS: usize = READ + PATCH + AGGREGATE + OUTPUT + SCAN_ROW;
// Each Map DataType adds key/value primitives beyond its owning StructField.
// There are4/4/2/2/2 maps in the five schemas; the null stats literal adds1.
pub(super) const DATA_TYPES: usize = SCHEMA_FIELDS + 2 * (4 + 4 + 2 + 2 + 2) + 1;

// commit filter: OR plus two NOT/IS_NULL/Column trees =7.
// commit patch: StructPatch + is_add's4 nodes + file_action_key's20:
// two Structs, five three-node coalesces (storageType appears in the null
// guard too), and the guard's ExpressionPredicate/Not/IsNull wrappers.
// winning-add filter3; metadata output Struct/StructPatch2; scan-row12.
pub(super) const EXPRESSIONS: usize = 7 + (1 + 4 + (2 + 5 * 3 + 3)) + 3 + 2 + 12;
// ColumnName components: commit filter4,is_add2,key28,aggregate4,
// winning-add filter1,metadata patch input1,nine scan-row columns*2.
pub(super) const PATH_COMPONENTS: usize = 4 + 2 + 28 + 4 + 1 + 1 + 9 * 2;
pub(super) const NODES: usize = 7;

// Fixed field names from actions, file_action_key, and SCAN_ROW_SCHEMA.
// defaultRowCommitVersion is the longest reached name (22 UTF-8 bytes).
// PM names (including protocol_version/metadata_version) are shorter.
const NAME_BYTES: usize = "defaultRowCommitVersion".len();

/// Protobuf upper bound, including Operation::QueryPlan, excluding file entries
/// and URL bytes (which are admitted separately from the immutable manifest).
/// Each FIELD includes its tag and the largest possible varint/length prefix.
/// Empty system metadata maps and absent statistics schemas contribute nothing.
pub(super) const FIXED_ENCODING: usize = {
    let field = super::plan_shape::FIELD_BOUND;
    // StructField: name/type/nullability + containing repeated-field envelope.
    // DataType: kind and up to Map key/value/nullability fields. Primitive and
    // Struct alternatives are smaller. Child fields/types are counted above.
    let schema = 4 * SCHEMA_FIELDS + 4 * DATA_TYPES;
    // Every reached expression/predicate uses <=4 non-child field envelopes;
    // child edges are included by their containing node's envelope. ColumnName
    // components add their own repeated strings. Two StructPatch maps each get
    // room for the one reached field transform (stats removal), its key/value
    // envelope, FieldTransform flags and insertion wrapper.
    let expressions = 4 * EXPRESSIONS + PATH_COMPONENTS + 2 * 6;
    // Node/operator/schema/output wrappers, six input edges, at most four PM
    // aggregate records each containing three ColumnNames, and query/plan.
    let operators = 5 * NODES + (NODES - 1) + 4 * 5 + 2;
    let names = (SCHEMA_FIELDS + PATH_COMPONENTS + 2 + 1) * NAME_BYTES;
    (schema + expressions + operators) * field + names
};

/// Sum of the reached schema-allocation sites. Each site is bounded by the
/// largest system schema (the42-field commit patch), including deep clones.
/// Summing sites admits their overlap without assuming LazyLock warmth.
const SCHEMA_SITES: &[&str] = &[
    "ADD_SCHEMA ToSchema",
    "ADD_FIELD deep clone",
    "REMOVE_FIELD ToSchema",
    "FILE_ACTION_KEY_FIELD schema",
    "SCAN_ROW_SCHEMA ToSchema",
    "METADATA_FIELD ToSchema",
    "PROTOCOL_FIELD ToSchema",
    "normalized_add_field schema_walk",
    "metadata projection input clone",
    "metadata projection schema_walk",
    "metadata output add-schema clone",
    "json_read_schema Add/Remove clones",
    "commit append key-field clone",
    "commit projection schema_walk",
    "aggregate key/value field clones",
    "PM versioned input clones",
    "PM aggregate output field clones",
];

/// Fixed producer heap owners, excluding manifest and file-dependent owners.
/// Schema construction, sparse patch lowering and ordinary PlanBuilder remain
/// unchanged. This sums source allocation sites, not measured heap peaks.
pub(super) fn fixed_owner_peak() -> Option<usize> {
    use std::mem::size_of;

    use super::json_schema_shape::{hash_peak, vector_peak};
    use crate::expressions::{ColumnName, Expression, ExpressionRef, Predicate};
    use crate::plans::ir::nodes::Agg;
    use crate::plans::ir::plan::PlanNode;
    use crate::schema::{ArrayType, MapType, SchemaRef, StructField, StructType};
    use crate::struct_patch::{ExpressionFieldPatch, ProjectionStructPatchBuilder};
    let word = size_of::<usize>();
    #[repr(C)]
    struct SharedOwner<T> {
        strong: usize,
        weak: usize,
        value: T,
    }
    const fn owner<T>() -> usize {
        size_of::<SharedOwner<T>>()
    }
    // The widest source schema has42 total fields, seven StructTypes and four
    // maps. PM additionally reaches three ArrayTypes. Each individual struct
    // index is bounded by42 entries; count all seven relocation/validation
    // peaks. Empty metadata maps allocate nothing. System names are ASCII.
    let index = vector_peak::<(usize, String, StructField)>(PATCH)?
        .checked_add(hash_peak::<usize>(PATCH)?)?
        .checked_add(hash_peak::<String>(PATCH)?)?
        .checked_add(vector_peak::<StructField>(PATCH)?)?;
    let names = PATCH.checked_mul(NAME_BYTES)?;
    let schema = index
        .checked_mul(7)?
        .checked_add(names.checked_mul(2)?)? // field names and IndexMap keys
        .checked_add(vector_peak::<u8>(NAME_BYTES)?.checked_mul(PATCH)?)? // lowercase validation
        .checked_add(7 * (owner::<StructType>() + "struct".len()))?
        .checked_add(4 * (size_of::<MapType>() + "map".len()))?
        .checked_add(3 * (size_of::<ArrayType>() + "array".len()))?;
    let schemas = schema.checked_mul(SCHEMA_SITES.len())?;
    // Every node in the five expression trees gets an Expression/Predicate Arc
    // and at most one child-vector slot per tree edge. Using the larger enum
    // also bounds predicate wrappers stored by value inside Expression.
    let expression_header = owner::<Expression>().max(owner::<Predicate>());
    let expressions = EXPRESSIONS
        .checked_mul(expression_header)?
        .checked_add(vector_peak::<ExpressionRef>(EXPRESSIONS)?)?;
    // The source has58 final ColumnName components. Construction of a joined
    // name can retain original leaf, prefix and joined path. build_plan clones
    // aggregate/group paths once; two further complete path sets cover those
    // input/output containers. Other expression references clone only Arcs.
    let paths = vector_peak::<String>(3)?
        .checked_add(3 * NAME_BYTES)?
        .checked_mul(PATH_COMPONENTS)?
        .checked_mul(3 + 2)?;
    // Five contains_col probes (three output, two commit) create two-component
    // names, including the deliberately absent parsed-stats/partition fields.
    // Their field_at failure owns format! + Error::generic's String clone.
    let probes = vector_peak::<String>(2)?
        .checked_add(2 * "partitionValues_parsed".len())?
        .checked_mul(5)?;
    let diagnostic = "Could not resolve column '': field '' not found in schema Cannot resolve column '': intermediate field '' is not a struct type".len()
        .checked_add(3 * (3 * NAME_BYTES + 3))?;
    let diagnostics = vector_peak::<u8>(diagnostic)?
        .checked_mul(2)?
        .checked_mul(5)?;
    // Sparse builders: one stats-drop map entry and two appended projection
    // items. FieldPatchOp's largest payload is ProjectionItem; retain its tag
    // and insert_after Vec even though only Drop is selected here.
    type Item = (StructField, ExpressionRef);
    let field_patch_width = size_of::<Item>() + word + size_of::<Vec<Item>>();
    let patch_buckets = 4usize; // hashbrown minimum for the single entry
    let patch_map = patch_buckets
        .checked_mul(size_of::<String>() + field_patch_width + 1)?
        .checked_add(2 * 16 - 1)?
        .checked_mul(2)?;
    let patches = patch_map
        .checked_add(hash_peak::<(String, ExpressionFieldPatch)>(1)?)?
        .checked_add(2 * "stats".len())?
        .checked_add(vector_peak::<Item>(2)?)?
        .checked_add(vector_peak::<ExpressionRef>(2)?)?
        .checked_add(3 * size_of::<ProjectionStructPatchBuilder<'_>>())?;
    // Seven BuilderNode Arcs have the PlanNode layout plus a SchemaRef; then
    // build_plan allocates PlanNode Vec and its pointer->index hash map. The
    // six one-input Vecs exist in both the builder DAG and emitted plan.
    let graph = NODES
        .checked_mul(owner::<(PlanNode, SchemaRef)>())?
        .checked_add(vector_peak::<PlanNode>(NODES)?)?
        .checked_add(hash_peak::<(*const (), usize)>(NODES)?)?
        .checked_add(vector_peak::<usize>(1)?.checked_mul(2 * (NODES - 1))?)?;
    // At most four aggregate operands, two alias Strings, one grouping path,
    // and two one-column source-constant name vectors (build + emitted clone).
    let aggregate = vector_peak::<(Agg, Option<String>)>(4)?
        .checked_add(vector_peak::<Agg>(4)?.checked_mul(2)?)?
        .checked_add(vector_peak::<ColumnName>(1)?.checked_mul(2)?)?
        .checked_add(2 * NAME_BYTES)?;
    let source_names = vector_peak::<String>(1)?
        .checked_add("version".len())?
        .checked_mul(2)?;
    // union_all's one-element input and present Vecs, and references()'s
    // borrowed-name HashSet for two filters and three projections. A path can
    // be visited at most once per expression-tree node at each call site.
    let validation = vector_peak::<crate::PlanBuilder>(1)?
        .checked_add(vector_peak::<std::sync::Arc<()>>(1)?)?
        .checked_add(hash_peak::<&ColumnName>(EXPRESSIONS)?.checked_mul(2 + 3)?)?;
    [
        schemas,
        expressions,
        paths,
        probes,
        diagnostics,
        patches,
        graph,
        aggregate,
        source_names,
        validation,
        size_of::<crate::Error>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

/// Semantic element/byte traversal bound for the ordinary closed producers.
/// The same named construction sites used by fixed_owner_peak also bound
/// construction work. Hash-table collision comparisons are included rather
/// than relying on expected constant-time lookup. This does not count machine
/// instructions inside allocation or hashing as separate semantic work units.
pub(super) fn work_units(files: usize, path_bytes: usize) -> Option<usize> {
    // The widest individual struct is Add/Remove (eleven direct children).
    // Metadata has nine, protocol four, scan-row six and DV five. Four is the
    // maximum fixed schema depth, including the enclosing system root.
    const STRUCT_WIDTH: usize = 11;
    const SCHEMA_DEPTH: usize = 4;
    // StructType::try_new: visit/recursive metadata validation, original name
    // copy, lowercase key and IndexMap key; hash both keys and compare at most
    // STRUCT_WIDTH names in each of the two collision chains.
    let field_work = SCHEMA_DEPTH + 3 * NAME_BYTES + 2 * NAME_BYTES + 2 * STRUCT_WIDTH * NAME_BYTES;
    let schemas = SCHEMA_SITES
        .len()
        .checked_mul(PATCH)?
        .checked_mul(field_work)?;
    // references() is reached by two filters and three projects. The largest
    // distinct reference set is the nine scan-row paths; commit patch has
    // eight (the storageType guard repeats a path). Bound each component's
    // creation/hash/name copy and every possible set collision comparison.
    let reference_paths = 9;
    let reference_passes = 2 + 3;
    let references = PATH_COMPONENTS
        .checked_mul(1 + 3 * NAME_BYTES + reference_paths * NAME_BYTES)?
        .checked_add(EXPRESSIONS)?
        .checked_mul(reference_passes)?;
    // Resolving all paths additionally traverses the schema's own field index.
    let resolution = PATH_COMPONENTS.checked_mul(1 + STRUCT_WIDTH * NAME_BYTES)?;
    // Construct expression nodes/child edges, sparse patches, and builder DAG;
    // build_plan visits nodes/edges, emits them, then PlanShape validates them.
    let graph = EXPRESSIONS
        .checked_mul(2)?
        .checked_add(2 * 2)?
        .checked_add(NODES.checked_mul(3)?)?
        .checked_add((NODES - 1) * 3)?;
    // Both task constructors preflight once and the admitted producer repeats
    // that preflight: one owner fold each. Canonical URL/encoding totals are O(1).
    // Then cover merge, reverse, version tagging, source collection, emitted
    // ScanJson clone, and the i64 version check each touch every selected file.
    let file_visits = 2 + 1 + 1 + 1 + 1 + 1 + 1;
    let file_work = files.checked_mul(file_visits + 25 + "json".len())?;
    // ParsedLogPath cover clone, ScanFile URL clone and emitted ScanJson clone.
    let copied_paths = path_bytes.checked_mul(3)?;
    [
        schemas,
        references,
        resolution,
        graph,
        file_work,
        copied_paths,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

/// Snapshot-only manifest validation and conversion, prepaid before either runs.
/// The canonical prefix/25-byte immediate-child invariant is already checked by
/// discovery. No descriptor walk is needed to compute this bound.
pub(super) fn manifest_work(files: usize, root_bytes: usize) -> Option<usize> {
    let paths = root_bytes.checked_add(25)?.checked_mul(files)?;
    const PATH_WALKS: &[&str] = &[
        "manifest canonical prefix",
        "segment canonical prefix",
        "segment URL base copy",
        "segment exact URL comparison",
        "ParsedLogPath ancestor splitting",
        "ParsedLogPath ancestor delta-log comparisons",
    ];
    const FILE_WALKS: &[&str] = &[
        "manifest validation/owner sum",
        "segment owner preflight",
        "to_log_segment owner preflight",
        "segment map/collect",
    ];
    // Manifest digits/suffix; URL relative join; ParsedLogPath filename copy,
    // split, version parse, extension copy and immediate parent comparison.
    let filename = 25 + 25 + 25 + 25 + 20 + 4 + "_delta_log".len();
    // The single log-root join/validation and latest ParsedLogPath clone. All
    // commit URLs have this exact length; no sum/max lookup is necessary.
    let roots = root_bytes
        .checked_mul(3)?
        .checked_add(root_bytes.checked_add(25 + 25 + 4)?)?;
    paths
        .checked_mul(PATH_WALKS.len())?
        .checked_add(files.checked_mul(FILE_WALKS.len().checked_add(filename)?)?)?
        .checked_add(roots)
}
