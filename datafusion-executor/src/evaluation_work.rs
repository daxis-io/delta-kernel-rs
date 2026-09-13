//! Prepaid semantic element/byte work for the sealed DataFusion path.
//! These counts describe selected source walks, not allocator instructions.
use crate::closed_plan_facts::ClosedPlanFacts;
use delta_kernel::tasks::{OperationFailure, Resource, ResourceExhausted, TaskLimits};

fn checked(value: Option<usize>, limits: TaskLimits) -> Result<usize, OperationFailure> {
    value.ok_or_else(|| {
        ResourceExhausted {
            resource: Resource::WorkUnits,
            limit: limits.limit(Resource::WorkUnits),
            observed: usize::MAX,
        }
        .into()
    })
}

// One unit per semantic file visit or byte visited/copied. Allocator internals
// are not instruction-counted. These are the reached source walks, not memory
// envelope multipliers. URL length dominates decoded Path length.
const PREPARATION_FILE_WALKS: &[&str] = &[
    "host input-size fold",
    "LogInput read-bound preflight",
    "LogInput read/identity loop",
    "store owner path-length fold",
    "store owner longest-path fold",
];
const PATH_DECODE_BYTE_WALKS: &[&str] = &[
    "percent escape discovery",
    "unchanged decoded prefix copy",
    "remaining percent decoder input",
    "remaining decoded output",
    "decoded UTF8 validation",
    "Path segment splitting",
    "PathPart dot comparison",
    "PathPart dot-dot comparison",
    "PathPart character validation",
    "Path owned string copy",
];
const PLANNING_FILE_WALKS: &[&str] = &[
    "store owner path-length fold",
    "store owner longest-path fold",
    "store owner construction",
    "ScanJson descriptor lowering",
    "physical config clone",
];
const PLANNING_PATH_WALKS: &[&str] = &[
    "store URL scheme search",
    "store URL authority search",
    "exact descriptor URL comparison",
    "origin URL clone",
    "origin path replacement",
    "physical config path clone",
];
const EXECUTION_FILE_WALKS: &[&str] = &[
    "FileStream local group clone",
    "local work queue pop",
    "whole-file morsel",
    "partition projection opener",
    "partition value clone",
    "partition literal construction",
    "JsonSource opener",
    "cached store lookup",
    "bounded read trace",
];
const EXECUTION_PATH_WALKS: &[&str] = &[
    "local group path clone",
    "cached exact path comparison",
    "GetResult ObjectMeta path clone",
];

pub(crate) fn planning_files(f: &ClosedPlanFacts) -> Option<usize> {
    // Path::from_url_path runs once in store construction and once in lowering.
    // Canonical cached-store construction reads 20 digits + '.json' + separator.
    let byte_walks = PATH_DECODE_BYTE_WALKS
        .len()
        .checked_mul(2)?
        .checked_add(PLANNING_PATH_WALKS.len())?;
    f.file_url_bytes.checked_mul(byte_walks)?.checked_add(
        f.files
            .checked_mul(PLANNING_FILE_WALKS.len().checked_add(20 + 5 + 1)?)?,
    )
}

pub(crate) fn execution_files(f: &ClosedPlanFacts) -> Option<usize> {
    // ProjectionOpener rewrites partition columns, visits input_file_name,
    // builds its projector and copies aliases for each file even for zero rows.
    // The closed root projection is bounded by the recorded total schema fields;
    // partition-index lookup has exactly one candidate: the admitted version column.
    const PROJECTION_WALKS: &[&str] = &[
        "partition rewrite",
        "input_file_name rewrite",
        "projector fields",
    ];
    let projection = f
        .fields
        .checked_mul(PROJECTION_WALKS.len())?
        .checked_add(f.fields)?
        .checked_add(f.name_bytes.checked_mul(2)?)?; // rewritten alias + output field name
    let per_file = EXECUTION_FILE_WALKS
        .len()
        .checked_add(20 + 5 + 1)?
        .checked_add(projection)?;
    f.files
        .checked_mul(per_file)?
        .checked_add(f.file_url_bytes.checked_mul(EXECUTION_PATH_WALKS.len())?)
}

pub(crate) fn preparation(
    f: &ClosedPlanFacts,
    limits: TaskLimits,
) -> Result<usize, OperationFailure> {
    // Source owner walks: Kernel->Arrow conversion and its preflight; output
    // header tree; decoder tree/tape/diagnostic tree; FIRST_VALUE candidate,
    // compact/emit trees; key output/null trees; terminal page/rebind trees.
    // Each visit processes a schema element; fixed-width arithmetic inside a
    // visit is not counted as another semantic element.
    const WALKS: &[&str] = &[
        "conversion bound",
        "conversion",
        "headers",
        "decoder",
        "tape",
        "diagnostics",
        "candidate",
        "compact/emit",
        "key output",
        "key null",
        "page",
        "rebind",
    ];
    checked(
        f.types
            .checked_mul(WALKS.len())
            .and_then(|n| n.checked_add(f.fields))
            .and_then(|n| n.checked_add(f.files.checked_mul(PREPARATION_FILE_WALKS.len())?)),
        limits,
    )
}

pub(crate) fn planning(f: &ClosedPlanFacts, limits: TaskLimits) -> Result<usize, OperationFailure> {
    let units = (|| -> Option<usize> {
        let nodes = f
            .expressions
            .checked_add(f.path_components.checked_mul(2)?)?
            .checked_add(f.constructed_fields.checked_mul(2)?)?
            .checked_add(f.root_fields.checked_mul(2)?)?
            .checked_add(f.aggregates.checked_mul(12)?)?
            .checked_add(f.cases.checked_mul(3)?)?;
        // Both original/projected CASE trees are revisited at each depth.
        let copies = 1usize.checked_shl(f.case_depth as u32)?;
        const WALKS: &[&str] = &[
            "lower",
            "coalesce rewrite",
            "physical expression",
            "async detection",
            "CASE collect",
            "CASE project",
            "mapping normalize",
            "equivalence normalize",
            "constant detection",
            "indirect projection",
            "projection fields",
            "aggregate argument/state fields",
        ];
        let walks = nodes.checked_mul(copies)?.checked_mul(WALKS.len())?;
        // Fixed schema/name lookup and collision comparisons. The system's
        // widest Struct bounds candidates; all names are fixed ASCII names.
        let names = f.name_bytes.checked_mul(f.max_struct_fields)?;
        // CASE return_field formats its original body (not both retained
        // projected bodies). Summing all descendants and nesting depth covers
        // nested return fields; Coalesce repeats its left argument in WHEN/THEN.
        let rendered = f
            .path_components
            .checked_add(f.constructed_fields)?
            .checked_mul(
                f.max_name_bytes
                    .checked_mul(2)?
                    .checked_add("get_field(, )Utf8(\"\")@18446744073709551615".len())?,
            )?
            .checked_add(
                f.expressions
                    .checked_mul("CASE WHEN  THEN  ELSE  END".len())?,
            )?
            .checked_mul(f.case_depth)?;
        // Four invariant walks, two full-child statistics walks, and sorted
        // column/index classification. No interval analysis or ordered input.
        let physical = f.nodes.checked_add(2)?;
        let properties = physical
            .checked_mul(4)?
            .checked_add(physical.checked_mul(2)?.checked_mul(f.max_struct_fields)?)?
            .checked_add(f.root_fields.checked_mul(f.max_root_fields)?)?;
        walks
            .checked_add(names)?
            .checked_add(rendered)?
            .checked_add(properties)?
            .checked_add(planning_files(f)?)
    })();
    checked(units, limits)
}

pub(crate) fn execution(
    f: &ClosedPlanFacts,
    rows: usize,
    bytes: usize,
    limits: TaskLimits,
) -> Result<usize, OperationFailure> {
    let units = (|| -> Option<usize> {
        // Complete input is consumed by the first aggregate pull. Prepay it
        // ONCE, including all possible emitted groups, not again per page.
        // Lexical framing is prepaid separately before LogInput::load.
        const BYTE_PASSES: &[&str] = &[
            "tape unescape",
            "decoder values",
            "diagnostic concat",
            "coalesce merge",
            "key encode",
            "key owner",
            "partial candidate",
            "partial compact",
            "partial state",
            "final candidate",
            "final compact",
            "final emit",
        ];
        let payload = bytes.checked_mul(BYTE_PASSES.len())?;
        // Four projects, two filters, two aggregate stages process schema
        // elements; evaluate both branches where CASE semantics can require it.
        let elements = f
            .types
            .checked_mul(rows)?
            .checked_mul(4 + 2 + 2)?
            .checked_add(f.expressions.checked_mul(rows)?.checked_mul(2)?)?;
        let collisions = if f.grouping_keys == 0 {
            0
        } else {
            // GroupValuesRows can compare each candidate with every existing
            // group in a colliding table. Total key bytes are source-derived
            // by the same encoder bound used in row-key allocation admission.
            crate::row_key_allocation::encoded_key_bytes(rows, bytes)?
                .checked_mul(rows)?
                .checked_mul(2)?
        };
        payload
            .checked_add(elements)?
            .checked_add(collisions)?
            .checked_add(execution_files(f)?)
    })();
    checked(units, limits)
}
