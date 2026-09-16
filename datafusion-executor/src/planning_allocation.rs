//! Owners constructed by the concrete planner for the two sealed JSON producers.
//!
//! Counts come from borrowed Kernel IR, never from an allocated logical/physical
//! plan. CASE reconstruction is bounded by its producer depth, including the
//! second reconstruction in ProjectionMapping. Runtime buffers are separate.
use std::mem::size_of;
use std::sync::Arc;

use datafusion::arrow::datatypes::{Field, FieldRef};
use datafusion::common::{ColumnStatistics, ScalarValue, Statistics};
use datafusion::logical_expr::{Expr, LogicalPlan};
use datafusion::physical_expr::equivalence::EquivalenceClass;
use datafusion::physical_expr::expressions::{
    BinaryExpr, CaseExpr, CastExpr, Column, IsNotNullExpr, IsNullExpr, Literal, NotExpr,
};
use datafusion::physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion::physical_plan::{ExecutionPlan, PlanProperties};
use datafusion::physical_planner::DefaultPhysicalPlanner;
use delta_kernel::tasks::{OperationFailure, Resource, ResourceExhausted, TaskLimits};

use crate::closed_plan_facts::ClosedPlanFacts;
use crate::json_arrays::vec_peak;

type Physical = Arc<dyn PhysicalExpr>;
// ArcInner<T> is repr(C): two atomic usize counts followed by T. A
// repr(C) witness preserves both payload and trailing alignment on wasm32.
#[repr(C)]
struct SharedOwner<T> {
    strong: usize,
    weak: usize,
    value: T,
}
const fn owner<T>() -> usize {
    size_of::<SharedOwner<T>>()
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PlanningEnvelope {
    /// Physical schemas, expressions, operator and equivalence containers.
    pub retained: usize,
    /// Construction peak, including retained physical and lowered logical IR.
    pub construction: usize,
}

/// A partition-independent hashbrown bound. Each table reserves at 7/8 load;
/// power-of-two rounding is less than twice the requested bucket count. Doubling
/// and an old/new reallocation overlap give fewer than eight buckets per live
/// entry. Separate control groups/minimum buckets are charged PER table, so
/// combining small independent tables cannot hide their minimum allocations.
fn hash_partitions(entries: usize, tables: usize, width: usize) -> Option<usize> {
    let buckets = entries
        .checked_mul(8)?
        .checked_add(tables.checked_mul(8)?)?;
    buckets
        .checked_mul(width.checked_add(1)?)?
        .checked_add(tables.checked_mul(2 * 16)?)
}
fn vectors(entries: usize, containers: usize, width: usize) -> Option<usize> {
    // vec_peak is linear above RawVec's minimum; adding each container's
    // minimum separately covers any partition of the aggregate entry count.
    vec_peak(entries, width)?.checked_add(containers.checked_mul(vec_peak(0, width)?)?)
}

// IndexMap's dense Bucket<K,V> includes HashValue(usize), separately from
// the hashbrown index buckets. The tuple layouts include Rust padding.
fn property_owners(members: usize, nodes: usize) -> Option<usize> {
    let reverse = vectors(members, nodes, size_of::<(usize, Physical, usize)>())?
        .checked_add(hash_partitions(members, nodes, size_of::<usize>())?)?;
    let classes = vectors(members, nodes, size_of::<EquivalenceClass>())?
        .checked_add(vectors(members, members, size_of::<(usize, Physical)>())?)?
        .checked_add(hash_partitions(members, members, size_of::<usize>())?)?;
    let projection = vectors(
        members,
        nodes,
        size_of::<(usize, Physical, Vec<(Physical, usize)>)>(),
    )?
    .checked_add(hash_partitions(members, nodes, size_of::<usize>())?)?
    .checked_add(vectors(members, members, size_of::<(Physical, usize)>())?)?;
    // Each aggregate contributes one unique group constraint; projection can
    // propagate it through every remaining node. Empty orderings need no heap.
    let constraints = vectors(nodes, nodes, size_of::<datafusion::common::Constraint>())?
        .checked_add(vectors(members, nodes, size_of::<usize>())?)?;
    reverse
        .checked_add(classes)?
        .checked_add(projection)?
        .checked_add(constraints)
}

pub(crate) fn preflight(
    f: &ClosedPlanFacts,
    schema_owners: usize,
    limits: TaskLimits,
) -> Result<PlanningEnvelope, OperationFailure> {
    if !matches!(f.nodes, 2 | 7) || f.case_depth > 2 {
        return Err(OperationFailure::malformed_response());
    }
    let mut b = Bound { bytes: 0, limits };
    // ScanJson schema-order projection and second aggregate stage.
    let physical_nodes = f.nodes + 2;
    // Per path component: accessor function and field-name literal (root Column
    // replaces these at component0). Constructed fields include every patch
    // pass-through and generated name/value pair; root outputs add cast/alias.
    // Four extra leaves per aggregate cover cloned ordering key and its generated
    // boolean filter; the full twelve also covers the aggregate, alias and state output Columns.
    let base = f
        .expressions
        .checked_add(
            f.path_components
                .checked_mul(2)
                .ok_or_else(|| b.overflow())?,
        )
        .and_then(|n| n.checked_add(f.constructed_fields.checked_mul(2)?))
        .and_then(|n| n.checked_add(f.root_fields.checked_mul(2)?))
        .and_then(|n| n.checked_add(f.aggregates.checked_mul(12)?))
        .and_then(|n| n.checked_add(f.cases.checked_mul(3)?))
        .ok_or_else(|| b.overflow())?;
    // A rebuilt CASE retains both original and projected bodies. At depth d
    // this recurrence is T(d)<=2*T(d-1). ProjectionMapping normalizes Columns
    // unconditionally, retaining a second rebuilt tree. Its named-struct
    // decomposition adds accessor/literal/target-column owners per field.
    let copies = 1usize
        .checked_shl(f.case_depth as u32)
        .ok_or_else(|| b.overflow())?;
    let exprs = base
        .checked_mul(copies)
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(f.fields.checked_mul(4)?))
        .ok_or_else(|| b.overflow())?;
    let header = [
        owner::<CaseExpr>(),
        owner::<ScalarFunctionExpr>(),
        owner::<BinaryExpr>(),
        owner::<CastExpr>(),
        owner::<Column>(),
        owner::<Literal>(),
        owner::<IsNullExpr>(),
        owner::<IsNotNullExpr>(),
        owner::<NotExpr>(),
    ]
    .into_iter()
    .max()
    .unwrap();
    b.add(exprs.checked_mul(header))?;
    // Child vectors, CASE when/then pairs and projection indices. Charge one
    // container minimum per expression and the tree's edges. A when/then pair
    // has the same storage as its two independently counted child references.
    b.add(vectors(exprs, exprs, size_of::<Physical>()))?;
    b.add(vectors(
        f.cases.checked_mul(copies).ok_or_else(|| b.overflow())?,
        exprs,
        size_of::<usize>(),
    ))?;
    // All physical names here are fixed field/builtin names, not user strings.
    // Field-name literal ScalarValue and Column clones each own their String.
    let name = f.max_name_bytes.max("named_struct".len());
    b.add(exprs.checked_mul(vec_peak(name, 1).ok_or_else(|| b.overflow())?))?;
    // Five schema ownership sites: initial conversion, declared casts, relaxed
    // physical nullability, projector return fields and aggregate input schema.
    // Nested Arrow Fields clones share their Arcs; charging complete conversion
    // owners at each site covers that sharing and conversion scratch as well.
    b.add(schema_owners.checked_mul(5))?;
    // Aggregate state fields: value/version/is_set for Partial, final value,
    // and another state_fields call for merge-expression execution setup.
    let state_fields = f.aggregates.checked_mul(7).ok_or_else(|| b.overflow())?;
    b.add(state_fields.checked_mul(owner::<Field>()))?;
    b.add(vectors(
        state_fields,
        f.aggregates.checked_mul(3).ok_or_else(|| b.overflow())?,
        size_of::<FieldRef>(),
    ))?;
    b.add(state_fields.checked_mul(
        vec_peak(name + "[first_value_is_set]".len(), 1).ok_or_else(|| b.overflow())?,
    ))?;
    b.add(f.aggregates.checked_mul(owner::<
        datafusion::physical_expr::aggregate::AggregateFunctionExpr,
    >()))?;
    // FIRST_VALUE args/filter/order owners, both retained input-field vectors,
    // order fields/types and Partial/Final aggregate/group/filter Arc slices.
    b.add(vectors(
        f.aggregates.checked_mul(12).ok_or_else(|| b.overflow())?,
        physical_nodes,
        size_of::<ScalarValue>(),
    ))?;
    // Human display is retained separately from the short alias. Three paths
    // (value/sentinel/key) plus a second key appear in FILTER/ORDER BY; quoted
    // path names can double quotes. Fixed syntax is enumerated here.
    let display = f.name_bytes.checked_mul(2).and_then(|n| n.checked_add(
        "first_value() FILTER (WHERE  IS NOT NULL AND  IS NOT NULL) ORDER BY  DESC NULLS LAST".len()))
        .ok_or_else(|| b.overflow())?;
    b.add(
        f.aggregates
            .checked_mul(vec_peak(display, 1).ok_or_else(|| b.overflow())?),
    )?;
    // Root projections contribute one mapping per output; constructed_fields
    // covers named-struct decomposition, including nested/guarded constructors
    // which do not actually decompose. State constants add separate entries.
    let members = f
        .root_fields
        .checked_add(f.constructed_fields)
        .and_then(|n| n.checked_add(f.aggregates.checked_mul(2)?))
        .and_then(|n| n.checked_add(f.grouping_keys))
        .ok_or_else(|| b.overflow())?;
    // Complete selected property containers, with dense and raw hash-index
    // owners assigned individually. This same subtotal is used for clones.
    let property_bytes = property_owners(members, physical_nodes).ok_or_else(|| b.overflow())?;
    b.add(Some(property_bytes))?;
    b.add(physical_nodes.checked_mul(owner::<PlanProperties>()))?;
    let operator = [
        owner::<datafusion::physical_plan::projection::ProjectionExec>(),
        owner::<datafusion::physical_plan::filter::FilterExec>(),
        owner::<datafusion::physical_plan::aggregates::AggregateExec>(),
        owner::<datafusion_datasource::source::DataSourceExec>(),
    ]
    .into_iter()
    .max()
    .unwrap();
    b.add(physical_nodes.checked_mul(operator))?;
    // The admitted lowering uses existing local constructors, never builtin
    // LazyLocks. UserDefined/VariadicAny/Any(1) signatures contain no heap
    // vectors. Each call owns Arc<ScalarUDF> plus Arc<implementation>;
    // ProjectionMapping adds one local GetField implementation per decomposition.
    let builtins = f
        .fields
        .checked_mul(2)
        .and_then(|n| n.checked_add(f.path_components))
        .and_then(|n| n.checked_add(f.cases))
        .ok_or_else(|| b.overflow())?;
    let implementation = [
        owner::<datafusion::functions::core::getfield::GetFieldFunc>(),
        owner::<datafusion::functions::core::named_struct::NamedStructFunc>(),
        owner::<datafusion::functions::core::coalesce::CoalesceFunc>(),
    ]
    .into_iter()
    .max()
    .unwrap();
    b.add(builtins.checked_mul(owner::<datafusion::logical_expr::ScalarUDF>() + implementation))?;
    b.add(f.aggregates.checked_mul(2).and_then(|n| {
        n.checked_mul(
            owner::<datafusion::logical_expr::AggregateUDF>()
                + owner::<datafusion::functions_aggregate::first_last::FirstValue>(),
        )
    }))?;
    let retained = b.bytes;
    // Original logical expressions coexist with their lowered physical owners.
    // The per-operator builders clone expressions for schema calculation and
    // aggregate alias removal. Keep original, field resolution and builder
    // destination, plus aggregate display's clone at construction peak.
    b.add(
        base.checked_mul(size_of::<Expr>() + size_of::<String>())
            .and_then(|n| n.checked_mul(4)),
    )?;
    b.add(physical_nodes.checked_mul(owner::<LogicalPlan>()))?;
    b.add(vectors(
        physical_nodes,
        1,
        DefaultPhysicalPlanner::initial_plan_node_layout().size(),
    ))?;
    b.add(vectors(
        physical_nodes,
        1,
        size_of::<(Option<usize>, &LogicalPlan)>(),
    ))?;
    b.add(vectors(1, 1, size_of::<usize>()))?;
    // Source and destination equivalence containers overlap during project,
    // plus find_longest_permutation's clone. Reuse the complete retained
    // container portion rather than assuming projected properties share heaps.
    b.add(property_bytes.checked_mul(2))?;
    // EquivalenceGroup::project builds new_classes before consuming them
    // into the result. Each class owns its own IndexSet backing. The optional
    // indirect-projection AugmentedMapping is counted rather than assumed away.
    b.add(vectors(
        members,
        physical_nodes,
        size_of::<(usize, Physical, EquivalenceClass)>(),
    ))?;
    b.add(hash_partitions(members, physical_nodes, size_of::<usize>()))?;
    b.add(vectors(members, members, size_of::<(usize, Physical)>()))?;
    b.add(hash_partitions(members, members, size_of::<usize>()))?;
    b.add(vectors(
        members,
        physical_nodes,
        size_of::<datafusion::physical_expr::equivalence::ConstExpr>(),
    ))?;
    b.add(vectors(
        members,
        physical_nodes,
        size_of::<(
            usize,
            &Physical,
            (&Vec<(Physical, usize)>, Option<&EquivalenceClass>),
        )>(),
    ))?;
    b.add(hash_partitions(members, physical_nodes, size_of::<usize>()))?;
    // NamedStructFunc::struct_field_mapping and ProjectionMapping's caller:
    // literal_args, outer (Vec<ScalarValue>,index), each one-name inner Vec,
    // and cloned names coexist with the accessor args/literals counted above.
    b.add(vectors(
        f.fields.checked_mul(2).ok_or_else(|| b.overflow())?,
        f.fields,
        size_of::<Option<ScalarValue>>(),
    ))?;
    b.add(vectors(
        f.fields,
        f.fields,
        size_of::<(Vec<ScalarValue>, usize)>(),
    ))?;
    b.add(vectors(f.fields, f.fields, size_of::<ScalarValue>()))?;
    b.add(f.name_bytes.checked_mul(2))?;
    // Arrow conversion charges do not contain DFSchema's qualifiers. Every
    // root field gets an Option<TableReference>; all are None in this path.
    b.add(vectors(
        f.fields,
        physical_nodes,
        size_of::<Option<datafusion::common::TableReference>>(),
    ))?;
    b.add(physical_nodes.checked_mul(owner::<datafusion::common::DFSchema>()))?;
    // CASE root inputs have at most three columns in the sealed producer;
    // BTreeSet's selected Rust B=6 node holds eleven keys, so this is one
    // leaf, with parent pointer, parent index and length rounded to word
    // alignment. No edge array/internal node is reachable at this width.
    let rebuilt_cases = f
        .cases
        .checked_mul(copies)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| b.overflow())?;
    b.add(rebuilt_cases.checked_mul((3 + 11) * size_of::<usize>()))?;
    b.add(hash_partitions(
        rebuilt_cases.checked_mul(11).ok_or_else(|| b.overflow())?,
        rebuilt_cases,
        size_of::<usize>(),
    ))?;
    b.add(
        vectors(
            rebuilt_cases.checked_mul(11).ok_or_else(|| b.overflow())?,
            rebuilt_cases,
            size_of::<(usize, usize)>(),
        )
        .and_then(|n| n.checked_mul(3)),
    )?;
    // Dense column map, sorted iterator and stable-sort temporary above;
    // projection indices and original/projected body child slots were retained.
    // Both filters walk their entire (bounded) child subtrees for statistics.
    // NOT/IS NULL makes interval check_support false; no interval graph or
    // typed-null struct arrays are constructed. Unknown column stats are
    // inline Precision::Absent; version constants are inline Int64.
    let stats = physical_nodes.checked_mul(2).ok_or_else(|| b.overflow())?;
    let width = 11usize.max(
        f.aggregates
            .checked_mul(3)
            .and_then(|n| n.checked_add(f.grouping_keys))
            .ok_or_else(|| b.overflow())?,
    );
    b.add(stats.checked_mul(owner::<Statistics>()))?;
    b.add(vectors(
        stats.checked_mul(width).ok_or_else(|| b.overflow())?,
        stats,
        size_of::<ColumnStatistics>(),
    ))?;
    b.add(Some(
        2 * (owner::<
            std::cell::RefCell<std::collections::HashMap<(usize, Option<usize>), Arc<Statistics>>>,
        >()),
    ))?;
    b.add(hash_partitions(
        stats,
        2,
        size_of::<((usize, Option<usize>), Arc<Statistics>)>(),
    ))?;
    b.add(vectors(stats, stats, size_of::<Arc<Statistics>>()))?;
    // Invariants walk children, ordering, distribution and boolean property
    // Vecs. Four passes in debug, three in release; using four covers both.
    b.add(vectors(
        physical_nodes * 4,
        physical_nodes * 4,
        size_of::<Arc<dyn ExecutionPlan>>(),
    ))?;
    b.finish()?;
    Ok(PlanningEnvelope {
        retained,
        construction: b.bytes,
    })
}
struct Bound {
    bytes: usize,
    limits: TaskLimits,
}
impl Bound {
    fn overflow(&self) -> OperationFailure {
        ResourceExhausted {
            resource: Resource::MetadataAllocatedBytes,
            limit: self.limits.limit(Resource::MetadataAllocatedBytes),
            observed: usize::MAX,
        }
        .into()
    }
    fn add(&mut self, n: Option<usize>) -> Result<(), OperationFailure> {
        self.bytes = n
            .and_then(|n| self.bytes.checked_add(n))
            .ok_or_else(|| self.overflow())?;
        Ok(())
    }
    fn finish(&self) -> Result<(), OperationFailure> {
        if self.bytes > self.limits.limit(Resource::MetadataAllocatedBytes) {
            return Err(ResourceExhausted {
                resource: Resource::MetadataAllocatedBytes,
                limit: self.limits.limit(Resource::MetadataAllocatedBytes),
                observed: self.bytes,
            }
            .into());
        }
        Ok(())
    }
}
