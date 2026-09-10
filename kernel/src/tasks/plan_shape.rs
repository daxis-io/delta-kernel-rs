use std::error::Error;
use std::fmt::{self, Write};

use super::{Resource, ResourceExhausted, TaskLimits};
use crate::expressions::{ColumnName, Expression, Predicate, Scalar};
use crate::plans::ir::nodes::{Agg, Operator, ScanFile};
use crate::plans::ir::plan::Plan;
use crate::schema::{ArrayType, DataType, MapType, PrimitiveType, StructField, StructType};

// The largest field number in the plan wire grammar is 19 (two tag bytes). A protobuf
// varint or length prefix uses at most ten bytes. Charging both also bounds fixed-width fields.
const FIELD: usize = 2 + 10;
// Bound the Rust call stack independently of caller-configured resource allowances.
const MAX_NESTING: usize = 64;

/// Structural measurements of a borrowed plan, without retaining any of its allocations.
///
/// This is not an admitted plan or permission to execute one. It checks topology and bounded
/// traversal of the retained IR, rejects opaque/unknown expressions, and bounds the protobuf
/// representation. This size bound does not assert a lossless wire round-trip (the current
/// protobuf conversion omits the child of a cast). Operator type compatibility and canonical
/// allocation ownership are separate requirements. Nonempty public metadata and struct-patch
/// hash maps are rejected before traversal: their reported capacity is not a backing bound. The
/// encoded bound includes the `Operation::QueryPlan` envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanShape {
    nodes: usize,
    depth: usize,
    encoded_bytes: usize,
    work_units: usize,
}

impl PlanShape {
    /// Inspects `plan` under `limits`, before cloning or serializing any of its contents.
    ///
    /// `scratch` is caller-owned workspace with at least `plan.nodes.len()` entries. Only that
    /// prefix is written; on failure it may be partially overwritten. This method allocates
    /// nothing. The caller is responsible for admitting the workspace's backing storage. All
    /// input indices must precede their node and fit the protobuf's `u32`. Sources have no
    /// inputs, unary operators one, semi joins two, and unions at least one. Disconnected earlier
    /// nodes are allowed and inspected. Nested IR traversal is limited by `SchemaDepth` and a
    /// hard stack-safety ceiling of 64; topology uses `PlanDepth` independently.
    ///
    /// Returns a source-free error for invalid structure, unsupported expressions, exhausted
    /// budgets or insufficient scratch space. Successful inspection does not charge a task's
    /// cumulative ledger; a composing driver must account these reported work units and its
    /// simultaneous live storage. No source graph is dropped or retained by this call.
    pub fn check(
        plan: &Plan,
        limits: &TaskLimits,
        scratch: &mut [usize],
    ) -> Result<Self, PlanShapeError> {
        Self::check_inner(plan, limits, scratch, false)
    }

    /// Inspects structure and literal compatibility without allocating or retaining `plan`.
    ///
    /// Uses the same caller-owned `scratch` and resource limits as [`Self::check`]. Also checks
    /// Values row widths, scan constants, scalar container types/nullability, decimal ranges and
    /// array/map type tags. Struct literals must match declared field order. Nonempty metadata
    /// and field-patch hash maps return [`PlanShapeError::UnadmittedMap`] before traversal.
    /// Comparisons consume work units before inspecting their inputs.
    ///
    /// This is still a measurement, not an admitted plan. General operator/expression typing,
    /// schema-name uniqueness under Unicode lowercase, metadata-column schema invariants and
    /// canonical allocation ownership require separate admission. Standalone struct literals
    /// reject exact duplicate names; Unicode-fold duplicates are not checked here.
    ///
    /// Returns [`PlanShapeError::InvalidLiteral`] for incompatible literals or source constants,
    /// or the structural/resource errors documented by [`Self::check`].
    pub fn check_literals(
        plan: &Plan,
        limits: &TaskLimits,
        scratch: &mut [usize],
    ) -> Result<Self, PlanShapeError> {
        Self::check_inner(plan, limits, scratch, true)
    }

    /// Number of plan nodes inspected, including disconnected nodes.
    pub fn nodes(&self) -> usize {
        self.nodes
    }

    /// Longest path through the topology, with a source at depth one.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Conservative protobuf size bound, including the query-operation envelope.
    pub fn encoded_bytes(&self) -> usize {
        self.encoded_bytes
    }

    /// Traversal work, including input edges, formatting writes and admitted comparisons.
    pub fn work_units(&self) -> usize {
        self.work_units
    }

    fn check_inner(
        plan: &Plan,
        limits: &TaskLimits,
        scratch: &mut [usize],
        literals: bool,
    ) -> Result<Self, PlanShapeError> {
        if plan.nodes.is_empty() {
            return Err(PlanShapeError::Empty);
        }
        let mut walk = Walk {
            limits,
            encoded: 0,
            work: 0,
            schema_nodes: 0,
            literals,
        };
        walk.check(Resource::PlanNodes, plan.nodes.len())?;
        let depths =
            scratch
                .get_mut(..plan.nodes.len())
                .ok_or(PlanShapeError::ScratchTooSmall {
                    required: plan.nodes.len(),
                })?;
        walk.add(FIELD)?; // Operation.query_plan
        let mut depth = 0;
        for (index, node) in plan.nodes.iter().enumerate() {
            walk.enter(1)?; // Plan.nodes
            let arity = match &node.op {
                Operator::ScanParquet(_) | Operator::ScanJson(_) | Operator::Values(_) => {
                    node.inputs.is_empty()
                }
                Operator::Project(_)
                | Operator::Filter(_)
                | Operator::DynamicScan(_)
                | Operator::Aggregate(_) => node.inputs.len() == 1,
                Operator::SemiJoin(_) => node.inputs.len() == 2,
                Operator::UnionAll(_) => !node.inputs.is_empty(),
            };
            if !arity {
                return Err(PlanShapeError::InvalidArity { node: index });
            }
            walk.work(node.inputs.len())?;
            let mut node_depth: usize = 1;
            walk.add(FIELD)?; // packed PlanNode.inputs
            for &input in &node.inputs {
                if input >= index || u32::try_from(input).is_err() {
                    return Err(PlanShapeError::InvalidInput { node: index });
                }
                node_depth = node_depth.max(
                    depths[input]
                        .checked_add(1)
                        .ok_or_else(|| walk.exhausted(Resource::PlanDepth))?,
                );
                walk.add(5)?; // uint32, at most five varint bytes
            }
            walk.check(Resource::PlanDepth, node_depth)?;
            depth = depth.max(node_depth);
            depths[index] = node_depth;
            walk.operator(&node.op)?;
        }
        Ok(Self {
            nodes: plan.nodes.len(),
            depth,
            encoded_bytes: walk.encoded,
            work_units: walk.work,
        })
    }
}

/// A structural preflight failure containing no caller-owned payload or allocated message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlanShapeError {
    /// A plan must have a terminal node.
    Empty,
    /// An operator's input count is invalid.
    InvalidArity {
        /// Index of the invalid node.
        node: usize,
    },
    /// An input is not a preceding node or cannot be encoded without truncation.
    InvalidInput {
        /// Index of the node containing the input.
        node: usize,
    },
    /// Opaque callbacks and unknown executable expressions cannot enter task mode.
    UnsupportedExpression,
    /// A literal payload, declared type, Values row or scan constant is incompatible.
    InvalidLiteral,
    /// A nonempty public hash map has no producer-proven traversal bound.
    UnadmittedMap,
    /// A checked resource allowance or the nesting ceiling was exceeded.
    ResourceExhausted(ResourceExhausted),
    /// The caller-provided workspace cannot hold the topology depths.
    ScratchTooSmall {
        /// Required number of `usize` entries.
        required: usize,
    },
}

impl fmt::Display for PlanShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("empty plan"),
            Self::InvalidArity { node } => write!(f, "invalid input count at plan node {node}"),
            Self::InvalidInput { node } => write!(f, "invalid input index at plan node {node}"),
            Self::UnsupportedExpression => f.write_str("unsupported task expression"),
            Self::InvalidLiteral => f.write_str("invalid task literal"),
            Self::UnadmittedMap => f.write_str("task map requires producer admission"),
            Self::ResourceExhausted(error) => error.fmt(f),
            Self::ScratchTooSmall { required } => {
                write!(f, "plan scratch requires {required} entries")
            }
        }
    }
}

impl Error for PlanShapeError {}

struct Walk<'a> {
    limits: &'a TaskLimits,
    encoded: usize,
    work: usize,
    schema_nodes: usize,
    literals: bool,
}

impl Walk<'_> {
    fn exhausted(&self, resource: Resource) -> PlanShapeError {
        PlanShapeError::ResourceExhausted(ResourceExhausted {
            resource,
            limit: self.limits.limit(resource),
            observed: usize::MAX,
        })
    }

    fn check(&self, resource: Resource, observed: usize) -> Result<(), PlanShapeError> {
        let limit = self.limits.limit(resource);
        if observed > limit {
            return Err(PlanShapeError::ResourceExhausted(ResourceExhausted {
                resource,
                limit,
                observed,
            }));
        }
        Ok(())
    }

    fn work(&mut self, amount: usize) -> Result<(), PlanShapeError> {
        self.work = self
            .work
            .checked_add(amount)
            .ok_or_else(|| self.exhausted(Resource::WorkUnits))?;
        self.check(Resource::WorkUnits, self.work)
    }

    fn add(&mut self, amount: usize) -> Result<(), PlanShapeError> {
        self.encoded = self
            .encoded
            .checked_add(amount)
            .ok_or_else(|| self.exhausted(Resource::PlanEncodedBytes))?;
        self.check(Resource::PlanEncodedBytes, self.encoded)
    }

    fn enter(&mut self, depth: usize) -> Result<(), PlanShapeError> {
        let limit = self.limits.limit(Resource::SchemaDepth).min(MAX_NESTING);
        if depth > limit {
            return Err(PlanShapeError::ResourceExhausted(ResourceExhausted {
                resource: Resource::SchemaDepth,
                limit,
                observed: depth,
            }));
        }
        self.work(1)?;
        self.add(FIELD)
    }

    fn schema_enter(&mut self, depth: usize) -> Result<(), PlanShapeError> {
        self.schema_nodes = self
            .schema_nodes
            .checked_add(1)
            .ok_or_else(|| self.exhausted(Resource::SchemaNodes))?;
        self.check(Resource::SchemaNodes, self.schema_nodes)?;
        self.enter(depth)
    }

    fn string(&mut self, value: &str) -> Result<(), PlanShapeError> {
        self.add(FIELD)?;
        self.add(value.len())
    }

    fn column(&mut self, value: &ColumnName) -> Result<(), PlanShapeError> {
        self.add(FIELD)?;
        self.work(value.path().len())?;
        for segment in value.path() {
            self.string(segment)?;
        }
        Ok(())
    }

    fn operator(&mut self, op: &Operator) -> Result<(), PlanShapeError> {
        self.add(FIELD * 2)?; // PlanNode.op and Operator's oneof payload
        match op {
            Operator::ScanParquet(n) => self.scan(&n.files, &n.file_constant_columns, &n.schema),
            Operator::ScanJson(n) => self.scan(&n.files, &n.file_constant_columns, &n.schema),
            Operator::Values(n) => {
                self.schema(&n.schema, 1)?;
                for row in &n.rows {
                    self.enter(1)?;
                    if self.literals && row.len() != n.schema.fields().len() {
                        return Err(PlanShapeError::InvalidLiteral);
                    }
                    let mut fields = n.schema.fields();
                    for value in row {
                        let expected = if self.literals { fields.next() } else { None };
                        self.scalar_as(value, 1, expected)?;
                    }
                }
                Ok(())
            }
            Operator::Project(n) => {
                self.expression(&n.expr, 1)?;
                self.schema(&n.schema, 1)
            }
            Operator::Filter(n) => self.predicate(&n.predicate, 1),
            Operator::DynamicScan(n) => {
                self.schema(&n.schema, 1)?;
                self.add(FIELD)?; // file_type
                self.string(n.base_url.as_str())?;
                self.work(n.file_constant_columns.len())?;
                for name in &n.file_constant_columns {
                    self.string(name)?;
                }
                for column in [
                    &n.path_column,
                    &n.file_size_column,
                    &n.last_modified_column,
                    &n.dv_column,
                ] {
                    self.column(column)?;
                }
                Ok(())
            }
            Operator::Aggregate(n) => {
                self.schema(&n.schema, 1)?;
                self.work(n.group_by.len())?;
                for column in &n.group_by {
                    self.column(column)?;
                }
                for agg in &n.aggs {
                    self.enter(1)?;
                    self.add(FIELD)?; // Agg.func
                    match agg {
                        Agg::Min(c) | Agg::Max(c) | Agg::Sum(c) | Agg::Count(c) => {
                            self.column(c)?
                        }
                        Agg::CountStar => (),
                        Agg::MinNonNullBy(a) | Agg::MaxNonNullBy(a) => {
                            self.column(&a.value)?;
                            self.column(&a.null_sentinel)?;
                            self.column(&a.key)?;
                        }
                    }
                }
                Ok(())
            }
            Operator::SemiJoin(n) => {
                self.add(FIELD)?;
                self.work(n.probe_keys.len())?;
                self.work(n.build_keys.len())?;
                for column in n.probe_keys.iter().chain(&n.build_keys) {
                    self.column(column)?;
                }
                Ok(())
            }
            Operator::UnionAll(_) => Ok(()),
        }
    }

    fn scan(
        &mut self,
        files: &[ScanFile],
        constants: &[String],
        schema: &StructType,
    ) -> Result<(), PlanShapeError> {
        self.schema(schema, 1)?;
        self.work(constants.len())?;
        for (index, name) in constants.iter().enumerate() {
            self.string(name)?;
            if self.literals {
                for previous in &constants[..index] {
                    if self.same_string(name, previous)? {
                        return Err(PlanShapeError::InvalidLiteral);
                    }
                }
                self.literal_field(schema, name)?
                    .ok_or(PlanShapeError::InvalidLiteral)?;
            }
        }
        for file in files {
            self.enter(1)?;
            self.add(FIELD * 3)?; // FileMeta message and its numeric fields
            self.string(file.meta.location.as_str())?;
            if self.literals && file.file_constants.len() != constants.len() {
                return Err(PlanShapeError::InvalidLiteral);
            }
            for (index, value) in file.file_constants.iter().enumerate() {
                let expected = if self.literals {
                    let name = &constants[index];
                    self.literal_field(schema, name)?
                } else {
                    None
                };
                self.scalar_as(value, 1, expected)?;
            }
        }
        Ok(())
    }

    fn schema(&mut self, schema: &StructType, depth: usize) -> Result<(), PlanShapeError> {
        self.schema_enter(depth)?;
        for field in schema.fields() {
            self.field(field, depth)?;
        }
        Ok(())
    }

    fn field(&mut self, field: &StructField, depth: usize) -> Result<(), PlanShapeError> {
        self.schema_enter(depth)?;
        // Reported HashMap capacity can shrink after deletion without releasing its buckets.
        // Even iteration or an absent-key lookup needs a bound minted by the producer.
        if !field.metadata.is_empty() {
            return Err(PlanShapeError::UnadmittedMap);
        }
        self.string(&field.name)?;
        self.data_type(&field.data_type, depth)?;
        self.add(FIELD) // nullable
    }

    fn data_type(&mut self, ty: &DataType, depth: usize) -> Result<(), PlanShapeError> {
        self.schema_enter(depth)?;
        match ty {
            DataType::Primitive(p) => {
                self.add(FIELD)?; // PrimitiveType
                match p {
                    PrimitiveType::Decimal(d) => {
                        self.decimal_type(d)?;
                        self.add(FIELD * 3)
                    }
                    #[cfg(feature = "geo-type-in-dev")]
                    PrimitiveType::Geometry(g) => {
                        self.add(FIELD)?;
                        self.string(g.crs())
                    }
                    #[cfg(feature = "geo-type-in-dev")]
                    PrimitiveType::Geography(g) => {
                        self.add(FIELD * 2)?;
                        self.string(g.crs())
                    }
                    PrimitiveType::String
                    | PrimitiveType::Long
                    | PrimitiveType::Integer
                    | PrimitiveType::Short
                    | PrimitiveType::Byte
                    | PrimitiveType::Float
                    | PrimitiveType::Double
                    | PrimitiveType::Boolean
                    | PrimitiveType::Binary
                    | PrimitiveType::Date
                    | PrimitiveType::Timestamp
                    | PrimitiveType::TimestampNtz
                    | PrimitiveType::Void
                    | PrimitiveType::IntervalYearMonth
                    | PrimitiveType::IntervalDayTime => self.add(FIELD),
                }
            }
            DataType::Array(a) => self.array_type(a, depth),
            DataType::Map(m) => self.map_type(m, depth),
            DataType::Struct(s) | DataType::Variant(s) => self.schema(s, depth + 1),
        }
    }

    fn array_type(&mut self, ty: &ArrayType, depth: usize) -> Result<(), PlanShapeError> {
        self.schema_enter(depth)?;
        if self.literals && ty.type_name != "array" {
            return Err(PlanShapeError::InvalidLiteral);
        }
        self.add(FIELD)?;
        self.data_type(ty.element_type(), depth + 1)
    }

    fn map_type(&mut self, ty: &MapType, depth: usize) -> Result<(), PlanShapeError> {
        self.schema_enter(depth)?;
        if self.literals && ty.type_name != "map" {
            return Err(PlanShapeError::InvalidLiteral);
        }
        self.add(FIELD)?;
        self.data_type(ty.key_type(), depth + 1)?;
        self.data_type(ty.value_type(), depth + 1)
    }

    fn scalar(&mut self, value: &Scalar, depth: usize) -> Result<(), PlanShapeError> {
        self.scalar_as(value, depth, None)
    }

    fn scalar_as(
        &mut self,
        value: &Scalar,
        depth: usize,
        expected: Option<&StructField>,
    ) -> Result<(), PlanShapeError> {
        self.scalar_typed(
            value,
            depth,
            expected.map(|f| (f.data_type(), f.is_nullable())),
        )
    }

    fn scalar_typed(
        &mut self,
        value: &Scalar,
        depth: usize,
        expected: Option<(&DataType, bool)>,
    ) -> Result<(), PlanShapeError> {
        self.enter(depth)?;
        if self.literals {
            if let Some((ty, nullable)) = expected {
                if (!nullable && value.is_null()) || !self.scalar_type_matches(value, ty, depth)? {
                    return Err(PlanShapeError::InvalidLiteral);
                }
            }
        }
        match value {
            Scalar::String(s) => self.string(s),
            Scalar::Binary(b) => {
                self.add(FIELD)?;
                self.add(b.len())
            }
            Scalar::Decimal(d) => {
                self.decimal_type(d.ty())?;
                if self.literals && d.bits().unsigned_abs() >= 10u128.pow(u32::from(d.precision()))
                {
                    return Err(PlanShapeError::InvalidLiteral);
                }
                self.add(FIELD * 5 + 16)
            }
            Scalar::Null(ty) => self.data_type(ty, depth + 1),
            Scalar::Struct(s) => {
                self.add(FIELD)?;
                for field in s.fields() {
                    self.field(field, depth + 1)?;
                }
                if self.literals && s.fields().len() != s.values().len() {
                    return Err(PlanShapeError::InvalidLiteral);
                }
                if self.literals {
                    for (index, field) in s.fields().iter().enumerate() {
                        for previous in &s.fields()[..index] {
                            if self.same_string(field.name(), previous.name())? {
                                return Err(PlanShapeError::InvalidLiteral);
                            }
                        }
                    }
                }
                for (index, value) in s.values().iter().enumerate() {
                    let expected = if self.literals {
                        s.fields().get(index)
                    } else {
                        None
                    };
                    self.scalar_as(value, depth + 1, expected)?;
                }
                Ok(())
            }
            Scalar::Array(a) => {
                self.add(FIELD)?;
                self.array_type(a.array_type(), depth + 1)?;
                for value in a.array_elements() {
                    self.scalar_typed(
                        value,
                        depth + 1,
                        Some((
                            a.array_type().element_type(),
                            a.array_type().contains_null(),
                        )),
                    )?;
                }
                Ok(())
            }
            Scalar::Map(m) => {
                self.add(FIELD)?;
                self.map_type(m.map_type(), depth + 1)?;
                for (key, value) in m.pairs() {
                    self.add(FIELD)?;
                    self.scalar_typed(key, depth + 1, Some((m.map_type().key_type(), false)))?;
                    self.scalar_typed(
                        value,
                        depth + 1,
                        Some((
                            m.map_type().value_type(),
                            m.map_type().value_contains_null(),
                        )),
                    )?;
                }
                Ok(())
            }
            Scalar::Integer(_)
            | Scalar::Long(_)
            | Scalar::Short(_)
            | Scalar::Byte(_)
            | Scalar::Float(_)
            | Scalar::Double(_)
            | Scalar::Boolean(_)
            | Scalar::Timestamp(_)
            | Scalar::TimestampNtz(_)
            | Scalar::Date(_)
            | Scalar::IntervalYearMonth(_)
            | Scalar::IntervalDayTime(_) => self.add(FIELD),
        }
    }

    fn literal_field<'b>(
        &mut self,
        schema: &'b StructType,
        name: &str,
    ) -> Result<Option<&'b StructField>, PlanShapeError> {
        // ponytail: metered linear lookup; use an admitted index if wide literal sources need it.
        for field in schema.fields() {
            if self.same_string(name, field.name())? {
                return Ok(Some(field));
            }
        }
        Ok(None)
    }

    fn decimal_type(&self, ty: &crate::schema::DecimalType) -> Result<(), PlanShapeError> {
        if self.literals
            && (ty.precision() == 0 || ty.precision() > 38 || ty.scale() > ty.precision())
        {
            return Err(PlanShapeError::InvalidLiteral);
        }
        Ok(())
    }

    fn comparison(&mut self, depth: usize) -> Result<(), PlanShapeError> {
        let limit = self.limits.limit(Resource::SchemaDepth).min(MAX_NESTING);
        if depth > limit {
            return Err(PlanShapeError::ResourceExhausted(ResourceExhausted {
                resource: Resource::SchemaDepth,
                limit,
                observed: depth,
            }));
        }
        self.work(1)
    }

    fn same_string(&mut self, left: &str, right: &str) -> Result<bool, PlanShapeError> {
        self.work(1)?;
        self.work(left.len())?;
        self.work(right.len())?;
        Ok(left == right)
    }

    fn scalar_type_matches(
        &mut self,
        scalar: &Scalar,
        ty: &DataType,
        depth: usize,
    ) -> Result<bool, PlanShapeError> {
        self.comparison(depth)?;
        let primitive = match scalar {
            Scalar::Integer(_) => DataType::INTEGER,
            Scalar::Long(_) => DataType::LONG,
            Scalar::Short(_) => DataType::SHORT,
            Scalar::Byte(_) => DataType::BYTE,
            Scalar::Float(_) => DataType::FLOAT,
            Scalar::Double(_) => DataType::DOUBLE,
            Scalar::String(_) => DataType::STRING,
            Scalar::Boolean(_) => DataType::BOOLEAN,
            Scalar::Timestamp(_) => DataType::TIMESTAMP,
            Scalar::TimestampNtz(_) => DataType::TIMESTAMP_NTZ,
            Scalar::IntervalYearMonth(_) => DataType::INTERVAL_YEAR_MONTH,
            Scalar::IntervalDayTime(_) => DataType::INTERVAL_DAY_TIME,
            Scalar::Date(_) => DataType::DATE,
            Scalar::Binary(_) => DataType::BINARY,
            Scalar::Decimal(d) => DataType::from(*d.ty()),
            Scalar::Null(actual) => return self.same_type(actual, ty, depth),
            Scalar::Struct(s) => {
                let DataType::Struct(expected) = ty else {
                    return Ok(false);
                };
                if s.fields().len() != expected.fields().len() {
                    return Ok(false);
                }
                for (actual, expected) in s.fields().iter().zip(expected.fields()) {
                    if !self.same_field(actual, expected, depth + 1)? {
                        return Ok(false);
                    }
                }
                return Ok(true);
            }
            Scalar::Array(a) => {
                return match ty {
                    DataType::Array(expected) => self.same_array(a.array_type(), expected, depth),
                    _ => Ok(false),
                }
            }
            Scalar::Map(m) => {
                return match ty {
                    DataType::Map(expected) => self.same_map(m.map_type(), expected, depth),
                    _ => Ok(false),
                }
            }
        };
        // The scalar branch constructs only an inline primitive type, never a container.
        Ok(&primitive == ty)
    }

    fn same_type(
        &mut self,
        left: &DataType,
        right: &DataType,
        depth: usize,
    ) -> Result<bool, PlanShapeError> {
        self.comparison(depth)?;
        match (left, right) {
            (DataType::Primitive(a), DataType::Primitive(b)) => {
                #[cfg(feature = "geo-type-in-dev")]
                match (a, b) {
                    (PrimitiveType::Geometry(a), PrimitiveType::Geometry(b)) => {
                        return self.same_string(a.crs(), b.crs());
                    }
                    (PrimitiveType::Geography(a), PrimitiveType::Geography(b)) => {
                        self.work(a.crs().len())?;
                        self.work(b.crs().len())?;
                    }
                    _ => (),
                }
                Ok(a == b)
            }
            (DataType::Array(a), DataType::Array(b)) => self.same_array(a, b, depth),
            (DataType::Map(a), DataType::Map(b)) => self.same_map(a, b, depth),
            (DataType::Struct(a), DataType::Struct(b))
            | (DataType::Variant(a), DataType::Variant(b)) => {
                if a.fields().len() != b.fields().len() {
                    return Ok(false);
                }
                for (a, b) in a.fields().zip(b.fields()) {
                    if !self.same_field(a, b, depth + 1)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn same_array(
        &mut self,
        a: &ArrayType,
        b: &ArrayType,
        depth: usize,
    ) -> Result<bool, PlanShapeError> {
        self.comparison(depth)?;
        Ok(self.same_string(&a.type_name, &b.type_name)?
            && a.contains_null() == b.contains_null()
            && self.same_type(a.element_type(), b.element_type(), depth + 1)?)
    }

    fn same_map(&mut self, a: &MapType, b: &MapType, depth: usize) -> Result<bool, PlanShapeError> {
        self.comparison(depth)?;
        Ok(self.same_string(&a.type_name, &b.type_name)?
            && a.value_contains_null() == b.value_contains_null()
            && self.same_type(a.key_type(), b.key_type(), depth + 1)?
            && self.same_type(a.value_type(), b.value_type(), depth + 1)?)
    }

    fn same_field(
        &mut self,
        a: &StructField,
        b: &StructField,
        depth: usize,
    ) -> Result<bool, PlanShapeError> {
        self.comparison(depth)?;
        if !a.metadata.is_empty() || !b.metadata.is_empty() {
            return Err(PlanShapeError::UnadmittedMap);
        }
        Ok(self.same_string(a.name(), b.name())?
            && a.is_nullable() == b.is_nullable()
            && self.same_type(a.data_type(), b.data_type(), depth)?)
    }

    fn expression(&mut self, expr: &Expression, depth: usize) -> Result<(), PlanShapeError> {
        self.enter(depth)?;
        match expr {
            Expression::Opaque(_) | Expression::Unknown(_) => {
                Err(PlanShapeError::UnsupportedExpression)
            }
            Expression::Literal(s) => self.scalar(s, depth + 1),
            Expression::Column(c) => self.column(c),
            Expression::Predicate(p) => self.predicate(p, depth + 1),
            Expression::Struct(exprs, nullability) => {
                self.add(FIELD)?;
                for expr in exprs {
                    self.expression(expr, depth + 1)?;
                }
                if let Some(expr) = nullability {
                    self.expression(expr, depth + 1)?;
                }
                Ok(())
            }
            Expression::StructPatch(patch) => {
                if !patch.field_patches.is_empty() {
                    return Err(PlanShapeError::UnadmittedMap);
                }
                self.add(FIELD)?;
                if let Some(path) = &patch.input_path {
                    self.column(path)?;
                }
                for expr in patch.prepended_fields.iter().chain(&patch.appended_fields) {
                    self.expression(expr, depth + 1)?;
                }
                Ok(())
            }
            Expression::Unary(u) => {
                self.add(FIELD * 2)?;
                self.expression(&u.expr, depth + 1)
            }
            Expression::Binary(b) => {
                self.add(FIELD * 2)?;
                self.expression(&b.left, depth + 1)?;
                self.expression(&b.right, depth + 1)
            }
            Expression::Variadic(v) => {
                self.add(FIELD * 2)?;
                for expr in &v.exprs {
                    self.expression(expr, depth + 1)?;
                }
                Ok(())
            }
            Expression::ParseJson(p) => {
                self.add(FIELD)?;
                self.expression(&p.json_expr, depth + 1)?;
                self.schema(&p.output_schema, depth + 1)
            }
            Expression::MapToStruct(m) => {
                self.add(FIELD)?;
                self.expression(&m.map_expr, depth + 1)
            }
            Expression::Cast(c) => {
                // The current wire uses an Unknown string for Cast. Still inspect the retained
                // child and target first, including nodes that this lossy wire conversion omits.
                self.expression(&c.expr, depth + 1)?;
                self.data_type(&c.target, depth + 1)?;
                self.add(FIELD)?;
                self.formatted(format_args!("cast_to_{}", c.target))
            }
        }
    }

    fn predicate(&mut self, pred: &Predicate, depth: usize) -> Result<(), PlanShapeError> {
        self.enter(depth)?;
        match pred {
            Predicate::Opaque(_) | Predicate::Unknown(_) => {
                Err(PlanShapeError::UnsupportedExpression)
            }
            Predicate::BooleanExpression(e) => self.expression(e, depth + 1),
            Predicate::Not(p) => self.predicate(p, depth + 1),
            Predicate::Unary(u) => {
                self.add(FIELD * 2)?;
                self.expression(&u.expr, depth + 1)
            }
            Predicate::Binary(b) => {
                self.add(FIELD * 2)?;
                self.expression(&b.left, depth + 1)?;
                self.expression(&b.right, depth + 1)
            }
            Predicate::Junction(j) => {
                self.add(FIELD * 2)?;
                for pred in &j.preds {
                    self.predicate(pred, depth + 1)?;
                }
                Ok(())
            }
        }
    }

    fn formatted(&mut self, args: fmt::Arguments<'_>) -> Result<(), PlanShapeError> {
        let mut writer = SizeWriter {
            walk: self,
            error: None,
        };
        let result = writer.write_fmt(args);
        match writer.error {
            Some(error) => Err(error),
            None if result.is_err() => Err(PlanShapeError::UnsupportedExpression),
            None => Ok(()),
        }
    }
}

struct SizeWriter<'a, 'b> {
    walk: &'a mut Walk<'b>,
    error: Option<PlanShapeError>,
}

impl Write for SizeWriter<'_, '_> {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        self.walk
            .work(1)
            .and_then(|()| self.walk.add(value.len()))
            .map_err(|error| {
                self.error = Some(error);
                fmt::Error
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_counters_reject_overflow_even_with_maximum_limits() {
        let limits = TaskLimits::qualification()
            .with_limit(Resource::PlanEncodedBytes, usize::MAX)
            .with_limit(Resource::WorkUnits, usize::MAX)
            .with_limit(Resource::SchemaNodes, usize::MAX);
        let mut walk = Walk {
            limits: &limits,
            encoded: usize::MAX,
            work: usize::MAX,
            schema_nodes: usize::MAX,
            literals: false,
        };
        for (resource, result) in [
            (Resource::PlanEncodedBytes, walk.add(1)),
            (Resource::WorkUnits, walk.work(1)),
            (Resource::SchemaNodes, walk.schema_enter(1)),
        ] {
            assert_eq!(
                result,
                Err(PlanShapeError::ResourceExhausted(ResourceExhausted {
                    resource,
                    limit: usize::MAX,
                    observed: usize::MAX,
                }))
            );
        }
    }
}
