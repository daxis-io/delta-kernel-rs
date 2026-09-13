//! Borrowed resource inputs for the two closed JSON plan producers.
//!
//! This is not an admission constructor: it accepts only an existing AdmittedPlan
//! with immutable JSON provenance. It allocates no IR, schema, names or work queue.
//! Counting repeated Arc references deliberately covers repeated lowering work.

use delta_kernel::expressions::{ColumnName, Expression, Predicate, Scalar, VariadicExpressionOp};
use delta_kernel::plans::ir::nodes::{Agg, Operator};
use delta_kernel::schema::{DataType, StructType};
use delta_kernel::tasks::{
    AdmittedPlan, OperationFailure, Resource, ResourceExhausted, TaskLimits,
};

#[derive(Default, Debug)]
pub(crate) struct ClosedPlanFacts {
    pub nodes: usize,
    pub edges: usize,
    pub fields: usize,
    pub root_fields: usize,
    pub max_root_fields: usize,
    pub max_struct_fields: usize,
    pub constructed_fields: usize,
    pub types: usize,
    pub expressions: usize,
    pub path_components: usize,
    pub name_bytes: usize,
    pub literal_bytes: usize,
    pub files: usize,
    pub file_url_bytes: usize,
    pub aggregates: usize,
    pub grouping_keys: usize,
    pub struct_patches: usize,
    pub cases: usize,
    pub case_branches: usize,
    pub case_depth: usize,
    pub max_name_bytes: usize,
    pub coalesce_utf8: usize,
    pub coalesce_int32: usize,
    pub boolean_junctions: usize,
    pub work: usize,
}

impl ClosedPlanFacts {
    pub(crate) fn inspect(
        plan: &AdmittedPlan,
        limits: TaskLimits,
    ) -> Result<Self, OperationFailure> {
        let manifest = plan
            .log_identity_manifest()
            .ok_or_else(OperationFailure::malformed_response)?;
        if manifest.files().is_empty() || !matches!(plan.plan().nodes.len(), 2 | 7) {
            return Err(OperationFailure::malformed_response());
        }
        let mut facts = Self::default();
        for (index, node) in plan.plan().nodes.iter().enumerate() {
            facts.step(1, limits)?;
            add(&mut facts.nodes, 1, limits)?;
            add(&mut facts.edges, node.inputs.len(), limits)?;
            if node.inputs.iter().any(|input| *input >= index) {
                return Err(OperationFailure::malformed_response());
            }
            let input_schema = node
                .inputs
                .first()
                .and_then(|parent| crate::plan::closed_input_schema(plan.plan(), *parent));
            match &node.op {
                Operator::ScanJson(scan) => {
                    if !node.inputs.is_empty() || scan.files.len() != manifest.files().len() {
                        return Err(OperationFailure::malformed_response());
                    }
                    facts.schema(&scan.schema, 0, limits)?;
                    for name in &scan.file_constant_columns {
                        facts.name(name, limits)?;
                    }
                    for file in &scan.files {
                        facts.step(1, limits)?;
                        add(&mut facts.files, 1, limits)?;
                        add(
                            &mut facts.file_url_bytes,
                            file.meta.location.as_str().len(),
                            limits,
                        )?;
                        for value in &file.file_constants {
                            facts.scalar(value, 0, limits)?;
                        }
                    }
                }
                Operator::Project(project) => {
                    facts.schema(&project.schema, 0, limits)?;
                    facts.expression(
                        &project.expr,
                        0,
                        0,
                        input_schema.ok_or_else(OperationFailure::malformed_response)?,
                        limits,
                    )?;
                    // Top-level StructColumns::into_guarded_columns clones a
                    // guard into every output field; packed nested structs use
                    // one CASE, counted by expression() below.
                    let guarded = match project.expr.as_ref() {
                        Expression::Struct(_, guard) => guard.is_some(),
                        Expression::StructPatch(patch) => patch.input_path().is_some(),
                        _ => false,
                    };
                    if guarded {
                        let extra = project.schema.num_fields().saturating_sub(1);
                        add(&mut facts.cases, extra, limits)?;
                        add(&mut facts.case_branches, extra, limits)?;
                    }
                }
                Operator::Filter(filter) => facts.predicate(
                    &filter.predicate,
                    0,
                    0,
                    input_schema.ok_or_else(OperationFailure::malformed_response)?,
                    limits,
                )?,
                Operator::Aggregate(aggregate) => {
                    facts.schema(&aggregate.schema, 0, limits)?;
                    for key in &aggregate.group_by {
                        add(&mut facts.grouping_keys, 1, limits)?;
                        facts.path(key, limits)?;
                    }
                    for aggregate in &aggregate.aggs {
                        facts.step(1, limits)?;
                        add(&mut facts.aggregates, 1, limits)?;
                        let Agg::MaxNonNullBy(operands) = aggregate else {
                            return Err(OperationFailure::malformed_response());
                        };
                        for path in [&operands.value, &operands.null_sentinel, &operands.key] {
                            facts.path(path, limits)?;
                        }
                    }
                }
                // No-checkpoint producers eliminate union/join arms in Kernel.
                // This inventory never broadens the executor's admitted subset.
                _ => return Err(OperationFailure::malformed_response()),
            }
        }
        if facts.files != manifest.files().len() {
            return Err(OperationFailure::malformed_response());
        }
        Ok(facts)
    }

    fn step(&mut self, amount: usize, limits: TaskLimits) -> Result<(), OperationFailure> {
        add(&mut self.work, amount, limits)?;
        check(Resource::WorkUnits, self.work, limits)
    }
    fn name(&mut self, name: &str, limits: TaskLimits) -> Result<(), OperationFailure> {
        self.max_name_bytes = self.max_name_bytes.max(name.len());
        self.step(name.len(), limits)?;
        add(&mut self.name_bytes, name.len(), limits)
    }
    fn path(&mut self, path: &ColumnName, limits: TaskLimits) -> Result<(), OperationFailure> {
        for part in path.as_ref() {
            add(&mut self.path_components, 1, limits)?;
            self.name(part, limits)?;
        }
        Ok(())
    }
    fn schema(
        &mut self,
        schema: &StructType,
        depth: usize,
        limits: TaskLimits,
    ) -> Result<(), OperationFailure> {
        depth_check(depth, limits)?;
        self.step(1, limits)?;
        self.max_struct_fields = self.max_struct_fields.max(schema.num_fields());
        if depth == 0 {
            add(&mut self.root_fields, schema.num_fields(), limits)?;
            self.max_root_fields = self.max_root_fields.max(schema.num_fields());
        }
        for field in schema.fields() {
            if !field.metadata.is_empty() {
                return Err(OperationFailure::malformed_response());
            }
            add(&mut self.fields, 1, limits)?;
            self.name(&field.name, limits)?;
            self.data_type(&field.data_type, depth + 1, limits)?;
        }
        Ok(())
    }
    fn data_type(
        &mut self,
        data_type: &DataType,
        depth: usize,
        limits: TaskLimits,
    ) -> Result<(), OperationFailure> {
        depth_check(depth, limits)?;
        self.step(1, limits)?;
        add(&mut self.types, 1, limits)?;
        match data_type {
            DataType::Primitive(_) => Ok(()),
            DataType::Struct(schema) => self.schema(schema, depth, limits),
            DataType::Array(array) => self.data_type(array.element_type(), depth + 1, limits),
            DataType::Map(map) => {
                self.data_type(map.key_type(), depth + 1, limits)?;
                self.data_type(map.value_type(), depth + 1, limits)
            }
            _ => Err(OperationFailure::malformed_response()),
        }
    }
    fn scalar(
        &mut self,
        value: &Scalar,
        depth: usize,
        limits: TaskLimits,
    ) -> Result<(), OperationFailure> {
        self.step(1, limits)?;
        match value {
            Scalar::Integer(_) | Scalar::Long(_) | Scalar::Boolean(_) => Ok(()),
            Scalar::String(value) => {
                self.step(value.len(), limits)?;
                add(&mut self.literal_bytes, value.len(), limits)
            }
            Scalar::Null(data_type) => self.data_type(data_type, depth + 1, limits),
            _ => Err(OperationFailure::malformed_response()),
        }
    }
    fn expression(
        &mut self,
        expression: &Expression,
        depth: usize,
        case_depth: usize,
        input_schema: &StructType,
        limits: TaskLimits,
    ) -> Result<(), OperationFailure> {
        depth_check(depth, limits)?;
        self.step(1, limits)?;
        add(&mut self.expressions, 1, limits)?;
        let introduces_case = match expression {
            Expression::Struct(_, guard) => guard.is_some(),
            Expression::StructPatch(patch) => patch.input_path().is_some(),
            Expression::Variadic(v) => v.op == VariadicExpressionOp::Coalesce && v.exprs.len() > 1,
            _ => false,
        };
        let case_depth = case_depth
            .checked_add(usize::from(introduces_case))
            .ok_or_else(OperationFailure::malformed_response)?;
        self.case_depth = self.case_depth.max(case_depth);
        // The only nested CASE is the key's DV guard around primitive
        // coalesces. Bound projected-body reconstruction before building it.
        if case_depth > 2 {
            return Err(OperationFailure::malformed_response());
        }
        match expression {
            Expression::Literal(value) => self.scalar(value, depth, limits)?,
            Expression::Column(path) => self.path(path, limits)?,
            Expression::Predicate(predicate) => {
                self.predicate(predicate, depth + 1, case_depth, input_schema, limits)?
            }
            Expression::Struct(fields, guard) => {
                add(&mut self.constructed_fields, fields.len(), limits)?;
                for field in fields {
                    self.expression(field, depth + 1, case_depth, input_schema, limits)?;
                }
                if let Some(guard) = guard {
                    add(&mut self.cases, 1, limits)?;
                    add(&mut self.case_branches, 1, limits)?;
                    self.expression(guard, depth + 1, case_depth, input_schema, limits)?;
                }
            }
            Expression::StructPatch(patch) => {
                add(&mut self.struct_patches, 1, limits)?;
                // Pass-through patch fields generate physical accessors even
                // though no child Expression exists in Kernel IR. Counting
                // dropped fields too is conservative. Inserted fields below
                // add their own slots as well as their recursive expressions.
                let source = match patch.input_path() {
                    Some(path) => match borrowed_column_type(input_schema, path) {
                        Some(DataType::Struct(schema)) => schema.as_ref(),
                        _ => return Err(OperationFailure::malformed_response()),
                    },
                    None => input_schema,
                };
                add(&mut self.constructed_fields, source.num_fields(), limits)?;
                add(
                    &mut self.constructed_fields,
                    patch.prepended_fields.len(),
                    limits,
                )?;
                add(
                    &mut self.constructed_fields,
                    patch.appended_fields.len(),
                    limits,
                )?;
                if let Some(path) = patch.input_path() {
                    self.path(path, limits)?;
                    add(&mut self.cases, 1, limits)?;
                    add(&mut self.case_branches, 1, limits)?;
                }
                for field in patch.prepended_fields.iter().chain(&patch.appended_fields) {
                    self.expression(field, depth + 1, case_depth, input_schema, limits)?;
                }
                for (name, patch) in &patch.field_patches {
                    self.name(name, limits)?;
                    add(&mut self.constructed_fields, patch.insertions.len(), limits)?;
                    for insertion in &patch.insertions {
                        self.expression(insertion, depth + 1, case_depth, input_schema, limits)?;
                    }
                }
            }
            Expression::Unary(unary) => {
                self.expression(&unary.expr, depth + 1, case_depth, input_schema, limits)?
            }
            Expression::Binary(binary) => {
                self.expression(&binary.left, depth + 1, case_depth, input_schema, limits)?;
                self.expression(&binary.right, depth + 1, case_depth, input_schema, limits)?;
            }
            Expression::Variadic(variadic) => {
                if variadic.op == VariadicExpressionOp::Coalesce && variadic.exprs.len() > 1 {
                    // These exact producers coalesce only two typed columns:
                    // path/storageType/pathOrInlineDv strings, or DV offset i32.
                    // Type drift is rejected before any DF expression is built.
                    let [left, right] = variadic.exprs.as_slice() else {
                        return Err(OperationFailure::malformed_response());
                    };
                    let (Expression::Column(left), Expression::Column(right)) = (left, right)
                    else {
                        return Err(OperationFailure::malformed_response());
                    };
                    let left = borrowed_column_type(input_schema, left)
                        .ok_or_else(OperationFailure::malformed_response)?;
                    let right = borrowed_column_type(input_schema, right)
                        .ok_or_else(OperationFailure::malformed_response)?;
                    if left != right {
                        return Err(OperationFailure::malformed_response());
                    }
                    match left {
                        DataType::Primitive(delta_kernel::schema::PrimitiveType::String) => {
                            add(&mut self.coalesce_utf8, 1, limits)?
                        }
                        DataType::Primitive(delta_kernel::schema::PrimitiveType::Integer) => {
                            add(&mut self.coalesce_int32, 1, limits)?
                        }
                        _ => return Err(OperationFailure::malformed_response()),
                    }
                    add(&mut self.cases, 1, limits)?;
                    add(&mut self.case_branches, variadic.exprs.len() - 1, limits)?;
                }
                for expression in &variadic.exprs {
                    self.expression(expression, depth + 1, case_depth, input_schema, limits)?;
                }
            }
            _ => return Err(OperationFailure::malformed_response()),
        }
        Ok(())
    }
    fn predicate(
        &mut self,
        predicate: &Predicate,
        depth: usize,
        case_depth: usize,
        input_schema: &StructType,
        limits: TaskLimits,
    ) -> Result<(), OperationFailure> {
        depth_check(depth, limits)?;
        self.step(1, limits)?;
        add(&mut self.expressions, 1, limits)?;
        match predicate {
            Predicate::BooleanExpression(expression) => {
                self.expression(expression, depth + 1, case_depth, input_schema, limits)?
            }
            Predicate::Not(predicate) => {
                self.predicate(predicate, depth + 1, case_depth, input_schema, limits)?
            }
            Predicate::Unary(unary) => {
                self.expression(&unary.expr, depth + 1, case_depth, input_schema, limits)?
            }
            Predicate::Binary(binary) => {
                self.expression(&binary.left, depth + 1, case_depth, input_schema, limits)?;
                self.expression(&binary.right, depth + 1, case_depth, input_schema, limits)?;
            }
            Predicate::Junction(junction) => {
                add(
                    &mut self.boolean_junctions,
                    junction.preds.len().saturating_sub(1),
                    limits,
                )?;
                for predicate in &junction.preds {
                    self.predicate(predicate, depth + 1, case_depth, input_schema, limits)?;
                }
            }
            _ => return Err(OperationFailure::malformed_response()),
        }
        Ok(())
    }
}
fn add(value: &mut usize, amount: usize, limits: TaskLimits) -> Result<(), OperationFailure> {
    *value = value.checked_add(amount).ok_or_else(|| {
        OperationFailure::from(ResourceExhausted {
            resource: Resource::WorkUnits,
            limit: limits.limit(Resource::WorkUnits),
            observed: usize::MAX,
        })
    })?;
    Ok(())
}
fn check(resource: Resource, observed: usize, limits: TaskLimits) -> Result<(), OperationFailure> {
    if observed > limits.limit(resource) {
        return Err(ResourceExhausted {
            resource,
            limit: limits.limit(resource),
            observed,
        }
        .into());
    }
    Ok(())
}
fn depth_check(depth: usize, limits: TaskLimits) -> Result<(), OperationFailure> {
    check(Resource::SchemaDepth, depth, limits)?;
    // Match the decoder's hard recursion ceiling even with maximal caller limits.
    if depth >= 64 {
        return Err(OperationFailure::malformed_response());
    }
    Ok(())
}

/// Error-free borrowed lookup: never formats a rejected ColumnName.
fn borrowed_column_type<'a>(mut schema: &'a StructType, path: &ColumnName) -> Option<&'a DataType> {
    let parts: &[String] = path.as_ref();
    let (last, parents) = parts.split_last()?;
    for part in parents {
        let DataType::Struct(child) = schema.field(part)?.data_type() else {
            return None;
        };
        schema = child;
    }
    Some(schema.field(last)?.data_type())
}
