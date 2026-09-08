//! Expression handling based on arrow-rs compute kernels.
use std::sync::Arc;

pub(crate) use evaluate_expression::extract_column;
use evaluate_expression::{evaluate_expression, evaluate_predicate};
use itertools::Itertools;
use tracing::debug;

use super::arrow_conversion::{TryFromKernel as _, TryIntoArrow as _};
use crate::arrow::array::{self, ArrayBuilder, ArrayRef, RecordBatch, StructArray};
use crate::arrow::datatypes::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema,
};
use crate::engine::arrow_data::{extract_record_batch, ArrowEngineData};
use crate::engine::arrow_utils::apply_schema::{apply_schema, apply_schema_to};
use crate::error::{DeltaResult, Error};
use crate::expressions::{Expression, ExpressionRef, PredicateRef, Scalar};
use crate::schema::{DataType, SchemaRef};
use crate::{EngineData, EvaluationHandler, ExpressionEvaluator, PredicateEvaluator};

pub mod evaluate_expression;
pub mod opaque;

#[cfg(test)]
mod tests;

#[derive(Debug)]
pub struct ArrowEvaluationHandler;

impl EvaluationHandler for ArrowEvaluationHandler {
    fn new_expression_evaluator(
        &self,
        schema: SchemaRef,
        expression: ExpressionRef,
        output_type: DataType,
    ) -> DeltaResult<Arc<dyn ExpressionEvaluator>> {
        Ok(Arc::new(DefaultExpressionEvaluator {
            _input_schema: schema,
            expression,
            output_type,
        }))
    }

    fn new_predicate_evaluator(
        &self,
        schema: SchemaRef,
        predicate: PredicateRef,
    ) -> DeltaResult<Arc<dyn PredicateEvaluator>> {
        Ok(Arc::new(DefaultPredicateEvaluator {
            _input_schema: schema,
            predicate,
        }))
    }

    /// Create a single-row array with all-null leaf values. Note that if a nested struct is
    /// included in the `output_type`, the entire struct will be NULL (instead of a not-null struct
    /// with NULL fields).
    fn null_row(&self, output_schema: SchemaRef) -> DeltaResult<Box<dyn EngineData>> {
        let fields = output_schema.fields();
        let arrays = fields
            .map(|field| Scalar::Null(field.data_type().clone()).to_array(1))
            .try_collect()?;
        let record_batch =
            RecordBatch::try_new(Arc::new(output_schema.as_ref().try_into_arrow()?), arrays)?;
        Ok(Box::new(ArrowEngineData::new(record_batch)))
    }

    fn create_many(
        &self,
        schema: SchemaRef,
        rows: &[&[Scalar]],
    ) -> DeltaResult<Box<dyn EngineData>> {
        let arrow_schema: Arc<ArrowSchema> = Arc::new(schema.as_ref().try_into_arrow()?);
        if rows.is_empty() {
            return Ok(Box::new(ArrowEngineData::new(RecordBatch::new_empty(
                arrow_schema,
            ))));
        }

        let num_rows = rows.len();
        let num_fields = schema.fields().len();
        for (row_idx, row) in rows.iter().enumerate() {
            if row.len() != num_fields {
                return Err(Error::generic(format!(
                    "Row {} has {} scalars but schema has {} fields",
                    row_idx,
                    row.len(),
                    num_fields
                )));
            }
        }

        let mut builders: Vec<Box<dyn ArrayBuilder>> = arrow_schema
            .fields()
            .iter()
            .map(|field| array::make_builder(field.data_type(), num_rows))
            .collect();

        let fields: Vec<_> = schema.fields().collect();
        for (col_idx, builder) in builders.iter_mut().enumerate() {
            let field_name = fields[col_idx].name();
            for (row_idx, row) in rows.iter().enumerate() {
                row[col_idx].append_to(builder.as_mut(), 1).map_err(|e| {
                    Error::generic(format!(
                        "Row {row_idx}, field '{field_name}' \
                            (expected type {}, got {}): {e}",
                        fields[col_idx].data_type(),
                        row[col_idx].data_type()
                    ))
                })?;
            }
        }

        let arrays: Vec<ArrayRef> = builders.into_iter().map(|mut b| b.finish()).collect();

        Ok(Box::new(ArrowEngineData::new(RecordBatch::try_new(
            arrow_schema,
            arrays,
        )?)))
    }
}

#[derive(Debug)]
pub struct DefaultExpressionEvaluator {
    _input_schema: SchemaRef,
    expression: ExpressionRef,
    output_type: DataType,
}

impl ExpressionEvaluator for DefaultExpressionEvaluator {
    fn evaluate(&self, batch: &dyn EngineData) -> DeltaResult<Box<dyn EngineData>> {
        debug!("Arrow evaluator evaluating: {:#?}", self.expression);
        let batch = extract_record_batch(batch)?;
        // TODO: make sure we have matching schemas for validation
        // if batch.schema().as_ref() != &input_schema {
        //     return Err(Error::Generic(format!(
        //         "input schema does not match batch schema: {:?} != {:?}",
        //         input_schema,
        //         batch.schema()
        //     )));
        // };
        let batch = match (self.expression.as_ref(), &self.output_type) {
            (Expression::StructPatch(patch), DataType::Struct(_)) if patch.is_empty() => {
                // Empty patch optimization: Skip expression evaluation and directly apply the
                // output schema to the input RecordBatch. This is used to cheaply apply a new
                // output schema to existing data without changing it, e.g. for column mapping.
                let array = match patch.input_path() {
                    None => Arc::new(StructArray::from(batch.clone())),
                    Some(path) => extract_column(batch, path)?,
                };
                apply_schema(&array, &self.output_type)?
            }
            (expr, output_type @ DataType::Struct(_)) => {
                let array_ref = evaluate_expression(expr, batch, Some(output_type))?;
                apply_schema(&array_ref, output_type)?
            }
            (expr, output_type) => {
                let array_ref = evaluate_expression(expr, batch, Some(output_type))?;
                let array_ref = apply_schema_to(&array_ref, output_type)?;
                let arrow_type = ArrowDataType::try_from_kernel(output_type)?;
                let schema = ArrowSchema::new(vec![ArrowField::new("output", arrow_type, true)]);
                RecordBatch::try_new(Arc::new(schema), vec![array_ref])?
            }
        };

        Ok(Box::new(ArrowEngineData::new(batch)))
    }
}

#[derive(Debug)]
pub struct DefaultPredicateEvaluator {
    _input_schema: SchemaRef,
    predicate: PredicateRef,
}

impl PredicateEvaluator for DefaultPredicateEvaluator {
    fn evaluate(&self, batch: &dyn EngineData) -> DeltaResult<Box<dyn EngineData>> {
        debug!("Arrow evaluator evaluating: {:#?}", self.predicate);
        let batch = extract_record_batch(batch)?;
        // TODO: make sure we have matching schemas for validation
        // if batch.schema().as_ref() != &input_schema {
        //     return Err(Error::Generic(format!(
        //         "input schema does not match batch schema: {:?} != {:?}",
        //         input_schema,
        //         batch.schema()
        //     )));
        // };
        let array = evaluate_predicate(&self.predicate, batch, false)?;
        let schema = ArrowSchema::new(vec![ArrowField::new(
            "output",
            ArrowDataType::Boolean,
            true,
        )]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)])?;
        Ok(Box::new(ArrowEngineData::new(batch)))
    }
}
