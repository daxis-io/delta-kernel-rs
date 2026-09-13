//! Transfer of source-admitted full array backing into Kernel evaluation pages.

use datafusion::arrow::array::{RecordBatch, StructArray};
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::execution::memory_pool::MemoryReservation;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::engine_data::RowVisitor;
use delta_kernel::expressions::{ArrayData, ColumnName};
use delta_kernel::schema::SchemaRef as KernelSchemaRef;
use delta_kernel::tasks::{
    AccountedEngineData, OperationFailure, Resource, ResourceExhausted, TaskLimits,
};
use delta_kernel::{DeltaResult, EngineData};
use std::mem::size_of;
use std::sync::Arc;

/// A bound established before a stream pull, independent of visible batch size.
/// `backing_rows` bounds the materialized aggregate's full backing, not its slice.
#[derive(Clone, Copy)]
pub(crate) struct PageEnvelope {
    pub retained: usize,
    pub rebind_peak: usize,
}
impl PageEnvelope {
    pub(crate) fn preflight(
        schema: &SchemaRef,
        backing_rows: usize,
        input_bytes: usize,
        schema_owners: usize,
        limits: TaskLimits,
    ) -> Result<Self, OperationFailure> {
        let overflow = || ResourceExhausted {
            resource: Resource::EvaluationPageBytes,
            limit: limits.limit(Resource::EvaluationPageBytes),
            observed: usize::MAX,
        };
        let arrays =
            crate::json_arrays::output_owner_peak(schema, backing_rows, input_bytes, limits)?;
        // Original full backing plus expected/actual schema fields. The schema
        // conversion component is supplied from this exact producer's declared
        // schema, before its Arrow conversion. No size() observation is used.
        let retained = arrays
            .checked_add(schema_owners.checked_mul(2).ok_or_else(overflow)?)
            .and_then(|n| n.checked_add(size_of::<AdmittedPageData>()))
            .and_then(|n| n.checked_add(size_of::<MemoryReservation>() + 2 * size_of::<usize>()))
            .and_then(|n| n.checked_add(crate::execution_aux::host_registration_peak()?))
            .ok_or_else(overflow)?;
        // fix_nested_null_masks reconstructs struct containers and combines
        // null masks; Arrow's checked struct cast constructs the second tree.
        // Their primitive/Map/List payloads remain shared with the input.
        // A zero-payload tree conservatively includes empty/null array buffers,
        // slice headers and per-child Arc vectors, including alignment minima.
        let containers = crate::json_arrays::output_owner_peak(schema, 1, 0, limits)?;
        let rebind_peak = containers
            .checked_mul(2)
            .and_then(|n| n.checked_add(size_of::<StructArray>()))
            .and_then(|n| {
                n.checked_add(crate::json_arrays::vec_peak(
                    schema.fields().len(),
                    size_of::<datafusion::arrow::array::ArrayRef>(),
                )?)
            })
            .ok_or_else(overflow)?;
        let page = retained
            .checked_add(size_of::<Box<dyn AccountedEngineData>>())
            .ok_or_else(overflow)?;
        if page > limits.limit(Resource::EvaluationPageBytes) {
            return Err(ResourceExhausted {
                resource: Resource::EvaluationPageBytes,
                limit: limits.limit(Resource::EvaluationPageBytes),
                observed: page,
            }
            .into());
        }
        Ok(Self {
            retained,
            rebind_peak,
        })
    }
}

/// Every transferred page retains the same host reservation until its last
/// engine-data owner is dropped. EOF/cancellation can release the stream while
/// Kernel still owns admitted scan pages; these owners cannot free the lease.
pub(crate) struct AdmittedPageData {
    data: ArrowEngineData,
    bytes: usize,
    host_retained_bytes: usize,
    lease: Arc<MemoryReservation>,
}
impl AdmittedPageData {
    pub(crate) fn with_host_retained_bytes(mut self, bytes: usize) -> Self {
        self.host_retained_bytes = bytes;
        self
    }
    pub(crate) fn new(
        batch: RecordBatch,
        expected: SchemaRef,
        envelope: PageEnvelope,
        lease: Arc<MemoryReservation>,
    ) -> Result<Self, OperationFailure> {
        if batch.num_rows() > 1 {
            return Err(OperationFailure::malformed_response());
        }
        let batch = crate::result_schema::rebind(batch, expected)?;
        Ok(Self {
            data: ArrowEngineData::new(batch),
            bytes: envelope.retained,
            host_retained_bytes: 0,
            lease,
        })
    }
}
impl AccountedEngineData for AdmittedPageData {
    fn host_retained_bytes(&self) -> usize {
        self.host_retained_bytes
    }
    fn accounted_bytes(&self) -> Result<usize, ResourceExhausted> {
        Ok(self.bytes)
    }
}
// Generic EngineData transformations are outside the selected task execution;
// preserve their ordinary semantics and propagate the existing backing lease.
struct LeasedData {
    data: Box<dyn EngineData>,
    lease: Arc<MemoryReservation>,
}
impl EngineData for AdmittedPageData {
    fn len(&self) -> usize {
        self.data.len()
    }
    fn visit_rows(&self, columns: &[ColumnName], visitor: &mut dyn RowVisitor) -> DeltaResult<()> {
        self.data.visit_rows(columns, visitor)
    }
    fn append_columns(
        &self,
        schema: KernelSchemaRef,
        columns: Vec<ArrayData>,
    ) -> DeltaResult<Box<dyn EngineData>> {
        Ok(Box::new(LeasedData {
            data: self.data.append_columns(schema, columns)?,
            lease: self.lease.clone(),
        }))
    }
    fn apply_selection_vector(
        self: Box<Self>,
        selection: Vec<bool>,
    ) -> DeltaResult<Box<dyn EngineData>> {
        let Self { data, lease, .. } = *self;
        Ok(Box::new(LeasedData {
            data: Box::new(data).apply_selection_vector(selection)?,
            lease,
        }))
    }
    fn has_field(&self, name: &ColumnName) -> bool {
        self.data.has_field(name)
    }
}
// LeasedData already contains a trait object; avoid boxing that object again.
impl EngineData for LeasedData {
    fn len(&self) -> usize {
        self.data.len()
    }
    fn visit_rows(&self, columns: &[ColumnName], visitor: &mut dyn RowVisitor) -> DeltaResult<()> {
        self.data.visit_rows(columns, visitor)
    }
    fn append_columns(
        &self,
        schema: KernelSchemaRef,
        columns: Vec<ArrayData>,
    ) -> DeltaResult<Box<dyn EngineData>> {
        Ok(Box::new(Self {
            data: self.data.append_columns(schema, columns)?,
            lease: self.lease.clone(),
        }))
    }
    fn apply_selection_vector(
        self: Box<Self>,
        selection: Vec<bool>,
    ) -> DeltaResult<Box<dyn EngineData>> {
        let Self { data, lease } = *self;
        Ok(Box::new(Self {
            data: data.apply_selection_vector(selection)?,
            lease,
        }))
    }
    fn has_field(&self, name: &ColumnName) -> bool {
        self.data.has_field(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json_arrays::tests::observe_allocations;
    use datafusion::arrow::array::Int64Array;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
    #[test]
    fn page_preflight_and_last_owner_reservation_lifetime() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let limits = TaskLimits::qualification();
        let (envelope, allocations) =
            observe_allocations(|| PageEnvelope::preflight(&schema, 20, 160, 1024, limits));
        assert_eq!(allocations, 0);
        let envelope = envelope.unwrap();
        let page_bytes = envelope.retained + size_of::<Box<dyn AccountedEngineData>>();
        for (limit, allowed) in [(page_bytes, true), (page_bytes - 1, false)] {
            let (result, allocations) = observe_allocations(|| {
                PageEnvelope::preflight(
                    &schema,
                    20,
                    160,
                    1024,
                    limits.with_limit(Resource::EvaluationPageBytes, limit),
                )
            });
            assert_eq!(allocations, 0);
            assert_eq!(result.is_ok(), allowed);
        }
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        let reservation = MemoryConsumer::new("KernelJsonHost").register(&pool);
        reservation
            .try_grow(envelope.retained + envelope.rebind_peak)
            .unwrap();
        let lease = Arc::new(reservation);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(0..20))],
        )
        .unwrap()
        .slice(19, 1);
        let page = AdmittedPageData::new(batch, schema, envelope, lease.clone()).unwrap();
        assert_eq!(page.accounted_bytes().unwrap(), envelope.retained);
        drop(lease);
        assert!(pool.reserved() > 0);
        let selected = Box::new(page).apply_selection_vector(vec![true]).unwrap();
        assert!(pool.reserved() > 0);
        assert_eq!(selected.len(), 1);
        drop(selected);
        assert_eq!(pool.reserved(), 0);
    }
}
