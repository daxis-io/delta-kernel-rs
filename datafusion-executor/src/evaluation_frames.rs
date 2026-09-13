//! Compiler frame layouts and selected futures-util0.3.32 queue owners.
//! This component excludes physical operators, decoder buffers and host state.

use datafusion::physical_planner::DefaultPhysicalPlanner;
use datafusion_datasource_json::source::JsonSource;
use delta_kernel::tasks::{OperationFailure, Resource, ResourceExhausted, TaskLimits};
use std::alloc::Layout;

/// Bound a Rust aggregate of known fields regardless of field reordering. Each
/// field contributes its size rounded to the aggregate alignment; that covers
/// all inter-field and trailing padding without assuming repr(Rust) ordering.
pub(crate) fn aggregate(fields: &[Layout]) -> Option<usize> {
    let align = fields.iter().map(Layout::align).max()?;
    fields.iter().try_fold(0usize, |total, field| {
        total.checked_add(field.size().checked_add(align - 1)? & !(align - 1))
    })
}

pub(crate) fn peak(limits: TaskLimits) -> Result<usize, OperationFailure> {
    let [initial, _leaf, optional_leaf] = DefaultPhysicalPlanner::planning_future_layouts();
    let word = Layout::new::<usize>();
    let flag = Layout::new::<bool>();
    let overflow = || ResourceExhausted {
        resource: Resource::MetadataAllocatedBytes,
        limit: limits.limit(Resource::MetadataAllocatedBytes),
        observed: usize::MAX,
    };
    // futures-util/stream/futures_unordered/task.rs Task<F>: Option<F>,
    // next_all,prev_all,len_all,next_ready_to_run,Weak(queue),queued,woken.
    // The Arc allocation prefixes two atomic counters. The sentinel contains
    // the SAME Option<F> storage even though its future is None.
    let task = aggregate(&[
        word,
        word,
        optional_leaf,
        word,
        word,
        word,
        word,
        word,
        flag,
        flag,
    ])
    .ok_or_else(overflow)?;
    // ready_to_run_queue.rs: AtomicWaker, head, tail, Arc(stub), plus Arc counters.
    let queue = aggregate(&[
        word,
        word,
        Layout::new::<futures::task::AtomicWaker>(),
        word,
        word,
        word,
    ])
    .ok_or_else(overflow)?;
    // One leaf, one sentinel. Concurrency1 does not eliminate this queue.
    let planner = initial
        .size()
        .checked_add(task.checked_mul(2).ok_or_else(overflow)?)
        .and_then(|n| n.checked_add(queue))
        .ok_or_else(overflow)?;
    let planner = datafusion::logical_expr::physical_planning_context::PhysicalPlanningContext::empty_owner_layouts()
        .iter().try_fold(planner, |n, layout| n.checked_add(layout.size()))
        .ok_or_else(overflow)?;
    let json = JsonSource::ndjson_frame_layouts()
        .iter()
        .try_fold(0usize, |n, layout| n.checked_add(layout.size()))
        .ok_or_else(overflow)?;
    let projection = datafusion_datasource::projection::ProjectionOpener::frame_layouts()
        .iter()
        .try_fold(0usize, |n, layout| n.checked_add(layout.size()))
        .ok_or_else(overflow)?;
    let aggregate =
        datafusion::physical_plan::aggregates::AggregateExec::unordered_stream_layouts();
    let global = aggregate[0]
        .size()
        .checked_add(aggregate[1].size())
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(overflow)?;
    let grouped = aggregate[2]
        .size()
        .checked_add(aggregate[3].size())
        .ok_or_else(overflow)?;
    let files = datafusion_datasource::file_stream::FileStream::frame_layouts()
        .into_iter()
        .chain(datafusion_datasource::file_stream::FileStream::file_opener_adapter_layouts())
        .try_fold(0usize, |n, layout| n.checked_add(layout.size()))
        .ok_or_else(overflow)?;
    // Closed scan producer: three projections plus ScanJson's schema-order
    // projection, two filters. PM has fewer. EnsureRequirements/Sanity do not
    // duplicate projection/filter operators. Empty stream replacements at EOF
    // coexist transiently with the old input in each filter and global stage.
    let relational = datafusion::physical_plan::projection::ProjectionExec::stream_frame_layout()
        .size()
        .checked_mul(4)
        .and_then(|n| {
            n.checked_add(
                datafusion::physical_plan::filter::FilterExec::stream_frame_layout()
                    .size()
                    .checked_mul(2)?,
            )
        })
        .and_then(|n| {
            n.checked_add(
                std::mem::size_of::<datafusion::physical_plan::EmptyRecordBatchStream>()
                    .checked_mul(4)?,
            )
        })
        // DataSourceExec always wraps the file stream in BatchSplitStream,
        // including batch_size1 (its retained batch shares already charged backing).
        .and_then(|n| {
            n.checked_add(std::mem::size_of::<
                datafusion::physical_plan::stream::BatchSplitStream,
            >())
        })
        // Arc<JsonOpener> control counters.
        .and_then(|n| n.checked_add(2 * std::mem::size_of::<usize>()))
        .ok_or_else(overflow)?;
    let bytes = planner
        .checked_add(json)
        .and_then(|n| n.checked_add(files))
        .and_then(|n| n.checked_add(relational))
        .and_then(|n| n.checked_add(projection))
        .and_then(|n| n.checked_add(global.max(grouped)))
        .and_then(|n| n.checked_add(crate::json_scan::provider_future_layout().size()))
        .ok_or_else(overflow)?;
    if bytes > limits.limit(Resource::MetadataAllocatedBytes) {
        return Err(ResourceExhausted {
            resource: Resource::MetadataAllocatedBytes,
            limit: limits.limit(Resource::MetadataAllocatedBytes),
            observed: bytes,
        }
        .into());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json_arrays::tests::observe_allocations;
    use delta_kernel::tasks::FailureKind;
    use std::mem::size_of;

    #[test]
    fn actual_frame_layout_admission_allocates_nothing() {
        let limits = TaskLimits::qualification();
        let (result, allocations) = observe_allocations(|| peak(limits));
        assert_eq!(allocations, 0);
        let bytes = result.unwrap();
        assert!(bytes > size_of::<usize>());
        for (limit, pass) in [(bytes, true), (bytes - 1, false), (0, false)] {
            let (result, allocations) = observe_allocations(|| {
                peak(limits.with_limit(Resource::MetadataAllocatedBytes, limit))
            });
            assert_eq!(allocations, 0);
            if pass {
                assert_eq!(result.unwrap(), bytes);
            } else {
                assert!(
                    matches!(result.unwrap_err().kind(), FailureKind::ResourceExhausted(e)
                if e.resource == Resource::MetadataAllocatedBytes && e.observed == bytes)
                );
            }
        }
    }
}
