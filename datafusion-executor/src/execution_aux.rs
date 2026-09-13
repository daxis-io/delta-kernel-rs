//! Reached execution metrics, context and registration owners for closed JSON plans.
//! Plan expressions, file queues, schemas and retained batches are separate components.

use crate::evaluation_frames::aggregate;
use crate::json_arrays::vec_peak;
use chrono::{DateTime, Utc};
use datafusion::execution::{
    memory_pool::{MemoryConsumer, MemoryPool},
    TaskContext,
};
use datafusion::physical_plan::metrics::{Metric, MetricsSet};
use delta_kernel::tasks::{OperationFailure, Resource, ResourceExhausted, TaskLimits};
use std::alloc::Layout;
use std::mem::size_of;
use std::sync::{atomic::AtomicUsize, Arc};

/// Backing of the host's MemoryReservation registration, retained by pages.
pub(crate) fn host_registration_peak() -> Option<usize> {
    let word = Layout::new::<usize>();
    aggregate(&[
        word,
        word,
        Layout::new::<Arc<dyn MemoryPool>>(),
        Layout::new::<MemoryConsumer>(),
    ])?
    .checked_add(vec_peak("KernelJsonHost".len(), 1)?)
}

pub(crate) fn peak(limits: TaskLimits) -> Result<usize, OperationFailure> {
    let overflow = || ResourceExhausted {
        resource: Resource::MetadataAllocatedBytes,
        limit: limits.limit(Resource::MetadataAllocatedBytes),
        observed: usize::MAX,
    };
    let word = Layout::new::<usize>();
    // parking_lot0.12 RawMutex is one AtomicU8. Its Mutex<T> has that
    // raw lock plus UnsafeCell<T>. Layout composition includes Arc counters,
    // all alignment and any Rust field reordering; no mutex is constructed.
    let set = aggregate(&[
        word,
        word,
        Layout::new::<std::sync::atomic::AtomicU8>(),
        Layout::new::<MetricsSet>(),
    ])
    .ok_or_else(overflow)?;
    let timestamp = aggregate(&[
        word,
        word,
        Layout::new::<std::sync::atomic::AtomicU8>(),
        Layout::new::<Option<DateTime<Utc>>>(),
    ])
    .ok_or_else(overflow)?;
    let metric = aggregate(&[word, word, Layout::new::<Metric>()]).ok_or_else(overflow)?;
    let atomic = aggregate(&[word, word, Layout::new::<AtomicUsize>()]).ok_or_else(overflow)?;
    // Maximal live-add pipeline: file source, four projections, two filters,
    // partial and final aggregates. PM has fewer operators. Each baseline
    // creates two timestamps and four time/count atomics (six Metric Arcs).
    let operators = 1usize + 4 + 2 + 2;
    // Source FileStream adds eight; DataSourceExec adds one split counter.
    // Filters add one two-counter ratio each. Both hash stages add three spill
    // counters and four GroupBy times even with DiskManager disabled. Partial
    // adds one two-counter reduction ratio; probe disabled at threshold1.0.
    let extra_metrics = 8usize + 1 + 2 + 2 * (3 + 4) + 1;
    let extra_atomics = 8usize + 1 + 2 * 2 + 2 * (3 + 4) + 2;
    let metrics = operators * 6 + extra_metrics;
    let atomics = operators * 4 + extra_atomics;
    // Registering into separate MetricsSet Vecs can each hit RawVec's minimum
    // capacity. Sum the per-set bound, then the additive growth contribution
    // for all registrations. Labels are empty; all reached names are static Cow.
    let metric_vectors = vec_peak(0, size_of::<Arc<Metric>>())
        .and_then(|n| n.checked_mul(operators))
        .and_then(|n| n.checked_add(vec_peak(metrics, size_of::<Arc<Metric>>())?))
        .ok_or_else(overflow)?;
    let metrics = set
        .checked_mul(operators)
        .and_then(|n| n.checked_add(metric.checked_mul(metrics)?))
        .and_then(|n| n.checked_add(atomic.checked_mul(atomics)?))
        .and_then(|n| n.checked_add(timestamp.checked_mul(operators * 2)?))
        .and_then(|n| n.checked_add(metric_vectors))
        .ok_or_else(overflow)?;
    // SessionState::task_ctx allocates one Arc<TaskContext> and clones the fixed
    // session id. SessionConfig shares Arc<ConfigOptions>; the private state's
    // extension map and all four function maps are empty, so cloning has no
    // bucket or caller callback allocation. RuntimeEnv is also Arc-cloned.
    let context = aggregate(&[word, word, Layout::new::<TaskContext>()])
        .and_then(|n| n.checked_add(vec_peak("kernel_json_task".len(), 1)?))
        .ok_or_else(overflow)?;
    // MemoryConsumer::register owns Arc<SharedRegistration> with exactly the
    // pool Arc and MemoryConsumer. One per aggregate stage plus the host's
    // KernelJsonHost reservation. The formatted stage names include [0].
    // Pool-internal accounting remains the caller pool's resource contract.
    let registration = aggregate(&[
        word,
        word,
        Layout::new::<Arc<dyn MemoryPool>>(),
        Layout::new::<MemoryConsumer>(),
    ])
    .ok_or_else(overflow)?;
    let registrations = registration
        .checked_mul(3)
        .and_then(|n| {
            n.checked_add(vec_peak("PartialHashAggregateStream[0]".len(), 1)?.checked_mul(2)?)
        })
        .and_then(|n| n.checked_add(vec_peak("KernelJsonHost".len(), 1)?))
        .ok_or_else(overflow)?;
    let bytes = metrics
        .checked_add(context)
        .and_then(|n| n.checked_add(registrations))
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
    #[test]
    fn fixed_execution_auxiliary_preflight_allocates_nothing() {
        let limits = TaskLimits::qualification();
        let (result, allocations) = observe_allocations(|| peak(limits));
        assert_eq!(allocations, 0);
        let bytes = result.unwrap();
        for (limit, allowed) in [(bytes, true), (bytes - 1, false), (0, false)] {
            let (result, allocations) = observe_allocations(|| {
                peak(limits.with_limit(Resource::MetadataAllocatedBytes, limit))
            });
            assert_eq!(allocations, 0);
            assert_eq!(result.is_ok(), allowed);
        }
    }
}
