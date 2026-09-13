//! Whole-log ownership before creating a JSON decoder or execution stream.

use std::mem::size_of;
use std::sync::Arc;

use delta_kernel::tasks::{
    AdmittedRead, LogIdentityManifest, OperationFailure, Resource, ResourceExhausted, TaskLimits,
};

use crate::json_framing::{self, JsonFraming};
use crate::log_storage::{AdmittedJsonLogStorage, JsonLogReadLimits};

/// Admitted, identity-checked owners; no decoder has been constructed yet.
pub(crate) struct LogInput {
    pub manifest: Arc<LogIdentityManifest>,
    pub reads: Vec<AdmittedRead>,
    pub framing: JsonFraming,
    pub retained_bytes: usize,
}

impl LogInput {
    /// Preflights the full selected history, then reads each exact observed version once.
    /// `other_retained` includes the task plan and other simultaneous host-owned state.
    pub(crate) async fn load(
        manifest: Arc<LogIdentityManifest>,
        storage: &dyn AdmittedJsonLogStorage,
        limits: TaskLimits,
        other_retained: usize,
    ) -> Result<Self, OperationFailure> {
        let overflow = || exhausted(Resource::TaskStateBytes, limits, usize::MAX);
        let slots = manifest
            .files()
            .len()
            .checked_mul(size_of::<AdmittedRead>())
            .ok_or_else(overflow)?;
        let mut retained = other_retained
            .checked_add(manifest.retained_bytes())
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .and_then(|n| n.checked_add(slots))
            .ok_or_else(overflow)?;
        check(Resource::TaskStateBytes, limits, retained)?;
        let mut total = 0usize;
        for file in manifest.files() {
            let bounds = JsonLogReadLimits::try_new(file, limits, total, total, retained)?;
            total = total.checked_add(bounds.exact_bytes).ok_or_else(overflow)?;
            retained = retained
                .checked_add(bounds.exact_bytes)
                .ok_or_else(overflow)?;
        }
        // Lexical work is admitted for all selected bytes before any provider read.
        check(Resource::WorkUnits, limits, total)?;
        let final_retained = retained;
        retained -= total;
        let mut reads = Vec::with_capacity(manifest.files().len());
        let mut requested = 0usize;
        let mut framing = JsonFraming::default();
        for file in manifest.files() {
            let bounds = JsonLogReadLimits::try_new(file, limits, requested, requested, retained)?;
            requested += bounds.exact_bytes;
            let read = storage.read_log(file, bounds).await?;
            if read.identity != file.identity
                || read.offset != 0
                || !read.eof
                || read.bytes.len() != bounds.exact_bytes
                || read.bytes.capacity() != bounds.exact_bytes
            {
                return Err(OperationFailure::malformed_response());
            }
            let observed = json_framing::preflight(&read.bytes, limits)?;
            framing.records = framing
                .records
                .checked_add(observed.records)
                .ok_or_else(overflow)?;
            framing.tokens = framing
                .tokens
                .checked_add(observed.tokens)
                .ok_or_else(overflow)?;
            check(Resource::Records, limits, framing.records)?;
            check(Resource::WorkUnits, limits, framing.tokens)?;
            framing.max_record_bytes = framing.max_record_bytes.max(observed.max_record_bytes);
            framing.max_depth = framing.max_depth.max(observed.max_depth);
            retained += bounds.exact_bytes;
            reads.push(read);
        }
        Ok(Self {
            manifest,
            reads,
            framing,
            retained_bytes: final_retained,
        })
    }
}

fn check(resource: Resource, limits: TaskLimits, observed: usize) -> Result<(), OperationFailure> {
    if observed > limits.limit(resource) {
        return Err(exhausted(resource, limits, observed));
    }
    Ok(())
}

fn exhausted(resource: Resource, limits: TaskLimits, observed: usize) -> OperationFailure {
    ResourceExhausted {
        resource,
        limit: limits.limit(resource),
        observed,
    }
    .into()
}
