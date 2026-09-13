//! Source-derived bounds for the selected Arrow JSON tape, with batch size one.

use std::mem::size_of;

use delta_kernel::tasks::{OperationFailure, Resource, ResourceExhausted, TaskLimits};

use crate::json_framing::JsonFraming;

/// Tape ownership only. Array decoders, output and execution require separate admission.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TapeEnvelope {
    pub retained: usize,
    pub peak: usize,
}

impl TapeEnvelope {
    /// Bounds initial reservations, growth and simultaneous old/new reallocations before
    /// constructing Arrow's tape. `fields` is its complete flattened fixed schema field count.
    pub(crate) fn preflight(
        fields: usize,
        framing: JsonFraming,
        limits: TaskLimits,
    ) -> Result<Self, OperationFailure> {
        Self::derive(fields, framing)
            .filter(|b| b.peak <= limits.limit(Resource::MetadataAllocatedBytes))
            .ok_or_else(|| {
                ResourceExhausted {
                    resource: Resource::MetadataAllocatedBytes,
                    limit: limits.limit(Resource::MetadataAllocatedBytes),
                    observed: Self::derive(fields, framing).map_or(usize::MAX, |b| b.peak),
                }
                .into()
            })
    }

    fn derive(fields: usize, framing: JsonFraming) -> Option<Self> {
        // TapeDecoder::new(1, fields): 2+2F elements, 1+2F offsets, 16F bytes,
        // ten stack entries. Each decoded element consumes at least one input byte;
        // strings/numbers consume at least one byte per offset. Both have a null sentinel.
        let twice_fields = fields.checked_mul(2)?;
        let entries = framing.max_record_bytes.checked_add(1)?;
        let element_bytes = grown(twice_fields.checked_add(2)?, entries, 8, 4)?;
        let offset_bytes = grown(twice_fields.checked_add(1)?, entries, size_of::<usize>(), 4)?;
        let string_bytes = grown(fields.checked_mul(16)?, framing.max_record_bytes, 1, 8)?;
        // Object key parsing pushes Value, Colon, String; Escape/Unicode adds one state.
        // All enclosing containers have one state each, so depth+4 is a ceiling.
        let stack_bytes = grown(10, framing.max_depth.checked_add(4)?, 8, 4)?;
        let backing = element_bytes
            .checked_add(offset_bytes)?
            .checked_add(string_bytes)?
            .checked_add(stack_bytes)?;
        // TapeDecoder has four Vecs and two usize fields, all at usize alignment.
        let fixed = size_of::<Vec<u8>>()
            .checked_mul(4)?
            .checked_add(size_of::<usize>().checked_mul(2)?)?;
        let retained = fixed.checked_add(backing)?;
        // A reallocating allocator can keep the old allocation until the new one exists.
        // Charging both ceilings for each buffer covers every such transition.
        let peak = retained.checked_add(backing)?;
        Some(Self { retained, peak })
    }
}

// Rust 1.97 RawVec::grow_amortized: max(2*old, required, MIN_NON_ZERO_CAP).
// If growth occurs, old < maximum required length, so new <= max(2*maximum, minimum).
// Keep the initial reservation when it is larger. This also covers bulk extend jumps.
fn grown(initial: usize, maximum: usize, element_size: usize, minimum: usize) -> Option<usize> {
    let capacity = initial.max(maximum.checked_mul(2)?).max(minimum);
    let bytes = capacity.checked_mul(element_size)?;
    (bytes <= isize::MAX as usize).then_some(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use delta_kernel::tasks::FailureKind;

    #[test]
    fn tape_refuses_exact_peak_minus_one_before_construction() {
        let framing = JsonFraming {
            records: 1,
            tokens: 20,
            max_record_bytes: 127,
            max_depth: 3,
        };
        let envelope = TapeEnvelope::preflight(30, framing, TaskLimits::qualification()).unwrap();
        assert!(envelope.peak > envelope.retained);
        TapeEnvelope::preflight(
            30,
            framing,
            TaskLimits::qualification().with_limit(Resource::MetadataAllocatedBytes, envelope.peak),
        )
        .unwrap();
        let failure = TapeEnvelope::preflight(
            30,
            framing,
            TaskLimits::qualification()
                .with_limit(Resource::MetadataAllocatedBytes, envelope.peak - 1),
        )
        .unwrap_err();
        assert!(matches!(failure.kind(), FailureKind::ResourceExhausted(e)
            if e.resource == Resource::MetadataAllocatedBytes && e.observed == envelope.peak));
    }

    #[test]
    fn tape_checked_arithmetic_rejects_overflow() {
        assert!(TapeEnvelope::preflight(
            usize::MAX,
            JsonFraming::default(),
            TaskLimits::qualification()
        )
        .is_err());
        assert!(TapeEnvelope::preflight(
            0,
            JsonFraming {
                max_record_bytes: usize::MAX,
                ..JsonFraming::default()
            },
            TaskLimits::qualification()
        )
        .is_err());
    }

    #[test]
    fn bulk_jump_then_doubling_is_covered() {
        // Initial 10 -> bulk reserve 31 -> subsequent push grows to 62.
        assert!(grown(10, 32, 8, 4).unwrap() >= 62 * 8);
    }
}
