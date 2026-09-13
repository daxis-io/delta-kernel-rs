use std::cell::Cell;

use super::ResourceExhausted;

// Keep the domain, default limit and counter index in one table.
macro_rules! resources {
    ($(#[doc = $doc:literal] $name:ident = $limit:expr),+ $(,)?) => {
        /// An independently limited resource used by an operation.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[non_exhaustive]
        pub enum Resource { $(#[doc = $doc] $name),+ }

        impl Resource {
            /// The supported accounting domains, in counter-storage order.
            pub const ALL: &'static [Self] = &[$(Self::$name),+];
        }

        const QUALIFICATION: [usize; Resource::ALL.len()] = [$($limit),+];
    };
}

resources! {
    /// Internal evaluation pages, including empty EOF pages.
    EvaluationPages = 4096,
    /// Cumulative internal evaluation batches.
    EvaluationBatches = 1_000_000,
    /// Cumulative internal evaluation rows, including unselected rows.
    EvaluationRows = 1_000_000,
    /// Cumulative conservative evaluation backing and container bytes.
    EvaluationBytes = 512 << 20,
    /// Dispatched task requests.
    Requests = 4096,
    /// Allocated continuation-token bytes.
    ContinuationBytes = 4096,
    /// Entries in one listing page.
    ListingEntries = 256,
    /// Descriptor backing and container bytes in one listing page.
    ListingDescriptorBytes = 256 << 10,
    /// Ranges in one read operation.
    ReadSlices = 16,
    /// Chunks in one read response.
    ReadChunks = 64,
    /// Backing bytes in one read response.
    ReadPayloadBytes = 4 << 20,
    /// Batches in one internal evaluation page.
    EvaluationPageBatches = 8,
    /// Rows in one internal evaluation page.
    EvaluationPageRows = 65536,
    /// Conservative backing bytes in one internal evaluation page.
    EvaluationPageBytes = 8 << 20,
    /// Retained plan nodes.
    PlanNodes = 4096,
    /// Plan topology depth.
    PlanDepth = 64,
    /// Checked encoded-size bound of an admitted plan.
    PlanEncodedBytes = 2 << 20,
    /// Encoded footer bytes.
    FooterBytes = 8 << 20,
    /// Decoded schema nodes.
    SchemaNodes = 4096,
    /// Decoded schema depth.
    SchemaDepth = 64,
    /// Decoded row groups.
    RowGroups = 16384,
    /// Decoded column chunks.
    ColumnChunks = 262144,
    /// Conservative decoded metadata allocation bytes.
    MetadataAllocatedBytes = 32 << 20,
    /// Encoded page-index bytes.
    PageIndexBytes = 4 << 20,
    /// Decoded page-index entries.
    PageIndexEntries = 1_000_000,
    /// Live task-owned state, including its fixed storage.
    TaskStateBytes = 32 << 20,
    /// Cumulative retained log descriptors.
    LogDescriptors = 16384,
    /// Cumulative descriptor backing and container bytes.
    DescriptorBytes = 256 << 20,
    /// Retained bytes of an incomplete JSON record.
    PartialJsonBytes = 1 << 20,
    /// Cumulative decoded records.
    Records = 1_000_000,
    /// Cumulative received input bytes.
    InputBytes = 256 << 20,
    /// Cumulative requested read bytes, charged before I/O.
    RequestedReadBytes = 256 << 20,
    /// Cumulative decoded backing bytes.
    DecodedBytes = 512 << 20,
    /// Records processed in one CPU turn.
    TurnRecords = 1024,
    /// Plan nodes processed in one CPU turn.
    TurnPlanNodes = 256,
    /// Input bytes processed in one CPU turn.
    TurnInputBytes = 1 << 20,
    /// Cooperative yields.
    Yields = 65536,
    /// CPU entries, including start and matching successful resumptions.
    CpuTurns = 4096 + 65536 + 1,
    /// Cumulative semantic work units.
    WorkUnits = 1_000_000,
    /// Live caller-output queue slots.
    OutputQueuedBatches = 2,
    /// Cumulative caller-output chunks.
    OutputChunks = 1024,
    /// Cumulative caller-output backing bytes.
    OutputBytes = 64 << 20,
}

/// Finite per-domain allowances for an operation.
///
/// A domain can constrain a per-operation observation, cumulative work, or live ownership.
/// These measurements are distinct; callers choose the corresponding accounting method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskLimits {
    limits: [usize; Resource::ALL.len()],
}

impl TaskLimits {
    /// Returns the finite qualification profile, including its 4096-request limit.
    pub fn qualification() -> Self {
        Self {
            limits: QUALIFICATION,
        }
    }

    /// Replaces one allowance. Zero deliberately rejects any positive use of that domain.
    pub fn with_limit(mut self, resource: Resource, limit: usize) -> Self {
        self.limits[resource as usize] = limit;
        self
    }

    /// Projects the decoder allowances into an allocation-free footer request payload.
    pub fn footer_limits(&self) -> FooterLimits {
        FooterLimits {
            footer_bytes: self.limit(Resource::FooterBytes),
            schema_nodes: self.limit(Resource::SchemaNodes),
            schema_depth: self.limit(Resource::SchemaDepth),
            row_groups: self.limit(Resource::RowGroups),
            column_chunks: self.limit(Resource::ColumnChunks),
            metadata_allocated_bytes: self.limit(Resource::MetadataAllocatedBytes),
            page_index_bytes: self.limit(Resource::PageIndexBytes),
            page_index_entries: self.limit(Resource::PageIndexEntries),
        }
    }

    /// Returns the configured allowance for a domain.
    pub fn limit(&self, resource: Resource) -> usize {
        self.limits[resource as usize]
    }
}

/// Decoder-specific allowances carried by a limited footer request.
///
/// The injected reader enforces each bound before its payload fetch, decoding or allocation.
/// A bounded trailer read may precede footer-length validation. Zero rejects positive use of the
/// corresponding domain; no unlimited sentinel is implied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct FooterLimits {
    /// Encoded footer bytes.
    pub footer_bytes: usize,
    /// Decoded schema nodes.
    pub schema_nodes: usize,
    /// Decoded schema depth.
    pub schema_depth: usize,
    /// Decoded row groups.
    pub row_groups: usize,
    /// Decoded column chunks.
    pub column_chunks: usize,
    /// Conservative decoded metadata allocation bytes.
    pub metadata_allocated_bytes: usize,
    /// Encoded page-index bytes.
    pub page_index_bytes: usize,
    /// Decoded page-index entries.
    pub page_index_entries: usize,
}

/// Separate cumulative work and live-ownership measurements for one domain.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ResourceUsage {
    consumed: usize,
    live: usize,
    peak_live: usize,
}

/// Read-only accounting observations exposed by a task machine.
pub struct TaskUsage<'a> {
    pub(super) accounting: &'a TaskAccounting,
}

impl TaskUsage<'_> {
    /// Returns one numeric snapshot without exposing mutable accounting operations.
    pub fn usage(&self, resource: Resource) -> ResourceUsage {
        self.accounting.usage(resource)
    }
}

impl ResourceUsage {
    /// Returns cumulative successful charges, including work whose storage has been released.
    pub fn consumed(self) -> usize {
        self.consumed
    }

    /// Returns the current owned amount reported by its allocation owner.
    pub fn live(self) -> usize {
        self.live
    }

    /// Returns the largest successfully admitted live amount.
    pub fn peak_live(self) -> usize {
        self.peak_live
    }
}

/// Operation-local checked counters; these are ownership charges, not process-memory readings.
///
/// Owners check or charge before allocating or growing and report live ownership at handoffs.
/// This ledger does not discover allocations or establish producer admission retroactively.
/// Shared borrows allow semantic states to charge work without replacing limits or resetting
/// cumulative protocol counters. The ledger is movable between threads but is not shared across
/// concurrent tasks; query execution uses the driver's shared allocation authority.
pub struct TaskAccounting {
    limits: TaskLimits,
    usage: [Cell<ResourceUsage>; Resource::ALL.len()],
}

impl TaskAccounting {
    /// Creates empty counters for the supplied finite allowances without allocating.
    pub fn new(limits: TaskLimits) -> Self {
        Self {
            limits,
            usage: std::array::from_fn(|_| Cell::new(ResourceUsage::default())),
        }
    }

    /// Returns a numeric snapshot of one domain.
    pub fn usage(&self, resource: Resource) -> ResourceUsage {
        self.usage[resource as usize].get()
    }

    /// Returns the immutable configured allowance for a domain.
    pub fn limit(&self, resource: Resource) -> usize {
        self.limits.limit(resource)
    }

    /// Checks a per-operation amount without changing cumulative or live accounting.
    ///
    /// Returns a typed exhaustion error when `amount` exceeds the configured allowance.
    pub fn check(&self, resource: Resource, amount: usize) -> Result<(), ResourceExhausted> {
        let limit = self.limits.limit(resource);
        if amount > limit {
            return Err(ResourceExhausted {
                resource,
                limit,
                observed: amount,
            });
        }
        Ok(())
    }

    /// Consumes cumulative work before performing it; successful charges are never refunded.
    ///
    /// Overflow or limit exhaustion leaves the counter unchanged and returns a typed error.
    pub fn charge(&self, resource: Resource, amount: usize) -> Result<(), ResourceExhausted> {
        let total = self.next_charge(resource, amount)?;
        let mut usage = self.usage(resource);
        usage.consumed = total;
        self.usage[resource as usize].set(usage);
        Ok(())
    }

    /// Admits the owner's next live amount and updates its peak, without refunding work.
    ///
    /// Owners call this before growing and again when releasing storage. Exhaustion preserves
    /// the previous observation. This cannot validate an allocation already made by a producer.
    pub fn set_live(&self, resource: Resource, amount: usize) -> Result<(), ResourceExhausted> {
        self.check(resource, amount)?;
        let mut usage = self.usage(resource);
        usage.live = amount;
        usage.peak_live = usage.peak_live.max(amount);
        self.usage[resource as usize].set(usage);
        Ok(())
    }

    pub(super) fn next_charge(
        &self,
        resource: Resource,
        amount: usize,
    ) -> Result<usize, ResourceExhausted> {
        let total = self
            .usage(resource)
            .consumed
            .checked_add(amount)
            .ok_or(ResourceExhausted {
                resource,
                limit: self.limits.limit(resource),
                observed: usize::MAX,
            })?;
        self.check(resource, total)?;
        Ok(total)
    }

    pub(super) fn clear_live(&self, fixed_task_bytes: usize) {
        for counter in &self.usage {
            let mut usage = counter.get();
            usage.live = 0;
            counter.set(usage);
        }
        // Construction already admitted this storage; dropping semantic state cannot release it.
        let counter = &self.usage[Resource::TaskStateBytes as usize];
        let mut usage = counter.get();
        usage.live = fixed_task_bytes;
        counter.set(usage);
    }
}

/// Work-only loan of one task's ledger while its exact request is pending.
///
/// The task constructs this view; callers cannot forge its key or replace its
/// accounting authority. Holding the borrow prevents resuming/cancelling the
/// task. Hosts prepay semantic work before executing it; successful charges
/// remain consumed if a completion future is abandoned or fails.
pub struct PendingWork<'a> {
    pub(super) key: super::RequestKey,
    pub(super) accounting: &'a TaskAccounting,
}
impl PendingWork<'_> {
    /// The only request for which this loan authorizes host work.
    pub fn key(&self) -> super::RequestKey {
        self.key
    }
    /// Kernel task owners that remain live while the host completes this request.
    /// This read-only observation excludes the transferred request. Hosts must
    /// compose it with their request and execution owners before allocation.
    pub fn retained_task_bytes(&self) -> usize {
        self.accounting.usage(Resource::TaskStateBytes).live()
    }
    /// Remaining cumulative work, including all prior Kernel and host charges.
    pub fn remaining(&self) -> usize {
        self.accounting
            .limit(Resource::WorkUnits)
            .saturating_sub(self.accounting.usage(Resource::WorkUnits).consumed())
    }
    /// Admits and permanently charges work before the corresponding stage.
    pub fn charge(&self, units: usize) -> Result<(), ResourceExhausted> {
        self.accounting.charge(Resource::WorkUnits, units)
    }
}
