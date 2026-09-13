use std::mem::size_of;

use super::{FailureKind, OperationFailure, Resource, ResourceExhausted, TaskProtocolError};
use crate::{EngineData, Error};

/// Opaque engine data with conservative physical-allocation accounting.
///
/// Engines include owned containers and the full backing allocations of sliced and shared buffers.
/// Charging shared allocations repeatedly is allowed; reporting only visible slice lengths is not.
/// The charge must remain valid while the batch is owned by a page. Implementations use checked
/// arithmetic and return [`ResourceExhausted`] on overflow. This method must not allocate or
/// perform I/O; an engine may calculate the charge while admitting the batch into its own memory
/// budget.
///
/// An engine must bound allocations before producing the batch. This interface supports admission
/// into Kernel and does not retroactively establish safe allocation or bound total process memory.
pub trait AccountedEngineData: EngineData {
    /// Returns the physical-allocation charge, or a typed accounting-overflow error.
    fn accounted_bytes(&self) -> Result<usize, ResourceExhausted>;
    /// Source-admitted host execution owners still live across task resumption.
    /// Separate from page backing: these do not consume the page-byte domain.
    /// The producing host reserves the remaining task allocation authority in
    /// its caller pool before handoff. Zero preserves hosts with no live state.
    fn host_retained_bytes(&self) -> usize {
        0
    }
}

/// Validated finite limits for one internal evaluation page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvaluationPageLimits {
    batches: usize,
    rows: usize,
    bytes: usize,
}

impl EvaluationPageLimits {
    /// Sets maximum batches, rows and accounted bytes per page.
    ///
    /// Returns [`TaskProtocolError::InvalidLimits`] for zero batch/row allowances, an
    /// unrepresentable batch container, or a byte allowance smaller than that container.
    pub fn new(batches: usize, rows: usize, bytes: usize) -> Result<Self, TaskProtocolError> {
        let container = batches.checked_mul(size_of::<Box<dyn AccountedEngineData>>());
        if batches == 0
            || rows == 0
            || container.is_none_or(|n| n > bytes || n > isize::MAX as usize)
        {
            return Err(TaskProtocolError::InvalidLimits);
        }
        Ok(Self {
            batches,
            rows,
            bytes,
        })
    }

    /// Returns the maximum batches per page.
    pub fn batches(&self) -> usize {
        self.batches
    }

    /// Returns the maximum rows per page, including logically unselected rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the maximum batch and container allocation charge per page.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Finite per-page and cumulative limits for a single internal evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvaluationLimits {
    page: EvaluationPageLimits,
    pages: usize,
    batches: usize,
    rows: usize,
    bytes: usize,
}

impl EvaluationLimits {
    /// Maximum cumulative pages, including EOF attempts.
    pub fn pages(self) -> usize {
        self.pages
    }
    /// Maximum cumulative batches.
    pub fn batches(self) -> usize {
        self.batches
    }
    /// Maximum cumulative rows, including unselected rows.
    pub fn rows(self) -> usize {
        self.rows
    }
    /// Maximum cumulative backing bytes, including released pages.
    pub fn bytes(self) -> usize {
        self.bytes
    }

    /// Sets per-page limits and cumulative page, batch, row and byte allowances.
    ///
    /// Zero cumulative allowances deliberately cause exhaustion when that resource is needed.
    /// A full page does not establish EOF: leave room for another page and one attempted batch
    /// pull. An EOF pull consumes no batch allowance, but requires admission before the pull.
    pub fn new(
        page: EvaluationPageLimits,
        pages: usize,
        batches: usize,
        rows: usize,
        bytes: usize,
    ) -> Self {
        Self {
            page,
            pages,
            batches,
            rows,
            bytes,
        }
    }

    /// Returns the limits supplied to the engine for each page.
    pub fn page(&self) -> EvaluationPageLimits {
        self.page
    }
}

/// An immutable, validated page of accounted engine batches.
///
/// Dropping the page releases its owned batches. Engines must admit their own allocations before
/// handing over a page; this type validates the transfer and retains no iterator or engine handler.
pub struct EvaluationPage {
    batches: Vec<Box<dyn AccountedEngineData>>,
    rows: usize,
    bytes: usize,
}

impl EvaluationPage {
    /// Validates actual container capacity and every batch's rows and physical bytes.
    ///
    /// On exhaustion or accounting overflow, drops all supplied batches and returns a typed
    /// operational failure. Validation does not allocate or undo an engine's earlier allocations.
    pub fn try_new(
        batches: Vec<Box<dyn AccountedEngineData>>,
        limits: EvaluationPageLimits,
    ) -> Result<Self, OperationFailure> {
        charge(
            0,
            batches.len(),
            limits.batches,
            Resource::EvaluationBatches,
        )?;
        let mut bytes = container_bytes(batches.capacity(), limits.bytes)?;
        let mut rows = 0;
        for batch in &batches {
            rows = charge(rows, batch.len(), limits.rows, Resource::EvaluationRows)?;
            bytes = charge(
                bytes,
                batch.accounted_bytes()?,
                limits.bytes,
                Resource::EvaluationBytes,
            )?;
        }
        Ok(Self {
            batches,
            rows,
            bytes,
        })
    }

    /// Borrows the batches without exposing mutable ownership of the validated container.
    pub fn batches(&self) -> &[Box<dyn AccountedEngineData>] {
        &self.batches
    }

    /// Transfers validated batch ownership without copying or allocating.
    pub fn into_batches(self) -> Vec<Box<dyn AccountedEngineData>> {
        self.batches
    }

    /// Returns the total rows, including logically unselected rows.
    pub fn num_rows(&self) -> usize {
        self.rows
    }

    /// Returns the physical batch and actual container allocation charge.
    pub fn accounted_bytes(&self) -> usize {
        self.bytes
    }
}

impl std::fmt::Debug for EvaluationPage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvaluationPage")
            .field("batches", &self.batches.len())
            .field("rows", &self.rows)
            .field("bytes", &self.bytes)
            .finish()
    }
}

/// Cumulative work charged by a native evaluation reader, including released pages.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EvaluationUsage {
    pages: usize,
    batches: usize,
    rows: usize,
    bytes: usize,
}

impl EvaluationUsage {
    /// Admits a page and its full requested container before either a synchronous source pull
    /// or an asynchronous host call. An attempted EOF page has the same admission requirement.
    pub(super) fn begin_page(
        &mut self,
        limits: EvaluationLimits,
    ) -> Result<usize, OperationFailure> {
        let pages = charge(self.pages, 1, limits.pages, Resource::EvaluationPages)?;
        charge(self.batches, 1, limits.batches, Resource::EvaluationBatches)?;
        let requested = container_bytes(limits.page.batches, limits.page.bytes)?;
        let bytes = charge(
            self.bytes,
            requested,
            limits.bytes,
            Resource::EvaluationBytes,
        )?;
        self.pages = pages;
        self.bytes = bytes;
        Ok(requested)
    }

    // A host has already admitted all batches. Check the original page and cumulative bounds
    // again; do not rely on the host having used the same limits in EvaluationPage::try_new.
    pub(super) fn record_page(
        &mut self,
        limits: EvaluationLimits,
        page: Option<&EvaluationPage>,
    ) -> Result<(), OperationFailure> {
        let Some(page) = page else {
            return Ok(());
        };
        charge(
            0,
            page.batches.len(),
            limits.page.batches,
            Resource::EvaluationBatches,
        )?;
        charge(0, page.rows, limits.page.rows, Resource::EvaluationRows)?;
        charge(0, page.bytes, limits.page.bytes, Resource::EvaluationBytes)?;
        let actual_container = container_bytes(page.batches.capacity(), limits.page.bytes)?;
        let reserved_container = container_bytes(limits.page.batches, limits.page.bytes)?;
        let payload = page
            .bytes
            .checked_sub(actual_container)
            .ok_or_else(OperationFailure::malformed_response)?;
        let additional = payload
            .checked_add(actual_container.saturating_sub(reserved_container))
            .ok_or_else(OperationFailure::malformed_response)?;
        let batches = charge(
            self.batches,
            page.batches.len(),
            limits.batches,
            Resource::EvaluationBatches,
        )?;
        let rows = charge(self.rows, page.rows, limits.rows, Resource::EvaluationRows)?;
        let bytes = charge(
            self.bytes,
            additional,
            limits.bytes,
            Resource::EvaluationBytes,
        )?;
        self.batches = batches;
        self.rows = rows;
        self.bytes = bytes;
        Ok(())
    }

    /// Returns pages begun, including the final empty EOF page if needed.
    pub fn pages(&self) -> usize {
        self.pages
    }

    /// Returns batches admitted across all pages.
    pub fn batches(&self) -> usize {
        self.batches
    }

    /// Returns rows admitted across all pages.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns cumulative admitted batch and container bytes, including released allocations.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

type BatchIterator =
    Box<dyn Iterator<Item = Result<Box<dyn AccountedEngineData>, OperationFailure>> + Send>;

/// Native driver component that pages an already admitted, lazy internal evaluation.
///
/// The source owns execution state and must enforce its own allocation and per-batch bounds before
/// returning a batch. The reader owns that iterator, pulls at most the requested batch count, and
/// performs no lookahead. It neither executes a plan nor adapts an unbounded `PlanExecutor` result.
/// Asynchronous engines page their own streams under the same limits before resuming Kernel tasks.
///
/// Dropping or cancelling the reader releases the source immediately. Returned pages belong to the
/// caller and are not retained here; the caller must drop them when its task is cancelled.
pub struct EvaluationReader {
    source: Option<BatchIterator>,
    limits: EvaluationLimits,
    usage: EvaluationUsage,
    failure: Option<FailureKind>,
}

impl EvaluationReader {
    /// Takes an engine-admitted source and finite limits without pulling the source.
    pub fn new(source: BatchIterator, limits: EvaluationLimits) -> Self {
        Self {
            source: Some(source),
            limits,
            usage: EvaluationUsage::default(),
            failure: None,
        }
    }

    /// Returns the next bounded page, or `None` once EOF is observed.
    ///
    /// Checks page/container admission before allocating or pulling. A batch that exceeds page or
    /// cumulative limits terminates evaluation and is dropped with the rest of the in-progress
    /// page. The original engine error transfers once; subsequent calls return only its category.
    /// No partial page is returned on failure. After cancellation, returns a failure whose
    /// [`OperationFailure::into_error`] is [`Error::Cancelled`].
    pub fn next_page(&mut self) -> Result<Option<EvaluationPage>, OperationFailure> {
        if let Some(kind) = self.failure {
            return Err(OperationFailure::terminal(kind));
        }
        if self.source.is_none() {
            return Ok(None);
        }
        match self.read_page() {
            Ok(page) => Ok(page),
            Err(error) => {
                self.source = None;
                self.failure = Some(error.kind());
                Err(error)
            }
        }
    }

    /// Immediately releases an active iterator; repeated calls preserve terminal state.
    ///
    /// Cancellation does not change an already completed or failed evaluation and cannot release
    /// pages already transferred to the caller. It performs no further iterator work.
    pub fn cancel(&mut self) {
        if self.source.take().is_some() {
            self.failure = Some(FailureKind::Cancelled);
        }
    }

    /// Returns separate cumulative accounting domains, not process-memory measurements.
    pub fn usage(&self) -> EvaluationUsage {
        self.usage
    }

    fn read_page(&mut self) -> Result<Option<EvaluationPage>, OperationFailure> {
        let limits = self.limits;
        let requested = self.usage.begin_page(limits)?;
        let mut batches = Vec::new();
        batches
            .try_reserve_exact(limits.page.batches)
            .map_err(|error| {
                OperationFailure::new(FailureKind::Engine, Error::generic_err(error))
            })?;
        let mut bytes = container_bytes(batches.capacity(), limits.page.bytes)?;
        self.usage.bytes = charge(
            self.usage.bytes,
            bytes
                .checked_sub(requested)
                .ok_or_else(OperationFailure::malformed_response)?,
            limits.bytes,
            Resource::EvaluationBytes,
        )?;
        let mut rows = 0;
        for _ in 0..limits.page.batches {
            let count = charge(
                self.usage.batches,
                1,
                limits.batches,
                Resource::EvaluationBatches,
            )?;
            let next = self.source.as_mut().and_then(|source| source.next());
            let Some(batch) = next else {
                self.source = None;
                break;
            };
            let batch = batch?;
            let batch_rows = batch.len();
            let batch_bytes = batch.accounted_bytes()?;
            rows = charge(rows, batch_rows, limits.page.rows, Resource::EvaluationRows)?;
            bytes = charge(
                bytes,
                batch_bytes,
                limits.page.bytes,
                Resource::EvaluationBytes,
            )?;
            let total_rows = charge(
                self.usage.rows,
                batch_rows,
                limits.rows,
                Resource::EvaluationRows,
            )?;
            let total_bytes = charge(
                self.usage.bytes,
                batch_bytes,
                limits.bytes,
                Resource::EvaluationBytes,
            )?;
            self.usage.batches = count;
            self.usage.rows = total_rows;
            self.usage.bytes = total_bytes;
            batches.push(batch);
        }
        if batches.is_empty() {
            Ok(None)
        } else {
            Ok(Some(EvaluationPage {
                batches,
                rows,
                bytes,
            }))
        }
    }
}

fn charge(
    used: usize,
    additional: usize,
    limit: usize,
    resource: Resource,
) -> Result<usize, ResourceExhausted> {
    match used.checked_add(additional) {
        Some(observed) if observed <= limit => Ok(observed),
        observed => Err(ResourceExhausted {
            resource,
            limit,
            observed: observed.unwrap_or(usize::MAX),
        }),
    }
}

fn container_bytes(capacity: usize, limit: usize) -> Result<usize, ResourceExhausted> {
    let bytes = capacity
        .checked_mul(size_of::<Box<dyn AccountedEngineData>>())
        .ok_or(ResourceExhausted {
            resource: Resource::EvaluationBytes,
            limit,
            observed: usize::MAX,
        })?;
    charge(0, bytes, limit, Resource::EvaluationBytes)
}
