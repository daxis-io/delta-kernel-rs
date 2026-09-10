//! Bounded, owned data exchanged by cooperative Kernel operations.
//!
//! Engines admit allocations before constructing responses. Validation here checks the transfer
//! into Kernel; it cannot undo an allocation already performed by an engine. Accounted bytes are
//! conservative backing-allocation charges, not total process memory.

mod accounting;
mod driver;
mod evaluation;
mod machine;
mod plan_admission;
mod plan_shape;
mod protocol;

use std::error::Error as StdError;
use std::fmt;

pub use accounting::{
    FooterLimits, Resource, ResourceUsage, TaskAccounting, TaskLimits, TaskUsage,
};
pub use driver::{AdmittedEvaluationSource, EvaluationDriver};
pub use evaluation::{
    AccountedEngineData, EvaluationLimits, EvaluationPage, EvaluationPageLimits, EvaluationReader,
    EvaluationUsage,
};
pub use machine::{TaskAction, TaskMachine, TaskState, TaskStatus};
pub use plan_admission::{AdmittedPlan, PlanAdmissionError, PlanMetadataEntry};
pub use plan_shape::{PlanShape, PlanShapeError};
pub use protocol::{
    CancelDisposition, CancelReason, CpuSlice, EvaluationKey, FileDescriptor, OperationTask,
    RequestKey, TaskId, TaskRequest, TaskRequestV1, TaskResponseV1, TaskStep,
};

use crate::Error;

/// A configured limit was exceeded, or its checked accounting overflowed.
///
/// On overflow, `observed` is `usize::MAX`, even when the limit is also `usize::MAX`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceExhausted {
    /// The exhausted accounting domain.
    pub resource: Resource,
    /// The configured limit.
    pub limit: usize,
    /// The attempted charge, saturated only when arithmetic overflowed.
    pub observed: usize,
}

impl fmt::Display for ResourceExhausted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} exhausted: limit {}, observed {}",
            self.resource, self.limit, self.observed
        )
    }
}

impl StdError for ResourceExhausted {}

/// Source-free terminal failure information, safe to retain after transferring an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FailureKind {
    /// A finite resource budget was exhausted.
    ResourceExhausted(ResourceExhausted),
    /// The engine failed for a reason represented by the original error.
    Engine,
    /// Execution was cancelled.
    Cancelled,
    /// A matching response has invalid contents or an inconsistent evaluation identity.
    MalformedResponse,
    /// A checked request identity cannot advance without wrapping.
    IdentityExhausted,
}

impl fmt::Display for FailureKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResourceExhausted(error) => error.fmt(f),
            Self::Engine => f.write_str("engine operation failed"),
            Self::Cancelled => f.write_str("operation cancelled"),
            Self::MalformedResponse => f.write_str("malformed task response"),
            Self::IdentityExhausted => f.write_str("task identity exhausted"),
        }
    }
}

impl StdError for FailureKind {}

/// An operational failure owning its original, non-cloneable Kernel error at most once.
#[derive(Debug)]
pub struct OperationFailure {
    kind: FailureKind,
    source: Option<Box<Error>>,
}

impl OperationFailure {
    /// Reports invalid contents of a matching response without retaining the payload.
    pub fn malformed_response() -> Self {
        Self::terminal(FailureKind::MalformedResponse)
    }

    /// Takes ownership of an engine error with its typed category.
    ///
    /// Drivers classify typed sources directly; they must not parse error messages. A Kernel
    /// cancellation, including Kernel backtrace wrappers, always retains the cancellation category
    /// regardless of the supplied `kind`.
    pub fn new(kind: FailureKind, source: Error) -> Self {
        let mut underlying = &source;
        while let Error::Backtraced { source, .. } = underlying {
            underlying = source;
        }
        let kind = if matches!(underlying, Error::Cancelled) {
            FailureKind::Cancelled
        } else {
            kind
        };
        Self {
            kind,
            source: Some(Box::new(source)),
        }
    }

    /// Returns the source-free category retained by terminal state.
    pub fn kind(&self) -> FailureKind {
        self.kind
    }

    /// Transfers the original error, canonicalizing cancellation to [`Error::Cancelled`].
    ///
    /// Locally detected failures without an original source use a typed generic-error source.
    /// Cancellation releases any original diagnostic context instead of wrapping the cancellation.
    pub fn into_error(self) -> Error {
        if self.kind == FailureKind::Cancelled {
            return Error::Cancelled;
        }
        match self.source {
            Some(source) => *source,
            None => Error::generic_err(self.kind),
        }
    }

    fn terminal(kind: FailureKind) -> Self {
        Self { kind, source: None }
    }
}

impl From<ResourceExhausted> for OperationFailure {
    fn from(error: ResourceExhausted) -> Self {
        Self::terminal(FailureKind::ResourceExhausted(error))
    }
}

impl fmt::Display for OperationFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.kind.fmt(f)
    }
}

impl StdError for OperationFailure {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source.as_deref().map(|error| error as &dyn StdError)
    }
}

/// Caller misuse or invalid configuration, distinct from an operational failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TaskProtocolError {
    /// Limits cannot represent a progressing, bounded page.
    InvalidLimits,
    /// The task has not been started.
    NotStarted,
    /// The task has already been started.
    AlreadyStarted,
    /// A request must be resumed or cancelled before more CPU work.
    PendingRequest,
    /// No outstanding request has the supplied key.
    WrongKey,
    /// The response variant does not match the outstanding request.
    WrongKind,
    /// A completed, failed or cancelled task cannot be resumed.
    Terminal,
    /// An identity contains a reserved zero value.
    InvalidIdentity,
    /// The process-local driver identity allocator has exhausted its range.
    IdentityExhausted,
}

impl fmt::Display for TaskProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimits => f.write_str("invalid task limits"),
            Self::NotStarted => f.write_str("task not started"),
            Self::AlreadyStarted => f.write_str("task already started"),
            Self::PendingRequest => f.write_str("task has an outstanding request"),
            Self::WrongKey => f.write_str("wrong task request key"),
            Self::WrongKind => f.write_str("wrong task response kind"),
            Self::Terminal => f.write_str("task is terminal"),
            Self::InvalidIdentity => f.write_str("task identities must be nonzero"),
            Self::IdentityExhausted => f.write_str("driver identity exhausted"),
        }
    }
}

impl StdError for TaskProtocolError {}
