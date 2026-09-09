use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{
    EvaluationPage, EvaluationPageLimits, FooterLimits, OperationFailure, TaskProtocolError,
};
use crate::ParquetFooter;

static NEXT_TASK_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_EVALUATION_ID: AtomicU64 = AtomicU64::new(1);

/// A nonzero process-local identity allocated by a driver and consumed by one task.
///
/// The token cannot be cloned or reconstructed from its numeric value. It is not a durable or
/// serializable identity, and does not authenticate effects from an untrusted driver.
#[derive(Debug, PartialEq, Eq)]
pub struct TaskId(u64);

impl TaskId {
    /// Allocates a fresh driver identity, or returns `IdentityExhausted` instead of wrapping.
    pub fn allocate() -> Result<Self, TaskProtocolError> {
        allocate_identity(&NEXT_TASK_ID).map(Self)
    }

    /// Returns the nonzero value for request routing and diagnostics.
    pub fn get(&self) -> u64 {
        self.0
    }
}

/// Identifies one outstanding request within one task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestKey {
    task: u64,
    request: u64,
}

impl RequestKey {
    /// Reconstructs a response key, rejecting either reserved zero value.
    ///
    /// Constructing a key does not create a task or authorize an effect. Drivers echo the key
    /// received from the task; resumption validates it against the outstanding request.
    pub fn new(task: u64, request: u64) -> Result<Self, TaskProtocolError> {
        if task == 0 || request == 0 {
            return Err(TaskProtocolError::InvalidIdentity);
        }
        Ok(Self { task, request })
    }

    /// Returns the task's driver-assigned identity.
    pub fn task_id(self) -> u64 {
        self.task
    }

    /// Returns the task-local request sequence number, starting at one.
    pub fn request_id(self) -> u64 {
        self.request
    }
}

/// A checked reference to evaluation state owned by the driver, not by a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvaluationKey {
    task: u64,
    evaluation: u64,
}

impl EvaluationKey {
    /// Allocates an evaluation identity for a nonzero task identity.
    ///
    /// Returns `InvalidIdentity` for zero or `IdentityExhausted` on allocator exhaustion.
    /// The driver must bind the result to its admitted evaluation before issuing pages.
    pub fn allocate(task: u64) -> Result<Self, TaskProtocolError> {
        if task == 0 {
            return Err(TaskProtocolError::InvalidIdentity);
        }
        Ok(Self {
            task,
            evaluation: allocate_identity(&NEXT_EVALUATION_ID)?,
        })
    }

    /// Returns the owning task identity.
    pub fn task_id(self) -> u64 {
        self.task
    }

    /// Returns the checked driver-local evaluation identity.
    pub fn evaluation_id(self) -> u64 {
        self.evaluation
    }
}

/// Validated work allowances for one cooperative CPU turn.
///
/// This is a work budget, not a wall-clock deadline. Implementers must yield inside wide schemas,
/// records and expression trees rather than only between outer operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuSlice {
    records: usize,
    bytes: usize,
    plan_nodes: usize,
}

impl CpuSlice {
    /// Sets record, input-byte and plan-node allowances.
    ///
    /// Returns `InvalidLimits` for zero allowances or values above the finite qualification
    /// turn ceilings: 1024 records, 1 MiB input and 256 plan nodes.
    pub fn new(records: usize, bytes: usize, plan_nodes: usize) -> Result<Self, TaskProtocolError> {
        if records == 0
            || records > 1024
            || bytes == 0
            || bytes > 1 << 20
            || plan_nodes == 0
            || plan_nodes > 256
        {
            return Err(TaskProtocolError::InvalidLimits);
        }
        Ok(Self {
            records,
            bytes,
            plan_nodes,
        })
    }

    /// Returns the maximum records processed in this turn.
    pub fn records(self) -> usize {
        self.records
    }

    /// Returns the maximum input bytes processed in this turn.
    pub fn bytes(self) -> usize {
        self.bytes
    }

    /// Returns the maximum plan nodes processed in this turn.
    pub fn plan_nodes(self) -> usize {
        self.plan_nodes
    }
}

/// Owned semantic metadata from an admitted listing source.
///
/// Producers bound discovery and allocation before creating descriptors. Consumers account the
/// actual container and string capacities before retaining a page. Backend version and identity
/// validation remain the driver's responsibility at the admitted I/O boundary.
pub struct FileDescriptor {
    /// The full object path, ordered using the storage path ordering.
    pub path: String,
    /// The object's byte length.
    pub size: u64,
    /// Last modification time in milliseconds since the Unix epoch.
    pub modification_time: i64,
}

/// Read-only work transferred to a driver after protocol admission.
///
/// Request materialization must admit owned bytes before allocation. The driver additionally
/// admits each effect's I/O and destination allocations before performing it. These payloads do
/// not adapt legacy unbounded storage or plan-executor results into admitted sources.
#[non_exhaustive]
pub enum TaskRequestV1 {
    /// Returns an admitted listing page without iterator lookahead.
    List {
        /// The listing root.
        root: String,
        /// The bounded cursor returned by the preceding page.
        continuation: Option<String>,
        /// Maximum descriptors in the response.
        entries: usize,
        /// Maximum descriptor backing and container bytes.
        descriptor_bytes: usize,
        /// Maximum cursor backing bytes.
        continuation_bytes: usize,
    },
    /// Reads exactly the requested range into admitted owned storage.
    Read {
        /// The admitted object path.
        path: String,
        /// The first byte offset.
        offset: u64,
        /// Exact requested length.
        length: usize,
    },
    /// Reads the size of an admitted object.
    Head {
        /// The admitted object path.
        path: String,
    },
    /// Uses the driver's injected limited Parquet reader.
    Footer {
        /// The admitted object path.
        path: String,
        /// Size established by the driver's identity-checked HEAD.
        size: u64,
        /// The footer, schema, row-group, metadata and page-index allowances.
        limits: FooterLimits,
    },
    /// Requests another page of an already admitted driver-owned evaluation.
    Evaluation {
        /// Identity bound to this task by the driver.
        evaluation: EvaluationKey,
        /// Bounds supplied to the producer before it constructs the page.
        limits: EvaluationPageLimits,
    },
}

/// An owned effect request and the exact key the driver must echo on resumption.
pub struct TaskRequest {
    /// The task identity and checked request sequence number.
    pub key: RequestKey,
    /// The semantic effect to execute.
    pub operation: TaskRequestV1,
}

/// Owned responses from an admitted driver.
///
/// The transition machine validates keys and variants before consuming pending state. Concrete
/// semantic states validate range shape, ordering, object identity, schema and actual backing
/// capacity before retaining payloads. The producer must establish allocation admission first;
/// in particular, a plain `ParquetFooter` alone does not prove limited-decoder provenance.
#[non_exhaustive]
pub enum TaskResponseV1 {
    /// An admitted listing page; `None` is definitive end-of-listing.
    Listing {
        /// Owned descriptors, including their container capacity.
        files: Vec<FileDescriptor>,
        /// The bounded continuation for the next page.
        continuation: Option<String>,
    },
    /// Exact owned read bytes. Visible length alone is not a backing-allocation charge.
    Bytes {
        /// Actual first byte offset.
        offset: u64,
        /// Response storage whose allocation was admitted by the driver.
        bytes: Vec<u8>,
        /// Whether the end of this range is the identity-checked object end.
        eof: bool,
    },
    /// Identity-checked object size.
    Head {
        /// Byte length of the admitted object.
        size: u64,
    },
    /// Metadata produced by the injected limited Parquet reader.
    Footer {
        /// The converted schema, requiring owner-local admission before task retention.
        footer: ParquetFooter,
    },
    /// One internal page or definitive EOF of the referenced evaluation.
    Evaluation {
        /// Exact identity from the request.
        evaluation: EvaluationKey,
        /// An admitted page; `None` establishes EOF without retaining a source.
        page: Option<EvaluationPage>,
    },
}

/// Reason for caller-driven cancellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CancelReason {
    /// The caller no longer needs the operation.
    Caller,
}

/// Cancellation result after the task has released its semantic state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelDisposition {
    /// The driver must release this pending effect and any response awaiting resumption.
    Cancelled(Option<RequestKey>),
    /// Cancellation has already released task-owned state.
    AlreadyCancelled,
    /// Completion or failure is immutable; cancellation did not change it.
    AlreadyTerminal,
}

/// The next owned outcome of a cooperative operation.
pub enum TaskStep<T> {
    /// The driver must perform this effect and resume with its exact key.
    Execute(TaskRequest),
    /// A CPU allowance was consumed; call `progress` in another turn.
    Yield,
    /// Transfers the completed output once.
    Complete(T),
    /// Transfers an operational failure and its original source once.
    Failed(OperationFailure),
    /// The task was cancelled before completion.
    Cancelled,
}

/// An owned, cooperative operation driven through explicit effects and CPU turns.
///
/// Implementations retain bounded semantic data and numeric identities only. Drivers own
/// handlers, iterators, futures, evaluation execution and cancellation mechanisms. Payload
/// construction requires producer admission; protocol validation cannot establish it afterward.
pub trait OperationTask {
    /// The output transferred exactly once on successful completion.
    type Output;

    /// Starts one CPU turn. Repeated starts return `AlreadyStarted`; a cancelled task stays
    /// cancelled.
    fn start(&mut self, cpu: CpuSlice) -> Result<TaskStep<Self::Output>, TaskProtocolError>;

    /// Advances one CPU turn, rejecting unstarted, pending or terminal tasks without mutation.
    fn progress(&mut self, cpu: CpuSlice) -> Result<TaskStep<Self::Output>, TaskProtocolError>;

    /// Resumes one matching effect. Wrong keys or variants preserve pending state; matching
    /// malformed responses and operational failures terminate the task and release its state.
    fn resume(
        &mut self,
        key: RequestKey,
        response: Result<TaskResponseV1, OperationFailure>,
        cpu: CpuSlice,
    ) -> Result<TaskStep<Self::Output>, TaskProtocolError>;

    /// Immediately releases task-owned state and returns the outstanding key for driver cleanup.
    fn cancel(&mut self, reason: CancelReason) -> CancelDisposition;
}

impl fmt::Debug for FileDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileDescriptor")
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for TaskRequestV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::List { .. } => "List(..)",
            Self::Read { .. } => "Read(..)",
            Self::Head { .. } => "Head(..)",
            Self::Footer { .. } => "Footer(..)",
            Self::Evaluation { .. } => "Evaluation(..)",
        })
    }
}

impl fmt::Debug for TaskRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskRequest")
            .field("key", &self.key)
            .field("operation", &self.operation)
            .finish()
    }
}

impl fmt::Debug for TaskResponseV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Listing { .. } => "Listing(..)",
            Self::Bytes { .. } => "Bytes(..)",
            Self::Head { .. } => "Head(..)",
            Self::Footer { .. } => "Footer(..)",
            Self::Evaluation { .. } => "Evaluation(..)",
        })
    }
}

impl<T> fmt::Debug for TaskStep<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Execute(request) => f.debug_tuple("Execute").field(request).finish(),
            Self::Yield => f.write_str("Yield"),
            Self::Complete(_) => f.write_str("Complete(..)"),
            Self::Failed(error) => f.debug_tuple("Failed").field(&error.kind()).finish(),
            Self::Cancelled => f.write_str("Cancelled"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum ResponseKind {
    List,
    Read,
    Head,
    Footer,
    Evaluation(EvaluationKey),
}

impl TaskRequestV1 {
    pub(super) fn response_kind(&self) -> ResponseKind {
        match self {
            Self::List { .. } => ResponseKind::List,
            Self::Read { .. } => ResponseKind::Read,
            Self::Head { .. } => ResponseKind::Head,
            Self::Footer { .. } => ResponseKind::Footer,
            Self::Evaluation { evaluation, .. } => ResponseKind::Evaluation(*evaluation),
        }
    }

    pub(super) fn owned_bytes(&self) -> Option<usize> {
        let backing = match self {
            Self::List {
                root, continuation, ..
            } => root
                .capacity()
                .checked_add(continuation.as_ref().map_or(0, String::capacity))?,
            Self::Read { path, .. } | Self::Head { path } | Self::Footer { path, .. } => {
                path.capacity()
            }
            Self::Evaluation { .. } => 0,
        };
        std::mem::size_of::<TaskRequest>().checked_add(backing)
    }
}

impl ResponseKind {
    pub(super) fn matches(self, response: &TaskResponseV1) -> bool {
        matches!(
            (self, response),
            (Self::List, TaskResponseV1::Listing { .. })
                | (Self::Read, TaskResponseV1::Bytes { .. })
                | (Self::Head, TaskResponseV1::Head { .. })
                | (Self::Footer, TaskResponseV1::Footer { .. })
                | (Self::Evaluation(_), TaskResponseV1::Evaluation { .. })
        )
    }

    pub(super) fn valid_identity(self, response: &TaskResponseV1) -> bool {
        match (self, response) {
            (Self::Evaluation(expected), TaskResponseV1::Evaluation { evaluation, .. }) => {
                expected == *evaluation
            }
            _ => true,
        }
    }
}

fn allocate_identity(counter: &AtomicU64) -> Result<u64, TaskProtocolError> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| {
            if id == 0 {
                None
            } else {
                id.checked_add(1)
            }
        })
        .map_err(|_| TaskProtocolError::IdentityExhausted)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::schema::StructType;

    #[test]
    fn response_kinds_match_only_their_variant_and_debug_does_not_visit_payloads() {
        let id = TaskId::allocate().unwrap();
        let evaluation = EvaluationKey::allocate(id.get()).unwrap();
        let kinds = [
            ResponseKind::List,
            ResponseKind::Read,
            ResponseKind::Head,
            ResponseKind::Footer,
            ResponseKind::Evaluation(evaluation),
        ];
        let responses = [
            TaskResponseV1::Listing {
                files: vec![],
                continuation: None,
            },
            TaskResponseV1::Bytes {
                offset: 0,
                bytes: vec![42; 100_000],
                eof: true,
            },
            TaskResponseV1::Head { size: 1 },
            TaskResponseV1::Footer {
                footer: ParquetFooter {
                    schema: Arc::new(StructType::try_new([]).unwrap()),
                },
            },
            TaskResponseV1::Evaluation {
                evaluation,
                page: None,
            },
        ];
        for (i, kind) in kinds.into_iter().enumerate() {
            for (j, response) in responses.iter().enumerate() {
                assert_eq!(kind.matches(response), i == j);
                assert!(format!("{response:?}").len() < 32);
            }
        }
        let request = TaskRequestV1::Head {
            path: "never-echo".repeat(8192),
        };
        assert_eq!(format!("{request:?}"), "Head(..)");
        struct NotDebug;
        assert_eq!(
            format!("{:?}", TaskStep::Complete(NotDebug)),
            "Complete(..)"
        );
    }

    #[test]
    fn identity_overflow_and_zero_never_wrap_or_issue_an_identity() {
        let ids = AtomicU64::new(u64::MAX - 1);
        assert_eq!(allocate_identity(&ids), Ok(u64::MAX - 1));
        for _ in 0..2 {
            assert_eq!(
                allocate_identity(&ids),
                Err(TaskProtocolError::IdentityExhausted)
            );
            assert_eq!(ids.load(Ordering::Relaxed), u64::MAX);
        }
        assert_eq!(
            allocate_identity(&AtomicU64::new(0)),
            Err(TaskProtocolError::IdentityExhausted)
        );
        assert_eq!(
            RequestKey::new(0, 1),
            Err(TaskProtocolError::InvalidIdentity)
        );
        assert_eq!(
            RequestKey::new(1, 0),
            Err(TaskProtocolError::InvalidIdentity)
        );
        assert_eq!(
            EvaluationKey::allocate(0),
            Err(TaskProtocolError::InvalidIdentity)
        );
    }
}
