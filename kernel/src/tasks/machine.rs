use std::fmt;
use std::mem::size_of;

use super::protocol::ResponseKind;
use super::{
    CancelDisposition, CancelReason, CpuSlice, FailureKind, OperationFailure, OperationTask,
    RequestKey, Resource, ResourceExhausted, TaskAccounting, TaskId, TaskLimits, TaskProtocolError,
    TaskRequest, TaskRequestV1, TaskResponseV1, TaskStep, TaskUsage,
};

/// An unkeyed semantic decision made during one CPU turn.
pub enum TaskAction<T> {
    /// Admit a request before calling `TaskState::take_request` to construct its payload.
    Request,
    /// Continue semantic work in another CPU turn.
    Yield,
    /// Transfer the final admitted output.
    Complete(T),
}

/// Bounded owned semantic state driven by the shared protocol machine.
///
/// This is an implementer contract, like `AccountedEngineData`: the machine enforces transitions
/// and protocol counters, while each implementation must prove its own CPU and allocation bounds.
/// State retains no driver, handler, iterator, future, callback or execution context. It may retain
/// admitted pages, descriptors and numeric identities. Dropping state must release that ownership
/// immediately; cancellation does not call a semantic-state hook.
///
/// Producers admit response allocations before resumption. State validates actual response shape,
/// object identity, ordering, nested schemas and backing capacity before retaining them. Use the
/// supplied accounting ledger before copying, allocating or growing. A shared ledger borrow cannot
/// replace its limits or reset cumulative protocol counters. Report only semantic work domains;
/// the machine owns request, yield, CPU-turn and total task-state accounting.
pub trait TaskState {
    /// Admitted output transferred once on completion.
    type Output;

    /// Reports conservative dynamic backing allocations without allocating or performing I/O.
    ///
    /// Include full backing owners and collection capacities, but exclude `size_of::<Self>()`,
    /// which the machine includes in fixed state storage. Return typed exhaustion on overflow.
    fn retained_bytes(&self) -> Result<usize, ResourceExhausted>;

    /// Performs at most one CPU slice and returns an unkeyed decision or terminal failure.
    ///
    /// Returning `Request` must not construct or allocate its outgoing payload. Preserve its
    /// bounded source data for `take_request`, which runs only after protocol admission succeeds.
    fn advance(
        &mut self,
        cpu: CpuSlice,
        accounting: &TaskAccounting,
    ) -> Result<TaskAction<Self::Output>, OperationFailure>;

    /// Materializes the requested effect after checked request-count and identity admission.
    ///
    /// Admit requested I/O and owned payload bytes before allocating or cloning them. Moving
    /// previously admitted ownership is allowed. Return a typed failure on exhaustion or invalid
    /// semantic state; the machine then drops all state without issuing an effect.
    fn take_request(
        &mut self,
        accounting: &TaskAccounting,
    ) -> Result<TaskRequestV1, OperationFailure>;

    /// Validates a matching response and performs at most one CPU slice.
    ///
    /// Variant and evaluation-identity checks have already run. Matching operational errors never
    /// reach state. Malformed payloads must return `OperationFailure::malformed_response`; resource
    /// failures must remain terminal rather than triggering semantic fallback.
    fn resume(
        &mut self,
        response: TaskResponseV1,
        cpu: CpuSlice,
        accounting: &TaskAccounting,
    ) -> Result<TaskAction<Self::Output>, OperationFailure>;
}

/// Bounded, source-free protocol status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// The task owns admitted state but has not run.
    New,
    /// The task may run CPU work or await its one pending response.
    Running,
    /// Output was transferred and state was released.
    Complete,
    /// The original failure was transferred; only its fixed category remains.
    Failed(FailureKind),
    /// Cancellation released semantic state.
    Cancelled,
}

/// Shared request/resume protocol owning exactly one semantic continuation.
///
/// The machine releases the continuation before returning completion, failure or cancellation.
/// Its terminal state retains fixed protocol fields and numeric accounting only. Payload admission
/// and CPU compliance remain the concrete `TaskState` implementer's obligations.
pub struct TaskMachine<S: TaskState> {
    id: TaskId,
    next_request: u64,
    status: TaskStatus,
    pending: Option<(RequestKey, ResponseKind)>,
    semantic: Option<S>,
    accounting: TaskAccounting,
}

impl<S: TaskState> TaskMachine<S> {
    /// Consumes a unique driver token and already admitted semantic state.
    ///
    /// Validates fixed storage and reported backing bytes before retention. Exhaustion drops
    /// the supplied state and returns a typed failure; it cannot undo prior producer allocation.
    pub fn new(id: TaskId, semantic: S, limits: TaskLimits) -> Result<Self, OperationFailure> {
        let task = Self {
            id,
            next_request: 1,
            status: TaskStatus::New,
            pending: None,
            semantic: Some(semantic),
            accounting: TaskAccounting::new(limits),
        };
        task.observe(None)?;
        Ok(task)
    }

    /// Returns the fixed protocol status without exposing semantic ownership.
    pub fn status(&self) -> TaskStatus {
        self.status
    }

    /// Returns the driver identity consumed when this task was constructed.
    pub fn task_id(&self) -> u64 {
        self.id.get()
    }

    /// Returns the one outstanding key, if any.
    pub fn pending_key(&self) -> Option<RequestKey> {
        self.pending.map(|p| p.0)
    }

    /// Borrows numeric accounting, including cumulative work retained after terminal cleanup.
    pub fn accounting(&self) -> TaskUsage<'_> {
        TaskUsage {
            accounting: &self.accounting,
        }
    }

    fn require_running(&self) -> Result<(), TaskProtocolError> {
        match self.status {
            TaskStatus::New => Err(TaskProtocolError::NotStarted),
            TaskStatus::Running => Ok(()),
            _ => Err(TaskProtocolError::Terminal),
        }
    }

    fn advance(&mut self, cpu: CpuSlice) -> TaskStep<S::Output> {
        let action = self
            .admit_turn(cpu)
            .map_err(OperationFailure::from)
            .and_then(|()| match self.semantic.as_mut() {
                Some(state) => state.advance(cpu, &self.accounting),
                None => Err(OperationFailure::malformed_response()),
            });
        self.finish_action(action)
    }

    fn admit_turn(&self, cpu: CpuSlice) -> Result<(), ResourceExhausted> {
        self.accounting
            .check(Resource::TurnRecords, cpu.records())?;
        self.accounting
            .check(Resource::TurnInputBytes, cpu.bytes())?;
        self.accounting
            .check(Resource::TurnPlanNodes, cpu.plan_nodes())?;
        self.accounting.charge(Resource::CpuTurns, 1)
    }

    fn finish_action(
        &mut self,
        action: Result<TaskAction<S::Output>, OperationFailure>,
    ) -> TaskStep<S::Output> {
        match action.and_then(|action| self.finish_checked(action)) {
            Ok(step) => step,
            Err(error) => self.fail(error),
        }
    }

    fn finish_checked(
        &mut self,
        action: TaskAction<S::Output>,
    ) -> Result<TaskStep<S::Output>, OperationFailure> {
        match action {
            TaskAction::Request => {
                let next = self
                    .next_request
                    .checked_add(1)
                    .ok_or_else(|| OperationFailure::terminal(FailureKind::IdentityExhausted))?;
                self.accounting.next_charge(Resource::Requests, 1)?;
                let operation = match self.semantic.as_mut() {
                    Some(state) => state.take_request(&self.accounting)?,
                    None => return Err(OperationFailure::malformed_response()),
                };
                let kind = operation.response_kind();
                if let ResponseKind::Evaluation(evaluation) = kind {
                    if evaluation.task_id() != self.id.get() {
                        return Err(OperationFailure::malformed_response());
                    }
                }
                self.observe(Some(&operation))?;
                let key = RequestKey::new(self.id.get(), self.next_request)
                    .map_err(|_| OperationFailure::terminal(FailureKind::IdentityExhausted))?;
                self.accounting.charge(Resource::Requests, 1)?;
                self.next_request = next;
                self.pending = Some((key, kind));
                Ok(TaskStep::Execute(TaskRequest { key, operation }))
            }
            TaskAction::Yield => {
                self.accounting.charge(Resource::Yields, 1)?;
                self.observe(None)?;
                Ok(TaskStep::Yield)
            }
            TaskAction::Complete(output) => {
                self.observe(None)?;
                self.release();
                self.status = TaskStatus::Complete;
                Ok(TaskStep::Complete(output))
            }
        }
    }

    fn observe(&self, outgoing: Option<&TaskRequestV1>) -> Result<(), OperationFailure> {
        let overflow = || ResourceExhausted {
            resource: Resource::TaskStateBytes,
            limit: self.accounting.limit(Resource::TaskStateBytes),
            observed: usize::MAX,
        };
        let dynamic = match &self.semantic {
            Some(state) => state.retained_bytes()?,
            None => 0,
        };
        let retained = size_of::<Self>()
            .checked_add(dynamic)
            .ok_or_else(overflow)?;
        if let Some(request) = outgoing {
            let total = retained
                .checked_add(request.owned_bytes().ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
            self.accounting.set_live(Resource::TaskStateBytes, total)?;
        }
        self.accounting
            .set_live(Resource::TaskStateBytes, retained)?;
        Ok(())
    }

    fn release(&mut self) {
        self.pending = None;
        self.semantic = None;
        self.accounting.clear_live(size_of::<Self>());
    }

    fn fail(&mut self, error: OperationFailure) -> TaskStep<S::Output> {
        self.release();
        self.status = TaskStatus::Failed(error.kind());
        TaskStep::Failed(error)
    }
}

impl<S: TaskState> OperationTask for TaskMachine<S> {
    fn pending_work(&self) -> Result<super::PendingWork<'_>, TaskProtocolError> {
        self.require_running()?;
        let key = self.pending_key().ok_or(TaskProtocolError::WrongKey)?;
        Ok(super::PendingWork {
            key,
            accounting: &self.accounting,
        })
    }

    type Output = S::Output;

    fn start(&mut self, cpu: CpuSlice) -> Result<TaskStep<Self::Output>, TaskProtocolError> {
        if self.status == TaskStatus::Cancelled {
            return Ok(TaskStep::Cancelled);
        }
        if self.status != TaskStatus::New {
            return Err(TaskProtocolError::AlreadyStarted);
        }
        self.status = TaskStatus::Running;
        Ok(self.advance(cpu))
    }

    fn progress(&mut self, cpu: CpuSlice) -> Result<TaskStep<Self::Output>, TaskProtocolError> {
        self.require_running()?;
        if self.pending.is_some() {
            return Err(TaskProtocolError::PendingRequest);
        }
        Ok(self.advance(cpu))
    }

    fn resume(
        &mut self,
        key: RequestKey,
        response: Result<TaskResponseV1, OperationFailure>,
        cpu: CpuSlice,
    ) -> Result<TaskStep<Self::Output>, TaskProtocolError> {
        self.require_running()?;
        let Some((expected, kind)) = self.pending else {
            return Err(TaskProtocolError::WrongKey);
        };
        if key != expected {
            return Err(TaskProtocolError::WrongKey);
        }
        if response.as_ref().is_ok_and(|r| !kind.matches(r)) {
            return Err(TaskProtocolError::WrongKind);
        }
        self.pending = None;
        let response = match response {
            Ok(response) => response,
            Err(error) => return Ok(self.fail(error)),
        };
        if !kind.valid_identity(&response) {
            return Ok(self.fail(OperationFailure::malformed_response()));
        }
        let action = self
            .admit_turn(cpu)
            .map_err(OperationFailure::from)
            .and_then(|()| match self.semantic.as_mut() {
                Some(state) => state.resume(response, cpu, &self.accounting),
                None => Err(OperationFailure::malformed_response()),
            });
        Ok(self.finish_action(action))
    }

    fn cancel(&mut self, _reason: CancelReason) -> CancelDisposition {
        match self.status {
            TaskStatus::Cancelled => CancelDisposition::AlreadyCancelled,
            TaskStatus::Complete | TaskStatus::Failed(_) => CancelDisposition::AlreadyTerminal,
            TaskStatus::New | TaskStatus::Running => {
                let pending = self.pending_key();
                self.release();
                self.status = TaskStatus::Cancelled;
                CancelDisposition::Cancelled(pending)
            }
        }
    }
}

impl<S: TaskState> fmt::Debug for TaskMachine<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskMachine")
            .field("task", &self.id)
            .field("status", &self.status)
            .field("pending", &self.pending_key())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;

    struct Issue(Arc<AtomicUsize>);
    impl TaskState for Issue {
        type Output = ();
        fn retained_bytes(&self) -> Result<usize, ResourceExhausted> {
            Ok(0)
        }
        fn advance(
            &mut self,
            _: CpuSlice,
            _: &TaskAccounting,
        ) -> Result<TaskAction<()>, OperationFailure> {
            Ok(TaskAction::Request)
        }
        fn take_request(&mut self, _: &TaskAccounting) -> Result<TaskRequestV1, OperationFailure> {
            self.0.fetch_add(1, Ordering::Relaxed);
            panic!("request exhaustion must precede materialization")
        }
        fn resume(
            &mut self,
            _: TaskResponseV1,
            _: CpuSlice,
            _: &TaskAccounting,
        ) -> Result<TaskAction<()>, OperationFailure> {
            panic!("no response expected")
        }
    }

    #[test]
    fn request_id_overflow_terminates_before_materialization() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut task = TaskMachine::new(
            TaskId::allocate().unwrap(),
            Issue(calls.clone()),
            TaskLimits::qualification(),
        )
        .unwrap();
        task.next_request = u64::MAX;
        let TaskStep::Failed(error) = task.start(CpuSlice::new(1, 1, 1).unwrap()).unwrap() else {
            panic!("expected exhaustion")
        };
        assert_eq!(error.kind(), FailureKind::IdentityExhausted);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert!(task.semantic.is_none());
        assert!(task.pending.is_none());
        assert_eq!(task.accounting().usage(Resource::Requests).consumed(), 0);
    }
}
