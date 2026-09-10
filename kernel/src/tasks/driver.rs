use super::{
    AccountedEngineData, AdmittedPlan, EvaluationKey, EvaluationLimits, EvaluationPageLimits,
    EvaluationReader, EvaluationUsage, OperationFailure, RequestKey, TaskId, TaskProtocolError,
    TaskRequest, TaskRequestV1, TaskResponseV1,
};

/// Lazy evaluation output whose allocations were admitted by its producer before each yield.
pub type AdmittedEvaluationSource =
    Box<dyn Iterator<Item = Result<Box<dyn AccountedEngineData>, OperationFailure>> + Send>;

struct ActiveEvaluation {
    key: EvaluationKey,
    page_limits: EvaluationPageLimits,
    reader: EvaluationReader,
}

#[derive(Debug, Clone, Copy)]
struct RequestSequence {
    next: u64,
}

impl RequestSequence {
    fn checked_next(self, task_id: u64, key: RequestKey) -> Result<u64, TaskProtocolError> {
        if key.task_id() != task_id || key.request_id() != self.next {
            return Err(TaskProtocolError::WrongKey);
        }
        self.next
            .checked_add(1)
            .ok_or(TaskProtocolError::IdentityExhausted)
    }
}

/// Driver-owned state for one task's admitted internal evaluation.
///
/// The driver separates request dispatch, effect completion and response transfer so callers can
/// yield between them. The compiler consumes an [`AdmittedPlan`] once and returns a lazy source
/// that admitted its own allocations. This component deliberately accepts no storage effects and
/// does not adapt an ordinary plan executor.
pub struct EvaluationDriver<C> {
    task_id: u64,
    sequence: RequestSequence,
    started: bool,
    effect: Option<TaskRequest>,
    response: Option<(RequestKey, Result<TaskResponseV1, OperationFailure>)>,
    active: Option<ActiveEvaluation>,
    usage: EvaluationUsage,
    compiler: C,
    terminal: bool,
    cancelled: bool,
}

impl<C> EvaluationDriver<C>
where
    C: FnMut(AdmittedPlan, EvaluationLimits) -> Result<AdmittedEvaluationSource, OperationFailure>,
{
    /// Allocates a fresh task identity and its initially idle evaluation driver.
    pub fn allocate(compiler: C) -> Result<(TaskId, Self), TaskProtocolError> {
        let id = TaskId::allocate()?;
        let task_id = id.get();
        Ok((
            id,
            Self {
                task_id,
                sequence: RequestSequence { next: 1 },
                started: false,
                effect: None,
                response: None,
                active: None,
                usage: EvaluationUsage::default(),
                compiler,
                terminal: false,
                cancelled: false,
            },
        ))
    }

    /// Allocates a checked evaluation identity owned by this driver's task.
    pub fn allocate_evaluation(&self) -> Result<EvaluationKey, TaskProtocolError> {
        if self.terminal || self.started {
            return Err(TaskProtocolError::Terminal);
        }
        EvaluationKey::allocate(self.task_id)
    }

    /// Validates and takes one effect request without executing it.
    ///
    /// Wrong task/request identities, duplicate starts, changed page limits and non-evaluation
    /// effects leave the existing driver state unchanged.
    pub fn dispatch(&mut self, request: TaskRequest) -> Result<RequestKey, TaskProtocolError> {
        if self.terminal {
            return Err(TaskProtocolError::Terminal);
        }
        if self.effect.is_some() || self.response.is_some() {
            return Err(TaskProtocolError::PendingRequest);
        }
        let next = self.sequence.checked_next(self.task_id, request.key)?;
        self.validate_operation(&request.operation)?;
        if matches!(request.operation, TaskRequestV1::EvaluationStart { .. }) {
            self.started = true;
        }
        self.sequence.next = next;
        let key = request.key;
        self.effect = Some(request);
        Ok(key)
    }

    /// Executes the dispatched effect once and retains its owned response for transfer.
    pub fn complete_effect(&mut self) -> Result<RequestKey, TaskProtocolError> {
        if self.cancelled {
            return Err(TaskProtocolError::Terminal);
        }
        if self.response.is_some() {
            return Err(TaskProtocolError::PendingRequest);
        }
        let missing = if self.terminal {
            TaskProtocolError::Terminal
        } else {
            TaskProtocolError::NotStarted
        };
        let request = self.effect.take().ok_or(missing)?;
        let key = request.key;
        let response = self.execute(request.operation);
        if response.is_err() {
            self.active = None;
            self.terminal = true;
        }
        self.response = Some((key, response));
        Ok(key)
    }

    /// Transfers one completed response, preserving it when the supplied key is wrong.
    pub fn take_response(
        &mut self,
        key: RequestKey,
    ) -> Result<Result<TaskResponseV1, OperationFailure>, TaskProtocolError> {
        match self.response.as_ref() {
            Some((actual, _)) if *actual != key => return Err(TaskProtocolError::WrongKey),
            Some(_) => (),
            None if self.terminal => return Err(TaskProtocolError::Terminal),
            None => return Err(TaskProtocolError::NotStarted),
        }
        match self.response.take() {
            Some((_, response)) => Ok(response),
            None => Err(TaskProtocolError::NotStarted),
        }
    }

    /// Drops the matching queued effect or response and releases the active source without pulls.
    pub fn cancel(&mut self, pending: Option<RequestKey>) -> Result<(), TaskProtocolError> {
        if self.cancelled {
            return Ok(());
        }
        if pending != self.outstanding_key() {
            return Err(TaskProtocolError::WrongKey);
        }
        self.effect = None;
        self.response = None;
        if let Some(mut active) = self.active.take() {
            active.reader.cancel();
            self.usage = active.reader.usage();
        }
        self.terminal = true;
        self.cancelled = true;
        Ok(())
    }

    /// Returns the queued effect or completed response key retained by the driver.
    pub fn outstanding_key(&self) -> Option<RequestKey> {
        self.effect
            .as_ref()
            .map(|request| request.key)
            .or_else(|| self.response.as_ref().map(|(key, _)| *key))
    }

    /// Returns the active evaluation identity while its lazy source is retained.
    pub fn active_evaluation(&self) -> Option<EvaluationKey> {
        self.active.as_ref().map(|active| active.key)
    }

    /// Returns cumulative evaluation work separately from live source ownership.
    pub fn evaluation_usage(&self) -> EvaluationUsage {
        self.usage
    }

    fn validate_operation(&self, operation: &TaskRequestV1) -> Result<(), TaskProtocolError> {
        match operation {
            TaskRequestV1::EvaluationStart { evaluation, .. } => {
                if evaluation.task_id() != self.task_id {
                    return Err(TaskProtocolError::WrongKey);
                }
                if self.started || self.active.is_some() {
                    return Err(TaskProtocolError::WrongKind);
                }
                Ok(())
            }
            TaskRequestV1::Evaluation { evaluation, limits } => {
                let active = self.active.as_ref().ok_or(TaskProtocolError::WrongKind)?;
                if *evaluation != active.key {
                    return Err(TaskProtocolError::WrongKey);
                }
                if *limits != active.page_limits {
                    return Err(TaskProtocolError::WrongKind);
                }
                Ok(())
            }
            _ => Err(TaskProtocolError::WrongKind),
        }
    }

    fn execute(&mut self, operation: TaskRequestV1) -> Result<TaskResponseV1, OperationFailure> {
        match operation {
            TaskRequestV1::EvaluationStart {
                evaluation,
                plan,
                limits,
            } => {
                let source = (self.compiler)(plan, limits)?;
                self.read_page(
                    evaluation,
                    limits.page(),
                    EvaluationReader::new(source, limits),
                )
            }
            TaskRequestV1::Evaluation { evaluation, .. } => {
                let active = self
                    .active
                    .take()
                    .ok_or_else(OperationFailure::malformed_response)?;
                self.read_page(evaluation, active.page_limits, active.reader)
            }
            _ => Err(OperationFailure::malformed_response()),
        }
    }

    fn read_page(
        &mut self,
        evaluation: EvaluationKey,
        page_limits: EvaluationPageLimits,
        mut reader: EvaluationReader,
    ) -> Result<TaskResponseV1, OperationFailure> {
        let page = reader.next_page();
        self.usage = reader.usage();
        match page {
            Ok(Some(page)) => {
                self.active = Some(ActiveEvaluation {
                    key: evaluation,
                    page_limits,
                    reader,
                });
                Ok(TaskResponseV1::Evaluation {
                    evaluation,
                    page: Some(page),
                })
            }
            Ok(None) => {
                self.terminal = true;
                Ok(TaskResponseV1::Evaluation {
                    evaluation,
                    page: None,
                })
            }
            Err(error) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RequestSequence;
    use crate::tasks::{RequestKey, TaskProtocolError};

    #[test]
    fn request_sequence_overflow_preserves_driver_state() {
        let sequence = RequestSequence { next: u64::MAX };
        let key = RequestKey::new(7, u64::MAX).unwrap();
        assert_eq!(
            sequence.checked_next(7, key),
            Err(TaskProtocolError::IdentityExhausted)
        );
        assert_eq!(sequence.next, u64::MAX);
    }
}
