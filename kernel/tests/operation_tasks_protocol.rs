#![cfg(feature = "operation-tasks")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use delta_kernel::engine_data::RowVisitor;
use delta_kernel::expressions::{ArrayData, ColumnName};
use delta_kernel::schema::SchemaRef;
use delta_kernel::tasks::{
    AccountedEngineData, AdmittedPlan, CancelDisposition, CancelReason, CpuSlice, EvaluationKey,
    EvaluationLimits, EvaluationPage, EvaluationPageLimits, EvaluationReader, FailureKind,
    OperationFailure, OperationTask, RequestKey, Resource, ResourceExhausted, TaskAccounting,
    TaskAction, TaskId, TaskLimits, TaskMachine, TaskProtocolError, TaskRequestV1, TaskResponseV1,
    TaskState, TaskStatus, TaskStep,
};
use delta_kernel::{DeltaResult, EngineData, Error};

#[test]
fn cpu_slices_require_progress_and_respect_the_qualification_turn_bounds() {
    for (records, bytes, nodes) in [
        (0, 1, 1),
        (1, 0, 1),
        (1025, 1, 1),
        (1, (1 << 20) + 1, 1),
        (1, 1, 0),
        (1, 1, 257),
    ] {
        assert_eq!(
            CpuSlice::new(records, bytes, nodes),
            Err(TaskProtocolError::InvalidLimits)
        );
    }
    assert!(CpuSlice::new(1, 1, 1).is_ok());
    assert!(CpuSlice::new(1024, 1 << 20, 256).is_ok());
}

#[test]
fn driver_task_ids_are_nonzero_and_distinct() {
    let first = TaskId::allocate().unwrap();
    let second = TaskId::allocate().unwrap();
    assert_ne!(first, second);
    assert_ne!(first.get(), 0);
    assert_ne!(second.get(), 0);
}

struct Replay {
    path: Option<String>,
    bytes: Vec<u8>,
    position: usize,
    sum: usize,
    drops: Arc<AtomicUsize>,
    requests: Arc<AtomicUsize>,
}

impl Drop for Replay {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

impl TaskState for Replay {
    type Output = usize;

    fn retained_bytes(&self) -> Result<usize, ResourceExhausted> {
        Ok(self.bytes.capacity() + self.path.as_ref().map_or(0, String::capacity))
    }

    fn advance(
        &mut self,
        cpu: CpuSlice,
        accounting: &TaskAccounting,
    ) -> Result<TaskAction<Self::Output>, OperationFailure> {
        if self.path.is_some() {
            return Ok(TaskAction::Request);
        }
        let end = (self.position + cpu.bytes().min(cpu.records())).min(self.bytes.len());
        accounting.charge(Resource::Records, end - self.position)?;
        while self.position < end {
            self.sum += usize::from(self.bytes[self.position]);
            self.position += 1;
        }
        Ok(if self.position < self.bytes.len() {
            TaskAction::Yield
        } else {
            TaskAction::Complete(self.sum)
        })
    }

    fn take_request(
        &mut self,
        accounting: &TaskAccounting,
    ) -> Result<TaskRequestV1, OperationFailure> {
        accounting.charge(Resource::RequestedReadBytes, 3)?;
        self.requests.fetch_add(1, Ordering::Relaxed);
        Ok(TaskRequestV1::Read {
            path: self.path.take().unwrap(),
            offset: 0,
            length: 3,
        })
    }

    fn resume(
        &mut self,
        response: TaskResponseV1,
        cpu: CpuSlice,
        accounting: &TaskAccounting,
    ) -> Result<TaskAction<Self::Output>, OperationFailure> {
        let TaskResponseV1::Bytes { offset, bytes, eof } = response else {
            panic!("machine must validate response kind before calling state")
        };
        if offset != 0 || bytes.len() != 3 || !eof {
            return Err(OperationFailure::malformed_response());
        }
        accounting.check(Resource::ReadPayloadBytes, bytes.capacity())?;
        accounting.charge(Resource::InputBytes, bytes.len())?;
        self.bytes = bytes;
        self.advance(cpu, accounting)
    }
}

fn replay(limits: TaskLimits) -> (TaskMachine<Replay>, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let drops = Arc::new(AtomicUsize::new(0));
    let requests = Arc::new(AtomicUsize::new(0));
    let state = Replay {
        path: Some("memory:///already-admitted-log".to_owned()),
        bytes: Vec::new(),
        position: 0,
        sum: 0,
        drops: drops.clone(),
        requests: requests.clone(),
    };
    (
        TaskMachine::new(TaskId::allocate().unwrap(), state, limits).unwrap(),
        drops,
        requests,
    )
}

fn cpu() -> CpuSlice {
    CpuSlice::new(1, 1, 1).unwrap()
}

fn bytes(offset: u64) -> Result<TaskResponseV1, OperationFailure> {
    Ok(TaskResponseV1::Bytes {
        offset,
        bytes: vec![1, 2, 3],
        eof: true,
    })
}

fn start(task: &mut TaskMachine<Replay>) -> RequestKey {
    let TaskStep::Execute(request) = task.start(cpu()).unwrap() else {
        panic!("expected read request")
    };
    assert_eq!(request.key.request_id(), 1);
    assert!(matches!(
        request.operation,
        TaskRequestV1::Read {
            offset: 0,
            length: 3,
            ..
        }
    ));
    request.key
}

#[test]
fn replay_yields_then_transfers_output_once_and_drops_its_owned_state() {
    let (mut task, drops, _) = replay(TaskLimits::qualification());
    assert_eq!(task.status(), TaskStatus::New);
    assert_eq!(
        task.progress(cpu()).unwrap_err(),
        TaskProtocolError::NotStarted
    );
    let key = start(&mut task);
    assert_eq!(
        task.start(cpu()).unwrap_err(),
        TaskProtocolError::AlreadyStarted
    );
    assert_eq!(
        task.progress(cpu()).unwrap_err(),
        TaskProtocolError::PendingRequest
    );
    assert!(matches!(
        task.resume(key, bytes(0), cpu()).unwrap(),
        TaskStep::Yield
    ));
    assert!(matches!(task.progress(cpu()).unwrap(), TaskStep::Yield));
    assert!(matches!(
        task.progress(cpu()).unwrap(),
        TaskStep::Complete(6)
    ));
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    assert_eq!(task.status(), TaskStatus::Complete);
    assert_eq!(
        task.resume(key, bytes(0), cpu()).unwrap_err(),
        TaskProtocolError::Terminal
    );
    assert_eq!(
        task.progress(cpu()).unwrap_err(),
        TaskProtocolError::Terminal
    );
    assert_eq!(
        task.cancel(CancelReason::Caller),
        CancelDisposition::AlreadyTerminal
    );
    assert_eq!(
        task.accounting().usage(Resource::TaskStateBytes).live(),
        std::mem::size_of_val(&task)
    );
    assert_eq!(task.accounting().usage(Resource::InputBytes).consumed(), 3);
}

#[test]
fn wrong_kind_stale_and_cross_task_keys_preserve_the_pending_request() {
    let (mut first, _, _) = replay(TaskLimits::qualification());
    let (mut second, _, _) = replay(TaskLimits::qualification());
    let key = start(&mut first);
    let other = start(&mut second);
    let wrong = RequestKey::new(key.task_id(), key.request_id() + 1).unwrap();
    for wrong_key in [other, wrong] {
        assert_eq!(
            first.resume(wrong_key, bytes(0), cpu()).unwrap_err(),
            TaskProtocolError::WrongKey
        );
        assert_eq!(first.pending_key(), Some(key));
    }
    assert_eq!(
        first
            .resume(key, Ok(TaskResponseV1::Head { size: 3 }), cpu())
            .unwrap_err(),
        TaskProtocolError::WrongKind
    );
    assert_eq!(first.pending_key(), Some(key));
    assert!(matches!(
        first.resume(key, bytes(0), cpu()).unwrap(),
        TaskStep::Yield
    ));
    assert_eq!(
        first.resume(key, bytes(0), cpu()).unwrap_err(),
        TaskProtocolError::WrongKey
    );
    assert_eq!(first.pending_key(), None);
}

#[test]
fn malformed_matching_response_is_terminal_and_releases_state() {
    let (mut task, drops, _) = replay(TaskLimits::qualification());
    let key = start(&mut task);
    let TaskStep::Failed(error) = task.resume(key, bytes(1), cpu()).unwrap() else {
        panic!("expected matching malformed failure")
    };
    assert_eq!(error.kind(), FailureKind::MalformedResponse);
    assert_eq!(
        task.status(),
        TaskStatus::Failed(FailureKind::MalformedResponse)
    );
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    assert_eq!(task.pending_key(), None);
    assert_eq!(
        task.cancel(CancelReason::Caller),
        CancelDisposition::AlreadyTerminal
    );
}

#[test]
fn cancellation_clears_state_before_dispatch_at_effect_completion_and_after_yield() {
    for handoff in 0..4 {
        let (mut task, drops, _) = replay(TaskLimits::qualification());
        let key = (handoff > 0).then(|| start(&mut task));
        let ready = (handoff == 2).then(|| bytes(0));
        if handoff == 3 {
            assert!(matches!(
                task.resume(key.unwrap(), bytes(0), cpu()).unwrap(),
                TaskStep::Yield
            ));
        }
        let outstanding = if handoff == 3 { None } else { key };
        assert_eq!(
            task.cancel(CancelReason::Caller),
            CancelDisposition::Cancelled(outstanding)
        );
        drop(ready); // The driver releases any completed response for the returned key.
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        assert_eq!(task.pending_key(), None);
        assert_eq!(
            task.cancel(CancelReason::Caller),
            CancelDisposition::AlreadyCancelled
        );
        assert!(matches!(task.start(cpu()).unwrap(), TaskStep::Cancelled));
        assert_eq!(
            task.accounting().usage(Resource::TaskStateBytes).live(),
            std::mem::size_of_val(&task)
        );
        if let Some(key) = key {
            assert_eq!(
                task.resume(key, bytes(0), cpu()).unwrap_err(),
                TaskProtocolError::Terminal
            );
        }
    }
}

#[derive(Debug)]
struct OriginalError(Arc<AtomicUsize>);

impl std::fmt::Display for OriginalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("original operational error")
    }
}
impl std::error::Error for OriginalError {}
impl Drop for OriginalError {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn matching_operational_error_transfers_original_source_once() {
    let (mut task, state_drops, _) = replay(TaskLimits::qualification());
    let drops = Arc::new(AtomicUsize::new(0));
    let key = start(&mut task);
    let original = Error::generic_err(OriginalError(drops.clone()));
    let response = Err(OperationFailure::new(FailureKind::Engine, original));
    let TaskStep::Failed(error) = task.resume(key, response, cpu()).unwrap() else {
        panic!("original error must transfer")
    };
    assert_eq!(state_drops.load(Ordering::Relaxed), 1);
    assert_eq!(drops.load(Ordering::Relaxed), 0);
    let error = error.into_error();
    assert!(matches!(&error, Error::GenericError { source } if source.is::<OriginalError>()));
    drop(error);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    assert_eq!(task.status(), TaskStatus::Failed(FailureKind::Engine));
    assert_eq!(
        task.progress(cpu()).unwrap_err(),
        TaskProtocolError::Terminal
    );
}

#[test]
fn request_admission_precedes_request_materialization_and_exhaustion_drops_state() {
    let limits = TaskLimits::qualification().with_limit(Resource::Requests, 0);
    let (mut task, drops, requests) = replay(limits);
    let TaskStep::Failed(error) = task.start(cpu()).unwrap() else {
        panic!("zero request budget must fail")
    };
    assert!(
        matches!(error.kind(), FailureKind::ResourceExhausted(e) if e.resource == Resource::Requests)
    );
    assert_eq!(requests.load(Ordering::Relaxed), 0);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    assert_eq!(task.pending_key(), None);
}

#[test]
fn configured_cpu_turn_limits_fail_before_semantic_work_or_request_materialization() {
    for resource in [
        Resource::TurnRecords,
        Resource::TurnInputBytes,
        Resource::TurnPlanNodes,
        Resource::CpuTurns,
    ] {
        let (mut task, drops, requests) =
            replay(TaskLimits::qualification().with_limit(resource, 0));
        let TaskStep::Failed(error) = task.start(cpu()).unwrap() else {
            panic!("configured turn budget must prevent semantic work")
        };
        assert!(
            matches!(error.kind(), FailureKind::ResourceExhausted(e) if e.resource == resource)
        );
        assert_eq!(requests.load(Ordering::Relaxed), 0);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn yield_exhaustion_is_terminal_and_cannot_return_partial_output() {
    let (mut task, drops, _) = replay(TaskLimits::qualification().with_limit(Resource::Yields, 0));
    let key = start(&mut task);
    let TaskStep::Failed(error) = task.resume(key, bytes(0), cpu()).unwrap() else {
        panic!("zero yield allowance must terminate")
    };
    assert!(
        matches!(error.kind(), FailureKind::ResourceExhausted(e) if e.resource == Resource::Yields)
    );
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    assert_eq!(
        task.progress(cpu()).unwrap_err(),
        TaskProtocolError::Terminal
    );
}

#[test]
fn every_accounting_domain_checks_boundary_overflow_and_keeps_live_separate() {
    for resource in Resource::ALL {
        let limits = TaskLimits::qualification().with_limit(*resource, 2);
        let accounting = TaskAccounting::new(limits);
        accounting.check(*resource, 2).unwrap();
        assert!(accounting.check(*resource, 3).is_err());
        accounting.charge(*resource, 2).unwrap();
        assert!(accounting.charge(*resource, 1).is_err());
        accounting.set_live(*resource, 2).unwrap();
        accounting.set_live(*resource, 0).unwrap();
        let usage = accounting.usage(*resource);
        assert_eq!(
            (usage.consumed(), usage.live(), usage.peak_live()),
            (2, 0, 2)
        );
        let accounting =
            TaskAccounting::new(TaskLimits::qualification().with_limit(*resource, usize::MAX));
        accounting.charge(*resource, usize::MAX).unwrap();
        let error = accounting.charge(*resource, 1).unwrap_err();
        assert_eq!(error.observed, usize::MAX);
        assert_eq!(accounting.usage(*resource).consumed(), usize::MAX);
    }
}

struct FinishAtOverflow(bool);

impl TaskState for FinishAtOverflow {
    type Output = ();
    fn retained_bytes(&self) -> Result<usize, ResourceExhausted> {
        Ok(if self.0 { usize::MAX } else { 0 })
    }
    fn advance(
        &mut self,
        _: CpuSlice,
        _: &TaskAccounting,
    ) -> Result<TaskAction<()>, OperationFailure> {
        self.0 = true;
        Ok(TaskAction::Complete(()))
    }
    fn take_request(&mut self, _: &TaskAccounting) -> Result<TaskRequestV1, OperationFailure> {
        panic!("no I/O")
    }
    fn resume(
        &mut self,
        _: TaskResponseV1,
        _: CpuSlice,
        _: &TaskAccounting,
    ) -> Result<TaskAction<()>, OperationFailure> {
        panic!("no I/O")
    }
}

#[test]
fn final_retained_state_overflow_cannot_become_successful_completion() {
    let mut task = TaskMachine::new(
        TaskId::allocate().unwrap(),
        FinishAtOverflow(false),
        TaskLimits::qualification().with_limit(Resource::TaskStateBytes, usize::MAX),
    )
    .unwrap();
    let TaskStep::Failed(error) = task.start(cpu()).unwrap() else {
        panic!("overflow cannot emit successful output")
    };
    assert!(
        matches!(error.kind(), FailureKind::ResourceExhausted(e) if e.resource == Resource::TaskStateBytes && e.observed == usize::MAX)
    );
    assert_eq!(
        task.accounting().usage(Resource::TaskStateBytes).live(),
        std::mem::size_of_val(&task)
    );
}

struct TestBatch {
    rows: usize,
    drops: Arc<AtomicUsize>,
}

impl Drop for TestBatch {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

impl EngineData for TestBatch {
    fn len(&self) -> usize {
        self.rows
    }
    fn visit_rows(&self, _: &[ColumnName], visitor: &mut dyn RowVisitor) -> DeltaResult<()> {
        visitor.visit(0, &[])
    }
    fn append_columns(&self, _: SchemaRef, _: Vec<ArrayData>) -> DeltaResult<Box<dyn EngineData>> {
        Err(Error::generic("empty fixture does not append columns"))
    }
    fn apply_selection_vector(self: Box<Self>, _: Vec<bool>) -> DeltaResult<Box<dyn EngineData>> {
        Ok(self)
    }
    fn has_field(&self, _: &ColumnName) -> bool {
        false
    }
}

impl AccountedEngineData for TestBatch {
    fn accounted_bytes(&self) -> Result<usize, ResourceExhausted> {
        Ok(64)
    }
}

struct EvaluationStartState {
    evaluation: EvaluationKey,
    plan: Option<AdmittedPlan>,
    rows: usize,
}

impl TaskState for EvaluationStartState {
    type Output = usize;

    fn retained_bytes(&self) -> Result<usize, ResourceExhausted> {
        Ok(self.plan.as_ref().map_or(0, AdmittedPlan::retained_bytes))
    }

    fn advance(
        &mut self,
        _: CpuSlice,
        _: &TaskAccounting,
    ) -> Result<TaskAction<Self::Output>, OperationFailure> {
        Ok(TaskAction::Request)
    }

    fn take_request(&mut self, _: &TaskAccounting) -> Result<TaskRequestV1, OperationFailure> {
        Ok(match self.plan.take() {
            Some(plan) => TaskRequestV1::EvaluationStart {
                evaluation: self.evaluation,
                plan,
                limits: evaluation_start_limits(),
            },
            None => TaskRequestV1::Evaluation {
                evaluation: self.evaluation,
                limits: evaluation_start_page_limits(),
            },
        })
    }

    fn resume(
        &mut self,
        response: TaskResponseV1,
        _: CpuSlice,
        _: &TaskAccounting,
    ) -> Result<TaskAction<Self::Output>, OperationFailure> {
        let TaskResponseV1::Evaluation { page, .. } = response else {
            panic!("machine must validate response kind before calling state")
        };
        let Some(page) = page else {
            return Ok(TaskAction::Complete(self.rows));
        };
        self.rows += page.num_rows();
        Ok(TaskAction::Request)
    }
}

#[test]
fn admitted_plan_transfers_once_then_evaluation_requests_use_only_driver_identity() {
    let limits = TaskLimits::qualification();
    let id = TaskId::allocate().unwrap();
    let evaluation = EvaluationKey::allocate(id.get()).unwrap();
    let plan = AdmittedPlan::try_i64_values("value", &[], &[1, 2, 3], &limits).unwrap();
    let plan_bytes = plan.retained_bytes();
    let mut task = TaskMachine::new(
        id,
        EvaluationStartState {
            evaluation,
            plan: Some(plan),
            rows: 0,
        },
        limits,
    )
    .unwrap();
    let fixed_task_bytes = std::mem::size_of_val(&task);
    assert_eq!(
        task.accounting().usage(Resource::TaskStateBytes).live(),
        fixed_task_bytes + plan_bytes
    );

    let TaskStep::Execute(start) = task.start(cpu()).unwrap() else {
        panic!("expected evaluation start")
    };
    let task_state = task.accounting().usage(Resource::TaskStateBytes);
    assert_eq!(task_state.live(), fixed_task_bytes);
    assert_eq!(
        task_state.peak_live(),
        fixed_task_bytes + std::mem::size_of_val(&start) + plan_bytes
    );
    let start_key = start.key;
    let TaskRequestV1::EvaluationStart {
        evaluation: actual,
        plan,
        limits: actual_limits,
    } = start.operation
    else {
        panic!("expected one-shot admitted plan transfer")
    };
    assert_eq!(actual, evaluation);
    assert_eq!(actual_limits, evaluation_start_limits());
    assert_eq!(plan.into_plan().nodes.len(), 1);

    assert_eq!(
        task.resume(start_key, Ok(TaskResponseV1::Head { size: 3 }), cpu())
            .unwrap_err(),
        TaskProtocolError::WrongKind
    );
    assert_eq!(task.pending_key(), Some(start_key));

    let drops = Arc::new(AtomicUsize::new(0));
    let page = EvaluationPage::try_new(
        vec![Box::new(TestBatch {
            rows: 3,
            drops: drops.clone(),
        })],
        evaluation_start_page_limits(),
    )
    .unwrap();
    let TaskStep::Execute(next) = task
        .resume(
            start_key,
            Ok(TaskResponseV1::Evaluation {
                evaluation,
                page: Some(page),
            }),
            cpu(),
        )
        .unwrap()
    else {
        panic!("expected identity-only evaluation request")
    };
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    assert!(matches!(
        next.operation,
        TaskRequestV1::Evaluation {
            evaluation: actual,
            ..
        } if actual == evaluation
    ));
    assert!(matches!(
        task.resume(
            next.key,
            Ok(TaskResponseV1::Evaluation {
                evaluation,
                page: None,
            }),
            cpu(),
        )
        .unwrap(),
        TaskStep::Complete(3)
    ));
}

fn evaluation_start_page_limits() -> EvaluationPageLimits {
    EvaluationPageLimits::new(1, 3, 128).unwrap()
}

fn evaluation_start_limits() -> EvaluationLimits {
    EvaluationLimits::new(evaluation_start_page_limits(), 4, 4, 12, 512)
}

struct FixedSource {
    batch: Option<Box<dyn AccountedEngineData>>,
    drops: Arc<AtomicUsize>,
}

impl Iterator for FixedSource {
    type Item = Result<Box<dyn AccountedEngineData>, OperationFailure>;
    fn next(&mut self) -> Option<Self::Item> {
        self.batch.take().map(Ok)
    }
}

impl Drop for FixedSource {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

struct PageState {
    key: EvaluationKey,
    page: Option<EvaluationPage>,
    resumes: Arc<AtomicUsize>,
}

impl TaskState for PageState {
    type Output = ();
    fn retained_bytes(&self) -> Result<usize, ResourceExhausted> {
        Ok(self
            .page
            .as_ref()
            .map_or(0, EvaluationPage::accounted_bytes))
    }
    fn advance(
        &mut self,
        _: CpuSlice,
        _: &TaskAccounting,
    ) -> Result<TaskAction<()>, OperationFailure> {
        Ok(TaskAction::Request)
    }
    fn take_request(&mut self, _: &TaskAccounting) -> Result<TaskRequestV1, OperationFailure> {
        Ok(TaskRequestV1::Evaluation {
            evaluation: self.key,
            limits: page_limits(),
        })
    }
    fn resume(
        &mut self,
        response: TaskResponseV1,
        _: CpuSlice,
        _: &TaskAccounting,
    ) -> Result<TaskAction<()>, OperationFailure> {
        self.resumes.fetch_add(1, Ordering::Relaxed);
        let TaskResponseV1::Evaluation { page, .. } = response else {
            panic!("expected evaluation page")
        };
        self.page = page;
        Ok(TaskAction::Yield)
    }
}

fn page_limits() -> EvaluationPageLimits {
    EvaluationPageLimits::new(1, 1, 128).unwrap()
}

#[test]
fn cancellation_drops_task_page_before_driver_releases_its_evaluation_source() {
    let id = TaskId::allocate().unwrap();
    let evaluation = EvaluationKey::allocate(id.get()).unwrap();
    let page_drops = Arc::new(AtomicUsize::new(0));
    let source_drops = Arc::new(AtomicUsize::new(0));
    let source = FixedSource {
        batch: Some(Box::new(TestBatch {
            rows: 0,
            drops: page_drops.clone(),
        })),
        drops: source_drops.clone(),
    };
    let mut driver = EvaluationReader::new(
        Box::new(source),
        EvaluationLimits::new(page_limits(), 4, 4, 4, 512),
    );
    let state = PageState {
        key: evaluation,
        page: None,
        resumes: Arc::new(AtomicUsize::new(0)),
    };
    let mut task = TaskMachine::new(id, state, TaskLimits::qualification()).unwrap();
    let TaskStep::Execute(first) = task.start(cpu()).unwrap() else {
        panic!("expected evaluation request")
    };
    let page = driver.next_page().unwrap();
    assert!(page.is_some());
    assert!(matches!(
        task.resume(
            first.key,
            Ok(TaskResponseV1::Evaluation { evaluation, page }),
            cpu()
        )
        .unwrap(),
        TaskStep::Yield
    ));
    let TaskStep::Execute(second) = task.progress(cpu()).unwrap() else {
        panic!("expected next page request")
    };
    assert_eq!(second.key.request_id(), first.key.request_id() + 1);
    assert_eq!(
        task.resume(
            first.key,
            Ok(TaskResponseV1::Evaluation {
                evaluation,
                page: None
            }),
            cpu()
        )
        .unwrap_err(),
        TaskProtocolError::WrongKey
    );
    assert_eq!(task.pending_key(), Some(second.key));
    assert_eq!(
        task.cancel(CancelReason::Caller),
        CancelDisposition::Cancelled(Some(second.key))
    );
    assert_eq!(page_drops.load(Ordering::Relaxed), 1);
    assert_eq!(source_drops.load(Ordering::Relaxed), 0);
    driver.cancel();
    assert_eq!(source_drops.load(Ordering::Relaxed), 1);
    assert_eq!(
        task.accounting().usage(Resource::TaskStateBytes).live(),
        std::mem::size_of_val(&task)
    );
}

#[test]
fn evaluation_identity_mismatch_is_terminal_before_semantic_resumption() {
    let id = TaskId::allocate().unwrap();
    let evaluation = EvaluationKey::allocate(id.get()).unwrap();
    let other = EvaluationKey::allocate(id.get()).unwrap();
    let resumes = Arc::new(AtomicUsize::new(0));
    let state = PageState {
        key: evaluation,
        page: None,
        resumes: resumes.clone(),
    };
    let mut task = TaskMachine::new(id, state, TaskLimits::qualification()).unwrap();
    let TaskStep::Execute(request) = task.start(cpu()).unwrap() else {
        panic!("expected evaluation request")
    };
    let TaskStep::Failed(error) = task
        .resume(
            request.key,
            Ok(TaskResponseV1::Evaluation {
                evaluation: other,
                page: None,
            }),
            cpu(),
        )
        .unwrap()
    else {
        panic!("expected identity failure")
    };
    assert_eq!(error.kind(), FailureKind::MalformedResponse);
    assert_eq!(resumes.load(Ordering::Relaxed), 0);
    assert_eq!(task.pending_key(), None);
}

struct InlineState {
    inline: [u8; 32768],
    heap: Vec<u8>,
    drops: Arc<AtomicUsize>,
}

impl Drop for InlineState {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

impl TaskState for InlineState {
    type Output = u8;
    fn retained_bytes(&self) -> Result<usize, ResourceExhausted> {
        Ok(self.heap.capacity())
    }
    fn advance(
        &mut self,
        _: CpuSlice,
        accounting: &TaskAccounting,
    ) -> Result<TaskAction<u8>, OperationFailure> {
        accounting.charge(Resource::Records, 1)?;
        Ok(TaskAction::Complete(self.inline[0]))
    }
    fn take_request(&mut self, _: &TaskAccounting) -> Result<TaskRequestV1, OperationFailure> {
        panic!("no I/O")
    }
    fn resume(
        &mut self,
        _: TaskResponseV1,
        _: CpuSlice,
        _: &TaskAccounting,
    ) -> Result<TaskAction<u8>, OperationFailure> {
        panic!("no I/O")
    }
}

#[test]
fn terminal_cleanup_releases_heap_but_keeps_inline_storage_charged() {
    for terminal in 0..3 {
        let drops = Arc::new(AtomicUsize::new(0));
        let state = InlineState {
            inline: [7; 32768],
            heap: vec![0; 4096],
            drops: drops.clone(),
        };
        let heap_bytes = state.heap.capacity();
        let limits = TaskLimits::qualification()
            .with_limit(Resource::Records, if terminal == 2 { 0 } else { 1 });
        let mut task = TaskMachine::new(TaskId::allocate().unwrap(), state, limits).unwrap();
        let fixed = std::mem::size_of_val(&task);
        assert!(fixed >= 32768);
        assert_eq!(
            task.accounting().usage(Resource::TaskStateBytes).live(),
            fixed + heap_bytes
        );
        match terminal {
            0 => assert_eq!(
                task.cancel(CancelReason::Caller),
                CancelDisposition::Cancelled(None)
            ),
            1 => assert!(matches!(task.start(cpu()).unwrap(), TaskStep::Complete(7))),
            _ => assert!(matches!(task.start(cpu()).unwrap(), TaskStep::Failed(_))),
        }
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        let usage = task.accounting().usage(Resource::TaskStateBytes);
        assert_eq!(usage.live(), fixed);
        assert_eq!(usage.peak_live(), fixed + heap_bytes);
        assert_eq!(
            task.accounting().usage(Resource::Records).consumed(),
            usize::from(terminal == 1)
        );
        assert_eq!(
            task.accounting().usage(Resource::CpuTurns).consumed(),
            usize::from(terminal != 0)
        );
        let counters: Vec<_> = Resource::ALL
            .iter()
            .map(|r| task.accounting().usage(*r))
            .collect();
        assert!(task.progress(cpu()).is_err());
        task.cancel(CancelReason::Caller);
        assert_eq!(
            counters,
            Resource::ALL
                .iter()
                .map(|r| task.accounting().usage(*r))
                .collect::<Vec<_>>()
        );
        drop(task);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }
}
