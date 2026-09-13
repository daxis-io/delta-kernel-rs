#![cfg(feature = "operation-tasks")]

use delta_kernel::tasks::*;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::task::Poll;

struct Host {
    responses: VecDeque<AdmittedAsyncEffect>,
    calls: Arc<AtomicUsize>,
    cancels: Arc<AtomicUsize>,
    expected: Arc<Mutex<Vec<Option<ObjectIdentity>>>>,
    cancelled: bool,
    work: usize,
}
impl Host {
    fn new(responses: impl IntoIterator<Item = AdmittedAsyncEffect>) -> Self {
        Self {
            responses: responses.into_iter().collect(),
            calls: Arc::default(),
            cancels: Arc::default(),
            expected: Arc::default(),
            cancelled: false,
            work: 1,
        }
    }
}
impl AdmittedAsyncHost for Host {
    fn complete<'a>(
        &'a mut self,
        _: &'a TaskRequestV1,
        identity: Option<ObjectIdentity>,
        _: EvaluationUsage,
        work: PendingWork<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<AdmittedAsyncEffect, OperationFailure>> + 'a>> {
        Box::pin(async move {
            work.charge(self.work)?;
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.expected.lock().unwrap().push(identity);
            let mut yielded = false;
            futures::future::poll_fn(|cx| {
                if yielded {
                    Poll::Ready(())
                } else {
                    yielded = true;
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            })
            .await;
            self.responses
                .pop_front()
                .ok_or_else(OperationFailure::malformed_response)
        })
    }
    fn cancel(&mut self) {
        if !self.cancelled {
            self.cancelled = true;
            self.cancels.fetch_add(1, Ordering::SeqCst);
            self.responses.clear();
        }
    }
}
struct Effects {
    requests: VecDeque<TaskRequestV1>,
    kernel_work: usize,
}
impl TaskState for Effects {
    type Output = ();
    fn retained_bytes(&self) -> Result<usize, ResourceExhausted> {
        let mut bytes = self.requests.capacity() * std::mem::size_of::<TaskRequestV1>();
        for request in &self.requests {
            bytes += match request {
                TaskRequestV1::Head { path } | TaskRequestV1::Read { path, .. } => path.capacity(),
                TaskRequestV1::EvaluationStart { plan, .. } => plan.retained_bytes(),
                _ => 0,
            };
        }
        Ok(bytes)
    }
    fn advance(
        &mut self,
        _: CpuSlice,
        accounting: &TaskAccounting,
    ) -> Result<TaskAction<()>, OperationFailure> {
        accounting.charge(Resource::WorkUnits, self.kernel_work)?;
        Ok(TaskAction::Request)
    }
    fn take_request(&mut self, _: &TaskAccounting) -> Result<TaskRequestV1, OperationFailure> {
        self.requests
            .pop_front()
            .ok_or_else(OperationFailure::malformed_response)
    }
    fn resume(
        &mut self,
        _: TaskResponseV1,
        _: CpuSlice,
        _: &TaskAccounting,
    ) -> Result<TaskAction<()>, OperationFailure> {
        Ok(if self.requests.is_empty() {
            TaskAction::Complete(())
        } else {
            TaskAction::Request
        })
    }
}
fn cpu() -> CpuSlice {
    CpuSlice::new(1024, 1 << 20, 256).unwrap()
}
fn machine(
    id: TaskId,
    requests: impl IntoIterator<Item = TaskRequestV1>,
    limits: TaskLimits,
    kernel_work: usize,
) -> TaskMachine<Effects> {
    TaskMachine::new(
        id,
        Effects {
            requests: requests.into_iter().collect(),
            kernel_work,
        },
        limits,
    )
    .unwrap()
}
fn effect(step: TaskStep<()>) -> TaskRequest {
    let TaskStep::Execute(request) = step else {
        panic!("request expected")
    };
    request
}
fn limits() -> EvaluationLimits {
    EvaluationLimits::new(EvaluationPageLimits::new(1, 1, 256).unwrap(), 4, 4, 4, 1024)
}
fn plan() -> AdmittedPlan {
    AdmittedPlan::try_i64_values("id", &[], &[1], &TaskLimits::qualification()).unwrap()
}

#[tokio::test]
async fn async_head_read_uses_shared_identity_and_shape_validation() {
    let original = ObjectIdentity::new([1; 32]);
    let replacement = ObjectIdentity::new([2; 32]);
    let host = Host::new([
        AdmittedAsyncEffect::Io(AdmittedIoEffect::Head(AdmittedHead {
            identity: original,
            size: 1,
        })),
        AdmittedAsyncEffect::Io(AdmittedIoEffect::Read(AdmittedRead {
            identity: replacement,
            offset: 0,
            bytes: vec![7],
            eof: true,
        })),
    ]);
    let observed = host.expected.clone();
    let cancels = host.cancels.clone();
    let (id, mut driver) =
        AsyncOperationDriver::allocate(host, TaskLimits::qualification()).unwrap();
    let mut task = machine(
        id,
        [
            TaskRequestV1::Head {
                path: "memory:///a".into(),
            },
            TaskRequestV1::Read {
                path: "memory:///a".into(),
                offset: 0,
                length: 1,
            },
        ],
        TaskLimits::qualification(),
        1,
    );
    let key = driver.dispatch(effect(task.start(cpu()).unwrap())).unwrap();
    driver
        .complete_effect(task.pending_work().unwrap())
        .await
        .unwrap();
    let response = driver.take_response(key).unwrap().unwrap();
    assert!(matches!(response, TaskResponseV1::Head { size: 1 }));
    let key = driver
        .dispatch(effect(task.resume(key, Ok(response), cpu()).unwrap()))
        .unwrap();
    driver
        .complete_effect(task.pending_work().unwrap())
        .await
        .unwrap();
    assert!(driver.take_response(key).unwrap().is_err());
    assert_eq!(*observed.lock().unwrap(), vec![None, Some(original)]);
    assert_eq!(cancels.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn dropped_completion_requires_cancellation_without_retrying_effect() {
    let host = Host::new([]);
    let calls = host.calls.clone();
    let cancels = host.cancels.clone();
    let (id, mut driver) =
        AsyncOperationDriver::allocate(host, TaskLimits::qualification()).unwrap();
    let mut task = machine(
        id,
        [TaskRequestV1::Head {
            path: "memory:///a".into(),
        }],
        TaskLimits::qualification(),
        1,
    );
    let key = driver.dispatch(effect(task.start(cpu()).unwrap())).unwrap();
    {
        let mut future = Box::pin(driver.complete_effect(task.pending_work().unwrap()));
        assert!(futures::poll!(future.as_mut()).is_pending());
    }
    assert_eq!(
        driver.complete_effect(task.pending_work().unwrap()).await,
        Err(TaskProtocolError::PendingRequest)
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(task.accounting().usage(Resource::WorkUnits).consumed(), 2);
    driver.cancel(Some(key)).unwrap();
    task.cancel(CancelReason::Caller);
    assert_eq!(task.accounting().usage(Resource::WorkUnits).consumed(), 2);
    assert_eq!(cancels.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn wrong_evaluation_identity_releases_host_and_preserves_response_key() {
    let (_, other) =
        AsyncOperationDriver::allocate(Host::new([]), TaskLimits::qualification()).unwrap();
    let foreign = other.allocate_evaluation().unwrap();
    let host = Host::new([AdmittedAsyncEffect::Evaluation {
        evaluation: foreign,
        page: None,
    }]);
    let cancels = host.cancels.clone();
    let (id, mut driver) =
        AsyncOperationDriver::allocate(host, TaskLimits::qualification()).unwrap();
    let evaluation = driver.allocate_evaluation().unwrap();
    let numeric_id = id.get();
    let mut task = machine(
        id,
        [TaskRequestV1::EvaluationStart {
            evaluation,
            plan: plan(),
            limits: limits(),
        }],
        TaskLimits::qualification(),
        1,
    );
    let key = driver.dispatch(effect(task.start(cpu()).unwrap())).unwrap();
    driver
        .complete_effect(task.pending_work().unwrap())
        .await
        .unwrap();
    assert!(matches!(
        driver.take_response(RequestKey::new(numeric_id, 2).unwrap()),
        Err(TaskProtocolError::WrongKey)
    ));
    assert!(driver.take_response(key).unwrap().is_err());
    assert_eq!(cancels.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn kernel_and_host_share_exact_work_limit_before_host_side_effects() {
    for (allowance, pass) in [(12, true), (11, false)] {
        let limits = TaskLimits::qualification().with_limit(Resource::WorkUnits, allowance);
        let mut host = Host::new([AdmittedAsyncEffect::Io(AdmittedIoEffect::Head(
            AdmittedHead {
                identity: ObjectIdentity::new([1; 32]),
                size: 1,
            },
        ))]);
        host.work = 7;
        let calls = host.calls.clone();
        let (id, mut driver) = AsyncOperationDriver::allocate(host, limits).unwrap();
        let mut task = machine(
            id,
            [TaskRequestV1::Head {
                path: "memory:///a".into(),
            }],
            limits,
            5,
        );
        let key = driver.dispatch(effect(task.start(cpu()).unwrap())).unwrap();
        driver
            .complete_effect(task.pending_work().unwrap())
            .await
            .unwrap();
        let response = driver.take_response(key).unwrap();
        assert_eq!(response.is_ok(), pass);
        assert_eq!(calls.load(Ordering::SeqCst), usize::from(pass));
        assert_eq!(
            task.accounting().usage(Resource::WorkUnits).consumed(),
            if pass { 12 } else { 5 }
        );
        if !pass {
            assert!(
                matches!(response.unwrap_err().kind(), FailureKind::ResourceExhausted(e)
            if e.resource == Resource::WorkUnits && e.observed == 12)
            );
        }
    }
}

#[tokio::test]
async fn wrong_and_stale_work_loans_do_not_invoke_host() {
    let host = Host::new([AdmittedAsyncEffect::Io(AdmittedIoEffect::Head(
        AdmittedHead {
            identity: ObjectIdentity::new([1; 32]),
            size: 1,
        },
    ))]);
    let calls = host.calls.clone();
    let limits = TaskLimits::qualification();
    let (id, mut driver) = AsyncOperationDriver::allocate(host, limits).unwrap();
    let numeric_id = id.get();
    let mut task = machine(
        id,
        [TaskRequestV1::Head {
            path: "memory:///a".into(),
        }],
        limits,
        1,
    );
    let key = driver.dispatch(effect(task.start(cpu()).unwrap())).unwrap();
    let mut other = machine(
        TaskId::allocate().unwrap(),
        [TaskRequestV1::Head {
            path: "memory:///b".into(),
        }],
        limits,
        1,
    );
    let _other_request = effect(other.start(cpu()).unwrap());
    assert_eq!(
        driver.complete_effect(other.pending_work().unwrap()).await,
        Err(TaskProtocolError::WrongKey)
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    driver
        .complete_effect(task.pending_work().unwrap())
        .await
        .unwrap();
    let _response = driver.take_response(key).unwrap();
    // Forge a second driver request without resuming the task's first one.
    // Its pending work loan remains tied to the first key and cannot authorize it.
    driver
        .dispatch(TaskRequest {
            key: RequestKey::new(numeric_id, 2).unwrap(),
            operation: TaskRequestV1::Head {
                path: "memory:///a".into(),
            },
        })
        .unwrap();
    assert_eq!(
        driver.complete_effect(task.pending_work().unwrap()).await,
        Err(TaskProtocolError::WrongKey)
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn pending_work_overflow_preserves_the_successful_charge() {
    let limits = TaskLimits::qualification().with_limit(Resource::WorkUnits, usize::MAX);
    let mut task = machine(
        TaskId::allocate().unwrap(),
        [TaskRequestV1::Head {
            path: "memory:///a".into(),
        }],
        limits,
        1,
    );
    assert!(task.pending_work().is_err());
    let _request = effect(task.start(cpu()).unwrap());
    let work = task.pending_work().unwrap();
    work.charge(usize::MAX - 1).unwrap();
    assert_eq!(work.remaining(), 0);
    assert_eq!(work.charge(1).unwrap_err().observed, usize::MAX);
    assert_eq!(
        task.accounting().usage(Resource::WorkUnits).consumed(),
        usize::MAX
    );
    task.cancel(CancelReason::Caller);
    assert!(task.pending_work().is_err());
    assert_eq!(
        task.accounting().usage(Resource::WorkUnits).consumed(),
        usize::MAX
    );
}

#[test]
fn synchronous_operation_task_implementation_needs_no_host_work_method() {
    struct ExistingTask;
    impl OperationTask for ExistingTask {
        type Output = ();
        fn start(&mut self, _: CpuSlice) -> Result<TaskStep<()>, TaskProtocolError> {
            Ok(TaskStep::Complete(()))
        }
        fn progress(&mut self, _: CpuSlice) -> Result<TaskStep<()>, TaskProtocolError> {
            Err(TaskProtocolError::Terminal)
        }
        fn resume(
            &mut self,
            _: RequestKey,
            _: Result<TaskResponseV1, OperationFailure>,
            _: CpuSlice,
        ) -> Result<TaskStep<()>, TaskProtocolError> {
            Err(TaskProtocolError::Terminal)
        }
        fn cancel(&mut self, _: CancelReason) -> CancelDisposition {
            CancelDisposition::AlreadyTerminal
        }
    }
    let mut task = ExistingTask;
    assert!(matches!(
        task.pending_work(),
        Err(TaskProtocolError::HostWorkUnavailable)
    ));
    assert!(matches!(task.start(cpu()).unwrap(), TaskStep::Complete(())));
}
