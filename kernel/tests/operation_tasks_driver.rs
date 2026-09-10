#![cfg(feature = "operation-tasks")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use delta_kernel::engine_data::RowVisitor;
use delta_kernel::expressions::{ArrayData, ColumnName};
use delta_kernel::schema::SchemaRef;
use delta_kernel::tasks::{
    AccountedEngineData, AdmittedEvaluationSource, AdmittedPlan, CancelDisposition, CancelReason,
    CpuSlice, EvaluationDriver, EvaluationKey, EvaluationLimits, EvaluationPageLimits, FailureKind,
    OperationFailure, OperationTask, RequestKey, ResourceExhausted, TaskAccounting, TaskAction,
    TaskLimits, TaskMachine, TaskProtocolError, TaskRequest, TaskRequestV1, TaskResponseV1,
    TaskState, TaskStep,
};
use delta_kernel::{DeltaResult, EngineData, Error};

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
        Err(Error::generic("driver fixture does not append columns"))
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

struct FixedSource {
    batch: Option<Box<dyn AccountedEngineData>>,
    pulls: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
}

impl Iterator for FixedSource {
    type Item = Result<Box<dyn AccountedEngineData>, OperationFailure>;

    fn next(&mut self) -> Option<Self::Item> {
        self.pulls.fetch_add(1, Ordering::Relaxed);
        self.batch.take().map(Ok)
    }
}

impl Drop for FixedSource {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

struct EvaluationState {
    evaluation: EvaluationKey,
    plan: Option<AdmittedPlan>,
    rows: usize,
}

impl TaskState for EvaluationState {
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
                limits: evaluation_limits(),
            },
            None => TaskRequestV1::Evaluation {
                evaluation: self.evaluation,
                limits: page_limits(),
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
            panic!("driver returned the wrong response kind")
        };
        match page {
            Some(page) => {
                self.rows += page.num_rows();
                Ok(TaskAction::Request)
            }
            None => Ok(TaskAction::Complete(self.rows)),
        }
    }
}

#[test]
fn driver_dispatches_one_plan_then_pages_by_checked_identity() {
    let compiles = Arc::new(AtomicUsize::new(0));
    let pulls = Arc::new(AtomicUsize::new(0));
    let batch_drops = Arc::new(AtomicUsize::new(0));
    let source_drops = Arc::new(AtomicUsize::new(0));
    let (id, mut driver) = EvaluationDriver::allocate({
        let compiles = compiles.clone();
        let pulls = pulls.clone();
        let batch_drops = batch_drops.clone();
        let source_drops = source_drops.clone();
        move |plan: AdmittedPlan, actual_limits: EvaluationLimits| {
            compiles.fetch_add(1, Ordering::Relaxed);
            assert_eq!(actual_limits, evaluation_limits());
            assert_eq!(plan.into_plan().nodes.len(), 1);
            Ok(Box::new(FixedSource {
                batch: Some(Box::new(TestBatch {
                    rows: 1,
                    drops: batch_drops.clone(),
                })),
                pulls: pulls.clone(),
                drops: source_drops.clone(),
            }) as AdmittedEvaluationSource)
        }
    })
    .unwrap();
    let task_id = id.get();
    let evaluation = driver.allocate_evaluation().unwrap();
    let plan =
        AdmittedPlan::try_i64_values("value", &[], &[7], &TaskLimits::qualification()).unwrap();
    let mut task = TaskMachine::new(
        id,
        EvaluationState {
            evaluation,
            plan: Some(plan),
            rows: 0,
        },
        TaskLimits::qualification(),
    )
    .unwrap();

    let TaskStep::Execute(start) = task.start(cpu()).unwrap() else {
        panic!("expected evaluation start")
    };
    let start_key = start.key;
    assert_eq!(driver.dispatch(start).unwrap(), start_key);
    assert_eq!(driver.outstanding_key(), Some(start_key));
    assert_eq!(compiles.load(Ordering::Relaxed), 0);
    assert_eq!(driver.complete_effect().unwrap(), start_key);
    assert_eq!(compiles.load(Ordering::Relaxed), 1);
    assert_eq!(pulls.load(Ordering::Relaxed), 1);
    assert_eq!(driver.active_evaluation(), Some(evaluation));
    let wrong_response_key = RequestKey::new(task_id, start_key.request_id() + 1).unwrap();
    assert!(matches!(
        driver.take_response(wrong_response_key),
        Err(TaskProtocolError::WrongKey)
    ));
    assert_eq!(driver.outstanding_key(), Some(start_key));
    let response = driver.take_response(start_key).unwrap();
    let TaskStep::Execute(next) = task.resume(start_key, response, cpu()).unwrap() else {
        panic!("expected identity-only page request")
    };

    let stale = TaskRequest {
        key: start_key,
        operation: TaskRequestV1::Evaluation {
            evaluation,
            limits: page_limits(),
        },
    };
    assert_eq!(driver.dispatch(stale), Err(TaskProtocolError::WrongKey));
    let other_task_id = delta_kernel::tasks::TaskId::allocate().unwrap().get();
    let other_task = RequestKey::new(other_task_id, next.key.request_id()).unwrap();
    let cross_task = TaskRequest {
        key: other_task,
        operation: TaskRequestV1::Evaluation {
            evaluation,
            limits: page_limits(),
        },
    };
    assert_eq!(
        driver.dispatch(cross_task),
        Err(TaskProtocolError::WrongKey)
    );
    let changed_limits = TaskRequest {
        key: next.key,
        operation: TaskRequestV1::Evaluation {
            evaluation,
            limits: EvaluationPageLimits::new(1, 2, 128).unwrap(),
        },
    };
    assert_eq!(
        driver.dispatch(changed_limits),
        Err(TaskProtocolError::WrongKind)
    );
    assert_eq!(pulls.load(Ordering::Relaxed), 1);
    assert_eq!(driver.active_evaluation(), Some(evaluation));

    let next_key = next.key;
    assert_eq!(driver.dispatch(next).unwrap(), next_key);
    assert_eq!(driver.complete_effect().unwrap(), next_key);
    assert_eq!(pulls.load(Ordering::Relaxed), 2);
    assert_eq!(source_drops.load(Ordering::Relaxed), 1);
    assert_eq!(driver.active_evaluation(), None);
    let response = driver.take_response(next_key).unwrap();
    assert!(matches!(
        task.resume(next_key, response, cpu()).unwrap(),
        TaskStep::Complete(1)
    ));
    let after_eof = TaskRequest {
        key: RequestKey::new(task_id, next_key.request_id() + 1).unwrap(),
        operation: TaskRequestV1::Evaluation {
            evaluation,
            limits: page_limits(),
        },
    };
    assert_eq!(driver.dispatch(after_eof), Err(TaskProtocolError::Terminal));
    assert_eq!(driver.complete_effect(), Err(TaskProtocolError::Terminal));
    assert!(matches!(
        driver.take_response(next_key),
        Err(TaskProtocolError::Terminal)
    ));
    assert_eq!(batch_drops.load(Ordering::Relaxed), 1);
    assert_eq!(driver.evaluation_usage().pages(), 2);
    assert_eq!(driver.evaluation_usage().batches(), 1);
    assert_eq!(driver.evaluation_usage().rows(), 1);
}

#[test]
fn cancellation_releases_a_completed_page_and_active_source_without_resumption() {
    let compiles = Arc::new(AtomicUsize::new(0));
    let pulls = Arc::new(AtomicUsize::new(0));
    let batch_drops = Arc::new(AtomicUsize::new(0));
    let source_drops = Arc::new(AtomicUsize::new(0));
    let (id, mut driver) = EvaluationDriver::allocate({
        let compiles = compiles.clone();
        let pulls = pulls.clone();
        let batch_drops = batch_drops.clone();
        let source_drops = source_drops.clone();
        move |_: AdmittedPlan, _: EvaluationLimits| {
            compiles.fetch_add(1, Ordering::Relaxed);
            Ok(Box::new(FixedSource {
                batch: Some(Box::new(TestBatch {
                    rows: 1,
                    drops: batch_drops.clone(),
                })),
                pulls: pulls.clone(),
                drops: source_drops.clone(),
            }) as AdmittedEvaluationSource)
        }
    })
    .unwrap();
    let evaluation = driver.allocate_evaluation().unwrap();
    let plan =
        AdmittedPlan::try_i64_values("value", &[], &[7], &TaskLimits::qualification()).unwrap();
    let mut task = TaskMachine::new(
        id,
        EvaluationState {
            evaluation,
            plan: Some(plan),
            rows: 0,
        },
        TaskLimits::qualification(),
    )
    .unwrap();
    let TaskStep::Execute(start) = task.start(cpu()).unwrap() else {
        panic!("expected evaluation start")
    };
    let key = start.key;
    driver.dispatch(start).unwrap();
    driver.complete_effect().unwrap();
    assert_eq!(pulls.load(Ordering::Relaxed), 1);
    assert_eq!(batch_drops.load(Ordering::Relaxed), 0);
    assert_eq!(source_drops.load(Ordering::Relaxed), 0);
    let wrong_key = RequestKey::new(key.task_id(), key.request_id() + 1).unwrap();
    assert_eq!(
        driver.cancel(Some(wrong_key)),
        Err(TaskProtocolError::WrongKey)
    );
    assert_eq!(driver.outstanding_key(), Some(key));
    assert_eq!(batch_drops.load(Ordering::Relaxed), 0);
    assert_eq!(source_drops.load(Ordering::Relaxed), 0);
    assert_eq!(
        task.cancel(CancelReason::Caller),
        CancelDisposition::Cancelled(Some(key))
    );
    driver.cancel(Some(key)).unwrap();
    assert_eq!(batch_drops.load(Ordering::Relaxed), 1);
    assert_eq!(source_drops.load(Ordering::Relaxed), 1);
    assert_eq!(driver.outstanding_key(), None);
    assert_eq!(driver.active_evaluation(), None);
    assert!(matches!(
        driver.take_response(key),
        Err(TaskProtocolError::Terminal)
    ));
    assert_eq!(compiles.load(Ordering::Relaxed), 1);
}

#[test]
fn cancellation_before_effect_completion_drops_the_plan_without_compiling() {
    let compiles = Arc::new(AtomicUsize::new(0));
    let (id, mut driver) = EvaluationDriver::allocate({
        let compiles = compiles.clone();
        move |_: AdmittedPlan, _: EvaluationLimits| {
            compiles.fetch_add(1, Ordering::Relaxed);
            unreachable!("cancelled queued plans must not compile")
        }
    })
    .unwrap();
    let evaluation = driver.allocate_evaluation().unwrap();
    let plan =
        AdmittedPlan::try_i64_values("value", &[], &[7], &TaskLimits::qualification()).unwrap();
    let mut task = TaskMachine::new(
        id,
        EvaluationState {
            evaluation,
            plan: Some(plan),
            rows: 0,
        },
        TaskLimits::qualification(),
    )
    .unwrap();
    let TaskStep::Execute(start) = task.start(cpu()).unwrap() else {
        panic!("expected evaluation start")
    };
    let key = start.key;
    driver.dispatch(start).unwrap();
    assert_eq!(
        task.cancel(CancelReason::Caller),
        CancelDisposition::Cancelled(Some(key))
    );
    driver.cancel(Some(key)).unwrap();
    assert_eq!(compiles.load(Ordering::Relaxed), 0);
    assert_eq!(driver.complete_effect(), Err(TaskProtocolError::Terminal));
}

#[test]
fn compiler_failure_transfers_the_original_error_once_and_terminates_the_driver() {
    let (id, mut driver) = EvaluationDriver::allocate(
        |_: AdmittedPlan, _: EvaluationLimits| -> Result<AdmittedEvaluationSource, _> {
            Err(OperationFailure::new(
                FailureKind::Engine,
                Error::generic("fixed compiler failure"),
            ))
        },
    )
    .unwrap();
    let evaluation = driver.allocate_evaluation().unwrap();
    let plan =
        AdmittedPlan::try_i64_values("value", &[], &[7], &TaskLimits::qualification()).unwrap();
    let mut task = TaskMachine::new(
        id,
        EvaluationState {
            evaluation,
            plan: Some(plan),
            rows: 0,
        },
        TaskLimits::qualification(),
    )
    .unwrap();
    let TaskStep::Execute(start) = task.start(cpu()).unwrap() else {
        panic!("expected evaluation start")
    };
    let key = start.key;
    driver.dispatch(start).unwrap();
    driver.complete_effect().unwrap();
    let response = driver.take_response(key).unwrap();
    let TaskStep::Failed(failure) = task.resume(key, response, cpu()).unwrap() else {
        panic!("compiler failure must terminate the task")
    };
    assert_eq!(failure.kind(), FailureKind::Engine);
    assert!(failure
        .into_error()
        .to_string()
        .contains("fixed compiler failure"));
    assert_eq!(driver.complete_effect(), Err(TaskProtocolError::Terminal));
    assert!(matches!(
        driver.take_response(key),
        Err(TaskProtocolError::Terminal)
    ));
}

fn page_limits() -> EvaluationPageLimits {
    EvaluationPageLimits::new(1, 1, 128).unwrap()
}

fn evaluation_limits() -> EvaluationLimits {
    EvaluationLimits::new(page_limits(), 2, 2, 1, 256)
}

fn cpu() -> CpuSlice {
    CpuSlice::new(1, 1, 1).unwrap()
}
