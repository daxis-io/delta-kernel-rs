use std::mem::size_of;

use super::json_materialization::{check, engine, exhausted};
use super::json_snapshot::task_evaluation_limits;
use super::plan_admission::JsonPlanBudget;
use super::*;
use crate::engine_data::{GetData, RowVisitor};
use crate::scan::state::ScanFile;
use crate::scan::ScanMetadata;
use crate::schema::{ColumnName, DataType};
use crate::snapshot::SnapshotRef;
use crate::{EngineData, FilteredEngineData};

/// One independently identified evaluation of Kernel's JSON-only live-add plan. Output is a
/// finite, admitted collection in the existing scan-row representation. No execution stream,
/// future, storage handler or callback is retained by the semantic state.
pub struct ScanMetadataTask {
    machine: TaskMachine<ScanState>,
}
impl ScanMetadataTask {
    /// Uses only a snapshot produced by `SnapshotLoadTask`, retaining the same immutable log
    /// identity manifest. Stats pruning and caller predicates are excluded from metadata replay.
    pub fn try_new(
        id: TaskId,
        evaluation: EvaluationKey,
        snapshot: SnapshotRef,
        limits: TaskLimits,
    ) -> Result<Self, OperationFailure> {
        if evaluation.task_id() != id.get() {
            return Err(OperationFailure::malformed_response());
        }
        let manifest = snapshot.log_identity_manifest.as_ref().ok_or_else(|| {
            engine(crate::Error::unsupported(
                "scan tasks require a task-built snapshot",
            ))
        })?;
        let budget = JsonPlanBudget::preflight_scan(
            manifest,
            snapshot.schema().as_ref(),
            snapshot.table_configuration(),
            snapshot.json_task_retained_bytes,
            &limits,
        )
        .map_err(OperationFailure::from)?;
        let slots = limits
            .limit(Resource::OutputChunks)
            .min(limits.limit(Resource::EvaluationBatches));
        let container = slots
            .checked_mul(size_of::<ScanMetadata>())
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| exhausted(Resource::TaskStateBytes, &limits))?;
        let live = snapshot
            .json_task_retained_bytes
            .checked_add(budget.backing)
            .and_then(|n| n.checked_add(container))
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .and_then(|n| n.checked_add(size_of::<ScanState>()))
            .ok_or_else(|| exhausted(Resource::TaskStateBytes, &limits))?;
        check(Resource::TaskStateBytes, live, &limits)?;
        let plan = AdmittedPlan::try_scan_json(snapshot.clone(), &limits)
            .map_err(OperationFailure::from)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(slots)
            .map_err(|e| engine(crate::Error::generic_err(e)))?;
        let state = ScanState {
            snapshot,
            evaluation,
            limits,
            plan: Some(plan),
            pending: Vec::new(),
            pending_bytes: 0,
            host_retained_bytes: 0,
            output,
            output_bytes: 0,
            done: false,
        };
        let machine = TaskMachine::new(id, state, limits)?;
        // Construction was preflighted against this fresh task's limits above.
        // Record that admitted work immediately, including cancellation before
        // start; it must not disappear from the cumulative task ledger.
        machine
            .accounting()
            .accounting
            .charge(Resource::WorkUnits, budget.work_units())?;
        Ok(Self { machine })
    }
    /// Fixed owners used by the existing scan-file visitor. Adapters reserve
    /// this before visiting task-produced metadata again for file conversion.
    pub fn file_visitor_owner_bytes(limits: TaskLimits) -> Result<usize, OperationFailure> {
        let bytes = super::json_visitor_allocation::scan_fixed_peak()
            .ok_or_else(|| exhausted(Resource::TaskStateBytes, &limits))?;
        check(Resource::TaskStateBytes, bytes, &limits)?;
        Ok(bytes)
    }
    /// Prepay the fixed setup and row/getter walks of a borrowed preflight
    /// followed by the existing ScanFile visitor, including empty batches.
    pub fn file_visitor_work(rows: usize, limits: TaskLimits) -> Result<usize, OperationFailure> {
        let units = super::json_visitor_allocation::scan_work(0, rows)
            .ok_or_else(|| exhausted(Resource::WorkUnits, &limits))?;
        check(Resource::WorkUnits, units, &limits)?;
        Ok(units)
    }
    /// Source-derived URL join owners for a borrowed selected file path.
    pub fn file_path_owner_bytes(
        root_bytes: usize,
        path: &str,
        limits: TaskLimits,
    ) -> Result<usize, OperationFailure> {
        let bytes = super::json_url_allocation::join_peak(root_bytes, path)
            .ok_or_else(|| exhausted(Resource::TaskStateBytes, &limits))?;
        check(Resource::TaskStateBytes, bytes, &limits)?;
        Ok(bytes)
    }
    /// Checked semantic work for selected path cloning, URL join and confinement.
    /// Prepay path.len before this function's allocation-free classification.
    pub fn file_path_work(
        root_bytes: usize,
        path: &str,
        limits: TaskLimits,
    ) -> Result<usize, OperationFailure> {
        let units = super::json_url_allocation::join_work(root_bytes, path)
            .ok_or_else(|| exhausted(Resource::WorkUnits, &limits))?;
        check(Resource::WorkUnits, units, &limits)?;
        Ok(units)
    }
    /// Borrows cumulative and peak accounting after output transfer or cancellation.
    pub fn accounting(&self) -> TaskUsage<'_> {
        self.machine.accounting()
    }
}
impl OperationTask for ScanMetadataTask {
    fn pending_work(&self) -> Result<PendingWork<'_>, TaskProtocolError> {
        self.machine.pending_work()
    }
    type Output = Box<[ScanMetadata]>;
    fn start(&mut self, cpu: CpuSlice) -> Result<TaskStep<Self::Output>, TaskProtocolError> {
        self.machine.start(cpu)
    }
    fn progress(&mut self, cpu: CpuSlice) -> Result<TaskStep<Self::Output>, TaskProtocolError> {
        self.machine.progress(cpu)
    }
    fn resume(
        &mut self,
        key: RequestKey,
        response: Result<TaskResponseV1, OperationFailure>,
        cpu: CpuSlice,
    ) -> Result<TaskStep<Self::Output>, TaskProtocolError> {
        self.machine.resume(key, response, cpu)
    }
    fn cancel(&mut self, reason: CancelReason) -> CancelDisposition {
        self.machine.cancel(reason)
    }
}
struct ScanState {
    snapshot: SnapshotRef,
    evaluation: EvaluationKey,
    limits: TaskLimits,
    plan: Option<AdmittedPlan>,
    pending: Vec<Box<dyn AccountedEngineData>>,
    pending_bytes: usize,
    host_retained_bytes: usize,
    output: Vec<ScanMetadata>,
    output_bytes: usize,
    done: bool,
}
struct Validation<'a> {
    root: &'a url::Url,
    unsupported: bool,
    malformed: bool,
}
fn validate_file(context: &mut Validation<'_>, file: ScanFile) {
    context.unsupported |=
        file.dv_info.has_vector() || file.transform.is_some() || !file.partition_values.is_empty();
    let inside = context.root.join(&file.path).is_ok_and(|url| {
        url.query().is_none()
            && url.fragment().is_none()
            && url.as_str().starts_with(context.root.as_str())
    });
    context.malformed |= file.path.is_empty() || file.size <= 0 || !inside;
}
/// Borrow before the ordinary ScanFileVisitor can clone/parse stats or DV data.
/// The producer has disabled stats and this task constructs no transform owners.
struct ScanPreflight<'a> {
    accounting: &'a TaskAccounting,
    root_bytes: usize,
    row_peak: usize,
    limits: TaskLimits,
    failure: Option<OperationFailure>,
}
impl RowVisitor for ScanPreflight<'_> {
    fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
        crate::scan::state::SCAN_ROW_LEAVES.as_ref()
    }
    fn visit<'a>(
        &mut self,
        rows: usize,
        getters: &[&'a dyn GetData<'a>],
    ) -> crate::DeltaResult<()> {
        if getters.len() != 14 {
            return Err(crate::Error::generic("malformed scan metadata page"));
        }
        for row in 0..rows {
            if getters[3].get_str(row, "scanFile.stats")?.is_some()
                || getters[4]
                    .get_str(row, "scanFile.deletionVector.storageType")?
                    .is_some()
            {
                self.failure = Some(engine(crate::Error::unsupported(
                    "stats and DVs are outside JSON operation tasks",
                )));
                break;
            }
            let partitions =
                getters[9].get_map(row, "scanFile.fileConstantValues.partitionValues")?;
            match partitions {
                Some(map) if map.keys().next().is_none() => {}
                Some(_) => {
                    self.failure = Some(engine(crate::Error::unsupported(
                        "partitioned files are outside JSON operation tasks",
                    )));
                    break;
                }
                None => {
                    self.failure = Some(OperationFailure::malformed_response());
                    break;
                }
            }
            let Some(path) = getters[0]
                .get_str(row, "scanFile.path")?
                .filter(|s| !s.is_empty())
            else {
                self.failure = Some(OperationFailure::malformed_response());
                break;
            };
            if getters[1]
                .get_long(row, "scanFile.size")?
                .is_none_or(|size| size <= 0)
                || getters[2].get_long(row, "add.modificationTime")?.is_none()
            {
                self.failure = Some(OperationFailure::malformed_response());
                break;
            }
            // The ordinary visitor materializes one path String and invokes
            // validate_file before constructing the next ScanFile. Partition
            // maps are empty; stats/DVs were rejected while still borrowed.
            // Classification inspects borrowed bytes and allocates nothing.
            // Prepay that walk before determining the parser branch.
            if let Err(error) = self.accounting.charge(Resource::WorkUnits, path.len()) {
                self.failure = Some(error.into());
                break;
            }
            let work = super::json_url_allocation::join_work(self.root_bytes, path);
            let admitted = work
                .ok_or_else(|| exhausted(Resource::WorkUnits, &self.limits))
                .and_then(|units| {
                    self.accounting
                        .charge(Resource::WorkUnits, units)
                        .map_err(Into::into)
                });
            if let Err(error) = admitted {
                self.failure = Some(error);
                break;
            }
            let peak = super::json_url_allocation::join_peak(self.root_bytes, path)
                .and_then(|n| n.checked_add(path.len()));
            match peak {
                Some(peak) => self.row_peak = self.row_peak.max(peak),
                None => {
                    self.failure = Some(exhausted(Resource::TaskStateBytes, &self.limits));
                    break;
                }
            }
        }
        Ok(())
    }
}

impl ScanState {
    fn admit_extra(
        &self,
        accounting: &TaskAccounting,
        bytes: usize,
    ) -> Result<(), OperationFailure> {
        let current = self
            .retained_bytes()?
            .checked_add(size_of::<TaskMachine<Self>>())
            .ok_or_else(|| exhausted(Resource::TaskStateBytes, &self.limits))?
            .max(accounting.usage(Resource::TaskStateBytes).live());
        let total = current
            .checked_add(bytes)
            .ok_or_else(|| exhausted(Resource::TaskStateBytes, &self.limits))?;
        accounting.check(Resource::TaskStateBytes, total)?;
        if self.host_retained_bytes != 0 {
            accounting.check(Resource::MetadataAllocatedBytes, total)?;
        }
        Ok(())
    }
}
impl TaskState for ScanState {
    type Output = Box<[ScanMetadata]>;
    fn retained_bytes(&self) -> Result<usize, ResourceExhausted> {
        [
            self.snapshot.json_task_retained_bytes,
            self.plan.as_ref().map_or(0, AdmittedPlan::retained_bytes),
            self.pending_bytes,
            self.output_bytes,
            self.host_retained_bytes,
            self.output
                .capacity()
                .saturating_mul(size_of::<ScanMetadata>()),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(ResourceExhausted {
            resource: Resource::TaskStateBytes,
            limit: self.limits.limit(Resource::TaskStateBytes),
            observed: usize::MAX,
        })
    }
    fn advance(
        &mut self,
        cpu: CpuSlice,
        accounting: &TaskAccounting,
    ) -> Result<TaskAction<Self::Output>, OperationFailure> {
        let mut rows_processed = 0usize;
        let mut bytes_processed = 0usize;
        while let Some(batch) = self.pending.first() {
            let rows = batch.len();
            let bytes = batch.accounted_bytes()?;
            let next_rows = rows_processed
                .checked_add(rows)
                .ok_or_else(|| exhausted(Resource::TurnRecords, &self.limits))?;
            let next_bytes = bytes_processed
                .checked_add(bytes)
                .ok_or_else(|| exhausted(Resource::TurnInputBytes, &self.limits))?;
            if next_rows > cpu.records() || next_bytes > cpu.bytes() {
                if rows_processed == 0 && bytes_processed == 0 {
                    return Err(ResourceExhausted {
                        resource: if next_rows > cpu.records() {
                            Resource::TurnRecords
                        } else {
                            Resource::TurnInputBytes
                        },
                        limit: if next_rows > cpu.records() {
                            cpu.records()
                        } else {
                            cpu.bytes()
                        },
                        observed: if next_rows > cpu.records() {
                            next_rows
                        } else {
                            next_bytes
                        },
                    }
                    .into());
                }
                return Ok(TaskAction::Yield);
            }
            if self.output.len() == self.output.capacity() {
                return Err(exhausted(Resource::OutputChunks, &self.limits));
            }
            // Admit fixed schema/getter traversal before even the borrowed
            // preflight visits rows. It obtains actual borrowed path lengths;
            // no data-dependent String, map or URL has been constructed yet.
            let fixed = super::json_visitor_allocation::scan_fixed_peak()
                .ok_or_else(|| exhausted(Resource::TaskStateBytes, &self.limits))?;
            self.admit_extra(accounting, fixed)?;
            accounting.charge(
                Resource::WorkUnits,
                super::json_visitor_allocation::scan_work(0, rows)
                    .ok_or_else(|| exhausted(Resource::WorkUnits, &self.limits))?,
            )?;
            let mut preflight = ScanPreflight {
                accounting,
                root_bytes: self.snapshot.table_root().as_str().len(),
                row_peak: 0,
                limits: self.limits,
                failure: None,
            };
            preflight.visit_rows_of(batch.as_ref()).map_err(engine)?;
            if let Some(failure) = preflight.failure {
                return Err(failure);
            }
            // Vec<bool> uses one byte per row (not a packed bitmap). Its exact
            // construction coexists with ordinary visitor scratch and one URL.
            let temporary = fixed
                .checked_add(preflight.row_peak)
                .and_then(|n| n.checked_add(rows))
                .ok_or_else(|| exhausted(Resource::TaskStateBytes, &self.limits))?;
            self.admit_extra(accounting, temporary)?;
            let batch = self.pending.remove(0);
            self.pending_bytes = self
                .pending_bytes
                .checked_sub(bytes)
                .ok_or_else(OperationFailure::malformed_response)?;
            let data: Box<dyn EngineData> = batch;
            let metadata = ScanMetadata {
                scan_files: FilteredEngineData::try_new(data, vec![true; rows]).map_err(engine)?,
                scan_file_transforms: Vec::new(),
            };
            let validation = metadata
                .visit_scan_files(
                    Validation {
                        root: self.snapshot.table_root(),
                        unsupported: false,
                        malformed: false,
                    },
                    validate_file,
                )
                .map_err(engine)?;
            if validation.unsupported {
                return Err(engine(crate::Error::unsupported(
                    "DVs, partitions and row transforms are outside JSON operation tasks",
                )));
            }
            if validation.malformed {
                return Err(OperationFailure::malformed_response());
            }
            let retained = bytes
                .checked_add(rows)
                .and_then(|n| n.checked_add(size_of::<ScanMetadata>()))
                .ok_or_else(|| exhausted(Resource::OutputBytes, &self.limits))?;
            accounting.charge(Resource::OutputChunks, 1)?;
            accounting.charge(Resource::OutputBytes, retained)?;
            self.output_bytes = self
                .output_bytes
                .checked_add(retained)
                .ok_or_else(|| exhausted(Resource::OutputBytes, &self.limits))?;
            self.output.push(metadata);
            rows_processed = next_rows;
            bytes_processed = next_bytes;
        }
        self.pending = Vec::new();
        self.pending_bytes = 0;
        if self.done {
            self.admit_extra(
                accounting,
                self.output.len().saturating_mul(size_of::<ScanMetadata>()),
            )?;
            return Ok(TaskAction::Complete(
                std::mem::take(&mut self.output).into_boxed_slice(),
            ));
        }
        Ok(TaskAction::Request)
    }
    fn take_request(&mut self, _: &TaskAccounting) -> Result<TaskRequestV1, OperationFailure> {
        if let Some(plan) = self.plan.take() {
            Ok(TaskRequestV1::EvaluationStart {
                evaluation: self.evaluation,
                plan,
                limits: task_evaluation_limits(&self.limits)?,
            })
        } else {
            Ok(TaskRequestV1::Evaluation {
                evaluation: self.evaluation,
                limits: task_evaluation_limits(&self.limits)?.page(),
            })
        }
    }
    fn resume(
        &mut self,
        response: TaskResponseV1,
        cpu: CpuSlice,
        accounting: &TaskAccounting,
    ) -> Result<TaskAction<Self::Output>, OperationFailure> {
        let TaskResponseV1::Evaluation { evaluation, page } = response else {
            return Err(OperationFailure::malformed_response());
        };
        if evaluation != self.evaluation {
            return Err(OperationFailure::malformed_response());
        }
        accounting.charge(Resource::EvaluationPages, 1)?;
        if let Some(page) = page {
            check(
                Resource::EvaluationPageBatches,
                page.batches().len(),
                &self.limits,
            )?;
            check(Resource::EvaluationPageRows, page.num_rows(), &self.limits)?;
            check(
                Resource::EvaluationPageBytes,
                page.accounted_bytes(),
                &self.limits,
            )?;
            accounting.charge(Resource::EvaluationBatches, page.batches().len())?;
            accounting.charge(Resource::EvaluationRows, page.num_rows())?;
            accounting.charge(Resource::EvaluationBytes, page.accounted_bytes())?;
            self.host_retained_bytes = self.host_retained_bytes.max(
                page.batches()
                    .iter()
                    .map(|batch| batch.host_retained_bytes())
                    .max()
                    .unwrap_or(0),
            );
            self.admit_extra(accounting, page.accounted_bytes())?;
            self.pending_bytes = page.accounted_bytes();
            self.pending = page.into_batches();
        } else {
            self.done = true;
        }
        self.advance(cpu, accounting)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;
    use crate::schema::{DataType, StructField, StructType};
    use crate::tasks::json_snapshot::tests::Batch;
    use crate::tasks::plan_admission::json_tests::snapshot;

    #[test]
    fn json_task_scan_producer_work_survives_cancel_before_start() {
        let snapshot =
            snapshot(StructType::try_new([StructField::nullable("a", DataType::LONG)]).unwrap());
        let limits = TaskLimits::qualification();
        let manifest = snapshot.log_identity_manifest.as_ref().unwrap();
        let budget = JsonPlanBudget::preflight_scan(
            manifest,
            snapshot.schema().as_ref(),
            snapshot.table_configuration(),
            snapshot.json_task_retained_bytes,
            &limits,
        )
        .unwrap();
        let id = TaskId::allocate().unwrap();
        let evaluation = EvaluationKey::allocate(id.get()).unwrap();
        let mut task = ScanMetadataTask::try_new(id, evaluation, snapshot, limits).unwrap();
        assert!(budget.work_units() > 0);
        assert_eq!(
            task.accounting().usage(Resource::WorkUnits).consumed(),
            budget.work_units()
        );
        assert_eq!(
            task.cancel(CancelReason::Caller),
            CancelDisposition::Cancelled(None)
        );
        assert_eq!(
            task.accounting().usage(Resource::WorkUnits).consumed(),
            budget.work_units()
        );
    }

    #[test]
    fn json_task_scan_page_and_output_preflight_share_live_ownership() {
        let snapshot =
            snapshot(StructType::try_new([StructField::nullable("a", DataType::LONG)]).unwrap());
        let bytes = 100_000;
        let temporary = super::json_visitor_allocation::scan_fixed_peak().unwrap();
        let output = Vec::with_capacity(1);
        let base = snapshot.json_task_retained_bytes
            + output.capacity() * size_of::<ScanMetadata>()
            + size_of::<TaskMachine<ScanState>>();
        let page_bytes = bytes + size_of::<Box<dyn AccountedEngineData>>();
        let limit = base + page_bytes + temporary - 1;
        assert!(base + page_bytes < limit && base + temporary < limit);
        let limits = TaskLimits::qualification().with_limit(Resource::TaskStateBytes, limit);
        let id = TaskId::allocate().unwrap();
        let evaluation = EvaluationKey::allocate(id.get()).unwrap();
        let state = ScanState {
            snapshot,
            evaluation,
            limits,
            plan: None,
            pending: Vec::new(),
            pending_bytes: 0,
            host_retained_bytes: 0,
            output,
            output_bytes: 0,
            done: false,
        };
        let mut machine = TaskMachine::new(id, state, limits).unwrap();
        let cpu = CpuSlice::new(1024, 1 << 20, 256).unwrap();
        let TaskStep::Execute(request) = machine.start(cpu).unwrap() else {
            panic!("evaluation expected")
        };
        let drops = Arc::new(AtomicUsize::new(0));
        let visits = Arc::new(AtomicUsize::new(0));
        let batch = Batch {
            schema: "{}".into(),
            bytes,
            drops: drops.clone(),
            visits: visits.clone(),
        };
        let page = EvaluationPage::try_new(
            vec![Box::new(batch)],
            task_evaluation_limits(&limits).unwrap().page(),
        )
        .unwrap();
        let TaskStep::Failed(failure) = machine
            .resume(
                request.key,
                Ok(TaskResponseV1::Evaluation {
                    evaluation,
                    page: Some(page),
                }),
                cpu,
            )
            .unwrap()
        else {
            panic!("combined admission must fail")
        };
        assert!(
            matches!(failure.kind(), FailureKind::ResourceExhausted(e) if e.resource == Resource::TaskStateBytes)
        );
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(visits.load(Ordering::SeqCst), 0);
        assert_eq!(
            machine.accounting().usage(Resource::TaskStateBytes).live(),
            size_of::<TaskMachine<ScanState>>()
        );
        assert_eq!(
            machine.progress(cpu).err(),
            Some(TaskProtocolError::Terminal)
        );
    }

    #[test]
    fn json_task_scan_extreme_output_slots_fail_without_panicking() {
        let snapshot =
            snapshot(StructType::try_new([StructField::nullable("a", DataType::LONG)]).unwrap());
        let slots = usize::MAX / (2 * size_of::<ScanMetadata>());
        let limits = TaskLimits::qualification()
            .with_limit(Resource::TaskStateBytes, usize::MAX)
            .with_limit(Resource::OutputChunks, slots)
            .with_limit(Resource::EvaluationBatches, slots);
        let id = TaskId::allocate().unwrap();
        let evaluation = EvaluationKey::allocate(id.get()).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ScanMetadataTask::try_new(id, evaluation, snapshot, limits)
        }));
        let failure = result
            .expect("owner sums must not overflow before checked admission")
            .err()
            .unwrap();
        assert!(matches!(failure.kind(), FailureKind::ResourceExhausted(e)
            if e.resource == Resource::TaskStateBytes && e.observed == usize::MAX));
    }

    #[test]
    fn json_task_scan_fixed_work_refuses_before_first_borrowed_visit() {
        let snapshot =
            snapshot(StructType::try_new([StructField::nullable("a", DataType::LONG)]).unwrap());
        let limits = TaskLimits::qualification().with_limit(Resource::WorkUnits, 0);
        let id = TaskId::allocate().unwrap();
        let evaluation = EvaluationKey::allocate(id.get()).unwrap();
        let state = ScanState {
            snapshot,
            evaluation,
            limits,
            plan: None,
            pending: Vec::new(),
            pending_bytes: 0,
            host_retained_bytes: 0,
            output: Vec::with_capacity(1),
            output_bytes: 0,
            done: false,
        };
        let mut machine = TaskMachine::new(id, state, limits).unwrap();
        let cpu = CpuSlice::new(1024, 1 << 20, 256).unwrap();
        let TaskStep::Execute(request) = machine.start(cpu).unwrap() else {
            panic!("evaluation expected")
        };
        let visits = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let batch = Batch {
            schema: "{}".into(),
            bytes: 100,
            drops: drops.clone(),
            visits: visits.clone(),
        };
        let page = EvaluationPage::try_new(
            vec![Box::new(batch)],
            task_evaluation_limits(&limits).unwrap().page(),
        )
        .unwrap();
        let TaskStep::Failed(error) = machine
            .resume(
                request.key,
                Ok(TaskResponseV1::Evaluation {
                    evaluation,
                    page: Some(page),
                }),
                cpu,
            )
            .unwrap()
        else {
            panic!("work admission must refuse")
        };
        assert!(
            matches!(error.kind(), FailureKind::ResourceExhausted(e) if e.resource == Resource::WorkUnits)
        );
        assert_eq!(visits.load(Ordering::SeqCst), 0);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn json_task_scan_plan_limit_retains_typed_category() {
        let snapshot =
            snapshot(StructType::try_new([StructField::nullable("a", DataType::LONG)]).unwrap());
        let id = TaskId::allocate().unwrap();
        let evaluation = EvaluationKey::allocate(id.get()).unwrap();
        let failure = ScanMetadataTask::try_new(
            id,
            evaluation,
            snapshot,
            TaskLimits::qualification().with_limit(Resource::PlanNodes, 1),
        )
        .err()
        .unwrap();
        assert!(
            matches!(failure.kind(), FailureKind::ResourceExhausted(e) if e.resource == Resource::PlanNodes)
        );
    }
}

#[cfg(test)]
mod borrowed_preflight_tests {
    use super::*;
    use crate::engine_data::MapItem;
    struct Text;
    impl<'a> GetData<'a> for Text {
        fn get_str(&'a self, _: usize, _: &str) -> crate::DeltaResult<Option<&'a str>> {
            Ok(Some("must not be cloned or JSON-decoded"))
        }
    }
    struct UnreadableMap;
    impl<'a> GetData<'a> for UnreadableMap {
        fn get_map(&'a self, _: usize, _: &str) -> crate::DeltaResult<Option<MapItem<'a>>> {
            panic!("stats/DV rejection must precede further materialization")
        }
    }
    #[test]
    fn stats_and_dv_refuse_before_parsing_or_cloning() {
        for index in [3, 4] {
            let mut getters: [&dyn GetData<'_>; 14] = [&(); 14];
            getters[index] = &Text;
            getters[9] = &UnreadableMap;
            let accounting = TaskAccounting::new(TaskLimits::qualification());
            let mut preflight = ScanPreflight {
                accounting: &accounting,
                root_bytes: 0,
                row_peak: 0,
                limits: TaskLimits::qualification(),
                failure: None,
            };
            preflight.visit(1, &getters).unwrap();
            assert!(matches!(
                preflight.failure.unwrap().into_error(),
                crate::Error::Unsupported(_)
            ));
        }
    }
}
