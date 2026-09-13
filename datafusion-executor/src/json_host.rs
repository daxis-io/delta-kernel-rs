//! Concrete local async host for Kernel's sealed no-checkpoint JSON tasks.
use crate::closed_plan_facts::ClosedPlanFacts;
use crate::evaluation_page::{AdmittedPageData, PageEnvelope};
use crate::log_storage::{
    AdmittedJsonLogStorage, JsonLogStorageRegistry, LogStorageAdmissionError,
};
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::execution::{
    context::SessionContext,
    memory_pool::{MemoryConsumer, MemoryReservation},
    object_store::ObjectStoreUrl,
};
use datafusion::physical_plan::SendableRecordBatchStream;
use delta_kernel::tasks::*;
use futures::StreamExt;
use std::alloc::Layout;
use std::future::Future;
use std::mem::size_of;
use std::pin::Pin;
use std::sync::Arc;

/// Admission failure before the host performs storage I/O.
#[derive(Debug)]
pub enum JsonHostAdmissionError {
    /// The capability is missing or no longer binds the ordinary registered Arc.
    Storage(LogStorageAdmissionError),
    /// Finite owner/work limits or the caller's memory pool refused admission.
    Operation(OperationFailure),
}
impl std::fmt::Display for JsonHostAdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Storage(e) => e.fmt(f),
            Self::Operation(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for JsonHostAdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Storage(e) => Some(e),
            Self::Operation(e) => Some(e),
        }
    }
}

/// Uses caller registrations and allocation authority; owns one finite metadata
/// evaluation. It cannot execute an arbitrary plan or fall back to storage I/O.
pub struct JsonTaskHost<'a> {
    session: &'a SessionContext,
    origin: &'a ObjectStoreUrl,
    storage: Arc<dyn AdmittedJsonLogStorage>,
    limits: TaskLimits,
    fixed: usize,
    lease: Option<Arc<MemoryReservation>>,
    active: Option<Active>,
    started: bool,
}
struct Active {
    evaluation: EvaluationKey,
    limits: EvaluationLimits,
    stream: SendableRecordBatchStream,
    expected: SchemaRef,
    page: PageEnvelope,
    host_retained_bytes: usize,
}
impl<'a> JsonTaskHost<'a> {
    /// Resolves the capability against the same registered store Arc and admits
    /// the concrete host/future/error headers before allocation or storage I/O.
    pub fn new(
        session: &'a SessionContext,
        origin: &'a ObjectStoreUrl,
        limits: TaskLimits,
    ) -> Result<Self, JsonHostAdmissionError> {
        let fixed = Self::fixed_peak(limits).map_err(JsonHostAdmissionError::Operation)?;
        let storage = JsonLogStorageRegistry::resolve(session, origin)
            .map_err(JsonHostAdmissionError::Storage)?;
        let reservation =
            MemoryConsumer::new("KernelJsonHost").register(&session.runtime_env().memory_pool);
        reservation
            .try_grow(fixed)
            .map_err(|e| JsonHostAdmissionError::Operation(engine_error(e)))?;
        Ok(Self {
            session,
            origin,
            storage,
            limits,
            fixed,
            lease: Some(Arc::new(reservation)),
            active: None,
            started: false,
        })
    }
    fn fixed_peak(limits: TaskLimits) -> Result<usize, OperationFailure> {
        let fixed = Self::completion_layout()
            .size()
            .checked_add(size_of::<Self>())
            .and_then(|n| n.checked_add(crate::execution_aux::host_registration_peak()?))
            .and_then(|n| n.checked_add(size_of::<MemoryReservation>() + 2 * size_of::<usize>()))
            // Capability refusals use fixed Ready error futures, even when
            // their response payload budget is zero. Success frames belong to
            // the provider's supplied response envelope.
            .and_then(|n| {
                n.checked_add(
                    size_of::<futures::future::Ready<Result<AdmittedRead, OperationFailure>>>()
                        + size_of::<
                            futures::future::Ready<Result<AdmittedListingPage, OperationFailure>>,
                        >(),
                )
            })
            // Original typed DataFusion/Kernel error plus the lease wrapper.
            .and_then(|n| {
                n.checked_add(
                    size_of::<datafusion::common::DataFusionError>()
                        + 2 * size_of::<delta_kernel::Error>()
                        + size_of::<LeasedFailure>(),
                )
            })
            .ok_or_else(|| {
                exhausted(
                    Resource::TaskStateBytes,
                    limits.limit(Resource::TaskStateBytes),
                    usize::MAX,
                )
            })?;
        crate::evaluation_admission::check(fixed, limits)?;
        Ok(fixed)
    }
    fn completion_layout() -> Layout {
        fn layout<A, F: Future>(_: impl FnOnce(A) -> F) -> Layout {
            Layout::new::<F>()
        }
        layout(
            |(host, request, work): (
                &'static mut JsonTaskHost<'static>,
                &'static TaskRequestV1,
                PendingWork<'static>,
            )| host.complete_guarded(request, EvaluationUsage::default(), work),
        )
    }
    async fn complete_guarded(
        &mut self,
        request: &TaskRequestV1,
        usage: EvaluationUsage,
        work: PendingWork<'_>,
    ) -> Result<AdmittedAsyncEffect, OperationFailure> {
        let result = self.complete_unboxed(request, usage, work).await;
        match result {
            Ok(effect) => Ok(effect),
            Err(error) => {
                let kind = error.kind();
                self.active = None;
                let source = LeasedFailure {
                    error,
                    _lease: self.lease.take(),
                };
                Err(OperationFailure::new(
                    kind,
                    delta_kernel::Error::GenericError {
                        source: Box::new(source),
                    },
                ))
            }
        }
    }
    fn verify_binding(&self) -> Result<(), OperationFailure> {
        let current =
            JsonLogStorageRegistry::resolve(self.session, self.origin).map_err(engine_error)?;
        if !Arc::ptr_eq(&current, &self.storage) {
            return Err(engine_error(LogStorageAdmissionError::StoreMismatch));
        }
        Ok(())
    }
    fn reserve(&self, bytes: usize) -> Result<(), OperationFailure> {
        crate::evaluation_admission::check(bytes, self.limits)?;
        let lease = self
            .lease
            .as_ref()
            .ok_or_else(OperationFailure::malformed_response)?;
        if bytes > lease.size() {
            lease.try_grow(bytes - lease.size()).map_err(engine_error)?;
        }
        Ok(())
    }
    async fn complete_unboxed(
        &mut self,
        request: &TaskRequestV1,
        usage: EvaluationUsage,
        work: PendingWork<'_>,
    ) -> Result<AdmittedAsyncEffect, OperationFailure> {
        self.verify_binding()?;
        let result = match request {
            TaskRequestV1::List {
                root,
                continuation,
                entries,
                descriptor_bytes,
                continuation_bytes,
            } if !self.started => {
                // Discovery is one bounded producer page; metadata byte/cursor
                // admission belongs to that capability before its allocations.
                let owners = [
                    self.fixed,
                    work.retained_task_bytes(),
                    size_of::<TaskRequest>(),
                    root.capacity(),
                    continuation.as_ref().map_or(0, String::capacity),
                    *descriptor_bytes,
                    *continuation_bytes,
                ]
                .into_iter()
                .try_fold(0usize, usize::checked_add)
                .ok_or_else(|| {
                    exhausted(
                        Resource::TaskStateBytes,
                        self.limits.limit(Resource::TaskStateBytes),
                        usize::MAX,
                    )
                })?;
                self.reserve(owners)?;
                work.charge(*entries)?;
                let page = self
                    .storage
                    .list(
                        root,
                        continuation.as_deref(),
                        *entries,
                        *descriptor_bytes,
                        *continuation_bytes,
                    )
                    .await?;
                // Before returning to OperationDriver::validate_listing: its
                // capacity fold, prefix/order checks, and both cursor comparisons.
                // The third per-file visit is this allocation-free prepayment loop.
                for file in &page.files {
                    let units = file
                        .path
                        .len()
                        .checked_mul(2)
                        .and_then(|n| n.checked_add(3))
                        .ok_or_else(|| {
                            exhausted(
                                Resource::WorkUnits,
                                self.limits.limit(Resource::WorkUnits),
                                usize::MAX,
                            )
                        })?;
                    work.charge(units)?;
                }
                let cursors = page
                    .continuation
                    .as_ref()
                    .map_or(0, String::len)
                    .checked_add(page.binding.as_ref().map_or(0, String::len))
                    .ok_or_else(|| {
                        exhausted(
                            Resource::WorkUnits,
                            self.limits.limit(Resource::WorkUnits),
                            usize::MAX,
                        )
                    })?;
                work.charge(cursors)?;
                Ok(AdmittedAsyncEffect::Io(AdmittedIoEffect::Listing(page)))
            }
            TaskRequestV1::EvaluationStart {
                evaluation,
                plan,
                limits,
            } if !self.started => {
                self.started = true;
                self.start(*evaluation, plan, *limits, &work).await?;
                self.pull(*evaluation, limits.page(), usage).await
            }
            TaskRequestV1::Evaluation { evaluation, limits } => {
                self.pull(*evaluation, *limits, usage).await
            }
            _ => Err(OperationFailure::malformed_response()),
        };
        result
    }
    async fn start(
        &mut self,
        evaluation: EvaluationKey,
        plan: &AdmittedPlan,
        evaluation_limits: EvaluationLimits,
        work: &PendingWork<'_>,
    ) -> Result<(), OperationFailure> {
        // A page must be possible before any log provider is called. The exact
        // backing bound is checked again after allocation-free framing.
        for (resource, available) in [
            (
                Resource::EvaluationBatches,
                evaluation_limits.page().batches(),
            ),
            (Resource::EvaluationRows, evaluation_limits.page().rows()),
        ] {
            if available == 0 {
                return Err(exhausted(resource, available, 1));
            }
        }
        work.charge(
            plan.json_host_inspection_work(&self.limits)
                .map_err(OperationFailure::from)?,
        )?;
        let facts = ClosedPlanFacts::inspect(plan, self.limits)?;
        work.charge(crate::evaluation_work::preparation(&facts, self.limits)?)?;
        let conversions = crate::evaluation_admission::schema_peak(plan, self.limits)?;
        let manifest = plan
            .log_identity_manifest()
            .ok_or_else(OperationFailure::malformed_response)?
            .clone();
        let bytes = manifest
            .files()
            .iter()
            .try_fold(0usize, |n, f| {
                usize::try_from(f.size)
                    .ok()
                    .and_then(|size| n.checked_add(size))
            })
            .ok_or_else(|| {
                exhausted(
                    Resource::InputBytes,
                    self.limits.limit(Resource::InputBytes),
                    usize::MAX,
                )
            })?;
        let input_headers = size_of::<crate::log_input::LogInput>()
            .checked_add(
                facts
                    .files
                    .checked_mul(size_of::<AdmittedRead>())
                    .ok_or_else(OperationFailure::malformed_response)?,
            )
            .ok_or_else(OperationFailure::malformed_response)?;
        let simultaneous = work
            .retained_task_bytes()
            .checked_add(size_of::<TaskRequest>())
            .ok_or_else(OperationFailure::malformed_response)?;
        // Before the provider future exists, reserve its full supplied response
        // allowance, including parser/header scratch. All prior log bodies can
        // coexist with that response. Decoder construction occurs afterwards.
        let initial = [
            self.fixed,
            simultaneous,
            plan.retained_bytes(),
            bytes,
            input_headers,
            conversions,
            self.limits.limit(Resource::ReadPayloadBytes),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(|| {
            exhausted(
                Resource::TaskStateBytes,
                self.limits.limit(Resource::TaskStateBytes),
                usize::MAX,
            )
        })?;
        self.reserve(initial)?;
        // The plan bound already contains this SAME immutable manifest Arc.
        // LogInput adds it once; remove only that shared component here.
        let other = plan
            .retained_bytes()
            .checked_sub(manifest.retained_bytes())
            .and_then(|n| n.checked_add(self.fixed))
            .and_then(|n| n.checked_add(simultaneous))
            .ok_or_else(OperationFailure::malformed_response)?;
        work.charge(bytes)?; // lexical framing before any byte is parsed
        let input = crate::log_input::LogInput::load(
            manifest.clone(),
            self.storage.as_ref(),
            self.limits,
            other,
        )
        .await?;
        let prepared =
            crate::evaluation_admission::prepare(plan, &input, &facts, conversions, self.limits)?;
        self.reserve(prepared.peak)?;
        let host_retained_bytes = prepared.peak.max(initial);
        // Kernel's individually source-preflighted materialization follows
        // page transfer while this stream remains alive. Reserve its finite
        // combined ceiling now; the scalar handoff makes Kernel check every
        // subsequent owner against that same ceiling before allocation.
        self.reserve(
            self.limits
                .limit(Resource::TaskStateBytes)
                .min(self.limits.limit(Resource::MetadataAllocatedBytes)),
        )?;
        work.charge(crate::evaluation_work::planning(&facts, self.limits)?)?;
        let store: Arc<dyn object_store::ObjectStore> = Arc::new(
            crate::log_store::AdmittedLogStore::try_new(input, facts.files, self.limits)?,
        );
        let logical = crate::plan::lower_plan(plan.plan(), Some((&manifest, &store)))
            .map_err(engine_error)?;
        let state = crate::metadata_session::new(self.session).map_err(engine_error)?;
        let planner = datafusion::physical_planner::DefaultPhysicalPlanner::default();
        let physical = planner
            .create_physical_plan_unboxed(&logical, &state)
            .await
            .map_err(engine_error)?;
        drop(logical);
        // Source and aggregate work is paid once before the first stream is
        // created/polled. Later pages only expose already covered output rows.
        work.charge(crate::evaluation_work::execution(
            &facts,
            prepared.input_rows,
            bytes,
            self.limits,
        )?)?;
        let stream = physical
            .execute(0, state.task_ctx())
            .map_err(engine_error)?;
        self.active = Some(Active {
            evaluation,
            limits: evaluation_limits,
            stream,
            expected: prepared.expected,
            page: prepared.page,
            host_retained_bytes,
        });
        Ok(())
    }
    async fn pull(
        &mut self,
        evaluation: EvaluationKey,
        page_limits: EvaluationPageLimits,
        usage: EvaluationUsage,
    ) -> Result<AdmittedAsyncEffect, OperationFailure> {
        let active = self
            .active
            .as_mut()
            .ok_or_else(OperationFailure::malformed_response)?;
        if active.evaluation != evaluation {
            return Err(OperationFailure::malformed_response());
        }
        let payload = active.page.retained;
        let page_bytes = payload
            .checked_add(size_of::<Box<dyn AccountedEngineData>>())
            .ok_or_else(OperationFailure::malformed_response)?;
        // The driver has already admitted the page/container and updated usage.
        // Admit one possible full-backed batch before polling, including EOF.
        for (resource, used, amount, limit) in [
            (
                Resource::EvaluationPageBytes,
                0,
                page_bytes,
                page_limits.bytes(),
            ),
            (Resource::EvaluationRows, 0, 1, page_limits.rows()),
            (Resource::EvaluationBatches, 0, 1, page_limits.batches()),
            (
                Resource::EvaluationRows,
                usage.rows(),
                1,
                active.limits.rows(),
            ),
            (
                Resource::EvaluationBatches,
                usage.batches(),
                1,
                active.limits.batches(),
            ),
            (
                Resource::EvaluationBytes,
                usage.bytes(),
                payload,
                active.limits.bytes(),
            ),
        ] {
            let observed = used
                .checked_add(amount)
                .ok_or_else(|| exhausted(resource, limit, usize::MAX))?;
            if observed > limit {
                return Err(exhausted(resource, limit, observed));
            }
        }
        let page = match active.stream.next().await {
            Some(batch) => {
                let data = AdmittedPageData::new(
                    batch.map_err(engine_error)?,
                    active.expected.clone(),
                    active.page,
                    self.lease
                        .as_ref()
                        .ok_or_else(OperationFailure::malformed_response)?
                        .clone(),
                )?
                .with_host_retained_bytes(active.host_retained_bytes);
                Some(EvaluationPage::try_new(vec![Box::new(data)], page_limits)?)
            }
            None => {
                self.active = None;
                None
            }
        };
        Ok(AdmittedAsyncEffect::Evaluation { evaluation, page })
    }
}
impl AdmittedAsyncHost for JsonTaskHost<'_> {
    fn complete<'a>(
        &'a mut self,
        request: &'a TaskRequestV1,
        _: Option<ObjectIdentity>,
        usage: EvaluationUsage,
        work: PendingWork<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<AdmittedAsyncEffect, OperationFailure>> + 'a>> {
        Box::pin(self.complete_guarded(request, usage, work))
    }

    fn cancel(&mut self) {
        self.active = None;
        self.lease = None;
    }
}
#[derive(Debug)]
struct LeasedFailure {
    error: OperationFailure,
    _lease: Option<Arc<MemoryReservation>>,
}
impl std::fmt::Display for LeasedFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}
impl std::error::Error for LeasedFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}
pub(crate) fn engine_error<E: std::error::Error + Send + Sync + 'static>(
    error: E,
) -> OperationFailure {
    OperationFailure::new(
        FailureKind::Engine,
        delta_kernel::Error::GenericError {
            source: Box::new(error),
        },
    )
}
fn exhausted(resource: Resource, limit: usize, observed: usize) -> OperationFailure {
    ResourceExhausted {
        resource,
        limit,
        observed,
    }
    .into()
}
