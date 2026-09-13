use std::mem::size_of;
use std::sync::Arc;
use url::Url;

use super::json_materialization::{check, engine, exhausted, preflight_pm};
use super::log_manifest::commit_version;
use super::plan_admission::JsonPlanBudget;
use super::*;
use crate::log_segment::protocol_metadata_replay::{task_pm_from_plan_output, PmCandidate};
use crate::log_segment::LogSegment;
use crate::snapshot::{Snapshot, SnapshotRef};
use crate::table_configuration::TableConfiguration;

/// Bounded JSON-only snapshot discovery and projected protocol/metadata replay. The host supplies
/// an independent task/evaluation identity and owns every external effect. Successful snapshots
/// retain immutable discovery provenance for a later independent `ScanMetadataTask`.
///
/// This narrow task accepts unpartitioned primitive fields with empty field metadata.
/// Even otherwise harmless field metadata such as a comment is rejected as unsupported;
/// it is never silently discarded. Checkpoints, reader features and row transforms
/// remain outside this JSON-only task path.
pub struct SnapshotLoadTask {
    machine: TaskMachine<SnapshotState>,
}

impl SnapshotLoadTask {
    /// Admits task storage before cloning the canonical URL or reserving descriptor slots.
    pub fn try_new(
        id: TaskId,
        evaluation: EvaluationKey,
        table_root: &Url,
        version: Option<u64>,
        limits: TaskLimits,
    ) -> Result<Self, OperationFailure> {
        if evaluation.task_id() != id.get()
            || table_root.query().is_some()
            || table_root.fragment().is_some()
            || table_root.cannot_be_a_base()
        {
            return Err(OperationFailure::malformed_response());
        }
        #[cfg(feature = "adaptive-metadata-in-dev")]
        return Err(engine(crate::Error::unsupported(
            "adaptive metadata is outside JSON operation tasks",
        )));
        let slots = limits.limit(Resource::LogDescriptors);
        if version.is_some_and(|v| v >= slots as u64 || v > i64::MAX as u64) {
            return Err(exhausted(Resource::LogDescriptors, &limits));
        }
        let roots = root_owner_peak(table_root.as_str().len())
            .ok_or_else(|| exhausted(Resource::TaskStateBytes, &limits))?;
        let allocation = slots
            .checked_mul(size_of::<FileDescriptor>())
            .and_then(|n| n.checked_add(roots))
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .and_then(|n| n.checked_add(size_of::<SnapshotState>()))
            .ok_or_else(|| exhausted(Resource::TaskStateBytes, &limits))?;
        check(Resource::TaskStateBytes, allocation, &limits)?;
        let mut table_root = table_root.clone();
        if !table_root.path().ends_with('/') {
            table_root
                .path_segments_mut()
                .map_err(|_| OperationFailure::malformed_response())?
                .push("");
        }
        let log_root: String = table_root
            .join("_delta_log/")
            .map_err(|_| OperationFailure::malformed_response())?
            .into();
        let mut files = Vec::new();
        files
            .try_reserve_exact(slots)
            .map_err(|e| engine(crate::Error::generic_err(e)))?;
        let state = SnapshotState {
            table_root,
            log_root,
            roots,
            version,
            evaluation,
            limits,
            files,
            file_path_capacity: 0,
            listing_path_capacity: 0,
            listing: Vec::new(),
            listing_index: 0,
            continuation: None,
            discovery_done: false,
            manifest: None,
            segment: None,
            segment_bytes: 0,
            plan: None,
            evaluating: false,
            page: None,
            page_index: 0,
            candidate: None,
            materialized_bytes: 0,
            host_retained_bytes: 0,
            evaluation_done: false,
        };
        Ok(Self {
            machine: TaskMachine::new(id, state, limits)?,
        })
    }
    /// Borrows cumulative and peak accounting after terminal cleanup.
    pub fn accounting(&self) -> TaskUsage<'_> {
        self.machine.accounting()
    }
}

/// url 2.5.8 Url has one owned serialization String. Clone retains the
/// canonical host; no IDNA/host parser runs again. PathSegmentsMut takes that
/// String by move, with an empty after_path owner because queries/fragments
/// were rejected above. Appending the empty segment adds at most one slash.
/// The fixed relative join copies the base serialization then appends only
/// ASCII `_delta_log/`; parse_relative/parse_file use no other heap owner on
/// this branch. Converting the resulting Url into String moves serialization.
/// Include old/new RawVec relocation for both simultaneous String owners.
fn root_owner_peak(bytes: usize) -> Option<usize> {
    use super::json_schema_shape::vector_peak;
    let canonical = bytes.checked_add(1)?;
    let log = canonical.checked_add("_delta_log/".len())?;
    vector_peak::<u8>(canonical)?.checked_add(vector_peak::<u8>(log)?)
}

impl OperationTask for SnapshotLoadTask {
    fn pending_work(&self) -> Result<PendingWork<'_>, TaskProtocolError> {
        self.machine.pending_work()
    }
    type Output = SnapshotRef;
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

struct SnapshotState {
    table_root: Url,
    log_root: String,
    roots: usize,
    version: Option<u64>,
    evaluation: EvaluationKey,
    limits: TaskLimits,
    files: Vec<FileDescriptor>,
    file_path_capacity: usize,
    listing_path_capacity: usize,
    listing: Vec<FileDescriptor>,
    listing_index: usize,
    continuation: Option<String>,
    discovery_done: bool,
    manifest: Option<Arc<LogIdentityManifest>>,
    segment: Option<LogSegment>,
    segment_bytes: usize,
    plan: Option<AdmittedPlan>,
    evaluating: bool,
    page: Option<EvaluationPage>,
    page_index: usize,
    candidate: Option<PmCandidate>,
    materialized_bytes: usize,
    host_retained_bytes: usize,
    evaluation_done: bool,
}

impl SnapshotState {
    fn admit_extra(
        &self,
        accounting: &TaskAccounting,
        extra: usize,
    ) -> Result<(), OperationFailure> {
        let current = self
            .retained_bytes()?
            .checked_add(size_of::<TaskMachine<Self>>())
            .ok_or_else(|| exhausted(Resource::TaskStateBytes, &self.limits))?
            .max(accounting.usage(Resource::TaskStateBytes).live());
        let total = current
            .checked_add(extra)
            .ok_or_else(|| exhausted(Resource::TaskStateBytes, &self.limits))?;
        accounting.check(Resource::TaskStateBytes, total)?;
        if self.host_retained_bytes != 0 {
            accounting.check(Resource::MetadataAllocatedBytes, total)?;
        }
        Ok(())
    }
    fn finish_discovery(
        &mut self,
        accounting: &TaskAccounting,
        cpu: CpuSlice,
    ) -> Result<(), OperationFailure> {
        if self
            .version
            .is_some_and(|v| self.files.len() as u64 != v + 1)
            || self.files.is_empty()
        {
            return Err(engine(crate::Error::MissingVersion));
        }
        if cpu.plan_nodes() < 8 {
            return Err(exhausted(Resource::TurnPlanNodes, &self.limits));
        }
        let producer_work =
            JsonPlanBudget::producer_work(self.files.len(), self.log_root.len(), &self.limits)
                .map_err(OperationFailure::from)?;
        let manifest_work =
            super::json_producer_shape::manifest_work(self.files.len(), self.log_root.len())
                .ok_or_else(|| exhausted(Resource::WorkUnits, &self.limits))?;
        let work = producer_work
            .checked_add(manifest_work)
            .ok_or_else(|| exhausted(Resource::WorkUnits, &self.limits))?;
        accounting.charge(Resource::WorkUnits, work)?;
        self.admit_extra(
            accounting,
            self.log_root.len() + size_of::<LogIdentityManifest>() + 2 * size_of::<usize>(),
        )?;
        let manifest = LogIdentityManifest::try_new(
            self.log_root.clone(),
            std::mem::take(&mut self.files),
            &self.limits,
        )?;
        self.file_path_capacity = 0; // ownership moved into the manifest
        let budget =
            JsonPlanBudget::preflight(&manifest, &self.limits).map_err(OperationFailure::from)?;
        let segment_bytes = manifest.segment_owner_peak(&self.limits)?;
        let simultaneous = manifest
            .retained_bytes()
            .checked_add(segment_bytes)
            .and_then(|n| n.checked_add(budget.backing))
            .ok_or_else(|| exhausted(Resource::TaskStateBytes, &self.limits))?;
        self.admit_extra(accounting, simultaneous)?;
        let manifest = Arc::new(manifest);
        let segment = manifest.to_log_segment(&self.table_root, &self.limits)?;
        let plan = AdmittedPlan::try_snapshot_json(&segment, manifest.clone(), &self.limits)
            .map_err(OperationFailure::from)?;
        self.segment_bytes = segment_bytes;
        self.manifest = Some(manifest);
        self.segment = Some(segment);
        self.plan = Some(plan);
        self.evaluating = true;
        Ok(())
    }

    fn finish_snapshot(
        &mut self,
        accounting: &TaskAccounting,
    ) -> Result<SnapshotRef, OperationFailure> {
        let candidate = self
            .candidate
            .take()
            .ok_or_else(|| engine(crate::Error::MissingMetadataAndProtocol))?;
        let (metadata_version, metadata) = candidate
            .metadata
            .ok_or_else(|| engine(crate::Error::MissingMetadata))?;
        let (protocol_version, protocol) = candidate
            .protocol
            .ok_or_else(|| engine(crate::Error::MissingProtocol))?;
        let manifest = self
            .manifest
            .as_ref()
            .ok_or_else(OperationFailure::malformed_response)?;
        if metadata_version != 0 {
            return Err(engine(crate::Error::unsupported(
                "metadata evolution is outside JSON operation tasks",
            )));
        }
        if protocol_version < 0 || protocol_version as u64 > manifest.version() {
            return Err(OperationFailure::malformed_response());
        }
        if protocol.min_reader_version() != 1
            || protocol.min_writer_version() > 2
            || protocol.reader_features().is_some_and(|f| !f.is_empty())
            || protocol.writer_features().is_some_and(|f| !f.is_empty())
        {
            return Err(engine(crate::Error::unsupported(
                "table protocol/features are outside JSON operation tasks",
            )));
        }
        if !metadata.partition_columns().is_empty() {
            return Err(engine(crate::Error::unsupported(
                "partitioned tables are outside JSON operation tasks",
            )));
        }
        // Reject recursive/metadata-bearing schema materializers before entering serde's
        // ordinary Schema decoder. The inspection retains only counters and bounded scratch.
        let shape = super::json_schema_shape::PrimitiveSchemaShape::inspect(
            metadata.schema_string(),
            &self.limits,
            |scratch| self.admit_extra(accounting, scratch),
        )?;
        let configuration_bytes = shape
            .configuration_bytes(&metadata, self.table_root.as_str().len())
            .ok_or_else(|| exhausted(Resource::MetadataAllocatedBytes, &self.limits))?;
        check(
            Resource::MetadataAllocatedBytes,
            configuration_bytes,
            &self.limits,
        )?;
        self.admit_extra(accounting, configuration_bytes)?;
        let configuration = TableConfiguration::try_new_json_task(
            metadata,
            protocol,
            self.table_root.clone(),
            manifest.version(),
        )
        .map_err(engine)?;
        self.materialized_bytes = self
            .materialized_bytes
            .checked_add(configuration_bytes)
            .ok_or_else(|| exhausted(Resource::TaskStateBytes, &self.limits))?;
        for field in configuration.logical_schema().fields() {
            if !field.metadata().is_empty()
                || !matches!(field.data_type(), crate::schema::DataType::Primitive(_))
            {
                return Err(engine(crate::Error::unsupported(
                    "JSON operation tasks require plain primitive fields",
                )));
            }
            use crate::schema::PrimitiveType;
            if matches!(
                field.data_type(),
                crate::schema::DataType::Primitive(
                    PrimitiveType::Void
                        | PrimitiveType::IntervalYearMonth
                        | PrimitiveType::IntervalDayTime
                )
            ) {
                return Err(engine(crate::Error::unsupported(
                    "primitive type is outside JSON operation tasks",
                )));
            }
        }
        let output_bytes = self
            .materialized_bytes
            .checked_add(self.segment_bytes)
            .and_then(|n| n.checked_add(manifest.retained_bytes()))
            .ok_or_else(|| exhausted(Resource::OutputBytes, &self.limits))?;
        accounting.charge(Resource::OutputBytes, output_bytes)?;
        let mut snapshot = Snapshot::new_with_crc(
            self.segment
                .take()
                .ok_or_else(OperationFailure::malformed_response)?,
            configuration,
            None,
            self.version.is_none(),
        )
        .map_err(engine)?;
        snapshot.json_task_retained_bytes = output_bytes;
        snapshot.log_identity_manifest = self.manifest.take();
        Ok(Arc::new(snapshot))
    }
}

impl TaskState for SnapshotState {
    type Output = SnapshotRef;
    fn retained_bytes(&self) -> Result<usize, ResourceExhausted> {
        let bytes = [
            self.roots,
            self.files
                .capacity()
                .saturating_mul(size_of::<FileDescriptor>()),
            self.listing
                .capacity()
                .saturating_mul(size_of::<FileDescriptor>()),
            self.file_path_capacity,
            self.listing_path_capacity,
            self.continuation.as_ref().map_or(0, String::capacity),
            self.manifest.as_ref().map_or(0, |m| m.retained_bytes()),
            self.segment_bytes,
            self.plan.as_ref().map_or(0, AdmittedPlan::retained_bytes),
            self.page
                .as_ref()
                .map_or(0, EvaluationPage::accounted_bytes),
            self.materialized_bytes,
            self.host_retained_bytes,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(ResourceExhausted {
            resource: Resource::TaskStateBytes,
            limit: self.limits.limit(Resource::TaskStateBytes),
            observed: usize::MAX,
        })?;
        Ok(bytes)
    }
    fn advance(
        &mut self,
        cpu: CpuSlice,
        accounting: &TaskAccounting,
    ) -> Result<TaskAction<Self::Output>, OperationFailure> {
        if !self.evaluating {
            let mut processed = 0;
            let mut bytes = 0usize;
            while self.listing_index < self.listing.len() && processed < cpu.records() {
                let file = &mut self.listing[self.listing_index];
                let next_bytes = bytes
                    .checked_add(file.path.len())
                    .ok_or_else(|| exhausted(Resource::TurnInputBytes, &self.limits))?;
                if next_bytes > cpu.bytes() {
                    if processed == 0 {
                        return Err(exhausted(Resource::TurnInputBytes, &self.limits));
                    }
                    break;
                }
                bytes = next_bytes;
                accounting.charge(Resource::Records, 1)?;
                // Borrow the relative suffix and validate a canonical commit name:
                // two prefix comparisons, then 20 digits and the five-byte suffix.
                let work = file
                    .path
                    .len()
                    .checked_mul(2)
                    .and_then(|n| n.checked_add(25 + 1))
                    .ok_or_else(|| exhausted(Resource::WorkUnits, &self.limits))?;
                accounting.charge(Resource::WorkUnits, work)?;
                let relative = file
                    .path
                    .strip_prefix(&self.log_root)
                    .ok_or_else(OperationFailure::malformed_response)?;
                if let Some(version) = commit_version(&self.log_root, &file.path) {
                    if self.version.is_some_and(|wanted| version > wanted) {
                        self.discovery_done = true;
                        self.listing_index = self.listing.len();
                        break;
                    }
                    if version != self.files.len() as u64 {
                        return Err(engine(crate::Error::MissingVersion));
                    }
                    if self.files.len() == self.files.capacity() {
                        return Err(exhausted(Resource::LogDescriptors, &self.limits));
                    }
                    accounting.charge(Resource::LogDescriptors, 1)?;
                    accounting.charge(
                        Resource::DescriptorBytes,
                        file.path.capacity() + size_of::<FileDescriptor>(),
                    )?;
                    self.file_path_capacity = self
                        .file_path_capacity
                        .checked_add(file.path.capacity())
                        .ok_or_else(|| exhausted(Resource::TaskStateBytes, &self.limits))?;
                    self.listing_path_capacity = self
                        .listing_path_capacity
                        .checked_sub(file.path.capacity())
                        .ok_or_else(OperationFailure::malformed_response)?;
                    self.files.push(FileDescriptor {
                        path: std::mem::take(&mut file.path),
                        size: file.size,
                        modification_time: file.modification_time,
                        identity: file.identity,
                    });
                    if self.version == Some(version) {
                        self.discovery_done = true;
                        self.listing_index = self.listing.len();
                        break;
                    }
                } else {
                    // str's two-way substring search is linear (at most two byte
                    // traversals); the two fixed prefix comparisons are separate.
                    let work = relative
                        .len()
                        .checked_mul(2)
                        .and_then(|n| n.checked_add("_sidecars/".len() + "_staged_commits/".len()))
                        .ok_or_else(|| exhausted(Resource::WorkUnits, &self.limits))?;
                    accounting.charge(Resource::WorkUnits, work)?;
                    if relative.contains(".checkpoint.")
                        || relative.starts_with("_sidecars/")
                        || relative.starts_with("_staged_commits/")
                    {
                        return Err(engine(crate::Error::unsupported(
                            "checkpoints and staged commits are outside JSON operation tasks",
                        )));
                    }
                }
                self.listing_index += 1;
                processed += 1;
            }
            if self.listing_index < self.listing.len() {
                return Ok(TaskAction::Yield);
            }
            self.listing = Vec::new();
            self.listing_path_capacity = 0;
            self.listing_index = 0;
            if !self.discovery_done {
                return Ok(TaskAction::Request);
            }
            self.finish_discovery(accounting, cpu)?;
            return Ok(TaskAction::Request);
        }
        if self.evaluation_done {
            return self.finish_snapshot(accounting).map(TaskAction::Complete);
        }
        if let Some(page) = &self.page {
            if self.page_index < page.batches().len() && cpu.records() == 0 {
                return Ok(TaskAction::Yield);
            }
            let mut processed = 0;
            while self.page_index < self.page.as_ref().map_or(0, |page| page.batches().len())
                && processed < cpu.records()
            {
                let batch = self
                    .page
                    .as_ref()
                    .and_then(|page| page.batches().get(self.page_index))
                    .ok_or_else(OperationFailure::malformed_response)?;
                if batch.len() != 0 {
                    if self.candidate.is_some() {
                        return Err(OperationFailure::malformed_response());
                    }
                    if batch.accounted_bytes()? > cpu.bytes() {
                        return Err(exhausted(Resource::TurnInputBytes, &self.limits));
                    }
                    let charge = preflight_pm(batch.as_ref(), &self.limits, |bytes| {
                        self.admit_extra(accounting, bytes)
                    })?;
                    accounting.charge(Resource::DecodedBytes, charge)?;
                    accounting.charge(Resource::WorkUnits, batch.accounted_bytes()?)?;
                    self.candidate =
                        Some(task_pm_from_plan_output(batch.as_ref()).map_err(engine)?);
                    self.materialized_bytes = charge;
                }
                self.page_index += 1;
                processed += 1;
            }
            if self.page_index < self.page.as_ref().map_or(0, |page| page.batches().len()) {
                return Ok(TaskAction::Yield);
            }
            self.page = None;
            self.page_index = 0;
        }
        Ok(TaskAction::Request)
    }
    fn take_request(
        &mut self,
        accounting: &TaskAccounting,
    ) -> Result<TaskRequestV1, OperationFailure> {
        if !self.evaluating {
            self.admit_extra(
                accounting,
                self.log_root.len() + self.continuation.as_ref().map_or(0, String::len),
            )?;
            return Ok(TaskRequestV1::List {
                root: self.log_root.clone(),
                continuation: self.continuation.clone(),
                entries: self.limits.limit(Resource::ListingEntries),
                descriptor_bytes: self.limits.limit(Resource::ListingDescriptorBytes),
                continuation_bytes: self.limits.limit(Resource::ContinuationBytes),
            });
        }
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
        match response {
            TaskResponseV1::Listing {
                files,
                continuation,
            } if !self.evaluating => {
                check(Resource::ListingEntries, files.len(), &self.limits)?;
                accounting.charge(Resource::WorkUnits, files.len())?; // capacity fold below
                let bytes = files
                    .capacity()
                    .checked_mul(size_of::<FileDescriptor>())
                    .and_then(|n| {
                        files
                            .iter()
                            .try_fold(n, |used, f| used.checked_add(f.path.capacity()))
                    })
                    .ok_or_else(|| exhausted(Resource::ListingDescriptorBytes, &self.limits))?;
                check(Resource::ListingDescriptorBytes, bytes, &self.limits)?;
                check(
                    Resource::ContinuationBytes,
                    continuation.as_ref().map_or(0, String::capacity),
                    &self.limits,
                )?;
                self.admit_extra(
                    accounting,
                    bytes + continuation.as_ref().map_or(0, String::capacity),
                )?;
                let mut previous = self.continuation.as_deref();
                for file in &files {
                    let work = file
                        .path
                        .len()
                        .checked_mul(2)
                        .and_then(|n| n.checked_add(1))
                        .ok_or_else(|| exhausted(Resource::WorkUnits, &self.limits))?;
                    accounting.charge(Resource::WorkUnits, work)?; // prefix and predecessor comparison
                    if !file.path.starts_with(&self.log_root)
                        || previous.is_some_and(|p| p >= file.path.as_str())
                    {
                        return Err(OperationFailure::malformed_response());
                    }
                    previous = Some(&file.path);
                }
                let expected = (files.len() == self.limits.limit(Resource::ListingEntries))
                    .then(|| files.last().map(|f| f.path.as_str()))
                    .flatten();
                accounting.charge(
                    Resource::WorkUnits,
                    continuation.as_ref().map_or(0, String::len),
                )?;
                if continuation.as_deref() != expected {
                    return Err(OperationFailure::malformed_response());
                }
                self.discovery_done = continuation.is_none();
                self.continuation = continuation;
                self.listing_path_capacity = bytes - files.capacity() * size_of::<FileDescriptor>();
                self.listing = files;
            }
            TaskResponseV1::Evaluation { evaluation, page }
                if self.evaluating && evaluation == self.evaluation =>
            {
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
                    if page.num_rows() > 1 {
                        return Err(OperationFailure::malformed_response());
                    }
                    self.host_retained_bytes = self.host_retained_bytes.max(
                        page.batches()
                            .iter()
                            .map(|batch| batch.host_retained_bytes())
                            .max()
                            .unwrap_or(0),
                    );
                    self.admit_extra(accounting, page.accounted_bytes())?;
                    self.page = Some(page);
                } else {
                    self.evaluation_done = true;
                }
            }
            _ => return Err(OperationFailure::malformed_response()),
        }
        self.advance(cpu, accounting)
    }
}

pub(super) fn task_evaluation_limits(
    limits: &TaskLimits,
) -> Result<EvaluationLimits, OperationFailure> {
    let page = EvaluationPageLimits::new(
        limits.limit(Resource::EvaluationPageBatches),
        limits.limit(Resource::EvaluationPageRows),
        limits.limit(Resource::EvaluationPageBytes),
    )
    .map_err(|e| engine(crate::Error::generic_err(e)))?;
    Ok(EvaluationLimits::new(
        page,
        limits.limit(Resource::EvaluationPages),
        limits.limit(Resource::EvaluationBatches),
        limits.limit(Resource::EvaluationRows),
        limits.limit(Resource::EvaluationBytes),
    ))
}
#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::engine_data::{GetData, RowVisitor};
    use crate::expressions::{ArrayData, ColumnName};
    use crate::schema::SchemaRef;
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub(crate) struct Batch {
        pub schema: String,
        pub bytes: usize,
        pub drops: Arc<AtomicUsize>,
        pub visits: Arc<AtomicUsize>,
    }
    impl Drop for Batch {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    impl<'a> GetData<'a> for Batch {
        fn get_str(&'a self, _: usize, _: &str) -> crate::DeltaResult<Option<&'a str>> {
            Ok(Some(&self.schema))
        }
    }
    impl crate::EngineData for Batch {
        fn len(&self) -> usize {
            1
        }
        fn visit_rows(
            &self,
            columns: &[ColumnName],
            visitor: &mut dyn RowVisitor,
        ) -> crate::DeltaResult<()> {
            match self.visits.fetch_add(1, Ordering::SeqCst) {
                0 => {
                    assert_eq!(columns.len(), 4, "borrowed protocol preflight expected");
                    visitor.visit(1, &[&(), &(), &(), &()])
                }
                1 => {
                    assert_eq!(columns.len(), 9, "borrowed metadata preflight expected");
                    visitor.visit(1, &[self, &(), &(), &(), &(), self, &(), &(), &()])
                }
                _ => panic!("owned action materialization must not run after rejected preflight"),
            }
        }
        fn append_columns(
            &self,
            _: SchemaRef,
            _: Vec<ArrayData>,
        ) -> crate::DeltaResult<Box<dyn crate::EngineData>> {
            Err(crate::Error::unsupported("test append"))
        }
        fn apply_selection_vector(
            self: Box<Self>,
            _: Vec<bool>,
        ) -> crate::DeltaResult<Box<dyn crate::EngineData>> {
            Ok(self)
        }
        fn has_field(&self, _: &ColumnName) -> bool {
            true
        }
    }
    impl AccountedEngineData for Batch {
        fn accounted_bytes(&self) -> Result<usize, ResourceExhausted> {
            Ok(self.bytes)
        }
    }
    fn state(limits: TaskLimits, evaluation: EvaluationKey) -> SnapshotState {
        SnapshotState {
            table_root: Url::parse("memory:///table/").unwrap(),
            log_root: String::new(),
            roots: 1024,
            version: None,
            evaluation,
            limits,
            files: Vec::new(),
            file_path_capacity: 0,
            listing_path_capacity: 0,
            listing: Vec::new(),
            listing_index: 0,
            continuation: None,
            discovery_done: true,
            manifest: None,
            segment: None,
            segment_bytes: 0,
            plan: None,
            evaluating: true,
            page: None,
            page_index: 0,
            candidate: None,
            materialized_bytes: 0,
            host_retained_bytes: 0,
            evaluation_done: false,
        }
    }
    fn cpu() -> CpuSlice {
        CpuSlice::new(1024, 1 << 20, 256).unwrap()
    }
    fn assert_resource(failure: &OperationFailure, resource: Resource) {
        assert!(
            matches!(failure.kind(), FailureKind::ResourceExhausted(e) if e.resource == resource),
            "{failure:?}"
        );
    }
    #[test]
    fn listing_path_validation_is_prepaid_at_exact_and_one_less_work() {
        let path = "memory:///table/_delta_log/00000000000000000000.json";
        let required = 1 + 1 + 2 * path.len(); // capacity fold, validation visit, two comparisons
        for available in [required - 1, required] {
            let limits = TaskLimits::qualification()
                .with_limit(Resource::ListingEntries, 2)
                .with_limit(Resource::WorkUnits, available);
            let id = TaskId::allocate().unwrap();
            let evaluation = EvaluationKey::allocate(id.get()).unwrap();
            let mut task = SnapshotLoadTask::try_new(
                id,
                evaluation,
                &Url::parse("memory:///table/").unwrap(),
                None,
                limits,
            )
            .unwrap();
            let TaskStep::Execute(request) = task.start(cpu()).unwrap() else {
                panic!("listing")
            };
            // One byte prevents advance from entering filename interpretation.
            let turn = CpuSlice::new(1, 1, 256).unwrap();
            let step = task
                .resume(
                    request.key,
                    Ok(TaskResponseV1::Listing {
                        files: vec![FileDescriptor {
                            path: path.into(),
                            size: 1,
                            modification_time: 0,
                            identity: ObjectIdentity::new([1; 32]),
                        }],
                        continuation: None,
                    }),
                    turn,
                )
                .unwrap();
            let TaskStep::Failed(error) = step else {
                panic!("bounded turn must refuse")
            };
            assert_resource(
                &error,
                if available == required {
                    Resource::TurnInputBytes
                } else {
                    Resource::WorkUnits
                },
            );
            assert_eq!(task.accounting().usage(Resource::Records).consumed(), 0);
            assert_eq!(
                task.accounting().usage(Resource::WorkUnits).consumed(),
                if available == required { required } else { 1 }
            );
        }
    }

    #[test]
    fn cached_path_capacities_follow_sliced_listing_cutoff_transfer_and_cancel() {
        let root = "memory:///table/_delta_log/";
        let limits = TaskLimits::qualification().with_limit(Resource::ListingEntries, 4);
        for cancel_after_partial in [true, false] {
            let id = TaskId::allocate().unwrap();
            let evaluation = EvaluationKey::allocate(id.get()).unwrap();
            let mut state = state(limits, evaluation);
            state.log_root = root.into();
            state.files = Vec::with_capacity(3);
            state.version = Some(1);
            state.evaluating = false;
            state.discovery_done = false;
            let listing: Vec<_> = (0..3)
                .map(|version| {
                    let mut path = format!("{root}{version:020}.json");
                    // Deliberately unequal spare capacities detect length-based
                    // accounting and subtraction of the wrong moved owner.
                    path.reserve(31 + version * 17);
                    FileDescriptor {
                        path,
                        size: 1,
                        modification_time: 0,
                        identity: ObjectIdentity::new([version as u8; 32]),
                    }
                })
                .collect();
            let capacities: Vec<_> = listing.iter().map(|f| f.path.capacity()).collect();
            let accounting = TaskAccounting::new(limits);
            let slice = CpuSlice::new(1, 1 << 20, 256).unwrap();
            assert!(matches!(
                state
                    .resume(
                        TaskResponseV1::Listing {
                            files: listing,
                            continuation: None
                        },
                        slice,
                        &accounting,
                    )
                    .unwrap(),
                TaskAction::Yield
            ));
            // The real accepted response has moved exactly one owner this turn.
            assert_eq!(state.files.len(), 1);
            assert_eq!(state.listing_index, 1);
            assert_eq!(state.file_path_capacity, capacities[0]);
            assert_eq!(state.listing_path_capacity, capacities[1] + capacities[2]);
            assert_eq!(
                state.file_path_capacity,
                state.files.iter().map(|f| f.path.capacity()).sum::<usize>()
            );
            assert_eq!(
                state.listing_path_capacity,
                state
                    .listing
                    .iter()
                    .map(|f| f.path.capacity())
                    .sum::<usize>()
            );
            if !cancel_after_partial {
                assert!(matches!(
                    state.advance(slice, &accounting).unwrap(),
                    TaskAction::Request
                ));
                // Version1 ends discovery: version2 is discarded, and accepted
                // version0/1 path owners transfer intact into the manifest.
                assert!(state.files.is_empty());
                assert!(state.listing.is_empty());
                assert_eq!(
                    state.file_path_capacity,
                    state.files.iter().map(|f| f.path.capacity()).sum::<usize>()
                );
                assert_eq!(
                    state.listing_path_capacity,
                    state
                        .listing
                        .iter()
                        .map(|f| f.path.capacity())
                        .sum::<usize>()
                );
                let files = state.manifest.as_ref().unwrap().files();
                assert_eq!(files.len(), 2);
                assert_eq!(
                    files.iter().map(|f| f.path.capacity()).sum::<usize>(),
                    capacities[0] + capacities[1]
                );
            }
            // Cancellation releases either partially moved listing state or
            // transferred manifest state through the actual machine lifecycle.
            let manifest = state.manifest.as_ref().map(Arc::downgrade);
            let mut machine = TaskMachine::new(id, state, limits).unwrap();
            let fixed = size_of::<TaskMachine<SnapshotState>>();
            assert!(machine.accounting().usage(Resource::TaskStateBytes).live() > fixed);
            assert!(matches!(
                machine.cancel(CancelReason::Caller),
                CancelDisposition::Cancelled(None)
            ));
            // The still-owned machine itself remains accounted after semantic
            // cleanup; no dynamic path or manifest owner may remain live.
            assert_eq!(
                machine.accounting().usage(Resource::TaskStateBytes).live(),
                fixed
            );
            assert!(
                machine
                    .accounting()
                    .usage(Resource::TaskStateBytes)
                    .peak_live()
                    > fixed
            );
            if let Some(manifest) = manifest {
                assert!(manifest.upgrade().is_none());
            }
        }
    }

    #[test]
    fn manifest_and_producer_work_is_prepaid_before_ownership_transfer() {
        let root = "memory:///table/_delta_log/";
        let profile = TaskLimits::qualification();
        let required = JsonPlanBudget::producer_work(64, root.len(), &profile).unwrap()
            + super::super::json_producer_shape::manifest_work(64, root.len()).unwrap();
        for available in [required - 1, required] {
            let limits = profile.with_limit(Resource::WorkUnits, available);
            let id = TaskId::allocate().unwrap();
            let evaluation = EvaluationKey::allocate(id.get()).unwrap();
            let mut state = state(limits, evaluation);
            state.log_root = root.into();
            state.files = (0..64)
                .map(|version| FileDescriptor {
                    path: format!("{root}{version:020}.json"),
                    size: 1,
                    modification_time: 0,
                    identity: ObjectIdentity::new([version as u8; 32]),
                })
                .collect();
            state.file_path_capacity = state.files.iter().map(|f| f.path.capacity()).sum();
            let accounting = TaskAccounting::new(limits);
            let result = state.finish_discovery(&accounting, cpu());
            if available < required {
                assert_resource(&result.unwrap_err(), Resource::WorkUnits);
                assert_eq!(
                    state.files.len(),
                    64,
                    "manifest must not take owners before admission"
                );
                assert!(state.manifest.is_none());
                assert!(state.plan.is_none());
                assert_eq!(accounting.usage(Resource::WorkUnits).consumed(), 0);
            } else {
                result.unwrap();
                assert!(state.files.is_empty());
                assert_eq!(state.file_path_capacity, 0);
                assert_eq!(state.manifest.as_ref().unwrap().files().len(), 64);
                assert!(state.plan.is_some());
                assert_eq!(accounting.usage(Resource::WorkUnits).consumed(), required);
            }
        }
    }

    #[test]
    fn json_task_root_owners_cover_canonical_join_and_boundary() {
        for input in [
            "memory:///table",
            "https://example.com/t%20able/",
            "file:///C:/table",
            "https://xn--bcher-kva.example/table",
        ] {
            let root = Url::parse(input).unwrap();
            let id = TaskId::allocate().unwrap();
            let evaluation = EvaluationKey::allocate(id.get()).unwrap();
            let limits = TaskLimits::qualification().with_limit(Resource::LogDescriptors, 1);
            let required = root_owner_peak(root.as_str().len()).unwrap()
                + size_of::<FileDescriptor>()
                + size_of::<SnapshotLoadTask>()
                + size_of::<SnapshotState>();
            let failure = SnapshotLoadTask::try_new(
                id,
                evaluation,
                &root,
                None,
                limits.with_limit(Resource::TaskStateBytes, required - 1),
            )
            .err()
            .unwrap();
            assert_resource(&failure, Resource::TaskStateBytes);
            let id = TaskId::allocate().unwrap();
            let evaluation = EvaluationKey::allocate(id.get()).unwrap();
            let task = SnapshotLoadTask::try_new(
                id,
                evaluation,
                &root,
                None,
                limits.with_limit(Resource::TaskStateBytes, required),
            )
            .unwrap();
            drop(task);
            let id = TaskId::allocate().unwrap();
            let evaluation = EvaluationKey::allocate(id.get()).unwrap();
            let mut task = SnapshotLoadTask::try_new(id, evaluation, &root, None, limits).unwrap();
            let TaskStep::Execute(request) = task.start(cpu()).unwrap() else {
                panic!("listing expected")
            };
            let TaskRequestV1::List { root: actual, .. } = request.operation else {
                panic!("listing expected")
            };
            let mut expected = root.clone();
            if !expected.path().ends_with('/') {
                expected.path_segments_mut().unwrap().push("");
            }
            assert_eq!(actual, expected.join("_delta_log/").unwrap().as_str());
            task.cancel(CancelReason::Caller);
        }
        assert_eq!(root_owner_peak(usize::MAX), None);
    }
    #[test]
    fn json_task_plan_and_constructor_exhaustion_preserve_category_and_cleanup() {
        let id = TaskId::allocate().unwrap();
        let evaluation = EvaluationKey::allocate(id.get()).unwrap();
        let root = Url::parse("memory:///table/").unwrap();
        let failure = SnapshotLoadTask::try_new(
            id,
            evaluation,
            &root,
            None,
            TaskLimits::qualification().with_limit(Resource::TaskStateBytes, 1),
        )
        .err()
        .unwrap();
        assert_resource(&failure, Resource::TaskStateBytes);
        let id = TaskId::allocate().unwrap();
        let evaluation = EvaluationKey::allocate(id.get()).unwrap();
        let mut task = SnapshotLoadTask::try_new(
            id,
            evaluation,
            &root,
            None,
            TaskLimits::qualification().with_limit(Resource::PlanNodes, 1),
        )
        .unwrap();
        let TaskStep::Execute(request) = task.start(cpu()).unwrap() else {
            panic!("listing expected")
        };
        let files = vec![FileDescriptor {
            path: "memory:///table/_delta_log/00000000000000000000.json".into(),
            size: 1,
            modification_time: 0,
            identity: ObjectIdentity::new([1; 32]),
        }];
        let TaskStep::Failed(failure) = task
            .resume(
                request.key,
                Ok(TaskResponseV1::Listing {
                    files,
                    continuation: None,
                }),
                cpu(),
            )
            .unwrap()
        else {
            panic!("failure expected")
        };
        assert_resource(&failure, Resource::PlanNodes);
        assert_eq!(
            task.accounting().usage(Resource::TaskStateBytes).live(),
            size_of::<TaskMachine<SnapshotState>>()
        );
        assert_eq!(
            task.progress(cpu()).err(),
            Some(TaskProtocolError::Terminal)
        );
    }
    #[test]
    fn json_task_schema_preflight_categories_survive_visitor_and_cleanup() {
        for resource in [
            Resource::SchemaDepth,
            Resource::SchemaNodes,
            Resource::PartialJsonBytes,
        ] {
            let limits = TaskLimits::qualification().with_limit(resource, 1);
            rejected_page(
                limits,
                "{\"type\":\"struct\",\"fields\":[]}",
                1024,
                Some(resource),
            );
        }
        rejected_page(TaskLimits::qualification(), "{\"unclosed\":", 1024, None);
    }
    #[test]
    fn json_task_incoming_page_and_materialization_must_fit_simultaneously() {
        let bytes = 100_000;
        let materialization =
            super::super::json_visitor_allocation::pm_fixed_peak().unwrap() + 2 * "{}".len();
        let base = 1024 + size_of::<TaskMachine<SnapshotState>>();
        let page = bytes + size_of::<Box<dyn AccountedEngineData>>();
        let limit = base + materialization + page - 1;
        assert!(base + page < limit && base + materialization < limit);
        rejected_page(
            TaskLimits::qualification().with_limit(Resource::TaskStateBytes, limit),
            "{}",
            bytes,
            Some(Resource::TaskStateBytes),
        );
    }
    #[test]
    fn json_host_and_kernel_fixed_preflight_compose_at_exact_boundary() {
        struct EmptyPage {
            visits: Arc<AtomicUsize>,
        }
        impl crate::EngineData for EmptyPage {
            fn len(&self) -> usize {
                0
            }
            fn visit_rows(
                &self,
                columns: &[ColumnName],
                visitor: &mut dyn RowVisitor,
            ) -> crate::DeltaResult<()> {
                self.visits.fetch_add(1, Ordering::SeqCst);
                match columns.len() {
                    4 => visitor.visit(0, &[&() as &dyn GetData<'_>; 4]),
                    9 => visitor.visit(0, &[&() as &dyn GetData<'_>; 9]),
                    _ => unreachable!(),
                }
            }
            fn append_columns(
                &self,
                _: SchemaRef,
                _: Vec<ArrayData>,
            ) -> crate::DeltaResult<Box<dyn crate::EngineData>> {
                unreachable!()
            }
            fn apply_selection_vector(
                self: Box<Self>,
                _: Vec<bool>,
            ) -> crate::DeltaResult<Box<dyn crate::EngineData>> {
                unreachable!()
            }
            fn has_field(&self, _: &ColumnName) -> bool {
                true
            }
        }
        impl AccountedEngineData for EmptyPage {
            fn accounted_bytes(&self) -> Result<usize, ResourceExhausted> {
                Ok(0)
            }
        }
        let fixed = super::super::json_visitor_allocation::pm_fixed_peak().unwrap();
        let host = fixed * 2;
        let id = TaskId::allocate().unwrap();
        let evaluation = EvaluationKey::allocate(id.get()).unwrap();
        let mut state = state(TaskLimits::qualification(), evaluation);
        state.host_retained_bytes = host;
        let kernel =
            state.retained_bytes().unwrap() - host + size_of::<TaskMachine<SnapshotState>>();
        let exact = kernel + host + fixed;
        assert!(kernel + fixed < exact - 1 && host < exact - 1);
        for resource in [Resource::TaskStateBytes, Resource::MetadataAllocatedBytes] {
            for (limit, expected) in [(exact, true), (exact - 1, false)] {
                state.limits = TaskLimits::qualification().with_limit(resource, limit);
                let accounting = TaskAccounting::new(state.limits);
                let visits = Arc::new(AtomicUsize::new(0));
                // Even an empty batch runs both selected column-map/getter
                // visitors. No dynamic action strings need materialization.
                let batch = EmptyPage {
                    visits: visits.clone(),
                };
                let first = preflight_pm(&batch, &state.limits, |bytes| {
                    state.admit_extra(&accounting, bytes)
                });
                assert_eq!(first.is_ok(), expected);
                if !expected {
                    assert_resource(&first.unwrap_err(), resource);
                    assert_eq!(visits.load(Ordering::SeqCst), 0);
                }
                if expected {
                    assert_eq!(visits.load(Ordering::SeqCst), 2);
                }
                drop(batch);
            }
        }
    }
    fn rejected_page(limits: TaskLimits, schema: &str, bytes: usize, resource: Option<Resource>) {
        let id = TaskId::allocate().unwrap();
        let evaluation = EvaluationKey::allocate(id.get()).unwrap();
        let mut machine = TaskMachine::new(id, state(limits, evaluation), limits).unwrap();
        let TaskStep::Execute(request) = machine.start(cpu()).unwrap() else {
            panic!("evaluation expected")
        };
        let drops = Arc::new(AtomicUsize::new(0));
        let visits = Arc::new(AtomicUsize::new(0));
        let batch = Batch {
            schema: schema.into(),
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
                cpu(),
            )
            .unwrap()
        else {
            panic!("preflight must fail")
        };
        if let Some(resource) = resource {
            assert_resource(&failure, resource);
        } else {
            assert_eq!(failure.kind(), FailureKind::MalformedResponse);
        }
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(visits.load(Ordering::SeqCst), 2);
        assert_eq!(
            machine.accounting().usage(Resource::TaskStateBytes).live(),
            size_of::<TaskMachine<SnapshotState>>()
        );
        assert_eq!(
            machine.progress(cpu()).err(),
            Some(TaskProtocolError::Terminal)
        );
    }
}
