//! Immutable task-local input for JsonSource. Provider reads happen only in LogInput::load.

use std::fmt;
use std::future::Future;
use std::mem::size_of;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use delta_kernel::tasks::{
    LogIdentityManifest, OperationFailure, Resource, ResourceExhausted, TaskLimits,
};
use futures::stream::{self, BoxStream, StreamExt};
use object_store::path::Path;
use object_store::{
    Attributes, CopyOptions, GetOptions, GetResult, GetResultPayload, ListResult, MultipartUpload,
    ObjectMeta, ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
};

use crate::log_input::LogInput;

#[derive(Debug)]
struct LogObject {
    meta: ObjectMeta,
    bytes: Bytes,
}

/// Trace entries borrow identity/path through the immutable manifest's index, without cloning URLs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LogReadTrace {
    pub index: usize,
    pub bytes: usize,
}
#[derive(Debug)]
struct ReadState {
    trace: Vec<LogReadTrace>,
    requested: usize,
}

#[derive(Debug)]
pub(crate) struct AdmittedLogStore {
    manifest: Arc<LogIdentityManifest>,
    objects: Vec<LogObject>,
    reads: Mutex<ReadState>,
    limits: TaskLimits,
    pub retained_bytes: usize,
}

impl AdmittedLogStore {
    /// Charges all immutable byte-owner headers, paths and fixed-capacity trace slots before
    /// building the wrapper. `pulls` is the reached physical scans' total file-open envelope.
    pub fn try_new(
        input: LogInput,
        pulls: usize,
        limits: TaskLimits,
    ) -> Result<Self, OperationFailure> {
        let total = Self::owner_peak(&input, pulls, limits)?;
        if input.reads.len() != input.manifest.files().len() {
            return Err(OperationFailure::malformed_response());
        }
        let mut objects = Vec::with_capacity(input.reads.len());
        for (index, (file, read)) in input.manifest.files().iter().zip(input.reads).enumerate() {
            if file.identity != read.identity
                || read.offset != 0
                || !read.eof
                || read.bytes.len() as u64 != file.size
                || read.bytes.capacity() != read.bytes.len()
            {
                return Err(OperationFailure::malformed_response());
            }
            let location = Path::from_url_path(
                url_path(&file.path).ok_or_else(OperationFailure::malformed_response)?,
            )
            .map_err(|_| OperationFailure::malformed_response())?;
            // The manifest already proves one canonical immediate-child name per
            // ascending version. Distinct checked version suffixes prove uniqueness
            // even after URL path decoding; no pairwise duplicate search is needed.
            if canonical_commit_index(&location) != Some(index) {
                return Err(OperationFailure::malformed_response());
            }
            let time = chrono::DateTime::from_timestamp_millis(file.modification_time)
                .ok_or_else(OperationFailure::malformed_response)?;
            objects.push(LogObject {
                meta: ObjectMeta {
                    location,
                    size: file.size,
                    last_modified: time,
                    e_tag: None,
                    version: None,
                },
                bytes: Bytes::from_owner(read.bytes),
            });
        }
        Ok(Self {
            manifest: input.manifest,
            objects,
            reads: Mutex::new(ReadState {
                trace: Vec::with_capacity(pulls),
                requested: 0,
            }),
            limits,
            retained_bytes: total,
        })
    }

    /// Allocation-free immutable wrapper proof, also used when composing the
    /// host's simultaneous owners before constructing this wrapper.
    pub(crate) fn owner_peak(
        input: &LogInput,
        pulls: usize,
        limits: TaskLimits,
    ) -> Result<usize, OperationFailure> {
        let overflow = || ResourceExhausted {
            resource: Resource::TaskStateBytes,
            limit: limits.limit(Resource::TaskStateBytes),
            observed: usize::MAX,
        };
        if pulls > limits.limit(Resource::Requests) {
            return Err(ResourceExhausted {
                resource: Resource::Requests,
                limit: limits.limit(Resource::Requests),
                observed: pulls,
            }
            .into());
        }
        let trace = pulls
            .checked_mul(size_of::<LogReadTrace>())
            .ok_or_else(overflow)?;
        let slots = input
            .reads
            .len()
            .checked_mul(size_of::<LogObject>())
            .ok_or_else(overflow)?;
        // bytes::Bytes::from_owner(Vec<u8>) allocates Owned<Vec<u8>>: AtomicUsize + Vec.
        let headers = input
            .reads
            .len()
            .checked_mul(size_of::<usize>() + size_of::<Vec<u8>>())
            .ok_or_else(overflow)?;
        let paths = input
            .manifest
            .files()
            .iter()
            .try_fold(0usize, |n, f| n.checked_add(f.path.len()))
            .ok_or_else(overflow)?;
        // from_url_path first percent-decodes (<= URL bytes) then Path::parse owns a String.
        // Retain both during construction. Each get additionally clones one metadata path;
        // charge all allowed gets up front, including their stream/future containers.
        let max_path = input
            .manifest
            .files()
            .iter()
            .map(|f| f.path.len())
            .max()
            .unwrap_or(0);
        let response = max_path
            .checked_add(size_of::<GetResult>())
            .and_then(|n| n.checked_add(size_of::<ResourceExhausted>()))
            .and_then(|n| {
                n.checked_add(size_of::<
                    futures::future::Ready<object_store::Result<GetResult>>,
                >())
            })
            .and_then(|n| {
                n.checked_add(size_of::<
                    stream::Once<futures::future::Ready<object_store::Result<Bytes>>>,
                >())
            })
            .ok_or_else(overflow)?;
        let responses = pulls.checked_mul(response).ok_or_else(overflow)?;
        let total = [
            input.retained_bytes,
            size_of::<Self>(),
            trace,
            slots,
            headers,
            paths,
            paths,
            responses,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(overflow)?;
        if total > limits.limit(Resource::TaskStateBytes) {
            return Err(ResourceExhausted {
                resource: Resource::TaskStateBytes,
                limit: limits.limit(Resource::TaskStateBytes),
                observed: total,
            }
            .into());
        }
        Ok(total)
    }

    pub fn with_trace<T>(
        &self,
        visit: impl FnOnce(&[LogReadTrace]) -> T,
    ) -> object_store::Result<T> {
        let state = self.reads.lock().map_err(|_| rejected())?;
        Ok(visit(&state.trace))
    }
}

/// Canonical twenty-digit commit suffix selects one immutable manifest slot.
/// Fixed suffix access and checked decimal accumulation never scan the root or allocate.
fn canonical_commit_index(location: &Path) -> Option<usize> {
    let path = location.as_ref().as_bytes();
    let start = path.len().checked_sub(25)?;
    if start == 0 || path[start - 1] != b'/' || &path[start + 20..] != b".json" {
        return None;
    }
    path[start..start + 20]
        .iter()
        .try_fold(0usize, |index, digit| {
            if !digit.is_ascii_digit() {
                return None;
            }
            index
                .checked_mul(10)?
                .checked_add(usize::from(*digit - b'0'))
        })
}

impl AdmittedLogStore {
    fn read_cached(&self, location: &Path, options: GetOptions) -> object_store::Result<GetResult> {
        if options.range.is_some()
            || options.head
            || options.if_match.is_some()
            || options.if_none_match.is_some()
            || options.if_modified_since.is_some()
            || options.if_unmodified_since.is_some()
            || options.version.is_some()
            || !options.extensions.is_empty()
        {
            return Err(rejected());
        }
        let index = canonical_commit_index(location).ok_or_else(rejected)?;
        let object = self.objects.get(index).ok_or_else(rejected)?;
        #[cfg(test)]
        tests::note_path_comparison();
        if &object.meta.location != location {
            return Err(rejected());
        }
        // All provider preconditions and complete-owner checks have already run. Revalidate
        // immutable provenance before charging a pull; no latest-object retry exists here.
        if object.bytes.len() as u64 != self.manifest.files()[index].size {
            return Err(rejected());
        }
        {
            let mut state = self.reads.lock().map_err(|_| rejected())?;
            if state.trace.len() == state.trace.capacity() {
                return Err(resource_error(
                    Resource::Requests,
                    state.trace.capacity(),
                    state.trace.len().saturating_add(1),
                ));
            }
            let requested = state
                .requested
                .checked_add(object.bytes.len())
                .ok_or_else(|| {
                    resource_error(
                        Resource::RequestedReadBytes,
                        self.limits.limit(Resource::RequestedReadBytes),
                        usize::MAX,
                    )
                })?;
            for resource in [Resource::RequestedReadBytes, Resource::InputBytes] {
                if requested > self.limits.limit(resource) {
                    return Err(resource_error(
                        resource,
                        self.limits.limit(resource),
                        requested,
                    ));
                }
            }
            state.requested = requested;
            state.trace.push(LogReadTrace {
                index,
                bytes: object.bytes.len(),
            });
        }
        Ok(GetResult {
            payload: GetResultPayload::Stream(
                stream::once(futures::future::ready(Ok(object.bytes.clone()))).boxed(),
            ),
            meta: object.meta.clone(),
            range: 0..object.meta.size,
            attributes: Attributes::default(),
            extensions: Default::default(),
        })
    }
}

// Manifest URLs have already passed canonical-root/identity validation. Borrow the path rather
// than allocating another Url parser output. Percent decoding remains Path's responsibility.
fn url_path(url: &str) -> Option<&str> {
    let scheme = url.find("://")?.checked_add(3)?;
    let path = url[scheme..].find('/')?.checked_add(scheme)?;
    Some(&url[path..])
}

impl fmt::Display for AdmittedLogStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("admitted JSON task input")
    }
}

#[async_trait]
impl ObjectStore for AdmittedLogStore {
    // The admitted bytes are already resident: make the future an explicitly sized Ready,
    // instead of async_trait's opaque capture frame. try_new charges this exact boxed type.
    fn get_opts<'life0, 'life1, 'async_trait>(
        &'life0 self,
        location: &'life1 Path,
        options: GetOptions,
    ) -> Pin<Box<dyn Future<Output = object_store::Result<GetResult>> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(futures::future::ready(self.read_cached(location, options)))
    }
    async fn put_opts(
        &self,
        _: &Path,
        _: PutPayload,
        _: PutOptions,
    ) -> object_store::Result<PutResult> {
        Err(rejected())
    }
    async fn put_multipart_opts(
        &self,
        _: &Path,
        _: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        Err(rejected())
    }
    fn delete_stream(
        &self,
        _: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        stream::once(futures::future::ready(Err(rejected()))).boxed()
    }
    fn list(&self, _: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        stream::once(futures::future::ready(Err(rejected()))).boxed()
    }
    async fn list_with_delimiter(&self, _: Option<&Path>) -> object_store::Result<ListResult> {
        Err(rejected())
    }
    async fn copy_opts(&self, _: &Path, _: &Path, _: CopyOptions) -> object_store::Result<()> {
        Err(rejected())
    }
}

#[derive(Debug)]
struct Rejected;
impl fmt::Display for Rejected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("JSON task store requires an admitted whole-file read")
    }
}
impl std::error::Error for Rejected {}
fn rejected() -> object_store::Error {
    object_store::Error::Generic {
        store: "admitted-json",
        source: Box::new(Rejected),
    }
}
fn resource_error(resource: Resource, limit: usize, observed: usize) -> object_store::Error {
    object_store::Error::Generic {
        store: "admitted-json",
        source: Box::new(ResourceExhausted {
            resource,
            limit,
            observed,
        }),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    std::thread_local! {
        static PATH_COMPARISONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }
    pub(crate) fn note_path_comparison() {
        PATH_COMPARISONS.with(|count| count.set(count.get() + 1));
    }
    pub(crate) fn take_path_comparisons() -> usize {
        PATH_COMPARISONS.with(|count| count.replace(0))
    }

    use super::*;
    use crate::json_framing;
    use delta_kernel::tasks::{AdmittedRead, FileDescriptor, ObjectIdentity};
    use object_store::ObjectStoreExt;

    fn input() -> LogInput {
        input_with_time(0)
    }

    fn input_with_time(modification_time: i64) -> LogInput {
        let limits = TaskLimits::qualification();
        let bytes = b"{\"id\":1}\n{\"id\":2}\n".to_vec();
        let identity = ObjectIdentity::new([7; 32]);
        let plan = admitted_plan(vec![FileDescriptor {
            path: "memory:///table/_delta_log/00000000000000000000.json".into(),
            size: bytes.len() as u64,
            modification_time,
            identity,
        }]);
        let manifest = plan.log_identity_manifest().unwrap().clone();
        LogInput {
            framing: json_framing::preflight(&bytes, limits).unwrap(),
            retained_bytes: manifest.retained_bytes()
                + bytes.capacity()
                + size_of::<AdmittedRead>(),
            manifest,
            reads: vec![AdmittedRead {
                bytes,
                identity,
                offset: 0,
                eof: true,
            }],
        }
    }

    pub(crate) fn admitted_plan(files: Vec<FileDescriptor>) -> delta_kernel::tasks::AdmittedPlan {
        admitted_plan_at(&url::Url::parse("memory:///table/").unwrap(), files)
    }

    pub(crate) fn admitted_plan_at(
        root: &url::Url,
        files: Vec<FileDescriptor>,
    ) -> delta_kernel::tasks::AdmittedPlan {
        let limits = TaskLimits::qualification();
        // Obtain provenance from the real closed task producer; no public arbitrary-manifest
        // constructor is added for the test or for the host.
        use delta_kernel::tasks::{
            CpuSlice, EvaluationKey, OperationTask, SnapshotLoadTask, TaskId, TaskRequestV1,
            TaskResponseV1, TaskStep,
        };
        let id = TaskId::allocate().unwrap();
        let evaluation = EvaluationKey::allocate(id.get()).unwrap();
        let cpu = CpuSlice::new(1024, 1 << 20, 256).unwrap();
        let mut task = SnapshotLoadTask::try_new(id, evaluation, root, None, limits).unwrap();
        let TaskStep::Execute(list) = task.start(cpu).unwrap() else {
            panic!("expected listing")
        };
        let mut step = task
            .resume(
                list.key,
                Ok(TaskResponseV1::Listing {
                    files,
                    continuation: None,
                }),
                cpu,
            )
            .unwrap();
        while matches!(step, TaskStep::Yield) {
            step = task.progress(cpu).unwrap();
        }
        let TaskStep::Execute(request) = step else {
            panic!("expected evaluation")
        };
        let TaskRequestV1::EvaluationStart { plan, .. } = request.operation else {
            panic!("expected admitted plan")
        };
        plan
    }

    #[test]
    fn observed_timestamp_extremes_are_typed_failures() {
        for timestamp in [i64::MIN, i64::MAX] {
            let error = AdmittedLogStore::try_new(
                input_with_time(timestamp),
                1,
                TaskLimits::qualification(),
            )
            .unwrap_err();
            assert_eq!(
                error.kind(),
                delta_kernel::tasks::FailureKind::MalformedResponse
            );
        }
        for timestamp in [-1001, -1, 0, 1, 1001] {
            let store = AdmittedLogStore::try_new(
                input_with_time(timestamp),
                1,
                TaskLimits::qualification(),
            )
            .unwrap();
            assert_eq!(
                store.objects[0].meta.last_modified.timestamp_millis(),
                timestamp
            );
        }
    }

    #[tokio::test]
    async fn immutable_whole_file_only_with_bounded_trace() {
        let store = AdmittedLogStore::try_new(input(), 1, TaskLimits::qualification()).unwrap();
        let path = Path::from("table/_delta_log/00000000000000000000.json");
        assert!(store.get(&Path::from("outside.json")).await.is_err());
        assert!(store.head(&path).await.is_err());
        assert!(store.get_range(&path, 0..1).await.is_err());
        assert_eq!(store.with_trace(|t| t.len()).unwrap(), 0);
        let bytes = store.get(&path).await.unwrap().bytes().await.unwrap();
        assert_eq!(&bytes[..], b"{\"id\":1}\n{\"id\":2}\n");
        assert_eq!(
            store
                .with_trace(|t| (t.len(), t[0].index, t[0].bytes))
                .unwrap(),
            (1, 0, bytes.len())
        );
        let error = store.get(&path).await.unwrap_err();
        let object_store::Error::Generic { source, .. } = error else {
            panic!("expected typed resource error")
        };
        assert!(source
            .downcast_ref::<ResourceExhausted>()
            .is_some_and(|e| e.resource == Resource::Requests));
        drop(store);
        // JsonSource may retain a chunk after the wrapper is dropped; the exact admitted owner
        // survives by Bytes refcount, without a borrowed backing or a replacement store read.
        assert_eq!(&bytes[..], b"{\"id\":1}\n{\"id\":2}\n");
    }

    #[test]
    fn wrapper_admits_before_owner_conversion_and_rejects_identity_mismatch() {
        let limits = TaskLimits::qualification();
        let bound = AdmittedLogStore::try_new(input(), 1, limits)
            .unwrap()
            .retained_bytes;
        AdmittedLogStore::try_new(
            input(),
            1,
            limits.with_limit(Resource::TaskStateBytes, bound),
        )
        .unwrap();
        assert!(AdmittedLogStore::try_new(
            input(),
            1,
            limits.with_limit(Resource::TaskStateBytes, bound - 1)
        )
        .is_err());
        let mut replaced = input();
        replaced.reads[0].identity = ObjectIdentity::new([8; 32]);
        assert!(AdmittedLogStore::try_new(replaced, 1, limits).is_err());
    }

    #[tokio::test]
    async fn json_source_executes_admitted_owners_at_batch_one() {
        use datafusion::arrow::datatypes::{DataType, Field, Schema};
        use datafusion::execution::context::SessionContext;
        use datafusion::execution::object_store::ObjectStoreUrl;
        use datafusion::physical_plan::collect;
        use datafusion_datasource::file_groups::FileGroup;
        use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
        use datafusion_datasource::source::DataSourceExec;
        use datafusion_datasource::{PartitionedFile, TableSchema};
        use datafusion_datasource_json::source::JsonSource;

        let input = input();
        let size = input.reads[0].bytes.len();
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        crate::json_arrays::DecoderEnvelope::preflight(
            &schema,
            input.framing,
            TaskLimits::qualification(),
        )
        .unwrap();
        let store =
            Arc::new(AdmittedLogStore::try_new(input, 1, TaskLimits::qualification()).unwrap());
        let source =
            Arc::new(JsonSource::new(TableSchema::from(schema)).with_object_store(store.clone()));
        let config =
            FileScanConfigBuilder::new(ObjectStoreUrl::parse("memory://").unwrap(), source)
                .with_file_group(FileGroup::new(vec![PartitionedFile::new(
                    "table/_delta_log/00000000000000000000.json",
                    size as u64,
                )]))
                .with_batch_size(Some(1))
                .build();
        let session = SessionContext::new();
        // DataSourceExec still resolves the caller's registered store before JsonSource's
        // task-local injection. This empty store cannot satisfy the read accidentally.
        session.runtime_env().register_object_store(
            &url::Url::parse("memory:///").unwrap(),
            Arc::new(object_store::memory::InMemory::new()),
        );
        let batches = collect(DataSourceExec::from_data_source(config), session.task_ctx())
            .await
            .unwrap();
        assert_eq!(batches.len(), 2);
        assert!(batches.iter().all(|b| b.num_rows() == 1));
        datafusion::assert_batches_eq!(
            ["+----+", "| id |", "+----+", "| 1  |", "| 2  |", "+----+"],
            &batches
        );
        assert_eq!(store.with_trace(|t| t.len()).unwrap(), 1);
    }
}

#[cfg(test)]
mod fixture_plan_tests {
    use super::*;
    use crate::json_framing::{self, JsonFraming};
    use datafusion::arrow::array::Int64Array;
    use datafusion::prelude::SessionContext;
    use delta_kernel::tasks::{AdmittedRead, FileDescriptor, ObjectIdentity};

    // This exercises the real closed producer and ordinary DataFusion lowering semantics.
    // It does not stand in for host pre-allocation or the final adapter qualification gates.
    #[tokio::test]
    async fn closed_protocol_metadata_plan_reads_the_frozen_delta_history() {
        const V0: &[u8] = include_bytes!("../tests/data/phase_d/00000000000000000000.json");
        const V1: &[u8] = include_bytes!("../tests/data/phase_d/00000000000000000001.json");
        let limits = TaskLimits::qualification();
        for logs in [&[V0][..], &[V0, V1][..]] {
            let files = logs
                .iter()
                .enumerate()
                .map(|(index, bytes)| FileDescriptor {
                    path: format!("memory:///table/_delta_log/{index:020}.json"),
                    size: bytes.len() as u64,
                    modification_time: 0,
                    identity: ObjectIdentity::new([index as u8; 32]),
                })
                .collect();
            let plan = super::tests::admitted_plan(files);
            let manifest = plan.log_identity_manifest().unwrap().clone();
            let mut framing = JsonFraming::default();
            let reads: Vec<_> = logs
                .iter()
                .enumerate()
                .map(|(index, bytes)| {
                    let observed = json_framing::preflight(bytes, limits).unwrap();
                    framing.records += observed.records;
                    framing.tokens += observed.tokens;
                    framing.max_depth = framing.max_depth.max(observed.max_depth);
                    framing.max_record_bytes =
                        framing.max_record_bytes.max(observed.max_record_bytes);
                    AdmittedRead {
                        bytes: bytes.to_vec(),
                        identity: ObjectIdentity::new([index as u8; 32]),
                        offset: 0,
                        eof: true,
                    }
                })
                .collect();
            let input = LogInput {
                retained_bytes: manifest.retained_bytes()
                    + plan.retained_bytes()
                    + reads.iter().map(|r| r.bytes.capacity()).sum::<usize>()
                    + reads.capacity() * size_of::<AdmittedRead>(),
                manifest: manifest.clone(),
                reads,
                framing,
            };
            let store = Arc::new(AdmittedLogStore::try_new(input, logs.len(), limits).unwrap());
            let injected: Arc<dyn ObjectStore> = store.clone();
            let logical =
                crate::plan::lower_plan(plan.plan(), Some((&manifest, &injected))).unwrap();
            let session = SessionContext::new();
            session.runtime_env().register_object_store(
                &url::Url::parse("memory:///").unwrap(),
                Arc::new(object_store::memory::InMemory::new()),
            );
            let state = crate::metadata_session::new(&session).unwrap();
            let physical = state.create_physical_plan(&logical).await.unwrap();
            let batches = datafusion::physical_plan::collect(physical, state.task_ctx())
                .await
                .unwrap();
            assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
            let batch = batches.iter().find(|b| b.num_rows() == 1).unwrap();
            for name in ["protocol_version", "metadata_version"] {
                let values = batch
                    .column_by_name(name)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                assert_eq!(values.value(0), 0);
            }
            assert_eq!(store.with_trace(|trace| trace.len()).unwrap(), logs.len());
        }
    }
}
