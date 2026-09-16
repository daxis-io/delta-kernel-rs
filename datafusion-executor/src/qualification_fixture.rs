//! Frozen finite fixture shared by native and browser qualification.
//! Caller-owned fixture state is constructed before opening a task. Every
//! provider response and future is admitted before its allocation.
use std::mem::size_of;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Poll;

use async_trait::async_trait;
use delta_kernel::tasks::*;
use futures::stream::BoxStream;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult,
};

use crate::log_storage::{AdmittedJsonLogStorage, JsonLogReadLimits, LogStorageFuture};

pub const ROOT: &str = "memory:///table/_delta_log/";
pub const V0: &[u8] = include_bytes!("../tests/data/phase_d/00000000000000000000.json");
pub const V1: &[u8] = include_bytes!("../tests/data/phase_d/00000000000000000001.json");
const UNSUPPORTED: &[u8] =
    include_bytes!("../tests/data/phase_d/unsupported-features-00000000000000000000.json");
const MALFORMED: &[u8] =
    include_bytes!("../tests/data/phase_d/malformed-00000000000000000001.json");
const SCHEMA_CHANGE: &[u8] =
    include_bytes!("../tests/data/phase_d/schema-change-00000000000000000001.json");
include!("qualification_fixture_identities.rs");

#[derive(Clone, Copy, Debug)]
pub enum LogVariant {
    Valid,
    MalformedLatest,
    UnsupportedFeatures,
    MetadataSchemaChange,
}
#[derive(Debug)]
struct Log {
    path: &'static str,
    bytes: &'static [u8],
    identity: ObjectIdentity,
}
#[derive(Debug)]
pub struct FixtureStore {
    ordinary: object_store::memory::InMemory,
    logs: [Log; 2],
    pub page_size: AtomicUsize,
    pub replace_reads: AtomicBool,
    pub missing_identity: AtomicBool,
    pub omit_version_zero: AtomicBool,
    pub lists: AtomicUsize,
    pub reads: [AtomicUsize; 2],
    pub ordinary_gets: AtomicUsize,
    pub ordinary_log_gets: AtomicUsize,
    pub ordinary_reads: Mutex<Vec<(String, Option<object_store::GetRange>)>>,
    pub pending: AtomicUsize,
    pub cancellations: AtomicUsize,
    pub allocations: AtomicUsize,
}
impl FixtureStore {
    pub fn new(variant: LogVariant) -> Self {
        let (zero, zero_id) = if matches!(variant, LogVariant::UnsupportedFeatures) {
            (UNSUPPORTED, UNSUPPORTED_IDENTITY)
        } else {
            (V0, V0_IDENTITY)
        };
        let (one, one_id) = if matches!(variant, LogVariant::MalformedLatest) {
            (MALFORMED, MALFORMED_IDENTITY)
        } else if matches!(variant, LogVariant::MetadataSchemaChange) {
            (SCHEMA_CHANGE, SCHEMA_CHANGE_IDENTITY)
        } else {
            (V1, V1_IDENTITY)
        };
        Self {
            ordinary: object_store::memory::InMemory::new(),
            logs: [
                Log {
                    path: "memory:///table/_delta_log/00000000000000000000.json",
                    bytes: zero,
                    identity: ObjectIdentity::new(zero_id),
                },
                Log {
                    path: "memory:///table/_delta_log/00000000000000000001.json",
                    bytes: one,
                    identity: ObjectIdentity::new(one_id),
                },
            ],
            page_size: AtomicUsize::new(1),
            replace_reads: AtomicBool::new(false),
            missing_identity: AtomicBool::new(false),
            omit_version_zero: AtomicBool::new(false),
            lists: AtomicUsize::new(0),
            reads: std::array::from_fn(|_| AtomicUsize::new(0)),
            ordinary_gets: AtomicUsize::new(0),
            ordinary_log_gets: AtomicUsize::new(0),
            ordinary_reads: Mutex::new(Vec::new()),
            pending: AtomicUsize::new(0),
            cancellations: AtomicUsize::new(0),
            allocations: AtomicUsize::new(0),
        }
    }
    fn list_shape(
        &self,
        root: &str,
        continuation: Option<&str>,
        entries: usize,
    ) -> Result<(usize, usize, bool), OperationFailure> {
        if root != ROOT || entries == 0 {
            return Err(OperationFailure::malformed_response());
        }
        let start = match continuation {
            None => usize::from(self.omit_version_zero.load(Ordering::SeqCst)),
            Some(path) if path == self.logs[0].path => 1,
            Some(path) if path == self.logs[1].path => 2,
            _ => return Err(OperationFailure::malformed_response()),
        };
        let count = (2 - start).min(entries);
        Ok((start, count, count == entries))
    }
    async fn list_unboxed(
        &self,
        start: usize,
        count: usize,
        more: bool,
    ) -> Result<AdmittedListingPage, OperationFailure> {
        self.lists.fetch_add(1, Ordering::SeqCst);
        let mut guard = Active::new(self);
        yield_once().await;
        self.allocations.fetch_add(1, Ordering::SeqCst);
        let files = self.logs[start..start + count]
            .iter()
            .map(|log| FileDescriptor {
                path: log.path.to_owned(),
                size: log.bytes.len() as u64,
                modification_time: 0,
                identity: log.identity,
            })
            .collect::<Vec<_>>();
        let continuation = more.then(|| self.logs[start + count - 1].path.to_owned());
        guard.complete = true;
        let binding = continuation.clone();
        Ok(AdmittedListingPage {
            files,
            continuation,
            binding,
        })
    }
    async fn read_unboxed(&self, index: usize) -> Result<AdmittedRead, OperationFailure> {
        self.reads[index].fetch_add(1, Ordering::SeqCst);
        let mut guard = Active::new(self);
        yield_once().await;
        // Observe replacement again after the asynchronous boundary. Never
        // return replacement bytes or silently refresh the admitted identity.
        if self.replace_reads.load(Ordering::SeqCst) {
            return Err(OperationFailure::malformed_response());
        }
        self.allocations.fetch_add(1, Ordering::SeqCst);
        let log = &self.logs[index];
        let bytes = log.bytes.to_vec(); // exact source length/capacity
        guard.complete = true;
        Ok(AdmittedRead {
            identity: log.identity,
            bytes,
            offset: 0,
            eof: true,
        })
    }
}
struct Active<'a> {
    store: &'a FixtureStore,
    complete: bool,
}
impl<'a> Active<'a> {
    fn new(store: &'a FixtureStore) -> Self {
        store.pending.fetch_add(1, Ordering::SeqCst);
        Self {
            store,
            complete: false,
        }
    }
}
impl Drop for Active<'_> {
    fn drop(&mut self) {
        self.store.pending.fetch_sub(1, Ordering::SeqCst);
        if !self.complete {
            self.store.cancellations.fetch_add(1, Ordering::SeqCst);
        }
    }
}
pub async fn yield_once() {
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
    .await
}
fn refusal<T: 'static>(
    resource: Resource,
    limit: usize,
    observed: usize,
) -> LogStorageFuture<'static, T> {
    Box::pin(futures::future::ready(Err(ResourceExhausted {
        resource,
        limit,
        observed,
    }
    .into())))
}
impl AdmittedJsonLogStorage for FixtureStore {
    fn list<'a>(
        &'a self,
        root: &'a str,
        continuation: Option<&'a str>,
        entries: usize,
        descriptor_bytes: usize,
        continuation_bytes: usize,
    ) -> LogStorageFuture<'a, AdmittedListingPage> {
        if self.missing_identity.load(Ordering::SeqCst) {
            const MESSAGE: &str = "immutable log object identity is unavailable";
            let needed = size_of::<
                futures::future::Ready<Result<AdmittedListingPage, OperationFailure>>,
            >() + size_of::<delta_kernel::Error>()
                + MESSAGE.len();
            if needed > descriptor_bytes {
                return refusal(Resource::ListingDescriptorBytes, descriptor_bytes, needed);
            }
            return Box::pin(futures::future::ready(Err(OperationFailure::new(
                FailureKind::Engine,
                delta_kernel::Error::Unsupported(MESSAGE.to_owned()),
            ))));
        }
        let (start, count, more) = match self.list_shape(root, continuation, entries) {
            Ok(shape) => shape,
            Err(error) => return Box::pin(futures::future::ready(Err(error))),
        };
        let future = self.list_unboxed(start, count, more);
        let needed = count * size_of::<FileDescriptor>()
            + self.logs[start..start + count]
                .iter()
                .map(|log| log.path.len())
                .sum::<usize>()
            + size_of_val(&future);
        if needed > descriptor_bytes {
            return refusal(Resource::ListingDescriptorBytes, descriptor_bytes, needed);
        }
        // Producer cursor plus the driver's independently owned binding copy.
        let cursor = if more {
            self.logs[start + count - 1].path.len() * 2
        } else {
            0
        };
        if cursor > continuation_bytes {
            return refusal(Resource::ContinuationBytes, continuation_bytes, cursor);
        }
        Box::pin(future)
    }
    fn read_log<'a>(
        &'a self,
        file: &'a FileDescriptor,
        limits: JsonLogReadLimits,
    ) -> LogStorageFuture<'a, AdmittedRead> {
        let Some(index) = self.logs.iter().position(|log| log.path == file.path) else {
            return Box::pin(futures::future::ready(Err(
                OperationFailure::malformed_response(),
            )));
        };
        let log = &self.logs[index];
        if file.identity != log.identity
            || file.size != log.bytes.len() as u64
            || file.modification_time != 0
        {
            return Box::pin(futures::future::ready(Err(
                OperationFailure::malformed_response(),
            )));
        }
        let future = self.read_unboxed(index);
        let response = log.bytes.len() + size_of_val(&future) + size_of::<AdmittedRead>();
        for (resource, bound, observed) in [
            (
                Resource::ReadPayloadBytes,
                limits.response_backing_bytes,
                response,
            ),
            (
                Resource::ListingDescriptorBytes,
                limits.identity_path_bytes,
                log.path.len(),
            ),
            (
                Resource::RequestedReadBytes,
                limits.remaining_requested_bytes,
                log.bytes.len(),
            ),
            (
                Resource::InputBytes,
                limits.decoder_input_bytes,
                log.bytes.len(),
            ),
            (
                Resource::TaskStateBytes,
                limits.retained_log_bytes,
                response,
            ),
        ] {
            if observed > bound {
                return refusal(resource, bound, observed);
            }
        }
        if limits.exact_bytes != log.bytes.len() {
            return Box::pin(futures::future::ready(Err(
                OperationFailure::malformed_response(),
            )));
        }
        Box::pin(future)
    }
}
impl std::fmt::Display for FixtureStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("phase-d frozen async fixture")
    }
}
#[async_trait]
impl ObjectStore for FixtureStore {
    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        self.ordinary_gets.fetch_add(1, Ordering::SeqCst);
        if location.as_ref().starts_with("table/_delta_log/") {
            self.ordinary_log_gets.fetch_add(1, Ordering::SeqCst);
        }
        self.ordinary_reads
            .lock()
            .unwrap()
            .push((location.to_string(), options.range.clone()));
        let mut guard = Active::new(self);
        yield_once().await;
        guard.complete = true;
        self.ordinary.get_opts(location, options).await
    }
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        options: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.ordinary.put_opts(location, payload, options).await
    }
    async fn put_multipart_opts(
        &self,
        location: &Path,
        options: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.ordinary.put_multipart_opts(location, options).await
    }
    fn delete_stream(
        &self,
        paths: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.ordinary.delete_stream(paths)
    }
    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        ObjectStore::list(&self.ordinary, prefix)
    }
    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        self.ordinary.list_with_delimiter(prefix).await
    }
    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.ordinary.copy_opts(from, to, options).await
    }
}

/// Installs both views of the same Arc into a caller-owned session configuration.
pub fn register(
    store: Arc<FixtureStore>,
    config: datafusion::execution::config::SessionConfig,
    runtime: &datafusion::execution::runtime_env::RuntimeEnv,
) -> datafusion::execution::config::SessionConfig {
    let origin = datafusion::execution::object_store::ObjectStoreUrl::parse("memory:///").unwrap();
    runtime.register_object_store(origin.as_ref(), store.clone());
    let mut registry = crate::log_storage::JsonLogStorageRegistry::default();
    registry.register(origin, store.clone(), store);
    config.with_extension(Arc::new(registry))
}
