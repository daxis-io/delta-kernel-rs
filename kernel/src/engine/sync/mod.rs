//! A simple, single-threaded [`Engine`] for prefetched or in-memory data.
//!
//! All I/O goes through an [`ObjectStore`]. On native targets, [`SyncEngine::new`] uses a
//! [`LocalFileSystem`] built lazily per URL. [`SyncEngine::new_with_store`] takes a supplied
//! store (for example an `InMemory` cache) and is the only constructor available on
//! `wasm32-unknown-unknown` and the Daxis browser stack's native test profile.
//!
//! On native targets, async object-store calls are driven via [`futures::executor::block_on`].
//! On `wasm32-unknown-unknown`, they are polled exactly once and must complete immediately.
//! This prevents a pending future from parking the browser's only thread. Cloud-backed stores
//! are therefore not supported here: browser callers must asynchronously prefetch data into
//! an immediately-ready cache-backed store before invoking the kernel's synchronous handlers.
//!
//! [`DefaultEngine`]: crate::engine::default::DefaultEngine
//! [`LocalFileSystem`]: crate::object_store::local::LocalFileSystem

use std::future::Future;
use std::sync::Arc;

use bytes::Bytes;
#[cfg(any(test, all(target_arch = "wasm32", target_os = "unknown")))]
use futures::FutureExt as _;
use itertools::Itertools;
use tracing::debug;
use url::Url;

use super::arrow_expression::ArrowEvaluationHandler;
use crate::engine::arrow_data::ArrowEngineData;
#[cfg(all(
    not(all(target_arch = "wasm32", target_os = "unknown")),
    any(
        not(feature = "daxis-browser-stack"),
        feature = "daxis-native-default-engine"
    )
))]
use crate::object_store::local::LocalFileSystem;
use crate::object_store::path::Path;
use crate::object_store::DynObjectStore;
// `ObjectStoreExt` is needed for `store.get()` etc. in arrow-58 mode where these methods moved
// off the `ObjectStore` trait. In arrow-57 mode the compat shim makes the import a no-op, so
// silence the resulting unused-import warning.
#[allow(unused_imports)]
use crate::object_store::ObjectStoreExt as _;
use crate::{
    DeltaResult, Engine, Error, EvaluationHandler, FileDataReadResultIterator, FileMeta,
    JsonHandler, ParquetHandler, PredicateRef, SchemaRef, StorageHandler,
};

pub(crate) mod json;
mod parquet;
mod storage;

/// A single-threaded implementation of [`Engine`]. See the module docs for supported stores.
pub struct SyncEngine {
    storage_handler: Arc<storage::SyncStorageHandler>,
    json_handler: Arc<json::SyncJsonHandler>,
    parquet_handler: Arc<parquet::SyncParquetHandler>,
    evaluation_handler: Arc<ArrowEvaluationHandler>,
}

impl SyncEngine {
    /// Create a SyncEngine that reads from the local filesystem via [`LocalFileSystem`].
    #[cfg(all(
        not(all(target_arch = "wasm32", target_os = "unknown")),
        any(
            not(feature = "daxis-browser-stack"),
            feature = "daxis-native-default-engine"
        )
    ))]
    pub fn new() -> Self {
        Self::new_inner(None)
    }

    /// Create a SyncEngine backed by `store`. See the module docs for the stricter
    /// immediately-ready store contract on `wasm32-unknown-unknown`.
    pub fn new_with_store(store: Arc<DynObjectStore>) -> Self {
        Self::new_inner(Some(store))
    }

    fn new_inner(store: Option<Arc<DynObjectStore>>) -> Self {
        SyncEngine {
            storage_handler: Arc::new(storage::SyncStorageHandler::new(store.clone())),
            json_handler: Arc::new(json::SyncJsonHandler::new(store.clone())),
            parquet_handler: Arc::new(parquet::SyncParquetHandler::new(store)),
            evaluation_handler: Arc::new(ArrowEvaluationHandler {}),
        }
    }
}

impl Engine for SyncEngine {
    fn evaluation_handler(&self) -> Arc<dyn EvaluationHandler> {
        self.evaluation_handler.clone()
    }

    fn storage_handler(&self) -> Arc<dyn StorageHandler> {
        self.storage_handler.clone()
    }

    fn parquet_handler(&self) -> Arc<dyn ParquetHandler> {
        self.parquet_handler.clone()
    }

    fn json_handler(&self) -> Arc<dyn JsonHandler> {
        self.json_handler.clone()
    }
}

/// Resolve the store, base URL, and path for `url`.
///
/// When `default_store` is `Some`, the user-provided store is returned with `url`'s decoded
/// path. The base URL is `url` with its path replaced by `/`, suitable for re-joining
/// listed [`Path`]s back into URLs.
///
/// When `default_store` is `None`, only `file://` URLs are accepted: a [`LocalFileSystem`]
/// is constructed rooted at the URL's resolved directory (the URL itself if it ends with
/// `/`, otherwise the parent), and the returned path is the simple filename (or empty for
/// directory URLs). Rooting at the parent avoids a url-crate quirk where path segments
/// containing reserved-looking characters (e.g. `:` in Windows drive letters or `~` in 8.3
/// short names like `RUNNER~1`) get URL-encoded by `path_segments_mut.extend(...)` but not
/// decoded back by `to_file_path`. By keeping such characters inside the store's base URL
/// (which is built once from a `PathBuf` and not re-processed), only simple filenames flow
/// through the broken round-trip.
pub(super) fn resolve_scope(
    default_store: Option<&Arc<DynObjectStore>>,
    url: &Url,
) -> DeltaResult<(Arc<DynObjectStore>, Url, Path)> {
    if let Some(store) = default_store {
        let mut base_url = url.clone();
        base_url.set_path("/");
        let path = Path::from_url_path(url.path())?;
        return Ok((store.clone(), base_url, path));
    }

    #[cfg(any(
        all(target_arch = "wasm32", target_os = "unknown"),
        all(
            not(all(target_arch = "wasm32", target_os = "unknown")),
            feature = "daxis-browser-stack",
            not(feature = "daxis-native-default-engine")
        )
    ))]
    {
        return Err(Error::generic(format!(
            "SyncEngine in the browser profile requires an explicit prefetched store for {url}"
        )));
    }

    #[cfg(all(
        not(all(target_arch = "wasm32", target_os = "unknown")),
        any(
            not(feature = "daxis-browser-stack"),
            feature = "daxis-native-default-engine"
        )
    ))]
    {
        if url.scheme() != "file" {
            return Err(Error::generic(format!(
                "SyncEngine without an explicit store can only access file:// URLs, got: {url}"
            )));
        }
        let file_path = url
            .to_file_path()
            .map_err(|()| Error::generic(format!("Invalid file URL: {url}")))?;
        // Use the deepest existing ancestor of the URL's directory as the store prefix:
        // `LocalFileSystem::new_with_prefix` canonicalizes its argument (which requires the path
        // to exist), and operating on a non-existent path is a valid case for `list_from` on a
        // not-yet-written `_delta_log/` directory or `head` on a missing file.
        let target_dir = if url.path().ends_with('/') {
            file_path.clone()
        } else {
            file_path
                .parent()
                .ok_or_else(|| Error::generic(format!("File URL has no parent: {url}")))?
                .to_path_buf()
        };
        let mut prefix = target_dir.as_path();
        while !prefix.exists() {
            prefix = prefix.parent().ok_or_else(|| {
                Error::generic(format!("No existing ancestor for {target_dir:?}"))
            })?;
        }
        let prefix = prefix.to_path_buf();
        let relative = file_path
            .strip_prefix(&prefix)
            .map_err(|e| Error::generic(format!("Failed to strip prefix: {e}")))?;
        let path = Path::from_iter(relative.components().filter_map(|c| match c {
            std::path::Component::Normal(s) => s.to_str().map(String::from),
            _ => None,
        }));
        let base_url = Url::from_directory_path(&prefix)
            .map_err(|()| Error::generic(format!("Could not URL-encode prefix {prefix:?}")))?;
        let store: Arc<DynObjectStore> = Arc::new(LocalFileSystem::new_with_prefix(&prefix)?);
        Ok((store, base_url, path))
    }
}

/// Fetch the contents of a file via [`resolve_scope`] and return them as bytes.
pub(super) fn get_bytes(
    default_store: Option<&Arc<DynObjectStore>>,
    location: &Url,
) -> DeltaResult<Bytes> {
    let (store, _, path) = resolve_scope(default_store, location)?;
    let get_result = drive_sync(store.get(&path), "get prefetched object")??;
    Ok(drive_sync(
        get_result.bytes(),
        "read prefetched object body",
    )??)
}

/// Write `data` to `location` via [`resolve_scope`].
///
/// For `file://` URLs the parent directory is created if missing (matching the local FS
/// behavior callers expect; `LocalFileSystem::put` itself does not create parents). When
/// `overwrite` is false, an existing file at `location` produces [`Error::FileAlreadyExists`].
pub(super) fn put_bytes(
    default_store: Option<&Arc<DynObjectStore>>,
    location: &Url,
    data: Bytes,
    overwrite: bool,
) -> DeltaResult<()> {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    if location.scheme() == "file" {
        if let Ok(file_path) = location.to_file_path() {
            if let Some(parent) = file_path.parent() {
                if !parent.exists() {
                    std::fs::create_dir_all(parent)?;
                }
            }
        }
    }
    let (store, _, object_path) = resolve_scope(default_store, location)?;
    let opts = if overwrite {
        crate::object_store::PutOptions::default()
    } else {
        crate::object_store::PutOptions {
            mode: crate::object_store::PutMode::Create,
            ..Default::default()
        }
    };
    drive_sync(
        store.put_opts(&object_path, data.into(), opts),
        "write synchronous object",
    )?
    .map_err(|e| match e {
        crate::object_store::Error::AlreadyExists { .. } => {
            Error::FileAlreadyExists(location.to_string())
        }
        other => Error::generic(other.to_string()),
    })?;
    Ok(())
}

#[cfg(any(test, all(target_arch = "wasm32", target_os = "unknown")))]
fn poll_immediate<F: Future>(future: F, operation: &str) -> DeltaResult<F::Output> {
    future.now_or_never().ok_or_else(|| {
        Error::generic(format!(
            "SyncEngine {operation} yielded Pending on wasm32-unknown-unknown; \
             browser handlers require an asynchronously prefetched, immediately-ready store"
        ))
    })
}

fn drive_sync<F: Future>(future: F, operation: &str) -> DeltaResult<F::Output> {
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    {
        poll_immediate(future, operation)
    }
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    {
        let _ = operation;
        Ok(futures::executor::block_on(future))
    }
}

/// Read each file as bytes and feed it to `try_create_from_bytes` to produce data batches.
fn read_files<F, I>(
    store: Option<&Arc<DynObjectStore>>,
    files: &[FileMeta],
    schema: SchemaRef,
    predicate: Option<PredicateRef>,
    mut try_create_from_bytes: F,
) -> DeltaResult<FileDataReadResultIterator>
where
    I: Iterator<Item = DeltaResult<ArrowEngineData>> + Send + 'static,
    F: FnMut(Bytes, SchemaRef, Option<PredicateRef>, String) -> DeltaResult<I> + Send + 'static,
{
    debug!("Reading files: {files:#?} with schema {schema:#?} and predicate {predicate:#?}");
    if files.is_empty() {
        return Ok(Box::new(std::iter::empty()));
    }
    let files = files.to_vec();
    let store = store.cloned();
    let result = files
        .into_iter()
        .map(move |file| {
            let location_string = file.location.to_string();
            let bytes = get_bytes(store.as_ref(), &file.location)?;
            try_create_from_bytes(bytes, schema.clone(), predicate.clone(), location_string)
        })
        .flatten_ok()
        .map(|data| Ok(Box::new(ArrowEngineData::new(data??.into())) as _));
    Ok(Box::new(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::tests::test_arrow_engine;

    #[test]
    fn browser_immediate_driver_never_parks_the_host_thread() {
        assert_eq!(
            poll_immediate(futures::future::ready(42), "ready test").unwrap(),
            42
        );

        let error = poll_immediate(futures::future::pending::<()>(), "pending test")
            .expect_err("a pending prefetched-store future must fail instead of parking");
        assert!(error.to_string().contains("pending test"));
        assert!(error.to_string().contains("yielded Pending"));
    }

    #[test]
    fn test_sync_engine() {
        let tmp = tempfile::tempdir().unwrap();
        let url = Url::from_directory_path(tmp.path()).unwrap();
        let engine = SyncEngine::new();
        test_arrow_engine(&engine, &url);
    }

    #[test]
    fn test_sync_engine_with_store() {
        let store = Arc::new(crate::object_store::memory::InMemory::new());
        let engine = SyncEngine::new_with_store(store);
        let url = Url::parse("memory:///test/").unwrap();
        test_arrow_engine(&engine, &url);
    }
}
