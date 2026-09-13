//! Caller capabilities for producer-admitted JSON log discovery and reads.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use datafusion::execution::context::SessionContext;
use datafusion::execution::object_store::ObjectStoreUrl;
use delta_kernel::tasks::{
    AdmittedListingPage, AdmittedRead, FileDescriptor, OperationFailure, Resource,
    ResourceExhausted, TaskLimits,
};
use object_store::ObjectStore;

/// Local host future. Dropping it must release provider state without additional pulls.
pub type LogStorageFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, OperationFailure>> + 'a>>;

/// Producer-side guarantees unavailable from a generic object-store registration.
///
/// Implementations must admit metadata, parser scratch and full backing owners before
/// allocation. Discovery is strictly ordered by canonical immediate-child URL and uses the
/// last returned path as continuation, with definitive EOF. Descriptor identity is a
/// domain-separated collision-resistant digest of a bounded strong provider version token.
/// Missing identity is unsupported; size and modification time are not identity substitutes.
pub trait AdmittedJsonLogStorage: Send + Sync {
    /// Returns a bounded ordered page, including separately bounded binding/cursor copies.
    /// Rejects unsupported identity or producer bounds before fetching or allocating.
    fn list<'a>(
        &'a self,
        root: &'a str,
        continuation: Option<&'a str>,
        entries: usize,
        descriptor_bytes: usize,
        continuation_bytes: usize,
    ) -> LogStorageFuture<'a, AdmittedListingPage>;

    /// Reads the complete observed version with provider preconditions, never an unconditional
    /// GET after HEAD. Returns exact owner capacity, offset zero and EOF; replacement fails
    /// without retry. Includes provider headers/body/parser backing in the supplied limits.
    fn read_log<'a>(
        &'a self,
        file: &'a FileDescriptor,
        limits: JsonLogReadLimits,
    ) -> LogStorageFuture<'a, AdmittedRead>;
}

/// Immutable projection of existing task budgets for a complete uncompressed log read.
#[derive(Debug, Clone, Copy)]
pub struct JsonLogReadLimits {
    /// Exact file size, checked against every relevant allowance before I/O.
    pub exact_bytes: usize,
    /// Remaining cumulative requested bytes, after earlier reads.
    pub remaining_requested_bytes: usize,
    /// Provider response backing, including headers and parser scratch.
    pub response_backing_bytes: usize,
    /// Path and provider identity backing.
    pub identity_path_bytes: usize,
    /// Remaining admitted decoder input bytes.
    pub decoder_input_bytes: usize,
    /// Remaining simultaneous whole-log ownership allowance.
    pub retained_log_bytes: usize,
}

impl JsonLogReadLimits {
    /// Checks a complete file against existing limits and cumulative/live ownership.
    /// Returns typed resource exhaustion before the producer starts I/O.
    pub fn try_new(
        file: &FileDescriptor,
        limits: TaskLimits,
        requested: usize,
        input: usize,
        retained: usize,
    ) -> Result<Self, OperationFailure> {
        let exact = usize::try_from(file.size).map_err(|_| ResourceExhausted {
            resource: Resource::ReadPayloadBytes,
            limit: limits.limit(Resource::ReadPayloadBytes),
            observed: usize::MAX,
        })?;
        let remaining = |resource, used: usize| -> Result<usize, OperationFailure> {
            let limit = limits.limit(resource);
            let observed = used.checked_add(exact).ok_or(ResourceExhausted {
                resource,
                limit,
                observed: usize::MAX,
            })?;
            if observed > limit {
                return Err(ResourceExhausted {
                    resource,
                    limit,
                    observed,
                }
                .into());
            }
            Ok(limit - used)
        };
        let retained_log_bytes = remaining(Resource::TaskStateBytes, retained)?;
        let response_backing_bytes =
            remaining(Resource::ReadPayloadBytes, 0)?.min(retained_log_bytes);
        Ok(Self {
            exact_bytes: exact,
            remaining_requested_bytes: remaining(Resource::RequestedReadBytes, requested)?,
            response_backing_bytes,
            identity_path_bytes: limits
                .limit(Resource::ListingDescriptorBytes)
                .min(response_backing_bytes),
            decoder_input_bytes: remaining(Resource::InputBytes, input)?,
            retained_log_bytes,
        })
    }
}

/// Immutable session extension binding capabilities to ordinary registered store instances.
/// Install with `SessionConfig::with_extension` after registering each corresponding store.
#[derive(Default)]
pub struct JsonLogStorageRegistry {
    entries: HashMap<ObjectStoreUrl, LogStorageBinding>,
}

#[derive(Clone)]
struct LogStorageBinding {
    store: Arc<dyn ObjectStore>,
    capability: Arc<dyn AdmittedJsonLogStorage>,
}

/// Typed rejection before discovery for missing or replaced caller registrations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogStorageAdmissionError {
    /// The session has no matching admitted storage capability.
    MissingCapability,
    /// The ordinary registration is absent or no longer the same Arc.
    StoreMismatch,
}

impl std::fmt::Display for LogStorageAdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::MissingCapability => "missing admitted JSON log storage capability",
            Self::StoreMismatch => "admitted JSON log storage registration mismatch",
        })
    }
}

impl std::error::Error for LogStorageAdmissionError {}

impl JsonLogStorageRegistry {
    /// Binds a capability to the same Arc installed in the caller's ordinary registry.
    /// This only updates the extension being assembled; it performs no storage I/O.
    pub fn register(
        &mut self,
        url: ObjectStoreUrl,
        store: Arc<dyn ObjectStore>,
        capability: Arc<dyn AdmittedJsonLogStorage>,
    ) {
        self.entries
            .insert(url, LogStorageBinding { store, capability });
    }

    /// Resolves a capability and verifies the current ordinary registration before I/O.
    /// Does not mutate the session or object-store registry.
    pub fn resolve(
        session: &SessionContext,
        url: &ObjectStoreUrl,
    ) -> Result<Arc<dyn AdmittedJsonLogStorage>, LogStorageAdmissionError> {
        let state = session.state_ref();
        let registry = state
            .read()
            .config()
            .get_extension::<Self>()
            .ok_or(LogStorageAdmissionError::MissingCapability)?;
        let binding = registry
            .entries
            .get(url)
            .ok_or(LogStorageAdmissionError::MissingCapability)?;
        let current = session
            .runtime_env()
            .object_store(url)
            .map_err(|_| LogStorageAdmissionError::StoreMismatch)?;
        if !Arc::ptr_eq(&current, &binding.store) {
            return Err(LogStorageAdmissionError::StoreMismatch);
        }
        Ok(Arc::clone(&binding.capability))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::execution::config::SessionConfig;
    use delta_kernel::tasks::{FailureKind, ObjectIdentity};
    use object_store::memory::InMemory;

    struct NoIo;
    impl AdmittedJsonLogStorage for NoIo {
        fn list<'a>(
            &'a self,
            _: &'a str,
            _: Option<&'a str>,
            _: usize,
            _: usize,
            _: usize,
        ) -> LogStorageFuture<'a, AdmittedListingPage> {
            panic!("registration resolution must not do I/O")
        }
        fn read_log<'a>(
            &'a self,
            _: &'a FileDescriptor,
            _: JsonLogReadLimits,
        ) -> LogStorageFuture<'a, AdmittedRead> {
            panic!("registration resolution must not do I/O")
        }
    }

    #[test]
    fn registration_requires_current_same_arc_before_io() {
        let url = ObjectStoreUrl::parse("memory://").unwrap();
        let empty = SessionContext::new();
        assert_eq!(
            JsonLogStorageRegistry::resolve(&empty, &url).err(),
            Some(LogStorageAdmissionError::MissingCapability)
        );
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let capability: Arc<dyn AdmittedJsonLogStorage> = Arc::new(NoIo);
        let mut registry = JsonLogStorageRegistry::default();
        registry.register(url.clone(), Arc::clone(&store), Arc::clone(&capability));
        let session = SessionContext::new_with_config(
            SessionConfig::new().with_extension(Arc::new(registry)),
        );
        assert_eq!(
            JsonLogStorageRegistry::resolve(&session, &url).err(),
            Some(LogStorageAdmissionError::StoreMismatch)
        );
        session.register_object_store(url.as_ref(), Arc::clone(&store));
        assert!(Arc::ptr_eq(
            &JsonLogStorageRegistry::resolve(&session, &url).unwrap(),
            &capability
        ));
        session.register_object_store(url.as_ref(), Arc::new(InMemory::new()));
        assert_eq!(
            JsonLogStorageRegistry::resolve(&session, &url).err(),
            Some(LogStorageAdmissionError::StoreMismatch)
        );
    }

    #[test]
    fn whole_log_limit_checks_cumulative_and_simultaneous_owners() {
        let file = FileDescriptor {
            path: "memory:///log".into(),
            size: 8,
            modification_time: 0,
            identity: ObjectIdentity::new([1; 32]),
        };
        for (resource, requested, input, retained) in [
            (Resource::RequestedReadBytes, 1, 0, 0),
            (Resource::InputBytes, 0, 1, 0),
            (Resource::TaskStateBytes, 0, 0, 1),
            (Resource::ReadPayloadBytes, 0, 0, 0),
        ] {
            let limit = if resource == Resource::ReadPayloadBytes {
                7
            } else {
                8
            };
            let failure = JsonLogReadLimits::try_new(
                &file,
                TaskLimits::qualification().with_limit(resource, limit),
                requested,
                input,
                retained,
            )
            .unwrap_err();
            assert!(
                matches!(failure.kind(), FailureKind::ResourceExhausted(e) if e.resource == resource)
            );
        }
        assert_eq!(
            JsonLogReadLimits::try_new(&file, TaskLimits::qualification(), 0, 0, 0)
                .unwrap()
                .exact_bytes,
            8
        );
    }
    #[test]
    fn whole_log_counter_overflow_rejects_even_with_maximum_limit() {
        let file = FileDescriptor {
            path: "memory:///log".into(),
            size: 1,
            modification_time: 0,
            identity: ObjectIdentity::new([1; 32]),
        };
        for resource in [
            Resource::RequestedReadBytes,
            Resource::InputBytes,
            Resource::TaskStateBytes,
        ] {
            let limits = TaskLimits::qualification().with_limit(resource, usize::MAX);
            let counters = |used| match resource {
                Resource::RequestedReadBytes => (used, 0, 0),
                Resource::InputBytes => (0, used, 0),
                Resource::TaskStateBytes => (0, 0, used),
                _ => unreachable!(),
            };
            let (requested, input, retained) = counters(usize::MAX - 1);
            JsonLogReadLimits::try_new(&file, limits, requested, input, retained).unwrap();
            let (requested, input, retained) = counters(usize::MAX);
            let failure =
                JsonLogReadLimits::try_new(&file, limits, requested, input, retained).unwrap_err();
            assert!(matches!(failure.kind(), FailureKind::ResourceExhausted(e)
                if e.resource == resource && e.limit == usize::MAX && e.observed == usize::MAX));
        }
    }
}
