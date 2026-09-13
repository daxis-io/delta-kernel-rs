use std::mem::size_of;

use super::{FileDescriptor, OperationFailure, Resource, ResourceExhausted, TaskLimits};

/// Immutable identity observations for one complete JSON history. Only the concrete snapshot
/// task can construct this provenance; scan tasks and their executors share the same owner.
#[derive(Debug)]
pub struct LogIdentityManifest {
    log_root: String,
    files: Vec<FileDescriptor>,
    version: u64,
    retained_bytes: usize,
}

impl LogIdentityManifest {
    /// Borrows selected commits in ascending version order, including their discovery identities.
    pub fn files(&self) -> &[FileDescriptor] {
        &self.files
    }

    /// Returns the canonical log-directory URL prefix used for discovery.
    pub fn log_root(&self) -> &str {
        &self.log_root
    }

    // try_new proves every full URL is root plus the canonical 25-byte filename.
    // Compute this exact sum without walking descriptors before work admission.
    pub(crate) fn path_bytes(&self) -> Option<usize> {
        self.log_root
            .len()
            .checked_add(25)?
            .checked_mul(self.files.len())
    }

    /// Returns the last version in the complete zero-based JSON history.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Returns the conservative backing charge retained by this immutable owner.
    pub fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    /// Source-derived peak for ordinary canonical LogSegment construction.
    /// Each ParsedLogPath owns FileMeta's URL, its 25-byte filename and four-byte
    /// extension. try_from additionally collects the one extension slice into a
    /// Vec. The selected parser branches have no host/IDNA or checkpoint owners.
    /// The latest-commit clone and log-root URL coexist with the full commit Vec.
    pub(crate) fn segment_owner_peak(
        &self,
        limits: &TaskLimits,
    ) -> Result<usize, OperationFailure> {
        use super::json_schema_shape::vector_peak;
        let compute = || -> Option<usize> {
            let mut peak = vector_peak::<crate::path::ParsedLogPath>(self.files.len())?
                .checked_add(vector_peak::<u8>(self.log_root.len())?)?
                .checked_add(size_of::<crate::log_segment::LogSegment>())?;
            let mut latest = 0;
            for file in &self.files {
                let path = vector_peak::<u8>(file.path.len())?
                    .checked_add(25 + "json".len())?
                    .checked_add(vector_peak::<&str>(1)?)?;
                peak = peak.checked_add(path)?;
                latest = file.path.len().checked_add(25 + "json".len())?;
            }
            peak.checked_add(latest)
        };
        let peak = compute().ok_or_else(|| exhausted(Resource::TaskStateBytes, limits))?;
        check(Resource::TaskStateBytes, peak, limits)?;
        check(Resource::MetadataAllocatedBytes, peak, limits)?;
        Ok(peak)
    }

    /// Materializes the same canonical segment used by synchronous snapshot construction.
    /// The caller also admits simultaneous task ownership against its ledger before this call.
    pub(crate) fn to_log_segment(
        &self,
        table_root: &url::Url,
        limits: &TaskLimits,
    ) -> Result<crate::log_segment::LogSegment, OperationFailure> {
        if table_root.cannot_be_a_base()
            || table_root.query().is_some()
            || table_root.fragment().is_some()
            || !table_root.path().ends_with('/')
            || self.log_root.strip_prefix(table_root.as_str()) != Some("_delta_log/")
        {
            return Err(OperationFailure::malformed_response());
        }
        self.segment_owner_peak(limits)?;
        // The table URL is already parsed and the manifest admitted exactly the
        // canonical immediate-child twenty-digit names. Fixed relative joins
        // preserve that host without re-entering URL/IDNA parsing on each file.
        let log_root = table_root
            .join("_delta_log/")
            .map_err(|_| OperationFailure::malformed_response())?;
        if log_root.as_str() != self.log_root {
            return Err(OperationFailure::malformed_response());
        }
        let commits = self
            .files
            .iter()
            .map(|file| {
                let name = file
                    .path
                    .strip_prefix(&self.log_root)
                    .ok_or_else(OperationFailure::malformed_response)?;
                let location = log_root
                    .join(name)
                    .map_err(|_| OperationFailure::malformed_response())?;
                if location.as_str() != file.path {
                    return Err(OperationFailure::malformed_response());
                }
                let meta = crate::FileMeta {
                    location,
                    size: file.size,
                    last_modified: file.modification_time,
                };
                crate::path::ParsedLogPath::try_from(meta)
                    .map_err(|error| OperationFailure::new(super::FailureKind::Engine, error))?
                    .ok_or_else(OperationFailure::malformed_response)
            })
            .collect::<Result<Vec<_>, OperationFailure>>()?;
        let latest = commits.last().cloned();
        Ok(crate::log_segment::LogSegment {
            end_version: self.version,
            checkpoint_version: None,
            log_root,
            listed: crate::log_segment_files::LogSegmentFiles {
                ascending_commit_files: commits,
                latest_commit_file: latest,
                max_published_version: Some(self.version),
                ..Default::default()
            },
            last_checkpoint_metadata: None,
        })
    }

    // Takes already admitted ownership; performs no allocations. The task separately admits
    // its Arc allocation and charges simultaneous owners before calling this constructor.
    pub(crate) fn try_new(
        log_root: String,
        files: Vec<FileDescriptor>,
        limits: &TaskLimits,
    ) -> Result<Self, OperationFailure> {
        if files.is_empty() {
            return Err(OperationFailure::new(
                super::FailureKind::Engine,
                crate::Error::MissingVersion,
            ));
        }
        if !log_root.ends_with('/') {
            return Err(OperationFailure::malformed_response());
        }
        check(Resource::LogDescriptors, files.len(), limits)?;
        let base = files
            .capacity()
            .checked_mul(size_of::<FileDescriptor>())
            .and_then(|n| n.checked_add(log_root.capacity()))
            .and_then(|n| n.checked_add(size_of::<Self>() + 2 * size_of::<usize>()))
            .ok_or_else(|| exhausted(Resource::TaskStateBytes, limits))?;
        let mut retained_bytes = base;
        for (index, file) in files.iter().enumerate() {
            if commit_version(&log_root, &file.path) != u64::try_from(index).ok() {
                return Err(OperationFailure::malformed_response());
            }
            retained_bytes = retained_bytes
                .checked_add(file.path.capacity())
                .ok_or_else(|| exhausted(Resource::TaskStateBytes, limits))?;
            check(Resource::TaskStateBytes, retained_bytes, limits)?;
        }
        let version =
            u64::try_from(files.len() - 1).map_err(|_| OperationFailure::malformed_response())?;
        Ok(Self {
            log_root,
            files,
            version,
            retained_bytes,
        })
    }
}

/// Parses only canonical, immediate-child, twenty-digit JSON commit names without allocating.
pub(crate) fn commit_version(root: &str, path: &str) -> Option<u64> {
    let name = path.strip_prefix(root)?;
    if name.len() != 25 || !name.ends_with(".json") {
        return None;
    }
    name.as_bytes()[..20]
        .iter()
        .try_fold(0u64, |version, digit| {
            if !digit.is_ascii_digit() {
                return None;
            }
            version
                .checked_mul(10)?
                .checked_add(u64::from(digit - b'0'))
        })
}

fn check(resource: Resource, observed: usize, limits: &TaskLimits) -> Result<(), OperationFailure> {
    if observed > limits.limit(resource) {
        return Err(ResourceExhausted {
            resource,
            limit: limits.limit(resource),
            observed,
        }
        .into());
    }
    Ok(())
}

fn exhausted(resource: Resource, limits: &TaskLimits) -> OperationFailure {
    ResourceExhausted {
        resource,
        limit: limits.limit(resource),
        observed: usize::MAX,
    }
    .into()
}

#[cfg(test)]
mod tests {
    use super::super::{FailureKind, ObjectIdentity};
    use super::*;
    #[test]
    fn canonical_segment_admission_boundary_and_root_binding() {
        let limits = TaskLimits::qualification();
        let root = url::Url::parse("https://xn--bcher-kva.example/t%20able/").unwrap();
        let log_root = root.join("_delta_log/").unwrap().to_string();
        let file = FileDescriptor {
            path: format!("{log_root}00000000000000000000.json"),
            size: 1,
            modification_time: 0,
            identity: ObjectIdentity::new([1; 32]),
        };
        let manifest = LogIdentityManifest::try_new(log_root, vec![file], &limits).unwrap();
        let peak = manifest.segment_owner_peak(&limits).unwrap();
        let segment = manifest
            .to_log_segment(&root, &limits.with_limit(Resource::TaskStateBytes, peak))
            .unwrap();
        assert_eq!(
            segment.listed.ascending_commit_files[0]
                .location
                .location
                .as_str(),
            manifest.files()[0].path
        );
        let failure = manifest
            .to_log_segment(
                &root,
                &limits.with_limit(Resource::TaskStateBytes, peak - 1),
            )
            .unwrap_err();
        assert!(matches!(failure.kind(), FailureKind::ResourceExhausted(e)
            if e.resource == Resource::TaskStateBytes && e.observed == peak));
        let other = url::Url::parse("https://other.example/table/").unwrap();
        assert_eq!(
            manifest.to_log_segment(&other, &limits).unwrap_err().kind(),
            FailureKind::MalformedResponse
        );
    }
}
