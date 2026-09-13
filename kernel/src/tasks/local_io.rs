use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path};
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use rustix::fs::{openat, Dir, Mode, OFlags};

use super::{
    AdmittedFooter, AdmittedHead, AdmittedIoSource, AdmittedListingPage, AdmittedRead, FailureKind,
    FileDescriptor, FooterLimits, ObjectIdentity, OperationFailure, Resource, ResourceExhausted,
    TaskLimits,
};
use crate::arrow::datatypes::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema,
};
use crate::engine::arrow_conversion::TryFromArrow as _;
use crate::parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use crate::parquet::errors::ParquetError;
use crate::parquet::file::metadata::{ParquetMetaDataLimits, ParquetMetaDataOptions};
use crate::schema::StructType;
use crate::{Error, ParquetFooter};

struct LocalEntry {
    descriptor: FileDescriptor,
}

/// Finite, eagerly admitted local-file index for native operation tasks.
pub struct LocalFileIoSource {
    root: Option<File>,
    root_text: String,
    entries: Vec<LocalEntry>,
}

impl LocalFileIoSource {
    /// Canonicalizes `root`, discovers a bounded file index, and sorts it by full path.
    pub fn try_new(root: impl AsRef<Path>, limits: TaskLimits) -> Result<Self, OperationFailure> {
        let requested_root = root.as_ref();
        if !requested_root.is_absolute() {
            return Err(engine(Error::generic(
                "local task root must be an absolute path",
            )));
        }
        let mut root_text = requested_root
            .to_str()
            .ok_or_else(|| engine(Error::generic("local task root is not UTF-8")))?
            .to_owned();
        while root_text.len() > 1 && root_text.ends_with('/') {
            root_text.pop();
        }
        if root_text != "/" {
            root_text.push('/');
        }
        let canonical_root = fs::canonicalize(requested_root).map_err(engine)?;
        let root = open_absolute_directory(&canonical_root)?;
        let count_limit = limits.limit(Resource::LogDescriptors);
        let byte_limit = limits.limit(Resource::DescriptorBytes);
        let entry_bytes = std::mem::size_of::<LocalEntry>();
        let capacity = count_limit.min(byte_limit / entry_bytes.max(1));
        let mut entries = Vec::new();
        entries.try_reserve_exact(capacity).map_err(|error| {
            engine(Error::generic(format!(
                "cannot reserve local discovery index: {error}"
            )))
        })?;
        let mut discovered = 0usize;
        let mut retained = entries
            .capacity()
            .checked_mul(entry_bytes)
            .ok_or_else(|| exhausted(Resource::DescriptorBytes, byte_limit, usize::MAX))?;
        check_limit(Resource::DescriptorBytes, byte_limit, retained)?;
        discover(
            &root,
            Path::new(""),
            &root_text,
            (count_limit, byte_limit),
            &mut discovered,
            &mut retained,
            &mut entries,
        )?;
        entries.sort_unstable_by(|left, right| left.descriptor.path.cmp(&right.descriptor.path));
        Ok(Self {
            root: Some(root),
            root_text,
            entries,
        })
    }
}

impl AdmittedIoSource for LocalFileIoSource {
    fn list(
        &mut self,
        root: &str,
        continuation: Option<&str>,
        entries: usize,
        descriptor_bytes: usize,
        continuation_bytes: usize,
    ) -> Result<AdmittedListingPage, OperationFailure> {
        if self.root_text.is_empty() || root != self.root_text || entries == 0 {
            return Err(OperationFailure::malformed_response());
        }
        let start = continuation.map_or(0, |after| {
            self.entries
                .partition_point(|entry| entry.descriptor.path.as_str() <= after)
        });
        let count = entries.min(self.entries.len().saturating_sub(start));
        let entry_bytes = std::mem::size_of::<FileDescriptor>();
        let minimum_container_bytes = count.checked_mul(entry_bytes).ok_or_else(|| {
            exhausted(
                Resource::ListingDescriptorBytes,
                descriptor_bytes,
                usize::MAX,
            )
        })?;
        check_limit(
            Resource::ListingDescriptorBytes,
            descriptor_bytes,
            minimum_container_bytes,
        )?;
        let mut files = Vec::new();
        files.try_reserve_exact(count).map_err(|error| {
            engine(Error::generic(format!(
                "cannot reserve local listing page: {error}"
            )))
        })?;
        let mut retained = files.capacity().checked_mul(entry_bytes).ok_or_else(|| {
            exhausted(
                Resource::ListingDescriptorBytes,
                descriptor_bytes,
                usize::MAX,
            )
        })?;
        check_limit(Resource::ListingDescriptorBytes, descriptor_bytes, retained)?;
        for entry in &self.entries[start..start + count] {
            let minimum_retained = retained
                .checked_add(entry.descriptor.path.len())
                .ok_or_else(|| {
                    exhausted(
                        Resource::ListingDescriptorBytes,
                        descriptor_bytes,
                        usize::MAX,
                    )
                })?;
            check_limit(
                Resource::ListingDescriptorBytes,
                descriptor_bytes,
                minimum_retained,
            )?;
            let path = admitted_string(
                &entry.descriptor.path,
                Resource::ListingDescriptorBytes,
                descriptor_bytes,
            )?;
            retained = retained.checked_add(path.capacity()).ok_or_else(|| {
                exhausted(
                    Resource::ListingDescriptorBytes,
                    descriptor_bytes,
                    usize::MAX,
                )
            })?;
            check_limit(Resource::ListingDescriptorBytes, descriptor_bytes, retained)?;
            files.push(FileDescriptor {
                path,
                size: entry.descriptor.size,
                modification_time: entry.descriptor.modification_time,
                identity: entry.descriptor.identity,
            });
        }
        let (continuation, binding) = if count == entries {
            let last = files
                .last()
                .ok_or_else(OperationFailure::malformed_response)?;
            check_limit(
                Resource::ContinuationBytes,
                continuation_bytes,
                last.path.len(),
            )?;
            (
                Some(admitted_string(
                    &last.path,
                    Resource::ContinuationBytes,
                    continuation_bytes,
                )?),
                Some(admitted_string(
                    &last.path,
                    Resource::ContinuationBytes,
                    continuation_bytes,
                )?),
            )
        } else {
            (None, None)
        };
        Ok(AdmittedListingPage {
            files,
            continuation,
            binding,
        })
    }

    fn read_exact(
        &mut self,
        path: &str,
        expected_identity: Option<ObjectIdentity>,
        offset: u64,
        length: usize,
    ) -> Result<AdmittedRead, OperationFailure> {
        let mut file = self.open_confined(path)?;
        let before = file.metadata().map_err(engine)?;
        let observed_identity = identity(&before);
        if expected_identity.is_some_and(|expected| expected != observed_identity) {
            return Err(engine(Error::generic("local object identity changed")));
        }
        let length_u64 = u64::try_from(length)
            .map_err(|error| engine(Error::generic(format!("range length overflow: {error}"))))?;
        let end = offset
            .checked_add(length_u64)
            .ok_or_else(|| engine(Error::generic("local read range overflow")))?;
        if end > before.len() {
            return Err(engine(Error::generic(
                "local read range exceeds object size",
            )));
        }
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(length).map_err(|error| {
            engine(Error::generic(format!(
                "cannot reserve exact local read destination: {error}"
            )))
        })?;
        if bytes.capacity() != length {
            return Err(engine(Error::generic(
                "local read destination capacity is not exact",
            )));
        }
        bytes.resize(length, 0);
        file.seek(SeekFrom::Start(offset)).map_err(engine)?;
        file.read_exact(&mut bytes).map_err(engine)?;
        let after = file.metadata().map_err(engine)?;
        if identity(&after) != observed_identity || after.len() != before.len() {
            return Err(engine(Error::generic(
                "local object changed during exact read",
            )));
        }
        Ok(AdmittedRead {
            identity: observed_identity,
            offset,
            bytes,
            eof: end == before.len(),
        })
    }

    fn head(&mut self, path: &str) -> Result<AdmittedHead, OperationFailure> {
        let file = self.open_confined(path)?;
        let metadata = file.metadata().map_err(engine)?;
        Ok(AdmittedHead {
            identity: identity(&metadata),
            size: metadata.len(),
        })
    }

    fn footer(
        &mut self,
        path: &str,
        expected_identity: ObjectIdentity,
        size: u64,
        limits: FooterLimits,
    ) -> Result<AdmittedFooter, OperationFailure> {
        if limits.schema_depth == 0 {
            return Err(exhausted(Resource::SchemaDepth, 0, 1));
        }
        let file = self.open_confined(path)?;
        let before = file.metadata().map_err(engine)?;
        let observed_identity = identity(&before);
        if observed_identity != expected_identity || before.len() != size {
            return Err(engine(Error::generic("local object identity changed")));
        }
        let parquet_limits = parquet_limits(limits)?;
        let options = ArrowReaderOptions::new()
            .with_skip_arrow_metadata(true)
            .with_metadata_options(ParquetMetaDataOptions::new().with_limits(parquet_limits));
        let metadata = ArrowReaderMetadata::load(&file, options).map_err(parquet_failure)?;
        check_kernel_schema_bound(metadata.schema(), limits.metadata_allocated_bytes)?;
        let schema = StructType::try_from_arrow(metadata.schema().as_ref())
            .map_err(|error| engine(Error::from(error)))?;
        let after = file.metadata().map_err(engine)?;
        if identity(&after) != observed_identity || after.len() != size {
            return Err(engine(Error::generic(
                "local object changed during footer decoding",
            )));
        }
        Ok(AdmittedFooter {
            identity: observed_identity,
            size,
            footer: ParquetFooter {
                schema: Arc::new(schema),
            },
        })
    }

    fn cancel(&mut self) {
        self.entries = Vec::new();
        self.root = None;
        self.root_text = String::new();
    }
}

impl LocalFileIoSource {
    fn open_confined(&self, path: &str) -> Result<File, OperationFailure> {
        if self.root_text.is_empty() {
            return Err(engine(Error::generic("local task source is cancelled")));
        }
        let root = self
            .root
            .as_ref()
            .ok_or_else(|| engine(Error::generic("local task source is cancelled")))?;
        let requested = Path::new(path);
        let visible_root = Path::new(&self.root_text);
        let relative = requested
            .strip_prefix(visible_root)
            .map_err(|_| engine(Error::generic("local object is outside the admitted root")))?;
        let mut components = relative.components().peekable();
        if components.peek().is_none() {
            return Err(engine(Error::generic(
                "local object path does not name a file",
            )));
        }
        let mut file = File::from(
            openat(root, ".", directory_flags(), Mode::empty()).map_err(rustix_failure)?,
        );
        while let Some(component) = components.next() {
            let Component::Normal(name) = component else {
                return Err(engine(Error::generic(
                    "local object is outside the admitted root",
                )));
            };
            let flags = if components.peek().is_some() {
                directory_flags()
            } else {
                file_flags()
            };
            file = File::from(openat(&file, name, flags, Mode::empty()).map_err(rustix_failure)?);
        }
        if !file.metadata().map_err(engine)?.is_file() {
            return Err(engine(Error::generic("local object is not a file")));
        }
        Ok(file)
    }
}

fn discover(
    directory: &File,
    relative_directory: &Path,
    requested_root: &str,
    limits: (usize, usize),
    discovered: &mut usize,
    retained: &mut usize,
    entries: &mut Vec<LocalEntry>,
) -> Result<(), OperationFailure> {
    let (count_limit, byte_limit) = limits;
    let directory_entries = Dir::read_from(directory).map_err(rustix_failure)?;
    for entry in directory_entries {
        let entry = entry.map_err(rustix_failure)?;
        let name = entry.file_name();
        if name.to_bytes() == b"." || name.to_bytes() == b".." {
            continue;
        }
        *discovered = discovered
            .checked_add(1)
            .ok_or_else(|| exhausted(Resource::LogDescriptors, count_limit, usize::MAX))?;
        check_limit(Resource::LogDescriptors, count_limit, *discovered)?;
        let file = File::from(
            openat(directory, name, file_flags(), Mode::empty()).map_err(rustix_failure)?,
        );
        let metadata = file.metadata().map_err(engine)?;
        let relative = relative_directory.join(std::ffi::OsStr::from_bytes(name.to_bytes()));
        if metadata.is_dir() {
            discover(
                &file,
                &relative,
                requested_root,
                limits,
                discovered,
                retained,
                entries,
            )?;
            continue;
        }
        if !metadata.is_file() {
            continue;
        }
        let visible_path = Path::new(requested_root).join(&relative);
        let path = visible_path
            .to_str()
            .ok_or_else(|| engine(Error::generic("local object path is not UTF-8")))?;
        let minimum_retained = retained
            .checked_add(path.len())
            .ok_or_else(|| exhausted(Resource::DescriptorBytes, byte_limit, usize::MAX))?;
        check_limit(Resource::DescriptorBytes, byte_limit, minimum_retained)?;
        let path = admitted_string(path, Resource::DescriptorBytes, byte_limit)?;
        *retained = retained
            .checked_add(path.capacity())
            .ok_or_else(|| exhausted(Resource::DescriptorBytes, byte_limit, usize::MAX))?;
        check_limit(Resource::DescriptorBytes, byte_limit, *retained)?;
        entries.push(LocalEntry {
            descriptor: FileDescriptor {
                path,
                size: metadata.len(),
                modification_time: modification_time_millis(&metadata)?,
                identity: identity(&metadata),
            },
        });
    }
    Ok(())
}

fn admitted_string(
    value: &str,
    resource: Resource,
    limit: usize,
) -> Result<String, OperationFailure> {
    check_limit(resource, limit, value.len())?;
    let mut admitted = String::new();
    admitted.try_reserve_exact(value.len()).map_err(|error| {
        engine(Error::generic(format!(
            "cannot reserve admitted local path: {error}"
        )))
    })?;
    admitted.push_str(value);
    check_limit(resource, limit, admitted.capacity())?;
    Ok(admitted)
}

fn modification_time_millis(metadata: &fs::Metadata) -> Result<i64, OperationFailure> {
    match metadata
        .modified()
        .map_err(engine)?
        .duration_since(UNIX_EPOCH)
    {
        Ok(elapsed) => i64::try_from(elapsed.as_millis()).map_err(|error| {
            engine(Error::generic(format!(
                "modification time overflow: {error}"
            )))
        }),
        Err(error) => i64::try_from(error.duration().as_millis())
            .ok()
            .and_then(i64::checked_neg)
            .ok_or_else(|| engine(Error::generic("modification time overflow"))),
    }
}

fn identity(metadata: &fs::Metadata) -> ObjectIdentity {
    let mut bytes = [0; 32];
    for (slot, value) in [
        metadata.dev(),
        metadata.ino(),
        metadata.ctime() as u64,
        metadata.ctime_nsec() as u64,
    ]
    .into_iter()
    .enumerate()
    {
        bytes[slot * 8..(slot + 1) * 8].copy_from_slice(&value.to_le_bytes());
    }
    ObjectIdentity::new(bytes)
}

fn parquet_limits(limits: FooterLimits) -> Result<ParquetMetaDataLimits, OperationFailure> {
    ParquetMetaDataLimits::builder()
        .with_footer_bytes(limits.footer_bytes)
        .with_schema_elements(limits.schema_nodes)
        .with_schema_depth(limits.schema_depth)
        .with_row_groups(limits.row_groups)
        .with_column_chunks_per_row_group(limits.column_chunks)
        .with_column_chunks(limits.column_chunks)
        .with_collection_elements(limits.metadata_allocated_bytes)
        .with_variable_width_bytes(limits.metadata_allocated_bytes)
        .with_total_variable_width_bytes(limits.metadata_allocated_bytes)
        .with_decoded_bytes(limits.metadata_allocated_bytes)
        .with_page_index_bytes(limits.page_index_bytes)
        .with_page_index_entries(limits.page_index_entries)
        .build()
        .map_err(parquet_failure)
}

fn check_kernel_schema_bound(schema: &ArrowSchema, limit: usize) -> Result<(), OperationFailure> {
    fn field_bound(field: &ArrowField, depth: usize) -> Option<usize> {
        let fixed = std::mem::size_of::<ArrowField>()
            .checked_add(std::mem::size_of::<crate::schema::StructField>())?
            .checked_add(8 * std::mem::size_of::<usize>())?;
        let metadata = field
            .metadata()
            .iter()
            .try_fold(0usize, |used, (key, value)| {
                used.checked_add(key.len())?.checked_add(value.len())
            })?;
        let nested_path = field.name().len().checked_mul(depth)?;
        let own = fixed
            .checked_add(field.name().len().checked_mul(2)?)?
            .checked_add(metadata.checked_mul(2)?)?
            .checked_add(nested_path)?;
        match field.data_type() {
            ArrowDataType::Struct(fields) => fields.iter().try_fold(own, |used, child| {
                used.checked_add(field_bound(child, depth.checked_add(1)?)?)
            }),
            ArrowDataType::List(child)
            | ArrowDataType::ListView(child)
            | ArrowDataType::LargeList(child)
            | ArrowDataType::LargeListView(child)
            | ArrowDataType::FixedSizeList(child, _) => {
                own.checked_add(field_bound(child, depth.checked_add(1)?)?)
            }
            ArrowDataType::Map(entries, _) => {
                own.checked_add(field_bound(entries, depth.checked_add(1)?)?)
            }
            ArrowDataType::Dictionary(_, value) => {
                own.checked_add(data_type_bound(value, depth.checked_add(1)?)?)
            }
            _ => Some(own),
        }
    }

    fn data_type_bound(data_type: &ArrowDataType, depth: usize) -> Option<usize> {
        match data_type {
            ArrowDataType::Struct(fields) => fields.iter().try_fold(0usize, |used, field| {
                used.checked_add(field_bound(field, depth)?)
            }),
            ArrowDataType::List(field)
            | ArrowDataType::ListView(field)
            | ArrowDataType::LargeList(field)
            | ArrowDataType::LargeListView(field)
            | ArrowDataType::FixedSizeList(field, _)
            | ArrowDataType::Map(field, _) => field_bound(field, depth),
            ArrowDataType::Dictionary(_, value) => data_type_bound(value, depth),
            _ => Some(std::mem::size_of::<crate::schema::DataType>()),
        }
    }

    let observed = schema.fields().iter().try_fold(0usize, |used, field| {
        used.checked_add(field_bound(field, 1)?)
    });
    check_limit(
        Resource::MetadataAllocatedBytes,
        limit,
        observed.unwrap_or(usize::MAX),
    )
}

fn parquet_failure(error: ParquetError) -> OperationFailure {
    if let ParquetError::ResourceExhausted {
        resource,
        limit,
        observed,
    } = error
    {
        let resource = match resource {
            "footer bytes" => Resource::FooterBytes,
            "schema elements" => Resource::SchemaNodes,
            "schema depth" => Resource::SchemaDepth,
            "row groups" => Resource::RowGroups,
            "column chunks" | "column chunks per row group" => Resource::ColumnChunks,
            "page index bytes" => Resource::PageIndexBytes,
            "page index entries" => Resource::PageIndexEntries,
            _ => Resource::MetadataAllocatedBytes,
        };
        return exhausted(resource, limit, observed);
    }
    engine(Error::from(error))
}

fn check_limit(resource: Resource, limit: usize, observed: usize) -> Result<(), OperationFailure> {
    if observed > limit {
        return Err(exhausted(resource, limit, observed));
    }
    Ok(())
}

fn exhausted(resource: Resource, limit: usize, observed: usize) -> OperationFailure {
    ResourceExhausted {
        resource,
        limit,
        observed,
    }
    .into()
}

fn engine(error: impl Into<Error>) -> OperationFailure {
    OperationFailure::new(FailureKind::Engine, error.into())
}

fn open_absolute_directory(path: &Path) -> Result<File, OperationFailure> {
    let mut directory = File::open("/").map_err(engine)?;
    for component in path.components() {
        match component {
            Component::RootDir => {}
            Component::Normal(name) => {
                directory = File::from(
                    openat(&directory, name, directory_flags(), Mode::empty())
                        .map_err(rustix_failure)?,
                );
            }
            _ => return Err(engine(Error::generic("invalid canonical local task root"))),
        }
    }
    Ok(directory)
}

fn directory_flags() -> OFlags {
    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC
}

fn file_flags() -> OFlags {
    OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK
}

fn rustix_failure(error: rustix::io::Errno) -> OperationFailure {
    engine(std::io::Error::from_raw_os_error(error.raw_os_error()))
}
