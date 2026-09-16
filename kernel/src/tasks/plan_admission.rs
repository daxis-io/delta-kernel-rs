use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::mem::size_of;
use std::sync::Arc;

use super::{PlanShape, PlanShapeError, Resource, ResourceExhausted, TaskLimits};
use crate::expressions::Scalar;
use crate::plans::ir::nodes::Values;
use crate::plans::ir::plan::{Plan, PlanNode};
use crate::schema::{ColumnMetadataKey, DataType, MetadataValue, StructField, StructType};

// hashbrown stores one extra control group after its buckets and may add up to one group minus
// one byte of alignment padding before them. Its widest selected group is 16 bytes.
const HASH_TABLE_CONTROL_GROUP: usize = 16;

/// One ordered metadata entry supplied by an allocation-owning plan producer.
///
/// The borrowed slice of entries is the admission source. Unlike a public `HashMap`, its length
/// bounds iteration even after hostile insertions and removals elsewhere. The admitted plan owns
/// exact clones of the sequence and materializes its ordinary Kernel metadata map only after all
/// size, work and duplicate checks succeed.
#[derive(Debug, Clone, Copy)]
pub struct PlanMetadataEntry<'a> {
    key: &'a str,
    value: &'a MetadataValue,
}

impl<'a> PlanMetadataEntry<'a> {
    /// Borrows one key and value from a producer-owned immutable sequence.
    pub fn new(key: &'a str, value: &'a MetadataValue) -> Self {
        Self { key, value }
    }
}

#[derive(Debug)]
struct OwnedMetadataEntry {
    key: String,
    value: MetadataValue,
}

#[derive(Clone, Copy)]
enum FixedValues<'a> {
    I64(&'a [i64]),
    String(&'a [&'a str]),
}

impl FixedValues<'_> {
    fn len(self) -> usize {
        match self {
            Self::I64(values) => values.len(),
            Self::String(values) => values.len(),
        }
    }
}

/// A plan whose retained allocations and executable shape were admitted before construction.
///
/// There is deliberately no conversion from [`Plan`]. Each constructor is an allocation owner
/// for the exact IR subset it accepts. More constructors can be added only with their own
/// producer-allocation and semantic proof.
#[derive(Debug)]
pub struct AdmittedPlan {
    plan: Plan,
    shape: PlanShape,
    retained_bytes: usize,
    metadata_bytes: usize,
    metadata_source: Box<[OwnedMetadataEntry]>,
    log_manifest: Option<Arc<super::LogIdentityManifest>>,
}

impl AdmittedPlan {
    /// Builds one non-nullable LONG `Values` source from borrowed fixed-width rows.
    ///
    /// This deliberately small producer exists for evaluation-driver qualification. Metadata is
    /// accepted as an ordered slice, checked for duplicate keys, and limited to scalar number,
    /// string and boolean values. Every failure precedes allocation and retains no caller input.
    /// General plans, nested JSON metadata and struct patches require separate admitted producers.
    pub fn try_i64_values(
        field_name: &str,
        metadata: &[PlanMetadataEntry<'_>],
        values: &[i64],
        limits: &TaskLimits,
    ) -> Result<Self, PlanAdmissionError> {
        Self::try_fixed_values(field_name, metadata, FixedValues::I64(values), limits)
    }

    /// Builds one non-nullable STRING `Values` source from borrowed strings.
    ///
    /// String bytes, row containers, schema, metadata, and the encoded plan are bounded before
    /// any producer allocation. The accepted metadata subset matches [`Self::try_i64_values`].
    ///
    /// # Errors
    ///
    /// Returns [`PlanAdmissionError`] when a limit is exceeded or metadata is unsupported or
    /// duplicated.
    pub fn try_string_values<'a>(
        field_name: &str,
        metadata: &[PlanMetadataEntry<'_>],
        values: &'a [&'a str],
        limits: &TaskLimits,
    ) -> Result<Self, PlanAdmissionError> {
        Self::try_fixed_values(field_name, metadata, FixedValues::String(values), limits)
    }

    fn try_fixed_values(
        field_name: &str,
        metadata: &[PlanMetadataEntry<'_>],
        values: FixedValues<'_>,
        limits: &TaskLimits,
    ) -> Result<Self, PlanAdmissionError> {
        let mut metadata_encoded = 0usize;
        let mut metadata_dynamic = 0usize;
        let mut comparison_work = 0usize;
        for (index, entry) in metadata.iter().enumerate() {
            if entry.key == ColumnMetadataKey::MetadataSpec.as_ref() {
                return Err(PlanAdmissionError::UnsupportedMetadata);
            }
            for previous in &metadata[..index] {
                comparison_work = checked_add(comparison_work, 1, Resource::WorkUnits, limits)?;
                comparison_work = checked_add(
                    comparison_work,
                    entry.key.len(),
                    Resource::WorkUnits,
                    limits,
                )?;
                comparison_work = checked_add(
                    comparison_work,
                    previous.key.len(),
                    Resource::WorkUnits,
                    limits,
                )?;
                if entry.key == previous.key {
                    return Err(PlanAdmissionError::DuplicateMetadata);
                }
            }
            let value_bytes = match entry.value {
                MetadataValue::Number(_) | MetadataValue::Boolean(_) => 10,
                MetadataValue::String(value) => {
                    metadata_dynamic = checked_add(
                        metadata_dynamic,
                        value.len(),
                        Resource::MetadataAllocatedBytes,
                        limits,
                    )?;
                    value.len()
                }
                MetadataValue::Other(_) => return Err(PlanAdmissionError::UnsupportedMetadata),
            };
            metadata_dynamic = checked_add(
                metadata_dynamic,
                entry.key.len(),
                Resource::MetadataAllocatedBytes,
                limits,
            )?;
            let entry_encoded = checked_add(
                6 * super::plan_shape::FIELD_BOUND,
                entry.key.len(),
                Resource::PlanEncodedBytes,
                limits,
            )?;
            let entry_encoded = checked_add(
                entry_encoded,
                value_bytes,
                Resource::PlanEncodedBytes,
                limits,
            )?;
            metadata_encoded = checked_add(
                metadata_encoded,
                entry_encoded,
                Resource::PlanEncodedBytes,
                limits,
            )?;
        }

        // The immutable sequence and the materialized map each own a key and string value clone.
        metadata_dynamic = checked_mul(
            metadata_dynamic,
            2,
            Resource::MetadataAllocatedBytes,
            limits,
        )?;
        let sequence_slots = checked_mul(
            metadata.len(),
            size_of::<OwnedMetadataEntry>(),
            Resource::MetadataAllocatedBytes,
            limits,
        )?;
        metadata_dynamic = checked_add(
            metadata_dynamic,
            sequence_slots,
            Resource::MetadataAllocatedBytes,
            limits,
        )?;
        let map_bytes = metadata_map_bound(metadata.len(), limits)?;
        metadata_dynamic = checked_add(
            metadata_dynamic,
            map_bytes,
            Resource::MetadataAllocatedBytes,
            limits,
        )?;
        check_limit(Resource::MetadataAllocatedBytes, metadata_dynamic, limits)?;

        let (data_type, value_bytes) = match values {
            FixedValues::I64(_) => (DataType::LONG, 0),
            FixedValues::String(values) => {
                let bytes = values.iter().try_fold(0usize, |used, value| {
                    checked_add(used, value.len(), Resource::TaskStateBytes, limits)
                })?;
                (DataType::STRING, bytes)
            }
        };
        let shape = PlanShape::fixed_values(
            field_name,
            &data_type,
            values.len(),
            value_bytes,
            metadata_encoded,
            comparison_work,
            limits,
        )?;
        let retained_bytes = fixed_plan_backing_bound(
            field_name.len(),
            metadata_dynamic,
            values.len(),
            value_bytes,
            limits,
        )?;
        check_limit(Resource::TaskStateBytes, retained_bytes, limits)?;

        let metadata_source: Box<[OwnedMetadataEntry]> = metadata
            .iter()
            .map(|entry| OwnedMetadataEntry {
                key: owned_string(entry.key),
                value: clone_metadata(entry.value),
            })
            .collect();
        let mut metadata_map = HashMap::with_capacity(metadata_source.len());
        for entry in &metadata_source {
            metadata_map.insert(owned_string(&entry.key), clone_metadata(&entry.value));
        }
        let mut field = StructField::not_null(owned_string(field_name), data_type);
        field.metadata = metadata_map;
        // A single primitive field cannot violate nested or duplicate-field schema invariants.
        let schema = Arc::new(StructType::new_unchecked([field]));
        let rows = match values {
            FixedValues::I64(values) => values
                .iter()
                .map(|value| vec![Scalar::Long(*value)])
                .collect(),
            FixedValues::String(values) => values
                .iter()
                .map(|value| vec![Scalar::String(owned_string(value))])
                .collect(),
        };
        let plan = Plan {
            nodes: vec![PlanNode::new(Values::new(schema, rows), vec![])],
        };
        debug_assert!(retained_bytes >= retained_backing_diagnostic(&plan, &metadata_source));
        Ok(Self {
            plan,
            shape,
            retained_bytes,
            metadata_bytes: metadata_dynamic,
            metadata_source,
            log_manifest: None,
        })
    }

    /// Returns the checked structural and encoded-size measurements.
    pub fn shape(&self) -> PlanShape {
        self.shape
    }

    /// Source bound for a host's borrowed inspection of the sealed JSON IR.
    /// Includes every system field/type/name/path and linear nested field lookup;
    /// it does not authorize arbitrary caller plans or allocate any IR owners.
    pub fn json_host_inspection_work(
        &self,
        limits: &TaskLimits,
    ) -> Result<usize, PlanAdmissionError> {
        let manifest = self
            .log_identity_manifest()
            .ok_or(PlanAdmissionError::InvalidShape)?;
        use super::json_producer_shape::{
            DATA_TYPES, EXPRESSIONS, NODES, PATH_COMPONENTS, SCHEMA_FIELDS,
        };
        let fixed = NODES
            + SCHEMA_FIELDS
            + DATA_TYPES
            + EXPRESSIONS
            + PATH_COMPONENTS
            + (SCHEMA_FIELDS + PATH_COMPONENTS) * "defaultRowCommitVersion".len()
            + PATH_COMPONENTS * 11;
        checked_add(
            fixed,
            checked_mul(manifest.files().len(), 4, Resource::WorkUnits, limits)?,
            Resource::WorkUnits,
            limits,
        )
    }

    /// Returns the conservative dynamic backing bytes owned by this plan.
    pub fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    /// Returns the checked dynamic bytes owned by metadata sequences and indexes.
    pub fn metadata_bytes(&self) -> usize {
        self.metadata_bytes
    }

    /// Returns the number of ordered metadata entries retained as producer provenance.
    pub fn metadata_entries(&self) -> usize {
        self.metadata_source.len()
    }

    /// Looks up fixed-source field metadata without exposing the mutable plan IR.
    pub fn field_metadata(&self, key: &str) -> Option<&MetadataValue> {
        let values = match &self.plan.nodes.first()?.op {
            crate::plans::ir::nodes::Operator::Values(values) => values,
            _ => return None,
        };
        values
            .schema
            .fields()
            .next()
            .and_then(|field| field.metadata().get(key))
    }

    /// Borrows the immutable JSON discovery provenance supplied by a concrete task producer.
    pub fn log_identity_manifest(&self) -> Option<&Arc<super::LogIdentityManifest>> {
        self.log_manifest.as_ref()
    }

    /// Borrows admitted IR during host compilation. The host is responsible for admitting any
    /// additional lowering allocations before creating them; this borrow grants no new producer.
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Transfers the admitted IR to its driver for compilation or execution.
    ///
    /// The caller becomes the allocation owner. Task code must move this value at most once and
    /// retain no clone of the source plan.
    pub fn into_plan(self) -> Plan {
        self.plan
    }
}

/// Source-free failure from a concrete admitted-plan producer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlanAdmissionError {
    /// A finite admission allowance was exceeded or checked arithmetic overflowed.
    ResourceExhausted(ResourceExhausted),
    /// Two ordered metadata entries use the same exact key.
    DuplicateMetadata,
    /// Nested arbitrary JSON requires the later bounded metadata decoder.
    UnsupportedMetadata,
    /// The concrete producer generated an invalid fixed plan shape.
    InvalidShape,
}

impl fmt::Display for PlanAdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResourceExhausted(error) => error.fmt(f),
            Self::DuplicateMetadata => f.write_str("duplicate task metadata key"),
            Self::UnsupportedMetadata => f.write_str("unsupported task metadata value"),
            Self::InvalidShape => f.write_str("invalid admitted plan shape"),
        }
    }
}

impl Error for PlanAdmissionError {}

impl From<PlanAdmissionError> for super::OperationFailure {
    fn from(error: PlanAdmissionError) -> Self {
        match error {
            PlanAdmissionError::ResourceExhausted(error) => error.into(),
            error => Self::new(super::FailureKind::Engine, crate::Error::generic_err(error)),
        }
    }
}

impl From<ResourceExhausted> for PlanAdmissionError {
    fn from(error: ResourceExhausted) -> Self {
        Self::ResourceExhausted(error)
    }
}

impl From<PlanShapeError> for PlanAdmissionError {
    fn from(error: PlanShapeError) -> Self {
        match error {
            PlanShapeError::ResourceExhausted(error) => Self::ResourceExhausted(error),
            _ => Self::InvalidShape,
        }
    }
}

fn check_limit(
    resource: Resource,
    observed: usize,
    limits: &TaskLimits,
) -> Result<(), PlanAdmissionError> {
    let limit = limits.limit(resource);
    if observed > limit {
        return Err(ResourceExhausted {
            resource,
            limit,
            observed,
        }
        .into());
    }
    Ok(())
}

fn checked_add(
    left: usize,
    right: usize,
    resource: Resource,
    limits: &TaskLimits,
) -> Result<usize, PlanAdmissionError> {
    let observed = left.checked_add(right).ok_or(ResourceExhausted {
        resource,
        limit: limits.limit(resource),
        observed: usize::MAX,
    })?;
    check_limit(resource, observed, limits)?;
    Ok(observed)
}

fn checked_mul(
    left: usize,
    right: usize,
    resource: Resource,
    limits: &TaskLimits,
) -> Result<usize, PlanAdmissionError> {
    let observed = left.checked_mul(right).ok_or(ResourceExhausted {
        resource,
        limit: limits.limit(resource),
        observed: usize::MAX,
    })?;
    check_limit(resource, observed, limits)?;
    Ok(observed)
}

fn metadata_map_bound(entries: usize, limits: &TaskLimits) -> Result<usize, PlanAdmissionError> {
    if entries == 0 {
        return Ok(0);
    }
    let minimum = checked_mul(entries, 8, Resource::MetadataAllocatedBytes, limits)?
        .checked_add(6)
        .ok_or(ResourceExhausted {
            resource: Resource::MetadataAllocatedBytes,
            limit: limits.limit(Resource::MetadataAllocatedBytes),
            observed: usize::MAX,
        })?
        / 7;
    let buckets = minimum
        .checked_next_power_of_two()
        .ok_or(ResourceExhausted {
            resource: Resource::MetadataAllocatedBytes,
            limit: limits.limit(Resource::MetadataAllocatedBytes),
            observed: usize::MAX,
        })?
        .max(4);
    let payload_and_controls = checked_mul(
        buckets,
        size_of::<(String, MetadataValue)>() + 1,
        Resource::MetadataAllocatedBytes,
        limits,
    )?;
    checked_add(
        payload_and_controls,
        2 * HASH_TABLE_CONTROL_GROUP - 1,
        Resource::MetadataAllocatedBytes,
        limits,
    )
}

fn fixed_plan_backing_bound(
    field_name_len: usize,
    metadata_bytes: usize,
    rows: usize,
    value_bytes: usize,
    limits: &TaskLimits,
) -> Result<usize, PlanAdmissionError> {
    let row_slots = checked_mul(
        rows,
        size_of::<Vec<Scalar>>() + size_of::<Scalar>(),
        Resource::TaskStateBytes,
        limits,
    )?;
    let mut schema_backing = checked_mul(
        4,
        size_of::<(String, StructField)>(),
        Resource::TaskStateBytes,
        limits,
    )?;
    for amount in [
        size_of::<StructType>(),
        2 * size_of::<usize>(),
        6,
        field_name_len,
        field_name_len,
    ] {
        schema_backing = checked_add(schema_backing, amount, Resource::TaskStateBytes, limits)?;
    }
    let plan_backing = size_of::<PlanNode>();
    let mut total = checked_add(row_slots, schema_backing, Resource::TaskStateBytes, limits)?;
    total = checked_add(total, plan_backing, Resource::TaskStateBytes, limits)?;
    total = checked_add(total, value_bytes, Resource::TaskStateBytes, limits)?;
    checked_add(total, metadata_bytes, Resource::TaskStateBytes, limits)
}

fn owned_string(value: &str) -> String {
    let mut owned = String::with_capacity(value.len());
    owned.push_str(value);
    owned
}

fn clone_metadata(value: &MetadataValue) -> MetadataValue {
    match value {
        MetadataValue::Number(value) => MetadataValue::Number(*value),
        MetadataValue::String(value) => MetadataValue::String(owned_string(value)),
        MetadataValue::Boolean(value) => MetadataValue::Boolean(*value),
        MetadataValue::Other(_) => unreachable!("rejected before allocation"),
    }
}

fn retained_backing_diagnostic(plan: &Plan, source: &[OwnedMetadataEntry]) -> usize {
    let Some(values) = plan.nodes.first().and_then(|node| match &node.op {
        crate::plans::ir::nodes::Operator::Values(values) => Some(values),
        _ => None,
    }) else {
        return usize::MAX;
    };
    let Some(field) = values.schema.fields().next() else {
        return usize::MAX;
    };
    let sequence = source.iter().fold(size_of_val(source), |bytes, entry| {
        bytes
            + entry.key.capacity()
            + match &entry.value {
                MetadataValue::String(value) => value.capacity(),
                _ => 0,
            }
    });
    let map_values = field.metadata().iter().fold(0, |bytes, (key, value)| {
        bytes
            + key.capacity()
            + match value {
                MetadataValue::String(value) => value.capacity(),
                _ => 0,
            }
    });
    sequence
        + map_values
        + values.rows.capacity() * size_of::<Vec<Scalar>>()
        + values
            .rows
            .iter()
            .map(|row| {
                row.capacity() * size_of::<Scalar>()
                    + row
                        .iter()
                        .map(|value| match value {
                            Scalar::String(value) => value.capacity(),
                            _ => 0,
                        })
                        .sum::<usize>()
            })
            .sum::<usize>()
        + field.name.capacity() * 2
        + plan.nodes.capacity() * size_of::<PlanNode>()
}

// These are closed allocation-owning producers, not admission of a caller-supplied Plan. Both
// builders below use fixed system schemas and fixed expression trees. JSON task scans disable
// stats, partitions and predicates. ScanBuilder scratch still depends on the user schema and
// configuration, and is admitted separately by preflight_scan before ScanBuilder is called.
// The envelope includes builder scratch, Arc/Vec/map containers, fixed system-schema clones and
// expressions. Per-file allowance includes ParsedLogPath/FileMeta/ScanFile/scalar containers and
// all URL/string clones. Admission precedes the first builder call; final inspection is diagnostic.
const JSON_PLAN_FIXED_ENCODING: usize = super::json_producer_shape::FIXED_ENCODING;
// PM is ScanJson + Aggregate. Live-add is ScanJson, Filter, Project,
// Aggregate, Filter, Project, and the existing scan-row Project.
const JSON_PLAN_MAX_NODES: usize = super::json_producer_shape::NODES;

pub(crate) struct JsonPlanBudget {
    pub(crate) backing: usize,
    fixed_backing: usize,
    encoded: usize,
    work: usize,
}

impl JsonPlanBudget {
    pub(super) fn work_units(&self) -> usize {
        self.work
    }

    pub(crate) fn producer_work(
        files: usize,
        root_bytes: usize,
        limits: &TaskLimits,
    ) -> Result<usize, PlanAdmissionError> {
        let work = root_bytes
            .checked_add(25)
            .and_then(|n| n.checked_mul(files))
            .and_then(|n| n.checked_add(root_bytes))
            .and_then(|paths| super::json_producer_shape::work_units(files, paths))
            .ok_or(PlanAdmissionError::ResourceExhausted(ResourceExhausted {
                resource: Resource::WorkUnits,
                limit: limits.limit(Resource::WorkUnits),
                observed: usize::MAX,
            }))?;
        check_limit(Resource::WorkUnits, work, limits)?;
        Ok(work)
    }

    pub(crate) fn preflight_scan(
        manifest: &super::LogIdentityManifest,
        schema: &StructType,
        configuration: &crate::table_configuration::TableConfiguration,
        snapshot_bytes: usize,
        limits: &TaskLimits,
    ) -> Result<Self, PlanAdmissionError> {
        let mut budget = Self::preflight(manifest, limits)?;
        check_limit(Resource::SchemaNodes, schema.num_fields(), limits)?;
        for field in schema.fields() {
            if !matches!(field.data_type(), crate::schema::DataType::Primitive(_))
                || !field.metadata().is_empty()
            {
                return Err(PlanAdmissionError::UnsupportedMetadata);
            }
        }
        let scratch = super::json_scan_allocation::scan_builder_peak(schema, configuration).ok_or(
            PlanAdmissionError::ResourceExhausted(ResourceExhausted {
                resource: Resource::TaskStateBytes,
                limit: limits.limit(Resource::TaskStateBytes),
                observed: usize::MAX,
            }),
        )?;
        budget.backing = checked_add(budget.backing, scratch, Resource::TaskStateBytes, limits)?;
        // The caller keeps the snapshot/manifest while ScanBuilder and its result coexist.
        checked_add(
            budget.backing,
            snapshot_bytes,
            Resource::TaskStateBytes,
            limits,
        )?;
        Ok(budget)
    }

    pub(crate) fn preflight(
        manifest: &super::LogIdentityManifest,
        limits: &TaskLimits,
    ) -> Result<Self, PlanAdmissionError> {
        // Fresh scan construction checks its complete known producer work before
        // its first descriptor/schema preflight. Snapshot construction prepays
        // this same amount on its existing cumulative ledger before calling us.
        let work = Self::producer_work(manifest.files().len(), manifest.log_root().len(), limits)?;
        check_limit(Resource::PlanNodes, JSON_PLAN_MAX_NODES, limits)?;
        check_limit(Resource::PlanDepth, JSON_PLAN_MAX_NODES, limits)?;
        // Five serialized system roots, their fields and nested DataTypes.
        // These are source shape counts, not a second schema allowance.
        check_limit(
            Resource::SchemaNodes,
            super::json_producer_shape::SCHEMA_FIELDS + super::json_producer_shape::DATA_TYPES + 5,
            limits,
        )?;
        check_limit(Resource::SchemaDepth, 4, limits)?;
        let file_owners = json_file_plan_owners(manifest).ok_or_else(|| {
            PlanAdmissionError::ResourceExhausted(ResourceExhausted {
                resource: Resource::TaskStateBytes,
                limit: limits.limit(Resource::TaskStateBytes),
                observed: usize::MAX,
            })
        })?;
        let fixed_backing = super::json_producer_shape::fixed_owner_peak().ok_or_else(|| {
            PlanAdmissionError::ResourceExhausted(ResourceExhausted {
                resource: Resource::TaskStateBytes,
                limit: limits.limit(Resource::TaskStateBytes),
                observed: usize::MAX,
            })
        })?;
        let backing = checked_add(fixed_backing, file_owners, Resource::TaskStateBytes, limits)?;
        let backing = checked_add(
            backing,
            manifest.retained_bytes(),
            Resource::TaskStateBytes,
            limits,
        )?;
        // plan.proto: repeated ScanFile envelope, FileMeta envelope + three
        // fields, scalar-constant envelope and Scalar::Long field: seven fields.
        // URL bytes are protobuf string bytes, without JSON escaping.
        let encoded_files = checked_mul(
            manifest.files().len(),
            7 * super::plan_shape::FIELD_BOUND,
            Resource::PlanEncodedBytes,
            limits,
        )?;
        let encoded_paths = manifest
            .path_bytes()
            .ok_or(PlanAdmissionError::ResourceExhausted(ResourceExhausted {
                resource: Resource::PlanEncodedBytes,
                limit: limits.limit(Resource::PlanEncodedBytes),
                observed: usize::MAX,
            }))?;
        let encoded = checked_add(
            JSON_PLAN_FIXED_ENCODING,
            encoded_files,
            Resource::PlanEncodedBytes,
            limits,
        )?;
        let encoded = checked_add(encoded, encoded_paths, Resource::PlanEncodedBytes, limits)?;
        check_limit(Resource::MetadataAllocatedBytes, fixed_backing, limits)?;
        Ok(Self {
            backing,
            fixed_backing,
            encoded,
            work,
        })
    }
}

/// File-dependent owners of the exact ordinary producer calls. The source
/// segment remains separately retained by the task. find_commit_cover_paths
/// clones one ParsedLogPath per file; version_tagged_scan_files clones its URL
/// into a ScanFile and owns one Long scalar Vec. scan_source moves those files
/// through Vec::from_iter; build_plan clones the ScanJson operator once.
/// Count both ScanFile containers plus the possible from_iter relocation,
/// all three URL clones, filename/extension clones, and both scalar Vec owners.
fn json_file_plan_owners(manifest: &super::LogIdentityManifest) -> Option<usize> {
    use super::json_schema_shape::vector_peak;
    use crate::plans::ir::nodes::ScanFile;
    let count = manifest.files().len();
    let containers = vector_peak::<crate::path::ParsedLogPath>(count)?
        .checked_add(vector_peak::<ScanFile>(count)?.checked_mul(3)?)?;
    manifest.files().iter().try_fold(containers, |total, file| {
        total
            .checked_add(file.path.len().checked_mul(3)?)?
            .checked_add(25 + "json".len())?
            .checked_add(vector_peak::<Scalar>(1)?.checked_mul(2)?)
    })
}

impl AdmittedPlan {
    pub(crate) fn try_snapshot_json(
        segment: &crate::log_segment::LogSegment,
        manifest: Arc<super::LogIdentityManifest>,
        limits: &TaskLimits,
    ) -> Result<Self, PlanAdmissionError> {
        let budget = JsonPlanBudget::preflight(&manifest, limits)?;
        if segment.checkpoint_version.is_some() || !segment.listed.checkpoint_parts.is_empty() {
            return Err(PlanAdmissionError::InvalidShape);
        }
        let plan = segment
            .protocol_metadata_plan()
            .map_err(|_| PlanAdmissionError::InvalidShape)?;
        Self::finish_json_producer(plan, manifest, budget, limits)
    }

    pub(crate) fn try_scan_json(
        snapshot: crate::snapshot::SnapshotRef,
        limits: &TaskLimits,
    ) -> Result<Self, PlanAdmissionError> {
        Self::scan_json_after_admission(snapshot, limits, |snapshot| {
            let scan = snapshot
                .scan_builder()
                .with_stats(crate::scan::StatsOptions::none())
                .build()
                .map_err(|_| PlanAdmissionError::InvalidShape)?;
            scan.json_task_scan_plan()
                .map_err(|_| PlanAdmissionError::InvalidShape)?
                .ok_or(PlanAdmissionError::InvalidShape)
        })
    }

    fn scan_json_after_admission(
        snapshot: crate::snapshot::SnapshotRef,
        limits: &TaskLimits,
        build: impl FnOnce(crate::snapshot::SnapshotRef) -> Result<Plan, PlanAdmissionError>,
    ) -> Result<Self, PlanAdmissionError> {
        let manifest = snapshot
            .log_identity_manifest
            .as_ref()
            .ok_or(PlanAdmissionError::InvalidShape)?;
        let budget = JsonPlanBudget::preflight_scan(
            manifest,
            snapshot.schema().as_ref(),
            snapshot.table_configuration(),
            snapshot.json_task_retained_bytes,
            limits,
        )?;
        let manifest = Arc::clone(manifest);
        let plan = build(snapshot)?;
        Self::finish_json_producer(plan, manifest, budget, limits)
    }

    fn finish_json_producer(
        plan: Plan,
        manifest: Arc<super::LogIdentityManifest>,
        budget: JsonPlanBudget,
        limits: &TaskLimits,
    ) -> Result<Self, PlanAdmissionError> {
        let shape = PlanShape::json_log_producer_shape(&plan, budget.encoded, budget.work, limits)
            .map_err(PlanAdmissionError::from)?;
        Ok(Self {
            plan,
            shape,
            retained_bytes: budget.backing,
            metadata_bytes: budget.fixed_backing,
            metadata_source: Box::new([]),
            log_manifest: Some(manifest),
        })
    }
}

#[cfg(test)]
pub(super) mod json_tests {
    use prost::Message;

    use super::*;
    use crate::schema::DataType;
    use crate::tasks::{FileDescriptor, LogIdentityManifest, ObjectIdentity};

    pub(crate) fn snapshot(schema: StructType) -> crate::snapshot::SnapshotRef {
        let limits = TaskLimits::qualification();
        let root = url::Url::parse("memory:///table/").unwrap();
        let manifest = Arc::new(
            LogIdentityManifest::try_new(
                "memory:///table/_delta_log/".into(),
                vec![FileDescriptor {
                    path: "memory:///table/_delta_log/00000000000000000000.json".into(),
                    size: 1024,
                    modification_time: 0,
                    identity: ObjectIdentity::new([1; 32]),
                }],
                &limits,
            )
            .unwrap(),
        );
        let metadata = crate::actions::Metadata::try_new(
            None,
            None,
            Arc::new(schema),
            vec![],
            0,
            Default::default(),
        )
        .unwrap();
        let protocol =
            serde_json::from_str("{\"minReaderVersion\":1,\"minWriterVersion\":2}").unwrap();
        let config = crate::table_configuration::TableConfiguration::try_new(
            metadata,
            protocol,
            root.clone(),
            0,
        )
        .unwrap();
        let mut snapshot = crate::snapshot::Snapshot::new_with_crc(
            manifest.to_log_segment(&root, &limits).unwrap(),
            config,
            None,
            true,
        )
        .unwrap();
        snapshot.json_task_retained_bytes = 1 << 20;
        snapshot.log_identity_manifest = Some(manifest);
        Arc::new(snapshot)
    }
    #[test]
    fn json_task_scan_schema_scratch_rejects_before_builder() {
        let schema =
            StructType::try_new((0..128).map(|i| {
                StructField::nullable(format!("f{i}{}", "x".repeat(1024)), DataType::LONG)
            }))
            .unwrap();
        let snapshot = snapshot(schema);
        let limits = TaskLimits::qualification().with_limit(Resource::TaskStateBytes, 8 << 20);
        // The old manifest-only envelope fits this allowance, but reached StateInfo scratch
        // and its simultaneous snapshot owner do not.
        assert!(JsonPlanBudget::preflight(
            snapshot.log_identity_manifest.as_ref().unwrap(),
            &limits
        )
        .is_ok());
        let called = std::cell::Cell::new(false);
        let failure = AdmittedPlan::scan_json_after_admission(snapshot, &limits, |_| {
            called.set(true);
            Err(PlanAdmissionError::InvalidShape)
        })
        .err()
        .unwrap();
        assert!(!called.get());
        assert!(
            matches!(failure, PlanAdmissionError::ResourceExhausted(e) if e.resource == Resource::TaskStateBytes)
        );
    }
    #[test]
    fn json_task_fixed_producers_encoding_diagnostics() {
        for fields in [1, 9, 128] {
            let schema = StructType::try_new(
                (0..fields).map(|i| StructField::nullable(format!("f{i}"), DataType::LONG)),
            )
            .unwrap();
            let snapshot = snapshot(schema);
            let limits = TaskLimits::qualification();
            let manifest = snapshot.log_identity_manifest.as_ref().unwrap();
            let pm = AdmittedPlan::try_snapshot_json(
                &manifest
                    .to_log_segment(snapshot.table_root(), &limits)
                    .unwrap(),
                manifest.clone(),
                &limits,
            )
            .unwrap();
            let scan = AdmittedPlan::try_scan_json(snapshot, &limits).unwrap();
            for (kind, plan) in [("pm", pm), ("scan", scan)] {
                let wire = crate::plans::proto::plan::Plan::from(plan.plan());
                assert!(wire.encoded_len() <= plan.shape.encoded_bytes());
                println!("json_task fixed producer kind={kind} user_fields={fields} nodes={} encoded={} encoded_bound={} backing_bound={}", plan.plan.nodes.len(), wire.encoded_len(), plan.shape.encoded_bytes(), plan.retained_bytes());
            }
        }
    }
}
