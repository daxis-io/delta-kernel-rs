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
        })
    }

    /// Returns the checked structural and encoded-size measurements.
    pub fn shape(&self) -> PlanShape {
        self.shape
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
