//! File groups, partition projection and provider owners for one JSON source.
use crate::{
    closed_plan_facts::ClosedPlanFacts,
    json_arrays::{shared_owner_bytes as owner, vec_peak},
};
use datafusion::arrow::datatypes::{FieldRef, Schema};
use datafusion::common::ScalarValue;
use datafusion::physical_expr::{
    expressions::{Column, Literal},
    projection::ProjectionExpr,
    PhysicalExpr,
};
use datafusion_datasource::{
    file_groups::FileGroup,
    file_scan_config::FileScanConfig,
    projection::{ProjectionOpener, SplitProjection},
    PartitionedFile, TableSchema,
};
use datafusion_datasource_json::source::JsonSource;
use delta_kernel::tasks::{OperationFailure, Resource, ResourceExhausted, TaskLimits};
use std::mem::size_of;
use std::sync::Arc;

pub(crate) fn peak(f: &ClosedPlanFacts, limits: TaskLimits) -> Result<usize, OperationFailure> {
    let overflow = || ResourceExhausted {
        resource: Resource::MetadataAllocatedBytes,
        limit: limits.limit(Resource::MetadataAllocatedBytes),
        observed: usize::MAX,
    };
    let bytes = (|| -> Option<usize> {
        // Original provider config, physical scan config clone, FileStream's
        // cloned local VecDeque, and active opener. Charging a complete group
        // at each site also covers transient prior/next-file overlap.
        let files = vec_peak(f.files, size_of::<PartitionedFile>())?
            .checked_add(f.file_url_bytes.checked_mul(4)?)?
            .checked_add(
                f.files
                    .checked_mul(vec_peak(1, size_of::<ScalarValue>())?)?,
            )?
            .checked_add(vec_peak(1, size_of::<FileGroup>())?)?
            .checked_mul(4)?;
        // Url::clone plus set_path on the already parsed origin; from_url_path
        // uses a percent-decoded String and Path::parse String simultaneously.
        // PartitionedFile::new_from_meta now moves that Path without encoding it.
        let paths = vec_peak(f.file_url_bytes, 1)?.checked_mul(2)?;
        let headers = crate::json_scan::provider_layout()
            .size()
            .checked_add(owner::<JsonSource>())?
            .checked_add(2 * (owner::<FileScanConfig>()))?
            .checked_add(owner::<ProjectionOpener>())?
            .checked_add(owner::<SplitProjection>())?
            .checked_add(3 * size_of::<TableSchema>())?;
        // SplitProjection creates file/partition Column maps, then rewrites
        // the all-column expression list. ProjectionOpener replaces version
        // with one Int64 Literal per file. Every producer schema's fields are
        // counted, so this dominates the single source's actual root width.
        let slots = f.fields.checked_mul(f.files.checked_add(3)?)?;
        let projections = vec_peak(slots, size_of::<ProjectionExpr>())?
            .checked_add(slots.checked_mul(owner::<Column>().max(owner::<Literal>()))?)?
            .checked_add(slots.checked_mul(f.max_name_bytes)?)?
            .checked_add(vec_peak(slots, size_of::<Arc<dyn PhysicalExpr>>())?)?;
        // The two temporary std HashMaps reserve by 7/8 occupancy and round
        // buckets to a power of two. Eight buckets per entry covers growth
        // and old/new coexistence; each has separate minimum/control groups.
        let maps = slots
            .checked_add(2)?
            .checked_mul(8)?
            .checked_mul(
                size_of::<(usize, Arc<dyn PhysicalExpr>)>().max(size_of::<(usize, String)>()) + 1,
            )?
            .checked_add(4 * 16)?;
        let sort = vec_peak(f.fields, size_of::<(String, usize)>())?
            .checked_mul(2)?
            .checked_add(f.fields.checked_mul(f.max_name_bytes)?)?;
        let schema = vec_peak(f.fields, size_of::<FieldRef>())?
            .checked_mul(4)?
            .checked_add(4 * (owner::<Schema>()))?;
        files
            .checked_add(paths)?
            .checked_add(headers)?
            .checked_add(projections)?
            .checked_add(maps)?
            .checked_add(sort)?
            .checked_add(schema)
    })()
    .ok_or_else(overflow)?;
    if bytes > limits.limit(Resource::MetadataAllocatedBytes) {
        return Err(ResourceExhausted {
            resource: Resource::MetadataAllocatedBytes,
            limit: limits.limit(Resource::MetadataAllocatedBytes),
            observed: bytes,
        }
        .into());
    }
    Ok(bytes)
}
