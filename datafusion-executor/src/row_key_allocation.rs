//! Arrow row payload accounting for Kernel's fixed file_action_key.
//!
//! The key has two Struct sentinels, three UTF-8 leaves and one i32 leaf.
//! DV-bearing keys must be bounded during evaluation even though the later
//! borrowed scan preflight rejects active DVs before materialization.

/// Bounds the sum of encoded key lengths from all selected records. `bytes`
/// bounds the combined decoded bytes of the three string leaves, not their
/// length in a single row. Each key chooses add/remove leaves by coalesce, so
/// the selected whole-log encoded byte count also bounds this decoded sum.
///
/// Arrow59 row/variable.rs uses 1+9*ceil(L/8) for L<=32 and
/// 4+33*ceil(L/32) otherwise. Both are <=9*L/8+36. Summing
/// over at most three strings per row gives ceil(9B/8)+108R.
/// Two Struct markers and the i32 marker+four bytes add seven bytes per row.
/// Null/empty strings each use one byte and satisfy the same inequality.
/// This excludes converter/Rows/offset/hash/array owners, which must be added
/// by the enclosing evaluation envelope; it is not a complete admission proof.
pub(crate) fn encoded_key_bytes(records: usize, bytes: usize) -> Option<usize> {
    // Calculate ceil(9B/8) without overflowing the intermediate 9B.
    let expanded = bytes
        .checked_add(bytes / 8)?
        .checked_add(usize::from(!bytes.is_multiple_of(8)))?;
    expanded.checked_add(records.checked_mul(3 * 36 + 2 + 1 + 4)?)
}

/// One reached GroupValuesRows instance, including its borrowed-schema codecs,
/// current/retained/replacement row backing, encoding scratch and emitted key
/// arrays. Partial and final aggregations each need their own instance; their
/// accumulator value/state arrays are outside this key-only envelope.
pub(crate) fn grouping_owner_peak(
    key: &datafusion::arrow::datatypes::Schema,
    records: usize,
    bytes: usize,
    limits: delta_kernel::tasks::TaskLimits,
) -> Result<usize, delta_kernel::tasks::OperationFailure> {
    use std::mem::size_of;

    use datafusion::arrow::array::{ArrayRef, StructArray};
    use datafusion::arrow::buffer::ScalarBuffer;
    use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
    use datafusion::arrow::row::{OwnedRow, Row, RowConverter, Rows, SortField};
    use datafusion::physical_plan::aggregates::group_values::GroupValuesRows;
    use delta_kernel::tasks::{OperationFailure, Resource, ResourceExhausted};

    use crate::json_arrays::vec_peak;

    // Only Kernel's concrete file_action_key, never a caller-selected group key.
    let malformed = || OperationFailure::malformed_response();
    if !key.metadata().is_empty() || key.fields().len() != 1 {
        return Err(malformed());
    }
    let outer = key.field(0);
    let DataType::Struct(fields) = outer.data_type() else {
        return Err(malformed());
    };
    if outer.name() != "file_action_key"
        || !outer.metadata().is_empty()
        || fields.len() != 2
        || fields[0].name() != "path"
        || fields[0].data_type() != &DataType::Utf8
        || fields[1].name() != "deletionVector"
    {
        return Err(malformed());
    }
    let DataType::Struct(dv) = fields[1].data_type() else {
        return Err(malformed());
    };
    if dv.len() != 3
        || dv[0].name() != "storageType"
        || dv[0].data_type() != &DataType::Utf8
        || dv[1].name() != "pathOrInlineDv"
        || dv[1].data_type() != &DataType::Utf8
        || dv[2].name() != "offset"
        || dv[2].data_type() != &DataType::Int32
        || fields
            .iter()
            .chain(dv.iter())
            .any(|f| !f.metadata().is_empty())
    {
        return Err(malformed());
    }
    let overflow = || -> OperationFailure {
        ResourceExhausted {
            resource: Resource::MetadataAllocatedBytes,
            limit: limits.limit(Resource::MetadataAllocatedBytes),
            observed: usize::MAX,
        }
        .into()
    };
    let peak = (|| -> Option<usize> {
        let word = size_of::<usize>();
        let arc = 2 * word;
        let payload = encoded_key_bytes(records, bytes)?;
        let rows = |count: usize, data: usize| -> Option<usize> {
            size_of::<Rows>()
                .checked_add(vec_peak(count.checked_add(1)?, word)?)?
                .checked_add(vec_peak(data, 1)?)
        };
        // Exact source capacities of rows_buffer are1000 rows and64000 bytes.
        // It remains live alongside retained keys and emit(First)'s replacement.
        let stored = rows(records.max(1000), payload.max(64 * 1000))?
            .checked_add(rows(records, payload)?.checked_mul(2)?)?;
        // Two nested Struct codecs produce separate temporary child Rows.
        // Each child key payload is a subset of the complete key encoding.
        let encoding = rows(records, payload)?
            .checked_mul(2)?
            .checked_add(vec_peak(records, word)?.checked_mul(3)?)?;
        // Three converters of arity1,2,3: six SortField/Codec/Encoder slots.
        // Codec's largest variant is (RowConverter,OwnedRow) or three Vecs;
        // Encoder's is (Rows,Row) or its Union fields. Add one aligned tag.
        // Include the actual enum width even though Union is never reached.
        let codec = (size_of::<RowConverter>() + size_of::<OwnedRow>())
            .max(3 * size_of::<Vec<usize>>())
            .checked_add(word)?;
        let encoder = (size_of::<Rows>() + size_of::<Row<'_>>())
            .max(
                2 * size_of::<Vec<usize>>()
                    + size_of::<ScalarBuffer<i8>>()
                    + size_of::<Option<ScalarBuffer<i32>>>(),
            )
            .checked_add(word)?;
        let converters = vec_peak(6, size_of::<SortField>())?
            .checked_mul(2)?
            .checked_add(3 * arc)?
            .checked_add(vec_peak(6, codec)?)?
            .checked_add(vec_peak(6, encoder)?)?;
        // Codec::Struct constructs two null-array sets and moves their encoded
        // buffers into two OwnedRows (including Vec->Box possible relocation).
        let templates = rows(1, encoded_key_bytes(1, 0)?)?
            .checked_mul(2)?
            .checked_add(vec_peak(encoded_key_bytes(1, 0)?, 1)?.checked_mul(2)?)?;
        // HashTable<(u64,usize)> uses <=7/8 load, power-of-two buckets and a
        // control byte per bucket. Include old/new tables, SIMD tail/alignment.
        let buckets = records
            .checked_mul(8)?
            .checked_add(6)?
            .checked_div(7)?
            .checked_next_power_of_two()?
            .max(4);
        let hash = buckets
            .checked_mul(size_of::<(u64, usize)>() + 1)?
            .checked_add(2 * 16 - 1)?
            .checked_mul(2)?;
        // intern: normalized ArrayRef Vec, hashes and caller group indices.
        // convert_rows: borrowed row-slice Vec; recursive convert_raw reuses
        // those slices and collects six result ArrayRefs across three levels.
        let scratch = vec_peak(1, size_of::<ArrayRef>())?
            // hashes_buffer plus one values_hashes Vec per nested Struct.
            .checked_add(vec_peak(records, size_of::<u64>())?.checked_mul(3)?)?
            .checked_add(vec_peak(records, word)?)?
            .checked_add(vec_peak(records, size_of::<&[u8]>())?)?
            .checked_add(vec_peak(6, size_of::<ArrayRef>())?)?;
        // decode_column corrects each Struct's child Fields: five cloned names,
        // transient Vec<Field>, Field Arcs and Vec/Arc slice of FieldRefs.
        let names = [
            "path",
            "deletionVector",
            "storageType",
            "pathOrInlineDv",
            "offset",
        ];
        let corrected = vec_peak(5, size_of::<Field>())?
            .checked_add(vec_peak(5, size_of::<FieldRef>())?.checked_mul(2)?)?
            .checked_add(5 * (size_of::<Field>() + arc))?
            .checked_add(names.iter().map(|s| s.len()).sum::<usize>())?
            .checked_add(2 * arc)?;
        // encode_array_if_necessary rebuilds both Struct wrappers and their
        // five child Arc entries even without dictionary/run-end key types.
        let rebound = (size_of::<StructArray>() + arc)
            .checked_mul(2)?
            .checked_add(vec_peak(5, size_of::<ArrayRef>())?)?;
        [
            stored,
            encoding,
            converters,
            templates,
            hash,
            scratch,
            corrected,
            rebound,
            size_of::<GroupValuesRows>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    })()
    .ok_or_else(overflow)?;
    // All emitted arrays retain full R-row backing, independent of page slices.
    // Two constructor null-array sites are additional to the ordinary output.
    let output = crate::json_arrays::output_owner_peak(key, records, bytes, limits)?;
    let nulls = crate::json_arrays::output_owner_peak(key, 1, 0, limits)?;
    let peak = peak
        .checked_add(output)
        .and_then(|n| nulls.checked_mul(2).and_then(|m| n.checked_add(m)))
        .ok_or_else(overflow)?;
    if peak > limits.limit(Resource::MetadataAllocatedBytes) {
        return Err(ResourceExhausted {
            resource: Resource::MetadataAllocatedBytes,
            limit: limits.limit(Resource::MetadataAllocatedBytes),
            observed: peak,
        }
        .into());
    }
    Ok(peak)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::array::{Array, ArrayRef, Int32Array, StringArray, StructArray};
    use datafusion::arrow::datatypes::{DataType, Field};
    use datafusion::arrow::row::{RowConverter, SortField};

    use super::*;

    #[test]
    fn key_bound_covers_arrow_block_boundaries_and_null_dv() {
        for length in [0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 1024] {
            let text = "x".repeat(length);
            let strings: ArrayRef =
                Arc::new(StringArray::from(vec![Some(text.as_str()), None, Some("")]));
            let dv = StructArray::new(
                vec![
                    Field::new("storageType", DataType::Utf8, true),
                    Field::new("pathOrInlineDv", DataType::Utf8, true),
                    Field::new("offset", DataType::Int32, true),
                ]
                .into(),
                vec![
                    strings.clone(),
                    strings.clone(),
                    Arc::new(Int32Array::from(vec![Some(1), None, Some(0)])),
                ],
                Some(vec![true, false, true].into()),
            );
            let key = StructArray::new(
                vec![
                    Field::new("path", DataType::Utf8, true),
                    Field::new("deletionVector", dv.data_type().clone(), true),
                ]
                .into(),
                vec![strings, Arc::new(dv)],
                None,
            );
            let (bound, allocations) =
                crate::json_arrays::tests::observe_allocations(|| encoded_key_bytes(3, 3 * length));
            assert_eq!(allocations, 0);
            let converter =
                RowConverter::new(vec![SortField::new(key.data_type().clone())]).unwrap();
            let rows = converter.convert_columns(&[Arc::new(key)]).unwrap();
            let encoded = rows.iter().map(|row| row.as_ref().len()).sum::<usize>();
            assert!(
                encoded <= bound.unwrap(),
                "length={length}, encoded={encoded}, bound={bound:?}"
            );
        }
    }

    #[test]
    fn grouped_key_owners_admit_before_construction_and_cover_partial_emit() {
        use datafusion::arrow::datatypes::Schema;
        use datafusion::logical_expr::EmitTo;
        use datafusion::physical_plan::aggregates::group_values::{GroupValues, GroupValuesRows};
        use delta_kernel::tasks::{FailureKind, Resource, TaskLimits};
        let paths = Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef;
        let dv = StructArray::new(
            vec![
                Field::new("storageType", DataType::Utf8, true),
                Field::new("pathOrInlineDv", DataType::Utf8, true),
                Field::new("offset", DataType::Int32, true),
            ]
            .into(),
            vec![
                paths.clone(),
                paths.clone(),
                Arc::new(Int32Array::from(vec![1, 2, 3])),
            ],
            None,
        );
        let key = Arc::new(StructArray::new(
            vec![
                Field::new("path", DataType::Utf8, true),
                Field::new("deletionVector", dv.data_type().clone(), true),
            ]
            .into(),
            vec![paths, Arc::new(dv)],
            None,
        )) as ArrayRef;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "file_action_key",
            key.data_type().clone(),
            true,
        )]));
        let limits = TaskLimits::qualification();
        let (peak, allocations) = crate::json_arrays::tests::observe_allocations(|| {
            grouping_owner_peak(&schema, 3, 9, limits)
        });
        assert_eq!(allocations, 0);
        let peak = peak.unwrap();
        assert!(grouping_owner_peak(
            &schema,
            3,
            9,
            limits.with_limit(Resource::MetadataAllocatedBytes, peak)
        )
        .is_ok());
        let (failure, allocations) = crate::json_arrays::tests::observe_allocations(|| {
            grouping_owner_peak(
                &schema,
                3,
                9,
                limits.with_limit(Resource::MetadataAllocatedBytes, peak - 1),
            )
        });
        assert_eq!(allocations, 0);
        assert!(
            matches!(failure.unwrap_err().kind(), FailureKind::ResourceExhausted(e)
            if e.resource == Resource::MetadataAllocatedBytes && e.observed == peak)
        );
        // Source formulas above establish preallocation admission. These ordinary
        // implementation diagnostics additionally exercise retained/replacement
        // Rows and decoded/corrected/rebound nested arrays on the selected pins.
        let mut groups = GroupValuesRows::try_new(schema.clone()).unwrap();
        let mut indices = Vec::new();
        groups.intern(&[key], &mut indices).unwrap();
        assert_eq!(indices, [0, 1, 2]);
        assert!(groups.size() < peak);
        assert_eq!(groups.emit(EmitTo::First(1)).unwrap()[0].len(), 1);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups.emit(EmitTo::All).unwrap()[0].len(), 2);
        assert!(groups.is_empty());
        assert!(grouping_owner_peak(&schema, usize::MAX, 0, limits).is_err());
        let wrong = Schema::new(vec![Field::new("arbitrary", DataType::Utf8, true)]);
        let (failure, allocations) = crate::json_arrays::tests::observe_allocations(|| {
            grouping_owner_peak(&wrong, 3, 9, limits)
        });
        assert_eq!(allocations, 0);
        assert!(matches!(
            failure.unwrap_err().kind(),
            FailureKind::MalformedResponse
        ));
    }

    #[test]
    fn key_size_arithmetic_rejects_overflow() {
        assert_eq!(encoded_key_bytes(0, 0), Some(0));
        assert_eq!(encoded_key_bytes(usize::MAX, 0), None);
        assert_eq!(encoded_key_bytes(1, usize::MAX), None);
    }
}
