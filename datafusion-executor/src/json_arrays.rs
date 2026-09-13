//! Array-builder ownership for the closed JSON system schemas.

use std::alloc::Layout;
use std::collections::HashMap;
use std::fmt;
use std::mem::size_of;
use std::sync::{Arc, Mutex};

use datafusion::arrow::array::{
    ArrayRef, BooleanArray, Int32Array, Int64Array, ListArray, MapArray, NullArray, RecordBatch,
    StringArray, StructArray,
};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef, Fields, Schema};
use delta_kernel::tasks::{FailureKind, OperationFailure, Resource, ResourceExhausted, TaskLimits};

use crate::json_allocation::TapeEnvelope;
use crate::json_framing::JsonFraming;

/// Complete tape, decoder tree, flush scratch, output and error-path envelope for one decoder.
/// Schema ownership is separately retained by the admitted plan and host.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DecoderEnvelope {
    pub peak: usize,
    pub output: usize,
    pub fields: usize,
}

impl DecoderEnvelope {
    /// Derives capacities without constructing a decoder or allocating a schema traversal Vec.
    /// Only the fixed producers' scalar/struct/list/map types are supported.
    pub(crate) fn preflight(
        schema: &Schema,
        framing: JsonFraming,
        limits: TaskLimits,
    ) -> Result<Self, OperationFailure> {
        let overflow = || -> OperationFailure {
            ResourceExhausted {
                resource: Resource::MetadataAllocatedBytes,
                limit: limits.limit(Resource::MetadataAllocatedBytes),
                observed: usize::MAX,
            }
            .into()
        };
        let mut budget = ArrayBudget::default();
        let root = DataType::Struct(schema.fields().clone());
        budget.visit(&root, 1, framing.max_record_bytes, 0, limits)?;
        let tape = TapeEnvelope::preflight(budget.fields, framing, limits)?;
        // flush's row-position Vec (batch size one), cloned root StructArray fields and
        // RecordBatch's column Vec coexist with the decoded root's own column Vec.
        let batch_vectors = vec_peak(schema.fields().len(), size_of::<ArrayRef>())
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(overflow)?;
        let flush = vec_peak(1, size_of::<u32>())
            .and_then(|n| n.checked_add(batch_vectors))
            .and_then(|n| n.checked_add(size_of::<RecordBatch>()))
            .ok_or_else(overflow)?;
        // ReaderBuilder::build_decoder calls flattened_fields(). Each child field is visited
        // once, and the iterator collection can grow according to RawVec's doubling rule.
        let flattened = vec_peak(budget.fields, size_of::<&Field>())
            .and_then(|n| n.checked_mul(budget.fields.checked_add(1)?))
            .ok_or_else(overflow)?;
        let errors = error_peak(
            framing.max_record_bytes,
            budget.error_context,
            budget.field_display,
        )
        .and_then(|n| n.checked_add(budget.format_scratch))
        .ok_or_else(overflow)?;
        let peak = [
            tape.peak,
            budget.decoders,
            budget.positions,
            budget.output,
            flush,
            flattened,
            errors,
            size_of::<datafusion::arrow::json::reader::Decoder>(),
            size_of::<datafusion::arrow::json::ReaderBuilder>(),
            size_of::<datafusion::arrow::error::ArrowError>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(overflow)?;
        check(Resource::MetadataAllocatedBytes, peak, limits)?;
        let output = budget.output.checked_add(flush).ok_or_else(overflow)?;
        check(Resource::EvaluationPageBytes, output, limits)?;
        Ok(Self {
            peak,
            output,
            fields: budget.fields,
        })
    }
}

/// One complete array tree with at most `rows` root rows and `bytes` variable
/// payload/child slots per field. This includes full backing, even when the
/// eventual RecordBatch is sliced to one visible row. Pipeline simultaneity,
/// scalar/aggregate owners and schema storage are admitted separately.
///
/// Reuses decoder output owners and additionally covers Arrow row decoding and
/// copy_array_data's aligned MutableBuffer/BufferBuilder allocations. Every
/// supported node has at most three buffers (validity, offsets, values). The
/// extra round-up is at most 63 bytes per buffer and each old/new growth peak
/// contains at most four times the requested size. This term is alignment
/// arithmetic from MutableBuffer, not an observed or empirical multiplier.
pub(crate) fn output_owner_peak(
    schema: &Schema,
    rows: usize,
    bytes: usize,
    limits: TaskLimits,
) -> Result<usize, OperationFailure> {
    output_type_owner_peak(
        &DataType::Struct(schema.fields().clone()),
        rows,
        bytes,
        limits,
    )
}

/// Borrowed array type entrypoint: no temporary Schema/Field/Vec owners.
pub(crate) fn output_type_owner_peak(
    data_type: &DataType,
    rows: usize,
    bytes: usize,
    limits: TaskLimits,
) -> Result<usize, OperationFailure> {
    let overflow = || {
        OperationFailure::from(ResourceExhausted {
            resource: Resource::MetadataAllocatedBytes,
            limit: limits.limit(Resource::MetadataAllocatedBytes),
            observed: usize::MAX,
        })
    };
    let mut budget = ArrayBudget::default();
    budget.visit(data_type, rows, bytes, 0, limits)?;
    let alignment = budget
        .fields
        .checked_add(1)
        .and_then(|fields| fields.checked_mul(3 * 4 * (64 - 1)))
        .ok_or_else(overflow)?;
    let columns = match data_type {
        DataType::Struct(fields) => fields.len(),
        _ => 1,
    };
    let columns = vec_peak(columns, size_of::<ArrayRef>()).ok_or_else(overflow)?;
    let peak = budget
        .output
        .checked_add(alignment)
        .and_then(|n| n.checked_add(columns))
        .and_then(|n| n.checked_add(size_of::<RecordBatch>()))
        .ok_or_else(overflow)?;
    check(Resource::MetadataAllocatedBytes, peak, limits)?;
    Ok(peak)
}

/// Reached MutableArrayData copy/concatenation owners for admitted system
/// arrays, plus ScalarValue::compact's recursive container reconstruction.
/// Input buffer payloads remain charged by their existing owners. `inputs`
/// bounds distinct input ArrayData trees (one for compact, <=R for concat).
pub(crate) fn copy_owner_peak(
    schema: &Schema,
    rows: usize,
    bytes: usize,
    inputs: usize,
    limits: TaskLimits,
) -> Result<usize, OperationFailure> {
    copy_type_owner_peak(
        &DataType::Struct(schema.fields().clone()),
        rows,
        bytes,
        inputs,
        limits,
    )
}

/// Same reached copy/compact owners for a borrowed value subtree.
pub(crate) fn copy_type_owner_peak(
    data_type: &DataType,
    rows: usize,
    bytes: usize,
    inputs: usize,
    limits: TaskLimits,
) -> Result<usize, OperationFailure> {
    mutable_type_owner_peak(data_type, rows, bytes, inputs, true, limits)
}

/// CASE merge reaches MutableArrayData directly, then freeze/make_array.
/// It does not reach ScalarValue::compact or the specialized concat dispatch.
/// Borrowed input payloads are retained separately; the two input ArrayData
/// trees and all new output/scratch owners are included here.
pub(crate) fn merge_type_owner_peak(
    data_type: &DataType,
    rows: usize,
    bytes: usize,
    limits: TaskLimits,
) -> Result<usize, OperationFailure> {
    mutable_type_owner_peak(data_type, rows, bytes, 2, false, limits)
}

fn mutable_type_owner_peak(
    data_type: &DataType,
    rows: usize,
    bytes: usize,
    inputs: usize,
    compact: bool,
    limits: TaskLimits,
) -> Result<usize, OperationFailure> {
    use datafusion::arrow::array::{ArrayData, MutableArrayData};
    use datafusion::arrow::buffer::Buffer;
    if inputs == 0 {
        return Err(OperationFailure::malformed_response());
    }
    let overflow = || -> OperationFailure {
        ResourceExhausted {
            resource: Resource::MetadataAllocatedBytes,
            limit: limits.limit(Resource::MetadataAllocatedBytes),
            observed: usize::MAX,
        }
        .into()
    };
    // All reached variable-width offsets are i32. Establish this before
    // copy_array_data's infallible expect around try_extend is reachable.
    let capacity = rows.max(bytes);
    if capacity > i32::MAX as usize {
        return Err(ResourceExhausted {
            resource: Resource::MetadataAllocatedBytes,
            limit: limits
                .limit(Resource::MetadataAllocatedBytes)
                .min(i32::MAX as usize),
            observed: capacity,
        }
        .into());
    }
    let mut budget = ArrayBudget::default();
    budget.visit(data_type, rows, capacity, 0, limits)?;
    let nodes = budget.fields.checked_add(1).ok_or_else(overflow)?;
    let output = output_type_owner_peak(data_type, rows, capacity, limits)?;
    let scratch = (|| -> Option<usize> {
        let word = size_of::<usize>();
        // to_data input trees, freeze's output tree and ArrayDataBuilder's
        // possible child-vector relocation. ArrayData owns up to two Buffer
        // entries per node; its validity Buffer is inline in the public type.
        let data = vec_peak(inputs.checked_add(2)?, size_of::<ArrayData>())?
            .checked_add(vec_peak(2, size_of::<Buffer>())?.checked_mul(inputs.checked_add(2)?)?)?;
        // One input-ref Vec and two boxed-closure Vecs per node. Selected
        // variable_size::build_extend captures two borrowed slices (four
        // words); null-bit closure captures &[u8] and &NullBuffer (three).
        // Primitive/list/boolean/struct alternatives have smaller captures.
        let closures = vec_peak(inputs, size_of::<&ArrayData>())?
            .checked_add(vec_peak(inputs, 2 * word)?.checked_mul(2)?)?
            .checked_add(inputs.checked_mul((4 + 3) * word)?)?
            // extend_nulls boxes one function pointer independent of inputs.
            .checked_add(word)?;
        // Count every MutableArrayData in its parent's growing child Vec.
        // The root can be stack-owned; including it keeps this composable with
        // a boxed future frame and bounds all one-child Map/List minima.
        let mutable = vec_peak(1, size_of::<MutableArrayData<'_>>())?;
        // arrow-select::concat dispatches Struct/Map/List to specialized
        // paths rather than MutableArrayData: typed input refs, borrowed child
        // arrays, optional sliced child owners and their child Arc vectors.
        // Across a tree there is at most one child edge per non-root node.
        let sliced_header = shared_owner_bytes::<StructArray>()
            .max(shared_owner_bytes::<MapArray>())
            .max(shared_owner_bytes::<ListArray>());
        let concat = if compact {
            vec_peak(inputs, word)?
                .checked_add(vec_peak(
                    inputs,
                    size_of::<&dyn datafusion::arrow::array::Array>(),
                )?)?
                .checked_add(vec_peak(inputs, sliced_header)?)?
                .checked_add(vec_peak(inputs, size_of::<ArrayRef>())?)?
        } else {
            0
        };
        data.checked_add(closures)?
            .checked_add(mutable)?
            .checked_add(concat)?
            .checked_mul(nodes)
    })()
    .ok_or_else(overflow)?;
    // Three reached tree owners: copied output, compact_view_buffers' rebuilt
    // Struct/Map/List containers, and Arc::make_mut's old-array clone before
    // assignment. Payload sharing makes this conservative; no buffer-size
    // observation is used for admission. new_buffers propagates root capacity
    // into collection children even when their actual value count is zero,
    // hence max(rows,bytes) above rather than decoded child length alone.
    // Direct merge has just its output tree: freeze moves buffer ownership
    // into ArrayData and make_array consumes it. compact additionally rebuilds
    // the two container trees described above.
    let output_trees = if compact { 3 } else { 1 };
    let peak = output
        .checked_mul(output_trees)
        .and_then(|n| n.checked_add(scratch))
        .ok_or_else(overflow)?;
    check(Resource::MetadataAllocatedBytes, peak, limits)?;
    Ok(peak)
}

#[derive(Default)]
struct ArrayBudget {
    decoders: usize,
    positions: usize,
    output: usize,
    fields: usize,
    error_context: usize,
    field_display: usize,
    format_scratch: usize,
}

impl ArrayBudget {
    fn visit(
        &mut self,
        data_type: &DataType,
        rows: usize,
        bytes: usize,
        depth: usize,
        limits: TaskLimits,
    ) -> Result<(), OperationFailure> {
        check(Resource::SchemaDepth, depth, limits)?;
        let overflow = || -> OperationFailure {
            ResourceExhausted {
                resource: Resource::MetadataAllocatedBytes,
                limit: limits.limit(Resource::MetadataAllocatedBytes),
                observed: usize::MAX,
            }
            .into()
        };
        let nullable = bitmap(rows).ok_or_else(overflow)?;
        let (decoder, output, positions) = match data_type {
            DataType::Struct(fields) => {
                let fixed = layout_sum(&[
                    Layout::new::<DataType>(),
                    Layout::new::<Vec<Box<dyn fmt::Debug>>>(),
                    Layout::new::<[bool; 3]>(),
                    Layout::new::<usize>(),
                    Layout::new::<Option<HashMap<String, usize>>>(),
                    Layout::new::<Vec<u32>>(),
                    Layout::new::<usize>(),
                ])
                .ok_or_else(overflow)?;
                let lookup = if fields.len() < 16 {
                    0
                } else {
                    let strings = fields
                        .iter()
                        .try_fold(0usize, |n, f| n.checked_add(f.name().len()))
                        .ok_or_else(overflow)?;
                    hash_table(fields.len(), size_of::<(String, usize)>())
                        .and_then(|n| n.checked_add(strings))
                        .ok_or_else(overflow)?
                };
                let decoder = fixed
                    .checked_add(lookup)
                    .and_then(|n| {
                        n.checked_add(vec_peak(fields.len(), size_of::<Box<dyn fmt::Debug>>())?)
                    })
                    .ok_or_else(overflow)?;
                let positions = rows
                    .checked_mul(fields.len())
                    .and_then(|n| vec_peak(n, size_of::<u32>()))
                    .ok_or_else(overflow)?;
                let output = arc_bytes::<StructArray>()
                    .and_then(|n| n.checked_add(nullable))
                    .and_then(|n| n.checked_add(vec_peak(fields.len(), size_of::<ArrayRef>())?))
                    .ok_or_else(overflow)?;
                for field in fields {
                    self.field(field, rows, bytes, depth + 1, limits)?;
                }
                (decoder, output, positions)
            }
            DataType::Map(entries, false) => {
                let DataType::Struct(fields) = entries.data_type() else {
                    return Err(unsupported());
                };
                if fields.len() != 2 {
                    return Err(unsupported());
                }
                // Map owns both growing key/value positions, its offsets and an entries
                // StructArray. Recursing through entries also overcharges a struct decoder.
                let positions = vec_peak(bytes.max(rows), size_of::<u32>())
                    .and_then(|n| n.checked_mul(2))
                    .ok_or_else(overflow)?;
                let offsets = rows
                    .checked_add(1)
                    .and_then(|n| n.checked_mul(size_of::<i32>()))
                    .and_then(aligned_buffer)
                    .ok_or_else(overflow)?;
                let output = arc_bytes::<MapArray>()
                    .and_then(|n| n.checked_add(nullable))
                    .and_then(|n| n.checked_add(offsets))
                    .ok_or_else(overflow)?;
                self.field(entries, bytes, bytes, depth + 1, limits)?;
                let decoder = layout_sum(&[
                    Layout::new::<FieldRef>(),
                    Layout::new::<Fields>(),
                    Layout::new::<[Box<dyn fmt::Debug>; 2]>(),
                    Layout::new::<[bool; 3]>(),
                ])
                .ok_or_else(overflow)?;
                (decoder, output, positions)
            }
            DataType::List(field) => {
                let positions = vec_peak(bytes.max(rows), size_of::<u32>()).ok_or_else(overflow)?;
                // List offsets are Vec<i32>, not an aligned BufferBuilder.
                let offsets = rows
                    .checked_add(1)
                    .and_then(|n| vec_buffer(n, size_of::<i32>()))
                    .ok_or_else(overflow)?;
                let output = arc_bytes::<ListArray>()
                    .and_then(|n| n.checked_add(nullable))
                    .and_then(|n| n.checked_add(offsets))
                    .ok_or_else(overflow)?;
                self.field(field, bytes, bytes, depth + 1, limits)?;
                let decoder = layout_sum(&[
                    Layout::new::<FieldRef>(),
                    Layout::new::<Box<dyn fmt::Debug>>(),
                    Layout::new::<[bool; 2]>(),
                ])
                .ok_or_else(overflow)?;
                (decoder, output, positions)
            }
            DataType::Utf8 => {
                // StringArrayDecoder computes the exact string byte sum before creating
                // Vec-backed value/offset builders. Coercion is off for this path.
                let values = vec_buffer(bytes, 1).ok_or_else(overflow)?;
                let offsets = rows
                    .checked_add(1)
                    .and_then(|n| vec_buffer(n, size_of::<i32>()))
                    .ok_or_else(overflow)?;
                let output = arc_bytes::<StringArray>()
                    .and_then(|n| n.checked_add(nullable))
                    .and_then(|n| n.checked_add(values))
                    .and_then(|n| n.checked_add(offsets))
                    .ok_or_else(overflow)?;
                (size_of::<[bool; 2]>(), output, 0)
            }
            DataType::Int32 | DataType::Int64 => {
                let (width, owner) = if data_type == &DataType::Int32 {
                    (size_of::<i32>(), arc_bytes::<Int32Array>())
                } else {
                    (size_of::<i64>(), arc_bytes::<Int64Array>())
                };
                let output = vec_buffer(rows, width)
                    .and_then(|n| n.checked_add(nullable))
                    .and_then(|n| n.checked_add(owner?))
                    .ok_or_else(overflow)?;
                let decoder = layout_sum(&[Layout::new::<DataType>(), Layout::new::<bool>()])
                    .ok_or_else(overflow)?;
                (decoder, output, 0)
            }
            DataType::Boolean => (
                size_of::<bool>(),
                bitmap(rows)
                    .and_then(|n| n.checked_add(nullable))
                    .and_then(|n| n.checked_add(arc_bytes::<BooleanArray>()?))
                    .ok_or_else(overflow)?,
                0,
            ),
            DataType::Null => (
                size_of::<bool>(),
                arc_bytes::<NullArray>().ok_or_else(overflow)?,
                0,
            ),
            _ => return Err(unsupported()),
        };
        self.decoders = self.decoders.checked_add(decoder).ok_or_else(overflow)?;
        self.output = self.output.checked_add(output).ok_or_else(overflow)?;
        self.positions = self.positions.checked_add(positions).ok_or_else(overflow)?;
        Ok(())
    }

    fn field(
        &mut self,
        field: &Field,
        rows: usize,
        bytes: usize,
        depth: usize,
        limits: TaskLimits,
    ) -> Result<(), OperationFailure> {
        if !field.metadata().is_empty() {
            return Err(unsupported());
        }
        let exhausted = || ResourceExhausted {
            resource: Resource::SchemaNodes,
            limit: limits.limit(Resource::SchemaNodes),
            observed: usize::MAX,
        };
        self.fields = self.fields.checked_add(1).ok_or_else(exhausted)?;
        check(Resource::SchemaNodes, self.fields, limits)?;
        self.error_context = self
            .error_context
            .checked_add("whilst decoding field '': ".len())
            .and_then(|n| n.checked_add(field.name().len()))
            .ok_or_else(exhausted)?;
        self.visit(field.data_type(), rows, bytes, depth, limits)?;
        // Never invoke Arrow Display here: nested DataType Display allocates Strings,
        // a Vec<String>, and a join buffer even when writing to a counting sink.
        let display = field_format(field).ok_or_else(exhausted)?;
        self.field_display = self.field_display.max(display.len);
        self.format_scratch = self.format_scratch.max(display.scratch);
        Ok(())
    }
}

/// Length and all intermediate heap storage of the selected Arrow formatting implementation.
/// Called only after the supported-type, metadata, depth and node checks above.
#[derive(Clone, Copy)]
struct FormatBound {
    len: usize,
    scratch: usize,
}

fn name_debug_len(name: &str) -> Option<usize> {
    // Debug string quotes plus at most ten ASCII bytes per input byte (\u{10ffff}).
    // This includes escaped controls, quotes and non-printing Unicode without formatting.
    name.len().checked_mul("\\u{10ffff}".len())?.checked_add(2)
}

fn field_format(field: &Field) -> Option<FormatBound> {
    let child = type_format(field.data_type())?;
    // Field Display also supports a nonzero i64 dictionary ID and ordered marker.
    // Include their maximum spellings even though closed producers use default IDs.
    let dict = ", dict_id: "
        .len()
        .checked_add("-9223372036854775808".len())?;
    let len = "Field { :  }"
        .len()
        .checked_add(name_debug_len(field.name())?)?
        .checked_add("nullable ".len())?
        .checked_add(child.len)?
        .checked_add(dict)?
        .checked_add(", dict_is_ordered".len())?;
    Some(FormatBound {
        len,
        scratch: child.scratch.checked_add(vec_peak(dict, 1)?)?,
    })
}

fn nested_field_format(field: &Field) -> Option<FormatBound> {
    let child = type_format(field.data_type())?;
    let len = name_debug_len(field.name())?
        .checked_add(": non-null ".len())?
        .checked_add(child.len)?;
    // format_field owns a formatted String while recursively formatting its DataType.
    // Empty FormatMetadata writes nothing and its String has zero capacity.
    Some(FormatBound {
        len,
        scratch: child.scratch.checked_add(vec_peak(len, 1)?)?,
    })
}

fn type_format(data_type: &DataType) -> Option<FormatBound> {
    let scalar = match data_type {
        DataType::Null => Some("Null"),
        DataType::Boolean => Some("Boolean"),
        DataType::Int32 => Some("Int32"),
        DataType::Int64 => Some("Int64"),
        DataType::Utf8 => Some("Utf8"),
        _ => None,
    };
    if let Some(name) = scalar {
        return Some(FormatBound {
            len: name.len(),
            scratch: 0,
        });
    }
    match data_type {
        DataType::Struct(fields) => {
            let mut joined = 0usize;
            let mut scratch = vec_peak(fields.len(), size_of::<String>())?;
            for (i, field) in fields.iter().enumerate() {
                let child = nested_field_format(field)?;
                joined = joined
                    .checked_add(child.len)?
                    .checked_add(if i == 0 { 0 } else { 2 })?;
                // Summing recursive peaks covers earlier retained formatted siblings too.
                scratch = scratch.checked_add(child.scratch)?;
            }
            // join allocates a second String while every formatted child is still live.
            Some(FormatBound {
                len: "Struct()".len().checked_add(joined)?,
                scratch: scratch.checked_add(vec_peak(joined, 1)?)?,
            })
        }
        DataType::Map(field, false) => {
            let child = nested_field_format(field)?;
            Some(FormatBound {
                len: "Map(, unsorted)".len().checked_add(child.len)?,
                scratch: child.scratch,
            })
        }
        DataType::List(field) => {
            let child = type_format(field.data_type())?;
            let name = if field.name() == "item" {
                0
            } else {
                ", field: ''".len().checked_add(field.name().len())?
            };
            Some(FormatBound {
                len: "List(non-null )"
                    .len()
                    .checked_add(child.len)?
                    .checked_add(name)?,
                scratch: child.scratch.checked_add(if name == 0 {
                    0
                } else {
                    vec_peak(name, 1)?
                })?,
            })
        }
        _ => None,
    }
}

fn layout_sum(fields: &[Layout]) -> Option<usize> {
    // Summing each field rounded independently and a final alignment pad bounds Rust's
    // field reordering without relying on a private decoder's compiler-specific offsets.
    let alignment = fields.iter().map(Layout::align).max().unwrap_or(1);
    fields.iter().try_fold(alignment - 1, |n, f| {
        n.checked_add(f.size())?.checked_add(f.align() - 1)
    })
}

#[repr(C)]
struct SharedOwner<T> {
    strong: usize,
    weak: usize,
    value: T,
}
pub(crate) const fn shared_owner_bytes<T>() -> usize {
    size_of::<SharedOwner<T>>()
}
fn arc_bytes<T>() -> Option<usize> {
    Some(shared_owner_bytes::<T>())
}

fn buffer_owner() -> Option<usize> {
    // Arrow Bytes: pointer, length, Deallocation (at most Arc<dyn Allocation> + usize),
    // optional pool Mutex<Option<Box<dyn MemoryReservation>>> and two Arc counters.
    layout_sum(&[
        Layout::new::<usize>(),
        Layout::new::<usize>(),
        Layout::new::<(Arc<dyn fmt::Debug>, usize)>(),
        Layout::new::<Mutex<Option<Box<dyn fmt::Debug>>>>(),
        Layout::new::<[usize; 2]>(),
    ])
}

pub(crate) fn vec_peak(elements: usize, width: usize) -> Option<usize> {
    let minimum = if width == 1 { 8 } else { 4 };
    elements
        .checked_mul(2)?
        .max(minimum)
        .checked_mul(width)?
        .checked_mul(2)
}
fn vec_buffer(elements: usize, width: usize) -> Option<usize> {
    vec_peak(elements, width)?.checked_add(buffer_owner()?)
}
fn aligned_buffer(bytes: usize) -> Option<usize> {
    // MutableBuffer rounds requests to 64 bytes; reserve doubles the previous capacity.
    // Include a possible old and new allocation and the immutable Bytes owner.
    let aligned = bytes.checked_add(63)? / 64 * 64;
    aligned
        .checked_mul(2)?
        .checked_mul(2)?
        .checked_add(buffer_owner()?)
}
fn bitmap(rows: usize) -> Option<usize> {
    aligned_buffer(rows.checked_add(7)? / 8)
}
fn hash_table(entries: usize, width: usize) -> Option<usize> {
    let buckets = if entries < 4 {
        4
    } else if entries < 8 {
        8
    } else if entries < 15 {
        16
    } else {
        (entries.checked_mul(8)? / 7).checked_next_power_of_two()?
    };
    // Hashbrown 0.17.1 uses <=16-byte groups on these targets. With (String,usize)
    // buckets, bucket bytes already meet control alignment; include trailing group bytes.
    buckets
        .checked_mul(width)?
        .checked_add(buckets)?
        .checked_add(16)
}
fn error_peak(bytes: usize, context: usize, field_display: usize) -> Option<usize> {
    // Tape serialization retains original spellings and can add one separator space per
    // token. There are no more tokens than input bytes. Then Tape::error formats another
    // String while that serialized String is still owned.
    let serialized = bytes.checked_add(bytes)?;
    let tape_error = serialized
        .checked_add("expected  got ".len())?
        .checked_add("field value".len())?;
    let primitive_error = bytes.checked_add("failed to parse \"\" as Int64".len())?;
    let null_error = field_display
        .checked_add("Encountered unmasked nulls in non-nullable StructArray child: ".len())?;
    let message = tape_error
        .max(primitive_error)
        .max(null_error)
        .checked_add(context)?;
    // Both old and new formatted messages can coexist while a field context is added.
    // vec_peak already includes String's growth and reallocation overlap.
    vec_peak(serialized.max(64), 1)?.checked_add(vec_peak(message, 1)?.checked_mul(2)?)
}
fn check(resource: Resource, observed: usize, limits: TaskLimits) -> Result<(), OperationFailure> {
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
fn unsupported() -> OperationFailure {
    OperationFailure::new(
        FailureKind::Engine,
        delta_kernel::Error::unsupported("JSON task decoder requires a closed system schema"),
    )
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) mod tests {
    use super::*;
    use std::alloc::{GlobalAlloc, System};
    use std::cell::Cell;

    // Regression detector only, not the allocation envelope: the latter is derived above
    // before any decoder construction. Thread-local counting isolates concurrent lib tests.
    struct ObservedAllocator;
    thread_local! {
        static OBSERVE: Cell<bool> = const { Cell::new(false) };
        static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    }
    fn allocated() {
        let _ = OBSERVE.try_with(|active| {
            if active.get() {
                let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
            }
        });
    }
    unsafe impl GlobalAlloc for ObservedAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            allocated();
            unsafe { System.alloc(layout) }
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            allocated();
            unsafe { System.alloc_zeroed(layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            allocated();
            unsafe { System.realloc(ptr, layout, size) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }
    #[global_allocator]
    static ALLOCATOR: ObservedAllocator = ObservedAllocator;

    pub(crate) fn observe_allocations<T>(operation: impl FnOnce() -> T) -> (T, usize) {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                OBSERVE.with(|active| active.set(false));
            }
        }
        ALLOCATIONS.with(|count| count.set(0));
        OBSERVE.with(|active| assert!(!active.replace(true), "nested allocation observation"));
        let reset = Reset;
        let result = operation();
        drop(reset);
        (result, ALLOCATIONS.with(Cell::get))
    }

    fn nested_schema() -> Schema {
        let entries = Field::new(
            "entries",
            DataType::Struct(
                vec![
                    Field::new("key", DataType::Utf8, false),
                    Field::new(
                        "value",
                        DataType::List(Arc::new(Field::new("named child", DataType::Int64, true))),
                        true,
                    ),
                ]
                .into(),
            ),
            false,
        );
        Schema::new(vec![Field::new(
            "root",
            DataType::Struct(
                vec![
                    Field::new("map", DataType::Map(Arc::new(entries), false), true),
                    Field::new("\n\"\\\u{1f}", DataType::Boolean, false),
                ]
                .into(),
            ),
            false,
        )])
    }

    #[test]
    fn copy_preflight_bounds_null_collection_initial_capacity_without_allocating() {
        use datafusion::arrow::array::Array;
        let schema = nested_schema();
        let limits = TaskLimits::qualification();
        let (peak, allocations) =
            observe_allocations(|| copy_owner_peak(&schema, 64, 0, 1, limits));
        assert_eq!(allocations, 0);
        let peak = peak.unwrap();
        assert!(copy_owner_peak(
            &schema,
            64,
            0,
            1,
            limits.with_limit(Resource::MetadataAllocatedBytes, peak)
        )
        .is_ok());
        let (error, allocations) = observe_allocations(|| {
            copy_owner_peak(
                &schema,
                64,
                0,
                1,
                limits.with_limit(Resource::MetadataAllocatedBytes, peak - 1),
            )
        });
        assert_eq!(allocations, 0);
        assert!(
            matches!(error.unwrap_err().kind(), FailureKind::ResourceExhausted(e)
            if e.resource == Resource::MetadataAllocatedBytes && e.observed == peak)
        );
        // MutableArrayData propagates64 root capacity into this empty map/list
        // tree. Exercise the actual selected copy and scalar-compact paths;
        // buffer-size diagnostics supplement the source derivation above.
        let array = datafusion::arrow::array::new_null_array(
            &DataType::Struct(schema.fields().clone()),
            64,
        );
        let copied = datafusion::common::scalar::copy_array_data(&array.to_data());
        let copied = datafusion::arrow::array::make_array(copied);
        assert_eq!(copied.len(), 64);
        assert_eq!(copied.null_count(), 64);
        assert!(copied.get_array_memory_size() < peak);
        let mut scalar = datafusion::common::ScalarValue::try_from_array(&copied, 0).unwrap();
        scalar.compact();
        assert!(scalar.is_null());
        let slices = [array.slice(0, 32), array.slice(32, 32)];
        let joined =
            datafusion::arrow::compute::concat(&[slices[0].as_ref(), slices[1].as_ref()]).unwrap();
        assert_eq!(joined.len(), 64);
        assert!(
            joined.get_array_memory_size() < copy_owner_peak(&schema, 64, 0, 2, limits).unwrap()
        );
        for (rows, bytes, inputs) in [
            (0, 0, 0),
            (usize::MAX, 0, 1),
            (1, usize::MAX, 1),
            (1, 0, usize::MAX),
        ] {
            let (error, allocations) =
                observe_allocations(|| copy_owner_peak(&schema, rows, bytes, inputs, limits));
            assert_eq!(allocations, 0);
            assert!(error.is_err());
        }
    }

    #[test]
    fn one_row_merge_preflight_and_filter_paths() {
        use datafusion::arrow::array::{new_null_array, Array};
        use datafusion::arrow::compute::{filter_record_batch, kernels::merge::merge};
        let schema = Arc::new(nested_schema());
        let data_type = DataType::Struct(schema.fields().clone());
        let limits = TaskLimits::qualification();
        let (bound, allocations) =
            observe_allocations(|| merge_type_owner_peak(&data_type, 1, 100, limits));
        assert_eq!(allocations, 0);
        let bound = bound.unwrap();
        for (limit, allowed) in [(bound, true), (bound - 1, false), (0, false)] {
            let (result, allocations) = observe_allocations(|| {
                merge_type_owner_peak(
                    &data_type,
                    1,
                    100,
                    limits.with_limit(Resource::MetadataAllocatedBytes, limit),
                )
            });
            assert_eq!(allocations, 0);
            assert_eq!(result.is_ok(), allowed);
        }
        assert!(bound < copy_type_owner_peak(&data_type, 1, 100, 2, limits).unwrap());
        let batch_schema = Arc::new(Schema::new(
            schema
                .fields()
                .iter()
                .map(|field| field.as_ref().clone().with_nullable(true))
                .collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(
            batch_schema,
            vec![new_null_array(schema.field(0).data_type(), 1)],
        )
        .unwrap();
        // For zero/one rows the only reachable Arrow strategies are All/None.
        // All slices; None constructs empty headers. merge still copies its
        // selected nested value, even though its mask has just one bit.
        for selected in [false, true] {
            let mask = BooleanArray::from(vec![selected]);
            let filtered = filter_record_batch(&batch, &mask).unwrap();
            assert_eq!(filtered.num_rows(), usize::from(selected));
            let other = filter_record_batch(&batch, &BooleanArray::from(vec![!selected])).unwrap();
            let truthy = StructArray::from(filtered);
            let falsy = StructArray::from(other);
            let merged = merge(&mask, &truthy, &falsy).unwrap();
            assert_eq!(merged.len(), 1);
            assert!(merged.get_array_memory_size() < bound);
        }
        let (error, allocations) =
            observe_allocations(|| merge_type_owner_peak(&data_type, 1, usize::MAX, limits));
        assert_eq!(allocations, 0);
        assert!(error.is_err());
    }

    #[test]
    fn nested_preflight_and_exact_boundary_never_format_or_allocate() {
        let schema = nested_schema();
        let framing = JsonFraming {
            records: 1,
            tokens: 20,
            max_record_bytes: 100,
            max_depth: 5,
        };
        let limits = TaskLimits::qualification();
        ALLOCATIONS.with(|n| n.set(0));
        OBSERVE.with(|active| active.set(true));
        let envelope = DecoderEnvelope::preflight(&schema, framing, limits);
        OBSERVE.with(|active| active.set(false));
        assert_eq!(ALLOCATIONS.with(Cell::get), 0);
        let envelope = envelope.unwrap();
        assert!(envelope.peak > envelope.output);
        assert_eq!(envelope.fields, 7);
        for (limit, allowed) in [(envelope.peak, true), (envelope.peak - 1, false)] {
            ALLOCATIONS.with(|n| n.set(0));
            OBSERVE.with(|active| active.set(true));
            let result = DecoderEnvelope::preflight(
                &schema,
                framing,
                limits.with_limit(Resource::MetadataAllocatedBytes, limit),
            );
            OBSERVE.with(|active| active.set(false));
            assert_eq!(ALLOCATIONS.with(Cell::get), 0);
            assert_eq!(result.is_ok(), allowed);
            if let Err(failure) = result {
                assert!(matches!(failure.kind(), FailureKind::ResourceExhausted(e)
                    if e.resource == Resource::MetadataAllocatedBytes && e.observed == envelope.peak));
            }
        }
    }

    #[test]
    fn full_backing_array_owner_preflight_is_allocation_free_at_boundary() {
        let schema = nested_schema();
        let limits = TaskLimits::qualification();
        let (peak, allocations) =
            observe_allocations(|| output_owner_peak(&schema, 128, 4096, limits));
        assert_eq!(allocations, 0);
        let peak = peak.unwrap();
        for (limit, expected) in [(peak, true), (peak - 1, false)] {
            let (result, allocations) = observe_allocations(|| {
                output_owner_peak(
                    &schema,
                    128,
                    4096,
                    limits.with_limit(Resource::MetadataAllocatedBytes, limit),
                )
            });
            assert_eq!(allocations, 0);
            assert_eq!(result.is_ok(), expected);
        }
        assert!(output_owner_peak(&schema, usize::MAX, 4096, limits).is_err());
    }

    #[test]
    fn derived_format_lengths_cover_actual_nested_arrow_messages() {
        let schema = nested_schema();
        let field = &schema.fields()[0];
        let bound = field_format(field).unwrap();
        // Oracle formatting deliberately runs only after the bound has been computed.
        assert!(bound.len >= field.to_string().len());
        let nested = type_format(field.data_type()).unwrap();
        assert!(nested.len >= field.data_type().to_string().len());
        assert!(nested.scratch > vec_peak(nested.len, 1).unwrap());
        assert!(name_debug_len("\u{1f}\n\"").unwrap() >= format!("{:?}", "\u{1f}\n\"").len());
    }

    #[test]
    fn array_arithmetic_overflow_and_unsupported_types_refuse() {
        assert!(vec_peak(usize::MAX, 8).is_none());
        assert!(aligned_buffer(usize::MAX).is_none());
        assert!(error_peak(usize::MAX, 0, 0).is_none());
        let schema = Schema::new(vec![Field::new("unsupported", DataType::Float64, true)]);
        assert!(DecoderEnvelope::preflight(
            &schema,
            JsonFraming::default(),
            TaskLimits::qualification()
        )
        .is_err());
    }
}
