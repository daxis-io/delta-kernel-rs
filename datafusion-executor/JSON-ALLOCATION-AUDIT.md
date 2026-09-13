# JSON admission audit

This audit is incomplete. The host execution path must remain unconnected until the array,
materialization, lowering and evaluation entries below have source-derived bounds and boundary
tests. Compiling these helpers is not an admission proof or query qualification.

The selected sources and Rust 1.97 allocation implementation are hashed in
`P/evidence/decoder-allocation-source-hashes.json`. Rust source provisioning is recorded separately
in `P/provisioning/rust197-allocation-source-01`. Arrow and object_store source are unchanged.

## Established framing and tape bounds

`json_framing` borrows a complete admitted owner. Its lexical walk does not build strings, maps,
lists or a token tape. It validates complete object documents, counts unknown fields, rejects
malformed framing and limits nesting before recursion. The recursion ceiling is the lesser of
the caller's existing SchemaDepth allowance and the existing qualification ceiling. Arrow retains
responsibility for Unicode surrogate pairing and action field types; Kernel interprets actions.

For a batch size of one, F flattened system-schema fields, B maximum complete record bytes, and
D observed nesting depth, `TapeDecoder::new` reserves:

| Owner | Initial elements | Maximum live elements | Element bytes |
|---|---:|---:|---:|
| tape elements | 2 + 2F | B + 1 | 8 |
| string/number offsets | 1 + 2F | B + 1 | sizeof(usize) |
| decoded string/number bytes | 16F | B | 1 |
| decoder stack | 10 | D + 4 | 8 |

The first tape element and offset are sentinels. Each subsequent tape element or offset consumes
at least one record byte. Unicode unescaping cannot increase the byte count. Each enclosing
container retains one state; object keys push Value, Colon, String, and an escape adds one state.
The selected TapeElement and DecoderState enum payloads fit eight bytes on the qualification
targets. TapeDecoder itself holds four Vecs and two usize fields.

Rust 1.97 `RawVec::grow_amortized` requests max(2*old, required, minimum). Thus capacity is bounded
by max(initial, 2*maximum_length, minimum), including bulk-extend jumps. Minimum capacity is 8
for byte elements and 4 for these other element widths. `json_allocation::TapeEnvelope` charges
both old and new backing during possible allocator relocation, includes fixed container storage,
uses checked arithmetic, and refuses before tape construction. This bound is only for the tape.

## Outstanding producer accounting

* `ReaderBuilder::build_decoder`: flattened-fields scratch, boxed decoder tree, struct field
  lookup maps and cloned names for structs reaching the lookup-map threshold.
* `StructArrayDecoder`: field-count times row-count positions, child decoder/result vectors,
  nullable masks, StructArray owners and RecordBatch conversion vectors.
* `MapArrayDecoder` and `ListLikeArrayDecoder`: growing child positions, offsets, child arrays,
  nullable masks and map entry StructArray owners. Child counts must derive from lexical input.
* `StringArrayDecoder`, primitive and boolean decoders: exact requested builder capacities,
  Arrow's 64-byte buffer alignment, buffer owner headers, null builders and returned array owners.
* Error paths: tape serialization and nested field-context strings coexist with decoder state.
* Kernel PM materialization: replace or establish the existing `64 * batch_bytes + 1 MiB`
  envelope using the reached visitor, serde schema/configuration and snapshot allocations.
* Kernel closed producers: establish fixed plan/node/schema and schema-dependent ScanBuilder
  envelopes from their allocation sites; existing guessed constants are not a completed proof.
* DataFusion: lowering, optimizer and physical planning scratch; in-flight aggregate/join/input
  ownership; pre-pull output/page/cumulative admission and reservations in the caller's pool.

`LogInput::load` currently establishes full-history read/input/work limits, exact version/size/EOF
validation and simultaneous whole-log ownership only. It does not create JsonSource or a decoder.
The public storage capability registration is approved, but no public table-opening implementation
is claimed by these helpers.

## Array and formatting implementation update

`json_arrays` now derives the decoder tree, positions, output buffers, flush scratch and formatting
owners for the supported system types. This supersedes the absence of array formulas in the
outstanding list above; integration with evaluation/transfer and final review remain outstanding.

Struct positions are field_count * row_count u32 entries. Map key and value positions coexist;
child positions cannot exceed the number of input bytes in a complete lexically admitted record.
Lists own their growing child positions and Vec<i32> offsets. String builders use Vec-backed
values and offsets; primitive builders use Vec-backed values. Null and Boolean builders use
64-byte aligned MutableBuffer storage. Each buffer ceiling includes possible old/new allocations
and immutable Bytes owner headers. Fixed private decoder layouts are bounded by the sum of their
source-declared fields and alignment padding, using public types' target-specific sizes.
ReaderBuilder flattened_fields constructs recursive Field::fields vectors: at most F inner vectors
and one outer vector, each bounded by F references. Decoder, ReaderBuilder and ArrowError fixed
storage is included. These bounds are intentionally separate from schema ownership transferred
with a page, which the host must retain and charge in full.

Tape::error serializes a subtree before formatting the error. Original decoded string/number
spellings plus at most one separator byte per token fit 2B bytes. The serializer's initial capacity
is 64 bytes. String growth overlap, a formatted message, and its replacement with a field context
coexist with tape and arrays. Field Display calls DataType Display, which allocates nested strings,
a Vec<String> and a joined String for Struct, a formatted field for Map, and a nonstandard child
name for List. Structural checked formulas now include every one of those owners. No Arrow
Display is invoked during preflight. Debug-name length is bounded by quotes plus the maximum
Unicode escape spelling per input byte. Empty metadata is required before this traversal.

The nested preflight test uses a thread-local allocator observer solely as a regression detector:
it verifies zero allocations for both admission at the derived peak and refusal at peak minus one.
The formulas are derived from source before construction, not fitted to observed decoder usage.
The actual Arrow formatter is invoked later only as a length oracle in a separate test.

## Task-local immutable store

`LogInput::load` performs conditional capability reads and complete lexical preflight before
`AdmittedLogStore` converts exact Vec owners to Bytes::from_owner. The selected bytes implementation
allocates Owned<Vec<u8>> (AtomicUsize plus Vec), charged before conversion. The store has no provider
handle and cannot retry a changed object. Immutable manifest identity, path and size stay bound.
Ranges, HEAD, unlisted paths and writes fail. Trace slots contain only manifest indexes/byte counts
and are reserved before use; get requests charge cumulative input/requested bytes before cloning
an immutable Bytes handle. One Once<Ready<Result<Bytes>>> stream is charged per admitted pull.

The get_opts implementation explicitly boxes Ready<Result<GetResult>>, whose target-specific
size is charged in try_new. It does not rely on an unmeasured async_trait capture frame. Metadata
path clones and typed resource-error boxes are included. Millisecond timestamps use chrono's
fallible from_timestamp_millis; SystemTime-to-DateTime's infallible conversion is not used.
Caller source/JsonSource execution futures outside this wrapper remain an evaluation-envelope
obligation, and the host must include simultaneous store/decoder/input state in its reservation.

Native evidence includes a real batch-one JsonSource execution with an empty ordinary registered
store; all bytes therefore came from the injected admitted owners. This is not the final Delta
adapter/native/browser gate. Original full task correctness and query/negative gates remain open.
