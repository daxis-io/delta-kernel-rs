//! Engine infrastructure shared by `Engine` implementations.
//!
//! The default Arrow/Tokio engine lives in the separate `delta_kernel_default_engine` crate.
//! `SyncEngine` is included only in test builds.

#[cfg(all(
    feature = "arrow-expression",
    any(feature = "arrow-58", feature = "arrow-59")
))]
use delta_kernel_derive::internal_api;

#[cfg(all(
    feature = "arrow-expression",
    any(feature = "arrow-58", feature = "arrow-59")
))]
use crate::parquet::arrow::arrow_reader::ArrowReaderOptions;
#[cfg(all(
    feature = "arrow-expression",
    any(feature = "arrow-58", feature = "arrow-59")
))]
use crate::parquet::arrow::arrow_writer::ArrowWriterOptions;

/// Returns the standard [`ArrowReaderOptions`] for all default engine parquet reads.
///
/// Skipping the embedded Arrow IPC schema avoids dependence on Arrow-specific metadata and
/// ensures that type resolution is driven by the kernel schema rather than the file's schema.
#[cfg(all(
    feature = "arrow-expression",
    any(feature = "arrow-58", feature = "arrow-59")
))]
#[internal_api]
pub(crate) fn reader_options() -> ArrowReaderOptions {
    ArrowReaderOptions::new().with_skip_arrow_metadata(true)
}

/// Returns the standard [`ArrowWriterOptions`] for all kernel parquet writes.
///
/// Omitting the Arrow IPC schema from the file metadata keeps Delta files interoperable with
/// non-Arrow readers and avoids encoding Arrow-specific type information.
#[cfg(all(
    feature = "arrow-expression",
    any(feature = "arrow-58", feature = "arrow-59")
))]
#[internal_api]
pub(crate) fn writer_options() -> ArrowWriterOptions {
    ArrowWriterOptions::new().with_skip_arrow_metadata(true)
}

#[cfg(feature = "arrow-conversion")]
pub mod arrow_conversion;

#[cfg(feature = "arrow-expression")]
pub mod arrow_expression;
#[cfg(all(feature = "arrow-expression", feature = "internal-api"))]
pub mod arrow_utils;
#[cfg(all(feature = "arrow-expression", not(feature = "internal-api")))]
pub(crate) mod arrow_utils;
#[cfg(all(feature = "internal-api", feature = "arrow-expression"))]
pub use self::arrow_utils::{parse_json, to_json_bytes};

// Plan-backed handlers use Arrow evaluation without depending on storage implementations.
#[cfg(all(feature = "declarative-plans", feature = "arrow-expression"))]
pub mod plans;

#[cfg(test)]
pub(crate) mod sync;

#[cfg(test)]
pub(crate) mod test_delegating;

#[cfg(feature = "arrow-engine-data")]
pub mod arrow_data;
#[cfg(feature = "arrow-engine-data")]
pub(crate) mod arrow_get_data;

#[cfg(all(feature = "arrow-expression", feature = "internal-api"))]
pub mod ensure_data_types;
#[cfg(all(feature = "arrow-expression", not(feature = "internal-api")))]
pub(crate) mod ensure_data_types;
#[cfg(feature = "default-engine-base")]
// module is always pub; trait inside is gated by #[internal_api]
pub mod parquet_row_group_skipping;
#[cfg(all(test, feature = "default-engine-base"))]
pub(crate) mod test_utils;
