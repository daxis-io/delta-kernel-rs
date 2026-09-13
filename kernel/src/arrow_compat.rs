//! This module re-exports the different versions of arrow, parquet, and object_store we support.

#[cfg(all(feature = "arrow-58", not(feature = "arrow-only")))]
pub use arrow_58 as arrow;
#[cfg(feature = "arrow-only")]
pub use arrow_59 as arrow;
#[cfg(all(feature = "arrow-58", not(feature = "arrow-59")))]
pub use parquet_58 as parquet;
#[cfg(feature = "arrow-59")]
pub use parquet_59 as parquet;

#[cfg(any(feature = "arrow-58", feature = "arrow-59"))]
pub mod object_store {
    #[cfg(not(feature = "arrow-59"))]
    pub use object_store_13::*;
    #[cfg(feature = "arrow-59")]
    pub use object_store_14::*;

    /// Constructs a GET response using the selected object-store version.
    ///
    /// The payload, metadata, range, and attributes are transferred unchanged. Runtime
    /// extensions, where supported, start empty. This function does not perform I/O.
    #[doc(hidden)]
    pub fn new_get_result(
        payload: GetResultPayload,
        meta: ObjectMeta,
        range: std::ops::Range<u64>,
        attributes: Attributes,
    ) -> GetResult {
        GetResult {
            payload,
            meta,
            range,
            attributes,
            #[cfg(feature = "arrow-59")]
            extensions: Default::default(),
        }
    }

    /// Constructs a PUT response with the supplied etag and version.
    ///
    /// Runtime extensions, where supported, start empty. This function does not perform I/O.
    #[doc(hidden)]
    pub fn new_put_result(e_tag: Option<String>, version: Option<String>) -> PutResult {
        PutResult {
            e_tag,
            version,
            #[cfg(feature = "arrow-59")]
            extensions: Default::default(),
        }
    }
}

#[cfg(all(
    feature = "need-arrow",
    not(feature = "arrow-58"),
    not(feature = "arrow-only")
))]
compile_error!(
    "Requested a feature that needs arrow without enabling arrow. Please enable `arrow-only`, `arrow-58` or `arrow-59`"
);

#[cfg(all(
    feature = "arrow-only",
    feature = "arrow-58",
    not(feature = "arrow-59")
))]
compile_error!(
    "Combining Arrow-only 59 with native Arrow 58 requires `arrow-59` for matching Parquet types"
);
