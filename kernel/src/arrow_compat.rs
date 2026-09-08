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
    pub use object_store_13::*;
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
