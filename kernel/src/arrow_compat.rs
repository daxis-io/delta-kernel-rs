//! This module re-exports the different versions of arrow, parquet, and object_store we support.

#[cfg(feature = "arrow-58")]
mod arrow_compat_shims {
    #[cfg(all(
        not(all(target_arch = "wasm32", target_os = "unknown")),
        any(not(feature = "daxis-browser-stack"), feature = "default-engine-base")
    ))]
    pub use arrow_58_native as arrow;
    #[cfg(all(
        not(all(target_arch = "wasm32", target_os = "unknown")),
        feature = "daxis-browser-stack",
        not(feature = "default-engine-base")
    ))]
    pub use arrow_58_stack as arrow;
    #[cfg(all(
        target_arch = "wasm32",
        target_os = "unknown",
        feature = "daxis-browser-stack"
    ))]
    pub use arrow_58_stack as arrow;
    #[cfg(all(
        not(all(target_arch = "wasm32", target_os = "unknown")),
        any(not(feature = "daxis-browser-stack"), feature = "default-engine-base")
    ))]
    pub use parquet_58_native as parquet;
    #[cfg(all(
        not(all(target_arch = "wasm32", target_os = "unknown")),
        feature = "daxis-browser-stack",
        not(feature = "default-engine-base")
    ))]
    pub use parquet_58_stack as parquet;
    #[cfg(all(
        target_arch = "wasm32",
        target_os = "unknown",
        feature = "daxis-browser-stack"
    ))]
    pub use parquet_58_stack as parquet;

    pub mod object_store {
        #[cfg(all(
            not(all(target_arch = "wasm32", target_os = "unknown")),
            any(not(feature = "daxis-browser-stack"), feature = "default-engine-base")
        ))]
        pub use object_store_13_native::*;
        #[cfg(all(
            not(all(target_arch = "wasm32", target_os = "unknown")),
            feature = "daxis-browser-stack",
            not(feature = "default-engine-base")
        ))]
        pub use object_store_13_stack::*;
        #[cfg(all(
            target_arch = "wasm32",
            target_os = "unknown",
            feature = "daxis-browser-stack"
        ))]
        pub use object_store_13_stack::*;
    }
}

#[cfg(all(
    target_arch = "wasm32",
    target_os = "unknown",
    feature = "arrow-58",
    not(feature = "daxis-browser-stack")
))]
compile_error!(
    "the Daxis stack requires `daxis-browser-stack` with `arrow-58` on wasm32-unknown-unknown"
);

#[cfg(all(feature = "arrow-57", not(feature = "arrow-58")))]
mod arrow_compat_shims {
    pub use arrow_57 as arrow;
    pub use parquet_57 as parquet;

    pub mod object_store {
        use std::future::Future;
        use std::ops::Range;

        pub use object_store_12::*;

        /// Compatibility extension trait mirroring `ObjectStoreExt` from object_store 0.13.
        ///
        /// Proxy methods that moved from `ObjectStore` (0.12) to `ObjectStoreExt` (0.13) so that
        /// callers can unconditionally `use ObjectStoreExt as _` even in arrow-57 mode. Otherwise,
        /// clippy would flag the import as unused in arrow-57 mode when callers only invoke the
        /// moved methods on an ObjectStore instance.
        pub trait ObjectStoreExt: ObjectStore {
            fn put(
                &self,
                location: &path::Path,
                payload: PutPayload,
            ) -> impl Future<Output = Result<PutResult>> + Send {
                async move { ObjectStore::put(self, location, payload).await }
            }

            fn get(&self, location: &path::Path) -> impl Future<Output = Result<GetResult>> + Send {
                async move { ObjectStore::get(self, location).await }
            }

            fn get_range(
                &self,
                location: &path::Path,
                range: Range<u64>,
            ) -> impl Future<Output = Result<bytes::Bytes>> + Send {
                async move { ObjectStore::get_range(self, location, range).await }
            }

            fn head(
                &self,
                location: &path::Path,
            ) -> impl Future<Output = Result<ObjectMeta>> + Send {
                async move { ObjectStore::head(self, location).await }
            }

            fn delete(&self, location: &path::Path) -> impl Future<Output = Result<()>> + Send {
                async move { ObjectStore::delete(self, location).await }
            }

            fn copy(
                &self,
                from: &path::Path,
                to: &path::Path,
            ) -> impl Future<Output = Result<()>> + Send {
                async move { ObjectStore::copy(self, from, to).await }
            }

            fn copy_if_not_exists(
                &self,
                from: &path::Path,
                to: &path::Path,
            ) -> impl Future<Output = Result<()>> + Send {
                async move { ObjectStore::copy_if_not_exists(self, from, to).await }
            }
        }

        impl<T: ObjectStore + ?Sized> ObjectStoreExt for T {}
    }
}

// if nothing is enabled but we need arrow because of some other feature flag, throw compile-time
// error
#[cfg(all(
    feature = "need-arrow",
    not(feature = "arrow-57"),
    not(feature = "arrow-58")
))]
compile_error!("Requested a feature that needs arrow without enabling arrow. Please enable the `arrow-57` or `arrow-58` feature");

#[cfg(any(feature = "arrow-57", feature = "arrow-58"))]
#[doc(hidden)]
pub use arrow_compat_shims::*;
