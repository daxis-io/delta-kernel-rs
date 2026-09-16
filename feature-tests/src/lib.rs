/// This is a compilation test to ensure that the default-engine feature flags are working
/// correctly.
///
/// Run (from workspace root) with:
/// 1. `cargo b -p feature_tests --features default-engine-rustls`
/// 2. `cargo b -p feature_tests --features default-engine-native-tls`
///
/// These run in our build CI.
pub fn test_default_engine_feature_flags() {
    #[cfg(any(
        feature = "default-engine-native-tls",
        feature = "default-engine-rustls"
    ))]
    {
        #[allow(unused_imports)]
        use delta_kernel_default_engine::DefaultEngine;
    }
}

/// Regression tests for rustls crypto provider conflicts.
///
/// rustls 0.23 panics at runtime if both `aws-lc-rs` and `ring` features are active and no
/// provider is explicitly installed. Each TLS backend must construct clients without that
/// conflict under the pinned object_store dependency graph.
///
/// Two APIs are tested because they behave differently:
///  - `rustls::ClientConfig::builder()` relies on auto-detection and panics on dual providers.
///  - `reqwest::Client::new()` explicitly constructs its provider, so it always succeeds.
#[cfg(test)]
mod tests {
    // Isolate each TLS backend. Workspace --all-features also enables the older Arrow
    // dependency graph, which can add a second crypto provider.
    #[test]
    #[cfg(any(
        all(
            feature = "default-engine-native-tls",
            not(feature = "default-engine-rustls")
        ),
        all(
            feature = "default-engine-rustls",
            not(feature = "default-engine-native-tls")
        )
    ))]
    fn test_tls_rustls_builder_no_dual_provider_panic() {
        let _config = rustls::ClientConfig::builder();
    }

    #[test]
    #[cfg(feature = "default-engine-native-tls")]
    fn test_native_tls_reqwest_client_no_panic() {
        let _client = reqwest::Client::new();
    }

    #[test]
    #[cfg(feature = "default-engine-rustls")]
    fn test_rustls_reqwest_client_no_panic() {
        let _client = reqwest::Client::new();
    }
}
