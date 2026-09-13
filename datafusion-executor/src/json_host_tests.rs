use crate::qualification_fixture::{FixtureStore, LogVariant};
use crate::JsonTaskHost;
use datafusion::execution::{
    config::SessionConfig,
    context::SessionContext,
    memory_pool::{GreedyMemoryPool, MemoryPool},
    object_store::ObjectStoreUrl,
    runtime_env::RuntimeEnvBuilder,
    session_state::SessionStateBuilder,
};
use delta_kernel::tasks::*;
use std::sync::{atomic::Ordering, Arc};

fn caller(store: Arc<FixtureStore>) -> (SessionContext, Arc<GreedyMemoryPool>) {
    let pool = Arc::new(GreedyMemoryPool::new(64 << 20));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(pool.clone())
        .build_arc()
        .unwrap();
    let config =
        crate::qualification_fixture::register(store, SessionConfig::new(), runtime.as_ref());
    // A minimal caller does not warm the default global builtin registries.
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_runtime_env(runtime)
        .build();
    (SessionContext::new_with_state(state), pool)
}
async fn drive<T: OperationTask>(
    task: &mut T,
    driver: &mut AsyncOperationDriver<JsonTaskHost<'_>>,
) -> Result<T::Output, OperationFailure> {
    let cpu = CpuSlice::new(1024, 1 << 20, 256).unwrap();
    let mut step = task.start(cpu).unwrap();
    for _ in 0..1000 {
        step = match step {
            TaskStep::Yield => task.progress(cpu).unwrap(),
            TaskStep::Execute(request) => {
                let key = driver.dispatch(request).unwrap();
                driver
                    .complete_effect(task.pending_work().unwrap())
                    .await
                    .unwrap();
                task.resume(key, driver.take_response(key).unwrap(), cpu)
                    .unwrap()
            }
            TaskStep::Complete(output) => return Ok(output),
            TaskStep::Failed(error) => return Err(error),
            TaskStep::Cancelled => panic!("unexpected cancellation"),
        };
    }
    panic!("bounded fixture task did not terminate")
}

#[tokio::test]
async fn concrete_host_latest_and_v0_use_actual_driver_pages_and_caller_pool() {
    for (version, expected_version, expected_paths) in [
        (None, 1, vec!["part-a.parquet"]),
        (Some(0), 0, vec!["part-a.parquet", "part-b.parquet"]),
    ] {
        let store = Arc::new(FixtureStore::new(LogVariant::Valid));
        let (session, pool) = caller(store.clone());
        let origin = ObjectStoreUrl::parse("memory:///").unwrap();
        let root = url::Url::parse("memory:///table/").unwrap();
        let limits = TaskLimits::qualification().with_limit(Resource::ListingEntries, 1);
        let host = JsonTaskHost::new(&session, &origin, limits).unwrap();
        let (id, mut driver) = AsyncOperationDriver::allocate(host, limits).unwrap();
        let evaluation = driver.allocate_evaluation().unwrap();
        let mut load = SnapshotLoadTask::try_new(id, evaluation, &root, version, limits).unwrap();
        let snapshot = drive(&mut load, &mut driver).await.unwrap();
        println!(
            "ADMITTED_HOST snapshot version={} work={} pages={}",
            snapshot.version(),
            load.accounting().usage(Resource::WorkUnits).consumed(),
            driver.evaluation_usage().pages()
        );
        assert_eq!(snapshot.version(), expected_version);
        assert_eq!(snapshot.schema().num_fields(), 9);
        drop(driver);
        assert_eq!(pool.reserved(), 0);
        let host = JsonTaskHost::new(&session, &origin, limits).unwrap();
        let (id, mut driver) = AsyncOperationDriver::allocate(host, limits).unwrap();
        let evaluation = driver.allocate_evaluation().unwrap();
        let mut scan = ScanMetadataTask::try_new(id, evaluation, snapshot, limits).unwrap();
        let metadata = drive(&mut scan, &mut driver).await.unwrap();
        println!(
            "ADMITTED_HOST scan version={} work={} pages={} batches={}",
            expected_version,
            scan.accounting().usage(Resource::WorkUnits).consumed(),
            driver.evaluation_usage().pages(),
            metadata.len()
        );
        drop(driver);
        assert!(
            pool.reserved() > 0,
            "retained pages keep their shared reservation"
        );
        let mut paths = Vec::new();
        for batch in &metadata {
            paths = batch
                .visit_scan_files(paths, |paths, file| {
                    assert_eq!(file.size, 3029);
                    assert!(!file.dv_info.has_vector());
                    assert!(file.partition_values.is_empty());
                    assert!(file.transform.is_none());
                    paths.push(file.path);
                })
                .unwrap();
        }
        paths.sort();
        assert_eq!(paths, expected_paths);
        drop(metadata);
        assert_eq!(pool.reserved(), 0);
        assert_eq!(
            store.lists.load(Ordering::SeqCst),
            if version.is_none() { 3 } else { 1 }
        );
        assert_eq!(store.reads[0].load(Ordering::SeqCst), 2);
        assert_eq!(
            store.reads[1].load(Ordering::SeqCst),
            if version.is_none() { 2 } else { 0 }
        );
        assert_eq!(
            store.ordinary_gets.load(Ordering::SeqCst),
            0,
            "ordinary store is empty; no log fallback"
        );
        assert_eq!(store.pending.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn concrete_host_rejects_replacement_and_releases_after_error_owner_drop() {
    let store = Arc::new(FixtureStore::new(LogVariant::Valid));
    store.replace_reads.store(true, Ordering::SeqCst);
    let (session, pool) = caller(store.clone());
    let origin = ObjectStoreUrl::parse("memory:///").unwrap();
    let limits = TaskLimits::qualification().with_limit(Resource::ListingEntries, 1);
    let host = JsonTaskHost::new(&session, &origin, limits).unwrap();
    let (id, mut driver) = AsyncOperationDriver::allocate(host, limits).unwrap();
    let evaluation = driver.allocate_evaluation().unwrap();
    let mut load = SnapshotLoadTask::try_new(
        id,
        evaluation,
        &url::Url::parse("memory:///table/").unwrap(),
        None,
        limits,
    )
    .unwrap();
    let error = drive(&mut load, &mut driver).await.unwrap_err();
    assert_eq!(error.kind(), FailureKind::MalformedResponse);
    assert_eq!(
        store.allocations.load(Ordering::SeqCst),
        3,
        "only admitted listing pages allocated"
    );
    assert_eq!(store.reads[0].load(Ordering::SeqCst), 1);
    assert_eq!(store.reads[1].load(Ordering::SeqCst), 0);
    assert_eq!(store.ordinary_gets.load(Ordering::SeqCst), 0);
    drop(driver);
    assert!(
        pool.reserved() > 0,
        "typed error owns its reservation until released"
    );
    drop(error);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn concrete_host_dropped_listing_future_cancels_without_producer_allocations() {
    let store = Arc::new(FixtureStore::new(LogVariant::Valid));
    let (session, pool) = caller(store.clone());
    let origin = ObjectStoreUrl::parse("memory:///").unwrap();
    let limits = TaskLimits::qualification().with_limit(Resource::ListingEntries, 1);
    let host = JsonTaskHost::new(&session, &origin, limits).unwrap();
    let (id, mut driver) = AsyncOperationDriver::allocate(host, limits).unwrap();
    let evaluation = driver.allocate_evaluation().unwrap();
    let mut load = SnapshotLoadTask::try_new(
        id,
        evaluation,
        &url::Url::parse("memory:///table/").unwrap(),
        None,
        limits,
    )
    .unwrap();
    let TaskStep::Execute(request) = load
        .start(CpuSlice::new(1024, 1 << 20, 256).unwrap())
        .unwrap()
    else {
        panic!("listing expected")
    };
    let key = driver.dispatch(request).unwrap();
    {
        let mut completion = Box::pin(driver.complete_effect(load.pending_work().unwrap()));
        assert!(futures::poll!(completion.as_mut()).is_pending());
        assert_eq!(store.pending.load(Ordering::SeqCst), 1);
    }
    assert_eq!(store.pending.load(Ordering::SeqCst), 0);
    assert_eq!(store.cancellations.load(Ordering::SeqCst), 1);
    assert_eq!(store.allocations.load(Ordering::SeqCst), 0);
    driver.cancel(Some(key)).unwrap();
    load.cancel(CancelReason::Caller);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn winning_v1_metadata_schema_change_is_unsupported_before_parquet_io() {
    // A minimal changed v1 schema reaches Kernel rejection within the unchanged
    // qualification work profile; the larger old fixture exhausted it first.
    let store = Arc::new(FixtureStore::new(LogVariant::MetadataSchemaChange));
    let (session, pool) = caller(store.clone());
    let origin = ObjectStoreUrl::parse("memory:///").unwrap();
    let limits = TaskLimits::qualification().with_limit(Resource::ListingEntries, 1);
    let host = JsonTaskHost::new(&session, &origin, limits).unwrap();
    let (id, mut driver) = AsyncOperationDriver::allocate(host, limits).unwrap();
    let evaluation = driver.allocate_evaluation().unwrap();
    let mut load = SnapshotLoadTask::try_new(
        id,
        evaluation,
        &url::Url::parse("memory:///table/").unwrap(),
        None,
        limits,
    )
    .unwrap();
    let error = drive(&mut load, &mut driver).await.unwrap_err();
    assert_eq!(error.kind(), FailureKind::Engine);
    let mut cause: &dyn std::error::Error = &error;
    loop {
        if matches!(
            cause.downcast_ref::<delta_kernel::Error>(),
            Some(delta_kernel::Error::Unsupported(_))
        ) {
            break;
        }
        cause = cause
            .source()
            .expect("typed Kernel Unsupported must survive task failure");
    }
    assert_eq!(store.reads[0].load(Ordering::SeqCst), 1);
    assert_eq!(store.reads[1].load(Ordering::SeqCst), 1);
    assert_eq!(
        store.ordinary_gets.load(Ordering::SeqCst),
        0,
        "no Parquet or ordinary-store read before rejection"
    );
    assert!(store.ordinary_reads.lock().unwrap().is_empty());
    drop(load);
    drop(driver);
    drop(error);
    assert_eq!(pool.reserved(), 0);
}
