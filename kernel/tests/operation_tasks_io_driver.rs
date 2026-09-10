#![cfg(feature = "operation-tasks")]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use delta_kernel::schema::StructType;
use delta_kernel::tasks::{
    AdmittedEvaluationSource, AdmittedFooter, AdmittedHead, AdmittedIoSource, AdmittedListingPage,
    AdmittedPlan, AdmittedRead, EvaluationLimits, FileDescriptor, FooterLimits, ObjectIdentity,
    OperationDriver, OperationFailure, RequestKey, Resource, TaskLimits, TaskProtocolError,
    TaskRequest, TaskRequestV1, TaskResponseV1,
};
use delta_kernel::ParquetFooter;

#[derive(Default)]
struct Counts {
    list_calls: AtomicUsize,
    list_pulls: AtomicUsize,
    reads: AtomicUsize,
    reads_with_identity: AtomicUsize,
    heads: AtomicUsize,
    footers: AtomicUsize,
    cancels: AtomicUsize,
}

struct FixedIo {
    files: VecDeque<FileDescriptor>,
    counts: Arc<Counts>,
    identity: ObjectIdentity,
    bad_read_shape: bool,
    change_identity: bool,
    change_head_on_repeat: bool,
    listing_fault: Option<ListingFault>,
}

#[derive(Default)]
struct InterleavedCounts {
    heads: AtomicUsize,
    read_expected: Mutex<Vec<Option<ObjectIdentity>>>,
}

struct InterleavedIo {
    counts: Arc<InterleavedCounts>,
}

impl AdmittedIoSource for InterleavedIo {
    fn list(
        &mut self,
        _: &str,
        _: Option<&str>,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<AdmittedListingPage, OperationFailure> {
        Err(OperationFailure::malformed_response())
    }

    fn read_exact(
        &mut self,
        path: &str,
        expected_identity: Option<ObjectIdentity>,
        offset: u64,
        length: usize,
    ) -> Result<AdmittedRead, OperationFailure> {
        let mut expected = self.counts.read_expected.lock().unwrap();
        let call = expected.len();
        expected.push(expected_identity);
        Ok(AdmittedRead {
            identity: interleaved_identity(path, call),
            offset,
            bytes: vec![0; length],
            eof: true,
        })
    }

    fn head(&mut self, path: &str) -> Result<AdmittedHead, OperationFailure> {
        let call = self.counts.heads.fetch_add(1, Ordering::Relaxed);
        Ok(AdmittedHead {
            identity: interleaved_identity(path, call),
            size: 1,
        })
    }

    fn footer(
        &mut self,
        _: &str,
        _: ObjectIdentity,
        _: u64,
        _: FooterLimits,
    ) -> Result<AdmittedFooter, OperationFailure> {
        Err(OperationFailure::malformed_response())
    }
}

fn interleaved_identity(path: &str, call: usize) -> ObjectIdentity {
    match (path, call) {
        ("memory:///a", 2) => ObjectIdentity::new([3; 32]),
        ("memory:///a", _) => ObjectIdentity::new([1; 32]),
        ("memory:///b", _) => ObjectIdentity::new([2; 32]),
        _ => panic!("unexpected path"),
    }
}

#[derive(Clone, Copy)]
enum ListingFault {
    Unsorted,
    EscapedRoot,
    WrongContinuation,
    OversizedBacking,
}

impl FixedIo {
    fn new(counts: Arc<Counts>) -> Self {
        Self {
            files: [
                descriptor("memory:///log/000.json"),
                descriptor("memory:///log/001.json"),
            ]
            .into(),
            counts,
            identity: ObjectIdentity::new([7; 32]),
            bad_read_shape: false,
            change_identity: false,
            change_head_on_repeat: false,
            listing_fault: None,
        }
    }
}

impl AdmittedIoSource for FixedIo {
    fn list(
        &mut self,
        root: &str,
        continuation: Option<&str>,
        entries: usize,
        _: usize,
        _: usize,
    ) -> Result<AdmittedListingPage, OperationFailure> {
        self.counts.list_calls.fetch_add(1, Ordering::Relaxed);
        assert_eq!(root, "memory:///log/");
        let _ = continuation;
        let mut files = Vec::with_capacity(entries);
        for _ in 0..entries {
            self.counts.list_pulls.fetch_add(1, Ordering::Relaxed);
            let Some(file) = self.files.pop_front() else {
                break;
            };
            files.push(file);
        }
        let continuation = (files.len() == entries).then(|| files.last().unwrap().path.clone());
        match self.listing_fault {
            Some(ListingFault::Unsorted) => files.swap(0, 1),
            Some(ListingFault::EscapedRoot) => files[0].path = "memory:///other/000.json".into(),
            Some(ListingFault::WrongContinuation) => {
                return Ok(AdmittedListingPage {
                    files,
                    continuation: Some("memory:///log/wrong.json".into()),
                    binding: continuation,
                });
            }
            Some(ListingFault::OversizedBacking) => {
                let mut path = String::with_capacity(8192);
                path.push_str("memory:///log/000.json");
                files[0].path = path;
            }
            None => {}
        }
        Ok(AdmittedListingPage {
            files,
            binding: continuation.clone(),
            continuation,
        })
    }

    fn read_exact(
        &mut self,
        path: &str,
        expected_identity: Option<ObjectIdentity>,
        offset: u64,
        length: usize,
    ) -> Result<AdmittedRead, OperationFailure> {
        assert_eq!(path, "memory:///checkpoint.parquet");
        if expected_identity.is_some() {
            assert_eq!(expected_identity, Some(self.identity));
            self.counts
                .reads_with_identity
                .fetch_add(1, Ordering::Relaxed);
        }
        self.counts.reads.fetch_add(1, Ordering::Relaxed);
        let mut bytes = vec![0; length];
        if self.bad_read_shape {
            bytes.push(0);
        }
        Ok(AdmittedRead {
            identity: if self.change_identity {
                ObjectIdentity::new([8; 32])
            } else {
                self.identity
            },
            offset,
            bytes,
            eof: offset + length as u64 == 4,
        })
    }

    fn head(&mut self, path: &str) -> Result<AdmittedHead, OperationFailure> {
        assert_eq!(path, "memory:///checkpoint.parquet");
        let call = self.counts.heads.fetch_add(1, Ordering::Relaxed);
        Ok(AdmittedHead {
            identity: if self.change_head_on_repeat && call > 0 {
                ObjectIdentity::new([8; 32])
            } else {
                self.identity
            },
            size: 4,
        })
    }

    fn footer(
        &mut self,
        path: &str,
        expected_identity: ObjectIdentity,
        size: u64,
        limits: FooterLimits,
    ) -> Result<AdmittedFooter, OperationFailure> {
        assert_eq!(path, "memory:///checkpoint.parquet");
        assert_eq!(expected_identity, self.identity);
        assert_eq!(size, 4);
        assert_eq!(limits, TaskLimits::qualification().footer_limits());
        self.counts.footers.fetch_add(1, Ordering::Relaxed);
        Ok(AdmittedFooter {
            identity: self.identity,
            size,
            footer: ParquetFooter {
                schema: Arc::new(StructType::try_new([]).unwrap()),
            },
        })
    }

    fn cancel(&mut self) {
        self.counts.cancels.fetch_add(1, Ordering::Relaxed);
        self.files.clear();
    }
}

#[test]
fn listing_pages_without_lookahead_and_validates_the_continuation() {
    let counts = Arc::new(Counts::default());
    let (id, mut driver) = driver(FixedIo::new(counts.clone()));
    let task = id.get();
    let first = request(
        task,
        1,
        TaskRequestV1::List {
            root: "memory:///log/".into(),
            continuation: None,
            entries: 2,
            descriptor_bytes: 4096,
            continuation_bytes: 256,
        },
    );
    let key = first.key;
    driver.dispatch(first).unwrap();
    driver.complete_effect().unwrap();
    let TaskResponseV1::Listing {
        files,
        continuation,
    } = driver.take_response(key).unwrap().unwrap()
    else {
        panic!("expected listing")
    };
    assert_eq!(files.len(), 2);
    assert_eq!(counts.list_pulls.load(Ordering::Relaxed), 2);

    let second = request(
        task,
        2,
        TaskRequestV1::List {
            root: "memory:///log/".into(),
            continuation,
            entries: 2,
            descriptor_bytes: 4096,
            continuation_bytes: 256,
        },
    );
    let key = second.key;
    driver.dispatch(second).unwrap();
    driver.complete_effect().unwrap();
    let TaskResponseV1::Listing {
        files,
        continuation,
    } = driver.take_response(key).unwrap().unwrap()
    else {
        panic!("expected listing EOF")
    };
    assert!(files.is_empty());
    assert!(continuation.is_none());
    assert_eq!(counts.list_pulls.load(Ordering::Relaxed), 3);
    assert_eq!(driver.io_usage().listing_pages(), 2);
    assert_eq!(driver.io_usage().descriptors(), 2);
}

#[test]
fn head_binds_exact_reads_and_limited_footers_to_one_object_identity() {
    let counts = Arc::new(Counts::default());
    let (id, mut driver) = driver(FixedIo::new(counts.clone()));
    let task = id.get();
    let head = request(
        task,
        1,
        TaskRequestV1::Head {
            path: "memory:///checkpoint.parquet".into(),
        },
    );
    let key = head.key;
    driver.dispatch(head).unwrap();
    driver.complete_effect().unwrap();
    assert!(matches!(
        driver.take_response(key).unwrap().unwrap(),
        TaskResponseV1::Head { size: 4 }
    ));

    let read = request(
        task,
        2,
        TaskRequestV1::Read {
            path: "memory:///checkpoint.parquet".into(),
            offset: 1,
            length: 2,
        },
    );
    let key = read.key;
    driver.dispatch(read).unwrap();
    driver.complete_effect().unwrap();
    assert!(matches!(
        driver.take_response(key).unwrap().unwrap(),
        TaskResponseV1::Bytes {
            offset: 1,
            bytes,
            eof: false
        } if bytes.len() == 2 && bytes.capacity() == 2
    ));

    let footer = request(
        task,
        3,
        TaskRequestV1::Footer {
            path: "memory:///checkpoint.parquet".into(),
            size: 4,
            limits: TaskLimits::qualification().footer_limits(),
        },
    );
    let key = footer.key;
    driver.dispatch(footer).unwrap();
    driver.complete_effect().unwrap();
    assert!(matches!(
        driver.take_response(key).unwrap().unwrap(),
        TaskResponseV1::Footer { .. }
    ));
    assert_eq!(counts.heads.load(Ordering::Relaxed), 1);
    assert_eq!(counts.reads.load(Ordering::Relaxed), 1);
    assert_eq!(counts.reads_with_identity.load(Ordering::Relaxed), 1);
    assert_eq!(counts.footers.load(Ordering::Relaxed), 1);
    assert_eq!(driver.io_usage().heads(), 1);
    assert_eq!(driver.io_usage().read_requests(), 1);
    assert_eq!(driver.io_usage().requested_read_bytes(), 2);
    assert_eq!(driver.io_usage().received_read_bytes(), 2);
    assert_eq!(driver.io_usage().footers(), 1);
}

#[test]
fn combined_driver_can_continue_with_io_after_evaluation_eof() {
    let counts = Arc::new(Counts::default());
    let source = FixedIo::new(counts.clone());
    let (id, mut driver) = OperationDriver::allocate_with_io(
        source,
        empty_compile as TestCompiler,
        TaskLimits::qualification(),
    )
    .unwrap();
    let evaluation = driver.allocate_evaluation().unwrap();
    let plan =
        AdmittedPlan::try_i64_values("value", &[], &[7], &TaskLimits::qualification()).unwrap();
    let start = request(
        id.get(),
        1,
        TaskRequestV1::EvaluationStart {
            evaluation,
            plan,
            limits: EvaluationLimits::new(
                delta_kernel::tasks::EvaluationPageLimits::new(1, 1, 128).unwrap(),
                1,
                1,
                1,
                128,
            ),
        },
    );
    let key = start.key;
    driver.dispatch(start).unwrap();
    driver.complete_effect().unwrap();
    assert!(matches!(
        driver.take_response(key).unwrap().unwrap(),
        TaskResponseV1::Evaluation {
            evaluation: actual,
            page: None
        } if actual == evaluation
    ));

    let head = request(
        id.get(),
        2,
        TaskRequestV1::Head {
            path: "memory:///checkpoint.parquet".into(),
        },
    );
    let key = head.key;
    driver.dispatch(head).unwrap();
    driver.complete_effect().unwrap();
    assert!(matches!(
        driver.take_response(key).unwrap().unwrap(),
        TaskResponseV1::Head { size: 4 }
    ));
    assert_eq!(counts.heads.load(Ordering::Relaxed), 1);
}

#[test]
fn malformed_incoming_listing_cursors_never_reach_the_source_or_consume_sequence() {
    for case in 0..3 {
        let counts = Arc::new(Counts::default());
        let (id, mut driver) = driver(FixedIo::new(counts.clone()));
        let task = id.get();
        let first = request(
            task,
            1,
            TaskRequestV1::List {
                root: "memory:///log/".into(),
                continuation: None,
                entries: 1,
                descriptor_bytes: 4096,
                continuation_bytes: 256,
            },
        );
        let key = first.key;
        driver.dispatch(first).unwrap();
        driver.complete_effect().unwrap();
        let TaskResponseV1::Listing {
            continuation: Some(valid),
            ..
        } = driver.take_response(key).unwrap().unwrap()
        else {
            panic!("expected full first page")
        };
        let (invalid, continuation_bytes) = match case {
            0 => {
                let mut value = String::with_capacity(4096);
                value.push_str(&valid);
                (value, 256)
            }
            1 => ("memory:///other/000.json".into(), 256),
            _ => ("memory:///log/unissued.json".into(), 256),
        };
        let malformed = request(
            task,
            2,
            TaskRequestV1::List {
                root: "memory:///log/".into(),
                continuation: Some(invalid),
                entries: 1,
                descriptor_bytes: 4096,
                continuation_bytes,
            },
        );
        assert_eq!(
            driver.dispatch(malformed),
            Err(TaskProtocolError::WrongKind)
        );
        assert_eq!(counts.list_calls.load(Ordering::Relaxed), 1);

        let valid = request(
            task,
            2,
            TaskRequestV1::List {
                root: "memory:///log/".into(),
                continuation: Some(valid),
                entries: 1,
                descriptor_bytes: 4096,
                continuation_bytes: 256,
            },
        );
        assert!(driver.dispatch(valid).is_ok());
    }
}

#[test]
fn changed_head_identity_after_a_read_fails_terminally() {
    let counts = Arc::new(Counts::default());
    let mut source = FixedIo::new(counts);
    source.change_head_on_repeat = true;
    let (id, mut driver) = driver(source);
    let task = id.get();
    for (sequence, operation) in [
        (
            1,
            TaskRequestV1::Head {
                path: "memory:///checkpoint.parquet".into(),
            },
        ),
        (
            2,
            TaskRequestV1::Read {
                path: "memory:///checkpoint.parquet".into(),
                offset: 0,
                length: 2,
            },
        ),
    ] {
        let request = request(task, sequence, operation);
        let key = request.key;
        driver.dispatch(request).unwrap();
        driver.complete_effect().unwrap();
        driver.take_response(key).unwrap().unwrap();
    }
    let changed = request(
        task,
        3,
        TaskRequestV1::Head {
            path: "memory:///checkpoint.parquet".into(),
        },
    );
    let key = changed.key;
    driver.dispatch(changed).unwrap();
    driver.complete_effect().unwrap();
    assert_eq!(
        driver.take_response(key).unwrap().unwrap_err().kind(),
        delta_kernel::tasks::FailureKind::MalformedResponse
    );
    assert_eq!(driver.complete_effect(), Err(TaskProtocolError::Terminal));
}

#[test]
fn a_reported_exact_eof_blocks_later_ranges_beyond_that_end() {
    let counts = Arc::new(Counts::default());
    let (id, mut driver) = driver(FixedIo::new(counts.clone()));
    let task = id.get();
    let first = request(
        task,
        1,
        TaskRequestV1::Read {
            path: "memory:///checkpoint.parquet".into(),
            offset: 0,
            length: 4,
        },
    );
    let key = first.key;
    driver.dispatch(first).unwrap();
    driver.complete_effect().unwrap();
    assert!(matches!(
        driver.take_response(key).unwrap().unwrap(),
        TaskResponseV1::Bytes { eof: true, .. }
    ));
    let beyond = request(
        task,
        2,
        TaskRequestV1::Read {
            path: "memory:///checkpoint.parquet".into(),
            offset: 4,
            length: 1,
        },
    );
    assert_eq!(driver.dispatch(beyond), Err(TaskProtocolError::WrongKind));
    assert_eq!(counts.reads.load(Ordering::Relaxed), 1);
}

#[test]
fn malformed_or_identity_changed_reads_fail_terminally() {
    for mutate in [false, true] {
        let counts = Arc::new(Counts::default());
        let mut source = FixedIo::new(counts);
        source.bad_read_shape = !mutate;
        source.change_identity = mutate;
        let (id, mut driver) = driver(source);
        let task = id.get();
        let head = request(
            task,
            1,
            TaskRequestV1::Head {
                path: "memory:///checkpoint.parquet".into(),
            },
        );
        let key = head.key;
        driver.dispatch(head).unwrap();
        driver.complete_effect().unwrap();
        driver.take_response(key).unwrap().unwrap();

        let read = request(
            task,
            2,
            TaskRequestV1::Read {
                path: "memory:///checkpoint.parquet".into(),
                offset: 0,
                length: 2,
            },
        );
        let key = read.key;
        driver.dispatch(read).unwrap();
        driver.complete_effect().unwrap();
        let failure = driver.take_response(key).unwrap().unwrap_err();
        assert_eq!(
            failure.kind(),
            delta_kernel::tasks::FailureKind::MalformedResponse
        );
        assert_eq!(driver.complete_effect(), Err(TaskProtocolError::Terminal));
    }
}

#[test]
fn interleaved_heads_retain_each_paths_identity() {
    let counts = Arc::new(InterleavedCounts::default());
    let (id, mut driver) = OperationDriver::allocate_with_io(
        InterleavedIo {
            counts: counts.clone(),
        },
        no_compile as TestCompiler,
        TaskLimits::qualification(),
    )
    .unwrap();
    for (sequence, path) in [(1, "memory:///a"), (2, "memory:///b")] {
        let request = request(
            id.get(),
            sequence,
            TaskRequestV1::Head { path: path.into() },
        );
        let key = request.key;
        driver.dispatch(request).unwrap();
        driver.complete_effect().unwrap();
        driver.take_response(key).unwrap().unwrap();
    }
    let changed = request(
        id.get(),
        3,
        TaskRequestV1::Head {
            path: "memory:///a".into(),
        },
    );
    let key = changed.key;
    driver.dispatch(changed).unwrap();
    driver.complete_effect().unwrap();
    assert_eq!(
        driver.take_response(key).unwrap().unwrap_err().kind(),
        delta_kernel::tasks::FailureKind::MalformedResponse
    );
    assert_eq!(counts.heads.load(Ordering::Relaxed), 3);
    assert_eq!(driver.complete_effect(), Err(TaskProtocolError::Terminal));
}

#[test]
fn interleaved_reads_reuse_each_paths_original_identity() {
    let counts = Arc::new(InterleavedCounts::default());
    let (id, mut driver) = OperationDriver::allocate_with_io(
        InterleavedIo {
            counts: counts.clone(),
        },
        no_compile as TestCompiler,
        TaskLimits::qualification(),
    )
    .unwrap();
    for (sequence, path) in [(1, "memory:///a"), (2, "memory:///b")] {
        let request = request(
            id.get(),
            sequence,
            TaskRequestV1::Read {
                path: path.into(),
                offset: 0,
                length: 1,
            },
        );
        let key = request.key;
        driver.dispatch(request).unwrap();
        driver.complete_effect().unwrap();
        driver.take_response(key).unwrap().unwrap();
    }
    let changed = request(
        id.get(),
        3,
        TaskRequestV1::Read {
            path: "memory:///a".into(),
            offset: 0,
            length: 1,
        },
    );
    let key = changed.key;
    driver.dispatch(changed).unwrap();
    driver.complete_effect().unwrap();
    assert_eq!(
        driver.take_response(key).unwrap().unwrap_err().kind(),
        delta_kernel::tasks::FailureKind::MalformedResponse
    );
    assert_eq!(
        *counts.read_expected.lock().unwrap(),
        vec![None, None, Some(ObjectIdentity::new([1; 32]))]
    );
    assert_eq!(driver.complete_effect(), Err(TaskProtocolError::Terminal));
}

#[test]
fn object_binding_limit_fails_before_another_source_call() {
    let counts = Arc::new(InterleavedCounts::default());
    let limits = TaskLimits::qualification().with_limit(Resource::Requests, 1);
    let (id, mut driver) = OperationDriver::allocate_with_io(
        InterleavedIo {
            counts: counts.clone(),
        },
        no_compile as TestCompiler,
        limits,
    )
    .unwrap();
    let first = request(
        id.get(),
        1,
        TaskRequestV1::Head {
            path: "memory:///a".into(),
        },
    );
    let key = first.key;
    driver.dispatch(first).unwrap();
    driver.complete_effect().unwrap();
    driver.take_response(key).unwrap().unwrap();

    let second = request(
        id.get(),
        2,
        TaskRequestV1::Head {
            path: "memory:///b".into(),
        },
    );
    let key = second.key;
    driver.dispatch(second).unwrap();
    driver.complete_effect().unwrap();
    assert!(matches!(
        driver.take_response(key).unwrap().unwrap_err().kind(),
        delta_kernel::tasks::FailureKind::ResourceExhausted(error)
            if error.resource == Resource::Requests && error.limit == 1 && error.observed == 2
    ));
    assert_eq!(counts.heads.load(Ordering::Relaxed), 1);
    assert_eq!(driver.complete_effect(), Err(TaskProtocolError::Terminal));
}

#[test]
fn malformed_listing_pages_fail_before_response_transfer() {
    for fault in [
        ListingFault::Unsorted,
        ListingFault::EscapedRoot,
        ListingFault::WrongContinuation,
        ListingFault::OversizedBacking,
    ] {
        let counts = Arc::new(Counts::default());
        let mut source = FixedIo::new(counts);
        source.listing_fault = Some(fault);
        let (id, mut driver) = driver(source);
        let request = request(
            id.get(),
            1,
            TaskRequestV1::List {
                root: "memory:///log/".into(),
                continuation: None,
                entries: 2,
                descriptor_bytes: 4096,
                continuation_bytes: 256,
            },
        );
        let key = request.key;
        driver.dispatch(request).unwrap();
        driver.complete_effect().unwrap();
        let failure = driver.take_response(key).unwrap().unwrap_err();
        assert_eq!(
            failure.kind(),
            delta_kernel::tasks::FailureKind::MalformedResponse
        );
        assert_eq!(driver.complete_effect(), Err(TaskProtocolError::Terminal));
    }
}

#[test]
fn nonzero_initial_range_requires_head_without_consuming_the_sequence() {
    let counts = Arc::new(Counts::default());
    let (id, mut driver) = driver(FixedIo::new(counts.clone()));
    let task = id.get();
    let read = request(
        task,
        1,
        TaskRequestV1::Read {
            path: "memory:///checkpoint.parquet".into(),
            offset: 1,
            length: 2,
        },
    );
    assert_eq!(driver.dispatch(read), Err(TaskProtocolError::WrongKind));
    assert_eq!(counts.reads.load(Ordering::Relaxed), 0);

    let head = request(
        task,
        1,
        TaskRequestV1::Head {
            path: "memory:///checkpoint.parquet".into(),
        },
    );
    assert!(driver.dispatch(head).is_ok());
}

#[test]
fn cancellation_releases_the_source_without_executing_a_queued_effect() {
    let counts = Arc::new(Counts::default());
    let (id, mut first_driver) = driver(FixedIo::new(counts.clone()));
    let first_request = request(
        id.get(),
        1,
        TaskRequestV1::List {
            root: "memory:///log/".into(),
            continuation: None,
            entries: 2,
            descriptor_bytes: 4096,
            continuation_bytes: 256,
        },
    );
    let key = first_request.key;
    first_driver.dispatch(first_request).unwrap();
    first_driver.cancel(Some(key)).unwrap();
    assert_eq!(counts.list_pulls.load(Ordering::Relaxed), 0);
    assert_eq!(counts.cancels.load(Ordering::Relaxed), 1);
    assert_eq!(
        first_driver.complete_effect(),
        Err(TaskProtocolError::Terminal)
    );

    let counts = Arc::new(Counts::default());
    let (id, mut driver) = driver(FixedIo::new(counts.clone()));
    let request = request(
        id.get(),
        1,
        TaskRequestV1::List {
            root: "memory:///log/".into(),
            continuation: None,
            entries: 2,
            descriptor_bytes: 4096,
            continuation_bytes: 256,
        },
    );
    let key = request.key;
    driver.dispatch(request).unwrap();
    driver.complete_effect().unwrap();
    driver.cancel(Some(key)).unwrap();
    assert_eq!(counts.list_pulls.load(Ordering::Relaxed), 2);
    assert_eq!(counts.cancels.load(Ordering::Relaxed), 1);
    assert!(matches!(
        driver.take_response(key),
        Err(TaskProtocolError::Terminal)
    ));
}

type TestCompiler =
    fn(AdmittedPlan, EvaluationLimits) -> Result<AdmittedEvaluationSource, OperationFailure>;
type TestDriver = OperationDriver<FixedIo, TestCompiler>;

fn driver(source: FixedIo) -> (delta_kernel::tasks::TaskId, TestDriver) {
    OperationDriver::allocate_with_io(
        source,
        no_compile as TestCompiler,
        TaskLimits::qualification(),
    )
    .unwrap()
}

fn no_compile(
    _: AdmittedPlan,
    _: EvaluationLimits,
) -> Result<AdmittedEvaluationSource, OperationFailure> {
    unreachable!("I/O tests do not compile plans")
}

fn empty_compile(
    _: AdmittedPlan,
    _: EvaluationLimits,
) -> Result<AdmittedEvaluationSource, OperationFailure> {
    Ok(Box::new(std::iter::empty()))
}

fn descriptor(path: &str) -> FileDescriptor {
    FileDescriptor {
        path: path.into(),
        size: 4,
        modification_time: 0,
    }
}

fn request(task: u64, sequence: u64, operation: TaskRequestV1) -> TaskRequest {
    TaskRequest {
        key: RequestKey::new(task, sequence).unwrap(),
        operation,
    }
}
