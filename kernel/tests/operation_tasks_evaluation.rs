#![cfg(feature = "operation-tasks")]

#[cfg(test)]
mod tests {
    use std::mem::size_of;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use delta_kernel::engine_data::RowVisitor;
    use delta_kernel::expressions::{ArrayData, ColumnName};
    use delta_kernel::schema::SchemaRef;
    use delta_kernel::tasks::{
        AccountedEngineData, EvaluationLimits, EvaluationPage, EvaluationPageLimits,
        EvaluationReader, FailureKind, OperationFailure, Resource, ResourceExhausted,
    };
    use delta_kernel::{DeltaResult, EngineData, Error};

    const SLOT: usize = size_of::<Box<dyn AccountedEngineData>>();

    struct Batch {
        rows: usize,
        bytes: usize,
        backing: Option<Vec<u8>>,
        drops: Arc<AtomicUsize>,
    }

    impl Drop for Batch {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl AccountedEngineData for Batch {
        fn accounted_bytes(&self) -> Result<usize, ResourceExhausted> {
            Ok(self
                .backing
                .as_ref()
                .map_or(self.bytes, |data| data.capacity() + size_of::<Self>()))
        }
    }

    impl EngineData for Batch {
        fn len(&self) -> usize {
            self.rows
        }
        fn visit_rows(
            &self,
            columns: &[ColumnName],
            visitor: &mut dyn RowVisitor,
        ) -> DeltaResult<()> {
            assert!(columns.is_empty());
            visitor.visit(self.rows, &[])
        }
        fn append_columns(
            &self,
            _: SchemaRef,
            _: Vec<ArrayData>,
        ) -> DeltaResult<Box<dyn EngineData>> {
            Err(Error::generic("test batch has no columns"))
        }
        fn apply_selection_vector(
            self: Box<Self>,
            selection: Vec<bool>,
        ) -> DeltaResult<Box<dyn EngineData>> {
            assert!(selection.is_empty());
            Ok(self)
        }
        fn has_field(&self, _: &ColumnName) -> bool {
            false
        }
    }

    fn batch(rows: usize, bytes: usize, drops: &Arc<AtomicUsize>) -> Box<dyn AccountedEngineData> {
        Box::new(Batch {
            rows,
            bytes,
            backing: None,
            drops: drops.clone(),
        })
    }

    struct Source {
        remaining: usize,
        pulls: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
        batch_drops: Arc<AtomicUsize>,
    }
    impl Iterator for Source {
        type Item = Result<Box<dyn AccountedEngineData>, OperationFailure>;
        fn next(&mut self) -> Option<Self::Item> {
            self.pulls.fetch_add(1, Ordering::Relaxed);
            if self.remaining == 0 {
                None
            } else {
                self.remaining -= 1;
                Some(Ok(batch(1, 64, &self.batch_drops)))
            }
        }
    }
    impl Drop for Source {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }
    fn source(count: usize) -> (Source, Arc<AtomicUsize>, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let pulls = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let batch_drops = Arc::new(AtomicUsize::new(0));
        (
            Source {
                remaining: count,
                pulls: pulls.clone(),
                drops: drops.clone(),
                batch_drops: batch_drops.clone(),
            },
            pulls,
            drops,
            batch_drops,
        )
    }
    fn limits(
        page_batches: usize,
        pages: usize,
        batches: usize,
        rows: usize,
        bytes: usize,
    ) -> EvaluationLimits {
        EvaluationLimits::new(
            EvaluationPageLimits::new(page_batches, 65536, 8 << 20).unwrap(),
            pages,
            batches,
            rows,
            bytes,
        )
    }
    fn resource(error: &OperationFailure, expected: Resource, limit: usize, observed: usize) {
        assert_eq!(
            error.kind(),
            FailureKind::ResourceExhausted(ResourceExhausted {
                resource: expected,
                limit,
                observed
            })
        );
    }

    #[test]
    fn page_limits_reject_non_progressing_or_unrepresentable_configuration() {
        assert!(EvaluationPageLimits::new(0, 1, SLOT).is_err());
        assert!(EvaluationPageLimits::new(1, 0, SLOT).is_err());
        assert!(EvaluationPageLimits::new(1, 1, SLOT - 1).is_err());
        assert!(EvaluationPageLimits::new(usize::MAX, 1, usize::MAX).is_err());
    }

    #[test]
    fn page_counts_physical_backing_and_actual_container_capacity() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut batches = Vec::with_capacity(4);
        let mut backing = Vec::with_capacity(4096);
        backing.push(1);
        let physical = backing.capacity() + size_of::<Batch>();
        batches.push(Box::new(Batch {
            rows: 1,
            bytes: 0,
            backing: Some(backing),
            drops: drops.clone(),
        }) as Box<dyn AccountedEngineData>);
        let bytes = batches.capacity() * SLOT + physical;
        let page =
            EvaluationPage::try_new(batches, EvaluationPageLimits::new(4, 1, bytes).unwrap())
                .unwrap();
        assert_eq!(page.accounted_bytes(), bytes);
        assert_eq!(page.num_rows(), 1);
        assert_eq!(page.batches().len(), 1);
        drop(page);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn page_rejects_each_boundary_and_drops_owned_batches() {
        for (count, rows, bytes, which) in [
            (2, 1, 0, Resource::EvaluationBatches),
            (1, 3, 0, Resource::EvaluationRows),
            (1, 1, 513, Resource::EvaluationBytes),
        ] {
            let drops = Arc::new(AtomicUsize::new(0));
            let batches = (0..count).map(|_| batch(rows, bytes, &drops)).collect();
            let error = EvaluationPage::try_new(
                batches,
                EvaluationPageLimits::new(1, 2, SLOT + 512).unwrap(),
            )
            .unwrap_err();
            assert!(
                matches!(error.kind(), FailureKind::ResourceExhausted(e) if e.resource == which)
            );
            assert_eq!(drops.load(Ordering::Relaxed), count);
        }
        let empty = Vec::<Box<dyn AccountedEngineData>>::with_capacity(4);
        let cap = empty.capacity();
        resource(
            &EvaluationPage::try_new(empty, EvaluationPageLimits::new(1, 1, SLOT).unwrap())
                .unwrap_err(),
            Resource::EvaluationBytes,
            SLOT,
            cap * SLOT,
        );
    }

    #[test]
    fn checked_accounting_never_wraps() {
        let drops = Arc::new(AtomicUsize::new(0));
        let error = EvaluationPage::try_new(
            vec![batch(1, usize::MAX, &drops)],
            EvaluationPageLimits::new(1, 1, usize::MAX).unwrap(),
        )
        .unwrap_err();
        resource(&error, Resource::EvaluationBytes, usize::MAX, usize::MAX);
        let error = EvaluationPage::try_new(
            vec![batch(usize::MAX, 0, &drops), batch(1, 0, &drops)],
            EvaluationPageLimits::new(2, usize::MAX, 2 * SLOT).unwrap(),
        )
        .unwrap_err();
        resource(&error, Resource::EvaluationRows, usize::MAX, usize::MAX);
        assert_eq!(drops.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn native_reader_does_not_look_ahead_and_releases_at_eof() {
        for size in [1, 2, 8] {
            let (input, pulls, drops, batch_drops) = source(size * 2);
            let mut reader = EvaluationReader::new(
                Box::new(input),
                limits(size, 3, size * 2 + 1, size * 2, 1 << 20),
            );
            for page_index in 1..=2 {
                let page = reader.next_page().unwrap().unwrap();
                assert_eq!(page.batches().len(), size);
                assert_eq!(pulls.load(Ordering::Relaxed), size * page_index);
                drop(page);
            }
            assert!(reader.next_page().unwrap().is_none());
            assert!(reader.next_page().unwrap().is_none());
            assert_eq!(pulls.load(Ordering::Relaxed), size * 2 + 1);
            assert_eq!(drops.load(Ordering::Relaxed), 1);
            assert_eq!(batch_drops.load(Ordering::Relaxed), size * 2);
            assert_eq!(reader.usage().pages(), 3);
            assert_eq!(reader.usage().batches(), size * 2);
            assert_eq!(reader.usage().rows(), size * 2);
            assert_eq!(reader.usage().bytes(), 3 * size * SLOT + size * 2 * 64);
        }
    }

    #[test]
    fn reader_charges_cumulative_batches_rows_bytes_across_pages() {
        for (batch_limit, row_limit, byte_limit, expected) in [
            (1, 9, 9999, Resource::EvaluationBatches),
            (9, 1, 9999, Resource::EvaluationRows),
            (9, 9, 2 * SLOT + 127, Resource::EvaluationBytes),
        ] {
            let (input, pulls, drops, batch_drops) = source(3);
            let mut reader = EvaluationReader::new(
                Box::new(input),
                limits(1, 8, batch_limit, row_limit, byte_limit),
            );
            drop(reader.next_page().unwrap().unwrap());
            let error = reader.next_page().unwrap_err();
            assert!(
                matches!(error.kind(), FailureKind::ResourceExhausted(e) if e.resource == expected)
            );
            let expected_pulls = if expected == Resource::EvaluationBatches {
                1
            } else {
                2
            };
            assert_eq!(pulls.load(Ordering::Relaxed), expected_pulls);
            assert_eq!(drops.load(Ordering::Relaxed), 1);
            assert_eq!(batch_drops.load(Ordering::Relaxed), expected_pulls);
            assert_eq!(reader.next_page().unwrap_err().kind(), error.kind());
            reader.cancel();
            assert_eq!(reader.next_page().unwrap_err().kind(), error.kind());
            assert_eq!(pulls.load(Ordering::Relaxed), expected_pulls);
        }
    }

    #[test]
    fn page_and_container_exhaustion_precedes_pulling() {
        for (pages, bytes, expected) in [
            (0, 9999, Resource::EvaluationPages),
            (1, SLOT - 1, Resource::EvaluationBytes),
        ] {
            let (input, pulls, drops, _) = source(1);
            let mut reader = EvaluationReader::new(Box::new(input), limits(1, pages, 1, 1, bytes));
            let error = reader.next_page().unwrap_err();
            assert!(
                matches!(error.kind(), FailureKind::ResourceExhausted(e) if e.resource == expected)
            );
            assert_eq!(pulls.load(Ordering::Relaxed), 0);
            assert_eq!(drops.load(Ordering::Relaxed), 1);
        }
    }

    #[test]
    fn cancel_drops_iterator_immediately_and_preserves_kernel_cancelled() {
        for consume in [false, true] {
            let (input, pulls, drops, batch_drops) = source(3);
            let mut reader = EvaluationReader::new(Box::new(input), limits(1, 4, 4, 4, 9999));
            let owned_page = if consume {
                reader.next_page().unwrap()
            } else {
                None
            };
            reader.cancel();
            reader.cancel();
            assert_eq!(drops.load(Ordering::Relaxed), 1);
            assert_eq!(pulls.load(Ordering::Relaxed), usize::from(consume));
            assert!(matches!(
                reader.next_page().unwrap_err().into_error(),
                Error::Cancelled
            ));
            assert_eq!(batch_drops.load(Ordering::Relaxed), 0);
            drop(owned_page);
            assert_eq!(batch_drops.load(Ordering::Relaxed), usize::from(consume));
        }
    }

    #[derive(Debug)]
    struct Original(Arc<AtomicUsize>);
    impl std::fmt::Display for Original {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("original")
        }
    }
    impl std::error::Error for Original {}
    impl Drop for Original {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
    #[test]
    fn operational_error_is_transferred_once_and_terminal_retains_no_source() {
        let drops = Arc::new(AtomicUsize::new(0));
        let failure = OperationFailure::new(
            FailureKind::Engine,
            Error::generic_err(Original(drops.clone())),
        );
        let mut reader = EvaluationReader::new(
            Box::new(std::iter::once(Err(failure))),
            limits(1, 1, 1, 1, 9999),
        );
        let first = reader.next_page().unwrap_err();
        assert!(std::error::Error::source(&first).is_some());
        assert!(std::error::Error::source(&reader.next_page().unwrap_err()).is_none());
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        match first.into_error() {
            Error::GenericError { source } => assert!(source.downcast_ref::<Original>().is_some()),
            _ => panic!("original error was replaced"),
        }
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn completion_is_immutable_under_cancellation() {
        let (input, pulls, drops, _) = source(0);
        let mut reader = EvaluationReader::new(Box::new(input), limits(1, 1, 1, 0, SLOT));
        assert!(reader.next_page().unwrap().is_none());
        reader.cancel();
        assert!(reader.next_page().unwrap().is_none());
        assert_eq!(pulls.load(Ordering::Relaxed), 1);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }
    #[test]
    fn failure_mid_page_releases_all_batches_and_the_source() {
        let (input, pulls, drops, batch_drops) = source(4);
        let mut reader = EvaluationReader::new(Box::new(input), limits(2, 3, 4, 1, 9999));
        resource(
            &reader.next_page().unwrap_err(),
            Resource::EvaluationRows,
            1,
            2,
        );
        assert_eq!(pulls.load(Ordering::Relaxed), 2);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        assert_eq!(batch_drops.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn engine_cancellation_overrides_generic_failure_classification() {
        let error = OperationFailure::new(FailureKind::Engine, Error::Cancelled);
        assert_eq!(error.kind(), FailureKind::Cancelled);
        assert!(matches!(error.into_error(), Error::Cancelled));
    }
    #[test]
    fn backtraced_cancellation_remains_cancelled_on_first_and_late_calls() {
        let wrapped = Error::Backtraced {
            source: Box::new(Error::Cancelled),
            backtrace: Box::new(std::backtrace::Backtrace::disabled()),
        };
        let failure = OperationFailure::new(FailureKind::Engine, wrapped);
        let mut reader = EvaluationReader::new(
            Box::new(std::iter::once(Err(failure))),
            limits(1, 1, 1, 1, 9999),
        );
        for _ in 0..2 {
            let error = reader.next_page().unwrap_err();
            assert_eq!(error.kind(), FailureKind::Cancelled);
            assert!(matches!(error.into_error(), Error::Cancelled));
        }
    }

    #[test]
    fn explicit_cancellation_category_dominates_its_engine_context() {
        let drops = Arc::new(AtomicUsize::new(0));
        let failure = OperationFailure::new(
            FailureKind::Cancelled,
            Error::generic_err(Original(drops.clone())),
        );
        assert!(matches!(failure.into_error(), Error::Cancelled));
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }
}
