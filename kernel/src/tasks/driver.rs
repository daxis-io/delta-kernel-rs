use super::{
    AccountedEngineData, AdmittedPlan, EvaluationKey, EvaluationLimits, EvaluationPageLimits,
    EvaluationReader, EvaluationUsage, FileDescriptor, FooterLimits, OperationFailure, RequestKey,
    Resource, ResourceExhausted, TaskId, TaskLimits, TaskProtocolError, TaskRequest, TaskRequestV1,
    TaskResponseV1,
};
use crate::ParquetFooter;

/// Lazy evaluation output whose allocations were admitted by its producer before each yield.
pub type AdmittedEvaluationSource =
    Box<dyn Iterator<Item = Result<Box<dyn AccountedEngineData>, OperationFailure>> + Send>;

/// Fixed-size identity used to bind reads and footer decoding to one observed object version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectIdentity([u8; 32]);

impl ObjectIdentity {
    /// Creates an identity from provider-owned version bytes or their collision-resistant digest.
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Returns the fixed-size identity bytes.
    pub fn bytes(self) -> [u8; 32] {
        self.0
    }
}

/// One listing page whose allocations were admitted by its source before construction.
pub struct AdmittedListingPage {
    /// Sorted descriptors confined to the requested root.
    pub files: Vec<FileDescriptor>,
    /// The last returned path for a full page, or `None` for definitive EOF.
    pub continuation: Option<String>,
    /// A separately admitted copy retained by the driver to authenticate the next request.
    pub binding: Option<String>,
}

/// One exact owned range whose destination allocation was admitted before I/O.
pub struct AdmittedRead {
    /// Identity observed while reading the range.
    pub identity: ObjectIdentity,
    /// Actual first byte offset.
    pub offset: u64,
    /// Exact owned bytes with capacity equal to the requested length.
    pub bytes: Vec<u8>,
    /// Whether the range ends at the identity-checked object end.
    pub eof: bool,
}

/// Identity and size returned by an admitted HEAD operation.
pub struct AdmittedHead {
    /// Identity observed with the size.
    pub identity: ObjectIdentity,
    /// Object byte length.
    pub size: u64,
}

/// Limited footer result bound to the identity and size used by its decoder.
pub struct AdmittedFooter {
    /// Identity observed while decoding.
    pub identity: ObjectIdentity,
    /// Identity-checked object size used by the decoder.
    pub size: u64,
    /// Decoded footer whose retained owners were admitted before allocation.
    pub footer: ParquetFooter,
}

/// Producer-admitted storage result, validated against its outstanding request by the driver.
pub enum AdmittedIoEffect {
    /// One bounded discovery page.
    Listing(AdmittedListingPage),
    /// An identity-checked exact read.
    Read(AdmittedRead),
    /// An observed identity and size.
    Head(AdmittedHead),
    /// A bounded footer decode.
    Footer(AdmittedFooter),
}

/// Checked cumulative storage-effect measurements retained independently of source ownership.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct IoUsage {
    listing_pages: usize,
    descriptors: usize,
    read_requests: usize,
    requested_read_bytes: usize,
    received_read_bytes: usize,
    heads: usize,
    footers: usize,
}

impl IoUsage {
    /// Returns completed listing pages, including empty EOF pages.
    pub fn listing_pages(self) -> usize {
        self.listing_pages
    }

    /// Returns descriptors produced across completed listing pages.
    pub fn descriptors(self) -> usize {
        self.descriptors
    }

    /// Returns completed exact-range reads.
    pub fn read_requests(self) -> usize {
        self.read_requests
    }

    /// Returns bytes requested by completed exact-range reads.
    pub fn requested_read_bytes(self) -> usize {
        self.requested_read_bytes
    }

    /// Returns admitted owned bytes received from completed exact-range reads.
    pub fn received_read_bytes(self) -> usize {
        self.received_read_bytes
    }

    /// Returns completed HEAD effects.
    pub fn heads(self) -> usize {
        self.heads
    }

    /// Returns completed limited-footer effects.
    pub fn footers(self) -> usize {
        self.footers
    }

    fn with_listing(self, descriptors: usize) -> Result<Self, OperationFailure> {
        Ok(Self {
            listing_pages: checked_increment(self.listing_pages)?,
            descriptors: checked_add(self.descriptors, descriptors)?,
            ..self
        })
    }

    fn with_read(self, bytes: usize) -> Result<Self, OperationFailure> {
        Ok(Self {
            read_requests: checked_increment(self.read_requests)?,
            requested_read_bytes: checked_add(self.requested_read_bytes, bytes)?,
            received_read_bytes: checked_add(self.received_read_bytes, bytes)?,
            ..self
        })
    }

    fn with_head(self) -> Result<Self, OperationFailure> {
        Ok(Self {
            heads: checked_increment(self.heads)?,
            ..self
        })
    }

    fn with_footer(self) -> Result<Self, OperationFailure> {
        Ok(Self {
            footers: checked_increment(self.footers)?,
            ..self
        })
    }
}

/// Explicit producer-admitted storage boundary for operation tasks.
///
/// Implementations admit discovery, destination, decoder and retained-result allocations before
/// performing them. Listing pulls at most `entries` items and returns the last path only for a full
/// page. Reads and footers compare `expected_identity` before allocating their result. This trait
/// is deliberately separate from the ordinary unbounded storage and Parquet handlers.
pub trait AdmittedIoSource {
    /// Produces one admitted, sorted listing page without lookahead.
    fn list(
        &mut self,
        root: &str,
        continuation: Option<&str>,
        entries: usize,
        descriptor_bytes: usize,
        continuation_bytes: usize,
    ) -> Result<AdmittedListingPage, OperationFailure>;

    /// Reads one exact range after checking any previously observed object identity.
    fn read_exact(
        &mut self,
        path: &str,
        expected_identity: Option<ObjectIdentity>,
        offset: u64,
        length: usize,
    ) -> Result<AdmittedRead, OperationFailure>;

    /// Observes one object's identity and size together.
    fn head(&mut self, path: &str) -> Result<AdmittedHead, OperationFailure>;

    /// Decodes one footer with the injected limited reader after checking object identity.
    fn footer(
        &mut self,
        path: &str,
        expected_identity: ObjectIdentity,
        size: u64,
        limits: FooterLimits,
    ) -> Result<AdmittedFooter, OperationFailure>;

    /// Releases provider iterators, buffers, decoder state and cancellation mechanisms.
    fn cancel(&mut self) {}
}

impl AdmittedIoSource for () {
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
        _: &str,
        _: Option<ObjectIdentity>,
        _: u64,
        _: usize,
    ) -> Result<AdmittedRead, OperationFailure> {
        Err(OperationFailure::malformed_response())
    }

    fn head(&mut self, _: &str) -> Result<AdmittedHead, OperationFailure> {
        Err(OperationFailure::malformed_response())
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

struct ActiveEvaluation {
    key: EvaluationKey,
    page_limits: EvaluationPageLimits,
    reader: EvaluationReader,
}

struct ObjectBinding {
    path: String,
    identity: ObjectIdentity,
    size: Option<u64>,
    head_observed: bool,
}

struct ObjectBindings {
    entries: Vec<ObjectBinding>,
    retained_bytes: usize,
    byte_limit: usize,
    entry_limit: usize,
}

impl ObjectBindings {
    fn new(limits: TaskLimits) -> Result<Self, TaskProtocolError> {
        let byte_limit = limits.limit(Resource::TaskStateBytes);
        let entry_limit = limits.limit(Resource::Requests);
        let capacity = entry_limit.min(byte_limit / std::mem::size_of::<ObjectBinding>());
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(capacity)
            .map_err(|_| TaskProtocolError::InvalidLimits)?;
        let retained_bytes = entries
            .capacity()
            .checked_mul(std::mem::size_of::<ObjectBinding>())
            .ok_or(TaskProtocolError::InvalidLimits)?;
        if entries.capacity() > capacity || retained_bytes > byte_limit {
            return Err(TaskProtocolError::InvalidLimits);
        }
        Ok(Self {
            entries,
            retained_bytes,
            byte_limit,
            entry_limit,
        })
    }

    fn get(&self, path: &str) -> Option<&ObjectBinding> {
        self.entries.iter().find(|binding| binding.path == path)
    }

    fn admit_new(&self, path: &String) -> Result<(), OperationFailure> {
        if self.get(path).is_some() {
            return Ok(());
        }
        if self.entries.len() >= self.entry_limit {
            return Err(ResourceExhausted {
                resource: Resource::Requests,
                limit: self.entry_limit,
                observed: self.entries.len().saturating_add(1),
            }
            .into());
        }
        if self.entries.len() == self.entries.capacity() {
            return Err(ResourceExhausted {
                resource: Resource::TaskStateBytes,
                limit: self.byte_limit,
                observed: self.byte_limit.saturating_add(1),
            }
            .into());
        }
        let observed = self.retained_bytes.saturating_add(path.capacity());
        if observed > self.byte_limit {
            return Err(ResourceExhausted {
                resource: Resource::TaskStateBytes,
                limit: self.byte_limit,
                observed,
            }
            .into());
        }
        Ok(())
    }

    fn bind(
        &mut self,
        path: String,
        identity: ObjectIdentity,
        size: Option<u64>,
        head_observed: bool,
    ) -> Result<(), OperationFailure> {
        if let Some(binding) = self.entries.iter_mut().find(|binding| binding.path == path) {
            if binding.identity != identity
                || binding
                    .size
                    .zip(size)
                    .is_some_and(|(expected, actual)| expected != actual)
            {
                return Err(OperationFailure::malformed_response());
            }
            binding.size = binding.size.or(size);
            binding.head_observed |= head_observed;
            return Ok(());
        }
        self.admit_new(&path)?;
        self.retained_bytes = self
            .retained_bytes
            .checked_add(path.capacity())
            .ok_or_else(OperationFailure::malformed_response)?;
        self.entries.push(ObjectBinding {
            path,
            identity,
            size,
            head_observed,
        });
        Ok(())
    }
}

struct ListingBinding {
    root: String,
    continuation: String,
}

#[derive(Debug, Clone, Copy)]
struct RequestSequence {
    next: u64,
}

impl RequestSequence {
    fn checked_next(self, task_id: u64, key: RequestKey) -> Result<u64, TaskProtocolError> {
        if key.task_id() != task_id || key.request_id() != self.next {
            return Err(TaskProtocolError::WrongKey);
        }
        self.next
            .checked_add(1)
            .ok_or(TaskProtocolError::IdentityExhausted)
    }
}

/// Driver-owned state for one task's admitted storage and internal evaluation effects.
///
/// The driver separates request dispatch, effect completion and response transfer so callers can
/// yield between them. Its explicit I/O source and plan compiler admit their own allocations. It
/// does not adapt ordinary storage handlers, Parquet handlers or plan executors.
pub struct OperationDriver<S, C> {
    task_id: u64,
    sequence: RequestSequence,
    started: bool,
    effect: Option<TaskRequest>,
    response: Option<(RequestKey, Result<TaskResponseV1, OperationFailure>)>,
    active: Option<ActiveEvaluation>,
    external_active: Option<(EvaluationKey, EvaluationLimits)>,
    source: Option<S>,
    objects: Option<ObjectBindings>,
    listing: Option<ListingBinding>,
    usage: EvaluationUsage,
    io_usage: IoUsage,
    compiler: Option<C>,
    terminal: bool,
    cancelled: bool,
}

/// Evaluation-only operation driver with no storage-effect source.
pub type EvaluationDriver<C> = OperationDriver<(), C>;

impl<C> OperationDriver<(), C>
where
    C: FnMut(AdmittedPlan, EvaluationLimits) -> Result<AdmittedEvaluationSource, OperationFailure>,
{
    /// Allocates a fresh task identity and its initially idle evaluation driver.
    pub fn allocate(compiler: C) -> Result<(TaskId, Self), TaskProtocolError> {
        Self::allocate_inner(None, None, compiler)
    }
}

impl<S, C> OperationDriver<S, C>
where
    S: AdmittedIoSource,
    C: FnMut(AdmittedPlan, EvaluationLimits) -> Result<AdmittedEvaluationSource, OperationFailure>,
{
    /// Allocates a fresh task identity and a driver with an explicit admitted I/O source.
    ///
    /// The request-count and task-state allowances bound a preallocated per-path identity table.
    pub fn allocate_with_io(
        source: S,
        compiler: C,
        limits: TaskLimits,
    ) -> Result<(TaskId, Self), TaskProtocolError> {
        Self::allocate_inner(Some(source), Some(ObjectBindings::new(limits)?), compiler)
    }

    fn allocate_inner(
        source: Option<S>,
        objects: Option<ObjectBindings>,
        compiler: C,
    ) -> Result<(TaskId, Self), TaskProtocolError> {
        let id = TaskId::allocate()?;
        let task_id = id.get();
        Ok((
            id,
            Self {
                task_id,
                sequence: RequestSequence { next: 1 },
                started: false,
                effect: None,
                response: None,
                active: None,
                external_active: None,
                source,
                objects,
                listing: None,
                usage: EvaluationUsage::default(),
                io_usage: IoUsage::default(),
                compiler: Some(compiler),
                terminal: false,
                cancelled: false,
            },
        ))
    }

    /// Allocates a checked evaluation identity owned by this driver's task.
    pub fn allocate_evaluation(&self) -> Result<EvaluationKey, TaskProtocolError> {
        if self.terminal || self.started {
            return Err(TaskProtocolError::Terminal);
        }
        EvaluationKey::allocate(self.task_id)
    }

    /// Validates and takes one effect request without executing it.
    ///
    /// Wrong task/request identities, duplicate starts, changed page limits and non-evaluation
    /// effects leave the existing driver state unchanged.
    pub fn dispatch(&mut self, request: TaskRequest) -> Result<RequestKey, TaskProtocolError> {
        if self.terminal {
            return Err(TaskProtocolError::Terminal);
        }
        if self.effect.is_some() || self.response.is_some() {
            return Err(TaskProtocolError::PendingRequest);
        }
        let next = self.sequence.checked_next(self.task_id, request.key)?;
        self.validate_operation(&request.operation)?;
        if matches!(request.operation, TaskRequestV1::EvaluationStart { .. }) {
            self.started = true;
        }
        self.sequence.next = next;
        let key = request.key;
        self.effect = Some(request);
        Ok(key)
    }

    /// Executes the dispatched effect once and retains its owned response for transfer.
    pub fn complete_effect(&mut self) -> Result<RequestKey, TaskProtocolError> {
        if self.cancelled {
            return Err(TaskProtocolError::Terminal);
        }
        if self.response.is_some() {
            return Err(TaskProtocolError::PendingRequest);
        }
        let missing = if self.terminal {
            TaskProtocolError::Terminal
        } else {
            TaskProtocolError::NotStarted
        };
        let request = self.effect.take().ok_or(missing)?;
        let key = request.key;
        let response = self.execute(request.operation);
        if response.is_err() {
            self.release_terminal_state();
        }
        self.response = Some((key, response));
        Ok(key)
    }

    /// Transfers one completed response, preserving it when the supplied key is wrong.
    pub fn take_response(
        &mut self,
        key: RequestKey,
    ) -> Result<Result<TaskResponseV1, OperationFailure>, TaskProtocolError> {
        match self.response.as_ref() {
            Some((actual, _)) if *actual != key => return Err(TaskProtocolError::WrongKey),
            Some(_) => (),
            None if self.terminal => return Err(TaskProtocolError::Terminal),
            None => return Err(TaskProtocolError::NotStarted),
        }
        match self.response.take() {
            Some((_, response)) => Ok(response),
            None => Err(TaskProtocolError::NotStarted),
        }
    }

    /// Drops the matching queued effect or response and releases the active source without pulls.
    pub fn cancel(&mut self, pending: Option<RequestKey>) -> Result<(), TaskProtocolError> {
        if self.cancelled {
            return Ok(());
        }
        if pending != self.outstanding_key() {
            return Err(TaskProtocolError::WrongKey);
        }
        self.effect = None;
        self.response = None;
        if let Some(mut active) = self.active.take() {
            active.reader.cancel();
            self.usage = active.reader.usage();
        }
        if let Some(mut source) = self.source.take() {
            source.cancel();
        }
        self.compiler = None;
        self.external_active = None;
        self.objects = None;
        self.listing = None;
        self.terminal = true;
        self.cancelled = true;
        Ok(())
    }

    /// Returns the queued effect or completed response key retained by the driver.
    pub fn outstanding_key(&self) -> Option<RequestKey> {
        self.effect
            .as_ref()
            .map(|request| request.key)
            .or_else(|| self.response.as_ref().map(|(key, _)| *key))
    }

    /// Returns the active evaluation identity while its lazy source is retained.
    pub fn active_evaluation(&self) -> Option<EvaluationKey> {
        self.active
            .as_ref()
            .map(|active| active.key)
            .or_else(|| self.external_active.map(|active| active.0))
    }

    /// Returns cumulative evaluation work separately from live source ownership.
    pub fn evaluation_usage(&self) -> EvaluationUsage {
        self.usage
    }

    /// Returns checked cumulative storage-effect measurements.
    pub fn io_usage(&self) -> IoUsage {
        self.io_usage
    }

    fn validate_operation(&self, operation: &TaskRequestV1) -> Result<(), TaskProtocolError> {
        match operation {
            TaskRequestV1::List {
                root,
                continuation,
                entries,
                descriptor_bytes,
                continuation_bytes,
            } => {
                let minimum_descriptors = entries
                    .checked_mul(std::mem::size_of::<FileDescriptor>())
                    .ok_or(TaskProtocolError::WrongKind)?;
                if self.source.is_none()
                    || root.is_empty()
                    || !root.ends_with('/')
                    || *entries == 0
                    || minimum_descriptors > *descriptor_bytes
                    || *continuation_bytes == 0
                {
                    return Err(TaskProtocolError::WrongKind);
                }
                match (continuation.as_ref(), self.listing.as_ref()) {
                    (None, None) => {}
                    (Some(actual), Some(expected))
                        if actual.capacity() <= *continuation_bytes
                            && !actual.is_empty()
                            && actual.starts_with(root)
                            && expected.root == *root
                            && expected.continuation == *actual => {}
                    _ => return Err(TaskProtocolError::WrongKind),
                }
                Ok(())
            }
            TaskRequestV1::Read {
                path,
                offset,
                length,
            } => {
                let end = offset
                    .checked_add(
                        (*length)
                            .try_into()
                            .map_err(|_| TaskProtocolError::WrongKind)?,
                    )
                    .ok_or(TaskProtocolError::WrongKind)?;
                if self.source.is_none() || path.is_empty() || *length == 0 {
                    return Err(TaskProtocolError::WrongKind);
                }
                let binding = self.objects.as_ref().and_then(|objects| objects.get(path));
                if let Some(size) = binding.and_then(|binding| binding.size) {
                    if end > size {
                        return Err(TaskProtocolError::WrongKind);
                    }
                } else if *offset != 0 && binding.is_none() {
                    return Err(TaskProtocolError::WrongKind);
                }
                Ok(())
            }
            TaskRequestV1::Head { path } => {
                if self.source.is_none() || path.is_empty() {
                    return Err(TaskProtocolError::WrongKind);
                }
                Ok(())
            }
            TaskRequestV1::Footer { path, size, .. } => {
                let Some(binding) = self.objects.as_ref().and_then(|objects| objects.get(path))
                else {
                    return Err(TaskProtocolError::WrongKind);
                };
                if self.source.is_none() || !binding.head_observed || binding.size != Some(*size) {
                    return Err(TaskProtocolError::WrongKind);
                }
                Ok(())
            }
            TaskRequestV1::EvaluationStart { evaluation, .. } => {
                if evaluation.task_id() != self.task_id {
                    return Err(TaskProtocolError::WrongKey);
                }
                if self.started || self.active.is_some() || self.external_active.is_some() {
                    return Err(TaskProtocolError::WrongKind);
                }
                Ok(())
            }
            TaskRequestV1::Evaluation { evaluation, limits } => {
                let (active_key, page_limits) = self
                    .active
                    .as_ref()
                    .map(|active| (active.key, active.page_limits))
                    .or_else(|| {
                        self.external_active
                            .map(|(key, limits)| (key, limits.page()))
                    })
                    .ok_or(TaskProtocolError::WrongKind)?;
                if *evaluation != active_key {
                    return Err(TaskProtocolError::WrongKey);
                }
                if *limits != page_limits {
                    return Err(TaskProtocolError::WrongKind);
                }
                Ok(())
            }
        }
    }

    fn execute(&mut self, operation: TaskRequestV1) -> Result<TaskResponseV1, OperationFailure> {
        match operation {
            operation @ (TaskRequestV1::List { .. }
            | TaskRequestV1::Read { .. }
            | TaskRequestV1::Head { .. }
            | TaskRequestV1::Footer { .. }) => self.execute_io(operation),
            TaskRequestV1::EvaluationStart {
                evaluation,
                plan,
                limits,
            } => {
                let compiler = self
                    .compiler
                    .as_mut()
                    .ok_or_else(OperationFailure::malformed_response)?;
                let source = compiler(plan, limits);
                self.compiler = None;
                self.read_page(
                    evaluation,
                    limits.page(),
                    EvaluationReader::new(source?, limits),
                )
            }
            TaskRequestV1::Evaluation { evaluation, .. } => {
                let active = self
                    .active
                    .take()
                    .ok_or_else(OperationFailure::malformed_response)?;
                self.read_page(evaluation, active.page_limits, active.reader)
            }
        }
    }

    fn execute_io(&mut self, operation: TaskRequestV1) -> Result<TaskResponseV1, OperationFailure> {
        let expected = self.prepare_io(&operation)?;
        let source = self
            .source
            .as_mut()
            .ok_or_else(OperationFailure::malformed_response)?;
        let effect = match &operation {
            TaskRequestV1::List {
                root,
                continuation,
                entries,
                descriptor_bytes,
                continuation_bytes,
            } => AdmittedIoEffect::Listing(source.list(
                root,
                continuation.as_deref(),
                *entries,
                *descriptor_bytes,
                *continuation_bytes,
            )?),
            TaskRequestV1::Read {
                path,
                offset,
                length,
            } => AdmittedIoEffect::Read(source.read_exact(path, expected, *offset, *length)?),
            TaskRequestV1::Head { path } => AdmittedIoEffect::Head(source.head(path)?),
            TaskRequestV1::Footer { path, size, limits } => {
                AdmittedIoEffect::Footer(source.footer(
                    path,
                    expected.ok_or_else(OperationFailure::malformed_response)?,
                    *size,
                    *limits,
                )?)
            }
            _ => return Err(OperationFailure::malformed_response()),
        };
        self.finish_io(operation, effect)
    }

    // Runs before either synchronous I/O or an asynchronous host effect. No allocations.
    fn prepare_io(
        &self,
        operation: &TaskRequestV1,
    ) -> Result<Option<ObjectIdentity>, OperationFailure> {
        match operation {
            TaskRequestV1::Read { path, .. } | TaskRequestV1::Head { path } => {
                let objects = self
                    .objects
                    .as_ref()
                    .ok_or_else(OperationFailure::malformed_response)?;
                objects.admit_new(path)?;
                Ok(objects.get(path).map(|binding| binding.identity))
            }
            TaskRequestV1::Footer { path, size, .. } => self
                .objects
                .as_ref()
                .and_then(|objects| objects.get(path))
                .filter(|binding| binding.head_observed && binding.size == Some(*size))
                .map(|binding| Some(binding.identity))
                .ok_or_else(OperationFailure::malformed_response),
            _ => Ok(None),
        }
    }

    // Both completion paths enter the same identity, ordering, capacity and usage validation.
    fn finish_io(
        &mut self,
        operation: TaskRequestV1,
        effect: AdmittedIoEffect,
    ) -> Result<TaskResponseV1, OperationFailure> {
        match operation {
            TaskRequestV1::List {
                root,
                continuation,
                entries,
                descriptor_bytes,
                continuation_bytes,
            } => {
                let AdmittedIoEffect::Listing(page) = effect else {
                    return Err(OperationFailure::malformed_response());
                };
                validate_listing(
                    &root,
                    continuation.as_deref(),
                    entries,
                    descriptor_bytes,
                    continuation_bytes,
                    &page,
                )?;
                self.io_usage = self.io_usage.with_listing(page.files.len())?;
                self.listing = page
                    .binding
                    .map(|continuation| ListingBinding { root, continuation });
                Ok(TaskResponseV1::Listing {
                    files: page.files,
                    continuation: page.continuation,
                })
            }
            TaskRequestV1::Read {
                path,
                offset,
                length,
            } => {
                let objects = self
                    .objects
                    .as_mut()
                    .ok_or_else(OperationFailure::malformed_response)?;
                objects.admit_new(&path)?;
                let binding = objects.get(&path);
                let expected_identity = binding.map(|binding| binding.identity);
                let known_size = binding.and_then(|binding| binding.size);
                let head_observed = binding.is_some_and(|binding| binding.head_observed);
                let AdmittedIoEffect::Read(read) = effect else {
                    return Err(OperationFailure::malformed_response());
                };
                let length_u64 =
                    u64::try_from(length).map_err(|_| OperationFailure::malformed_response())?;
                let end = offset
                    .checked_add(length_u64)
                    .ok_or_else(OperationFailure::malformed_response)?;
                let expected_eof = known_size.map(|size| end == size);
                if read.offset != offset
                    || read.bytes.len() != length
                    || read.bytes.capacity() != length
                    || expected_identity.is_some_and(|identity| identity != read.identity)
                    || expected_eof.is_some_and(|eof| eof != read.eof)
                {
                    return Err(OperationFailure::malformed_response());
                }
                objects.bind(
                    path,
                    read.identity,
                    read.eof.then_some(end).or(known_size),
                    head_observed,
                )?;
                self.io_usage = self.io_usage.with_read(length)?;
                Ok(TaskResponseV1::Bytes {
                    offset,
                    bytes: read.bytes,
                    eof: read.eof,
                })
            }
            TaskRequestV1::Head { path } => {
                let objects = self
                    .objects
                    .as_mut()
                    .ok_or_else(OperationFailure::malformed_response)?;
                objects.admit_new(&path)?;
                let previous = objects
                    .get(&path)
                    .map(|binding| (binding.identity, binding.size));
                let AdmittedIoEffect::Head(head) = effect else {
                    return Err(OperationFailure::malformed_response());
                };
                if previous.is_some_and(|(identity, size)| {
                    identity != head.identity || size.is_some_and(|size| size != head.size)
                }) {
                    return Err(OperationFailure::malformed_response());
                }
                let size = head.size;
                objects.bind(path, head.identity, Some(size), true)?;
                self.io_usage = self.io_usage.with_head()?;
                Ok(TaskResponseV1::Head { size })
            }
            TaskRequestV1::Footer {
                path,
                size,
                limits: _,
            } => {
                let identity = self
                    .objects
                    .as_ref()
                    .and_then(|objects| objects.get(&path))
                    .filter(|binding| binding.head_observed && binding.size == Some(size))
                    .map(|binding| binding.identity)
                    .ok_or_else(OperationFailure::malformed_response)?;
                let AdmittedIoEffect::Footer(footer) = effect else {
                    return Err(OperationFailure::malformed_response());
                };
                if footer.identity != identity || footer.size != size {
                    return Err(OperationFailure::malformed_response());
                }
                self.io_usage = self.io_usage.with_footer()?;
                Ok(TaskResponseV1::Footer {
                    footer: footer.footer,
                })
            }
            _ => Err(OperationFailure::malformed_response()),
        }
    }

    fn read_page(
        &mut self,
        evaluation: EvaluationKey,
        page_limits: EvaluationPageLimits,
        mut reader: EvaluationReader,
    ) -> Result<TaskResponseV1, OperationFailure> {
        let page = reader.next_page();
        self.usage = reader.usage();
        match page {
            Ok(Some(page)) => {
                self.active = Some(ActiveEvaluation {
                    key: evaluation,
                    page_limits,
                    reader,
                });
                Ok(TaskResponseV1::Evaluation {
                    evaluation,
                    page: Some(page),
                })
            }
            Ok(None) => {
                if self.source.is_none() {
                    self.release_terminal_state();
                }
                Ok(TaskResponseV1::Evaluation {
                    evaluation,
                    page: None,
                })
            }
            Err(error) => Err(error),
        }
    }

    fn release_terminal_state(&mut self) {
        self.active = None;
        self.external_active = None;
        self.source = None;
        self.compiler = None;
        self.external_active = None;
        self.objects = None;
        self.listing = None;
        self.terminal = true;
    }
}

fn checked_increment(value: usize) -> Result<usize, OperationFailure> {
    checked_add(value, 1)
}

fn checked_add(left: usize, right: usize) -> Result<usize, OperationFailure> {
    left.checked_add(right)
        .ok_or_else(OperationFailure::malformed_response)
}

fn validate_listing(
    root: &str,
    after: Option<&str>,
    entries: usize,
    descriptor_bytes: usize,
    continuation_bytes: usize,
    page: &AdmittedListingPage,
) -> Result<(), OperationFailure> {
    if page.files.len() > entries {
        return Err(OperationFailure::malformed_response());
    }
    let base = page
        .files
        .capacity()
        .checked_mul(std::mem::size_of::<FileDescriptor>())
        .ok_or_else(OperationFailure::malformed_response)?;
    let retained = page.files.iter().try_fold(base, |used, file| {
        used.checked_add(file.path.capacity())
            .ok_or_else(OperationFailure::malformed_response)
    })?;
    if retained > descriptor_bytes
        || page
            .continuation
            .as_ref()
            .is_some_and(|value| value.capacity() > continuation_bytes)
        || page
            .binding
            .as_ref()
            .is_some_and(|value| value.capacity() > continuation_bytes)
    {
        return Err(OperationFailure::malformed_response());
    }
    let mut previous = after;
    for file in &page.files {
        if !file.path.starts_with(root) || previous.is_some_and(|path| file.path.as_str() <= path) {
            return Err(OperationFailure::malformed_response());
        }
        previous = Some(&file.path);
    }
    let expected_continuation = (page.files.len() == entries)
        .then(|| page.files.last().map(|file| file.path.as_str()))
        .flatten();
    if page.continuation.as_deref() != expected_continuation
        || page.binding.as_deref() != expected_continuation
    {
        return Err(OperationFailure::malformed_response());
    }
    Ok(())
}

/// Producer-admitted result of an asynchronous external effect.
pub enum AdmittedAsyncEffect {
    /// Storage metadata or bytes, checked by the same validator as synchronous I/O.
    Io(AdmittedIoEffect),
    /// One page from the single evaluation owned by this task's host.
    Evaluation {
        /// Identity bound to the host's execution stream.
        evaluation: EvaluationKey,
        /// An admitted page, or definitive end of the stream.
        page: Option<super::EvaluationPage>,
    },
}

/// Host-owned asynchronous effects. Implementations retain their own futures and execution
/// streams, and admit allocations before creating returned values. They must respect both the
/// request's page limits and the remaining cumulative evaluation allowances described by `usage`.
/// A returned page is validated again by the driver; that validation is not allocation admission.
pub trait AdmittedAsyncHost {
    /// Performs one already dispatched effect. The request remains owned by the driver until
    /// completion, including the admitted plan on the single evaluation start.
    fn complete<'a>(
        &'a mut self,
        request: &'a TaskRequestV1,
        expected_identity: Option<ObjectIdentity>,
        usage: EvaluationUsage,
        work: super::PendingWork<'a>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<AdmittedAsyncEffect, OperationFailure>> + 'a>,
    >;

    /// Releases execution streams, provider state and outstanding host resources without pulls.
    fn cancel(&mut self);
}

type NoCompiler =
    fn(AdmittedPlan, EvaluationLimits) -> Result<AdmittedEvaluationSource, OperationFailure>;

fn no_sync_compiler(
    _: AdmittedPlan,
    _: EvaluationLimits,
) -> Result<AdmittedEvaluationSource, OperationFailure> {
    Err(OperationFailure::malformed_response())
}

/// Asynchronous completion over the existing request/response driver. Futures and streams belong
/// to `host`; Kernel task state contains only admitted data and identities. Dropping this driver
/// cancels the host. Dropping an in-flight completion future leaves its request pending and
/// requires cancellation, preventing a second evaluation start or duplicate external side effect.
pub struct AsyncOperationDriver<H: AdmittedAsyncHost> {
    driver: OperationDriver<(), NoCompiler>,
    host: H,
    in_flight: bool,
}

impl<H: AdmittedAsyncHost> AsyncOperationDriver<H> {
    /// Allocates an independent task identity and a bounded shared I/O validation ledger.
    pub fn allocate(host: H, limits: TaskLimits) -> Result<(TaskId, Self), TaskProtocolError> {
        let (id, driver) =
            OperationDriver::allocate_with_io((), no_sync_compiler as NoCompiler, limits)?;
        Ok((
            id,
            Self {
                driver,
                host,
                in_flight: false,
            },
        ))
    }

    /// Allocates the one evaluation identity before dispatch starts.
    pub fn allocate_evaluation(&self) -> Result<EvaluationKey, TaskProtocolError> {
        self.driver.allocate_evaluation()
    }

    /// Uses the existing request-key, ordering, kind and one-start validation.
    pub fn dispatch(&mut self, request: TaskRequest) -> Result<RequestKey, TaskProtocolError> {
        if self.in_flight {
            return Err(TaskProtocolError::PendingRequest);
        }
        self.driver.dispatch(request)
    }

    /// Awaits exactly one host effect and validates its admitted result before retaining it.
    pub async fn complete_effect(
        &mut self,
        work: super::PendingWork<'_>,
    ) -> Result<RequestKey, TaskProtocolError> {
        if self.in_flight || self.driver.response.is_some() {
            return Err(TaskProtocolError::PendingRequest);
        }
        if self.driver.terminal {
            return Err(TaskProtocolError::Terminal);
        }
        let request = self
            .driver
            .effect
            .as_ref()
            .ok_or(TaskProtocolError::NotStarted)?;
        let key = request.key;
        if work.key() != key {
            return Err(TaskProtocolError::WrongKey);
        }
        let prepared = self
            .driver
            .prepare_io(&request.operation)
            .and_then(|identity| {
                let limits = match &request.operation {
                    TaskRequestV1::EvaluationStart { limits, .. } => Some(*limits),
                    TaskRequestV1::Evaluation { .. } => {
                        self.driver.external_active.map(|active| active.1)
                    }
                    _ => None,
                };
                if let Some(limits) = limits {
                    self.driver.usage.begin_page(limits)?;
                }
                Ok(identity)
            });
        let result = match prepared {
            Ok(expected_identity) => {
                self.in_flight = true;
                let result = self
                    .host
                    .complete(
                        &request.operation,
                        expected_identity,
                        self.driver.usage,
                        work,
                    )
                    .await;
                self.in_flight = false;
                result
            }
            Err(error) => Err(error),
        };
        let request = self
            .driver
            .effect
            .take()
            .ok_or(TaskProtocolError::NotStarted)?;
        let response = result.and_then(|effect| self.finish(request.operation, effect));
        if response.is_err() {
            self.host.cancel();
            self.driver.release_terminal_state();
        }
        self.driver.response = Some((key, response));
        Ok(key)
    }

    fn finish(
        &mut self,
        request: TaskRequestV1,
        effect: AdmittedAsyncEffect,
    ) -> Result<TaskResponseV1, OperationFailure> {
        match (request, effect) {
            (
                request @ (TaskRequestV1::List { .. }
                | TaskRequestV1::Read { .. }
                | TaskRequestV1::Head { .. }
                | TaskRequestV1::Footer { .. }),
                AdmittedAsyncEffect::Io(effect),
            ) => self.driver.finish_io(request, effect),
            (
                TaskRequestV1::EvaluationStart {
                    evaluation, limits, ..
                },
                AdmittedAsyncEffect::Evaluation {
                    evaluation: actual,
                    page,
                },
            ) => {
                self.driver.compiler = None;
                self.finish_page(evaluation, actual, limits, page)
            }
            (
                TaskRequestV1::Evaluation { evaluation, .. },
                AdmittedAsyncEffect::Evaluation {
                    evaluation: actual,
                    page,
                },
            ) => {
                let limits = self
                    .driver
                    .external_active
                    .ok_or_else(OperationFailure::malformed_response)?
                    .1;
                self.finish_page(evaluation, actual, limits, page)
            }
            _ => Err(OperationFailure::malformed_response()),
        }
    }

    fn finish_page(
        &mut self,
        expected: EvaluationKey,
        actual: EvaluationKey,
        limits: EvaluationLimits,
        page: Option<super::EvaluationPage>,
    ) -> Result<TaskResponseV1, OperationFailure> {
        if expected != actual {
            return Err(OperationFailure::malformed_response());
        }
        self.driver.usage.record_page(limits, page.as_ref())?;
        self.driver.external_active = page.as_ref().map(|_| (expected, limits));
        Ok(TaskResponseV1::Evaluation {
            evaluation: expected,
            page,
        })
    }

    /// Transfers the response only for the exact outstanding key.
    pub fn take_response(
        &mut self,
        key: RequestKey,
    ) -> Result<Result<TaskResponseV1, OperationFailure>, TaskProtocolError> {
        self.driver.take_response(key)
    }

    /// Cancels pending work and drops host execution resources without polling.
    pub fn cancel(&mut self, key: Option<RequestKey>) -> Result<(), TaskProtocolError> {
        self.driver.cancel(key)?;
        self.host.cancel();
        self.in_flight = false;
        Ok(())
    }

    /// Returns cumulative validated evaluation usage after completion or cancellation.
    pub fn evaluation_usage(&self) -> EvaluationUsage {
        self.driver.evaluation_usage()
    }

    /// Returns cumulative validated storage usage after completion or cancellation.
    pub fn io_usage(&self) -> IoUsage {
        self.driver.io_usage()
    }
}

impl<H: AdmittedAsyncHost> Drop for AsyncOperationDriver<H> {
    fn drop(&mut self) {
        self.host.cancel();
        let _ = self.driver.cancel(self.driver.outstanding_key());
    }
}

#[cfg(test)]
mod tests {
    use super::{IoUsage, RequestSequence};
    use crate::tasks::{FailureKind, RequestKey, TaskProtocolError};

    #[test]
    fn request_sequence_overflow_preserves_driver_state() {
        let sequence = RequestSequence { next: u64::MAX };
        let key = RequestKey::new(7, u64::MAX).unwrap();
        assert_eq!(
            sequence.checked_next(7, key),
            Err(TaskProtocolError::IdentityExhausted)
        );
        assert_eq!(sequence.next, u64::MAX);
    }

    #[test]
    fn io_usage_overflow_is_atomic_and_never_wraps() {
        let listing = IoUsage {
            listing_pages: 1,
            descriptors: usize::MAX,
            ..IoUsage::default()
        };
        assert_eq!(
            listing.with_listing(1).unwrap_err().kind(),
            FailureKind::MalformedResponse
        );
        assert_eq!(listing.listing_pages(), 1);
        assert_eq!(listing.descriptors(), usize::MAX);

        let read = IoUsage {
            read_requests: 1,
            requested_read_bytes: usize::MAX,
            received_read_bytes: 1,
            ..IoUsage::default()
        };
        assert_eq!(
            read.with_read(1).unwrap_err().kind(),
            FailureKind::MalformedResponse
        );
        assert_eq!(read.read_requests(), 1);
        assert_eq!(read.requested_read_bytes(), usize::MAX);
        assert_eq!(read.received_read_bytes(), 1);

        let head = IoUsage {
            heads: usize::MAX,
            ..IoUsage::default()
        };
        assert_eq!(
            head.with_head().unwrap_err().kind(),
            FailureKind::MalformedResponse
        );
        assert_eq!(head.heads(), usize::MAX);

        let footer = IoUsage {
            footers: usize::MAX,
            ..IoUsage::default()
        };
        assert_eq!(
            footer.with_footer().unwrap_err().kind(),
            FailureKind::MalformedResponse
        );
        assert_eq!(footer.footers(), usize::MAX);
    }
}
