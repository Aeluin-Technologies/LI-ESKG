//! Append-only resolution ledger with atomic visibility and replay checks.

use std::hash::Hash;
use std::sync::Arc;

use hashbrown::{HashMap, HashSet};
use li_core::{
    CommandEnvelope, CommitVersion, DecisionAction, DecisionId,
    DecisionRecord, HostEntityRef, HostRelation, IdempotencyKey, IdentityId,
    IdentityReference, IdentityStatus, InferenceId, InferenceRecord,
    MaterializationId, ObservationEnvelope, ObservationId, ResolutionCommand,
    RevertMode, Timestamp, TransactionId, VersionInterval,
};
use thiserror::Error;
use tracing::instrument;

use crate::errors::StorageError;
use crate::keys::ColumnFamily;
use crate::postcard_adapter::{deserialize, serialize};
use crate::traits::{KvBackend, KvOp};

const LEDGER_PREFIX: &[u8] = b"txn/";

/// Error returned when a command violates the  ledger contract.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LedgerError {
    /// Physical backend or command-envelope encoding failed.
    #[error("durable ledger storage failed: {0}")]
    Storage(#[from] StorageError),
    /// Persisted envelopes are missing, unordered, or fail replay checks.
    #[error("durable resolution log is corrupt")]
    CorruptDurableLog,
    /// Producer snapshot was stale and scores must be recomputed.
    #[error("version conflict: expected {expected}, current {current}")]
    VersionConflict {
        /// Producer snapshot version.
        expected: CommitVersion,
        /// Current durable version.
        current: CommitVersion,
    },
    /// Commit version reached the maximum integer value.
    #[error("commit version space exhausted")]
    VersionExhausted,
    /// Idempotency key was reused for a different transaction payload.
    #[error("idempotency key collision")]
    IdempotencyCollision,
    /// Producer transaction identifier already exists.
    #[error("transaction {0:?} already exists")]
    DuplicateTransaction(TransactionId),
    /// Observation identifier already belongs to immutable evidence.
    #[error("observation {0:?} already exists and cannot be overwritten")]
    ImmutableObservation(ObservationId),
    /// Referenced observation does not exist.
    #[error("observation {0:?} does not exist")]
    MissingObservation(ObservationId),
    /// Correction references evidence that does not exist.
    #[error("superseded observation {0:?} does not exist")]
    MissingSupersededObservation(ObservationId),
    /// Identity identifier is already present.
    #[error("identity {0:?} already exists")]
    DuplicateIdentity(IdentityId),
    /// Referenced identity does not exist.
    #[error("identity {0:?} does not exist")]
    MissingIdentity(IdentityId),
    /// Lifecycle transition is invalid for the current status.
    #[error("identity {identity:?} cannot transition from {status:?}")]
    InvalidLifecycle {
        /// Identity whose transition was rejected.
        identity: IdentityId,
        /// Current durable status.
        status: IdentityStatus,
    },
    /// Inference or decision identifier is already durable.
    #[error("durable record identifier already exists")]
    DuplicateRecord,
    /// Inference, decision, and observation references do not agree.
    #[error("inference and decision lineage is inconsistent")]
    InvalidLineage,
    /// Required provider, model, calibration, solver, or policy metadata is
    /// missing.
    #[error("inference and decision provenance is incomplete")]
    IncompleteProvenance,
    /// New-identity action and allocated identity metadata disagree.
    #[error(
        "create-identity action requires exactly one new identity allocation"
    )]
    InvalidCreation,
    /// Latent assignment selected an unavailable identity.
    #[error("selected latent identity is unavailable")]
    InvalidAssignment,
    /// Durable record validity does not open at the transaction version.
    #[error("new durable records must open at the commit version")]
    InvalidValidity,
    /// Explicit dependency transaction does not exist or is inactive.
    #[error("dependency transaction {0:?} does not exist or is inactive")]
    MissingDependency(TransactionId),
    /// Merge command contains invalid endpoints or support.
    #[error("merge command is structurally or contextually invalid")]
    InvalidMerge,
    /// Split references a missing or inactive merge decision.
    #[error("split references an inactive merge decision")]
    InvalidSplit,
    /// Strict revert target still has active dependents.
    #[error("strict revert target has active dependents")]
    ActiveDependents,
    /// Revert target does not exist or is already inactive.
    #[error("revert target is missing or inactive")]
    InvalidRevert,
    /// Materialization identifier already exists.
    #[error("materialization {0:?} already exists")]
    DuplicateMaterialization(MaterializationId),
    /// Materialization receipt does not match a pending outbox entry.
    #[error("materialization receipt does not match a pending outbox entry")]
    InvalidReceipt,
}

/// Successful atomic commit result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitResult {
    /// Version at which the effects became visible.
    pub version: CommitVersion,
    /// Whether this result came from an existing idempotency key.
    pub replayed: bool,
}

/// Pending idempotent host write stored in the transactional outbox.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializationOutbox {
    /// Stable materialization identifier.
    pub id: MaterializationId,
    /// Decision responsible for the host write.
    pub decision: DecisionId,
    /// Native relation declared by the host schema profile.
    pub relation: HostRelation,
    /// Ledger version that enqueued the write.
    pub commit: CommitVersion,
    /// Whether a successful receipt has been appended.
    pub completed: bool,
}

/// Successful host materialization receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterializationReceipt {
    /// Stable materialization identifier.
    pub materialization: MaterializationId,
    /// Responsible decision.
    pub decision: DecisionId,
    /// Ledger version that recorded success.
    pub commit: CommitVersion,
}

/// Durable append-only merge decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeRecord {
    /// Stable merge decision identifier.
    pub decision: DecisionId,
    /// First merge endpoint.
    pub left: IdentityId,
    /// Second merge endpoint.
    pub right: IdentityId,
    /// Association decisions supporting the merge.
    pub support: Box<[DecisionId]>,
    /// Merge policy version.
    pub policy_version: u64,
    /// Merge loss or operating-point version.
    pub loss_version: u64,
    /// Half-open durable validity interval.
    pub validity: VersionInterval,
    /// Transaction that introduced the merge.
    pub transaction: TransactionId,
}

/// Immutable committed transaction and its current revision status.
#[derive(Debug, Clone, PartialEq)]
pub struct TransactionRecord {
    /// Commit version.
    pub version: CommitVersion,
    /// Original deterministic command envelope.
    pub envelope: Arc<CommandEnvelope>,
    /// Whether later reversion leaves its interpretation current.
    pub active: bool,
}

#[derive(Debug)]
struct ValidatedBatch {
    next_version: CommitVersion,
}

enum CommitPreparation {
    Replay(CommitResult),
    Apply(ValidatedBatch),
}

/// Batch-local lifecycle overlay that avoids cloning the complete identity
/// index for transactional validation.
struct IdentityOverlay<'a> {
    base: &'a HashMap<IdentityId, IdentityStatus>,
    changes: HashMap<IdentityId, IdentityStatus>,
    capacity_hint: usize,
}

impl<'a> IdentityOverlay<'a> {
    fn with_capacity(
        base: &'a HashMap<IdentityId, IdentityStatus>,
        capacity: usize,
    ) -> Self {
        Self {
            base,
            changes: HashMap::new(),
            capacity_hint: capacity,
        }
    }

    fn get(&self, identity: IdentityId) -> Option<IdentityStatus> {
        self.changes
            .get(&identity)
            .copied()
            .or_else(|| self.base.get(&identity).copied())
    }

    fn contains(&self, identity: IdentityId) -> bool {
        self.changes.contains_key(&identity) ||
            self.base.contains_key(&identity)
    }

    fn insert_new(
        &mut self,
        identity: IdentityId,
        status: IdentityStatus,
    ) -> bool {
        if self.contains(identity) {
            return false;
        }
        self.reserve_if_needed();
        self.changes.insert(identity, status);
        true
    }

    fn replace(&mut self, identity: IdentityId, status: IdentityStatus) {
        self.reserve_if_needed();
        self.changes.insert(identity, status);
    }

    fn reserve_if_needed(&mut self) {
        if self.changes.is_empty() && self.changes.capacity() == 0 {
            self.changes.reserve(self.capacity_hint);
        }
    }
}

fn reserve_once<T>(set: &mut HashSet<T>, capacity: usize)
where
    T: Eq + Hash,
{
    if set.is_empty() && set.capacity() == 0 {
        set.reserve(capacity);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CanonicalNode {
    parent: IdentityId,
    minimum: IdentityId,
    size: usize,
}

/// In-memory reference ledger implementing  append-only semantics.
#[derive(Debug, Clone, Default)]
pub struct MemoryLedger {
    version: CommitVersion,
    observations: HashMap<ObservationId, ObservationEnvelope>,
    observation_commits: HashMap<ObservationId, CommitVersion>,
    inferences: HashMap<InferenceId, Arc<InferenceRecord>>,
    decisions: HashMap<DecisionId, Arc<DecisionRecord>>,
    current_inference: HashMap<ObservationId, InferenceId>,
    current_decision: HashMap<ObservationId, DecisionId>,
    identities: HashMap<IdentityId, IdentityStatus>,
    promotions: HashMap<IdentityId, HostEntityRef>,
    merges: HashMap<DecisionId, MergeRecord>,
    canonical: HashMap<IdentityId, CanonicalNode>,
    outbox: HashMap<MaterializationId, MaterializationOutbox>,
    receipts: Vec<MaterializationReceipt>,
    transactions: Vec<TransactionRecord>,
    transaction_indexes: HashMap<TransactionId, usize>,
    dependents: HashMap<TransactionId, Vec<TransactionId>>,
    idempotency: HashMap<IdempotencyKey, (CommitResult, TransactionId)>,
}

/// Append-only durable ledger backed by an atomic key-value implementation.
///
/// Each successful commit persists the original command envelope under its
/// monotonically increasing commit version. The in-memory indexes are rebuilt
/// by deterministic replay when the adapter opens.
#[derive(Debug)]
pub struct DurableLedger<B> {
    backend: B,
    memory: MemoryLedger,
}

/// Durable ledger boundary required by the transactional  runtime.
pub trait ResolutionLedger {
    /// Returns the latest atomic commit version.
    fn current_version(&self) -> CommitVersion;

    /// Atomically commits a validated command envelope.
    fn commit_envelope(
        &mut self,
        envelope: CommandEnvelope,
    ) -> Result<CommitResult, LedgerError>;

    /// Borrows one transactional outbox entry.
    fn materialization(
        &self,
        id: MaterializationId,
    ) -> Option<&MaterializationOutbox>;
}

impl ResolutionLedger for MemoryLedger {
    fn current_version(&self) -> CommitVersion {
        self.version()
    }

    fn commit_envelope(
        &mut self,
        envelope: CommandEnvelope,
    ) -> Result<CommitResult, LedgerError> {
        self.commit(envelope)
    }

    fn materialization(
        &self,
        id: MaterializationId,
    ) -> Option<&MaterializationOutbox> {
        self.outbox(id)
    }
}

impl<B: KvBackend> ResolutionLedger for DurableLedger<B> {
    fn current_version(&self) -> CommitVersion {
        self.memory.current_version()
    }

    fn commit_envelope(
        &mut self,
        envelope: CommandEnvelope,
    ) -> Result<CommitResult, LedgerError> {
        let validated = match self.memory.prepare(&envelope)? {
            CommitPreparation::Replay(result) => return Ok(result),
            CommitPreparation::Apply(validated) => validated,
        };
        let result = CommitResult {
            version: validated.next_version,
            replayed: false,
        };
        let key = durable_key(validated.next_version);
        let value = serialize(&envelope)?;
        self.backend.apply_transaction(&[KvOp::Put {
            cf: ColumnFamily::ResolutionLedger,
            key: key.to_vec(),
            value,
        }])?;
        self.memory
            .apply(Arc::new(envelope), validated.next_version);
        Ok(result)
    }

    fn materialization(
        &self,
        id: MaterializationId,
    ) -> Option<&MaterializationOutbox> {
        self.memory.materialization(id)
    }
}

impl<B: KvBackend> DurableLedger<B> {
    /// Opens a durable ledger and reconstructs all indexes by ordered replay.
    ///
    /// # Errors
    ///
    /// Returns [`LedgerError::Storage`] for backend/decoding failures and
    /// [`LedgerError::CorruptDurableLog`] when persisted commit keys or
    /// envelopes violate monotonic replay.
    pub fn open(backend: B) -> Result<Self, LedgerError> {
        let records = backend
            .prefix_scan(ColumnFamily::ResolutionLedger, LEDGER_PREFIX)?;
        let mut memory = MemoryLedger::with_capacity(records.len(), 0);
        for (index, (key, value)) in records.iter().enumerate() {
            let version = decode_durable_key(key)
                .ok_or(LedgerError::CorruptDurableLog)?;
            let expected = u64::try_from(index)
                .ok()
                .and_then(|offset| offset.checked_add(1))
                .map(CommitVersion::new)
                .ok_or(LedgerError::CorruptDurableLog)?;
            if version != expected {
                return Err(LedgerError::CorruptDurableLog);
            }
            let envelope: CommandEnvelope = deserialize(value)?;
            let result = memory.commit(envelope)?;
            if result.version != version || result.replayed {
                return Err(LedgerError::CorruptDurableLog);
            }
        }
        Ok(Self { backend, memory })
    }

    /// Borrows the reconstructed  ledger indexes.
    pub const fn memory(&self) -> &MemoryLedger {
        &self.memory
    }

    /// Consumes the adapter and returns its physical backend.
    pub fn into_backend(self) -> B {
        self.backend
    }
}

fn durable_key(version: CommitVersion) -> [u8; 12] {
    let mut key = [0_u8; 12];
    key[..LEDGER_PREFIX.len()].copy_from_slice(LEDGER_PREFIX);
    key[LEDGER_PREFIX.len()..].copy_from_slice(&version.get().to_be_bytes());
    key
}

fn decode_durable_key(key: &[u8]) -> Option<CommitVersion> {
    if key.len() != 12 || !key.starts_with(LEDGER_PREFIX) {
        return None;
    }
    let bytes: [u8; 8] = key[LEDGER_PREFIX.len()..].try_into().ok()?;
    Some(CommitVersion::new(u64::from_be_bytes(bytes)))
}

impl MemoryLedger {
    /// Creates an empty ledger with capacities sized for expected hot indexes.
    pub fn with_capacity(records: usize, identities: usize) -> Self {
        Self {
            observations: HashMap::with_capacity(records),
            observation_commits: HashMap::with_capacity(records),
            inferences: HashMap::with_capacity(records),
            decisions: HashMap::with_capacity(records),
            current_inference: HashMap::with_capacity(records),
            current_decision: HashMap::with_capacity(records),
            identities: HashMap::with_capacity(identities),
            promotions: HashMap::new(),
            merges: HashMap::new(),
            canonical: HashMap::with_capacity(identities),
            outbox: HashMap::new(),
            receipts: Vec::new(),
            transactions: Vec::with_capacity(records),
            transaction_indexes: HashMap::with_capacity(records),
            dependents: HashMap::new(),
            idempotency: HashMap::with_capacity(records),
            version: CommitVersion::ZERO,
        }
    }

    /// Atomically validates and appends a complete resolution command batch.
    ///
    /// A version conflict returns before mutation so the caller must recompute
    /// candidates and scores from fresh snapshots. Replaying the same
    /// idempotency key and envelope returns the original version without
    /// effects.
    ///
    /// # Errors
    ///
    /// Returns [`LedgerError`] when any command or dependency violates the
    /// contextual  invariants.
    #[instrument(skip_all, fields(expected = envelope.expected_version().get()))]
    pub fn commit(
        &mut self,
        envelope: CommandEnvelope,
    ) -> Result<CommitResult, LedgerError> {
        let validated = match self.prepare(&envelope)? {
            CommitPreparation::Replay(result) => return Ok(result),
            CommitPreparation::Apply(validated) => validated,
        };
        self.apply(Arc::new(envelope), validated.next_version);
        Ok(CommitResult {
            version: validated.next_version,
            replayed: false,
        })
    }

    fn prepare(
        &self,
        envelope: &CommandEnvelope,
    ) -> Result<CommitPreparation, LedgerError> {
        if let Some((result, transaction)) =
            self.idempotency.get(&envelope.idempotency())
        {
            let same = self
                .transaction_indexes
                .get(transaction)
                .and_then(|index| self.transactions.get(*index))
                .is_some_and(|record| record.envelope.as_ref() == envelope);
            if same {
                return Ok(CommitPreparation::Replay(CommitResult {
                    version: result.version,
                    replayed: true,
                }));
            }
            return Err(LedgerError::IdempotencyCollision);
        }
        let validated = self.validate(envelope)?;
        Ok(CommitPreparation::Apply(validated))
    }

    fn validate(
        &self,
        envelope: &CommandEnvelope,
    ) -> Result<ValidatedBatch, LedgerError> {
        if envelope.expected_version() != self.version {
            return Err(LedgerError::VersionConflict {
                expected: envelope.expected_version(),
                current: self.version,
            });
        }
        if self
            .transaction_indexes
            .contains_key(&envelope.transaction())
        {
            return Err(LedgerError::DuplicateTransaction(
                envelope.transaction(),
            ));
        }
        let next_version = self
            .version
            .checked_next()
            .ok_or(LedgerError::VersionExhausted)?;
        for dependency in envelope.depends_on() {
            let active = self
                .transaction_indexes
                .get(dependency)
                .and_then(|index| self.transactions.get(*index))
                .is_some_and(|record| record.active);
            if !active {
                return Err(LedgerError::MissingDependency(*dependency));
            }
        }

        let mut pending_observations = HashSet::new();
        let mut identity_status = IdentityOverlay::with_capacity(
            &self.identities,
            envelope.commands().len(),
        );
        let mut pending_inferences = HashSet::new();
        let mut pending_decisions = HashSet::new();
        let mut pending_materializations = HashSet::new();

        for command in envelope.commands() {
            match command {
                ResolutionCommand::PersistObservation(observation) => {
                    reserve_once(
                        &mut pending_observations,
                        envelope.commands().len(),
                    );
                    let id = observation.id();
                    if self.observations.contains_key(&id) ||
                        !pending_observations.insert(id)
                    {
                        return Err(LedgerError::ImmutableObservation(id));
                    }
                    if let Some(superseded) = observation.supersedes() &&
                        !self.observations.contains_key(&superseded) &&
                        !pending_observations.contains(&superseded)
                    {
                        return Err(
                            LedgerError::MissingSupersededObservation(
                                superseded,
                            ),
                        );
                    }
                },
                ResolutionCommand::CommitResolution {
                    inference,
                    decision,
                    created_identity,
                } => {
                    reserve_once(
                        &mut pending_inferences,
                        envelope.commands().len(),
                    );
                    reserve_once(
                        &mut pending_decisions,
                        envelope.commands().len(),
                    );
                    if !self.observations.contains_key(&inference.observation) &&
                        !pending_observations
                            .contains(&inference.observation)
                    {
                        return Err(LedgerError::MissingObservation(
                            inference.observation,
                        ));
                    }
                    if self.inferences.contains_key(&inference.id) ||
                        !pending_inferences.insert(inference.id) ||
                        self.decisions.contains_key(&decision.id) ||
                        !pending_decisions.insert(decision.id)
                    {
                        return Err(LedgerError::DuplicateRecord);
                    }
                    if decision.inference != inference.id ||
                        inference.validity.begin() != next_version ||
                        inference.validity.end().is_some() ||
                        decision.validity.begin() != next_version ||
                        decision.validity.end().is_some()
                    {
                        return Err(LedgerError::InvalidLineage);
                    }
                    if inference.provenance.providers.is_empty() ||
                        inference.provenance.providers.iter().any(
                            |artifact| {
                                artifact.provider.0 == 0 ||
                                    artifact.schema.0 == 0 ||
                                    artifact.model_version == 0 ||
                                    artifact.calibration_id == 0
                            },
                        ) ||
                        inference.diagnostics.solver_version == 0 ||
                        decision.policy_version == 0 ||
                        decision.loss_version == 0
                    {
                        return Err(LedgerError::IncompleteProvenance);
                    }
                    match (&decision.action, created_identity) {
                        (DecisionAction::CreateIdentity, Some(identity)) => {
                            if !identity_status
                                .insert_new(*identity, IdentityStatus::Active)
                            {
                                return Err(LedgerError::DuplicateIdentity(
                                    *identity,
                                ));
                            }
                        },
                        (DecisionAction::CreateIdentity, None) => {
                            return Err(LedgerError::InvalidCreation);
                        },
                        (
                            DecisionAction::Assign(IdentityReference::Latent(
                                identity,
                            )),
                            None,
                        ) => {
                            let status =
                                identity_status.get(*identity).ok_or(
                                    LedgerError::MissingIdentity(*identity),
                                )?;
                            if !matches!(
                                status,
                                IdentityStatus::Active |
                                    IdentityStatus::Dormant
                            ) {
                                return Err(LedgerError::InvalidAssignment);
                            }
                        },
                        (
                            DecisionAction::Assign(IdentityReference::Known(
                                _,
                            )),
                            None,
                        ) |
                        (DecisionAction::RejectNoise, None) |
                        (DecisionAction::Abstain, None) => {},
                        (_, Some(_)) => {
                            return Err(LedgerError::InvalidCreation);
                        },
                    }
                },
                ResolutionCommand::CreateIdentity { identity, .. } => {
                    if !identity_status
                        .insert_new(*identity, IdentityStatus::Active)
                    {
                        return Err(LedgerError::DuplicateIdentity(*identity));
                    }
                },
                ResolutionCommand::Promote { identity, .. } => {
                    let status = identity_status
                        .get(*identity)
                        .ok_or(LedgerError::MissingIdentity(*identity))?;
                    if !matches!(
                        status,
                        IdentityStatus::Active | IdentityStatus::Dormant
                    ) {
                        return Err(LedgerError::InvalidLifecycle {
                            identity: *identity,
                            status,
                        });
                    }
                    identity_status
                        .replace(*identity, IdentityStatus::Resolved);
                },
                ResolutionCommand::Merge {
                    decision,
                    left,
                    right,
                    support,
                    ..
                } => {
                    reserve_once(
                        &mut pending_decisions,
                        envelope.commands().len(),
                    );
                    if left == right ||
                        self.decisions.contains_key(decision) ||
                        self.merges.contains_key(decision) ||
                        !pending_decisions.insert(*decision)
                    {
                        return Err(LedgerError::InvalidMerge);
                    }
                    for identity in [left, right] {
                        let status = identity_status
                            .get(*identity)
                            .ok_or(LedgerError::MissingIdentity(*identity))?;
                        if !matches!(
                            status,
                            IdentityStatus::Active |
                                IdentityStatus::Dormant |
                                IdentityStatus::Merged
                        ) {
                            return Err(LedgerError::InvalidMerge);
                        }
                    }
                    if support.iter().any(|id| {
                        !self.decisions.contains_key(id) &&
                            !pending_decisions.contains(id)
                    }) {
                        return Err(LedgerError::InvalidMerge);
                    }
                },
                ResolutionCommand::Split(plan) => {
                    if plan.close_merges.iter().any(|decision| {
                        !self.merges.get(decision).is_some_and(|record| {
                            record.validity.end().is_none()
                        })
                    }) {
                        return Err(LedgerError::InvalidSplit);
                    }
                    for partition in &plan.partitions {
                        if !identity_status.contains(partition.identity) {
                            return Err(LedgerError::MissingIdentity(
                                partition.identity,
                            ));
                        }
                        for observation in &partition.observations {
                            if !self.observations.contains_key(observation) &&
                                !pending_observations.contains(observation)
                            {
                                return Err(LedgerError::MissingObservation(
                                    *observation,
                                ));
                            }
                        }
                        identity_status.replace(
                            partition.identity,
                            IdentityStatus::Active,
                        );
                    }
                },
                ResolutionCommand::Dormant(identity) => {
                    Self::validate_lifecycle(
                        &mut identity_status,
                        *identity,
                        IdentityStatus::Active,
                        IdentityStatus::Dormant,
                    )?;
                },
                ResolutionCommand::Reactivate(identity) => {
                    Self::validate_lifecycle(
                        &mut identity_status,
                        *identity,
                        IdentityStatus::Dormant,
                        IdentityStatus::Active,
                    )?;
                },
                ResolutionCommand::Retire(identity) => {
                    let status = identity_status
                        .get(*identity)
                        .ok_or(LedgerError::MissingIdentity(*identity))?;
                    if !matches!(
                        status,
                        IdentityStatus::Active | IdentityStatus::Dormant
                    ) {
                        return Err(LedgerError::InvalidLifecycle {
                            identity: *identity,
                            status,
                        });
                    }
                    identity_status
                        .replace(*identity, IdentityStatus::Retired);
                },
                ResolutionCommand::Revert { transaction, mode } => {
                    let active = self
                        .transaction_indexes
                        .get(transaction)
                        .and_then(|index| self.transactions.get(*index))
                        .is_some_and(|record| record.active);
                    if !active {
                        return Err(LedgerError::InvalidRevert);
                    }
                    if *mode == RevertMode::Strict &&
                        self.dependents.get(transaction).is_some_and(
                            |dependents| {
                                dependents.iter().any(|dependent| {
                                    self.transaction_indexes
                                        .get(dependent)
                                        .and_then(|index| {
                                            self.transactions.get(*index)
                                        })
                                        .is_some_and(|record| record.active)
                                })
                            },
                        )
                    {
                        return Err(LedgerError::ActiveDependents);
                    }
                },
                ResolutionCommand::EnqueueMaterialization {
                    materialization,
                    decision,
                    ..
                } => {
                    reserve_once(
                        &mut pending_materializations,
                        envelope.commands().len(),
                    );
                    if self.outbox.contains_key(materialization) ||
                        !pending_materializations.insert(*materialization)
                    {
                        return Err(LedgerError::DuplicateMaterialization(
                            *materialization,
                        ));
                    }
                    if !self.decisions.contains_key(decision) &&
                        !pending_decisions.contains(decision)
                    {
                        return Err(LedgerError::InvalidLineage);
                    }
                },
                ResolutionCommand::Materialized {
                    materialization,
                    decision,
                } => {
                    let valid = self.outbox.get(materialization).is_some_and(
                        |entry| {
                            !entry.completed && entry.decision == *decision
                        },
                    );
                    if !valid {
                        return Err(LedgerError::InvalidReceipt);
                    }
                },
            }
        }
        Ok(ValidatedBatch { next_version })
    }

    fn validate_lifecycle(
        statuses: &mut IdentityOverlay<'_>,
        identity: IdentityId,
        expected: IdentityStatus,
        replacement: IdentityStatus,
    ) -> Result<(), LedgerError> {
        let status = statuses
            .get(identity)
            .ok_or(LedgerError::MissingIdentity(identity))?;
        if status != expected {
            return Err(LedgerError::InvalidLifecycle { identity, status });
        }
        statuses.replace(identity, replacement);
        Ok(())
    }

    fn apply(
        &mut self,
        envelope: Arc<CommandEnvelope>,
        version: CommitVersion,
    ) {
        for command in envelope.commands() {
            match command {
                ResolutionCommand::PersistObservation(observation) => {
                    self.observation_commits.insert(observation.id(), version);
                    self.observations
                        .insert(observation.id(), observation.clone());
                },
                ResolutionCommand::CommitResolution {
                    inference,
                    decision,
                    created_identity,
                } => {
                    self.close_current(inference.observation, version);
                    self.current_inference
                        .insert(inference.observation, inference.id);
                    self.current_decision
                        .insert(inference.observation, decision.id);
                    self.inferences.insert(inference.id, inference.clone());
                    self.decisions.insert(decision.id, decision.clone());
                    if let Some(identity) = created_identity {
                        self.identities
                            .insert(*identity, IdentityStatus::Active);
                        self.insert_canonical(*identity);
                    }
                },
                ResolutionCommand::CreateIdentity { identity, .. } => {
                    self.identities.insert(*identity, IdentityStatus::Active);
                    self.insert_canonical(*identity);
                },
                ResolutionCommand::Promote { identity, target } => {
                    self.identities
                        .insert(*identity, IdentityStatus::Resolved);
                    self.promotions.insert(*identity, target.clone());
                },
                ResolutionCommand::Merge {
                    decision,
                    left,
                    right,
                    support,
                    policy_version,
                    loss_version,
                } => {
                    self.merges.insert(
                        *decision,
                        MergeRecord {
                            decision: *decision,
                            left: *left,
                            right: *right,
                            support: support.clone(),
                            policy_version: *policy_version,
                            loss_version: *loss_version,
                            validity: VersionInterval::current(version),
                            transaction: envelope.transaction(),
                        },
                    );
                    if let Some((minimum, replaced)) =
                        self.union_canonical(*left, *right)
                    {
                        self.identities
                            .insert(replaced, IdentityStatus::Merged);
                        if self.identities.get(&minimum) ==
                            Some(&IdentityStatus::Merged)
                        {
                            self.identities
                                .insert(minimum, IdentityStatus::Active);
                        }
                    }
                },
                ResolutionCommand::Split(plan) => {
                    for decision in &plan.close_merges {
                        if let Some(record) = self.merges.get_mut(decision) {
                            let begin = record.validity.begin();
                            if let Ok(validity) =
                                VersionInterval::new(begin, Some(version))
                            {
                                record.validity = validity;
                            }
                        }
                    }
                    for partition in &plan.partitions {
                        self.identities.insert(
                            partition.identity,
                            IdentityStatus::Active,
                        );
                    }
                    self.rebuild_canonical();
                },
                ResolutionCommand::Dormant(identity) => {
                    self.identities.insert(*identity, IdentityStatus::Dormant);
                },
                ResolutionCommand::Reactivate(identity) => {
                    self.identities.insert(*identity, IdentityStatus::Active);
                },
                ResolutionCommand::Retire(identity) => {
                    self.identities.insert(*identity, IdentityStatus::Retired);
                },
                ResolutionCommand::Revert { transaction, mode } => {
                    if *mode != RevertMode::Compensating {
                        let targets = self.revert_targets(*transaction, *mode);
                        for target in targets {
                            self.deactivate_transaction(target, version);
                        }
                    }
                },
                ResolutionCommand::EnqueueMaterialization {
                    materialization,
                    decision,
                    relation,
                } => {
                    self.outbox.insert(
                        *materialization,
                        MaterializationOutbox {
                            id: *materialization,
                            decision: *decision,
                            relation: *relation,
                            commit: version,
                            completed: false,
                        },
                    );
                },
                ResolutionCommand::Materialized {
                    materialization,
                    decision,
                } => {
                    if let Some(entry) = self.outbox.get_mut(materialization) {
                        entry.completed = true;
                    }
                    self.receipts.push(MaterializationReceipt {
                        materialization: *materialization,
                        decision: *decision,
                        commit: version,
                    });
                },
            }
        }

        let transaction = envelope.transaction();
        let idempotency = envelope.idempotency();
        for dependency in envelope.depends_on() {
            self.dependents
                .entry(*dependency)
                .or_default()
                .push(transaction);
        }
        let index = self.transactions.len();
        self.transactions.push(TransactionRecord {
            version,
            envelope,
            active: true,
        });
        self.transaction_indexes.insert(transaction, index);
        let result = CommitResult {
            version,
            replayed: false,
        };
        self.idempotency.insert(idempotency, (result, transaction));
        self.version = version;
    }

    fn close_current(
        &mut self,
        observation: ObservationId,
        end: CommitVersion,
    ) {
        if let Some(inference_id) =
            self.current_inference.get(&observation).copied() &&
            let Some(record) = self.inferences.get_mut(&inference_id)
        {
            let begin = record.validity.begin();
            if let Ok(validity) = VersionInterval::new(begin, Some(end)) {
                Arc::make_mut(record).validity = validity;
            }
        }
        if let Some(decision_id) =
            self.current_decision.get(&observation).copied() &&
            let Some(record) = self.decisions.get_mut(&decision_id)
        {
            let begin = record.validity.begin();
            if let Ok(validity) = VersionInterval::new(begin, Some(end)) {
                Arc::make_mut(record).validity = validity;
            }
        }
    }

    fn revert_targets(
        &self,
        transaction: TransactionId,
        mode: RevertMode,
    ) -> Vec<TransactionId> {
        let mut output = vec![transaction];
        if mode != RevertMode::Cascade {
            return output;
        }
        let mut cursor = 0;
        let mut visited: HashSet<TransactionId> = HashSet::new();
        visited.insert(transaction);
        while cursor < output.len() {
            if let Some(dependents) = self.dependents.get(&output[cursor]) {
                for dependent in dependents {
                    if visited.insert(*dependent) {
                        output.push(*dependent);
                    }
                }
            }
            cursor = cursor.saturating_add(1);
        }
        output
    }

    fn deactivate_transaction(
        &mut self,
        transaction: TransactionId,
        end: CommitVersion,
    ) {
        let Some(index) = self.transaction_indexes.get(&transaction).copied()
        else {
            return;
        };
        let commands = self
            .transactions
            .get(index)
            .map(|record| record.envelope.commands().to_vec());
        if let Some(record) = self.transactions.get_mut(index) {
            record.active = false;
        }
        let Some(commands) = commands else {
            return;
        };
        for command in commands {
            match command {
                ResolutionCommand::CommitResolution {
                    inference,
                    decision,
                    ..
                } => {
                    if self.current_inference.get(&inference.observation) ==
                        Some(&inference.id)
                    {
                        self.close_current(inference.observation, end);
                        self.current_inference.remove(&inference.observation);
                        self.current_decision.remove(&inference.observation);
                    }
                    if let Some(record) = self.decisions.get_mut(&decision.id) &&
                        let Ok(validity) = VersionInterval::new(
                            record.validity.begin(),
                            Some(end),
                        )
                    {
                        Arc::make_mut(record).validity = validity;
                    }
                },
                ResolutionCommand::Merge { decision, .. } => {
                    if let Some(record) = self.merges.get_mut(&decision) &&
                        record.validity.end().is_none() &&
                        let Ok(validity) = VersionInterval::new(
                            record.validity.begin(),
                            Some(end),
                        )
                    {
                        record.validity = validity;
                    }
                },
                ResolutionCommand::EnqueueMaterialization {
                    materialization,
                    ..
                } => {
                    self.outbox.remove(&materialization);
                },
                _ => {},
            }
        }
        self.rebuild_canonical();
    }

    fn rebuild_canonical(&mut self) {
        self.canonical.clear();
        let identities: Vec<_> = self.identities.keys().copied().collect();
        for identity in identities {
            self.insert_canonical(identity);
        }
        let active_edges: Vec<(IdentityId, IdentityId)> = self
            .merges
            .values()
            .filter(|record| record.validity.end().is_none())
            .map(|record| (record.left, record.right))
            .collect();
        for (left, right) in active_edges {
            let _ = self.union_canonical(left, right);
        }
        let identities: Vec<_> = self.identities.keys().copied().collect();
        for identity in identities {
            if self
                .canonical(identity)
                .is_some_and(|minimum| minimum != identity)
            {
                self.identities.insert(identity, IdentityStatus::Merged);
            }
        }
    }

    fn insert_canonical(&mut self, identity: IdentityId) {
        self.canonical.insert(
            identity,
            CanonicalNode {
                parent: identity,
                minimum: identity,
                size: 1,
            },
        );
    }

    fn root(&self, identity: IdentityId) -> Option<IdentityId> {
        let mut current = identity;
        let mut remaining = self.canonical.len();
        while remaining > 0 {
            let node = self.canonical.get(&current)?;
            if node.parent == current {
                return Some(current);
            }
            current = node.parent;
            remaining = remaining.saturating_sub(1);
        }
        None
    }

    /// Unites two canonical classes by size and returns the new and replaced
    /// deterministic minima.
    fn union_canonical(
        &mut self,
        left: IdentityId,
        right: IdentityId,
    ) -> Option<(IdentityId, IdentityId)> {
        let left_root = self.root(left)?;
        let right_root = self.root(right)?;
        if left_root == right_root {
            return None;
        }
        let left_node = *self.canonical.get(&left_root)?;
        let right_node = *self.canonical.get(&right_root)?;
        let (parent, child, parent_node, child_node) = if left_node.size >
            right_node.size ||
            (left_node.size == right_node.size && left_root < right_root)
        {
            (left_root, right_root, left_node, right_node)
        } else {
            (right_root, left_root, right_node, left_node)
        };
        let minimum = parent_node.minimum.min(child_node.minimum);
        let replaced = parent_node.minimum.max(child_node.minimum);
        if let Some(node) = self.canonical.get_mut(&child) {
            node.parent = parent;
        }
        if let Some(node) = self.canonical.get_mut(&parent) {
            node.minimum = minimum;
            node.size = parent_node.size.saturating_add(child_node.size);
        }
        Some((minimum, replaced))
    }

    /// Returns the latest atomic commit version.
    pub const fn version(&self) -> CommitVersion {
        self.version
    }

    /// Borrows immutable evidence by stable identifier.
    pub fn observation(
        &self,
        id: ObservationId,
    ) -> Option<&ObservationEnvelope> {
        self.observations.get(&id)
    }

    /// Borrows the current decision for an observation.
    pub fn current_decision(
        &self,
        observation: ObservationId,
    ) -> Option<&DecisionRecord> {
        self.current_decision
            .get(&observation)
            .and_then(|id| self.decisions.get(id))
            .map(Arc::as_ref)
    }

    /// Borrows the decision valid at one historical commit version.
    pub fn decision_as_of(
        &self,
        observation: ObservationId,
        version: CommitVersion,
    ) -> Option<&DecisionRecord> {
        self.decisions
            .values()
            .find(|decision| {
                self.inferences.get(&decision.inference).is_some_and(
                    |inference| inference.observation == observation,
                ) && decision.validity.contains(version)
            })
            .map(Arc::as_ref)
    }

    /// Returns the deterministic representative of the active merge class.
    pub fn canonical(&self, identity: IdentityId) -> Option<IdentityId> {
        let root = self.root(identity)?;
        self.canonical.get(&root).map(|node| node.minimum)
    }

    /// Borrows one outbox entry for idempotent host delivery.
    pub fn outbox(
        &self,
        id: MaterializationId,
    ) -> Option<&MaterializationOutbox> {
        self.outbox.get(&id)
    }

    /// Appends observations visible by commit and event-time bounds into
    /// `output`.
    ///
    /// The caller owns and reuses `output`; results are sorted by stable ID
    /// for deterministic bitemporal queries.
    pub fn observations_as_of<'a>(
        &'a self,
        event_time: Timestamp,
        commit: CommitVersion,
        output: &mut Vec<&'a ObservationEnvelope>,
    ) {
        output.clear();
        output.extend(self.observations.values().filter(|observation| {
            observation.event_time() <= event_time &&
                self.observation_commits
                    .get(&observation.id())
                    .is_some_and(|version| *version <= commit)
        }));
        output.sort_unstable_by_key(|observation| observation.id());
    }

    /// Borrows the append-only ordered transaction log.
    pub fn transactions(&self) -> &[TransactionRecord] {
        &self.transactions
    }

    /// Computes the active transitive dependency closure into caller storage.
    pub fn dependency_closure(
        &self,
        transaction: TransactionId,
        output: &mut Vec<TransactionId>,
    ) {
        output.clear();
        output.extend(self.revert_targets(transaction, RevertMode::Cascade));
        output.sort_unstable();
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use li_core::{
        AssociationOutcome, BoundaryTreatment, ContentHash, DecisionAction,
        EvidenceError, IdempotencyKey, InferenceProvenance, Modality,
        NormalizedDistribution, OutcomeProbability, PayloadRef,
        ProviderArtifact, ProviderId, QualityMetadata, SchemaId,
        SolverDiagnostics, SolverStoppingReason, SourceId,
    };

    use super::*;
    use crate::MemoryKvBackend;

    #[derive(Debug, Default)]
    struct RejectingBackend;

    impl KvBackend for RejectingBackend {
        fn get(
            &self,
            _cf: ColumnFamily,
            _key: &[u8],
        ) -> Result<Option<Vec<u8>>, StorageError> {
            Ok(None)
        }

        fn prefix_scan(
            &self,
            _cf: ColumnFamily,
            _prefix: &[u8],
        ) -> Result<Vec<(Vec<u8>, Vec<u8>)>, StorageError> {
            Ok(Vec::new())
        }

        fn apply_transaction(
            &mut self,
            _batch: &[KvOp],
        ) -> Result<(), StorageError> {
            Err(StorageError::TransactionFailed)
        }
    }

    fn key(value: u8) -> Result<IdempotencyKey, li_core::CommandError> {
        IdempotencyKey::new([value; 16])
    }

    fn observation(
        id: u64,
        supersedes: Option<ObservationId>,
    ) -> Result<ObservationEnvelope, EvidenceError> {
        ObservationEnvelope::new(
            ObservationId(id),
            SourceId(1),
            Modality(1),
            Timestamp::from_micros(i64::try_from(id).unwrap_or(i64::MAX)),
            Timestamp::from_micros(i64::try_from(id).unwrap_or(i64::MAX)),
            PayloadRef::Inline(Bytes::from_static(b"payload")),
            QualityMetadata::Opaque {
                schema: SchemaId(1),
                bytes: Bytes::new(),
            },
            ContentHash::new([u8::try_from(id).unwrap_or(u8::MAX); 32]),
            supersedes,
        )
    }

    fn envelope(
        transaction: u64,
        expected: u64,
        commands: Vec<ResolutionCommand>,
    ) -> Result<CommandEnvelope, li_core::CommandError> {
        let key = key(u8::try_from(transaction).unwrap_or(u8::MAX))?;
        CommandEnvelope::new(
            TransactionId(transaction),
            CommitVersion::new(expected),
            key,
            Timestamp::from_micros(
                i64::try_from(transaction).unwrap_or(i64::MAX),
            ),
            Vec::new(),
            commands,
        )
    }

    fn inference_and_decision(
        version: u64,
        observation: ObservationId,
        action: DecisionAction,
    ) -> Result<
        (Arc<InferenceRecord>, Arc<DecisionRecord>),
        Box<dyn std::error::Error>,
    > {
        let target = match &action {
            DecisionAction::Assign(reference) => Some(reference.clone()),
            _ => None,
        };
        let mut entries = vec![
            OutcomeProbability {
                outcome: AssociationOutcome::New,
                probability: li_core::Probability::new(
                    if matches!(action, DecisionAction::CreateIdentity) {
                        0.8
                    } else {
                        0.1
                    },
                ),
            },
            OutcomeProbability {
                outcome: AssociationOutcome::Noise,
                probability: li_core::Probability::new(
                    if matches!(action, DecisionAction::CreateIdentity) {
                        0.2
                    } else {
                        0.1
                    },
                ),
            },
        ];
        if let Some(reference) = target {
            entries.push(OutcomeProbability {
                outcome: AssociationOutcome::Identity(reference),
                probability: li_core::Probability::new(0.8),
            });
        }
        if matches!(
            action,
            DecisionAction::RejectNoise | DecisionAction::Abstain
        ) {
            entries[0].probability = li_core::Probability::new(0.45);
            entries[1].probability = li_core::Probability::new(0.55);
        }
        let distribution = NormalizedDistribution::new(entries, None)?;
        let inference = InferenceRecord {
            id: InferenceId(version),
            observation,
            distribution: distribution.clone(),
            contributions: Vec::new().into_boxed_slice(),
            provenance: Arc::new(InferenceProvenance {
                providers: vec![ProviderArtifact {
                    provider: ProviderId(1),
                    schema: SchemaId(1),
                    model_version: 1,
                    calibration_id: 1,
                }]
                .into_boxed_slice(),
                candidate_version: 1,
                host_snapshot: CommitVersion::new(version.saturating_sub(1)),
                configuration_hash: ContentHash::new([1; 32]),
            }),
            diagnostics: Arc::new(SolverDiagnostics {
                solver_version: 1,
                tolerance: 1.0e-9,
                iterations: 1,
                residual: 0.0,
                damping_schedule: Vec::new().into_boxed_slice(),
                stopping_reason: SolverStoppingReason::Exact,
                boundary_treatment: BoundaryTreatment::Global,
                random_seed: None,
            }),
            validity: VersionInterval::current(CommitVersion::new(version)),
        };
        let decision = DecisionRecord::new(
            DecisionId(version),
            inference.id,
            action,
            1,
            1,
            VersionInterval::current(CommitVersion::new(version)),
            &distribution,
        )?;
        Ok((Arc::new(inference), Arc::new(decision)))
    }

    #[test]
    fn evidence_is_immutable_and_idempotent_replay_has_no_effect()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut ledger = MemoryLedger::default();
        let command = envelope(
            1,
            0,
            vec![ResolutionCommand::PersistObservation(observation(1, None)?)],
        )?;
        let first = ledger.commit(command.clone())?;
        let replay = ledger.commit(command)?;
        assert_eq!(first.version, CommitVersion::new(1));
        assert!(replay.replayed);
        assert_eq!(ledger.transactions().len(), 1);

        let overwrite = envelope(
            2,
            1,
            vec![ResolutionCommand::PersistObservation(observation(1, None)?)],
        )?;
        assert_eq!(
            ledger.commit(overwrite),
            Err(LedgerError::ImmutableObservation(ObservationId(1)))
        );
        assert_eq!(ledger.version(), CommitVersion::new(1));
        Ok(())
    }

    #[test]
    fn resolution_commit_closes_prior_decision_without_deleting_history()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut ledger = MemoryLedger::default();
        ledger.commit(envelope(
            1,
            0,
            vec![ResolutionCommand::PersistObservation(observation(1, None)?)],
        )?)?;
        let (inference, decision) = inference_and_decision(
            2,
            ObservationId(1),
            DecisionAction::CreateIdentity,
        )?;
        ledger.commit(envelope(
            2,
            1,
            vec![ResolutionCommand::CommitResolution {
                inference,
                decision,
                created_identity: Some(IdentityId(9)),
            }],
        )?)?;
        let (inference, decision) = inference_and_decision(
            3,
            ObservationId(1),
            DecisionAction::RejectNoise,
        )?;
        ledger.commit(envelope(
            3,
            2,
            vec![ResolutionCommand::CommitResolution {
                inference,
                decision,
                created_identity: None,
            }],
        )?)?;
        assert_eq!(
            ledger
                .current_decision(ObservationId(1))
                .map(|record| &record.action),
            Some(&DecisionAction::RejectNoise)
        );
        assert_eq!(
            ledger
                .decision_as_of(ObservationId(1), CommitVersion::new(2))
                .map(|record| &record.action),
            Some(&DecisionAction::CreateIdentity)
        );
        Ok(())
    }

    #[test]
    fn host_is_unchanged_until_a_matching_receipt_is_committed()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut ledger = MemoryLedger::default();
        ledger.commit(envelope(
            1,
            0,
            vec![ResolutionCommand::PersistObservation(observation(1, None)?)],
        )?)?;
        let (inference, decision) = inference_and_decision(
            2,
            ObservationId(1),
            DecisionAction::CreateIdentity,
        )?;
        ledger.commit(envelope(
            2,
            1,
            vec![
                ResolutionCommand::CommitResolution {
                    inference,
                    decision,
                    created_identity: Some(IdentityId(1)),
                },
                ResolutionCommand::EnqueueMaterialization {
                    materialization: MaterializationId(1),
                    decision: DecisionId(2),
                    relation: HostRelation::Occur {
                        event: li_core::EventId(1),
                        physical: li_core::PhysicalNodeId(1),
                    },
                },
            ],
        )?)?;
        assert_eq!(
            ledger
                .outbox(MaterializationId(1))
                .map(|entry| entry.completed),
            Some(false)
        );
        ledger.commit(envelope(
            3,
            2,
            vec![ResolutionCommand::Materialized {
                materialization: MaterializationId(1),
                decision: DecisionId(2),
            }],
        )?)?;
        assert_eq!(
            ledger
                .outbox(MaterializationId(1))
                .map(|entry| entry.completed),
            Some(true)
        );
        Ok(())
    }

    #[test]
    fn merge_canonicalization_is_deterministic_and_split_is_non_destructive()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut ledger = MemoryLedger::default();
        ledger.commit(envelope(
            1,
            0,
            vec![
                ResolutionCommand::CreateIdentity {
                    identity: IdentityId(9),
                    created_at: Timestamp::UNIX_EPOCH,
                },
                ResolutionCommand::CreateIdentity {
                    identity: IdentityId(3),
                    created_at: Timestamp::UNIX_EPOCH,
                },
            ],
        )?)?;
        ledger.commit(envelope(
            2,
            1,
            vec![ResolutionCommand::merge(
                DecisionId(10),
                IdentityId(9),
                IdentityId(3),
                Vec::new(),
                1,
                1,
            )?],
        )?)?;
        assert_eq!(ledger.canonical(IdentityId(9)), Some(IdentityId(3)));

        let split = li_core::SplitPlan::new(
            vec![DecisionId(10)],
            vec![
                li_core::SplitPartition {
                    identity: IdentityId(9),
                    observations: vec![ObservationId(90)].into_boxed_slice(),
                },
                li_core::SplitPartition {
                    identity: IdentityId(3),
                    observations: vec![ObservationId(30)].into_boxed_slice(),
                },
            ],
            Vec::new(),
        );
        assert!(split.is_ok());
        // Missing observations reject the entire split without closing merge
        // history.
        if let Ok(split) = split {
            let command =
                envelope(3, 2, vec![ResolutionCommand::Split(split)])?;
            assert_eq!(
                ledger.commit(command),
                Err(LedgerError::MissingObservation(ObservationId(90)))
            );
        }
        assert_eq!(ledger.canonical(IdentityId(9)), Some(IdentityId(3)));
        Ok(())
    }

    #[test]
    fn descending_merges_remain_balanced_and_choose_the_minimum()
    -> Result<(), Box<dyn std::error::Error>> {
        const IDENTITIES: u64 = 128;
        let mut ledger = MemoryLedger::with_capacity(
            IDENTITIES as usize,
            IDENTITIES as usize,
        );
        let creates = (1..=IDENTITIES)
            .map(|identity| ResolutionCommand::CreateIdentity {
                identity: IdentityId(identity),
                created_at: Timestamp::UNIX_EPOCH,
            })
            .collect();
        ledger.commit(envelope(1, 0, creates)?)?;
        for upper in (2..=IDENTITIES).rev() {
            let sequence = IDENTITIES.saturating_sub(upper).saturating_add(2);
            ledger.commit(envelope(
                sequence,
                sequence.saturating_sub(1),
                vec![ResolutionCommand::merge(
                    DecisionId(sequence),
                    IdentityId(upper),
                    IdentityId(upper - 1),
                    Vec::new(),
                    1,
                    1,
                )?],
            )?)?;
        }
        for identity in 1..=IDENTITIES {
            assert_eq!(
                ledger.canonical(IdentityId(identity)),
                Some(IdentityId(1))
            );
        }
        let mut maximum_depth = 0_usize;
        for identity in 1..=IDENTITIES {
            let mut current = IdentityId(identity);
            let mut depth = 0_usize;
            while let Some(node) = ledger.canonical.get(&current) {
                if node.parent == current {
                    break;
                }
                current = node.parent;
                depth = depth.saturating_add(1);
            }
            maximum_depth = maximum_depth.max(depth);
        }
        assert!(maximum_depth <= 7);
        Ok(())
    }

    #[test]
    fn stale_snapshot_rejects_the_whole_batch()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut ledger = MemoryLedger::default();
        ledger.commit(envelope(
            1,
            0,
            vec![ResolutionCommand::PersistObservation(observation(1, None)?)],
        )?)?;
        let stale = envelope(
            2,
            0,
            vec![ResolutionCommand::PersistObservation(observation(2, None)?)],
        )?;
        assert_eq!(
            ledger.commit(stale),
            Err(LedgerError::VersionConflict {
                expected: CommitVersion::ZERO,
                current: CommitVersion::new(1),
            })
        );
        assert!(ledger.observation(ObservationId(2)).is_none());
        Ok(())
    }

    #[test]
    fn durable_ledger_replays_ordered_envelopes_and_rejects_duplicate_effects()
    -> Result<(), Box<dyn std::error::Error>> {
        let backend = MemoryKvBackend::default();
        let mut ledger = DurableLedger::open(backend)?;
        let command = envelope(
            1,
            0,
            vec![ResolutionCommand::PersistObservation(observation(1, None)?)],
        )?;
        let committed = ledger.commit_envelope(command.clone())?;
        assert_eq!(committed.version, CommitVersion::new(1));

        let backend = ledger.into_backend();
        let mut reopened = DurableLedger::open(backend)?;
        assert!(reopened.memory().observation(ObservationId(1)).is_some());
        assert_eq!(reopened.current_version(), CommitVersion::new(1));
        let replay = reopened.commit_envelope(command)?;
        assert!(replay.replayed);
        assert_eq!(reopened.memory().transactions().len(), 1);
        Ok(())
    }

    #[test]
    fn backend_failure_does_not_publish_staged_ledger_state()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut ledger = DurableLedger::open(RejectingBackend)?;
        let command = envelope(
            1,
            0,
            vec![ResolutionCommand::PersistObservation(observation(1, None)?)],
        )?;
        assert_eq!(
            ledger.commit_envelope(command),
            Err(LedgerError::Storage(StorageError::TransactionFailed))
        );
        assert_eq!(ledger.current_version(), CommitVersion::ZERO);
        assert!(ledger.memory().observation(ObservationId(1)).is_none());
        Ok(())
    }

    #[cfg(feature = "rocksdb")]
    #[test]
    fn rocksdb_ledger_survives_close_and_reopen()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        {
            let backend = crate::RocksDbBackend::open(directory.path())?;
            let mut ledger = DurableLedger::open(backend)?;
            ledger.commit_envelope(envelope(
                1,
                0,
                vec![ResolutionCommand::PersistObservation(observation(
                    1, None,
                )?)],
            )?)?;
        }
        let backend = crate::RocksDbBackend::open(directory.path())?;
        let ledger = DurableLedger::open(backend)?;
        assert_eq!(ledger.current_version(), CommitVersion::new(1));
        assert!(ledger.memory().observation(ObservationId(1)).is_some());
        Ok(())
    }
}
