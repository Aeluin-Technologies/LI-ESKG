//! Transactional pipeline with coherent snapshots and typestate phases.

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arc_swap::ArcSwap;
use crossbeam_channel::{
    Receiver, Sender, TryRecvError, TrySendError, bounded,
};
use li_core::{
    BayesRiskPolicy, CommandEnvelope, CommandError, CommitVersion,
    ContentHash, DecisionAction, DecisionError, DecisionId, DecisionRecord,
    HostRelation, IdempotencyKey, IdentityId, IdentityReference, InferenceId,
    InferenceProvenance, InferenceRecord, MaterializationId,
    ObservationEnvelope, ProviderArtifact, ResolutionCommand, TransactionId,
    VersionInterval,
};
use li_factors::{FactorProvider, ProviderContext, ProviderError};
use li_inference::{
    SolverError, SolverScratch, SumProductConfig, SumProductSolver,
};
use li_storage::{CommitResult, LedgerError, ResolutionLedger};
use li_workspace::{
    CommittedAssociation, HotIdentity, HotWorkspace, WorkerScratch,
    WorkspaceError,
};
use thiserror::Error;
use tracing::{info_span, instrument};

/// Coherent read-mostly host and provider-index versions for one batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeSnapshot {
    /// Authoritative host snapshot version.
    pub host: CommitVersion,
    /// Candidate index snapshot/configuration version.
    pub candidates: u64,
    /// Hash of candidate, solver, and provider configuration.
    pub configuration_hash: ContentHash,
}

/// Domain-specific planner mapping accepted decisions to native host
/// relations.
pub trait MaterializationPlanner: Send + Sync {
    /// Returns a native host write plan, or `None` when no host fact is
    /// implied.
    fn plan(
        &self,
        observation: &ObservationEnvelope,
        decision: &DecisionAction,
    ) -> Option<HostRelation>;
}

/// Stable materializer failure retained while the outbox remains pending.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("host materializer failed with code {code}")]
pub struct MaterializationFailure {
    /// Deployment-specific stable diagnostic code.
    pub code: u32,
}

/// Idempotent authoritative host adapter.
pub trait HostMaterializer: Send {
    /// Writes one native relation using `materialization` as the idempotency
    /// key.
    fn materialize(
        &mut self,
        materialization: MaterializationId,
        relation: HostRelation,
        decision: DecisionId,
        commit: CommitVersion,
    ) -> Result<(), MaterializationFailure>;
}

/// Error returned before a resolution batch is durably complete.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// Empty collective batches have no inference semantics.
    #[error("resolution batch must not be empty")]
    EmptyBatch,
    /// Stable ID sequence reached `u64::MAX`.
    #[error("stable identifier sequence exhausted")]
    IdExhausted,
    /// Provider did not emit exactly one candidate group per observation.
    #[error(
        "provider {provider:?} emitted {actual} groups for batch length {expected}"
    )]
    IncompleteProvider {
        /// Provider that violated the batch contract.
        provider: li_core::ProviderId,
        /// Required observation group count.
        expected: usize,
        /// Emitted observation group count.
        actual: usize,
    },
    /// Structural command construction failed.
    #[error(transparent)]
    Command(#[from] CommandError),
    /// Durable ledger contract rejected the batch.
    #[error(transparent)]
    Ledger(#[from] LedgerError),
    /// Candidate/factor provider rejected input or emitted invalid data.
    #[error(transparent)]
    Provider(#[from] ProviderError),
    /// Factor graph compilation or inference failed.
    #[error(transparent)]
    Solver(#[from] SolverError),
    /// Decision action was inconsistent with its posterior domain.
    #[error(transparent)]
    Decision(#[from] DecisionError),
    /// Hot cache could not apply a durable association.
    #[error(transparent)]
    Workspace(#[from] WorkspaceError),
}

/// Successful batch result, including host writes left safely pending.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessResult {
    /// Commit that made immutable evidence visible.
    pub evidence_commit: CommitResult,
    /// Atomic inference, decision, lifecycle, and outbox commit.
    pub resolution_commit: CommitResult,
    /// Latest ledger version after successful materialization receipts.
    pub final_version: CommitVersion,
    /// Durable decisions in input observation order.
    pub decisions: Box<[Arc<DecisionRecord>]>,
    /// Host writes that failed and remain retryable in the outbox.
    pub pending_materializations: Box<[MaterializationId]>,
}

/// Marker for a received but unpersisted batch.
pub struct Received;
/// Marker for an evidence-persisted batch.
pub struct Persisted;
/// Marker for a batch with a normalized posterior.
pub struct Inferred;
/// Marker for a batch with explicit policy decisions.
pub struct Decided;
/// Marker for an atomically committed resolution batch.
pub struct Committed;

/// Small typestate token that makes conceptual phase order
/// non-interchangeable.
pub struct BatchPhase<S> {
    batch_len: usize,
    version: CommitVersion,
    marker: PhantomData<S>,
}

impl BatchPhase<Received> {
    fn new(batch_len: usize, version: CommitVersion) -> Self {
        Self {
            batch_len,
            version,
            marker: PhantomData,
        }
    }

    fn persisted(self, version: CommitVersion) -> BatchPhase<Persisted> {
        BatchPhase {
            batch_len: self.batch_len,
            version,
            marker: PhantomData,
        }
    }
}

impl BatchPhase<Persisted> {
    fn inferred(self) -> BatchPhase<Inferred> {
        BatchPhase {
            batch_len: self.batch_len,
            version: self.version,
            marker: PhantomData,
        }
    }
}

impl BatchPhase<Inferred> {
    fn decided(self) -> BatchPhase<Decided> {
        BatchPhase {
            batch_len: self.batch_len,
            version: self.version,
            marker: PhantomData,
        }
    }
}

impl BatchPhase<Decided> {
    fn committed(self, version: CommitVersion) -> BatchPhase<Committed> {
        BatchPhase {
            batch_len: self.batch_len,
            version,
            marker: PhantomData,
        }
    }
}

/// Bounded multi-producer ingestion queue with explicit overload feedback.
#[derive(Debug)]
pub struct BoundedIngress<T> {
    sender: Sender<T>,
    receiver: Receiver<T>,
}

impl<T> BoundedIngress<T> {
    /// Creates a positive-capacity bounded queue.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::EmptyBatch`] when `capacity` is zero.
    pub fn new(capacity: usize) -> Result<Self, PipelineError> {
        if capacity == 0 {
            return Err(PipelineError::EmptyBatch);
        }
        let (sender, receiver) = bounded(capacity);
        Ok(Self { sender, receiver })
    }

    /// Attempts to enqueue without blocking or allocating a new task.
    pub fn try_send(&self, item: T) -> Result<(), TrySendError<T>> {
        self.sender.try_send(item)
    }

    /// Attempts to receive one item without blocking.
    pub fn try_recv(&self) -> Result<T, TryRecvError> {
        self.receiver.try_recv()
    }

    /// Returns the current bounded queue depth.
    pub fn len(&self) -> usize {
        self.receiver.len()
    }

    /// Returns whether the queue currently contains no items.
    pub fn is_empty(&self) -> bool {
        self.receiver.is_empty()
    }
}

#[derive(Debug)]
struct Sequence(AtomicU64);

impl Sequence {
    fn new() -> Self {
        Self(AtomicU64::new(1))
    }

    fn next(&self) -> Result<u64, PipelineError> {
        self.0
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| PipelineError::IdExhausted)
    }
}

#[derive(Debug)]
struct IdPool {
    transactions: Sequence,
    inferences: Sequence,
    decisions: Sequence,
    identities: Sequence,
    materializations: Sequence,
}

impl IdPool {
    fn new() -> Self {
        Self {
            transactions: Sequence::new(),
            inferences: Sequence::new(),
            decisions: Sequence::new(),
            identities: Sequence::new(),
            materializations: Sequence::new(),
        }
    }

    fn envelope(
        &self,
        expected: CommitVersion,
        issued_at: li_core::Timestamp,
        depends_on: Vec<TransactionId>,
        commands: Vec<ResolutionCommand>,
    ) -> Result<CommandEnvelope, PipelineError> {
        let raw = self.transactions.next()?;
        let transaction = TransactionId(raw);
        let mut bytes = [0_u8; 16];
        bytes[..8].copy_from_slice(&raw.to_le_bytes());
        bytes[8..14].copy_from_slice(b"LIESKG");
        let idempotency = IdempotencyKey::new(bytes)?;
        Ok(CommandEnvelope::new(
            transaction,
            expected,
            idempotency,
            issued_at,
            depends_on,
            commands,
        )?)
    }
}

///  runtime connecting opaque providers, inference, policy, ledger, and
/// host.
pub struct ResolutionRuntime<L, P, M> {
    ledger: L,
    planner: P,
    materializer: M,
    providers: ArcSwap<Vec<Arc<dyn FactorProvider>>>,
    snapshot: ArcSwap<RuntimeSnapshot>,
    solver: SumProductSolver,
    policy: BayesRiskPolicy,
    workspace: HotWorkspace,
    scratch: WorkerScratch,
    solver_scratch: SolverScratch,
    ids: IdPool,
    history_capacity: usize,
    pending: Vec<MaterializationId>,
    associations: Vec<CommittedAssociation>,
}

impl<L, P, M> ResolutionRuntime<L, P, M>
where
    L: ResolutionLedger,
    P: MaterializationPlanner,
    M: HostMaterializer,
{
    /// Creates a  runtime with one configured solver and reusable worker
    /// pool.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::Solver`] for an invalid solver configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ledger: L,
        planner: P,
        materializer: M,
        providers: Vec<Arc<dyn FactorProvider>>,
        snapshot: RuntimeSnapshot,
        solver: SumProductConfig,
        policy: BayesRiskPolicy,
        workspace: HotWorkspace,
        scratch: WorkerScratch,
        history_capacity: usize,
    ) -> Result<Self, PipelineError> {
        Ok(Self {
            ledger,
            planner,
            materializer,
            providers: ArcSwap::from_pointee(providers),
            snapshot: ArcSwap::from_pointee(snapshot),
            solver: SumProductSolver::new(solver)?,
            policy,
            workspace,
            scratch,
            solver_scratch: SolverScratch::default(),
            ids: IdPool::new(),
            history_capacity,
            pending: Vec::new(),
            associations: Vec::new(),
        })
    }

    /// Atomically publishes provider and index snapshots for later batches.
    pub fn publish(
        &self,
        providers: Vec<Arc<dyn FactorProvider>>,
        snapshot: RuntimeSnapshot,
    ) {
        self.providers.store(Arc::new(providers));
        self.snapshot.store(Arc::new(snapshot));
    }

    /// Executes persist, snapshot, gate, compile, solve, decide, commit, and
    /// materialize.
    ///
    /// Failed host writes are returned as pending outbox identifiers; the
    /// accepted ledger decision remains durable and retryable.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] before resolution commit when a provider,
    /// solver, policy, or ledger invariant is violated.
    #[instrument(skip_all, fields(batch_len = observations.len()))]
    pub fn process_batch(
        &mut self,
        observations: &[ObservationEnvelope],
    ) -> Result<ProcessResult, PipelineError> {
        if observations.is_empty() {
            return Err(PipelineError::EmptyBatch);
        }
        let issued_at = observations
            .iter()
            .map(ObservationEnvelope::ingestion_time)
            .max()
            .unwrap_or(li_core::Timestamp::UNIX_EPOCH);
        let phase = BatchPhase::<Received>::new(
            observations.len(),
            self.ledger.current_version(),
        );
        let persist_commands = observations
            .iter()
            .cloned()
            .map(ResolutionCommand::PersistObservation)
            .collect();
        let persistence = self.ids.envelope(
            phase.version,
            issued_at,
            Vec::new(),
            persist_commands,
        )?;
        let persistence_tx = persistence.transaction();
        let evidence_commit = self.ledger.commit_envelope(persistence)?;
        let phase = phase.persisted(evidence_commit.version);

        self.scratch.reset(observations.len());
        self.scratch.observations.extend_from_slice(observations);
        let snapshot = self.snapshot.load_full();
        let providers = self.providers.load_full();
        let context = ProviderContext {
            host_snapshot: snapshot.host,
            candidate_snapshot: snapshot.candidates,
        };
        let _snapshot_span = info_span!(
            "li_eskg.inference",
            host_snapshot = snapshot.host.get(),
            candidate_snapshot = snapshot.candidates
        )
        .entered();

        for provider in providers.iter() {
            self.scratch.provider_candidates.reset(observations.len());
            provider.generate_candidates(
                &self.scratch.observations,
                context,
                &mut self.scratch.provider_candidates,
            )?;
            let actual = self.scratch.provider_candidates.observation_count();
            if actual != observations.len() {
                return Err(PipelineError::IncompleteProvider {
                    provider: provider.provider_id(),
                    expected: observations.len(),
                    actual,
                });
            }
            for index in 0..observations.len() {
                if let Some(candidates) =
                    self.scratch.provider_candidates.get(index)
                {
                    self.scratch.candidate_groups[index]
                        .extend_from_slice(candidates);
                }
            }
        }
        for index in 0..observations.len() {
            self.scratch.candidates.push_observation(
                index,
                observations.len(),
                self.scratch.candidate_groups[index].drain(..),
            )?;
        }
        for provider in providers.iter() {
            provider.emit_factors(
                &self.scratch.observations,
                &self.scratch.candidates,
                context,
                &mut self.scratch.factors,
            )?;
        }
        let posterior = self.solver.solve(
            &self.scratch.candidates,
            self.scratch.factors.as_slice(),
            &mut self.solver_scratch,
        )?;
        let phase = phase.inferred();

        let resolution_version = phase
            .version
            .checked_next()
            .ok_or(PipelineError::IdExhausted)?;
        let artifacts: Box<[ProviderArtifact]> = providers
            .iter()
            .map(|provider| provider.artifact())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let provenance = Arc::new(InferenceProvenance {
            providers: artifacts,
            candidate_version: snapshot.candidates,
            host_snapshot: snapshot.host,
            configuration_hash: snapshot.configuration_hash,
        });
        let diagnostics = Arc::new(posterior.diagnostics);
        let mut contribution_groups: Vec<Vec<li_core::ScoreContribution>> =
            (0..observations.len()).map(|_| Vec::new()).collect();
        for factor in self.scratch.factors.as_slice() {
            for variable in factor.variables() {
                let Ok(index) = usize::try_from(*variable) else {
                    continue;
                };
                if let Some(group) = contribution_groups.get_mut(index) {
                    group.extend_from_slice(factor.contributions());
                }
            }
        }
        let mut commands =
            Vec::with_capacity(observations.len().saturating_mul(2));
        let mut decisions = Vec::with_capacity(observations.len());
        let mut materializations = Vec::new();
        self.associations.clear();

        for ((observation, distribution), contributions) in observations
            .iter()
            .zip(posterior.distributions.into_vec())
            .zip(contribution_groups)
        {
            let action = self.policy.decide(&distribution);
            let inference_id = InferenceId(self.ids.inferences.next()?);
            let decision_id = DecisionId(self.ids.decisions.next()?);
            let decision = Arc::new(DecisionRecord::new(
                decision_id,
                inference_id,
                action.clone(),
                self.policy.policy_version,
                self.policy.loss_version,
                VersionInterval::current(resolution_version),
                &distribution,
            )?);
            let inference = Arc::new(InferenceRecord {
                id: inference_id,
                observation: observation.id(),
                distribution,
                contributions: contributions.into_boxed_slice(),
                provenance: Arc::clone(&provenance),
                diagnostics: Arc::clone(&diagnostics),
                validity: VersionInterval::current(resolution_version),
            });
            let created_identity = if action == DecisionAction::CreateIdentity
            {
                Some(IdentityId(self.ids.identities.next()?))
            } else {
                None
            };
            commands.push(ResolutionCommand::CommitResolution {
                inference,
                decision: Arc::clone(&decision),
                created_identity,
            });

            let active_identity = match (&action, created_identity) {
                (
                    DecisionAction::Assign(IdentityReference::Latent(
                        identity,
                    )),
                    _,
                ) => Some(*identity),
                (DecisionAction::CreateIdentity, Some(identity)) => {
                    Some(identity)
                },
                _ => None,
            };
            if let Some(identity) = active_identity {
                self.associations.push(CommittedAssociation {
                    observation: observation.id(),
                    identity,
                    version: resolution_version,
                    event_time: observation.event_time(),
                });
            }

            if let Some(relation) = self.planner.plan(observation, &action) {
                let materialization =
                    MaterializationId(self.ids.materializations.next()?);
                commands.push(ResolutionCommand::EnqueueMaterialization {
                    materialization,
                    decision: decision_id,
                    relation,
                });
                materializations.push((
                    materialization,
                    decision_id,
                    relation,
                ));
            }
            decisions.push(decision);
        }
        let phase = phase.decided();
        let resolution = self.ids.envelope(
            phase.version,
            issued_at,
            vec![persistence_tx],
            commands,
        )?;
        let resolution_commit = self.ledger.commit_envelope(resolution)?;
        let phase = phase.committed(resolution_commit.version);

        for association in self.associations.iter().copied() {
            if self.workspace.get(association.identity).is_none() {
                self.workspace.insert(HotIdentity::new(
                    association.identity,
                    4,
                    self.history_capacity,
                    association.event_time,
                    CommitVersion::ZERO,
                ));
            }
            self.workspace.apply_committed(association)?;
        }

        self.pending.clear();
        for (materialization, decision, relation) in materializations {
            if self
                .materializer
                .materialize(
                    materialization,
                    relation,
                    decision,
                    phase.version,
                )
                .is_err()
            {
                self.pending.push(materialization);
                continue;
            }
            self.commit_receipt(materialization, decision, issued_at)?;
        }
        Ok(ProcessResult {
            evidence_commit,
            resolution_commit,
            final_version: self.ledger.current_version(),
            decisions: decisions.into_boxed_slice(),
            pending_materializations: self.pending.clone().into_boxed_slice(),
        })
    }

    /// Retries one pending outbox entry and appends a receipt on success.
    ///
    /// # Errors
    ///
    /// Returns the materializer error without changing the ledger receipt set,
    /// or [`PipelineError`] if the receipt commit is rejected.
    pub fn retry_materialization(
        &mut self,
        materialization: MaterializationId,
        issued_at: li_core::Timestamp,
    ) -> Result<(), RetryError> {
        let entry = self
            .ledger
            .materialization(materialization)
            .cloned()
            .ok_or(RetryError::MissingOutbox(materialization))?;
        if entry.completed {
            return Ok(());
        }
        self.materializer.materialize(
            entry.id,
            entry.relation,
            entry.decision,
            entry.commit,
        )?;
        self.commit_receipt(entry.id, entry.decision, issued_at)?;
        Ok(())
    }

    fn commit_receipt(
        &mut self,
        materialization: MaterializationId,
        decision: DecisionId,
        issued_at: li_core::Timestamp,
    ) -> Result<(), PipelineError> {
        let envelope = self.ids.envelope(
            self.ledger.current_version(),
            issued_at,
            Vec::new(),
            vec![ResolutionCommand::Materialized {
                materialization,
                decision,
            }],
        )?;
        self.ledger.commit_envelope(envelope)?;
        Ok(())
    }

    /// Borrows the durable ledger.
    pub const fn ledger(&self) -> &L {
        &self.ledger
    }

    /// Borrows the active hot cache.
    pub const fn workspace(&self) -> &HotWorkspace {
        &self.workspace
    }
}

/// Error returned while retrying an already committed outbox entry.
#[derive(Debug, Error)]
pub enum RetryError {
    /// Requested outbox entry does not exist.
    #[error("materialization {0:?} is not present in the outbox")]
    MissingOutbox(MaterializationId),
    /// Authoritative host adapter still failed.
    #[error(transparent)]
    Materialization(#[from] MaterializationFailure),
    /// Receipt command or ledger validation failed.
    #[error(transparent)]
    Pipeline(#[from] PipelineError),
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use bytes::Bytes;
    use li_core::{
        EventId, EvidenceError, LossModel, Modality, PayloadRef,
        PhysicalNodeId, ProviderId, QualityMetadata, SchemaId, SourceId,
        Timestamp,
    };
    use li_factors::{CandidateBuffer, FactorBuffer, FactorTable};
    use li_storage::MemoryLedger;
    use smallvec::SmallVec;

    use super::*;

    struct UnaryProvider;

    impl FactorProvider for UnaryProvider {
        fn artifact(&self) -> ProviderArtifact {
            ProviderArtifact {
                provider: ProviderId(1),
                schema: SchemaId(1),
                model_version: 2,
                calibration_id: 3,
            }
        }

        fn generate_candidates(
            &self,
            batch: &[ObservationEnvelope],
            _context: ProviderContext,
            output: &mut CandidateBuffer,
        ) -> Result<(), ProviderError> {
            for index in 0..batch.len() {
                output.push_observation(index, batch.len(), [])?;
            }
            Ok(())
        }

        fn emit_factors(
            &self,
            batch: &[ObservationEnvelope],
            _candidates: &CandidateBuffer,
            _context: ProviderContext,
            output: &mut FactorBuffer,
        ) -> Result<(), ProviderError> {
            for index in 0..batch.len() {
                output.push(FactorTable::new(
                    SmallVec::from_slice(&[
                        u32::try_from(index).unwrap_or(u32::MAX)
                    ]),
                    SmallVec::from_slice(&[2]),
                    vec![0.0, -4.0],
                    Vec::new(),
                )?);
            }
            Ok(())
        }
    }

    struct OccurPlanner;

    impl MaterializationPlanner for OccurPlanner {
        fn plan(
            &self,
            _observation: &ObservationEnvelope,
            decision: &DecisionAction,
        ) -> Option<HostRelation> {
            if *decision == DecisionAction::CreateIdentity {
                Some(HostRelation::Occur {
                    event: EventId(1),
                    physical: PhysicalNodeId(1),
                })
            } else {
                None
            }
        }
    }

    struct Materializer {
        failures_left: AtomicUsize,
    }

    impl HostMaterializer for Materializer {
        fn materialize(
            &mut self,
            _materialization: MaterializationId,
            _relation: HostRelation,
            _decision: DecisionId,
            _commit: CommitVersion,
        ) -> Result<(), MaterializationFailure> {
            let failed = self
                .failures_left
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    value.checked_sub(1)
                })
                .is_ok();
            if failed {
                Err(MaterializationFailure { code: 1 })
            } else {
                Ok(())
            }
        }
    }

    fn observation(id: u64) -> Result<ObservationEnvelope, EvidenceError> {
        ObservationEnvelope::new(
            li_core::ObservationId(id),
            SourceId(1),
            Modality(1),
            Timestamp::from_micros(i64::try_from(id).unwrap_or(i64::MAX)),
            Timestamp::from_micros(i64::try_from(id).unwrap_or(i64::MAX)),
            PayloadRef::Inline(Bytes::from_static(b"x")),
            QualityMetadata::Opaque {
                schema: SchemaId(1),
                bytes: Bytes::new(),
            },
            ContentHash::new([1; 32]),
            None,
        )
    }

    fn runtime(
        failures: usize,
    ) -> Result<
        ResolutionRuntime<MemoryLedger, OccurPlanner, Materializer>,
        PipelineError,
    > {
        let loss = LossModel::new(4.0, 4.0, 4.0, 10.0)?;
        ResolutionRuntime::new(
            MemoryLedger::default(),
            OccurPlanner,
            Materializer {
                failures_left: AtomicUsize::new(failures),
            },
            vec![Arc::new(UnaryProvider)],
            RuntimeSnapshot {
                host: CommitVersion::ZERO,
                candidates: 1,
                configuration_hash: ContentHash::new([3; 32]),
            },
            SumProductConfig {
                solver_version: 1,
                max_iterations: 10,
                tolerance: 1.0e-9,
                damping: 0.0,
                candidate_log_prior: 0.0,
                new_log_prior: 0.0,
                noise_log_prior: -2.0,
                boundary_treatment: li_core::BoundaryTreatment::Global,
            },
            BayesRiskPolicy {
                policy_version: 1,
                loss_version: 1,
                loss,
            },
            HotWorkspace::with_capacity(4),
            WorkerScratch::with_capacity(4, 8, 4, 16),
            8,
        )
    }

    #[test]
    fn bounded_ingress_reports_backpressure() -> Result<(), PipelineError> {
        let queue = BoundedIngress::new(1)?;
        assert!(queue.try_send(1).is_ok());
        assert!(matches!(queue.try_send(2), Err(TrySendError::Full(2))));
        assert_eq!(queue.try_recv(), Ok(1));
        Ok(())
    }

    #[test]
    fn pipeline_persists_distinct_inference_and_decision_records()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut runtime = runtime(0)?;
        let result = runtime.process_batch(&[observation(1)?])?;
        assert_eq!(result.evidence_commit.version, CommitVersion::new(1));
        assert_eq!(result.resolution_commit.version, CommitVersion::new(2));
        assert_eq!(result.final_version, CommitVersion::new(3));
        assert!(result.pending_materializations.is_empty());
        assert_eq!(runtime.workspace().len(), 1);
        Ok(())
    }

    #[test]
    fn failed_host_write_stays_pending_and_can_be_retried()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut runtime = runtime(1)?;
        let result = runtime.process_batch(&[observation(1)?])?;
        assert_eq!(result.final_version, CommitVersion::new(2));
        assert_eq!(result.pending_materializations.len(), 1);
        runtime.retry_materialization(
            result.pending_materializations[0],
            Timestamp::from_micros(2),
        )?;
        assert_eq!(runtime.ledger().version(), CommitVersion::new(3));
        Ok(())
    }

    #[test]
    fn empty_batch_has_no_ledger_effect() -> Result<(), PipelineError> {
        let mut runtime = runtime(0)?;
        assert!(matches!(
            runtime.process_batch(&[]),
            Err(PipelineError::EmptyBatch)
        ));
        assert_eq!(runtime.ledger().version(), CommitVersion::ZERO);
        Ok(())
    }
}
