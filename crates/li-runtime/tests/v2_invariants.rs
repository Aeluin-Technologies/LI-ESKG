//! Cross-crate compliance tests for paper invariants I1-I9.

use std::sync::Arc;

use bytes::Bytes;
use li_core::{
    AssociationOutcome, BoundaryTreatment, CommandEnvelope, CommitVersion,
    ContentHash, DecisionAction, DecisionId, DecisionRecord, HostNodeId,
    HostRelation, IdempotencyKey, IdentityId, InferenceId,
    InferenceProvenance, InferenceRecord, MaterializationId, Modality,
    NormalizedDistribution, ObservationEnvelope, ObservationId,
    OutcomeProbability, PayloadRef, Probability, ProviderArtifact, ProviderId,
    QualityMetadata, ResolutionCommand, SchemaId, SolverDiagnostics,
    SolverStoppingReason, SourceId, SplitPartition, SplitPlan, Timestamp,
    TransactionId, VersionInterval,
};
use li_model::{
    AuthoritativeHostGraph, HostGraphError, HostSchemaProfile,
    MaterializationOutcome,
};
use li_storage::{LedgerError, MemoryLedger, ResolutionLedger};

fn observation(
    id: u64,
    supersedes: Option<ObservationId>,
) -> Result<ObservationEnvelope, li_core::EvidenceError> {
    ObservationEnvelope::new(
        ObservationId(id),
        SourceId(1),
        Modality(1),
        Timestamp::from_micros(i64::try_from(id).unwrap_or(i64::MAX)),
        Timestamp::from_micros(i64::try_from(id).unwrap_or(i64::MAX)),
        PayloadRef::Inline(Bytes::from_static(b"immutable")),
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
    CommandEnvelope::new(
        TransactionId(transaction),
        CommitVersion::new(expected),
        IdempotencyKey::new(
            [u8::try_from(transaction).unwrap_or(u8::MAX); 16],
        )?,
        Timestamp::from_micros(i64::try_from(transaction).unwrap_or(i64::MAX)),
        Vec::new(),
        commands,
    )
}

fn records(
    version: u64,
    observation: ObservationId,
    action: DecisionAction,
    provider: ProviderId,
) -> Result<
    (Arc<InferenceRecord>, Arc<DecisionRecord>),
    Box<dyn std::error::Error>,
> {
    let mut entries = vec![
        OutcomeProbability {
            outcome: AssociationOutcome::New,
            probability: Probability::new(0.1),
        },
        OutcomeProbability {
            outcome: AssociationOutcome::Noise,
            probability: Probability::new(0.1),
        },
    ];
    match &action {
        DecisionAction::Assign(target) => entries.push(OutcomeProbability {
            outcome: AssociationOutcome::Identity(target.clone()),
            probability: Probability::new(0.8),
        }),
        DecisionAction::CreateIdentity => {
            entries[0].probability = Probability::new(0.8);
            entries[1].probability = Probability::new(0.2);
        },
        DecisionAction::RejectNoise => {
            entries[0].probability = Probability::new(0.2);
            entries[1].probability = Probability::new(0.8);
        },
        DecisionAction::Abstain => {
            entries[0].probability = Probability::new(0.5);
            entries[1].probability = Probability::new(0.5);
        },
    }
    let distribution = NormalizedDistribution::new(entries, None)?;
    let inference = InferenceRecord {
        id: InferenceId(version),
        observation,
        distribution: distribution.clone(),
        contributions: Vec::new().into_boxed_slice(),
        provenance: Arc::new(InferenceProvenance {
            providers: vec![ProviderArtifact {
                provider,
                schema: SchemaId(1),
                model_version: 1,
                calibration_id: 1,
            }]
            .into_boxed_slice(),
            candidate_version: 1,
            host_snapshot: CommitVersion::new(version.saturating_sub(1)),
            configuration_hash: ContentHash::new([9; 32]),
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

fn profile() -> Result<HostSchemaProfile, li_model::HostSchemaError> {
    HostSchemaProfile::new([
        Arc::from("native:triggers"),
        Arc::from("native:leadsTo"),
        Arc::from("native:evolution"),
        Arc::from("native:contain"),
        Arc::from("native:occur"),
        Arc::from("native:influence"),
    ])
}

#[test]
fn i1_type_soundness_is_closed_and_failed_endpoints_do_not_mutate_host()
-> Result<(), Box<dyn std::error::Error>> {
    let relation = HostRelation::Triggers {
        state: li_core::StateId(1),
        event: li_core::EventId(1),
    };
    assert_eq!(
        relation.endpoints(),
        (
            HostNodeId::State(li_core::StateId(1)),
            HostNodeId::Event(li_core::EventId(1))
        )
    );
    let mut host = AuthoritativeHostGraph::with_capacity(profile()?, 1, 1);
    host.add_node(HostNodeId::State(li_core::StateId(1)))?;
    assert_eq!(
        host.materialize(
            relation,
            DecisionId(1),
            CommitVersion::new(1),
            MaterializationId(1),
        ),
        Err(HostGraphError::MissingNode(HostNodeId::Event(
            li_core::EventId(1)
        )))
    );
    assert_eq!(host.edge_count(), 0);
    Ok(())
}

#[test]
fn i2_committed_evidence_is_immutable()
-> Result<(), Box<dyn std::error::Error>> {
    let mut ledger = MemoryLedger::default();
    ledger.commit_envelope(envelope(
        1,
        0,
        vec![ResolutionCommand::PersistObservation(observation(1, None)?)],
    )?)?;
    let overwrite = envelope(
        2,
        1,
        vec![ResolutionCommand::PersistObservation(observation(1, None)?)],
    )?;
    assert_eq!(
        ledger.commit_envelope(overwrite),
        Err(LedgerError::ImmutableObservation(ObservationId(1)))
    );
    assert_eq!(ledger.current_version(), CommitVersion::new(1));
    Ok(())
}

#[test]
fn i3_only_one_decision_is_current_while_history_remains_queryable()
-> Result<(), Box<dyn std::error::Error>> {
    let mut ledger = MemoryLedger::default();
    ledger.commit_envelope(envelope(
        1,
        0,
        vec![ResolutionCommand::PersistObservation(observation(1, None)?)],
    )?)?;
    let (inference, decision) = records(
        2,
        ObservationId(1),
        DecisionAction::CreateIdentity,
        ProviderId(1),
    )?;
    ledger.commit_envelope(envelope(
        2,
        1,
        vec![ResolutionCommand::CommitResolution {
            inference,
            decision,
            created_identity: Some(IdentityId(1)),
        }],
    )?)?;
    let (inference, decision) = records(
        3,
        ObservationId(1),
        DecisionAction::RejectNoise,
        ProviderId(1),
    )?;
    ledger.commit_envelope(envelope(
        3,
        2,
        vec![ResolutionCommand::CommitResolution {
            inference,
            decision,
            created_identity: None,
        }],
    )?)?;
    assert!(matches!(
        ledger
            .current_decision(ObservationId(1))
            .map(|record| &record.action),
        Some(DecisionAction::RejectNoise)
    ));
    assert!(matches!(
        ledger
            .decision_as_of(ObservationId(1), CommitVersion::new(2))
            .map(|record| &record.action),
        Some(DecisionAction::CreateIdentity)
    ));
    Ok(())
}

#[test]
fn i4_incomplete_provenance_is_rejected_atomically()
-> Result<(), Box<dyn std::error::Error>> {
    let mut ledger = MemoryLedger::default();
    ledger.commit_envelope(envelope(
        1,
        0,
        vec![ResolutionCommand::PersistObservation(observation(1, None)?)],
    )?)?;
    let (mut inference, decision) = records(
        2,
        ObservationId(1),
        DecisionAction::CreateIdentity,
        ProviderId(1),
    )?;
    Arc::make_mut(&mut Arc::make_mut(&mut inference).provenance).providers =
        Vec::new().into_boxed_slice();
    assert_eq!(
        ledger.commit_envelope(envelope(
            2,
            1,
            vec![ResolutionCommand::CommitResolution {
                inference,
                decision,
                created_identity: Some(IdentityId(1)),
            }],
        )?),
        Err(LedgerError::IncompleteProvenance)
    );
    assert_eq!(ledger.current_version(), CommitVersion::new(1));
    Ok(())
}

#[test]
fn i5_merge_classes_have_one_deterministic_canonical_representative()
-> Result<(), Box<dyn std::error::Error>> {
    let mut ledger = MemoryLedger::default();
    ledger.commit_envelope(envelope(
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
    ledger.commit_envelope(envelope(
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
    assert_eq!(ledger.canonical(IdentityId(3)), Some(IdentityId(3)));
    Ok(())
}

#[test]
fn i6_split_is_non_destructive_and_preserves_prior_transactions()
-> Result<(), Box<dyn std::error::Error>> {
    let mut ledger = MemoryLedger::default();
    ledger.commit_envelope(envelope(
        1,
        0,
        vec![
            ResolutionCommand::PersistObservation(observation(1, None)?),
            ResolutionCommand::PersistObservation(observation(2, None)?),
            ResolutionCommand::CreateIdentity {
                identity: IdentityId(1),
                created_at: Timestamp::UNIX_EPOCH,
            },
            ResolutionCommand::CreateIdentity {
                identity: IdentityId(2),
                created_at: Timestamp::UNIX_EPOCH,
            },
        ],
    )?)?;
    ledger.commit_envelope(envelope(
        2,
        1,
        vec![ResolutionCommand::merge(
            DecisionId(10),
            IdentityId(1),
            IdentityId(2),
            Vec::new(),
            1,
            1,
        )?],
    )?)?;
    let split = SplitPlan::new(
        vec![DecisionId(10)],
        vec![
            SplitPartition {
                identity: IdentityId(1),
                observations: vec![ObservationId(1)].into_boxed_slice(),
            },
            SplitPartition {
                identity: IdentityId(2),
                observations: vec![ObservationId(2)].into_boxed_slice(),
            },
        ],
        Vec::new(),
    )?;
    ledger.commit_envelope(envelope(
        3,
        2,
        vec![ResolutionCommand::Split(split)],
    )?)?;
    assert_eq!(ledger.transactions().len(), 3);
    assert_eq!(ledger.canonical(IdentityId(1)), Some(IdentityId(1)));
    assert_eq!(ledger.canonical(IdentityId(2)), Some(IdentityId(2)));
    Ok(())
}

#[test]
fn i7_invalid_transaction_exposes_none_of_its_effects()
-> Result<(), Box<dyn std::error::Error>> {
    let mut ledger = MemoryLedger::default();
    ledger.commit_envelope(envelope(
        1,
        0,
        vec![ResolutionCommand::PersistObservation(observation(1, None)?)],
    )?)?;
    let invalid = envelope(
        2,
        1,
        vec![
            ResolutionCommand::PersistObservation(observation(2, None)?),
            ResolutionCommand::PersistObservation(observation(1, None)?),
        ],
    )?;
    assert_eq!(
        ledger.commit_envelope(invalid),
        Err(LedgerError::ImmutableObservation(ObservationId(1)))
    );
    assert!(ledger.observation(ObservationId(2)).is_none());
    assert_eq!(ledger.current_version(), CommitVersion::new(1));
    Ok(())
}

#[test]
fn i8_materialized_host_edge_carries_decision_commit_and_idempotency()
-> Result<(), Box<dyn std::error::Error>> {
    let mut host = AuthoritativeHostGraph::with_capacity(profile()?, 2, 1);
    host.add_node(HostNodeId::State(li_core::StateId(1)))?;
    host.add_node(HostNodeId::Event(li_core::EventId(1)))?;
    let relation = HostRelation::Triggers {
        state: li_core::StateId(1),
        event: li_core::EventId(1),
    };
    let first = host.materialize(
        relation,
        DecisionId(7),
        CommitVersion::new(4),
        MaterializationId(9),
    )?;
    assert!(matches!(first, MaterializationOutcome::Applied(_)));
    let edge = host.graph().edge_weights().next();
    assert!(edge.is_some_and(|edge| {
        edge.decision == DecisionId(7) &&
            edge.commit == CommitVersion::new(4) &&
            edge.materialization == MaterializationId(9)
    }));
    Ok(())
}

#[test]
fn i9_replaying_an_idempotency_key_has_no_additional_effect()
-> Result<(), Box<dyn std::error::Error>> {
    let mut ledger = MemoryLedger::default();
    let command = envelope(
        1,
        0,
        vec![ResolutionCommand::PersistObservation(observation(1, None)?)],
    )?;
    let first = ledger.commit_envelope(command.clone())?;
    let replay = ledger.commit_envelope(command)?;
    assert!(!first.replayed);
    assert!(replay.replayed);
    assert_eq!(ledger.transactions().len(), 1);
    assert_eq!(ledger.current_version(), CommitVersion::new(1));
    Ok(())
}
