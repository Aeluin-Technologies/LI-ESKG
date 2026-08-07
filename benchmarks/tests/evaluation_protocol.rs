//! Executable controlled checks corresponding to paper RQ1-RQ8.
//!
//! These tests validate mathematical kernels and experimental contracts. They
//! deliberately do not claim empirical superiority on an unpublished dataset.

use std::sync::Arc;

use benchmarks::{
    CalibrationMetrics, CalibrationSample, EvaluationPlan, ProtocolError,
    ScaleMeasurement,
};
use bytes::Bytes;
use li_core::{
    AssociationOutcome, BayesRiskPolicy, BoundaryTreatment, CommandEnvelope,
    CommitVersion, ContentHash, DecisionAction, DecisionId, DecisionRecord,
    HostNodeId, HostRelation, IdempotencyKey, IdentityId, IdentityReference,
    InferenceId, InferenceProvenance, InferenceRecord, LossModel,
    MaterializationId, Modality, NormalizedDistribution, ObservationEnvelope,
    ObservationId, OutcomeProbability, PayloadRef, Probability,
    ProviderArtifact, ProviderId, QualityMetadata, ResolutionCommand,
    SchemaId, SolverDiagnostics, SolverStoppingReason, SourceId,
    SplitPartition, SplitPlan, Timestamp, TransactionId, VersionInterval,
};
use li_factors::{CandidateBuffer, FactorTable};
use li_inference::{SolverScratch, SumProductConfig, SumProductSolver};
use li_model::{
    AuthoritativeHostGraph, HostSchemaProfile, InteroperabilityProjector,
    MaterializationOutcome, RdfProfile,
};
use li_storage::{
    DurableLedger, MemoryKvBackend, MemoryLedger, ResolutionLedger,
};
use smallvec::SmallVec;

fn observation(
    id: u64,
) -> Result<ObservationEnvelope, li_core::EvidenceError> {
    ObservationEnvelope::new(
        ObservationId(id),
        SourceId(1),
        Modality(1),
        Timestamp::from_micros(1),
        Timestamp::from_micros(2),
        PayloadRef::Inline(Bytes::from_static(b"evaluation")),
        QualityMetadata::Opaque {
            schema: SchemaId(1),
            bytes: Bytes::new(),
        },
        ContentHash::new([u8::try_from(id).unwrap_or(u8::MAX); 32]),
        None,
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

fn distribution() -> Result<NormalizedDistribution, li_core::DistributionError>
{
    NormalizedDistribution::new(
        vec![
            OutcomeProbability {
                outcome: AssociationOutcome::Identity(
                    IdentityReference::Latent(IdentityId(1)),
                ),
                probability: Probability::new(0.8),
            },
            OutcomeProbability {
                outcome: AssociationOutcome::New,
                probability: Probability::new(0.1),
            },
            OutcomeProbability {
                outcome: AssociationOutcome::Noise,
                probability: Probability::new(0.1),
            },
        ],
        None,
    )
}

fn inference(
    provider: ProviderId,
) -> Result<InferenceRecord, li_core::DistributionError> {
    Ok(InferenceRecord {
        id: InferenceId(provider.0),
        observation: ObservationId(1),
        distribution: distribution()?,
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
            host_snapshot: CommitVersion::ZERO,
            configuration_hash: ContentHash::new([7; 32]),
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
        validity: VersionInterval::current(CommitVersion::new(1)),
    })
}

fn policy() -> Result<BayesRiskPolicy, li_core::DecisionError> {
    Ok(BayesRiskPolicy {
        policy_version: 1,
        loss_version: 1,
        loss: LossModel::new(1.0, 1.0, 1.0, 10.0)?,
    })
}

fn solver(
    boundary_treatment: BoundaryTreatment,
) -> Result<SumProductSolver, li_inference::SolverError> {
    SumProductSolver::new(SumProductConfig {
        solver_version: 1,
        max_iterations: 20,
        tolerance: 1.0e-10,
        damping: 0.25,
        candidate_log_prior: 0.0,
        new_log_prior: 0.0,
        noise_log_prior: -8.0,
        boundary_treatment,
    })
}

fn candidates() -> Result<CandidateBuffer, li_factors::ProviderError> {
    let mut candidates = CandidateBuffer::with_capacity(2, 2);
    candidates.reset(2);
    for index in 0..2 {
        candidates.push_observation(
            index,
            2,
            [IdentityReference::Latent(IdentityId(1))],
        )?;
    }
    Ok(candidates)
}

fn host_profile() -> Result<HostSchemaProfile, li_model::HostSchemaError> {
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
fn rq1_collective_factor_corrects_a_controlled_pairwise_error()
-> Result<(), Box<dyn std::error::Error>> {
    let candidates = candidates()?;
    let unary = [
        FactorTable::new(
            SmallVec::from_slice(&[0]),
            SmallVec::from_slice(&[3]),
            vec![-0.2, 0.0, -8.0],
            Vec::new(),
        )?,
        FactorTable::new(
            SmallVec::from_slice(&[1]),
            SmallVec::from_slice(&[3]),
            vec![-0.2, 0.0, -8.0],
            Vec::new(),
        )?,
    ];
    let collective = FactorTable::new(
        SmallVec::from_slice(&[0, 1]),
        SmallVec::from_slice(&[3, 3]),
        vec![4.0, 0.0, -8.0, 0.0, 0.0, -8.0, -8.0, -8.0, -8.0],
        Vec::new(),
    )?;
    let solver = solver(BoundaryTreatment::Global)?;
    let independent =
        solver.solve(&candidates, &unary, &mut SolverScratch::default())?;
    let factors = [unary[0].clone(), unary[1].clone(), collective];
    let collective =
        solver.solve(&candidates, &factors, &mut SolverScratch::default())?;
    let policy = policy()?;
    assert!(independent.distributions.iter().all(|distribution| {
        matches!(policy.decide(distribution), DecisionAction::CreateIdentity)
    }));
    assert!(collective.distributions.iter().all(|distribution| {
        matches!(
            policy.decide(distribution),
            DecisionAction::Assign(IdentityReference::Latent(IdentityId(1)))
        )
    }));
    Ok(())
}

#[test]
fn rq2_calibration_metrics_match_proper_scoring_rule_definitions()
-> Result<(), ProtocolError> {
    let samples = [
        CalibrationSample::new(0.8, true)?,
        CalibrationSample::new(0.2, false)?,
    ];
    let metrics = CalibrationMetrics::compute(&samples, 2)?;
    assert!((metrics.negative_log_likelihood + 0.8_f64.ln()).abs() < 1.0e-12);
    assert!((metrics.brier - 0.04).abs() < 1.0e-12);
    assert!((metrics.expected_calibration_error - 0.2).abs() < 1.0e-12);
    Ok(())
}

#[test]
fn rq3_false_merge_can_be_split_without_deleting_history()
-> Result<(), Box<dyn std::error::Error>> {
    let mut ledger = MemoryLedger::default();
    ledger.commit_envelope(envelope(
        1,
        0,
        vec![
            ResolutionCommand::PersistObservation(observation(1)?),
            ResolutionCommand::PersistObservation(observation(2)?),
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
            DecisionId(20),
            IdentityId(1),
            IdentityId(2),
            Vec::new(),
            1,
            1,
        )?],
    )?)?;
    let split = SplitPlan::new(
        vec![DecisionId(20)],
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
fn rq4_boundary_approximations_are_never_reported_as_globally_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let candidates = candidates()?;
    for treatment in [
        BoundaryTreatment::CachedApproximation,
        BoundaryTreatment::TruncatedApproximation,
        BoundaryTreatment::OmittedApproximation,
    ] {
        let posterior = solver(treatment)?.solve(
            &candidates,
            &[],
            &mut SolverScratch::default(),
        )?;
        assert_eq!(posterior.diagnostics.boundary_treatment, treatment);
        assert!(!treatment.preserves_global_marginal());
    }
    Ok(())
}

#[test]
fn rq5_scale_cells_require_all_independent_dimensions_and_valid_tails() {
    let plan = EvaluationPlan::canonical();
    assert_eq!(plan.validate(), Ok(()));
    let invalid = ScaleMeasurement {
        active_identities: 1_000,
        candidates: 16,
        batch_size: 64,
        factor_arity: 3,
        iterations: 10,
        throughput: 50_000.0,
        p50_micros: 100.0,
        p95_micros: 90.0,
        p99_micros: 150.0,
        peak_resident_bytes: 1_000_000,
        storage_bytes: 2_000_000,
    };
    assert_eq!(
        invalid.validate(),
        Err(ProtocolError::InvalidScaleMeasurement)
    );
}

#[test]
fn rq6_durable_replay_and_host_materialization_are_idempotent()
-> Result<(), Box<dyn std::error::Error>> {
    let command = envelope(
        1,
        0,
        vec![ResolutionCommand::PersistObservation(observation(1)?)],
    )?;
    let mut durable = DurableLedger::open(MemoryKvBackend::default())?;
    durable.commit_envelope(command)?;
    let backend = durable.into_backend();
    let reopened = DurableLedger::open(backend)?;
    assert_eq!(reopened.current_version(), CommitVersion::new(1));
    assert_eq!(reopened.memory().transactions().len(), 1);

    let mut host =
        AuthoritativeHostGraph::with_capacity(host_profile()?, 2, 1);
    host.add_node(HostNodeId::State(li_core::StateId(1)))?;
    host.add_node(HostNodeId::Event(li_core::EventId(1)))?;
    let relation = HostRelation::Triggers {
        state: li_core::StateId(1),
        event: li_core::EventId(1),
    };
    host.materialize(
        relation,
        DecisionId(1),
        CommitVersion::new(1),
        MaterializationId(1),
    )?;
    let replay = host.materialize(
        relation,
        DecisionId(1),
        CommitVersion::new(1),
        MaterializationId(1),
    )?;
    assert!(matches!(replay, MaterializationOutcome::Replayed(_)));
    assert_eq!(host.edge_count(), 1);
    Ok(())
}

#[test]
fn rq7_canonical_factors_preserve_decisions_but_not_provider_provenance()
-> Result<(), Box<dyn std::error::Error>> {
    let first = inference(ProviderId(1))?;
    let second = inference(ProviderId(2))?;
    assert_eq!(first.distribution, second.distribution);
    assert_eq!(
        policy()?.decide(&first.distribution),
        policy()?.decide(&second.distribution)
    );
    assert_ne!(first.provenance.providers, second.provenance.providers);
    Ok(())
}

#[test]
fn rq8_exports_include_prov_nanopublication_shacl_and_native_round_trip()
-> Result<(), Box<dyn std::error::Error>> {
    let observation = observation(1)?;
    let inference = inference(ProviderId(1))?;
    let decision = DecisionRecord::new(
        DecisionId(1),
        inference.id,
        policy()?.decide(&inference.distribution),
        1,
        1,
        VersionInterval::current(CommitVersion::new(1)),
        &inference.distribution,
    )?;
    let projector = InteroperabilityProjector::new(
        "https://example.test/",
        RdfProfile::Rdf12,
    )?;
    let mut export = String::new();
    projector.project_nanopublication(
        &observation,
        &inference,
        &decision,
        "https://example.test/agent",
        &mut export,
    )?;
    assert!(export.contains("prov:wasGeneratedBy"));
    assert!(export.contains("np:Nanopublication"));
    assert!(export.contains("rdf:reifies"));
    projector.write_shacl(&mut export)?;
    assert!(export.contains("li:DecisionShape"));
    assert!(export.contains("sh:minCount 1"));
    assert!(export.contains("sh:maxCount 1"));

    let mut host =
        AuthoritativeHostGraph::with_capacity(host_profile()?, 2, 1);
    host.add_node(HostNodeId::State(li_core::StateId(1)))?;
    host.add_node(HostNodeId::Event(li_core::EventId(1)))?;
    host.materialize(
        HostRelation::Triggers {
            state: li_core::StateId(1),
            event: li_core::EventId(1),
        },
        decision.id,
        CommitVersion::new(1),
        MaterializationId(1),
    )?;
    assert!(host.graph().edge_weights().next().is_some_and(|edge| {
        edge.predicate.as_ref() == "native:triggers" &&
            edge.decision == decision.id
    }));
    Ok(())
}
