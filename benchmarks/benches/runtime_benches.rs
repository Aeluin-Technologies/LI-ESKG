//! Benchmark suite evaluating end-to-end runtime ingestion, belief propagation
//! inference, identity resolution, and workspace cleanup performance.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, criterion_group,
    criterion_main,
};
use li_core::{
    BayesRiskPolicy, BoundaryTreatment, CommandEnvelope, CommitVersion,
    ContentHash, DecisionAction, DecisionId, HostRelation, IdempotencyKey,
    IdentityId, IdentityReference, LossModel, MaterializationId, Modality,
    ObservationEnvelope, ObservationId, PayloadRef, ProviderArtifact,
    ProviderId, QualityMetadata, ResolutionCommand, SchemaId, SourceId,
    Timestamp, TransactionId,
};
use li_factors::{
    CandidateBuffer, FactorBuffer, FactorProvider, FactorTable,
    ProviderContext, ProviderError,
};
use li_inference::{SolverScratch, SumProductConfig, SumProductSolver};
use li_runtime::{
    HostMaterializer, MaterializationFailure, MaterializationPlanner,
    PipelineError, ResolutionRuntime, RuntimeSnapshot,
};
use li_storage::{MemoryLedger, ResolutionLedger};
use li_workspace::{HotIdentity, HotWorkspace, WorkerScratch};
use smallvec::SmallVec;

const MAX_INGESTION_CANDIDATES: usize = 32;

#[derive(Debug)]
struct BenchProvider {
    candidates: Box<[IdentityReference]>,
}

impl FactorProvider for BenchProvider {
    fn artifact(&self) -> ProviderArtifact {
        ProviderArtifact {
            provider: ProviderId(1),
            schema: SchemaId(1),
            model_version: 2,
            calibration_id: 1,
        }
    }

    fn generate_candidates(
        &self,
        batch: &[ObservationEnvelope],
        _context: ProviderContext,
        output: &mut CandidateBuffer,
    ) -> Result<(), ProviderError> {
        for index in 0..batch.len() {
            output.push_observation(
                index,
                batch.len(),
                self.candidates.iter().cloned(),
            )?;
        }
        Ok(())
    }

    fn emit_factors(
        &self,
        batch: &[ObservationEnvelope],
        candidates: &CandidateBuffer,
        _context: ProviderContext,
        output: &mut FactorBuffer,
    ) -> Result<(), ProviderError> {
        for index in 0..batch.len() {
            let cardinality = candidates
                .get(index)
                .map_or(2, |values| values.len().saturating_add(2));
            let cardinality_u16 = u16::try_from(cardinality)
                .map_err(|_| ProviderError::InvalidCardinality)?;
            let mut potentials = vec![-8.0; cardinality];
            if let Some(first) = potentials.first_mut() {
                *first = 0.0;
            }
            output.push(FactorTable::new(
                SmallVec::from_slice(&[u32::try_from(index)
                    .map_err(|_| ProviderError::InvalidCardinality)?]),
                SmallVec::from_slice(&[cardinality_u16]),
                potentials,
                Vec::new(),
            )?);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct NoPlanner;

impl MaterializationPlanner for NoPlanner {
    fn plan(
        &self,
        _observation: &ObservationEnvelope,
        _decision: &DecisionAction,
    ) -> Option<HostRelation> {
        None
    }
}

#[derive(Debug, Clone, Copy)]
struct NoMaterializer;

impl HostMaterializer for NoMaterializer {
    fn materialize(
        &mut self,
        _materialization: MaterializationId,
        _relation: HostRelation,
        _decision: DecisionId,
        _commit: CommitVersion,
    ) -> Result<(), MaterializationFailure> {
        Ok(())
    }
}

type BenchRuntime = ResolutionRuntime<MemoryLedger, NoPlanner, NoMaterializer>;

/// Creates a deterministic non-zero replay key.
fn key(value: u64) -> Option<IdempotencyKey> {
    let mut bytes = [0_u8; 16];
    bytes[..8].copy_from_slice(&value.to_le_bytes());
    IdempotencyKey::new(bytes).ok()
}

/// Creates an immutable observation envelope with a compact payload.
fn observation(
    id: u64,
    target: usize,
    step: usize,
) -> Option<ObservationEnvelope> {
    let event_time = id
        .saturating_mul(1_000)
        .saturating_add(u64::try_from(step).unwrap_or(u64::MAX));
    let mut payload = [0_u8; 16];
    payload[..8].copy_from_slice(
        &u64::try_from(target).unwrap_or(u64::MAX).to_le_bytes(),
    );
    payload[8..].copy_from_slice(
        &u64::try_from(step).unwrap_or(u64::MAX).to_le_bytes(),
    );
    ObservationEnvelope::new(
        ObservationId(id),
        SourceId(1),
        Modality(1),
        Timestamp::from_micros(i64::try_from(event_time).unwrap_or(i64::MAX)),
        Timestamp::from_micros(
            i64::try_from(event_time.saturating_add(1)).unwrap_or(i64::MAX),
        ),
        PayloadRef::Inline(Bytes::copy_from_slice(&payload)),
        QualityMetadata::Opaque {
            schema: SchemaId(1),
            bytes: Bytes::new(),
        },
        ContentHash::new([1; 32]),
        None,
    )
    .ok()
}

/// Generates the same target-by-observation scenario matrix as the V1 suite.
fn deterministic_dataset(
    targets: usize,
    observations_per_target: usize,
) -> Option<Vec<ObservationEnvelope>> {
    let capacity = targets.checked_mul(observations_per_target)?;
    let mut observations = Vec::with_capacity(capacity);
    let mut id = 1_u64;
    for target in 0..targets {
        for step in 0..observations_per_target {
            observations.push(observation(id, target, step)?);
            id = id.checked_add(1)?;
        }
    }
    observations.sort_unstable_by_key(ObservationEnvelope::event_time);
    Some(observations)
}

/// Creates a V2 runtime with an optional pre-existing active identity pool.
fn runtime(
    batch_capacity: usize,
    active_pool: usize,
    candidate_count: usize,
) -> Result<BenchRuntime, PipelineError> {
    let mut ledger = MemoryLedger::with_capacity(
        batch_capacity.saturating_mul(3).saturating_add(1),
        active_pool.saturating_add(batch_capacity),
    );
    let mut workspace = HotWorkspace::with_capacity(
        active_pool.saturating_add(batch_capacity),
    );
    if active_pool > 0 {
        let commands = (1..=active_pool as u64)
            .map(|id| ResolutionCommand::CreateIdentity {
                identity: IdentityId(id),
                created_at: Timestamp::UNIX_EPOCH,
            })
            .collect();
        let envelope = CommandEnvelope::new(
            TransactionId(10_000_000),
            CommitVersion::ZERO,
            key(10_000_000).ok_or(PipelineError::IdExhausted)?,
            Timestamp::UNIX_EPOCH,
            Vec::new(),
            commands,
        )?;
        ledger.commit_envelope(envelope)?;
        for id in 1..=active_pool as u64 {
            workspace.insert(HotIdentity::new(
                IdentityId(id),
                0,
                8,
                Timestamp::UNIX_EPOCH,
                ledger.current_version(),
            ));
        }
    }
    let first = (active_pool / 2).saturating_sub(candidate_count / 2);
    let candidates = (0..candidate_count)
        .map(|offset| {
            IdentityReference::Latent(IdentityId(
                u64::try_from(first.saturating_add(offset).saturating_add(1))
                    .unwrap_or(u64::MAX),
            ))
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let provider: Arc<dyn FactorProvider> =
        Arc::new(BenchProvider { candidates });
    let loss = LossModel::new(4.0, 4.0, 4.0, 10.0)?;
    ResolutionRuntime::new(
        ledger,
        NoPlanner,
        NoMaterializer,
        vec![provider],
        RuntimeSnapshot {
            host: CommitVersion::ZERO,
            candidates: 1,
            configuration_hash: ContentHash::new([2; 32]),
        },
        SumProductConfig {
            solver_version: 2,
            max_iterations: 20,
            tolerance: 1.0e-9,
            damping: 0.0,
            candidate_log_prior: 0.0,
            new_log_prior: 0.0,
            noise_log_prior: -4.0,
            boundary_treatment: BoundaryTreatment::Global,
        },
        BayesRiskPolicy {
            policy_version: 2,
            loss_version: 2,
            loss,
        },
        workspace,
        WorkerScratch::with_capacity(
            batch_capacity,
            batch_capacity.saturating_mul(candidate_count),
            batch_capacity,
            batch_capacity.saturating_mul(candidate_count.saturating_add(2)),
        ),
        8,
    )
}

fn bench_pipeline_ingestion(c: &mut Criterion) {
    let mut group =
        c.benchmark_group("RuntimePipeline::IngestionAndResolution");
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    for (targets, observations_per_target) in
        [(100_usize, 100_usize), (1_000, 100), (10_000, 10)]
    {
        let Some(observations) =
            deterministic_dataset(targets, observations_per_target)
        else {
            continue;
        };
        let total = observations.len() as u64;
        group.throughput(Throughput::Elements(total));
        group.bench_with_input(
            BenchmarkId::new(
                "targets_x_obs",
                format!("{targets}x{observations_per_target}"),
            ),
            &observations,
            |b, batch| {
                b.iter_batched(
                    || runtime(batch.len(), 0, 0),
                    |runtime| {
                        black_box(runtime.and_then(|mut runtime| {
                            runtime.process_batch(batch)
                        }))
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_pipeline_indexed_candidate_scaling(c: &mut Criterion) {
    let mut group =
        c.benchmark_group("RuntimePipeline::FixedCandidatePoolScaling");
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    for active_pool in [10_000_usize, 100_000] {
        let Ok(mut runtime) =
            runtime(1, active_pool, MAX_INGESTION_CANDIDATES)
        else {
            continue;
        };
        let mut sequence = 1_000_000_u64;
        group
            .throughput(Throughput::Elements(MAX_INGESTION_CANDIDATES as u64));
        group.bench_with_input(
            BenchmarkId::new("pool_size_fixed_32_candidates", active_pool),
            &active_pool,
            |b, _| {
                b.iter(|| {
                    sequence = sequence.saturating_add(1);
                    let result = observation(sequence, 0, 0)
                        .ok_or(PipelineError::IdExhausted)
                        .and_then(|observation| {
                            runtime.process_batch(&[observation])
                        });
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

fn bench_pipeline_eviction_lifecycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("RuntimePipeline::EvictionAndCleanup");
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);

    for count in [10_000_u64, 100_000] {
        let mut evicted = Vec::with_capacity((count / 2) as usize);
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(
            BenchmarkId::new("evict_expired_50_percent", count),
            &count,
            |b, &node_count| {
                b.iter_batched(
                    || {
                        let mut workspace =
                            HotWorkspace::with_capacity(node_count as usize);
                        for id in 1..=node_count {
                            let activation =
                                if id % 2 == 0 { 100 } else { 1_000_000 };
                            workspace.insert(HotIdentity::new(
                                IdentityId(id),
                                0,
                                4,
                                Timestamp::from_micros(activation),
                                CommitVersion::ZERO,
                            ));
                        }
                        workspace
                    },
                    |mut workspace| {
                        workspace.evict_before(
                            Timestamp::from_micros(500_000),
                            &mut evicted,
                        );
                        black_box(evicted.len());
                        black_box(workspace)
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

/// Creates the two-variable factor graph used by V2-only solver benchmarks.
fn collective_fixture() -> Option<(CandidateBuffer, FactorTable)> {
    let mut candidates = CandidateBuffer::with_capacity(2, 2);
    candidates.reset(2);
    candidates
        .push_observation(0, 2, [IdentityReference::Latent(IdentityId(1))])
        .ok()?;
    candidates
        .push_observation(1, 2, [IdentityReference::Latent(IdentityId(2))])
        .ok()?;
    let factor = FactorTable::new(
        SmallVec::from_slice(&[0, 1]),
        SmallVec::from_slice(&[3, 3]),
        vec![0.0, -4.0, -4.0, -4.0, 0.0, -4.0, -4.0, -4.0, 0.0],
        Vec::new(),
    )
    .ok()?;
    Some((candidates, factor))
}

fn bench_v2_collective_inference(c: &mut Criterion) {
    let Some((candidates, factor)) = collective_fixture() else {
        return;
    };
    let config = |boundary_treatment| SumProductConfig {
        solver_version: 2,
        max_iterations: 20,
        tolerance: 1.0e-10,
        damping: 0.25,
        candidate_log_prior: 0.0,
        new_log_prior: -1.0,
        noise_log_prior: -2.0,
        boundary_treatment,
    };
    let Ok(exact_solver) =
        SumProductSolver::new(config(BoundaryTreatment::Global))
    else {
        return;
    };
    let Ok(loopy_solver) =
        SumProductSolver::new(config(BoundaryTreatment::CachedApproximation))
    else {
        return;
    };
    let mut exact_scratch = SolverScratch::default();
    c.bench_function("V2::inference::exact_tree", |b| {
        b.iter(|| {
            black_box(exact_solver.solve(
                &candidates,
                std::slice::from_ref(&factor),
                &mut exact_scratch,
            ))
        });
    });
    let loopy_factors = [factor.clone(), factor];
    let mut loopy_scratch = SolverScratch::default();
    c.bench_function("V2::inference::loopy_cycle", |b| {
        b.iter(|| {
            black_box(loopy_solver.solve(
                &candidates,
                &loopy_factors,
                &mut loopy_scratch,
            ))
        });
    });
}

criterion_group!(
    benches,
    bench_pipeline_ingestion,
    bench_pipeline_indexed_candidate_scaling,
    bench_pipeline_eviction_lifecycle,
    bench_v2_collective_inference
);
criterion_main!(benches);
