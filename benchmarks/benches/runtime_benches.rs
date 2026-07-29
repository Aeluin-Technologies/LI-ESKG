//! Benchmark suite evaluating end-to-end runtime ingestion, belief propagation
//! inference, identity resolution, and workspace cleanup performance.

use std::boxed::Box;
use std::hint::black_box;
use std::time::Duration;
use std::vec::Vec;

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, criterion_group,
    criterion_main,
};
use li_core::belief::BeliefState;
use li_core::events::RuntimeEvent;
use li_core::ids::{IdentityId, ObservationId};
use li_core::observation::{Evidence, Modality, Observation, Timestamp};
use li_core::ontology::Vertex;
use li_core::probability::{Confidence, Probability};
use li_core::relation::Relation;
use li_factors::compiler::{DirectMapDecision, FactorCompiler};
use li_factors::factor::{Factor, FactorScope};
use li_model::operations::GraphOperation;
use li_runtime::engine::{EngineConfig, RuntimeEngine};
use li_runtime::executor::ExecutionSink;
use li_workspace::{ActiveWorkspace, InMemoryWorkspace};
use rand::RngExt;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;

const MAX_INGESTION_CANDIDATES: usize = 32;

#[derive(Debug, Clone, PartialEq)]
pub struct BenchGpsPayload {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BenchTrackSummary {
    pub last_x: f64,
    pub last_y: f64,
    pub last_timestamp: Timestamp,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BenchIdentityNode {
    pub id: IdentityId,
    pub created_at: Timestamp,
}

pub struct BenchSpatioTemporalFactor {
    pub candidate_id: IdentityId,
    pub obs_x: f64,
    pub obs_y: f64,
    pub obs_time: i64,
    pub belief_x: f64,
    pub belief_y: f64,
    pub belief_time: i64,
}

/// Computes the unary spatio-temporal likelihood used by both inference paths.
#[inline]
fn spatio_temporal_likelihood(
    obs_x: f64,
    obs_y: f64,
    obs_time: i64,
    belief_x: f64,
    belief_y: f64,
    belief_time: i64,
) -> Probability {
    let dx = obs_x - belief_x;
    let dy = obs_y - belief_y;
    let dist = (dx * dx + dy * dy).sqrt();
    let dt_sec = (obs_time - belief_time).abs().max(1) as f64 / 1_000.0;
    let speed = dist / dt_sec;

    if speed > 300.0 {
        Probability::new(0.001)
    } else {
        Probability::new((-dist / 100.0).exp().clamp(0.01, 0.99))
    }
}

impl FactorScope for BenchSpatioTemporalFactor {
    fn scope(&self) -> &[IdentityId] {
        core::slice::from_ref(&self.candidate_id)
    }
}

impl Factor for BenchSpatioTemporalFactor {
    fn evaluate(&self, assignment: &[IdentityId]) -> Probability {
        if assignment.is_empty() || assignment[0] != self.candidate_id {
            return Probability::new(0.01);
        }

        spatio_temporal_likelihood(
            self.obs_x,
            self.obs_y,
            self.obs_time,
            self.belief_x,
            self.belief_y,
            self.belief_time,
        )
    }
}

pub struct BenchFactorCompiler;

impl FactorCompiler<BenchGpsPayload, BenchTrackSummary>
    for BenchFactorCompiler
{
    fn compile_factors(
        &self,
        evidence: &Evidence<BenchGpsPayload>,
        active_beliefs: &[&BeliefState<BenchTrackSummary>],
    ) -> Vec<Box<dyn Factor>> {
        let mut factors: Vec<Box<dyn Factor>> =
            Vec::with_capacity(active_beliefs.len());
        for belief in active_beliefs {
            factors.push(Box::new(BenchSpatioTemporalFactor {
                candidate_id: belief.identity,
                obs_x: evidence.observation.payload.x,
                obs_y: evidence.observation.payload.y,
                obs_time: evidence.observation.timestamp.as_micros(),
                belief_x: belief.summary.last_x,
                belief_y: belief.summary.last_y,
                belief_time: belief.summary.last_timestamp.as_micros(),
            }));
        }
        factors
    }

    fn try_direct_map(
        &self,
        evidence: &Evidence<BenchGpsPayload>,
        active_beliefs: &[&BeliefState<BenchTrackSummary>],
        decision_threshold: Probability,
    ) -> DirectMapDecision {
        const BACKGROUND_LIKELIHOOD: f64 = 0.01;

        let observation = &evidence.observation;
        let mut best: Option<(IdentityId, f64)> = None;
        for belief in active_beliefs {
            let likelihood = spatio_temporal_likelihood(
                observation.payload.x,
                observation.payload.y,
                observation.timestamp.as_micros(),
                belief.summary.last_x,
                belief.summary.last_y,
                belief.summary.last_timestamp.as_micros(),
            )
            .value();
            let marginal = likelihood / (likelihood + BACKGROUND_LIKELIHOOD);
            if marginal < decision_threshold.value() {
                continue;
            }

            let replace = match best {
                Some((best_id, best_score)) => {
                    marginal > best_score ||
                        (marginal == best_score &&
                            belief.identity < best_id)
                },
                None => true,
            };
            if replace {
                best = Some((belief.identity, marginal));
            }
        }

        match best {
            Some((identity, _)) => DirectMapDecision::Assign(identity),
            None => DirectMapDecision::CreateIdentity,
        }
    }
}

type BenchWorkspace = InMemoryWorkspace<BenchTrackSummary>;

#[derive(Default, Clone, Debug)]
pub struct BenchGraphSink {
    pub identities: Vec<BenchIdentityNode>,
    pub observations: Vec<Observation<BenchGpsPayload>>,
    pub relations: Vec<(Vertex, Relation, Vertex, Timestamp)>,
}

impl ExecutionSink<BenchGpsPayload, (), BenchTrackSummary> for BenchGraphSink {
    type Error = ();

    fn execute_batch(
        &mut self,
        operations: &[GraphOperation<
            BenchGpsPayload,
            (),
            BenchTrackSummary,
        >],
    ) -> Result<(), Self::Error> {
        for op in operations {
            match op {
                GraphOperation::CommitIdentity { id, created_at } => {
                    self.identities.push(BenchIdentityNode {
                        id: *id,
                        created_at: *created_at,
                    });
                },
                GraphOperation::CommitObservation(obs) => {
                    self.observations.push(obs.clone())
                },
                GraphOperation::CommitRelation {
                    source,
                    relation,
                    target,
                    created_at,
                } => self.relations.push((
                    *source,
                    *relation,
                    *target,
                    *created_at,
                )),
                _ => {},
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct BenchNoopSink;

impl ExecutionSink<BenchGpsPayload, (), BenchTrackSummary> for BenchNoopSink {
    type Error = ();

    #[inline]
    fn execute_batch(
        &mut self,
        operations: &[GraphOperation<
            BenchGpsPayload,
            (),
            BenchTrackSummary,
        >],
    ) -> Result<(), Self::Error> {
        black_box(operations);
        Ok(())
    }
}

fn generate_deterministic_dataset(
    num_targets: usize,
    obs_per_target: usize,
    seed: u64,
) -> Vec<Observation<BenchGpsPayload>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut observations = Vec::with_capacity(num_targets * obs_per_target);
    let mut obs_id = 1u64;

    for target in 0..num_targets {
        let base_x = (target as f64) * 500.0;
        let base_y = (target as f64) * 300.0;
        let mut current_time = 1_000_000i64;

        for step in 0..obs_per_target {
            current_time += rng.random_range(500..3_000);
            let noise_x = rng.random_range(-15.0..15.0);
            let noise_y = rng.random_range(-15.0..15.0);

            observations.push(Observation {
                id: ObservationId(obs_id),
                modality: Modality(1),
                timestamp: Timestamp::from_millis(current_time),
                confidence: Confidence::new(rng.random_range(0.80..0.98)),
                payload: BenchGpsPayload {
                    x: base_x + (step as f64 * 5.0) + noise_x,
                    y: base_y + (step as f64 * 2.0) + noise_y,
                },
            });
            obs_id += 1;
        }
    }

    observations.sort_by_key(|o| o.timestamp.as_micros());
    observations
}

fn bench_pipeline_ingestion(c: &mut Criterion) {
    let mut group =
        c.benchmark_group("RuntimePipeline::IngestionAndResolution");
    group.warm_up_time(Duration::from_secs(3));
    // Ten samples avoid long-run warnings while retaining Criterion's minimum
    // statistical sample count. The optimized pipeline completes every
    // scenario comfortably inside this measurement window.
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    let scenarios = [(100, 100), (1000, 100), (10_000, 10)];
    for &(num_targets, obs_per_target) in &scenarios {
        let total_obs = (num_targets * obs_per_target) as u64;
        let observations =
            generate_deterministic_dataset(num_targets, obs_per_target, 1337);

        group.throughput(Throughput::Elements(total_obs));
        group.bench_with_input(
            BenchmarkId::new(
                "targets_x_obs",
                format!("{num_targets}x{obs_per_target}"),
            ),
            &observations,
            |b, obs_list| {
                b.iter_batched(
                    || {
                        let config = EngineConfig {
                            decision_threshold: 0.35,
                            direct_assignment_threshold: 0.95,
                        };
                        let engine: RuntimeEngine<
                            BenchGpsPayload,
                            (),
                            BenchTrackSummary,
                            BenchWorkspace,
                            BenchFactorCompiler,
                            BenchGraphSink,
                        > = RuntimeEngine::new(
                            config,
                            obs_list.len() + 10,
                            BenchWorkspace::default(),
                            BenchFactorCompiler,
                            BenchGraphSink::default(),
                        );
                        (engine, obs_list.clone())
                    },
                    |(mut engine, stream)| {
                        for obs in stream {
                            // Model the bounded neighborhood produced by an
                            // upstream metric index before factor compilation.
                            let identities =
                                &engine.executor().sink().identities;
                            let candidate_start = identities
                                .len()
                                .saturating_sub(MAX_INGESTION_CANDIDATES);
                            let candidate_ids: Vec<IdentityId> = identities
                                [candidate_start..]
                                .iter()
                                .map(|n| n.id)
                                .collect();

                            let obs_x = obs.payload.x;
                            let obs_y = obs.payload.y;
                            let obs_timestamp = obs.timestamp;

                            let evidence = Evidence {
                                observation: obs,
                                candidates: candidate_ids,
                            };

                            let submission = engine.submit_event(
                                RuntimeEvent::Observation(evidence),
                            );
                            debug_assert!(submission.is_ok());

                            let tick = engine.tick::<()>();
                            debug_assert!(matches!(tick, Ok(true)));

                            if let Some(last_rel) =
                                engine.executor().sink().relations.last() &&
                                let Vertex::Identity(assigned_id) =
                                    last_rel.2
                            {
                                engine.workspace_mut().insert(BeliefState {
                                    identity: assigned_id,
                                    summary: BenchTrackSummary {
                                        last_x: obs_x,
                                        last_y: obs_y,
                                        last_timestamp: obs_timestamp,
                                    },
                                    posterior: Probability::new(0.90),
                                    last_update: obs_timestamp,
                                });
                            }
                        }
                        black_box(engine)
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

    for active_pool_size in &[10_000, 100_000] {
        let size = *active_pool_size;
        let mut workspace = BenchWorkspace::with_capacity(size);

        for id in 1..=size as u64 {
            let target_id = IdentityId(id);
            workspace.insert(BeliefState {
                identity: target_id,
                summary: BenchTrackSummary {
                    last_x: (id as f64) * 10.0,
                    last_y: (id as f64) * 10.0,
                    last_timestamp: Timestamp::from_millis(1000),
                },
                posterior: Probability::new(0.85),
                last_update: Timestamp::from_millis(1000),
            });
        }

        let candidate_start = (size as u64 / 2)
            .saturating_sub((MAX_INGESTION_CANDIDATES / 2) as u64);
        let candidates: Vec<IdentityId> = (candidate_start..
            candidate_start + MAX_INGESTION_CANDIDATES as u64)
            .map(IdentityId)
            .collect();
        let config = EngineConfig {
            decision_threshold: 0.35,
            direct_assignment_threshold: 0.95,
        };
        let mut engine: RuntimeEngine<
            BenchGpsPayload,
            (),
            BenchTrackSummary,
            BenchWorkspace,
            BenchFactorCompiler,
            BenchNoopSink,
        > = RuntimeEngine::new(
            config,
            1,
            workspace,
            BenchFactorCompiler,
            BenchNoopSink,
        );
        let evidence = Evidence {
            observation: Observation {
                id: ObservationId(999_999),
                modality: Modality(1),
                timestamp: Timestamp::from_millis(2000),
                confidence: Confidence::new(0.92),
                payload: BenchGpsPayload {
                    x: (size as f64 / 2.0) * 10.0,
                    y: (size as f64 / 2.0) * 10.0,
                },
            },
            candidates,
        };

        group
            .throughput(Throughput::Elements(MAX_INGESTION_CANDIDATES as u64));
        group.bench_with_input(
            BenchmarkId::new("pool_size_fixed_32_candidates", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let event =
                        RuntimeEvent::Observation(black_box(evidence.clone()));
                    let submission = engine.submit_event(event);
                    debug_assert!(submission.is_ok());

                    let tick = engine.tick::<()>();
                    debug_assert!(matches!(tick, Ok(true)));
                    black_box(engine.workspace());
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

    for total_nodes in &[10_000, 100_000] {
        let count = *total_nodes as u64;
        group.throughput(Throughput::Elements(count));

        group.bench_with_input(
            BenchmarkId::new("evict_expired_50_percent", count),
            &count,
            |b, &node_count| {
                b.iter_batched(
                    || {
                        let mut workspace =
                            BenchWorkspace::with_capacity(node_count as usize);
                        for id in 1..=node_count {
                            let last_update =
                                if id % 2 == 0 { 100 } else { 1_000_000 };
                            workspace.insert(BeliefState {
                                identity: IdentityId(id),
                                summary: BenchTrackSummary {
                                    last_x: 0.0,
                                    last_y: 0.0,
                                    last_timestamp: Timestamp::from_millis(
                                        last_update,
                                    ),
                                },
                                posterior: Probability::new(0.8),
                                last_update: Timestamp::from_millis(
                                    last_update,
                                ),
                            });
                        }
                        workspace
                    },
                    |mut ws| {
                        let evicted = ws.evict_expired(
                            Timestamp::from_millis(1_000_000),
                            500_000,
                        );
                        black_box(evicted);
                        black_box(ws);
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_pipeline_ingestion,
    bench_pipeline_indexed_candidate_scaling,
    bench_pipeline_eviction_lifecycle
);
criterion_main!(benches);
