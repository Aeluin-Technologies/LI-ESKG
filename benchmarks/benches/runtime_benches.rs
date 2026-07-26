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
use li_core::ids::{IdentityId, ObservationId, VertexId};
use li_core::observation::{Evidence, Modality, Observation, Timestamp};
use li_core::probability::{Confidence, Probability};
use li_core::relation::Relation;
use li_factors::compiler::FactorCompiler;
use li_factors::factor::{Factor, FactorScope};
use li_model::ontology::IdentityNode;
use li_model::operations::GraphOperation;
use li_runtime::engine::{EngineConfig, RuntimeEngine};
use li_runtime::executor::ExecutionSink;
use li_workspace::checkpoint::WorkspaceSnapshot;
use li_workspace::workspace::ActiveWorkspace;
use rand::RngExt;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;

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

pub struct BenchSpatioTemporalFactor {
    pub candidate_id: IdentityId,
    pub obs_x: f64,
    pub obs_y: f64,
    pub obs_time: i64,
    pub belief_x: f64,
    pub belief_y: f64,
    pub belief_time: i64,
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

        let dx = self.obs_x - self.belief_x;
        let dy = self.obs_y - self.belief_y;
        let dist = (dx * dx + dy * dy).sqrt();

        let dt_sec =
            ((self.obs_time - self.belief_time).abs().max(1)) as f64 / 1_000.0;
        let speed = dist / dt_sec;

        if speed > 300.0 {
            Probability::new(0.001)
        } else {
            let likelihood = (-dist / 100.0).exp();
            Probability::new(likelihood.clamp(0.01, 0.99))
        }
    }
}

pub struct BenchFactorCompiler;

impl FactorCompiler<BenchGpsPayload, BenchTrackSummary>
    for BenchFactorCompiler
{
    fn compile_factors(
        &self,
        evidence: &Evidence<BenchGpsPayload>,
        active_beliefs: &[BeliefState<BenchTrackSummary>],
    ) -> Vec<Box<dyn Factor>> {
        let mut factors: Vec<Box<dyn Factor>> = Vec::new();
        for &cand_id in &evidence.candidates {
            if let Some(belief) =
                active_beliefs.iter().find(|b| b.identity == cand_id)
            {
                factors.push(Box::new(BenchSpatioTemporalFactor {
                    candidate_id: cand_id,
                    obs_x: evidence.observation.payload.x,
                    obs_y: evidence.observation.payload.y,
                    obs_time: evidence.observation.timestamp.as_micros(),
                    belief_x: belief.summary.last_x,
                    belief_y: belief.summary.last_y,
                    belief_time: belief.summary.last_timestamp.as_micros(),
                }));
            }
        }
        factors
    }
}

#[derive(Default, Clone)]
pub struct BenchWorkspace {
    beliefs: Vec<BeliefState<BenchTrackSummary>>,
}

impl ActiveWorkspace for BenchWorkspace {
    type Summary = BenchTrackSummary;

    fn insert(&mut self, belief: BeliefState<Self::Summary>) {
        if let Some(existing) = self
            .beliefs
            .iter_mut()
            .find(|b| b.identity == belief.identity)
        {
            *existing = belief;
        } else {
            self.beliefs.push(belief);
        }
    }

    fn get(&self, id: IdentityId) -> Option<&BeliefState<Self::Summary>> {
        self.beliefs.iter().find(|b| b.identity == id)
    }

    fn get_mut(
        &mut self,
        id: IdentityId,
    ) -> Option<&mut BeliefState<Self::Summary>> {
        self.beliefs.iter_mut().find(|b| b.identity == id)
    }

    fn active_beliefs(&self) -> Vec<&BeliefState<Self::Summary>> {
        self.beliefs.iter().collect()
    }

    fn evict_expired(
        &mut self,
        current_time: Timestamp,
        ttl: i64,
    ) -> Vec<BeliefState<Self::Summary>> {
        let mut retained = Vec::new();
        let mut evicted = Vec::new();
        for b in self.beliefs.drain(..) {
            if current_time.as_micros() - b.last_update.as_micros() > ttl {
                evicted.push(b);
            } else {
                retained.push(b);
            }
        }
        self.beliefs = retained;
        evicted
    }

    fn create_snapshot(
        &self,
        current_time: Timestamp,
    ) -> WorkspaceSnapshot<Self::Summary> {
        WorkspaceSnapshot {
            timestamp: current_time,
            active_states: self.beliefs.clone(),
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct BenchGraphSink {
    pub identities: Vec<IdentityNode>,
    pub observations: Vec<Observation<BenchGpsPayload>>,
    pub relations: Vec<(VertexId, Relation, VertexId, Timestamp)>,
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
                GraphOperation::CommitIdentity(node) => {
                    self.identities.push(node.clone())
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

    // Augmentation du temps de mesure pour absorber le flux complet sur de
    // grands volumes
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(25));
    group.sample_size(15);

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
                            let candidate_ids: Vec<IdentityId> = engine
                                .executor()
                                .sink()
                                .identities
                                .iter()
                                .map(|n| n.id)
                                .collect();

                            let obs_x = obs.payload.x;
                            let obs_y = obs.payload.y;
                            let obs_timestamp = obs.timestamp;

                            // Passage direct par valeur sans appel à .clone()
                            // sur l'observation
                            let evidence = Evidence {
                                observation: obs,
                                candidates: candidate_ids,
                            };

                            engine
                                .submit_event(RuntimeEvent::Observation(
                                    evidence,
                                ))
                                .unwrap();

                            engine.tick::<()>().unwrap();

                            if let Some(last_rel) =
                                engine.executor().sink().relations.last()
                            {
                                let assigned_id = IdentityId(last_rel.2.0);
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

fn bench_pipeline_scaling_active_beliefs(c: &mut Criterion) {
    let mut group = c.benchmark_group("RuntimePipeline::CandidateScaling");

    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);

    for active_candidates in &[10_000, 100_000] {
        let size = *active_candidates;
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::new("single_obs_against_active_pool", size),
            &size,
            |b, &candidate_count| {
                b.iter_batched(
                    || {
                        let config = EngineConfig {
                            decision_threshold: 0.35,
                            direct_assignment_threshold: 0.95,
                        };
                        let mut workspace = BenchWorkspace::default();
                        let mut sink = BenchGraphSink::default();

                        let candidates: Vec<IdentityId> = (1..=
                            candidate_count as u64)
                            .map(|id| {
                                let target_id = IdentityId(id);
                                sink.identities.push(IdentityNode {
                                    id: target_id,
                                    created_at: Timestamp::from_millis(1000),
                                });
                                workspace.insert(BeliefState {
                                    identity: target_id,
                                    summary: BenchTrackSummary {
                                        last_x: (id as f64) * 10.0,
                                        last_y: (id as f64) * 10.0,
                                        last_timestamp: Timestamp::from_millis(
                                            1000,
                                        ),
                                    },
                                    posterior: Probability::new(0.85),
                                    last_update: Timestamp::from_millis(1000),
                                });
                                target_id
                            })
                            .collect();

                        let engine: RuntimeEngine<
                            BenchGpsPayload,
                            (),
                            BenchTrackSummary,
                            BenchWorkspace,
                            BenchFactorCompiler,
                            BenchGraphSink,
                        > = RuntimeEngine::new(
                            config,
                            10,
                            workspace,
                            BenchFactorCompiler,
                            sink,
                        );

                        let obs = Observation {
                            id: ObservationId(999_999),
                            modality: Modality(1),
                            timestamp: Timestamp::from_millis(2000),
                            confidence: Confidence::new(0.92),
                            payload: BenchGpsPayload { x: 500.0, y: 500.0 },
                        };

                        let evidence = Evidence {
                            observation: obs,
                            candidates,
                        };

                        (engine, evidence)
                    },
                    |(mut engine, evidence)| {
                        engine
                            .submit_event(RuntimeEvent::Observation(evidence))
                            .unwrap();
                        engine.tick::<()>().unwrap();
                        black_box(engine);
                    },
                    BatchSize::LargeInput,
                );
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
                        let mut workspace = BenchWorkspace::default();
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
    bench_pipeline_scaling_active_beliefs,
    bench_pipeline_eviction_lifecycle
);
criterion_main!(benches);
