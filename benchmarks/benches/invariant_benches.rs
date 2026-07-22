//! Benchmark suite for invariants.

use std::collections::HashMap;
use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use li_core::ids::{IdentityId, ObservationId, VertexId};
use li_core::ontology::Vertex;
use li_core::relation::Relation;
use li_core::{Confidence, Modality, Observation, Timestamp};
use li_model::graph::KnowledgeGraph;
use li_model::invariants::{
    IdentityUniquenessInvariant, Invariant, ObservationPartitionInvariant,
};
use li_model::ontology::Edge;
use li_model::queries::{
    IdentitySetQuery, NeighborhoodQuery, SupportSetQuery,
};
use rand::TryRng;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;

pub struct MockGraph {
    pub identities: Vec<IdentityId>,
    pub support_sets: HashMap<u64, Vec<Observation<()>>>,
    pub edges: HashMap<u64, Vec<Edge>>,
}

impl KnowledgeGraph for MockGraph {
    type EventPayload = ();
    type ObservationPayload = ();
    type StatePayload = ();

    fn vertex_type(&self, _id: VertexId) -> Option<Vertex> {
        None
    }

    fn apply(
        &mut self,
        _op: li_model::operations::GraphOperation<(), (), ()>,
    ) {
        // Left blank intentionally for benchmarking.
    }
}

impl IdentitySetQuery for MockGraph {
    fn all_identities(&self) -> Vec<IdentityId> {
        self.identities.clone()
    }
}

impl SupportSetQuery for MockGraph {
    fn query_support_set(&self, id: IdentityId) -> Vec<&Observation<()>> {
        self.support_sets
            .get(&id.0)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }
}

impl NeighborhoodQuery for MockGraph {
    type EdgeRef<'a> = &'a Edge;

    fn out_edges(&self, id: VertexId) -> Vec<&Edge> {
        self.edges
            .get(&id.0)
            .map(|e| e.iter().collect())
            .unwrap_or_default()
    }
}

fn generate_valid_bench_graph(
    num_identities: u64,
    obs_per_identity: u64,
    noise_edges: u64,
) -> MockGraph {
    let mut identities = Vec::with_capacity(num_identities as usize);
    let mut support_sets = HashMap::with_capacity(num_identities as usize);
    let mut edges: HashMap<u64, Vec<Edge>> = HashMap::with_capacity(
        (num_identities * obs_per_identity + noise_edges) as usize,
    );

    let mut rng = ChaCha8Rng::seed_from_u64(1337);
    let mut current_obs_id = num_identities + 1;

    for id_idx in 1..=num_identities {
        let identity_id = IdentityId(id_idx);
        identities.push(identity_id);

        let mut current_identity_support =
            Vec::with_capacity(obs_per_identity as usize);

        for _ in 0..obs_per_identity {
            let obs_raw_id = current_obs_id;
            current_obs_id += 1;

            current_identity_support.push(Observation {
                id: ObservationId(obs_raw_id),
                modality: Modality(1),
                timestamp: Timestamp(obs_raw_id.try_into().unwrap()),
                confidence: Confidence(0.7),
                payload: (),
            });

            let support_edge = Edge {
                relation: Relation::Supports,
                target: VertexId(identity_id.0),
                source: VertexId(obs_raw_id),
                created_at: Timestamp(0),
            };

            edges.entry(obs_raw_id).or_default().push(support_edge);
        }

        support_sets.insert(identity_id.0, current_identity_support);
    }

    let total_vertices = current_obs_id;
    for _ in 0..noise_edges {
        let src = (rng.try_next_u64().unwrap() % num_identities) + 1;
        let tgt = (rng.try_next_u64().unwrap() % total_vertices) + 1;

        let noise_edge = Edge {
            relation: Relation::Supports,
            target: VertexId(tgt),
            source: VertexId(src),
            created_at: Timestamp(0),
        };

        edges.entry(src).or_default().push(noise_edge);
    }

    MockGraph {
        identities,
        support_sets,
        edges,
    }
}

fn bench_observation_partition(c: &mut Criterion) {
    let mut group = c.benchmark_group("Observation Partition Verification");

    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));

    for size in &[50, 250, 1000, 1_000_000] {
        let num_identities = *size;
        let obs_per_identity = 5;
        let noise_edges = num_identities * 15;

        let graph = generate_valid_bench_graph(
            num_identities,
            obs_per_identity,
            noise_edges,
        );
        let invariant = ObservationPartitionInvariant;

        group.bench_with_input(
            BenchmarkId::new("identities_count", num_identities),
            &graph,
            |b, g| {
                b.iter(|| {
                    black_box(invariant.verify(g));
                });
            },
        );
    }
    group.finish();
}

fn bench_identity_uniqueness(c: &mut Criterion) {
    let mut group = c.benchmark_group("Identity Uniqueness Verification");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));

    for size in &[50, 250, 1000, 1_000_000] {
        let num_identities = *size;
        let obs_per_identity = 5;
        let noise_edges = num_identities * 15;

        let graph = generate_valid_bench_graph(
            num_identities,
            obs_per_identity,
            noise_edges,
        );
        let invariant = IdentityUniquenessInvariant;

        group.bench_with_input(
            BenchmarkId::new("identities_count", num_identities),
            &graph,
            |b, g| {
                b.iter(|| {
                    black_box(invariant.verify(g));
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_observation_partition,
    bench_identity_uniqueness
);
criterion_main!(benches);
