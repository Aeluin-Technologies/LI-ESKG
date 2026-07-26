//! Benchmark suite for LI-ESKG graph invariant verification.

use std::hint::black_box;
use std::time::Duration;

use criterion::{
    BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use li_core::ids::{EventId, IdentityId, ObservationId, StateId};
use li_core::observation::{Modality, Observation, Timestamp};
use li_core::ontology::Vertex;
use li_core::probability::Confidence;
use li_core::relation::Relation;
use li_model::graph::{KnowledgeGraph, PetGraphStore};
use li_model::invariants::{
    CausalAcyclicityInvariant, IdentityUniquenessInvariant, Invariant,
    ObservationPartitionInvariant,
};
use li_model::operations::GraphOperation;

/// Generates a synthetic knowledge graph containing `num_entities` nodes
/// and `obs_per_identity` observations per identity in a single pass.
fn generate_bench_graph(
    num_entities: u64,
    obs_per_identity: u64,
) -> PetGraphStore<(), (), ()> {
    let mut store = PetGraphStore::new();

    let ops_per_entity = 3 + 1 + (obs_per_identity * 2);
    let total_ops = (num_entities * ops_per_entity) as usize;
    let mut ops = Vec::with_capacity(total_ops);

    let mut obs_counter = num_entities + 1;

    for idx in 1..=num_entities {
        let identity_id = IdentityId(idx);
        let event_id = EventId(idx);
        let state_id = StateId(idx);

        ops.push(GraphOperation::CommitIdentity {
            id: identity_id,
            created_at: Timestamp::default(),
        });
        ops.push(GraphOperation::CommitEvent {
            id: event_id,
            timestamp: Timestamp::from_millis(
                idx.try_into().unwrap_or(i64::MAX),
            ),
            payload: (),
        });
        ops.push(GraphOperation::CommitState {
            id: state_id,
            timestamp: Timestamp::from_millis(
                idx.try_into().unwrap_or(i64::MAX),
            ),
            payload: (),
        });

        if idx > 1 {
            ops.push(GraphOperation::CommitRelation {
                source: Vertex::Event(EventId(idx - 1)),
                relation: Relation::Influence,
                target: Vertex::Event(event_id),
                created_at: Timestamp::default(),
            });
        }

        for _ in 0..obs_per_identity {
            let obs_id = ObservationId(obs_counter);
            obs_counter += 1;

            ops.push(GraphOperation::CommitObservation(Observation {
                id: obs_id,
                modality: Modality(1),
                timestamp: Timestamp::from_millis(
                    obs_id.0.try_into().unwrap_or(i64::MAX),
                ),
                confidence: Confidence::new(0.7),
                payload: (),
            }));

            ops.push(GraphOperation::CommitRelation {
                source: Vertex::Observation(obs_id),
                relation: Relation::Supports,
                target: Vertex::Identity(identity_id),
                created_at: Timestamp::default(),
            });
        }
    }

    store
        .apply_batch(ops)
        .expect("Failed to initialize benchmark graph store");
    store
}

fn bench_observation_partition(c: &mut Criterion) {
    let mut group = c.benchmark_group("Observation Partition Verification");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(6));
    group.sample_size(10);

    for &size in &[10_000, 50_000, 100_000] {
        let graph = generate_bench_graph(size, 5);
        let invariant = ObservationPartitionInvariant;

        group.throughput(Throughput::Elements(size));
        group.bench_with_input(
            BenchmarkId::new("observation", size),
            &graph,
            |b, g| {
                b.iter(|| black_box(invariant.verify(g)));
            },
        );
    }
    group.finish();
}

fn bench_identity_uniqueness(c: &mut Criterion) {
    let mut group = c.benchmark_group("Identity Uniqueness Verification");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(6));
    group.sample_size(10);

    for &size in &[10_000, 50_000, 100_000] {
        let graph = generate_bench_graph(size, 5);
        let invariant = IdentityUniquenessInvariant;

        group.throughput(Throughput::Elements(size));
        group.bench_with_input(
            BenchmarkId::new("uniqueness", size),
            &graph,
            |b, g| {
                b.iter(|| black_box(invariant.verify(g)));
            },
        );
    }
    group.finish();
}

fn bench_causal_acyclicity(c: &mut Criterion) {
    let mut group = c.benchmark_group("Causal Acyclicity Verification");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(10);

    for &size in &[10_000, 50_000, 100_000] {
        let graph = generate_bench_graph(size, 5);
        let invariant = CausalAcyclicityInvariant;

        group.throughput(Throughput::Elements(size));
        group.bench_with_input(
            BenchmarkId::new("acyclicit", size),
            &graph,
            |b, g| {
                b.iter(|| black_box(invariant.verify(g)));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_observation_partition,
    bench_identity_uniqueness,
    bench_causal_acyclicity
);
criterion_main!(benches);
