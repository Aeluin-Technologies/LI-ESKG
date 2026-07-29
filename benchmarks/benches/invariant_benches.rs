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

/// Converts a benchmark identifier to a saturating timestamp scalar.
fn timestamp_value(value: u64) -> i64 {
    value.min(i64::MAX as u64) as i64
}

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
            timestamp: Timestamp::from_millis(timestamp_value(idx)),
            payload: (),
        });
        ops.push(GraphOperation::CommitState {
            id: state_id,
            timestamp: Timestamp::from_millis(timestamp_value(idx)),
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
                timestamp: Timestamp::from_millis(timestamp_value(obs_id.0)),
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

    let initialization = store.apply_batch(ops);
    assert!(
        initialization.is_ok(),
        "benchmark graph initialization failed: {initialization:?}"
    );
    store
}

fn bench_observation_partition(c: &mut Criterion) {
    let mut group = c.benchmark_group("Observation Partition Verification");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(6));
    group.sample_size(10);

    const OBSERVATIONS_PER_IDENTITY: u64 = 5;
    for &size in &[10_000, 50_000, 100_000] {
        let graph = generate_bench_graph(size, OBSERVATIONS_PER_IDENTITY);
        let invariant = ObservationPartitionInvariant;
        let observation_count = size.saturating_mul(OBSERVATIONS_PER_IDENTITY);

        group.throughput(Throughput::Elements(observation_count));
        group.bench_with_input(
            BenchmarkId::new("observations", observation_count),
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

    const OBSERVATIONS_PER_IDENTITY: u64 = 5;
    for &size in &[10_000, 50_000, 100_000] {
        let graph = generate_bench_graph(size, OBSERVATIONS_PER_IDENTITY);
        let invariant = IdentityUniquenessInvariant;
        let node_count = size.saturating_mul(3 + OBSERVATIONS_PER_IDENTITY);

        group.throughput(Throughput::Elements(node_count));
        group.bench_with_input(
            BenchmarkId::new("nodes", node_count),
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

    const OBSERVATIONS_PER_IDENTITY: u64 = 5;
    for &size in &[10_000, 50_000, 100_000] {
        let graph = generate_bench_graph(size, OBSERVATIONS_PER_IDENTITY);
        let invariant = CausalAcyclicityInvariant;
        let support_relations = size.saturating_mul(OBSERVATIONS_PER_IDENTITY);
        let causal_relations = size.saturating_sub(1);
        let relation_count =
            support_relations.saturating_add(causal_relations);

        group.throughput(Throughput::Elements(relation_count));
        group.bench_with_input(
            BenchmarkId::new("relations", relation_count),
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
