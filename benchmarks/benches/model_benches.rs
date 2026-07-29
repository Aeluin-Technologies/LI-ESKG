//! Benchmarks for allocation-bounded core and transactional graph operations.

use std::hint::black_box;

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, criterion_group,
    criterion_main,
};
use li_core::belief::BoundedHistory;
use li_core::ids::{EventId, IdentityId, ObservationId};
use li_core::observation::{Modality, Observation, Timestamp};
use li_core::ontology::Vertex;
use li_core::probability::Confidence;
use li_core::relation::Relation;
use li_model::graph::{KnowledgeGraph, PetGraphStore};
use li_model::operations::{GraphOperation, IdentityAssignment};

type BenchStore = PetGraphStore<[f32; 4], (), ()>;

const TARGET_IDENTITY: IdentityId = IdentityId(1);
const DUPLICATE_IDENTITY: IdentityId = IdentityId(2);

/// Creates a fixed-size observation payload for graph materialization.
fn observation(id: u64) -> Observation<[f32; 4]> {
    Observation::new(
        ObservationId(id),
        Modality(1),
        Timestamp::from_micros(id as i64),
        Confidence::new(0.95),
        [id as f32, 1.0, 2.0, 3.0],
    )
}

/// Creates a pre-reserved graph containing one active identity.
fn store_with_identity(
    identity: IdentityId,
    node_capacity: usize,
    edge_capacity: usize,
) -> BenchStore {
    let mut store = PetGraphStore::with_capacity(node_capacity, edge_capacity);
    let result = store.apply_batch([GraphOperation::CommitIdentity {
        id: identity,
        created_at: Timestamp::UNIX_EPOCH,
    }]);
    assert!(
        result.is_ok(),
        "benchmark identity setup failed: {result:?}"
    );
    store
}

/// Builds a pre-reserved identity support star of the requested degree.
fn support_store(degree: usize, identity: IdentityId) -> BenchStore {
    let mut store =
        store_with_identity(identity, degree.saturating_add(1), degree);
    for offset in 0..degree {
        let id = (offset as u64).saturating_add(1);
        let result = store.materialize_observation(
            observation(id),
            IdentityAssignment::Existing(identity),
        );
        assert!(result.is_ok(), "benchmark support setup failed: {result:?}");
    }
    store
}

/// Builds the target and duplicate identities used by merge benchmarks.
fn merge_store(degree: usize) -> BenchStore {
    let mut store =
        PetGraphStore::with_capacity(degree.saturating_add(2), degree);
    let identities = [
        GraphOperation::CommitIdentity {
            id: TARGET_IDENTITY,
            created_at: Timestamp::UNIX_EPOCH,
        },
        GraphOperation::CommitIdentity {
            id: DUPLICATE_IDENTITY,
            created_at: Timestamp::UNIX_EPOCH,
        },
    ];
    let result = store.apply_batch(identities);
    assert!(result.is_ok(), "benchmark merge setup failed: {result:?}");

    for offset in 0..degree {
        let id = (offset as u64).saturating_add(1);
        let result = store.materialize_observation(
            observation(id),
            IdentityAssignment::Existing(DUPLICATE_IDENTITY),
        );
        assert!(
            result.is_ok(),
            "benchmark merge incidence setup failed: {result:?}"
        );
    }
    store
}

/// Creates a batch whose final relation fails after nodes and an edge commit.
fn late_failure_batch() -> [GraphOperation<[f32; 4], (), ()>; 5] {
    let timestamp = Timestamp::from_micros(1);
    [
        GraphOperation::CommitObservation(observation(1)),
        GraphOperation::CommitEvent {
            id: EventId(1),
            timestamp,
            payload: (),
        },
        GraphOperation::CommitEvent {
            id: EventId(2),
            timestamp,
            payload: (),
        },
        GraphOperation::CommitRelation {
            source: Vertex::Event(EventId(1)),
            relation: Relation::Influence,
            target: Vertex::Event(EventId(2)),
            created_at: timestamp,
        },
        GraphOperation::CommitRelation {
            source: Vertex::Event(EventId(2)),
            relation: Relation::Influence,
            target: Vertex::Event(EventId(3)),
            created_at: timestamp,
        },
    ]
}

/// Measures full-ring appends after the one-time allocation has completed.
fn bench_bounded_history_append(c: &mut Criterion) {
    let mut group = c.benchmark_group("BoundedHistory::steady_state_push");
    group.throughput(Throughput::Elements(1));

    for capacity in [1_usize, 8, 64, 256, 1_024] {
        let mut history = BoundedHistory::new(capacity);
        for value in 0..capacity {
            let _evicted = history.push(value as u64);
        }
        let mut value = capacity as u64;

        group.bench_with_input(
            BenchmarkId::from_parameter(capacity),
            &capacity,
            |bencher, _| {
                bencher.iter(|| {
                    value = value.wrapping_add(1);
                    black_box(history.push(black_box(value)))
                });
            },
        );
    }
    group.finish();
}

/// Measures the atomic new-identity and existing-identity commit paths.
fn bench_observation_materialization(c: &mut Criterion) {
    let mut group =
        c.benchmark_group("PetGraphStore::materialize_observation");
    group.throughput(Throughput::Elements(1));

    group.bench_function("new_identity_pre_reserved", |bencher| {
        bencher.iter_batched_ref(
            || BenchStore::with_capacity(2, 1),
            |store| {
                let result = store.materialize_observation(
                    observation(1),
                    IdentityAssignment::New(TARGET_IDENTITY),
                );
                debug_assert!(result.is_ok());
                black_box(result)
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("existing_identity_pre_reserved", |bencher| {
        bencher.iter_batched_ref(
            || store_with_identity(TARGET_IDENTITY, 2, 1),
            |store| {
                let result = store.materialize_observation(
                    observation(1),
                    IdentityAssignment::Existing(TARGET_IDENTITY),
                );
                debug_assert!(result.is_ok());
                black_box(result)
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// Measures reference-only traversal of observation support stars.
fn bench_zero_copy_support_iteration(c: &mut Criterion) {
    let mut group =
        c.benchmark_group("PetGraphStore::supporting_observations");

    for degree in [1_usize, 8, 64, 512, 4_096] {
        let store = support_store(degree, TARGET_IDENTITY);
        group.throughput(Throughput::Elements(degree as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(degree),
            &degree,
            |bencher, _| {
                bencher.iter(|| {
                    let support =
                        store.supporting_observations(TARGET_IDENTITY);
                    debug_assert!(support.is_ok());
                    let checksum = support.map_or(0_u64, |observations| {
                        observations.fold(0_u64, |accumulator, item| {
                            accumulator.wrapping_add(item.id.0)
                        })
                    });
                    black_box(checksum)
                });
            },
        );
    }
    group.finish();
}

/// Measures identity canonicalization across representative incident degrees.
fn bench_identity_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("PetGraphStore::merge_identities");

    for degree in [1_usize, 8, 64, 512] {
        let template = merge_store(degree);
        group.throughput(Throughput::Elements(degree as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(degree),
            &degree,
            |bencher, _| {
                bencher.iter_batched_ref(
                    || template.clone(),
                    |store| {
                        let result = store.merge_identities(
                            TARGET_IDENTITY,
                            DUPLICATE_IDENTITY,
                        );
                        debug_assert!(result.is_ok());
                        black_box(result)
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

/// Measures rollback when the last operation in a transaction fails.
fn bench_atomic_late_failure_rollback(c: &mut Criterion) {
    let mut group = c.benchmark_group("PetGraphStore::late_failure_rollback");
    group.throughput(Throughput::Elements(5));

    let mut store = BenchStore::with_capacity(3, 2);
    group.bench_function("node_and_edge_journal", |bencher| {
        bencher.iter(|| {
            let result = store.apply_batch(late_failure_batch());
            debug_assert!(result.is_err());
            debug_assert_eq!(store.node_count(), 0);
            debug_assert_eq!(store.edge_count(), 0);
            black_box(result)
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_bounded_history_append,
    bench_observation_materialization,
    bench_zero_copy_support_iteration,
    bench_identity_merge,
    bench_atomic_late_failure_rollback
);
criterion_main!(benches);
