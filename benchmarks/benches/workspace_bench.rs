//! Benchmark suite for `InMemoryWorkspace` operations using Criterion.

use std::hint::black_box;

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, criterion_group,
    criterion_main,
};
use li_core::belief::BeliefState;
use li_core::ids::IdentityId;
use li_core::observation::Timestamp;
use li_core::probability::Probability;
use li_workspace::{ActiveWorkspace, InMemoryWorkspace};
use rand::RngExt;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;

/// Represents $b_i = (\theta_i, \Sigma_i)$ from Theorem 7 of the LI-ESKG
/// paper. Stores continuous state embedding (mean vector) and spatial motion
/// dynamics (covariance).
#[derive(Debug, Clone, PartialEq)]
pub struct TrackingSummary {
    pub mean: [f64; 4],
    pub covariance: [f64; 16],
}

impl Default for TrackingSummary {
    fn default() -> Self {
        Self {
            mean: [0.0, 0.0, 1.0, 1.0],
            covariance: [0.1; 16],
        }
    }
}

fn setup_workspace(
    count: u64,
    base_timestamp: i64,
    rng: &mut ChaCha8Rng,
) -> InMemoryWorkspace<TrackingSummary> {
    let mut workspace = InMemoryWorkspace::new();
    for i in 1..=count {
        let ts_offset = rng.random_range(-1000..1000);
        workspace.insert(BeliefState {
            identity: IdentityId(i),
            summary: TrackingSummary::default(),
            posterior: Probability::new(0.95),
            last_update: Timestamp(base_timestamp + ts_offset),
        });
    }
    workspace
}

fn bench_workspace_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("InMemoryWorkspace::insert");

    for size in &[100, 1_000, 10_000, 100_000] {
        let size = *size;
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::new("insert_existing_tree", size),
            &size,
            |b, &size| {
                let mut rng = ChaCha8Rng::seed_from_u64(1337);
                let workspace =
                    setup_workspace(size as u64, 1_000_000, &mut rng);

                let new_belief = BeliefState {
                    identity: IdentityId((size / 2) as u64),
                    summary: TrackingSummary::default(),
                    posterior: Probability::new(0.99),
                    last_update: Timestamp(1_000_100),
                };

                b.iter_batched(
                    || (workspace.clone(), new_belief.clone()),
                    |(mut ws, belief)| {
                        ws.insert(belief);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_workspace_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("InMemoryWorkspace::get");

    for size in &[100, 1_000, 10_000, 100_000] {
        let size = *size;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let workspace = setup_workspace(size as u64, 1_000_000, &mut rng);

        let target_id = IdentityId((size / 2) as u64);

        group.bench_with_input(
            BenchmarkId::new("hit_middle_key", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let res = workspace.get(target_id);
                    black_box(res);
                });
            },
        );

        let missing_id = IdentityId((size + 9999) as u64);
        group.bench_with_input(
            BenchmarkId::new("miss_nonexistent_key", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let res = workspace.get(missing_id);
                    black_box(res);
                });
            },
        );
    }
    group.finish();
}

fn bench_workspace_active_beliefs(c: &mut Criterion) {
    let mut group = c.benchmark_group("InMemoryWorkspace::active_beliefs");

    for size in &[100, 1_000, 10_000, 100_000] {
        let size = *size;
        let mut rng = ChaCha8Rng::seed_from_u64(1337);
        let workspace = setup_workspace(size as u64, 1_000_000, &mut rng);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, _| {
                b.iter(|| {
                    let beliefs = workspace.active_beliefs();
                    black_box(beliefs);
                });
            },
        );
    }
    group.finish();
}

fn bench_workspace_eviction(c: &mut Criterion) {
    let mut group = c.benchmark_group("InMemoryWorkspace::evict_expired");

    for size in &[100, 1_000, 10_000, 50_000] {
        let size = *size as u64;

        group.bench_with_input(
            BenchmarkId::new("evict_ratio_10_percent", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let mut ws = InMemoryWorkspace::new();
                        for i in 1..=size {
                            let last_update =
                                if i <= size / 10 { 100 } else { 1_000_000 };
                            ws.insert(BeliefState {
                                identity: IdentityId(i),
                                summary: TrackingSummary::default(),
                                posterior: Probability::new(0.8),
                                last_update: Timestamp(last_update),
                            });
                        }
                        ws
                    },
                    |mut ws| {
                        let evicted =
                            ws.evict_expired(Timestamp(1_000_000), 500_000);
                        black_box(evicted);
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("evict_ratio_50_percent", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let mut ws = InMemoryWorkspace::new();
                        for i in 1..=size {
                            let last_update =
                                if i % 2 == 0 { 100 } else { 1_000_000 };
                            ws.insert(BeliefState {
                                identity: IdentityId(i),
                                summary: TrackingSummary::default(),
                                posterior: Probability::new(0.8),
                                last_update: Timestamp(last_update),
                            });
                        }
                        ws
                    },
                    |mut ws| {
                        let evicted =
                            ws.evict_expired(Timestamp(1_000_000), 500_000);
                        black_box(evicted);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_workspace_snapshot(c: &mut Criterion) {
    let mut group = c.benchmark_group("InMemoryWorkspace::create_snapshot");

    for size in &[100, 1_000, 10_000, 100_000] {
        let size = *size;
        let mut rng = ChaCha8Rng::seed_from_u64(1337);
        let workspace = setup_workspace(size as u64, 1_000_000, &mut rng);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, _| {
                b.iter(|| {
                    let snapshot =
                        workspace.create_snapshot(Timestamp(1_000_000));
                    black_box(snapshot);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_workspace_insert,
    bench_workspace_lookup,
    bench_workspace_active_beliefs,
    bench_workspace_eviction,
    bench_workspace_snapshot
);
criterion_main!(benches);
