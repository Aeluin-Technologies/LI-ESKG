//! Benchmark suite for `InMemoryWorkspace` operations.

use std::hint::black_box;
use std::time::Duration;

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
            last_update: Timestamp::from_millis(base_timestamp + ts_offset),
        });
    }
    workspace
}

fn bench_workspace_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("InMemoryWorkspace::insert");

    // Plus de temps et d'échantillons pour garantir la stabilité
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);

    const BATCH_SIZE: u64 = 1_000;

    for size in &[100_000, 1_000_000] {
        let size = *size;
        group.throughput(Throughput::Elements(BATCH_SIZE));

        group.bench_with_input(
            BenchmarkId::new("insert_batch_1000", size),
            &size,
            |b, _| {
                // Utilisation d'un lot de 1 000 éléments :
                // On amortit le clone de setup sur 1000 insertions
                b.iter_batched(
                    || {
                        let mut rng = ChaCha8Rng::seed_from_u64(1337);
                        let ws =
                            setup_workspace(size as u64, 1_000_000, &mut rng);

                        let batch: Vec<_> = (0..BATCH_SIZE)
                            .map(|i| BeliefState {
                                identity: IdentityId(
                                    (size as u64) + i + 10_000,
                                ),
                                summary: TrackingSummary::default(),
                                posterior: Probability::new(0.99),
                                last_update: Timestamp::from_millis(1_000_100),
                            })
                            .collect();

                        (ws, batch)
                    },
                    |(mut ws, batch)| {
                        for belief in batch {
                            ws.insert(belief);
                        }
                        black_box(ws);
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_workspace_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("InMemoryWorkspace::get");

    // Les lookups sont en nanosecondes : 50 échantillons apportent une
    // excellente métrique
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    for size in &[100_000, 1_000_000] {
        let size = *size;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let workspace = setup_workspace(size as u64, 1_000_000, &mut rng);

        let target_id = IdentityId((size / 2) as u64);

        group.bench_with_input(
            BenchmarkId::new("hit_middle_key", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(workspace.get(target_id));
                });
            },
        );

        let missing_id = IdentityId((size + 9999) as u64);
        group.bench_with_input(
            BenchmarkId::new("miss_nonexistent_key", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(workspace.get(missing_id));
                });
            },
        );
    }
    group.finish();
}

fn bench_workspace_active_beliefs(c: &mut Criterion) {
    let mut group = c.benchmark_group("InMemoryWorkspace::active_beliefs");

    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(12));
    group.sample_size(30);

    for size in &[100_000, 1_000_000] {
        let size = *size;
        let mut rng = ChaCha8Rng::seed_from_u64(1337);
        let workspace = setup_workspace(size as u64, 1_000_000, &mut rng);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(workspace.active_beliefs());
                });
            },
        );
    }
    group.finish();
}

fn bench_workspace_eviction(c: &mut Criterion) {
    let mut group = c.benchmark_group("InMemoryWorkspace::evict_expired");

    // L'éviction nécessite une copie fraîche du workspace par échantillon :
    // On passe à 20s de temps de mesure pour laisser le temps nécessaire sur
    // 1M d'éléments
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(15);

    for size in &[100_000, 1_000_000] {
        let size_u64 = *size as u64;

        let mut template_10 = InMemoryWorkspace::new();
        for i in 1..=size_u64 {
            let last_update = if i <= size_u64 / 10 { 100 } else { 1_000_000 };
            template_10.insert(BeliefState {
                identity: IdentityId(i),
                summary: TrackingSummary::default(),
                posterior: Probability::new(0.8),
                last_update: Timestamp::from_millis(last_update),
            });
        }

        group.bench_with_input(
            BenchmarkId::new("evict_ratio_10_percent", size_u64),
            &size_u64,
            |b, _| {
                b.iter_batched(
                    || template_10.clone(),
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

        let mut template_50 = InMemoryWorkspace::new();
        for i in 1..=size_u64 {
            let last_update = if i % 2 == 0 { 100 } else { 1_000_000 };
            template_50.insert(BeliefState {
                identity: IdentityId(i),
                summary: TrackingSummary::default(),
                posterior: Probability::new(0.8),
                last_update: Timestamp::from_millis(last_update),
            });
        }

        group.bench_with_input(
            BenchmarkId::new("evict_ratio_50_percent", size_u64),
            &size_u64,
            |b, _| {
                b.iter_batched(
                    || template_50.clone(),
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

fn bench_workspace_snapshot(c: &mut Criterion) {
    let mut group = c.benchmark_group("InMemoryWorkspace::create_snapshot");

    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(12));
    group.sample_size(30);

    for size in &[100_000, 1_000_000] {
        let size = *size;
        let mut rng = ChaCha8Rng::seed_from_u64(1337);
        let workspace = setup_workspace(size as u64, 1_000_000, &mut rng);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(workspace.create_snapshot(black_box(
                        Timestamp::from_millis(1_000_000),
                    )));
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
