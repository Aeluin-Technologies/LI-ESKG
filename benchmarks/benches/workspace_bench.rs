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
use li_workspace::{
    ActiveWorkspace, InMemoryWorkspace, SpatialGridConfig, SpatialGridIndex,
    SpatialIndexError, SpatialMatch, SpatialPoint,
};
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
    let mut workspace = InMemoryWorkspace::with_capacity(count as usize);
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

fn setup_eviction_workspace(
    count: u64,
    expired_numerator: u64,
    expired_denominator: u64,
) -> InMemoryWorkspace<TrackingSummary> {
    let mut workspace = InMemoryWorkspace::with_capacity(count as usize);
    let expired_count =
        count.saturating_mul(expired_numerator) / expired_denominator;
    for id in 1..=count {
        let last_update = if id <= expired_count { 100 } else { 1_000_000 };
        workspace.insert(BeliefState {
            identity: IdentityId(id),
            summary: TrackingSummary::default(),
            posterior: Probability::new(0.8),
            last_update: Timestamp::from_millis(last_update),
        });
    }
    workspace
}

/// Builds a fixed-density spatial pool and an interior query point.
fn setup_spatial_index(
    count: usize,
) -> Result<(SpatialGridIndex, SpatialPoint), SpatialIndexError> {
    let config = SpatialGridConfig::try_new(8.0, 4.0, 32)?;
    let mut index = SpatialGridIndex::with_capacity(config, count);
    let width = (count as f64).sqrt().ceil() as usize;

    for offset in 0..count {
        let x = (offset % width) as f64;
        let y = (offset / width) as f64;
        index.insert(
            IdentityId(offset as u64 + 1),
            SpatialPoint::try_new(x, y)?,
        )?;
    }

    let center = width as f64 / 2.0 + 0.25;
    Ok((index, SpatialPoint::try_new(center, center)?))
}

fn bench_workspace_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("InMemoryWorkspace::insert");

    // Use longer sampling to stabilize large-workspace measurements.
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
                // Batch inserts amortize setup cloning across 1,000 elements.
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

    // Fast lookups need additional samples for a stable estimate.
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

fn bench_workspace_indexed_candidate_lookup(c: &mut Criterion) {
    const CANDIDATE_COUNT: usize = 32;

    let mut group =
        c.benchmark_group("InMemoryWorkspace::indexed_candidate_lookup");
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(30);
    group.throughput(Throughput::Elements(CANDIDATE_COUNT as u64));

    for size in &[10_000, 100_000, 1_000_000] {
        let size = *size;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let workspace = setup_workspace(size as u64, 1_000_000, &mut rng);
        let stride = (size as u64 / CANDIDATE_COUNT as u64).max(1);
        let candidates: [IdentityId; CANDIDATE_COUNT] =
            core::array::from_fn(|index| {
                IdentityId(1 + stride * index as u64)
            });

        group.bench_with_input(
            BenchmarkId::new("fixed_32_candidates_in_pool", size),
            &size,
            |b, _| {
                b.iter(|| {
                    for identity in &candidates {
                        black_box(workspace.get(black_box(*identity)));
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_spatial_candidate_lookup(c: &mut Criterion) {
    const MAX_CANDIDATES: usize = 32;

    let mut group = c.benchmark_group("SpatialGridIndex::query_into");
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(30);
    group.throughput(Throughput::Elements(1));

    for size in &[10_000, 100_000, 1_000_000] {
        let size = *size;
        let setup = setup_spatial_index(size);
        let Ok((index, query_point)) = setup else {
            continue;
        };
        let mut matches: Vec<SpatialMatch> =
            Vec::with_capacity(MAX_CANDIDATES);

        group.bench_with_input(
            BenchmarkId::new("fixed_density_top_32", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = index.query_into(
                        black_box(query_point),
                        black_box(&mut matches),
                    );
                    debug_assert!(result.is_ok());
                    debug_assert!(matches.len() <= MAX_CANDIDATES);
                    black_box(matches.as_slice());
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

    // Eviction requires a fresh workspace for every sample.
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10);

    for size in &[100_000, 500_000, 1_000_000] {
        let size_u64 = *size as u64;
        let templates = [
            (
                "evict_ratio_0_percent",
                setup_eviction_workspace(size_u64, 0, 1),
            ),
            (
                "evict_ratio_50_percent",
                setup_eviction_workspace(size_u64, 1, 2),
            ),
            (
                "evict_ratio_100_percent",
                setup_eviction_workspace(size_u64, 1, 1),
            ),
        ];

        for (label, template) in &templates {
            group.throughput(Throughput::Elements(size_u64));
            group.bench_with_input(
                BenchmarkId::new(*label, size_u64),
                &size_u64,
                |b, _| {
                    b.iter_batched(
                        || template.clone(),
                        |mut ws| {
                            let evicted = ws.evict_expired(
                                Timestamp::from_millis(1_000_000),
                                500_000,
                            );
                            drop(black_box(evicted));
                            drop(black_box(ws));
                        },
                        BatchSize::LargeInput,
                    );
                },
            );
        }

        let refill_template = setup_eviction_workspace(size_u64, 1, 2);
        let mut eviction_buffer = Vec::with_capacity((size_u64 / 2) as usize);
        group.throughput(Throughput::Elements(size_u64 + size_u64 / 2));
        group.bench_with_input(
            BenchmarkId::new("evict_and_refill_50_percent", size_u64),
            &size_u64,
            |b, _| {
                b.iter_batched(
                    || refill_template.clone(),
                    |mut ws| {
                        ws.evict_expired_into(
                            Timestamp::from_millis(1_000_000),
                            500_000,
                            &mut eviction_buffer,
                        );
                        for belief in eviction_buffer.drain(..) {
                            ws.insert(belief);
                        }
                        black_box(eviction_buffer.capacity());
                        drop(black_box(ws));
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
    bench_workspace_indexed_candidate_lookup,
    bench_spatial_candidate_lookup,
    bench_workspace_active_beliefs,
    bench_workspace_eviction,
    bench_workspace_snapshot
);
criterion_main!(benches);
