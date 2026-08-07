//! Benchmark suite for `InMemoryWorkspace` operations.

use std::hint::black_box;
use std::time::Duration;

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, criterion_group,
    criterion_main,
};
use li_core::{
    CommitVersion, IdentityId, IdentityReference, ObservationId, Timestamp,
};
use li_factors::CandidateBuffer;
use li_workspace::{
    CommittedAssociation, HotIdentity, HotWorkspace, PublishedSnapshot,
    WorkerScratch,
};

/// Builds a pre-reserved V2 workspace with deterministic activation times.
fn setup_workspace(count: u64, base_timestamp: i64) -> HotWorkspace {
    let mut workspace = HotWorkspace::with_capacity(count as usize);
    for id in 1..=count {
        let offset = i64::try_from(id % 2_000).unwrap_or(i64::MAX) - 1_000;
        workspace.insert(HotIdentity::new(
            IdentityId(id),
            0,
            4,
            Timestamp::from_micros(base_timestamp.saturating_add(offset)),
            CommitVersion::ZERO,
        ));
    }
    workspace
}

/// Builds a workspace with a controlled fraction older than the cutoff.
fn setup_eviction_workspace(
    count: u64,
    expired_numerator: u64,
    expired_denominator: u64,
) -> HotWorkspace {
    let mut workspace = HotWorkspace::with_capacity(count as usize);
    let expired_count =
        count.saturating_mul(expired_numerator) / expired_denominator;
    for id in 1..=count {
        let activation = if id <= expired_count { 100 } else { 1_000_000 };
        workspace.insert(HotIdentity::new(
            IdentityId(id),
            0,
            4,
            Timestamp::from_micros(activation),
            CommitVersion::ZERO,
        ));
    }
    workspace
}

fn bench_workspace_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("InMemoryWorkspace::insert");
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);
    const BATCH_SIZE: u64 = 1_000;

    for size in [100_000_usize, 1_000_000] {
        group.throughput(Throughput::Elements(BATCH_SIZE));
        group.bench_with_input(
            BenchmarkId::new("insert_batch_1000", size),
            &size,
            |b, &pool_size| {
                b.iter_batched(
                    || setup_workspace(pool_size as u64, 1_000_000),
                    |mut workspace| {
                        for offset in 0..BATCH_SIZE {
                            workspace.insert(HotIdentity::new(
                                IdentityId(pool_size as u64 + offset + 10_000),
                                0,
                                4,
                                Timestamp::from_micros(1_000_100),
                                CommitVersion::ZERO,
                            ));
                        }
                        black_box(workspace)
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
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    for size in [100_000_usize, 1_000_000] {
        let workspace = setup_workspace(size as u64, 1_000_000);
        let target = IdentityId((size / 2) as u64);
        group.bench_with_input(
            BenchmarkId::new("hit_middle_key", size),
            &size,
            |b, _| b.iter(|| black_box(workspace.get(black_box(target)))),
        );
        let missing = IdentityId((size + 9_999) as u64);
        group.bench_with_input(
            BenchmarkId::new("miss_nonexistent_key", size),
            &size,
            |b, _| b.iter(|| black_box(workspace.get(black_box(missing)))),
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

    for size in [10_000_usize, 100_000, 1_000_000] {
        let workspace = setup_workspace(size as u64, 1_000_000);
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

fn bench_candidate_canonicalization(c: &mut Criterion) {
    const MAX_CANDIDATES: usize = 32;
    let mut group = c.benchmark_group("V2::CandidateBuffer::canonicalize");
    group.throughput(Throughput::Elements(MAX_CANDIDATES as u64));

    for pool_size in [10_000_u64, 100_000, 1_000_000] {
        let stride = (pool_size / MAX_CANDIDATES as u64).max(1);
        let candidates: Vec<_> = (0..MAX_CANDIDATES)
            .rev()
            .map(|index| {
                IdentityReference::Latent(IdentityId(
                    1 + stride * index as u64,
                ))
            })
            .collect();
        let mut output = CandidateBuffer::with_capacity(1, MAX_CANDIDATES);
        group.bench_with_input(
            BenchmarkId::new("fixed_32_candidates_in_pool", pool_size),
            &pool_size,
            |b, _| {
                b.iter(|| {
                    output.reset(1);
                    black_box(output.push_observation(
                        0,
                        1,
                        candidates.iter().cloned(),
                    ))
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

    for size in [100_000_usize, 1_000_000] {
        let workspace = setup_workspace(size as u64, 1_000_000);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, _| {
                b.iter(|| {
                    let checksum =
                        workspace.values().fold(0_u64, |acc, state| {
                            acc.wrapping_add(state.identity().0)
                        });
                    black_box(checksum)
                });
            },
        );
    }
    group.finish();
}

fn bench_workspace_eviction(c: &mut Criterion) {
    let mut group = c.benchmark_group("InMemoryWorkspace::evict_expired");
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10);

    for size in [100_000_u64, 500_000, 1_000_000] {
        for (label, numerator) in [
            ("evict_ratio_0_percent", 0_u64),
            ("evict_ratio_50_percent", 1),
            ("evict_ratio_100_percent", 2),
        ] {
            let mut evicted = Vec::with_capacity(size as usize);
            group.throughput(Throughput::Elements(size));
            group.bench_with_input(
                BenchmarkId::new(label, size),
                &size,
                |b, &count| {
                    b.iter_batched(
                        || setup_eviction_workspace(count, numerator, 2),
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

        let mut evicted = Vec::with_capacity((size / 2) as usize);
        group.throughput(Throughput::Elements(size + size / 2));
        group.bench_with_input(
            BenchmarkId::new("evict_and_refill_50_percent", size),
            &size,
            |b, &count| {
                b.iter_batched(
                    || setup_eviction_workspace(count, 1, 2),
                    |mut workspace| {
                        workspace.evict_before(
                            Timestamp::from_micros(500_000),
                            &mut evicted,
                        );
                        for identity in evicted.drain(..) {
                            workspace.insert(identity);
                        }
                        black_box(workspace)
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

    for size in [100_000_usize, 1_000_000] {
        let identities: Vec<_> = (1..=size as u64).map(IdentityId).collect();
        let snapshot = PublishedSnapshot::new(identities);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, _| {
                b.iter(|| black_box(snapshot.load()));
            },
        );
    }
    group.finish();
}

fn bench_v2_committed_update_and_scratch_reuse(c: &mut Criterion) {
    c.bench_function("V2::HotWorkspace::apply_committed", |b| {
        b.iter_batched(
            || setup_workspace(1, 0),
            |mut workspace| {
                black_box(workspace.apply_committed(CommittedAssociation {
                    observation: ObservationId(1),
                    identity: IdentityId(1),
                    version: CommitVersion::new(1),
                    event_time: Timestamp::from_micros(1),
                }))
            },
            BatchSize::SmallInput,
        );
    });

    let mut scratch = WorkerScratch::with_capacity(256, 2_048, 512, 8_192);
    c.bench_function("V2::WorkerScratch::reset", |b| {
        b.iter(|| {
            scratch.reset(black_box(256));
            black_box(scratch.observations.capacity())
        });
    });
}

criterion_group!(
    benches,
    bench_workspace_insert,
    bench_workspace_lookup,
    bench_workspace_indexed_candidate_lookup,
    bench_candidate_canonicalization,
    bench_workspace_active_beliefs,
    bench_workspace_eviction,
    bench_workspace_snapshot,
    bench_v2_committed_update_and_scratch_reuse
);
criterion_main!(benches);
