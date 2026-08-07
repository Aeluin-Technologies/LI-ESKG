//! Benchmark suite for LI-ESKG graph invariant verification.

use std::hint::black_box;
use std::time::Duration;

use bytes::Bytes;
use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, criterion_group,
    criterion_main,
};
use li_core::{
    CommandEnvelope, CommitVersion, ContentHash, IdempotencyKey, IdentityId,
    Modality, ObservationEnvelope, ObservationId, PayloadRef, QualityMetadata,
    ResolutionCommand, SchemaId, SourceId, Timestamp, TransactionId,
};
use li_storage::MemoryLedger;

/// Creates deterministic immutable evidence for one benchmark observation.
fn observation(id: u64) -> Option<ObservationEnvelope> {
    let timestamp =
        Timestamp::from_micros(i64::try_from(id).unwrap_or(i64::MAX));
    ObservationEnvelope::new(
        ObservationId(id),
        SourceId(1),
        Modality(1),
        timestamp,
        timestamp,
        PayloadRef::Inline(Bytes::from_static(b"benchmark")),
        QualityMetadata::Opaque {
            schema: SchemaId(1),
            bytes: Bytes::new(),
        },
        ContentHash::new([1; 32]),
        None,
    )
    .ok()
}

/// Creates a non-zero stable replay key from a sequence number.
fn key(sequence: u64) -> Option<IdempotencyKey> {
    let mut bytes = [0_u8; 16];
    bytes[..8].copy_from_slice(&sequence.to_le_bytes());
    IdempotencyKey::new(bytes).ok()
}

/// Wraps commands in a versioned atomic envelope.
fn envelope(
    sequence: u64,
    version: CommitVersion,
    dependencies: Vec<TransactionId>,
    commands: Vec<ResolutionCommand>,
) -> Option<CommandEnvelope> {
    CommandEnvelope::new(
        TransactionId(sequence),
        version,
        key(sequence)?,
        Timestamp::from_micros(i64::try_from(sequence).unwrap_or(i64::MAX)),
        dependencies,
        commands,
    )
    .ok()
}

fn bench_observation_partition(c: &mut Criterion) {
    let mut group =
        c.benchmark_group("V2::ResolutionLedger::persist_observation_batch");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(6));
    group.sample_size(10);
    const OBSERVATIONS_PER_IDENTITY: u64 = 5;

    for size in [10_000_u64, 50_000, 100_000] {
        let observation_count = size.saturating_mul(OBSERVATIONS_PER_IDENTITY);
        let commands: Option<Vec<_>> = (1..=observation_count)
            .map(|id| {
                observation(id).map(ResolutionCommand::PersistObservation)
            })
            .collect();
        let Some(batch) = commands.and_then(|commands| {
            envelope(1, CommitVersion::ZERO, Vec::new(), commands)
        }) else {
            continue;
        };
        group.throughput(Throughput::Elements(observation_count));
        group.bench_with_input(
            BenchmarkId::new("observations", observation_count),
            &batch,
            |b, command| {
                b.iter_batched(
                    || command.clone(),
                    |command| {
                        let mut ledger = MemoryLedger::with_capacity(
                            observation_count as usize,
                            0,
                        );
                        black_box(ledger.commit(command))
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_identity_uniqueness(c: &mut Criterion) {
    let mut group =
        c.benchmark_group("V2::ResolutionLedger::create_identity_batch");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(6));
    group.sample_size(10);

    for size in [10_000_u64, 50_000, 100_000] {
        let commands = (1..=size)
            .map(|id| ResolutionCommand::CreateIdentity {
                identity: IdentityId(id),
                created_at: Timestamp::UNIX_EPOCH,
            })
            .collect();
        let Some(batch) =
            envelope(1, CommitVersion::ZERO, Vec::new(), commands)
        else {
            continue;
        };
        group.throughput(Throughput::Elements(size));
        group.bench_with_input(
            BenchmarkId::new("nodes", size),
            &batch,
            |b, command| {
                b.iter_batched(
                    || command.clone(),
                    |command| {
                        let mut ledger =
                            MemoryLedger::with_capacity(1, size as usize);
                        black_box(ledger.commit(command))
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

/// Builds an append-only transaction dependency chain of `size` records.
fn dependency_chain(size: u64) -> Option<MemoryLedger> {
    let mut ledger = MemoryLedger::with_capacity(size as usize, 0);
    for sequence in 1..=size {
        let dependencies = if sequence == 1 {
            Vec::new()
        } else {
            vec![TransactionId(sequence - 1)]
        };
        let command =
            ResolutionCommand::PersistObservation(observation(sequence)?);
        let batch =
            envelope(sequence, ledger.version(), dependencies, vec![command])?;
        ledger.commit(batch).ok()?;
    }
    Some(ledger)
}

fn bench_causal_acyclicity(c: &mut Criterion) {
    let mut group =
        c.benchmark_group("V2::ResolutionLedger::dependency_closure");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(10);

    for size in [10_000_u64, 50_000, 100_000] {
        let Some(ledger) = dependency_chain(size) else {
            continue;
        };
        let mut closure = Vec::with_capacity(size as usize);
        group.throughput(Throughput::Elements(size));
        group.bench_with_input(
            BenchmarkId::new("relations", size),
            &ledger,
            |b, ledger| {
                b.iter(|| {
                    ledger.dependency_closure(TransactionId(1), &mut closure);
                    black_box(closure.len())
                });
            },
        );
    }
    group.finish();
}

fn bench_v2_idempotent_replay(c: &mut Criterion) {
    let Some(command) = observation(1).and_then(|observation| {
        envelope(
            1,
            CommitVersion::ZERO,
            Vec::new(),
            vec![ResolutionCommand::PersistObservation(observation)],
        )
    }) else {
        return;
    };
    let mut ledger = MemoryLedger::with_capacity(1, 0);
    if ledger.commit(command.clone()).is_err() {
        return;
    }
    c.bench_function("V2::ResolutionLedger::idempotent_replay", |b| {
        b.iter(|| black_box(ledger.commit(command.clone())));
    });
}

criterion_group!(
    benches,
    bench_observation_partition,
    bench_identity_uniqueness,
    bench_causal_acyclicity,
    bench_v2_idempotent_replay
);
criterion_main!(benches);
