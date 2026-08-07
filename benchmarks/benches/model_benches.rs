//! Benchmarks for allocation-bounded core and transactional graph operations.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, criterion_group,
    criterion_main,
};
use li_core::{
    BoundedHistory, CommandEnvelope, CommitVersion, DecisionId, EventId,
    HostNodeId, HostRelation, IdempotencyKey, IdentityId, MaterializationId,
    PhysicalNodeId, ResolutionCommand, Timestamp, TransactionId,
};
use li_model::{AuthoritativeHostGraph, HostSchemaProfile};
use li_storage::MemoryLedger;
use petgraph::stable_graph::NodeIndex;
use petgraph::visit::EdgeRef;

/// Creates the complete native host predicate profile used by benchmarks.
fn profile() -> Option<HostSchemaProfile> {
    HostSchemaProfile::new([
        Arc::from("native:triggers"),
        Arc::from("native:leadsTo"),
        Arc::from("native:evolution"),
        Arc::from("native:contain"),
        Arc::from("native:occur"),
        Arc::from("native:influence"),
    ])
    .ok()
}

/// Creates a deterministic non-zero replay key.
fn key(value: u64) -> Option<IdempotencyKey> {
    let mut bytes = [0_u8; 16];
    bytes[..8].copy_from_slice(&value.to_le_bytes());
    IdempotencyKey::new(bytes).ok()
}

/// Creates one V2 command envelope at the supplied ledger version.
fn envelope(
    sequence: u64,
    version: CommitVersion,
    commands: Vec<ResolutionCommand>,
) -> Option<CommandEnvelope> {
    CommandEnvelope::new(
        TransactionId(sequence),
        version,
        key(sequence)?,
        Timestamp::from_micros(i64::try_from(sequence).unwrap_or(i64::MAX)),
        Vec::new(),
        commands,
    )
    .ok()
}

/// Builds a host support star and returns its zero-copy traversal root.
fn support_graph(
    degree: usize,
) -> Option<(AuthoritativeHostGraph, NodeIndex<u32>)> {
    let mut graph = AuthoritativeHostGraph::with_capacity(
        profile()?,
        degree.saturating_add(1),
        degree,
    );
    let physical = graph
        .add_node(HostNodeId::Physical(PhysicalNodeId(1)))
        .ok()?;
    for offset in 0..degree {
        let id = u64::try_from(offset).ok()?.saturating_add(1);
        graph.add_node(HostNodeId::Event(EventId(id))).ok()?;
        graph
            .materialize(
                HostRelation::Occur {
                    event: EventId(id),
                    physical: PhysicalNodeId(1),
                },
                DecisionId(id),
                CommitVersion::new(1),
                MaterializationId(id),
            )
            .ok()?;
    }
    Some((graph, physical))
}

/// Builds a chain of active merge classes for revision scaling.
fn merge_ledger(degree: usize) -> Option<MemoryLedger> {
    let mut ledger = MemoryLedger::with_capacity(
        degree.saturating_mul(2).saturating_add(4),
        degree.saturating_add(2),
    );
    let identities = (0..degree.saturating_add(2))
        .map(|offset| ResolutionCommand::CreateIdentity {
            identity: IdentityId(
                u64::try_from(offset).unwrap_or(u64::MAX) + 1,
            ),
            created_at: Timestamp::UNIX_EPOCH,
        })
        .collect();
    ledger
        .commit(envelope(1, CommitVersion::ZERO, identities)?)
        .ok()?;
    for offset in 0..degree {
        let sequence = u64::try_from(offset).ok()?.saturating_add(2);
        let command = ResolutionCommand::merge(
            DecisionId(sequence),
            IdentityId(1),
            IdentityId(sequence),
            Vec::new(),
            1,
            1,
        )
        .ok()?;
        let next = envelope(sequence, ledger.version(), vec![command])?;
        ledger.commit(next).ok()?;
    }
    Some(ledger)
}

fn bench_bounded_history_append(c: &mut Criterion) {
    let mut group = c.benchmark_group("BoundedHistory::steady_state_push");
    group.throughput(Throughput::Elements(1));
    for capacity in [1_usize, 8, 64, 256, 1_024] {
        let mut history = BoundedHistory::new(capacity);
        for value in 0..capacity {
            let _ = history.push(value as u64);
        }
        let mut value = capacity as u64;
        group.bench_with_input(
            BenchmarkId::from_parameter(capacity),
            &capacity,
            |b, _| {
                b.iter(|| {
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
    let Some(host_profile) = profile() else {
        return;
    };
    let mut group =
        c.benchmark_group("V2::AuthoritativeHostGraph::materialize");
    group.throughput(Throughput::Elements(1));
    group.bench_function("new_identity_pre_reserved", |b| {
        b.iter_batched(
            || {
                AuthoritativeHostGraph::with_capacity(
                    host_profile.clone(),
                    2,
                    1,
                )
            },
            |mut graph| {
                let first = graph.add_node(HostNodeId::Event(EventId(1)));
                let second =
                    graph.add_node(HostNodeId::Physical(PhysicalNodeId(1)));
                let result = if first.is_ok() && second.is_ok() {
                    graph.materialize(
                        HostRelation::Occur {
                            event: EventId(1),
                            physical: PhysicalNodeId(1),
                        },
                        DecisionId(1),
                        CommitVersion::new(1),
                        MaterializationId(1),
                    )
                } else {
                    return black_box(graph);
                };
                let _ = black_box(result);
                black_box(graph)
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("existing_identity_pre_reserved", |b| {
        b.iter_batched(
            || {
                let mut graph = AuthoritativeHostGraph::with_capacity(
                    host_profile.clone(),
                    2,
                    1,
                );
                let _ = graph.add_node(HostNodeId::Event(EventId(1)));
                let _ =
                    graph.add_node(HostNodeId::Physical(PhysicalNodeId(1)));
                graph
            },
            |mut graph| {
                black_box(graph.materialize(
                    HostRelation::Occur {
                        event: EventId(1),
                        physical: PhysicalNodeId(1),
                    },
                    DecisionId(1),
                    CommitVersion::new(1),
                    MaterializationId(1),
                ))
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// Measures reference-only traversal of observation support stars.
fn bench_zero_copy_support_iteration(c: &mut Criterion) {
    let mut group =
        c.benchmark_group("V2::AuthoritativeHostGraph::incoming_relations");
    for degree in [1_usize, 8, 64, 512, 4_096] {
        let Some((graph, physical)) = support_graph(degree) else {
            continue;
        };
        group.throughput(Throughput::Elements(degree as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(degree),
            &degree,
            |b, _| {
                b.iter(|| {
                    let checksum = graph
                        .graph()
                        .edges_directed(
                            physical,
                            petgraph::Direction::Incoming,
                        )
                        .fold(0_u64, |acc, edge| {
                            acc.wrapping_add(u64::from(
                                edge.id().index() as u32
                            ))
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
    let mut group =
        c.benchmark_group("V2::ResolutionLedger::merge_identities");

    for degree in [1_usize, 8, 64, 512] {
        let Some(template) = merge_ledger(degree) else {
            continue;
        };
        group.throughput(Throughput::Elements(degree as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(degree),
            &degree,
            |b, &merge_degree| {
                b.iter_batched(
                    || template.clone(),
                    |mut ledger| {
                        let sequence = u64::try_from(merge_degree)
                            .unwrap_or(u64::MAX)
                            .saturating_add(10_000);
                        let command = ResolutionCommand::merge(
                            DecisionId(sequence),
                            IdentityId(1),
                            IdentityId(
                                u64::try_from(merge_degree)
                                    .unwrap_or(u64::MAX)
                                    .saturating_add(2),
                            ),
                            Vec::new(),
                            1,
                            1,
                        );
                        let result = command.ok().and_then(|command| {
                            envelope(sequence, ledger.version(), vec![command])
                        });
                        black_box(result.map(|batch| ledger.commit(batch)))
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
    let Some(batch) = envelope(
        1,
        CommitVersion::ZERO,
        vec![
            ResolutionCommand::CreateIdentity {
                identity: IdentityId(1),
                created_at: Timestamp::UNIX_EPOCH,
            },
            ResolutionCommand::CreateIdentity {
                identity: IdentityId(2),
                created_at: Timestamp::UNIX_EPOCH,
            },
            ResolutionCommand::CreateIdentity {
                identity: IdentityId(3),
                created_at: Timestamp::UNIX_EPOCH,
            },
            ResolutionCommand::CreateIdentity {
                identity: IdentityId(4),
                created_at: Timestamp::UNIX_EPOCH,
            },
            ResolutionCommand::CreateIdentity {
                identity: IdentityId(1),
                created_at: Timestamp::UNIX_EPOCH,
            },
        ],
    ) else {
        return;
    };
    let mut group =
        c.benchmark_group("V2::ResolutionLedger::late_failure_rollback");
    group.throughput(Throughput::Elements(5));
    group.bench_function("node_and_edge_journal", |b| {
        b.iter_batched(
            MemoryLedger::default,
            |mut ledger| {
                let result = ledger.commit(batch.clone());
                debug_assert!(result.is_err());
                debug_assert_eq!(ledger.version(), CommitVersion::ZERO);
                black_box(result)
            },
            BatchSize::SmallInput,
        );
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
