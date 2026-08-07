# LI-ESKG

[Paper](papers/paper.pdf) | [GitHub](https://github.com/Aeluin-Technologies/LI-ESKG)

LI-ESKG integrates uncertain entity resolution with an authoritative
event-state knowledge graph. The implementation separates three planes:

- an active, allocation-reusing probabilistic workspace;
- an append-only resolution ledger containing immutable evidence, inference,
  decisions, revisions, dependencies, outbox entries, and receipts;
- a host-only graph containing accepted native event-state relations.

## Benchmarks

Benchmarked on a M2 Apple chip.

| Benchmark | Scale | Execution Time | Throughput |
| :--- | :--- | :--- | :--- |
| **Observation Partition** | 500,000 obs | 69.8 ms | 7.16 Melem/s |
| **Identity Uniqueness** | 400,000 nodes | 2.30 ms | 173.77 Melem/s |
| **Causal Acyclicity** | 300,000 rels | 3.95 ms | 76.05 Melem/s |
| **Batch Insertion** | 1,000,000 items | 76.4 µs | 13.09 Gelem/s |
| **Active Belief Query** | 1,000,000 items | 4.37 ms | 228.78 Melem/s |
| **Runtime Pipeline** | 10,000x10 | 167 ms | 598.8 Kelem/s |
