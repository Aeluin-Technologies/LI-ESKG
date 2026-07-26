//! Verification of Theorem 2 (Causal Acyclicity).

use petgraph::algo::toposort;

use crate::graph::PetGraphStore;
use crate::invariants::Invariant;
use crate::projection::{EventStateProjection, GraphProjection};

/// Asserts that the causal subgraph $R_{\text{eskg}}$ forms a Directed Acyclic
/// Graph (DAG).
pub struct CausalAcyclicityInvariant;

impl<P: Clone, E: Clone, S: Clone> Invariant<PetGraphStore<P, E, S>>
    for CausalAcyclicityInvariant
{
    fn verify(&self, graph: &PetGraphStore<P, E, S>) -> bool {
        let projected = EventStateProjection::project(graph);
        // `toposort` is implemented iteratively on the heap, preventing stack
        // overflow on deep paths.
        toposort(&projected.graph, None).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use li_core::ids::EventId;
    use li_core::observation::Timestamp;
    use li_core::ontology::Vertex;
    use li_core::relation::Relation;

    use super::*;
    use crate::graph::KnowledgeGraph;
    use crate::operations::GraphOperation;

    #[test]
    fn test_valid_causal_dag() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let ops = vec![
            GraphOperation::CommitEvent {
                id: EventId(1),
                timestamp: Timestamp::default(),
                payload: (),
            },
            GraphOperation::CommitEvent {
                id: EventId(2),
                timestamp: Timestamp::default(),
                payload: (),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Event(EventId(1)),
                relation: Relation::Influence,
                target: Vertex::Event(EventId(2)),
                created_at: Timestamp::default(),
            },
        ];
        store.apply_batch(ops).expect("Failed to apply ops");

        let invariant = CausalAcyclicityInvariant;
        assert!(invariant.verify(&store));
    }

    #[test]
    fn test_detects_causal_cycle() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let ops = vec![
            GraphOperation::CommitEvent {
                id: EventId(1),
                timestamp: Timestamp::default(),
                payload: (),
            },
            GraphOperation::CommitEvent {
                id: EventId(2),
                timestamp: Timestamp::default(),
                payload: (),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Event(EventId(1)),
                relation: Relation::Influence,
                target: Vertex::Event(EventId(2)),
                created_at: Timestamp::default(),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Event(EventId(2)),
                relation: Relation::Influence,
                target: Vertex::Event(EventId(1)),
                created_at: Timestamp::default(),
            },
        ];
        store.apply_batch(ops).expect("Failed to apply ops");

        let invariant = CausalAcyclicityInvariant;
        assert!(!invariant.verify(&store));
    }

    #[test]
    fn test_deep_dag_stack_safety() {
        let num_entities = 100_000;
        let mut store = PetGraphStore::<(), (), ()>::new();
        let mut ops = Vec::with_capacity(num_entities * 2);

        for idx in 1..=num_entities {
            ops.push(GraphOperation::CommitEvent {
                id: EventId(idx as u64),
                timestamp: Timestamp::default(),
                payload: (),
            });
        }

        for idx in 1..num_entities {
            ops.push(GraphOperation::CommitRelation {
                source: Vertex::Event(EventId(idx as u64)),
                relation: Relation::Influence,
                target: Vertex::Event(EventId((idx + 1) as u64)),
                created_at: Timestamp::default(),
            });
        }

        store
            .apply_batch(ops)
            .expect("Failed to initialize deep DAG");

        let invariant = CausalAcyclicityInvariant;
        // Verify that 100k-deep DAG succeeds without overflowing the thread
        // stack
        assert!(invariant.verify(&store));
    }
}
