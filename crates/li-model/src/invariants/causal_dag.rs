//! Verification that the deterministic Event-State subgraph is acyclic.

use petgraph::algo::toposort;
use petgraph::visit::EdgeFiltered;

use crate::graph::PetGraphStore;
use crate::invariants::{Invariant, InvariantViolation};

/// Asserts that the causal subgraph $R_{\text{eskg}}$ forms a directed acyclic
/// graph.
pub struct CausalAcyclicityInvariant;

impl<P, E, S> Invariant<PetGraphStore<P, E, S>> for CausalAcyclicityInvariant {
    fn validate(
        &self,
        graph: &PetGraphStore<P, E, S>,
    ) -> Result<(), InvariantViolation> {
        let causal_graph = EdgeFiltered::from_fn(&graph.raw_graph, |edge| {
            edge.weight().relation.is_eskg_relation()
        });

        if toposort(&causal_graph, None).is_err() {
            Err(InvariantViolation::CausalCycle)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use li_core::ids::{EventId, IdentityId};
    use li_core::observation::Timestamp;
    use li_core::ontology::Vertex;
    use li_core::relation::Relation;

    use super::*;
    use crate::graph::KnowledgeGraph;
    use crate::ontology::{EdgeData, NodeData};
    use crate::operations::GraphOperation;

    #[test]
    fn accepts_valid_causal_dag() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let operations = vec![
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
        assert!(store.apply_batch(operations).is_ok());

        let invariant = CausalAcyclicityInvariant;
        assert_eq!(invariant.validate(&store), Ok(()));
        assert!(invariant.verify(&store));
    }

    #[test]
    fn reports_causal_cycle() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let operations = vec![
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
        assert!(store.apply_batch(operations).is_ok());

        let invariant = CausalAcyclicityInvariant;
        assert_eq!(
            invariant.validate(&store),
            Err(InvariantViolation::CausalCycle)
        );
        assert!(!invariant.verify(&store));
    }

    #[test]
    fn ignores_cycles_composed_only_of_li_relations() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let timestamp = Timestamp::from_secs(1);
        let operations = vec![
            GraphOperation::CommitIdentity {
                id: IdentityId(1),
                created_at: timestamp,
            },
            GraphOperation::CommitIdentity {
                id: IdentityId(2),
                created_at: timestamp,
            },
            GraphOperation::CommitRelation {
                source: Vertex::Identity(IdentityId(1)),
                relation: Relation::AssociatedWith,
                target: Vertex::Identity(IdentityId(2)),
                created_at: timestamp,
            },
            GraphOperation::CommitRelation {
                source: Vertex::Identity(IdentityId(2)),
                relation: Relation::AssociatedWith,
                target: Vertex::Identity(IdentityId(1)),
                created_at: timestamp,
            },
        ];
        assert!(store.apply_batch(operations).is_ok());

        let invariant = CausalAcyclicityInvariant;
        assert_eq!(invariant.validate(&store), Ok(()));
    }

    #[test]
    fn detects_self_loop() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let timestamp = Timestamp::from_secs(1);
        assert!(
            store
                .apply_batch([GraphOperation::CommitEvent {
                    id: EventId(1),
                    timestamp,
                    payload: (),
                }])
                .is_ok()
        );
        let event_index =
            store.index_map.get(&Vertex::Event(EventId(1))).copied();
        if let Some(event_index) = event_index {
            store.raw_graph.add_edge(
                event_index,
                event_index,
                EdgeData {
                    relation: Relation::Influence,
                    created_at: timestamp,
                },
            );
        }

        let invariant = CausalAcyclicityInvariant;
        assert_eq!(
            invariant.validate(&store),
            Err(InvariantViolation::CausalCycle)
        );
    }

    #[test]
    fn validates_deep_causal_chain_without_using_call_stack() {
        const EVENT_COUNT: usize = 50_000;

        let timestamp = Timestamp::UNIX_EPOCH;
        let mut store = PetGraphStore::<(), (), ()>::with_capacity(
            EVENT_COUNT,
            EVENT_COUNT,
        );
        let mut previous = None;

        for raw_id in 0..EVENT_COUNT {
            let id = EventId(raw_id as u64);
            let vertex = Vertex::Event(id);
            let index = store.raw_graph.add_node(NodeData::Event {
                id,
                timestamp,
                payload: (),
            });
            store.index_map.insert(vertex, index);
            if let Some(source) = previous {
                store.raw_graph.add_edge(
                    source,
                    index,
                    EdgeData {
                        relation: Relation::Influence,
                        created_at: timestamp,
                    },
                );
            }
            previous = Some(index);
        }

        assert_eq!(CausalAcyclicityInvariant.validate(&store), Ok(()));
    }
}
