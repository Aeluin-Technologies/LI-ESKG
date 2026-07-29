//! Verification that identity support sets partition all observations.

use li_core::ontology::Vertex;
use li_core::relation::Relation;
use petgraph::Direction;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};

use crate::graph::PetGraphStore;
use crate::invariants::{Invariant, InvariantViolation};
use crate::ontology::NodeData;

/// Asserts that every observation supports exactly one identity and every
/// identity has a non-empty support set.
pub struct ObservationPartitionInvariant;

impl<P, E, S> Invariant<PetGraphStore<P, E, S>>
    for ObservationPartitionInvariant
{
    fn validate(
        &self,
        store: &PetGraphStore<P, E, S>,
    ) -> Result<(), InvariantViolation> {
        for edge in store.raw_graph.edge_references() {
            if edge.weight().relation != Relation::Supports {
                continue;
            }

            let source = store.raw_graph[edge.source()].vertex();
            let target = store.raw_graph[edge.target()].vertex();
            if !matches!(source, Vertex::Observation(_)) ||
                !matches!(target, Vertex::Identity(_))
            {
                return Err(InvariantViolation::InvalidSupportEndpoints {
                    origin: source,
                    target,
                });
            }
        }

        for node_index in store.raw_graph.node_indices() {
            match &store.raw_graph[node_index] {
                NodeData::Observation(observation) => {
                    let actual = store
                        .raw_graph
                        .edges_directed(node_index, Direction::Outgoing)
                        .filter(|edge| {
                            edge.weight().relation == Relation::Supports
                        })
                        .count();
                    if actual != 1 {
                        return Err(
                            InvariantViolation::ObservationSupportCardinality {
                                observation: observation.id,
                                actual,
                            },
                        );
                    }
                },
                NodeData::Identity { id, .. } => {
                    let actual = store
                        .raw_graph
                        .edges_directed(node_index, Direction::Incoming)
                        .filter(|edge| {
                            edge.weight().relation == Relation::Supports
                        })
                        .count();
                    if actual == 0 {
                        return Err(
                            InvariantViolation::IdentitySupportCardinality {
                                identity: *id,
                                actual,
                            },
                        );
                    }
                },
                NodeData::Event { .. } | NodeData::State { .. } => {},
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use li_core::ids::{IdentityId, ObservationId};
    use li_core::observation::{Modality, Observation, Timestamp};
    use li_core::probability::Confidence;

    use super::*;
    use crate::graph::KnowledgeGraph;
    use crate::ontology::EdgeData;
    use crate::operations::GraphOperation;

    fn observation(id: u64) -> Observation<()> {
        Observation::new(
            ObservationId(id),
            Modality(1),
            Timestamp::from_secs(id as i64),
            Confidence::new(1.0),
            (),
        )
    }

    #[test]
    fn accepts_empty_graph() {
        let store = PetGraphStore::<(), (), ()>::new();
        let invariant = ObservationPartitionInvariant;

        assert_eq!(invariant.validate(&store), Ok(()));
    }

    #[test]
    fn accepts_disjoint_complete_support_sets() {
        let timestamp = Timestamp::from_secs(10);
        let mut store = PetGraphStore::<(), (), ()>::new();
        let operations = vec![
            GraphOperation::CommitObservation(observation(1)),
            GraphOperation::CommitObservation(observation(2)),
            GraphOperation::CommitIdentity {
                id: IdentityId(1),
                created_at: timestamp,
            },
            GraphOperation::CommitIdentity {
                id: IdentityId(2),
                created_at: timestamp,
            },
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(1)),
                relation: Relation::Supports,
                target: Vertex::Identity(IdentityId(1)),
                created_at: timestamp,
            },
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(2)),
                relation: Relation::Supports,
                target: Vertex::Identity(IdentityId(2)),
                created_at: timestamp,
            },
        ];
        assert!(store.apply_batch(operations).is_ok());

        let invariant = ObservationPartitionInvariant;
        assert_eq!(invariant.validate(&store), Ok(()));
        assert!(invariant.verify(&store));
    }

    #[test]
    fn reports_observation_without_support() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        assert!(
            store
                .apply_batch([GraphOperation::CommitObservation(observation(
                    1
                ))])
                .is_ok()
        );

        let invariant = ObservationPartitionInvariant;
        assert_eq!(
            invariant.validate(&store),
            Err(InvariantViolation::ObservationSupportCardinality {
                observation: ObservationId(1),
                actual: 0,
            })
        );
    }

    #[test]
    fn reports_identity_without_support() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        assert!(
            store
                .apply_batch([GraphOperation::CommitIdentity {
                    id: IdentityId(1),
                    created_at: Timestamp::from_secs(1),
                }])
                .is_ok()
        );

        let invariant = ObservationPartitionInvariant;
        assert_eq!(
            invariant.validate(&store),
            Err(InvariantViolation::IdentitySupportCardinality {
                identity: IdentityId(1),
                actual: 0,
            })
        );
    }

    #[test]
    fn reports_observation_supporting_multiple_identities() {
        let timestamp = Timestamp::from_secs(10);
        let mut store = PetGraphStore::<(), (), ()>::new();
        let operations = vec![
            GraphOperation::CommitObservation(observation(1)),
            GraphOperation::CommitIdentity {
                id: IdentityId(1),
                created_at: timestamp,
            },
            GraphOperation::CommitIdentity {
                id: IdentityId(2),
                created_at: timestamp,
            },
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(1)),
                relation: Relation::Supports,
                target: Vertex::Identity(IdentityId(1)),
                created_at: timestamp,
            },
        ];
        assert!(store.apply_batch(operations).is_ok());
        let observation_index = store
            .index_map
            .get(&Vertex::Observation(ObservationId(1)))
            .copied();
        let second_identity_index = store
            .index_map
            .get(&Vertex::Identity(IdentityId(2)))
            .copied();
        if let (Some(observation_index), Some(second_identity_index)) =
            (observation_index, second_identity_index)
        {
            store.raw_graph.add_edge(
                observation_index,
                second_identity_index,
                EdgeData {
                    relation: Relation::Supports,
                    created_at: timestamp,
                },
            );
        }

        let invariant = ObservationPartitionInvariant;
        assert_eq!(
            invariant.validate(&store),
            Err(InvariantViolation::ObservationSupportCardinality {
                observation: ObservationId(1),
                actual: 2,
            })
        );
    }

    #[test]
    fn reports_invalid_support_endpoints_in_raw_graph() {
        let timestamp = Timestamp::from_secs(1);
        let mut store = PetGraphStore::<(), (), ()>::new();
        let operations = [
            GraphOperation::CommitIdentity {
                id: IdentityId(1),
                created_at: timestamp,
            },
            GraphOperation::CommitIdentity {
                id: IdentityId(2),
                created_at: timestamp,
            },
        ];
        assert!(store.apply_batch(operations).is_ok());

        let source = store
            .index_map
            .get(&Vertex::Identity(IdentityId(1)))
            .copied();
        let target = store
            .index_map
            .get(&Vertex::Identity(IdentityId(2)))
            .copied();
        if let (Some(source), Some(target)) = (source, target) {
            store.raw_graph.add_edge(
                source,
                target,
                EdgeData {
                    relation: Relation::Supports,
                    created_at: timestamp,
                },
            );
        }

        let invariant = ObservationPartitionInvariant;
        assert_eq!(
            invariant.validate(&store),
            Err(InvariantViolation::InvalidSupportEndpoints {
                origin: Vertex::Identity(IdentityId(1)),
                target: Vertex::Identity(IdentityId(2)),
            })
        );
    }
}
