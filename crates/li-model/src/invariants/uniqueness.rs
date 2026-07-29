//! Verification that identity nodes have one canonical graph representation.

use li_core::ontology::Vertex;

use crate::graph::PetGraphStore;
use crate::invariants::{Invariant, InvariantViolation};
use crate::ontology::NodeData;

/// Asserts that each identity is represented by exactly one node and that the
/// identity index resolves to that canonical node.
pub struct IdentityUniquenessInvariant;

impl<P, E, S> Invariant<PetGraphStore<P, E, S>>
    for IdentityUniquenessInvariant
{
    fn validate(
        &self,
        store: &PetGraphStore<P, E, S>,
    ) -> Result<(), InvariantViolation> {
        for node_index in store.raw_graph.node_indices() {
            if let NodeData::Identity { id, .. } = &store.raw_graph[node_index]
            {
                let vertex = Vertex::Identity(*id);
                if store.index_map.get(&vertex).copied() != Some(node_index) {
                    return Err(InvariantViolation::NonCanonicalIdentity {
                        identity: *id,
                    });
                }
            }
        }

        for (vertex, node_index) in &store.index_map {
            let Vertex::Identity(identity) = vertex else {
                continue;
            };
            let Some(NodeData::Identity { id, .. }) =
                store.raw_graph.node_weight(*node_index)
            else {
                return Err(InvariantViolation::StaleIdentityIndex {
                    identity: *identity,
                });
            };
            if id != identity {
                return Err(InvariantViolation::StaleIdentityIndex {
                    identity: *identity,
                });
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
    use li_core::relation::Relation;

    use super::*;
    use crate::graph::KnowledgeGraph;
    use crate::operations::GraphOperation;

    #[test]
    fn accepts_canonical_identity_index() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let operations = [
            GraphOperation::CommitIdentity {
                id: IdentityId(1),
                created_at: Timestamp::from_secs(1),
            },
            GraphOperation::CommitIdentity {
                id: IdentityId(2),
                created_at: Timestamp::from_secs(2),
            },
        ];
        assert!(store.apply_batch(operations).is_ok());

        let invariant = IdentityUniquenessInvariant;
        assert_eq!(invariant.validate(&store), Ok(()));
        assert!(invariant.verify(&store));
    }

    #[test]
    fn simultaneous_same_modality_observations_do_not_imply_identity_collision()
     {
        let timestamp = Timestamp::from_secs(1);
        let mut store = PetGraphStore::<(), (), ()>::new();
        let operations = vec![
            GraphOperation::CommitObservation(Observation::new(
                ObservationId(1),
                Modality(1),
                timestamp,
                Confidence::new(1.0),
                (),
            )),
            GraphOperation::CommitObservation(Observation::new(
                ObservationId(2),
                Modality(1),
                timestamp,
                Confidence::new(1.0),
                (),
            )),
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

        let invariant = IdentityUniquenessInvariant;
        assert_eq!(invariant.validate(&store), Ok(()));
    }

    #[test]
    fn reports_duplicate_raw_identity_node() {
        let identity = IdentityId(1);
        let timestamp = Timestamp::from_secs(1);
        let mut store = PetGraphStore::<(), (), ()>::new();
        assert!(
            store
                .apply_batch([GraphOperation::CommitIdentity {
                    id: identity,
                    created_at: timestamp,
                }])
                .is_ok()
        );
        store.raw_graph.add_node(NodeData::Identity {
            id: identity,
            created_at: timestamp,
        });

        let invariant = IdentityUniquenessInvariant;
        assert_eq!(
            invariant.validate(&store),
            Err(InvariantViolation::NonCanonicalIdentity { identity })
        );
    }

    #[test]
    fn reports_missing_identity_index_entry() {
        let identity = IdentityId(1);
        let mut store = PetGraphStore::<(), (), ()>::new();
        assert!(
            store
                .apply_batch([GraphOperation::CommitIdentity {
                    id: identity,
                    created_at: Timestamp::from_secs(1),
                }])
                .is_ok()
        );
        store.index_map.remove(&Vertex::Identity(identity));

        let invariant = IdentityUniquenessInvariant;
        assert_eq!(
            invariant.validate(&store),
            Err(InvariantViolation::NonCanonicalIdentity { identity })
        );
    }

    #[test]
    fn reports_stale_identity_index_entry() {
        let timestamp = Timestamp::from_secs(1);
        let mut store = PetGraphStore::<(), (), ()>::new();
        assert!(
            store
                .apply_batch([GraphOperation::CommitIdentity {
                    id: IdentityId(1),
                    created_at: timestamp,
                }])
                .is_ok()
        );
        let canonical = store
            .index_map
            .get(&Vertex::Identity(IdentityId(1)))
            .copied();
        if let Some(canonical) = canonical {
            store
                .index_map
                .insert(Vertex::Identity(IdentityId(2)), canonical);
        }

        let invariant = IdentityUniquenessInvariant;
        assert_eq!(
            invariant.validate(&store),
            Err(InvariantViolation::StaleIdentityIndex {
                identity: IdentityId(2),
            })
        );
    }
}
