//! Specialized query traits for efficient topological inspections without
//! allocations.

use li_core::ids::IdentityId;
use li_core::observation::Observation;
use li_core::ontology::Vertex;
use li_core::relation::Relation;
use petgraph::Direction;
use petgraph::visit::EdgeRef;

use crate::graph::PetGraphStore;
use crate::ontology::NodeData;

/// Fast zero-copy support set query.
pub trait SupportSetQuery<P> {
    /// Recovers references to all empirical observations supporting an
    /// identity.
    fn query_support_set(&self, identity: IdentityId) -> Vec<&Observation<P>>;
}

impl<P, E, S> SupportSetQuery<P> for PetGraphStore<P, E, S> {
    fn query_support_set(&self, identity: IdentityId) -> Vec<&Observation<P>> {
        let target_vertex = Vertex::Identity(identity);
        let target_idx = match self.index_map.get(&target_vertex) {
            Some(idx) => *idx,
            None => return Vec::new(),
        };

        let mut support = Vec::new();
        for edge in self
            .raw_graph
            .edges_directed(target_idx, Direction::Incoming)
        {
            if edge.weight().relation == Relation::Supports &&
                let NodeData::Observation(obs) =
                    &self.raw_graph[edge.source()]
            {
                support.push(obs);
            }
        }
        support
    }
}

/// Outgoing topological boundary query.
pub trait NeighborhoodQuery {
    /// Recovers all outgoing directed relations from a source vertex.
    fn out_edges(&self, vertex: Vertex) -> Vec<(Vertex, Relation, Vertex)>;
}

impl<P, E, S> NeighborhoodQuery for PetGraphStore<P, E, S> {
    fn out_edges(&self, vertex: Vertex) -> Vec<(Vertex, Relation, Vertex)> {
        let src_idx = match self.index_map.get(&vertex) {
            Some(idx) => *idx,
            None => return Vec::new(),
        };

        let mut edges = Vec::new();
        for edge in self.raw_graph.edges_directed(src_idx, Direction::Outgoing)
        {
            let tgt_vertex = self.raw_graph[edge.target()].vertex();
            // NOTE: this clones `u64`.
            edges.push((vertex, edge.weight().relation, tgt_vertex));
        }
        edges
    }
}

/// Query listing active identities in the graph.
pub trait IdentitySetQuery {
    /// Returns all allocated identity identifiers.
    fn all_identities(&self) -> Vec<IdentityId>;
}

impl<P, E, S> IdentitySetQuery for PetGraphStore<P, E, S> {
    fn all_identities(&self) -> Vec<IdentityId> {
        self.index_map
            .keys()
            .filter_map(|v| match v {
                Vertex::Identity(id) => Some(*id),
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use li_core::ids::{IdentityId, ObservationId};
    use li_core::observation::{Modality, Observation, Timestamp};
    use li_core::ontology::Vertex;
    use li_core::probability::Confidence;
    use li_core::relation::Relation;

    use super::*;
    use crate::graph::{KnowledgeGraph, PetGraphStore};
    use crate::operations::GraphOperation;

    #[test]
    fn test_zero_copy_identity_set_query_empty() {
        let store = PetGraphStore::<(), (), ()>::new();
        let identities = IdentitySetQuery::all_identities(&store);
        assert!(identities.is_empty());
    }

    #[test]
    fn test_zero_copy_identity_set_query_populated() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let ops = vec![
            GraphOperation::CommitIdentity {
                id: IdentityId(1),
                created_at: Timestamp::from_secs(100),
            },
            GraphOperation::CommitIdentity {
                id: IdentityId(2),
                created_at: Timestamp::from_secs(101),
            },
        ];
        assert!(store.apply_batch(ops).is_ok());

        let mut identities = IdentitySetQuery::all_identities(&store);
        identities.sort();
        assert_eq!(identities, vec![IdentityId(1), IdentityId(2)]);
    }

    #[test]
    fn test_zero_copy_support_set_query_unregistered_identity() {
        let store = PetGraphStore::<(), (), ()>::new();
        let support =
            SupportSetQuery::query_support_set(&store, IdentityId(999));
        assert!(support.is_empty());
    }

    #[test]
    fn test_zero_copy_support_set_query_multiple_observations() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let obs1 = Observation::new(
            ObservationId(1),
            Modality(1),
            Timestamp::from_secs(100),
            Confidence::new(0.8),
            (),
        );
        let obs2 = Observation::new(
            ObservationId(2),
            Modality(1),
            Timestamp::from_secs(101),
            Confidence::new(0.9),
            (),
        );

        let ops = vec![
            GraphOperation::CommitObservation(obs1),
            GraphOperation::CommitObservation(obs2),
            GraphOperation::CommitIdentity {
                id: IdentityId(1),
                created_at: Timestamp::from_secs(102),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(1)),
                relation: Relation::Supports,
                target: Vertex::Identity(IdentityId(1)),
                created_at: Timestamp::from_secs(103),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(2)),
                relation: Relation::Supports,
                target: Vertex::Identity(IdentityId(1)),
                created_at: Timestamp::from_secs(104),
            },
        ];
        assert!(store.apply_batch(ops).is_ok());

        let support =
            SupportSetQuery::query_support_set(&store, IdentityId(1));
        assert_eq!(support.len(), 2);
        let mut obs_ids: Vec<_> = support.iter().map(|o| o.id).collect();
        obs_ids.sort();
        assert_eq!(obs_ids, vec![ObservationId(1), ObservationId(2)]);
    }
}
