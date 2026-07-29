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
    fn query_support_set<'a>(
        &'a self,
        identity: IdentityId,
    ) -> impl Iterator<Item = &'a Observation<P>>
    where
        P: 'a;
}

impl<P, E, S> SupportSetQuery<P> for PetGraphStore<P, E, S> {
    fn query_support_set<'a>(
        &'a self,
        identity: IdentityId,
    ) -> impl Iterator<Item = &'a Observation<P>>
    where
        P: 'a,
    {
        let target_vertex = Vertex::Identity(identity);
        self.index_map
            .get(&target_vertex)
            .into_iter()
            .flat_map(|&target_idx| {
                self.raw_graph
                    .edges_directed(target_idx, Direction::Incoming)
            })
            .filter_map(|edge| {
                if edge.weight().relation != Relation::Supports {
                    return None;
                }
                match &self.raw_graph[edge.source()] {
                    NodeData::Observation(observation) => Some(observation),
                    _ => None,
                }
            })
    }
}

/// Outgoing topological boundary query.
pub trait NeighborhoodQuery {
    /// Recovers all outgoing directed relations from a source vertex.
    fn out_edges(
        &self,
        vertex: Vertex,
    ) -> impl Iterator<Item = (Vertex, Relation, Vertex)>;
}

impl<P, E, S> NeighborhoodQuery for PetGraphStore<P, E, S> {
    fn out_edges(
        &self,
        vertex: Vertex,
    ) -> impl Iterator<Item = (Vertex, Relation, Vertex)> {
        self.index_map
            .get(&vertex)
            .into_iter()
            .flat_map(|&source| {
                self.raw_graph.edges_directed(source, Direction::Outgoing)
            })
            .map(move |edge| {
                (
                    vertex,
                    edge.weight().relation,
                    self.raw_graph[edge.target()].vertex(),
                )
            })
    }
}

/// Query listing active identities in the graph.
pub trait IdentitySetQuery {
    /// Returns all allocated identity identifiers.
    fn all_identities(&self) -> impl Iterator<Item = IdentityId>;
}

impl<P, E, S> IdentitySetQuery for PetGraphStore<P, E, S> {
    fn all_identities(&self) -> impl Iterator<Item = IdentityId> {
        self.raw_graph.node_weights().filter_map(|node| match node {
            NodeData::Identity { id, .. } => Some(*id),
            _ => None,
        })
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
        assert_eq!(IdentitySetQuery::all_identities(&store).count(), 0);
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

        let mut identities: Vec<_> =
            IdentitySetQuery::all_identities(&store).collect();
        identities.sort_unstable();
        assert_eq!(identities, vec![IdentityId(1), IdentityId(2)]);
    }

    #[test]
    fn test_zero_copy_support_set_query_unregistered_identity() {
        let store = PetGraphStore::<(), (), ()>::new();
        assert_eq!(
            SupportSetQuery::query_support_set(&store, IdentityId(999))
                .count(),
            0
        );
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

        let support: Vec<_> =
            SupportSetQuery::query_support_set(&store, IdentityId(1))
                .collect();
        assert_eq!(support.len(), 2);
        let mut obs_ids: Vec<_> = support.iter().map(|o| o.id).collect();
        obs_ids.sort();
        assert_eq!(obs_ids, vec![ObservationId(1), ObservationId(2)]);
    }

    #[test]
    fn test_zero_copy_neighborhood_query_reports_outgoing_relations() {
        let timestamp = Timestamp::from_secs(100);
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
            GraphOperation::CommitRelation {
                source: Vertex::Identity(IdentityId(1)),
                relation: Relation::AssociatedWith,
                target: Vertex::Identity(IdentityId(2)),
                created_at: timestamp,
            },
        ];
        assert!(store.apply_batch(operations).is_ok());

        let edges: Vec<_> = NeighborhoodQuery::out_edges(
            &store,
            Vertex::Identity(IdentityId(1)),
        )
        .collect();
        assert_eq!(
            edges,
            vec![(
                Vertex::Identity(IdentityId(1)),
                Relation::AssociatedWith,
                Vertex::Identity(IdentityId(2)),
            )]
        );
        assert_eq!(
            NeighborhoodQuery::out_edges(
                &store,
                Vertex::Identity(IdentityId(999)),
            )
            .count(),
            0
        );
    }
}
