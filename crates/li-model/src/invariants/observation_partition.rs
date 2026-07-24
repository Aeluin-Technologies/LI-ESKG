//! Verification of Theorem 4 (Observation Partition).

use alloc::vec::Vec;
use core::ops::Deref;

use li_core::ids::VertexId;
use li_core::relation::Relation;
use rayon::prelude::*;

use crate::graph::KnowledgeGraph;
use crate::invariants::Invariant;
use crate::ontology::Edge;
use crate::queries::{IdentitySetQuery, NeighborhoodQuery, SupportSetQuery};

/// Asserts that the support sets of active identities form a strict partition
/// of the active observation space.
pub struct ObservationPartitionInvariant;

impl<G> Invariant<G> for ObservationPartitionInvariant
where
    G: KnowledgeGraph
        + NeighborhoodQuery
        + IdentitySetQuery
        + SupportSetQuery
        + Sync,
    for<'a> <G as NeighborhoodQuery>::EdgeRef<'a>: Deref<Target = Edge>,
{
    /// Evaluates partition properties by inspecting active identity support
    /// sets in parallel and tracking cross-boundary uniqueness constraints.
    fn verify(&self, graph: &G) -> bool {
        let active_identities = IdentitySetQuery::all_identities(graph);
        let local_results: Option<Vec<Vec<_>>> = active_identities
            .into_par_iter()
            .map(|id| {
                let support = SupportSetQuery::query_support_set(graph, id);

                if support.is_empty() {
                    return None;
                }

                let mut local_obs_ids = Vec::with_capacity(support.len());

                for obs in support {
                    let vid = VertexId(obs.id.0);
                    let out_edges = NeighborhoodQuery::out_edges(graph, vid);
                    let mut supports_relation_count = 0;

                    for edge_ref in out_edges {
                        let edge: &Edge = edge_ref.deref();
                        if edge.relation == Relation::Supports {
                            supports_relation_count += 1;
                            if edge.target != VertexId(id.0) {
                                return None;
                            }
                        }
                    }

                    if supports_relation_count != 1 {
                        return None;
                    }

                    local_obs_ids.push(obs.id);
                }

                Some(local_obs_ids)
            })
            .collect();

        let obs_lists: Vec<Vec<_>> = match local_results {
            Some(lists) => lists,
            None => return false,
        };

        let mut all_obs_ids: Vec<_> =
            obs_lists.into_iter().flatten().collect();
        all_obs_ids.par_sort_unstable();

        let has_duplicates = all_obs_ids.par_windows(2).any(|w| w[0] == w[1]);
        !has_duplicates
    }
}

#[cfg(test)]
mod tests {
    use li_core::IdentityId;
    use li_core::ids::ObservationId;
    use li_core::observation::{Modality, Observation, Timestamp};
    use li_core::probability::Confidence;

    use super::*;
    use crate::memory::MemoryGraph;
    use crate::ontology::IdentityNode;
    use crate::operations::GraphOperation;

    fn create_observation(id: u64) -> Observation<()> {
        Observation {
            id: ObservationId(id),
            modality: Modality(1),
            timestamp: Timestamp(0),
            confidence: Confidence(0.7),
            payload: (),
        }
    }

    #[test]
    fn test_empty_graph_passes_vacuously() {
        let graph: MemoryGraph<(), (), ()> = MemoryGraph::new();
        assert!(ObservationPartitionInvariant.verify(&graph));
    }

    #[test]
    fn test_perfect_disjoint_partition_passes() {
        let mut graph: MemoryGraph<(), (), ()> = MemoryGraph::new();
        let id_a = IdentityId(10);
        let id_b = IdentityId(20);
        let obs_1 = create_observation(1);
        let obs_2 = create_observation(2);

        graph.apply(GraphOperation::CommitIdentity(IdentityNode {
            id: id_a,
            created_at: Timestamp(0),
        }));
        graph.apply(GraphOperation::CommitIdentity(IdentityNode {
            id: id_b,
            created_at: Timestamp(0),
        }));
        graph.apply(GraphOperation::CommitObservation(obs_1.clone()));
        graph.apply(GraphOperation::CommitObservation(obs_2.clone()));

        graph.apply(GraphOperation::CommitRelation {
            source: VertexId(obs_1.id.0),
            relation: Relation::Supports,
            target: VertexId(id_a.0),
            created_at: Timestamp(0),
        });
        graph.apply(GraphOperation::CommitRelation {
            source: VertexId(obs_2.id.0),
            relation: Relation::Supports,
            target: VertexId(id_b.0),
            created_at: Timestamp(0),
        });

        assert!(ObservationPartitionInvariant.verify(&graph));
    }

    #[test]
    fn test_empty_identity_support_set_fails() {
        let mut graph: MemoryGraph<(), (), ()> = MemoryGraph::new();
        let id_a = IdentityId(10);
        graph.apply(GraphOperation::CommitIdentity(IdentityNode {
            id: id_a,
            created_at: Timestamp(0),
        }));

        assert!(!ObservationPartitionInvariant.verify(&graph));
    }

    #[test]
    fn test_shared_observation_across_identities_fails() {
        let mut graph: MemoryGraph<(), (), ()> = MemoryGraph::new();
        let id_a = IdentityId(10);
        let id_b = IdentityId(20);
        let shared_obs = create_observation(100);

        graph.apply(GraphOperation::CommitIdentity(IdentityNode {
            id: id_a,
            created_at: Timestamp(0),
        }));
        graph.apply(GraphOperation::CommitIdentity(IdentityNode {
            id: id_b,
            created_at: Timestamp(0),
        }));
        graph.apply(GraphOperation::CommitObservation(shared_obs.clone()));

        graph.apply(GraphOperation::CommitRelation {
            source: VertexId(shared_obs.id.0),
            relation: Relation::Supports,
            target: VertexId(id_a.0),
            created_at: Timestamp(0),
        });
        graph.apply(GraphOperation::CommitRelation {
            source: VertexId(shared_obs.id.0),
            relation: Relation::Supports,
            target: VertexId(id_b.0),
            created_at: Timestamp(0),
        });

        assert!(!ObservationPartitionInvariant.verify(&graph));
    }

    #[test]
    fn test_mismatched_edge_target_fails() {
        let mut graph: MemoryGraph<(), (), ()> = MemoryGraph::new();
        let id_a = IdentityId(10);
        let obs = create_observation(1);

        graph.apply(GraphOperation::CommitIdentity(IdentityNode {
            id: id_a,
            created_at: Timestamp(0),
        }));
        graph.apply(GraphOperation::CommitObservation(obs.clone()));

        graph.apply(GraphOperation::CommitRelation {
            source: VertexId(obs.id.0),
            relation: Relation::Supports,
            target: VertexId(999),
            created_at: Timestamp(0),
        });

        assert!(!ObservationPartitionInvariant.verify(&graph));
    }
}
