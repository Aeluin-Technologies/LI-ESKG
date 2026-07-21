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
        let active_identities = graph.all_identities();
        let local_results: Option<Vec<Vec<_>>> = active_identities
            .into_par_iter()
            .map(|id| {
                let support = graph.query_support_set(id);

                if support.is_empty() {
                    return None;
                }

                let mut local_obs_ids = Vec::with_capacity(support.len());

                for obs in support {
                    let vid = VertexId(obs.id.0);
                    let out_edges = graph.out_edges(vid);
                    let mut supports_relation_count = 0;

                    for edge_ref in out_edges {
                        let edge = edge_ref.deref();
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

        let obs_lists = match local_results {
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
    use alloc::collections::BTreeMap;
    use alloc::vec;
    use alloc::vec::Vec;

    use li_core::IdentityId;
    use li_core::ids::ObservationId;
    use li_core::observation::{Modality, Observation, Timestamp};
    use li_core::ontology::Vertex;
    use li_core::probability::Confidence;

    use super::*;

    struct MockGraph {
        identities: Vec<IdentityId>,
        support_sets: BTreeMap<IdentityId, Vec<Observation<()>>>,
        out_edges: BTreeMap<VertexId, Vec<Edge>>,
    }

    impl KnowledgeGraph for MockGraph {
        type EventPayload = ();
        type ObservationPayload = ();
        type StatePayload = ();

        fn vertex_type(&self, _id: VertexId) -> Option<Vertex> {
            None
        }

        fn apply(
            &mut self,
            _op: crate::operations::GraphOperation<(), (), ()>,
        ) {
        }
    }

    impl IdentitySetQuery for MockGraph {
        fn all_identities(&self) -> Vec<IdentityId> {
            self.identities.clone()
        }
    }

    impl SupportSetQuery for MockGraph {
        fn query_support_set(
            &self,
            identity: IdentityId,
        ) -> Vec<&Observation<()>> {
            self.support_sets
                .get(&identity)
                .map(|v| v.iter().collect())
                .unwrap_or_default()
        }
    }

    impl NeighborhoodQuery for MockGraph {
        type EdgeRef<'a>
            = &'a Edge
        where
            Self: 'a;

        fn out_edges<'a>(
            &'a self,
            source: VertexId,
        ) -> Vec<Self::EdgeRef<'a>> {
            self.out_edges
                .get(&source)
                .map(|v| v.iter().collect())
                .unwrap_or_default()
        }
    }

    fn create_observation(id: u64) -> Observation<()> {
        Observation {
            id: ObservationId(id),
            modality: Modality(1),
            timestamp: Timestamp(0),
            confidence: Confidence(0.7),
            payload: (),
        }
    }

    fn create_edge(source: u64, target: u64, relation: Relation) -> Edge {
        Edge {
            source: VertexId(source),
            relation,
            target: VertexId(target),
            created_at: Timestamp(0),
        }
    }

    #[test]
    fn test_empty_graph_passes_vacuously() {
        let graph = MockGraph {
            identities: Vec::new(),
            support_sets: BTreeMap::new(),
            out_edges: BTreeMap::new(),
        };
        assert!(ObservationPartitionInvariant.verify(&graph));
    }

    #[test]
    fn test_perfect_disjoint_partition_passes() {
        let id_a = IdentityId(10);
        let id_b = IdentityId(20);
        let obs_1 = create_observation(1);
        let obs_2 = create_observation(2);

        let mut support_sets = BTreeMap::new();
        support_sets.insert(id_a, vec![obs_1]);
        support_sets.insert(id_b, vec![obs_2]);

        let mut out_edges = BTreeMap::new();
        out_edges.insert(
            VertexId(1),
            vec![create_edge(1, id_a.0, Relation::Supports)],
        );
        out_edges.insert(
            VertexId(2),
            vec![create_edge(2, id_b.0, Relation::Supports)],
        );

        let graph = MockGraph {
            identities: vec![id_a, id_b],
            support_sets,
            out_edges,
        };

        assert!(ObservationPartitionInvariant.verify(&graph));
    }

    #[test]
    fn test_empty_identity_support_set_fails() {
        let id_a = IdentityId(10);

        let graph = MockGraph {
            identities: vec![id_a],
            support_sets: BTreeMap::new(),
            out_edges: BTreeMap::new(),
        };

        // Fails because every active identity must have a non-empty support
        // set.
        assert!(!ObservationPartitionInvariant.verify(&graph));
    }

    #[test]
    fn test_shared_observation_across_identities_fails() {
        let id_a = IdentityId(10);
        let id_b = IdentityId(20);
        let shared_obs = create_observation(100);

        let mut support_sets = BTreeMap::new();
        // Overlap: Both tracking hypotheses pull the exact same physical
        // measurement
        support_sets.insert(id_a, vec![shared_obs.clone()]);
        support_sets.insert(id_b, vec![shared_obs]);

        let mut out_edges = BTreeMap::new();
        out_edges.insert(
            VertexId(100),
            vec![
                create_edge(100, id_a.0, Relation::Supports),
                create_edge(100, id_b.0, Relation::Supports),
            ],
        );

        let graph = MockGraph {
            identities: vec![id_a, id_b],
            support_sets,
            out_edges,
        };

        // Fails because the subsets are not disjoint.
        assert!(!ObservationPartitionInvariant.verify(&graph));
    }

    #[test]
    fn test_mismatched_edge_target_fails() {
        let id_a = IdentityId(10);
        let obs = create_observation(1);

        let mut support_sets = BTreeMap::new();
        support_sets.insert(id_a, vec![obs]);

        let mut out_edges = BTreeMap::new();
        // Malformed topology: Query returns it for Identity A, but the actual
        // edge targets 999
        out_edges.insert(
            VertexId(1),
            vec![create_edge(1, 999, Relation::Supports)],
        );

        let graph = MockGraph {
            identities: vec![id_a],
            support_sets,
            out_edges,
        };

        assert!(!ObservationPartitionInvariant.verify(&graph));
    }
}
