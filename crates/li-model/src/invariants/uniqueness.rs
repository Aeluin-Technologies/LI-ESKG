//! Verification of Theorem 3 (Identity Uniqueness).

use alloc::vec::Vec;

use rayon::prelude::*;

use crate::graph::KnowledgeGraph;
use crate::invariants::Invariant;
use crate::queries::{IdentitySetQuery, SupportSetQuery};

/// Asserts that distinct active identity entities map to distinct physical
/// objects.
///
/// This invariant enforces that for any pair of unique identity nodes $i_a,
/// i_b \in I$ where $i_a \neq i_b$, their historical observation support sets
/// contain no concurrent measurements originating from the same sensor
/// modality channel:
///
/// $$\forall i_a, i_b \in I, \; i_a \neq i_b \implies \forall o_1 \in
/// \text{supp}(i_a), \forall o_2 \in \text{supp}(i_b), \; t(o_1) = t(o_2)
/// \implies m(o_1) \neq m(o_2)$$
pub struct IdentityUniquenessInvariant;

impl<G> Invariant<G> for IdentityUniquenessInvariant
where
    G: KnowledgeGraph + SupportSetQuery + IdentitySetQuery + Sync,
{
    /// Evaluates structural identity constraints across active sets in
    /// parallel.
    fn verify(&self, graph: &G) -> bool {
        let active_identities = IdentitySetQuery::all_identities(graph);
        let local_footprints: Option<Vec<Vec<_>>> = active_identities
            .into_par_iter()
            .map(|id| {
                let support = SupportSetQuery::query_support_set(graph, id);
                let mut footprints = Vec::with_capacity(support.len());

                for obs in support {
                    footprints.push(((obs.timestamp, obs.modality), id));
                }

                Some(footprints)
            })
            .collect();

        let footprints_lists: Vec<Vec<_>> = match local_footprints {
            Some(lists) => lists,
            None => return false,
        };

        let mut all_footprints: Vec<_> =
            footprints_lists.into_iter().flatten().collect();
        all_footprints.par_sort_unstable_by_key(|(fp, _)| *fp);

        let has_conflict = all_footprints.par_windows(2).any(|w| {
            let (fp1, id1) = &w[0];
            let (fp2, id2) = &w[1];

            fp1 == fp2 && id1 != id2
        });

        !has_conflict
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use li_core::ids::{IdentityId, ObservationId, VertexId};
    use li_core::observation::{Modality, Observation, Timestamp};
    use li_core::probability::Confidence;
    use li_core::relation::Relation;

    use super::*;
    use crate::memory::MemoryGraph;
    use crate::ontology::IdentityNode;
    use crate::operations::GraphOperation;

    fn create_mock_observation(
        id: u64,
        time: i64,
        modality: u32,
    ) -> Observation<()> {
        Observation {
            id: ObservationId(id),
            modality: Modality(modality),
            timestamp: Timestamp(time),
            confidence: Confidence(0.7),
            payload: (),
        }
    }

    fn build_graph(
        identities: Vec<(IdentityId, Vec<Observation<()>>)>,
    ) -> MemoryGraph<(), (), ()> {
        let mut graph = MemoryGraph::new();
        for (id, obs_list) in identities {
            graph.apply(GraphOperation::CommitIdentity(IdentityNode {
                id,
                created_at: Timestamp(0),
            }));
            for obs in obs_list {
                let obs_vid = VertexId(obs.id.0);
                graph.apply(GraphOperation::CommitObservation(obs));
                graph.apply(GraphOperation::CommitRelation {
                    source: obs_vid,
                    relation: Relation::Supports,
                    target: VertexId(id.0),
                    created_at: Timestamp(0),
                });
            }
        }
        graph
    }

    #[test]
    fn test_empty_graph_passes() {
        let graph = build_graph(Vec::new());
        assert!(IdentityUniquenessInvariant.verify(&graph));
    }

    #[test]
    fn test_single_identity_passes() {
        let id = IdentityId(1);
        let obs = create_mock_observation(10, 1710000000, 1);
        let graph = build_graph(vec![(id, vec![obs])]);
        assert!(IdentityUniquenessInvariant.verify(&graph));
    }

    #[test]
    fn test_same_time_different_modality_passes() {
        let id_a = IdentityId(1);
        let id_b = IdentityId(2);
        let obs_a = create_mock_observation(10, 1710000000, 1);
        let obs_b = create_mock_observation(11, 1710000000, 2);
        let graph =
            build_graph(vec![(id_a, vec![obs_a]), (id_b, vec![obs_b])]);
        assert!(IdentityUniquenessInvariant.verify(&graph));
    }

    #[test]
    fn test_same_modality_different_time_passes() {
        let id_a = IdentityId(1);
        let id_b = IdentityId(2);
        let obs_a = create_mock_observation(10, 1710000000, 1);
        let obs_b = create_mock_observation(11, 171005000, 1);
        let graph =
            build_graph(vec![(id_a, vec![obs_a]), (id_b, vec![obs_b])]);
        assert!(IdentityUniquenessInvariant.verify(&graph));
    }

    #[test]
    fn test_direct_spatiotemporal_collision_fails() {
        let id_a = IdentityId(1);
        let id_b = IdentityId(2);
        let obs_a = create_mock_observation(10, 1710000000, 1);
        let obs_b = create_mock_observation(11, 1710000000, 1);
        let graph =
            build_graph(vec![(id_a, vec![obs_a]), (id_b, vec![obs_b])]);
        assert!(!IdentityUniquenessInvariant.verify(&graph));
    }

    #[test]
    fn test_self_duplicate_observation_footprint_passes() {
        let id = IdentityId(1);
        let obs_a = create_mock_observation(10, 1710000000, 1);
        let obs_b = create_mock_observation(10, 1710000000, 1);
        let graph = build_graph(vec![(id, vec![obs_a, obs_b])]);
        assert!(IdentityUniquenessInvariant.verify(&graph));
    }
}
