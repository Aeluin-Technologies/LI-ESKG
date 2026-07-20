//! Verification of Theorem 3 (Identity Uniqueness).

use alloc::collections::BTreeMap;

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
    G: KnowledgeGraph + SupportSetQuery + IdentitySetQuery,
{
    /// Evaluates structural identity constraints across active sets.
    fn verify(&self, graph: &G) -> bool {
        let mut global_footprints = BTreeMap::new();

        for id in graph.all_identities() {
            let support = graph.query_support_set(id);

            for obs in support {
                let footprint = (obs.timestamp, obs.modality);

                if let Some(&existing_identity_id) =
                    global_footprints.get(&footprint)
                {
                    if existing_identity_id != id {
                        return false;
                    }
                } else {
                    global_footprints.insert(footprint, id);
                }
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use li_core::ids::{IdentityId, ObservationId, VertexId};
    use li_core::observation::{Modality, Observation, Timestamp};
    use li_core::ontology::Vertex;
    use li_core::probability::Confidence;

    use super::*;
    use crate::graph::KnowledgeGraph;

    struct MockGraph {
        identities: Vec<IdentityId>,
        support_sets: BTreeMap<IdentityId, Vec<Observation<()>>>,
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
                .map(|vec| vec.iter().collect())
                .unwrap_or_default()
        }
    }

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

    #[test]
    fn test_empty_graph_passes() {
        let graph = MockGraph {
            identities: Vec::new(),
            support_sets: BTreeMap::new(),
        };

        assert!(IdentityUniquenessInvariant.verify(&graph));
    }

    #[test]
    fn test_single_identity_passes() {
        let id = IdentityId(1);
        let obs = create_mock_observation(10, 1710000000, 1);

        let mut support_sets = BTreeMap::new();
        support_sets.insert(id, vec![obs]);

        let graph = MockGraph {
            identities: vec![id],
            support_sets,
        };

        assert!(IdentityUniquenessInvariant.verify(&graph));
    }

    #[test]
    fn test_same_time_different_modality_passes() {
        // Theorem 3 allows concurrent measurements if they stem from separate
        // sensor channels.
        let id_a = IdentityId(1);
        let id_b = IdentityId(2);

        let obs_a = create_mock_observation(10, 1710000000, 1); // Modality 1
        let obs_b = create_mock_observation(11, 1710000000, 2); // Modality 2

        let mut support_sets = BTreeMap::new();
        support_sets.insert(id_a, vec![obs_a]);
        support_sets.insert(id_b, vec![obs_b]);

        let graph = MockGraph {
            identities: vec![id_a, id_b],
            support_sets,
        };

        assert!(IdentityUniquenessInvariant.verify(&graph));
    }

    #[test]
    fn test_same_modality_different_time_passes() {
        let id_a = IdentityId(1);
        let id_b = IdentityId(2);

        let obs_a = create_mock_observation(10, 1710000000, 1); // Time T1
        let obs_b = create_mock_observation(11, 1710005000, 1); // Time T2

        let mut support_sets = BTreeMap::new();
        support_sets.insert(id_a, vec![obs_a]);
        support_sets.insert(id_b, vec![obs_b]);

        let graph = MockGraph {
            identities: vec![id_a, id_b],
            support_sets,
        };

        assert!(IdentityUniquenessInvariant.verify(&graph));
    }

    #[test]
    fn test_direct_spatiotemporal_collision_fails() {
        let id_a = IdentityId(1);
        let id_b = IdentityId(2);

        // Two distinct identities claim the same snapshot on the same channel.
        let obs_a = create_mock_observation(10, 1710000000, 1);
        let obs_b = create_mock_observation(11, 1710000000, 1);

        let mut support_sets = BTreeMap::new();
        support_sets.insert(id_a, vec![obs_a]);
        support_sets.insert(id_b, vec![obs_b]);

        let graph = MockGraph {
            identities: vec![id_a, id_b],
            support_sets,
        };

        assert!(!IdentityUniquenessInvariant.verify(&graph));
    }

    #[test]
    fn test_self_duplicate_observation_footprint_passes() {
        // If an identity's query returns duplicate observations or entries
        // with the exact same timestamp/modality, it should not trip a
        // self-collision.
        let id = IdentityId(1);
        let obs_a = create_mock_observation(10, 1710000000, 1);
        let obs_b = create_mock_observation(10, 1710000000, 1); // Duplicate tracking

        let mut support_sets = BTreeMap::new();
        support_sets.insert(id, vec![obs_a, obs_b]);

        let graph = MockGraph {
            identities: vec![id],
            support_sets,
        };

        assert!(IdentityUniquenessInvariant.verify(&graph));
    }
}
