//! Verification of Theorem 4 (Observation Partition) with adaptive execution.

use std::collections::HashSet;

use li_core::ids::IdentityId;
use li_core::ontology::Vertex;
use li_core::relation::Relation;
use petgraph::Direction;
use petgraph::visit::EdgeRef;
use rayon::prelude::*;

use crate::graph::PetGraphStore;
use crate::invariants::{Invariant, PARALLEL_EXECUTION_THRESHOLD};
use crate::queries::{IdentitySetQuery, SupportSetQuery};

/// Asserts that the support sets of active identities form a strict partition
/// over active observations.
pub struct ObservationPartitionInvariant;

impl<P: Sync + Send + Clone, E: Sync + Send + Clone, S: Sync + Send + Clone>
    Invariant<PetGraphStore<P, E, S>> for ObservationPartitionInvariant
{
    fn verify(&self, store: &PetGraphStore<P, E, S>) -> bool {
        let active_identities = IdentitySetQuery::all_identities(store);
        let total_nodes = store.raw_graph.node_count();

        let process_identity = |id: IdentityId| -> Option<Vec<_>> {
            let support = SupportSetQuery::query_support_set(store, id);
            let mut local_obs_ids = Vec::with_capacity(support.len());

            for obs in support {
                let obs_vertex = Vertex::Observation(obs.id);
                let obs_idx = match store.index_map.get(&obs_vertex) {
                    Some(idx) => *idx,
                    None => return None,
                };

                let mut supports_relation_count = 0;
                for edge in store
                    .raw_graph
                    .edges_directed(obs_idx, Direction::Outgoing)
                {
                    if edge.weight().relation == Relation::Supports {
                        supports_relation_count += 1;
                        let target_vertex =
                            store.raw_graph[edge.target()].vertex();
                        if target_vertex != Vertex::Identity(id) {
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
        };

        let is_parallel = total_nodes >= PARALLEL_EXECUTION_THRESHOLD;

        let obs_lists: Vec<Vec<_>> = if is_parallel {
            let local_results: Option<Vec<Vec<_>>> = active_identities
                .into_par_iter()
                .map(process_identity)
                .collect();

            match local_results {
                Some(lists) => lists,
                None => return false,
            }
        } else {
            let mut lists = Vec::with_capacity(active_identities.len());
            for id in active_identities {
                match process_identity(id) {
                    Some(local) => lists.push(local),
                    None => return false,
                }
            }
            lists
        };

        let mut all_obs_ids: Vec<_> =
            obs_lists.into_iter().flatten().collect();

        if is_parallel {
            all_obs_ids.par_sort_unstable();
            !all_obs_ids.par_windows(2).any(|w| w[0] == w[1])
        } else {
            let mut seen = HashSet::with_capacity(all_obs_ids.len());
            all_obs_ids.into_iter().all(|obs_id| seen.insert(obs_id))
        }
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
    use crate::graph::KnowledgeGraph;
    use crate::operations::GraphOperation;

    #[test]
    fn test_partition_empty_graph_valid() {
        let store = PetGraphStore::<(), (), ()>::new();
        let invariant = ObservationPartitionInvariant;
        assert!(invariant.verify(&store));
    }

    #[test]
    fn test_partition_disjoint_support_sets_valid() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let obs1 = Observation::new(
            ObservationId(1),
            Modality(1),
            Timestamp::from_secs(100),
            Confidence::new(1.0),
            (),
        );
        let obs2 = Observation::new(
            ObservationId(2),
            Modality(2),
            Timestamp::from_secs(101),
            Confidence::new(1.0),
            (),
        );

        let ops = vec![
            GraphOperation::CommitObservation(obs1),
            GraphOperation::CommitObservation(obs2),
            GraphOperation::CommitIdentity {
                id: IdentityId(1),
                created_at: Timestamp::from_secs(102),
            },
            GraphOperation::CommitIdentity {
                id: IdentityId(2),
                created_at: Timestamp::from_secs(103),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(1)),
                relation: Relation::Supports,
                target: Vertex::Identity(IdentityId(1)),
                created_at: Timestamp::from_secs(104),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(2)),
                relation: Relation::Supports,
                target: Vertex::Identity(IdentityId(2)),
                created_at: Timestamp::from_secs(105),
            },
        ];
        assert!(store.apply_batch(ops).is_ok());

        let invariant = ObservationPartitionInvariant;
        assert!(invariant.verify(&store));
    }

    #[test]
    fn test_partition_shared_observation_invalid() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let obs = Observation::new(
            ObservationId(1),
            Modality(1),
            Timestamp::from_secs(100),
            Confidence::new(1.0),
            (),
        );

        let ops = vec![
            GraphOperation::CommitObservation(obs),
            GraphOperation::CommitIdentity {
                id: IdentityId(1),
                created_at: Timestamp::from_secs(101),
            },
            GraphOperation::CommitIdentity {
                id: IdentityId(2),
                created_at: Timestamp::from_secs(102),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(1)),
                relation: Relation::Supports,
                target: Vertex::Identity(IdentityId(1)),
                created_at: Timestamp::from_secs(103),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(1)),
                relation: Relation::Supports,
                target: Vertex::Identity(IdentityId(2)),
                created_at: Timestamp::from_secs(104),
            },
        ];
        assert!(store.apply_batch(ops).is_ok());

        let invariant = ObservationPartitionInvariant;
        assert!(!invariant.verify(&store));
    }

    #[test]
    fn test_partition_observation_without_supports_relation() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let obs = Observation::new(
            ObservationId(1),
            Modality(1),
            Timestamp::from_secs(100),
            Confidence::new(1.0),
            (),
        );

        let ops = vec![
            GraphOperation::CommitObservation(obs),
            GraphOperation::CommitIdentity {
                id: IdentityId(1),
                created_at: Timestamp::from_secs(101),
            },
        ];
        assert!(store.apply_batch(ops).is_ok());

        let invariant = ObservationPartitionInvariant;
        assert!(invariant.verify(&store));
    }
}
