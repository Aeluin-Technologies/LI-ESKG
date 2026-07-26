//! Verification of Theorem 3 (Identity Uniqueness) with adaptive execution.

use std::collections::HashMap;

use li_core::ids::IdentityId;
use rayon::prelude::*;

use crate::graph::PetGraphStore;
use crate::invariants::{Invariant, PARALLEL_EXECUTION_THRESHOLD};
use crate::queries::{IdentitySetQuery, SupportSetQuery};

/// Asserts that distinct active identity entities map to distinct physical
/// objects.
pub struct IdentityUniquenessInvariant;

impl<P: Sync + Send + Clone, E: Sync + Send + Clone, S: Sync + Send + Clone>
    Invariant<PetGraphStore<P, E, S>> for IdentityUniquenessInvariant
{
    fn verify(&self, store: &PetGraphStore<P, E, S>) -> bool {
        let active_identities = IdentitySetQuery::all_identities(store);
        let total_nodes = store.raw_graph.node_count();

        let extract_footprints = |id: IdentityId| -> Vec<_> {
            let support = SupportSetQuery::query_support_set(store, id);
            support
                .into_iter()
                .map(|obs| ((obs.timestamp, obs.modality), id))
                .collect()
        };

        let is_parallel = total_nodes >= PARALLEL_EXECUTION_THRESHOLD;

        let footprints_lists: Vec<Vec<_>> = if is_parallel {
            active_identities
                .into_par_iter()
                .map(extract_footprints)
                .collect()
        } else {
            active_identities
                .into_iter()
                .map(extract_footprints)
                .collect()
        };

        let mut all_footprints: Vec<_> =
            footprints_lists.into_iter().flatten().collect();

        if is_parallel {
            all_footprints.par_sort_unstable_by_key(|(fp, _)| *fp);
            let has_conflict = all_footprints.par_windows(2).any(|w| {
                let (fp1, id1) = &w[0];
                let (fp2, id2) = &w[1];
                fp1 == fp2 && id1 != id2
            });
            !has_conflict
        } else {
            let mut seen = HashMap::with_capacity(all_footprints.len());
            !all_footprints.into_iter().any(|(fp, id)| {
                if let Some(existing_id) = seen.insert(fp, id) {
                    existing_id != id
                } else {
                    false
                }
            })
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
    fn test_uniqueness_no_conflicts_valid() {
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
            Modality(1),
            Timestamp::from_secs(200),
            Confidence::new(1.0),
            (),
        );

        let ops = vec![
            GraphOperation::CommitObservation(obs1),
            GraphOperation::CommitObservation(obs2),
            GraphOperation::CommitIdentity {
                id: IdentityId(1),
                created_at: Timestamp::from_secs(100),
            },
            GraphOperation::CommitIdentity {
                id: IdentityId(2),
                created_at: Timestamp::from_secs(200),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(1)),
                relation: Relation::Supports,
                target: Vertex::Identity(IdentityId(1)),
                created_at: Timestamp::from_secs(101),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(2)),
                relation: Relation::Supports,
                target: Vertex::Identity(IdentityId(2)),
                created_at: Timestamp::from_secs(201),
            },
        ];
        assert!(store.apply_batch(ops).is_ok());

        let invariant = IdentityUniquenessInvariant;
        assert!(invariant.verify(&store));
    }

    #[test]
    fn test_uniqueness_spatiotemporal_footprint_conflict() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let shared_time = Timestamp::from_secs(100);
        let shared_modality = Modality(5);

        let obs1 = Observation::new(
            ObservationId(1),
            shared_modality,
            shared_time,
            Confidence::new(0.9),
            (),
        );
        let obs2 = Observation::new(
            ObservationId(2),
            shared_modality,
            shared_time,
            Confidence::new(0.95),
            (),
        );

        let ops = vec![
            GraphOperation::CommitObservation(obs1),
            GraphOperation::CommitObservation(obs2),
            GraphOperation::CommitIdentity {
                id: IdentityId(10),
                created_at: Timestamp::from_secs(100),
            },
            GraphOperation::CommitIdentity {
                id: IdentityId(20),
                created_at: Timestamp::from_secs(100),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(1)),
                relation: Relation::Supports,
                target: Vertex::Identity(IdentityId(10)),
                created_at: Timestamp::from_secs(101),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(2)),
                relation: Relation::Supports,
                target: Vertex::Identity(IdentityId(20)),
                created_at: Timestamp::from_secs(101),
            },
        ];
        assert!(store.apply_batch(ops).is_ok());

        let invariant = IdentityUniquenessInvariant;
        assert!(!invariant.verify(&store));
    }

    #[test]
    fn test_uniqueness_same_timestamp_different_modality_valid() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let shared_time = Timestamp::from_secs(100);

        let obs1 = Observation::new(
            ObservationId(1),
            Modality(1),
            shared_time,
            Confidence::new(1.0),
            (),
        );
        let obs2 = Observation::new(
            ObservationId(2),
            Modality(2),
            shared_time,
            Confidence::new(1.0),
            (),
        );

        let ops = vec![
            GraphOperation::CommitObservation(obs1),
            GraphOperation::CommitObservation(obs2),
            GraphOperation::CommitIdentity {
                id: IdentityId(1),
                created_at: Timestamp::from_secs(100),
            },
            GraphOperation::CommitIdentity {
                id: IdentityId(2),
                created_at: Timestamp::from_secs(100),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(1)),
                relation: Relation::Supports,
                target: Vertex::Identity(IdentityId(1)),
                created_at: Timestamp::from_secs(101),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(2)),
                relation: Relation::Supports,
                target: Vertex::Identity(IdentityId(2)),
                created_at: Timestamp::from_secs(101),
            },
        ];
        assert!(store.apply_batch(ops).is_ok());

        let invariant = IdentityUniquenessInvariant;
        assert!(invariant.verify(&store));
    }
}
