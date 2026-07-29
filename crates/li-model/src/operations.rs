//! Formal transactional primitives driving mutations in the persistent graph
//! layer.

use li_core::ids::{EventId, IdentityId, StateId};
use li_core::observation::{Observation, Timestamp};
use li_core::ontology::Vertex;
use li_core::relation::Relation;

/// Resolved assignment for an incoming observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityAssignment {
    /// Attach the observation to an identity already present in the graph.
    Existing(IdentityId),
    /// Create a new identity and attach the observation to it atomically.
    New(IdentityId),
}

/// Atomic graph operational primitives updating topology or payload records.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphOperation<P, E, S> {
    /// Commits an empirical observation node to $O$.
    CommitObservation(Observation<P>),
    /// Commits a resolved latent identity node to $I$.
    CommitIdentity {
        id: IdentityId,
        created_at: Timestamp,
    },
    /// Commits a temporal event node to $E$.
    CommitEvent {
        id: EventId,
        timestamp: Timestamp,
        payload: E,
    },
    /// Commits a state snapshot node to $S$.
    CommitState {
        id: StateId,
        timestamp: Timestamp,
        payload: S,
    },
    /// Establishes a directed semantic edge between two domain vertices in
    /// $R$.
    CommitRelation {
        source: Vertex,
        relation: Relation,
        target: Vertex,
        created_at: Timestamp,
    },
    /// Canonicalizes two active identities according to Algorithm 2.
    ///
    /// The target absorbs every incident relation from the duplicate before
    /// the duplicate node is removed. Implementations must execute this
    /// operation atomically and keep it isolated from additive operations in
    /// the same batch.
    MergeIdentities {
        /// Canonical identity that remains active.
        target: IdentityId,
        /// Duplicate identity removed after its relations are rewired.
        duplicate: IdentityId,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_assignment_distinguishes_existing_and_new_identities() {
        let identity = IdentityId(7);

        assert_eq!(
            IdentityAssignment::Existing(identity),
            IdentityAssignment::Existing(identity)
        );
        assert_ne!(
            IdentityAssignment::Existing(identity),
            IdentityAssignment::New(identity)
        );
    }

    #[test]
    fn merge_operation_preserves_both_identity_roles() {
        let operation = GraphOperation::<(), (), ()>::MergeIdentities {
            target: IdentityId(1),
            duplicate: IdentityId(2),
        };

        assert_eq!(
            operation,
            GraphOperation::MergeIdentities {
                target: IdentityId(1),
                duplicate: IdentityId(2),
            }
        );
    }
}
