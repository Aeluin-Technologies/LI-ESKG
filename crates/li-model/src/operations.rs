//! Formal transactional primitives driving mutations in the persistent graph
//! layer.

use li_core::ids::{EventId, IdentityId, StateId};
use li_core::observation::{Observation, Timestamp};
use li_core::ontology::Vertex;
use li_core::relation::Relation;

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
}
