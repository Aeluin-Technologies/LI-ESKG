//! Transactional mutations mapping the structural transitions.

use li_core::ids::VertexId;
use li_core::observation::{Observation, Timestamp};
use li_core::relation::Relation;

use crate::ontology::{EventNode, IdentityNode, StateNode};

/// Atomic operations allowed to transition the state of the persistent graph.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphOperation<P, E, S> {
    /// Commits an immutable empirical observation node to $O$.
    CommitObservation(Observation<P>),
    /// Commits a newly resolved latent identity node to $I$.
    CommitIdentity(IdentityNode),
    /// Commits a causal event node to $E$.
    CommitEvent(EventNode<E>),
    /// Commits a static state node to $S$.
    CommitState(StateNode<S>),
    /// Establishes a directed relation between two vertices in $R$.
    CommitRelation {
        source: VertexId,
        relation: Relation,
        target: VertexId,
        created_at: Timestamp,
    },
}
