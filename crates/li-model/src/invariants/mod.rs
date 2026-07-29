//! Verification suites validating mathematical constraints and theorems over
//! the knowledge graph.

use li_core::ids::{IdentityId, ObservationId};
use li_core::ontology::Vertex;
use thiserror::Error;

use crate::graph::KnowledgeGraph;

/// A diagnostic violation of a formal graph invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum InvariantViolation {
    /// The Event-State causal subgraph contains a directed cycle.
    #[error("Event-State causal relations contain a directed cycle")]
    CausalCycle,
    /// An observation does not support exactly one identity.
    #[error(
        "Observation {observation:?} has {actual} Supports relations; expected 1"
    )]
    ObservationSupportCardinality {
        /// Observation with an invalid support cardinality.
        observation: ObservationId,
        /// Number of outgoing Supports relations.
        actual: usize,
    },
    /// An identity has no supporting observation.
    #[error(
        "Identity {identity:?} has {actual} supporting observations; expected at least 1"
    )]
    IdentitySupportCardinality {
        /// Identity with an empty support set.
        identity: IdentityId,
        /// Number of incoming Supports relations.
        actual: usize,
    },
    /// A Supports relation has an invalid ontology endpoint.
    #[error(
        "Supports relation has invalid endpoints {origin:?} -> {target:?}"
    )]
    InvalidSupportEndpoints {
        /// Source endpoint of the relation.
        origin: Vertex,
        /// Target endpoint of the relation.
        target: Vertex,
    },
    /// An identity node is absent from or disagrees with the canonical index.
    #[error("Identity node {identity:?} is not canonical in the graph index")]
    NonCanonicalIdentity {
        /// Identity whose graph node is not indexed canonically.
        identity: IdentityId,
    },
    /// An identity index entry does not resolve to its canonical graph node.
    #[error("Identity index entry {identity:?} is stale or mismatched")]
    StaleIdentityIndex {
        /// Identity whose index entry is invalid.
        identity: IdentityId,
    },
}

/// Formal mathematical constraint enforced over graph state.
pub trait Invariant<G: KnowledgeGraph> {
    /// Validates the graph and returns a diagnostic invariant violation.
    fn validate(&self, graph: &G) -> Result<(), InvariantViolation>;

    /// Returns whether the graph satisfies the invariant.
    #[inline]
    fn verify(&self, graph: &G) -> bool {
        self.validate(graph).is_ok()
    }
}

pub mod causal_dag;
pub mod observation_partition;
pub mod uniqueness;

pub use causal_dag::CausalAcyclicityInvariant;
pub use observation_partition::ObservationPartitionInvariant;
pub use uniqueness::IdentityUniquenessInvariant;
