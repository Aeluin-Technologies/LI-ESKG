//! Verification suites for theoretical guarantees and system invariants.

use crate::graph::KnowledgeGraph;

/// Formal mathematical constraint enforced over the graph state.
pub trait Invariant<G: KnowledgeGraph> {
    /// Verifies if the current graph configuration satisfies the theorem
    /// constraints.
    fn verify(&self, graph: &G) -> bool;
}

pub mod observation_partition;
pub mod uniqueness;

pub use observation_partition::ObservationPartitionInvariant;
pub use uniqueness::IdentityUniquenessInvariant;
