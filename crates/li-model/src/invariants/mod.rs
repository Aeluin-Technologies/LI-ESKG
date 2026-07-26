//! Verification suites validating mathematical constraints and theorems over
//! the knowledge graph.

use crate::graph::KnowledgeGraph;

/// Threshold count of total nodes triggering parallel verification via Rayon.
pub const PARALLEL_EXECUTION_THRESHOLD: usize = 5_000;

/// Formal mathematical constraint enforced over the graph state.
pub trait Invariant<G: KnowledgeGraph> {
    /// Verifies if the graph configuration satisfies theorem constraints.
    fn verify(&self, graph: &G) -> bool;
}

pub mod causal_dag;
pub mod observation_partition;
pub mod uniqueness;

pub use causal_dag::CausalAcyclicityInvariant;
pub use observation_partition::ObservationPartitionInvariant;
pub use uniqueness::IdentityUniquenessInvariant;
