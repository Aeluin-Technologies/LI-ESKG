//! Graph homomorphism mappings for causal subgraphs.

use alloc::collections::BTreeSet;

use crate::graph::KnowledgeGraph;
use crate::ontology::Edge;

/// Structurally isolated causal graph containing zero identity uncertainty.
#[derive(Debug, Clone)]
pub struct EventStateGraph {
    pub edges: BTreeSet<Edge>,
}

/// Representation of the projection homomorphism $\pi$.
pub trait GraphProjection<Input: KnowledgeGraph> {
    /// Projects the global graph into the original Event-State causal
    /// subspace.
    fn project(graph: &Input) -> EventStateGraph;
}

/// Implementation of Theorem 2 (Projection Preservation).
pub struct EventStateProjection;
