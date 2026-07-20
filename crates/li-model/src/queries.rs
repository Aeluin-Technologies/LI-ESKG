//! Decoupled lookup queries for topological inspection.

use alloc::vec::Vec;

use li_core::VertexId;
use li_core::ids::IdentityId;
use li_core::observation::Observation;

use crate::graph::KnowledgeGraph;

/// Query for extracting the empirical observations supporting an identity
/// hypothesis.
pub trait SupportSetQuery: KnowledgeGraph {
    /// Recovers the support set $\text{supp}(i) = \{o \in O \mid (o,
    /// \text{supports}, i) \in R\}$.
    fn query_support_set(
        &self,
        identity: IdentityId,
    ) -> Vec<&Observation<Self::ObservationPayload>>;
}

/// Query for scanning the immediate directed structural boundaries of a
/// vertex.
pub trait NeighborhoodQuery: KnowledgeGraph {
    type EdgeRef<'a>
    where
        Self: 'a;

    /// Extracts the local forward neighborhood edges of a query vertex.
    fn out_edges<'a>(&'a self, source: VertexId) -> Vec<Self::EdgeRef<'a>>;
}

/// Query for discovering all allocated identity nodes within the active graph
/// partition.
pub trait IdentitySetQuery: KnowledgeGraph {
    /// Returns a list of all active identity identifiers within the current
    /// tracking window.
    fn all_identities(&self) -> Vec<IdentityId>;
}
