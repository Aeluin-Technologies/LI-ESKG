//! Mathematical graph interface decoupling topology from storage drivers.

use alloc::vec::Vec;

use li_core::ids::{IdentityId, VertexId};
use li_core::observation::Observation;
use li_core::ontology::Vertex;

use crate::ontology::Edge;
use crate::operations::GraphOperation;

/// Interface representing the persistent knowledge graph $G = (V, R)$.
pub trait KnowledgeGraph {
    /// Payload type associated with graph observations.
    type ObservationPayload;
    /// Payload type associated with graph events.
    type EventPayload;
    /// Payload type associated with graph state nodes.
    type StatePayload;
    /// Error type returned by fallible graph storage operations.
    type Error;

    /// Evaluates the ontological typology of a vertex within the set $V$.
    fn vertex_type(&self, id: VertexId)
    -> Result<Option<Vertex>, Self::Error>;

    /// Transitions the graph state by applying a formal operational primitive.
    fn apply(
        &mut self,
        op: GraphOperation<
            Self::ObservationPayload,
            Self::EventPayload,
            Self::StatePayload,
        >,
    ) {
        let _ = self.apply_batch(&[op]);
    }

    /// Applies a batch of formal operational primitives sequentially to
    /// transition the graph state.
    fn apply_batch(
        &mut self,
        ops: &[GraphOperation<
            Self::ObservationPayload,
            Self::EventPayload,
            Self::StatePayload,
        >],
    ) -> Result<(), Self::Error>;

    /// Queries the support set of observations linked to a given identity.
    fn query_support_set(
        &self,
        identity: IdentityId,
    ) -> Result<Vec<Observation<Self::ObservationPayload>>, Self::Error>;

    /// Retrieves all outgoing edges originating from a source vertex.
    fn out_edges(&self, source: VertexId) -> Result<Vec<Edge>, Self::Error>;

    /// Enumerates all identity identifiers defined in the graph $V$.
    fn all_identities(&self) -> Result<Vec<IdentityId>, Self::Error>;
}
