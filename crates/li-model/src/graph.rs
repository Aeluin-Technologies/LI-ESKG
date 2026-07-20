//! Mathematical graph interface decoupling topology from storage drivers.

use li_core::ids::VertexId;
use li_core::ontology::Vertex;

use crate::operations::GraphOperation;

/// Interface representing the persistent knowledge graph $G = (V, R)$.
pub trait KnowledgeGraph {
    type ObservationPayload;
    type EventPayload;
    type StatePayload;

    /// Evaluates the ontological typology of a vertex within the set $V$.
    fn vertex_type(&self, id: VertexId) -> Option<Vertex>;

    /// Transitions the graph state by applying a formal operational primitive.
    fn apply(
        &mut self,
        op: GraphOperation<
            Self::ObservationPayload,
            Self::EventPayload,
            Self::StatePayload,
        >,
    );
}
