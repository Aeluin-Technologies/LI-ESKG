//! Formal schema models for partitioned vertex sets.

use serde::{Deserialize, Serialize};

use crate::ids::{
    EventId, IdentityId, ObservationId, StateId, VertexId, VertexKind,
};

/// Compile-time enforcement of the partitioned vertex set $V = O \sqcup I
/// \sqcup E \sqcup S$.
#[derive(Deserialize, Serialize, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Vertex {
    /// Empirical observation node ($O$).
    Observation(ObservationId),
    /// Latent identity hypothesis node ($I$).
    Identity(IdentityId),
    /// Temporal event node ($E$).
    Event(EventId),
    /// Causal entity state node ($S$).
    State(StateId),
}

impl Vertex {
    /// Extracts the raw numeric value of the encapsulated node identifier.
    pub fn raw_id(&self) -> u64 {
        match self {
            Self::Observation(id) => id.0,
            Self::Identity(id) => id.0,
            Self::Event(id) => id.0,
            Self::State(id) => id.0,
        }
    }

    /// Converts the typed vertex enum into a generic `VertexId`.
    pub fn vertex_id(&self) -> VertexId {
        match self {
            Self::Observation(id) => VertexId::from(*id),
            Self::Identity(id) => VertexId::from(*id),
            Self::Event(id) => VertexId::from(*id),
            Self::State(id) => VertexId::from(*id),
        }
    }

    /// Returns the ontology partition containing this vertex.
    pub const fn kind(&self) -> VertexKind {
        match self {
            Self::Observation(_) => VertexKind::Observation,
            Self::Identity(_) => VertexKind::Identity,
            Self::Event(_) => VertexKind::Event,
            Self::State(_) => VertexKind::State,
        }
    }
}

impl From<ObservationId> for Vertex {
    fn from(id: ObservationId) -> Self {
        Self::Observation(id)
    }
}

impl From<IdentityId> for Vertex {
    fn from(id: IdentityId) -> Self {
        Self::Identity(id)
    }
}

impl From<EventId> for Vertex {
    fn from(id: EventId) -> Self {
        Self::Event(id)
    }
}

impl From<StateId> for Vertex {
    fn from(id: StateId) -> Self {
        Self::State(id)
    }
}

impl From<Vertex> for VertexId {
    fn from(vertex: Vertex) -> Self {
        vertex.vertex_id()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_conversion_preserves_disjoint_partitions() {
        let observation = Vertex::from(ObservationId(11));
        let identity = Vertex::from(IdentityId(11));

        assert_ne!(observation, identity);
        assert_ne!(observation.vertex_id(), identity.vertex_id());
        assert_eq!(observation.kind(), VertexKind::Observation);
        assert_eq!(identity.kind(), VertexKind::Identity);
        assert_eq!(observation.raw_id(), 11);
    }

    #[test]
    fn every_vertex_maps_to_the_expected_tagged_identifier() {
        let cases = [
            (
                Vertex::Observation(ObservationId(1)),
                VertexKind::Observation,
            ),
            (Vertex::Identity(IdentityId(2)), VertexKind::Identity),
            (Vertex::Event(EventId(3)), VertexKind::Event),
            (Vertex::State(StateId(4)), VertexKind::State),
        ];

        for (vertex, expected_kind) in cases {
            assert_eq!(vertex.vertex_id().kind(), expected_kind);
            assert_eq!(vertex.vertex_id().raw(), vertex.raw_id());
        }
    }
}
