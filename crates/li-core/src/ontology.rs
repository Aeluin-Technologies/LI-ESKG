//! Formal schema models for partitioned vertex sets.

use serde::{Deserialize, Serialize};

use crate::ids::{EventId, IdentityId, ObservationId, StateId, VertexId};

/// Compile-time enforcement of the partitioned vertex set $V = O \sqcup I
/// \sqcup E \sqcup S$.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq, Hash)]
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
        VertexId(self.raw_id())
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
