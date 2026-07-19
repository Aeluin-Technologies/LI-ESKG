//! Formal schema validation models for vertex partitioning.

use crate::ids::{EventId, IdentityId, ObservationId, StateId};

/// Compile-time enforcement of the partitioned vertex set $V = O \sqcup I
/// \sqcup E \sqcup S$.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Vertex {
    /// Empirical observation node entry ($O$).
    Observation(ObservationId),
    /// Latent identity hypothesis node entry ($I$).
    Identity(IdentityId),
    /// Temporal event node entry ($E$).
    Event(EventId),
    /// Causal entity state node entry ($S$).
    State(StateId),
}
