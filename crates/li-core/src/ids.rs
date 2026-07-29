//! Strong-typed identifiers for graph entities and vertices.

use core::fmt;

use serde::{Deserialize, Serialize};

/// Unique identifier for a latent identity hypothesis node.
#[derive(
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
)]
pub struct IdentityId(pub u64);

/// Unique identifier for an empirical observation node.
#[derive(
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
)]
pub struct ObservationId(pub u64);

/// Unique identifier for a causal event node.
#[derive(
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
)]
pub struct EventId(pub u64);

/// Unique identifier for an entity state node.
#[derive(
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
)]
pub struct StateId(pub u64);

/// Stable discriminator for a vertex partition in the LI-ESKG ontology.
#[repr(u8)]
#[derive(
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
)]
pub enum VertexKind {
    /// Empirical observation partition.
    Observation = 0,
    /// Latent identity partition.
    Identity = 1,
    /// Causal event partition.
    Event = 2,
    /// Entity state partition.
    State = 3,
}

impl VertexKind {
    /// Returns the stable numeric tag used by persistent key encodings.
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// Returns the stable display prefix for this vertex partition.
    const fn prefix(self) -> &'static str {
        match self {
            Self::Observation => "O",
            Self::Identity => "I",
            Self::Event => "E",
            Self::State => "S",
        }
    }
}

impl fmt::Display for VertexKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.prefix())
    }
}

/// Tagged node identifier that preserves the disjoint ontology partition.
///
/// Unlike a raw numeric identifier, this representation cannot alias vertices
/// from different partitions that happen to use the same numeric value.
#[repr(C)]
#[derive(
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
)]
pub struct VertexId {
    kind: VertexKind,
    raw: u64,
}

impl VertexId {
    /// Creates a tagged vertex identifier.
    ///
    /// # Arguments
    ///
    /// * `kind` - Ontology partition containing the vertex.
    /// * `raw` - Numeric identifier unique within the partition.
    pub const fn new(kind: VertexKind, raw: u64) -> Self {
        Self { kind, raw }
    }

    /// Returns the ontology partition containing the vertex.
    pub const fn kind(self) -> VertexKind {
        self.kind
    }

    /// Returns the numeric identifier within the vertex partition.
    pub const fn raw(self) -> u64 {
        self.raw
    }
}

impl From<IdentityId> for VertexId {
    fn from(id: IdentityId) -> Self {
        Self::new(VertexKind::Identity, id.0)
    }
}

impl From<ObservationId> for VertexId {
    fn from(id: ObservationId) -> Self {
        Self::new(VertexKind::Observation, id.0)
    }
}

impl From<EventId> for VertexId {
    fn from(id: EventId) -> Self {
        Self::new(VertexKind::Event, id.0)
    }
}

impl From<StateId> for VertexId {
    fn from(id: StateId) -> Self {
        Self::new(VertexKind::State, id.0)
    }
}

impl fmt::Display for IdentityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "I#{}", self.0)
    }
}

impl fmt::Display for ObservationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "O#{}", self.0)
    }
}

impl fmt::Display for EventId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E#{}", self.0)
    }
}

impl fmt::Display for StateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S#{}", self.0)
    }
}

impl fmt::Display for VertexId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}#{}", self.kind.prefix(), self.raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_vertex_ids_preserve_partition_tags() {
        let observation = VertexId::from(ObservationId(7));
        let identity = VertexId::from(IdentityId(7));
        let event = VertexId::from(EventId(7));
        let state = VertexId::from(StateId(7));

        assert_ne!(observation, identity);
        assert_ne!(identity, event);
        assert_ne!(event, state);
        assert_eq!(observation.kind(), VertexKind::Observation);
        assert_eq!(observation.kind().code(), 0);
        assert_eq!(observation.raw(), 7);
    }

    #[test]
    fn typed_and_vertex_displays_are_stable() {
        assert_eq!(IdentityId(3).to_string(), "I#3");
        assert_eq!(ObservationId(3).to_string(), "O#3");
        assert_eq!(EventId(3).to_string(), "E#3");
        assert_eq!(StateId(3).to_string(), "S#3");
        assert_eq!(VertexId::from(StateId(3)).to_string(), "S#3");
        assert_eq!(VertexKind::Identity.to_string(), "I");
    }

    #[test]
    fn vertex_id_constructor_exposes_tag_and_raw_value() {
        let vertex = VertexId::new(VertexKind::Event, u64::MAX);

        assert_eq!(vertex.kind(), VertexKind::Event);
        assert_eq!(vertex.raw(), u64::MAX);
    }
}
