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

/// Unified raw node identifier for graph storage engines.
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
pub struct VertexId(pub u64);

impl From<IdentityId> for VertexId {
    fn from(id: IdentityId) -> Self {
        Self(id.0)
    }
}

impl From<ObservationId> for VertexId {
    fn from(id: ObservationId) -> Self {
        Self(id.0)
    }
}

impl From<EventId> for VertexId {
    fn from(id: EventId) -> Self {
        Self(id.0)
    }
}

impl From<StateId> for VertexId {
    fn from(id: StateId) -> Self {
        Self(id.0)
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
        write!(f, "V#{}", self.0)
    }
}
