//! Strong-typed identifiers for graph entities and vertices.

/// Unique identifier for a latent identity hypothesis node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IdentityId(pub u64);

/// Unique identifier for an empirical observation node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObservationId(pub u64);

/// Unique identifier for a causal event node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EventId(pub u64);

/// Unique identifier for an entity state node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StateId(pub u64);

/// Unified identifier for any vertex within the global knowledge graph
/// topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VertexId(pub u64);
