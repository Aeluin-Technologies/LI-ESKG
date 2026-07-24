//! Formal node and edge schema definitions for the persistent graph layer.

use li_core::ids::{EventId, IdentityId, StateId, VertexId};
use li_core::observation::Timestamp;
use li_core::relation::Relation;
use serde::{Deserialize, Serialize};

/// Persistent record of a resolved latent identity node.
/// Represents the node $i \in I$ inside the graph topology.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq)]
pub struct IdentityNode {
    pub id: IdentityId,
    pub created_at: Timestamp,
}

/// Persistent record of a causal event occurrence node.
/// Represents the node $e \in E$ driving state transitions.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq)]
pub struct EventNode<E> {
    pub id: EventId,
    pub timestamp: Timestamp,
    pub payload: E,
}

/// Persistent record of an entity state node.
/// Represents the node $s \in S$ mapping historical intervals.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq)]
pub struct StateNode<S> {
    pub id: StateId,
    pub timestamp: Timestamp,
    pub payload: S,
}

/// Historic, directed semantic relation between two vertices.
/// Defines the tuple $(v_1, \text{relation}, v_2) \in R$ at a specific point
/// in time.
#[derive(
    Deserialize, Serialize, Debug, Clone, PartialEq, Eq, PartialOrd, Ord,
)]
pub struct Edge {
    pub source: VertexId,
    pub relation: Relation,
    pub target: VertexId,
    pub created_at: Timestamp,
}
