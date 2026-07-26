//! Formal node and edge payload representations encapsulated within petgraph
//! storage.

use li_core::ids::{EventId, IdentityId, StateId};
use li_core::observation::{Observation, Timestamp};
use li_core::ontology::Vertex;
use li_core::relation::Relation;
use serde::{Deserialize, Serialize};

/// Encapsulates heterogeneous vertex payloads directly within petgraph nodes.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeData<P, E, S> {
    /// Empirical observation node ($O$).
    Observation(Observation<P>),
    /// Latent identity hypothesis node ($I$).
    Identity {
        id: IdentityId,
        created_at: Timestamp,
    },
    /// Causal event occurrence node ($E$).
    Event {
        id: EventId,
        timestamp: Timestamp,
        payload: E,
    },
    /// Entity state snapshot node ($S$).
    State {
        id: StateId,
        timestamp: Timestamp,
        payload: S,
    },
}

impl<P, E, S> NodeData<P, E, S> {
    /// Resolves the domain typed `Vertex` enum corresponding to this node.
    pub fn vertex(&self) -> Vertex {
        match self {
            Self::Observation(obs) => Vertex::Observation(obs.id),
            Self::Identity { id, .. } => Vertex::Identity(*id),
            Self::Event { id, .. } => Vertex::Event(*id),
            Self::State { id, .. } => Vertex::State(*id),
        }
    }

    /// Extracts the raw underlying numeric ID value.
    pub fn raw_id(&self) -> u64 {
        self.vertex().raw_id()
    }
}

/// Persistent directed edge payload storing the semantic relation type and
/// timestamp.
#[derive(
    Serialize, Deserialize, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
pub struct EdgeData {
    /// Semantic edge classification.
    pub relation: Relation,
    /// Temporal marker when the edge was established.
    pub created_at: Timestamp,
}
