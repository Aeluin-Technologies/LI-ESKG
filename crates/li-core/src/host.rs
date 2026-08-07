//! Authoritative host-graph references and type-safe ESKG relation roles.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ids::{
    CommitVersion, EventId, PhysicalNodeId, SemanticNodeId, StateId,
};

/// Lightweight reference to an authoritative entity at one coherent snapshot.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct HostEntityRef {
    backend: u32,
    key: Arc<str>,
    snapshot: CommitVersion,
}

/// Error returned when an authoritative host reference is malformed.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HostReferenceError {
    /// The backend-specific key was empty.
    #[error("host entity key must not be empty")]
    EmptyKey,
}

impl HostEntityRef {
    /// Creates a validated reference to an authoritative host entity.
    ///
    /// # Arguments
    ///
    /// * `backend` - Deployment-local host adapter code.
    /// * `key` - Opaque backend key, preserved without interpretation.
    /// * `snapshot` - Host snapshot used to resolve the key.
    ///
    /// # Errors
    ///
    /// Returns [`HostReferenceError::EmptyKey`] when `key` is empty.
    pub fn new(
        backend: u32,
        key: impl Into<Arc<str>>,
        snapshot: CommitVersion,
    ) -> Result<Self, HostReferenceError> {
        let key = key.into();
        if key.is_empty() {
            return Err(HostReferenceError::EmptyKey);
        }
        Ok(Self {
            backend,
            key,
            snapshot,
        })
    }

    /// Returns the deployment-local host adapter code.
    pub const fn backend(&self) -> u32 {
        self.backend
    }

    /// Returns the opaque backend key without allocation.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Returns the coherent host snapshot version.
    pub const fn snapshot(&self) -> CommitVersion {
        self.snapshot
    }
}

/// Authoritative host node partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HostNodeId {
    /// Event-state graph state node.
    State(StateId),
    /// Event-state graph event node.
    Event(EventId),
    /// Authoritative physical-entity node.
    Physical(PhysicalNodeId),
    /// Authoritative semantic-knowledge node.
    Semantic(SemanticNodeId),
}

/// Closed set of motivating ESKG host predicate roles.
#[repr(u8)]
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
)]
pub enum HostRelationRole {
    /// State activates or enables an event: `State -> Event`.
    Triggers = 0,
    /// Event produces or leads to a resulting state: `Event -> State`.
    LeadsTo = 1,
    /// Direct state evolution: `State -> State`.
    Evolution = 2,
    /// Composite-event containment: `Event -> Event`.
    Contain = 3,
    /// Event-space attribution: `Event -> Physical`.
    Occur = 4,
    /// Host process or quality influence: `Event -> Physical`.
    Influence = 5,
}

impl HostRelationRole {
    /// Returns all required host roles in their stable code order.
    pub const ALL: [Self; 6] = [
        Self::Triggers,
        Self::LeadsTo,
        Self::Evolution,
        Self::Contain,
        Self::Occur,
        Self::Influence,
    ];

    /// Returns the stable compact role code.
    pub const fn code(self) -> u8 {
        self as u8
    }
}

/// Type-safe host relation whose endpoints are valid by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HostRelation {
    /// `Triggers(state, event)`.
    Triggers {
        /// Enabling state.
        state: StateId,
        /// Triggered event.
        event: EventId,
    },
    /// `LeadsTo(event, state)`.
    LeadsTo {
        /// Producing event.
        event: EventId,
        /// Resulting state.
        state: StateId,
    },
    /// `Evolution(from, to)`.
    Evolution {
        /// Earlier state.
        from: StateId,
        /// Later state.
        to: StateId,
    },
    /// `Contain(container, contained)`.
    Contain {
        /// Composite event.
        container: EventId,
        /// Contained event.
        contained: EventId,
    },
    /// `Occur(event, physical)`.
    Occur {
        /// Located event.
        event: EventId,
        /// Physical location or entity.
        physical: PhysicalNodeId,
    },
    /// `Influence(event, physical)`.
    Influence {
        /// Influencing event.
        event: EventId,
        /// Influenced physical node.
        physical: PhysicalNodeId,
    },
}

impl HostRelation {
    /// Returns the host-schema role represented by this relation.
    pub const fn role(self) -> HostRelationRole {
        match self {
            Self::Triggers { .. } => HostRelationRole::Triggers,
            Self::LeadsTo { .. } => HostRelationRole::LeadsTo,
            Self::Evolution { .. } => HostRelationRole::Evolution,
            Self::Contain { .. } => HostRelationRole::Contain,
            Self::Occur { .. } => HostRelationRole::Occur,
            Self::Influence { .. } => HostRelationRole::Influence,
        }
    }

    /// Returns the typed source and target nodes without dynamic validation.
    pub const fn endpoints(self) -> (HostNodeId, HostNodeId) {
        match self {
            Self::Triggers { state, event } => {
                (HostNodeId::State(state), HostNodeId::Event(event))
            },
            Self::LeadsTo { event, state } => {
                (HostNodeId::Event(event), HostNodeId::State(state))
            },
            Self::Evolution { from, to } => {
                (HostNodeId::State(from), HostNodeId::State(to))
            },
            Self::Contain {
                container,
                contained,
            } => (HostNodeId::Event(container), HostNodeId::Event(contained)),
            Self::Occur { event, physical } |
            Self::Influence { event, physical } => {
                (HostNodeId::Event(event), HostNodeId::Physical(physical))
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_host_keys_are_unrepresentable() {
        assert_eq!(
            HostEntityRef::new(1, "", CommitVersion::ZERO),
            Err(HostReferenceError::EmptyKey)
        );
        let reference =
            HostEntityRef::new(7, "person:42", CommitVersion::new(3));
        assert!(reference.is_ok());
        if let Ok(reference) = reference {
            assert_eq!(reference.backend(), 7);
            assert_eq!(reference.key(), "person:42");
            assert_eq!(reference.snapshot(), CommitVersion::new(3));
        }
    }

    #[test]
    fn relation_endpoints_match_the_v2_host_contract() {
        let cases = [
            (
                HostRelation::Triggers {
                    state: StateId(1),
                    event: EventId(2),
                },
                HostRelationRole::Triggers,
                (HostNodeId::State(StateId(1)), HostNodeId::Event(EventId(2))),
            ),
            (
                HostRelation::Occur {
                    event: EventId(2),
                    physical: PhysicalNodeId(3),
                },
                HostRelationRole::Occur,
                (
                    HostNodeId::Event(EventId(2)),
                    HostNodeId::Physical(PhysicalNodeId(3)),
                ),
            ),
        ];

        for (relation, role, endpoints) in cases {
            assert_eq!(relation.role(), role);
            assert_eq!(relation.endpoints(), endpoints);
        }
        assert_eq!(HostRelationRole::ALL.len(), 6);
    }
}
