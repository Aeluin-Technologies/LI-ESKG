//! Strongly typed identifiers for evidence, identities, records, and host
//! entities.

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

/// Unique identifier for an authoritative physical-entity node.
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
pub struct PhysicalNodeId(pub u64);

/// Unique identifier for an authoritative semantic-knowledge node.
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
pub struct SemanticNodeId(pub u64);

/// Stable identifier for a durable inference record.
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
pub struct InferenceId(pub u64);

/// Stable identifier for a durable decision record.
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
pub struct DecisionId(pub u64);

/// Stable identifier for a resolution transaction.
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
pub struct TransactionId(pub u64);

/// Stable identifier for a materialization outbox entry and receipt.
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
pub struct MaterializationId(pub u64);

/// Stable identifier for an opaque factor provider.
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
pub struct ProviderId(pub u64);

/// Stable identifier for a provider-owned payload schema.
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
pub struct SchemaId(pub u64);

/// Stable identifier for an observation source.
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
pub struct SourceId(pub u64);

/// Totally ordered durable commit version.
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
    Default,
)]
pub struct CommitVersion(u64);

impl CommitVersion {
    /// Initial empty-ledger version.
    pub const ZERO: Self = Self(0);

    /// Creates a version from its durable integer representation.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the durable integer representation.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Computes the next commit version without wrapping.
    pub const fn checked_next(self) -> Option<Self> {
        match self.0.checked_add(1) {
            Some(value) => Some(Self(value)),
            None => None,
        }
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

impl fmt::Display for CommitVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "v{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_displays_are_stable() {
        assert_eq!(IdentityId(3).to_string(), "I#3");
        assert_eq!(ObservationId(3).to_string(), "O#3");
        assert_eq!(EventId(3).to_string(), "E#3");
        assert_eq!(StateId(3).to_string(), "S#3");
    }

    #[test]
    fn commit_version_never_wraps() {
        assert_eq!(
            CommitVersion::ZERO.checked_next(),
            Some(CommitVersion::new(1))
        );
        assert_eq!(CommitVersion::new(u64::MAX).checked_next(), None);
        assert_eq!(CommitVersion::new(42).to_string(), "v42");
    }
}
