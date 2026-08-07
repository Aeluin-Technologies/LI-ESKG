//! Type-state lifecycle for latent identity hypotheses.

use std::marker::PhantomData;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::host::HostEntityRef;
use crate::ids::{CommitVersion, IdentityId};
use crate::observation::Timestamp;

mod sealed {
    pub trait Sealed {}
}

/// Sealed marker implemented by valid compile-time identity states.
pub trait IdentityState: sealed::Sealed {
    /// Runtime status corresponding to the marker.
    const STATUS: IdentityStatus;
}

/// Active identity marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Active;

/// Dormant identity marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dormant;

/// Resolved identity marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolved;

/// Merged historical identity marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Merged;

/// Retired identity marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Retired;

macro_rules! identity_states {
    ($($marker:ty => $status:ident),+ $(,)?) => {
        $(
            impl sealed::Sealed for $marker {}
            impl IdentityState for $marker {
                const STATUS: IdentityStatus = IdentityStatus::$status;
            }
        )+
    };
}

identity_states! {
    Active => Active,
    Dormant => Dormant,
    Resolved => Resolved,
    Merged => Merged,
    Retired => Retired,
}

/// Durable runtime identity lifecycle status.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IdentityStatus {
    /// Receives committed observations in the hot workspace.
    Active = 0,
    /// Removed from hot belief state but eligible for retrieval.
    Dormant = 1,
    /// Promoted or mapped to an authoritative host entity.
    Resolved = 2,
    /// Historical member of an active merge equivalence class.
    Merged = 3,
    /// Excluded from ordinary candidate retrieval.
    Retired = 4,
}

/// Error returned by a lifecycle transition that would create an invalid
/// state.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum IdentityTransitionError {
    /// A merge attempted to target the same identity.
    #[error("identity cannot be merged into itself")]
    SelfMerge,
}

/// Latent identity whose legal transitions are selected by `S` at compile
/// time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity<S: IdentityState> {
    id: IdentityId,
    created_at: Timestamp,
    applied_version: CommitVersion,
    host: Option<HostEntityRef>,
    canonical: Option<IdentityId>,
    marker: PhantomData<S>,
}

impl Identity<Active> {
    /// Creates a new active explanatory hypothesis.
    pub const fn new(
        id: IdentityId,
        created_at: Timestamp,
        applied_version: CommitVersion,
    ) -> Self {
        Self {
            id,
            created_at,
            applied_version,
            host: None,
            canonical: None,
            marker: PhantomData,
        }
    }

    /// Moves the hypothesis out of the hot workspace without retiring it.
    pub fn dormant(self, version: CommitVersion) -> Identity<Dormant> {
        self.transition(version, None, None)
    }

    /// Resolves the hypothesis to an authoritative host entity.
    pub fn resolve(
        self,
        host: HostEntityRef,
        version: CommitVersion,
    ) -> Identity<Resolved> {
        self.transition(version, Some(host), None)
    }

    /// Closes this identity into another merge-class member.
    ///
    /// # Errors
    ///
    /// Returns [`IdentityTransitionError::SelfMerge`] when `canonical` equals
    /// the identity being closed.
    pub fn merge(
        self,
        canonical: IdentityId,
        version: CommitVersion,
    ) -> Result<Identity<Merged>, IdentityTransitionError> {
        if canonical == self.id {
            return Err(IdentityTransitionError::SelfMerge);
        }
        Ok(self.transition(version, None, Some(canonical)))
    }

    /// Retires the identity from ordinary candidate retrieval.
    pub fn retire(self, version: CommitVersion) -> Identity<Retired> {
        self.transition(version, None, None)
    }
}

impl Identity<Dormant> {
    /// Reactivates a dormant identity into the hot workspace.
    pub fn reactivate(self, version: CommitVersion) -> Identity<Active> {
        self.transition(version, None, None)
    }

    /// Resolves a dormant hypothesis to an authoritative host entity.
    pub fn resolve(
        self,
        host: HostEntityRef,
        version: CommitVersion,
    ) -> Identity<Resolved> {
        self.transition(version, Some(host), None)
    }

    /// Retires a dormant identity.
    pub fn retire(self, version: CommitVersion) -> Identity<Retired> {
        self.transition(version, None, None)
    }
}

impl<S: IdentityState> Identity<S> {
    /// Returns the stable latent identity identifier.
    pub const fn id(&self) -> IdentityId {
        self.id
    }

    /// Returns the original creation event time.
    pub const fn created_at(&self) -> Timestamp {
        self.created_at
    }

    /// Returns the latest lifecycle commit applied to this typed view.
    pub const fn applied_version(&self) -> CommitVersion {
        self.applied_version
    }

    /// Returns the compile-time lifecycle status.
    pub const fn status(&self) -> IdentityStatus {
        S::STATUS
    }

    /// Borrows the promotion target when this is a resolved identity.
    pub const fn host(&self) -> Option<&HostEntityRef> {
        self.host.as_ref()
    }

    /// Returns the selected merge representative for a merged identity.
    pub const fn canonical(&self) -> Option<IdentityId> {
        self.canonical
    }

    fn transition<N: IdentityState>(
        self,
        applied_version: CommitVersion,
        host: Option<HostEntityRef>,
        canonical: Option<IdentityId>,
    ) -> Identity<N> {
        Identity {
            id: self.id,
            created_at: self.created_at,
            applied_version,
            host,
            canonical,
            marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_state_transitions_preserve_identity_and_versions() {
        let active = Identity::new(
            IdentityId(1),
            Timestamp::from_micros(5),
            CommitVersion::new(1),
        );
        let dormant = active.dormant(CommitVersion::new(2));
        assert_eq!(dormant.status(), IdentityStatus::Dormant);
        let active = dormant.reactivate(CommitVersion::new(3));
        assert_eq!(active.id(), IdentityId(1));
        assert_eq!(active.created_at(), Timestamp::from_micros(5));
        assert_eq!(active.applied_version(), CommitVersion::new(3));
    }

    #[test]
    fn resolved_and_merged_states_require_their_payloads() {
        let active = Identity::new(
            IdentityId(4),
            Timestamp::UNIX_EPOCH,
            CommitVersion::ZERO,
        );
        let host = HostEntityRef::new(1, "physical:4", CommitVersion::new(8));
        assert!(host.is_ok());
        if let Ok(host) = host {
            let resolved =
                active.clone().resolve(host.clone(), CommitVersion::new(9));
            assert_eq!(resolved.host(), Some(&host));
            assert_eq!(resolved.status(), IdentityStatus::Resolved);
        }

        assert_eq!(
            active.clone().merge(IdentityId(4), CommitVersion::new(2)),
            Err(IdentityTransitionError::SelfMerge)
        );
        let merged = active.merge(IdentityId(7), CommitVersion::new(2));
        assert!(merged.is_ok());
        if let Ok(merged) = merged {
            assert_eq!(merged.canonical(), Some(IdentityId(7)));
        }
    }
}
