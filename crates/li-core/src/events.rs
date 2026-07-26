//! Engine stimuli and transitions driving the execution pipeline loop.

use crate::ids::{IdentityId, ObservationId};
use crate::observation::Evidence;

/// State machine transition triggers driving the core runtime execution loop.
#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeEvent<P> {
    /// Stimulus package signaling the arrival of a new empirical evidence
    /// item.
    Observation(Evidence<P>),
    /// Operational directive executing structural merge transitions of
    /// identity nodes.
    IdentityMerged {
        /// Target persistent identity node absorbing historical records.
        target: IdentityId,
        /// Duplicate identity node to be truncated and deleted from active
        /// partitions.
        duplicate: IdentityId,
    },
    /// Operational directive executing the scission of an observation link.
    IdentitySplit {
        /// Source identity currently holding the observation.
        source: IdentityId,
        /// Empirical observation to detach.
        observation: ObservationId,
        /// Destination identity node or `None` if a new identity must be
        /// instantiated.
        destination: Option<IdentityId>,
    },
    /// Operational directive registering a new identity allocation sequence.
    IdentityCreated(IdentityId),
    /// Signal instructing the runtime to execute workspace checkpoint
    /// serialization.
    Checkpoint,
    /// Termination signal executing immediate system loop shutdown.
    Shutdown,
}

impl<P> RuntimeEvent<P> {
    /// Validates whether the runtime event payload is structurally sound and
    /// non-self-referential.
    pub fn is_valid(&self) -> bool {
        match self {
            Self::IdentityMerged { target, duplicate } => target != duplicate,
            Self::IdentitySplit {
                source,
                destination: Some(dest),
                ..
            } => source != dest,
            _ => true,
        }
    }

    /// Construct a validated `IdentityMerged` event.
    /// Returns `None` if `target == duplicate`.
    pub fn merged(target: IdentityId, duplicate: IdentityId) -> Option<Self> {
        if target == duplicate {
            None
        } else {
            Some(Self::IdentityMerged { target, duplicate })
        }
    }

    /// Construct a validated `IdentitySplit` event.
    /// Returns `None` if `source == destination`.
    pub fn split(
        source: IdentityId,
        observation: ObservationId,
        destination: Option<IdentityId>,
    ) -> Option<Self> {
        if destination == Some(source) {
            None
        } else {
            Some(Self::IdentitySplit {
                source,
                observation,
                destination,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_event_prevention() {
        assert!(
            RuntimeEvent::<()>::merged(IdentityId(1), IdentityId(1)).is_none()
        );
        assert!(
            RuntimeEvent::<()>::merged(IdentityId(1), IdentityId(2)).is_some()
        );

        assert!(
            RuntimeEvent::<()>::split(
                IdentityId(1),
                ObservationId(10),
                Some(IdentityId(1))
            )
            .is_none()
        );
        assert!(
            RuntimeEvent::<()>::split(
                IdentityId(1),
                ObservationId(10),
                Some(IdentityId(2))
            )
            .is_some()
        );
    }
}
