//! Engine stimuli and transitions for driving the execution pipeline loop.

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
        /// Target persistent identity node absorbing the historical records.
        target: IdentityId,
        /// Duplicate identity node to be truncated and deleted from active
        /// partitions.
        duplicate: IdentityId,
    },
    /// Operational directive executing the scission (unlinking) of a specific
    /// observation. Re-allocates the detached evidence to a distinct or
    /// new identity hypothesis to prevent orphan nodes.
    IdentitySplit {
        /// The identity currently holding the wrong association.
        source: IdentityId,
        /// The specific empirical observation to unlink.
        observation: ObservationId,
        /// The target identity destination (or `None` if a brand new identity
        /// node must be generated).
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
