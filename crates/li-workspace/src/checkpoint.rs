//! Serialized snapshots for fault tolerance and emergency recovery.

use alloc::vec::Vec;

use li_core::belief::BeliefState;
use li_core::observation::Timestamp;

/// Immutable snapshot of the active belief layer $B_t$ at a discrete time
/// point.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkspaceSnapshot<S> {
    /// Timestamp marking the snapshot instantiation.
    pub timestamp: Timestamp,
    /// Vector of serialized active tracking hypothesis states.
    pub active_states: Vec<BeliefState<S>>,
}
