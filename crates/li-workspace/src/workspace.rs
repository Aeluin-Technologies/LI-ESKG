//! Abstract interface representing the ephemeral active belief tracking memory
//! layer.

use alloc::vec::Vec;

use li_core::belief::BeliefState;
use li_core::ids::IdentityId;
use li_core::observation::Timestamp;

use crate::checkpoint::WorkspaceSnapshot;

/// Management layer for maintaining the bounded active set of tracking
/// hypotheses $B_t = \{b_1, \dots, b_n\}$.
pub trait ActiveWorkspace {
    /// Aggregate statistical summary of belief layer.
    type Summary;

    /// Inserts or updates an active belief state hypothesis.
    fn insert(&mut self, belief: BeliefState<Self::Summary>);

    /// Retrieves a reference to an active belief state by identity identifier.
    fn get(&self, id: IdentityId) -> Option<&BeliefState<Self::Summary>>;

    /// Retrieves a mutable reference to an active belief state by identity
    /// identifier.
    fn get_mut(
        &mut self,
        id: IdentityId,
    ) -> Option<&mut BeliefState<Self::Summary>>;

    /// Returns references to all currently active belief states.
    fn active_beliefs(&self) -> Vec<&BeliefState<Self::Summary>>;

    /// Identifies and removes expired belief states exceeding the threshold
    /// $\tau$.
    fn evict_expired(
        &mut self,
        current_time: Timestamp,
        ttl_microseconds: i64,
    ) -> Vec<BeliefState<Self::Summary>>;

    /// Removes expired beliefs into a caller-owned reusable buffer.
    ///
    /// Implementations should override this method when they can fill
    /// `output` directly. The default preserves compatibility but may allocate
    /// a temporary vector through [`Self::evict_expired`].
    ///
    /// # Arguments
    ///
    /// * `current_time` - Timestamp used as the expiration baseline.
    /// * `ttl_microseconds` - Strict time-to-live threshold in microseconds.
    /// * `output` - Buffer cleared before evicted beliefs are appended.
    fn evict_expired_into(
        &mut self,
        current_time: Timestamp,
        ttl_microseconds: i64,
        output: &mut Vec<BeliefState<Self::Summary>>,
    ) {
        output.clear();
        output.extend(self.evict_expired(current_time, ttl_microseconds));
    }

    /// Instantiates a snapshot of the current active belief layer.
    fn create_snapshot(
        &self,
        current_time: Timestamp,
    ) -> WorkspaceSnapshot<Self::Summary>;
}
