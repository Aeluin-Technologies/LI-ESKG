//! Retention and eviction policies for active belief states.

use li_core::belief::BeliefState;
use li_core::observation::Timestamp;

/// Policy interface determining when a belief state should be evicted from the
/// active layer $B_t$.
pub trait EvictionPolicy<S> {
    /// Evaluates if a belief state has exceeded the maximum time-to-live
    /// threshold $\tau$.
    fn should_evict(
        &self,
        belief: &BeliefState<S>,
        current_time: Timestamp,
        ttl_microseconds: i64,
    ) -> bool;
}

/// Standard time-decay eviction policy asserting $t_{\text{current}} -
/// t_{\text{last}} > \tau$.
pub struct TemporalEvictionPolicy;

impl<S> EvictionPolicy<S> for TemporalEvictionPolicy {
    fn should_evict(
        &self,
        belief: &BeliefState<S>,
        current_time: Timestamp,
        ttl_microseconds: i64,
    ) -> bool {
        current_time.0 - belief.last_update.0 > ttl_microseconds
    }
}
