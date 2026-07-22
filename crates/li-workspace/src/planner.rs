//! Inference scheduling based on information gain and temporal thresholds.

use li_core::belief::BeliefState;
use li_core::ids::IdentityId;

/// Evaluates whether incoming evidence warrants executing Belief Propagation.
pub trait InferencePlanner<S> {
    /// Evaluates information-theoretic criteria to decide if factor graph
    /// inference must run.
    fn should_trigger_inference(
        &self,
        candidates: &[IdentityId],
        active_beliefs: &[BeliefState<S>],
    ) -> bool;
}
