//! Abstract observational likelihood interfaces.

use li_core::belief::BeliefState;
use li_core::observation::Observation;
use li_core::probability::Probability;

/// Pairwise likelihood evaluator computing $\phi(o_t, b_i)$.
pub trait PairwiseCompatibility<P, S>: Send + Sync {
    /// Computes the compatibility probability between an observation and an
    /// active belief state.
    fn evaluate(
        &self,
        observation: &Observation<P>,
        belief: &BeliefState<S>,
    ) -> Probability;
}
