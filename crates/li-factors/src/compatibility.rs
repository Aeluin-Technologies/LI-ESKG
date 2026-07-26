//! Abstract observational likelihood interfaces.

use li_core::belief::BeliefState;
use li_core::observation::Observation;
use li_core::probability::Probability;

/// Pairwise likelihood evaluator computing $\phi(o_t, b_i)$ between
/// observations and active beliefs.
pub trait PairwiseCompatibility<P, S>: Send + Sync {
    /// Computes the compatibility probability $\phi(o_t, b_i) \in [0, 1]$.
    ///
    /// # Arguments
    ///
    /// * `observation` - Incoming observation payload $o_t$.
    /// * `belief` - Active belief state $b_i$.
    fn evaluate(
        &self,
        observation: &Observation<P>,
        belief: &BeliefState<S>,
    ) -> Probability;

    /// Computes the natural log-likelihood $\ln \phi(o_t, b_i) \in (-\infty,
    /// 0]$.
    ///
    /// Default implementation delegates to [`PairwiseCompatibility::evaluate`]
    /// and converts to log-domain.
    ///
    /// # Arguments
    ///
    /// * `observation` - Incoming observation payload $o_t$.
    /// * `belief` - Active belief state $b_i$.
    fn evaluate_log(
        &self,
        observation: &Observation<P>,
        belief: &BeliefState<S>,
    ) -> f64 {
        self.evaluate(observation, belief).to_log()
    }

    /// Checks whether an observation and belief state pair exceeds a
    /// compatibility threshold.
    ///
    /// # Arguments
    ///
    /// * `observation` - Incoming observation payload $o_t$.
    /// * `belief` - Active belief state $b_i$.
    /// * `threshold` - Minimum required probability.
    fn is_compatible(
        &self,
        observation: &Observation<P>,
        belief: &BeliefState<S>,
        threshold: Probability,
    ) -> bool {
        self.evaluate(observation, belief) >= threshold
    }
}
