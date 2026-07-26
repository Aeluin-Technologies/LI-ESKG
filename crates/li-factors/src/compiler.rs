//! Interface for compiling factor sets from incoming evidence packages.

use li_core::belief::BeliefState;
use li_core::observation::Evidence;

use crate::factor::Factor;

/// Abstract compiler translating evidence packages into active factor nodes.
pub trait FactorCompiler<P, S> {
    /// Constructs the set of factor potential nodes $\Phi_t = \{\phi_1, \dots,
    /// \phi_m\}$.
    ///
    /// # Arguments
    ///
    /// * `evidence` - Evidence package $E_t$ containing incoming observations
    ///   and candidate sets.
    /// * `active_beliefs` - Currently active belief states $B_t$ in the system
    ///   graph.
    fn compile_factors(
        &self,
        evidence: &Evidence<P>,
        active_beliefs: &[BeliefState<S>],
    ) -> Vec<Box<dyn Factor>>;

    /// Filters active belief states using the candidate blocking identifiers
    /// provided in the evidence package.
    ///
    /// Reduces factor construction complexity from $O(|I|)$ to
    /// $O(|I_{\text{candidates}}|)$.
    ///
    /// # Arguments
    ///
    /// * `evidence` - Evidence package $E_t$.
    /// * `active_beliefs` - Full set of active belief states.
    fn filter_candidates<'a>(
        &self,
        evidence: &Evidence<P>,
        active_beliefs: &'a [BeliefState<S>],
    ) -> Vec<&'a BeliefState<S>> {
        if evidence.candidates.is_empty() {
            active_beliefs.iter().collect()
        } else {
            active_beliefs
                .iter()
                .filter(|b| evidence.candidates.contains(&b.identity))
                .collect()
        }
    }
}
