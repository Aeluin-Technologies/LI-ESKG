//! Abstract observational likelihood interfaces.

use std::collections::HashMap;

use li_core::belief::BeliefState;
use li_core::ids::IdentityId;
use li_core::observation::Observation;
use li_core::probability::Probability;

/// Normalised joint likelihood distribution over $k$ candidates and a residual
/// noise state.
#[derive(Debug, Clone, PartialEq)]
pub struct KCandidateDistribution {
    candidate_probabilities: HashMap<IdentityId, Probability>,
    background_probability: Probability,
}

impl KCandidateDistribution {
    /// Creates a new normalized [`KCandidateDistribution`].
    ///
    /// # Arguments
    ///
    /// * `candidate_probabilities` - Likelihood mapping for candidate
    ///   identities.
    /// * `background_probability` - Unassigned hypothesis probability.
    ///
    /// # Returns
    ///
    /// Constructed [`KCandidateDistribution`].
    pub fn new(
        candidate_probabilities: HashMap<IdentityId, Probability>,
        background_probability: Probability,
    ) -> Self {
        Self {
            candidate_probabilities,
            background_probability,
        }
    }

    /// Consumes distribution to return inner candidate probability map.
    #[inline]
    pub fn into_parts(
        self,
    ) -> (HashMap<IdentityId, Probability>, Probability) {
        (self.candidate_probabilities, self.background_probability)
    }

    /// Returns reference to candidate probability mapping.
    #[inline]
    pub fn candidates(&self) -> &HashMap<IdentityId, Probability> {
        &self.candidate_probabilities
    }

    /// Returns the background/unassigned probability.
    #[inline]
    pub fn background(&self) -> Probability {
        self.background_probability
    }
}

/// Evaluator computing categorical joint likelihood distributions $\phi(o_t,
/// B_k)$.
pub trait MultiCandidateCompatibility<P, S>: Send + Sync {
    /// Computes joint probabilities for $k$ candidate beliefs simultaneously.
    ///
    /// # Arguments
    ///
    /// * `observation` - Incoming observation payload $o_t$.
    /// * `beliefs` - Active candidate belief states $[b_1, \dots, b_k]$.
    ///
    /// # Returns
    ///
    /// A normalized [`KCandidateDistribution`].
    fn evaluate_joint(
        &self,
        observation: &Observation<P>,
        beliefs: &[&BeliefState<S>],
    ) -> KCandidateDistribution;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_candidate_distribution_parts() {
        let id1 = IdentityId(10);
        let mut map = HashMap::new();
        map.insert(id1, Probability::new(0.8));

        let dist =
            KCandidateDistribution::new(map.clone(), Probability::new(0.2));

        assert_eq!(dist.candidates(), &map);
        assert_eq!(dist.background(), Probability::new(0.2));

        let (extracted_map, extracted_bg) = dist.into_parts();
        assert_eq!(extracted_map, map);
        assert_eq!(extracted_bg, Probability::new(0.2));
    }
}
