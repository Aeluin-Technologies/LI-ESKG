//! Maximum A Posteriori (MAP) identity assignment decision estimation.

use li_core::ids::IdentityId;
use li_core::probability::Probability;

use crate::posterior::PosteriorDistribution;

/// Optimal identity candidate assignment resulting from MAP decision
/// estimation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapAssignment {
    /// Identity candidate that satisfied the selection criteria, if found.
    pub selected_identity: Option<IdentityId>,
}

/// Estimator computing MAP assignment selection over posterior probability
/// distributions.
pub struct MapEstimator;

impl MapEstimator {
    /// Extracts the MAP existing identity when it beats the new-identity
    /// assignment and decision threshold.
    ///
    /// # Arguments
    ///
    /// * `posteriors` - Calculated marginal posterior distributions over
    ///   candidate identities.
    /// * `decision_threshold` - Minimum probability threshold required to
    ///   confirm an assignment.
    ///
    /// # Returns
    ///
    /// A [`MapAssignment`] containing the selected identity if a candidate
    /// exceeds the threshold.
    pub fn estimate_map(
        &self,
        posteriors: &PosteriorDistribution,
        decision_threshold: Probability,
    ) -> MapAssignment {
        let mut best_identity = None;
        let mut max_prob = posteriors.new_identity_probability;

        for marginal in &posteriors.marginals {
            if marginal.probability.0 > decision_threshold.0 &&
                marginal.probability.0 > max_prob.0
            {
                max_prob = marginal.probability;
                best_identity = Some(marginal.identity);
            }
        }

        MapAssignment {
            selected_identity: best_identity,
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use li_core::ids::IdentityId;
    use li_core::probability::Probability;

    use crate::map::MapEstimator;
    use crate::posterior::{MarginalPosterior, PosteriorDistribution};

    #[test]
    fn test_map_empty_posteriors() {
        let estimator = MapEstimator;
        let posteriors = PosteriorDistribution {
            marginals: Vec::new(),
            new_identity_probability: Probability::new(1.0),
        };
        let assignment =
            estimator.estimate_map(&posteriors, Probability::new(0.5));

        assert_eq!(assignment.selected_identity, None);
    }

    #[test]
    fn test_map_all_below_threshold() {
        let estimator = MapEstimator;
        let posteriors = PosteriorDistribution {
            marginals: alloc::vec![
                MarginalPosterior {
                    identity: IdentityId(1),
                    probability: Probability::new(0.3),
                },
                MarginalPosterior {
                    identity: IdentityId(2),
                    probability: Probability::new(0.4),
                },
            ],
            new_identity_probability: Probability::new(0.2),
        };

        let assignment =
            estimator.estimate_map(&posteriors, Probability::new(0.5));
        assert_eq!(assignment.selected_identity, None);
    }

    #[test]
    fn test_map_selection_above_threshold() {
        let estimator = MapEstimator;
        let posteriors = PosteriorDistribution {
            marginals: alloc::vec![
                MarginalPosterior {
                    identity: IdentityId(1),
                    probability: Probability::new(0.6),
                },
                MarginalPosterior {
                    identity: IdentityId(2),
                    probability: Probability::new(0.85),
                },
            ],
            new_identity_probability: Probability::new(0.1),
        };

        let assignment =
            estimator.estimate_map(&posteriors, Probability::new(0.5));
        assert_eq!(assignment.selected_identity, Some(IdentityId(2)));
    }

    #[test]
    fn test_map_exact_threshold_edge_case() {
        let estimator = MapEstimator;
        let posteriors = PosteriorDistribution {
            marginals: alloc::vec![MarginalPosterior {
                identity: IdentityId(1),
                probability: Probability::new(0.5),
            }],
            new_identity_probability: Probability::new(0.1),
        };

        let assignment =
            estimator.estimate_map(&posteriors, Probability::new(0.5));
        assert_eq!(assignment.selected_identity, None);
    }
}
