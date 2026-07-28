//! Global Maximum A Posteriori (MAP) identity assignment decision estimation.

use li_core::ids::IdentityId;
use li_core::probability::Probability;

use crate::posterior::PosteriorDistribution;

/// Optimal identity candidate assignment resulting from MAP decision
/// estimation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapAssignment {
    /// Selected candidate identity maximizing posterior probability above
    /// threshold, if found.
    pub selected_identity: Option<IdentityId>,
}

/// Estimator computing MAP decision selection over posterior probability
/// distributions.
pub struct MapEstimator;

impl Default for MapEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl MapEstimator {
    /// Instantiates a new MAP decision estimator.
    pub fn new() -> Self {
        Self
    }

    /// Extracts the optimal candidate identity maximizing posterior
    /// probability above threshold.
    ///
    /// # Arguments
    ///
    /// * `posteriors` - Calculated marginal posterior distributions over
    ///   candidate identities.
    /// * `decision_threshold` - Minimum probability threshold required to
    ///   confirm assignment.
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
        if posteriors.marginals.is_empty() {
            return MapAssignment {
                selected_identity: None,
            };
        }

        let best_candidate = posteriors
            .marginals
            .iter()
            .filter(|m| m.probability.value() >= decision_threshold.value())
            .max_by(|a, b| {
                a.probability
                    .value()
                    .partial_cmp(&b.probability.value())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        MapAssignment {
            selected_identity: best_candidate.map(|m| m.identity),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::posterior::MarginalPosterior;

    #[test]
    fn test_map_respects_threshold_and_selects_max() {
        let estimator = MapEstimator::new();
        let m1 = MarginalPosterior {
            identity: IdentityId(1),
            probability: Probability::new(0.60),
            log_odds: 0.40,
        };
        let m2 = MarginalPosterior {
            identity: IdentityId(2),
            probability: Probability::new(0.85),
            log_odds: 1.73,
        };
        let posteriors = PosteriorDistribution::new(vec![m1, m2]);

        let assignment =
            estimator.estimate_map(&posteriors, Probability::new(0.70));
        assert_eq!(assignment.selected_identity, Some(IdentityId(2)));
    }

    #[test]
    fn test_map_rejects_all_below_threshold() {
        let estimator = MapEstimator::new();
        let m1 = MarginalPosterior {
            identity: IdentityId(1),
            probability: Probability::new(0.40),
            log_odds: -0.40,
        };
        let posteriors = PosteriorDistribution::new(vec![m1]);

        let assignment =
            estimator.estimate_map(&posteriors, Probability::new(0.50));
        assert_eq!(assignment.selected_identity, None);
    }

    #[test]
    fn test_map_empty_input() {
        let estimator = MapEstimator::new();
        let posteriors = PosteriorDistribution::default();
        let assignment =
            estimator.estimate_map(&posteriors, Probability::new(0.50));
        assert_eq!(assignment.selected_identity, None);
    }
}
