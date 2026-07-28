//! Marginal and joint posterior probability representations over identity
//! assignment hypotheses.

use li_core::ids::IdentityId;
use li_core::probability::Probability;
use ordered_float::OrderedFloat;

/// Single variable marginal posterior distribution capturing probability and
/// log-odds.
#[derive(Debug, Clone, PartialEq)]
pub struct MarginalPosterior {
    /// Identity candidate evaluated by this marginal probability.
    pub identity: IdentityId,
    /// Marginal probability $P(z_i = 1)$.
    pub probability: Probability,
    /// Log-odds score $\ln \frac{P(z_i = 1)}{P(z_i = 0)}$.
    pub log_odds: f64,
}

/// Collection of evaluated marginal posteriors $P(Z_t)$ across candidate
/// variables.
#[derive(Debug, Clone, Default)]
pub struct PosteriorDistribution {
    /// Vector of calculated marginal probabilities for evaluated identity
    /// candidates.
    pub marginals: Vec<MarginalPosterior>,
    is_sorted_by_id: bool,
}

impl PartialEq for PosteriorDistribution {
    fn eq(&self, other: &Self) -> bool {
        self.marginals == other.marginals
    }
}

impl PosteriorDistribution {
    /// Instantiates a new posterior distribution container.
    ///
    /// # Arguments
    ///
    /// * `marginals` - Vector of marginal probability calculations.
    pub fn new(marginals: Vec<MarginalPosterior>) -> Self {
        Self {
            marginals,
            is_sorted_by_id: false,
        }
    }

    /// Sorts marginal candidates in descending order of probability in-place
    /// without heap allocations.
    pub fn sort_by_probability_desc(&mut self) {
        self.marginals.sort_unstable_by(|a, b| {
            OrderedFloat(b.probability.value())
                .cmp(&OrderedFloat(a.probability.value()))
        });
        self.is_sorted_by_id = false;
    }

    /// Sorts marginal candidates by identity identifier to enable binary
    /// search lookups.
    pub fn sort_by_identity(&mut self) {
        self.marginals.sort_unstable_by_key(|m| m.identity);
        self.is_sorted_by_id = true;
    }

    /// Searches for a specific identity marginal using binary search if
    /// pre-sorted, or linear search.
    ///
    /// # Arguments
    ///
    /// * `identity` - Target identity candidate identifier.
    /// * `is_sorted_by_id` - Flag indicating if marginals are sorted by
    ///   identity.
    pub fn find_marginal(
        &self,
        identity: IdentityId,
    ) -> Option<&MarginalPosterior> {
        if self.is_sorted_by_id {
            self.marginals
                .binary_search_by_key(&identity, |m| m.identity)
                .ok()
                .map(|idx| &self.marginals[idx])
        } else {
            self.marginals.iter().find(|m| m.identity == identity)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_posterior_sorting_performance() {
        let m1 = MarginalPosterior {
            identity: IdentityId(1),
            probability: Probability::new(0.3),
            log_odds: -0.84,
        };
        let m2 = MarginalPosterior {
            identity: IdentityId(2),
            probability: Probability::new(0.9),
            log_odds: 2.19,
        };
        let m3 = MarginalPosterior {
            identity: IdentityId(3),
            probability: Probability::new(0.5),
            log_odds: 0.0,
        };

        let mut dist = PosteriorDistribution::new(vec![m1, m3, m2]);
        dist.sort_by_probability_desc();

        assert_eq!(dist.marginals[0].identity, IdentityId(2));
        assert_eq!(dist.marginals[1].identity, IdentityId(3));
        assert_eq!(dist.marginals[2].identity, IdentityId(1));
    }

    #[test]
    fn test_binary_search_lookup() {
        let m1 = MarginalPosterior {
            identity: IdentityId(5),
            probability: Probability::new(0.8),
            log_odds: 1.38,
        };
        let m2 = MarginalPosterior {
            identity: IdentityId(12),
            probability: Probability::new(0.2),
            log_odds: -1.38,
        };

        let mut dist =
            PosteriorDistribution::new(vec![m2.clone(), m1.clone()]);
        dist.sort_by_identity();

        assert_eq!(dist.find_marginal(IdentityId(5)), Some(&m1));
        assert_eq!(dist.find_marginal(IdentityId(99)), None);
    }

    #[test]
    fn test_empty_distribution_edge_cases() {
        let mut dist = PosteriorDistribution::default();
        dist.sort_by_probability_desc();
        assert!(dist.marginals.is_empty());
        assert_eq!(dist.find_marginal(IdentityId(1)), None);
    }
}
