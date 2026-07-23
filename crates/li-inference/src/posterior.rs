//! Structures representing marginal and joint posterior distributions
//! $P(Z_t)$.

use alloc::vec::Vec;

use li_core::ids::IdentityId;
use li_core::probability::Probability;

/// Single variable marginal probability assignment $P(z_i = \text{true})$.
#[derive(Debug, Clone, PartialEq)]
pub struct MarginalPosterior {
    pub identity: IdentityId,
    pub probability: Probability,
}

/// Collection of calculated marginal posterior probabilities $P(Z_t)$.
#[derive(Debug, Clone, PartialEq)]
pub struct PosteriorDistribution {
    pub marginals: Vec<MarginalPosterior>,
}
