//! Structures representing active tracking configurations within the ephemeral
//! layer.

use serde::{Deserialize, Serialize};

use crate::ids::IdentityId;
use crate::observation::Timestamp;
use crate::probability::Probability;

/// State representation of a tracking hypothesis inside the active layer.
/// Matches the theoretical formulation $b_i = (\theta, \Sigma, \Lambda)$.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BeliefState<S> {
    /// Target latent identity identifier.
    pub identity: IdentityId,
    /// Modality-agnostic rolling statistical summary data.
    pub summary: S,
    /// Current calculated marginal posterior probability value.
    pub posterior: Probability,
    /// Temporal marker of the latest update or reinforcement.
    pub last_update: Timestamp,
}

impl<S> BeliefState<S> {
    /// Instantiates a new active belief state structure.
    pub fn new(
        identity: IdentityId,
        summary: S,
        posterior: Probability,
        last_update: Timestamp,
    ) -> Self {
        Self {
            identity,
            summary,
            posterior,
            last_update,
        }
    }

    /// Updates the marginal posterior value and update timestamp.
    ///
    /// # Arguments
    ///
    /// * `posterior` - New posterior probability score.
    /// * `timestamp` - Microsecond update timestamp.
    pub fn update_posterior(
        &mut self,
        posterior: Probability,
        timestamp: Timestamp,
    ) {
        self.posterior = posterior;
        self.last_update = timestamp;
    }
}
