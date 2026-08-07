//! Bayes-risk policy actions and durable decision records.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ids::{DecisionId, InferenceId};
use crate::inference::{
    AssociationOutcome, IdentityReference, NormalizedDistribution,
    VersionInterval,
};

/// Operational action selected separately from probabilistic inference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionAction {
    /// Assign evidence to a gated latent or known identity.
    Assign(IdentityReference),
    /// Create a new latent identity hypothesis.
    CreateIdentity,
    /// Reject evidence as noise or non-entity input.
    RejectNoise,
    /// Defer commitment for review or later evidence.
    Abstain,
}

/// Error returned by a malformed or incomplete decision policy.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum DecisionError {
    /// A loss was NaN, infinite, or negative.
    #[error("loss value {field} must be finite and non-negative")]
    InvalidLoss {
        /// Stable name of the rejected loss field.
        field: &'static str,
    },
    /// An assignment selected a target outside the inference domain.
    #[error(
        "selected target was not present in the supporting inference domain"
    )]
    TargetOutsideDomain,
}

/// Explicit one-step loss model used by the default Bayes-risk policy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LossModel {
    false_assignment: f64,
    false_new: f64,
    false_rejection: f64,
    abstention: f64,
}

impl LossModel {
    /// Creates a validated non-negative loss model.
    ///
    /// # Arguments
    ///
    /// * `false_assignment` - Cost when an identity assignment is wrong.
    /// * `false_new` - Cost when a new identity is created incorrectly.
    /// * `false_rejection` - Cost when valid evidence is rejected as noise.
    /// * `abstention` - Review or delay cost of abstention.
    ///
    /// # Errors
    ///
    /// Returns [`DecisionError::InvalidLoss`] for non-finite or negative
    /// costs.
    pub fn new(
        false_assignment: f64,
        false_new: f64,
        false_rejection: f64,
        abstention: f64,
    ) -> Result<Self, DecisionError> {
        let values = [
            ("false_assignment", false_assignment),
            ("false_new", false_new),
            ("false_rejection", false_rejection),
            ("abstention", abstention),
        ];
        for (field, value) in values {
            if !value.is_finite() || value < 0.0 {
                return Err(DecisionError::InvalidLoss { field });
            }
        }
        Ok(Self {
            false_assignment,
            false_new,
            false_rejection,
            abstention,
        })
    }

    /// Computes expected one-step loss for an action and normalized posterior.
    pub fn expected_loss(
        self,
        action: &DecisionAction,
        distribution: &NormalizedDistribution,
    ) -> f64 {
        if matches!(action, DecisionAction::Abstain) {
            return self.abstention;
        }

        distribution
            .entries()
            .iter()
            .fold(0.0, |accumulator, entry| {
                let correct = match (action, &entry.outcome) {
                    (
                        DecisionAction::Assign(selected),
                        AssociationOutcome::Identity(actual),
                    ) => selected == actual,
                    (
                        DecisionAction::CreateIdentity,
                        AssociationOutcome::New,
                    ) |
                    (
                        DecisionAction::RejectNoise,
                        AssociationOutcome::Noise,
                    ) => true,
                    _ => false,
                };
                if correct {
                    accumulator
                } else {
                    let loss = match action {
                        DecisionAction::Assign(_) => self.false_assignment,
                        DecisionAction::CreateIdentity => self.false_new,
                        DecisionAction::RejectNoise => self.false_rejection,
                        DecisionAction::Abstain => self.abstention,
                    };
                    entry.probability.value().mul_add(loss, accumulator)
                }
            })
    }
}

/// Deterministic myopic Bayes-risk decision policy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BayesRiskPolicy {
    /// Stable policy implementation/configuration version.
    pub policy_version: u64,
    /// Stable loss model or operating-point version.
    pub loss_version: u64,
    /// Explicit loss values used by the policy.
    pub loss: LossModel,
}

impl BayesRiskPolicy {
    /// Selects the minimum-risk action with a stable tie order.
    ///
    /// Identity assignments follow canonical distribution order, followed by
    /// new identity, noise rejection, and abstention. This ordering makes
    /// deterministic replay independent of hash-map iteration order.
    pub fn decide(
        self,
        distribution: &NormalizedDistribution,
    ) -> DecisionAction {
        let mut best = DecisionAction::CreateIdentity;
        let mut best_loss = self.loss.expected_loss(&best, distribution);

        for entry in distribution.entries() {
            let AssociationOutcome::Identity(reference) = &entry.outcome
            else {
                continue;
            };
            let candidate = DecisionAction::Assign(reference.clone());
            let risk = self.loss.expected_loss(&candidate, distribution);
            if risk < best_loss {
                best = candidate;
                best_loss = risk;
            }
        }

        for candidate in [DecisionAction::RejectNoise, DecisionAction::Abstain]
        {
            let risk = self.loss.expected_loss(&candidate, distribution);
            if risk < best_loss {
                best = candidate;
                best_loss = risk;
            }
        }
        best
    }
}

/// Immutable policy result linked to one durable inference record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionRecord {
    /// Stable decision record identifier.
    pub id: DecisionId,
    /// Supporting durable inference record.
    pub inference: InferenceId,
    /// Selected operational action.
    pub action: DecisionAction,
    /// Policy implementation/configuration version.
    pub policy_version: u64,
    /// Loss-model or documented operating-point version.
    pub loss_version: u64,
    /// Durable validity interval.
    pub validity: VersionInterval,
}

impl DecisionRecord {
    /// Creates a decision proven to belong to the supporting domain.
    ///
    /// # Errors
    ///
    /// Returns [`DecisionError::TargetOutsideDomain`] if an assignment target
    /// is absent from `distribution`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: DecisionId,
        inference: InferenceId,
        action: DecisionAction,
        policy_version: u64,
        loss_version: u64,
        validity: VersionInterval,
        distribution: &NormalizedDistribution,
    ) -> Result<Self, DecisionError> {
        if let DecisionAction::Assign(target) = &action {
            let outcome = AssociationOutcome::Identity(target.clone());
            if distribution.probability(&outcome).is_none() {
                return Err(DecisionError::TargetOutsideDomain);
            }
        }
        Ok(Self {
            id,
            inference,
            action,
            policy_version,
            loss_version,
            validity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::{CommitVersion, IdentityId};

    // Local helpers live in a nested module to keep test construction explicit
    // without exposing unchecked constructors to production code.
    mod li_core_test_support {
        use crate::ids::IdentityId;
        use crate::inference::{
            AssociationOutcome, IdentityReference, NormalizedDistribution,
            OutcomeProbability,
        };
        use crate::probability::Probability;

        pub fn distribution()
        -> Result<NormalizedDistribution, crate::inference::DistributionError>
        {
            let entries = vec![
                OutcomeProbability {
                    outcome: AssociationOutcome::Identity(
                        IdentityReference::Latent(IdentityId(1)),
                    ),
                    probability: Probability::new(0.7),
                },
                OutcomeProbability {
                    outcome: AssociationOutcome::New,
                    probability: Probability::new(0.2),
                },
                OutcomeProbability {
                    outcome: AssociationOutcome::Noise,
                    probability: Probability::new(0.1),
                },
            ];
            NormalizedDistribution::new(entries, None)
        }
    }

    #[test]
    fn loss_model_rejects_invalid_costs() {
        assert!(LossModel::new(f64::NAN, 1.0, 1.0, 1.0).is_err());
        assert!(LossModel::new(-1.0, 1.0, 1.0, 1.0).is_err());
        assert!(LossModel::new(10.0, 5.0, 5.0, 1.0).is_ok());
    }

    #[test]
    fn bayes_risk_can_abstain_when_false_assignment_is_expensive() {
        let loss = LossModel::new(100.0, 100.0, 100.0, 1.0);
        assert!(loss.is_ok());
        if let Ok(loss) = loss {
            let policy = BayesRiskPolicy {
                policy_version: 2,
                loss_version: 4,
                loss,
            };
            let distribution = li_core_test_support::distribution();
            assert!(distribution.is_ok());
            if let Ok(distribution) = distribution {
                assert_eq!(
                    policy.decide(&distribution),
                    DecisionAction::Abstain
                );
            }
        }
    }

    #[test]
    fn decision_record_rejects_targets_outside_domain() {
        let distribution = li_core_test_support::distribution();
        assert!(distribution.is_ok());
        let Ok(distribution) = distribution else {
            return;
        };
        let decision = DecisionRecord::new(
            DecisionId(1),
            InferenceId(2),
            DecisionAction::Assign(IdentityReference::Latent(IdentityId(99))),
            1,
            1,
            VersionInterval::current(CommitVersion::new(1)),
            &distribution,
        );
        assert_eq!(decision, Err(DecisionError::TargetOutsideDomain));
    }
}
