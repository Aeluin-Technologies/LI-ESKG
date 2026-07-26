//! Abstract potential function traits for factor graph scope evaluations.

use li_core::ids::IdentityId;
use li_core::probability::Probability;
use thiserror::Error;

/// Errors occurring during factor graph scope evaluations and domain
/// assignments.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum FactorError {
    #[error(
        "Scope length mismatch: factor expects {expected} variables, got {actual}"
    )]
    ScopeLengthMismatch { expected: usize, actual: usize },

    #[error("Invalid assignment index at position {index}")]
    InvalidAssignmentIndex { index: usize },

    #[error("Factor domain scope cannot be empty")]
    EmptyScope,
}

/// Interface exposing the variable scope $Z_\phi$ associated with a factor
/// node.
pub trait FactorScope {
    /// Returns the sequence of identity identifiers forming the domain scope
    /// $Z_\phi$.
    fn scope(&self) -> &[IdentityId];

    /// Validates whether a proposed identity assignment slice conforms to the
    /// scope dimension.
    #[inline]
    fn validate_assignment(
        &self,
        assignment: &[IdentityId],
    ) -> Result<(), FactorError> {
        let expected = self.scope().len();
        let actual = assignment.len();
        if expected != actual {
            Err(FactorError::ScopeLengthMismatch { expected, actual })
        } else {
            Ok(())
        }
    }
}

/// Abstract potential function $\phi_i(Z_\phi)$ evaluating variable
/// assignments.
pub trait Factor: FactorScope + Send + Sync {
    /// Evaluates the local potential score given a variable assignment.
    fn evaluate(&self, assignment: &[IdentityId]) -> Probability;

    /// Evaluates the natural log-potential $\ln \phi_i(Z_\phi)$ given a
    /// variable assignment.
    #[inline]
    fn evaluate_log(&self, assignment: &[IdentityId]) -> f64 {
        self.evaluate(assignment).to_log()
    }

    /// Evaluates local potential score with scope boundary checking.
    #[inline]
    fn evaluate_checked(
        &self,
        assignment: &[IdentityId],
    ) -> Result<Probability, FactorError> {
        self.validate_assignment(assignment)?;
        Ok(self.evaluate(assignment))
    }
}

/// Concrete factor implementation representing pairwise
/// observation-to-identity potentials.
pub struct PairwiseFactor {
    // Avoid heap allocation by using a fixed-size array for single-item
    // scopes
    scope: [IdentityId; 1],
    compatibility_score: Probability,
}

impl PairwiseFactor {
    /// Creates a new [`PairwiseFactor`] node.
    pub fn new(
        target_identity: IdentityId,
        compatibility_score: Probability,
    ) -> Self {
        Self {
            scope: [target_identity],
            compatibility_score,
        }
    }
}

impl FactorScope for PairwiseFactor {
    #[inline]
    fn scope(&self) -> &[IdentityId] {
        &self.scope
    }
}

impl Factor for PairwiseFactor {
    #[inline]
    fn evaluate(&self, assignment: &[IdentityId]) -> Probability {
        if assignment == self.scope {
            self.compatibility_score
        } else {
            Probability::ZERO
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pairwise_factor_scope_validation() {
        let id = IdentityId(1);
        let factor = PairwiseFactor::new(id, Probability::new(0.8));

        assert_eq!(factor.scope(), &[id]);
        assert!(factor.validate_assignment(&[id]).is_ok());

        let err = factor.validate_assignment(&[]).unwrap_err();
        assert_eq!(
            err,
            FactorError::ScopeLengthMismatch {
                expected: 1,
                actual: 0
            }
        );
    }

    #[test]
    fn test_checked_evaluation() {
        let id1 = IdentityId(1);
        let id2 = IdentityId(2);
        let score = Probability::new(0.95);
        let factor = PairwiseFactor::new(id1, score);

        assert_eq!(factor.evaluate_checked(&[id1]).unwrap(), score);
        assert_eq!(
            factor.evaluate_checked(&[id2]).unwrap(),
            Probability::ZERO
        );
        assert!(factor.evaluate_checked(&[id1, id2]).is_err());
    }
}
