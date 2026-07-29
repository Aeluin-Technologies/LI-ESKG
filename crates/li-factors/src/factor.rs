//! Abstract potential function traits for factor graph scope evaluations.

use std::collections::HashMap;

use li_core::ids::IdentityId;
use li_core::probability::Probability;
use thiserror::Error;

/// Errors occurring during factor graph scope evaluations and domain
/// assignments.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum FactorError {
    /// Error returned when an assignment length does not match expected scope
    /// bounds.
    #[error(
        "Scope length mismatch: factor expects {expected} variables, got {actual}"
    )]
    ScopeLengthMismatch { expected: usize, actual: usize },

    /// Error returned when an invalid variable index is queried.
    #[error("Invalid assignment index at position {index}")]
    InvalidAssignmentIndex { index: usize },

    /// Error returned when attempting to construct a factor with an empty
    /// scope.
    #[error("Factor domain scope cannot be empty")]
    EmptyScope,

    /// Error returned when duplicate identity identifiers are present in the
    /// scope.
    #[error("Duplicate identity detected in factor scope: {identity_id:?}")]
    DuplicateScopeIdentity { identity_id: IdentityId },

    /// Error returned when multiple candidates from the same scope are
    /// selected simultaneously.
    #[error(
        "Invalid candidate assignment: mutual exclusion constraint violated"
    )]
    MutualExclusionViolation,
}

/// Interface exposing the variable scope $Z_\phi$ associated with a factor
/// node.
pub trait FactorScope {
    /// Returns the sequence of identity identifiers forming the domain scope
    /// $Z_\phi$.
    fn scope(&self) -> &[IdentityId];

    /// Validates whether a proposed identity assignment slice conforms to the
    /// scope dimension.
    ///
    /// # Arguments
    ///
    /// * `assignment` - Slice of assigned identity identifiers to validate.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if valid, or a [`FactorError`] on failure.
    #[inline]
    fn validate_assignment(
        &self,
        _assignment: &[IdentityId],
    ) -> Result<(), FactorError> {
        if self.scope().is_empty() {
            Err(FactorError::EmptyScope)
        } else {
            Ok(())
        }
    }
}

/// Abstract potential function evaluating variable assignments over $k$
/// candidates.
pub trait Factor: FactorScope + Send + Sync {
    /// Evaluates the local potential score given a variable assignment.
    ///
    /// # Arguments
    ///
    /// * `assignment` - Active identity assignments for the current
    ///   observation hypothesis.
    ///
    /// # Returns
    ///
    /// The computed evaluation [`Probability`].
    fn evaluate(&self, assignment: &[IdentityId]) -> Probability;

    /// Evaluates the natural log-potential given a variable assignment.
    ///
    /// # Arguments
    ///
    /// * `assignment` - Active identity assignments for the current
    ///   observation hypothesis.
    ///
    /// # Returns
    ///
    /// The natural logarithm of the evaluated probability in $(-\infty, 0]$.
    #[inline]
    fn evaluate_log(&self, assignment: &[IdentityId]) -> f64 {
        self.evaluate(assignment).to_log()
    }

    /// Evaluates local potential score with scope boundary checking.
    ///
    /// # Arguments
    ///
    /// * `assignment` - Active identity assignments to evaluate.
    ///
    /// # Returns
    ///
    /// Result containing the evaluated [`Probability`] or a [`FactorError`].
    #[inline]
    fn evaluate_checked(
        &self,
        assignment: &[IdentityId],
    ) -> Result<Probability, FactorError> {
        self.validate_assignment(assignment)?;
        Ok(self.evaluate(assignment))
    }
}

/// Native $k$-candidate categorical factor enforcing mutual exclusion across
/// candidate identities.
#[derive(Debug)]
pub struct CategoricalFactor {
    scope: Vec<IdentityId>,
    candidate_probabilities: HashMap<IdentityId, Probability>,
    background_probability: Probability,
}

impl CategoricalFactor {
    /// Constructs a new [`CategoricalFactor`] node over k candidates.
    ///
    /// # Arguments
    ///
    /// * `candidates` - Map of candidate identities to their marginal
    ///   likelihoods.
    /// * `background_probability` - Residual probability assigned to
    ///   unassigned/noise state.
    ///
    /// # Returns
    ///
    /// Result containing the initialized [`CategoricalFactor`] or a
    /// [`FactorError`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut probs = HashMap::new();
    /// probs.insert(IdentityId(1), Probability::new(0.6));
    /// probs.insert(IdentityId(2), Probability::new(0.3));
    /// let factor = CategoricalFactor::new(probs, Probability::new(0.1));
    /// assert!(factor.is_ok());
    /// ```
    pub fn new(
        candidates: HashMap<IdentityId, Probability>,
        background_probability: Probability,
    ) -> Result<Self, FactorError> {
        if candidates.is_empty() {
            return Err(FactorError::EmptyScope);
        }

        let scope = candidates.keys().copied().collect();

        Ok(Self {
            scope,
            candidate_probabilities: candidates,
            background_probability,
        })
    }

    /// Returns the background/unassigned probability score.
    #[inline]
    pub fn background_probability(&self) -> Probability {
        self.background_probability
    }
}

impl FactorScope for CategoricalFactor {
    #[inline]
    fn scope(&self) -> &[IdentityId] {
        &self.scope
    }
}

impl Factor for CategoricalFactor {
    fn evaluate(&self, assignment: &[IdentityId]) -> Probability {
        let mut selected = None;
        for identity in assignment {
            let Some(probability) =
                self.candidate_probabilities.get(identity).copied()
            else {
                continue;
            };
            if selected.is_some() {
                return Probability::ZERO;
            }
            selected = Some(probability);
        }

        match selected {
            Some(probability) => probability,
            None => self.background_probability,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_categorical_factor_empty_scope() {
        let candidates = HashMap::new();
        let result = CategoricalFactor::new(candidates, Probability::new(0.1));
        assert_eq!(result.err(), Some(FactorError::EmptyScope));
    }

    #[test]
    fn test_categorical_factor_single_candidate_selection()
    -> Result<(), FactorError> {
        let id1 = IdentityId(1);
        let id2 = IdentityId(2);
        let mut candidates = HashMap::new();
        candidates.insert(id1, Probability::new(0.7));
        candidates.insert(id2, Probability::new(0.2));

        let factor =
            CategoricalFactor::new(candidates, Probability::new(0.1))?;

        assert_eq!(factor.evaluate(&[id1]), Probability::new(0.7));
        assert_eq!(factor.evaluate(&[id2]), Probability::new(0.2));
        Ok(())
    }

    #[test]
    fn test_categorical_factor_mutual_exclusion_violation()
    -> Result<(), FactorError> {
        let id1 = IdentityId(1);
        let id2 = IdentityId(2);
        let mut candidates = HashMap::new();
        candidates.insert(id1, Probability::new(0.5));
        candidates.insert(id2, Probability::new(0.4));

        let factor =
            CategoricalFactor::new(candidates, Probability::new(0.1))?;

        assert_eq!(factor.evaluate(&[id1, id2]), Probability::ZERO);
        Ok(())
    }

    #[test]
    fn test_categorical_factor_background_fallback() -> Result<(), FactorError>
    {
        let id1 = IdentityId(1);
        let id_out = IdentityId(99);
        let mut candidates = HashMap::new();
        candidates.insert(id1, Probability::new(0.8));

        let factor =
            CategoricalFactor::new(candidates, Probability::new(0.2))?;

        assert_eq!(factor.evaluate(&[]), Probability::new(0.2));
        assert_eq!(factor.evaluate(&[id_out]), Probability::new(0.2));
        Ok(())
    }

    #[test]
    fn test_categorical_factor_log_evaluation() -> Result<(), FactorError> {
        let id1 = IdentityId(1);
        let mut candidates = HashMap::new();
        candidates.insert(id1, Probability::new(1.0));

        let factor =
            CategoricalFactor::new(candidates, Probability::new(0.0))?;

        assert_eq!(factor.evaluate_log(&[id1]), 0.0);
        assert_eq!(factor.evaluate_log(&[]), f64::NEG_INFINITY);
        Ok(())
    }
}
