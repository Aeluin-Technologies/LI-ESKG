//! Abstract potential function traits for factor graph scope evaluations.

use li_core::ids::IdentityId;
use li_core::probability::Probability;

/// Interface exposing the variable scope $Z_\phi$ associated with a factor
/// node.
pub trait FactorScope {
    /// Returns the sequence of identity identifiers forming the domain scope
    /// $Z_\phi$.
    fn scope(&self) -> &[IdentityId];
}

/// Abstract potential function $\phi_i(Z_\phi)$ evaluating variable
/// assignments.
pub trait Factor: FactorScope + Send + Sync {
    /// Evaluates the local potential score given a variable assignment.
    fn evaluate(&self, assignment: &[IdentityId]) -> Probability;
}
