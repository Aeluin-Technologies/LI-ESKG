//! Interface for compiling factor sets from incoming evidence packages.

use alloc::boxed::Box;
use alloc::vec::Vec;

use li_core::belief::BeliefState;
use li_core::observation::Evidence;

use crate::factor::Factor;

/// Abstract compiler translating evidence packages into active factor nodes.
pub trait FactorCompiler<P, S> {
    /// Constructs the set of factor potential nodes $\Phi_t = \{\phi_1, \dots,
    /// \phi_m\}$.
    fn compile_factors(
        &self,
        evidence: &Evidence<P>,
        active_beliefs: &[BeliefState<S>],
    ) -> Vec<Box<dyn Factor>>;
}
