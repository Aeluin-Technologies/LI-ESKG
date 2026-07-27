//! Interface for compiling factor sets from incoming evidence packages.

use std::sync::Arc;

use li_core::belief::BeliefState;
use li_core::observation::Evidence;

use crate::compatibility::MultiCandidateCompatibility;
use crate::factor::{CategoricalFactor, Factor};

/// Abstract compiler translating evidence packages into active $k$-candidate
/// factor nodes.
pub trait FactorCompiler<P, S> {
    /// Constructs active factor potential nodes $\Phi_t = \{\phi_1, \dots,
    /// \phi_m\}$.
    ///
    /// # Arguments
    ///
    /// * `evidence` - Evidence package containing incoming observation and
    ///   candidate scope.
    /// * `active_beliefs` - Active belief states currently present in graph
    ///   storage.
    ///
    /// # Returns
    ///
    /// Vector of dynamic trait objects implementing [`Factor`].
    fn compile_factors(
        &self,
        evidence: &Evidence<P>,
        active_beliefs: &[BeliefState<S>],
    ) -> Vec<Box<dyn Factor>>;

    /// Filters active belief states using candidate blocking identifiers from
    /// evidence.
    ///
    /// # Arguments
    ///
    /// * `evidence` - Evidence package $E_t$.
    /// * `active_beliefs` - Full active belief states set.
    ///
    /// # Returns
    ///
    /// Filtered list of references to matching candidate belief states.
    fn filter_candidates<'a>(
        &self,
        evidence: &Evidence<P>,
        active_beliefs: &'a [BeliefState<S>],
    ) -> Vec<&'a BeliefState<S>> {
        if evidence.candidates.is_empty() {
            active_beliefs.iter().collect()
        } else {
            active_beliefs
                .iter()
                .filter(|b| evidence.candidates.contains(&b.identity))
                .collect()
        }
    }
}

/// Production compiler creating native [`CategoricalFactor`] nodes for
/// multi-candidate evidence.
pub struct CategoricalFactorCompiler<P, S> {
    evaluator: Arc<dyn MultiCandidateCompatibility<P, S>>,
}

impl<P, S> CategoricalFactorCompiler<P, S> {
    /// Creates a new [`CategoricalFactorCompiler`].
    ///
    /// # Arguments
    ///
    /// * `evaluator` - Thread-safe multi-candidate evaluator.
    pub fn new(evaluator: Arc<dyn MultiCandidateCompatibility<P, S>>) -> Self {
        Self { evaluator }
    }
}

impl<P: Send + Sync, S: Send + Sync> FactorCompiler<P, S>
    for CategoricalFactorCompiler<P, S>
{
    fn compile_factors(
        &self,
        evidence: &Evidence<P>,
        active_beliefs: &[BeliefState<S>],
    ) -> Vec<Box<dyn Factor>> {
        let candidate_beliefs =
            self.filter_candidates(evidence, active_beliefs);
        let mut factors: Vec<Box<dyn Factor>> = Vec::with_capacity(1);

        let distribution = self
            .evaluator
            .evaluate_joint(&evidence.observation, &candidate_beliefs);
        let (candidate_probs, bg_prob) = distribution.into_parts();

        if let Ok(factor) = CategoricalFactor::new(candidate_probs, bg_prob) {
            factors.push(Box::new(factor));
        }

        factors
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use li_core::ids::{IdentityId, ObservationId};
    use li_core::observation::{Modality, Observation, Timestamp};
    use li_core::probability::{Confidence, Probability};

    use super::*;
    use crate::compatibility::KCandidateDistribution;

    struct DummyEvaluator;

    impl MultiCandidateCompatibility<(), ()> for DummyEvaluator {
        fn evaluate_joint(
            &self,
            _observation: &Observation<()>,
            beliefs: &[&BeliefState<()>],
        ) -> KCandidateDistribution {
            let mut probs = HashMap::new();
            for b in beliefs {
                probs.insert(b.identity, Probability::new(0.5));
            }
            KCandidateDistribution::new(probs, Probability::new(0.1))
        }
    }

    fn mock_observation() -> Observation<()> {
        Observation::new(
            ObservationId(1),
            Modality(1),
            Timestamp::UNIX_EPOCH,
            Confidence::new(1.0),
            (),
        )
    }

    #[test]
    fn test_compiler_empty_candidates_uses_all_active_beliefs() {
        let evaluator = Arc::new(DummyEvaluator);
        let compiler = CategoricalFactorCompiler::new(evaluator);

        let evidence = Evidence::new(mock_observation(), vec![]);
        let beliefs = vec![BeliefState {
            identity: IdentityId(1),
            summary: (),
            posterior: Probability::new(0.8),
            last_update: Timestamp::UNIX_EPOCH,
        }];

        let compiled = compiler.compile_factors(&evidence, &beliefs);
        assert_eq!(compiled.len(), 1);
        assert_eq!(compiled[0].scope(), &[IdentityId(1)]);
    }

    #[test]
    fn test_compiler_no_active_beliefs_produces_no_factors() {
        let evaluator = Arc::new(DummyEvaluator);
        let compiler = CategoricalFactorCompiler::new(evaluator);

        let evidence = Evidence::new(mock_observation(), vec![]);
        let beliefs: Vec<BeliefState<()>> = vec![];

        let compiled = compiler.compile_factors(&evidence, &beliefs);
        assert_eq!(compiled.len(), 0);
    }

    #[test]
    fn test_compiler_filters_candidates_correctly() {
        let evaluator = Arc::new(DummyEvaluator);
        let compiler = CategoricalFactorCompiler::new(evaluator);

        let id1 = IdentityId(1);
        let id2 = IdentityId(2);

        let mut evidence = Evidence::new(mock_observation(), vec![id1]);

        let belief1 = BeliefState {
            identity: id1,
            summary: (),
            posterior: Probability::new(0.8),
            last_update: Timestamp::UNIX_EPOCH,
        };
        let belief2 = BeliefState {
            identity: id2,
            summary: (),
            posterior: Probability::new(0.8),
            last_update: Timestamp::UNIX_EPOCH,
        };

        let beliefs = vec![belief1, belief2];
        let filtered = compiler.filter_candidates(&evidence, &beliefs);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].identity, id1);

        evidence.candidates.clear();
        let unfiltered = compiler.filter_candidates(&evidence, &beliefs);
        assert_eq!(unfiltered.len(), 2);
    }
}
