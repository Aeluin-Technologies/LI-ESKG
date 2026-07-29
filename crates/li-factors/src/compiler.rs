//! Interface for compiling factor sets from incoming evidence packages.

use std::sync::Arc;

use li_core::belief::BeliefState;
use li_core::ids::IdentityId;
use li_core::observation::Evidence;
use li_core::probability::Probability;

use crate::compatibility::MultiCandidateCompatibility;
use crate::factor::{CategoricalFactor, Factor};

/// Result of an analytic MAP shortcut supplied by a factor compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectMapDecision {
    /// The compiler cannot evaluate this factor family analytically.
    Unsupported,
    /// The observation is assigned to an existing identity.
    Assign(IdentityId),
    /// The background hypothesis wins and a new identity must be created.
    CreateIdentity,
}

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
        active_beliefs: &[&BeliefState<S>],
    ) -> Vec<Box<dyn Factor>>;

    /// Attempts an exact MAP decision without materializing a factor graph.
    ///
    /// Compilers should implement this method only when their factor family
    /// has a closed-form MAP solution equivalent to generic inference.
    /// [`DirectMapDecision::Assign`] must reference one of `active_beliefs`.
    ///
    /// # Arguments
    ///
    /// * `evidence` - Evidence package containing the current observation.
    /// * `active_beliefs` - Borrowed candidate beliefs resolved by the caller.
    /// * `decision_threshold` - Minimum probability required for assignment.
    ///
    /// # Returns
    ///
    /// A direct decision, or [`DirectMapDecision::Unsupported`] when generic
    /// factor-graph inference is required.
    fn try_direct_map(
        &self,
        _evidence: &Evidence<P>,
        _active_beliefs: &[&BeliefState<S>],
        _decision_threshold: Probability,
    ) -> DirectMapDecision {
        DirectMapDecision::Unsupported
    }

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
        active_beliefs: &[&'a BeliefState<S>],
    ) -> Vec<&'a BeliefState<S>> {
        if evidence.candidates.is_empty() {
            active_beliefs.to_vec()
        } else {
            active_beliefs
                .iter()
                .copied()
                .filter(|belief| {
                    evidence.candidates.contains(&belief.identity)
                })
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
        active_beliefs: &[&BeliefState<S>],
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

    fn try_direct_map(
        &self,
        evidence: &Evidence<P>,
        active_beliefs: &[&BeliefState<S>],
        decision_threshold: Probability,
    ) -> DirectMapDecision {
        if active_beliefs.is_empty() {
            return DirectMapDecision::CreateIdentity;
        }

        let threshold = decision_threshold.value();
        let mut best: Option<(IdentityId, f64)> = None;

        let mut select = |identity: IdentityId, probability: Probability| {
            let score = probability.value();
            if score < threshold {
                return;
            }

            let replace = match best {
                Some((best_identity, best_score)) => {
                    score > best_score ||
                        (score == best_score && identity < best_identity)
                },
                None => true,
            };
            if replace {
                best = Some((identity, score));
            }
        };
        let background = self
            .evaluator
            .evaluate_joint_stream(
                &evidence.observation,
                active_beliefs,
                &mut select,
            )
            .value();

        match best {
            Some((identity, score)) if score > background => {
                DirectMapDecision::Assign(identity)
            },
            _ => DirectMapDecision::CreateIdentity,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

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

    struct BackgroundEvaluator;

    impl MultiCandidateCompatibility<(), ()> for BackgroundEvaluator {
        fn evaluate_joint(
            &self,
            _observation: &Observation<()>,
            beliefs: &[&BeliefState<()>],
        ) -> KCandidateDistribution {
            let mut probs = HashMap::with_capacity(beliefs.len());
            for belief in beliefs {
                probs.insert(belief.identity, Probability::new(0.2));
            }
            KCandidateDistribution::new(probs, Probability::new(0.8))
        }
    }

    struct StreamingEvaluator {
        materialized_calls: Arc<AtomicUsize>,
    }

    impl MultiCandidateCompatibility<(), ()> for StreamingEvaluator {
        fn evaluate_joint(
            &self,
            _observation: &Observation<()>,
            _beliefs: &[&BeliefState<()>],
        ) -> KCandidateDistribution {
            self.materialized_calls.fetch_add(1, Ordering::Relaxed);
            KCandidateDistribution::new(HashMap::new(), Probability::ONE)
        }

        fn evaluate_joint_stream(
            &self,
            _observation: &Observation<()>,
            beliefs: &[&BeliefState<()>],
            emit: &mut dyn FnMut(IdentityId, Probability),
        ) -> Probability {
            for belief in beliefs {
                emit(belief.identity, Probability::new(0.9));
            }
            Probability::new(0.1)
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
        let beliefs = [BeliefState {
            identity: IdentityId(1),
            summary: (),
            posterior: Probability::new(0.8),
            last_update: Timestamp::UNIX_EPOCH,
        }];

        let belief_refs = beliefs.iter().collect::<Vec<_>>();
        let compiled = compiler.compile_factors(&evidence, &belief_refs);
        assert_eq!(compiled.len(), 1);
        assert_eq!(compiled[0].scope(), &[IdentityId(1)]);
    }

    #[test]
    fn test_compiler_no_active_beliefs_produces_no_factors() {
        let evaluator = Arc::new(DummyEvaluator);
        let compiler = CategoricalFactorCompiler::new(evaluator);

        let evidence = Evidence::new(mock_observation(), vec![]);
        let beliefs: Vec<BeliefState<()>> = vec![];

        let belief_refs = beliefs.iter().collect::<Vec<_>>();
        let compiled = compiler.compile_factors(&evidence, &belief_refs);
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

        let beliefs = [belief1, belief2];
        let belief_refs = beliefs.iter().collect::<Vec<_>>();
        let filtered = compiler.filter_candidates(&evidence, &belief_refs);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].identity, id1);

        evidence.candidates.clear();
        let unfiltered = compiler.filter_candidates(&evidence, &belief_refs);
        assert_eq!(unfiltered.len(), 2);
    }

    #[test]
    fn test_direct_map_is_deterministic_for_equal_probabilities() {
        let compiler =
            CategoricalFactorCompiler::new(Arc::new(DummyEvaluator));
        let evidence = Evidence::new(
            mock_observation(),
            vec![IdentityId(9), IdentityId(3)],
        );
        let first = BeliefState {
            identity: IdentityId(9),
            summary: (),
            posterior: Probability::new(0.5),
            last_update: Timestamp::UNIX_EPOCH,
        };
        let second = BeliefState {
            identity: IdentityId(3),
            summary: (),
            posterior: Probability::new(0.5),
            last_update: Timestamp::UNIX_EPOCH,
        };
        let beliefs = [&first, &second];

        assert_eq!(
            compiler.try_direct_map(
                &evidence,
                &beliefs,
                Probability::new(0.4),
            ),
            DirectMapDecision::Assign(IdentityId(3))
        );
    }

    #[test]
    fn test_direct_map_prefers_background_hypothesis() {
        let compiler =
            CategoricalFactorCompiler::new(Arc::new(BackgroundEvaluator));
        let evidence = Evidence::new(mock_observation(), vec![IdentityId(1)]);
        let belief = BeliefState {
            identity: IdentityId(1),
            summary: (),
            posterior: Probability::new(0.5),
            last_update: Timestamp::UNIX_EPOCH,
        };

        assert_eq!(
            compiler.try_direct_map(
                &evidence,
                &[&belief],
                Probability::new(0.1),
            ),
            DirectMapDecision::CreateIdentity
        );
    }

    #[test]
    fn test_direct_map_respects_decision_threshold() {
        let compiler =
            CategoricalFactorCompiler::new(Arc::new(DummyEvaluator));
        let evidence = Evidence::new(mock_observation(), vec![IdentityId(1)]);
        let belief = BeliefState {
            identity: IdentityId(1),
            summary: (),
            posterior: Probability::new(0.5),
            last_update: Timestamp::UNIX_EPOCH,
        };

        assert_eq!(
            compiler.try_direct_map(
                &evidence,
                &[&belief],
                Probability::new(0.6),
            ),
            DirectMapDecision::CreateIdentity
        );
    }

    #[test]
    fn test_direct_map_uses_allocation_free_streaming_evaluator() {
        let materialized_calls = Arc::new(AtomicUsize::new(0));
        let compiler =
            CategoricalFactorCompiler::new(Arc::new(StreamingEvaluator {
                materialized_calls: Arc::clone(&materialized_calls),
            }));
        let evidence = Evidence::new(mock_observation(), vec![IdentityId(5)]);
        let belief = BeliefState {
            identity: IdentityId(5),
            summary: (),
            posterior: Probability::new(0.5),
            last_update: Timestamp::UNIX_EPOCH,
        };

        assert_eq!(
            compiler.try_direct_map(
                &evidence,
                &[&belief],
                Probability::new(0.8),
            ),
            DirectMapDecision::Assign(IdentityId(5))
        );
        assert_eq!(materialized_calls.load(Ordering::Relaxed), 0);
    }
}
