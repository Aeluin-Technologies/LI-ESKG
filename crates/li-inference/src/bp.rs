//! Exact Sum-Product inference for one observation's categorical assignment.

use alloc::vec::Vec;

use li_core::IdentityId;
use li_core::probability::Probability;

use crate::factor_graph::FactorGraph;
use crate::posterior::{MarginalPosterior, PosteriorDistribution};

/// Configuration retained for compatibility with iterative BP deployments.
#[derive(Debug, Clone, Copy)]
pub struct BpConfig {
    /// Maximum iterations permitted by an iterative solver implementation.
    pub max_iterations: usize,
    /// Convergence tolerance used by an iterative solver implementation.
    pub convergence_threshold: f64,
}

/// Computes exact categorical marginals for a localized observation graph.
pub struct BeliefPropagationSolver {
    /// Operational settings shared with iterative solver implementations.
    pub config: BpConfig,
}

#[derive(Clone, Copy)]
struct LogMass {
    zero_factors: usize,
    log_value: f64,
}

impl LogMass {
    fn add(&mut self, probability: Probability) {
        if probability.0 > 0.0 {
            self.log_value += probability.0.ln();
        } else {
            self.zero_factors += 1;
        }
    }

    fn replace(&mut self, old: Probability, new: Probability) {
        if old.0 > 0.0 {
            self.log_value -= old.0.ln();
        } else {
            self.zero_factors -= 1;
        }
        self.add(new);
    }

    fn value(self) -> f64 {
        if self.zero_factors == 0 {
            self.log_value
        } else {
            f64::NEG_INFINITY
        }
    }
}

impl BeliefPropagationSolver {
    /// Creates a solver with the supplied operational configuration.
    pub fn new(config: BpConfig) -> Self {
        Self { config }
    }

    /// Computes exact candidate and new-identity posterior probabilities.
    ///
    /// Each factor supplies a potential for its active candidate values and an
    /// inactive potential for the `new identity` assignment. With one incoming
    /// observation this is a tree, so a single Sum-Product reduction produces
    /// the exact marginal without iterative message buffers.
    pub fn solve(&self, graph: &FactorGraph) -> PosteriorDistribution {
        let _ = self.config;
        let mut base_mass = LogMass {
            zero_factors: 0,
            log_value: 0.0,
        };

        for factor in &graph.factors {
            base_mass.add(factor.evaluate(&[IdentityId(0)]));
        }

        let mut candidate_masses = Vec::with_capacity(graph.variables.len());
        for _ in &graph.variables {
            candidate_masses.push(base_mass);
        }

        for factor_index in 0..graph.factors.len() {
            let factor = &graph.factors[factor_index];
            let inactive = factor.evaluate(&[IdentityId(0)]);

            for variable_index in graph
                .factor_scope(crate::factor_graph::FactorIndex(factor_index))
            {
                let candidate =
                    graph.variables[variable_index.0].candidate_identity;
                let active =
                    factor.evaluate(core::slice::from_ref(&candidate));
                candidate_masses[variable_index.0].replace(inactive, active);
            }
        }

        let new_identity_log_mass = base_mass.value();
        let mut max_log_mass = new_identity_log_mass;
        for mass in &candidate_masses {
            max_log_mass = max_log_mass.max(mass.value());
        }

        if !max_log_mass.is_finite() {
            return PosteriorDistribution {
                marginals: graph
                    .variables
                    .iter()
                    .map(|variable| MarginalPosterior {
                        identity: variable.candidate_identity,
                        probability: Probability::new(0.0),
                    })
                    .collect(),
                new_identity_probability: Probability::new(1.0),
            };
        }

        let mut normalizer =
            normalized_mass(new_identity_log_mass, max_log_mass);
        for mass in &candidate_masses {
            normalizer += normalized_mass(mass.value(), max_log_mass);
        }

        let inverse_normalizer = if normalizer > 0.0 {
            normalizer.recip()
        } else {
            0.0
        };
        let mut marginals = Vec::with_capacity(graph.variables.len());
        for (variable, mass) in graph.variables.iter().zip(candidate_masses) {
            marginals.push(MarginalPosterior {
                identity: variable.candidate_identity,
                probability: Probability::new(
                    normalized_mass(mass.value(), max_log_mass) *
                        inverse_normalizer,
                ),
            });
        }

        PosteriorDistribution {
            marginals,
            new_identity_probability: Probability::new(
                normalized_mass(new_identity_log_mass, max_log_mass) *
                    inverse_normalizer,
            ),
        }
    }
}

fn normalized_mass(log_mass: f64, max_log_mass: f64) -> f64 {
    if log_mass.is_finite() {
        (log_mass - max_log_mass).exp()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;

    use li_core::ids::IdentityId;
    use li_core::probability::Probability;

    use crate::bp::{BeliefPropagationSolver, BpConfig};
    use crate::factor_graph::FactorGraph;
    use crate::factor_graph::tests::ConstantFactor;

    #[test]
    fn test_bp_empty_graph() {
        let solver = BeliefPropagationSolver::new(BpConfig {
            max_iterations: 10,
            convergence_threshold: 0.001,
        });

        let posteriors = solver.solve(&FactorGraph::new());

        assert!(posteriors.marginals.is_empty());
        assert_eq!(posteriors.new_identity_probability, Probability::new(1.0));
    }

    #[test]
    fn test_bp_isolated_candidate_is_uniform_with_new_identity() {
        let solver = BeliefPropagationSolver::new(BpConfig {
            max_iterations: 5,
            convergence_threshold: 0.01,
        });
        let graph = FactorGraph::from_candidates(&[IdentityId(100)]);

        let posteriors = solver.solve(&graph);

        assert_eq!(posteriors.marginals.len(), 1);
        assert_eq!(posteriors.marginals[0].identity, IdentityId(100));
        assert_eq!(posteriors.marginals[0].probability, Probability::new(0.5));
        assert_eq!(posteriors.new_identity_probability, Probability::new(0.5));
    }

    #[test]
    fn test_bp_normalizes_candidates_against_new_identity() {
        let solver = BeliefPropagationSolver::new(BpConfig {
            max_iterations: 10,
            convergence_threshold: 0.001,
        });
        let mut graph = FactorGraph::from_candidates(&[IdentityId(1)]);
        let factor = Box::new(ConstantFactor {
            scope_ids: alloc::vec![IdentityId(1)],
            value: 0.0,
        });
        let factor_index = graph.add_factor(factor);

        assert_eq!(factor_index.map(|index| index.0), Some(0));

        let posteriors = solver.solve(&graph);

        assert_eq!(posteriors.marginals[0].probability, Probability::new(0.0));
        assert_eq!(posteriors.new_identity_probability, Probability::new(1.0));
    }
}
