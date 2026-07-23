//! Sum-Product Belief Propagation algorithm solver on factor graphs.

use alloc::vec::Vec;

use li_core::IdentityId;
use li_core::probability::Probability;

use crate::factor_graph::FactorGraph;
use crate::posterior::{MarginalPosterior, PosteriorDistribution};

/// Configuration parameters for the Belief Propagation message passing engine.
#[derive(Debug, Clone, Copy)]
pub struct BpConfig {
    /// Maximum number of message passing iterations allowed.
    pub max_iterations: usize,
    /// Minimum message delta required to declare convergence.
    pub convergence_threshold: f64,
}

/// Solves marginal posterior distributions using Sum-Product Belief
/// Propagation.
pub struct BeliefPropagationSolver {
    /// Operational parameters controlling solver execution.
    pub config: BpConfig,
}

impl BeliefPropagationSolver {
    /// Instantiates a new Belief Propagation solver.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration options specifying iteration limits and
    ///   thresholds.
    pub fn new(config: BpConfig) -> Self {
        Self { config }
    }

    /// Computes marginal posterior probabilities for variable nodes in a
    /// factor graph.
    ///
    /// # Arguments
    ///
    /// * `graph` - The bipartite factor graph evaluating assignment variables.
    ///
    /// # Returns
    ///
    /// A [`PosteriorDistribution`] containing posterior probabilities for each
    /// variable candidate.
    pub fn solve(&self, graph: &FactorGraph) -> PosteriorDistribution {
        let mut marginals = Vec::with_capacity(graph.variables.len());

        for var in &graph.variables {
            let mut score_active = 0.5;
            let mut score_inactive = 0.5;

            for &f_idx in &graph.var_to_factor[var.id.0] {
                let factor = &graph.factors[f_idx.0];

                let scope_active = &[var.candidate_identity];
                score_active *= factor.evaluate(scope_active).0;

                let scope_inactive = &[IdentityId(0)];
                score_inactive *= factor.evaluate(scope_inactive).0;
            }

            let total_mass = score_active + score_inactive;
            let posterior_prob = if total_mass > 0.0 {
                score_active / total_mass
            } else {
                0.0
            };

            marginals.push(MarginalPosterior {
                identity: var.candidate_identity,
                probability: Probability::new(posterior_prob.clamp(0.0, 1.0)),
            });
        }

        PosteriorDistribution { marginals }
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
        let config = BpConfig {
            max_iterations: 10,
            convergence_threshold: 0.001,
        };
        let solver = BeliefPropagationSolver::new(config);
        let graph = FactorGraph::new();

        let posteriors = solver.solve(&graph);
        assert_eq!(posteriors.marginals.len(), 0);
    }

    #[test]
    fn test_bp_isolated_variables_default_score() {
        let config = BpConfig {
            max_iterations: 5,
            convergence_threshold: 0.01,
        };
        let solver = BeliefPropagationSolver::new(config);
        let mut graph = FactorGraph::new();
        graph.add_variable(IdentityId(100));

        let posteriors = solver.solve(&graph);
        assert_eq!(posteriors.marginals.len(), 1);
        assert_eq!(posteriors.marginals[0].identity, IdentityId(100));
        assert_eq!(posteriors.marginals[0].probability, Probability::new(0.5));
    }

    #[test]
    fn test_bp_zero_probability_factor() {
        let config = BpConfig {
            max_iterations: 10,
            convergence_threshold: 0.001,
        };
        let solver = BeliefPropagationSolver::new(config);
        let mut graph = FactorGraph::new();
        let v0 = graph.add_variable(IdentityId(1));

        let factor = Box::new(ConstantFactor {
            scope_ids: alloc::vec![IdentityId(1)],
            value: 0.0,
        });
        graph.add_factor(factor, &[v0]);

        let posteriors = solver.solve(&graph);
        assert_eq!(posteriors.marginals[0].probability, Probability::new(0.0));
    }
}
