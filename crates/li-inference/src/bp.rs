//! Parallel Log-Domain Sum-Product Belief Propagation solver robust against
//! NaN and infinite values.

use li_core::ids::IdentityId;
use li_core::probability::Probability;
use rayon::prelude::*;

use crate::factor_graph::FactorGraph;
use crate::posterior::{MarginalPosterior, PosteriorDistribution};

/// Configuration parameters for the Belief Propagation message passing engine.
#[derive(Debug, Clone, Copy)]
pub struct BpConfig {
    /// Maximum number of message passing iterations allowed.
    pub max_iterations: usize,
    /// Minimum message delta threshold required to declare convergence.
    pub convergence_threshold: f64,
}

impl Default for BpConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            convergence_threshold: 1e-6,
        }
    }
}

/// Solves marginal posterior distributions using Log-Domain Sum-Product Belief
/// Propagation.
pub struct BeliefPropagationSolver {
    /// Operational parameters controlling execution limits and convergence
    /// criteria.
    pub config: BpConfig,
}

impl BeliefPropagationSolver {
    /// Instantiates a new Belief Propagation solver.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration options controlling iteration limits and
    ///   convergence.
    pub fn new(config: BpConfig) -> Self {
        Self { config }
    }

    /// Computes LogSumExp over logarithmic values with explicit checks
    /// preventing NaN propagation.
    pub fn log_sum_exp(logs: &[f64]) -> f64 {
        if logs.is_empty() {
            return f64::NEG_INFINITY;
        }

        let mut max_val = f64::NEG_INFINITY;
        let mut has_pos_infinity = false;

        for &val in logs {
            if val == f64::INFINITY {
                has_pos_infinity = true;
            } else if val > max_val {
                max_val = val;
            }
        }

        if has_pos_infinity {
            return f64::INFINITY;
        }

        if max_val == f64::NEG_INFINITY {
            return f64::NEG_INFINITY;
        }

        let sum_exp: f64 = logs
            .iter()
            .map(|&x| {
                if x == f64::NEG_INFINITY {
                    0.0
                } else {
                    (x - max_val).exp()
                }
            })
            .sum();

        max_val + sum_exp.ln()
    }

    /// Solves marginal posterior probabilities for variable nodes in a factor
    /// graph.
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
        let num_vars = graph.variables.len();
        let num_factors = graph.factors.len();

        if num_vars == 0 {
            return PosteriorDistribution::default();
        }

        let mut var_to_factor_msg: Vec<Vec<[f64; 2]>> = graph
            .var_adjacencies
            .iter()
            .map(|f_list| vec![[0.0, 0.0]; f_list.len()])
            .collect();

        let mut factor_to_var_msg: Vec<Vec<[f64; 2]>> = graph
            .factor_adjacencies
            .iter()
            .map(|v_list| vec![[0.0, 0.0]; v_list.len()])
            .collect();

        for _iteration in 0..self.config.max_iterations {
            let mut max_delta: f64 = 0.0;

            for (v_idx, v_msgs) in
                var_to_factor_msg.iter_mut().enumerate().take(num_vars)
            {
                let adjacencies = &graph.var_adjacencies[v_idx];
                for (f_pos, _edge) in adjacencies.iter().enumerate() {
                    for state in 0..2 {
                        let mut sum_log = 0.0;
                        for (other_f_pos, other_edge) in
                            adjacencies.iter().enumerate()
                        {
                            if other_f_pos == f_pos {
                                continue;
                            }
                            sum_log += factor_to_var_msg
                                [other_edge.factor_idx.0]
                                [other_edge.pos_in_factor][state];
                        }
                        v_msgs[f_pos][state] = sum_log;
                    }

                    let max_m = v_msgs[f_pos][0].max(v_msgs[f_pos][1]);
                    if max_m.is_finite() {
                        v_msgs[f_pos][0] -= max_m;
                        v_msgs[f_pos][1] -= max_m;
                    }
                }
            }

            let next_factor_to_var_msg: Vec<Vec<[f64; 2]>> = (0..num_factors)
                .into_par_iter()
                .map(|f_idx| {
                    let factor = &graph.factors[f_idx];
                    let scoped_edges = &graph.factor_adjacencies[f_idx];
                    let num_scoped = scoped_edges.len();
                    debug_assert!(
                        num_scoped < 30,
                        "Factor scope too large for dense enumeration"
                    );
                    let mut new_msgs = vec![[0.0, 0.0]; num_scoped];

                    for (target_pos, _target_edge) in
                        scoped_edges.iter().enumerate()
                    {
                        for (target_state, target_msg_slot) in
                            new_msgs[target_pos].iter_mut().enumerate()
                        {
                            let mut combination_logs = Vec::with_capacity(
                                1 << num_scoped.saturating_sub(1),
                            );
                            let num_combinations = 1 << num_scoped;

                            let mut assignments_buf =
                                vec![IdentityId(0); num_scoped];

                            for comb in 0..num_combinations {
                                let current_target_state =
                                    (comb >> target_pos) & 1;
                                if current_target_state != target_state {
                                    continue;
                                }

                                assignments_buf.clear();
                                let mut incoming_sum = 0.0;

                                for (other_pos, other_edge) in
                                    scoped_edges.iter().enumerate()
                                {
                                    let state = (comb >> other_pos) & 1;
                                    if state == 1 {
                                        assignments_buf.push(
                                            graph.variables
                                                [other_edge.var_idx.0]
                                                .candidate_identity,
                                        );
                                    }
                                    if other_pos != target_pos {
                                        incoming_sum += var_to_factor_msg
                                            [other_edge.var_idx.0]
                                            [other_edge.pos_in_var][state];
                                    }
                                }

                                let factor_prob = factor
                                    .evaluate(&assignments_buf)
                                    .value()
                                    .max(1e-12);
                                combination_logs
                                    .push(factor_prob.ln() + incoming_sum);
                            }

                            *target_msg_slot =
                                Self::log_sum_exp(&combination_logs);
                        }

                        let max_m = new_msgs[target_pos][0]
                            .max(new_msgs[target_pos][1]);
                        if max_m.is_finite() {
                            new_msgs[target_pos][0] -= max_m;
                            new_msgs[target_pos][1] -= max_m;
                        }
                    }

                    new_msgs
                })
                .collect();

            for f_idx in 0..num_factors {
                for (v_pos, msgs) in
                    next_factor_to_var_msg[f_idx].iter().enumerate()
                {
                    let d0 =
                        (msgs[0] - factor_to_var_msg[f_idx][v_pos][0]).abs();
                    let d1 =
                        (msgs[1] - factor_to_var_msg[f_idx][v_pos][1]).abs();
                    max_delta = max_delta.max(d0).max(d1);
                }
            }

            factor_to_var_msg = next_factor_to_var_msg;

            if max_delta < self.config.convergence_threshold {
                break;
            }
        }

        let mut marginals = Vec::with_capacity(num_vars);
        for (v_idx, var) in graph.variables.iter().enumerate() {
            let mut log_score = [0.0, 0.0];
            for edge in &graph.var_adjacencies[v_idx] {
                log_score[0] += factor_to_var_msg[edge.factor_idx.0]
                    [edge.pos_in_factor][0];
                log_score[1] += factor_to_var_msg[edge.factor_idx.0]
                    [edge.pos_in_factor][1];
            }

            let log_z = Self::log_sum_exp(&log_score);
            let prob_active = if log_z == f64::NEG_INFINITY {
                0.0
            } else {
                (log_score[1] - log_z).exp()
            };
            let log_odds = if log_score[1] == log_score[0] {
                0.0
            } else {
                log_score[1] - log_score[0]
            };

            marginals.push(MarginalPosterior {
                identity: var.candidate_identity,
                probability: Probability::new(prob_active.clamp(0.0, 1.0)),
                log_odds,
            });
        }

        PosteriorDistribution::new(marginals)
    }
}

#[cfg(test)]
mod tests {
    use li_core::ids::IdentityId;

    use super::*;

    #[test]
    fn test_log_sum_exp_robustness() {
        assert_eq!(
            BeliefPropagationSolver::log_sum_exp(&[]),
            f64::NEG_INFINITY
        );
        assert_eq!(
            BeliefPropagationSolver::log_sum_exp(&[
                f64::NEG_INFINITY,
                f64::NEG_INFINITY
            ]),
            f64::NEG_INFINITY
        );
        assert_eq!(
            BeliefPropagationSolver::log_sum_exp(&[1.0, f64::INFINITY]),
            f64::INFINITY
        );

        let res = BeliefPropagationSolver::log_sum_exp(&[0.0, 0.0]);
        assert!((res - (2.0f64).ln()).abs() < 1e-9);
    }

    #[test]
    fn test_bp_single_variable_eval() {
        let mut fg = FactorGraph::new();
        let id = IdentityId(100);
        let v_idx = fg.add_variable(id);

        struct TestFactor(IdentityId);

        impl li_factors::factor::FactorScope for TestFactor {
            fn scope(&self) -> &[IdentityId] {
                core::slice::from_ref(&self.0)
            }
        }

        impl li_factors::factor::Factor for TestFactor {
            fn evaluate(&self, assignments: &[IdentityId]) -> Probability {
                if assignments.contains(&self.0) {
                    Probability::new(0.9)
                } else {
                    Probability::new(0.1)
                }
            }
        }

        fg.add_factor(Box::new(TestFactor(id)), &[v_idx]);

        let solver = BeliefPropagationSolver::new(BpConfig::default());
        let posteriors = solver.solve(&fg);

        let marginal = posteriors.find_marginal(id).unwrap();
        assert!((marginal.probability.value() - 0.9).abs() < 1e-2);
    }
}
