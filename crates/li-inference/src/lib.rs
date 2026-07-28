//! Latent Identity ESKG Inference Engine (`li-inference`).
//!
//! This crate implements the ephemeral factor graph compilation, Sum-Product
//! Belief Propagation, MAP decision estimation, and the operational pipeline
//! orchestration specified in Algorithm 1.

pub mod bp;
pub mod factor_graph;
pub mod map;
pub mod posterior;
pub mod scheduler;

pub use bp::{BeliefPropagationSolver, BpConfig};
pub use factor_graph::{
    FactorEdge, FactorGraph, FactorIndex, VarEdge, VarIndex, VariableNode,
};
pub use map::{MapAssignment, MapEstimator};
pub use posterior::{MarginalPosterior, PosteriorDistribution};
pub use scheduler::{
    IdentityGenerator, OperationalPipeline, PipelineScheduler,
};

#[cfg(test)]
mod tests {
    use li_core::ids::IdentityId;
    use li_core::probability::Probability;

    use super::*;

    #[test]
    fn test_crate_exports_and_full_inference_loop() {
        let mut fg = FactorGraph::new();
        let id1 = IdentityId(1);
        let v1 = fg.add_variable(id1);

        struct TestFactor(IdentityId);

        impl li_factors::factor::FactorScope for TestFactor {
            fn scope(&self) -> &[IdentityId] {
                core::slice::from_ref(&self.0)
            }
        }

        impl li_factors::factor::Factor for TestFactor {
            fn evaluate(&self, assignments: &[IdentityId]) -> Probability {
                if assignments.contains(&self.0) {
                    Probability::new(0.95)
                } else {
                    Probability::new(0.05)
                }
            }
        }

        fg.add_factor(Box::new(TestFactor(id1)), &[v1]);

        let solver = BeliefPropagationSolver::new(BpConfig::default());
        let posteriors = solver.solve(&fg);

        let estimator = MapEstimator::new();
        let assignment =
            estimator.estimate_map(&posteriors, Probability::new(0.80));

        assert_eq!(assignment.selected_identity, Some(id1));
    }
}
