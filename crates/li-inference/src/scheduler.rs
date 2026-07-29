//! Operational inference pipeline orchestration for observation handling,
//! workspace sync, and transactional commits.

use li_core::belief::BeliefState;
use li_core::ids::IdentityId;
use li_core::observation::Evidence;
use li_core::ontology::Vertex;
use li_core::probability::Probability;
use li_core::relation::Relation;
use li_factors::compiler::FactorCompiler;
use li_model::graph::KnowledgeGraph;
use li_model::operations::GraphOperation;
use li_workspace::workspace::ActiveWorkspace;

use crate::bp::BeliefPropagationSolver;
use crate::factor_graph::FactorGraph;
use crate::map::MapEstimator;

/// Trait defining decoupled unique identity identifier generators.
pub trait IdentityGenerator {
    /// Generates the next unique identity identifier.
    fn next_identity_id(&mut self) -> IdentityId;
}

/// Interface defining pipeline schedulers processing empirical observation
/// evidence.
pub trait PipelineScheduler<P, E, S, G, W, C>
where
    G: KnowledgeGraph<
            ObservationPayload = P,
            EventPayload = E,
            StatePayload = S,
        >,
    W: ActiveWorkspace<Summary = S>,
    C: FactorCompiler<P, S>,
{
    /// Processes an incoming evidence package, updates workspace, and returns
    /// transactional commits.
    ///
    /// # Arguments
    ///
    /// * `evidence` - Empirical evidence package containing candidate identity
    ///   hypotheses.
    /// * `graph` - Target knowledge graph to query and update.
    /// * `workspace` - Active workspace holding current belief states.
    /// * `compiler` - Compiler translating evidence into factor potentials.
    /// * `id_gen` - Generator producing unique identity identifiers.
    ///
    /// # Returns
    ///
    /// Result containing vector of committed [`GraphOperation`] items or
    /// underlying graph error.
    fn process_observation(
        &self,
        evidence: Evidence<P>,
        graph: &mut G,
        workspace: &mut W,
        compiler: &C,
        id_gen: &mut dyn IdentityGenerator,
    ) -> Result<Vec<GraphOperation<P, E, S>>, G::Error>;
}

/// Pipeline orchestrator handling factor compilation, inference execution,
/// workspace updates, and graph commits.
pub struct OperationalPipeline {
    /// Belief Propagation solver instance.
    pub bp_solver: BeliefPropagationSolver,
    /// MAP decision estimator instance.
    pub map_estimator: MapEstimator,
    /// Decision threshold required to commit an identity assignment.
    pub decision_threshold: Probability,
}

impl OperationalPipeline {
    /// Instantiates a new operational pipeline orchestrator.
    ///
    /// # Arguments
    ///
    /// * `bp_solver` - Configured Belief Propagation solver.
    /// * `decision_threshold` - Minimum probability threshold for confirming
    ///   candidate identity.
    pub fn new(
        bp_solver: BeliefPropagationSolver,
        decision_threshold: Probability,
    ) -> Self {
        Self {
            bp_solver,
            map_estimator: MapEstimator::new(),
            decision_threshold,
        }
    }
}

impl<P, E, S, G, W, C> PipelineScheduler<P, E, S, G, W, C>
    for OperationalPipeline
where
    P: Clone,
    E: Clone,
    S: Clone + Default,
    G: KnowledgeGraph<
            ObservationPayload = P,
            EventPayload = E,
            StatePayload = S,
        >,
    W: ActiveWorkspace<Summary = S>,
    C: FactorCompiler<P, S>,
{
    fn process_observation(
        &self,
        evidence: Evidence<P>,
        graph: &mut G,
        workspace: &mut W,
        compiler: &C,
        id_gen: &mut dyn IdentityGenerator,
    ) -> Result<Vec<GraphOperation<P, E, S>>, G::Error> {
        let mut active_beliefs = Vec::with_capacity(evidence.candidates.len());
        for &candidate_id in &evidence.candidates {
            if let Some(belief) = workspace.get(candidate_id) {
                active_beliefs.push(belief);
            }
        }

        let mut factor_graph =
            FactorGraph::with_capacity(evidence.candidates.len(), 8);
        let mut var_map = Vec::with_capacity(evidence.candidates.len());

        for &candidate_id in &evidence.candidates {
            let v_idx = factor_graph.add_variable(candidate_id);
            var_map.push((candidate_id, v_idx));
        }

        let factors = compiler.compile_factors(&evidence, &active_beliefs);
        for factor in factors {
            let scope_ids = factor.scope();
            let mut scope_indices = Vec::with_capacity(scope_ids.len());

            for &id in scope_ids {
                if let Some(&(_, v_idx)) =
                    var_map.iter().find(|(c_id, _)| *c_id == id)
                {
                    scope_indices.push(v_idx);
                }
            }

            if !scope_indices.is_empty() {
                factor_graph.add_factor(factor, &scope_indices);
            }
        }

        let posteriors = self.bp_solver.solve(&factor_graph);

        let map_assignment = self
            .map_estimator
            .estimate_map(&posteriors, self.decision_threshold);

        for marginal in &posteriors.marginals {
            if let Some(belief) = workspace.get_mut(marginal.identity) {
                belief.posterior = marginal.probability;
                belief.last_update = evidence.observation.timestamp;
            }
        }

        let mut operations = Vec::with_capacity(4);

        let obs_op =
            GraphOperation::CommitObservation(evidence.observation.clone());
        operations.push(obs_op);

        let target_identity = match map_assignment.selected_identity {
            Some(existing_id) => existing_id,
            None => {
                let new_identity_id = id_gen.next_identity_id();
                let id_op = GraphOperation::CommitIdentity {
                    id: new_identity_id,
                    created_at: evidence.observation.timestamp,
                };
                operations.push(id_op);

                let new_belief = BeliefState {
                    identity: new_identity_id,
                    summary: S::default(),
                    posterior: Probability::new(1.0),
                    last_update: evidence.observation.timestamp,
                };
                workspace.insert(new_belief);

                new_identity_id
            },
        };

        let rel_op = GraphOperation::CommitRelation {
            source: Vertex::Observation(evidence.observation.id),
            relation: Relation::Supports,
            target: Vertex::Identity(target_identity),
            created_at: evidence.observation.timestamp,
        };
        operations.push(rel_op);

        graph.apply_batch(operations.clone())?;

        Ok(operations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bp::BpConfig;

    struct SequentialIdGen(u64);
    impl IdentityGenerator for SequentialIdGen {
        fn next_identity_id(&mut self) -> IdentityId {
            self.0 += 1;
            IdentityId(self.0)
        }
    }

    #[test]
    fn test_sequential_id_generator() {
        let mut generator = SequentialIdGen(100);
        assert_eq!(generator.next_identity_id(), IdentityId(101));
        assert_eq!(generator.next_identity_id(), IdentityId(102));
    }

    #[test]
    fn test_pipeline_construction() {
        let solver = BeliefPropagationSolver::new(BpConfig::default());
        let pipeline = OperationalPipeline::new(solver, Probability::new(0.8));
        assert_eq!(pipeline.decision_threshold.value(), 0.8);
    }
}
