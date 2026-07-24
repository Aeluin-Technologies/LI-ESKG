//! Operational inference pipeline orchestration for observation handling and
//! state updates.

use alloc::vec::Vec;

use li_core::ids::{IdentityId, VertexId};
use li_core::observation::Evidence;
use li_core::probability::Probability;
use li_core::relation::Relation;
use li_factors::compiler::FactorCompiler;
use li_model::graph::KnowledgeGraph;
use li_model::ontology::IdentityNode;
use li_model::operations::GraphOperation;
use li_workspace::workspace::ActiveWorkspace;

use crate::bp::BeliefPropagationSolver;
use crate::factor_graph::FactorGraph;
use crate::map::MapEstimator;

/// Interface defining pipeline schedulers capable of processing empirical
/// observation evidence.
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
    /// Processes an incoming evidence package and executes full inference and
    /// belief updating.
    ///
    /// # Arguments
    ///
    /// * `evidence` - Empirical evidence package containing candidate identity
    ///   hypotheses.
    /// * `graph` - Target knowledge graph to query and update.
    /// * `workspace` - Active workspace holding current belief states.
    /// * `compiler` - Compiler translating evidence into factor potentials.
    ///
    /// # Returns
    ///
    /// A vector of [`GraphOperation`] items representing structural commits
    /// applied to the graph.
    fn process_observation(
        &self,
        evidence: Evidence<P>,
        graph: &mut G,
        workspace: &mut W,
        compiler: &C,
    ) -> Vec<GraphOperation<P, E, S>>;
}

/// Pipeline orchestrator handling factor compilation, inference execution, and
/// graph commits.
pub struct OperationalPipeline<BP> {
    /// Belief propagation solver instance.
    pub bp_solver: BP,
    /// MAP estimator instance.
    pub map_estimator: MapEstimator,
    /// Decision threshold required to commit an identity assignment.
    pub decision_threshold: Probability,
}

impl<BP> OperationalPipeline<BP> {
    /// Instantiates a new operational pipeline orchestrator.
    ///
    /// # Arguments
    ///
    /// * `bp_solver` - Belief Propagation solver implementation.
    /// * `decision_threshold` - Minimum probability threshold for confirming
    ///   candidate identity.
    pub fn new(bp_solver: BP, decision_threshold: Probability) -> Self {
        Self {
            bp_solver,
            map_estimator: MapEstimator,
            decision_threshold,
        }
    }
}

impl<P, E, S, G, W, C> PipelineScheduler<P, E, S, G, W, C>
    for OperationalPipeline<BeliefPropagationSolver>
where
    P: Clone,
    E: Clone,
    S: Clone,
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
    ) -> Vec<GraphOperation<P, E, S>> {
        let mut active_beliefs = Vec::with_capacity(evidence.candidates.len());
        for &candidate_id in &evidence.candidates {
            let vertex_exists = matches!(
                graph.vertex_type(VertexId(candidate_id.0)),
                Ok(Some(_))
            );

            if let Some(belief) = workspace.get(candidate_id) {
                active_beliefs.push(belief.clone());
            } else if vertex_exists {
                // Topology vertex exists in persistent storage but is inactive
                // in workspace.
            }
        }

        let mut factor_graph = FactorGraph::new();
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

        let mut operations = Vec::new();

        let obs_op =
            GraphOperation::CommitObservation(evidence.observation.clone());
        graph.apply(obs_op.clone());
        operations.push(obs_op);

        let target_identity = match map_assignment.selected_identity {
            Some(existing_id) => existing_id,
            None => {
                let new_identity_id = IdentityId(evidence.observation.id.0);
                let identity_node = IdentityNode {
                    id: new_identity_id,
                    created_at: evidence.observation.timestamp,
                };

                let id_op = GraphOperation::CommitIdentity(identity_node);
                graph.apply(id_op.clone());
                operations.push(id_op);

                new_identity_id
            },
        };

        let rel_op = GraphOperation::CommitRelation {
            source: VertexId(evidence.observation.id.0),
            relation: Relation::Supports,
            target: VertexId(target_identity.0),
            created_at: evidence.observation.timestamp,
        };
        graph.apply(rel_op.clone());
        operations.push(rel_op);

        operations
    }
}

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;
    use alloc::vec::Vec;
    use core::convert::Infallible;

    use li_core::belief::BeliefState;
    use li_core::ids::{IdentityId, ObservationId, VertexId};
    use li_core::observation::{Evidence, Modality, Observation, Timestamp};
    use li_core::ontology::Vertex;
    use li_core::probability::{Confidence, Probability};
    use li_core::relation::Relation;
    use li_factors::compiler::FactorCompiler;
    use li_factors::factor::{Factor, FactorScope};
    use li_model::Edge;
    use li_model::graph::KnowledgeGraph;
    use li_model::operations::GraphOperation;
    use li_workspace::InMemoryWorkspace;
    use li_workspace::workspace::ActiveWorkspace;

    use crate::bp::{BeliefPropagationSolver, BpConfig};
    use crate::scheduler::{OperationalPipeline, PipelineScheduler};

    #[derive(Clone)]
    struct MockPayload;

    #[derive(Clone)]
    struct MockSummary;

    struct MockGraph {
        vertices: Vec<VertexId>,
    }

    impl KnowledgeGraph for MockGraph {
        type Error = Infallible;
        type EventPayload = MockPayload;
        type ObservationPayload = MockPayload;
        type StatePayload = MockSummary;

        fn vertex_type(
            &self,
            id: VertexId,
        ) -> Result<Option<Vertex>, Self::Error> {
            if self.vertices.contains(&id) {
                Ok(Some(Vertex::Identity(IdentityId(id.0))))
            } else {
                Ok(None)
            }
        }

        fn apply_batch(
            &mut self,
            _ops: &[GraphOperation<
                Self::ObservationPayload,
                Self::EventPayload,
                Self::StatePayload,
            >],
        ) -> Result<(), Self::Error> {
            // Pas d'appel à self.apply() pour éviter la récursion mutuelle
            Ok(())
        }

        fn query_support_set(
            &self,
            _identity: IdentityId,
        ) -> Result<Vec<Observation<Self::ObservationPayload>>, Self::Error>
        {
            Ok(Vec::new())
        }

        fn out_edges(
            &self,
            _source: VertexId,
        ) -> Result<Vec<Edge>, Self::Error> {
            Ok(Vec::new())
        }

        fn all_identities(&self) -> Result<Vec<IdentityId>, Self::Error> {
            Ok(Vec::new())
        }
    }

    struct PassThroughFactor {
        scope: Vec<IdentityId>,
        val: f64,
    }

    impl FactorScope for PassThroughFactor {
        fn scope(&self) -> &[IdentityId] {
            &self.scope
        }
    }

    impl Factor for PassThroughFactor {
        fn evaluate(&self, assignment: &[IdentityId]) -> Probability {
            let prior = 0.5;
            let likelihood_true = self.val;
            let likelihood_false = 1.0 - self.val;

            if assignment.is_empty() || assignment == [IdentityId(0)] {
                let unnormalized = likelihood_false * (1.0 - prior);
                let evidence_total = (likelihood_true * prior) +
                    (likelihood_false * (1.0 - prior));
                Probability::new(unnormalized / evidence_total)
            } else {
                let unnormalized = likelihood_true * prior;
                let evidence_total = (likelihood_true * prior) +
                    (likelihood_false * (1.0 - prior));
                Probability::new(unnormalized / evidence_total)
            }
        }
    }

    struct MockCompiler {
        return_value: f64,
    }

    impl FactorCompiler<MockPayload, MockSummary> for MockCompiler {
        fn compile_factors(
            &self,
            evidence: &Evidence<MockPayload>,
            _active_beliefs: &[BeliefState<MockSummary>],
        ) -> Vec<Box<dyn Factor>> {
            let mut factors: Vec<Box<dyn Factor>> = Vec::new();
            for &c in &evidence.candidates {
                factors.push(Box::new(PassThroughFactor {
                    scope: alloc::vec![c],
                    val: self.return_value,
                }));
            }
            factors
        }
    }

    #[test]
    fn test_scheduler_no_candidates_creates_new_identity() {
        let bp_solver = BeliefPropagationSolver::new(BpConfig {
            max_iterations: 5,
            convergence_threshold: 0.01,
        });
        let pipeline =
            OperationalPipeline::new(bp_solver, Probability::new(0.7));

        let mut graph = MockGraph {
            vertices: Vec::new(),
        };
        let mut workspace = InMemoryWorkspace::<MockSummary>::new();
        let compiler = MockCompiler { return_value: 0.9 };

        let evidence = Evidence {
            observation: Observation {
                id: ObservationId(50),
                modality: Modality(1),
                timestamp: Timestamp(1000),
                confidence: Confidence(0.9),
                payload: MockPayload,
            },
            candidates: Vec::new(),
        };

        let ops = pipeline.process_observation(
            evidence,
            &mut graph,
            &mut workspace,
            &compiler,
        );

        assert_eq!(ops.len(), 3);
        match &ops[1] {
            GraphOperation::CommitIdentity(node) => {
                assert_eq!(node.id, IdentityId(50));
            },
            _ => panic!("Expected CommitIdentity operation"),
        }
    }

    #[test]
    fn test_scheduler_candidates_below_threshold_creates_new_identity() {
        let bp_solver = BeliefPropagationSolver::new(BpConfig {
            max_iterations: 5,
            convergence_threshold: 0.01,
        });
        let pipeline =
            OperationalPipeline::new(bp_solver, Probability::new(0.8));

        let mut graph = MockGraph {
            vertices: alloc::vec![VertexId(10)],
        };
        let mut workspace = InMemoryWorkspace::<MockSummary>::new();
        workspace.insert(BeliefState {
            identity: IdentityId(10),
            summary: MockSummary,
            posterior: Probability::new(0.1),
            last_update: Timestamp(0),
        });

        let compiler = MockCompiler { return_value: 0.2 };

        let evidence = Evidence {
            observation: Observation {
                id: ObservationId(99),
                modality: Modality(1),
                timestamp: Timestamp(2000),
                confidence: Confidence(0.9),
                payload: MockPayload,
            },
            candidates: alloc::vec![IdentityId(10)],
        };

        let ops = pipeline.process_observation(
            evidence,
            &mut graph,
            &mut workspace,
            &compiler,
        );

        assert_eq!(ops.len(), 3);
        match &ops[1] {
            GraphOperation::CommitIdentity(node) => {
                assert_eq!(node.id, IdentityId(99));
            },
            _ => panic!("Expected CommitIdentity creation"),
        }
    }

    #[test]
    fn test_scheduler_candidate_matches_threshold() {
        let bp_solver = BeliefPropagationSolver::new(BpConfig {
            max_iterations: 5,
            convergence_threshold: 0.01,
        });
        let pipeline =
            OperationalPipeline::new(bp_solver, Probability::new(0.5));

        let mut graph = MockGraph {
            vertices: alloc::vec![VertexId(10)],
        };
        let mut workspace = InMemoryWorkspace::<MockSummary>::new();
        workspace.insert(BeliefState {
            identity: IdentityId(10),
            summary: MockSummary,
            posterior: Probability::new(0.1),
            last_update: Timestamp(0),
        });

        let compiler = MockCompiler { return_value: 0.95 };

        let evidence = Evidence {
            observation: Observation {
                id: ObservationId(200),
                modality: Modality(1),
                timestamp: Timestamp(3000),
                confidence: Confidence(0.9),
                payload: MockPayload,
            },
            candidates: alloc::vec![IdentityId(10)],
        };

        let ops = pipeline.process_observation(
            evidence,
            &mut graph,
            &mut workspace,
            &compiler,
        );

        assert_eq!(ops.len(), 2);

        match &ops[0] {
            GraphOperation::CommitObservation(obs) => {
                assert_eq!(obs.id, ObservationId(200))
            },
            _ => panic!("Op 0 should be CommitObservation"),
        }

        match &ops[1] {
            GraphOperation::CommitRelation {
                source,
                relation,
                target,
                ..
            } => {
                assert_eq!(*source, VertexId(200));
                assert_eq!(*relation, Relation::Supports);
                assert_eq!(*target, VertexId(10));
            },
            _ => panic!("Op 2 should be CommitRelation"),
        }

        let updated_belief = workspace.get(IdentityId(10)).unwrap();
        assert_eq!(updated_belief.last_update, Timestamp(3000));
        assert_eq!(updated_belief.posterior, Probability::new(0.95));
    }
}
