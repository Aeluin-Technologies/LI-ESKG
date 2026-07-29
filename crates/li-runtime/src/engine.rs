//! Reactive engine coordinating planner, inference, workspace, and storage
//! execution.

use alloc::vec::Vec;
use core::marker::PhantomData;
use std::collections::HashMap;
use std::collections::hash_map::Entry;

use li_core::belief::BeliefState;
use li_core::events::RuntimeEvent;
use li_core::ids::IdentityId;
use li_core::observation::{Evidence, Timestamp};
use li_core::ontology::Vertex;
use li_core::probability::Probability;
use li_core::relation::Relation;
use li_factors::compiler::{DirectMapDecision, FactorCompiler};
use li_inference::bp::{BeliefPropagationSolver, BpConfig};
use li_inference::factor_graph::FactorGraph;
use li_inference::map::MapEstimator;
use li_model::operations::GraphOperation;
use li_workspace::workspace::ActiveWorkspace;
use smallvec::SmallVec;

use crate::channels::EventQueue;
use crate::dispatcher::{DispatchOutcome, EventDispatcher};
use crate::executor::{ExecutionSink, OperationExecutor};

/// Configuration parameters governing threshold decisions within the engine.
#[derive(Debug, Clone, Copy)]
pub struct EngineConfig {
    /// Probability threshold required to assign an identity hypothesis via
    /// MAP decision.
    pub decision_threshold: f64,
    /// Confidence threshold enabling direct assignment, bypassing
    /// probabilistic factor inference.
    pub direct_assignment_threshold: f64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            decision_threshold: 0.85,
            direct_assignment_threshold: 0.98,
        }
    }
}

/// Reactive state machine reactor orchestrating the execution loop of the
/// runtime.
pub struct RuntimeEngine<P, E, S, W, FC, Sink> {
    config: EngineConfig,
    queue: EventQueue<RuntimeEvent<P>>,
    dispatcher: EventDispatcher,
    workspace: W,
    compiler: FC,
    executor: OperationExecutor<Sink>,
    operations: Vec<GraphOperation<P, E, S>>,
    bp_config: BpConfig,
    next_identity_id: u64,
    is_running: bool,
    _phantom: PhantomData<(E, S)>,
}

impl<P, E, S, W, FC, Sink> RuntimeEngine<P, E, S, W, FC, Sink> {
    /// Instantiates a new `RuntimeEngine` with explicit configuration and
    /// initial state.
    ///
    /// # Arguments
    ///
    /// * `config` - Decision and operation thresholds.
    /// * `capacity` - Maximum capacity for the internal event queue.
    /// * `workspace` - Active working memory state.
    /// * `compiler` - Factor compiler implementation.
    /// * `sink` - Persistence execution sink destination.
    pub fn new(
        config: EngineConfig,
        capacity: usize,
        workspace: W,
        compiler: FC,
        sink: Sink,
    ) -> Self {
        Self {
            config,
            queue: EventQueue::new(capacity),
            dispatcher: EventDispatcher::new(),
            workspace,
            compiler,
            executor: OperationExecutor::new(sink),
            operations: Vec::with_capacity(3),
            bp_config: BpConfig {
                max_iterations: 10,
                convergence_threshold: 0.001,
            },
            next_identity_id: 1000,
            is_running: true,
            _phantom: PhantomData,
        }
    }

    /// Submits a runtime event into the internal processing queue.
    ///
    /// # Arguments
    ///
    /// * `event` - The event payload to enqueue.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on successful enqueue, or `Err(event)` if the queue is
    /// saturated.
    pub fn submit_event(
        &mut self,
        event: RuntimeEvent<P>,
    ) -> Result<(), RuntimeEvent<P>> {
        self.queue.push(event)
    }

    /// Executes a single event loop step from the internal queue.
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if an event was processed, `Ok(false)` if queue was
    /// empty or engine stopped.
    pub fn tick<Err>(&mut self) -> Result<bool, Err>
    where
        Sink: ExecutionSink<P, E, S, Error = Err>,
        W: ActiveWorkspace<Summary = S>,
        FC: FactorCompiler<P, S>,
    {
        if !self.is_running {
            return Ok(false);
        }

        let event = match self.queue.pop() {
            Some(ev) => ev,
            None => return Ok(false),
        };

        match self.dispatcher.dispatch(event) {
            DispatchOutcome::EvaluateObservation(evidence) => {
                self.process_observation(evidence)?;
            },
            DispatchOutcome::MergeIdentities { target, duplicate } => {
                self.operations.clear();
                self.operations.push(GraphOperation::MergeIdentities {
                    target,
                    duplicate,
                });
                let result = self.executor.commit_slice(&self.operations);
                self.operations.clear();
                result?;
            },
            DispatchOutcome::TriggerCheckpoint => {
                self.workspace.create_snapshot(Timestamp::default());
            },
            DispatchOutcome::Shutdown => {
                self.is_running = false;
            },
            DispatchOutcome::NoOp => {},
        }

        Ok(true)
    }

    /// Processes an incoming observation evidence through planning, factor
    /// inference, and graph commit.
    ///
    /// # Arguments
    ///
    /// * `evidence` - Empirical evidence observation package.
    fn process_observation<Err>(
        &mut self,
        evidence: Evidence<P>,
    ) -> Result<(), Err>
    where
        Sink: ExecutionSink<P, E, S, Error = Err>,
        W: ActiveWorkspace<Summary = S>,
        FC: FactorCompiler<P, S>,
    {
        self.operations.clear();
        let timestamp = evidence.observation.timestamp;
        let obs_id = evidence.observation.id;

        let selected_identity = if evidence.candidates.is_empty() {
            None
        } else {
            let active_beliefs: SmallVec<[&BeliefState<S>; 32]> = evidence
                .candidates
                .iter()
                .filter_map(|&identity| self.workspace.get(identity))
                .collect();

            match self.compiler.try_direct_map(
                &evidence,
                &active_beliefs,
                Probability::new(self.config.decision_threshold),
            ) {
                DirectMapDecision::Assign(identity) => Some(identity),
                DirectMapDecision::CreateIdentity => None,
                DirectMapDecision::Unsupported => {
                    let factors = self
                        .compiler
                        .compile_factors(&evidence, &active_beliefs);

                    let mut fg = FactorGraph::with_capacity(
                        evidence.candidates.len(),
                        factors.len(),
                    );
                    let mut variable_indices =
                        HashMap::with_capacity(evidence.candidates.len());
                    for &cand_id in &evidence.candidates {
                        if let Entry::Vacant(entry) =
                            variable_indices.entry(cand_id)
                        {
                            entry.insert(fg.add_variable(cand_id));
                        }
                    }

                    let mut factor_scope =
                        Vec::with_capacity(evidence.candidates.len());
                    for factor in factors {
                        factor_scope.clear();
                        for identity in factor.scope() {
                            if let Some(index) = variable_indices.get(identity)
                            {
                                factor_scope.push(*index);
                            }
                        }
                        if factor_scope.len() == factor.scope().len() {
                            fg.add_factor(factor, &factor_scope);
                        }
                    }

                    let solver = BeliefPropagationSolver::new(self.bp_config);
                    let posteriors = solver.solve(&fg);
                    let estimator = MapEstimator;
                    estimator
                        .estimate_map(
                            &posteriors,
                            Probability::new(self.config.decision_threshold),
                        )
                        .selected_identity
                },
            }
        };

        let target_identity = match selected_identity {
            Some(identity) => identity,
            None => {
                let identity = IdentityId(self.next_identity_id);
                self.next_identity_id += 1;
                self.operations.push(GraphOperation::CommitIdentity {
                    id: identity,
                    created_at: timestamp,
                });
                identity
            },
        };

        self.operations
            .push(GraphOperation::CommitObservation(evidence.observation));
        self.operations.push(GraphOperation::CommitRelation {
            source: Vertex::Observation(obs_id),
            relation: Relation::Supports,
            target: Vertex::Identity(target_identity),
            created_at: timestamp,
        });

        let result = self.executor.commit_slice(&self.operations);
        self.operations.clear();
        result
    }

    /// Returns a reference to the active workspace.
    pub fn workspace(&self) -> &W {
        &self.workspace
    }

    /// Returns a mutable reference to the active workspace.
    pub fn workspace_mut(&mut self) -> &mut W {
        &mut self.workspace
    }

    /// Returns a reference to the operation executor.
    pub fn executor(&self) -> &OperationExecutor<Sink> {
        &self.executor
    }

    /// Returns a mutable reference to the operation executor.
    pub fn executor_mut(&mut self) -> &mut OperationExecutor<Sink> {
        &mut self.executor
    }

    /// Returns `true` if the engine is currently active and processing events.
    pub fn is_running(&self) -> bool {
        self.is_running
    }
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeMap;
    use alloc::vec;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use li_core::belief::BeliefState;
    use li_core::ids::ObservationId;
    use li_core::observation::{Modality, Observation};
    use li_core::probability::Confidence;
    use li_factors::factor::{CategoricalFactor, Factor};
    use li_workspace::checkpoint::WorkspaceSnapshot;

    use super::*;

    struct DummyWorkspace;
    impl ActiveWorkspace for DummyWorkspace {
        type Summary = ();

        fn insert(&mut self, _belief: BeliefState<()>) {}

        fn get(&self, _id: IdentityId) -> Option<&BeliefState<()>> {
            None
        }

        fn get_mut(
            &mut self,
            _id: IdentityId,
        ) -> Option<&mut BeliefState<()>> {
            None
        }

        fn active_beliefs(&self) -> Vec<&BeliefState<()>> {
            Vec::new()
        }

        fn evict_expired(
            &mut self,
            _current_time: Timestamp,
            _ttl: i64,
        ) -> Vec<BeliefState<()>> {
            Vec::new()
        }

        fn create_snapshot(
            &self,
            current_time: Timestamp,
        ) -> WorkspaceSnapshot<()> {
            WorkspaceSnapshot {
                timestamp: current_time,
                active_states: Vec::new(),
            }
        }
    }

    struct DummyCompiler;
    impl FactorCompiler<(), ()> for DummyCompiler {
        fn compile_factors(
            &self,
            _evidence: &Evidence<()>,
            _active_beliefs: &[&BeliefState<()>],
        ) -> Vec<alloc::boxed::Box<dyn Factor>> {
            Vec::new()
        }
    }

    struct DummySink;
    impl ExecutionSink<(), (), ()> for DummySink {
        type Error = ();

        fn execute_batch(
            &mut self,
            _operations: &[GraphOperation<(), (), ()>],
        ) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    #[test]
    fn test_engine_shutdown() {
        let ws = DummyWorkspace;
        let compiler = DummyCompiler;
        let sink = DummySink;
        let mut engine = RuntimeEngine::new(
            EngineConfig::default(),
            10,
            ws,
            compiler,
            sink,
        );

        assert!(engine.is_running());
        assert_eq!(engine.submit_event(RuntimeEvent::Shutdown), Ok(()));
        assert_eq!(engine.tick::<()>(), Ok(true));
        assert!(!engine.is_running());
        assert_eq!(engine.tick::<()>(), Ok(false));
    }

    #[test]
    fn test_engine_empty_tick() {
        let ws = DummyWorkspace;
        let compiler = DummyCompiler;
        let sink = DummySink;
        let mut engine: RuntimeEngine<
            (),
            (),
            (),
            DummyWorkspace,
            DummyCompiler,
            DummySink,
        > = RuntimeEngine::new(
            EngineConfig::default(),
            10,
            ws,
            compiler,
            sink,
        );

        assert_eq!(engine.tick::<()>(), Ok(false));
    }

    struct CandidateWorkspace {
        beliefs: BTreeMap<IdentityId, BeliefState<()>>,
    }

    impl ActiveWorkspace for CandidateWorkspace {
        type Summary = ();

        fn insert(&mut self, belief: BeliefState<()>) {
            self.beliefs.insert(belief.identity, belief);
        }

        fn get(&self, id: IdentityId) -> Option<&BeliefState<()>> {
            self.beliefs.get(&id)
        }

        fn get_mut(&mut self, id: IdentityId) -> Option<&mut BeliefState<()>> {
            self.beliefs.get_mut(&id)
        }

        fn active_beliefs(&self) -> Vec<&BeliefState<()>> {
            self.beliefs.values().collect()
        }

        fn evict_expired(
            &mut self,
            _current_time: Timestamp,
            _ttl: i64,
        ) -> Vec<BeliefState<()>> {
            Vec::new()
        }

        fn create_snapshot(
            &self,
            current_time: Timestamp,
        ) -> WorkspaceSnapshot<()> {
            WorkspaceSnapshot {
                timestamp: current_time,
                active_states: self.beliefs.values().cloned().collect(),
            }
        }
    }

    struct StrongCompiler;

    impl FactorCompiler<(), ()> for StrongCompiler {
        fn compile_factors(
            &self,
            evidence: &Evidence<()>,
            active_beliefs: &[&BeliefState<()>],
        ) -> Vec<alloc::boxed::Box<dyn Factor>> {
            let mut probabilities =
                HashMap::with_capacity(active_beliefs.len());
            for belief in active_beliefs {
                if evidence.candidates.contains(&belief.identity) {
                    probabilities
                        .insert(belief.identity, Probability::new(0.99));
                }
            }

            match CategoricalFactor::new(probabilities, Probability::new(0.01))
            {
                Ok(factor) => vec![alloc::boxed::Box::new(factor)],
                Err(_) => Vec::new(),
            }
        }
    }

    struct DirectCompiler {
        fallback_calls: Arc<AtomicUsize>,
    }

    impl FactorCompiler<(), ()> for DirectCompiler {
        fn compile_factors(
            &self,
            _evidence: &Evidence<()>,
            _active_beliefs: &[&BeliefState<()>],
        ) -> Vec<alloc::boxed::Box<dyn Factor>> {
            self.fallback_calls.fetch_add(1, Ordering::Relaxed);
            Vec::new()
        }

        fn try_direct_map(
            &self,
            _evidence: &Evidence<()>,
            active_beliefs: &[&BeliefState<()>],
            _decision_threshold: Probability,
        ) -> DirectMapDecision {
            match active_beliefs.first() {
                Some(belief) => DirectMapDecision::Assign(belief.identity),
                None => DirectMapDecision::CreateIdentity,
            }
        }
    }

    #[derive(Default)]
    struct CapturingSink {
        operations: Vec<GraphOperation<(), (), ()>>,
    }

    impl ExecutionSink<(), (), ()> for CapturingSink {
        type Error = ();

        fn execute_batch(
            &mut self,
            operations: &[GraphOperation<(), (), ()>],
        ) -> Result<(), Self::Error> {
            self.operations.extend_from_slice(operations);
            Ok(())
        }
    }

    #[test]
    fn test_engine_connects_compiled_factor_scope_to_candidate() {
        let candidate = IdentityId(7);
        let timestamp = Timestamp::from_secs(10);
        let mut beliefs = BTreeMap::new();
        beliefs.insert(
            candidate,
            BeliefState::new(candidate, (), Probability::new(0.8), timestamp),
        );
        let workspace = CandidateWorkspace { beliefs };
        let mut engine = RuntimeEngine::new(
            EngineConfig::default(),
            1,
            workspace,
            StrongCompiler,
            CapturingSink::default(),
        );
        let evidence = Evidence::new(
            Observation::new(
                ObservationId(1),
                Modality(1),
                timestamp,
                Confidence::new(1.0),
                (),
            ),
            vec![candidate],
        );

        assert_eq!(
            engine.submit_event(RuntimeEvent::Observation(evidence)),
            Ok(())
        );
        assert_eq!(engine.tick::<()>(), Ok(true));

        let operations = &engine.executor().sink().operations;
        assert!(!operations.iter().any(|operation| {
            matches!(operation, GraphOperation::CommitIdentity { .. })
        }));
        assert!(operations.iter().any(|operation| {
            matches!(
                operation,
                GraphOperation::CommitRelation {
                    relation: Relation::Supports,
                    target: Vertex::Identity(identity),
                    ..
                } if *identity == candidate
            )
        }));
    }

    #[test]
    fn test_engine_uses_direct_map_without_compiling_factor_graph() {
        let candidate = IdentityId(17);
        let timestamp = Timestamp::from_secs(10);
        let mut beliefs = BTreeMap::new();
        beliefs.insert(
            candidate,
            BeliefState::new(candidate, (), Probability::new(0.8), timestamp),
        );
        let fallback_calls = Arc::new(AtomicUsize::new(0));
        let compiler = DirectCompiler {
            fallback_calls: Arc::clone(&fallback_calls),
        };
        let mut engine = RuntimeEngine::new(
            EngineConfig::default(),
            1,
            CandidateWorkspace { beliefs },
            compiler,
            CapturingSink::default(),
        );
        let evidence = Evidence::new(
            Observation::new(
                ObservationId(1),
                Modality(1),
                timestamp,
                Confidence::new(1.0),
                (),
            ),
            vec![candidate],
        );

        assert_eq!(
            engine.submit_event(RuntimeEvent::Observation(evidence)),
            Ok(())
        );
        assert_eq!(engine.tick::<()>(), Ok(true));
        assert_eq!(fallback_calls.load(Ordering::Relaxed), 0);
        assert!(engine.executor().sink().operations.iter().any(|operation| {
            matches!(
                operation,
                GraphOperation::CommitRelation {
                    target: Vertex::Identity(identity),
                    ..
                } if *identity == candidate
            )
        }));
    }
}
