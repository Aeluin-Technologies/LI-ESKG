//! Reactive engine coordinating planner, inference, workspace, and storage
//! execution.

use alloc::vec::Vec;
use core::marker::PhantomData;

use li_core::events::RuntimeEvent;
use li_core::ids::IdentityId;
use li_core::observation::{Evidence, Timestamp};
use li_core::ontology::Vertex;
use li_core::probability::Probability;
use li_core::relation::Relation;
use li_factors::compiler::FactorCompiler;
use li_inference::bp::{BeliefPropagationSolver, BpConfig};
use li_inference::factor_graph::FactorGraph;
use li_inference::map::MapEstimator;
use li_model::operations::GraphOperation;
use li_workspace::workspace::ActiveWorkspace;

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
        P: Clone,
        S: Clone,
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
                let ops = Vec::from([GraphOperation::CommitRelation {
                    source: Vertex::Identity(target),
                    relation: Relation::Refines,
                    target: Vertex::Identity(duplicate),
                    created_at: Timestamp::default(),
                }]);
                self.executor.commit(ops)?;
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
        P: Clone,
        S: Clone,
    {
        let mut ops = Vec::new();
        let timestamp = evidence.observation.timestamp;
        let obs_id = evidence.observation.id;

        let target_identity = if evidence.candidates.is_empty() {
            let new_id = IdentityId(self.next_identity_id);
            self.next_identity_id += 1;
            ops.push(GraphOperation::CommitIdentity {
                id: new_id,
                created_at: timestamp,
            });
            new_id
        } else {
            let active_beliefs_refs = self.workspace.active_beliefs();
            let active_beliefs: Vec<_> =
                active_beliefs_refs.into_iter().cloned().collect();
            let factors =
                self.compiler.compile_factors(&evidence, &active_beliefs);

            let mut fg = FactorGraph::new();
            for &cand_id in &evidence.candidates {
                fg.add_variable(cand_id);
            }
            for factor in factors {
                fg.add_factor(factor, &[]);
            }

            let solver = BeliefPropagationSolver::new(self.bp_config);
            let posteriors = solver.solve(&fg);
            let estimator = MapEstimator;
            let assignment = estimator.estimate_map(
                &posteriors,
                Probability::new(self.config.decision_threshold),
            );

            match assignment.selected_identity {
                Some(id) => id,
                None => {
                    let new_id = IdentityId(self.next_identity_id);
                    ops.push(GraphOperation::CommitIdentity {
                        id: new_id,
                        created_at: timestamp,
                    });
                    self.next_identity_id += 1;
                    new_id
                },
            }
        };

        ops.push(GraphOperation::CommitObservation(evidence.observation));
        ops.push(GraphOperation::CommitRelation {
            source: Vertex::Observation(obs_id),
            relation: Relation::Supports,
            target: Vertex::Identity(target_identity),
            created_at: timestamp,
        });

        self.executor.commit(ops)
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
    use li_core::belief::BeliefState;
    use li_factors::factor::Factor;
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
            _active_beliefs: &[BeliefState<()>],
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
}
