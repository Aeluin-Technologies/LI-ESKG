//! End-to-end validation of the paper's evidence-to-graph pipeline.

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use chrono::Duration;
use li_core::{
    BeliefState, BoundedHistory, Confidence, EventId, Evidence, IdentityId,
    Modality, Observation, ObservationId, ObservationModel, Probability,
    Relation, StateId, Timestamp, Vertex,
};
use li_factors::compiler::{CategoricalFactorCompiler, DirectMapDecision};
use li_factors::{
    FactorCompiler, KCandidateDistribution, MultiCandidateCompatibility,
};
use li_inference::{
    BeliefPropagationSolver, BpConfig, FactorGraph, MapAssignment,
    MapEstimator, PosteriorDistribution, VarIndex,
};
use li_model::invariants::{
    CausalAcyclicityInvariant, IdentityUniquenessInvariant,
    ObservationPartitionInvariant,
};
use li_model::operations::IdentityAssignment;
use li_model::{
    EventStateGraph, EventStateProjection, GraphOperation, GraphProjection,
    Invariant, KnowledgeGraph, PetGraphStore,
};
use li_workspace::{
    ActiveWorkspace, CandidateGenerator, InMemoryWorkspace,
    SpatialCoordinates, SpatialGridConfig, SpatialGridIndex,
    SpatialIndexError, SpatialPoint,
};

const HISTORY_CAPACITY: usize = 2;
const BACKGROUND_WEIGHT: f64 = 0.05;

type ProjectionEdge = (Vertex, Relation, Vertex);
type ProjectionSignature = (HashSet<Vertex>, HashSet<ProjectionEdge>);

/// Scalar measurement used by the test observation modality.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Reading {
    value: f64,
}

impl SpatialCoordinates for Reading {
    fn spatial_point(&self) -> Result<SpatialPoint, SpatialIndexError> {
        SpatialPoint::try_new(self.value, 0.0)
    }
}

/// Bounded rolling summary maintained by the observation model.
#[derive(Debug, Clone, PartialEq)]
struct RollingSummary {
    history: BoundedHistory<f64>,
    sum: f64,
}

impl RollingSummary {
    /// Creates an empty summary with a fixed history limit.
    fn new(capacity: usize) -> Self {
        Self {
            history: BoundedHistory::new(capacity),
            sum: 0.0,
        }
    }

    /// Records a value and removes the evicted value from the rolling sum.
    fn record(&mut self, value: f64) {
        if self.history.capacity() == 0 {
            let _rejected = self.history.push(value);
            return;
        }

        if let Some(evicted) = self.history.push(value) {
            self.sum -= evicted;
        }
        self.sum += value;
    }

    /// Returns the rolling arithmetic mean, if the history is non-empty.
    fn mean(&self) -> Option<f64> {
        if self.history.is_empty() {
            None
        } else {
            Some(self.sum / self.history.len() as f64)
        }
    }
}

impl Default for RollingSummary {
    fn default() -> Self {
        Self::new(HISTORY_CAPACITY)
    }
}

/// Allocation-conscious scalar observation model used throughout the test.
struct ScalarObservationModel;

impl ObservationModel<Reading> for ScalarObservationModel {
    type Error = Infallible;
    type Summary = RollingSummary;

    fn likelihood(
        &self,
        summary: &Self::Summary,
        payload: &Reading,
    ) -> Probability {
        match summary.mean() {
            Some(mean) => {
                Probability::new((-(payload.value - mean).abs()).exp())
            },
            None => Probability::ONE,
        }
    }

    fn update(
        &self,
        summary: &mut Self::Summary,
        observation: &Observation<Reading>,
    ) -> Result<(), Self::Error> {
        summary.record(observation.payload.value);
        Ok(())
    }

    fn merge(
        &self,
        target: &mut Self::Summary,
        other: &Self::Summary,
    ) -> Result<(), Self::Error> {
        for value in other.history.iter().copied() {
            target.record(value);
        }
        Ok(())
    }

    fn decay(
        &self,
        _summary: &mut Self::Summary,
        _delta: Duration,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn checkpoint(
        &self,
        summary: &Self::Summary,
        output: &mut Vec<u8>,
    ) -> Result<(), Self::Error> {
        output.push(summary.history.len() as u8);
        for value in summary.history.iter() {
            output.extend_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }
}

/// Normalized categorical compatibility backed by the observation model.
struct ScalarCompatibility {
    model: Arc<ScalarObservationModel>,
}

impl MultiCandidateCompatibility<Reading, RollingSummary>
    for ScalarCompatibility
{
    fn evaluate_joint(
        &self,
        observation: &Observation<Reading>,
        beliefs: &[&BeliefState<RollingSummary>],
    ) -> KCandidateDistribution {
        let candidate_weight = |belief: &&BeliefState<RollingSummary>| {
            self.model
                .likelihood(&belief.summary, &observation.payload)
                .value() *
                observation.confidence.value()
        };
        let total_weight = beliefs.iter().map(candidate_weight).sum::<f64>() +
            BACKGROUND_WEIGHT;
        let mut candidates = HashMap::with_capacity(beliefs.len());

        for belief in beliefs {
            let probability =
                candidate_weight(belief) / total_weight.max(f64::MIN_POSITIVE);
            candidates.insert(belief.identity, Probability::new(probability));
        }

        KCandidateDistribution::new(
            candidates,
            Probability::new(
                BACKGROUND_WEIGHT / total_weight.max(f64::MIN_POSITIVE),
            ),
        )
    }

    fn evaluate_joint_stream(
        &self,
        observation: &Observation<Reading>,
        beliefs: &[&BeliefState<RollingSummary>],
        emit: &mut dyn FnMut(IdentityId, Probability),
    ) -> Probability {
        let candidate_weight = |belief: &&BeliefState<RollingSummary>| {
            self.model
                .likelihood(&belief.summary, &observation.payload)
                .value() *
                observation.confidence.value()
        };
        let total_weight = beliefs.iter().map(candidate_weight).sum::<f64>() +
            BACKGROUND_WEIGHT;
        let denominator = total_weight.max(f64::MIN_POSITIVE);

        for belief in beliefs {
            emit(
                belief.identity,
                Probability::new(candidate_weight(belief) / denominator),
            );
        }
        Probability::new(BACKGROUND_WEIGHT / denominator)
    }
}

/// Observable details from one ephemeral inference execution.
struct InferenceOutcome {
    assignment: MapAssignment,
    posteriors: PosteriorDistribution,
    variable_count: usize,
    factor_count: usize,
    connected_scope_edges: usize,
}

/// Test-only failures for missing pipeline state that assertions cannot
/// express.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipelineTestError {
    MissingBelief(IdentityId),
    MissingMarginal(IdentityId),
    MissingScopeVariable(IdentityId),
    ExpectedExistingAssignment,
    MissingProjectionEndpoint,
    MissingProjectionWeight,
}

impl fmt::Display for PipelineTestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingBelief(identity) => {
                write!(formatter, "missing belief for {identity}")
            },
            Self::MissingMarginal(identity) => {
                write!(formatter, "missing marginal for {identity}")
            },
            Self::MissingScopeVariable(identity) => {
                write!(
                    formatter,
                    "factor scope has no variable for {identity}"
                )
            },
            Self::ExpectedExistingAssignment => {
                formatter.write_str("MAP returned the background assignment")
            },
            Self::MissingProjectionEndpoint => {
                formatter.write_str("projected edge has no endpoints")
            },
            Self::MissingProjectionWeight => {
                formatter.write_str("projected edge has no weight")
            },
        }
    }
}

impl Error for PipelineTestError {}

/// Compiles evidence, connects every factor scope, runs BP, and estimates MAP.
fn infer(
    evidence: &Evidence<Reading>,
    workspace: &InMemoryWorkspace<RollingSummary>,
    compiler: &CategoricalFactorCompiler<Reading, RollingSummary>,
    solver: &BeliefPropagationSolver,
    estimator: &MapEstimator,
    threshold: Probability,
) -> Result<InferenceOutcome, PipelineTestError> {
    let active_beliefs = workspace.active_beliefs();
    let factors = compiler.compile_factors(evidence, &active_beliefs);
    let mut factor_graph =
        FactorGraph::with_capacity(evidence.candidates.len(), factors.len());
    let mut variables = Vec::<(IdentityId, VarIndex)>::with_capacity(
        evidence.candidates.len(),
    );

    for candidate in evidence.candidates.iter().copied() {
        variables.push((candidate, factor_graph.add_variable(candidate)));
    }

    for factor in factors {
        let mut scope = Vec::with_capacity(factor.scope().len());
        for identity in factor.scope().iter().copied() {
            let variable = variables
                .iter()
                .find_map(|(candidate, index)| {
                    (*candidate == identity).then_some(*index)
                })
                .ok_or(PipelineTestError::MissingScopeVariable(identity))?;
            scope.push(variable);
        }
        factor_graph.add_factor(factor, &scope);
    }

    let variable_count = factor_graph.variables.len();
    let factor_count = factor_graph.factors.len();
    let connected_scope_edges =
        factor_graph.factor_adjacencies.iter().map(Vec::len).sum();
    let posteriors = solver.solve(&factor_graph);
    let assignment = estimator.estimate_map(&posteriors, threshold);

    Ok(InferenceOutcome {
        assignment,
        posteriors,
        variable_count,
        factor_count,
        connected_scope_edges,
    })
}

/// Builds an immutable observation with the test modality.
fn observation(id: u64, seconds: i64, value: f64) -> Observation<Reading> {
    Observation::new(
        ObservationId(id),
        Modality(1),
        Timestamp::from_secs(seconds),
        Confidence::new(1.0),
        Reading { value },
    )
}

/// Captures the exact typed topology of an Event-State projection.
fn projection_signature(
    projection: &EventStateGraph<u32, u32>,
) -> Result<ProjectionSignature, PipelineTestError> {
    let graph = projection.graph();
    let mut nodes = HashSet::with_capacity(graph.node_count());
    let mut edges = HashSet::with_capacity(graph.edge_count());

    for node in graph.node_weights() {
        nodes.insert(node.vertex());
    }
    for edge in graph.edge_indices() {
        let (source, target) = graph
            .edge_endpoints(edge)
            .ok_or(PipelineTestError::MissingProjectionEndpoint)?;
        let relation = graph
            .edge_weight(edge)
            .ok_or(PipelineTestError::MissingProjectionWeight)?
            .relation;
        edges.insert((
            graph[source].vertex(),
            relation,
            graph[target].vertex(),
        ));
    }

    Ok((nodes, edges))
}

/// Runs the complete paper pipeline through inference and materialization.
#[test]
fn paper_pipeline_preserves_graph_invariants_and_bounded_state()
-> Result<(), Box<dyn Error>> {
    let model = Arc::new(ScalarObservationModel);
    let compatibility = Arc::new(ScalarCompatibility {
        model: Arc::clone(&model),
    });
    let compiler = CategoricalFactorCompiler::new(compatibility);
    let solver = BeliefPropagationSolver::new(BpConfig {
        max_iterations: 32,
        convergence_threshold: 1e-9,
    });
    let estimator = MapEstimator::new();
    let threshold = Probability::new(0.8);
    let canonical = IdentityId(100);
    let duplicate = IdentityId(200);
    let mut workspace = InMemoryWorkspace::<RollingSummary>::new();
    let mut graph = PetGraphStore::<Reading, u32, u32>::with_capacity(12, 12);

    // With no candidate variables, BP and MAP select the background path.
    let first = Evidence::new(observation(1, 1, 10.0), Vec::new());
    let first_inference = infer(
        &first, &workspace, &compiler, &solver, &estimator, threshold,
    )?;
    assert_eq!(first_inference.variable_count, 0);
    assert_eq!(first_inference.factor_count, 0);
    assert_eq!(first_inference.connected_scope_edges, 0);
    assert_eq!(first_inference.assignment.selected_identity, None);

    let mut canonical_summary = RollingSummary::default();
    model.update(&mut canonical_summary, &first.observation)?;
    workspace.insert(BeliefState::new(
        canonical,
        canonical_summary,
        Probability::ONE,
        first.observation.timestamp,
    ));
    assert_eq!(
        graph.materialize_observation(
            first.observation,
            IdentityAssignment::New(canonical),
        )?,
        canonical
    );

    // A distant observation makes the background hypothesis win over its
    // candidate and therefore materializes a second identity.
    let distant = Evidence::new(observation(2, 2, 25.0), vec![canonical]);
    let distant_beliefs = distant
        .candidates
        .iter()
        .filter_map(|identity| workspace.get(*identity))
        .collect::<Vec<_>>();
    assert_eq!(
        compiler.try_direct_map(&distant, &distant_beliefs, threshold),
        DirectMapDecision::CreateIdentity
    );
    let distant_inference = infer(
        &distant, &workspace, &compiler, &solver, &estimator, threshold,
    )?;
    assert_eq!(distant_inference.variable_count, 1);
    assert_eq!(distant_inference.factor_count, 1);
    assert_eq!(distant_inference.connected_scope_edges, 1);
    assert_eq!(distant_inference.assignment.selected_identity, None);

    let mut duplicate_summary = RollingSummary::default();
    model.update(&mut duplicate_summary, &distant.observation)?;
    workspace.insert(BeliefState::new(
        duplicate,
        duplicate_summary,
        Probability::ONE,
        distant.observation.timestamp,
    ));
    assert_eq!(
        graph.materialize_observation(
            distant.observation,
            IdentityAssignment::New(duplicate),
        )?,
        duplicate
    );

    // Spatial lookup remains local while evidence construction canonicalizes
    // any duplicate identifiers supplied by an upstream index.
    let deduplicated =
        Evidence::new(observation(99, 3, 10.2), vec![canonical, canonical]);
    assert_eq!(deduplicated.candidates, vec![canonical]);
    let index_config = SpatialGridConfig::try_new(1.0, 1.0, 4)?;
    let mut candidate_index = SpatialGridIndex::with_capacity(index_config, 2);
    candidate_index.insert(canonical, SpatialPoint::try_new(10.0, 0.0)?)?;
    candidate_index.insert(duplicate, SpatialPoint::try_new(25.0, 0.0)?)?;
    let mut candidate_buffer = Vec::with_capacity(4);
    let candidate_capacity = candidate_buffer.capacity();

    // Two nearby observations exercise indexed candidate extraction,
    // connected categorical factors, BP, direct MAP, belief updates, and
    // graph materialization.
    for nearby_observation in
        [observation(3, 3, 10.2), observation(4, 4, 10.4)]
    {
        candidate_index.generate_candidates_into(
            &nearby_observation,
            &mut candidate_buffer,
        );
        assert_eq!(candidate_buffer, vec![canonical]);
        assert_eq!(candidate_buffer.capacity(), candidate_capacity);
        let evidence =
            Evidence::new(nearby_observation, candidate_buffer.clone());
        let candidate_beliefs = evidence
            .candidates
            .iter()
            .filter_map(|identity| workspace.get(*identity))
            .collect::<Vec<_>>();
        assert_eq!(
            compiler.try_direct_map(&evidence, &candidate_beliefs, threshold,),
            DirectMapDecision::Assign(canonical)
        );
        let inference = infer(
            &evidence, &workspace, &compiler, &solver, &estimator, threshold,
        )?;
        assert_eq!(inference.variable_count, 1);
        assert_eq!(inference.factor_count, 1);
        assert_eq!(inference.connected_scope_edges, 1);
        let selected = inference
            .assignment
            .selected_identity
            .ok_or(PipelineTestError::ExpectedExistingAssignment)?;
        assert_eq!(selected, canonical);
        let posterior = inference
            .posteriors
            .find_marginal(canonical)
            .map(|marginal| marginal.probability)
            .ok_or(PipelineTestError::MissingMarginal(canonical))?;
        assert!(posterior.value() >= threshold.value());

        let belief = workspace
            .get_mut(canonical)
            .ok_or(PipelineTestError::MissingBelief(canonical))?;
        model.update(&mut belief.summary, &evidence.observation)?;
        belief.update_posterior(posterior, evidence.observation.timestamp);
        assert_eq!(
            graph.materialize_observation(
                evidence.observation,
                IdentityAssignment::Existing(selected),
            )?,
            canonical
        );
    }

    let canonical_belief = workspace
        .get(canonical)
        .ok_or(PipelineTestError::MissingBelief(canonical))?;
    assert_eq!(
        canonical_belief.summary.history.capacity(),
        HISTORY_CAPACITY
    );
    assert_eq!(canonical_belief.summary.history.len(), HISTORY_CAPACITY);
    assert_eq!(
        canonical_belief
            .summary
            .history
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![10.2, 10.4]
    );

    // A zero-sized history rejects data without allocating or retaining it.
    let mut zero_history = BoundedHistory::new(0);
    assert_eq!(zero_history.push(7_u8), Some(7_u8));
    assert_eq!(zero_history.capacity(), 0);
    assert!(zero_history.is_empty());

    let causal_timestamp = Timestamp::from_secs(5);
    graph.apply_batch([
        GraphOperation::CommitEvent {
            id: EventId(1),
            timestamp: causal_timestamp,
            payload: 11,
        },
        GraphOperation::CommitEvent {
            id: EventId(2),
            timestamp: causal_timestamp,
            payload: 12,
        },
        GraphOperation::CommitState {
            id: StateId(1),
            timestamp: causal_timestamp,
            payload: 21,
        },
        GraphOperation::CommitState {
            id: StateId(2),
            timestamp: causal_timestamp,
            payload: 22,
        },
        GraphOperation::CommitRelation {
            source: Vertex::Event(EventId(1)),
            relation: Relation::Influence,
            target: Vertex::Event(EventId(2)),
            created_at: causal_timestamp,
        },
        GraphOperation::CommitRelation {
            source: Vertex::Event(EventId(1)),
            relation: Relation::Trigger,
            target: Vertex::State(StateId(1)),
            created_at: causal_timestamp,
        },
        GraphOperation::CommitRelation {
            source: Vertex::State(StateId(1)),
            relation: Relation::Evolution,
            target: Vertex::State(StateId(2)),
            created_at: causal_timestamp,
        },
        GraphOperation::CommitRelation {
            source: Vertex::Observation(ObservationId(1)),
            relation: Relation::ObservedDuring,
            target: Vertex::State(StateId(1)),
            created_at: causal_timestamp,
        },
    ])?;

    ObservationPartitionInvariant.validate(&graph)?;
    IdentityUniquenessInvariant.validate(&graph)?;
    CausalAcyclicityInvariant.validate(&graph)?;

    let projection_before = EventStateProjection::project(&graph);
    let signature_before = projection_signature(&projection_before)?;
    assert_eq!(signature_before.0.len(), 4);
    assert_eq!(signature_before.1.len(), 3);
    assert!(signature_before.1.contains(&(
        Vertex::Event(EventId(1)),
        Relation::Influence,
        Vertex::Event(EventId(2)),
    )));
    assert!(signature_before.1.contains(&(
        Vertex::Event(EventId(1)),
        Relation::Trigger,
        Vertex::State(StateId(1)),
    )));
    assert!(signature_before.1.contains(&(
        Vertex::State(StateId(1)),
        Relation::Evolution,
        Vertex::State(StateId(2)),
    )));

    // Algorithm 2 merges both the rolling summaries and graph support sets.
    let duplicate_belief = workspace
        .beliefs
        .remove(&duplicate)
        .ok_or(PipelineTestError::MissingBelief(duplicate))?;
    let canonical_belief = workspace
        .get_mut(canonical)
        .ok_or(PipelineTestError::MissingBelief(canonical))?;
    model.merge(&mut canonical_belief.summary, &duplicate_belief.summary)?;
    canonical_belief.update_posterior(
        Probability::new(
            canonical_belief
                .posterior
                .value()
                .max(duplicate_belief.posterior.value()),
        ),
        duplicate_belief.last_update,
    );

    graph.apply_batch([GraphOperation::MergeIdentities {
        target: canonical,
        duplicate,
    }])?;

    assert_eq!(workspace.len(), 1);
    assert!(workspace.get(duplicate).is_none());
    assert!(!graph.contains_vertex(Vertex::Identity(duplicate)));
    let mut support_ids: Vec<_> = graph
        .supporting_observations(canonical)?
        .map(|observation| observation.id)
        .collect();
    support_ids.sort_unstable();
    assert_eq!(
        support_ids,
        vec![
            ObservationId(1),
            ObservationId(2),
            ObservationId(3),
            ObservationId(4),
        ]
    );

    let merged_belief = workspace
        .get(canonical)
        .ok_or(PipelineTestError::MissingBelief(canonical))?;
    assert_eq!(merged_belief.summary.history.len(), HISTORY_CAPACITY);
    assert_eq!(
        merged_belief
            .summary
            .history
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![10.4, 25.0]
    );
    let mut checkpoint = Vec::with_capacity(17);
    model.checkpoint(&merged_belief.summary, &mut checkpoint)?;
    assert_eq!(checkpoint.len(), 17);

    ObservationPartitionInvariant.validate(&graph)?;
    IdentityUniquenessInvariant.validate(&graph)?;
    CausalAcyclicityInvariant.validate(&graph)?;
    let projection_after = EventStateProjection::project(&graph);
    assert_eq!(projection_signature(&projection_after)?, signature_before);

    Ok(())
}
