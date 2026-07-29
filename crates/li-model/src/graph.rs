//! Mathematical graph interface decoupling topology from storage drivers.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use li_core::ids::IdentityId;
use li_core::observation::Observation;
use li_core::ontology::Vertex;
use li_core::relation::Relation;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableGraph;
use petgraph::visit::EdgeRef;
use petgraph::{Directed, Direction};
use smallvec::SmallVec;
use thiserror::Error;

use crate::ontology::{EdgeData, NodeData};
use crate::operations::{GraphOperation, IdentityAssignment};

/// Concrete in-memory graph representation used by [`PetGraphStore`].
pub type RawGraph<P, E, S> =
    StableGraph<NodeData<P, E, S>, EdgeData, Directed, u32>;

/// Statistics produced by an identity canonicalization operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MergeOutcome {
    /// Relations moved from the duplicate to the canonical identity.
    pub rewired_relations: usize,
    /// Relations omitted because the canonical identity already had them.
    pub coalesced_relations: usize,
}

/// Structural and topological errors produced during graph state transitions.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GraphError {
    /// Attempted to establish an edge violating domain/codomain ontology
    /// schema rules.
    #[error("Invalid relation {relation:?} between {source:?} and {target:?}")]
    InvalidRelationTransition {
        relation: Relation,
        r#source: Vertex,
        target: Vertex,
    },

    /// Referenced source or target vertex does not exist in the active
    /// ontology set V.
    #[error("Vertex not found in graph topology: {vertex:?}")]
    VertexNotFound { vertex: Vertex },

    /// Attempted to insert a vertex whose typed domain ID already exists in V.
    #[error("Vertex already registered in ontology: {vertex:?}")]
    DuplicateVertex { vertex: Vertex },

    /// Attempted to insert an already registered semantic relation.
    #[error(
        "Relation {relation:?} already exists between {origin:?} and {target:?}"
    )]
    DuplicateRelation {
        /// Origin vertex of the duplicate relation.
        origin: Vertex,
        /// Semantic relation that already exists.
        relation: Relation,
        /// Destination vertex of the duplicate relation.
        target: Vertex,
    },

    /// Attempted to assign an observation that already supports an identity.
    #[error(
        "Observation {observation:?} already supports identity {identity:?}"
    )]
    ObservationAlreadyAssigned {
        /// Observation with an existing support owner.
        observation: li_core::ids::ObservationId,
        /// Existing support owner.
        identity: IdentityId,
    },

    /// Attempted to merge an identity into itself.
    #[error("Cannot merge identity {identity:?} into itself")]
    SelfIdentityMerge {
        /// Identity present in both merge roles.
        identity: IdentityId,
    },

    /// Merge operations must be isolated so every backend can apply them
    /// atomically.
    #[error(
        "Identity merge operations must be committed in an isolated batch"
    )]
    MergeOperationMustBeIsolated,

    /// The node index and typed vertex index disagree.
    #[error("Graph index is inconsistent for vertex {vertex:?}")]
    CorruptIndex {
        /// Vertex whose mapped node is absent or has a different type.
        vertex: Vertex,
    },
}

#[derive(Debug, Clone, Copy)]
enum Undo {
    Node {
        vertex: Vertex,
        index: NodeIndex<u32>,
    },
    Edge(EdgeIndex<u32>),
}

type IdentityRewire = (NodeIndex<u32>, NodeIndex<u32>, EdgeData);

/// Abstract interface representing the persistent knowledge graph G = (V, R).
pub trait KnowledgeGraph {
    /// Modality observation payload type.
    type ObservationPayload;
    /// Event occurrence payload type.
    type EventPayload;
    /// Entity state snapshot payload type.
    type StatePayload;
    /// Fallible operation error type.
    type Error;

    /// Verifies if a domain vertex exists in V and returns its typed
    /// representation.
    fn vertex_type(
        &self,
        vertex: Vertex,
    ) -> Result<Option<Vertex>, Self::Error>;

    /// Applies a batch of formal operational primitives sequentially to
    /// transition graph state. Consumes operations to prevent unnecessary
    /// cloning.
    fn apply_batch(
        &mut self,
        ops: impl IntoIterator<
            Item = GraphOperation<
                Self::ObservationPayload,
                Self::EventPayload,
                Self::StatePayload,
            >,
        >,
    ) -> Result<(), Self::Error>;

    /// Traverses incoming edges to recover the support set supp(i)
    /// supporting an identity.
    fn query_support_set(
        &self,
        identity: IdentityId,
    ) -> Result<Vec<Observation<Self::ObservationPayload>>, Self::Error>
    where
        Self::ObservationPayload: Clone;

    /// Enumerates all active identity identifiers in I.
    fn all_identities(&self) -> Result<Vec<IdentityId>, Self::Error>;
}

/// Memory graph backed natively by `petgraph`.
#[derive(Debug, Clone)]
pub struct PetGraphStore<P, E, S> {
    pub(crate) raw_graph: RawGraph<P, E, S>,
    pub(crate) index_map: HashMap<Vertex, NodeIndex<u32>>,
}

impl<P, E, S> Default for PetGraphStore<P, E, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P, E, S> PetGraphStore<P, E, S> {
    /// Instantiates an empty `PetGraphStore` instance.
    pub fn new() -> Self {
        Self::with_capacity(0, 0)
    }

    /// Instantiates a graph with capacity for a known working set.
    ///
    /// # Arguments
    ///
    /// * `node_capacity` - Expected number of vertices.
    /// * `edge_capacity` - Expected number of semantic relations.
    pub fn with_capacity(node_capacity: usize, edge_capacity: usize) -> Self {
        Self {
            raw_graph: StableGraph::with_capacity(
                node_capacity,
                edge_capacity,
            ),
            index_map: HashMap::with_capacity(node_capacity),
        }
    }

    /// Reserves storage before a real-time ingestion burst.
    ///
    /// Calling this method outside the hot path lets subsequent commits reuse
    /// graph and index capacity.
    pub fn reserve(
        &mut self,
        additional_nodes: usize,
        additional_edges: usize,
    ) {
        self.raw_graph.reserve_nodes(additional_nodes);
        self.raw_graph.reserve_edges(additional_edges);
        self.index_map.reserve(additional_nodes);
    }

    /// Returns a read-only view of the underlying petgraph.
    pub fn graph(&self) -> &RawGraph<P, E, S> {
        &self.raw_graph
    }

    /// Returns the number of registered vertices.
    pub fn node_count(&self) -> usize {
        self.raw_graph.node_count()
    }

    /// Returns the number of registered relations.
    pub fn edge_count(&self) -> usize {
        self.raw_graph.edge_count()
    }

    /// Returns `true` when the typed vertex exists.
    pub fn contains_vertex(&self, vertex: Vertex) -> bool {
        self.index_map.contains_key(&vertex)
    }

    /// Returns the immutable data associated with a typed vertex.
    pub fn node_data(&self, vertex: Vertex) -> Option<&NodeData<P, E, S>> {
        self.index_map
            .get(&vertex)
            .and_then(|&index| self.raw_graph.node_weight(index))
    }

    /// Iterates over active identity identifiers without allocating.
    pub fn identity_ids(&self) -> impl Iterator<Item = IdentityId> + '_ {
        self.raw_graph.node_weights().filter_map(|node| match node {
            NodeData::Identity { id, .. } => Some(*id),
            _ => None,
        })
    }

    /// Iterates over an identity's supporting observations without cloning.
    pub fn supporting_observations(
        &self,
        identity: IdentityId,
    ) -> Result<impl Iterator<Item = &Observation<P>> + '_, GraphError> {
        let vertex = Vertex::Identity(identity);
        let target = self.checked_index(vertex)?;

        Ok(self
            .raw_graph
            .edges_directed(target, Direction::Incoming)
            .filter_map(|edge| {
                if edge.weight().relation != Relation::Supports {
                    return None;
                }
                match &self.raw_graph[edge.source()] {
                    NodeData::Observation(observation) => Some(observation),
                    _ => None,
                }
            }))
    }

    /// Atomically commits an observation and its resolved identity assignment.
    ///
    /// The `New` path creates the identity in the same transaction; the
    /// `Existing` path rolls the observation back if the target is absent.
    pub fn materialize_observation(
        &mut self,
        observation: Observation<P>,
        assignment: IdentityAssignment,
    ) -> Result<IdentityId, GraphError> {
        let observation_id = observation.id;
        let timestamp = observation.timestamp;

        match assignment {
            IdentityAssignment::Existing(identity) => {
                self.apply_batch([
                    GraphOperation::CommitObservation(observation),
                    GraphOperation::CommitRelation {
                        source: Vertex::Observation(observation_id),
                        relation: Relation::Supports,
                        target: Vertex::Identity(identity),
                        created_at: timestamp,
                    },
                ])?;
                Ok(identity)
            },
            IdentityAssignment::New(identity) => {
                self.apply_batch([
                    GraphOperation::CommitObservation(observation),
                    GraphOperation::CommitIdentity {
                        id: identity,
                        created_at: timestamp,
                    },
                    GraphOperation::CommitRelation {
                        source: Vertex::Observation(observation_id),
                        relation: Relation::Supports,
                        target: Vertex::Identity(identity),
                        created_at: timestamp,
                    },
                ])?;
                Ok(identity)
            },
        }
    }

    /// Merges a duplicate identity into a canonical target.
    ///
    /// All incoming and outgoing LI relations are moved to `target`, duplicate
    /// relations are coalesced, and the duplicate vertex is removed. Since
    /// causal ESKG relations have only event/state endpoints, this operation
    /// cannot alter the event-state projection.
    pub fn merge_identities(
        &mut self,
        target: IdentityId,
        duplicate: IdentityId,
    ) -> Result<MergeOutcome, GraphError> {
        if target == duplicate {
            return Err(GraphError::SelfIdentityMerge { identity: target });
        }

        let target_vertex = Vertex::Identity(target);
        let duplicate_vertex = Vertex::Identity(duplicate);
        let target_index = self.checked_index(target_vertex)?;
        let duplicate_index = self.checked_index(duplicate_vertex)?;

        let incident_capacity = self
            .raw_graph
            .edges_directed(duplicate_index, Direction::Incoming)
            .count()
            .saturating_add(
                self.raw_graph
                    .edges_directed(duplicate_index, Direction::Outgoing)
                    .count(),
            );
        let mut rewires: SmallVec<[IdentityRewire; 8]> =
            SmallVec::with_capacity(incident_capacity);

        for edge in self
            .raw_graph
            .edges_directed(duplicate_index, Direction::Incoming)
        {
            let source = edge.source();
            if source != target_index && source != duplicate_index {
                rewires.push((source, target_index, *edge.weight()));
            }
        }

        for edge in self
            .raw_graph
            .edges_directed(duplicate_index, Direction::Outgoing)
        {
            let destination = edge.target();
            if destination != target_index && destination != duplicate_index {
                rewires.push((target_index, destination, *edge.weight()));
            }
        }

        if self.raw_graph.node_weight(duplicate_index).is_none() {
            return Err(GraphError::CorruptIndex {
                vertex: duplicate_vertex,
            });
        }

        let _removed = self.raw_graph.remove_node(duplicate_index);
        self.index_map.remove(&duplicate_vertex);

        let mut rewired_relations = 0usize;
        let mut coalesced_relations = 0usize;
        for (source, destination, edge) in rewires {
            if self.relation_exists(source, edge.relation, destination) {
                coalesced_relations = coalesced_relations.saturating_add(1);
            } else {
                self.raw_graph.add_edge(source, destination, edge);
                rewired_relations = rewired_relations.saturating_add(1);
            }
        }

        Ok(MergeOutcome {
            rewired_relations,
            coalesced_relations,
        })
    }

    fn checked_index(
        &self,
        vertex: Vertex,
    ) -> Result<NodeIndex<u32>, GraphError> {
        let index = *self
            .index_map
            .get(&vertex)
            .ok_or(GraphError::VertexNotFound { vertex })?;
        match self.raw_graph.node_weight(index) {
            Some(node) if node.vertex() == vertex => Ok(index),
            _ => Err(GraphError::CorruptIndex { vertex }),
        }
    }

    fn relation_exists(
        &self,
        source: NodeIndex<u32>,
        relation: Relation,
        target: NodeIndex<u32>,
    ) -> bool {
        self.raw_graph
            .edges_connecting(source, target)
            .any(|edge| edge.weight().relation == relation)
    }

    fn support_owner(
        &self,
        observation: NodeIndex<u32>,
    ) -> Option<IdentityId> {
        self.raw_graph
            .edges_directed(observation, Direction::Outgoing)
            .find_map(|edge| {
                if edge.weight().relation != Relation::Supports {
                    return None;
                }
                match &self.raw_graph[edge.target()] {
                    NodeData::Identity { id, .. } => Some(*id),
                    _ => None,
                }
            })
    }

    fn insert_node(
        &mut self,
        vertex: Vertex,
        node: NodeData<P, E, S>,
    ) -> Result<Undo, GraphError> {
        match self.index_map.entry(vertex) {
            Entry::Occupied(_) => Err(GraphError::DuplicateVertex { vertex }),
            Entry::Vacant(entry) => {
                let index = self.raw_graph.add_node(node);
                entry.insert(index);
                Ok(Undo::Node { vertex, index })
            },
        }
    }

    /// Processes an individual graph operation commit by consuming it,
    /// enforcing schema validity and uniqueness without heap allocations.
    fn commit_op(
        &mut self,
        op: GraphOperation<P, E, S>,
    ) -> Result<Undo, GraphError> {
        match op {
            GraphOperation::CommitObservation(obs) => {
                let vertex = Vertex::Observation(obs.id);
                self.insert_node(vertex, NodeData::Observation(obs))
            },
            GraphOperation::CommitIdentity { id, created_at } => {
                let vertex = Vertex::Identity(id);
                self.insert_node(vertex, NodeData::Identity { id, created_at })
            },
            GraphOperation::CommitEvent {
                id,
                timestamp,
                payload,
            } => {
                let vertex = Vertex::Event(id);
                self.insert_node(
                    vertex,
                    NodeData::Event {
                        id,
                        timestamp,
                        payload,
                    },
                )
            },
            GraphOperation::CommitState {
                id,
                timestamp,
                payload,
            } => {
                let vertex = Vertex::State(id);
                self.insert_node(
                    vertex,
                    NodeData::State {
                        id,
                        timestamp,
                        payload,
                    },
                )
            },
            GraphOperation::CommitRelation {
                source,
                relation,
                target,
                created_at,
            } => {
                if !relation.is_valid_transition(&source, &target) {
                    return Err(GraphError::InvalidRelationTransition {
                        relation,
                        source,
                        target,
                    });
                }

                let src_idx = self.checked_index(source)?;
                let tgt_idx = self.checked_index(target)?;

                if relation == Relation::Supports &&
                    let Some(identity) = self.support_owner(src_idx)
                {
                    let Vertex::Observation(observation) = source else {
                        return Err(GraphError::InvalidRelationTransition {
                            relation,
                            source,
                            target,
                        });
                    };
                    return Err(GraphError::ObservationAlreadyAssigned {
                        observation,
                        identity,
                    });
                }

                if self.relation_exists(src_idx, relation, tgt_idx) {
                    return Err(GraphError::DuplicateRelation {
                        origin: source,
                        relation,
                        target,
                    });
                }

                Ok(Undo::Edge(self.raw_graph.add_edge(
                    src_idx,
                    tgt_idx,
                    EdgeData {
                        relation,
                        created_at,
                    },
                )))
            },
            GraphOperation::MergeIdentities { .. } => {
                Err(GraphError::MergeOperationMustBeIsolated)
            },
        }
    }

    fn rollback(&mut self, journal: &mut SmallVec<[Undo; 8]>) {
        while let Some(undo) = journal.pop() {
            match undo {
                Undo::Edge(index) => {
                    let _removed = self.raw_graph.remove_edge(index);
                },
                Undo::Node { vertex, index } => {
                    let _removed = self.raw_graph.remove_node(index);
                    self.index_map.remove(&vertex);
                },
            }
        }
    }
}

impl<P, E, S> KnowledgeGraph for PetGraphStore<P, E, S> {
    type Error = GraphError;
    type EventPayload = E;
    type ObservationPayload = P;
    type StatePayload = S;

    fn vertex_type(
        &self,
        vertex: Vertex,
    ) -> Result<Option<Vertex>, Self::Error> {
        if self.index_map.contains_key(&vertex) {
            self.checked_index(vertex).map(|_| Some(vertex))
        } else {
            Ok(None)
        }
    }

    fn apply_batch(
        &mut self,
        ops: impl IntoIterator<Item = GraphOperation<P, E, S>>,
    ) -> Result<(), Self::Error> {
        let mut operations = ops.into_iter();
        let Some(first) = operations.next() else {
            return Ok(());
        };

        let first = match first {
            GraphOperation::MergeIdentities { target, duplicate } => {
                if operations.next().is_some() {
                    return Err(GraphError::MergeOperationMustBeIsolated);
                }
                self.merge_identities(target, duplicate)?;
                return Ok(());
            },
            operation => operation,
        };

        let mut journal = SmallVec::<[Undo; 8]>::new();
        journal.push(self.commit_op(first)?);

        for operation in operations {
            if matches!(operation, GraphOperation::MergeIdentities { .. }) {
                self.rollback(&mut journal);
                return Err(GraphError::MergeOperationMustBeIsolated);
            }

            match self.commit_op(operation) {
                Ok(undo) => journal.push(undo),
                Err(error) => {
                    self.rollback(&mut journal);
                    return Err(error);
                },
            }
        }
        Ok(())
    }

    fn query_support_set(
        &self,
        identity: IdentityId,
    ) -> Result<Vec<Observation<P>>, Self::Error>
    where
        P: Clone,
    {
        self.supporting_observations(identity)
            .map(|observations| observations.cloned().collect())
    }

    fn all_identities(&self) -> Result<Vec<IdentityId>, Self::Error> {
        Ok(self.identity_ids().collect())
    }
}

#[cfg(test)]
mod tests {
    use li_core::ids::{EventId, IdentityId, ObservationId, StateId};
    use li_core::observation::{Modality, Observation, Timestamp};
    use li_core::probability::Confidence;

    use super::*;

    #[test]
    fn test_duplicate_vertex_rejection() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let obs = Observation::new(
            ObservationId(1),
            Modality(1),
            Timestamp::from_secs(100),
            Confidence::new(0.9),
            (),
        );

        let op1 = GraphOperation::CommitObservation(obs.clone());
        let op2 = GraphOperation::CommitObservation(obs);

        assert!(store.apply_batch(vec![op1]).is_ok());
        let result = store.apply_batch(vec![op2]);
        assert_eq!(
            result,
            Err(GraphError::DuplicateVertex {
                vertex: Vertex::Observation(ObservationId(1))
            })
        );
    }

    #[test]
    fn test_invalid_relation_transition() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let obs = Observation::new(
            ObservationId(1),
            Modality(1),
            Timestamp::from_secs(100),
            Confidence::new(0.9),
            (),
        );

        let setup = vec![
            GraphOperation::CommitObservation(obs),
            GraphOperation::CommitIdentity {
                id: IdentityId(1),
                created_at: Timestamp::from_secs(100),
            },
        ];
        assert!(store.apply_batch(setup).is_ok());

        let invalid_op = GraphOperation::CommitRelation {
            source: Vertex::Identity(IdentityId(1)),
            relation: Relation::Supports,
            target: Vertex::Observation(ObservationId(1)),
            created_at: Timestamp::from_secs(101),
        };

        let result = store.apply_batch(vec![invalid_op]);
        assert_eq!(
            result,
            Err(GraphError::InvalidRelationTransition {
                relation: Relation::Supports,
                source: Vertex::Identity(IdentityId(1)),
                target: Vertex::Observation(ObservationId(1)),
            })
        );
    }

    #[test]
    fn test_missing_vertex_in_relation() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let op = GraphOperation::CommitRelation {
            source: Vertex::Observation(ObservationId(99)),
            relation: Relation::Supports,
            target: Vertex::Identity(IdentityId(1)),
            created_at: Timestamp::from_secs(100),
        };

        let result = store.apply_batch(vec![op]);
        assert_eq!(
            result,
            Err(GraphError::VertexNotFound {
                vertex: Vertex::Observation(ObservationId(99))
            })
        );
    }

    #[test]
    fn test_vertex_type_reports_corrupt_index() {
        let identity = IdentityId(1);
        let vertex = Vertex::Identity(identity);
        let mut store = PetGraphStore::<(), (), ()>::new();
        assert!(
            store
                .apply_batch([GraphOperation::CommitIdentity {
                    id: identity,
                    created_at: Timestamp::UNIX_EPOCH,
                }])
                .is_ok()
        );

        if let Some(index) = store.index_map.get(&vertex).copied() {
            let _removed = store.raw_graph.remove_node(index);
        }

        assert_eq!(
            store.vertex_type(vertex),
            Err(GraphError::CorruptIndex { vertex })
        );
    }

    #[test]
    fn test_query_support_set_non_existent_identity() {
        let store = PetGraphStore::<(), (), ()>::new();
        let result = store.query_support_set(IdentityId(404));
        assert_eq!(
            result,
            Err(GraphError::VertexNotFound {
                vertex: Vertex::Identity(IdentityId(404))
            })
        );
    }

    #[test]
    fn test_query_support_set_empty() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let op = GraphOperation::CommitIdentity {
            id: IdentityId(1),
            created_at: Timestamp::from_secs(100),
        };
        assert!(store.apply_batch(vec![op]).is_ok());

        assert_eq!(
            store
                .query_support_set(IdentityId(1))
                .map(|support| support.is_empty()),
            Ok(true)
        );
    }

    #[test]
    fn test_all_identities_filtering() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let ops = vec![
            GraphOperation::CommitIdentity {
                id: IdentityId(10),
                created_at: Timestamp::from_secs(100),
            },
            GraphOperation::CommitEvent {
                id: EventId(1),
                timestamp: Timestamp::from_secs(101),
                payload: (),
            },
            GraphOperation::CommitState {
                id: StateId(1),
                timestamp: Timestamp::from_secs(102),
                payload: (),
            },
            GraphOperation::CommitIdentity {
                id: IdentityId(20),
                created_at: Timestamp::from_secs(103),
            },
        ];
        assert!(store.apply_batch(ops).is_ok());

        let identities = store.all_identities().map(|mut identities| {
            identities.sort_unstable();
            identities
        });
        assert_eq!(identities, Ok(vec![IdentityId(10), IdentityId(20)]));
    }

    #[test]
    fn batch_failure_rolls_back_all_prior_mutations() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let observation = Observation::new(
            ObservationId(1),
            Modality(1),
            Timestamp::from_secs(100),
            Confidence::new(0.9),
            (),
        );

        let result = store.apply_batch([
            GraphOperation::CommitObservation(observation),
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(1)),
                relation: Relation::Supports,
                target: Vertex::Identity(IdentityId(99)),
                created_at: Timestamp::from_secs(100),
            },
        ]);

        assert_eq!(
            result,
            Err(GraphError::VertexNotFound {
                vertex: Vertex::Identity(IdentityId(99))
            })
        );
        assert_eq!(store.node_count(), 0);
        assert_eq!(store.edge_count(), 0);
        assert!(!store.contains_vertex(Vertex::Observation(ObservationId(1))));
    }

    #[test]
    fn materialization_rolls_back_observation_when_target_is_missing() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let observation = Observation::new(
            ObservationId(8),
            Modality(1),
            Timestamp::from_secs(100),
            Confidence::new(0.9),
            (),
        );

        let result = store.materialize_observation(
            observation,
            IdentityAssignment::Existing(IdentityId(404)),
        );

        assert_eq!(
            result,
            Err(GraphError::VertexNotFound {
                vertex: Vertex::Identity(IdentityId(404))
            })
        );
        assert_eq!(store.node_count(), 0);
    }

    #[test]
    fn materialization_creates_identity_and_unique_support_atomically() {
        let mut store = PetGraphStore::<(), (), ()>::with_capacity(2, 1);
        let observation = Observation::new(
            ObservationId(8),
            Modality(1),
            Timestamp::from_secs(100),
            Confidence::new(0.9),
            (),
        );

        let assigned = store.materialize_observation(
            observation,
            IdentityAssignment::New(IdentityId(12)),
        );

        assert_eq!(assigned, Ok(IdentityId(12)));
        assert_eq!(store.node_count(), 2);
        assert_eq!(store.edge_count(), 1);
        let support_ids =
            store
                .supporting_observations(IdentityId(12))
                .map(|support| {
                    support.map(|observation| observation.id).collect()
                });
        assert_eq!(support_ids, Ok(vec![ObservationId(8)]));
    }

    #[test]
    fn second_support_owner_is_rejected_without_mutation() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let observation = Observation::new(
            ObservationId(1),
            Modality(1),
            Timestamp::from_secs(100),
            Confidence::new(1.0),
            (),
        );
        let setup = [
            GraphOperation::CommitObservation(observation),
            GraphOperation::CommitIdentity {
                id: IdentityId(1),
                created_at: Timestamp::from_secs(100),
            },
            GraphOperation::CommitIdentity {
                id: IdentityId(2),
                created_at: Timestamp::from_secs(100),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(1)),
                relation: Relation::Supports,
                target: Vertex::Identity(IdentityId(1)),
                created_at: Timestamp::from_secs(100),
            },
        ];
        assert!(store.apply_batch(setup).is_ok());

        let result = store.apply_batch([GraphOperation::CommitRelation {
            source: Vertex::Observation(ObservationId(1)),
            relation: Relation::Supports,
            target: Vertex::Identity(IdentityId(2)),
            created_at: Timestamp::from_secs(101),
        }]);

        assert_eq!(
            result,
            Err(GraphError::ObservationAlreadyAssigned {
                observation: ObservationId(1),
                identity: IdentityId(1),
            })
        );
        assert_eq!(store.edge_count(), 1);
    }

    #[test]
    fn duplicate_semantic_relation_is_rejected_without_mutation() {
        let timestamp = Timestamp::from_secs(100);
        let source = Vertex::Identity(IdentityId(1));
        let target = Vertex::Identity(IdentityId(2));
        let mut store = PetGraphStore::<(), (), ()>::new();
        assert!(
            store
                .apply_batch([
                    GraphOperation::CommitIdentity {
                        id: IdentityId(1),
                        created_at: timestamp,
                    },
                    GraphOperation::CommitIdentity {
                        id: IdentityId(2),
                        created_at: timestamp,
                    },
                    GraphOperation::CommitRelation {
                        source,
                        relation: Relation::AssociatedWith,
                        target,
                        created_at: timestamp,
                    },
                ])
                .is_ok()
        );

        let result = store.apply_batch([GraphOperation::CommitRelation {
            source,
            relation: Relation::AssociatedWith,
            target,
            created_at: Timestamp::from_secs(101),
        }]);

        assert_eq!(
            result,
            Err(GraphError::DuplicateRelation {
                origin: source,
                relation: Relation::AssociatedWith,
                target,
            })
        );
        assert_eq!(store.edge_count(), 1);
    }

    #[test]
    fn merge_rewires_support_and_coalesces_existing_relations() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let target = IdentityId(1);
        let duplicate = IdentityId(2);
        let related = IdentityId(3);
        let origin = IdentityId(4);
        let setup = vec![
            GraphOperation::CommitIdentity {
                id: target,
                created_at: Timestamp::from_secs(1),
            },
            GraphOperation::CommitIdentity {
                id: duplicate,
                created_at: Timestamp::from_secs(1),
            },
            GraphOperation::CommitIdentity {
                id: related,
                created_at: Timestamp::from_secs(1),
            },
            GraphOperation::CommitIdentity {
                id: origin,
                created_at: Timestamp::from_secs(1),
            },
            GraphOperation::CommitObservation(Observation::new(
                ObservationId(10),
                Modality(1),
                Timestamp::from_secs(1),
                Confidence::new(1.0),
                (),
            )),
            GraphOperation::CommitObservation(Observation::new(
                ObservationId(20),
                Modality(1),
                Timestamp::from_secs(2),
                Confidence::new(1.0),
                (),
            )),
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(10)),
                relation: Relation::Supports,
                target: Vertex::Identity(target),
                created_at: Timestamp::from_secs(1),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(20)),
                relation: Relation::Supports,
                target: Vertex::Identity(duplicate),
                created_at: Timestamp::from_secs(2),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Identity(target),
                relation: Relation::AssociatedWith,
                target: Vertex::Identity(related),
                created_at: Timestamp::from_secs(2),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Identity(duplicate),
                relation: Relation::AssociatedWith,
                target: Vertex::Identity(related),
                created_at: Timestamp::from_secs(3),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Identity(origin),
                relation: Relation::AssociatedWith,
                target: Vertex::Identity(duplicate),
                created_at: Timestamp::from_secs(3),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Identity(target),
                relation: Relation::Refines,
                target: Vertex::Identity(duplicate),
                created_at: Timestamp::from_secs(3),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Identity(duplicate),
                relation: Relation::AssociatedWith,
                target: Vertex::Identity(target),
                created_at: Timestamp::from_secs(3),
            },
        ];
        assert!(store.apply_batch(setup).is_ok());

        let outcome = store.merge_identities(target, duplicate);

        assert_eq!(
            outcome,
            Ok(MergeOutcome {
                rewired_relations: 2,
                coalesced_relations: 1,
            })
        );
        assert!(!store.contains_vertex(Vertex::Identity(duplicate)));
        let support_ids =
            store.supporting_observations(target).map(|support| {
                let mut ids: Vec<_> =
                    support.map(|observation| observation.id).collect();
                ids.sort_unstable();
                ids
            });
        assert_eq!(
            support_ids,
            Ok(vec![ObservationId(10), ObservationId(20)])
        );
        assert_eq!(store.edge_count(), 4);
        let origin_edges = store.checked_index(Vertex::Identity(origin)).map(
            |origin_index| {
                store
                    .raw_graph
                    .edges_directed(origin_index, Direction::Outgoing)
                    .map(|edge| {
                        (
                            edge.weight().relation,
                            store.raw_graph[edge.target()].vertex(),
                        )
                    })
                    .collect::<Vec<_>>()
            },
        );
        assert_eq!(
            origin_edges,
            Ok(vec![(Relation::AssociatedWith, Vertex::Identity(target),)])
        );
    }

    #[test]
    fn merge_rejects_self_and_missing_identities() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        assert_eq!(
            store.merge_identities(IdentityId(1), IdentityId(1)),
            Err(GraphError::SelfIdentityMerge {
                identity: IdentityId(1)
            })
        );
        assert_eq!(
            store.merge_identities(IdentityId(1), IdentityId(2)),
            Err(GraphError::VertexNotFound {
                vertex: Vertex::Identity(IdentityId(1))
            })
        );
    }

    #[test]
    fn merge_must_be_the_only_operation_in_a_batch() {
        let mut store = PetGraphStore::<(), (), ()>::new();
        let result = store.apply_batch([
            GraphOperation::CommitIdentity {
                id: IdentityId(1),
                created_at: Timestamp::from_secs(1),
            },
            GraphOperation::MergeIdentities {
                target: IdentityId(1),
                duplicate: IdentityId(2),
            },
        ]);

        assert_eq!(result, Err(GraphError::MergeOperationMustBeIsolated));
        assert_eq!(store.node_count(), 0);
    }
}
