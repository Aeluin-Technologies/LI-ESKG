//! Mathematical graph interface decoupling topology from storage drivers.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use li_core::ids::IdentityId;
use li_core::observation::Observation;
use li_core::ontology::Vertex;
use li_core::relation::Relation;
use petgraph::Directed;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableGraph;
use petgraph::visit::EdgeRef;
use thiserror::Error;

use crate::ontology::{EdgeData, NodeData};
use crate::operations::GraphOperation;

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
}

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
    ) -> Result<Vec<Observation<Self::ObservationPayload>>, Self::Error>;

    /// Enumerates all active identity identifiers in I.
    fn all_identities(&self) -> Result<Vec<IdentityId>, Self::Error>;
}

/// Memory graph backed natively by `petgraph`.
#[derive(Debug, Clone)]
pub struct PetGraphStore<P, E, S> {
    /// Internal petgraph storage mapping node data and directed edge
    /// metadata.
    pub raw_graph: StableGraph<NodeData<P, E, S>, EdgeData, Directed, u32>,
    /// Hash Index map resolving typed domain vertices to internal
    /// `NodeIndex<u32>`.
    pub index_map: HashMap<Vertex, NodeIndex<u32>>,
}

impl<P, E, S> Default for PetGraphStore<P, E, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P, E, S> PetGraphStore<P, E, S> {
    /// Instantiates an empty `PetGraphStore` instance.
    pub fn new() -> Self {
        Self {
            raw_graph: StableGraph::default(),
            index_map: HashMap::new(),
        }
    }

    /// Processes an individual graph operation commit by consuming it,
    /// enforcing schema validity and uniqueness without heap allocations.
    fn commit_op(
        &mut self,
        op: GraphOperation<P, E, S>,
    ) -> Result<(), GraphError> {
        match op {
            GraphOperation::CommitObservation(obs) => {
                let vertex = Vertex::Observation(obs.id);
                match self.index_map.entry(vertex) {
                    Entry::Occupied(_) => {
                        return Err(GraphError::DuplicateVertex { vertex });
                    },
                    Entry::Vacant(entry) => {
                        let idx = self
                            .raw_graph
                            .add_node(NodeData::Observation(obs));
                        entry.insert(idx);
                    },
                }
            },
            GraphOperation::CommitIdentity { id, created_at } => {
                let vertex = Vertex::Identity(id);
                match self.index_map.entry(vertex) {
                    Entry::Occupied(_) => {
                        return Err(GraphError::DuplicateVertex { vertex });
                    },
                    Entry::Vacant(entry) => {
                        let idx = self
                            .raw_graph
                            .add_node(NodeData::Identity { id, created_at });
                        entry.insert(idx);
                    },
                }
            },
            GraphOperation::CommitEvent {
                id,
                timestamp,
                payload,
            } => {
                let vertex = Vertex::Event(id);
                match self.index_map.entry(vertex) {
                    Entry::Occupied(_) => {
                        return Err(GraphError::DuplicateVertex { vertex });
                    },
                    Entry::Vacant(entry) => {
                        let idx = self.raw_graph.add_node(NodeData::Event {
                            id,
                            timestamp,
                            payload,
                        });
                        entry.insert(idx);
                    },
                }
            },
            GraphOperation::CommitState {
                id,
                timestamp,
                payload,
            } => {
                let vertex = Vertex::State(id);
                match self.index_map.entry(vertex) {
                    Entry::Occupied(_) => {
                        return Err(GraphError::DuplicateVertex { vertex });
                    },
                    Entry::Vacant(entry) => {
                        let idx = self.raw_graph.add_node(NodeData::State {
                            id,
                            timestamp,
                            payload,
                        });
                        entry.insert(idx);
                    },
                }
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

                let src_idx = *self
                    .index_map
                    .get(&source)
                    .ok_or(GraphError::VertexNotFound { vertex: source })?;
                let tgt_idx = *self
                    .index_map
                    .get(&target)
                    .ok_or(GraphError::VertexNotFound { vertex: target })?;

                self.raw_graph.add_edge(
                    src_idx,
                    tgt_idx,
                    EdgeData {
                        relation,
                        created_at,
                    },
                );
            },
        }
        Ok(())
    }
}

impl<P: Clone, E, S> KnowledgeGraph for PetGraphStore<P, E, S> {
    type Error = GraphError;
    type EventPayload = E;
    type ObservationPayload = P;
    type StatePayload = S;

    fn vertex_type(
        &self,
        vertex: Vertex,
    ) -> Result<Option<Vertex>, Self::Error> {
        Ok(self.index_map.get(&vertex).map(|_| vertex))
    }

    fn apply_batch(
        &mut self,
        ops: impl IntoIterator<Item = GraphOperation<P, E, S>>,
    ) -> Result<(), Self::Error> {
        for op in ops {
            self.commit_op(op)?;
        }
        Ok(())
    }

    fn query_support_set(
        &self,
        identity: IdentityId,
    ) -> Result<Vec<Observation<P>>, Self::Error> {
        let target_vertex = Vertex::Identity(identity);
        let target_idx = *self.index_map.get(&target_vertex).ok_or(
            GraphError::VertexNotFound {
                vertex: target_vertex,
            },
        )?;

        let mut support = Vec::new();
        for edge in self
            .raw_graph
            .edges_directed(target_idx, petgraph::Direction::Incoming)
        {
            if edge.weight().relation == Relation::Supports &&
                let NodeData::Observation(obs) =
                    &self.raw_graph[edge.source()]
            {
                support.push(obs.clone());
            }
        }
        Ok(support)
    }

    fn all_identities(&self) -> Result<Vec<IdentityId>, Self::Error> {
        let identities = self
            .index_map
            .keys()
            .filter_map(|v| match v {
                Vertex::Identity(id) => Some(*id),
                _ => None,
            })
            .collect();
        Ok(identities)
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

        let support = store.query_support_set(IdentityId(1)).unwrap();
        assert!(support.is_empty());
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

        let mut identities = store.all_identities().unwrap();
        identities.sort();
        assert_eq!(identities, vec![IdentityId(10), IdentityId(20)]);
    }
}
