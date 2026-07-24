//! Concrete in-memory graph representation utilizing standard tree
//! collections.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::convert::Infallible;

use li_core::ids::{EventId, IdentityId, ObservationId, StateId, VertexId};
use li_core::observation::Observation;
use li_core::ontology::Vertex;
use li_core::relation::Relation;

use crate::graph::KnowledgeGraph;
use crate::ontology::{Edge, EventNode, IdentityNode, StateNode};
use crate::operations::GraphOperation;
use crate::projection::{
    EventStateGraph, EventStateProjection, GraphProjection,
};
use crate::queries::{IdentitySetQuery, NeighborhoodQuery, SupportSetQuery};

/// In-memory tree-backed implementation of the `KnowledgeGraph` trait.
///
/// # Examples
///
/// ```
/// use li_model::MemoryGraph;
/// use li_model::graph::KnowledgeGraph;
/// use li_model::operations::GraphOperation;
/// use li_core::ids::{IdentityId, VertexId};
/// use li_model::ontology::IdentityNode;
/// use li_core::observation::Timestamp;
///
/// // Create a new empty knowledge graph
/// let mut graph: MemoryGraph<(), (), ()> = MemoryGraph::new();
///
/// // Commit an identity node
/// let id = IdentityId(42);
/// let op = GraphOperation::CommitIdentity(IdentityNode {
///     id,
///     created_at: Timestamp(1710000000),
/// });
/// graph.apply(op);
///
/// // Check that the central ontology registry accurately tracks the vertex type
/// assert!(graph.vertex_type(VertexId(42)).unwrap().is_some());
/// ```
#[derive(Debug, Clone)]
pub struct MemoryGraph<P, E, S> {
    pub ontology: BTreeMap<VertexId, Vertex>,
    pub observations: BTreeMap<ObservationId, Observation<P>>,
    pub identities: BTreeMap<IdentityId, IdentityNode>,
    pub events: BTreeMap<EventId, EventNode<E>>,
    pub states: BTreeMap<StateId, StateNode<S>>,
    pub out_edges: BTreeMap<VertexId, BTreeSet<Edge>>,
    pub in_edges: BTreeMap<VertexId, BTreeSet<Edge>>,
}

impl<P, E, S> Default for MemoryGraph<P, E, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P, E, S> MemoryGraph<P, E, S> {
    pub fn new() -> Self {
        Self {
            ontology: BTreeMap::new(),
            observations: BTreeMap::new(),
            identities: BTreeMap::new(),
            events: BTreeMap::new(),
            states: BTreeMap::new(),
            out_edges: BTreeMap::new(),
            in_edges: BTreeMap::new(),
        }
    }

    /// Internal mutation primitive used to process individual operational
    /// commits.
    fn commit_op(&mut self, op: GraphOperation<P, E, S>)
    where
        P: Clone,
        E: Clone,
        S: Clone,
    {
        match op {
            GraphOperation::CommitObservation(obs) => {
                let vid = VertexId(obs.id.0);
                self.ontology.insert(vid, Vertex::Observation(obs.id));
                self.observations.insert(obs.id, obs);
            },
            GraphOperation::CommitIdentity(identity) => {
                let vid = VertexId(identity.id.0);
                self.ontology.insert(vid, Vertex::Identity(identity.id));
                self.identities.insert(identity.id, identity);
            },
            GraphOperation::CommitEvent(event) => {
                let vid = VertexId(event.id.0);
                self.ontology.insert(vid, Vertex::Event(event.id));
                self.events.insert(event.id, event);
            },
            GraphOperation::CommitState(state) => {
                let vid = VertexId(state.id.0);
                self.ontology.insert(vid, Vertex::State(state.id));
                self.states.insert(state.id, state);
            },
            GraphOperation::CommitRelation {
                source,
                relation,
                target,
                created_at,
            } => {
                let edge = Edge {
                    source,
                    relation,
                    target,
                    created_at,
                };
                self.out_edges
                    .entry(source)
                    .or_default()
                    .insert(edge.clone());
                self.in_edges.entry(target).or_default().insert(edge);
            },
        }
    }
}

impl<P: Clone, E: Clone, S: Clone> KnowledgeGraph for MemoryGraph<P, E, S> {
    type Error = Infallible;
    type EventPayload = E;
    type ObservationPayload = P;
    type StatePayload = S;

    /// Resolves the semantic variant of a vertex if it exists in the ontology.
    fn vertex_type(
        &self,
        id: VertexId,
    ) -> Result<Option<Vertex>, Self::Error> {
        Ok(self.ontology.get(&id).cloned())
    }

    /// Mutates the graph state by applying a batch of operational primitives.
    fn apply_batch(
        &mut self,
        ops: &[GraphOperation<P, E, S>],
    ) -> Result<(), Self::Error> {
        for op in ops {
            self.commit_op(op.clone());
        }
        Ok(())
    }

    /// Traverses incoming edges to fetch all historical observations
    /// validating an identity.
    fn query_support_set(
        &self,
        identity: IdentityId,
    ) -> Result<Vec<Observation<P>>, Self::Error> {
        let vid = VertexId(identity.0);
        let mut support = Vec::new();
        if let Some(edges) = self.in_edges.get(&vid) {
            for edge in edges {
                if edge.relation == Relation::Supports &&
                    let Some(Vertex::Observation(oid)) =
                        self.ontology.get(&edge.source) &&
                    let Some(obs) = self.observations.get(oid)
                {
                    support.push(obs.clone());
                }
            }
        }
        Ok(support)
    }

    /// Retrieves all directed edges originating from a source vertex.
    fn out_edges(&self, source: VertexId) -> Result<Vec<Edge>, Self::Error> {
        Ok(self
            .out_edges
            .get(&source)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default())
    }

    /// Extracts all identity identifiers contained in the graph.
    fn all_identities(&self) -> Result<Vec<IdentityId>, Self::Error> {
        Ok(self.identities.keys().cloned().collect())
    }
}

impl<P: Clone, E: Clone, S: Clone> SupportSetQuery for MemoryGraph<P, E, S> {
    fn query_support_set(&self, identity: IdentityId) -> Vec<&Observation<P>> {
        let vid = VertexId(identity.0);
        let mut support = Vec::new();
        if let Some(edges) = self.in_edges.get(&vid) {
            for edge in edges {
                if edge.relation == Relation::Supports &&
                    let Some(Vertex::Observation(oid)) =
                        self.ontology.get(&edge.source) &&
                    let Some(obs) = self.observations.get(oid)
                {
                    support.push(obs);
                }
            }
        }
        support
    }
}

impl<P: Clone, E: Clone, S: Clone> NeighborhoodQuery for MemoryGraph<P, E, S> {
    type EdgeRef<'a>
        = &'a Edge
    where
        Self: 'a;

    fn out_edges<'a>(&'a self, source: VertexId) -> Vec<Self::EdgeRef<'a>> {
        self.out_edges
            .get(&source)
            .map(|set| set.iter().collect())
            .unwrap_or_default()
    }
}

impl<P: Clone, E: Clone, S: Clone> GraphProjection<MemoryGraph<P, E, S>>
    for EventStateProjection
where
    MemoryGraph<P, E, S>: KnowledgeGraph,
{
    fn project(graph: &MemoryGraph<P, E, S>) -> EventStateGraph {
        let mut edges = BTreeSet::new();
        for (source_vid, target_edges) in &graph.out_edges {
            match graph.ontology.get(source_vid) {
                Some(Vertex::Event(_)) | Some(Vertex::State(_)) => {
                    for edge in target_edges {
                        match graph.ontology.get(&edge.target) {
                            Some(Vertex::Event(_)) |
                            Some(Vertex::State(_)) => {
                                edges.insert(edge.clone());
                            },
                            _ => {},
                        }
                    }
                },
                _ => {},
            }
        }
        EventStateGraph { edges }
    }
}

impl<P: Clone, E: Clone, S: Clone> IdentitySetQuery for MemoryGraph<P, E, S> {
    fn all_identities(&self) -> Vec<IdentityId> {
        self.identities.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use li_core::observation::{Modality, Timestamp};
    use li_core::probability::Confidence;

    use super::*;

    fn dummy_confidence() -> Confidence {
        Confidence(0.78)
    }

    #[test]
    fn test_symmetric_edges_on_relation_commit() {
        let mut graph: MemoryGraph<(), (), ()> = MemoryGraph::new();
        let source = VertexId(1);
        let target = VertexId(2);
        let timestamp = Timestamp(1000);

        let op = GraphOperation::CommitRelation {
            source,
            relation: Relation::Supports,
            target,
            created_at: timestamp,
        };
        graph.apply(op);

        let out_edges = KnowledgeGraph::out_edges(&graph, source).unwrap();
        assert_eq!(out_edges.len(), 1);
        assert_eq!(out_edges[0].target, target);

        assert!(graph.in_edges.contains_key(&target));
        let in_edge =
            graph.in_edges.get(&target).unwrap().iter().next().unwrap();
        assert_eq!(in_edge.source, source);
    }

    #[test]
    fn test_query_support_set() {
        let mut graph: MemoryGraph<&'static str, (), ()> = MemoryGraph::new();
        let identity_id = IdentityId(100);
        let obs_id = ObservationId(1);

        graph.apply(GraphOperation::CommitIdentity(IdentityNode {
            id: identity_id,
            created_at: Timestamp(500),
        }));

        graph.apply(GraphOperation::CommitObservation(Observation {
            id: obs_id,
            modality: Modality(1),
            timestamp: Timestamp(500),
            confidence: dummy_confidence(),
            payload: "sensor_reading",
        }));

        graph.apply(GraphOperation::CommitRelation {
            source: VertexId(obs_id.0),
            relation: Relation::Supports,
            target: VertexId(identity_id.0),
            created_at: Timestamp(501),
        });

        let support =
            KnowledgeGraph::query_support_set(&graph, identity_id).unwrap();
        assert_eq!(support.len(), 1);
        assert_eq!(support[0].payload, "sensor_reading");
    }

    #[test]
    fn test_event_state_projection_filters_correctly() {
        let mut graph: MemoryGraph<(), &'static str, &'static str> =
            MemoryGraph::new();

        let event_id = EventId(10);
        let state_id = StateId(20);
        let obs_id = ObservationId(30);

        graph.apply(GraphOperation::CommitEvent(EventNode {
            id: event_id,
            timestamp: Timestamp(100),
            payload: "state_transition_trigger",
        }));
        graph.apply(GraphOperation::CommitState(StateNode {
            id: state_id,
            timestamp: Timestamp(101),
            payload: "active_state",
        }));
        graph.apply(GraphOperation::CommitObservation(Observation {
            id: obs_id,
            modality: Modality(1),
            timestamp: Timestamp(100),
            confidence: dummy_confidence(),
            payload: (),
        }));

        graph.apply(GraphOperation::CommitRelation {
            source: VertexId(event_id.0),
            relation: Relation::Supports,
            target: VertexId(state_id.0),
            created_at: Timestamp(102),
        });

        graph.apply(GraphOperation::CommitRelation {
            source: VertexId(obs_id.0),
            relation: Relation::Supports,
            target: VertexId(event_id.0),
            created_at: Timestamp(102),
        });

        let projected_graph = EventStateProjection::project(&graph);

        assert_eq!(projected_graph.edges.len(), 1);
        let remaining_edge = projected_graph.edges.iter().next().unwrap();
        assert_eq!(remaining_edge.source, VertexId(event_id.0));
        assert_eq!(remaining_edge.target, VertexId(state_id.0));
    }

    #[test]
    fn test_identity_set_query() {
        let mut graph: MemoryGraph<(), (), ()> = MemoryGraph::new();

        graph.apply(GraphOperation::CommitIdentity(IdentityNode {
            id: IdentityId(1),
            created_at: Timestamp(0),
        }));
        graph.apply(GraphOperation::CommitIdentity(IdentityNode {
            id: IdentityId(2),
            created_at: Timestamp(0),
        }));

        let mut identities = KnowledgeGraph::all_identities(&graph).unwrap();
        identities.sort();
        assert_eq!(identities, alloc::vec![IdentityId(1), IdentityId(2)]);
    }
}
