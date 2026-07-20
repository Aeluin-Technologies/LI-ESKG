//! Concrete in-memory graph representation utilizing standard tree
//! collections.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use li_core::ids::{EventId, IdentityId, ObservationId, StateId, VertexId};
use li_core::observation::Observation;
use li_core::ontology::Vertex;
use li_core::relation::Relation;

use crate::IdentitySetQuery;
use crate::graph::KnowledgeGraph;
use crate::ontology::{Edge, EventNode, IdentityNode, StateNode};
use crate::operations::GraphOperation;
use crate::projection::{
    EventStateGraph, EventStateProjection, GraphProjection,
};
use crate::queries::{NeighborhoodQuery, SupportSetQuery};

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
/// assert!(graph.vertex_type(VertexId(42)).is_some());
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
}

impl<P: Clone, E: Clone, S: Clone> KnowledgeGraph for MemoryGraph<P, E, S> {
    type EventPayload = E;
    type ObservationPayload = P;
    type StatePayload = S;

    /// Resolves the semantic variant of a vertex if it exists in the ontology.
    fn vertex_type(&self, id: VertexId) -> Option<Vertex> {
        self.ontology.get(&id).cloned()
    }

    /// Mutates the graph state by appending a new node or building a directed
    /// link.
    ///
    /// # Invariants
    /// * **Node Commits:** Explicitly registers the node's unique ID into the
    ///   central ontology registry before inserting the core payload.
    /// * **Relation Commits:** Edges are symmetrically indexed in both
    ///   `out_edges` and `in_edges` to prevent orphan relationships.
    fn apply(&mut self, op: GraphOperation<P, E, S>) {
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

impl<P: Clone, E: Clone, S: Clone> SupportSetQuery for MemoryGraph<P, E, S> {
    /// Traverses incoming edges to fetch all historical observations
    /// validating an identity.
    ///
    /// Filters strictly for `Relation::Supports` references originating from a
    /// `Vertex::Observation` variant.
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

impl<P, E, S> NeighborhoodQuery for MemoryGraph<P, E, S>
where
    P: Clone,
    E: Clone,
    S: Clone,
{
    type EdgeRef<'a>
        = &'a Edge
    where
        Self: 'a;

    /// Retrieves references to all directed edges originating from this
    /// vertex. Returns an empty vector if the vertex has no outbound
    /// activity.
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
    /// Isolates causal pathways by filtering out identity/observation
    /// tracking.
    ///
    /// # Returns
    /// A localized `EventStateGraph` containing exclusively the interactions
    /// between Event nodes and State nodes.
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
    /// Extact all identities contained on the [`MemoryGraph`].
    fn all_identities(&self) -> Vec<IdentityId> {
        self.identities.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use li_core::observation::{Modality, Timestamp};
    use li_core::probability::Confidence;

    use super::*;

    // Helper to generate a dummy confidence score (assuming it implements
    // Default or a simple new fn)
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

        // Ensure the edge is written to both index directions symmetrically
        let out_edges = graph.out_edges(source);
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

        // 1. Commit the Identity Node
        graph.apply(GraphOperation::CommitIdentity(IdentityNode {
            id: identity_id,
            created_at: Timestamp(500),
        }));

        // 2. Commit an Observation Node
        graph.apply(GraphOperation::CommitObservation(Observation {
            id: obs_id,
            modality: Modality(1),
            timestamp: Timestamp(500),
            confidence: dummy_confidence(),
            payload: "sensor_reading",
        }));

        // 3. Link them via a Supports relation
        graph.apply(GraphOperation::CommitRelation {
            source: VertexId(obs_id.0),
            relation: Relation::Supports,
            target: VertexId(identity_id.0),
            created_at: Timestamp(501),
        });

        // 4. Query and verify the verification pathway
        let support = graph.query_support_set(identity_id);
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

        // Populate mixed nodes (Causal vs Empirical)
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

        // Edge A: Event -> State (Should be kept in projection)
        graph.apply(GraphOperation::CommitRelation {
            source: VertexId(event_id.0),
            relation: Relation::Supports, /* Use valid enum variants
                                           * matching your runtime */
            target: VertexId(state_id.0),
            created_at: Timestamp(102),
        });

        // Edge B: Observation -> Event (Should be stripped out)
        graph.apply(GraphOperation::CommitRelation {
            source: VertexId(obs_id.0),
            relation: Relation::Supports,
            target: VertexId(event_id.0),
            created_at: Timestamp(102),
        });

        // Execute Projection
        let projected_graph = EventStateProjection::project(&graph);

        // Assert that only the Causal edge remains
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

        let mut identities = graph.all_identities();
        identities.sort();
        assert_eq!(identities, alloc::vec![IdentityId(1), IdentityId(2)]);
    }
}
