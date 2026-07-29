//! Causal projection homomorphisms isolating the deterministic
//! $R_{\text{eskg}}$ subgraph.

use li_core::ids::{EventId, StateId};
use li_core::observation::Timestamp;
use li_core::ontology::Vertex;
use petgraph::Directed;
use petgraph::stable_graph::StableGraph;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, NodeIndexable};

use crate::graph::PetGraphStore;
use crate::ontology::{EdgeData, NodeData};

/// A node in the deterministic Event-State projection.
#[derive(Debug, Clone, PartialEq)]
pub enum EventStateNode<E, S> {
    /// Causal event occurrence.
    Event {
        /// Strongly typed event identifier.
        id: EventId,
        /// Time at which the event occurred.
        timestamp: Timestamp,
        /// Domain-specific event payload.
        payload: E,
    },
    /// Entity state snapshot.
    State {
        /// Strongly typed state identifier.
        id: StateId,
        /// Time represented by the state.
        timestamp: Timestamp,
        /// Domain-specific state payload.
        payload: S,
    },
}

impl<E, S> EventStateNode<E, S> {
    /// Returns the strongly typed ontology vertex represented by this node.
    #[inline]
    pub fn vertex(&self) -> Vertex {
        match self {
            Self::Event { id, .. } => Vertex::Event(*id),
            Self::State { id, .. } => Vertex::State(*id),
        }
    }
}

/// Structurally isolated causal graph containing zero identity uncertainty.
#[derive(Debug, Clone)]
pub struct EventStateGraph<E, S> {
    graph: StableGraph<EventStateNode<E, S>, EdgeData, Directed, u32>,
}

impl<E, S> EventStateGraph<E, S> {
    /// Returns a read-only view of the projected graph.
    #[inline]
    pub fn graph(
        &self,
    ) -> &StableGraph<EventStateNode<E, S>, EdgeData, Directed, u32> {
        &self.graph
    }
}

/// Interface defining the projection homomorphism $\pi$.
pub trait GraphProjection<P, E, S> {
    /// Materializes the Event-State causal subspace from the global graph.
    fn project(store: &PetGraphStore<P, E, S>) -> EventStateGraph<E, S>
    where
        E: Clone,
        S: Clone;
}

/// Homomorphism mapping implementation enforcing projection preservation.
pub struct EventStateProjection;

impl<P, E, S> GraphProjection<P, E, S> for EventStateProjection {
    fn project(store: &PetGraphStore<P, E, S>) -> EventStateGraph<E, S>
    where
        E: Clone,
        S: Clone,
    {
        let node_count = store.raw_graph.node_count();
        let edge_count = store.raw_graph.edge_count();
        let mut projected = StableGraph::with_capacity(node_count, edge_count);
        let mut node_mapping = vec![None; store.raw_graph.node_bound()];

        for source_index in store.raw_graph.node_indices() {
            let projected_index = match &store.raw_graph[source_index] {
                NodeData::Event {
                    id,
                    timestamp,
                    payload,
                } => Some(projected.add_node(EventStateNode::Event {
                    id: *id,
                    timestamp: *timestamp,
                    payload: payload.clone(),
                })),
                NodeData::State {
                    id,
                    timestamp,
                    payload,
                } => Some(projected.add_node(EventStateNode::State {
                    id: *id,
                    timestamp: *timestamp,
                    payload: payload.clone(),
                })),
                NodeData::Observation(_) | NodeData::Identity { .. } => None,
            };
            node_mapping[source_index.index()] = projected_index;
        }

        for edge in store.raw_graph.edge_references() {
            if !edge.weight().relation.is_eskg_relation() {
                continue;
            }

            let source = node_mapping[edge.source().index()];
            let target = node_mapping[edge.target().index()];
            if let (Some(source), Some(target)) = (source, target) {
                projected.add_edge(source, target, *edge.weight());
            }
        }

        EventStateGraph { graph: projected }
    }
}

#[cfg(test)]
mod tests {
    use li_core::ids::{IdentityId, ObservationId};
    use li_core::observation::{Modality, Observation};
    use li_core::probability::Confidence;
    use li_core::relation::Relation;

    use super::*;
    use crate::graph::KnowledgeGraph;
    use crate::operations::GraphOperation;

    #[test]
    fn projection_preserves_exact_event_state_topology() {
        let timestamp = Timestamp::from_secs(10);
        let mut store = PetGraphStore::<(), u32, u32>::new();
        let operations = vec![
            GraphOperation::CommitEvent {
                id: EventId(1),
                timestamp,
                payload: 11,
            },
            GraphOperation::CommitEvent {
                id: EventId(2),
                timestamp,
                payload: 12,
            },
            GraphOperation::CommitState {
                id: StateId(1),
                timestamp,
                payload: 21,
            },
            GraphOperation::CommitState {
                id: StateId(2),
                timestamp,
                payload: 22,
            },
            GraphOperation::CommitIdentity {
                id: IdentityId(1),
                created_at: timestamp,
            },
            GraphOperation::CommitObservation(Observation::new(
                ObservationId(1),
                Modality(1),
                timestamp,
                Confidence::new(1.0),
                (),
            )),
            GraphOperation::CommitRelation {
                source: Vertex::Event(EventId(1)),
                relation: Relation::Influence,
                target: Vertex::Event(EventId(2)),
                created_at: timestamp,
            },
            GraphOperation::CommitRelation {
                source: Vertex::Event(EventId(1)),
                relation: Relation::Trigger,
                target: Vertex::State(StateId(1)),
                created_at: timestamp,
            },
            GraphOperation::CommitRelation {
                source: Vertex::State(StateId(1)),
                relation: Relation::Lead,
                target: Vertex::State(StateId(2)),
                created_at: timestamp,
            },
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(1)),
                relation: Relation::Supports,
                target: Vertex::Identity(IdentityId(1)),
                created_at: timestamp,
            },
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(1)),
                relation: Relation::ObservedDuring,
                target: Vertex::State(StateId(1)),
                created_at: timestamp,
            },
        ];
        assert!(store.apply_batch(operations).is_ok());

        let projection = EventStateProjection::project(&store);
        let graph = projection.graph();
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.edge_count(), 3);

        for expected in [
            Vertex::Event(EventId(1)),
            Vertex::Event(EventId(2)),
            Vertex::State(StateId(1)),
            Vertex::State(StateId(2)),
        ] {
            assert!(
                graph.node_weights().any(|node| node.vertex() == expected)
            );
        }

        let expected_edges = [
            (
                Vertex::Event(EventId(1)),
                Relation::Influence,
                Vertex::Event(EventId(2)),
            ),
            (
                Vertex::Event(EventId(1)),
                Relation::Trigger,
                Vertex::State(StateId(1)),
            ),
            (
                Vertex::State(StateId(1)),
                Relation::Lead,
                Vertex::State(StateId(2)),
            ),
        ];
        for (source, relation, target) in expected_edges {
            let matching_edges = graph
                .edge_references()
                .filter(|edge| {
                    graph[edge.source()].vertex() == source &&
                        edge.weight().relation == relation &&
                        graph[edge.target()].vertex() == target
                })
                .count();
            assert_eq!(matching_edges, 1);
        }
    }

    #[test]
    fn event_state_node_reports_typed_vertex() {
        let node = EventStateNode::<(), ()>::State {
            id: StateId(7),
            timestamp: Timestamp::UNIX_EPOCH,
            payload: (),
        };

        assert_eq!(node.vertex(), Vertex::State(StateId(7)));
    }
}
