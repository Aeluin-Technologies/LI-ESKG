//! Causal projection homomorphisms isolating the deterministic
//! $R_{\text{eskg}}$ subgraph.

use std::collections::HashMap;

use petgraph::Directed;
use petgraph::stable_graph::StableGraph;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};

use crate::graph::PetGraphStore;
use crate::ontology::{EdgeData, NodeData};

/// Structurally isolated causal graph containing zero identity uncertainty.
#[derive(Debug, Clone)]
pub struct EventStateGraph<E, S> {
    /// Internal petgraph storing exclusively Event ($E$) and State ($S$)
    /// nodes with $R_{\text{eskg}}$ relations.
    pub graph: StableGraph<NodeData<(), E, S>, EdgeData, Directed, u32>,
}

/// Interface defining the projection homomorphism $\pi$.
pub trait GraphProjection<P, E, S> {
    /// Projects the global knowledge graph into the original Event-State
    /// causal subspace $R_{\text{eskg}}$.
    fn project(store: &PetGraphStore<P, E, S>) -> EventStateGraph<E, S>
    where
        E: Clone,
        S: Clone;
}

/// Homomorphism mapping implementation enforcing Theorem 2 (Projection
/// Preservation).
pub struct EventStateProjection;

impl<P, E, S> GraphProjection<P, E, S> for EventStateProjection {
    fn project(store: &PetGraphStore<P, E, S>) -> EventStateGraph<E, S>
    where
        E: Clone,
        S: Clone,
    {
        let node_count = store.raw_graph.node_count();
        let edge_count = store.raw_graph.edge_count();

        let mut proj_graph =
            StableGraph::with_capacity(node_count, edge_count);
        let mut node_mapping = HashMap::with_capacity(node_count);

        for idx in store.raw_graph.node_indices() {
            let node_data = &store.raw_graph[idx];
            match node_data {
                NodeData::Event {
                    id,
                    timestamp,
                    payload,
                } => {
                    let new_idx = proj_graph.add_node(NodeData::Event {
                        id: *id,
                        timestamp: *timestamp,
                        payload: payload.clone(),
                    });
                    node_mapping.insert(idx, new_idx);
                },
                NodeData::State {
                    id,
                    timestamp,
                    payload,
                } => {
                    let new_idx = proj_graph.add_node(NodeData::State {
                        id: *id,
                        timestamp: *timestamp,
                        payload: payload.clone(),
                    });
                    node_mapping.insert(idx, new_idx);
                },
                _ => {},
            }
        }

        for edge in store.raw_graph.edge_references() {
            if edge.weight().relation.is_eskg_relation() &&
                let (Some(&src_proj), Some(&tgt_proj)) = (
                    node_mapping.get(&edge.source()),
                    node_mapping.get(&edge.target()),
                )
            {
                proj_graph.add_edge(src_proj, tgt_proj, edge.weight().clone());
            }
        }

        EventStateGraph { graph: proj_graph }
    }
}
