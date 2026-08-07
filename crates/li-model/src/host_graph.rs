//! Authoritative host-only ESKG profile and petgraph reference adapter.

use std::collections::BTreeSet;
use std::sync::Arc;

use hashbrown::HashMap;
use li_core::{
    CommitVersion, DecisionId, HostNodeId, HostRelation, HostRelationRole,
    MaterializationId,
};
use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error returned by a malformed deployment host-schema profile.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HostSchemaError {
    /// A required native predicate identifier was empty.
    #[error("host predicate for {0:?} must not be empty")]
    EmptyPredicate(HostRelationRole),
    /// Two roles mapped to the same native predicate and lost role
    /// distinction.
    #[error("host predicate identifiers must be unique across the six roles")]
    DuplicatePredicate,
}

/// Deployment mapping from six motivating roles to native host predicates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostSchemaProfile {
    predicates: [Arc<str>; 6],
}

impl HostSchemaProfile {
    /// Creates a complete, unambiguous native predicate mapping.
    ///
    /// The array order is [`HostRelationRole::ALL`]. No `eskg:` namespace is
    /// invented by the library.
    ///
    /// # Errors
    ///
    /// Returns [`HostSchemaError`] for empty or duplicate predicates.
    pub fn new(predicates: [Arc<str>; 6]) -> Result<Self, HostSchemaError> {
        let mut unique = BTreeSet::new();
        for (role, predicate) in
            HostRelationRole::ALL.into_iter().zip(&predicates)
        {
            if predicate.is_empty() {
                return Err(HostSchemaError::EmptyPredicate(role));
            }
            if !unique.insert(predicate.as_ref()) {
                return Err(HostSchemaError::DuplicatePredicate);
            }
        }
        Ok(Self { predicates })
    }

    /// Returns the native predicate declared for `role`.
    pub fn predicate(&self, role: HostRelationRole) -> &str {
        &self.predicates[usize::from(role.code())]
    }
}

/// Traceable authoritative edge materialized from an accepted decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostEdge {
    /// Native host role.
    pub role: HostRelationRole,
    /// Deployment-specific predicate identifier.
    pub predicate: Arc<str>,
    /// Responsible resolution decision.
    pub decision: DecisionId,
    /// Ledger commit that enqueued the materialization.
    pub commit: CommitVersion,
    /// Idempotent materialization identifier.
    pub materialization: MaterializationId,
}

/// Result of an idempotent authoritative host write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterializationOutcome {
    /// A new host relation was written.
    Applied(EdgeIndex<u32>),
    /// The identical materialization was already present.
    Replayed(EdgeIndex<u32>),
}

/// Error returned while applying an authoritative host write.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HostGraphError {
    /// Node identifier already exists in the host graph.
    #[error("host node {0:?} already exists")]
    DuplicateNode(HostNodeId),
    /// Relation endpoint is absent from the coherent host snapshot.
    #[error("host node {0:?} does not exist")]
    MissingNode(HostNodeId),
    /// An idempotency identifier was reused for different materialization
    /// data.
    #[error("materialization idempotency collision")]
    IdempotencyCollision,
}

/// Petgraph-backed authoritative host graph containing no transient LI
/// objects.
#[derive(Debug)]
pub struct AuthoritativeHostGraph {
    profile: HostSchemaProfile,
    graph: StableDiGraph<HostNodeId, HostEdge, u32>,
    indexes: HashMap<HostNodeId, NodeIndex<u32>>,
    materializations: HashMap<MaterializationId, EdgeIndex<u32>>,
}

impl AuthoritativeHostGraph {
    /// Creates an empty host graph with reserved topology capacity.
    pub fn with_capacity(
        profile: HostSchemaProfile,
        nodes: usize,
        edges: usize,
    ) -> Self {
        Self {
            profile,
            graph: StableDiGraph::with_capacity(nodes, edges),
            indexes: HashMap::with_capacity(nodes),
            materializations: HashMap::with_capacity(edges),
        }
    }

    /// Adds one authoritative host node.
    ///
    /// # Errors
    ///
    /// Returns [`HostGraphError::DuplicateNode`] when the typed identifier is
    /// already present.
    pub fn add_node(
        &mut self,
        node: HostNodeId,
    ) -> Result<NodeIndex<u32>, HostGraphError> {
        if self.indexes.contains_key(&node) {
            return Err(HostGraphError::DuplicateNode(node));
        }
        let index = self.graph.add_node(node);
        self.indexes.insert(node, index);
        Ok(index)
    }

    /// Applies one accepted native host relation idempotently.
    ///
    /// Only this method mutates authoritative edges; candidate generation,
    /// inference, decision, and ledger commit remain host-isolated.
    ///
    /// # Errors
    ///
    /// Returns [`HostGraphError`] for missing endpoints or idempotency
    /// mismatch.
    pub fn materialize(
        &mut self,
        relation: HostRelation,
        decision: DecisionId,
        commit: CommitVersion,
        materialization: MaterializationId,
    ) -> Result<MaterializationOutcome, HostGraphError> {
        if let Some(index) =
            self.materializations.get(&materialization).copied()
        {
            let same = self.graph.edge_weight(index).is_some_and(|edge| {
                edge.role == relation.role() &&
                    edge.decision == decision &&
                    edge.commit == commit &&
                    self.graph.edge_endpoints(index).is_some_and(
                        |(source, target)| {
                            let endpoints = relation.endpoints();
                            self.graph.node_weight(source) ==
                                Some(&endpoints.0) &&
                                self.graph.node_weight(target) ==
                                    Some(&endpoints.1)
                        },
                    )
            });
            if same {
                return Ok(MaterializationOutcome::Replayed(index));
            }
            return Err(HostGraphError::IdempotencyCollision);
        }

        let (source, target) = relation.endpoints();
        let source_index = self
            .indexes
            .get(&source)
            .copied()
            .ok_or(HostGraphError::MissingNode(source))?;
        let target_index = self
            .indexes
            .get(&target)
            .copied()
            .ok_or(HostGraphError::MissingNode(target))?;
        let edge = HostEdge {
            role: relation.role(),
            predicate: Arc::from(self.profile.predicate(relation.role())),
            decision,
            commit,
            materialization,
        };
        let index = self.graph.add_edge(source_index, target_index, edge);
        self.materializations.insert(materialization, index);
        Ok(MaterializationOutcome::Applied(index))
    }

    /// Returns the number of authoritative nodes.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Returns the number of successfully materialized host relations.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Borrows the underlying host-only petgraph for zero-copy traversal.
    pub const fn graph(&self) -> &StableDiGraph<HostNodeId, HostEdge, u32> {
        &self.graph
    }
}

#[cfg(test)]
mod tests {
    use li_core::{EventId, PhysicalNodeId, StateId};

    use super::*;

    fn profile() -> Result<HostSchemaProfile, HostSchemaError> {
        HostSchemaProfile::new([
            Arc::from("native:triggers"),
            Arc::from("native:leadsTo"),
            Arc::from("native:evolution"),
            Arc::from("native:contain"),
            Arc::from("native:occur"),
            Arc::from("native:influence"),
        ])
    }

    #[test]
    fn profile_requires_six_distinct_native_predicates() {
        let duplicate = HostSchemaProfile::new([
            Arc::from("x"),
            Arc::from("x"),
            Arc::from("2"),
            Arc::from("3"),
            Arc::from("4"),
            Arc::from("5"),
        ]);
        assert_eq!(duplicate, Err(HostSchemaError::DuplicatePredicate));
    }

    #[test]
    fn host_graph_contains_no_observation_or_identity_partition()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut graph =
            AuthoritativeHostGraph::with_capacity(profile()?, 3, 1);
        graph.add_node(HostNodeId::Event(EventId(1)))?;
        graph.add_node(HostNodeId::State(StateId(2)))?;
        graph.add_node(HostNodeId::Physical(PhysicalNodeId(3)))?;
        assert_eq!(graph.node_count(), 3);
        Ok(())
    }

    #[test]
    fn materialization_is_traceable_and_idempotent()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut graph =
            AuthoritativeHostGraph::with_capacity(profile()?, 2, 1);
        graph.add_node(HostNodeId::Event(EventId(1)))?;
        graph.add_node(HostNodeId::Physical(PhysicalNodeId(2)))?;
        let relation = HostRelation::Occur {
            event: EventId(1),
            physical: PhysicalNodeId(2),
        };
        let first = graph.materialize(
            relation,
            DecisionId(7),
            CommitVersion::new(3),
            MaterializationId(9),
        )?;
        let replay = graph.materialize(
            relation,
            DecisionId(7),
            CommitVersion::new(3),
            MaterializationId(9),
        )?;
        assert!(matches!(first, MaterializationOutcome::Applied(_)));
        assert!(matches!(replay, MaterializationOutcome::Replayed(_)));
        assert_eq!(graph.edge_count(), 1);
        Ok(())
    }

    #[test]
    fn missing_endpoint_leaves_host_graph_unchanged()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut graph =
            AuthoritativeHostGraph::with_capacity(profile()?, 1, 1);
        graph.add_node(HostNodeId::Event(EventId(1)))?;
        let result = graph.materialize(
            HostRelation::Occur {
                event: EventId(1),
                physical: PhysicalNodeId(2),
            },
            DecisionId(7),
            CommitVersion::new(3),
            MaterializationId(9),
        );
        assert_eq!(
            result,
            Err(HostGraphError::MissingNode(HostNodeId::Physical(
                PhysicalNodeId(2)
            )))
        );
        assert_eq!(graph.edge_count(), 0);
        Ok(())
    }
}
