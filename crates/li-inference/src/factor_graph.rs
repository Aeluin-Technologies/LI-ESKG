//! Bipartite factor graph representation with pre-computed adjacency positions
//! for O(1) message passing.

use li_core::ids::IdentityId;
use li_factors::factor::Factor;

/// Index referencing a variable node within a [`FactorGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VarIndex(pub usize);

/// Index referencing a factor node within a [`FactorGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FactorIndex(pub usize);

/// Variable node representing a candidate identity binary assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariableNode {
    /// Unique index of this variable node.
    pub id: VarIndex,
    /// Candidate identity identifier evaluated by this node.
    pub candidate_identity: IdentityId,
}

/// Pre-computed adjacency connection from a variable node to a connected
/// factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VarEdge {
    /// Connected factor index.
    pub factor_idx: FactorIndex,
    /// Index of this variable in the target factor's scope array.
    pub pos_in_factor: usize,
}

/// Pre-computed adjacency connection from a factor node to a connected
/// variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FactorEdge {
    /// Connected variable index.
    pub var_idx: VarIndex,
    /// Index of this factor in the target variable's connected factors array.
    pub pos_in_var: usize,
}

/// Bipartite graph structuring dynamic assignment variables and connected
/// factor potentials.
pub struct FactorGraph {
    /// List of variable nodes contained in the graph.
    pub variables: Vec<VariableNode>,
    /// List of evaluated factor potentials contained in the graph.
    pub factors: Vec<Box<dyn Factor>>,
    /// Pre-indexed adjacencies from variables to connected factors.
    pub var_adjacencies: Vec<Vec<VarEdge>>,
    /// Pre-indexed adjacencies from factors to connected variables.
    pub factor_adjacencies: Vec<Vec<FactorEdge>>,
}

impl Default for FactorGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl FactorGraph {
    /// Constructs an empty factor graph.
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            factors: Vec::new(),
            var_adjacencies: Vec::new(),
            factor_adjacencies: Vec::new(),
        }
    }

    /// Constructs an empty factor graph with pre-allocated memory capacity.
    ///
    /// # Arguments
    ///
    /// * `var_capacity` - Estimated number of variable nodes.
    /// * `factor_capacity` - Estimated number of factor nodes.
    pub fn with_capacity(var_capacity: usize, factor_capacity: usize) -> Self {
        Self {
            variables: Vec::with_capacity(var_capacity),
            factors: Vec::with_capacity(factor_capacity),
            var_adjacencies: Vec::with_capacity(var_capacity),
            factor_adjacencies: Vec::with_capacity(factor_capacity),
        }
    }

    /// Adds a candidate identity variable to the graph.
    ///
    /// # Arguments
    ///
    /// * `candidate_identity` - The target candidate identity.
    ///
    /// # Returns
    ///
    /// Assigned [`VarIndex`] for the variable.
    pub fn add_variable(
        &mut self,
        candidate_identity: IdentityId,
    ) -> VarIndex {
        let idx = VarIndex(self.variables.len());
        self.variables.push(VariableNode {
            id: idx,
            candidate_identity,
        });
        self.var_adjacencies.push(Vec::with_capacity(4));
        idx
    }

    /// Connects a factor potential to a variable scope with pre-computed
    /// adjacency offsets.
    ///
    /// # Arguments
    ///
    /// * `factor` - Boxed factor potential implementation.
    /// * `scopes` - Slice of variable indices governed by this factor.
    ///
    /// # Returns
    ///
    /// Assigned [`FactorIndex`] for the factor node.
    pub fn add_factor(
        &mut self,
        factor: Box<dyn Factor>,
        scopes: &[VarIndex],
    ) -> FactorIndex {
        let f_idx = FactorIndex(self.factors.len());
        self.factors.push(factor);

        let mut factor_edges = Vec::with_capacity(scopes.len());

        for (pos_in_factor, &v_idx) in scopes.iter().enumerate() {
            let pos_in_var = self.var_adjacencies[v_idx.0].len();

            self.var_adjacencies[v_idx.0].push(VarEdge {
                factor_idx: f_idx,
                pos_in_factor,
            });

            factor_edges.push(FactorEdge {
                var_idx: v_idx,
                pos_in_var,
            });
        }

        self.factor_adjacencies.push(factor_edges);
        f_idx
    }
}

#[cfg(test)]
mod tests {
    use li_core::probability::Probability;
    use li_factors::factor::FactorScope;

    use super::*;

    #[derive(Debug)]
    struct DummyFactor {
        scope: Vec<IdentityId>,
    }

    impl FactorScope for DummyFactor {
        fn scope(&self) -> &[IdentityId] {
            &self.scope
        }
    }

    impl Factor for DummyFactor {
        fn evaluate(&self, _assignments: &[IdentityId]) -> Probability {
            Probability::new(1.0)
        }
    }

    #[test]
    fn test_precomputed_adjacency_offsets() {
        let mut fg = FactorGraph::new();
        let v0 = fg.add_variable(IdentityId(10));
        let v1 = fg.add_variable(IdentityId(20));

        let f0 = fg.add_factor(
            Box::new(DummyFactor {
                scope: vec![IdentityId(10), IdentityId(20)],
            }),
            &[v0, v1],
        );

        assert_eq!(fg.var_adjacencies[v0.0][0].factor_idx, f0);
        assert_eq!(fg.var_adjacencies[v0.0][0].pos_in_factor, 0);
        assert_eq!(fg.factor_adjacencies[f0.0][0].var_idx, v0);
        assert_eq!(fg.factor_adjacencies[f0.0][0].pos_in_var, 0);

        assert_eq!(fg.var_adjacencies[v1.0][0].factor_idx, f0);
        assert_eq!(fg.var_adjacencies[v1.0][0].pos_in_factor, 1);
        assert_eq!(fg.factor_adjacencies[f0.0][1].var_idx, v1);
        assert_eq!(fg.factor_adjacencies[f0.0][1].pos_in_var, 0);
    }

    #[test]
    fn test_empty_graph_creation() {
        let fg = FactorGraph::with_capacity(10, 10);
        assert!(fg.variables.is_empty());
        assert!(fg.factors.is_empty());
    }
}
