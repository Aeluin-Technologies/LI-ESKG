//! Bipartite factor graph representation linking assignment variables and
//! potential factors.

use alloc::boxed::Box;
use alloc::vec::Vec;

use li_core::ids::IdentityId;
use li_factors::factor::Factor;

/// Strongly-typed index referencing a variable node within a [`FactorGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VarIndex(pub usize);

/// Strongly-typed index referencing a factor node within a [`FactorGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FactorIndex(pub usize);

/// Variable node representing a candidate identity assignment hypothesis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariableNode {
    /// Unique index of this variable in the host factor graph.
    pub id: VarIndex,
    /// Identity candidate evaluating active or inactive state.
    pub candidate_identity: IdentityId,
}

/// Bipartite graph structuring dynamic assignment variables and connected
/// factor potentials.
pub struct FactorGraph {
    /// List of variable nodes in the graph.
    pub variables: Vec<VariableNode>,
    /// List of evaluated factor potentials in the graph.
    pub factors: Vec<Box<dyn Factor>>,
    /// Mapping from variable node index to connected factor node indices.
    pub var_to_factor: Vec<Vec<FactorIndex>>,
    /// Mapping from factor node index to scope variable node indices.
    pub factor_to_var: Vec<Vec<VarIndex>>,
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
            var_to_factor: Vec::new(),
            factor_to_var: Vec::new(),
        }
    }

    /// Adds a new assignment variable candidate to the factor graph.
    ///
    /// # Arguments
    ///
    /// * `candidate_identity` - The identity identifier associated with this
    ///   variable node.
    ///
    /// # Returns
    ///
    /// The assigned [`VarIndex`] for the newly created variable node.
    pub fn add_variable(
        &mut self,
        candidate_identity: IdentityId,
    ) -> VarIndex {
        let idx = VarIndex(self.variables.len());
        self.variables.push(VariableNode {
            id: idx,
            candidate_identity,
        });
        self.var_to_factor.push(Vec::new());
        idx
    }

    /// Adds a potential factor node connected to a scope of variable nodes.
    ///
    /// # Arguments
    ///
    /// * `factor` - Boxed implementation of the factor potential.
    /// * `scopes` - Slice of variable indices governed by this factor.
    ///
    /// # Returns
    ///
    /// The assigned [`FactorIndex`] for the newly added factor.
    pub fn add_factor(
        &mut self,
        factor: Box<dyn Factor>,
        scopes: &[VarIndex],
    ) -> FactorIndex {
        let f_idx = FactorIndex(self.factors.len());
        self.factors.push(factor);

        let mut var_indices = Vec::with_capacity(scopes.len());
        for &v_idx in scopes {
            var_indices.push(v_idx);
            self.var_to_factor[v_idx.0].push(f_idx);
        }
        self.factor_to_var.push(var_indices);
        f_idx
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use alloc::boxed::Box;
    use alloc::vec::Vec;

    use li_core::ids::IdentityId;
    use li_core::probability::Probability;
    use li_factors::factor::{Factor, FactorScope};

    use crate::factor_graph::FactorGraph;

    pub(crate) struct ConstantFactor {
        pub scope_ids: Vec<IdentityId>,
        pub value: f64,
    }

    impl FactorScope for ConstantFactor {
        fn scope(&self) -> &[IdentityId] {
            &self.scope_ids
        }
    }

    impl Factor for ConstantFactor {
        fn evaluate(&self, _assignment: &[IdentityId]) -> Probability {
            Probability::new(self.value)
        }
    }

    #[test]
    fn test_factor_graph_empty() {
        let graph = FactorGraph::new();
        assert_eq!(graph.variables.len(), 0);
        assert_eq!(graph.factors.len(), 0);
        assert_eq!(graph.var_to_factor.len(), 0);
        assert_eq!(graph.factor_to_var.len(), 0);
    }

    #[test]
    fn test_factor_graph_add_variables_and_factors() {
        let mut graph = FactorGraph::new();
        let v0 = graph.add_variable(IdentityId(1));
        let v1 = graph.add_variable(IdentityId(2));

        let factor = Box::new(ConstantFactor {
            scope_ids: alloc::vec![IdentityId(1), IdentityId(2)],
            value: 1.0,
        });

        let f0 = graph.add_factor(factor, &[v0, v1]);

        assert_eq!(v0.0, 0);
        assert_eq!(v1.0, 1);
        assert_eq!(f0.0, 0);
        assert_eq!(graph.var_to_factor[0], alloc::vec![f0]);
        assert_eq!(graph.var_to_factor[1], alloc::vec![f0]);
        assert_eq!(graph.factor_to_var[0], alloc::vec![v0, v1]);
    }

    #[test]
    fn test_factor_graph_disconnected_variable() {
        let mut graph = FactorGraph::new();
        let v0 = graph.add_variable(IdentityId(10));

        assert_eq!(graph.var_to_factor[v0.0].len(), 0);
    }
}
