//! Compact factor graph for one observation's categorical identity assignment.

use alloc::boxed::Box;
use alloc::vec::Vec;

use li_core::ids::IdentityId;
use li_factors::factor::Factor;

/// Strongly typed index referencing a candidate value in a [`FactorGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VarIndex(pub usize);

/// Strongly typed index referencing a factor in a [`FactorGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FactorIndex(pub usize);

/// Candidate value in the incoming observation's assignment domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariableNode {
    /// Unique index of this candidate in the local factor graph.
    pub id: VarIndex,
    /// Existing identity represented by this assignment value.
    pub candidate_identity: IdentityId,
}

/// Compact local factor graph for the assignment variable
/// `Z_t in I_t union {new}`.
///
/// The graph stores all factor scopes in one contiguous buffer. This avoids a
/// heap allocation for every candidate-factor edge on the runtime hot path.
pub struct FactorGraph {
    /// Candidate values in deterministic identity order.
    pub variables: Vec<VariableNode>,
    /// Compatibility factors compiled for the local candidate neighborhood.
    pub factors: Vec<Box<dyn Factor>>,
    scope_variables: Vec<VarIndex>,
    scope_offsets: Vec<usize>,
}

impl Default for FactorGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl FactorGraph {
    /// Creates an empty local factor graph.
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            factors: Vec::new(),
            scope_variables: Vec::new(),
            scope_offsets: Vec::from([0]),
        }
    }

    /// Builds a graph whose assignment domain contains the supplied
    /// candidates.
    ///
    /// Candidates are sorted and deduplicated so factor lookup is logarithmic
    /// and posterior ordering is deterministic.
    pub fn from_candidates(candidates: &[IdentityId]) -> Self {
        let mut identities = candidates.to_vec();
        identities.sort_unstable();
        identities.dedup();

        let mut graph = Self {
            variables: Vec::with_capacity(identities.len()),
            factors: Vec::new(),
            scope_variables: Vec::new(),
            scope_offsets: Vec::from([0]),
        };

        for candidate_identity in identities {
            graph.add_variable(candidate_identity);
        }

        graph
    }

    /// Adds an assignment value to the local domain.
    ///
    /// Callers constructing graphs incrementally must add values in ascending
    /// identity order before adding factors.
    pub fn add_variable(
        &mut self,
        candidate_identity: IdentityId,
    ) -> VarIndex {
        let index = VarIndex(self.variables.len());
        self.variables.push(VariableNode {
            id: index,
            candidate_identity,
        });
        index
    }

    /// Adds a factor and connects it to candidate values in its scope.
    ///
    /// Candidate identities outside this graph's local domain are ignored. A
    /// factor without any local values is discarded because it cannot affect
    /// this observation's posterior.
    pub fn add_factor(
        &mut self,
        factor: Box<dyn Factor>,
    ) -> Option<FactorIndex> {
        let scope_start = self.scope_variables.len();
        for identity in factor.scope() {
            if let Some(index) = self.variable_index(*identity) {
                self.scope_variables.push(index);
            }
        }

        if scope_start == self.scope_variables.len() {
            return None;
        }

        let factor_index = FactorIndex(self.factors.len());
        self.factors.push(factor);
        self.scope_offsets.push(self.scope_variables.len());
        Some(factor_index)
    }

    /// Returns the index of a candidate identity in the sorted domain.
    pub fn variable_index(&self, identity: IdentityId) -> Option<VarIndex> {
        self.variables
            .binary_search_by_key(&identity, |variable| {
                variable.candidate_identity
            })
            .ok()
            .map(VarIndex)
    }

    /// Returns the candidate values connected to a factor.
    pub fn factor_scope(&self, factor: FactorIndex) -> &[VarIndex] {
        let start = self.scope_offsets[factor.0];
        let end = self.scope_offsets[factor.0 + 1];
        &self.scope_variables[start..end]
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use alloc::boxed::Box;
    use alloc::vec::Vec;

    use li_core::ids::IdentityId;
    use li_core::probability::Probability;
    use li_factors::factor::{Factor, FactorScope};

    use crate::factor_graph::{FactorGraph, FactorIndex, VarIndex};

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

        assert!(graph.variables.is_empty());
        assert!(graph.factors.is_empty());
    }

    #[test]
    fn test_factor_graph_sorts_candidates_and_stores_flat_scope() {
        let mut graph = FactorGraph::from_candidates(&[
            IdentityId(2),
            IdentityId(1),
            IdentityId(2),
        ]);
        let factor = Box::new(ConstantFactor {
            scope_ids: alloc::vec![IdentityId(1), IdentityId(2)],
            value: 1.0,
        });

        let factor_index = graph.add_factor(factor);

        assert_eq!(graph.variables.len(), 2);
        assert_eq!(graph.variables[0].candidate_identity, IdentityId(1));
        assert_eq!(graph.variables[1].candidate_identity, IdentityId(2));
        assert_eq!(factor_index.map(|index| index.0), Some(0));
        assert_eq!(
            graph.factor_scope(FactorIndex(0)),
            &[VarIndex(0), VarIndex(1)]
        );
    }

    #[test]
    fn test_factor_graph_discards_out_of_domain_factor() {
        let mut graph = FactorGraph::from_candidates(&[IdentityId(10)]);
        let factor = Box::new(ConstantFactor {
            scope_ids: alloc::vec![IdentityId(11)],
            value: 1.0,
        });

        assert_eq!(graph.add_factor(factor), None);
        assert!(graph.factors.is_empty());
    }
}
