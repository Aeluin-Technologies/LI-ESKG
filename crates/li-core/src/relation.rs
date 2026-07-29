//! Semantic relation typings and topology schema validation.

use serde::{Deserialize, Serialize};

use crate::ontology::Vertex;

/// Classification of semantic edges allowed within the knowledge graph schema.
#[derive(
    Deserialize,
    Serialize,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
)]
pub enum Relation {
    /// Causal event triggering a state transition ($E \times S \to
    /// R_{\text{eskg}}$).
    Trigger,
    /// Temporal sequencing between subsequent states ($S \times S \to
    /// R_{\text{eskg}}$).
    Lead,
    /// Evolution between successive entity states ($S \times S \to
    /// R_{\text{eskg}}$).
    Evolution,
    /// Mereological decomposition of composite states ($S \times S \to
    /// R_{\text{eskg}}$).
    Contain,
    /// Spatio-temporal anchoring of events ($E \times S \to R_{\text{eskg}}$).
    Occur,
    /// Causal interaction link between distinct events ($E \times E \to
    /// R_{\text{eskg}}$).
    Influence,

    /// Temporal binding of evidence to a window ($O \times S \to
    /// R_{\text{LI}}$).
    ObservedDuring,
    /// Direct semantic property attribution ($O \times S \to R_{\text{LI}}$).
    Describes,
    /// Empirical evidence supporting an identity ($(o, \text{supports}, i) \in
    /// R_{\text{LI}}$).
    Supports,
    /// Structural merging of historical identities ($I \times I \to
    /// R_{\text{LI}}$).
    Refines,
    /// Probabilistic association between entities ($I \times I \to
    /// R_{\text{LI}}$).
    AssociatedWith,
}

impl Relation {
    /// Returns `true` if the edge belongs to the persistent ESKG causal
    /// subgraph ($R_{\text{eskg}}$).
    pub fn is_eskg_relation(&self) -> bool {
        matches!(
            self,
            Self::Trigger |
                Self::Lead |
                Self::Evolution |
                Self::Contain |
                Self::Occur |
                Self::Influence
        )
    }

    /// Returns `true` if the edge belongs to the LI operational subgraph
    /// ($R_{\text{LI}}$).
    pub fn is_li_relation(&self) -> bool {
        !self.is_eskg_relation()
    }

    /// Validates whether a directed edge `(source, relation, target)` respects
    /// the schema invariants.
    ///
    /// # Arguments
    ///
    /// * `source` - Origin vertex.
    /// * `target` - Destination vertex.
    ///
    /// # Examples
    ///
    /// ```
    /// use li_core::ids::{EventId, StateId};
    /// use li_core::ontology::Vertex;
    /// use li_core::relation::Relation;
    ///
    /// let event = Vertex::Event(EventId(1));
    /// let state = Vertex::State(StateId(2));
    /// assert!(Relation::Trigger.is_valid_transition(&event, &state));
    /// assert!(!Relation::Trigger.is_valid_transition(&state, &event));
    /// ```
    pub fn is_valid_transition(
        &self,
        source: &Vertex,
        target: &Vertex,
    ) -> bool {
        match self {
            Self::Trigger => matches!(
                (source, target),
                (Vertex::Event(_), Vertex::State(_))
            ),
            Self::Lead => {
                matches!(
                    (source, target),
                    (Vertex::State(_), Vertex::State(_))
                ) && source != target
            },
            Self::Evolution => {
                matches!(
                    (source, target),
                    (Vertex::State(_), Vertex::State(_))
                ) && source != target
            },
            Self::Contain => {
                matches!(
                    (source, target),
                    (Vertex::State(_), Vertex::State(_))
                ) && source != target
            },
            Self::Occur => matches!(
                (source, target),
                (Vertex::Event(_), Vertex::State(_))
            ),
            Self::Influence => {
                matches!(
                    (source, target),
                    (Vertex::Event(_), Vertex::Event(_))
                ) && source != target
            },

            Self::ObservedDuring => {
                matches!(
                    (source, target),
                    (Vertex::Observation(_), Vertex::State(_))
                )
            },
            Self::Describes => {
                matches!(
                    (source, target),
                    (Vertex::Observation(_), Vertex::State(_))
                )
            },
            Self::Supports => matches!(
                (source, target),
                (Vertex::Observation(_), Vertex::Identity(_))
            ),
            Self::Refines => {
                matches!(
                    (source, target),
                    (Vertex::Identity(_), Vertex::Identity(_))
                ) && source != target
            },
            Self::AssociatedWith => {
                matches!(
                    (source, target),
                    (Vertex::Identity(_), Vertex::Identity(_))
                ) && source != target
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::*;

    #[test]
    fn relation_domains_are_exhaustively_enforced() {
        let source_vertices = [
            Vertex::Observation(ObservationId(1)),
            Vertex::Identity(IdentityId(1)),
            Vertex::Event(EventId(1)),
            Vertex::State(StateId(1)),
        ];
        let target_vertices = [
            Vertex::Observation(ObservationId(2)),
            Vertex::Identity(IdentityId(2)),
            Vertex::Event(EventId(2)),
            Vertex::State(StateId(2)),
        ];
        let cases = [
            (Relation::Trigger, 2, 3),
            (Relation::Lead, 3, 3),
            (Relation::Evolution, 3, 3),
            (Relation::Contain, 3, 3),
            (Relation::Occur, 2, 3),
            (Relation::Influence, 2, 2),
            (Relation::ObservedDuring, 0, 3),
            (Relation::Describes, 0, 3),
            (Relation::Supports, 0, 1),
            (Relation::Refines, 1, 1),
            (Relation::AssociatedWith, 1, 1),
        ];

        for (relation, expected_source, expected_target) in cases {
            for (source_index, source) in source_vertices.iter().enumerate() {
                for (target_index, target) in
                    target_vertices.iter().enumerate()
                {
                    assert_eq!(
                        relation.is_valid_transition(source, target),
                        source_index == expected_source &&
                            target_index == expected_target,
                        "unexpected domain for {relation:?}: {source:?} -> {target:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn same_partition_structural_self_loops_are_rejected() {
        let cases = [
            (Relation::Lead, Vertex::State(StateId(7))),
            (Relation::Evolution, Vertex::State(StateId(7))),
            (Relation::Contain, Vertex::State(StateId(7))),
            (Relation::Influence, Vertex::Event(EventId(7))),
            (Relation::Refines, Vertex::Identity(IdentityId(7))),
            (Relation::AssociatedWith, Vertex::Identity(IdentityId(7))),
        ];

        for (relation, vertex) in cases {
            assert!(!relation.is_valid_transition(&vertex, &vertex));
        }
    }

    #[test]
    fn relation_layers_partition_the_relation_set() {
        let eskg = [
            Relation::Trigger,
            Relation::Lead,
            Relation::Evolution,
            Relation::Contain,
            Relation::Occur,
            Relation::Influence,
        ];
        let li = [
            Relation::ObservedDuring,
            Relation::Describes,
            Relation::Supports,
            Relation::Refines,
            Relation::AssociatedWith,
        ];

        for relation in eskg {
            assert!(relation.is_eskg_relation());
            assert!(!relation.is_li_relation());
        }
        for relation in li {
            assert!(!relation.is_eskg_relation());
            assert!(relation.is_li_relation());
        }
    }
}
