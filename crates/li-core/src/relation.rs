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
    /// Structural tracking of entity properties ($I \times S \to
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
            Self::Lead => matches!(
                (source, target),
                (Vertex::State(_), Vertex::State(_))
            ),
            Self::Evolution => matches!(
                (source, target),
                (Vertex::Identity(_), Vertex::State(_))
            ),
            Self::Contain => matches!(
                (source, target),
                (Vertex::State(_), Vertex::State(_))
            ),
            Self::Occur => matches!(
                (source, target),
                (Vertex::Event(_), Vertex::State(_))
            ),
            Self::Influence => matches!(
                (source, target),
                (Vertex::Event(_), Vertex::Event(_))
            ),

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
                )
            },
            Self::AssociatedWith => {
                matches!(
                    (source, target),
                    (Vertex::Identity(_), Vertex::Identity(_))
                )
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::*;

    #[test]
    fn test_schema_validations() {
        let obs = Vertex::Observation(ObservationId(1));
        let ident = Vertex::Identity(IdentityId(2));
        let state = Vertex::State(StateId(3));

        assert!(Relation::Supports.is_valid_transition(&obs, &ident));
        assert!(!Relation::Supports.is_valid_transition(&ident, &obs));
        assert!(!Relation::Supports.is_valid_transition(&obs, &state));
    }
}
