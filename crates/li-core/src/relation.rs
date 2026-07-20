//! Semantic relation typings for edge classification.

/// Classification of semantic edges allowed within the knowledge graph schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Relation {
    /// Causal event triggering a state transition ($E \times S \to R_{eskg}$).
    Trigger,
    /// Temporal sequencing between subsequent states ($S \times S \to
    /// R_{eskg}$).
    Lead,
    /// Structural tracking of entity properties ($I \times S \to R_{eskg}$).
    Evolution,
    /// Mereological decomposition of composite states ($S \times S \to
    /// R_{eskg}$).
    Contain,
    /// Spatio-temporal anchoring of events ($E \times S \to R_{eskg}$).
    Occur,
    /// Causal interaction link between distinct events ($E \times E \to
    /// R_{eskg}$).
    Influence,

    /// Temporal binding of evidence to a window ($O \times S \to R_{LI}$).
    ObservedDuring,
    /// Direct semantic property attribution ($O \times S \to R_{LI}$).
    Describes,
    /// Empirical evidence supporting an identity ($(o, \text{supports}, i) \in
    /// R_{LI}$).
    Supports,
    /// Structural merging of historical identities ($I \times I \to R_{LI}$).
    Refines,
    /// Probabilistic association between entities ($I \times I \to R_{LI}$).
    AssociatedWith,
}
