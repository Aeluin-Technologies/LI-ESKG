//! # LI-ESKG knowledge model
//!
//! It defines the core graph ontology, relational edge schemas, transactional
//! graph operations, and algebraic invariants matching the paper's theorems.

#![deny(unsafe_code)]

pub mod graph;
pub mod invariants;
pub mod ontology;
pub mod operations;
pub mod projection;
pub mod queries;
pub mod spatial;

pub use graph::{
    GraphError, KnowledgeGraph, MergeOutcome, PetGraphStore, RawGraph,
};
pub use invariants::{Invariant, InvariantViolation};
pub use ontology::{EdgeData, NodeData};
pub use operations::{GraphOperation, IdentityAssignment};
pub use projection::{
    EventStateGraph, EventStateNode, EventStateProjection, GraphProjection,
};
pub use spatial::{
    Covariance2D, GeoPoint, SpatialComponent, SpatialError, SpatialGaussian2D,
};
