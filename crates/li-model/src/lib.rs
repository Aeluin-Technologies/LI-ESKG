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

pub use graph::{GraphError, KnowledgeGraph, PetGraphStore};
pub use ontology::{EdgeData, NodeData};
pub use operations::GraphOperation;
pub use projection::{EventStateGraph, EventStateProjection, GraphProjection};
