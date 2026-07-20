//! # LI-ESKG knowledge model
//!
//! It defines the core graph ontology, relational edge schemas, transactional
//! graph operations, and algebraic invariants matching the paper's theorems.

#![no_std]
#![deny(unsafe_code)]

extern crate alloc;

pub mod graph;
pub mod invariants;
pub mod memory;
pub mod ontology;
pub mod operations;
pub mod projection;
pub mod queries;

pub use graph::*;
pub use invariants::*;
pub use memory::*;
pub use ontology::*;
pub use operations::*;
pub use projection::*;
pub use queries::*;
