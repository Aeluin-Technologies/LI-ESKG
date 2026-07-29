//! # LI-ESKG domain primitives
//!
//! This crate contains the pure mathematical definitions, algebraic data
//! types, and core primitives of the Latent Identity Event-State Knowledge
//! Graph (LI-ESKG) framework.

#![deny(unsafe_code, missing_docs)]

pub mod belief;
pub mod events;
pub mod ids;
pub mod observation;
pub mod ontology;
pub mod probability;
pub mod relation;

pub use belief::*;
pub use events::*;
pub use ids::*;
pub use observation::*;
pub use ontology::*;
pub use probability::*;
pub use relation::*;
