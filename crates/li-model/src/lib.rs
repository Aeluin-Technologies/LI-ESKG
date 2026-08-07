//! # LI-ESKG knowledge model
//!
//! It defines the core graph ontology, relational edge schemas, transactional
//! graph operations, and algebraic invariants matching the paper's theorems.

#![deny(unsafe_code)]

pub mod host_graph;
pub mod interoperability;
pub mod spatial;

pub use host_graph::{
    AuthoritativeHostGraph, HostEdge, HostGraphError, HostSchemaError,
    HostSchemaProfile, MaterializationOutcome,
};
pub use interoperability::{
    InteroperabilityProjector, ProjectionError, RdfProfile,
};
pub use spatial::{
    Covariance2D, GeoPoint, SpatialComponent, SpatialError, SpatialGaussian2D,
};
