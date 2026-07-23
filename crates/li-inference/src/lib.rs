//! # LI-ESKG probabilistic inference engine
//!
//! This crate implements the ephemeral factor graph compilation, Sum-Product
//! Belief Propagation, MAP decision estimation, and the operational pipeline
//! orchestration specified in Algorithm 1.

#![no_std]

extern crate alloc;

pub mod bp;
pub mod factor_graph;
pub mod map;
pub mod posterior;
pub mod scheduler;

pub use bp::*;
pub use factor_graph::*;
pub use map::*;
pub use posterior::*;
pub use scheduler::*;
