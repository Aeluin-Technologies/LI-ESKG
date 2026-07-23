//! # LI-ESKG factor layer
//!
//! This crate provides abstract traits for local factor functions
//! $\phi_i(Z_\phi)$ and factor graph compilation interfaces. It contains no
//! domain-specific spatial, temporal, or semantic assumptions.

#![no_std]
#![deny(unsafe_code, missing_docs)]

extern crate alloc;

pub mod compatibility;
pub mod compiler;
pub mod factor;

pub use compatibility::*;
pub use compiler::*;
pub use factor::*;
