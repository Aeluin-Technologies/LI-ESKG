//! # LI-ESKG workspace
//!
//! This crate implements the ephemeral Belief Layer ($B_t$).
//! It manages tracking hypotheses, integrates upstream metric space candidate
//! generators, controls inference scheduling via information-theoretic
//! planning, and handles time-based eviction.

#![deny(unsafe_code, missing_docs)]

pub mod hot;

pub use hot::*;
