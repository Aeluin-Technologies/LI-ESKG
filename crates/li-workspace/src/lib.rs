//! # LI-ESKG workspace
//!
//! This crate implements the ephemeral Belief Layer ($B_t$).
//! It manages tracking hypotheses, integrates upstream metric space candidate
//! generators, controls inference scheduling via information-theoretic
//! planning, and handles time-based eviction.

#![no_std]
#![deny(unsafe_code, missing_docs)]

extern crate alloc;

pub mod candidate;
pub mod checkpoint;
pub mod eviction;
pub mod memory;
pub mod planner;
pub mod workspace;

pub use candidate::*;
pub use checkpoint::*;
pub use eviction::*;
pub use memory::*;
pub use planner::*;
pub use workspace::*;
