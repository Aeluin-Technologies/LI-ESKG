//! Latent Identity ESKG Inference Engine (`li-inference`).
//!
//! This crate implements allocation-reusing log-domain Sum-Product for
//! collective factor graphs, including exact-tree and loopy diagnostics.

pub mod solver;

pub use solver::{
    BatchPosterior, SolverError, SolverScratch, SumProductConfig,
    SumProductSolver,
};
