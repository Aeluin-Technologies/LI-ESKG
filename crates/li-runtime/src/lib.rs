//! # LI-ESKG execution runtime
//!
//! This crate provides execution orchestration and event-driven pipeline loops
//! for the Latent Identity Event-State Knowledge Graph (LI-ESKG) framework.

#![deny(unsafe_code, missing_docs)]

pub mod pipeline;

pub use pipeline::{
    BatchPhase, BoundedIngress, Committed, Decided, HostMaterializer,
    Inferred, MaterializationFailure, MaterializationPlanner, Persisted,
    PipelineError, ProcessResult, Received, ResolutionRuntime, RetryError,
    RuntimeSnapshot,
};
