//! # LI-ESKG execution runtime
//!
//! This crate provides execution orchestration and event-driven pipeline loops
//! for the Latent Identity Event-State Knowledge Graph (LI-ESKG) framework.

#![deny(unsafe_code, missing_docs)]

extern crate alloc;

pub mod channels;
pub mod dispatcher;
pub mod engine;
pub mod executor;

pub use channels::EventQueue;
pub use dispatcher::{DispatchOutcome, EventDispatcher};
pub use engine::{EngineConfig, RuntimeEngine};
pub use executor::{ExecutionSink, OperationExecutor};
