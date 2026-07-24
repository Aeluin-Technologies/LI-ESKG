//! # LI-ESKG storage layer
//!
//! This crate defines persistent storage driver abstractions and database
//! sinks for the Latent Identity Event-State Knowledge Graph (LI-ESKG)
//! framework.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_code)]

extern crate alloc;

pub mod backend;
pub mod errors;
pub mod keys;
pub mod postcard_adapter;
pub mod store;
pub mod traits;

pub use backend::MemoryKvBackend;
#[cfg(feature = "std")]
pub use backend::RocksDbBackend;
pub use errors::StorageError;
pub use store::StorageEngine;
pub use traits::{CheckpointStore, KvBackend, WalStore};
