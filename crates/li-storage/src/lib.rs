//! # LI-ESKG storage layer
//!
//! This crate defines persistent storage driver abstractions and database
//! sinks for the Latent Identity Event-State Knowledge Graph (LI-ESKG)
//! framework.

#![deny(unsafe_code)]

pub mod backend;
pub mod errors;
pub mod keys;
pub mod ledger;
pub mod postcard_adapter;
pub mod traits;

pub use backend::MemoryKvBackend;
#[cfg(feature = "rocksdb")]
pub use backend::RocksDbBackend;
pub use errors::StorageError;
pub use ledger::{
    CommitResult, DurableLedger, LedgerError, MaterializationOutbox,
    MaterializationReceipt, MemoryLedger, MergeRecord, ResolutionLedger,
    TransactionRecord,
};
pub use traits::KvBackend;
