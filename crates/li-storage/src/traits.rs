//! Abstraction layer for Key-Value physical backends, workspace checkpoints,
//! and WAL loggers.

use alloc::vec::Vec;

use li_core::belief::BeliefState;
use li_core::observation::Evidence;

use crate::errors::StorageError;
use crate::keys::ColumnFamily;

pub type KvRecord = (Vec<u8>, Vec<u8>);

/// Represents an atomic key-value operation within a storage batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvOp {
    /// Inserts or updates a key-value entry in a Column Family.
    Put {
        cf: ColumnFamily,
        key: Vec<u8>,
        value: Vec<u8>,
    },
    /// Deletes a key from a Column Family.
    Delete { cf: ColumnFamily, key: Vec<u8> },
}

/// Interface for physical Key-Value storage drivers operating in no-std or std
/// environments.
pub trait KvBackend {
    /// Reads an entry by key from a Column Family.
    fn get(
        &self,
        cf: ColumnFamily,
        key: &[u8],
    ) -> Result<Option<Vec<u8>>, StorageError>;

    /// Scans a Column Family for keys sharing a specific byte prefix.
    fn prefix_scan(
        &self,
        cf: ColumnFamily,
        prefix: &[u8],
    ) -> Result<Vec<KvRecord>, StorageError>;

    /// Executes an atomic transaction batch against the backend.
    fn apply_transaction(
        &mut self,
        batch: &[KvOp],
    ) -> Result<(), StorageError>;
}

/// Interface for persisting active belief states.
pub trait CheckpointStore<S> {
    /// Persists a vector of active belief states.
    fn save_checkpoint(
        &mut self,
        beliefs: &[BeliefState<S>],
    ) -> Result<(), StorageError>;

    /// Restores the latest persisted active belief states.
    fn load_latest_checkpoint(
        &self,
    ) -> Result<Vec<BeliefState<S>>, StorageError>;
}

/// Interface for Write-Ahead Log operations.
pub trait WalStore<P> {
    /// Appends incoming evidence into the WAL and returns its sequence number.
    fn append_evidence(
        &mut self,
        evidence: &Evidence<P>,
    ) -> Result<u64, StorageError>;

    /// Reads evidence entries starting from a given sequence number.
    fn read_delta(
        &self,
        from_sequence: u64,
    ) -> Result<Vec<Evidence<P>>, StorageError>;

    /// Truncates log entries up to a given sequence boundary.
    fn truncate_up_to(&mut self, sequence: u64) -> Result<(), StorageError>;
}
