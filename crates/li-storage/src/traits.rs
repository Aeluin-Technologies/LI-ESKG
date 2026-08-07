//! Atomic key-value backend abstraction for durable  envelope storage.

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

/// Interface for physical key-value storage drivers.
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
