//! RocksDB driver implementation mapping native driver errors to static
//! StorageError variants.

use alloc::vec::Vec;
use std::path::Path;

use rocksdb::{ColumnFamilyDescriptor, DB, Options, WriteBatch};

use crate::errors::StorageError;
use crate::keys::ColumnFamily;
use crate::traits::{KvBackend, KvOp};

/// RocksDB backend wrapper implementing KvBackend for OS environments.
pub struct RocksDbBackend {
    db: DB,
}

impl RocksDbBackend {
    /// Opens or creates a RocksDB instance with required Column Families at
    /// path.
    ///
    /// Args:
    ///   path: File system path to storage directory.
    ///
    /// Returns:
    ///   Configured RocksDbBackend instance.
    ///
    /// Errors:
    ///   StorageError::BackendError on initialization failure.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, StorageError> {
        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);

        let cfs = vec![
            ColumnFamilyDescriptor::new(
                ColumnFamily::Ontology.as_str(),
                Options::default(),
            ),
            ColumnFamilyDescriptor::new(
                ColumnFamily::Observations.as_str(),
                Options::default(),
            ),
            ColumnFamilyDescriptor::new(
                ColumnFamily::Identities.as_str(),
                Options::default(),
            ),
            ColumnFamilyDescriptor::new(
                ColumnFamily::Events.as_str(),
                Options::default(),
            ),
            ColumnFamilyDescriptor::new(
                ColumnFamily::States.as_str(),
                Options::default(),
            ),
            ColumnFamilyDescriptor::new(
                ColumnFamily::OutEdges.as_str(),
                Options::default(),
            ),
            ColumnFamilyDescriptor::new(
                ColumnFamily::InEdges.as_str(),
                Options::default(),
            ),
            ColumnFamilyDescriptor::new(
                ColumnFamily::Checkpoints.as_str(),
                Options::default(),
            ),
            ColumnFamilyDescriptor::new(
                ColumnFamily::WalObservations.as_str(),
                Options::default(),
            ),
        ];

        let db =
            DB::open_cf_descriptors(&db_opts, path, cfs).map_err(|_| {
                StorageError::BackendError("Failed to open RocksDB database")
            })?;

        Ok(Self { db })
    }
}

impl KvBackend for RocksDbBackend {
    fn get(
        &self,
        cf: ColumnFamily,
        key: &[u8],
    ) -> Result<Option<Vec<u8>>, StorageError> {
        let handle = self
            .db
            .cf_handle(cf.as_str())
            .ok_or(StorageError::BackendError("Column family not found"))?;

        self.db.get_cf(&handle, key).map_err(|_| {
            StorageError::BackendError("RocksDB get_cf operation failed")
        })
    }

    fn prefix_scan(
        &self,
        cf: ColumnFamily,
        prefix: &[u8],
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>, StorageError> {
        let handle = self
            .db
            .cf_handle(cf.as_str())
            .ok_or(StorageError::BackendError("Column family not found"))?;

        let mut mode = rocksdb::IteratorMode::Start;
        if !prefix.is_empty() {
            mode = rocksdb::IteratorMode::From(
                prefix,
                rocksdb::Direction::Forward,
            );
        }

        let iter = self.db.iterator_cf(&handle, mode);
        let mut results = Vec::new();

        for item in iter {
            let (k, v) = item.map_err(|_| {
                StorageError::BackendError("RocksDB iterator error")
            })?;
            if k.starts_with(prefix) {
                results.push((k.to_vec(), v.to_vec()));
            } else {
                break;
            }
        }

        Ok(results)
    }

    fn apply_transaction(
        &mut self,
        batch: &[KvOp],
    ) -> Result<(), StorageError> {
        let mut write_batch = WriteBatch::default();

        for op in batch {
            match op {
                KvOp::Put { cf, key, value } => {
                    let handle = self.db.cf_handle(cf.as_str()).ok_or(
                        StorageError::BackendError("Column family not found"),
                    )?;
                    write_batch.put_cf(&handle, key, value);
                },
                KvOp::Delete { cf, key } => {
                    let handle = self.db.cf_handle(cf.as_str()).ok_or(
                        StorageError::BackendError("Column family not found"),
                    )?;
                    write_batch.delete_cf(&handle, key);
                },
            }
        }

        self.db.write(write_batch).map_err(|_| {
            StorageError::BackendError("RocksDB write_batch commit failed")
        })
    }
}
