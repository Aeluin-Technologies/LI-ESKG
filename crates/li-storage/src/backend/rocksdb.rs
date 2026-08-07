//! RocksDB driver implementation mapping native driver errors to static
//! StorageError variants.

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
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, StorageError> {
        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);

        let cfs = vec![ColumnFamilyDescriptor::new(
            ColumnFamily::ResolutionLedger.as_str(),
            Options::default(),
        )];

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

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::keys::ColumnFamily;
    use crate::traits::{KvBackend, KvOp};

    #[test]
    fn test_rocksdb_open_put_get_delete()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let mut backend = RocksDbBackend::open(dir.path())?;
        let cf = ColumnFamily::ResolutionLedger;
        let key = b"id_key".to_vec();
        let value = b"id_val".to_vec();

        assert_eq!(backend.get(cf, &key)?, None);

        let batch = vec![KvOp::Put {
            cf,
            key: key.clone(),
            value: value.clone(),
        }];
        assert!(backend.apply_transaction(&batch).is_ok());

        assert_eq!(backend.get(cf, &key)?, Some(value));

        let del_batch = vec![KvOp::Delete {
            cf,
            key: key.clone(),
        }];
        assert!(backend.apply_transaction(&del_batch).is_ok());

        assert_eq!(backend.get(cf, &key)?, None);
        Ok(())
    }

    #[test]
    fn test_rocksdb_prefix_scan_and_persistence()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let cf = ColumnFamily::ResolutionLedger;

        {
            let mut backend = RocksDbBackend::open(dir.path())?;
            let batch = vec![
                KvOp::Put {
                    cf,
                    key: b"wal_001".to_vec(),
                    value: b"payload1".to_vec(),
                },
                KvOp::Put {
                    cf,
                    key: b"wal_002".to_vec(),
                    value: b"payload2".to_vec(),
                },
                KvOp::Put {
                    cf,
                    key: b"other_001".to_vec(),
                    value: b"payload3".to_vec(),
                },
            ];
            backend.apply_transaction(&batch)?;

            let scan = backend.prefix_scan(cf, b"wal_")?;
            assert_eq!(scan.len(), 2);
            assert_eq!(scan[0], (b"wal_001".to_vec(), b"payload1".to_vec()));
            assert_eq!(scan[1], (b"wal_002".to_vec(), b"payload2".to_vec()));
        }

        {
            let backend = RocksDbBackend::open(dir.path())?;
            assert_eq!(
                backend.get(cf, b"wal_001")?,
                Some(b"payload1".to_vec())
            );
        }
        Ok(())
    }

    #[test]
    fn test_rocksdb_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let mut backend = RocksDbBackend::open(dir.path())?;
        let cf = ColumnFamily::ResolutionLedger;

        assert!(backend.apply_transaction(&[]).is_ok());
        assert_eq!(backend.get(cf, b"non_existent")?, None);
        assert!(backend.prefix_scan(cf, b"non_existent")?.is_empty());
        Ok(())
    }
}
