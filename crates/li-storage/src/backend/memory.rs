//! In-memory BTreeMap implementation of KvBackend for no-std targets.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::errors::StorageError;
use crate::keys::ColumnFamily;
use crate::traits::{KvBackend, KvOp};

/// In-memory tree-backed driver implementation for embedded environments and
/// tests.
#[derive(Debug, Clone, Default)]
pub struct MemoryKvBackend {
    tables: BTreeMap<ColumnFamily, BTreeMap<Vec<u8>, Vec<u8>>>,
}

impl MemoryKvBackend {
    /// Constructs a new empty [`MemoryKvBackend`].
    pub fn new() -> Self {
        Self {
            tables: BTreeMap::new(),
        }
    }
}

impl KvBackend for MemoryKvBackend {
    fn get(
        &self,
        cf: ColumnFamily,
        key: &[u8],
    ) -> Result<Option<Vec<u8>>, StorageError> {
        Ok(self
            .tables
            .get(&cf)
            .and_then(|table| table.get(key).cloned()))
    }

    fn prefix_scan(
        &self,
        cf: ColumnFamily,
        prefix: &[u8],
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>, StorageError> {
        let mut results = Vec::new();
        if let Some(table) = self.tables.get(&cf) {
            for (k, v) in table.range(prefix.to_vec()..) {
                if k.starts_with(prefix) {
                    results.push((k.clone(), v.clone()));
                } else {
                    break;
                }
            }
        }
        Ok(results)
    }

    fn apply_transaction(
        &mut self,
        batch: &[KvOp],
    ) -> Result<(), StorageError> {
        for op in batch {
            match op {
                KvOp::Put { cf, key, value } => {
                    self.tables
                        .entry(*cf)
                        .or_default()
                        .insert(key.clone(), value.clone());
                },
                KvOp::Delete { cf, key } => {
                    if let Some(table) = self.tables.get_mut(cf) {
                        table.remove(key);
                    }
                },
            }
        }
        Ok(())
    }
}
