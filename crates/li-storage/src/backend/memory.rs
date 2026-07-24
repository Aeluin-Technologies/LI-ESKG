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

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::keys::ColumnFamily;
    use crate::traits::{KvBackend, KvOp};

    #[test]
    fn test_memory_put_get_delete() {
        let mut backend = MemoryKvBackend::new();
        let cf = ColumnFamily::Ontology;
        let key = b"key1".to_vec();
        let value = b"value1".to_vec();

        assert_eq!(backend.get(cf, &key).unwrap(), None);

        let batch = vec![KvOp::Put {
            cf,
            key: key.clone(),
            value: value.clone(),
        }];
        assert!(backend.apply_transaction(&batch).is_ok());

        assert_eq!(backend.get(cf, &key).unwrap(), Some(value));

        let del_batch = vec![KvOp::Delete {
            cf,
            key: key.clone(),
        }];
        assert!(backend.apply_transaction(&del_batch).is_ok());

        assert_eq!(backend.get(cf, &key).unwrap(), None);
    }

    #[test]
    fn test_memory_prefix_scan() {
        let mut backend = MemoryKvBackend::new();
        let cf = ColumnFamily::Observations;

        let batch = vec![
            KvOp::Put {
                cf,
                key: b"pref_1".to_vec(),
                value: b"v1".to_vec(),
            },
            KvOp::Put {
                cf,
                key: b"pref_2".to_vec(),
                value: b"v2".to_vec(),
            },
            KvOp::Put {
                cf,
                key: b"other_1".to_vec(),
                value: b"v3".to_vec(),
            },
        ];
        backend.apply_transaction(&batch).unwrap();

        let results = backend.prefix_scan(cf, b"pref_").unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (b"pref_1".to_vec(), b"v1".to_vec()));
        assert_eq!(results[1], (b"pref_2".to_vec(), b"v2".to_vec()));

        let empty_prefix = backend.prefix_scan(cf, b"").unwrap();
        assert_eq!(empty_prefix.len(), 3);

        let missing_prefix = backend.prefix_scan(cf, b"nomatch").unwrap();
        assert!(missing_prefix.is_empty());
    }

    #[test]
    fn test_memory_edge_cases() {
        let mut backend = MemoryKvBackend::new();
        let cf = ColumnFamily::States;

        assert!(backend.apply_transaction(&[]).is_ok());

        let del_nonexistent = vec![KvOp::Delete {
            cf,
            key: b"absent".to_vec(),
        }];
        assert!(backend.apply_transaction(&del_nonexistent).is_ok());

        assert_eq!(backend.get(cf, b"absent").unwrap(), None);
        assert!(backend.prefix_scan(cf, b"absent").unwrap().is_empty());
    }
}
