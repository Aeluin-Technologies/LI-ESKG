//! Postcard binary serialization and deserialization wrappers.

use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::errors::StorageError;

/// Serializes a reference to a type into a binary byte vector using postcard.
///
/// Args:
///   val: Reference to serializable structure.
///
/// Returns:
///   Encoded byte vector.
///
/// Errors:
///   StorageError::SerializationFailed on encoding failure.
pub fn serialize<T: Serialize>(val: &T) -> Result<Vec<u8>, StorageError> {
    postcard::to_allocvec(val).map_err(|_| StorageError::SerializationFailed)
}

/// Deserializes an instance of T from a binary byte slice using postcard.
///
/// Args:
///   bytes: Encoded byte slice.
///
/// Returns:
///   Deserialized structure T.
///
/// Errors:
///   StorageError::DeserializationFailed on decoding failure.
pub fn deserialize<'a, T: Deserialize<'a>>(
    bytes: &'a [u8],
) -> Result<T, StorageError> {
    postcard::from_bytes(bytes)
        .map_err(|_| StorageError::DeserializationFailed)
}
