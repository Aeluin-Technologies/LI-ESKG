//! Postcard binary serialization and deserialization wrappers.

use serde::{Deserialize, Serialize};

use crate::errors::StorageError;

/// Serializes a reference to a type into a binary byte vector using postcard.
pub fn serialize<T: Serialize>(val: &T) -> Result<Vec<u8>, StorageError> {
    postcard::to_allocvec(val).map_err(|_| StorageError::SerializationFailed)
}

/// Deserializes an instance of T from a binary byte slice using postcard.
pub fn deserialize<'a, T: Deserialize<'a>>(
    bytes: &'a [u8],
) -> Result<T, StorageError> {
    postcard::from_bytes(bytes)
        .map_err(|_| StorageError::DeserializationFailed)
}
