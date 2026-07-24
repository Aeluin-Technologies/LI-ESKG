//! Type-safe error handling for storage operations, encoding, and backend
//! interactions.

use thiserror::Error;

/// Storage error variants.
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageError {
    /// Failure during postcard binary serialization.
    #[error("Failed to serialize object with postcard")]
    SerializationFailed,

    /// Failure during postcard binary deserialization.
    #[error("Failed to deserialize object with postcard")]
    DeserializationFailed,

    /// Failure during key encoding or prefix generation.
    #[error("Failed to encode binary key")]
    KeyEncodingError,

    /// Target key or entity was not found in storage.
    #[error("Requested entity not found")]
    NotFound,

    /// Atomic transaction execution failure.
    #[error("Atomic transaction batch failed")]
    TransactionFailed,

    /// Storage backend runtime error.
    #[error("Storage backend error: {0}")]
    BackendError(&'static str),
}
