//! Type-safe error handling for storage operations, encoding, and backend
//! interactions.

use li_core::ids::IdentityId;
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

    /// An identity required by a canonicalization operation does not exist.
    #[error("Identity {identity:?} was not found for canonicalization")]
    IdentityNotFound {
        /// Missing canonical or duplicate identity.
        identity: IdentityId,
    },

    /// A canonicalization operation used the same identity for both roles.
    #[error("Cannot merge identity {identity:?} into itself")]
    SelfIdentityMerge {
        /// Identity present in both merge roles.
        identity: IdentityId,
    },

    /// A merge was combined with another operation in one logical batch.
    #[error(
        "Identity merge operations must be committed in an isolated batch"
    )]
    MergeOperationMustBeIsolated,

    /// A persisted identity record disagrees with its tagged ontology key.
    #[error("Stored identity {identity:?} has inconsistent ontology records")]
    CorruptIdentityRecord {
        /// Identity whose storage records disagree.
        identity: IdentityId,
    },

    /// An edge value disagrees with its directional storage key.
    #[error("Stored edge payload does not match its directional key")]
    CorruptEdgeRecord,

    /// Atomic transaction execution failure.
    #[error("Atomic transaction batch failed")]
    TransactionFailed,

    /// Storage backend runtime error.
    #[error("Storage backend error: {0}")]
    BackendError(&'static str),
}
