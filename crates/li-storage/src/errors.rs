//! Physical persistence and envelope-encoding errors.

use thiserror::Error;

/// Storage failures exposed by key-value backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum StorageError {
    /// Postcard serialization failed.
    #[error("failed to serialize a resolution envelope")]
    SerializationFailed,
    /// Postcard deserialization failed.
    #[error("failed to deserialize a resolution envelope")]
    DeserializationFailed,
    /// Atomic backend transaction failed.
    #[error("atomic storage transaction failed")]
    TransactionFailed,
    /// Backend-specific operation failed.
    #[error("storage backend error: {0}")]
    BackendError(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_have_stable_nonempty_diagnostics() {
        let errors = [
            StorageError::SerializationFailed,
            StorageError::DeserializationFailed,
            StorageError::TransactionFailed,
            StorageError::BackendError("read failed"),
        ];
        assert!(errors.iter().all(|error| !error.to_string().is_empty()));
    }
}
