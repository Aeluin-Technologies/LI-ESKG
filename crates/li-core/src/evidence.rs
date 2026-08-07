//! Immutable observation envelopes and modality-specific quality metadata.

use std::sync::Arc;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ids::{ObservationId, SchemaId, SourceId};
use crate::observation::{Modality, Timestamp};
use crate::probability::Probability;

/// Collision-resistant digest of immutable observation content.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
)]
pub struct ContentHash([u8; 32]);

impl ContentHash {
    /// Creates a content hash from its exact 256-bit representation.
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Returns the digest bytes by reference.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Immutable payload location kept outside the numeric hot path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PayloadRef {
    /// Shared inline bytes for compact payloads crossing pipeline stages.
    Inline(Bytes),
    /// Backend-owned large object referenced by an opaque key.
    External {
        /// Object-store or blob-segment adapter code.
        backend: u32,
        /// Opaque object key.
        key: Arc<str>,
        /// Exact payload length in bytes.
        length: u64,
    },
}

/// Error returned while constructing an immutable observation envelope.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EvidenceError {
    /// An external payload reference used an empty object key.
    #[error("external payload key must not be empty")]
    EmptyPayloadKey,
    /// A quality field was NaN, infinite, or outside its declared domain.
    #[error("invalid quality metadata: {field}")]
    InvalidQuality {
        /// Stable name of the rejected quality field.
        field: &'static str,
    },
    /// A correction attempted to supersede itself.
    #[error("an observation cannot supersede itself")]
    SelfSupersession,
}

impl PayloadRef {
    /// Creates a validated external payload reference.
    ///
    /// # Errors
    ///
    /// Returns [`EvidenceError::EmptyPayloadKey`] when `key` is empty.
    pub fn external(
        backend: u32,
        key: impl Into<Arc<str>>,
        length: u64,
    ) -> Result<Self, EvidenceError> {
        let key = key.into();
        if key.is_empty() {
            return Err(EvidenceError::EmptyPayloadKey);
        }
        Ok(Self::External {
            backend,
            key,
            length,
        })
    }
}

/// Quality and uncertainty metadata with modality-specific semantics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QualityMetadata {
    /// Position covariance in row-major `[xx, xy, yx, yy]` order.
    PositionCovariance([f64; 4]),
    /// Detector class and localization probabilities.
    Detection {
        /// Calibrated class probability.
        class_probability: Probability,
        /// Localization quality, distinct from class probability.
        localization_quality: Probability,
    },
    /// Text mention byte boundaries and calibrated extraction probability.
    TextMention {
        /// Inclusive UTF-8 byte start.
        start: u32,
        /// Exclusive UTF-8 byte end.
        end: u32,
        /// Calibrated extraction probability.
        score: Probability,
    },
    /// Provider-owned encoded metadata with an explicit schema.
    Opaque {
        /// Schema that defines the byte representation.
        schema: SchemaId,
        /// Immutable provider-owned bytes.
        bytes: Bytes,
    },
}

impl QualityMetadata {
    /// Creates a finite, symmetric, positive-semidefinite 2D covariance.
    ///
    /// # Arguments
    ///
    /// * `matrix` - Row-major covariance entries `[xx, xy, yx, yy]`.
    ///
    /// # Errors
    ///
    /// Returns [`EvidenceError::InvalidQuality`] for non-finite, asymmetric,
    /// negative-diagonal, or negative-determinant matrices.
    pub fn position_covariance(
        matrix: [f64; 4],
    ) -> Result<Self, EvidenceError> {
        let finite = matrix.iter().all(|value| value.is_finite());
        let symmetric = (matrix[1] - matrix[2]).abs() <= f64::EPSILON;
        let determinant =
            matrix[0].mul_add(matrix[3], -(matrix[1] * matrix[2]));
        if !finite ||
            !symmetric ||
            matrix[0] < 0.0 ||
            matrix[3] < 0.0 ||
            determinant < 0.0
        {
            return Err(EvidenceError::InvalidQuality {
                field: "position covariance",
            });
        }
        Ok(Self::PositionCovariance(matrix))
    }

    /// Creates text mention quality with a non-empty byte interval.
    ///
    /// # Errors
    ///
    /// Returns [`EvidenceError::InvalidQuality`] when `start >= end`.
    pub fn text_mention(
        start: u32,
        end: u32,
        score: Probability,
    ) -> Result<Self, EvidenceError> {
        if start >= end {
            return Err(EvidenceError::InvalidQuality {
                field: "text mention interval",
            });
        }
        Ok(Self::TextMention { start, end, score })
    }
}

/// Immutable V2 evidence tuple `(u, source, modality, te, ts, payload,
/// quality, hash)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationEnvelope {
    id: ObservationId,
    source: SourceId,
    modality: Modality,
    event_time: Timestamp,
    ingestion_time: Timestamp,
    payload: PayloadRef,
    quality: QualityMetadata,
    content_hash: ContentHash,
    supersedes: Option<ObservationId>,
}

impl ObservationEnvelope {
    /// Creates an immutable observation with separate event and ingestion
    /// clocks.
    ///
    /// No ordering constraint is imposed between the clocks: delayed data and
    /// imperfect source clocks may place either timestamp first.
    ///
    /// # Errors
    ///
    /// Returns [`EvidenceError::SelfSupersession`] when a correction points to
    /// its own identifier.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: ObservationId,
        source: SourceId,
        modality: Modality,
        event_time: Timestamp,
        ingestion_time: Timestamp,
        payload: PayloadRef,
        quality: QualityMetadata,
        content_hash: ContentHash,
        supersedes: Option<ObservationId>,
    ) -> Result<Self, EvidenceError> {
        if supersedes == Some(id) {
            return Err(EvidenceError::SelfSupersession);
        }
        Ok(Self {
            id,
            source,
            modality,
            event_time,
            ingestion_time,
            payload,
            quality,
            content_hash,
            supersedes,
        })
    }

    /// Returns the stable observation identifier.
    pub const fn id(&self) -> ObservationId {
        self.id
    }

    /// Returns the stable source code.
    pub const fn source(&self) -> SourceId {
        self.source
    }

    /// Returns the compact modality code.
    pub const fn modality(&self) -> Modality {
        self.modality
    }

    /// Returns the observed-world event time.
    pub const fn event_time(&self) -> Timestamp {
        self.event_time
    }

    /// Returns the system ingestion time.
    pub const fn ingestion_time(&self) -> Timestamp {
        self.ingestion_time
    }

    /// Borrows the immutable payload reference.
    pub const fn payload(&self) -> &PayloadRef {
        &self.payload
    }

    /// Borrows modality-specific quality metadata.
    pub const fn quality(&self) -> &QualityMetadata {
        &self.quality
    }

    /// Returns the immutable content digest.
    pub const fn content_hash(&self) -> ContentHash {
        self.content_hash
    }

    /// Returns the earlier observation corrected by this record, if any.
    pub const fn supersedes(&self) -> Option<ObservationId> {
        self.supersedes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quality() -> QualityMetadata {
        QualityMetadata::Opaque {
            schema: SchemaId(1),
            bytes: Bytes::new(),
        }
    }

    #[test]
    fn covariance_rejects_invalid_matrices() {
        assert!(
            QualityMetadata::position_covariance([1.0, 0.0, 0.0, 2.0]).is_ok()
        );
        assert!(
            QualityMetadata::position_covariance([1.0, 2.0, 0.0, 1.0])
                .is_err()
        );
        assert!(
            QualityMetadata::position_covariance([1.0, 2.0, 2.0, 1.0])
                .is_err()
        );
        assert!(
            QualityMetadata::position_covariance([f64::NAN, 0.0, 0.0, 1.0])
                .is_err()
        );
    }

    #[test]
    fn payload_and_text_ranges_reject_empty_values() {
        assert_eq!(
            PayloadRef::external(1, "", 0),
            Err(EvidenceError::EmptyPayloadKey)
        );
        assert!(PayloadRef::external(1, "blob-1", 0).is_ok());
        assert!(
            QualityMetadata::text_mention(4, 4, Probability::ONE).is_err()
        );
        assert!(QualityMetadata::text_mention(4, 5, Probability::ONE).is_ok());
    }

    #[test]
    fn observations_keep_two_clocks_and_reject_self_correction() {
        let payload = PayloadRef::Inline(Bytes::from_static(b"evidence"));
        let result = ObservationEnvelope::new(
            ObservationId(9),
            SourceId(2),
            Modality(3),
            Timestamp::from_micros(20),
            Timestamp::from_micros(10),
            payload.clone(),
            quality(),
            ContentHash::new([7; 32]),
            None,
        );
        assert!(result.is_ok());
        if let Ok(observation) = result {
            assert_eq!(observation.event_time(), Timestamp::from_micros(20));
            assert_eq!(
                observation.ingestion_time(),
                Timestamp::from_micros(10)
            );
            assert_eq!(observation.payload(), &payload);
        }

        let self_correction = ObservationEnvelope::new(
            ObservationId(9),
            SourceId(2),
            Modality(3),
            Timestamp::UNIX_EPOCH,
            Timestamp::UNIX_EPOCH,
            PayloadRef::Inline(Bytes::new()),
            quality(),
            ContentHash::new([0; 32]),
            Some(ObservationId(9)),
        );
        assert_eq!(self_correction, Err(EvidenceError::SelfSupersession));
    }
}
