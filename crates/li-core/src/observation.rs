//! Data structures for empirical evidence and raw data ingestion.

use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::ids::{IdentityId, ObservationId};
use crate::probability::Confidence;

/// Monotonically increasing UNIX timestamp measured in microseconds.
#[derive(
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
)]
pub struct Timestamp(pub i64);

/// Unique modality identifier for incoming observation channels.
#[derive(
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
)]
pub struct Modality(pub u32);

/// Immutable empirical observation representing a physical world measurement.
/// Matches the mathematical definition $o = (m, \rho, t, \sigma)$ from the
/// paper.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Observation<P> {
    /// Unique identifier of the observation.
    pub id: ObservationId,
    /// Channel modality type indicator.
    pub modality: Modality,
    /// Temporal marker of the occurrence.
    pub timestamp: Timestamp,
    /// Perception extraction confidence score.
    pub confidence: Confidence,
    /// Modality-specific data payload.
    pub payload: P,
}

/// Structural evidence package combining an observation with its pre-filtered
/// candidates. Serves as the primary operational data transfer object across
/// the Python-Rust boundary.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Evidence<P> {
    /// Underlying empirical observation data.
    pub observation: Observation<P>,
    /// Candidates extracted from upstream metric space indexing pipelines.
    pub candidates: Vec<IdentityId>,
}
