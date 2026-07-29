//! Data structures for empirical evidence, temporal tracking, and raw data
//! ingestion.

use std::error::Error;
use std::fmt;
use std::ops::{Add, Sub};
use std::vec::Vec;

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Deserializer, Serialize};

use crate::ids::{IdentityId, ObservationId};
use crate::probability::Confidence;

/// Signed timestamp measured in microseconds since the UNIX epoch.
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
pub struct Timestamp(i64);

/// Error returned when a timestamp cannot be represented by `chrono`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimestampRangeError {
    micros: i64,
}

impl TimestampRangeError {
    /// Returns the rejected UNIX timestamp in microseconds.
    pub const fn micros(self) -> i64 {
        self.micros
    }
}

impl fmt::Display for TimestampRangeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "timestamp {}µs is outside chrono's supported range",
            self.micros
        )
    }
}

impl Error for TimestampRangeError {}

impl Timestamp {
    /// Largest representable timestamp.
    pub const MAX: Self = Self(i64::MAX);
    /// Smallest representable timestamp.
    pub const MIN: Self = Self(i64::MIN);
    /// UNIX epoch timestamp constant (0 microseconds).
    pub const UNIX_EPOCH: Self = Self(0);

    /// Instantiates a timestamp from raw microseconds.
    ///
    /// # Arguments
    ///
    /// * `micros` - Microseconds since UNIX epoch.
    pub const fn from_micros(micros: i64) -> Self {
        Self(micros)
    }

    /// Instantiates a timestamp from milliseconds.
    ///
    /// # Arguments
    ///
    /// * `millis` - Milliseconds since UNIX epoch.
    pub fn from_millis(millis: i64) -> Self {
        Self(millis.saturating_mul(1_000))
    }

    /// Instantiates a timestamp from seconds.
    ///
    /// # Arguments
    ///
    /// * `secs` - Seconds since UNIX epoch.
    pub fn from_secs(secs: i64) -> Self {
        Self(secs.saturating_mul(1_000_000))
    }

    /// Instantiates a timestamp from a `chrono::DateTime<Utc>`.
    ///
    /// # Arguments
    ///
    /// * `dt` - UTC datetime instance.
    pub fn from_datetime(dt: DateTime<Utc>) -> Self {
        Self(dt.timestamp_micros())
    }

    /// Returns the timestamp value in microseconds.
    pub const fn as_micros(&self) -> i64 {
        self.0
    }

    /// Returns the timestamp value in milliseconds.
    pub fn as_millis(&self) -> i64 {
        self.0.div_euclid(1_000)
    }

    /// Returns the timestamp value in seconds.
    pub fn as_secs(&self) -> i64 {
        self.0.div_euclid(1_000_000)
    }

    /// Returns the timestamp value as fractional continuous seconds (`f64`).
    pub fn as_secs_f64(&self) -> f64 {
        self.0 as f64 / 1_000_000.0
    }

    /// Converts the timestamp to a `chrono::DateTime<Utc>`.
    /// Returns `None` if the timestamp is out of range for `DateTime`.
    pub fn to_datetime(&self) -> Option<DateTime<Utc>> {
        DateTime::from_timestamp_micros(self.0)
    }

    /// Computes the duration elapsed since an earlier timestamp.
    /// Returns a zero duration if `earlier` is greater than `self`.
    ///
    /// # Arguments
    ///
    /// * `earlier` - The prior timestamp baseline.
    pub fn duration_since(&self, earlier: &Self) -> Duration {
        if self.0 >= earlier.0 {
            Duration::microseconds(self.0.saturating_sub(earlier.0))
        } else {
            Duration::zero()
        }
    }
}

impl Default for Timestamp {
    fn default() -> Self {
        Self::UNIX_EPOCH
    }
}

impl From<DateTime<Utc>> for Timestamp {
    fn from(dt: DateTime<Utc>) -> Self {
        Self::from_datetime(dt)
    }
}

impl TryFrom<Timestamp> for DateTime<Utc> {
    type Error = TimestampRangeError;

    fn try_from(ts: Timestamp) -> Result<Self, Self::Error> {
        ts.to_datetime().ok_or(TimestampRangeError { micros: ts.0 })
    }
}

impl Sub<Timestamp> for Timestamp {
    type Output = Duration;

    fn sub(self, rhs: Timestamp) -> Self::Output {
        Duration::microseconds(self.0.saturating_sub(rhs.0))
    }
}

impl Add<Duration> for Timestamp {
    type Output = Self;

    fn add(self, rhs: Duration) -> Self::Output {
        match rhs.num_microseconds() {
            Some(micros) => Self(self.0.saturating_add(micros)),
            None if rhs < Duration::zero() => Self::MIN,
            None => Self::MAX,
        }
    }
}

impl Sub<Duration> for Timestamp {
    type Output = Self;

    fn sub(self, rhs: Duration) -> Self::Output {
        match rhs.num_microseconds() {
            Some(micros) => Self(self.0.saturating_sub(micros)),
            None if rhs < Duration::zero() => Self::MAX,
            None => Self::MIN,
        }
    }
}

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

impl<P> Observation<P> {
    /// Instantiates a new empirical observation item.
    pub fn new(
        id: ObservationId,
        modality: Modality,
        timestamp: Timestamp,
        confidence: Confidence,
        payload: P,
    ) -> Self {
        Self {
            id,
            modality,
            timestamp,
            confidence,
            payload,
        }
    }
}

/// Structural evidence package combining an observation with pre-filtered
/// candidate identity nodes.
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct Evidence<P> {
    /// Underlying empirical observation data.
    pub observation: Observation<P>,
    /// Candidates extracted from upstream metric space indexing pipelines.
    pub candidates: Vec<IdentityId>,
}

impl<P> Evidence<P> {
    /// Instantiates a new evidence package, automatically deduplicating
    /// candidate IDs.
    ///
    /// # Arguments
    ///
    /// * `observation` - Inner empirical observation.
    /// * `candidates` - List of candidate identity IDs.
    pub fn new(
        observation: Observation<P>,
        mut candidates: Vec<IdentityId>,
    ) -> Self {
        candidates.sort_unstable();
        candidates.dedup();
        Self {
            observation,
            candidates,
        }
    }
}

#[derive(Deserialize)]
struct EvidenceRepresentation<P> {
    observation: Observation<P>,
    candidates: Vec<IdentityId>,
}

impl<'de, P> Deserialize<'de> for Evidence<P>
where
    P: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let representation =
            EvidenceRepresentation::deserialize(deserializer)?;
        Ok(Self::new(
            representation.observation,
            representation.candidates,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_conversions_preserve_precision() {
        let ts_micros = Timestamp::from_micros(1_500_000);
        assert_eq!(ts_micros.as_micros(), 1_500_000);
        assert_eq!(ts_micros.as_millis(), 1_500);
        assert_eq!(ts_micros.as_secs(), 1);
        assert_eq!(ts_micros.as_secs_f64(), 1.5);

        let ts_millis = Timestamp::from_millis(2_000);
        assert_eq!(ts_millis.as_secs(), 2);

        let ts_secs = Timestamp::from_secs(10);
        assert_eq!(ts_secs.as_micros(), 10_000_000);
    }

    #[test]
    fn timestamp_chrono_interop_is_fallible() {
        let datetime = DateTime::from_timestamp_micros(1_600_000_000_000_000);

        if let Some(datetime) = datetime {
            let timestamp = Timestamp::from_datetime(datetime);
            assert_eq!(timestamp.to_datetime(), Some(datetime));

            let duration = Duration::seconds(5);
            let future = timestamp + duration;
            assert_eq!((future - timestamp).num_seconds(), 5);
        } else {
            assert!(
                datetime.is_some(),
                "test timestamp must be representable"
            );
        }

        let out_of_range = DateTime::<Utc>::try_from(Timestamp::MAX);
        assert!(out_of_range.is_err());
    }

    #[test]
    fn timestamp_arithmetic_saturates_at_numeric_extremes() {
        let one_microsecond = Duration::microseconds(1);

        assert_eq!(Timestamp::MAX + one_microsecond, Timestamp::MAX);
        assert_eq!(Timestamp::MIN - one_microsecond, Timestamp::MIN);
        assert_eq!(Timestamp::UNIX_EPOCH + Duration::MAX, Timestamp::MAX);
        assert_eq!(Timestamp::UNIX_EPOCH + Duration::MIN, Timestamp::MIN);
        assert_eq!(Timestamp::UNIX_EPOCH - Duration::MAX, Timestamp::MIN);
        assert_eq!(Timestamp::UNIX_EPOCH - Duration::MIN, Timestamp::MAX);
        assert_eq!(
            (Timestamp::MAX - Timestamp::MIN).num_microseconds(),
            Some(i64::MAX)
        );
        assert_eq!(
            (Timestamp::MIN - Timestamp::MAX).num_microseconds(),
            Some(i64::MIN)
        );
        assert_eq!(
            Timestamp::MAX.duration_since(&Timestamp::MIN),
            Duration::microseconds(i64::MAX)
        );
        assert_eq!(
            Timestamp::MIN.duration_since(&Timestamp::MAX),
            Duration::zero()
        );
    }

    #[test]
    fn negative_epoch_conversions_use_euclidean_division() {
        let timestamp = Timestamp::from_micros(-1);

        assert_eq!(timestamp.as_millis(), -1);
        assert_eq!(timestamp.as_secs(), -1);
        assert_eq!(Timestamp::from_micros(-1_000_001).as_secs(), -2);
    }

    #[test]
    fn candidate_deduplication_is_deterministic() {
        let obs = Observation::new(
            ObservationId(1),
            Modality(1),
            Timestamp::from_secs(100),
            Confidence::new(0.9),
            (),
        );

        let evidence = Evidence::new(
            obs,
            vec![IdentityId(2), IdentityId(1), IdentityId(2)],
        );
        assert_eq!(evidence.candidates, vec![IdentityId(1), IdentityId(2)]);
    }

    #[test]
    fn evidence_deserialization_canonicalizes_candidates() {
        let serialized = r#"{
            "observation": {
                "id": 1,
                "modality": 2,
                "timestamp": 3,
                "confidence": 0.9,
                "payload": null
            },
            "candidates": [4, 2, 4, 3, 2]
        }"#;
        let evidence = serde_json::from_str::<Evidence<()>>(serialized);

        assert!(evidence.is_ok());
        if let Ok(evidence) = evidence {
            assert_eq!(
                evidence.candidates,
                vec![IdentityId(2), IdentityId(3), IdentityId(4)]
            );
        }
    }
}
