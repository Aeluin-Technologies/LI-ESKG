//! Data structures for empirical evidence, temporal tracking, and raw data
//! ingestion.

use alloc::vec::Vec;
use core::ops::{Add, Sub};

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::{IdentityId, ObservationId};
use crate::probability::Confidence;

/// Monotonically increasing timestamp measured in microseconds since the UNIX
/// epoch. Encapsulates `chrono` types for multi-scale time manipulation.
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

impl Timestamp {
    /// UNIX epoch timestamp constant (0 microseconds).
    pub const UNIX_EPOCH: Self = Self(0);

    /// Instantiates a timestamp from raw microseconds.
    ///
    /// # Arguments
    ///
    /// * `micros` - Microseconds since UNIX epoch.
    pub fn from_micros(micros: i64) -> Self {
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
    pub fn as_micros(&self) -> i64 {
        self.0
    }

    /// Returns the timestamp value in milliseconds.
    pub fn as_millis(&self) -> i64 {
        self.0 / 1_000
    }

    /// Returns the timestamp value in seconds.
    pub fn as_secs(&self) -> i64 {
        self.0 / 1_000_000
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
            Duration::microseconds(self.0 - earlier.0)
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
    type Error = ();

    fn try_from(ts: Timestamp) -> Result<Self, Self::Error> {
        ts.to_datetime().ok_or(())
    }
}

impl Sub<Timestamp> for Timestamp {
    type Output = Duration;

    fn sub(self, rhs: Timestamp) -> Self::Output {
        Duration::microseconds(self.0 - rhs.0)
    }
}

impl Add<Duration> for Timestamp {
    type Output = Self;

    fn add(self, rhs: Duration) -> Self::Output {
        let micros = rhs.num_microseconds().unwrap_or(0);
        Self(self.0.saturating_add(micros))
    }
}

impl Sub<Duration> for Timestamp {
    type Output = Self;

    fn sub(self, rhs: Duration) -> Self::Output {
        let micros = rhs.num_microseconds().unwrap_or(0);
        Self(self.0.saturating_sub(micros))
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
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_conversions() {
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
    fn test_timestamp_chrono_interop() {
        let dt =
            DateTime::from_timestamp_micros(1_600_000_000_000_000).unwrap();
        let ts = Timestamp::from_datetime(dt);

        assert_eq!(ts.to_datetime(), Some(dt));

        let duration = Duration::seconds(5);
        let ts_future = ts + duration;
        assert_eq!((ts_future - ts).num_seconds(), 5);
    }

    #[test]
    fn test_candidate_deduplication() {
        let obs = Observation::new(
            ObservationId(1),
            Modality(1),
            Timestamp::from_secs(100),
            Confidence::new(0.9),
            (),
        );

        let evidence = Evidence::new(
            obs,
            alloc::vec![IdentityId(2), IdentityId(1), IdentityId(2)],
        );
        assert_eq!(
            evidence.candidates,
            alloc::vec![IdentityId(1), IdentityId(2)]
        );
    }
}
