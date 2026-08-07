//! Time and modality primitives shared by immutable evidence envelopes.

use std::error::Error;
use std::fmt;
use std::ops::{Add, Sub};

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

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
            "timestamp {}us is outside chrono's supported range",
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
    /// UNIX epoch timestamp.
    pub const UNIX_EPOCH: Self = Self(0);

    /// Constructs a timestamp from microseconds since the UNIX epoch.
    pub const fn from_micros(micros: i64) -> Self {
        Self(micros)
    }

    /// Constructs a timestamp from milliseconds with saturating conversion.
    pub fn from_millis(millis: i64) -> Self {
        Self(millis.saturating_mul(1_000))
    }

    /// Constructs a timestamp from seconds with saturating conversion.
    pub fn from_secs(seconds: i64) -> Self {
        Self(seconds.saturating_mul(1_000_000))
    }

    /// Constructs a timestamp from a UTC datetime.
    pub fn from_datetime(datetime: DateTime<Utc>) -> Self {
        Self(datetime.timestamp_micros())
    }

    /// Returns microseconds since the UNIX epoch.
    pub const fn as_micros(self) -> i64 {
        self.0
    }

    /// Returns floor milliseconds since the UNIX epoch.
    pub fn as_millis(self) -> i64 {
        self.0.div_euclid(1_000)
    }

    /// Returns floor seconds since the UNIX epoch.
    pub fn as_secs(self) -> i64 {
        self.0.div_euclid(1_000_000)
    }

    /// Returns continuous seconds since the UNIX epoch.
    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 / 1_000_000.0
    }

    /// Converts into `chrono`, returning `None` outside its supported range.
    pub fn to_datetime(self) -> Option<DateTime<Utc>> {
        DateTime::from_timestamp_micros(self.0)
    }

    /// Returns non-negative elapsed time since `earlier`.
    pub fn duration_since(self, earlier: Self) -> Duration {
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
    fn from(datetime: DateTime<Utc>) -> Self {
        Self::from_datetime(datetime)
    }
}

impl TryFrom<Timestamp> for DateTime<Utc> {
    type Error = TimestampRangeError;

    fn try_from(timestamp: Timestamp) -> Result<Self, Self::Error> {
        timestamp.to_datetime().ok_or(TimestampRangeError {
            micros: timestamp.0,
        })
    }
}

impl Sub<Timestamp> for Timestamp {
    type Output = Duration;

    fn sub(self, other: Timestamp) -> Self::Output {
        Duration::microseconds(self.0.saturating_sub(other.0))
    }
}

impl Add<Duration> for Timestamp {
    type Output = Self;

    fn add(self, duration: Duration) -> Self::Output {
        match duration.num_microseconds() {
            Some(micros) => Self(self.0.saturating_add(micros)),
            None if duration < Duration::zero() => Self::MIN,
            None => Self::MAX,
        }
    }
}

impl Sub<Duration> for Timestamp {
    type Output = Self;

    fn sub(self, duration: Duration) -> Self::Output {
        match duration.num_microseconds() {
            Some(micros) => Self(self.0.saturating_sub(micros)),
            None if duration < Duration::zero() => Self::MAX,
            None => Self::MIN,
        }
    }
}

/// Stable identifier for an observation modality or channel.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_epoch_conversions_use_euclidean_division() {
        let timestamp = Timestamp::from_micros(-1);
        assert_eq!(timestamp.as_millis(), -1);
        assert_eq!(timestamp.as_secs(), -1);
    }

    #[test]
    fn arithmetic_saturates_at_numeric_extremes() {
        assert_eq!(Timestamp::MAX + Duration::microseconds(1), Timestamp::MAX);
        assert_eq!(Timestamp::MIN - Duration::microseconds(1), Timestamp::MIN);
        assert_eq!(
            Timestamp::UNIX_EPOCH.duration_since(Timestamp::from_micros(1)),
            Duration::zero()
        );
    }

    #[test]
    fn chrono_conversion_is_fallible_and_precise() {
        let timestamp = Timestamp::from_micros(1_234_567);
        let converted = DateTime::<Utc>::try_from(timestamp);
        assert!(converted.is_ok());
        if let Ok(converted) = converted {
            assert_eq!(Timestamp::from(converted), timestamp);
        }
        assert!(DateTime::<Utc>::try_from(Timestamp::MAX).is_err());
    }
}
