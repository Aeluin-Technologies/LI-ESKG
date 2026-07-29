//! Mathematical primitives for probabilistic reasoning and scoring metrics.

use std::error::Error;
use std::fmt;

use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer, Serialize};

/// Validation error returned when constructing a probability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbabilityError {
    /// The value is NaN or infinite.
    NonFinite,
    /// The value lies outside the inclusive probability interval.
    OutOfRange,
}

impl fmt::Display for ProbabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => {
                formatter.write_str("probability must be finite")
            },
            Self::OutOfRange => {
                formatter.write_str("probability must be within [0.0, 1.0]")
            },
        }
    }
}

impl Error for ProbabilityError {}

/// Validation error returned when constructing a confidence score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceError {
    /// The value is NaN or infinite.
    NonFinite,
    /// The value is negative.
    Negative,
}

impl fmt::Display for ConfidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => {
                formatter.write_str("confidence must be finite")
            },
            Self::Negative => {
                formatter.write_str("confidence must be non-negative")
            },
        }
    }
}

impl Error for ConfidenceError {}

/// A continuous probability value within `[0.0, 1.0]`.
#[derive(Serialize, Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Probability(f64);

impl Probability {
    /// Absolute certainty probability constant.
    pub const ONE: Self = Self(1.0);
    /// Absolute zero probability constant.
    pub const ZERO: Self = Self(0.0);

    /// Instantiates a probability by saturating the input to `[0.0, 1.0]`.
    ///
    /// NaN and negative infinity map to zero; positive infinity maps to one.
    /// Use [`Self::try_new`] when invalid input must be reported.
    ///
    /// # Arguments
    ///
    /// * `value` - Continuous floating point number. If `value` is `NaN`, it
    ///   defaults to `0.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use li_core::Probability;
    ///
    /// let p = Probability::new(0.85);
    /// assert_eq!(p.value(), 0.85);
    /// ```
    pub fn new(value: f64) -> Self {
        if value.is_nan() || value <= 0.0 {
            Self::ZERO
        } else if value >= 1.0 {
            Self::ONE
        } else {
            Self(value)
        }
    }

    /// Constructs a probability after validating that the input is finite and
    /// within `[0.0, 1.0]`.
    ///
    /// # Arguments
    ///
    /// * `value` - Candidate probability value.
    ///
    /// # Errors
    ///
    /// Returns [`ProbabilityError`] when `value` is non-finite or out of
    /// range.
    pub fn try_new(value: f64) -> Result<Self, ProbabilityError> {
        if !value.is_finite() {
            Err(ProbabilityError::NonFinite)
        } else if !(0.0..=1.0).contains(&value) {
            Err(ProbabilityError::OutOfRange)
        } else if value == 0.0 {
            Ok(Self::ZERO)
        } else if value == 1.0 {
            Ok(Self::ONE)
        } else {
            Ok(Self(value))
        }
    }

    /// Returns the underlying raw `f64` scalar value.
    pub const fn value(&self) -> f64 {
        self.0
    }

    /// Converts the probability score into log-domain representation.
    /// Returns `-f64::INFINITY` if probability is zero.
    pub fn to_log(&self) -> f64 {
        if self.0 == 0.0 {
            f64::NEG_INFINITY
        } else {
            self.0.ln()
        }
    }

    /// Instantiates a saturated probability from a log-domain scalar.
    ///
    /// NaN and negative infinity map to zero. Positive log values, including
    /// positive infinity, map to one. Use [`Self::try_from_log`] to report an
    /// invalid log probability.
    ///
    /// # Arguments
    ///
    /// * `log_value` - Logarithmic probability scalar value.
    pub fn from_log(log_value: f64) -> Self {
        if log_value.is_nan() || log_value == f64::NEG_INFINITY {
            Self::ZERO
        } else if log_value >= 0.0 {
            Self::ONE
        } else {
            Self::new(log_value.exp())
        }
    }

    /// Constructs a probability from a valid log-domain value.
    ///
    /// # Arguments
    ///
    /// * `log_value` - Log probability in `[-∞, 0.0]`.
    ///
    /// # Errors
    ///
    /// Returns [`ProbabilityError`] for NaN, positive infinity, or a positive
    /// log value.
    pub fn try_from_log(log_value: f64) -> Result<Self, ProbabilityError> {
        if log_value == f64::NEG_INFINITY {
            Ok(Self::ZERO)
        } else if !log_value.is_finite() {
            Err(ProbabilityError::NonFinite)
        } else if log_value > 0.0 {
            Err(ProbabilityError::OutOfRange)
        } else {
            Self::try_new(log_value.exp())
        }
    }
}

impl<'de> Deserialize<'de> for Probability {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = f64::deserialize(deserializer)?;
        Self::try_new(value).map_err(D::Error::custom)
    }
}

impl Default for Probability {
    fn default() -> Self {
        Self::ZERO
    }
}

/// An unnormalized non-negative confidence score representing sensor or
/// pipeline measurement certainty.
#[derive(Serialize, Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Confidence(f64);

impl Confidence {
    /// Zero confidence constant.
    pub const ZERO: Self = Self(0.0);

    /// Instantiates a saturated non-negative confidence score.
    ///
    /// Negative and NaN values map to zero. Positive infinity maps to
    /// `f64::MAX`. Use [`Self::try_new`] when invalid input must be reported.
    ///
    /// # Arguments
    ///
    /// * `value` - Raw confidence score value.
    pub fn new(value: f64) -> Self {
        if value.is_nan() || value <= 0.0 {
            Self::ZERO
        } else if value == f64::INFINITY {
            Self(f64::MAX)
        } else {
            Self(value)
        }
    }

    /// Constructs a confidence score after validating it is finite and
    /// non-negative.
    ///
    /// # Arguments
    ///
    /// * `value` - Candidate confidence score.
    ///
    /// # Errors
    ///
    /// Returns [`ConfidenceError`] when `value` is non-finite or negative.
    pub fn try_new(value: f64) -> Result<Self, ConfidenceError> {
        if !value.is_finite() {
            Err(ConfidenceError::NonFinite)
        } else if value < 0.0 {
            Err(ConfidenceError::Negative)
        } else if value == 0.0 {
            Ok(Self::ZERO)
        } else {
            Ok(Self(value))
        }
    }

    /// Returns the underlying raw `f64` confidence score.
    pub const fn value(&self) -> f64 {
        self.0
    }
}

impl<'de> Deserialize<'de> for Confidence {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = f64::deserialize(deserializer)?;
        Self::try_new(value).map_err(D::Error::custom)
    }
}

impl Default for Confidence {
    fn default() -> Self {
        Self::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probability_saturates_invalid_inputs() {
        assert_eq!(Probability::new(1.5).value(), 1.0);
        assert_eq!(Probability::new(-0.5).value(), 0.0);
        assert_eq!(Probability::new(f64::NAN).value(), 0.0);
        assert_eq!(Probability::new(f64::INFINITY), Probability::ONE);
        assert_eq!(Probability::new(f64::NEG_INFINITY), Probability::ZERO);
        assert_eq!(Probability::new(0.42).value(), 0.42);
    }

    #[test]
    fn probability_fallible_construction_rejects_invalid_values() {
        assert_eq!(
            Probability::try_new(f64::NAN),
            Err(ProbabilityError::NonFinite)
        );
        assert_eq!(
            Probability::try_new(f64::INFINITY),
            Err(ProbabilityError::NonFinite)
        );
        assert_eq!(
            Probability::try_new(-f64::EPSILON),
            Err(ProbabilityError::OutOfRange)
        );
        assert_eq!(
            Probability::try_new(1.0 + f64::EPSILON),
            Err(ProbabilityError::OutOfRange)
        );
        assert_eq!(Probability::try_new(0.0), Ok(Probability::ZERO));
        assert_eq!(Probability::try_new(1.0), Ok(Probability::ONE));
    }

    #[test]
    fn log_conversion_preserves_boundaries_and_round_trips() {
        let p = Probability::new(0.5);
        let log_p = p.to_log();
        let recovered = Probability::from_log(log_p);

        assert!((p.value() - recovered.value()).abs() < 1e-10);
        assert_eq!(Probability::ZERO.to_log(), f64::NEG_INFINITY);
        assert_eq!(
            Probability::try_from_log(f64::NEG_INFINITY),
            Ok(Probability::ZERO)
        );
        assert_eq!(Probability::try_from_log(0.0), Ok(Probability::ONE));
        assert_eq!(
            Probability::try_from_log(f64::INFINITY),
            Err(ProbabilityError::NonFinite)
        );
        assert_eq!(
            Probability::try_from_log(f64::NAN),
            Err(ProbabilityError::NonFinite)
        );
        assert_eq!(
            Probability::try_from_log(f64::EPSILON),
            Err(ProbabilityError::OutOfRange)
        );
    }

    #[test]
    fn confidence_construction_is_finite_and_saturating() {
        assert_eq!(Confidence::new(-10.0).value(), 0.0);
        assert_eq!(Confidence::new(f64::NAN).value(), 0.0);
        assert_eq!(Confidence::new(f64::INFINITY).value(), f64::MAX);
        assert_eq!(Confidence::new(12.5).value(), 12.5);
        assert_eq!(Confidence::try_new(-1.0), Err(ConfidenceError::Negative));
        assert_eq!(
            Confidence::try_new(f64::INFINITY),
            Err(ConfidenceError::NonFinite)
        );
        assert_eq!(Confidence::try_new(12.5), Ok(Confidence::new(12.5)));
    }

    #[test]
    fn serde_deserialization_rejects_invalid_scalar_representations() {
        use serde::de::value::{Error, F64Deserializer};

        let invalid_probability = serde_json::from_str::<Probability>("1.1");
        let negative_probability = serde_json::from_str::<Probability>("-0.1");
        let invalid_confidence = serde_json::from_str::<Confidence>("-1.0");
        let valid_probability = serde_json::from_str::<Probability>("0.75");
        let nan_probability =
            Probability::deserialize(F64Deserializer::<Error>::new(f64::NAN));
        let infinite_confidence =
            Confidence::deserialize(F64Deserializer::<Error>::new(
                f64::INFINITY,
            ));

        assert!(invalid_probability.is_err());
        assert!(negative_probability.is_err());
        assert!(invalid_confidence.is_err());
        assert!(nan_probability.is_err());
        assert!(infinite_confidence.is_err());
        assert!(valid_probability.is_ok());
        if let Ok(valid_probability) = valid_probability {
            assert_eq!(valid_probability, Probability::new(0.75));
        }
    }
}
