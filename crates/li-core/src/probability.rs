//! Mathematical primitives for probabilistic reasoning and scoring metrics.

use serde::{Deserialize, Serialize};

/// A continuous probability value within `[0.0, 1.0]`.
#[derive(
    Serialize, Deserialize, Debug, Clone, Copy, PartialEq, PartialOrd,
)]
pub struct Probability(f64);

impl Probability {
    /// Absolute certainty probability constant.
    pub const ONE: Self = Self(1.0);
    /// Absolute zero probability constant.
    pub const ZERO: Self = Self(0.0);

    /// Instantiates a new probability value.
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
        if value.is_nan() || value < 0.0 {
            Self::ZERO
        } else if value > 1.0 {
            Self::ONE
        } else {
            Self(value)
        }
    }

    /// Returns the underlying raw `f64` scalar value.
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Converts the probability score into log-domain representation.
    /// Returns `-f64::INFINITY` if probability is zero.
    pub fn to_log(&self) -> f64 {
        if self.0 == 0.0 {
            f64::NEG_INFINITY
        } else {
            libm::log(self.0)
        }
    }

    /// Instantiates a probability value from a log-domain scalar.
    ///
    /// # Arguments
    ///
    /// * `log_value` - Logarithmic probability scalar value.
    pub fn from_log(log_value: f64) -> Self {
        if log_value.is_nan() || log_value == f64::NEG_INFINITY {
            Self::ZERO
        } else {
            Self::new(libm::exp(log_value))
        }
    }
}

impl Default for Probability {
    fn default() -> Self {
        Self::ZERO
    }
}

/// An unnormalized non-negative confidence score representing sensor or
/// pipeline measurement certainty.
#[derive(
    Serialize, Deserialize, Debug, Clone, Copy, PartialEq, PartialOrd,
)]
pub struct Confidence(f64);

impl Confidence {
    /// Zero confidence constant.
    pub const ZERO: Self = Self(0.0);

    /// Instantiates a new confidence score.
    /// Clamps input to `0.0` if negative or `NaN`.
    ///
    /// # Arguments
    ///
    /// * `value` - Raw confidence score value.
    pub fn new(value: f64) -> Self {
        if value.is_nan() || value < 0.0 {
            Self::ZERO
        } else {
            Self(value)
        }
    }

    /// Returns the underlying raw `f64` confidence score.
    pub fn value(&self) -> f64 {
        self.0
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
    fn test_probability_bounds_and_nan() {
        assert_eq!(Probability::new(1.5).value(), 1.0);
        assert_eq!(Probability::new(-0.5).value(), 0.0);
        assert_eq!(Probability::new(f64::NAN).value(), 0.0);
        assert_eq!(Probability::new(0.42).value(), 0.42);
    }

    #[test]
    fn test_log_conversion() {
        let p = Probability::new(0.5);
        let log_p = p.to_log();
        let recovered = Probability::from_log(log_p);
        assert!((p.value() - recovered.value()).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_validation() {
        assert_eq!(Confidence::new(-10.0).value(), 0.0);
        assert_eq!(Confidence::new(f64::NAN).value(), 0.0);
        assert_eq!(Confidence::new(12.5).value(), 12.5);
    }
}
