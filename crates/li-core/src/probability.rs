//! Mathematical primitives for probabilistic reasoning and scoring metrics.

/// A continuous probability value.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Probability(pub f64);

impl Probability {
    /// Instantiates a new probability value.
    /// Clamps input into the valid interval [0.0, 1.0].
    pub fn new(value: f64) -> Self {
        if value < 0.0 {
            Self(0.0)
        } else if value > 1.0 {
            Self(1.0)
        } else {
            Self(value)
        }
    }
}

/// An unnormalized confidence score representing measurement or prediction
/// certainty.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Confidence(pub f64);
