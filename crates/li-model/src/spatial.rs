//! Spatial models and fixed-size geometric primitives for location-aware
//! identity inference.

use thiserror::Error;

const EARTH_RADIUS_METERS: f64 = 6_371_000.0;

/// Validation errors produced by spatial model constructors and operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SpatialError {
    /// Latitude is NaN or infinite.
    #[error("Latitude must be finite")]
    NonFiniteLatitude,
    /// Latitude is outside the WGS84 range.
    #[error("Latitude must be within [-90, 90] degrees")]
    LatitudeOutOfRange,
    /// Longitude is NaN or infinite.
    #[error("Longitude must be finite")]
    NonFiniteLongitude,
    /// Longitude is outside the WGS84 range.
    #[error("Longitude must be within [-180, 180] degrees")]
    LongitudeOutOfRange,
    /// Altitude is NaN or infinite.
    #[error("Altitude must be finite when provided")]
    NonFiniteAltitude,
    /// Positional accuracy is NaN or infinite.
    #[error("Positional accuracy must be finite")]
    NonFiniteAccuracy,
    /// Positional accuracy is negative.
    #[error("Positional accuracy must be non-negative")]
    NegativeAccuracy,
    /// A Cartesian mean contains a NaN or infinite coordinate.
    #[error("Spatial mean coordinates must be finite")]
    NonFiniteMean,
    /// A covariance entry is NaN or infinite.
    #[error("Covariance entries must be finite")]
    NonFiniteCovariance,
    /// A covariance matrix is not positive semidefinite.
    #[error("Covariance matrix must be positive semidefinite")]
    InvalidCovariance,
}

/// A validated or saturating geographic coordinate in WGS84.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoPoint {
    /// Latitude in decimal degrees.
    pub latitude: f64,
    /// Longitude in decimal degrees.
    pub longitude: f64,
    /// Optional altitude above mean sea level in meters.
    pub altitude_meters: Option<f64>,
}

impl GeoPoint {
    /// Creates a point by saturating angular coordinates to WGS84 bounds.
    ///
    /// NaN angles become zero, infinite angles saturate to the corresponding
    /// bound, and a non-finite altitude is discarded. Use [`Self::try_new`]
    /// when invalid input must be reported.
    #[inline]
    pub fn new(
        latitude: f64,
        longitude: f64,
        altitude_meters: Option<f64>,
    ) -> Self {
        Self {
            latitude: saturate_angle(latitude, -90.0, 90.0),
            longitude: saturate_angle(longitude, -180.0, 180.0),
            altitude_meters: altitude_meters.filter(|value| value.is_finite()),
        }
    }

    /// Creates a point after validating all WGS84 coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialError`] when a coordinate is non-finite or outside its
    /// WGS84 range.
    pub fn try_new(
        latitude: f64,
        longitude: f64,
        altitude_meters: Option<f64>,
    ) -> Result<Self, SpatialError> {
        let point = Self {
            latitude,
            longitude,
            altitude_meters,
        };
        point.validate()?;
        Ok(point)
    }

    /// Validates that the point contains finite, in-range WGS84 coordinates.
    pub fn validate(&self) -> Result<(), SpatialError> {
        if !self.latitude.is_finite() {
            return Err(SpatialError::NonFiniteLatitude);
        }
        if !(-90.0..=90.0).contains(&self.latitude) {
            return Err(SpatialError::LatitudeOutOfRange);
        }
        if !self.longitude.is_finite() {
            return Err(SpatialError::NonFiniteLongitude);
        }
        if !(-180.0..=180.0).contains(&self.longitude) {
            return Err(SpatialError::LongitudeOutOfRange);
        }
        if self
            .altitude_meters
            .is_some_and(|altitude| !altitude.is_finite())
        {
            return Err(SpatialError::NonFiniteAltitude);
        }
        Ok(())
    }

    /// Computes great-circle surface distance using the Haversine formula.
    ///
    /// Returns NaN when either public point value is invalid. Call
    /// [`Self::try_haversine_distance`] to receive a diagnostic error.
    pub fn haversine_distance(&self, other: &Self) -> f64 {
        self.try_haversine_distance(other).unwrap_or(f64::NAN)
    }

    /// Computes great-circle surface distance after validating both points.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialError`] if either point contains an invalid
    /// coordinate.
    pub fn try_haversine_distance(
        &self,
        other: &Self,
    ) -> Result<f64, SpatialError> {
        self.validate()?;
        other.validate()?;

        let latitude = self.latitude.to_radians();
        let other_latitude = other.latitude.to_radians();
        let latitude_delta = (other.latitude - self.latitude).to_radians();
        let longitude_delta = (other.longitude - self.longitude).to_radians();

        let half_latitude_sine = (latitude_delta * 0.5).sin();
        let half_longitude_sine = (longitude_delta * 0.5).sin();
        let haversine = (half_latitude_sine * half_latitude_sine +
            latitude.cos() *
                other_latitude.cos() *
                half_longitude_sine *
                half_longitude_sine)
            .clamp(0.0, 1.0);
        let central_angle =
            2.0 * haversine.sqrt().atan2((1.0 - haversine).max(0.0).sqrt());

        Ok(EARTH_RADIUS_METERS * central_angle)
    }
}

/// A geographic observation with isotropic positional uncertainty.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialComponent {
    /// Geographical position using WGS84 coordinates.
    pub position: GeoPoint,
    /// Positional standard deviation in meters.
    pub accuracy_meters: f64,
}

impl SpatialComponent {
    /// Creates a spatial component with a finite, saturating accuracy.
    ///
    /// Negative and NaN accuracy values become zero. Positive infinity and
    /// excessively large values saturate below the square-overflow boundary.
    /// Use [`Self::try_new`] when invalid input must be reported.
    #[inline]
    pub fn new(position: GeoPoint, accuracy_meters: f64) -> Self {
        let maximum_accuracy = f64::MAX.sqrt() * 0.5;
        let accuracy_meters = if accuracy_meters.is_nan() ||
            accuracy_meters.is_sign_negative()
        {
            0.0
        } else {
            accuracy_meters.min(maximum_accuracy)
        };
        Self {
            position,
            accuracy_meters,
        }
    }

    /// Creates a spatial component after validating its point and accuracy.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialError`] for an invalid point, non-finite accuracy, or
    /// negative accuracy.
    pub fn try_new(
        position: GeoPoint,
        accuracy_meters: f64,
    ) -> Result<Self, SpatialError> {
        position.validate()?;
        if !accuracy_meters.is_finite() {
            return Err(SpatialError::NonFiniteAccuracy);
        }
        if accuracy_meters < 0.0 {
            return Err(SpatialError::NegativeAccuracy);
        }
        Ok(Self {
            position,
            accuracy_meters,
        })
    }

    /// Validates the spatial point and uncertainty.
    pub fn validate(&self) -> Result<(), SpatialError> {
        self.position.validate()?;
        if !self.accuracy_meters.is_finite() {
            return Err(SpatialError::NonFiniteAccuracy);
        }
        if self.accuracy_meters < 0.0 {
            return Err(SpatialError::NegativeAccuracy);
        }
        Ok(())
    }

    /// Evaluates a Gaussian compatibility log-likelihood.
    ///
    /// Invalid public field values produce negative infinity so malformed
    /// evidence can never become a preferred identity assignment.
    pub fn evaluate_log_likelihood(&self, target: &Self) -> f64 {
        self.try_evaluate_log_likelihood(target)
            .unwrap_or(f64::NEG_INFINITY)
    }

    /// Evaluates a Gaussian compatibility log-likelihood with diagnostics.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialError`] if either component contains invalid values.
    pub fn try_evaluate_log_likelihood(
        &self,
        target: &Self,
    ) -> Result<f64, SpatialError> {
        self.validate()?;
        target.validate()?;
        let distance =
            self.position.try_haversine_distance(&target.position)?;
        let combined_deviation =
            self.accuracy_meters.hypot(target.accuracy_meters);

        if combined_deviation <= f64::EPSILON {
            return Ok(if distance <= f64::EPSILON {
                0.0
            } else {
                f64::NEG_INFINITY
            });
        }

        let standardized_distance = distance / combined_deviation;
        if standardized_distance.is_finite() {
            Ok(-0.5 * standardized_distance * standardized_distance)
        } else {
            Ok(f64::NEG_INFINITY)
        }
    }
}

/// A symmetric fixed-size two-dimensional covariance matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Covariance2D {
    xx: f64,
    xy: f64,
    yy: f64,
}

impl Covariance2D {
    /// Creates a positive-semidefinite covariance matrix.
    ///
    /// # Arguments
    ///
    /// * `xx` - Variance along the x-axis.
    /// * `xy` - Symmetric x/y covariance.
    /// * `yy` - Variance along the y-axis.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialError`] for non-finite entries or a matrix that is not
    /// positive semidefinite.
    pub fn try_new(xx: f64, xy: f64, yy: f64) -> Result<Self, SpatialError> {
        if !xx.is_finite() || !xy.is_finite() || !yy.is_finite() {
            return Err(SpatialError::NonFiniteCovariance);
        }
        if xx < 0.0 || yy < 0.0 {
            return Err(SpatialError::InvalidCovariance);
        }

        let correlation = if xx == 0.0 || yy == 0.0 {
            if xy == 0.0 {
                0.0
            } else {
                return Err(SpatialError::InvalidCovariance);
            }
        } else {
            (xy / xx.sqrt()) / yy.sqrt()
        };
        if !correlation.is_finite() || correlation.abs() > 1.0 {
            return Err(SpatialError::InvalidCovariance);
        }

        Ok(Self { xx, xy, yy })
    }

    /// Creates an isotropic covariance with equal x/y variance.
    pub fn try_isotropic(variance: f64) -> Result<Self, SpatialError> {
        Self::try_new(variance, 0.0, variance)
    }

    /// Returns variance along the x-axis.
    #[inline]
    pub fn xx(&self) -> f64 {
        self.xx
    }

    /// Returns the symmetric x/y covariance.
    #[inline]
    pub fn xy(&self) -> f64 {
        self.xy
    }

    /// Returns variance along the y-axis.
    #[inline]
    pub fn yy(&self) -> f64 {
        self.yy
    }
}

/// A fixed-size Gaussian spatial summary in a local Cartesian frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialGaussian2D {
    mean_meters: [f64; 2],
    covariance: Covariance2D,
}

impl SpatialGaussian2D {
    /// Creates a Gaussian spatial summary without heap allocation.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialError::NonFiniteMean`] when either mean coordinate is
    /// NaN or infinite.
    pub fn try_new(
        mean_meters: [f64; 2],
        covariance: Covariance2D,
    ) -> Result<Self, SpatialError> {
        if !mean_meters.iter().all(|coordinate| coordinate.is_finite()) {
            return Err(SpatialError::NonFiniteMean);
        }
        Ok(Self {
            mean_meters,
            covariance,
        })
    }

    /// Returns the local Cartesian mean in meters.
    #[inline]
    pub fn mean_meters(&self) -> [f64; 2] {
        self.mean_meters
    }

    /// Returns the spatial covariance.
    #[inline]
    pub fn covariance(&self) -> Covariance2D {
        self.covariance
    }

    /// Evaluates covariance-aware Gaussian compatibility.
    ///
    /// Invalid combined covariance values produce negative infinity.
    pub fn evaluate_log_likelihood(&self, target: &Self) -> f64 {
        self.try_evaluate_log_likelihood(target)
            .unwrap_or(f64::NEG_INFINITY)
    }

    /// Evaluates covariance-aware Gaussian compatibility with diagnostics.
    ///
    /// The calculation uses a scaled analytic inverse of the combined 2x2
    /// covariance and performs no heap allocation.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialError`] when adding the covariances produces
    /// non-finite values or a non-positive-semidefinite matrix.
    pub fn try_evaluate_log_likelihood(
        &self,
        target: &Self,
    ) -> Result<f64, SpatialError> {
        let xx = self.covariance.xx + target.covariance.xx;
        let xy = self.covariance.xy + target.covariance.xy;
        let yy = self.covariance.yy + target.covariance.yy;
        if !xx.is_finite() || !xy.is_finite() || !yy.is_finite() {
            return Err(SpatialError::NonFiniteCovariance);
        }

        let delta_x = self.mean_meters[0] - target.mean_meters[0];
        let delta_y = self.mean_meters[1] - target.mean_meters[1];
        if !delta_x.is_finite() || !delta_y.is_finite() {
            return Ok(f64::NEG_INFINITY);
        }
        if delta_x == 0.0 && delta_y == 0.0 {
            return Ok(0.0);
        }

        let covariance_scale = xx.abs().max(xy.abs()).max(yy.abs());
        if covariance_scale == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        let normalized_xx = xx / covariance_scale;
        let normalized_xy = xy / covariance_scale;
        let normalized_yy = yy / covariance_scale;
        let determinant =
            normalized_xx * normalized_yy - normalized_xy * normalized_xy;
        if determinant < -64.0 * f64::EPSILON {
            return Err(SpatialError::InvalidCovariance);
        }
        if determinant <= 0.0 {
            return Ok(f64::NEG_INFINITY);
        }

        let delta_scale = delta_x.abs().max(delta_y.abs());
        let normalized_delta_x = delta_x / delta_scale;
        let normalized_delta_y = delta_y / delta_scale;
        let normalized_quadratic = normalized_yy *
            normalized_delta_x *
            normalized_delta_x -
            2.0 * normalized_xy * normalized_delta_x * normalized_delta_y +
            normalized_xx * normalized_delta_y * normalized_delta_y;
        if normalized_quadratic < -64.0 * f64::EPSILON {
            return Err(SpatialError::InvalidCovariance);
        }

        let scaled_delta = delta_scale / covariance_scale.sqrt();
        let quadratic =
            scaled_delta * scaled_delta * normalized_quadratic.max(0.0) /
                determinant;
        if quadratic.is_finite() {
            Ok(-0.5 * quadratic)
        } else {
            Ok(f64::NEG_INFINITY)
        }
    }
}

#[inline]
fn saturate_angle(value: f64, minimum: f64, maximum: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(minimum, maximum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saturating_point_constructor_bounds_coordinates() {
        let upper = GeoPoint::new(105.0, 210.0, Some(10.0));
        assert_eq!(upper.latitude, 90.0);
        assert_eq!(upper.longitude, 180.0);

        let lower = GeoPoint::new(-105.0, -210.0, None);
        assert_eq!(lower.latitude, -90.0);
        assert_eq!(lower.longitude, -180.0);
    }

    #[test]
    fn saturating_point_constructor_sanitizes_non_finite_values() {
        let point = GeoPoint::new(f64::NAN, f64::INFINITY, Some(f64::NAN));

        assert_eq!(point.latitude, 0.0);
        assert_eq!(point.longitude, 180.0);
        assert_eq!(point.altitude_meters, None);
    }

    #[test]
    fn fallible_point_constructor_rejects_invalid_values() {
        assert_eq!(
            GeoPoint::try_new(f64::NAN, 0.0, None),
            Err(SpatialError::NonFiniteLatitude)
        );
        assert_eq!(
            GeoPoint::try_new(0.0, 181.0, None),
            Err(SpatialError::LongitudeOutOfRange)
        );
        assert_eq!(
            GeoPoint::try_new(0.0, 0.0, Some(f64::INFINITY)),
            Err(SpatialError::NonFiniteAltitude)
        );
    }

    #[test]
    fn haversine_distance_is_zero_for_identical_points() {
        let point = GeoPoint::new(48.8566, 2.3522, None);

        assert!(point.haversine_distance(&point) < f64::EPSILON);
    }

    #[test]
    fn haversine_distance_matches_known_coordinates() {
        let paris = GeoPoint::new(48.8566, 2.3522, None);
        let london = GeoPoint::new(51.5074, -0.1278, None);
        let distance = paris.haversine_distance(&london);

        assert!((distance - 343_556.0).abs() < 1_000.0);
    }

    #[test]
    fn haversine_distance_handles_antipodal_rounding() {
        let origin = GeoPoint::new(0.0, 0.0, None);
        let antipode = GeoPoint::new(0.0, 180.0, None);
        let near_antipode = GeoPoint::new(0.000_001, 179.999_999, None);
        let expected = std::f64::consts::PI * EARTH_RADIUS_METERS;

        let antipodal_distance = origin.haversine_distance(&antipode);
        let near_antipodal_distance =
            origin.haversine_distance(&near_antipode);
        assert!((antipodal_distance - expected).abs() < 1.0);
        assert!(near_antipodal_distance.is_finite());
        assert!(near_antipodal_distance <= expected);
    }

    #[test]
    fn haversine_distance_reports_public_nan_field() {
        let invalid = GeoPoint {
            latitude: f64::NAN,
            longitude: 0.0,
            altitude_meters: None,
        };
        let valid = GeoPoint::new(0.0, 0.0, None);

        assert_eq!(
            invalid.try_haversine_distance(&valid),
            Err(SpatialError::NonFiniteLatitude)
        );
        assert!(invalid.haversine_distance(&valid).is_nan());
    }

    #[test]
    fn spatial_component_supports_saturating_and_fallible_construction() {
        let point = GeoPoint::new(0.0, 0.0, None);
        assert_eq!(SpatialComponent::new(point, -15.0).accuracy_meters, 0.0);
        assert_eq!(
            SpatialComponent::try_new(point, -1.0),
            Err(SpatialError::NegativeAccuracy)
        );
        assert_eq!(
            SpatialComponent::try_new(point, f64::INFINITY),
            Err(SpatialError::NonFiniteAccuracy)
        );
    }

    #[test]
    fn isotropic_likelihood_handles_zero_variance() {
        let first = GeoPoint::new(48.8566, 2.3522, None);
        let second = GeoPoint::new(48.8567, 2.3523, None);
        let first_component = SpatialComponent::new(first, 0.0);
        let same_component = SpatialComponent::new(first, 0.0);
        let second_component = SpatialComponent::new(second, 0.0);

        assert_eq!(
            first_component.evaluate_log_likelihood(&same_component),
            0.0
        );
        assert_eq!(
            first_component.evaluate_log_likelihood(&second_component),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn isotropic_likelihood_decays_with_distance() {
        let first =
            SpatialComponent::new(GeoPoint::new(0.0, 0.0, None), 100.0);
        let second =
            SpatialComponent::new(GeoPoint::new(0.0, 0.001, None), 100.0);
        let likelihood = first.evaluate_log_likelihood(&second);

        assert!(likelihood < 0.0);
        assert!(likelihood.is_finite());
    }

    #[test]
    fn covariance_constructor_rejects_invalid_matrices() {
        assert_eq!(
            Covariance2D::try_new(-1.0, 0.0, 1.0),
            Err(SpatialError::InvalidCovariance)
        );
        assert_eq!(
            Covariance2D::try_new(1.0, 2.0, 1.0),
            Err(SpatialError::InvalidCovariance)
        );
        assert_eq!(
            Covariance2D::try_new(1.0, f64::NAN, 1.0),
            Err(SpatialError::NonFiniteCovariance)
        );
    }

    #[test]
    fn covariance_likelihood_matches_mahalanobis_distance()
    -> Result<(), SpatialError> {
        let covariance = Covariance2D::try_isotropic(4.0)?;
        let first = SpatialGaussian2D::try_new([0.0, 0.0], covariance)?;
        let second = SpatialGaussian2D::try_new([3.0, 4.0], covariance)?;
        let likelihood = first.try_evaluate_log_likelihood(&second)?;

        assert!((likelihood - -1.5625).abs() < 1.0e-12);
        Ok(())
    }

    #[test]
    fn covariance_likelihood_preserves_highly_anisotropic_axis()
    -> Result<(), SpatialError> {
        let covariance = Covariance2D::try_new(1.0, 0.0, 1.0e-16)?;
        let first = SpatialGaussian2D::try_new([0.0, 0.0], covariance)?;
        let second = SpatialGaussian2D::try_new([0.0, 1.0e-8], covariance)?;
        let likelihood = first.try_evaluate_log_likelihood(&second)?;

        assert!((likelihood - -0.25).abs() < 1.0e-12);
        Ok(())
    }

    #[test]
    fn singular_covariance_only_accepts_identical_means()
    -> Result<(), SpatialError> {
        let covariance = Covariance2D::try_isotropic(0.0)?;
        let origin = SpatialGaussian2D::try_new([0.0, 0.0], covariance)?;
        let same = SpatialGaussian2D::try_new([0.0, 0.0], covariance)?;
        let displaced = SpatialGaussian2D::try_new([1.0, 0.0], covariance)?;

        assert_eq!(origin.evaluate_log_likelihood(&same), 0.0);
        assert_eq!(
            origin.evaluate_log_likelihood(&displaced),
            f64::NEG_INFINITY
        );
        Ok(())
    }

    #[test]
    fn gaussian_constructor_rejects_non_finite_mean()
    -> Result<(), SpatialError> {
        let covariance = Covariance2D::try_isotropic(1.0)?;

        assert_eq!(
            SpatialGaussian2D::try_new([f64::NAN, 0.0], covariance),
            Err(SpatialError::NonFiniteMean)
        );
        Ok(())
    }
}
