//! Spatial models and geometric primitives for geography-aware processing.

/// Represents a 3D or 2D geographical coordinate using the WGS84 reference
/// system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoPoint {
    /// Latitude in decimal degrees, clamped to `[-90.0, 90.0]`.
    pub latitude: f64,
    /// Longitude in decimal degrees, clamped to `[-180.0, 180.0]`.
    pub longitude: f64,
    /// Optional altitude above mean sea level in meters.
    pub altitude_meters: Option<f64>,
}

impl GeoPoint {
    /// Creates a new [`GeoPoint`] with optional altitude parameter.
    ///
    /// # Arguments
    ///
    /// * `latitude` - Latitude in decimal degrees.
    /// * `longitude` - Longitude in decimal degrees.
    /// * `altitude_meters` - Optional altitude in meters above sea level.
    ///
    /// # Examples
    ///
    /// ```
    /// use li_model::spatial::GeoPoint;
    ///
    /// let point = GeoPoint::new(48.8566, 2.3522, Some(35.0));
    /// assert_eq!(point.latitude, 48.8566);
    /// ```
    #[inline]
    pub fn new(
        latitude: f64,
        longitude: f64,
        altitude_meters: Option<f64>,
    ) -> Self {
        Self {
            latitude: latitude.clamp(-90.0, 90.0),
            longitude: longitude.clamp(-180.0, 180.0),
            altitude_meters,
        }
    }

    /// Computes the great-circle distance to another point using the Haversine
    /// formula.
    ///
    /// # Arguments
    ///
    /// * `other` - The target [`GeoPoint`] destination.
    ///
    /// # Returns
    ///
    /// The geodesic distance between the two points in meters.
    ///
    /// # Examples
    ///
    /// ```
    /// use li_model::spatial::GeoPoint;
    ///
    /// let paris = GeoPoint::new(48.8566, 2.3522, None);
    /// let london = GeoPoint::new(51.5074, -0.1278, None);
    /// let distance = paris.haversine_distance(&london);
    /// assert!((distance - 343556.0).abs() < 1000.0);
    /// ```
    pub fn haversine_distance(&self, other: &Self) -> f64 {
        const EARTH_RADIUS_METERS: f64 = 6_371_000.0;

        let lat1 = self.latitude.to_radians();
        let lat2 = other.latitude.to_radians();
        let delta_lat = (other.latitude - self.latitude).to_radians();
        let delta_lon = (other.longitude - self.longitude).to_radians();

        let a = (delta_lat / 2.0).sin().powi(2) +
            lat1.cos() * lat2.cos() * (delta_lon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

        EARTH_RADIUS_METERS * c
    }
}

/// Represents a spatial payload containing geographic positioning and
/// positional uncertainty.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialComponent {
    /// Geographical position using WGS84 coordinates.
    pub position: GeoPoint,
    /// Positional accuracy standard deviation in meters.
    pub accuracy_meters: f64,
}

impl SpatialComponent {
    /// Creates a new [`SpatialComponent`].
    ///
    /// # Arguments
    ///
    /// * `position` - Geographical location.
    /// * `accuracy_meters` - Standard deviation or accuracy radius in meters.
    ///
    /// # Examples
    ///
    /// ```
    /// use li_model::spatial::{GeoPoint, SpatialComponent};
    ///
    /// let location = GeoPoint::new(48.8566, 2.3522, None);
    /// let component = SpatialComponent::new(location, 5.0);
    /// assert_eq!(component.accuracy_meters, 5.0);
    /// ```
    #[inline]
    pub fn new(position: GeoPoint, accuracy_meters: f64) -> Self {
        Self {
            position,
            accuracy_meters: accuracy_meters.max(0.0),
        }
    }

    /// Evaluates the natural log-likelihood between two spatial components
    /// using a Gaussian kernel decay model.
    ///
    /// # Arguments
    ///
    /// * `target` - Target [`SpatialComponent`] to compare against.
    ///
    /// # Returns
    ///
    /// Natural logarithm of the likelihood in the range `(-\infty, 0]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use li_model::spatial::{GeoPoint, SpatialComponent};
    ///
    /// let obs = SpatialComponent::new(GeoPoint::new(48.8566, 2.3522, None), 5.0);
    /// let state =
    ///     SpatialComponent::new(GeoPoint::new(48.8567, 2.3523, None), 10.0);
    /// let log_likelihood = obs.evaluate_log_likelihood(&state);
    /// assert!(log_likelihood <= 0.0);
    /// ```
    pub fn evaluate_log_likelihood(&self, target: &Self) -> f64 {
        let distance = self.position.haversine_distance(&target.position);
        let combined_variance =
            self.accuracy_meters.powi(2) + target.accuracy_meters.powi(2);

        if combined_variance <= f64::EPSILON {
            if distance <= f64::EPSILON {
                0.0
            } else {
                f64::NEG_INFINITY
            }
        } else {
            -0.5 * (distance.powi(2) / combined_variance)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geopoint_clamping_edge_cases() {
        let point_over = GeoPoint::new(105.0, 210.0, Some(10.0));
        assert_eq!(point_over.latitude, 90.0);
        assert_eq!(point_over.longitude, 180.0);

        let point_under = GeoPoint::new(-105.0, -210.0, None);
        assert_eq!(point_under.latitude, -90.0);
        assert_eq!(point_under.longitude, -180.0);
    }

    #[test]
    fn test_haversine_distance_identical_points() {
        let point = GeoPoint::new(48.8566, 2.3522, None);
        let distance = point.haversine_distance(&point);
        assert!(distance < f64::EPSILON);
    }

    #[test]
    fn test_haversine_distance_known_coordinates() {
        let paris = GeoPoint::new(48.8566, 2.3522, None);
        let london = GeoPoint::new(51.5074, -0.1278, None);
        let distance = paris.haversine_distance(&london);

        let expected_distance_meters = 343_556.0;
        assert!((distance - expected_distance_meters).abs() < 1000.0);
    }

    #[test]
    fn test_haversine_antipodal_points() {
        let north_pole = GeoPoint::new(90.0, 0.0, None);
        let south_pole = GeoPoint::new(-90.0, 0.0, None);
        let distance = north_pole.haversine_distance(&south_pole);

        let half_earth_circumference = std::f64::consts::PI * 6_371_000.0;
        assert!((distance - half_earth_circumference).abs() < 1.0);
    }

    #[test]
    fn test_spatial_component_negative_accuracy() {
        let location = GeoPoint::new(0.0, 0.0, None);
        let component = SpatialComponent::new(location, -15.0);
        assert_eq!(component.accuracy_meters, 0.0);
    }

    #[test]
    fn test_log_likelihood_zero_variance_identical_location() {
        let location = GeoPoint::new(48.8566, 2.3522, None);
        let obs = SpatialComponent::new(location, 0.0);
        let state = SpatialComponent::new(location, 0.0);

        assert_eq!(obs.evaluate_log_likelihood(&state), 0.0);
    }

    #[test]
    fn test_log_likelihood_zero_variance_different_location() {
        let p1 = GeoPoint::new(48.8566, 2.3522, None);
        let p2 = GeoPoint::new(48.8567, 2.3523, None);
        let obs = SpatialComponent::new(p1, 0.0);
        let state = SpatialComponent::new(p2, 0.0);

        assert_eq!(obs.evaluate_log_likelihood(&state), f64::NEG_INFINITY);
    }

    #[test]
    fn test_log_likelihood_decay_value() {
        let p1 = GeoPoint::new(0.0, 0.0, None);
        let p2 = GeoPoint::new(0.0, 0.001, None);
        let obs = SpatialComponent::new(p1, 100.0);
        let state = SpatialComponent::new(p2, 100.0);

        let log_lh = obs.evaluate_log_likelihood(&state);
        assert!(log_lh < 0.0);
        assert!(log_lh.is_finite());
    }
}
