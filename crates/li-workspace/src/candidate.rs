//! Allocation-conscious candidate extraction and spatial indexing.

use alloc::vec::Vec;
use core::cmp::Ordering;
use core::error::Error;
use core::fmt;

use hashbrown::HashMap;
use li_core::ids::IdentityId;
use li_core::observation::Observation;

/// Maximum number of grid cells visited by one spatial query.
///
/// Bounding the window prevents a malformed configuration from turning a
/// local lookup into an effectively unbounded scan.
const MAX_QUERY_CELL_COUNT: u64 = 1_000_000;

/// Abstract interface for upstream metric-space candidate indices.
///
/// This formalizes the mapping $f_{\text{index}}: o_t \mapsto
/// \{i \in I \mid d(\operatorname{emb}(o_t),
/// \operatorname{emb}(i)) < \epsilon\}$.
pub trait CandidateGenerator<P> {
    /// Generates candidate identity identifiers for an observation.
    ///
    /// Implementations with a reusable-buffer API should override
    /// [`Self::generate_candidates_into`] and may implement this method as a
    /// convenience allocation.
    fn generate_candidates(
        &self,
        observation: &Observation<P>,
    ) -> Vec<IdentityId>;

    /// Writes candidate identifiers into a caller-owned reusable buffer.
    ///
    /// The default implementation preserves compatibility with generators
    /// that only implement [`Self::generate_candidates`]. Implementations on
    /// latency-sensitive paths should override this method to avoid the
    /// temporary allocation.
    ///
    /// # Arguments
    ///
    /// * `observation` - Observation used to search the index.
    /// * `output` - Buffer cleared before candidate identifiers are written.
    fn generate_candidates_into(
        &self,
        observation: &Observation<P>,
        output: &mut Vec<IdentityId>,
    ) {
        output.clear();
        output.extend(self.generate_candidates(observation));
    }
}

/// Validated point in a two-dimensional metric space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialPoint {
    x: f64,
    y: f64,
}

impl SpatialPoint {
    /// Creates a finite spatial point.
    ///
    /// # Arguments
    ///
    /// * `x` - Horizontal metric coordinate.
    /// * `y` - Vertical metric coordinate.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialIndexError::NonFiniteCoordinate`] when either
    /// coordinate is NaN or infinite.
    pub fn try_new(x: f64, y: f64) -> Result<Self, SpatialIndexError> {
        if !x.is_finite() || !y.is_finite() {
            return Err(SpatialIndexError::NonFiniteCoordinate);
        }
        Ok(Self { x, y })
    }

    /// Returns the horizontal coordinate.
    pub const fn x(self) -> f64 {
        self.x
    }

    /// Returns the vertical coordinate.
    pub const fn y(self) -> f64 {
        self.y
    }

    /// Returns the squared Euclidean distance to another point.
    ///
    /// Squared distance avoids a square root in the query hot path.
    pub fn squared_distance(self, other: Self) -> f64 {
        let delta_x = self.x - other.x;
        let delta_y = self.y - other.y;
        delta_x.mul_add(delta_x, delta_y * delta_y)
    }
}

/// Extracts validated spatial coordinates from an observation payload.
///
/// Applications implement this trait for their local payload type, allowing
/// [`SpatialGridIndex`] to implement [`CandidateGenerator`] without dynamic
/// dispatch or an allocated embedding.
pub trait SpatialCoordinates {
    /// Returns the metric-space position represented by this payload.
    ///
    /// # Errors
    ///
    /// Returns a [`SpatialIndexError`] when the payload does not contain a
    /// valid indexable position.
    fn spatial_point(&self) -> Result<SpatialPoint, SpatialIndexError>;
}

/// Validated fixed-grid search configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialGridConfig {
    cell_size: f64,
    search_radius: f64,
    search_radius_squared: f64,
    cell_radius: i64,
    max_candidates: usize,
}

impl SpatialGridConfig {
    /// Creates a fixed-cell spatial-index configuration.
    ///
    /// # Arguments
    ///
    /// * `cell_size` - Positive finite width and height of each grid cell.
    /// * `search_radius` - Non-negative finite exact Euclidean search radius.
    /// * `max_candidates` - Maximum number of candidates returned per query.
    ///   Zero disables candidate results without disabling index updates.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid scalar values or a cell window exceeding
    /// the hard query-work bound.
    pub fn try_new(
        cell_size: f64,
        search_radius: f64,
        max_candidates: usize,
    ) -> Result<Self, SpatialIndexError> {
        if !cell_size.is_finite() || cell_size <= 0.0 {
            return Err(SpatialIndexError::InvalidCellSize);
        }
        if !search_radius.is_finite() || search_radius < 0.0 {
            return Err(SpatialIndexError::InvalidSearchRadius);
        }
        let search_radius_squared = search_radius * search_radius;
        if !search_radius_squared.is_finite() {
            return Err(SpatialIndexError::InvalidSearchRadius);
        }

        let cell_radius_value = (search_radius / cell_size).ceil();
        if !cell_radius_value.is_finite() ||
            cell_radius_value > u64::MAX as f64
        {
            return Err(SpatialIndexError::QueryWindowTooLarge);
        }
        let cell_radius_u64 = cell_radius_value as u64;
        let diameter = cell_radius_u64
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(SpatialIndexError::QueryWindowTooLarge)?;
        let query_cell_count = diameter
            .checked_mul(diameter)
            .ok_or(SpatialIndexError::QueryWindowTooLarge)?;
        if query_cell_count > MAX_QUERY_CELL_COUNT ||
            cell_radius_u64 > i64::MAX as u64
        {
            return Err(SpatialIndexError::QueryWindowTooLarge);
        }

        Ok(Self {
            cell_size,
            search_radius,
            search_radius_squared,
            cell_radius: cell_radius_u64 as i64,
            max_candidates,
        })
    }

    /// Returns the grid cell size.
    pub const fn cell_size(self) -> f64 {
        self.cell_size
    }

    /// Returns the exact Euclidean search radius.
    pub const fn search_radius(self) -> f64 {
        self.search_radius
    }

    /// Returns the maximum number of candidates emitted by one query.
    pub const fn max_candidates(self) -> usize {
        self.max_candidates
    }
}

/// Error reported by validated spatial-index operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialIndexError {
    /// Grid cell size is zero, negative, NaN, or infinite.
    InvalidCellSize,
    /// Search radius is negative, NaN, or infinite.
    InvalidSearchRadius,
    /// A point contains a NaN or infinite coordinate.
    NonFiniteCoordinate,
    /// A coordinate cannot be represented by the integer grid.
    CoordinateOutOfRange,
    /// The configured neighborhood exceeds the bounded query-work limit.
    QueryWindowTooLarge,
}

impl fmt::Display for SpatialIndexError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidCellSize => {
                "spatial grid cell size must be positive and finite"
            },
            Self::InvalidSearchRadius => {
                "spatial search radius must be non-negative and finite"
            },
            Self::NonFiniteCoordinate => "spatial coordinates must be finite",
            Self::CoordinateOutOfRange => {
                "spatial coordinate is outside the supported grid range"
            },
            Self::QueryWindowTooLarge => {
                "spatial query window exceeds the bounded cell limit"
            },
        };
        formatter.write_str(message)
    }
}

impl Error for SpatialIndexError {}

/// Ranked identity returned by a spatial query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialMatch {
    identity: IdentityId,
    squared_distance: f64,
}

impl SpatialMatch {
    /// Returns the matched identity identifier.
    pub const fn identity(self) -> IdentityId {
        self.identity
    }

    /// Returns the squared Euclidean distance used for ranking.
    pub const fn squared_distance(self) -> f64 {
        self.squared_distance
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CellKey {
    x: i64,
    y: i64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct IndexedPoint {
    point: SpatialPoint,
    cell: CellKey,
}

/// Fixed-cell spatial hash index with bounded deterministic nearest lookup.
///
/// Query cost is proportional to the configured neighboring cells and their
/// local occupancy rather than the total identity count. Buckets and identity
/// locations are retained in separate hash tables so updates never require a
/// global scan.
#[derive(Debug, Clone)]
pub struct SpatialGridIndex {
    config: SpatialGridConfig,
    cells: HashMap<CellKey, Vec<IdentityId>>,
    locations: HashMap<IdentityId, IndexedPoint>,
}

impl SpatialGridIndex {
    /// Creates an empty spatial grid.
    ///
    /// # Arguments
    ///
    /// * `config` - Validated cell, radius, and result-bound configuration.
    pub fn new(config: SpatialGridConfig) -> Self {
        Self {
            config,
            cells: HashMap::new(),
            locations: HashMap::new(),
        }
    }

    /// Creates an empty grid with capacity for known identity cardinality.
    ///
    /// Reserving up front avoids hash-table growth on the ingestion path.
    pub fn with_capacity(
        config: SpatialGridConfig,
        identity_capacity: usize,
    ) -> Self {
        Self {
            config,
            cells: HashMap::with_capacity(identity_capacity),
            locations: HashMap::with_capacity(identity_capacity),
        }
    }

    /// Returns the active configuration.
    pub const fn config(&self) -> SpatialGridConfig {
        self.config
    }

    /// Returns the number of indexed identities.
    pub fn len(&self) -> usize {
        self.locations.len()
    }

    /// Returns whether the index contains no identities.
    pub fn is_empty(&self) -> bool {
        self.locations.is_empty()
    }

    /// Reserves storage for additional identities and occupied cells.
    ///
    /// # Arguments
    ///
    /// * `additional` - Number of identities expected to be inserted.
    pub fn reserve(&mut self, additional: usize) {
        self.locations.reserve(additional);
        self.cells.reserve(additional);
    }

    /// Returns whether an identity is present in the index.
    pub fn contains(&self, identity: IdentityId) -> bool {
        self.locations.contains_key(&identity)
    }

    /// Returns the current point of an indexed identity.
    pub fn point(&self, identity: IdentityId) -> Option<SpatialPoint> {
        self.locations.get(&identity).map(|entry| entry.point)
    }

    /// Inserts an identity or atomically moves its existing index entry.
    ///
    /// Coordinate-to-cell validation completes before any existing entry is
    /// changed. Updating an identity in its current cell does not touch the
    /// bucket, while cross-cell moves remove the previous membership.
    ///
    /// # Arguments
    ///
    /// * `identity` - Stable identity identifier to insert or update.
    /// * `point` - Current metric-space point for the identity.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialIndexError::CoordinateOutOfRange`] when the point
    /// cannot be represented by the configured grid. The existing entry
    /// remains unchanged on validation failure.
    pub fn insert(
        &mut self,
        identity: IdentityId,
        point: SpatialPoint,
    ) -> Result<(), SpatialIndexError> {
        let destination = self.cell_for(point)?;
        let previous = self.locations.get(&identity).copied();

        if let Some(previous_entry) = previous {
            if previous_entry.cell == destination {
                self.locations.insert(
                    identity,
                    IndexedPoint {
                        point,
                        cell: destination,
                    },
                );
                return Ok(());
            }
            self.remove_from_cell(previous_entry.cell, identity);
        }

        self.cells.entry(destination).or_default().push(identity);
        self.locations.insert(
            identity,
            IndexedPoint {
                point,
                cell: destination,
            },
        );
        Ok(())
    }

    /// Removes an identity and its cell membership.
    ///
    /// Returns `true` when the identity was indexed.
    pub fn remove(&mut self, identity: IdentityId) -> bool {
        let Some(indexed) = self.locations.remove(&identity) else {
            return false;
        };
        self.remove_from_cell(indexed.cell, identity);
        true
    }

    /// Clears all identities while retaining hash-table allocations.
    pub fn clear(&mut self) {
        self.cells.clear();
        self.locations.clear();
    }

    /// Finds exact-radius candidates into a caller-reused ranked buffer.
    ///
    /// Results are ordered by increasing squared distance and then increasing
    /// identity identifier, making output deterministic across hash seeds and
    /// bucket update order. The buffer reserves at most the configured result
    /// bound on its first call and does not grow on subsequent calls.
    ///
    /// # Arguments
    ///
    /// * `point` - Query point.
    /// * `output` - Buffer cleared and filled with at most `max_candidates`.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialIndexError::CoordinateOutOfRange`] when the query
    /// point cannot be represented by the configured grid.
    pub fn query_into(
        &self,
        point: SpatialPoint,
        output: &mut Vec<SpatialMatch>,
    ) -> Result<(), SpatialIndexError> {
        output.clear();
        if self.config.max_candidates == 0 {
            return Ok(());
        }
        output.reserve_exact(self.config.max_candidates);

        self.visit_matches(point, |candidate| {
            Self::insert_ranked_match(
                output,
                candidate,
                self.config.max_candidates,
            );
        })
    }

    /// Finds candidate identifiers into a caller-reused buffer.
    ///
    /// This compatibility path preserves the same bounded deterministic order
    /// as [`Self::query_into`] without allocating a temporary match buffer.
    /// Prefer [`Self::query_into`] when distances are consumed downstream,
    /// because retaining the rank records avoids repeated location lookups.
    pub fn query_ids_into(
        &self,
        point: SpatialPoint,
        output: &mut Vec<IdentityId>,
    ) -> Result<(), SpatialIndexError> {
        output.clear();
        if self.config.max_candidates == 0 {
            return Ok(());
        }
        output.reserve_exact(self.config.max_candidates);

        self.visit_matches(point, |candidate| {
            let insertion = match output.iter().position(|identity| {
                self.compare_identity_to_match(*identity, candidate, point) ==
                    Ordering::Greater
            }) {
                Some(index) => index,
                None => output.len(),
            };

            if output.len() < self.config.max_candidates {
                output.insert(insertion, candidate.identity);
            } else if insertion < self.config.max_candidates {
                let _ = output.pop();
                output.insert(insertion, candidate.identity);
            }
        })
    }

    fn cell_for(
        &self,
        point: SpatialPoint,
    ) -> Result<CellKey, SpatialIndexError> {
        Ok(CellKey {
            x: Self::cell_component(point.x, self.config.cell_size)?,
            y: Self::cell_component(point.y, self.config.cell_size)?,
        })
    }

    fn cell_component(
        coordinate: f64,
        cell_size: f64,
    ) -> Result<i64, SpatialIndexError> {
        let scaled = (coordinate / cell_size).floor();
        let maximum_exclusive = -(i64::MIN as f64);
        if !scaled.is_finite() ||
            scaled < i64::MIN as f64 ||
            scaled >= maximum_exclusive
        {
            return Err(SpatialIndexError::CoordinateOutOfRange);
        }
        Ok(scaled as i64)
    }

    fn remove_from_cell(&mut self, cell: CellKey, identity: IdentityId) {
        let should_remove = if let Some(bucket) = self.cells.get_mut(&cell) {
            if let Some(position) =
                bucket.iter().position(|candidate| *candidate == identity)
            {
                let _ = bucket.swap_remove(position);
            }
            bucket.is_empty()
        } else {
            false
        };
        if should_remove {
            self.cells.remove(&cell);
        }
    }

    fn visit_matches(
        &self,
        point: SpatialPoint,
        mut visitor: impl FnMut(SpatialMatch),
    ) -> Result<(), SpatialIndexError> {
        let center = self.cell_for(point)?;
        let radius = self.config.cell_radius;

        for delta_y in -radius..=radius {
            let Some(cell_y) = center.y.checked_add(delta_y) else {
                continue;
            };
            for delta_x in -radius..=radius {
                let Some(cell_x) = center.x.checked_add(delta_x) else {
                    continue;
                };
                let key = CellKey {
                    x: cell_x,
                    y: cell_y,
                };
                let Some(bucket) = self.cells.get(&key) else {
                    continue;
                };
                for identity in bucket {
                    let Some(indexed) = self.locations.get(identity) else {
                        continue;
                    };
                    let squared_distance =
                        point.squared_distance(indexed.point);
                    if squared_distance <= self.config.search_radius_squared {
                        visitor(SpatialMatch {
                            identity: *identity,
                            squared_distance,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn insert_ranked_match(
        output: &mut Vec<SpatialMatch>,
        candidate: SpatialMatch,
        limit: usize,
    ) {
        let insertion = match output.binary_search_by(|existing| {
            Self::compare_matches(*existing, candidate)
        }) {
            Ok(index) | Err(index) => index,
        };
        if output.len() < limit {
            output.insert(insertion, candidate);
        } else if insertion < limit {
            let _ = output.pop();
            output.insert(insertion, candidate);
        }
    }

    fn compare_matches(left: SpatialMatch, right: SpatialMatch) -> Ordering {
        left.squared_distance
            .total_cmp(&right.squared_distance)
            .then_with(|| left.identity.cmp(&right.identity))
    }

    fn compare_identity_to_match(
        &self,
        identity: IdentityId,
        right: SpatialMatch,
        query: SpatialPoint,
    ) -> Ordering {
        let Some(indexed) = self.locations.get(&identity) else {
            return identity.cmp(&right.identity);
        };
        let left = SpatialMatch {
            identity,
            squared_distance: query.squared_distance(indexed.point),
        };
        left.squared_distance
            .total_cmp(&right.squared_distance)
            .then_with(|| identity.cmp(&right.identity))
    }
}

impl<P> CandidateGenerator<P> for SpatialGridIndex
where
    P: SpatialCoordinates,
{
    fn generate_candidates(
        &self,
        observation: &Observation<P>,
    ) -> Vec<IdentityId> {
        let mut candidates = Vec::with_capacity(self.config.max_candidates);
        self.generate_candidates_into(observation, &mut candidates);
        candidates
    }

    fn generate_candidates_into(
        &self,
        observation: &Observation<P>,
        output: &mut Vec<IdentityId>,
    ) {
        output.clear();
        let Ok(point) = observation.payload.spatial_point() else {
            return;
        };
        let _ = self.query_ids_into(point, output);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use li_core::ids::ObservationId;
    use li_core::observation::{Modality, Timestamp};
    use li_core::probability::Confidence;

    use super::*;

    #[derive(Debug, Clone, Copy)]
    struct Payload {
        x: f64,
        y: f64,
    }

    impl SpatialCoordinates for Payload {
        fn spatial_point(&self) -> Result<SpatialPoint, SpatialIndexError> {
            SpatialPoint::try_new(self.x, self.y)
        }
    }

    fn config(
        cell_size: f64,
        radius: f64,
        max_candidates: usize,
    ) -> Result<SpatialGridConfig, SpatialIndexError> {
        SpatialGridConfig::try_new(cell_size, radius, max_candidates)
    }

    #[test]
    fn configuration_rejects_invalid_and_unbounded_windows() {
        assert_eq!(
            config(0.0, 1.0, 1),
            Err(SpatialIndexError::InvalidCellSize)
        );
        assert_eq!(
            config(f64::NAN, 1.0, 1),
            Err(SpatialIndexError::InvalidCellSize)
        );
        assert_eq!(
            config(1.0, -1.0, 1),
            Err(SpatialIndexError::InvalidSearchRadius)
        );
        assert_eq!(
            config(1.0, f64::INFINITY, 1),
            Err(SpatialIndexError::InvalidSearchRadius)
        );
        assert_eq!(
            config(f64::MAX, f64::MAX, 1),
            Err(SpatialIndexError::InvalidSearchRadius)
        );
        assert_eq!(
            config(f64::MIN_POSITIVE, f64::MAX, 1),
            Err(SpatialIndexError::InvalidSearchRadius)
        );
        assert_eq!(
            config(1.0, 1_000.0, 1),
            Err(SpatialIndexError::QueryWindowTooLarge)
        );
    }

    #[test]
    fn point_rejects_non_finite_coordinates() {
        assert_eq!(
            SpatialPoint::try_new(f64::NAN, 0.0),
            Err(SpatialIndexError::NonFiniteCoordinate)
        );
        assert_eq!(
            SpatialPoint::try_new(0.0, f64::INFINITY),
            Err(SpatialIndexError::NonFiniteCoordinate)
        );
    }

    #[test]
    fn query_crosses_cell_boundaries_and_applies_exact_radius()
    -> Result<(), SpatialIndexError> {
        let mut index = SpatialGridIndex::new(config(10.0, 2.0, 8)?);
        index.insert(IdentityId(1), SpatialPoint::try_new(10.5, 2.0)?)?;
        index.insert(IdentityId(2), SpatialPoint::try_new(11.1, 2.0)?)?;

        let mut output = Vec::new();
        index.query_into(SpatialPoint::try_new(9.0, 2.0)?, &mut output)?;

        assert_eq!(output.len(), 1);
        assert_eq!(output[0].identity(), IdentityId(1));
        assert_eq!(output[0].squared_distance(), 2.25);
        Ok(())
    }

    #[test]
    fn query_is_bounded_and_deterministic_for_equal_distances()
    -> Result<(), SpatialIndexError> {
        let mut index = SpatialGridIndex::new(config(2.0, 2.0, 2)?);
        index.insert(IdentityId(9), SpatialPoint::try_new(-1.0, 0.0)?)?;
        index.insert(IdentityId(3), SpatialPoint::try_new(1.0, 0.0)?)?;
        index.insert(IdentityId(5), SpatialPoint::try_new(0.0, 1.0)?)?;

        let mut output = Vec::new();
        index.query_into(SpatialPoint::try_new(0.0, 0.0)?, &mut output)?;

        assert_eq!(
            output
                .iter()
                .map(|candidate| candidate.identity())
                .collect::<Vec<_>>(),
            vec![IdentityId(3), IdentityId(5)]
        );

        let mut identities = Vec::new();
        index.query_ids_into(
            SpatialPoint::try_new(0.0, 0.0)?,
            &mut identities,
        )?;
        assert_eq!(identities, vec![IdentityId(3), IdentityId(5)]);
        Ok(())
    }

    #[test]
    fn moving_identity_removes_stale_bucket_entries()
    -> Result<(), SpatialIndexError> {
        let mut index = SpatialGridIndex::new(config(1.0, 0.5, 4)?);
        let identity = IdentityId(7);
        let original = SpatialPoint::try_new(0.0, 0.0)?;
        let destination = SpatialPoint::try_new(20.0, 20.0)?;

        index.insert(identity, original)?;
        index.insert(identity, original)?;
        index.insert(identity, destination)?;

        let mut output = Vec::new();
        index.query_into(original, &mut output)?;
        assert!(output.is_empty());
        index.query_into(destination, &mut output)?;
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].identity(), identity);
        assert!(index.remove(identity));
        assert!(!index.remove(identity));
        assert!(index.is_empty());
        Ok(())
    }

    #[test]
    fn failed_move_keeps_previous_entry_unchanged()
    -> Result<(), SpatialIndexError> {
        let mut index = SpatialGridIndex::new(config(1.0, 1.0, 4)?);
        let identity = IdentityId(11);
        let original = SpatialPoint::try_new(4.0, 5.0)?;
        index.insert(identity, original)?;

        let invalid = SpatialPoint::try_new(f64::MAX, 0.0)?;
        assert_eq!(
            index.insert(identity, invalid),
            Err(SpatialIndexError::CoordinateOutOfRange)
        );
        assert_eq!(index.point(identity), Some(original));

        let mut output = Vec::new();
        index.query_into(original, &mut output)?;
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].identity(), identity);
        Ok(())
    }

    #[test]
    fn reusable_output_capacity_stabilizes_after_first_query()
    -> Result<(), SpatialIndexError> {
        let mut index = SpatialGridIndex::new(config(1.0, 1.0, 16)?);
        index.insert(IdentityId(1), SpatialPoint::try_new(0.0, 0.0)?)?;
        let mut output = Vec::new();
        let query = SpatialPoint::try_new(0.0, 0.0)?;

        index.query_into(query, &mut output)?;
        let first_capacity = output.capacity();
        index.query_into(query, &mut output)?;

        assert!(first_capacity >= 16);
        assert_eq!(output.capacity(), first_capacity);
        Ok(())
    }

    #[test]
    fn candidate_generator_reuses_id_buffer_and_clears_on_invalid_payload()
    -> Result<(), SpatialIndexError> {
        let mut index = SpatialGridIndex::new(config(1.0, 1.0, 4)?);
        index.insert(IdentityId(2), SpatialPoint::try_new(0.0, 0.0)?)?;
        let valid = Observation::new(
            ObservationId(1),
            Modality(1),
            Timestamp::UNIX_EPOCH,
            Confidence::new(1.0),
            Payload { x: 0.0, y: 0.0 },
        );
        let invalid = Observation::new(
            ObservationId(2),
            Modality(1),
            Timestamp::UNIX_EPOCH,
            Confidence::new(1.0),
            Payload {
                x: f64::NAN,
                y: 0.0,
            },
        );
        let mut output = Vec::with_capacity(4);

        index.generate_candidates_into(&valid, &mut output);
        assert_eq!(output, vec![IdentityId(2)]);
        let capacity = output.capacity();
        index.generate_candidates_into(&invalid, &mut output);
        assert!(output.is_empty());
        assert_eq!(output.capacity(), capacity);
        Ok(())
    }

    #[test]
    fn zero_result_limit_skips_lookup_and_clear_retains_index_usability()
    -> Result<(), SpatialIndexError> {
        let mut index = SpatialGridIndex::new(config(1.0, 1.0, 0)?);
        let point = SpatialPoint::try_new(0.0, 0.0)?;
        index.insert(IdentityId(1), point)?;
        let mut output = vec![SpatialMatch {
            identity: IdentityId(99),
            squared_distance: 0.0,
        }];

        index.query_into(point, &mut output)?;
        assert!(output.is_empty());
        index.clear();
        assert!(index.is_empty());
        assert!(!index.contains(IdentityId(1)));
        Ok(())
    }
}
