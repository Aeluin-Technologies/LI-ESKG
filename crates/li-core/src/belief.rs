//! Structures representing active tracking configurations within the ephemeral
//! layer.

use std::collections::VecDeque;
use std::vec::Vec;

use chrono::Duration;
use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer, Serialize};

use crate::ids::IdentityId;
use crate::observation::{Observation, Timestamp};
use crate::probability::Probability;

/// Fixed-limit FIFO history that allocates its storage once and never grows.
///
/// Pushing into a full history evicts and returns the oldest entry. A zero
/// capacity history rejects every pushed entry by returning it immediately.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BoundedHistory<T> {
    entries: VecDeque<T>,
    capacity: usize,
}

impl<T> BoundedHistory<T> {
    /// Allocates storage for at most `capacity` history entries.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of retained entries.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Returns the configured logical entry limit.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of retained entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no entries are retained.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the oldest retained entry, if present.
    pub fn front(&self) -> Option<&T> {
        self.entries.front()
    }

    /// Returns the newest retained entry, if present.
    pub fn back(&self) -> Option<&T> {
        self.entries.back()
    }

    /// Returns an iterator over entries from oldest to newest.
    pub fn iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = &T> + ExactSizeIterator {
        self.entries.iter()
    }

    /// Returns the two contiguous slices backing the ring buffer.
    pub fn as_slices(&self) -> (&[T], &[T]) {
        self.entries.as_slices()
    }

    /// Removes all entries while retaining the allocated storage.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Appends an entry and returns the evicted oldest entry, if any.
    ///
    /// # Arguments
    ///
    /// * `entry` - Value to append to the history.
    pub fn push(&mut self, entry: T) -> Option<T> {
        if self.capacity == 0 {
            return Some(entry);
        }

        let evicted = if self.entries.len() == self.capacity {
            self.entries.pop_front()
        } else {
            None
        };
        self.entries.push_back(entry);
        evicted
    }
}

impl<T> Default for BoundedHistory<T> {
    fn default() -> Self {
        Self::new(0)
    }
}

#[derive(Deserialize)]
struct BoundedHistoryRepresentation<T> {
    entries: Vec<T>,
    capacity: usize,
}

impl<'de, T> Deserialize<'de> for BoundedHistory<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let representation =
            BoundedHistoryRepresentation::<T>::deserialize(deserializer)?;
        if representation.entries.len() > representation.capacity {
            return Err(D::Error::custom(
                "bounded history contains more entries than its capacity",
            ));
        }

        let mut history = Self::new(representation.capacity);
        history.entries.extend(representation.entries);
        Ok(history)
    }
}

/// Uniform execution interface for modality-specific rolling belief summaries.
///
/// Implementations operate on caller-owned summaries and buffers so the hot
/// path can reuse allocations across observations and checkpoints.
pub trait ObservationModel<P>: Send + Sync {
    /// Rolling statistical summary maintained for this modality.
    type Summary;
    /// Domain error returned by state mutation and serialization operations.
    type Error;

    /// Computes the observation likelihood for the current summary.
    ///
    /// # Arguments
    ///
    /// * `summary` - Current bounded statistical summary.
    /// * `payload` - Incoming modality-specific payload.
    fn likelihood(&self, summary: &Self::Summary, payload: &P) -> Probability;

    /// Incorporates an immutable observation into a rolling summary.
    ///
    /// # Arguments
    ///
    /// * `summary` - Summary to update in place.
    /// * `observation` - Incoming immutable empirical observation.
    ///
    /// # Errors
    ///
    /// Returns the model-specific error when the update cannot be applied.
    fn update(
        &self,
        summary: &mut Self::Summary,
        observation: &Observation<P>,
    ) -> Result<(), Self::Error>;

    /// Merges another summary into a target summary without cloning it.
    ///
    /// # Arguments
    ///
    /// * `target` - Summary that receives the merged statistics.
    /// * `other` - Summary consumed logically through a shared reference.
    ///
    /// # Errors
    ///
    /// Returns the model-specific error when the summaries are incompatible.
    fn merge(
        &self,
        target: &mut Self::Summary,
        other: &Self::Summary,
    ) -> Result<(), Self::Error>;

    /// Applies temporal decay to a summary.
    ///
    /// # Arguments
    ///
    /// * `summary` - Summary to decay in place.
    /// * `delta` - Non-negative elapsed time since the previous update.
    ///
    /// # Errors
    ///
    /// Returns the model-specific error when decay cannot be computed.
    fn decay(
        &self,
        summary: &mut Self::Summary,
        delta: Duration,
    ) -> Result<(), Self::Error>;

    /// Appends a checkpoint encoding to a caller-reused byte buffer.
    ///
    /// Implementations must not clear `output`; callers may compose several
    /// model checkpoints into one preallocated buffer.
    ///
    /// # Arguments
    ///
    /// * `summary` - Summary to serialize.
    /// * `output` - Reusable destination buffer.
    ///
    /// # Errors
    ///
    /// Returns the model-specific serialization error.
    fn checkpoint(
        &self,
        summary: &Self::Summary,
        output: &mut Vec<u8>,
    ) -> Result<(), Self::Error>;
}

/// State representation of a tracking hypothesis inside the active layer.
/// Matches the theoretical formulation $b_i = (\theta, \Sigma, \Lambda)$.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BeliefState<S> {
    /// Target latent identity identifier.
    pub identity: IdentityId,
    /// Modality-agnostic rolling statistical summary data.
    pub summary: S,
    /// Current calculated marginal posterior probability value.
    pub posterior: Probability,
    /// Temporal marker of the latest update or reinforcement.
    pub last_update: Timestamp,
}

impl<S> BeliefState<S> {
    /// Instantiates a new active belief state structure.
    pub fn new(
        identity: IdentityId,
        summary: S,
        posterior: Probability,
        last_update: Timestamp,
    ) -> Self {
        Self {
            identity,
            summary,
            posterior,
            last_update,
        }
    }

    /// Updates the marginal posterior value and update timestamp.
    ///
    /// # Arguments
    ///
    /// * `posterior` - New posterior probability score.
    /// * `timestamp` - Microsecond update timestamp.
    pub fn update_posterior(
        &mut self,
        posterior: Probability,
        timestamp: Timestamp,
    ) {
        self.posterior = posterior;
        self.last_update = self.last_update.max(timestamp);
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use super::*;
    use crate::ids::ObservationId;
    use crate::observation::Modality;
    use crate::probability::Confidence;

    struct CounterModel;

    impl ObservationModel<u8> for CounterModel {
        type Error = Infallible;
        type Summary = u16;

        fn likelihood(
            &self,
            summary: &Self::Summary,
            payload: &u8,
        ) -> Probability {
            Probability::new(
                f64::from(*payload) / f64::from((*summary).max(1)),
            )
        }

        fn update(
            &self,
            summary: &mut Self::Summary,
            observation: &Observation<u8>,
        ) -> Result<(), Self::Error> {
            *summary = summary.saturating_add(u16::from(observation.payload));
            Ok(())
        }

        fn merge(
            &self,
            target: &mut Self::Summary,
            other: &Self::Summary,
        ) -> Result<(), Self::Error> {
            *target = target.saturating_add(*other);
            Ok(())
        }

        fn decay(
            &self,
            summary: &mut Self::Summary,
            _delta: Duration,
        ) -> Result<(), Self::Error> {
            *summary /= 2;
            Ok(())
        }

        fn checkpoint(
            &self,
            summary: &Self::Summary,
            output: &mut Vec<u8>,
        ) -> Result<(), Self::Error> {
            output.extend_from_slice(&summary.to_le_bytes());
            Ok(())
        }
    }

    #[test]
    fn bounded_history_evicts_oldest_without_growing_storage() {
        let mut history = BoundedHistory::new(3);
        let allocated_capacity = history.entries.capacity();

        for value in 0..100 {
            history.push(value);
            assert_eq!(history.entries.capacity(), allocated_capacity);
        }

        assert_eq!(history.len(), 3);
        assert_eq!(
            history.iter().copied().collect::<Vec<_>>(),
            vec![97, 98, 99]
        );
        assert_eq!(history.front(), Some(&97));
        assert_eq!(history.back(), Some(&99));

        history.clear();
        assert!(history.is_empty());
        assert_eq!(history.entries.capacity(), allocated_capacity);
    }

    #[test]
    fn zero_capacity_history_returns_every_entry() {
        let mut history = BoundedHistory::new(0);

        assert_eq!(history.push(7), Some(7));
        assert!(history.is_empty());
        assert_eq!(history.capacity(), 0);
    }

    #[test]
    fn bounded_history_deserialization_enforces_capacity() {
        let invalid = serde_json::from_str::<BoundedHistory<u8>>(
            r#"{"entries":[1,2],"capacity":1}"#,
        );
        let valid = serde_json::from_str::<BoundedHistory<u8>>(
            r#"{"entries":[1,2],"capacity":3}"#,
        );

        assert!(invalid.is_err());
        assert!(valid.is_ok());
        if let Ok(valid) = valid {
            assert_eq!(valid.iter().copied().collect::<Vec<_>>(), vec![1, 2]);
            assert_eq!(valid.capacity(), 3);
        }
    }

    #[test]
    fn observation_model_reuses_checkpoint_buffer() {
        let model = CounterModel;
        let observation = Observation::new(
            ObservationId(1),
            Modality(1),
            Timestamp::from_micros(10),
            Confidence::new(1.0),
            4,
        );
        let mut summary = 4;

        assert!(model.update(&mut summary, &observation).is_ok());
        assert!(model.merge(&mut summary, &2).is_ok());
        assert!(model.decay(&mut summary, Duration::microseconds(1)).is_ok());
        assert_eq!(model.likelihood(&summary, &4), Probability::new(0.8));

        let mut checkpoint = Vec::with_capacity(8);
        let allocated_capacity = checkpoint.capacity();
        assert!(model.checkpoint(&summary, &mut checkpoint).is_ok());
        checkpoint.clear();
        assert!(model.checkpoint(&summary, &mut checkpoint).is_ok());
        assert_eq!(checkpoint.capacity(), allocated_capacity);
        assert_eq!(checkpoint.as_slice(), &5_u16.to_le_bytes());
    }

    #[test]
    fn posterior_updates_do_not_regress_last_update_time() {
        let mut belief = BeliefState::new(
            IdentityId(1),
            (),
            Probability::new(0.4),
            Timestamp::from_micros(20),
        );

        belief.update_posterior(
            Probability::new(0.8),
            Timestamp::from_micros(10),
        );
        assert_eq!(belief.posterior, Probability::new(0.8));
        assert_eq!(belief.last_update, Timestamp::from_micros(20));

        belief.update_posterior(
            Probability::new(0.9),
            Timestamp::from_micros(30),
        );
        assert_eq!(belief.last_update, Timestamp::from_micros(30));
    }
}
