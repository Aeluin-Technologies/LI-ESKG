//! Fixed-capacity history storage for active workspace summaries.

use std::collections::VecDeque;

use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer, Serialize};

/// FIFO history that allocates for a fixed logical capacity and never grows.
///
/// Pushing into a full history evicts the oldest entry. A zero-capacity
/// history returns every pushed entry immediately.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BoundedHistory<T> {
    entries: VecDeque<T>,
    capacity: usize,
}

impl<T> BoundedHistory<T> {
    /// Allocates storage for at most `capacity` history entries.
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

    /// Returns whether the history is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the oldest retained entry.
    pub fn front(&self) -> Option<&T> {
        self.entries.front()
    }

    /// Returns the newest retained entry.
    pub fn back(&self) -> Option<&T> {
        self.entries.back()
    }

    /// Iterates from the oldest entry to the newest.
    pub fn iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = &T> + ExactSizeIterator {
        self.entries.iter()
    }

    /// Returns the two contiguous slices backing the ring buffer.
    pub fn as_slices(&self) -> (&[T], &[T]) {
        self.entries.as_slices()
    }

    /// Clears retained entries without releasing capacity.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Appends an entry and returns the evicted oldest entry, if any.
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
struct HistoryRepresentation<T> {
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
            HistoryRepresentation::<T>::deserialize(deserializer)?;
        if representation.entries.len() > representation.capacity {
            return Err(D::Error::custom(
                "history contains more entries than its declared capacity",
            ));
        }
        let mut history = Self::new(representation.capacity);
        history.entries.extend(representation.entries);
        Ok(history)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_history_evicts_oldest_without_growing() {
        let mut history = BoundedHistory::new(2);
        assert_eq!(history.push(1), None);
        assert_eq!(history.push(2), None);
        let capacity = history.entries.capacity();
        assert_eq!(history.push(3), Some(1));
        assert_eq!(history.iter().copied().collect::<Vec<_>>(), vec![2, 3]);
        assert_eq!(history.entries.capacity(), capacity);
    }

    #[test]
    fn zero_capacity_and_invalid_serialized_state_are_rejected() {
        let mut history = BoundedHistory::new(0);
        assert_eq!(history.push(7), Some(7));
        assert!(history.is_empty());
        let invalid = serde_json::from_str::<BoundedHistory<u8>>(
            r#"{"entries":[1,2],"capacity":1}"#,
        );
        assert!(invalid.is_err());
    }
}
