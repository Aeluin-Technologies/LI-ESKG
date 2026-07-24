//! Fixed-capacity event queue primitives designed for no-std execution
//! environments.

use alloc::collections::VecDeque;

/// Bounded synchronous FIFO queue for buffering runtime events in no-std
/// environments.
#[derive(Debug, Clone)]
pub struct EventQueue<T> {
    capacity: usize,
    ring: VecDeque<T>,
}

impl<T> EventQueue<T> {
    /// Creates a new bounded event queue with a fixed capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of elements the queue can hold.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            ring: VecDeque::with_capacity(capacity),
        }
    }

    /// Pushes an item to the back of the queue if capacity allows.
    ///
    /// # Arguments
    ///
    /// * `item` - The item to insert into the queue.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or `Err(item)` if the queue is at full
    /// capacity.
    pub fn push(&mut self, item: T) -> Result<(), T> {
        if self.ring.len() >= self.capacity {
            Err(item)
        } else {
            self.ring.push_back(item);
            Ok(())
        }
    }

    /// Removes and returns the item at the front of the queue.
    ///
    /// # Returns
    ///
    /// Returns `Some(T)` if an element exists, or `None` if the queue is
    /// empty.
    pub fn pop(&mut self) -> Option<T> {
        self.ring.pop_front()
    }

    /// Returns the number of items currently stored in the queue.
    pub fn len(&self) -> usize {
        self.ring.len()
    }

    /// Checks whether the queue contains no elements.
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// Returns the maximum capacity of the queue.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_push_pop_fifo() {
        let mut q = EventQueue::new(2);
        assert!(q.is_empty());
        assert_eq!(q.push(10), Ok(()));
        assert_eq!(q.push(20), Ok(()));
        assert_eq!(q.len(), 2);
        assert_eq!(q.pop(), Some(10));
        assert_eq!(q.pop(), Some(20));
        assert_eq!(q.pop(), None);
        assert!(q.is_empty());
    }

    #[test]
    fn test_queue_overflow_edge_case() {
        let mut q = EventQueue::new(1);
        assert_eq!(q.push(1), Ok(()));
        assert_eq!(q.push(2), Err(2));
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_zero_capacity_edge_case() {
        let mut q = EventQueue::new(0);
        assert_eq!(q.push(42), Err(42));
        assert_eq!(q.pop(), None);
    }
}
