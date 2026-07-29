//! Flat, allocation-reusing implementation of the `ActiveWorkspace` trait.

use alloc::vec::Vec;

use hashbrown::HashMap;
use li_core::belief::BeliefState;
use li_core::ids::IdentityId;
use li_core::observation::Timestamp;

use crate::checkpoint::WorkspaceSnapshot;
use crate::workspace::ActiveWorkspace;

/// Returns whether a belief exceeds its time-to-live without overflowing.
#[inline]
fn is_expired<S>(
    belief: &BeliefState<S>,
    current_time: Timestamp,
    ttl_microseconds: i64,
) -> bool {
    current_time
        .as_micros()
        .saturating_sub(belief.last_update.as_micros()) >
        ttl_microseconds
}

/// Holds active identity hypotheses in memory during real-time tracking,
/// providing expected constant-time lookup and reusable flat storage.
#[derive(Debug, Clone)]
pub struct InMemoryWorkspace<S> {
    /// Maps persistent identity identifiers to their active belief states.
    ///
    /// Iteration order is intentionally unspecified. Callers that need a
    /// stable presentation order should sort their resulting identifiers.
    pub beliefs: HashMap<IdentityId, BeliefState<S>>,
}

impl<S> InMemoryWorkspace<S> {
    /// Constructs a new, empty [`InMemoryWorkspace`].
    #[inline]
    pub fn new() -> Self {
        Self {
            beliefs: HashMap::new(),
        }
    }

    /// Constructs an empty workspace with space for at least `capacity`
    /// beliefs.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Minimum number of beliefs that can be inserted without
    ///   growing the backing table.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            beliefs: HashMap::with_capacity(capacity),
        }
    }

    /// Returns the total number of elements stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.beliefs.len()
    }

    /// Returns `true` if the workspace contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.beliefs.is_empty()
    }

    /// Returns the number of beliefs that fit before the table reallocates.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.beliefs.capacity()
    }

    /// Reserves capacity for at least `additional` more beliefs.
    ///
    /// # Arguments
    ///
    /// * `additional` - Number of entries that should fit in addition to the
    ///   current workspace length.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.beliefs.reserve(additional);
    }
}

impl<S> Default for InMemoryWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Clone> ActiveWorkspace for InMemoryWorkspace<S> {
    type Summary = S;

    /// Inserts or updates an active belief state hypothesis in the workspace.
    ///
    /// If an entry for the belief's identity already exists, it is overwritten
    /// with the new state.
    ///
    /// # Arguments
    ///
    /// * `belief` - The [`BeliefState`] instance to be stored.
    ///
    /// # Complexity
    ///
    /// * Time: expected $\mathcal{O}(1)$
    /// * Space: $\mathcal{O}(1)$ auxiliary allocation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use li_core::belief::BeliefState;
    /// use li_core::ids::IdentityId;
    /// use li_core::observation::Timestamp;
    /// use li_core::probability::Probability;
    /// use li_workspace::{ActiveWorkspace, InMemoryWorkspace};
    ///
    /// let mut workspace = InMemoryWorkspace::<[f32; 2]>::new();
    /// let belief = BeliefState {
    ///     identity: IdentityId(101),
    ///     summary: [0.5, 0.8],
    ///     posterior: Probability::new(0.95),
    ///     last_update: Timestamp::from_millis(1000000),
    /// };
    ///
    /// workspace.insert(belief);
    /// assert!(workspace.get(IdentityId(101)).is_some());
    /// ```
    fn insert(&mut self, belief: BeliefState<S>) {
        self.beliefs.insert(belief.identity, belief);
    }

    /// Retrieves an immutable reference to an active belief state by its
    /// identifier.
    ///
    /// # Arguments
    ///
    /// * `id` - The [`IdentityId`] corresponding to the desired belief state.
    ///
    /// # Returns
    ///
    /// An `Option` containing a reference to the [`BeliefState`] if found, or
    /// `None`.
    ///
    /// # Complexity
    ///
    /// * Time: expected $\mathcal{O}(1)$
    /// * Space: $\mathcal{O}(1)$
    fn get(&self, id: IdentityId) -> Option<&BeliefState<S>> {
        self.beliefs.get(&id)
    }

    /// Retrieves a mutable reference to an active belief state by its
    /// identifier.
    ///
    /// # Arguments
    ///
    /// * `id` - The [`IdentityId`] corresponding to the target belief state.
    ///
    /// # Returns
    ///
    /// An `Option` containing a mutable reference to the [`BeliefState`] if
    /// found, or `None`.
    ///
    /// # Complexity
    ///
    /// * Time: expected $\mathcal{O}(1)$
    /// * Space: $\mathcal{O}(1)$
    fn get_mut(&mut self, id: IdentityId) -> Option<&mut BeliefState<S>> {
        self.beliefs.get_mut(&id)
    }

    /// Collects immutable references to all currently active belief states.
    ///
    /// # Returns
    ///
    /// A `Vec` of references to all active [`BeliefState`] items.
    ///
    /// # Complexity
    ///
    /// * Time: $\mathcal{O}(|B_t|)$
    /// * Space: $\mathcal{O}(|B_t|)$ for the returned reference list.
    fn active_beliefs(&self) -> Vec<&BeliefState<S>> {
        self.beliefs.values().collect()
    }

    /// Evaluates all managed belief states against the temporal eviction
    /// policy and removes expired entries.
    ///
    /// Evicted beliefs are returned so they can be committed to the persistent
    /// graph or processed by the runtime.
    ///
    /// # Arguments
    ///
    /// * `current_time` - The baseline timestamp $t_{\text{current}}$ used for
    ///   expiration evaluation.
    /// * `ttl_microseconds` - The time-to-live threshold $\tau$ expressed in
    ///   microseconds.
    ///
    /// # Returns
    ///
    /// A `Vec` containing all evicted [`BeliefState`] instances.
    ///
    /// # Complexity
    ///
    /// * Time: expected $\mathcal{O}(|B_t|)$
    /// * Space: $\mathcal{O}(k)$ where $k$ is the number of evicted items.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use li_core::belief::BeliefState;
    /// use li_core::ids::IdentityId;
    /// use li_core::observation::Timestamp;
    /// use li_core::probability::Probability;
    /// use li_workspace::{ActiveWorkspace, InMemoryWorkspace};
    ///
    /// let mut workspace = InMemoryWorkspace::<()>::new();
    /// workspace.insert(BeliefState {
    ///     identity: IdentityId(1),
    ///     summary: (),
    ///     posterior: Probability::new(0.8),
    ///     last_update: Timestamp::from_millis(100),
    /// });
    ///
    /// // Evict entries older than 50 microseconds relative to timestamp 200
    /// let evicted = workspace.evict_expired(Timestamp::from_millis(200), 50);
    /// assert_eq!(evicted.len(), 1);
    /// assert!(workspace.get(IdentityId(1)).is_none());
    /// ```
    fn evict_expired(
        &mut self,
        current_time: Timestamp,
        ttl_microseconds: i64,
    ) -> Vec<BeliefState<S>> {
        let mut evicted = Vec::new();
        self.evict_expired_into(current_time, ttl_microseconds, &mut evicted);
        evicted
    }

    fn evict_expired_into(
        &mut self,
        current_time: Timestamp,
        ttl_microseconds: i64,
        evicted: &mut Vec<BeliefState<S>>,
    ) {
        evicted.clear();
        let expired_count = self
            .beliefs
            .values()
            .filter(|belief| {
                is_expired(belief, current_time, ttl_microseconds)
            })
            .count();

        if expired_count == 0 {
            return;
        }

        evicted.reserve(expired_count);
        if expired_count == self.beliefs.len() {
            evicted.extend(self.beliefs.drain().map(|(_, belief)| belief));
            return;
        }

        evicted.extend(
            self.beliefs
                .extract_if(|_, belief| {
                    is_expired(belief, current_time, ttl_microseconds)
                })
                .map(|(_, belief)| belief),
        );
    }

    /// Creates an immutable, serializable snapshot of the current workspace
    /// state $B_t$.
    ///
    /// # Arguments
    ///
    /// * `current_time` - The timestamp to mark the creation of the snapshot.
    ///
    /// # Returns
    ///
    /// A [`WorkspaceSnapshot`] containing cloned states of all active beliefs.
    ///
    /// # Complexity
    ///
    /// * Time: $\mathcal{O}(|B_t|)$ assuming $O(1)$ cloning cost for `S`.
    /// * Space: $\mathcal{O}(|B_t|)$ for the snapshot buffer allocation.
    fn create_snapshot(
        &self,
        current_time: Timestamp,
    ) -> WorkspaceSnapshot<S> {
        WorkspaceSnapshot {
            timestamp: current_time,
            active_states: self.beliefs.values().cloned().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use li_core::belief::BeliefState;
    use li_core::ids::IdentityId;
    use li_core::observation::Timestamp;
    use li_core::probability::Probability;

    use super::*;

    fn mock_belief(id: u64, timestamp: i64) -> BeliefState<u64> {
        BeliefState {
            identity: IdentityId(id),
            summary: id * 100,
            posterior: Probability::new(0.95),
            last_update: Timestamp::from_millis(timestamp),
        }
    }

    #[test]
    fn test_new_workspace_is_empty() {
        let workspace = InMemoryWorkspace::<u64>::new();
        assert_eq!(workspace.len(), 0);
        assert!(workspace.is_empty());
        assert!(workspace.active_beliefs().is_empty());
    }

    #[test]
    fn test_default_implementation() {
        let workspace = InMemoryWorkspace::<u64>::default();
        assert!(workspace.is_empty());
    }

    #[test]
    fn test_insert_and_get() {
        let mut workspace = InMemoryWorkspace::<u64>::new();
        let belief = mock_belief(42, 1000);

        workspace.insert(belief);

        assert_eq!(workspace.len(), 1);
        assert!(!workspace.is_empty());

        let fetched = workspace.get(IdentityId(42));
        assert!(fetched.is_some());
        assert_eq!(
            fetched.map(|belief| belief.identity),
            Some(IdentityId(42))
        );
        assert_eq!(fetched.map(|belief| belief.summary), Some(4200));
    }

    #[test]
    fn test_get_non_existent_returns_none() {
        let workspace = InMemoryWorkspace::<u64>::new();
        assert!(workspace.get(IdentityId(999)).is_none());
    }

    #[test]
    fn test_insert_overwrite_updates_state_and_preserves_len() {
        let mut workspace = InMemoryWorkspace::<u64>::new();

        let initial = mock_belief(1, 1000);
        let updated = BeliefState {
            identity: IdentityId(1),
            summary: 9999,
            posterior: Probability::new(0.99),
            last_update: Timestamp::from_millis(2000),
        };

        workspace.insert(initial);
        assert_eq!(workspace.len(), 1);

        workspace.insert(updated);
        assert_eq!(workspace.len(), 1);

        assert_eq!(
            workspace.get(IdentityId(1)).map(|belief| belief.summary),
            Some(9999)
        );
        assert_eq!(
            workspace
                .get(IdentityId(1))
                .map(|belief| belief.last_update),
            Some(Timestamp::from_millis(2000))
        );
    }

    #[test]
    fn test_get_mut_modifies_belief_in_place() {
        let mut workspace = InMemoryWorkspace::<u64>::new();
        workspace.insert(mock_belief(1, 1000));

        if let Some(belief) = workspace.get_mut(IdentityId(1)) {
            belief.summary = 5555;
        }

        assert_eq!(
            workspace.get(IdentityId(1)).map(|belief| belief.summary),
            Some(5555)
        );
    }

    #[test]
    fn test_evict_expired_selective() {
        let mut workspace = InMemoryWorkspace::<u64>::new();

        workspace.insert(mock_belief(1, 100));
        workspace.insert(mock_belief(2, 100));
        workspace.insert(mock_belief(3, 1000));

        assert_eq!(workspace.len(), 3);

        // Delta for 1 & 2 is 900 ms (900_000 us). Delta for 3 is 0 ms (0 us).
        // TTL threshold set to 500 ms (500_000 us).
        let evicted =
            workspace.evict_expired(Timestamp::from_millis(1000), 500_000);

        assert_eq!(evicted.len(), 2);
        let mut evicted_ids: Vec<_> =
            evicted.iter().map(|belief| belief.identity).collect();
        evicted_ids.sort_unstable();
        assert_eq!(evicted_ids, [IdentityId(1), IdentityId(2)]);
        assert_eq!(workspace.len(), 1);

        assert!(workspace.get(IdentityId(1)).is_none());
        assert!(workspace.get(IdentityId(2)).is_none());
        assert!(workspace.get(IdentityId(3)).is_some());
    }

    #[test]
    fn test_evict_expired_on_empty_workspace() {
        let mut workspace = InMemoryWorkspace::<u64>::new();
        let evicted =
            workspace.evict_expired(Timestamp::from_millis(1000), 100_000);

        assert!(evicted.is_empty());
        assert_eq!(workspace.len(), 0);
    }

    #[test]
    fn test_evict_expired_none_matching() {
        let mut workspace = InMemoryWorkspace::<u64>::new();
        workspace.insert(mock_belief(1, 1000));
        workspace.insert(mock_belief(2, 1000));

        // Elapsed time is 50 ms (50_000 us). TTL threshold set to 100 ms
        // (100_000 us).
        let evicted =
            workspace.evict_expired(Timestamp::from_millis(1050), 100_000);

        assert!(evicted.is_empty());
        assert_eq!(workspace.len(), 2);
    }

    #[test]
    fn test_evict_expired_all_matching() {
        let mut workspace = InMemoryWorkspace::<u64>::new();
        workspace.insert(mock_belief(1, 100));
        workspace.insert(mock_belief(2, 200));

        // Elapsed times are 900 ms and 800 ms. TTL threshold set to 100 ms
        // (100_000 us).
        let evicted =
            workspace.evict_expired(Timestamp::from_millis(1000), 100_000);

        assert_eq!(evicted.len(), 2);
        assert_eq!(workspace.len(), 0);
        assert!(workspace.is_empty());
    }

    #[test]
    fn test_evict_expired_uses_strict_ttl_boundary() {
        let mut workspace = InMemoryWorkspace::<u64>::new();
        workspace.insert(mock_belief(1, 900));
        workspace.insert(mock_belief(2, 899));

        let evicted =
            workspace.evict_expired(Timestamp::from_millis(1000), 100_000);

        assert_eq!(evicted.len(), 1);
        assert!(workspace.get(IdentityId(1)).is_some());
        assert!(workspace.get(IdentityId(2)).is_none());
    }

    #[test]
    fn test_evict_expired_handles_extreme_timestamps() {
        let mut workspace = InMemoryWorkspace::<u64>::new();
        workspace.insert(BeliefState {
            identity: IdentityId(1),
            summary: 100,
            posterior: Probability::new(0.95),
            last_update: Timestamp::MIN,
        });

        let evicted = workspace.evict_expired(Timestamp::MAX, i64::MAX - 1);

        assert_eq!(evicted.len(), 1);
        assert!(workspace.is_empty());
    }

    #[test]
    fn test_partial_eviction_retains_capacity_for_refill() {
        const CAPACITY: usize = 256;
        let mut workspace = InMemoryWorkspace::<u64>::with_capacity(CAPACITY);
        assert!(workspace.capacity() >= CAPACITY);

        for id in 0..CAPACITY {
            let timestamp = if id % 2 == 0 { 100 } else { 1_000 };
            workspace.insert(mock_belief(id as u64, timestamp));
        }
        let allocated_capacity = workspace.capacity();

        let evicted =
            workspace.evict_expired(Timestamp::from_millis(1_000), 500_000);
        assert_eq!(evicted.len(), CAPACITY / 2);
        assert!(workspace.capacity() >= workspace.len());

        let refill_count = evicted.len();
        for id in CAPACITY..CAPACITY + refill_count {
            workspace.insert(mock_belief(id as u64, 1_000));
        }
        assert_eq!(workspace.len(), CAPACITY);
        assert!(workspace.capacity() >= workspace.len());
        assert!(workspace.capacity() <= allocated_capacity);
    }

    #[test]
    fn test_full_eviction_retains_capacity_for_refill() {
        const CAPACITY: usize = 128;
        let mut workspace = InMemoryWorkspace::<u64>::with_capacity(CAPACITY);
        for id in 0..CAPACITY {
            workspace.insert(mock_belief(id as u64, 100));
        }
        let allocated_capacity = workspace.capacity();

        let evicted =
            workspace.evict_expired(Timestamp::from_millis(1_000), 100_000);
        assert_eq!(evicted.len(), CAPACITY);
        assert_eq!(workspace.capacity(), allocated_capacity);

        for id in CAPACITY..CAPACITY * 2 {
            workspace.insert(mock_belief(id as u64, 1_000));
        }
        assert_eq!(workspace.capacity(), allocated_capacity);
    }

    #[test]
    fn test_evict_expired_into_reuses_and_clears_output_buffer() {
        let mut workspace = InMemoryWorkspace::<u64>::with_capacity(4);
        workspace.insert(mock_belief(1, 100));
        workspace.insert(mock_belief(2, 100));
        workspace.insert(mock_belief(3, 1_000));
        let mut evicted = Vec::with_capacity(2);

        workspace.evict_expired_into(
            Timestamp::from_millis(1_000),
            500_000,
            &mut evicted,
        );
        assert_eq!(evicted.len(), 2);
        let allocated_capacity = evicted.capacity();

        workspace.evict_expired_into(
            Timestamp::from_millis(1_000),
            500_000,
            &mut evicted,
        );
        assert!(evicted.is_empty());
        assert_eq!(evicted.capacity(), allocated_capacity);
    }

    #[test]
    fn test_reserve_preallocates_additional_slots() {
        let mut workspace = InMemoryWorkspace::<u64>::new();
        workspace.insert(mock_belief(1, 100));
        workspace.reserve(255);

        let reserved_capacity = workspace.capacity();
        assert!(reserved_capacity >= 256);
        for id in 2..=256 {
            workspace.insert(mock_belief(id, 100));
        }
        assert_eq!(workspace.capacity(), reserved_capacity);
    }

    #[test]
    fn test_create_snapshot() {
        let mut workspace = InMemoryWorkspace::<u64>::new();
        workspace.insert(mock_belief(10, 500));
        workspace.insert(mock_belief(20, 600));

        let now = Timestamp::from_millis(1000);
        let snapshot = workspace.create_snapshot(now);

        assert_eq!(snapshot.timestamp, now);
        assert_eq!(snapshot.active_states.len(), 2);

        let ids: Vec<u64> = snapshot
            .active_states
            .iter()
            .map(|b| b.identity.0)
            .collect();
        assert!(ids.contains(&10));
        assert!(ids.contains(&20));
    }

    #[test]
    fn test_large_workspace_operations() {
        let total_items = 20_500;
        let mut workspace =
            InMemoryWorkspace::<u64>::with_capacity(total_items);

        for i in 0..total_items {
            let ts = if i % 2 == 0 { 100 } else { 1000 };
            workspace.insert(mock_belief(i as u64, ts));
        }

        assert_eq!(workspace.len(), total_items);

        let active = workspace.active_beliefs();
        assert_eq!(active.len(), total_items);

        let snapshot = workspace.create_snapshot(Timestamp::from_millis(2000));
        assert_eq!(snapshot.active_states.len(), total_items);

        let evicted =
            workspace.evict_expired(Timestamp::from_millis(2000), 1_500_000);

        let expected_evicted = total_items / 2;
        assert_eq!(evicted.len(), expected_evicted);
        assert_eq!(workspace.len(), total_items - expected_evicted);
    }
}
