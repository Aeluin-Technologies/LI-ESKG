//! Tree-backed in-memory implementation of the `ActiveWorkspace` trait.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::hash::{Hash, Hasher};

use li_core::belief::BeliefState;
use li_core::ids::IdentityId;
use li_core::observation::Timestamp;
use rayon::prelude::*;

use crate::checkpoint::WorkspaceSnapshot;
use crate::eviction::{EvictionPolicy, TemporalEvictionPolicy};
use crate::workspace::ActiveWorkspace;

/// Number of internal shards used to partition belief states.
const NUM_SHARDS: usize = 16;

/// Threshold below which sequential iteration is used.
const PARALLEL_THRESHOLD: usize = 10_000;

/// Holds active identity hypotheses in memory during real-time tracking,
/// providing logarithmic lookup, insertion, and key-ordered operations.
#[derive(Debug, Clone)]
pub struct InMemoryWorkspace<S> {
    /// Map linking persistent identity identifiers to their active belief
    /// states.
    pub shards: Box<[BTreeMap<IdentityId, BeliefState<S>>]>,
    /// Bitmask (16 bits) tracking non-empty shards to skip empty iteration.
    active_mask: u16,
    /// Total number of active beliefs across all shards.
    total_len: usize,
}

impl<S> InMemoryWorkspace<S> {
    /// Constructs a new, empty [`InMemoryWorkspace`].
    pub fn new() -> Self {
        let mut shards = Vec::with_capacity(NUM_SHARDS);
        for _ in 0..NUM_SHARDS {
            shards.push(BTreeMap::new());
        }
        Self {
            shards: shards.into_boxed_slice(),
            active_mask: 0,
            total_len: 0,
        }
    }

    #[inline]
    fn shard_index(id: IdentityId) -> usize {
        struct FastHasher(u64);
        impl Hasher for FastHasher {
            #[inline]
            fn finish(&self) -> u64 {
                self.0
            }

            #[inline]
            fn write(&mut self, bytes: &[u8]) {
                for &b in bytes {
                    self.0 = self
                        .0
                        .wrapping_add(b as u64)
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15);
                }
            }

            #[inline]
            fn write_u64(&mut self, i: u64) {
                let mut x = i;
                x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                self.0 = x ^ (x >> 31);
            }
        }

        let mut hasher = FastHasher(0);
        id.hash(&mut hasher);
        (hasher.finish() as usize) & (NUM_SHARDS - 1)
    }

    /// Fast iterator over active (non-empty) shards using hardware trailing
    /// zeros instruction (`tzcnt`).
    #[inline]
    fn active_shards_iter(
        &self,
    ) -> impl Iterator<Item = (usize, &BTreeMap<IdentityId, BeliefState<S>>)>
    {
        let mut mask = self.active_mask;
        core::iter::from_fn(move || {
            if mask == 0 {
                None
            } else {
                let tz = mask.trailing_zeros() as usize;
                mask &= mask - 1; // Clear lowest set bit.
                Some((tz, &self.shards[tz]))
            }
        })
    }

    /// Returns the total number of elements stored across all shards.
    #[inline]
    pub fn len(&self) -> usize {
        self.total_len
    }

    /// Returns `true` if the workspace contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_len == 0
    }
}

impl<S> Default for InMemoryWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Clone + Send + Sync> ActiveWorkspace for InMemoryWorkspace<S> {
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
    /// * Time: $\mathcal{O}(\log |B_t|)$
    /// * Space: $\mathcal{O}(1)$ auxiliary allocation.
    fn insert(&mut self, belief: BeliefState<S>) {
        let idx = Self::shard_index(belief.identity);
        if self.shards[idx].insert(belief.identity, belief).is_none() {
            self.total_len += 1;
            self.active_mask |= 1 << idx;
        }
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
    /// * Time: $\mathcal{O}(\log |B_t|)$
    /// * Space: $\mathcal{O}(1)$
    fn get(&self, id: IdentityId) -> Option<&BeliefState<S>> {
        let idx = Self::shard_index(id);
        self.shards[idx].get(&id)
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
    /// * Time: $\mathcal{O}(\log |B_t|)$
    /// * Space: $\mathcal{O}(1)$
    fn get_mut(&mut self, id: IdentityId) -> Option<&mut BeliefState<S>> {
        let idx = Self::shard_index(id);
        self.shards[idx].get_mut(&id)
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
        let mut out = Vec::with_capacity(self.total_len);

        if self.total_len < PARALLEL_THRESHOLD {
            for (_, shard) in self.active_shards_iter() {
                out.extend(shard.values());
            }
        } else {
            out.par_extend(
                self.shards.par_iter().flat_map_iter(|shard| shard.values()),
            );
        }

        out
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
    /// * Time: $\mathcal{O}(|B_t| \log |B_t|)$
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
    ///     last_update: Timestamp(100),
    /// });
    ///
    /// // Evict entries older than 50 microseconds relative to timestamp 200
    /// let evicted = workspace.evict_expired(Timestamp(200), 50);
    /// assert_eq!(evicted.len(), 1);
    /// assert!(workspace.get(IdentityId(1)).is_none());
    /// ```
    fn evict_expired(
        &mut self,
        current_time: Timestamp,
        ttl_microseconds: i64,
    ) -> Vec<BeliefState<S>> {
        let evicted: Vec<BeliefState<S>> =
            if self.total_len < PARALLEL_THRESHOLD {
                let policy = TemporalEvictionPolicy;
                let mut res = Vec::new();

                let mut mask = self.active_mask;
                while mask != 0 {
                    let idx = mask.trailing_zeros() as usize;
                    mask &= mask - 1;

                    let shard = &mut self.shards[idx];
                    let mut expired_keys = Vec::new();

                    for (id, belief) in shard.iter() {
                        if policy.should_evict(
                            belief,
                            current_time,
                            ttl_microseconds,
                        ) {
                            expired_keys.push(*id);
                        }
                    }

                    for key in expired_keys {
                        if let Some(belief) = shard.remove(&key) {
                            res.push(belief);
                        }
                    }

                    if shard.is_empty() {
                        self.active_mask &= !(1 << idx);
                    }
                }
                res
            } else {
                let items: Vec<BeliefState<S>> = self
                    .shards
                    .par_iter_mut()
                    .flat_map_iter(|shard| {
                        let policy = TemporalEvictionPolicy;
                        let mut expired_keys = Vec::new();

                        for (id, belief) in shard.iter() {
                            if policy.should_evict(
                                belief,
                                current_time,
                                ttl_microseconds,
                            ) {
                                expired_keys.push(*id);
                            }
                        }

                        let mut evicted_items =
                            Vec::with_capacity(expired_keys.len());
                        for key in expired_keys {
                            if let Some(belief) = shard.remove(&key) {
                                evicted_items.push(belief);
                            }
                        }

                        evicted_items
                    })
                    .collect();

                for (idx, shard) in self.shards.iter().enumerate() {
                    if shard.is_empty() {
                        self.active_mask &= !(1 << idx);
                    }
                }

                items
            };

        self.total_len -= evicted.len();
        evicted
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
        let mut active_states = Vec::with_capacity(self.total_len);

        if self.total_len < PARALLEL_THRESHOLD {
            for (_, shard) in self.active_shards_iter() {
                active_states.extend(shard.values().cloned());
            }
        } else {
            active_states.par_extend(
                self.shards
                    .par_iter()
                    .flat_map_iter(|shard| shard.values().cloned()),
            );
        }

        WorkspaceSnapshot {
            timestamp: current_time,
            active_states,
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
            last_update: Timestamp(timestamp),
        }
    }

    #[test]
    fn test_new_workspace_is_empty() {
        let workspace = InMemoryWorkspace::<u64>::new();
        assert_eq!(workspace.len(), 0);
        assert!(workspace.is_empty());
        assert!(workspace.active_beliefs().is_empty());
        assert_eq!(workspace.shards.len(), NUM_SHARDS);
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
        assert_eq!(fetched.unwrap().identity, IdentityId(42));
        assert_eq!(fetched.unwrap().summary, 4200);
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
            last_update: Timestamp(2000),
        };

        workspace.insert(initial);
        assert_eq!(workspace.len(), 1);

        workspace.insert(updated);
        assert_eq!(workspace.len(), 1);

        let retrieved = workspace.get(IdentityId(1)).unwrap();
        assert_eq!(retrieved.summary, 9999);
        assert_eq!(retrieved.last_update, Timestamp(2000));
    }

    #[test]
    fn test_get_mut_modifies_belief_in_place() {
        let mut workspace = InMemoryWorkspace::<u64>::new();
        workspace.insert(mock_belief(1, 1000));

        if let Some(belief) = workspace.get_mut(IdentityId(1)) {
            belief.summary = 5555;
        }

        assert_eq!(workspace.get(IdentityId(1)).unwrap().summary, 5555);
    }

    #[test]
    fn test_evict_expired_selective() {
        let mut workspace = InMemoryWorkspace::<u64>::new();

        workspace.insert(mock_belief(1, 100));
        workspace.insert(mock_belief(2, 100));
        workspace.insert(mock_belief(3, 1000));

        assert_eq!(workspace.len(), 3);

        let evicted = workspace.evict_expired(Timestamp(1000), 500);

        assert_eq!(evicted.len(), 2);
        assert_eq!(workspace.len(), 1);

        assert!(workspace.get(IdentityId(1)).is_none());
        assert!(workspace.get(IdentityId(2)).is_none());
        assert!(workspace.get(IdentityId(3)).is_some());
    }

    #[test]
    fn test_evict_expired_on_empty_workspace() {
        let mut workspace = InMemoryWorkspace::<u64>::new();
        let evicted = workspace.evict_expired(Timestamp(1000), 100);

        assert!(evicted.is_empty());
        assert_eq!(workspace.len(), 0);
    }

    #[test]
    fn test_evict_expired_none_matching() {
        let mut workspace = InMemoryWorkspace::<u64>::new();
        workspace.insert(mock_belief(1, 1000));
        workspace.insert(mock_belief(2, 1000));

        let evicted = workspace.evict_expired(Timestamp(1050), 100);

        assert!(evicted.is_empty());
        assert_eq!(workspace.len(), 2);
    }

    #[test]
    fn test_evict_expired_all_matching() {
        let mut workspace = InMemoryWorkspace::<u64>::new();
        workspace.insert(mock_belief(1, 100));
        workspace.insert(mock_belief(2, 200));

        let evicted = workspace.evict_expired(Timestamp(1000), 100);

        assert_eq!(evicted.len(), 2);
        assert_eq!(workspace.len(), 0);
        assert!(workspace.is_empty());
    }

    #[test]
    fn test_create_snapshot() {
        let mut workspace = InMemoryWorkspace::<u64>::new();
        workspace.insert(mock_belief(10, 500));
        workspace.insert(mock_belief(20, 600));

        let now = Timestamp(1000);
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
    fn test_parallel_execution_above_threshold() {
        let mut workspace = InMemoryWorkspace::<u64>::new();
        let total_items = PARALLEL_THRESHOLD + 500;

        for i in 0..total_items {
            let ts = if i % 2 == 0 { 100 } else { 1000 };
            workspace.insert(mock_belief(i as u64, ts));
        }

        assert_eq!(workspace.len(), total_items);

        let active = workspace.active_beliefs();
        assert_eq!(active.len(), total_items);

        let snapshot = workspace.create_snapshot(Timestamp(2000));
        assert_eq!(snapshot.active_states.len(), total_items);

        let evicted = workspace.evict_expired(Timestamp(2000), 1000);

        let expected_evicted = total_items / 2;
        assert_eq!(evicted.len(), expected_evicted);
        assert_eq!(workspace.len(), total_items - expected_evicted);
    }
}
