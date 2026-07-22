//! Tree-backed in-memory implementation of the `ActiveWorkspace` trait.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use li_core::belief::BeliefState;
use li_core::ids::IdentityId;
use li_core::observation::Timestamp;

use crate::checkpoint::WorkspaceSnapshot;
use crate::eviction::{EvictionPolicy, TemporalEvictionPolicy};
use crate::workspace::ActiveWorkspace;

/// Holds active identity hypotheses in memory during real-time tracking,
/// providing logarithmic lookup, insertion, and key-ordered operations.
#[derive(Debug, Clone)]
pub struct InMemoryWorkspace<S> {
    /// Map linking persistent identity identifiers to their active belief
    /// states.
    pub beliefs: BTreeMap<IdentityId, BeliefState<S>>,
}

impl<S> InMemoryWorkspace<S> {
    /// Constructs a new, empty [`InMemoryWorkspace`].
    pub fn new() -> Self {
        Self {
            beliefs: BTreeMap::new(),
        }
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
    /// * Time: $\mathcal{O}(\log |B_t|)$
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
    ///     last_update: Timestamp(1000000),
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
    /// * Time: $\mathcal{O}(\log |B_t|)$
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
    /// * Time: $\mathcal{O}(\log |B_t|)$
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
        let policy = TemporalEvictionPolicy;
        let mut expired_keys = Vec::new();

        for (id, belief) in self.beliefs.iter() {
            if policy.should_evict(belief, current_time, ttl_microseconds) {
                expired_keys.push(*id);
            }
        }

        let mut evicted = Vec::with_capacity(expired_keys.len());
        for key in expired_keys {
            if let Some(belief) = self.beliefs.remove(&key) {
                evicted.push(belief);
            }
        }

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
        WorkspaceSnapshot {
            timestamp: current_time,
            active_states: self.beliefs.values().cloned().collect(),
        }
    }
}
