//! Cache-oriented V2 active workspace and reusable worker scratch storage.

use std::sync::Arc;

use arc_swap::ArcSwap;
use bytes::Bytes;
use hashbrown::HashMap;
use li_core::{
    BoundedHistory, CommitVersion, ContentHash, IdentityId, IdentityReference,
    ObservationEnvelope, ObservationId, ProviderId, SchemaId, Timestamp,
};
use li_factors::{CandidateBuffer, FactorBuffer};
use thiserror::Error;

/// Provider-owned opaque sufficient or bounded summary handle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummaryHandle {
    /// Provider responsible for interpreting the bytes.
    pub provider: ProviderId,
    /// Provider payload schema.
    pub schema: SchemaId,
    /// Cheaply cloned opaque key or immutable summary bytes.
    pub opaque_key: Bytes,
    /// Provider summary version.
    pub version: u64,
    /// Digest protecting the provider-owned representation.
    pub content_hash: ContentHash,
}

/// Compact active identity state; durable audit data remains in the ledger.
#[derive(Debug, Clone, PartialEq)]
pub struct HotIdentity {
    identity: IdentityId,
    summaries: Vec<SummaryHandle>,
    history: BoundedHistory<ObservationId>,
    last_activation: Timestamp,
    applied_version: CommitVersion,
}

impl HotIdentity {
    /// Creates compact hot state with reserved provider and history bounds.
    pub fn new(
        identity: IdentityId,
        summary_capacity: usize,
        history_capacity: usize,
        last_activation: Timestamp,
        applied_version: CommitVersion,
    ) -> Self {
        Self {
            identity,
            summaries: Vec::with_capacity(summary_capacity),
            history: BoundedHistory::new(history_capacity),
            last_activation,
            applied_version,
        }
    }

    /// Returns the latent identity identifier.
    pub const fn identity(&self) -> IdentityId {
        self.identity
    }

    /// Borrows provider summaries without copying their bytes.
    pub fn summaries(&self) -> &[SummaryHandle] {
        &self.summaries
    }

    /// Borrows bounded recent observation history.
    pub const fn history(&self) -> &BoundedHistory<ObservationId> {
        &self.history
    }

    /// Returns the latest activation event time.
    pub const fn last_activation(&self) -> Timestamp {
        self.last_activation
    }

    /// Returns the ledger version applied to this cache entry.
    pub const fn applied_version(&self) -> CommitVersion {
        self.applied_version
    }

    /// Replaces or appends one provider/schema summary without a hash lookup.
    pub fn upsert_summary(&mut self, summary: SummaryHandle) {
        if let Some(existing) = self.summaries.iter_mut().find(|existing| {
            existing.provider == summary.provider &&
                existing.schema == summary.schema
        }) {
            *existing = summary;
        } else {
            self.summaries.push(summary);
        }
    }
}

/// Proof object produced only after an association ledger commit succeeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommittedAssociation {
    /// Assigned observation.
    pub observation: ObservationId,
    /// Selected active identity.
    pub identity: IdentityId,
    /// Commit version that made the decision current.
    pub version: CommitVersion,
    /// Observation event time used for activation ordering.
    pub event_time: Timestamp,
}

/// Error returned when cache state would diverge from the durable ledger.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum WorkspaceError {
    /// Association referenced an identity absent from the hot workspace.
    #[error("active identity {0:?} is missing from the workspace")]
    MissingIdentity(IdentityId),
    /// Cache update attempted to apply an older or duplicate commit.
    #[error("workspace commit version must advance monotonically")]
    StaleCommit,
}

/// Active V2 belief cache with bounded history and allocation reuse.
#[derive(Debug)]
pub struct HotWorkspace {
    identities: HashMap<IdentityId, HotIdentity>,
}

impl HotWorkspace {
    /// Creates an empty workspace with reserved identity capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            identities: HashMap::with_capacity(capacity),
        }
    }

    /// Inserts or replaces one reconstructed active cache entry.
    pub fn insert(&mut self, identity: HotIdentity) -> Option<HotIdentity> {
        self.identities.insert(identity.identity, identity)
    }

    /// Borrows an active cache entry.
    pub fn get(&self, identity: IdentityId) -> Option<&HotIdentity> {
        self.identities.get(&identity)
    }

    /// Iterates over active cache entries without allocating or cloning.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &HotIdentity> {
        self.identities.values()
    }

    /// Applies a committed association; proposals cannot mutate workspace
    /// state.
    ///
    /// # Errors
    ///
    /// Returns [`WorkspaceError`] if the identity is missing or the durable
    /// commit does not advance the entry version.
    pub fn apply_committed(
        &mut self,
        association: CommittedAssociation,
    ) -> Result<Option<ObservationId>, WorkspaceError> {
        let state = self
            .identities
            .get_mut(&association.identity)
            .ok_or(WorkspaceError::MissingIdentity(association.identity))?;
        if association.version <= state.applied_version {
            return Err(WorkspaceError::StaleCommit);
        }
        state.applied_version = association.version;
        state.last_activation =
            state.last_activation.max(association.event_time);
        Ok(state.history.push(association.observation))
    }

    /// Evicts identities whose last activation is strictly before `cutoff`.
    ///
    /// Removed entries are appended to `output`, which is cleared but retains
    /// its allocation for the next eviction pass.
    pub fn evict_before(
        &mut self,
        cutoff: Timestamp,
        output: &mut Vec<HotIdentity>,
    ) {
        output.clear();
        self.identities
            .extract_if(|_, state| state.last_activation < cutoff)
            .for_each(|(_, state)| output.push(state));
    }

    /// Returns the number of active cache entries.
    pub fn len(&self) -> usize {
        self.identities.len()
    }

    /// Returns whether the active cache is empty.
    pub fn is_empty(&self) -> bool {
        self.identities.is_empty()
    }
}

/// Atomically published read-mostly provider or candidate-index snapshot.
#[derive(Debug)]
pub struct PublishedSnapshot<T> {
    current: ArcSwap<T>,
}

impl<T> PublishedSnapshot<T> {
    /// Publishes an initial immutable snapshot.
    pub fn new(snapshot: T) -> Self {
        Self {
            current: ArcSwap::from_pointee(snapshot),
        }
    }

    /// Loads one coherent snapshot for a complete inference request.
    pub fn load(&self) -> Arc<T> {
        self.current.load_full()
    }

    /// Atomically replaces the snapshot for later requests.
    pub fn publish(&self, snapshot: T) {
        self.current.store(Arc::new(snapshot));
    }
}

/// Per-worker allocation pool reused across complete collective batches.
#[derive(Debug)]
pub struct WorkerScratch {
    /// Loaded immutable observation envelopes.
    pub observations: Vec<ObservationEnvelope>,
    /// Flat canonical candidate ranges.
    pub candidates: CandidateBuffer,
    /// Per-provider candidate output reused before canonical aggregation.
    pub provider_candidates: CandidateBuffer,
    /// Reused aggregation groups for multiple providers.
    pub candidate_groups: Vec<Vec<IdentityReference>>,
    /// Validated dense factor tables.
    pub factors: FactorBuffer,
    /// Contiguous log-domain solver messages.
    pub messages: Vec<f64>,
    /// Batch-local factor adjacency offsets.
    pub adjacency_offsets: Vec<u32>,
    /// Reusable partial-selection indexes.
    pub selection: Vec<u32>,
}

impl WorkerScratch {
    /// Creates reusable buffers sized from measured deployment percentiles.
    pub fn with_capacity(
        batch: usize,
        candidates: usize,
        factors: usize,
        messages: usize,
    ) -> Self {
        Self {
            observations: Vec::with_capacity(batch),
            candidates: CandidateBuffer::with_capacity(batch, candidates),
            provider_candidates: CandidateBuffer::with_capacity(
                batch, candidates,
            ),
            candidate_groups: (0..batch)
                .map(|_| {
                    Vec::with_capacity(
                        candidates.checked_div(batch.max(1)).unwrap_or(0),
                    )
                })
                .collect(),
            factors: FactorBuffer::with_capacity(factors),
            messages: Vec::with_capacity(messages),
            adjacency_offsets: Vec::with_capacity(factors.saturating_add(1)),
            selection: Vec::with_capacity(candidates),
        }
    }

    /// Clears a completed batch without releasing any allocation.
    pub fn reset(&mut self, batch_len: usize) {
        self.observations.clear();
        self.candidates.reset(batch_len);
        self.provider_candidates.reset(batch_len);
        if self.candidate_groups.len() < batch_len {
            self.candidate_groups.resize_with(batch_len, Vec::new);
        }
        for group in &mut self.candidate_groups[..batch_len] {
            group.clear();
        }
        self.factors.clear();
        self.messages.clear();
        self.adjacency_offsets.clear();
        self.selection.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hot(identity: u64, history: usize, version: u64) -> HotIdentity {
        HotIdentity::new(
            IdentityId(identity),
            2,
            history,
            Timestamp::from_micros(10),
            CommitVersion::new(version),
        )
    }

    #[test]
    fn workspace_updates_only_with_advancing_commits() {
        let mut workspace = HotWorkspace::with_capacity(1);
        workspace.insert(hot(1, 1, 1));
        let committed = CommittedAssociation {
            observation: ObservationId(2),
            identity: IdentityId(1),
            version: CommitVersion::new(2),
            event_time: Timestamp::from_micros(20),
        };
        assert_eq!(workspace.apply_committed(committed), Ok(None));
        assert_eq!(
            workspace.apply_committed(committed),
            Err(WorkspaceError::StaleCommit)
        );
        assert_eq!(
            workspace
                .get(IdentityId(1))
                .map(HotIdentity::last_activation),
            Some(Timestamp::from_micros(20))
        );
    }

    #[test]
    fn bounded_history_and_eviction_reuse_output_capacity() {
        let mut workspace = HotWorkspace::with_capacity(2);
        workspace.insert(hot(1, 1, 0));
        workspace.insert(hot(2, 1, 0));
        let mut evicted = Vec::with_capacity(2);
        let capacity = evicted.capacity();
        workspace.evict_before(Timestamp::from_micros(11), &mut evicted);
        assert_eq!(evicted.len(), 2);
        assert_eq!(evicted.capacity(), capacity);
        assert!(workspace.is_empty());
    }

    #[test]
    fn published_snapshot_is_coherent_and_replaceable() {
        let snapshot = PublishedSnapshot::new(vec![1_u8, 2]);
        let first = snapshot.load();
        snapshot.publish(vec![3]);
        assert_eq!(first.as_slice(), &[1, 2]);
        assert_eq!(snapshot.load().as_slice(), &[3]);
    }

    #[test]
    fn values_borrows_every_active_identity_without_copying() {
        let mut workspace = HotWorkspace::with_capacity(2);
        workspace.insert(hot(1, 1, 0));
        workspace.insert(hot(2, 1, 0));
        let identity_sum = workspace
            .values()
            .map(|identity| identity.identity().0)
            .sum::<u64>();
        assert_eq!(identity_sum, 3);
        assert_eq!(workspace.values().len(), 2);
    }

    #[test]
    fn worker_scratch_retains_top_level_allocations() {
        let mut scratch = WorkerScratch::with_capacity(4, 8, 3, 16);
        let capacities = (
            scratch.observations.capacity(),
            scratch.messages.capacity(),
            scratch.selection.capacity(),
        );
        scratch.reset(4);
        assert_eq!(
            capacities,
            (
                scratch.observations.capacity(),
                scratch.messages.capacity(),
                scratch.selection.capacity(),
            )
        );
    }
}
