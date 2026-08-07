//! Validatable append-only resolution commands and dependency-aware revisions.

use std::collections::BTreeSet;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::decision::DecisionRecord;
use crate::evidence::ObservationEnvelope;
use crate::host::{HostEntityRef, HostRelation};
use crate::ids::{
    CommitVersion, DecisionId, IdentityId, MaterializationId, ObservationId,
    TransactionId,
};
use crate::inference::InferenceRecord;
use crate::observation::Timestamp;

/// Stable replay key supplied by a command producer.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
)]
pub struct IdempotencyKey([u8; 16]);

impl IdempotencyKey {
    /// Creates a non-zero replay key.
    ///
    /// # Errors
    ///
    /// Returns [`CommandError::ZeroIdempotencyKey`] for the reserved all-zero
    /// key.
    pub fn new(bytes: [u8; 16]) -> Result<Self, CommandError> {
        if bytes == [0; 16] {
            return Err(CommandError::ZeroIdempotencyKey);
        }
        Ok(Self(bytes))
    }

    /// Returns the exact stable key bytes.
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

/// Reversal behavior when active dependents exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RevertMode {
    /// Reject the command when the target has active dependents.
    Strict,
    /// Close the complete transitive dependency closure.
    Cascade,
    /// Append an explicit semantic replacement when inversion is impossible.
    Compensating,
}

/// One non-empty split partition and its observation allocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitPartition {
    /// Identity that represents this partition after the split.
    pub identity: IdentityId,
    /// Observations retained or reassigned into the partition.
    pub observations: Box<[ObservationId]>,
}

/// Complete dependency-aware split plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitPlan {
    /// Active merge decisions whose validity intervals must close.
    pub close_merges: Box<[DecisionId]>,
    /// Intended identity/observation partition.
    pub partitions: Box<[SplitPartition]>,
    /// Downstream transactions to invalidate or mark ambiguous.
    pub invalidate_dependents: Box<[TransactionId]>,
}

/// Structural command construction error.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CommandError {
    /// Reserved all-zero idempotency key was supplied.
    #[error("idempotency key must not be all zero")]
    ZeroIdempotencyKey,
    /// Merge endpoints were identical.
    #[error("merge endpoints must be distinct")]
    SelfMerge,
    /// Split did not contain at least two non-empty partitions.
    #[error("split requires at least two non-empty partitions")]
    IncompleteSplit,
    /// An identity or observation appeared in more than one split partition.
    #[error("split partitions must be disjoint")]
    OverlappingSplit,
    /// Atomic transaction did not contain a command.
    #[error("resolution transaction must contain at least one command")]
    EmptyBatch,
}

impl SplitPlan {
    /// Creates a complete intended split partition and dependency plan.
    ///
    /// # Errors
    ///
    /// Returns [`CommandError`] for fewer than two non-empty partitions or for
    /// repeated identity/observation membership.
    pub fn new(
        close_merges: Vec<DecisionId>,
        partitions: Vec<SplitPartition>,
        invalidate_dependents: Vec<TransactionId>,
    ) -> Result<Self, CommandError> {
        if partitions.len() < 2 ||
            partitions.iter().any(|part| part.observations.is_empty())
        {
            return Err(CommandError::IncompleteSplit);
        }
        let mut identities = BTreeSet::new();
        let mut observations = BTreeSet::new();
        for partition in &partitions {
            if !identities.insert(partition.identity) {
                return Err(CommandError::OverlappingSplit);
            }
            for observation in &partition.observations {
                if !observations.insert(*observation) {
                    return Err(CommandError::OverlappingSplit);
                }
            }
        }
        Ok(Self {
            close_merges: close_merges.into_boxed_slice(),
            partitions: partitions.into_boxed_slice(),
            invalidate_dependents: invalidate_dependents.into_boxed_slice(),
        })
    }
}

/// Valid structural command payload; contextual ledger checks happen at
/// commit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResolutionCommand {
    /// Append immutable evidence before inference.
    PersistObservation(ObservationEnvelope),
    /// Atomically append one inference and its distinct policy decision.
    CommitResolution {
        /// Durable normalized inference record.
        inference: Arc<InferenceRecord>,
        /// Durable policy decision supported by `inference`.
        decision: Arc<DecisionRecord>,
        /// Identity allocated when the selected action is `CreateIdentity`.
        created_identity: Option<IdentityId>,
    },
    /// Create a latent identity explicitly.
    CreateIdentity {
        /// New stable identity identifier.
        identity: IdentityId,
        /// Event time used for the lifecycle descriptor.
        created_at: Timestamp,
    },
    /// Promote a latent identity to an authoritative host entity.
    Promote {
        /// Latent identity leaving the active workspace.
        identity: IdentityId,
        /// Authoritative promotion target.
        target: HostEntityRef,
    },
    /// Append an explicit merge decision with supporting decisions.
    Merge {
        /// Stable merge decision identifier.
        decision: DecisionId,
        /// First merge-class member.
        left: IdentityId,
        /// Second merge-class member.
        right: IdentityId,
        /// Association decisions supporting the merge.
        support: Box<[DecisionId]>,
        /// Merge policy version.
        policy_version: u64,
        /// Merge loss model or operating-point version.
        loss_version: u64,
    },
    /// Apply an intended split and dependent invalidations.
    Split(SplitPlan),
    /// Remove an active identity from the hot workspace.
    Dormant(IdentityId),
    /// Return a dormant identity to the hot workspace.
    Reactivate(IdentityId),
    /// Exclude an identity from ordinary retrieval.
    Retire(IdentityId),
    /// Revert, cascade-close, or compensate an earlier transaction.
    Revert {
        /// Earlier transaction being revised.
        transaction: TransactionId,
        /// Dependency handling strategy.
        mode: RevertMode,
    },
    /// Record a successful idempotent host materialization receipt.
    Materialized {
        /// Outbox/materialization identifier.
        materialization: MaterializationId,
        /// Decision responsible for the host write.
        decision: DecisionId,
    },
    /// Append a transactional outbox plan without mutating the host graph.
    EnqueueMaterialization {
        /// Stable outbox/materialization identifier.
        materialization: MaterializationId,
        /// Decision responsible for the future host write.
        decision: DecisionId,
        /// Native host relation to write idempotently.
        relation: HostRelation,
    },
}

impl ResolutionCommand {
    /// Creates a structurally valid non-self merge command.
    ///
    /// # Errors
    ///
    /// Returns [`CommandError::SelfMerge`] for identical endpoints.
    pub fn merge(
        decision: DecisionId,
        left: IdentityId,
        right: IdentityId,
        support: Vec<DecisionId>,
        policy_version: u64,
        loss_version: u64,
    ) -> Result<Self, CommandError> {
        if left == right {
            return Err(CommandError::SelfMerge);
        }
        Ok(Self::Merge {
            decision,
            left,
            right,
            support: support.into_boxed_slice(),
            policy_version,
            loss_version,
        })
    }
}

/// Command envelope carrying conflict and deterministic replay metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommandEnvelope {
    /// Producer-assigned transaction identifier.
    transaction: TransactionId,
    /// Snapshot version from which all scores and decisions were computed.
    expected_version: CommitVersion,
    /// Stable replay key.
    idempotency: IdempotencyKey,
    /// Ingestion timestamp of the command.
    issued_at: Timestamp,
    /// Earlier transactions whose outputs were consumed by this batch.
    depends_on: Box<[TransactionId]>,
    /// Atomic command batch made visible at one commit version.
    commands: Box<[ResolutionCommand]>,
}

impl CommandEnvelope {
    /// Creates a non-empty atomic command envelope.
    ///
    /// # Errors
    ///
    /// Returns [`CommandError::EmptyBatch`] when no command is supplied.
    pub fn new(
        transaction: TransactionId,
        expected_version: CommitVersion,
        idempotency: IdempotencyKey,
        issued_at: Timestamp,
        depends_on: Vec<TransactionId>,
        commands: Vec<ResolutionCommand>,
    ) -> Result<Self, CommandError> {
        if commands.is_empty() {
            return Err(CommandError::EmptyBatch);
        }
        Ok(Self {
            transaction,
            expected_version,
            idempotency,
            issued_at,
            depends_on: depends_on.into_boxed_slice(),
            commands: commands.into_boxed_slice(),
        })
    }

    /// Returns the producer transaction identifier.
    pub const fn transaction(&self) -> TransactionId {
        self.transaction
    }

    /// Returns the coherent snapshot version used by the command producer.
    pub const fn expected_version(&self) -> CommitVersion {
        self.expected_version
    }

    /// Returns the stable replay key.
    pub const fn idempotency(&self) -> IdempotencyKey {
        self.idempotency
    }

    /// Returns the command ingestion timestamp.
    pub const fn issued_at(&self) -> Timestamp {
        self.issued_at
    }

    /// Borrows explicit transaction dependencies.
    pub fn depends_on(&self) -> &[TransactionId] {
        &self.depends_on
    }

    /// Borrows the atomic command batch.
    pub fn commands(&self) -> &[ResolutionCommand] {
        &self.commands
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idempotency_and_merge_smart_constructors_reject_impossible_values() {
        assert_eq!(
            IdempotencyKey::new([0; 16]),
            Err(CommandError::ZeroIdempotencyKey)
        );
        assert!(IdempotencyKey::new([1; 16]).is_ok());
        assert_eq!(
            ResolutionCommand::merge(
                DecisionId(1),
                IdentityId(1),
                IdentityId(1),
                Vec::new(),
                1,
                1,
            ),
            Err(CommandError::SelfMerge)
        );
        let key = IdempotencyKey::new([1; 16]);
        assert!(key.is_ok());
        if let Ok(key) = key {
            assert_eq!(
                CommandEnvelope::new(
                    TransactionId(1),
                    CommitVersion::ZERO,
                    key,
                    Timestamp::UNIX_EPOCH,
                    Vec::new(),
                    Vec::new(),
                ),
                Err(CommandError::EmptyBatch)
            );
        }
    }

    #[test]
    fn split_plan_requires_disjoint_nonempty_partitions() {
        let incomplete = SplitPlan::new(
            Vec::new(),
            vec![SplitPartition {
                identity: IdentityId(1),
                observations: vec![ObservationId(1)].into_boxed_slice(),
            }],
            Vec::new(),
        );
        assert_eq!(incomplete, Err(CommandError::IncompleteSplit));

        let overlap = SplitPlan::new(
            Vec::new(),
            vec![
                SplitPartition {
                    identity: IdentityId(1),
                    observations: vec![ObservationId(1)].into_boxed_slice(),
                },
                SplitPartition {
                    identity: IdentityId(2),
                    observations: vec![ObservationId(1)].into_boxed_slice(),
                },
            ],
            Vec::new(),
        );
        assert_eq!(overlap, Err(CommandError::OverlappingSplit));

        let valid = SplitPlan::new(
            Vec::new(),
            vec![
                SplitPartition {
                    identity: IdentityId(1),
                    observations: vec![ObservationId(1)].into_boxed_slice(),
                },
                SplitPartition {
                    identity: IdentityId(2),
                    observations: vec![ObservationId(2)].into_boxed_slice(),
                },
            ],
            Vec::new(),
        );
        assert!(valid.is_ok());
    }
}
