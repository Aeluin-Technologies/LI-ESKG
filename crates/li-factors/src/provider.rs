//! Provider-agnostic candidate and collective factor buffers.

use std::ops::Range;

use li_core::{
    CommitVersion, IdentityReference, ObservationEnvelope, ProviderArtifact,
    ProviderId, ScoreContribution,
};
use smallvec::SmallVec;
use thiserror::Error;

/// Error returned by provider dispatch or reusable buffer validation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ProviderError {
    /// Provider emitted candidates for an observation outside the batch.
    #[error("observation index {index} is outside batch length {batch_len}")]
    ObservationOutOfBounds { index: usize, batch_len: usize },
    /// Provider emitted candidate groups out of observation order.
    #[error(
        "candidate groups must be emitted in non-decreasing observation order"
    )]
    CandidateOrder,
    /// Factor scope was empty.
    #[error("factor scope must not be empty")]
    EmptyFactorScope,
    /// Factor referenced one batch-local variable more than once.
    #[error("factor scope contains a duplicate variable")]
    DuplicateFactorVariable,
    /// Factor cardinality was zero or overflowed the dense table size.
    #[error("invalid factor cardinality")]
    InvalidCardinality,
    /// Dense log-potential table length did not match scope cardinalities.
    #[error("factor table expects {expected} values, received {actual}")]
    FactorTableLength { expected: usize, actual: usize },
    /// Factor contained NaN or infinite log potentials.
    #[error("factor log potentials must be finite")]
    NonFinitePotential,
    /// Provider-specific failure with a stable diagnostic code.
    #[error("provider {provider:?} failed with code {code}")]
    Provider { provider: ProviderId, code: u32 },
}

/// Coherent read-only versions held for a complete provider batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderContext {
    /// Authoritative host snapshot version.
    pub host_snapshot: CommitVersion,
    /// Candidate index snapshot/configuration version.
    pub candidate_snapshot: u64,
}

/// Flat candidate storage with one canonical range per observation.
#[derive(Debug, Default, Clone)]
pub struct CandidateBuffer {
    offsets: Vec<u32>,
    candidates: Vec<IdentityReference>,
    next_observation: usize,
}

impl CandidateBuffer {
    /// Creates reusable candidate storage with reserved batch and candidate
    /// capacity.
    pub fn with_capacity(
        batch_capacity: usize,
        candidate_capacity: usize,
    ) -> Self {
        Self {
            offsets: Vec::with_capacity(batch_capacity.saturating_add(1)),
            candidates: Vec::with_capacity(candidate_capacity),
            next_observation: 0,
        }
    }

    /// Clears logical contents while retaining all allocations.
    pub fn reset(&mut self, batch_len: usize) {
        self.offsets.clear();
        self.offsets.reserve(batch_len.saturating_add(1));
        self.offsets.push(0);
        self.candidates.clear();
        self.next_observation = 0;
    }

    /// Appends one observation's candidates and canonicalizes duplicates.
    ///
    /// Providers must call this exactly once for each observation, including
    /// empty candidate groups, in ascending observation order.
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError`] for an invalid or out-of-order index.
    pub fn push_observation<I>(
        &mut self,
        observation_index: usize,
        batch_len: usize,
        candidates: I,
    ) -> Result<(), ProviderError>
    where
        I: IntoIterator<Item = IdentityReference>,
    {
        if observation_index >= batch_len {
            return Err(ProviderError::ObservationOutOfBounds {
                index: observation_index,
                batch_len,
            });
        }
        if observation_index != self.next_observation {
            return Err(ProviderError::CandidateOrder);
        }
        let start = self.candidates.len();
        self.candidates.extend(candidates);
        self.candidates[start..].sort_unstable();
        let end_before_dedup = self.candidates.len();
        let mut write = start;
        for read in start..end_before_dedup {
            if write == start ||
                self.candidates[read] != self.candidates[write - 1]
            {
                self.candidates.swap(write, read);
                write = write.saturating_add(1);
            }
        }
        self.candidates.truncate(write);
        let end = u32::try_from(self.candidates.len())
            .map_err(|_| ProviderError::InvalidCardinality)?;
        self.offsets.push(end);
        self.next_observation = self.next_observation.saturating_add(1);
        Ok(())
    }

    /// Borrows candidates for one observation without allocation.
    pub fn get(
        &self,
        observation_index: usize,
    ) -> Option<&[IdentityReference]> {
        let start =
            usize::try_from(*self.offsets.get(observation_index)?).ok()?;
        let end = usize::try_from(
            *self.offsets.get(observation_index.saturating_add(1))?,
        )
        .ok()?;
        self.candidates.get(start..end)
    }

    /// Returns the number of completed observation groups.
    pub fn observation_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Returns the total number of canonical candidates.
    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }
}

/// Dense factor table over batch-local assignment variables.
#[derive(Debug, Clone, PartialEq)]
pub struct FactorTable {
    variables: SmallVec<[u32; 4]>,
    cardinalities: SmallVec<[u16; 4]>,
    log_potentials: Box<[f64]>,
    contributions: Box<[ScoreContribution]>,
}

impl FactorTable {
    /// Creates and validates a dense row-major log-potential table.
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError`] for empty scope, invalid cardinalities, table
    /// size mismatch, or non-finite potentials.
    pub fn new(
        variables: SmallVec<[u32; 4]>,
        cardinalities: SmallVec<[u16; 4]>,
        log_potentials: Vec<f64>,
        contributions: Vec<ScoreContribution>,
    ) -> Result<Self, ProviderError> {
        if variables.is_empty() || variables.len() != cardinalities.len() {
            return Err(ProviderError::EmptyFactorScope);
        }
        let mut canonical_variables = variables.clone();
        canonical_variables.sort_unstable();
        if canonical_variables
            .windows(2)
            .any(|pair| pair[0] == pair[1])
        {
            return Err(ProviderError::DuplicateFactorVariable);
        }
        let mut expected = 1_usize;
        for cardinality in &cardinalities {
            if *cardinality == 0 {
                return Err(ProviderError::InvalidCardinality);
            }
            expected = expected
                .checked_mul(usize::from(*cardinality))
                .ok_or(ProviderError::InvalidCardinality)?;
        }
        if expected != log_potentials.len() {
            return Err(ProviderError::FactorTableLength {
                expected,
                actual: log_potentials.len(),
            });
        }
        if log_potentials.iter().any(|value| !value.is_finite()) {
            return Err(ProviderError::NonFinitePotential);
        }
        Ok(Self {
            variables,
            cardinalities,
            log_potentials: log_potentials.into_boxed_slice(),
            contributions: contributions.into_boxed_slice(),
        })
    }

    /// Borrows batch-local variable indexes in scope order.
    pub fn variables(&self) -> &[u32] {
        &self.variables
    }

    /// Borrows domain cardinalities in scope order.
    pub fn cardinalities(&self) -> &[u16] {
        &self.cardinalities
    }

    /// Borrows the dense row-major log-potential table.
    pub fn log_potentials(&self) -> &[f64] {
        &self.log_potentials
    }

    /// Borrows typed statistical contributions retained for provenance.
    pub fn contributions(&self) -> &[ScoreContribution] {
        &self.contributions
    }
}

/// Caller-owned collection of compiled provider factor tables.
#[derive(Debug, Default)]
pub struct FactorBuffer {
    factors: Vec<FactorTable>,
}

impl FactorBuffer {
    /// Creates a reusable factor buffer.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            factors: Vec::with_capacity(capacity),
        }
    }

    /// Clears factors while retaining table-vector capacity.
    pub fn clear(&mut self) {
        self.factors.clear();
    }

    /// Appends one validated factor table.
    pub fn push(&mut self, factor: FactorTable) {
        self.factors.push(factor);
    }

    /// Borrows all emitted factors.
    pub fn as_slice(&self) -> &[FactorTable] {
        &self.factors
    }

    /// Returns a factor subrange for provider-specific batching.
    pub fn range(&self, range: Range<usize>) -> Option<&[FactorTable]> {
        self.factors.get(range)
    }
}

/// Opaque provider boundary dispatched once per batch, not per candidate.
pub trait FactorProvider: Send + Sync {
    /// Returns provider, schema, model, and calibration provenance.
    fn artifact(&self) -> ProviderArtifact;

    /// Returns the stable provider implementation identifier.
    fn provider_id(&self) -> ProviderId {
        self.artifact().provider
    }

    /// Emits finite gated identity candidates into caller-owned storage.
    fn generate_candidates(
        &self,
        batch: &[ObservationEnvelope],
        context: ProviderContext,
        output: &mut CandidateBuffer,
    ) -> Result<(), ProviderError>;

    /// Emits typed unary or collective factors into caller-owned storage.
    fn emit_factors(
        &self,
        batch: &[ObservationEnvelope],
        candidates: &CandidateBuffer,
        context: ProviderContext,
        output: &mut FactorBuffer,
    ) -> Result<(), ProviderError>;
}

#[cfg(test)]
mod tests {
    use li_core::{IdentityId, ScoreSemantics};

    use super::*;

    #[test]
    fn candidate_buffer_canonicalizes_and_reuses_capacity() {
        let mut buffer = CandidateBuffer::with_capacity(2, 4);
        buffer.reset(2);
        let capacity = buffer.candidates.capacity();
        assert!(
            buffer
                .push_observation(
                    0,
                    2,
                    [
                        IdentityReference::Latent(IdentityId(2)),
                        IdentityReference::Latent(IdentityId(1)),
                        IdentityReference::Latent(IdentityId(2)),
                    ],
                )
                .is_ok()
        );
        assert!(buffer.push_observation(1, 2, []).is_ok());
        assert_eq!(
            buffer.get(0),
            Some(
                [
                    IdentityReference::Latent(IdentityId(1)),
                    IdentityReference::Latent(IdentityId(2)),
                ]
                .as_slice()
            )
        );
        buffer.reset(2);
        assert_eq!(buffer.candidates.capacity(), capacity);
    }

    #[test]
    fn candidate_groups_reject_out_of_order_and_out_of_range_indexes() {
        let mut buffer = CandidateBuffer::default();
        buffer.reset(2);
        assert_eq!(
            buffer.push_observation(1, 2, []),
            Err(ProviderError::CandidateOrder)
        );
        assert_eq!(
            buffer.push_observation(2, 2, []),
            Err(ProviderError::ObservationOutOfBounds {
                index: 2,
                batch_len: 2
            })
        );
    }

    #[test]
    fn factor_table_checks_dense_shape_and_finite_values() {
        let contribution = ScoreContribution::new(
            -1.0,
            ScoreSemantics::LogPotential,
            ProviderId(1),
            1,
            1,
            "test",
        );
        assert!(contribution.is_ok());
        let contributions = contribution.into_iter().collect();
        let valid = FactorTable::new(
            SmallVec::from_slice(&[0, 1]),
            SmallVec::from_slice(&[2, 3]),
            vec![0.0; 6],
            contributions,
        );
        assert!(valid.is_ok());
        assert_eq!(
            FactorTable::new(
                SmallVec::from_slice(&[0]),
                SmallVec::from_slice(&[2]),
                vec![0.0],
                Vec::new(),
            ),
            Err(ProviderError::FactorTableLength {
                expected: 2,
                actual: 1
            })
        );
    }
}
