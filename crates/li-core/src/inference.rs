//! Typed statistical contributions and durable inference records.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::evidence::ContentHash;
use crate::host::HostEntityRef;
use crate::ids::{
    CommitVersion, IdentityId, InferenceId, ObservationId, ProviderId,
    SchemaId,
};
use crate::probability::Probability;

/// Known or latent identity reference admitted to a gated domain.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum IdentityReference {
    /// Active latent identity hypothesis.
    Latent(IdentityId),
    /// Lightweight authoritative host-entity reference.
    Known(HostEntityRef),
}

/// One value in the finite inference domain `C(o) union {new, noise}`.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum AssociationOutcome {
    /// Assignment to a gated identity reference.
    Identity(IdentityReference),
    /// Evidence for an identity not represented by the gated candidates.
    New,
    /// Invalid, irrelevant, or non-entity evidence.
    Noise,
}

/// Statistical meaning of one numeric provider contribution.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScoreSemantics {
    /// Log probability of evidence under a declared model.
    LogLikelihood = 0,
    /// Log ratio of two declared evidence models.
    LogLikelihoodRatio = 1,
    /// General factor-graph log potential.
    LogPotential = 2,
    /// Calibrated posterior probability in `[0, 1]`.
    CalibratedPosterior = 3,
}

/// Error returned when a score lacks valid statistical semantics.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ScoreError {
    /// The numeric value was NaN or infinite.
    #[error("score contribution must be finite")]
    NonFinite,
    /// A calibrated posterior was outside `[0, 1]`.
    #[error("calibrated posterior must be in [0, 1]")]
    PosteriorOutOfRange,
    /// Statistical validity-domain description was empty.
    #[error("score validity domain must not be empty")]
    EmptyValidityDomain,
}

/// One provider-owned, statistically typed factor contribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreContribution {
    value: f64,
    semantics: ScoreSemantics,
    provider: ProviderId,
    model_version: u64,
    calibration_id: u64,
    validity_domain: Arc<str>,
}

impl ScoreContribution {
    /// Creates a finite contribution suitable for factor compilation.
    ///
    /// Raw similarities are deliberately absent from [`ScoreSemantics`]; a
    /// provider must calibrate or transform them before this constructor.
    ///
    /// # Errors
    ///
    /// Returns [`ScoreError`] when the numeric or semantic contract is
    /// invalid.
    pub fn new(
        value: f64,
        semantics: ScoreSemantics,
        provider: ProviderId,
        model_version: u64,
        calibration_id: u64,
        validity_domain: impl Into<Arc<str>>,
    ) -> Result<Self, ScoreError> {
        if !value.is_finite() {
            return Err(ScoreError::NonFinite);
        }
        if semantics == ScoreSemantics::CalibratedPosterior &&
            !(0.0..=1.0).contains(&value)
        {
            return Err(ScoreError::PosteriorOutOfRange);
        }
        let validity_domain = validity_domain.into();
        if validity_domain.is_empty() {
            return Err(ScoreError::EmptyValidityDomain);
        }
        Ok(Self {
            value,
            semantics,
            provider,
            model_version,
            calibration_id,
            validity_domain,
        })
    }

    /// Returns the finite numeric contribution.
    pub const fn value(&self) -> f64 {
        self.value
    }

    /// Returns the declared statistical semantics.
    pub const fn semantics(&self) -> ScoreSemantics {
        self.semantics
    }

    /// Returns the provider identifier.
    pub const fn provider(&self) -> ProviderId {
        self.provider
    }

    /// Returns the model version used by the provider.
    pub const fn model_version(&self) -> u64 {
        self.model_version
    }

    /// Returns the calibration artifact identifier.
    pub const fn calibration_id(&self) -> u64 {
        self.calibration_id
    }

    /// Borrows the documented statistical validity domain.
    pub fn validity_domain(&self) -> &str {
        &self.validity_domain
    }
}

/// One normalized outcome and its probability mass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutcomeProbability {
    /// Domain outcome.
    pub outcome: AssociationOutcome,
    /// Normalized probability mass.
    pub probability: Probability,
}

/// Explicit residual mass retained when alternatives are truncated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResidualMass {
    /// Probability mass omitted from individual alternatives.
    pub probability: Probability,
    /// Deterministic truncation rule and parameters.
    pub rule: Arc<str>,
}

/// Error returned when constructing a normalized finite distribution.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum DistributionError {
    /// The same outcome appeared more than once.
    #[error("inference distribution contains a duplicate outcome")]
    DuplicateOutcome,
    /// The mandatory `new` or `noise` alternative was absent.
    #[error("inference domain must contain both new and noise")]
    MissingOpenWorldOutcome,
    /// Explicit and residual probability mass did not sum to one.
    #[error("probability mass sums to {sum}, expected 1")]
    NotNormalized {
        /// Observed total probability mass.
        sum: f64,
    },
    /// A residual mass was supplied without a truncation rule.
    #[error("residual mass requires a non-empty truncation rule")]
    EmptyTruncationRule,
}

/// Canonical normalized distribution over a finite association domain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedDistribution {
    entries: Box<[OutcomeProbability]>,
    residual: Option<ResidualMass>,
}

impl NormalizedDistribution {
    /// Maximum accepted floating-point normalization error.
    pub const NORMALIZATION_TOLERANCE: f64 = 1.0e-10;

    /// Sorts, validates, and freezes a finite outcome distribution.
    ///
    /// # Arguments
    ///
    /// * `entries` - Explicit candidate, `new`, and `noise` masses.
    /// * `residual` - Omitted mass and deterministic truncation rule.
    ///
    /// # Errors
    ///
    /// Returns [`DistributionError`] for duplicate, incomplete, or
    /// non-normalized domains.
    pub fn new(
        mut entries: Vec<OutcomeProbability>,
        residual: Option<ResidualMass>,
    ) -> Result<Self, DistributionError> {
        if residual.as_ref().is_some_and(|mass| mass.rule.is_empty()) {
            return Err(DistributionError::EmptyTruncationRule);
        }
        entries
            .sort_unstable_by(|left, right| left.outcome.cmp(&right.outcome));
        if entries
            .windows(2)
            .any(|pair| pair[0].outcome == pair[1].outcome)
        {
            return Err(DistributionError::DuplicateOutcome);
        }
        let has_new = entries
            .iter()
            .any(|entry| entry.outcome == AssociationOutcome::New);
        let has_noise = entries
            .iter()
            .any(|entry| entry.outcome == AssociationOutcome::Noise);
        if !has_new || !has_noise {
            return Err(DistributionError::MissingOpenWorldOutcome);
        }

        let explicit = entries
            .iter()
            .map(|entry| entry.probability.value())
            .sum::<f64>();
        let sum = explicit +
            residual
                .as_ref()
                .map_or(0.0, |mass| mass.probability.value());
        if (sum - 1.0).abs() > Self::NORMALIZATION_TOLERANCE {
            return Err(DistributionError::NotNormalized { sum });
        }
        Ok(Self {
            entries: entries.into_boxed_slice(),
            residual,
        })
    }

    /// Borrows canonical entries without allocation.
    pub fn entries(&self) -> &[OutcomeProbability] {
        &self.entries
    }

    /// Borrows truncation metadata, when present.
    pub const fn residual(&self) -> Option<&ResidualMass> {
        self.residual.as_ref()
    }

    /// Finds an explicit outcome probability using canonical binary search.
    pub fn probability(
        &self,
        outcome: &AssociationOutcome,
    ) -> Option<Probability> {
        self.entries
            .binary_search_by(|entry| entry.outcome.cmp(outcome))
            .ok()
            .map(|index| self.entries[index].probability)
    }
}

/// Half-open durable validity interval `[begin, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionInterval {
    begin: CommitVersion,
    end: Option<CommitVersion>,
}

/// Error returned for an empty or reversed validity interval.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("validity interval end must be greater than begin")]
pub struct VersionIntervalError;

impl VersionInterval {
    /// Creates a validated half-open commit interval.
    ///
    /// # Errors
    ///
    /// Returns [`VersionIntervalError`] when a finite end is not after begin.
    pub fn new(
        begin: CommitVersion,
        end: Option<CommitVersion>,
    ) -> Result<Self, VersionIntervalError> {
        if end.is_some_and(|end| end <= begin) {
            return Err(VersionIntervalError);
        }
        Ok(Self { begin, end })
    }

    /// Creates a current interval with no closing version.
    pub const fn current(begin: CommitVersion) -> Self {
        Self { begin, end: None }
    }

    /// Returns the inclusive opening version.
    pub const fn begin(self) -> CommitVersion {
        self.begin
    }

    /// Returns the exclusive closing version.
    pub const fn end(self) -> Option<CommitVersion> {
        self.end
    }

    /// Returns whether `version` belongs to the interval.
    pub fn contains(self, version: CommitVersion) -> bool {
        version >= self.begin && self.end.is_none_or(|end| version < end)
    }
}

/// Solver completion status recorded for exactness and approximation audits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverStoppingReason {
    /// Exact algorithm completed on an acyclic factor graph.
    Exact,
    /// Approximate messages met the configured residual tolerance.
    Converged,
    /// Iteration budget was exhausted.
    IterationLimit,
    /// Approximate messages oscillated under the configured damping schedule.
    Oscillation,
    /// Solver rejected invalid numeric input.
    NumericalFailure,
}

/// Treatment of factors crossing a local inference boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryTreatment {
    /// Full global factor graph was solved.
    Global,
    /// Exact induced boundary potential was included.
    ExactInduced,
    /// No factor crossed the local boundary.
    NoCrossBoundary,
    /// Induced boundary potential is constant for relevant local states.
    ConstantInduced,
    /// Cached boundary messages approximate the induced potential.
    CachedApproximation,
    /// Boundary support or messages were truncated.
    TruncatedApproximation,
    /// Cross-boundary information was omitted.
    OmittedApproximation,
}

impl BoundaryTreatment {
    /// Returns whether this treatment preserves the global marginal by
    /// contract.
    pub const fn preserves_global_marginal(self) -> bool {
        matches!(
            self,
            Self::Global |
                Self::ExactInduced |
                Self::NoCrossBoundary |
                Self::ConstantInduced
        )
    }
}

/// Durable solver diagnostics required for approximate inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolverDiagnostics {
    /// Solver implementation/configuration version.
    pub solver_version: u64,
    /// Configured convergence tolerance.
    pub tolerance: f64,
    /// Completed full message-passing iterations.
    pub iterations: u32,
    /// Final maximum message residual.
    pub residual: f64,
    /// Applied damping values by schedule step.
    pub damping_schedule: Box<[f64]>,
    /// Exact or approximate stopping reason.
    pub stopping_reason: SolverStoppingReason,
    /// Exact or approximate treatment of local cross-boundary factors.
    pub boundary_treatment: BoundaryTreatment,
    /// Seed for deterministic replay of randomized procedures.
    pub random_seed: Option<u64>,
}

/// One provider/model/calibration artifact used by an inference run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderArtifact {
    /// Provider implementation identifier.
    pub provider: ProviderId,
    /// Provider payload schema.
    pub schema: SchemaId,
    /// Provider statistical model version.
    pub model_version: u64,
    /// Calibration artifact identifier.
    pub calibration_id: u64,
}

/// Versioned provenance shared by a durable inference record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceProvenance {
    /// Every provider/model/calibration artifact used by the factor graph.
    pub providers: Box<[ProviderArtifact]>,
    /// Candidate generation/index version.
    pub candidate_version: u64,
    /// Coherent host snapshot used for candidate generation.
    pub host_snapshot: CommitVersion,
    /// Hash of the provider/index configuration.
    pub configuration_hash: ContentHash,
}

/// Immutable durable inference result, distinct from its policy decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceRecord {
    /// Stable record identifier.
    pub id: InferenceId,
    /// Observation whose assignment was inferred.
    pub observation: ObservationId,
    /// Normalized distribution over the complete gated domain.
    pub distribution: NormalizedDistribution,
    /// Typed contributions retained for provenance and replay.
    pub contributions: Box<[ScoreContribution]>,
    /// Provider, model, calibration, candidate, and snapshot versions.
    pub provenance: Arc<InferenceProvenance>,
    /// Solver diagnostics and replay data.
    pub diagnostics: Arc<SolverDiagnostics>,
    /// Durable validity interval.
    pub validity: VersionInterval,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probability(value: f64) -> Probability {
        Probability::new(value)
    }

    #[test]
    fn score_contributions_reject_raw_invalid_numbers() {
        assert_eq!(
            ScoreContribution::new(
                f64::NAN,
                ScoreSemantics::LogPotential,
                ProviderId(1),
                1,
                1,
                "all"
            ),
            Err(ScoreError::NonFinite)
        );
        assert_eq!(
            ScoreContribution::new(
                1.1,
                ScoreSemantics::CalibratedPosterior,
                ProviderId(1),
                1,
                1,
                "all"
            ),
            Err(ScoreError::PosteriorOutOfRange)
        );
        assert!(
            ScoreContribution::new(
                -12.0,
                ScoreSemantics::LogLikelihoodRatio,
                ProviderId(1),
                2,
                3,
                "camera-v2"
            )
            .is_ok()
        );
    }

    #[test]
    fn normalized_distribution_requires_open_world_states_and_unique_mass() {
        let valid = NormalizedDistribution::new(
            vec![
                OutcomeProbability {
                    outcome: AssociationOutcome::Noise,
                    probability: probability(0.1),
                },
                OutcomeProbability {
                    outcome: AssociationOutcome::Identity(
                        IdentityReference::Latent(IdentityId(4)),
                    ),
                    probability: probability(0.7),
                },
                OutcomeProbability {
                    outcome: AssociationOutcome::New,
                    probability: probability(0.2),
                },
            ],
            None,
        );
        assert!(valid.is_ok());
        if let Ok(valid) = valid {
            assert_eq!(
                valid.probability(&AssociationOutcome::New),
                Some(probability(0.2))
            );
        }

        let missing_noise = NormalizedDistribution::new(
            vec![OutcomeProbability {
                outcome: AssociationOutcome::New,
                probability: Probability::ONE,
            }],
            None,
        );
        assert_eq!(
            missing_noise,
            Err(DistributionError::MissingOpenWorldOutcome)
        );
    }

    #[test]
    fn residual_mass_and_duplicate_edges_are_validated() {
        let duplicate = NormalizedDistribution::new(
            vec![
                OutcomeProbability {
                    outcome: AssociationOutcome::New,
                    probability: probability(0.4),
                },
                OutcomeProbability {
                    outcome: AssociationOutcome::New,
                    probability: probability(0.4),
                },
                OutcomeProbability {
                    outcome: AssociationOutcome::Noise,
                    probability: probability(0.2),
                },
            ],
            None,
        );
        assert_eq!(duplicate, Err(DistributionError::DuplicateOutcome));

        let truncated = NormalizedDistribution::new(
            vec![
                OutcomeProbability {
                    outcome: AssociationOutcome::New,
                    probability: probability(0.3),
                },
                OutcomeProbability {
                    outcome: AssociationOutcome::Noise,
                    probability: probability(0.2),
                },
            ],
            Some(ResidualMass {
                probability: probability(0.5),
                rule: Arc::from("top-0 candidates"),
            }),
        );
        assert!(truncated.is_ok());
    }

    #[test]
    fn version_intervals_are_half_open_and_nonempty() {
        assert!(
            VersionInterval::new(
                CommitVersion::new(3),
                Some(CommitVersion::new(3))
            )
            .is_err()
        );
        let interval = VersionInterval::new(
            CommitVersion::new(3),
            Some(CommitVersion::new(5)),
        );
        assert!(interval.is_ok());
        if let Ok(interval) = interval {
            assert!(interval.contains(CommitVersion::new(3)));
            assert!(!interval.contains(CommitVersion::new(5)));
        }
    }

    #[test]
    fn only_exact_boundary_conditions_claim_global_marginal_preservation() {
        assert!(BoundaryTreatment::ExactInduced.preserves_global_marginal());
        assert!(
            BoundaryTreatment::NoCrossBoundary.preserves_global_marginal()
        );
        assert!(
            !BoundaryTreatment::CachedApproximation
                .preserves_global_marginal()
        );
        assert!(
            !BoundaryTreatment::OmittedApproximation
                .preserves_global_marginal()
        );
    }
}
