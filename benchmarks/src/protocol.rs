//! Typed experimental design and metric kernels for the paper's eight RQs.

use std::collections::BTreeSet;

use thiserror::Error;

/// One falsifiable research question from the V2 evaluation protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResearchQuestion {
    /// Collective resolution against independent pairwise matching.
    Resolution,
    /// Empirical calibration of reported probabilities.
    Calibration,
    /// Recovery from false assignments and merges without history loss.
    Revision,
    /// Accuracy and latency cost of local boundary approximations.
    Locality,
    /// Throughput, latency, memory, and storage scaling.
    Scale,
    /// Exact reconstruction after injected crashes.
    Recovery,
    /// Decision equivalence and provenance distinction across providers.
    ProviderAgnosticism,
    /// Standards conformance and host-adapter round trips.
    Interoperability,
}

impl ResearchQuestion {
    /// Complete ordered RQ1-RQ8 set.
    pub const ALL: [Self; 8] = [
        Self::Resolution,
        Self::Calibration,
        Self::Revision,
        Self::Locality,
        Self::Scale,
        Self::Recovery,
        Self::ProviderAgnosticism,
        Self::Interoperability,
    ];
}

/// Required comparison system from the paper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Baseline {
    /// Independent pairwise matcher.
    IndependentPairwise,
    /// Calibrated Fellegi-Sunter model.
    FellegiSunter,
    /// Bayesian or graphical entity-resolution model.
    BayesianGraphical,
    /// Incremental linkage method.
    IncrementalLinkage,
    /// Tracking baseline for trajectory datasets.
    Tracking,
    /// LI-ESKG with collective factors removed.
    WithoutCollectiveFactors,
    /// Oracle candidate generator measuring gate upper bound.
    OracleCandidates,
}

impl Baseline {
    /// Complete required baseline set.
    pub const ALL: [Self; 7] = [
        Self::IndependentPairwise,
        Self::FellegiSunter,
        Self::BayesianGraphical,
        Self::IncrementalLinkage,
        Self::Tracking,
        Self::WithoutCollectiveFactors,
        Self::OracleCandidates,
    ];
}

/// Required resolution-quality metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResolutionMetric {
    /// Pairwise precision.
    PairwisePrecision,
    /// Pairwise recall.
    PairwiseRecall,
    /// Pairwise harmonic mean.
    PairwiseF1,
    /// B-cubed cluster score.
    BCubed,
    /// Adjusted Rand index.
    AdjustedRand,
    /// Fraction of erroneous merges.
    FalseMergeRate,
    /// Fraction of erroneous splits.
    FalseSplitRate,
    /// Precision of new-identity decisions.
    NewIdentityPrecision,
}

impl ResolutionMetric {
    /// Complete required resolution metric set.
    pub const ALL: [Self; 8] = [
        Self::PairwisePrecision,
        Self::PairwiseRecall,
        Self::PairwiseF1,
        Self::BCubed,
        Self::AdjustedRand,
        Self::FalseMergeRate,
        Self::FalseSplitRate,
        Self::NewIdentityPrecision,
    ];
}

/// Required calibration and selective-prediction metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CalibrationMetric {
    /// Negative log likelihood.
    NegativeLogLikelihood,
    /// Brier score.
    Brier,
    /// Expected calibration error.
    ExpectedCalibrationError,
    /// Reliability diagram data.
    ReliabilityDiagram,
    /// Risk-coverage curve.
    RiskCoverage,
}

impl CalibrationMetric {
    /// Complete required calibration metric set.
    pub const ALL: [Self; 5] = [
        Self::NegativeLogLikelihood,
        Self::Brier,
        Self::ExpectedCalibrationError,
        Self::ReliabilityDiagram,
        Self::RiskCoverage,
    ];
}

/// Required runtime, memory, and persistence metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SystemMetric {
    /// Observations processed per unit time.
    Throughput,
    /// Median latency.
    MedianLatency,
    /// 95th-percentile latency.
    P95Latency,
    /// 99th-percentile latency.
    P99Latency,
    /// Peak resident memory.
    PeakResidentMemory,
    /// Allocated bytes per observation.
    AllocatedBytesPerObservation,
    /// Bounded queue depth.
    QueueDepth,
    /// Checkpoint pause duration.
    CheckpointPause,
    /// Crash recovery duration.
    RecoveryTime,
    /// Durable write amplification.
    WriteAmplification,
    /// Durable storage growth.
    StorageGrowth,
}

impl SystemMetric {
    /// Complete required systems metric set.
    pub const ALL: [Self; 11] = [
        Self::Throughput,
        Self::MedianLatency,
        Self::P95Latency,
        Self::P99Latency,
        Self::PeakResidentMemory,
        Self::AllocatedBytesPerObservation,
        Self::QueueDepth,
        Self::CheckpointPause,
        Self::RecoveryTime,
        Self::WriteAmplification,
        Self::StorageGrowth,
    ];
}

/// Synthetic generator controls required to expose confounding variables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SyntheticControls {
    /// Number of ground-truth entities.
    pub identities: usize,
    /// Motion-noise standard deviation.
    pub motion_noise: f64,
    /// Sensor-noise standard deviation.
    pub sensor_noise: f64,
    /// Missing-modality probability.
    pub missing_modality_rate: f64,
    /// Correlated-error strength.
    pub correlated_error_rate: f64,
    /// Clutter observations per true observation.
    pub clutter_rate: f64,
    /// Whether arrival times vary.
    pub variable_arrival: bool,
    /// Whether dormancy and reappearance are present.
    pub dormancy_and_reappearance: bool,
    /// Whether merge and split truth events are present.
    pub merge_and_split_events: bool,
}

/// Minimum structural properties of a real evaluation dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealDatasetProfile {
    /// Dataset contains event timestamps.
    pub timestamps: bool,
    /// Dataset contains repeated observations per entity.
    pub repeated_observations: bool,
    /// Number of modalities.
    pub modalities: usize,
    /// Number of relational contexts when modalities are limited.
    pub relational_contexts: usize,
}

/// Complete reproducible experiment declaration for RQ1-RQ8.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationPlan {
    /// RQs included in the experiment.
    pub questions: BTreeSet<ResearchQuestion>,
    /// Synthetic generator controls.
    pub synthetic: SyntheticControls,
    /// Real-data structural profile.
    pub real: RealDatasetProfile,
    /// Required comparison systems.
    pub baselines: BTreeSet<Baseline>,
    /// Required resolution metrics.
    pub resolution_metrics: BTreeSet<ResolutionMetric>,
    /// Required calibration metrics.
    pub calibration_metrics: BTreeSet<CalibrationMetric>,
    /// Required systems metrics.
    pub system_metrics: BTreeSet<SystemMetric>,
    /// Ground truth is inaccessible to gating and inference.
    pub ground_truth_blinded: bool,
    /// Every provider factor is removed in at least one ablation.
    pub ablate_each_provider_factor: bool,
    /// Boundary cached, truncated, and omitted variants are evaluated.
    pub boundary_ablations: bool,
    /// Abstention, ANN, reversible merge, and provenance are ablated.
    pub architectural_ablations: bool,
    /// Crashes are injected before and after ledger writes.
    pub ledger_fault_injection: bool,
    /// Crashes are injected before and after host materialization.
    pub host_fault_injection: bool,
    /// Dependency versions, compiler settings, target, and hardware are
    /// fixed.
    pub reproducibility_metadata: bool,
}

impl EvaluationPlan {
    /// Builds the complete protocol with all mandatory design dimensions.
    pub fn canonical() -> Self {
        Self {
            questions: ResearchQuestion::ALL.into_iter().collect(),
            synthetic: SyntheticControls {
                identities: 1_000,
                motion_noise: 1.0,
                sensor_noise: 1.0,
                missing_modality_rate: 0.1,
                correlated_error_rate: 0.1,
                clutter_rate: 0.1,
                variable_arrival: true,
                dormancy_and_reappearance: true,
                merge_and_split_events: true,
            },
            real: RealDatasetProfile {
                timestamps: true,
                repeated_observations: true,
                modalities: 2,
                relational_contexts: 0,
            },
            baselines: Baseline::ALL.into_iter().collect(),
            resolution_metrics: ResolutionMetric::ALL.into_iter().collect(),
            calibration_metrics: CalibrationMetric::ALL.into_iter().collect(),
            system_metrics: SystemMetric::ALL.into_iter().collect(),
            ground_truth_blinded: true,
            ablate_each_provider_factor: true,
            boundary_ablations: true,
            architectural_ablations: true,
            ledger_fault_injection: true,
            host_fault_injection: true,
            reproducibility_metadata: true,
        }
    }

    /// Validates mandatory datasets, baselines, ablations, and fault points.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError`] when the design could not answer all RQs or
    /// leaks ground truth into inference.
    pub fn validate(&self) -> Result<(), ProtocolError> {
        if !ResearchQuestion::ALL
            .iter()
            .all(|question| self.questions.contains(question))
        {
            return Err(ProtocolError::MissingResearchQuestion);
        }
        if self.synthetic.identities == 0 ||
            !valid_rate(self.synthetic.missing_modality_rate) ||
            !valid_rate(self.synthetic.correlated_error_rate) ||
            !valid_rate(self.synthetic.clutter_rate) ||
            !valid_non_negative(self.synthetic.motion_noise) ||
            !valid_non_negative(self.synthetic.sensor_noise) ||
            !self.synthetic.variable_arrival ||
            !self.synthetic.dormancy_and_reappearance ||
            !self.synthetic.merge_and_split_events
        {
            return Err(ProtocolError::IncompleteSyntheticControls);
        }
        if !self.real.timestamps ||
            !self.real.repeated_observations ||
            (self.real.modalities < 2 && self.real.relational_contexts < 2)
        {
            return Err(ProtocolError::InvalidRealDataset);
        }
        if !self.ground_truth_blinded {
            return Err(ProtocolError::GroundTruthLeakage);
        }
        if !Baseline::ALL
            .iter()
            .all(|baseline| self.baselines.contains(baseline))
        {
            return Err(ProtocolError::MissingBaseline);
        }
        if !ResolutionMetric::ALL
            .iter()
            .all(|metric| self.resolution_metrics.contains(metric)) ||
            !CalibrationMetric::ALL
                .iter()
                .all(|metric| self.calibration_metrics.contains(metric)) ||
            !SystemMetric::ALL
                .iter()
                .all(|metric| self.system_metrics.contains(metric))
        {
            return Err(ProtocolError::MissingMetric);
        }
        if !self.ablate_each_provider_factor ||
            !self.boundary_ablations ||
            !self.architectural_ablations
        {
            return Err(ProtocolError::MissingAblation);
        }
        if !self.ledger_fault_injection || !self.host_fault_injection {
            return Err(ProtocolError::MissingFaultPoint);
        }
        if !self.reproducibility_metadata {
            return Err(ProtocolError::MissingReproducibilityMetadata);
        }
        Ok(())
    }
}

/// Binary pairwise confusion counts used for resolution metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairwiseCounts {
    /// Correctly predicted coreferent pairs.
    pub true_positive: u64,
    /// Incorrectly merged pairs.
    pub false_positive: u64,
    /// Missed coreferent pairs.
    pub false_negative: u64,
}

impl PairwiseCounts {
    /// Returns pairwise precision, using zero for an empty prediction set.
    pub fn precision(self) -> f64 {
        ratio(
            self.true_positive,
            self.true_positive.saturating_add(self.false_positive),
        )
    }

    /// Returns pairwise recall, using zero when no positive truth exists.
    pub fn recall(self) -> f64 {
        ratio(
            self.true_positive,
            self.true_positive.saturating_add(self.false_negative),
        )
    }

    /// Returns the harmonic mean of pairwise precision and recall.
    pub fn f1(self) -> f64 {
        let precision = self.precision();
        let recall = self.recall();
        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }
}

/// One calibrated binary forecast and its observed outcome.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationSample {
    probability: f64,
    outcome: bool,
}

impl CalibrationSample {
    /// Creates a valid binary forecast.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError::InvalidProbability`] outside `[0, 1]`.
    pub fn new(
        probability: f64,
        outcome: bool,
    ) -> Result<Self, ProtocolError> {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(ProtocolError::InvalidProbability);
        }
        Ok(Self {
            probability,
            outcome,
        })
    }
}

/// Proper scoring rules and expected calibration error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationMetrics {
    /// Mean negative log likelihood.
    pub negative_log_likelihood: f64,
    /// Mean binary Brier score.
    pub brier: f64,
    /// Equal-width expected calibration error.
    pub expected_calibration_error: f64,
}

impl CalibrationMetrics {
    /// Computes NLL, Brier score, and equal-width ECE in one pass.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError::EmptySamples`] or
    /// [`ProtocolError::InvalidBinCount`] for invalid inputs.
    pub fn compute(
        samples: &[CalibrationSample],
        bins: usize,
    ) -> Result<Self, ProtocolError> {
        if samples.is_empty() {
            return Err(ProtocolError::EmptySamples);
        }
        if bins == 0 {
            return Err(ProtocolError::InvalidBinCount);
        }
        let mut nll = 0.0;
        let mut brier = 0.0;
        let mut counts = vec![0_u64; bins];
        let mut probability_sums = vec![0.0_f64; bins];
        let mut outcome_sums = vec![0_u64; bins];
        for sample in samples {
            let target = if sample.outcome { 1.0 } else { 0.0 };
            let selected = if sample.outcome {
                sample.probability
            } else {
                1.0 - sample.probability
            };
            nll -= selected.max(f64::MIN_POSITIVE).ln();
            let residual = sample.probability - target;
            brier = residual.mul_add(residual, brier);
            let scaled = sample.probability * bins as f64;
            let index = (scaled as usize).min(bins - 1);
            counts[index] = counts[index].saturating_add(1);
            probability_sums[index] += sample.probability;
            outcome_sums[index] =
                outcome_sums[index].saturating_add(u64::from(sample.outcome));
        }
        let count = samples.len() as f64;
        let mut ece = 0.0;
        for index in 0..bins {
            if counts[index] == 0 {
                continue;
            }
            let bin_count = counts[index] as f64;
            let confidence = probability_sums[index] / bin_count;
            let frequency = outcome_sums[index] as f64 / bin_count;
            ece += (bin_count / count) * (confidence - frequency).abs();
        }
        Ok(Self {
            negative_log_likelihood: nll / count,
            brier: brier / count,
            expected_calibration_error: ece,
        })
    }
}

/// One RQ5 scale cell and its measured systems response.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleMeasurement {
    /// Active identity count.
    pub active_identities: usize,
    /// Candidate count per observation.
    pub candidates: usize,
    /// Collective batch size.
    pub batch_size: usize,
    /// Maximum factor arity.
    pub factor_arity: usize,
    /// Solver iteration count.
    pub iterations: u32,
    /// Observations processed per second.
    pub throughput: f64,
    /// Median latency in microseconds.
    pub p50_micros: f64,
    /// 95th-percentile latency in microseconds.
    pub p95_micros: f64,
    /// 99th-percentile latency in microseconds.
    pub p99_micros: f64,
    /// Peak resident bytes.
    pub peak_resident_bytes: u64,
    /// Durable storage bytes.
    pub storage_bytes: u64,
}

impl ScaleMeasurement {
    /// Validates positive dimensions and monotonic finite latency quantiles.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError::InvalidScaleMeasurement`] for malformed data.
    pub fn validate(self) -> Result<Self, ProtocolError> {
        if self.active_identities == 0 ||
            self.batch_size == 0 ||
            self.factor_arity == 0 ||
            self.iterations == 0 ||
            !valid_non_negative(self.throughput) ||
            self.throughput == 0.0 ||
            !valid_non_negative(self.p50_micros) ||
            !valid_non_negative(self.p95_micros) ||
            !valid_non_negative(self.p99_micros) ||
            self.p50_micros > self.p95_micros ||
            self.p95_micros > self.p99_micros
        {
            return Err(ProtocolError::InvalidScaleMeasurement);
        }
        Ok(self)
    }
}

/// Invalid experimental design or metric input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ProtocolError {
    /// At least one RQ was omitted.
    #[error("evaluation plan must cover RQ1 through RQ8")]
    MissingResearchQuestion,
    /// Synthetic generator does not control every mandated dimension.
    #[error("synthetic generator controls are incomplete or invalid")]
    IncompleteSyntheticControls,
    /// Real dataset lacks time, repetition, or multimodal/relational context.
    #[error("real dataset does not satisfy the minimum protocol")]
    InvalidRealDataset,
    /// Ground truth would be visible to gating or inference.
    #[error("ground truth must remain blinded from candidate generation")]
    GroundTruthLeakage,
    /// A required baseline is absent.
    #[error("evaluation plan omits a required baseline")]
    MissingBaseline,
    /// A required quality, calibration, or systems metric is absent.
    #[error("evaluation plan omits a required metric")]
    MissingMetric,
    /// A required ablation is absent.
    #[error("evaluation plan omits a required ablation")]
    MissingAblation,
    /// A required crash-injection boundary is absent.
    #[error("evaluation plan omits a required fault-injection boundary")]
    MissingFaultPoint,
    /// Dependency, build, target, or hardware metadata is absent.
    #[error("evaluation plan lacks reproducibility metadata")]
    MissingReproducibilityMetadata,
    /// Forecast probability was outside `[0, 1]`.
    #[error("calibration probability must be finite and within [0, 1]")]
    InvalidProbability,
    /// Metric computation received no observations.
    #[error("metric computation requires at least one sample")]
    EmptySamples,
    /// ECE requires at least one bin.
    #[error("calibration bin count must be positive")]
    InvalidBinCount,
    /// RQ5 dimensions or measurements were invalid.
    #[error("scale measurement dimensions or quantiles are invalid")]
    InvalidScaleMeasurement,
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn valid_rate(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn valid_non_negative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_plan_covers_the_complete_protocol() {
        assert_eq!(EvaluationPlan::canonical().validate(), Ok(()));
    }

    #[test]
    fn pairwise_metrics_match_hand_computed_values() {
        let counts = PairwiseCounts {
            true_positive: 3,
            false_positive: 1,
            false_negative: 2,
        };
        assert!((counts.precision() - 0.75).abs() < f64::EPSILON);
        assert!((counts.recall() - 0.6).abs() < f64::EPSILON);
        assert!((counts.f1() - 2.0 / 3.0).abs() < f64::EPSILON);
    }
}
