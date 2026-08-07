//! Allocation-reusing log-domain Sum-Product solver for collective factors.

use li_core::{
    AssociationOutcome, BoundaryTreatment, NormalizedDistribution,
    OutcomeProbability, Probability, SolverDiagnostics, SolverStoppingReason,
};
use li_factors::{CandidateBuffer, FactorTable};
use petgraph::unionfind::UnionFind;
use thiserror::Error;

/// Error returned by factor-graph compilation or numeric inference.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SolverError {
    /// Factor referenced a variable outside the candidate batch.
    #[error("factor variable {variable} is outside batch length {batch_len}")]
    VariableOutOfBounds {
        /// Invalid batch-local variable index.
        variable: usize,
        /// Number of variables in the batch.
        batch_len: usize,
    },
    /// Factor cardinality differs from the candidate domain cardinality.
    #[error("factor cardinality does not match variable domain")]
    CardinalityMismatch,
    /// Message buffer size overflowed `usize`.
    #[error("factor graph message storage size overflow")]
    CapacityOverflow,
    /// Solver configuration contains invalid numeric or iteration bounds.
    #[error("invalid solver configuration")]
    InvalidConfiguration,
    /// Log-domain normalization encountered no finite support.
    #[error("factor graph has no finite assignment support")]
    NoFiniteSupport,
    /// Final marginal could not satisfy the normalized distribution contract.
    #[error("posterior distribution is invalid: {0}")]
    Distribution(#[from] li_core::DistributionError),
}

/// Deterministic loopy or exact-tree Sum-Product configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SumProductConfig {
    /// Stable solver implementation/configuration version.
    pub solver_version: u64,
    /// Maximum full iterations for cyclic graphs.
    pub max_iterations: u32,
    /// Maximum probability-space message residual for convergence.
    pub tolerance: f64,
    /// Constant damping in `[0, 1)` applied to new messages.
    pub damping: f64,
    /// Log prior shared by each gated candidate.
    pub candidate_log_prior: f64,
    /// Explicit log prior mass for `new`.
    pub new_log_prior: f64,
    /// Explicit log prior mass for `noise`.
    pub noise_log_prior: f64,
    /// Exact or approximate handling of excluded cross-boundary factors.
    pub boundary_treatment: BoundaryTreatment,
}

impl SumProductConfig {
    /// Validates finite priors, iteration count, tolerance, and damping.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::InvalidConfiguration`] for invalid bounds.
    pub fn validate(self) -> Result<Self, SolverError> {
        let priors = [
            self.candidate_log_prior,
            self.new_log_prior,
            self.noise_log_prior,
        ];
        if self.max_iterations == 0 ||
            !self.tolerance.is_finite() ||
            self.tolerance <= 0.0 ||
            !self.damping.is_finite() ||
            !(0.0..1.0).contains(&self.damping) ||
            priors.iter().any(|value| !value.is_finite())
        {
            return Err(SolverError::InvalidConfiguration);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone)]
struct Edge {
    variable: usize,
    factor: usize,
    factor_position: usize,
    message_offset: usize,
    cardinality: usize,
}

/// Reusable message and topology storage for one worker.
#[derive(Debug, Default)]
pub struct SolverScratch {
    edges: Vec<Edge>,
    factor_edges: Vec<Vec<usize>>,
    variable_edges: Vec<Vec<usize>>,
    variable_cardinalities: Vec<usize>,
    factor_strides: Vec<Vec<usize>>,
    variable_to_factor: Vec<f64>,
    factor_to_variable: Vec<f64>,
    next_messages: Vec<f64>,
    marginals: Vec<f64>,
}

impl SolverScratch {
    /// Clears graph-specific contents while retaining top-level allocations.
    pub fn clear(&mut self) {
        self.edges.clear();
        for edges in &mut self.factor_edges {
            edges.clear();
        }
        self.factor_edges.clear();
        for edges in &mut self.variable_edges {
            edges.clear();
        }
        self.variable_edges.clear();
        self.variable_cardinalities.clear();
        for strides in &mut self.factor_strides {
            strides.clear();
        }
        self.factor_strides.clear();
        self.variable_to_factor.clear();
        self.factor_to_variable.clear();
        self.next_messages.clear();
        self.marginals.clear();
    }
}

/// Batch marginal result with exactness and convergence diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchPosterior {
    /// One normalized assignment distribution per observation.
    pub distributions: Box<[NormalizedDistribution]>,
    /// Exact or approximate solver diagnostics.
    pub diagnostics: SolverDiagnostics,
}

/// Log-domain Sum-Product implementation over validated dense factors.
#[derive(Debug, Clone, Copy)]
pub struct SumProductSolver {
    config: SumProductConfig,
}

impl SumProductSolver {
    /// Creates a solver after validating all numeric configuration.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::InvalidConfiguration`] for invalid settings.
    pub fn new(config: SumProductConfig) -> Result<Self, SolverError> {
        Ok(Self {
            config: config.validate()?,
        })
    }

    /// Solves a collective batch, reusing caller-owned message storage.
    ///
    /// Domains place canonical gated candidates first, then `new`, then
    /// `noise`. Acyclic bipartite graphs run enough synchronous passes for
    /// exact tree marginals; cyclic graphs obey `max_iterations`.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError`] for invalid topology, cardinality, or numeric
    /// support.
    pub fn solve(
        self,
        candidates: &CandidateBuffer,
        factors: &[FactorTable],
        scratch: &mut SolverScratch,
    ) -> Result<BatchPosterior, SolverError> {
        self.compile(candidates, factors, scratch)?;
        let (acyclic, exact_iterations) =
            self.topology(candidates.observation_count(), scratch);
        let iteration_limit = if acyclic {
            exact_iterations.max(1)
        } else {
            self.config.max_iterations
        };
        // Damping changes the fixed-point trajectory and therefore prevents a
        // finite tree schedule from being exact. It remains available only for
        // cyclic graphs, where the result is explicitly approximate.
        let effective_damping =
            if acyclic { 0.0 } else { self.config.damping };
        let mut residual = f64::INFINITY;
        let mut completed = 0_u32;
        for iteration in 0..iteration_limit {
            residual =
                self.factor_pass(factors, scratch, effective_damping)?;
            residual =
                residual.max(self.variable_pass(scratch, effective_damping)?);
            completed = iteration.saturating_add(1);
            if !acyclic && residual <= self.config.tolerance {
                break;
            }
        }

        let stopping_reason = if acyclic {
            SolverStoppingReason::Exact
        } else if residual <= self.config.tolerance {
            SolverStoppingReason::Converged
        } else {
            SolverStoppingReason::IterationLimit
        };
        let distributions = self.collect(candidates, scratch)?;
        let damping_len = usize::try_from(completed)
            .map_err(|_| SolverError::CapacityOverflow)?;
        Ok(BatchPosterior {
            distributions: distributions.into_boxed_slice(),
            diagnostics: SolverDiagnostics {
                solver_version: self.config.solver_version,
                tolerance: self.config.tolerance,
                iterations: completed,
                residual,
                damping_schedule: vec![effective_damping; damping_len]
                    .into_boxed_slice(),
                stopping_reason,
                boundary_treatment: self.config.boundary_treatment,
                random_seed: None,
            },
        })
    }

    fn compile(
        self,
        candidates: &CandidateBuffer,
        factors: &[FactorTable],
        scratch: &mut SolverScratch,
    ) -> Result<(), SolverError> {
        scratch.clear();
        let variables = candidates.observation_count();
        scratch.variable_edges.resize_with(variables, Vec::new);
        scratch.variable_cardinalities.reserve(variables);
        for variable in 0..variables {
            let cardinality = candidates
                .get(variable)
                .map_or(2, |domain| domain.len().saturating_add(2));
            scratch.variable_cardinalities.push(cardinality);
        }
        scratch.factor_edges.resize_with(factors.len(), Vec::new);
        scratch.factor_strides.reserve(factors.len());

        let mut message_len = 0_usize;
        for (factor_index, factor) in factors.iter().enumerate() {
            let scope_len = factor.variables().len();
            let mut strides = vec![1_usize; scope_len];
            if scope_len > 1 {
                for position in (0..scope_len - 1).rev() {
                    let next = position + 1;
                    strides[position] = strides[next]
                        .checked_mul(usize::from(factor.cardinalities()[next]))
                        .ok_or(SolverError::CapacityOverflow)?;
                }
            }
            scratch.factor_strides.push(strides);
            for (position, variable) in factor.variables().iter().enumerate() {
                let variable = usize::try_from(*variable).map_err(|_| {
                    SolverError::VariableOutOfBounds {
                        variable: usize::MAX,
                        batch_len: variables,
                    }
                })?;
                if variable >= variables {
                    return Err(SolverError::VariableOutOfBounds {
                        variable,
                        batch_len: variables,
                    });
                }
                let cardinality = scratch.variable_cardinalities[variable];
                if cardinality != usize::from(factor.cardinalities()[position])
                {
                    return Err(SolverError::CardinalityMismatch);
                }
                let edge_index = scratch.edges.len();
                scratch.edges.push(Edge {
                    variable,
                    factor: factor_index,
                    factor_position: position,
                    message_offset: message_len,
                    cardinality,
                });
                scratch.factor_edges[factor_index].push(edge_index);
                scratch.variable_edges[variable].push(edge_index);
                message_len = message_len
                    .checked_add(cardinality)
                    .ok_or(SolverError::CapacityOverflow)?;
            }
        }
        scratch.variable_to_factor.resize(message_len, 0.0);
        scratch.factor_to_variable.resize(message_len, 0.0);
        scratch.next_messages.resize(message_len, 0.0);
        Ok(())
    }

    /// Detects cycles and returns a safe exact pass bound for the largest
    /// connected component.
    fn topology(
        self,
        variables: usize,
        scratch: &SolverScratch,
    ) -> (bool, u32) {
        let nodes = variables.saturating_add(scratch.factor_edges.len());
        let mut components = UnionFind::new(nodes);
        let mut acyclic = true;
        for edge in &scratch.edges {
            let factor = variables.saturating_add(edge.factor);
            if !components.union(edge.variable, factor) {
                acyclic = false;
            }
        }
        let mut component_edges = vec![0_usize; nodes];
        for edge in &scratch.edges {
            let root = components.find(edge.variable);
            component_edges[root] = component_edges[root].saturating_add(1);
        }
        let largest = component_edges.into_iter().max().unwrap_or(0);
        let passes = u32::try_from(largest.saturating_add(1))
            .unwrap_or(u32::MAX)
            .max(1);
        (acyclic, passes)
    }

    fn factor_pass(
        self,
        factors: &[FactorTable],
        scratch: &mut SolverScratch,
        damping: f64,
    ) -> Result<f64, SolverError> {
        let mut maximum_residual = 0.0_f64;
        for edge_index in 0..scratch.edges.len() {
            let edge = &scratch.edges[edge_index];
            let factor = &factors[edge.factor];
            let range =
                edge.message_offset..edge.message_offset + edge.cardinality;
            scratch.next_messages[range.clone()].fill(f64::NEG_INFINITY);
            let strides = &scratch.factor_strides[edge.factor];

            for (flat_index, potential) in
                factor.log_potentials().iter().enumerate()
            {
                let state = (flat_index / strides[edge.factor_position]) %
                    edge.cardinality;
                let mut value = *potential;
                for other_edge_index in &scratch.factor_edges[edge.factor] {
                    if *other_edge_index == edge_index {
                        continue;
                    }
                    let other = &scratch.edges[*other_edge_index];
                    let other_state = (flat_index /
                        strides[other.factor_position]) %
                        other.cardinality;
                    value += scratch.variable_to_factor
                        [other.message_offset + other_state];
                }
                let output =
                    &mut scratch.next_messages[edge.message_offset + state];
                *output = log_add_exp(*output, value);
            }
            normalize_log(&mut scratch.next_messages[range.clone()])?;
            maximum_residual = maximum_residual.max(damp_and_residual(
                &scratch.next_messages[range.clone()],
                &mut scratch.factor_to_variable[range],
                damping,
            )?);
        }
        Ok(maximum_residual)
    }

    fn variable_pass(
        self,
        scratch: &mut SolverScratch,
        damping: f64,
    ) -> Result<f64, SolverError> {
        let mut maximum_residual = 0.0_f64;
        for edge_index in 0..scratch.edges.len() {
            let edge = &scratch.edges[edge_index];
            let range =
                edge.message_offset..edge.message_offset + edge.cardinality;
            for state in 0..edge.cardinality {
                let mut value = self.log_prior(edge.cardinality, state);
                for other_edge_index in &scratch.variable_edges[edge.variable]
                {
                    if *other_edge_index == edge_index {
                        continue;
                    }
                    let other = &scratch.edges[*other_edge_index];
                    value += scratch.factor_to_variable
                        [other.message_offset + state];
                }
                scratch.next_messages[edge.message_offset + state] = value;
            }
            normalize_log(&mut scratch.next_messages[range.clone()])?;
            maximum_residual = maximum_residual.max(damp_and_residual(
                &scratch.next_messages[range.clone()],
                &mut scratch.variable_to_factor[range],
                damping,
            )?);
        }
        Ok(maximum_residual)
    }

    fn log_prior(self, cardinality: usize, state: usize) -> f64 {
        if state.saturating_add(2) == cardinality {
            self.config.new_log_prior
        } else if state.saturating_add(1) == cardinality {
            self.config.noise_log_prior
        } else {
            self.config.candidate_log_prior
        }
    }

    fn collect(
        self,
        candidates: &CandidateBuffer,
        scratch: &mut SolverScratch,
    ) -> Result<Vec<NormalizedDistribution>, SolverError> {
        let mut output =
            Vec::with_capacity(scratch.variable_cardinalities.len());
        for variable in 0..scratch.variable_cardinalities.len() {
            let cardinality = scratch.variable_cardinalities[variable];
            scratch.marginals.clear();
            scratch.marginals.resize(cardinality, 0.0);
            for state in 0..cardinality {
                let mut value = self.log_prior(cardinality, state);
                for edge_index in &scratch.variable_edges[variable] {
                    let edge = &scratch.edges[*edge_index];
                    value += scratch.factor_to_variable
                        [edge.message_offset + state];
                }
                scratch.marginals[state] = value;
            }
            normalize_log(&mut scratch.marginals)?;
            let domain = candidates.get(variable).unwrap_or(&[]);
            let mut entries = Vec::with_capacity(cardinality);
            for (state, reference) in domain.iter().enumerate() {
                entries.push(OutcomeProbability {
                    outcome: AssociationOutcome::Identity(reference.clone()),
                    probability: Probability::new(
                        scratch.marginals[state].exp(),
                    ),
                });
            }
            entries.push(OutcomeProbability {
                outcome: AssociationOutcome::New,
                probability: Probability::new(
                    scratch.marginals[cardinality - 2].exp(),
                ),
            });
            entries.push(OutcomeProbability {
                outcome: AssociationOutcome::Noise,
                probability: Probability::new(
                    scratch.marginals[cardinality - 1].exp(),
                ),
            });
            output.push(NormalizedDistribution::new(entries, None)?);
        }
        Ok(output)
    }
}

fn log_add_exp(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY {
        return right;
    }
    if right == f64::NEG_INFINITY {
        return left;
    }
    let maximum = left.max(right);
    maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
}

fn normalize_log(values: &mut [f64]) -> Result<(), SolverError> {
    let normalizer =
        values.iter().copied().fold(f64::NEG_INFINITY, log_add_exp);
    if !normalizer.is_finite() {
        return Err(SolverError::NoFiniteSupport);
    }
    for value in values {
        *value -= normalizer;
    }
    Ok(())
}

fn damp_and_residual(
    next: &[f64],
    current: &mut [f64],
    damping: f64,
) -> Result<f64, SolverError> {
    let mut residual = 0.0_f64;
    for (next_value, current_value) in next.iter().zip(current.iter_mut()) {
        let old_probability = current_value.exp();
        let new_probability = next_value.exp();
        let mixed = damping
            .mul_add(old_probability, (1.0 - damping) * new_probability);
        if !mixed.is_finite() || mixed <= 0.0 {
            return Err(SolverError::NoFiniteSupport);
        }
        residual = residual.max((new_probability - old_probability).abs());
        *current_value = mixed.ln();
    }
    normalize_log(current)?;
    Ok(residual)
}

#[cfg(test)]
mod tests {
    use li_core::{IdentityId, IdentityReference};
    use smallvec::SmallVec;

    use super::*;

    fn candidates() -> Result<CandidateBuffer, li_factors::ProviderError> {
        let mut candidates = CandidateBuffer::with_capacity(2, 2);
        candidates.reset(2);
        candidates.push_observation(
            0,
            2,
            [IdentityReference::Latent(IdentityId(1))],
        )?;
        candidates.push_observation(
            1,
            2,
            [IdentityReference::Latent(IdentityId(2))],
        )?;
        Ok(candidates)
    }

    fn config() -> SumProductConfig {
        SumProductConfig {
            solver_version: 2,
            max_iterations: 20,
            tolerance: 1.0e-10,
            damping: 0.0,
            candidate_log_prior: 0.0,
            new_log_prior: -1.0,
            noise_log_prior: -2.0,
            boundary_treatment: BoundaryTreatment::Global,
        }
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        let mut invalid = config();
        invalid.max_iterations = 0;
        assert!(matches!(
            SumProductSolver::new(invalid),
            Err(SolverError::InvalidConfiguration)
        ));
    }

    #[test]
    fn acyclic_collective_factor_reports_exact_marginals()
    -> Result<(), Box<dyn std::error::Error>> {
        let candidates = candidates()?;
        let factor = FactorTable::new(
            SmallVec::from_slice(&[0, 1]),
            SmallVec::from_slice(&[3, 3]),
            vec![0.0, -4.0, -4.0, -4.0, 0.0, -4.0, -4.0, -4.0, 0.0],
            Vec::new(),
        )?;
        let mut tree_config = config();
        tree_config.damping = 0.75;
        let solver = SumProductSolver::new(tree_config)?;
        let posterior = solver.solve(
            &candidates,
            &[factor],
            &mut SolverScratch::default(),
        )?;
        assert_eq!(
            posterior.diagnostics.stopping_reason,
            SolverStoppingReason::Exact
        );
        assert_eq!(posterior.distributions.len(), 2);
        assert!(
            posterior
                .diagnostics
                .damping_schedule
                .iter()
                .all(|damping| *damping == 0.0)
        );
        for distribution in &posterior.distributions {
            let sum = distribution
                .entries()
                .iter()
                .map(|entry| entry.probability.value())
                .sum::<f64>();
            assert!(
                (sum - 1.0).abs() <=
                    NormalizedDistribution::NORMALIZATION_TOLERANCE
            );
        }
        Ok(())
    }

    #[test]
    fn disconnected_unary_forest_uses_largest_component_pass_bound()
    -> Result<(), Box<dyn std::error::Error>> {
        const VARIABLES: usize = 128;
        let mut candidates = CandidateBuffer::with_capacity(VARIABLES, 0);
        candidates.reset(VARIABLES);
        let mut factors = Vec::with_capacity(VARIABLES);
        for index in 0..VARIABLES {
            candidates.push_observation(index, VARIABLES, [])?;
            factors.push(FactorTable::new(
                SmallVec::from_slice(&[u32::try_from(index)?]),
                SmallVec::from_slice(&[2]),
                vec![0.0, -1.0],
                Vec::new(),
            )?);
        }
        let posterior = SumProductSolver::new(config())?.solve(
            &candidates,
            &factors,
            &mut SolverScratch::default(),
        )?;
        assert_eq!(posterior.diagnostics.iterations, 2);
        assert_eq!(
            posterior.diagnostics.stopping_reason,
            SolverStoppingReason::Exact
        );
        Ok(())
    }

    #[test]
    fn cardinality_mismatch_fails_before_message_updates()
    -> Result<(), Box<dyn std::error::Error>> {
        let candidates = candidates()?;
        let factor = FactorTable::new(
            SmallVec::from_slice(&[0]),
            SmallVec::from_slice(&[2]),
            vec![0.0, 0.0],
            Vec::new(),
        )?;
        let solver = SumProductSolver::new(config())?;
        assert_eq!(
            solver.solve(
                &candidates,
                &[factor],
                &mut SolverScratch::default()
            ),
            Err(SolverError::CardinalityMismatch)
        );
        Ok(())
    }

    #[test]
    fn cyclic_graph_reports_approximation_and_retains_damping()
    -> Result<(), Box<dyn std::error::Error>> {
        let candidates = candidates()?;
        let potentials =
            vec![0.0, -3.0, -3.0, -3.0, 0.0, -3.0, -3.0, -3.0, 0.0];
        let first = FactorTable::new(
            SmallVec::from_slice(&[0, 1]),
            SmallVec::from_slice(&[3, 3]),
            potentials.clone(),
            Vec::new(),
        )?;
        let second = FactorTable::new(
            SmallVec::from_slice(&[0, 1]),
            SmallVec::from_slice(&[3, 3]),
            potentials,
            Vec::new(),
        )?;
        let mut loopy_config = config();
        loopy_config.max_iterations = 2;
        loopy_config.damping = 0.25;
        loopy_config.boundary_treatment =
            BoundaryTreatment::TruncatedApproximation;
        let posterior = SumProductSolver::new(loopy_config)?.solve(
            &candidates,
            &[first, second],
            &mut SolverScratch::default(),
        )?;

        assert_ne!(
            posterior.diagnostics.stopping_reason,
            SolverStoppingReason::Exact
        );
        assert!(
            posterior
                .diagnostics
                .damping_schedule
                .iter()
                .all(|damping| *damping == 0.25)
        );
        assert_eq!(
            posterior.diagnostics.boundary_treatment,
            BoundaryTreatment::TruncatedApproximation
        );
        Ok(())
    }

    #[test]
    fn exact_tree_matches_exhaustive_factor_product()
    -> Result<(), Box<dyn std::error::Error>> {
        let candidates = candidates()?;
        let unary_left = [-0.3, -1.2, -2.1];
        let unary_right = [-0.7, -0.2, -2.4];
        let pair = [0.4, -0.6, -1.0, -0.8, 0.5, -1.1, -1.2, -1.3, 0.2];
        let factors = [
            FactorTable::new(
                SmallVec::from_slice(&[0]),
                SmallVec::from_slice(&[3]),
                unary_left.to_vec(),
                Vec::new(),
            )?,
            FactorTable::new(
                SmallVec::from_slice(&[1]),
                SmallVec::from_slice(&[3]),
                unary_right.to_vec(),
                Vec::new(),
            )?,
            FactorTable::new(
                SmallVec::from_slice(&[0, 1]),
                SmallVec::from_slice(&[3, 3]),
                pair.to_vec(),
                Vec::new(),
            )?,
        ];
        let posterior = SumProductSolver::new(config())?.solve(
            &candidates,
            &factors,
            &mut SolverScratch::default(),
        )?;

        let priors = [0.0_f64, -1.0, -2.0];
        let mut left = [0.0_f64; 3];
        let mut right = [0.0_f64; 3];
        let mut normalizer = 0.0_f64;
        for left_state in 0..3 {
            for right_state in 0..3 {
                let weight = (priors[left_state] +
                    priors[right_state] +
                    unary_left[left_state] +
                    unary_right[right_state] +
                    pair[left_state * 3 + right_state])
                    .exp();
                left[left_state] += weight;
                right[right_state] += weight;
                normalizer += weight;
            }
        }
        for value in &mut left {
            *value /= normalizer;
        }
        for value in &mut right {
            *value /= normalizer;
        }
        let outcomes = [
            AssociationOutcome::Identity(IdentityReference::Latent(
                IdentityId(1),
            )),
            AssociationOutcome::New,
            AssociationOutcome::Noise,
        ];
        for (state, outcome) in outcomes.iter().enumerate() {
            let computed_left = posterior.distributions[0]
                .probability(outcome)
                .map(|probability| probability.value());
            assert!(computed_left.is_some_and(|value| {
                (value - left[state]).abs() <= 1.0e-10
            }));
        }
        let right_outcomes = [
            AssociationOutcome::Identity(IdentityReference::Latent(
                IdentityId(2),
            )),
            AssociationOutcome::New,
            AssociationOutcome::Noise,
        ];
        for (state, outcome) in right_outcomes.iter().enumerate() {
            let computed_right = posterior.distributions[1]
                .probability(outcome)
                .map(|probability| probability.value());
            assert!(computed_right.is_some_and(|value| {
                (value - right[state]).abs() <= 1.0e-10
            }));
        }
        Ok(())
    }

    #[test]
    fn isolated_variable_uses_explicit_new_and_noise_priors()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut candidates = CandidateBuffer::default();
        candidates.reset(1);
        candidates.push_observation(0, 1, [])?;
        let posterior = SumProductSolver::new(config())?.solve(
            &candidates,
            &[],
            &mut SolverScratch::default(),
        )?;
        let distribution = &posterior.distributions[0];
        let new = distribution.probability(&AssociationOutcome::New);
        let noise = distribution.probability(&AssociationOutcome::Noise);
        assert!(new.is_some_and(|new| noise.is_some_and(|noise| new > noise)));
        Ok(())
    }
}
