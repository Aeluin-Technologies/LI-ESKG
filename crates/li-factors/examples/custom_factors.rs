//! Example demonstrating multi-candidate spatial compatibility evaluation
//! and categorical $k$-factor instantiation.

use std::collections::HashMap;

use li_core::belief::BeliefState;
use li_core::ids::{IdentityId, ObservationId};
use li_core::observation::{Modality, Observation, Timestamp};
use li_core::probability::{Confidence, Probability};
use li_factors::compatibility::{
    KCandidateDistribution, MultiCandidateCompatibility,
};
use li_factors::factor::{
    CategoricalFactor, Factor, FactorError, FactorScope,
};

/// Modality-specific spatial measurement payload.
#[derive(Clone, Debug)]
pub struct CustomSpatialPayload {
    pub x: f64,
    pub y: f64,
}

/// Target summary state containing historical position estimates.
#[derive(Clone, Debug)]
pub struct CustomSummaryState {
    pub last_x: f64,
    pub last_y: f64,
}

/// Unary temporal decay factor over a single target identity scope.
pub struct UserTemporalFactor {
    pub scope_ids: Vec<IdentityId>,
    pub decay_rate: f64,
    pub delta_t: f64,
}

impl UserTemporalFactor {
    /// Constructs a new [`UserTemporalFactor`].
    ///
    /// # Arguments
    ///
    /// * `identity` - Target identity identifier.
    /// * `decay_rate` - Exponential temporal decay coefficient.
    /// * `delta_t` - Elapsed time delta in seconds.
    pub fn new(identity: IdentityId, decay_rate: f64, delta_t: f64) -> Self {
        Self {
            scope_ids: vec![identity],
            decay_rate,
            delta_t,
        }
    }
}

impl FactorScope for UserTemporalFactor {
    fn scope(&self) -> &[IdentityId] {
        &self.scope_ids
    }
}

impl Factor for UserTemporalFactor {
    fn evaluate(&self, _assignment: &[IdentityId]) -> Probability {
        let val = (-self.decay_rate * self.delta_t).exp();
        Probability::new(val)
    }
}

/// Spatial evaluator deriving joint categorical candidate distributions over
/// $k$ candidates.
pub struct UserSpatialEvaluator {
    pub max_distance: f64,
    pub background_prob: Probability,
}

impl UserSpatialEvaluator {
    fn probability(
        &self,
        observation: &Observation<CustomSpatialPayload>,
        belief: &BeliefState<CustomSummaryState>,
    ) -> Probability {
        let dx = observation.payload.x - belief.summary.last_x;
        let dy = observation.payload.y - belief.summary.last_y;
        let distance = dx.hypot(dy);

        if distance > self.max_distance {
            Probability::ZERO
        } else {
            Probability::new(1.0 - (distance / self.max_distance))
        }
    }
}

impl MultiCandidateCompatibility<CustomSpatialPayload, CustomSummaryState>
    for UserSpatialEvaluator
{
    fn evaluate_joint(
        &self,
        observation: &Observation<CustomSpatialPayload>,
        beliefs: &[&BeliefState<CustomSummaryState>],
    ) -> KCandidateDistribution {
        let mut candidate_probs = HashMap::with_capacity(beliefs.len());

        for belief in beliefs {
            candidate_probs.insert(
                belief.identity,
                self.probability(observation, belief),
            );
        }

        KCandidateDistribution::new(candidate_probs, self.background_prob)
    }

    fn evaluate_joint_stream(
        &self,
        observation: &Observation<CustomSpatialPayload>,
        beliefs: &[&BeliefState<CustomSummaryState>],
        emit: &mut dyn FnMut(IdentityId, Probability),
    ) -> Probability {
        for belief in beliefs {
            emit(belief.identity, self.probability(observation, belief));
        }
        self.background_prob
    }
}

fn main() -> Result<(), FactorError> {
    let evaluator = UserSpatialEvaluator {
        max_distance: 100.0,
        background_prob: Probability::new(0.05),
    };

    let obs = Observation::new(
        ObservationId(1),
        Modality(1),
        Timestamp::from_millis(1000),
        Confidence::new(0.95),
        CustomSpatialPayload { x: 10.0, y: 20.0 },
    );

    let belief_a = BeliefState::new(
        IdentityId(100),
        CustomSummaryState {
            last_x: 12.0,
            last_y: 20.0,
        },
        Probability::new(0.8),
        Timestamp::from_millis(900),
    );

    let belief_b = BeliefState::new(
        IdentityId(101),
        CustomSummaryState {
            last_x: 50.0,
            last_y: 20.0,
        },
        Probability::new(0.6),
        Timestamp::from_millis(900),
    );

    let candidate_beliefs = vec![&belief_a, &belief_b];

    let distribution = evaluator.evaluate_joint(&obs, &candidate_beliefs);
    let (candidate_probs, bg_prob) = distribution.into_parts();

    let factor = CategoricalFactor::new(candidate_probs, bg_prob)?;

    let score_a = factor.evaluate(&[belief_a.identity]);
    let score_b = factor.evaluate(&[belief_b.identity]);
    let score_bg = factor.evaluate(&[]);

    println!("Categorical Factor Scope: {:?}", factor.scope());
    println!(
        "Evaluation score for Identity {}: {:.4}",
        belief_a.identity.0,
        score_a.value()
    );
    println!(
        "Evaluation score for Identity {}: {:.4}",
        belief_b.identity.0,
        score_b.value()
    );
    println!(
        "Evaluation score for Unassigned/Background: {:.4}",
        score_bg.value()
    );

    assert!(score_a.value() > score_b.value());
    Ok(())
}
