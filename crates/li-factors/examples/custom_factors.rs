//! Example demonstrating custom spatial and temporal factor implementations.

use li_core::belief::BeliefState;
use li_core::ids::IdentityId;
use li_core::observation::{Observation, Timestamp};
use li_core::probability::Probability;
use li_factors::compatibility::PairwiseCompatibility;
use li_factors::factor::{Factor, FactorScope};

#[derive(Clone)]
pub struct CustomSpatialPayload {
    pub x: f64,
    pub y: f64,
}

#[derive(Clone)]
pub struct CustomSummaryState {
    pub last_x: f64,
    pub last_y: f64,
}

pub struct UserTemporalFactor {
    pub scope_ids: [IdentityId; 1],
    pub decay_rate: f64,
    pub delta_t: f64,
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

pub struct UserSpatialEvaluator {
    pub max_distance: f64,
}

impl PairwiseCompatibility<CustomSpatialPayload, CustomSummaryState>
    for UserSpatialEvaluator
{
    fn evaluate(
        &self,
        observation: &Observation<CustomSpatialPayload>,
        belief: &BeliefState<CustomSummaryState>,
    ) -> Probability {
        let dx = observation.payload.x - belief.summary.last_x;
        let dy = observation.payload.y - belief.summary.last_y;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist > self.max_distance {
            Probability::new(0.0)
        } else {
            Probability::new(1.0 - (dist / self.max_distance))
        }
    }
}

fn main() {
    let evaluator = UserSpatialEvaluator {
        max_distance: 100.0,
    };

    let obs = Observation {
        id: li_core::ids::ObservationId(1),
        modality: li_core::observation::Modality(1),
        timestamp: Timestamp::from_millis(1000),
        confidence: li_core::probability::Confidence::new(0.95),
        payload: CustomSpatialPayload { x: 10.0, y: 20.0 },
    };

    let belief = BeliefState {
        identity: IdentityId(100),
        summary: CustomSummaryState {
            last_x: 12.0,
            last_y: 20.0,
        },
        posterior: Probability::new(0.8),
        last_update: Timestamp::from_millis(900),
    };

    let score = evaluator.evaluate(&obs, &belief);
    assert!(score.value() > 0.0);

    println!(
        "Evaluated spatial compatibility score between observation {} and identity {}: {:.4}",
        obs.id.0,
        belief.identity.0,
        score.value()
    );
}
