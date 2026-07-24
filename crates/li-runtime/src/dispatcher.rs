//! Event routing and dispatching mechanism for incoming runtime signals.

use li_core::events::RuntimeEvent;
use li_core::ids::IdentityId;
use li_core::observation::Evidence;

/// Result of an event dispatching phase indicating the necessary operational
/// directive.
#[derive(Debug, Clone, PartialEq)]
pub enum DispatchOutcome<P> {
    /// Process observation via candidate evaluation and probabilistic
    /// inference.
    EvaluateObservation(Evidence<P>),
    /// Directly merge two existing identity nodes in the graph.
    MergeIdentities {
        /// The target identity node to merge into.
        target: IdentityId,
        /// The duplicate identity node to be merged and removed.
        duplicate: IdentityId,
    },
    /// Trigger a state persistence checkpoint.
    TriggerCheckpoint,
    /// Initiate graceful runtime termination.
    Shutdown,
    /// No action required or event ignored.
    NoOp,
}

/// Event dispatcher routing incoming system events to operational directives.
#[derive(Debug, Default, Clone, Copy)]
pub struct EventDispatcher;

impl EventDispatcher {
    /// Creates a new instance of `EventDispatcher`.
    pub fn new() -> Self {
        Self
    }

    /// Routes a runtime event to its corresponding execution directive.
    ///
    /// # Arguments
    ///
    /// * `event` - The incoming runtime event to be evaluated.
    ///
    /// # Returns
    ///
    /// Returns a `DispatchOutcome` directing subsequent processing steps.
    pub fn dispatch<P>(&self, event: RuntimeEvent<P>) -> DispatchOutcome<P> {
        match event {
            RuntimeEvent::Observation(evidence) => {
                DispatchOutcome::EvaluateObservation(evidence)
            },
            RuntimeEvent::IdentityMerged { target, duplicate } => {
                DispatchOutcome::MergeIdentities { target, duplicate }
            },
            RuntimeEvent::Checkpoint => DispatchOutcome::TriggerCheckpoint,
            RuntimeEvent::Shutdown => DispatchOutcome::Shutdown,
            _ => DispatchOutcome::NoOp,
        }
    }
}

#[cfg(test)]
mod tests {
    use li_core::ids::{IdentityId, ObservationId};
    use li_core::observation::{Modality, Observation, Timestamp};
    use li_core::probability::Confidence;

    use super::*;

    #[test]
    fn test_dispatch_observation() {
        let dispatcher = EventDispatcher::new();
        let evidence = Evidence {
            observation: Observation {
                id: ObservationId(1),
                modality: Modality(1),
                timestamp: Timestamp(100),
                confidence: Confidence(0.99),
                payload: 42u64,
            },
            candidates: alloc::vec![IdentityId(10)],
        };
        let event: RuntimeEvent<u64> =
            RuntimeEvent::Observation(evidence.clone());
        assert_eq!(
            dispatcher.dispatch(event),
            DispatchOutcome::EvaluateObservation(evidence)
        );
    }

    #[test]
    fn test_dispatch_control_events() {
        let dispatcher = EventDispatcher::new();
        let cp: RuntimeEvent<()> = RuntimeEvent::Checkpoint;
        let sd: RuntimeEvent<()> = RuntimeEvent::Shutdown;

        assert_eq!(
            dispatcher.dispatch(cp),
            DispatchOutcome::TriggerCheckpoint
        );
        assert_eq!(dispatcher.dispatch(sd), DispatchOutcome::Shutdown);
    }
}
