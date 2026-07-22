//! Interface for upstream metric-space nearest neighbor candidate extraction.

use alloc::vec::Vec;

use li_core::ids::IdentityId;
use li_core::observation::Observation;

/// Abstract interface for upstream vector search indices (e.g., Faiss, HNSW,
/// ScaNN). Formalizes the mapping $f_{\text{index}}: o_t \mapsto \{i \in I
/// \mid d(\text{emb}(o_t), \text{emb}(i)) < \epsilon\}$.
pub trait CandidateGenerator<P> {
    /// Generates candidate identity identifiers corresponding to an empirical
    /// observation.
    fn generate_candidates(
        &self,
        observation: &Observation<P>,
    ) -> Vec<IdentityId>;
}
