//! Binary key encoding primitives and Column Family partition definitions.

use li_core::ids::{VertexId, VertexKind};
use li_core::relation::Relation;

/// Number of bytes in a tagged vertex key.
pub const VERTEX_KEY_LEN: usize = 9;
/// Number of bytes in a complete directed edge key.
pub const EDGE_KEY_LEN: usize = VERTEX_KEY_LEN * 2 + 1;
/// Number of bytes in an incoming target-and-relation prefix.
pub const IN_EDGE_PREFIX_LEN: usize = VERTEX_KEY_LEN + 1;

/// Logical column families isolating data types in the physical storage layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ColumnFamily {
    /// Mapping VertexId -> Vertex variant code.
    Ontology,
    /// Mapping ObservationId -> Observation payload bytes.
    Observations,
    /// Mapping IdentityId -> IdentityNode payload bytes.
    Identities,
    /// Mapping EventId -> EventNode payload bytes.
    Events,
    /// Mapping StateId -> StateNode payload bytes.
    States,
    /// Mapping (SourceVertexId | RelationCode | TargetVertexId) -> Edge bytes.
    OutEdges,
    /// Mapping (TargetVertexId | RelationCode | SourceVertexId) -> Edge bytes.
    InEdges,
    /// Mapping IdentityId -> BeliefState payload bytes.
    Checkpoints,
    /// Mapping MonotonicSequence -> Evidence payload bytes.
    WalObservations,
}

impl ColumnFamily {
    /// Returns the string name representation of [`ColumnFamily`].
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ontology => "ontology",
            Self::Observations => "observations",
            Self::Identities => "identities",
            Self::Events => "events",
            Self::States => "states",
            Self::OutEdges => "out_edges",
            Self::InEdges => "in_edges",
            Self::Checkpoints => "checkpoints",
            Self::WalObservations => "wal_observations",
        }
    }
}

/// Encodes a 64-bit integer into a big-endian byte array to preserve numerical
/// ordering.
pub fn encode_u64(val: u64) -> [u8; 8] {
    val.to_be_bytes()
}

/// Decodes a 64-bit unsigned integer from a big-endian byte slice.
pub fn decode_u64(bytes: &[u8]) -> Option<u64> {
    if bytes.len() < 8 {
        return None;
    }
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[..8]);
    Some(u64::from_be_bytes(arr))
}

/// Encodes a tagged vertex identifier without losing its ontology partition.
///
/// The partition tag leads the big-endian raw identifier so distinct vertex
/// kinds with the same numeric identifier remain separate and keys retain a
/// stable lexicographic order.
pub fn encode_vertex_id(vertex: VertexId) -> [u8; VERTEX_KEY_LEN] {
    let mut key = [0u8; VERTEX_KEY_LEN];
    key[0] = vertex.kind().code();
    key[1..].copy_from_slice(&vertex.raw().to_be_bytes());
    key
}

/// Decodes a tagged vertex identifier from a storage key prefix.
///
/// Returns `None` when the key is truncated or contains an unknown partition
/// tag.
pub fn decode_vertex_id(bytes: &[u8]) -> Option<VertexId> {
    if bytes.len() < VERTEX_KEY_LEN {
        return None;
    }

    let kind = match bytes[0] {
        0 => VertexKind::Observation,
        1 => VertexKind::Identity,
        2 => VertexKind::Event,
        3 => VertexKind::State,
        _ => return None,
    };
    decode_u64(&bytes[1..]).map(|raw| VertexId::new(kind, raw))
}

/// Encodes a Relation variant into a 1-byte numerical code.
pub fn encode_relation(relation: Relation) -> u8 {
    match relation {
        Relation::Trigger => 0,
        Relation::Lead => 1,
        Relation::Evolution => 2,
        Relation::Contain => 3,
        Relation::Occur => 4,
        Relation::Influence => 5,
        Relation::ObservedDuring => 6,
        Relation::Describes => 7,
        Relation::Supports => 8,
        Relation::Refines => 9,
        Relation::AssociatedWith => 10,
    }
}

/// Encodes an outgoing edge into a fixed-size composite binary key.
pub fn encode_out_edge_key(
    source: VertexId,
    relation: Relation,
    target: VertexId,
) -> [u8; EDGE_KEY_LEN] {
    let mut key = [0u8; EDGE_KEY_LEN];
    key[..VERTEX_KEY_LEN].copy_from_slice(&encode_vertex_id(source));
    key[VERTEX_KEY_LEN] = encode_relation(relation);
    key[VERTEX_KEY_LEN + 1..].copy_from_slice(&encode_vertex_id(target));
    key
}

/// Encodes an incoming edge into a fixed-size composite binary key.
pub fn encode_in_edge_key(
    target: VertexId,
    relation: Relation,
    source: VertexId,
) -> [u8; EDGE_KEY_LEN] {
    let mut key = [0u8; EDGE_KEY_LEN];
    key[..VERTEX_KEY_LEN].copy_from_slice(&encode_vertex_id(target));
    key[VERTEX_KEY_LEN] = encode_relation(relation);
    key[VERTEX_KEY_LEN + 1..].copy_from_slice(&encode_vertex_id(source));
    key
}

/// Encodes a prefix for scanning every outgoing edge from a source vertex.
pub fn encode_out_edge_prefix(source: VertexId) -> [u8; VERTEX_KEY_LEN] {
    encode_vertex_id(source)
}

/// Encodes a prefix for scanning every incoming edge to a target vertex.
pub fn encode_in_edge_target_prefix(target: VertexId) -> [u8; VERTEX_KEY_LEN] {
    encode_vertex_id(target)
}

/// Encodes a prefix key for scanning incoming edges by target and
/// relation.
pub fn encode_in_edge_prefix(
    target: VertexId,
    relation: Relation,
) -> [u8; IN_EDGE_PREFIX_LEN] {
    let mut prefix = [0u8; IN_EDGE_PREFIX_LEN];
    prefix[..VERTEX_KEY_LEN].copy_from_slice(&encode_vertex_id(target));
    prefix[VERTEX_KEY_LEN] = encode_relation(relation);
    prefix
}

#[cfg(test)]
mod tests {
    use li_core::ids::{EventId, IdentityId, ObservationId, StateId};

    use super::*;

    #[test]
    fn tagged_vertex_keys_round_trip_without_cross_partition_aliases() {
        let vertices = [
            VertexId::from(ObservationId(1)),
            VertexId::from(IdentityId(1)),
            VertexId::from(EventId(1)),
            VertexId::from(StateId(1)),
        ];
        let mut keys = vertices.map(encode_vertex_id);

        keys.sort_unstable();

        assert!(keys.windows(2).all(|pair| pair[0] != pair[1]));
        for vertex in vertices {
            assert_eq!(
                decode_vertex_id(&encode_vertex_id(vertex)),
                Some(vertex)
            );
        }
    }

    #[test]
    fn edge_prefixes_cover_only_the_requested_tagged_endpoint() {
        let observation = VertexId::from(ObservationId(7));
        let identity = VertexId::from(IdentityId(7));
        let target = VertexId::from(StateId(9));
        let relation = Relation::Describes;
        let observation_key =
            encode_out_edge_key(observation, relation, target);
        let identity_key = encode_out_edge_key(identity, relation, target);

        assert!(
            observation_key.starts_with(&encode_out_edge_prefix(observation))
        );
        assert!(
            !identity_key.starts_with(&encode_out_edge_prefix(observation))
        );

        let incoming = encode_in_edge_key(target, relation, observation);
        assert!(incoming.starts_with(&encode_in_edge_target_prefix(target)));
        assert!(
            incoming.starts_with(&encode_in_edge_prefix(target, relation))
        );
    }

    #[test]
    fn vertex_decoder_rejects_truncated_and_unknown_tags() {
        assert_eq!(decode_vertex_id(&[0; VERTEX_KEY_LEN - 1]), None);

        let mut invalid = [0u8; VERTEX_KEY_LEN];
        invalid[0] = u8::MAX;
        assert_eq!(decode_vertex_id(&invalid), None);
    }
}
