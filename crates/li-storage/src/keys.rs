//! Binary key encoding primitives and Column Family partition definitions.

use alloc::vec::Vec;

use li_core::ids::VertexId;
use li_core::relation::Relation;

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

/// Encodes an outgoing edge into a 17-byte composite binary key.
///
/// Args:
///   source: Source vertex identifier.
///   relation: Edge relation classification.
///   target: Target vertex identifier.
///
/// Returns:
///   Composite key byte vector.
pub fn encode_out_edge_key(
    source: VertexId,
    relation: Relation,
    target: VertexId,
) -> Vec<u8> {
    let mut key = Vec::with_capacity(17);
    key.extend_from_slice(&encode_u64(source.0));
    key.push(encode_relation(relation));
    key.extend_from_slice(&encode_u64(target.0));
    key
}

/// Encodes an incoming edge into a 17-byte composite binary key.
///
/// Args:
///   target: Target vertex identifier.
///   relation: Edge relation classification.
///   source: Source vertex identifier.
///
/// Returns:
///   Composite key byte vector.
pub fn encode_in_edge_key(
    target: VertexId,
    relation: Relation,
    source: VertexId,
) -> Vec<u8> {
    let mut key = Vec::with_capacity(17);
    key.extend_from_slice(&encode_u64(target.0));
    key.push(encode_relation(relation));
    key.extend_from_slice(&encode_u64(source.0));
    key
}

/// Encodes a 9-byte prefix key for scanning incoming edges by target and
/// relation.
///
/// Args:
///   target: Target vertex identifier.
///   relation: Edge relation classification.
///
/// Returns:
///   Prefix byte vector.
pub fn encode_in_edge_prefix(target: VertexId, relation: Relation) -> Vec<u8> {
    let mut prefix = Vec::with_capacity(9);
    prefix.extend_from_slice(&encode_u64(target.0));
    prefix.push(encode_relation(relation));
    prefix
}
