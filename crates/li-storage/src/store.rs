//! Unified storage engine implementing KnowledgeGraph, CheckpointStore, and
//! WalStore.

use alloc::vec::Vec;
use core::marker::PhantomData;

use li_core::belief::BeliefState;
use li_core::ids::{EventId, IdentityId, ObservationId, StateId, VertexId};
use li_core::observation::{Evidence, Observation, Timestamp};
use li_core::ontology::Vertex;
use li_core::relation::Relation;
use li_model::graph::KnowledgeGraph;
use li_model::operations::GraphOperation;
use serde::{Deserialize, Serialize};

use crate::errors::StorageError;
use crate::keys::{
    ColumnFamily, EDGE_KEY_LEN, decode_u64, encode_in_edge_key,
    encode_in_edge_prefix, encode_in_edge_target_prefix, encode_out_edge_key,
    encode_out_edge_prefix, encode_u64, encode_vertex_id,
};
use crate::postcard_adapter::{deserialize, serialize};
use crate::traits::{CheckpointStore, KvBackend, KvOp, WalStore};

/// Directed ontology relation edge connecting a source vertex to a target
/// vertex.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    /// Source vertex in the ontology graph.
    pub source: Vertex,
    /// Semantic relationship connecting source and target.
    pub relation: Relation,
    /// Target vertex in the ontology graph.
    pub target: Vertex,
    /// Timestamp when this relation was established.
    pub created_at: Timestamp,
}

/// Internal layout for identity node serialization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct IdentityNode {
    pub id: IdentityId,
    pub created_at: Timestamp,
}

/// Internal layout for event node serialization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct EventNode<E> {
    pub id: EventId,
    pub timestamp: Timestamp,
    pub payload: E,
}

/// Internal layout for state node serialization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct StateNode<S> {
    pub id: StateId,
    pub timestamp: Timestamp,
    pub payload: S,
}

/// Helper extracting the underlying [`VertexId`] representation from a domain
/// [`Vertex`].
fn vertex_to_id(vertex: Vertex) -> VertexId {
    vertex.vertex_id()
}

/// Replaces a duplicate identity endpoint with its canonical target.
///
/// Relations directly connecting the two merge identities, duplicate
/// self-loops, and non-LI relations are removed rather than rewired.
fn rewire_identity_edge(
    mut edge: Edge,
    target: IdentityId,
    duplicate: IdentityId,
) -> Option<Edge> {
    if !edge.relation.is_li_relation() {
        return None;
    }

    let target_vertex = Vertex::Identity(target);
    let duplicate_vertex = Vertex::Identity(duplicate);
    if edge.source == duplicate_vertex {
        if edge.target == target_vertex || edge.target == duplicate_vertex {
            return None;
        }
        edge.source = target_vertex;
    } else if edge.target == duplicate_vertex {
        if edge.source == target_vertex || edge.source == duplicate_vertex {
            return None;
        }
        edge.target = target_vertex;
    } else {
        return None;
    }

    Some(edge)
}

/// Primary engine managing persistence over any KvBackend implementation.
pub struct StorageEngine<B, P, E, S> {
    backend: B,
    next_wal_seq: u64,
    _phantom: PhantomData<(P, E, S)>,
}

impl<B: KvBackend, P, E, S> StorageEngine<B, P, E, S> {
    /// Constructs a StorageEngine wrapping the given backend driver.
    ///
    /// ## Returns
    /// New StorageEngine instance.
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            next_wal_seq: 1,
            _phantom: PhantomData,
        }
    }

    /// Helper for retrieving observation instances by ObservationId.
    pub fn fetch_observation(
        &self,
        id: ObservationId,
    ) -> Result<Option<Observation<P>>, StorageError>
    where
        P: for<'a> Deserialize<'a>,
    {
        let key = encode_u64(id.0);
        match self.backend.get(ColumnFamily::Observations, &key)? {
            Some(bytes) => Ok(Some(deserialize(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Retrieves all outgoing edges originating from a source vertex.
    pub fn out_edges(
        &self,
        source: VertexId,
    ) -> Result<Vec<Edge>, StorageError> {
        let prefix = encode_out_edge_prefix(source);
        let records =
            self.backend.prefix_scan(ColumnFamily::OutEdges, &prefix)?;

        let mut edges = Vec::with_capacity(records.len());
        for (_key, bytes) in records {
            let edge: Edge = deserialize(&bytes)?;
            edges.push(edge);
        }
        Ok(edges)
    }

    /// Merges a duplicate identity into a canonical target atomically.
    ///
    /// The operation rewires all other incident LI relations, preserves an
    /// existing canonical relation when keys collide, removes links between
    /// the two identities, and deletes the duplicate node records.
    ///
    /// # Arguments
    ///
    /// * `target` - Canonical identity that remains active.
    /// * `duplicate` - Identity removed after its relations are rewired.
    ///
    /// # Errors
    ///
    /// Returns a structured error for a self-merge, a missing identity,
    /// inconsistent persisted records, serialization failure, or backend
    /// failure.
    pub fn merge_identities(
        &mut self,
        target: IdentityId,
        duplicate: IdentityId,
    ) -> Result<(), StorageError> {
        if target == duplicate {
            return Err(StorageError::SelfIdentityMerge { identity: target });
        }

        self.ensure_identity_exists(target)?;
        self.ensure_identity_exists(duplicate)?;

        let duplicate_vertex = Vertex::Identity(duplicate);
        let duplicate_id = VertexId::from(duplicate);
        let out_prefix = encode_out_edge_prefix(duplicate_id);
        let in_prefix = encode_in_edge_target_prefix(duplicate_id);
        let outgoing = self
            .backend
            .prefix_scan(ColumnFamily::OutEdges, &out_prefix)?;
        let incoming = self
            .backend
            .prefix_scan(ColumnFamily::InEdges, &in_prefix)?;
        let incident_capacity = outgoing.len().saturating_add(incoming.len());
        let mut old_out_keys =
            Vec::<[u8; EDGE_KEY_LEN]>::with_capacity(incident_capacity);
        let mut old_in_keys =
            Vec::<[u8; EDGE_KEY_LEN]>::with_capacity(incident_capacity);
        let mut rewired = Vec::<([u8; EDGE_KEY_LEN], Edge)>::with_capacity(
            incident_capacity,
        );

        for (stored_key, bytes) in outgoing {
            let edge: Edge = deserialize(&bytes)?;
            let source = vertex_to_id(edge.source);
            let destination = vertex_to_id(edge.target);
            let out_key =
                encode_out_edge_key(source, edge.relation, destination);
            if stored_key.as_slice() != out_key.as_slice() ||
                edge.source != duplicate_vertex
            {
                return Err(StorageError::CorruptEdgeRecord);
            }

            old_out_keys.push(out_key);
            old_in_keys.push(encode_in_edge_key(
                destination,
                edge.relation,
                source,
            ));
            if let Some(replacement) =
                rewire_identity_edge(edge, target, duplicate)
            {
                let replacement_key = encode_out_edge_key(
                    vertex_to_id(replacement.source),
                    replacement.relation,
                    vertex_to_id(replacement.target),
                );
                rewired.push((replacement_key, replacement));
            }
        }

        for (stored_key, bytes) in incoming {
            let edge: Edge = deserialize(&bytes)?;
            let source = vertex_to_id(edge.source);
            let destination = vertex_to_id(edge.target);
            let in_key =
                encode_in_edge_key(destination, edge.relation, source);
            if stored_key.as_slice() != in_key.as_slice() ||
                edge.target != duplicate_vertex
            {
                return Err(StorageError::CorruptEdgeRecord);
            }

            old_out_keys.push(encode_out_edge_key(
                source,
                edge.relation,
                destination,
            ));
            old_in_keys.push(in_key);
            if let Some(replacement) =
                rewire_identity_edge(edge, target, duplicate)
            {
                let replacement_key = encode_out_edge_key(
                    vertex_to_id(replacement.source),
                    replacement.relation,
                    vertex_to_id(replacement.target),
                );
                rewired.push((replacement_key, replacement));
            }
        }

        old_out_keys.sort_unstable();
        old_out_keys.dedup();
        old_in_keys.sort_unstable();
        old_in_keys.dedup();
        rewired.sort_unstable_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.created_at.cmp(&right.1.created_at))
        });
        rewired.dedup_by(|left, right| left.0 == right.0);

        let operation_capacity = old_out_keys
            .len()
            .saturating_add(old_in_keys.len())
            .saturating_add(rewired.len().saturating_mul(2))
            .saturating_add(2);
        let mut batch = Vec::with_capacity(operation_capacity);
        for key in old_out_keys {
            batch.push(KvOp::Delete {
                cf: ColumnFamily::OutEdges,
                key: key.to_vec(),
            });
        }
        for key in old_in_keys {
            batch.push(KvOp::Delete {
                cf: ColumnFamily::InEdges,
                key: key.to_vec(),
            });
        }
        batch.push(KvOp::Delete {
            cf: ColumnFamily::Identities,
            key: encode_u64(duplicate.0).to_vec(),
        });
        batch.push(KvOp::Delete {
            cf: ColumnFamily::Ontology,
            key: encode_vertex_id(duplicate_id).to_vec(),
        });

        for (out_key, edge) in rewired {
            let in_key = encode_in_edge_key(
                vertex_to_id(edge.target),
                edge.relation,
                vertex_to_id(edge.source),
            );
            let existing_out =
                self.backend.get(ColumnFamily::OutEdges, &out_key)?;
            let existing_in =
                self.backend.get(ColumnFamily::InEdges, &in_key)?;

            match (existing_out, existing_in) {
                (Some(_), Some(_)) => {},
                (Some(value), None) => batch.push(KvOp::Put {
                    cf: ColumnFamily::InEdges,
                    key: in_key.to_vec(),
                    value,
                }),
                (None, Some(value)) => batch.push(KvOp::Put {
                    cf: ColumnFamily::OutEdges,
                    key: out_key.to_vec(),
                    value,
                }),
                (None, None) => {
                    let value = serialize(&edge)?;
                    batch.push(KvOp::Put {
                        cf: ColumnFamily::OutEdges,
                        key: out_key.to_vec(),
                        value: value.clone(),
                    });
                    batch.push(KvOp::Put {
                        cf: ColumnFamily::InEdges,
                        key: in_key.to_vec(),
                        value,
                    });
                },
            }
        }

        self.backend.apply_transaction(&batch)
    }

    /// Validates the paired identity and tagged ontology records.
    fn ensure_identity_exists(
        &self,
        identity: IdentityId,
    ) -> Result<(), StorageError> {
        let identity_key = encode_u64(identity.0);
        if self
            .backend
            .get(ColumnFamily::Identities, &identity_key)?
            .is_none()
        {
            return Err(StorageError::IdentityNotFound { identity });
        }

        let ontology_key = encode_vertex_id(VertexId::from(identity));
        let Some(bytes) =
            self.backend.get(ColumnFamily::Ontology, &ontology_key)?
        else {
            return Err(StorageError::IdentityNotFound { identity });
        };
        let stored: Vertex = deserialize(&bytes)?;
        if stored != Vertex::Identity(identity) {
            return Err(StorageError::CorruptIdentityRecord { identity });
        }

        Ok(())
    }
}

impl<B, P, E, S> KnowledgeGraph for StorageEngine<B, P, E, S>
where
    B: KvBackend,
    P: Serialize + for<'a> Deserialize<'a> + Clone,
    E: Serialize + for<'a> Deserialize<'a> + Clone,
    S: Serialize + for<'a> Deserialize<'a> + Clone,
{
    type Error = StorageError;
    type EventPayload = E;
    type ObservationPayload = P;
    type StatePayload = S;

    /// Evaluates the ontological typology of a vertex within the set V.
    fn vertex_type(
        &self,
        vertex: Vertex,
    ) -> Result<Option<Vertex>, Self::Error> {
        let vid = vertex_to_id(vertex);
        let key = encode_vertex_id(vid);
        match self.backend.get(ColumnFamily::Ontology, &key)? {
            Some(bytes) => Ok(Some(deserialize(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Applies a batch of formal operational primitives sequentially to
    /// transition the graph state.
    fn apply_batch(
        &mut self,
        ops: impl IntoIterator<Item = GraphOperation<P, E, S>>,
    ) -> Result<(), Self::Error> {
        let mut operations = ops.into_iter();
        let Some(first) = operations.next() else {
            return Ok(());
        };
        let first = match first {
            GraphOperation::MergeIdentities { target, duplicate } => {
                if operations.next().is_some() {
                    return Err(StorageError::MergeOperationMustBeIsolated);
                }
                return self.merge_identities(target, duplicate);
            },
            operation => operation,
        };
        let mut batch = Vec::new();

        for op in core::iter::once(first).chain(operations) {
            match op {
                GraphOperation::CommitObservation(obs) => {
                    let key = encode_u64(obs.id.0);
                    let vkey = encode_vertex_id(VertexId::from(obs.id));

                    let v_bytes = serialize(&Vertex::Observation(obs.id))?;
                    let obs_bytes = serialize(&obs)?;

                    batch.push(KvOp::Put {
                        cf: ColumnFamily::Ontology,
                        key: vkey.to_vec(),
                        value: v_bytes,
                    });
                    batch.push(KvOp::Put {
                        cf: ColumnFamily::Observations,
                        key: key.to_vec(),
                        value: obs_bytes,
                    });
                },
                GraphOperation::CommitIdentity { id, created_at } => {
                    let key = encode_u64(id.0);
                    let vkey = encode_vertex_id(VertexId::from(id));

                    let v_bytes = serialize(&Vertex::Identity(id))?;
                    let node = IdentityNode { id, created_at };
                    let id_bytes = serialize(&node)?;

                    batch.push(KvOp::Put {
                        cf: ColumnFamily::Ontology,
                        key: vkey.to_vec(),
                        value: v_bytes,
                    });
                    batch.push(KvOp::Put {
                        cf: ColumnFamily::Identities,
                        key: key.to_vec(),
                        value: id_bytes,
                    });
                },
                GraphOperation::CommitEvent {
                    id,
                    timestamp,
                    payload,
                } => {
                    let key = encode_u64(id.0);
                    let vkey = encode_vertex_id(VertexId::from(id));

                    let v_bytes = serialize(&Vertex::Event(id))?;
                    let node = EventNode {
                        id,
                        timestamp,
                        payload,
                    };
                    let evt_bytes = serialize(&node)?;

                    batch.push(KvOp::Put {
                        cf: ColumnFamily::Ontology,
                        key: vkey.to_vec(),
                        value: v_bytes,
                    });
                    batch.push(KvOp::Put {
                        cf: ColumnFamily::Events,
                        key: key.to_vec(),
                        value: evt_bytes,
                    });
                },
                GraphOperation::CommitState {
                    id,
                    timestamp,
                    payload,
                } => {
                    let key = encode_u64(id.0);
                    let vkey = encode_vertex_id(VertexId::from(id));

                    let v_bytes = serialize(&Vertex::State(id))?;
                    let node = StateNode {
                        id,
                        timestamp,
                        payload,
                    };
                    let st_bytes = serialize(&node)?;

                    batch.push(KvOp::Put {
                        cf: ColumnFamily::Ontology,
                        key: vkey.to_vec(),
                        value: v_bytes,
                    });
                    batch.push(KvOp::Put {
                        cf: ColumnFamily::States,
                        key: key.to_vec(),
                        value: st_bytes,
                    });
                },
                GraphOperation::CommitRelation {
                    source,
                    relation,
                    target,
                    created_at,
                } => {
                    let edge = Edge {
                        source,
                        relation,
                        target,
                        created_at,
                    };
                    let edge_bytes = serialize(&edge)?;

                    let source_id = vertex_to_id(source);
                    let target_id = vertex_to_id(target);

                    let out_key =
                        encode_out_edge_key(source_id, relation, target_id);
                    let in_key =
                        encode_in_edge_key(target_id, relation, source_id);

                    batch.push(KvOp::Put {
                        cf: ColumnFamily::OutEdges,
                        key: out_key.to_vec(),
                        value: edge_bytes.clone(),
                    });
                    batch.push(KvOp::Put {
                        cf: ColumnFamily::InEdges,
                        key: in_key.to_vec(),
                        value: edge_bytes,
                    });
                },
                GraphOperation::MergeIdentities { .. } => {
                    return Err(StorageError::MergeOperationMustBeIsolated);
                },
            }
        }

        self.backend.apply_transaction(&batch)
    }

    /// Queries the support set of observations linked to a given identity.
    fn query_support_set(
        &self,
        identity: IdentityId,
    ) -> Result<Vec<Observation<P>>, Self::Error> {
        let prefix = encode_in_edge_prefix(
            VertexId::from(identity),
            Relation::Supports,
        );
        let records =
            self.backend.prefix_scan(ColumnFamily::InEdges, &prefix)?;

        let mut observations = Vec::with_capacity(records.len());
        for (_key, edge_bytes) in records {
            let edge: Edge = deserialize(&edge_bytes)?;
            if let Vertex::Observation(obs_id) = edge.source &&
                let Some(obs) = self.fetch_observation(obs_id)?
            {
                observations.push(obs);
            }
        }
        Ok(observations)
    }

    /// Enumerates all identity identifiers defined in the graph V.
    fn all_identities(&self) -> Result<Vec<IdentityId>, Self::Error> {
        let records =
            self.backend.prefix_scan(ColumnFamily::Identities, &[])?;
        let mut ids = Vec::with_capacity(records.len());
        for (key, _val) in records {
            if let Some(raw_id) = decode_u64(&key) {
                ids.push(IdentityId(raw_id));
            }
        }
        Ok(ids)
    }
}

impl<B, P, E, S> CheckpointStore<S> for StorageEngine<B, P, E, S>
where
    B: KvBackend,
    S: Serialize + for<'a> Deserialize<'a>,
{
    fn save_checkpoint(
        &mut self,
        beliefs: &[BeliefState<S>],
    ) -> Result<(), StorageError> {
        let mut batch = Vec::with_capacity(beliefs.len());

        for belief in beliefs {
            let key = encode_u64(belief.identity.0);
            let value = serialize(belief)?;
            batch.push(KvOp::Put {
                cf: ColumnFamily::Checkpoints,
                key: key.to_vec(),
                value,
            });
        }

        self.backend.apply_transaction(&batch)
    }

    fn load_latest_checkpoint(
        &self,
    ) -> Result<Vec<BeliefState<S>>, StorageError> {
        let records =
            self.backend.prefix_scan(ColumnFamily::Checkpoints, &[])?;
        let mut beliefs = Vec::with_capacity(records.len());

        for (_key, bytes) in records {
            let belief: BeliefState<S> = deserialize(&bytes)?;
            beliefs.push(belief);
        }
        Ok(beliefs)
    }
}

impl<B, P, E, S> WalStore<P> for StorageEngine<B, P, E, S>
where
    B: KvBackend,
    P: Serialize + for<'a> Deserialize<'a>,
{
    fn append_evidence(
        &mut self,
        evidence: &Evidence<P>,
    ) -> Result<u64, StorageError> {
        let seq = self.next_wal_seq;
        let key = encode_u64(seq);
        let value = serialize(evidence)?;

        let batch = [KvOp::Put {
            cf: ColumnFamily::WalObservations,
            key: key.to_vec(),
            value,
        }];

        self.backend.apply_transaction(&batch)?;
        self.next_wal_seq += 1;
        Ok(seq)
    }

    fn read_delta(
        &self,
        from_sequence: u64,
    ) -> Result<Vec<Evidence<P>>, StorageError> {
        let records = self
            .backend
            .prefix_scan(ColumnFamily::WalObservations, &[])?;

        let mut deltas = Vec::new();
        for (key, bytes) in records {
            if let Some(seq) = decode_u64(&key) &&
                seq >= from_sequence
            {
                let evidence: Evidence<P> = deserialize(&bytes)?;
                deltas.push(evidence);
            }
        }
        Ok(deltas)
    }

    fn truncate_up_to(&mut self, sequence: u64) -> Result<(), StorageError> {
        let records = self
            .backend
            .prefix_scan(ColumnFamily::WalObservations, &[])?;
        let mut batch = Vec::new();

        for (key, _val) in records {
            if let Some(seq) = decode_u64(&key) &&
                seq <= sequence
            {
                batch.push(KvOp::Delete {
                    cf: ColumnFamily::WalObservations,
                    key,
                });
            }
        }

        self.backend.apply_transaction(&batch)
    }
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeMap;
    use alloc::vec;
    use alloc::vec::Vec;

    use li_core::ids::{
        EventId, IdentityId, ObservationId, StateId, VertexId,
    };
    use li_core::observation::{Evidence, Modality, Observation, Timestamp};
    use li_core::probability::{Confidence, Probability};
    use li_core::relation::Relation;

    use super::*;
    use crate::traits::KvRecord;

    struct MemoryBackend {
        storage: BTreeMap<ColumnFamily, BTreeMap<Vec<u8>, Vec<u8>>>,
        transactions: usize,
    }

    impl MemoryBackend {
        fn new() -> Self {
            Self {
                storage: BTreeMap::new(),
                transactions: 0,
            }
        }
    }

    impl KvBackend for MemoryBackend {
        fn get(
            &self,
            cf: ColumnFamily,
            key: &[u8],
        ) -> Result<Option<Vec<u8>>, StorageError> {
            Ok(self.storage.get(&cf).and_then(|map| map.get(key).cloned()))
        }

        fn apply_transaction(
            &mut self,
            batch: &[KvOp],
        ) -> Result<(), StorageError> {
            self.transactions = self.transactions.saturating_add(1);
            for op in batch {
                match op {
                    KvOp::Put { cf, key, value } => {
                        self.storage
                            .entry(*cf)
                            .or_default()
                            .insert(key.clone(), value.clone());
                    },
                    KvOp::Delete { cf, key } => {
                        if let Some(map) = self.storage.get_mut(cf) {
                            map.remove(key);
                        }
                    },
                }
            }
            Ok(())
        }

        fn prefix_scan(
            &self,
            cf: ColumnFamily,
            prefix: &[u8],
        ) -> Result<Vec<KvRecord>, StorageError> {
            let mut results = Vec::new();
            if let Some(map) = self.storage.get(&cf) {
                for (k, v) in map.iter() {
                    if k.starts_with(prefix) {
                        results.push((k.clone(), v.clone()));
                    }
                }
            }
            Ok(results)
        }
    }

    fn create_test_engine() -> StorageEngine<MemoryBackend, (), (), ()> {
        StorageEngine::new(MemoryBackend::new())
    }

    fn mock_observation(id: u64) -> Observation<()> {
        Observation {
            id: ObservationId(id),
            modality: Modality(1),
            timestamp: Timestamp::from_millis(1000),
            confidence: Confidence::new(0.95),
            payload: (),
        }
    }

    #[test]
    fn test_commit_and_query_vertices_happy_path() {
        let mut engine = create_test_engine();

        let obs = mock_observation(1);
        let identity_id = IdentityId(10);
        let event_id = EventId(20);
        let state_id = StateId(30);

        let ops = vec![
            GraphOperation::CommitObservation(obs.clone()),
            GraphOperation::CommitIdentity {
                id: identity_id,
                created_at: Timestamp::from_millis(1000),
            },
            GraphOperation::CommitEvent {
                id: event_id,
                timestamp: Timestamp::from_millis(1000),
                payload: (),
            },
            GraphOperation::CommitState {
                id: state_id,
                timestamp: Timestamp::from_millis(1000),
                payload: (),
            },
        ];

        assert!(engine.apply_batch(ops).is_ok());

        assert_eq!(
            engine
                .vertex_type(Vertex::Observation(ObservationId(1)))
                .unwrap(),
            Some(Vertex::Observation(ObservationId(1)))
        );
        assert_eq!(
            engine
                .vertex_type(Vertex::Identity(IdentityId(10)))
                .unwrap(),
            Some(Vertex::Identity(IdentityId(10)))
        );
        assert_eq!(
            engine.vertex_type(Vertex::Event(EventId(20))).unwrap(),
            Some(Vertex::Event(EventId(20)))
        );
        assert_eq!(
            engine.vertex_type(Vertex::State(StateId(30))).unwrap(),
            Some(Vertex::State(StateId(30)))
        );

        let fetched_obs = engine.fetch_observation(ObservationId(1)).unwrap();
        assert_eq!(fetched_obs, Some(obs));
    }

    #[test]
    fn tagged_vertices_with_equal_raw_ids_coexist_in_ontology() {
        let mut engine = create_test_engine();
        let observation = mock_observation(1);
        let vertices = [
            Vertex::Observation(ObservationId(1)),
            Vertex::Identity(IdentityId(1)),
            Vertex::Event(EventId(1)),
            Vertex::State(StateId(1)),
        ];
        let operations = vec![
            GraphOperation::CommitObservation(observation),
            GraphOperation::CommitIdentity {
                id: IdentityId(1),
                created_at: Timestamp::from_secs(1),
            },
            GraphOperation::CommitEvent {
                id: EventId(1),
                timestamp: Timestamp::from_secs(1),
                payload: (),
            },
            GraphOperation::CommitState {
                id: StateId(1),
                timestamp: Timestamp::from_secs(1),
                payload: (),
            },
        ];

        assert_eq!(engine.apply_batch(operations), Ok(()));
        for vertex in vertices {
            assert_eq!(engine.vertex_type(vertex), Ok(Some(vertex)));
        }
        assert_eq!(
            engine
                .backend
                .prefix_scan(ColumnFamily::Ontology, &[])
                .map(|records| records.len()),
            Ok(4)
        );
    }

    #[test]
    fn test_relations_and_support_set_happy_path() -> Result<(), StorageError>
    {
        let mut engine = create_test_engine();

        let obs = mock_observation(100);

        let ops = vec![
            GraphOperation::CommitObservation(obs.clone()),
            GraphOperation::CommitIdentity {
                id: IdentityId(200),
                created_at: Timestamp::from_millis(1000),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(100)),
                relation: Relation::Supports,
                target: Vertex::Identity(IdentityId(200)),
                created_at: Timestamp::from_millis(1000),
            },
        ];

        assert!(engine.apply_batch(ops).is_ok());

        let out_edges =
            engine.out_edges(VertexId::from(ObservationId(100)))?;
        assert_eq!(out_edges.len(), 1);
        assert_eq!(
            out_edges[0].source,
            Vertex::Observation(ObservationId(100))
        );
        assert_eq!(out_edges[0].relation, Relation::Supports);
        assert_eq!(out_edges[0].target, Vertex::Identity(IdentityId(200)));

        let support = engine.query_support_set(IdentityId(200))?;
        assert_eq!(support.len(), 1);
        assert_eq!(support[0], obs);
        Ok(())
    }

    #[test]
    fn test_all_identities_happy_path() {
        let mut engine = create_test_engine();

        let ops = vec![
            GraphOperation::CommitIdentity {
                id: IdentityId(1),
                created_at: Timestamp::from_millis(100),
            },
            GraphOperation::CommitIdentity {
                id: IdentityId(2),
                created_at: Timestamp::from_millis(200),
            },
        ];

        assert!(engine.apply_batch(ops).is_ok());

        let mut ids = engine.all_identities().unwrap();
        ids.sort_by_key(|i| i.0);
        assert_eq!(ids, vec![IdentityId(1), IdentityId(2)]);
    }

    #[test]
    fn merge_rewires_both_directions_and_coalesces_existing_edges() {
        let mut engine = create_test_engine();
        let target = IdentityId(1);
        let duplicate = IdentityId(2);
        let related = IdentityId(3);
        let existing_timestamp = Timestamp::from_secs(10);
        let setup = vec![
            GraphOperation::CommitIdentity {
                id: target,
                created_at: Timestamp::from_secs(1),
            },
            GraphOperation::CommitIdentity {
                id: duplicate,
                created_at: Timestamp::from_secs(2),
            },
            GraphOperation::CommitIdentity {
                id: related,
                created_at: Timestamp::from_secs(3),
            },
            GraphOperation::CommitObservation(mock_observation(10)),
            GraphOperation::CommitObservation(mock_observation(20)),
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(10)),
                relation: Relation::Supports,
                target: Vertex::Identity(target),
                created_at: Timestamp::from_secs(4),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Observation(ObservationId(20)),
                relation: Relation::Supports,
                target: Vertex::Identity(duplicate),
                created_at: Timestamp::from_secs(5),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Identity(target),
                relation: Relation::AssociatedWith,
                target: Vertex::Identity(related),
                created_at: existing_timestamp,
            },
            GraphOperation::CommitRelation {
                source: Vertex::Identity(duplicate),
                relation: Relation::AssociatedWith,
                target: Vertex::Identity(related),
                created_at: Timestamp::from_secs(11),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Identity(related),
                relation: Relation::AssociatedWith,
                target: Vertex::Identity(duplicate),
                created_at: Timestamp::from_secs(12),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Identity(duplicate),
                relation: Relation::AssociatedWith,
                target: Vertex::Identity(target),
                created_at: Timestamp::from_secs(13),
            },
            GraphOperation::CommitRelation {
                source: Vertex::Identity(target),
                relation: Relation::Refines,
                target: Vertex::Identity(duplicate),
                created_at: Timestamp::from_secs(14),
            },
        ];
        assert_eq!(engine.apply_batch(setup), Ok(()));
        assert_eq!(engine.backend.transactions, 1);

        let merge = [GraphOperation::MergeIdentities { target, duplicate }];
        assert_eq!(engine.apply_batch(merge), Ok(()));
        assert_eq!(engine.backend.transactions, 2);
        assert_eq!(engine.vertex_type(Vertex::Identity(duplicate)), Ok(None));
        assert_eq!(
            engine.all_identities().map(|mut identities| {
                identities.sort_unstable();
                identities
            }),
            Ok(vec![target, related])
        );
        assert_eq!(
            engine.query_support_set(target).map(|mut observations| {
                observations.sort_unstable_by_key(|item| item.id);
                observations
                    .into_iter()
                    .map(|item| item.id)
                    .collect::<Vec<_>>()
            }),
            Ok(vec![ObservationId(10), ObservationId(20)])
        );

        let target_edges = engine.out_edges(VertexId::from(target));
        assert!(matches!(
            target_edges,
            Ok(edges)
                if edges.len() == 1 &&
                    edges[0].target == Vertex::Identity(related) &&
                    edges[0].created_at == existing_timestamp
        ));
        let related_edges = engine.out_edges(VertexId::from(related));
        assert!(matches!(
            related_edges,
            Ok(edges)
                if edges.len() == 1 &&
                    edges[0].target == Vertex::Identity(target)
        ));
        assert!(matches!(
            engine.out_edges(VertexId::from(duplicate)),
            Ok(edges) if edges.is_empty()
        ));
        let duplicate_in_prefix =
            encode_in_edge_target_prefix(VertexId::from(duplicate));
        assert!(matches!(
            engine
                .backend
                .prefix_scan(ColumnFamily::InEdges, &duplicate_in_prefix),
            Ok(records) if records.is_empty()
        ));
    }

    #[test]
    fn merge_rejects_self_missing_and_mixed_operations_without_writes() {
        let mut empty = create_test_engine();
        assert_eq!(
            empty.merge_identities(IdentityId(7), IdentityId(7)),
            Err(StorageError::SelfIdentityMerge {
                identity: IdentityId(7)
            })
        );
        assert_eq!(
            empty.merge_identities(IdentityId(7), IdentityId(8)),
            Err(StorageError::IdentityNotFound {
                identity: IdentityId(7)
            })
        );
        assert_eq!(empty.backend.transactions, 0);

        let mixed = [
            GraphOperation::CommitIdentity {
                id: IdentityId(1),
                created_at: Timestamp::from_secs(1),
            },
            GraphOperation::MergeIdentities {
                target: IdentityId(1),
                duplicate: IdentityId(2),
            },
        ];
        assert_eq!(
            empty.apply_batch(mixed),
            Err(StorageError::MergeOperationMustBeIsolated)
        );
        assert_eq!(empty.backend.transactions, 0);
        assert_eq!(
            empty.vertex_type(Vertex::Identity(IdentityId(1))),
            Ok(None)
        );

        assert_eq!(
            empty.apply_batch([GraphOperation::CommitIdentity {
                id: IdentityId(1),
                created_at: Timestamp::from_secs(1),
            }]),
            Ok(())
        );
        assert_eq!(
            empty.merge_identities(IdentityId(1), IdentityId(2)),
            Err(StorageError::IdentityNotFound {
                identity: IdentityId(2)
            })
        );
        assert_eq!(empty.backend.transactions, 1);
    }

    #[test]
    fn test_checkpoint_store_happy_path() {
        let mut engine = create_test_engine();

        let beliefs = vec![
            BeliefState {
                identity: IdentityId(1),
                summary: (),
                posterior: Probability::new(0.8),
                last_update: Timestamp::from_millis(500),
            },
            BeliefState {
                identity: IdentityId(2),
                summary: (),
                posterior: Probability::new(0.3),
                last_update: Timestamp::from_millis(600),
            },
        ];

        assert!(engine.save_checkpoint(&beliefs).is_ok());

        let loaded = engine.load_latest_checkpoint().unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn test_wal_store_happy_path() {
        let mut engine = create_test_engine();

        let ev1 = Evidence {
            observation: mock_observation(10),
            candidates: vec![IdentityId(1)],
        };
        let ev2 = Evidence {
            observation: mock_observation(11),
            candidates: vec![IdentityId(2)],
        };

        let seq1 = engine.append_evidence(&ev1).unwrap();
        let seq2 = engine.append_evidence(&ev2).unwrap();

        assert_eq!(seq1, 1);
        assert_eq!(seq2, 2);

        let deltas = engine.read_delta(1).unwrap();
        assert_eq!(deltas.len(), 2);

        let deltas_from_2 = engine.read_delta(2).unwrap();
        assert_eq!(deltas_from_2.len(), 1);

        assert!(engine.truncate_up_to(1).is_ok());

        let deltas_after_trunc = engine.read_delta(1).unwrap();
        assert_eq!(deltas_after_trunc.len(), 1);
    }

    #[test]
    fn test_edge_case_fetch_nonexistent_entities() -> Result<(), StorageError>
    {
        let engine = create_test_engine();

        assert_eq!(engine.fetch_observation(ObservationId(999))?, None);
        assert_eq!(
            engine.vertex_type(Vertex::Identity(IdentityId(999)))?,
            None
        );
        assert!(
            engine
                .out_edges(VertexId::from(IdentityId(999)))?
                .is_empty()
        );
        assert!(engine.query_support_set(IdentityId(999))?.is_empty());
        Ok(())
    }

    #[test]
    fn test_edge_case_empty_batch_and_empty_queries() {
        let mut engine = create_test_engine();

        let empty_ops: Vec<GraphOperation<(), (), ()>> = Vec::new();
        assert!(engine.apply_batch(empty_ops).is_ok());
        assert!(engine.all_identities().unwrap().is_empty());
        assert!(engine.load_latest_checkpoint().unwrap().is_empty());
        assert!(engine.read_delta(0).unwrap().is_empty());
    }

    #[test]
    fn test_edge_case_wal_truncate_nonexistent_sequences() {
        let mut engine = create_test_engine();

        let ev = Evidence {
            observation: mock_observation(1),
            candidates: vec![],
        };
        engine.append_evidence(&ev).unwrap();

        assert!(engine.truncate_up_to(999).is_ok());
        assert!(engine.read_delta(0).unwrap().is_empty());
    }
}
