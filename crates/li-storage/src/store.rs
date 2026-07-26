//! Unified storage engine implementing KnowledgeGraph, CheckpointStore, and
//! WalStore.

use alloc::vec::Vec;
use core::marker::PhantomData;

use li_core::belief::BeliefState;
use li_core::ids::{IdentityId, ObservationId, VertexId};
use li_core::observation::{Evidence, Observation};
use li_core::ontology::Vertex;
use li_core::relation::Relation;
use li_model::graph::KnowledgeGraph;
use li_model::ontology::Edge;
use li_model::operations::GraphOperation;
use serde::{Deserialize, Serialize};

use crate::errors::StorageError;
use crate::keys::{
    ColumnFamily, decode_u64, encode_in_edge_key, encode_in_edge_prefix,
    encode_out_edge_key, encode_u64,
};
use crate::postcard_adapter::{deserialize, serialize};
use crate::traits::{CheckpointStore, KvBackend, KvOp, WalStore};

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

    /// Evaluates the ontological typology of a vertex within the set $V$.
    fn vertex_type(
        &self,
        id: VertexId,
    ) -> Result<Option<Vertex>, Self::Error> {
        let key = encode_u64(id.0);
        match self.backend.get(ColumnFamily::Ontology, &key)? {
            Some(bytes) => Ok(Some(deserialize(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Applies a batch of formal operational primitives sequentially to
    /// transition the graph state.
    fn apply_batch(
        &mut self,
        ops: &[GraphOperation<P, E, S>],
    ) -> Result<(), Self::Error> {
        let mut batch = Vec::with_capacity(ops.len() * 3);

        for op in ops {
            match op {
                GraphOperation::CommitObservation(obs) => {
                    let vid = VertexId(obs.id.0);
                    let key = encode_u64(obs.id.0);
                    let vkey = encode_u64(vid.0);

                    let v_bytes = serialize(&Vertex::Observation(obs.id))?;
                    let obs_bytes = serialize(obs)?;

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
                GraphOperation::CommitIdentity(identity) => {
                    let vid = VertexId(identity.id.0);
                    let key = encode_u64(identity.id.0);
                    let vkey = encode_u64(vid.0);

                    let v_bytes = serialize(&Vertex::Identity(identity.id))?;
                    let id_bytes = serialize(identity)?;

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
                GraphOperation::CommitEvent(event) => {
                    let vid = VertexId(event.id.0);
                    let key = encode_u64(event.id.0);
                    let vkey = encode_u64(vid.0);

                    let v_bytes = serialize(&Vertex::Event(event.id))?;
                    let evt_bytes = serialize(event)?;

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
                GraphOperation::CommitState(state) => {
                    let vid = VertexId(state.id.0);
                    let key = encode_u64(state.id.0);
                    let vkey = encode_u64(vid.0);

                    let v_bytes = serialize(&Vertex::State(state.id))?;
                    let st_bytes = serialize(state)?;

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
                        source: *source,
                        relation: *relation,
                        target: *target,
                        created_at: *created_at,
                    };
                    let edge_bytes = serialize(&edge)?;

                    let out_key =
                        encode_out_edge_key(*source, *relation, *target);
                    let in_key =
                        encode_in_edge_key(*target, *relation, *source);

                    batch.push(KvOp::Put {
                        cf: ColumnFamily::OutEdges,
                        key: out_key,
                        value: edge_bytes.clone(),
                    });
                    batch.push(KvOp::Put {
                        cf: ColumnFamily::InEdges,
                        key: in_key,
                        value: edge_bytes,
                    });
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
        let prefix =
            encode_in_edge_prefix(VertexId(identity.0), Relation::Supports);
        let records =
            self.backend.prefix_scan(ColumnFamily::InEdges, &prefix)?;

        let mut observations = Vec::with_capacity(records.len());
        for (_key, edge_bytes) in records {
            let edge: Edge = deserialize(&edge_bytes)?;
            if let Some(obs) =
                self.fetch_observation(ObservationId(edge.source.0))?
            {
                observations.push(obs);
            }
        }
        Ok(observations)
    }

    /// Retrieves all outgoing edges originating from a source vertex.
    fn out_edges(&self, source: VertexId) -> Result<Vec<Edge>, Self::Error> {
        let prefix = encode_u64(source.0);
        let records =
            self.backend.prefix_scan(ColumnFamily::OutEdges, &prefix)?;

        let mut edges = Vec::with_capacity(records.len());
        for (_key, bytes) in records {
            let edge: Edge = deserialize(&bytes)?;
            edges.push(edge);
        }
        Ok(edges)
    }

    /// Enumerates all identity identifiers defined in the graph $V$.
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
    use li_model::ontology::{EventNode, IdentityNode, StateNode};

    use super::*;
    use crate::traits::KvRecord;

    struct MemoryBackend {
        storage: BTreeMap<ColumnFamily, BTreeMap<Vec<u8>, Vec<u8>>>,
    }

    impl MemoryBackend {
        fn new() -> Self {
            Self {
                storage: BTreeMap::new(),
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
        let identity = IdentityNode {
            id: IdentityId(10),
            created_at: Timestamp::from_millis(1000),
        };
        let event = EventNode {
            id: EventId(20),
            timestamp: Timestamp::from_millis(1000),
            payload: (),
        };
        let state = StateNode {
            id: StateId(30),
            timestamp: Timestamp::from_millis(1000),
            payload: (),
        };

        let ops = vec![
            GraphOperation::CommitObservation(obs.clone()),
            GraphOperation::CommitIdentity(identity),
            GraphOperation::CommitEvent(event),
            GraphOperation::CommitState(state),
        ];

        assert!(engine.apply_batch(&ops).is_ok());

        assert_eq!(
            engine.vertex_type(VertexId(1)).unwrap(),
            Some(Vertex::Observation(ObservationId(1)))
        );
        assert_eq!(
            engine.vertex_type(VertexId(10)).unwrap(),
            Some(Vertex::Identity(IdentityId(10)))
        );
        assert_eq!(
            engine.vertex_type(VertexId(20)).unwrap(),
            Some(Vertex::Event(EventId(20)))
        );
        assert_eq!(
            engine.vertex_type(VertexId(30)).unwrap(),
            Some(Vertex::State(StateId(30)))
        );

        let fetched_obs = engine.fetch_observation(ObservationId(1)).unwrap();
        assert_eq!(fetched_obs, Some(obs));
    }

    #[test]
    fn test_relations_and_support_set_happy_path() {
        let mut engine = create_test_engine();

        let obs = mock_observation(100);
        let identity = IdentityNode {
            id: IdentityId(200),
            created_at: Timestamp::from_millis(1000),
        };

        let ops = vec![
            GraphOperation::CommitObservation(obs.clone()),
            GraphOperation::CommitIdentity(identity),
            GraphOperation::CommitRelation {
                source: VertexId(100),
                relation: Relation::Supports,
                target: VertexId(200),
                created_at: Timestamp::from_millis(1000),
            },
        ];

        assert!(engine.apply_batch(&ops).is_ok());

        let out_edges = engine.out_edges(VertexId(100)).unwrap();
        assert_eq!(out_edges.len(), 1);
        assert_eq!(out_edges[0].source, VertexId(100));
        assert_eq!(out_edges[0].relation, Relation::Supports);
        assert_eq!(out_edges[0].target, VertexId(200));

        let support = engine.query_support_set(IdentityId(200)).unwrap();
        assert_eq!(support.len(), 1);
        assert_eq!(support[0], obs);
    }

    #[test]
    fn test_all_identities_happy_path() {
        let mut engine = create_test_engine();

        let ops = vec![
            GraphOperation::CommitIdentity(IdentityNode {
                id: IdentityId(1),
                created_at: Timestamp::from_millis(100),
            }),
            GraphOperation::CommitIdentity(IdentityNode {
                id: IdentityId(2),
                created_at: Timestamp::from_millis(200),
            }),
        ];

        assert!(engine.apply_batch(&ops).is_ok());

        let mut ids = engine.all_identities().unwrap();
        ids.sort_by_key(|i| i.0);
        assert_eq!(ids, vec![IdentityId(1), IdentityId(2)]);
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
    fn test_edge_case_fetch_nonexistent_entities() {
        let engine = create_test_engine();

        assert_eq!(
            engine.fetch_observation(ObservationId(999)).unwrap(),
            None
        );
        assert_eq!(engine.vertex_type(VertexId(999)).unwrap(), None);
        assert!(engine.out_edges(VertexId(999)).unwrap().is_empty());
        assert!(
            engine
                .query_support_set(IdentityId(999))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn test_edge_case_empty_batch_and_empty_queries() {
        let mut engine = create_test_engine();

        assert!(engine.apply_batch(&[]).is_ok());
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
