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
    /// Args:
    ///   backend: Key-value driver instance.
    ///
    /// Returns:
    ///   New StorageEngine instance.
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
