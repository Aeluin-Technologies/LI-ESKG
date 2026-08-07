//! Deterministic RDF, PROV-O, nanopublication, and SHACL export profiles.

use std::fmt::{self, Write};
use std::sync::Arc;

use li_core::{
    AssociationOutcome, DecisionAction, DecisionRecord, IdentityReference,
    InferenceRecord, ObservationEnvelope,
};
use thiserror::Error;

/// RDF compatibility profile used for uncertain proposition representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RdfProfile {
    /// RDF 1.1 reification-compatible output.
    Rdf11,
    /// RDF 1.2 triple terms and reifiers.
    Rdf12,
}

/// Error returned by interoperability configuration or serialization.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ProjectionError {
    /// Base IRI was empty, relative, or contained whitespace.
    #[error("projection base IRI must be absolute and contain no whitespace")]
    InvalidBaseIri,
    /// Inference and decision lineage did not match the observation.
    #[error("projection input lineage is inconsistent")]
    InvalidLineage,
    /// Writing into the provided string buffer failed.
    #[error("projection formatting failed")]
    Formatting,
}

impl From<fmt::Error> for ProjectionError {
    fn from(_: fmt::Error) -> Self {
        Self::Formatting
    }
}

/// Deterministic standards projection configured for one deployment namespace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InteroperabilityProjector {
    base: Arc<str>,
    profile: RdfProfile,
}

impl InteroperabilityProjector {
    /// Creates a projector for an absolute deployment-owned base IRI.
    ///
    /// # Errors
    ///
    /// Returns [`ProjectionError::InvalidBaseIri`] for an invalid base.
    pub fn new(
        base: impl Into<Arc<str>>,
        profile: RdfProfile,
    ) -> Result<Self, ProjectionError> {
        let base = base.into();
        if !valid_absolute_iri(&base) {
            return Err(ProjectionError::InvalidBaseIri);
        }
        Ok(Self { base, profile })
    }

    /// Writes a lossless decision/provenance projection into caller storage.
    ///
    /// Candidate propositions are discussed through RDF reification and are
    /// not asserted as current host facts. `output` is cleared but retains its
    /// allocation.
    ///
    /// # Errors
    ///
    /// Returns [`ProjectionError::InvalidLineage`] when record references do
    /// not agree.
    pub fn project_decision(
        &self,
        observation: &ObservationEnvelope,
        inference: &InferenceRecord,
        decision: &DecisionRecord,
        output: &mut String,
    ) -> Result<(), ProjectionError> {
        if inference.observation != observation.id() ||
            decision.inference != inference.id
        {
            return Err(ProjectionError::InvalidLineage);
        }
        output.clear();
        self.write_prefixes(output)?;
        writeln!(
            output,
            "<{}observation/{}> a prov:Entity, li:Observation ; li:source {} ; li:eventTime {} ; li:ingestionTime {} ; li:contentHash \"{}\" .",
            self.base,
            observation.id().0,
            observation.source().0,
            observation.event_time().as_micros(),
            observation.ingestion_time().as_micros(),
            Hex(observation.content_hash().as_bytes()),
        )?;
        writeln!(
            output,
            "<{}inference/{}> a prov:Activity, li:Inference ; prov:used <{}observation/{}> ; li:hostSnapshot {} ; li:candidateVersion {} ; li:solverVersion {} ; li:stoppingReason \"{:?}\" .",
            self.base,
            inference.id.0,
            self.base,
            observation.id().0,
            inference.provenance.host_snapshot.get(),
            inference.provenance.candidate_version,
            inference.diagnostics.solver_version,
            inference.diagnostics.stopping_reason,
        )?;
        for (index, artifact) in
            inference.provenance.providers.iter().enumerate()
        {
            writeln!(
                output,
                "<{}inference/{}> li:usedProvider [ a prov:Entity ; li:providerId {} ; li:schemaId {} ; li:modelVersion {} ; li:calibrationId {} ; li:ordinal {} ] .",
                self.base,
                inference.id.0,
                artifact.provider.0,
                artifact.schema.0,
                artifact.model_version,
                artifact.calibration_id,
                index,
            )?;
        }
        for (index, contribution) in inference.contributions.iter().enumerate()
        {
            writeln!(
                output,
                "_:score{} a li:ScoreContribution ; li:value {:.17} ; li:scoreSemantics \"{:?}\" ; li:providerId {} ; li:modelVersion {} ; li:calibrationId {} ; li:validityDomain \"{}\" .",
                index,
                contribution.value(),
                contribution.semantics(),
                contribution.provider().0,
                contribution.model_version(),
                contribution.calibration_id(),
                TurtleLiteral(contribution.validity_domain()),
            )?;
            writeln!(
                output,
                "<{}inference/{}> li:hasScoreContribution _:score{} .",
                self.base, inference.id.0, index,
            )?;
        }
        for (index, entry) in
            inference.distribution.entries().iter().enumerate()
        {
            self.write_alternative(
                output,
                observation,
                inference,
                index,
                &entry.outcome,
                entry.probability.value(),
            )?;
        }
        writeln!(
            output,
            "<{}decision/{}> a prov:Activity, li:Decision ; prov:used <{}inference/{}> ; li:policyVersion {} ; li:lossVersion {} ; li:action \"{}\" .",
            self.base,
            decision.id.0,
            self.base,
            inference.id.0,
            decision.policy_version,
            decision.loss_version,
            ActionLabel(&decision.action),
        )?;
        Ok(())
    }

    /// Writes one immutable nanopublication exchange unit.
    ///
    /// The assertion graph records the selected action, while provenance and
    /// publication-info graphs retain the responsible records and versions.
    pub fn project_nanopublication(
        &self,
        observation: &ObservationEnvelope,
        inference: &InferenceRecord,
        decision: &DecisionRecord,
        publisher_iri: &str,
        output: &mut String,
    ) -> Result<(), ProjectionError> {
        if !valid_absolute_iri(publisher_iri) {
            return Err(ProjectionError::InvalidBaseIri);
        }
        self.project_decision(observation, inference, decision, output)?;
        writeln!(
            output,
            "\n<{}nanopub/{}> a np:Nanopublication ; np:hasAssertion <{}nanopub/{}/assertion> ; np:hasProvenance <{}nanopub/{}/provenance> ; np:hasPublicationInfo <{}nanopub/{}/publicationInfo> .",
            self.base,
            decision.id.0,
            self.base,
            decision.id.0,
            self.base,
            decision.id.0,
            self.base,
            decision.id.0,
        )?;
        writeln!(
            output,
            "<{}nanopub/{}/assertion> {{ <{}observation/{}> li:hasDecision <{}decision/{}> . }}",
            self.base,
            decision.id.0,
            self.base,
            observation.id().0,
            self.base,
            decision.id.0,
        )?;
        writeln!(
            output,
            "<{}nanopub/{}/provenance> {{ <{}nanopub/{}/assertion> prov:wasGeneratedBy <{}decision/{}> . }}",
            self.base,
            decision.id.0,
            self.base,
            decision.id.0,
            self.base,
            decision.id.0,
        )?;
        writeln!(
            output,
            "<{}nanopub/{}/publicationInfo> {{ <{}nanopub/{}> prov:wasAttributedTo <{}> . }}",
            self.base, decision.id.0, self.base, decision.id.0, publisher_iri,
        )?;
        Ok(())
    }

    /// Writes baseline SHACL shapes for exported decision records.
    pub fn write_shacl(
        &self,
        output: &mut String,
    ) -> Result<(), ProjectionError> {
        output.clear();
        self.write_prefixes(output)?;
        writeln!(
            output,
            "li:DecisionShape a sh:NodeShape ; sh:targetClass li:Decision ;\n  sh:property [ sh:path prov:used ; sh:minCount 1 ; sh:maxCount 1 ; sh:class li:Inference ] ;\n  sh:property [ sh:path li:policyVersion ; sh:minCount 1 ; sh:maxCount 1 ] ;\n  sh:property [ sh:path li:lossVersion ; sh:minCount 1 ; sh:maxCount 1 ] ;\n  sh:property [ sh:path li:action ; sh:minCount 1 ; sh:maxCount 1 ] ."
        )?;
        writeln!(
            output,
            "li:InferenceShape a sh:NodeShape ; sh:targetClass li:Inference ;\n  sh:property [ sh:path prov:used ; sh:minCount 1 ; sh:maxCount 1 ; sh:class li:Observation ] ;\n  sh:property [ sh:path li:usedProvider ; sh:minCount 1 ] ;\n  sh:property [ sh:path li:hostSnapshot ; sh:minCount 1 ; sh:maxCount 1 ] ;\n  sh:property [ sh:path li:candidateVersion ; sh:minCount 1 ; sh:maxCount 1 ] ;\n  sh:property [ sh:path li:solverVersion ; sh:minCount 1 ; sh:maxCount 1 ] ;\n  sh:property [ sh:path li:stoppingReason ; sh:minCount 1 ; sh:maxCount 1 ] ."
        )?;
        writeln!(
            output,
            "li:ProviderArtifactShape a sh:NodeShape ; sh:targetObjectsOf li:usedProvider ;\n  sh:property [ sh:path li:providerId ; sh:minCount 1 ; sh:maxCount 1 ] ;\n  sh:property [ sh:path li:schemaId ; sh:minCount 1 ; sh:maxCount 1 ] ;\n  sh:property [ sh:path li:modelVersion ; sh:minCount 1 ; sh:maxCount 1 ] ;\n  sh:property [ sh:path li:calibrationId ; sh:minCount 1 ; sh:maxCount 1 ] ."
        )?;
        writeln!(
            output,
            "li:ScoreContributionShape a sh:NodeShape ; sh:targetClass li:ScoreContribution ;\n  sh:property [ sh:path li:value ; sh:minCount 1 ; sh:maxCount 1 ] ;\n  sh:property [ sh:path li:scoreSemantics ; sh:minCount 1 ; sh:maxCount 1 ; sh:in ( \"LogLikelihood\" \"LogLikelihoodRatio\" \"LogPotential\" \"CalibratedPosterior\" ) ] ;\n  sh:property [ sh:path li:modelVersion ; sh:minCount 1 ; sh:maxCount 1 ] ;\n  sh:property [ sh:path li:calibrationId ; sh:minCount 1 ; sh:maxCount 1 ] ."
        )?;
        writeln!(
            output,
            "li:AlternativeShape a sh:NodeShape ; sh:targetClass li:Alternative ;\n  sh:property [ sh:path li:probability ; sh:minCount 1 ; sh:maxCount 1 ] ;\n  sh:property [ sh:path prov:wasGeneratedBy ; sh:minCount 1 ; sh:maxCount 1 ; sh:class li:Inference ] ."
        )?;
        writeln!(
            output,
            "li:PromotionShape a sh:NodeShape ; sh:targetClass li:Promotion ;\n  sh:property [ sh:path li:promotionTarget ; sh:minCount 1 ; sh:maxCount 1 ] ;\n  sh:property [ sh:path prov:wasGeneratedBy ; sh:minCount 1 ; sh:maxCount 1 ; sh:class li:Decision ] ."
        )?;
        writeln!(
            output,
            "li:RevisionShape a sh:NodeShape ; sh:targetClass li:Revision ;\n  sh:or ( [ sh:property [ sh:path prov:invalidated ; sh:minCount 1 ] ] [ sh:property [ sh:path li:revokes ; sh:minCount 1 ] ] ) ."
        )?;
        writeln!(
            output,
            "li:MaterializationShape a sh:NodeShape ; sh:targetClass li:Materialization ;\n  sh:property [ sh:path li:responsibleDecision ; sh:minCount 1 ; sh:maxCount 1 ; sh:class li:Decision ] ;\n  sh:property [ sh:path li:commitVersion ; sh:minCount 1 ; sh:maxCount 1 ] ;\n  sh:property [ sh:path li:hostPredicate ; sh:minCount 1 ; sh:maxCount 1 ] ."
        )?;
        Ok(())
    }

    fn write_prefixes(
        &self,
        output: &mut String,
    ) -> Result<(), ProjectionError> {
        writeln!(output, "@prefix li: <{}vocab/> .", self.base)?;
        writeln!(output, "@prefix prov: <http://www.w3.org/ns/prov#> .")?;
        writeln!(
            output,
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> ."
        )?;
        writeln!(output, "@prefix sh: <http://www.w3.org/ns/shacl#> .")?;
        writeln!(output, "@prefix np: <http://www.nanopub.org/nschema#> .")?;
        Ok(())
    }

    fn write_alternative(
        &self,
        output: &mut String,
        observation: &ObservationEnvelope,
        inference: &InferenceRecord,
        index: usize,
        outcome: &AssociationOutcome,
        probability: f64,
    ) -> Result<(), ProjectionError> {
        let object = OutcomeTerm {
            base: &self.base,
            outcome,
        };
        match self.profile {
            RdfProfile::Rdf11 => writeln!(
                output,
                "_:alternative{} a li:Alternative, rdf:Statement ; rdf:subject <{}observation/{}> ; rdf:predicate li:candidateTarget ; rdf:object {} ; li:probability {:.17} ; prov:wasGeneratedBy <{}inference/{}> .",
                index,
                self.base,
                observation.id().0,
                object,
                probability,
                self.base,
                inference.id.0,
            )?,
            RdfProfile::Rdf12 => writeln!(
                output,
                "_:alternative{} a li:Alternative ; rdf:reifies <<( <{}observation/{}> li:candidateTarget {} )>> ; li:probability {:.17} ; prov:wasGeneratedBy <{}inference/{}> .",
                index,
                self.base,
                observation.id().0,
                object,
                probability,
                self.base,
                inference.id.0,
            )?,
        }
        Ok(())
    }
}

struct Hex<'a>(&'a [u8]);

struct TurtleLiteral<'a>(&'a str);

impl fmt::Display for TurtleLiteral<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for character in self.0.chars() {
            match character {
                '\\' => formatter.write_str("\\\\")?,
                '"' => formatter.write_str("\\\"")?,
                '\n' => formatter.write_str("\\n")?,
                '\r' => formatter.write_str("\\r")?,
                '\t' => formatter.write_str("\\t")?,
                _ => formatter.write_char(character)?,
            }
        }
        Ok(())
    }
}

fn valid_absolute_iri(value: &str) -> bool {
    !value.is_empty() &&
        value.contains(':') &&
        !value.chars().any(|character| {
            character.is_whitespace() || "<>\"{}".contains(character)
        })
}

impl fmt::Display for Hex<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

struct ActionLabel<'a>(&'a DecisionAction);

impl fmt::Display for ActionLabel<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.0 {
            DecisionAction::Assign(_) => "assign",
            DecisionAction::CreateIdentity => "new",
            DecisionAction::RejectNoise => "noise",
            DecisionAction::Abstain => "abstain",
        })
    }
}

struct OutcomeTerm<'a> {
    base: &'a str,
    outcome: &'a AssociationOutcome,
}

impl fmt::Display for OutcomeTerm<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.outcome {
            AssociationOutcome::Identity(IdentityReference::Latent(
                identity,
            )) => {
                write!(formatter, "<{}identity/{}>", self.base, identity.0)
            },
            AssociationOutcome::Identity(IdentityReference::Known(
                reference,
            )) => {
                write!(
                    formatter,
                    "<{}host/{}/{}>",
                    self.base,
                    reference.backend(),
                    Hex(reference.key().as_bytes())
                )
            },
            AssociationOutcome::New => formatter.write_str("li:newIdentity"),
            AssociationOutcome::Noise => formatter.write_str("li:noise"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use li_core::{
        BoundaryTreatment, CommitVersion, ContentHash, DecisionId,
        EvidenceError, InferenceId, InferenceProvenance, Modality,
        NormalizedDistribution, ObservationId, OutcomeProbability, PayloadRef,
        Probability, ProviderArtifact, ProviderId, QualityMetadata, SchemaId,
        ScoreContribution, ScoreSemantics, SolverDiagnostics,
        SolverStoppingReason, SourceId, Timestamp, VersionInterval,
    };

    use super::*;

    fn observation() -> Result<ObservationEnvelope, EvidenceError> {
        ObservationEnvelope::new(
            ObservationId(1),
            SourceId(2),
            Modality(3),
            Timestamp::from_micros(4),
            Timestamp::from_micros(5),
            PayloadRef::Inline(Bytes::from_static(b"x")),
            QualityMetadata::Opaque {
                schema: SchemaId(1),
                bytes: Bytes::new(),
            },
            ContentHash::new([0xab; 32]),
            None,
        )
    }

    fn records()
    -> Result<(InferenceRecord, DecisionRecord), Box<dyn std::error::Error>>
    {
        let distribution = NormalizedDistribution::new(
            vec![
                OutcomeProbability {
                    outcome: AssociationOutcome::New,
                    probability: Probability::new(0.8),
                },
                OutcomeProbability {
                    outcome: AssociationOutcome::Noise,
                    probability: Probability::new(0.2),
                },
            ],
            None,
        )?;
        let inference = InferenceRecord {
            id: InferenceId(1),
            observation: ObservationId(1),
            distribution: distribution.clone(),
            contributions: Vec::new().into_boxed_slice(),
            provenance: Arc::new(InferenceProvenance {
                providers: vec![ProviderArtifact {
                    provider: ProviderId(1),
                    schema: SchemaId(2),
                    model_version: 3,
                    calibration_id: 4,
                }]
                .into_boxed_slice(),
                candidate_version: 5,
                host_snapshot: CommitVersion::new(6),
                configuration_hash: ContentHash::new([7; 32]),
            }),
            diagnostics: Arc::new(SolverDiagnostics {
                solver_version: 8,
                tolerance: 1.0e-9,
                iterations: 1,
                residual: 0.0,
                damping_schedule: Vec::new().into_boxed_slice(),
                stopping_reason: SolverStoppingReason::Exact,
                boundary_treatment: BoundaryTreatment::Global,
                random_seed: None,
            }),
            validity: VersionInterval::current(CommitVersion::new(1)),
        };
        let decision = DecisionRecord::new(
            DecisionId(2),
            inference.id,
            DecisionAction::CreateIdentity,
            9,
            10,
            VersionInterval::current(CommitVersion::new(1)),
            &distribution,
        )?;
        Ok((inference, decision))
    }

    #[test]
    fn rdf11_projection_is_deterministic_and_does_not_assert_candidates()
    -> Result<(), Box<dyn std::error::Error>> {
        let projector = InteroperabilityProjector::new(
            "https://example.test/li/",
            RdfProfile::Rdf11,
        )?;
        let observation = observation()?;
        let (inference, decision) = records()?;
        let mut first = String::with_capacity(2048);
        let mut second = String::with_capacity(2048);
        projector.project_decision(
            &observation,
            &inference,
            &decision,
            &mut first,
        )?;
        projector.project_decision(
            &observation,
            &inference,
            &decision,
            &mut second,
        )?;
        assert_eq!(first, second);
        assert!(first.contains("rdf:Statement"));
        assert!(first.contains("prov:used"));
        assert!(first.contains("li:calibrationId 4"));
        Ok(())
    }

    #[test]
    fn rdf12_nanopublication_and_shacl_profiles_are_emitted()
    -> Result<(), Box<dyn std::error::Error>> {
        let projector =
            InteroperabilityProjector::new("urn:li:test:", RdfProfile::Rdf12)?;
        let observation = observation()?;
        let (inference, decision) = records()?;
        let mut output = String::new();
        projector.project_nanopublication(
            &observation,
            &inference,
            &decision,
            "https://publisher.test/agent",
            &mut output,
        )?;
        assert!(output.contains("rdf:reifies <<("));
        assert!(output.contains("np:Nanopublication"));
        projector.write_shacl(&mut output)?;
        assert!(output.contains("sh:NodeShape"));
        assert!(output.contains("sh:maxCount 1"));
        assert!(output.contains("li:ProviderArtifactShape"));
        assert!(output.contains("li:MaterializationShape"));
        Ok(())
    }

    #[test]
    fn score_validity_domains_are_escaped_as_turtle_literals()
    -> Result<(), Box<dyn std::error::Error>> {
        let projector =
            InteroperabilityProjector::new("urn:li:test:", RdfProfile::Rdf12)?;
        let observation = observation()?;
        let (mut inference, decision) = records()?;
        inference.contributions = vec![ScoreContribution::new(
            0.5,
            ScoreSemantics::CalibratedPosterior,
            ProviderId(1),
            2,
            3,
            "domain \"quoted\"\nnext",
        )?]
        .into_boxed_slice();
        let mut output = String::new();
        projector.project_decision(
            &observation,
            &inference,
            &decision,
            &mut output,
        )?;
        assert!(output.contains("domain \\\"quoted\\\"\\nnext"));
        Ok(())
    }

    #[test]
    fn invalid_lineage_is_rejected_before_output()
    -> Result<(), Box<dyn std::error::Error>> {
        let projector =
            InteroperabilityProjector::new("urn:li:test:", RdfProfile::Rdf11)?;
        let mut observation = observation()?;
        let (inference, decision) = records()?;
        observation = ObservationEnvelope::new(
            ObservationId(99),
            observation.source(),
            observation.modality(),
            observation.event_time(),
            observation.ingestion_time(),
            observation.payload().clone(),
            observation.quality().clone(),
            observation.content_hash(),
            None,
        )?;
        let mut output = String::from("stale");
        assert_eq!(
            projector.project_decision(
                &observation,
                &inference,
                &decision,
                &mut output
            ),
            Err(ProjectionError::InvalidLineage)
        );
        assert_eq!(output, "stale");
        Ok(())
    }
}
