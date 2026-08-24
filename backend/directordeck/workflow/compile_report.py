from __future__ import annotations

"""Typed immutable evidence projected by the public compile report.

These contracts sit below the compiler and API schemas so both layers consume
the exact resolutions/evaluations produced during graph construction.  They do
not contain host observations beyond the already-bounded capability result.
"""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .contracts import (
    Backend,
    ContractModel,
    FeatureResolution,
    Identifier,
    JsonObject,
    ModelFamily,
    PositiveVersion,
    ResolvedFeatureImplementation,
    Sha256Digest,
)


class CompiledCapabilityReason(ContractModel):
    code: Identifier
    feature_id: Identifier | None = None
    segment_id: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    unit_id: Annotated[str, Field(min_length=1, max_length=256)] | None = None
    backend: Backend | None = None
    rule: Identifier
    message: Annotated[str, Field(min_length=1, max_length=4_096)]
    remediation: Annotated[str, Field(min_length=1, max_length=4_096)]
    safe_details: JsonObject = Field(default_factory=dict)


class CompiledCapabilityEvaluation(ContractModel):
    available: bool
    reasons: tuple[CompiledCapabilityReason, ...] = ()
    verified_contracts: tuple[Identifier, ...] = ()
    runtime_fingerprints: tuple[Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")], ...] = ()

    @model_validator(mode="after")
    def validate_evaluation(self) -> "CompiledCapabilityEvaluation":
        if self.available == bool(self.reasons):
            raise ValueError("capability availability must agree with reasons")
        if len(self.verified_contracts) != len(set(self.verified_contracts)):
            raise ValueError("verified capability contracts must be unique")
        if len(self.runtime_fingerprints) != len(set(self.runtime_fingerprints)):
            raise ValueError("capability runtime fingerprints must be unique")
        return self


class CompiledFeatureResolution(ContractModel):
    segment_id: Annotated[str, Field(min_length=1, max_length=128)]
    unit_id: Annotated[str, Field(min_length=1, max_length=256)]
    feature_id: Identifier
    version: PositiveVersion
    backend: Backend
    family: ModelFamily
    template_id: Identifier
    resolution: FeatureResolution
    adapter_fingerprint: Sha256Digest
    capability: CompiledCapabilityEvaluation


class CompiledFeatureNotice(ContractModel):
    segment_id: Annotated[str, Field(min_length=1, max_length=128)]
    unit_id: Annotated[str, Field(min_length=1, max_length=256)]
    feature_id: Identifier
    message: Annotated[str, Field(min_length=1, max_length=4_096)]


class CompiledUnitDigest(ContractModel):
    unit_id: Annotated[str, Field(min_length=1, max_length=256)]
    digest: JsonObject

    @model_validator(mode="after")
    def validate_digest(self) -> "CompiledUnitDigest":
        if set(self.digest) != {"algorithm", "value"}:
            raise ValueError("compiled unit digest has an unknown shape")
        if self.digest.get("algorithm") != "sha256-canonical-json-v1":
            raise ValueError("compiled unit digest must use canonical SHA-256")
        value = self.digest.get("value")
        if (
            not isinstance(value, str)
            or len(value) != 71
            or not value.startswith("sha256-")
            or any(character not in "0123456789abcdef" for character in value[7:])
        ):
            raise ValueError("compiled unit digest value is invalid")
        return self


class CompiledExecutionReportV2(ContractModel):
    source: Literal["v4_native_compile_adapter_v2"]
    host_capability_revision: Sha256Digest | None = None
    manifest: JsonObject
    plans: tuple[JsonObject, ...]
    families: tuple[ModelFamily, ...]
    unit_effective_execution_digests: tuple[CompiledUnitDigest, ...]
    feature_resolutions: Annotated[
        tuple[CompiledFeatureResolution, ...],
        Field(min_length=1, max_length=8_192),
    ]
    notices: Annotated[tuple[CompiledFeatureNotice, ...], Field(max_length=256)] = ()

    @model_validator(mode="after")
    def validate_report_identity(self) -> "CompiledExecutionReportV2":
        units = [item.unit_id for item in self.unit_effective_execution_digests]
        if len(units) != len(set(units)):
            raise ValueError("compiled report unit digests must be unique")
        resolution_keys = [
            (item.segment_id, item.unit_id, item.feature_id, item.version)
            for item in self.feature_resolutions
        ]
        if len(resolution_keys) != len(set(resolution_keys)):
            raise ValueError("compiled feature resolution identities must be unique")
        known = {
            (item.segment_id, item.unit_id, item.feature_id)
            for item in self.feature_resolutions
        }
        if any(
            (notice.segment_id, notice.unit_id, notice.feature_id) not in known
            for notice in self.notices
        ):
            raise ValueError("compiled notice has no matching feature resolution")
        return self


class NodeEmissionEvidenceV3(ContractModel):
    node_id: Annotated[str, Field(min_length=1, max_length=128)]
    class_type: Annotated[str, Field(min_length=1, max_length=128)]
    feature_id: Identifier
    implementation_id: Identifier


class CompiledFeatureUseV3(ContractModel):
    segment_id: Annotated[str, Field(min_length=1, max_length=128)]
    unit_id: Annotated[str, Field(min_length=1, max_length=256)]
    feature_id: Identifier
    version: PositiveVersion
    backend: Backend
    family: ModelFamily
    template_id: Identifier
    state: Literal["inactive", "applicable"]
    config_source: Literal["definition_default", "project", "segment", "context"]
    reason_code: Identifier | None = None
    implementation: ResolvedFeatureImplementation | None = None
    execution_identity: JsonObject | None = None
    runtime_pool_identity: JsonObject | None = None
    node_emissions: tuple[NodeEmissionEvidenceV3, ...] = ()

    @model_validator(mode="after")
    def validate_state(self) -> "CompiledFeatureUseV3":
        applicable = self.state == "applicable"
        if applicable != (self.implementation is not None):
            raise ValueError("applicable feature use requires one implementation")
        if applicable != (self.execution_identity is not None):
            raise ValueError("applicable feature use requires execution identity")
        if applicable and self.reason_code is not None:
            raise ValueError("applicable feature use cannot have inactive reason")
        if not applicable and (
            self.reason_code is None
            or self.runtime_pool_identity is not None
            or self.node_emissions
        ):
            raise ValueError("inactive feature use may contain only its reason")
        if any(item.feature_id != self.feature_id for item in self.node_emissions):
            raise ValueError("node emission owner does not match feature use")
        if self.implementation is not None and any(
            item.implementation_id != self.implementation.implementation_id
            for item in self.node_emissions
        ):
            raise ValueError("node emission implementation does not match feature use")
        return self


class CompiledExecutionReportV3(ContractModel):
    source: Literal["bundle6_native_compile_v3"] = "bundle6_native_compile_v3"
    manifest: JsonObject
    plans: tuple[JsonObject, ...]
    families: tuple[ModelFamily, ...]
    unit_effective_execution_digests: tuple[CompiledUnitDigest, ...]
    feature_resolutions: Annotated[
        tuple[CompiledFeatureUseV3, ...],
        Field(min_length=1, max_length=8_192),
    ]
    notices: Annotated[tuple[CompiledFeatureNotice, ...], Field(max_length=256)] = ()
    advisories: Annotated[tuple[CompiledCapabilityReason, ...], Field(max_length=256)] = ()

    @model_validator(mode="after")
    def validate_report_identity(self) -> "CompiledExecutionReportV3":
        units = [item.unit_id for item in self.unit_effective_execution_digests]
        if len(units) != len(set(units)):
            raise ValueError("compiled report unit digests must be unique")
        keys = [
            (item.segment_id, item.unit_id, item.feature_id, item.version)
            for item in self.feature_resolutions
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("compiled feature-use identities must be unique")
        node_ids = [
            (item.unit_id, node.node_id)
            for item in self.feature_resolutions
            for node in item.node_emissions
        ]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("each emitted node must have one feature owner")
        return self


__all__ = [
    "CompiledCapabilityEvaluation",
    "CompiledCapabilityReason",
    "CompiledExecutionReportV2",
    "CompiledExecutionReportV3",
    "CompiledFeatureUseV3",
    "CompiledFeatureNotice",
    "CompiledFeatureResolution",
    "CompiledUnitDigest",
    "NodeEmissionEvidenceV3",
]
