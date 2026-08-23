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


__all__ = [
    "CompiledCapabilityEvaluation",
    "CompiledCapabilityReason",
    "CompiledExecutionReportV2",
    "CompiledFeatureNotice",
    "CompiledFeatureResolution",
    "CompiledUnitDigest",
]
