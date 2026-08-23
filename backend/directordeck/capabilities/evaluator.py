from __future__ import annotations

"""The single live-host capability evaluator used by Stage-5 consumers."""

import re
from typing import Annotated

from pydantic import Field, model_validator

from ..workflow.contracts import (
    Backend,
    CapabilitySet,
    CompileContext,
    ContractModel,
    FeatureResolution,
    HostCapabilitySnapshot,
    Identifier,
    JsonObject,
    NodeContractRegistry,
    OperationalReadiness,
    RuntimeFingerprint,
    Sha256Digest,
    canonical_sha256,
)
from ..workflow.node_contracts import V4_OUTPUT_NEUTRAL_NODE_CLASSES


# Capability names are release contracts, not arbitrary keys supplied by a
# host snapshot.  A new tool/package therefore needs an explicit DirectorDeck
# release before an interpreter may depend on it.  This keeps a misspelling or
# a forged snapshot member from silently turning into an available feature.
REGISTERED_MEDIA_CAPABILITIES = frozenset({"ffmpeg", "ffprobe"})
REGISTERED_PACKAGE_CAPABILITIES = frozenset(
    {"ray", "static_ffmpeg", "torch", "xfuser"}
)
REGISTERED_RAYLIGHT_CAPABILITIES = frozenset({"cleanup", "installation"})
STRICT_ATTENTION_PYTORCH_RUNTIME_PROBE = "strict_attention.pytorch"
STRICT_ATTENTION_CK_INT8_RUNTIME_PROBE = "strict_attention.ck_int8"
STRICT_H3_SAGE_RUNTIME_PROBE = "strict_h3_sage"
_DEVICE_BOUND_RUNTIME_PROBES = frozenset(
    {
        STRICT_ATTENTION_CK_INT8_RUNTIME_PROBE,
        STRICT_H3_SAGE_RUNTIME_PROBE,
    }
)
_GLOBAL_RUNTIME_PROBES = frozenset({STRICT_ATTENTION_PYTORCH_RUNTIME_PROBE})
_RUNTIME_GPU_SCOPE = re.compile(r"^gpu_(0|[1-9][0-9]{0,2})$")


def _runtime_device_scope(device: str) -> str:
    if device in {"any", "default", "cpu"}:
        return device
    match = re.fullmatch(r"gpu:(0|[1-9][0-9]{0,2})", device)
    if match is None or int(match.group(1)) > 255:
        raise ValueError("runtime probe device must be any/default/cpu/gpu:N")
    return f"gpu_{match.group(1)}"


def runtime_probe_key(probe: str, *, device: str | None = None) -> str:
    """Build the stable evidence key shared by provider and evaluator."""

    if probe in _GLOBAL_RUNTIME_PROBES:
        if device is not None:
            raise ValueError("global runtime probe cannot select a device")
        return probe
    if probe not in _DEVICE_BOUND_RUNTIME_PROBES:
        raise ValueError("unknown Director runtime probe")
    if device is None:
        raise ValueError("device-bound runtime probe requires a device scope")
    return f"{probe}.{_runtime_device_scope(device)}"


def contextual_runtime_capability_id(probe: str, ctx: CompileContext) -> str:
    """Resolve one interpreter requirement from its exact placement context.

    Catalog contexts deliberately have no binding and therefore use the
    provider's derived ``any`` observation. Compile/preflight contexts bind the
    saved Standard placement exactly.
    """

    if probe in _GLOBAL_RUNTIME_PROBES:
        return f"runtime.{runtime_probe_key(probe)}"
    binding = getattr(ctx, "binding", None)
    device = getattr(binding, "device", None)
    if device is None:
        device = "any"
    if not isinstance(device, str):
        raise TypeError("runtime capability placement must be a string")
    return f"runtime.{runtime_probe_key(probe, device=device)}"


def _registered_runtime_probe_key(member: str) -> bool:
    if member in _GLOBAL_RUNTIME_PROBES:
        return True
    for probe in _DEVICE_BOUND_RUNTIME_PROBES:
        prefix = f"{probe}."
        if not member.startswith(prefix):
            continue
        scope = member.removeprefix(prefix)
        if scope in {"any", "default", "cpu"}:
            return True
        match = _RUNTIME_GPU_SCOPE.fullmatch(scope)
        return match is not None and int(match.group(1)) <= 255
    return False


class CapabilityReason(ContractModel):
    """Stable, privacy-safe failure shape shared by catalog and preflight."""

    code: Identifier
    feature_id: Identifier | None = None
    segment_id: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    unit_id: Annotated[str, Field(min_length=1, max_length=256)] | None = None
    backend: Backend | None = None
    rule: Identifier
    message: str = Field(min_length=1, max_length=4_096)
    remediation: str = Field(min_length=1, max_length=4_096)
    safe_details: JsonObject = Field(default_factory=dict)


class CapabilityEvaluation(ContractModel):
    available: bool
    reasons: tuple[CapabilityReason, ...] = ()
    verified_contracts: tuple[Identifier, ...] = ()
    runtime_fingerprints: tuple[RuntimeFingerprint, ...] = ()

    @model_validator(mode="after")
    def _validate_result(self) -> "CapabilityEvaluation":
        if self.available == bool(self.reasons):
            raise ValueError("available capability cannot contain reasons")
        if len(self.verified_contracts) != len(set(self.verified_contracts)):
            raise ValueError("verified contracts must be unique")
        if len(self.runtime_fingerprints) != len(set(self.runtime_fingerprints)):
            raise ValueError("runtime fingerprints must be unique")
        return self


def resolution_adapter_fingerprint(
    *,
    feature_id: str,
    feature_version: int,
    ctx: CompileContext,
    resolution: FeatureResolution,
) -> Sha256Digest:
    """Fingerprint only the exact resolved adapter selection for one context."""

    return canonical_sha256(
        {
            "schema_version": 1,
            "feature_id": feature_id,
            "feature_version": feature_version,
            "backend": ctx.backend,
            "family": ctx.family,
            "implementations": tuple(
                implementation.model_dump(mode="json")
                for implementation in resolution.implementations
            ),
        }
    )


def _reason(
    code: str,
    *,
    feature_id: str,
    ctx: CompileContext,
    segment_id: str | None,
    unit_id: str | None,
    rule: str,
    message: str,
    remediation: str,
    safe_details: dict[str, object] | None = None,
) -> CapabilityReason:
    return CapabilityReason(
        code=code,
        feature_id=feature_id,
        segment_id=segment_id,
        unit_id=unit_id,
        backend=ctx.backend,
        rule=rule,
        message=message,
        remediation=remediation,
        safe_details=safe_details or {},
    )


class CapabilityEvaluator:
    """Cross-bind resolution, release contracts, and observed host evidence."""

    __slots__ = ("_registry",)

    def __init__(self, registry: NodeContractRegistry) -> None:
        if not isinstance(registry, NodeContractRegistry):
            raise TypeError("capability evaluator requires NodeContractRegistry")
        self._registry = registry

    @property
    def registry(self) -> NodeContractRegistry:
        return self._registry

    def evaluate(
        self,
        *,
        feature_id: str,
        ctx: CompileContext,
        resolution: FeatureResolution,
        required_capabilities: CapabilitySet,
        snapshot: HostCapabilitySnapshot,
        readiness: OperationalReadiness | None,
        segment_id: str | None = None,
        unit_id: str | None = None,
    ) -> CapabilityEvaluation:
        if not isinstance(resolution, FeatureResolution):
            raise TypeError("capability evaluation requires FeatureResolution")
        if not isinstance(required_capabilities, CapabilitySet):
            raise TypeError("capability evaluation requires CapabilitySet")
        if not isinstance(snapshot, HostCapabilitySnapshot):
            raise TypeError("capability evaluation requires HostCapabilitySnapshot")
        if readiness is not None and not isinstance(readiness, OperationalReadiness):
            raise TypeError("readiness must be OperationalReadiness or None")

        reasons: list[CapabilityReason] = []
        verified_contracts: list[str] = []

        expected_node_capability_ids = tuple(
            "node." + "".join(
                character if character.isalnum() or character in "_.:-" else "_"
                for character in implementation.class_type
            )
            for implementation in resolution.implementations
        )
        declared_node_capability_ids = required_capabilities.in_namespace(
            "node"
        )
        if (
            declared_node_capability_ids != expected_node_capability_ids
            or (
                resolution.state == "noop"
                and bool(required_capabilities.ids)
            )
        ):
            reasons.append(
                _reason(
                    "capability_declaration_mismatch",
                    feature_id=feature_id,
                    ctx=ctx,
                    segment_id=segment_id,
                    unit_id=unit_id,
                    rule="interpreter_required_capabilities",
                    message="The feature capability declaration does not match its resolved adapters.",
                    remediation="Update the feature interpreter and template as one versioned change.",
                )
            )

        expected_node_capability_id_set = set(expected_node_capability_ids)
        for capability_id in required_capabilities.ids:
            namespace, _, member = capability_id.partition(".")
            if namespace == "node":
                if capability_id not in expected_node_capability_id_set:
                    reasons.append(
                        _reason(
                            "unknown_capability",
                            feature_id=feature_id,
                            ctx=ctx,
                            segment_id=segment_id,
                            unit_id=unit_id,
                            rule="registered_capability",
                            message="The feature declares an unsupported capability.",
                            remediation="Update the feature interpreter and DirectorDeck as one versioned release.",
                            safe_details={"capability_id": capability_id},
                        )
                    )
                continue
            if namespace == "media" and member in REGISTERED_MEDIA_CAPABILITIES:
                observed = snapshot.media_tool_status.get(member)
                if observed is None or not observed.available:
                    reasons.append(
                        _reason(
                            "media_tool_unavailable",
                            feature_id=feature_id,
                            ctx=ctx,
                            segment_id=segment_id,
                            unit_id=unit_id,
                            rule="host_media_tool_status",
                            message="A required media tool is unavailable.",
                            remediation="Install the required media tools and run preflight again.",
                            safe_details={"tool": member},
                        )
                    )
                continue
            if namespace == "package" and member in REGISTERED_PACKAGE_CAPABILITIES:
                observed = snapshot.importable_packages.get(member)
                if observed is None or not observed.importable:
                    reasons.append(
                        _reason(
                            "package_unavailable",
                            feature_id=feature_id,
                            ctx=ctx,
                            segment_id=segment_id,
                            unit_id=unit_id,
                            rule="host_package_importability",
                            message="A required runtime package is unavailable.",
                            remediation="Install the required runtime package and restart ComfyUI.",
                            safe_details={"package": member},
                        )
                    )
                continue
            if namespace == "runtime" and _registered_runtime_probe_key(member):
                # Runtime probes are advisory observations.  They help explain
                # a later ComfyUI execution failure, but never predictively
                # reject a compile or submission.
                continue
            if (
                namespace == "raylight"
                and member in REGISTERED_RAYLIGHT_CAPABILITIES
            ):
                # Package/provenance state is advisory.  Only an objectively
                # absent class_type is known to be unexecutable before ComfyUI
                # receives the prompt.
                if (
                    member == "cleanup"
                    and "DirectorDeckRayKill" not in snapshot.node_registry
                ):
                    reasons.append(
                        _reason(
                            "raylight_cleanup_unavailable",
                            feature_id=feature_id,
                            ctx=ctx,
                            segment_id=segment_id,
                            unit_id=unit_id,
                            rule="raylight_cleanup_contract",
                            message="The required Director RayKill node is unavailable.",
                            remediation=(
                                "Enable DirectorDeck's bundled RayLight nodes and "
                                "restart ComfyUI."
                            ),
                            safe_details={"class_type": "DirectorDeckRayKill"},
                        )
                    )
                continue
            reasons.append(
                _reason(
                    "unknown_capability",
                    feature_id=feature_id,
                    ctx=ctx,
                    segment_id=segment_id,
                    unit_id=unit_id,
                    rule="registered_capability",
                    message="The feature declares an unsupported capability.",
                    remediation="Update the feature interpreter and DirectorDeck as one versioned release.",
                    safe_details={"capability_id": capability_id},
                )
            )

        if resolution.state == "noop":
            return CapabilityEvaluation(
                available=not reasons,
                reasons=tuple(reasons),
            )

        requested_gpu_indices = _runtime_gpu_indices(ctx)
        cuda_gpu_indices = frozenset(
            item.logical_index
            for item in snapshot.gpu_inventory
            if item.backend == "cuda"
        )
        if ctx.backend == "raylight" and len(cuda_gpu_indices) < 2:
            reasons.append(
                _reason(
                    "raylight_cuda_unavailable",
                    feature_id=feature_id,
                    ctx=ctx,
                    segment_id=segment_id,
                    unit_id=unit_id,
                    rule="logical_gpu_inventory",
                    message="RayLight requires at least two logical CUDA GPUs.",
                    remediation="Use a host with at least two CUDA GPUs or select the Standard backend.",
                    safe_details={"cuda_gpu_count": len(cuda_gpu_indices)},
                )
            )
        else:
            invalid_gpu_indices = tuple(
                index
                for index in requested_gpu_indices
                if index not in cuda_gpu_indices
            )
            if invalid_gpu_indices:
                reasons.append(
                    _reason(
                        "invalid_runtime_gpu_indices",
                        feature_id=feature_id,
                        ctx=ctx,
                        segment_id=segment_id,
                        unit_id=unit_id,
                        rule="logical_gpu_inventory",
                        message="The selected logical CUDA GPU indices are unavailable.",
                        remediation="Select only logical CUDA GPUs reported by the current host.",
                        safe_details={"invalid_indices": invalid_gpu_indices},
                    )
                )

        for implementation in resolution.implementations:
            try:
                contract = self._registry.validate_implementation(
                    implementation,
                    output_affecting=(
                        implementation.class_type
                        not in V4_OUTPUT_NEUTRAL_NODE_CLASSES
                    ),
                    model_family=ctx.family,
                    backend=ctx.backend,
                )
            except (KeyError, ValueError):
                reasons.append(
                    _reason(
                        "adapter_contract_mismatch",
                        feature_id=feature_id,
                        ctx=ctx,
                        segment_id=segment_id,
                        unit_id=unit_id,
                        rule="node_contract_registry",
                        message="The resolved adapter does not match DirectorDeck's compiler contract.",
                        remediation="Refresh DirectorDeck's feature configuration and try again.",
                        safe_details={"class_type": implementation.class_type},
                    )
                )
                continue

            if implementation.class_type not in snapshot.node_registry:
                reasons.append(
                    _reason(
                        "node_unavailable",
                        feature_id=feature_id,
                        ctx=ctx,
                        segment_id=segment_id,
                        unit_id=unit_id,
                        rule="host_node_registry",
                        message="The mapped ComfyUI node is unavailable.",
                        remediation="Install or enable the mapped node and restart ComfyUI.",
                        safe_details={"class_type": implementation.class_type},
                    )
                )
                continue
            # Module/source/provenance, object-info compatibility and live
            # fingerprints are advisory.  A present class_type is submitted to
            # ComfyUI, which is the authority that executes and validates it.
            if contract.contract_id not in verified_contracts:
                verified_contracts.append(contract.contract_id)

        return CapabilityEvaluation(
            available=not reasons,
            reasons=tuple(reasons),
            verified_contracts=tuple(verified_contracts),
            # Retained in the schema for persisted-report compatibility only;
            # live module fingerprints are no longer generated or compared.
            runtime_fingerprints=(),
        )


def _runtime_gpu_indices(ctx: CompileContext) -> tuple[int, ...]:
    if ctx.backend != "raylight":
        return ()
    binding = getattr(ctx, "binding", None)
    raylight = getattr(binding, "raylight", None)
    values = getattr(raylight, "gpu_select", ())
    if not isinstance(values, (tuple, list)):
        return ()
    return tuple(index for index in values if type(index) is int and index >= 0)


__all__ = [
    "CapabilityEvaluation",
    "CapabilityEvaluator",
    "CapabilityReason",
    "REGISTERED_MEDIA_CAPABILITIES",
    "REGISTERED_PACKAGE_CAPABILITIES",
    "REGISTERED_RAYLIGHT_CAPABILITIES",
    "STRICT_ATTENTION_CK_INT8_RUNTIME_PROBE",
    "STRICT_ATTENTION_PYTORCH_RUNTIME_PROBE",
    "STRICT_H3_SAGE_RUNTIME_PROBE",
    "contextual_runtime_capability_id",
    "resolution_adapter_fingerprint",
    "runtime_probe_key",
]
