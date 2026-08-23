from __future__ import annotations

"""Bounded runtime and control evidence captured by new v5 jobs.

The persisted ``jobs.settings_snapshot`` column is historical wire storage,
not permission to copy the complete live settings authority. New jobs record
only selected placement, adapters actually resolved into their plan, and the
minimal immutable client routing evidence needed to recover progress monitors.
"""

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from ..schemas import LoraFeatureParams, RuntimeSettingsV3, UnifiedTimelineDraftV5
from .compile_report import CompiledExecutionReportV2, CompiledFeatureResolution
from .contracts import ContractModel, FrozenMap, JsonValue, ModelFamily
from .execution import CompiledExecutionPlan
from .lora_factory import LoraLoaderBindingKey, ResolvedLoraAdapter
from .v5_compat import (
    V5RayLightRuntimeProjection,
    V5RuntimeCurrentnessProjection,
    V5RuntimeFamilyProjection,
    project_v5_runtime_currentness,
    resolve_v5_lora_adapters,
)


class JobRayLightRuntimeProjectionV1(ContractModel):
    gpu_select: Annotated[tuple[int, ...], Field(min_length=2, max_length=64)]
    ulysses_degree: Annotated[int, Field(ge=1, le=64)]
    ring_degree: Annotated[int, Field(ge=1, le=64)]
    cfg_degree: Annotated[int, Field(ge=1, le=64)]
    dp_degree: Annotated[int, Field(ge=1, le=64)]
    fsdp: bool
    cpu_offload: bool

    @field_validator("gpu_select", mode="before")
    @classmethod
    def restore_wire_gpu_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class JobRuntimeFamilyProjectionV1(ContractModel):
    family: ModelFamily
    backend: Literal["standard", "raylight"]
    device: Annotated[str, Field(min_length=1, max_length=128)]
    raylight_profile: JobRayLightRuntimeProjectionV1 | None

    @model_validator(mode="after")
    def validate_backend_projection(self) -> "JobRuntimeFamilyProjectionV1":
        if (self.backend == "raylight") != (self.raylight_profile is not None):
            raise ValueError("RayLight backend and runtime profile must agree")
        return self


class JobRuntimeProjectionV1(ContractModel):
    memory_policy: Literal["keep_resident"]
    raylight_residency_policy: Literal[
        "release_after_sampling", "keep_until_switch"
    ] | None
    multi_gpu_enabled: bool | None
    families: Annotated[
        tuple[JobRuntimeFamilyProjectionV1, ...],
        Field(min_length=1, max_length=2),
    ]
    clip_device: Annotated[str, Field(min_length=1, max_length=128)]
    video_vae_device: Annotated[str, Field(min_length=1, max_length=128)]
    audio_vae_device: Annotated[str, Field(min_length=1, max_length=128)] | None

    @field_validator("families", mode="before")
    @classmethod
    def restore_wire_family_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_projection(self) -> "JobRuntimeProjectionV1":
        families = [item.family for item in self.families]
        if len(families) != len(set(families)):
            raise ValueError("job runtime family projections must be unique")
        uses_raylight = any(item.backend == "raylight" for item in self.families)
        if uses_raylight != (self.raylight_residency_policy is not None):
            raise ValueError("RayLight residency projection is inconsistent")
        if uses_raylight != (self.multi_gpu_enabled is not None):
            raise ValueError("RayLight multi-GPU projection is inconsistent")
        return self


class ResolvedJobLoraAdapterV1(ContractModel):
    family: ModelFamily
    backend: Literal["standard", "raylight"]
    adapter_id: Annotated[str, Field(min_length=1, max_length=64)]
    binding: LoraLoaderBindingKey | None
    class_type: Annotated[str, Field(min_length=1, max_length=256)]
    node_contract_id: Annotated[str, Field(min_length=1, max_length=256)]
    semantic_version: Annotated[str, Field(min_length=1, max_length=128)]
    options: FrozenMap[str, JsonValue] = Field(default_factory=dict)
    runtime_fingerprint: Annotated[
        str,
        Field(pattern=r"^sha256:[0-9a-f]{64}$"),
    ]

    @model_validator(mode="after")
    def validate_binding(self) -> "ResolvedJobLoraAdapterV1":
        if self.backend == "standard":
            if self.binding is None or self.binding.family != self.family:
                raise ValueError("Standard job LoRA requires its exact binding")
        elif self.binding is not None or self.adapter_id != "ray_lora":
            raise ValueError("RayLight job LoRA must use fixed adapter evidence")
        return self


class JobControlEvidenceV1(ContractModel):
    """Immutable control-plane routing captured when the job is admitted."""

    progress_client_id: Annotated[
        str,
        Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$"),
    ]


class JobRuntimeSnapshotV1(ContractModel):
    snapshot_schema_version: Literal[1] = 1
    runtime_projection: JobRuntimeProjectionV1
    resolved_lora_adapters: Annotated[
        tuple[ResolvedJobLoraAdapterV1, ...], Field(max_length=2)
    ] = ()
    # Optional only so bounded snapshots written before this control evidence
    # was introduced remain readable. Every newly admitted job populates it.
    control_evidence: JobControlEvidenceV1 | None = None

    @field_validator("resolved_lora_adapters", mode="before")
    @classmethod
    def restore_wire_adapter_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_adapters(self) -> "JobRuntimeSnapshotV1":
        families = [item.family for item in self.resolved_lora_adapters]
        if len(families) != len(set(families)):
            raise ValueError("resolved job LoRA adapter families must be unique")
        runtime_families = {item.family for item in self.runtime_projection.families}
        if not set(families) <= runtime_families:
            raise ValueError("job LoRA adapter is outside the runtime projection")
        return self

    def family_map(self) -> dict[ModelFamily, JobRuntimeFamilyProjectionV1]:
        return {item.family: item for item in self.runtime_projection.families}

    def lora_adapter_map(self) -> dict[ModelFamily, ResolvedJobLoraAdapterV1]:
        return {item.family: item for item in self.resolved_lora_adapters}

    def has_same_execution_identity(self, other: "JobRuntimeSnapshotV1") -> bool:
        """Compare execution evidence without treating monitor routing as work."""

        return (
            self.runtime_projection == other.runtime_projection
            and self.resolved_lora_adapters == other.resolved_lora_adapters
        )


def validate_job_runtime_snapshot_creative_binding(
    snapshot: JobRuntimeSnapshotV1,
    draft: UnifiedTimelineDraftV5,
    segment_ids: list[str] | None,
) -> None:
    """Cross-bind immutable runtime evidence to its captured creative input.

    The runtime snapshot deliberately omits the live mapping table. Historical
    readers can still prove that every selected family, placement and resolved
    adapter belongs to the same captured v5 job without consulting current
    settings or re-resolving a mapping.
    """

    enabled = tuple(segment for segment in draft.segments if segment.enabled)
    if segment_ids is None:
        selected = enabled
    else:
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("captured segment selection contains duplicates")
        selected_ids = set(segment_ids)
        enabled_ids = {segment.id for segment in enabled}
        if not selected_ids or not selected_ids <= enabled_ids:
            raise ValueError("captured segment selection is not an enabled subset")
        selected = tuple(
            segment for segment in enabled if segment.id in selected_ids
        )
    if not selected:
        raise ValueError("captured segment selection contains no enabled segment")

    selected_families = {
        family
        for family in ("fl2va", "ref2va")
        if any(segment.mode == family for segment in selected)
    }
    family_map = snapshot.family_map()
    if set(family_map) != selected_families:
        raise ValueError("job runtime projection does not match selected families")

    lora_selection = draft.features.project.get("lora")
    active_loras: dict[ModelFamily, tuple[str, str]] = {}
    if lora_selection is not None and lora_selection.enabled:
        params = LoraFeatureParams.model_validate(lora_selection.params)
        for family in selected_families:
            family_selection = params.by_family[family]
            if not family_selection.enabled:
                continue
            model_filename = getattr(draft.model_stack, family).filename
            if model_filename is None or family_selection.filename is None:
                raise ValueError("active job LoRA binding is incomplete")
            active_loras[family] = (
                model_filename,
                family_selection.filename,
            )

    adapter_map = snapshot.lora_adapter_map()
    if set(adapter_map) != set(active_loras):
        raise ValueError("job runtime adapter evidence does not match active LoRA slots")
    for family, (model_filename, lora_filename) in active_loras.items():
        runtime_family = family_map[family]
        adapter = adapter_map[family]
        if adapter.backend != runtime_family.backend:
            raise ValueError("job runtime adapter backend does not match placement")
        if adapter.backend == "standard":
            expected_binding = LoraLoaderBindingKey(
                family=family,
                model_filename=model_filename,
                lora_filename=lora_filename,
            )
            if adapter.binding != expected_binding:
                raise ValueError("job runtime adapter binding does not match creative input")


def _ray_projection(
    value: V5RayLightRuntimeProjection | None,
) -> JobRayLightRuntimeProjectionV1 | None:
    if value is None:
        return None
    return JobRayLightRuntimeProjectionV1(
        gpu_select=value.gpu_select,
        ulysses_degree=value.ulysses_degree,
        ring_degree=value.ring_degree,
        cfg_degree=value.cfg_degree,
        dp_degree=value.dp_degree,
        fsdp=value.fsdp,
        cpu_offload=value.cpu_offload,
    )


def _family_projection(
    value: V5RuntimeFamilyProjection,
) -> JobRuntimeFamilyProjectionV1:
    return JobRuntimeFamilyProjectionV1(
        family=value.family,
        backend=value.backend,
        device=value.device,
        raylight_profile=_ray_projection(value.raylight_profile),
    )


def _runtime_projection(
    value: V5RuntimeCurrentnessProjection,
) -> JobRuntimeProjectionV1:
    return JobRuntimeProjectionV1(
        memory_policy=value.memory_policy,  # type: ignore[arg-type]
        raylight_residency_policy=value.raylight_residency_policy,  # type: ignore[arg-type]
        multi_gpu_enabled=value.multi_gpu_enabled,
        families=tuple(_family_projection(item) for item in value.families),
        clip_device=value.clip_device,
        video_vae_device=value.video_vae_device,
        audio_vae_device=value.audio_vae_device,
    )


def _compiled_report(plan: CompiledExecutionPlan) -> CompiledExecutionReportV2:
    report = plan.compile_report
    if isinstance(report, CompiledExecutionReportV2):
        return report
    return CompiledExecutionReportV2.model_validate(report)


def _active_lora_resolutions(
    report: CompiledExecutionReportV2,
    family: ModelFamily,
) -> tuple[CompiledFeatureResolution, ...]:
    return tuple(
        item
        for item in report.feature_resolutions
        if item.feature_id == "lora"
        and item.family == family
        and item.resolution.state == "active"
    )


def _snapshot_adapter(
    family: ModelFamily,
    expected: ResolvedLoraAdapter,
    report: CompiledExecutionReportV2,
    *,
    lora_filename: str,
    strength: float,
) -> ResolvedJobLoraAdapterV1:
    candidates = _active_lora_resolutions(report, family)
    if not candidates:
        raise ValueError("compiled plan omitted an active LoRA resolution")
    if expected.adapter.backend == "standard":
        if expected.binding is None:
            raise ValueError("resolved Standard LoRA omitted its exact binding")
        expected_report_binding: dict[str, object] = expected.binding.model_dump(
            mode="json"
        )
    else:
        if expected.binding is not None:
            raise ValueError("resolved RayLight LoRA unexpectedly has a Standard binding")
        expected_report_binding = {
            "backend": "raylight",
            "family": family,
            "lora_filename": lora_filename,
        }
    snapshots: set[tuple[str, str, str, str, str]] = set()
    implementation = None
    for candidate in candidates:
        details = candidate.resolution.resolution_details
        if (
            candidate.backend != expected.adapter.backend
            or candidate.family != family
            or details.get("backend") != expected.adapter.backend
            or details.get("family") != family
        ):
            raise ValueError("compiled LoRA backend/family drifted from creative input")
        if details.get("adapter_id") != expected.adapter.adapter_id:
            raise ValueError("compiled LoRA adapter id drifted")
        if details.get("binding") != expected_report_binding:
            raise ValueError("compiled LoRA binding drifted from creative input")
        if details.get("strength") != strength:
            raise ValueError("compiled LoRA strength drifted from creative input")
        if details.get("loader_options") != dict(expected.options):
            raise ValueError("compiled LoRA loader options drifted")
        implementations = candidate.resolution.implementations
        if len(implementations) != 1:
            raise ValueError("compiled LoRA must resolve exactly one implementation")
        item = implementations[0]
        if item.class_type != expected.adapter.class_type:
            raise ValueError("compiled LoRA implementation drifted from adapter")
        snapshots.add(
            (
                item.class_type,
                item.implementation_id,
                item.semantic_version,
                item.runtime_fingerprint,
                candidate.adapter_fingerprint,
            )
        )
        implementation = item
    if len(snapshots) != 1 or implementation is None:
        raise ValueError("compiled family LoRA evidence is inconsistent")
    return ResolvedJobLoraAdapterV1(
        family=family,
        backend=expected.adapter.backend,
        adapter_id=expected.adapter.adapter_id,
        binding=expected.binding,
        class_type=implementation.class_type,
        node_contract_id=implementation.implementation_id,
        semantic_version=implementation.semantic_version,
        options=dict(expected.options),
        runtime_fingerprint=implementation.runtime_fingerprint,
    )


def build_job_runtime_snapshot(
    draft: UnifiedTimelineDraftV5,
    segment_ids: list[str] | None,
    settings: RuntimeSettingsV3,
    plan: CompiledExecutionPlan,
) -> JobRuntimeSnapshotV1:
    runtime = project_v5_runtime_currentness(draft, segment_ids, settings)
    expected = resolve_v5_lora_adapters(draft, settings, segment_ids)
    report = _compiled_report(plan)
    lora_selection = draft.features.project.get("lora")
    lora_params = (
        LoraFeatureParams.model_validate(lora_selection.params)
        if lora_selection is not None and lora_selection.enabled
        else None
    )
    adapters_list: list[ResolvedJobLoraAdapterV1] = []
    for item in expected:
        if lora_params is None:
            raise ValueError("resolved job LoRA has no active creative selection")
        family_selection = lora_params.by_family[item.family]
        if not family_selection.enabled or family_selection.filename is None:
            raise ValueError("resolved job LoRA has no active family binding")
        adapters_list.append(
            _snapshot_adapter(
                item.family,
                item.resolution,
                report,
                lora_filename=family_selection.filename,
                strength=family_selection.strength,
            )
        )
    adapters = tuple(adapters_list)
    active_families = {
        item.family
        for item in report.feature_resolutions
        if item.feature_id == "lora" and item.resolution.state == "active"
    }
    if active_families != {item.family for item in adapters}:
        raise ValueError("compiled plan contains unexpected active LoRA evidence")
    return JobRuntimeSnapshotV1(
        runtime_projection=_runtime_projection(runtime),
        resolved_lora_adapters=adapters,
        control_evidence=JobControlEvidenceV1(
            progress_client_id=settings.client_id,
        ),
    )


__all__ = [
    "JobRayLightRuntimeProjectionV1",
    "JobControlEvidenceV1",
    "JobRuntimeFamilyProjectionV1",
    "JobRuntimeProjectionV1",
    "JobRuntimeSnapshotV1",
    "ResolvedJobLoraAdapterV1",
    "build_job_runtime_snapshot",
    "validate_job_runtime_snapshot_creative_binding",
]
