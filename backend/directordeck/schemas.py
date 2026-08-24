from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
import json
import math
import re
import secrets
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from .h3_capabilities import H3_REFERENCE_LIMITS
from .workflow.compile_report import (
    CompiledCapabilityEvaluation,
    CompiledCapabilityReason,
    CompiledFeatureUseV3,
    CompiledFeatureNotice,
)
from .workflow.canonical import utf16_sort_key
from .workflow.contracts import FeatureResolution


GenerationMode = Literal["t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"]
TimelineMode = Literal["fl2va", "ref2va"]
SchedulerName = Literal["simple", "normal", "karras", "beta"]
JobMode = GenerationMode | Literal["timeline"]
JobStatus = Literal[
    "queued",
    "preparing",
    "running",
    "succeeded",
    "failed",
    "cancelling",
    "cancelled",
]
AssetKind = Literal["image", "audio", "video"]
DeviceTarget = Annotated[str, Field(pattern=r"^(default|cpu|gpu:(0|[1-9][0-9]*))$")]
VaeDeviceTarget = Annotated[str, Field(pattern=r"^(default|gpu:(0|[1-9][0-9]*))$")]
StandardLoraLoader: TypeAlias = Literal[
    "dedicated", "bypass_model_only", "model_only", "minimax_h3_turbo"
]

MODE_ORDER: tuple[GenerationMode, ...] = ("t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v")
MINIMAX_H3_PROMPT_MAX_CHARACTERS = 7_000
# Timeline revisions cross the JSON boundary and are compared exactly by the
# browser. Keep the durable counter inside JavaScript's safe-integer range just
# like timeline seeds; silently rounding a CAS value would defeat the lock.
MAX_TIMELINE_REVISION = 2**53 - 1
MAX_FEATURES_PER_SCOPE = 64
MAX_FEATURE_PARAMETER_DEPTH = 8
MAX_FEATURE_OBJECT_KEYS = 64
MAX_FEATURE_ARRAY_ITEMS = 128
MAX_FEATURE_STRING_LENGTH = 4_096
MAX_FEATURE_PARAMS_BYTES = 65_536
_FEATURE_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,63}$")
MODEL_ROUTE: dict[GenerationMode | TimelineMode, TimelineMode] = {
    "t2v": "fl2va",
    "i2v": "fl2va",
    "fl2v": "fl2va",
    "r2v": "ref2va",
    "v2v": "ref2va",
    "rv2v": "ref2va",
    "fl2va": "fl2va",
    "ref2va": "ref2va",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class StorageStatusRead(StrictModel):
    """The fixed database location of this embedded Director process."""

    active_database_path: str


class ModelBinding(StrictModel):
    filename: Annotated[str, Field(min_length=1, max_length=1024)]
    device: DeviceTarget = "default"


class RayLightProfile(StrictModel):
    gpu_select: Annotated[list[Annotated[int, Field(ge=0, le=255)]], Field(min_length=1, max_length=8)] = Field(
        default_factory=lambda: [0]
    )
    ulysses_degree: Annotated[int, Field(ge=1, le=8)] = 1
    ring_degree: Annotated[int, Field(ge=1, le=8)] = 1
    cfg_degree: Literal[1] = 1
    dp_degree: Literal[1] = 1
    # FSDP stays fail-closed in native timeline v1. The installed RayLight
    # cleanup path has not yet proved that FSDP actor CUDA allocations are
    # released before a later Standard unit uses the same endpoint.
    fsdp: Literal[False] = False
    cpu_offload: Literal[False] = False

    @model_validator(mode="after")
    def validate_topology(self) -> "RayLightProfile":
        if len(set(self.gpu_select)) != len(self.gpu_select):
            raise ValueError("raylight gpu_select entries must be unique")
        world_size = len(self.gpu_select)
        topology_size = (
            self.ulysses_degree
            * self.ring_degree
            * self.cfg_degree
            * self.dp_degree
        )
        if topology_size != world_size:
            raise ValueError(
                "raylight topology product (ulysses_degree * ring_degree * "
                "cfg_degree * dp_degree) must equal selected GPU count"
            )
        return self


class StandardLoraLoaderOverride(StrictModel):
    """A visible user decision scoped to one exact LoRA/base-model pair."""

    loader: StandardLoraLoader
    lora_name: Annotated[str, Field(min_length=1, max_length=1024)]
    model_filename: Annotated[str, Field(min_length=1, max_length=1024)]


class DiffusionModelBinding(ModelBinding):
    """A diffusion-model slot and its optional model-only LoRA.

    LoRAs are kept on the two shared diffusion-family settings slots rather
    than on mode drafts.  This preserves the product rule that model/runtime
    configuration is shared while every mode's creative configuration stays
    independent.
    """

    lora_name: Annotated[str, Field(min_length=1, max_length=1024)] | None = None
    lora_strength: Annotated[float, Field(ge=-10, le=10, allow_inf_nan=False)] = 1.0
    lora_loader: Literal[
        "auto", "dedicated", "bypass_model_only", "model_only", "minimax_h3_turbo"
    ] = "auto"
    # This is deliberately separate from the obsolete ``lora_loader`` wire
    # field.  Old databases and browser WAL entries may still carry a hidden
    # non-auto value there; only this new, visible exact mapping authorizes a
    # Standard loader choice.
    standard_lora_loader_override: StandardLoraLoaderOverride | None = None
    lora_low_vram: bool = False
    backend: Literal["auto", "standard", "raylight"] = "auto"
    raylight: RayLightProfile = Field(default_factory=RayLightProfile)

    @model_validator(mode="after")
    def validate_standard_lora_loader_override(self) -> "DiffusionModelBinding":
        override = self.standard_lora_loader_override
        if override is not None and (
            self.lora_name is None
            or override.lora_name != self.lora_name
            or override.model_filename != self.filename
        ):
            raise ValueError(
                "standard_lora_loader_override must match the selected "
                "lora_name and model filename"
            )
        return self


class VaeModelBinding(StrictModel):
    filename: Annotated[str, Field(min_length=1, max_length=1024)]
    device: VaeDeviceTarget = "default"


class SettingsModels(StrictModel):
    fl2va: DiffusionModelBinding
    ref2va: DiffusionModelBinding
    clip: ModelBinding
    video_vae: VaeModelBinding
    audio_vae: VaeModelBinding


class ModelFileSelection(StrictModel):
    """One creative model-file choice; ``None`` is a saved incomplete state."""

    filename: Annotated[str, Field(min_length=1, max_length=1024)] | None


class DiffusionModelSelection(ModelFileSelection):
    """A diffusion-family creative selection, intentionally without placement."""


class ModelStack(StrictModel):
    """All project-owned model selections for a v5 creative document."""

    fl2va: DiffusionModelSelection
    ref2va: DiffusionModelSelection
    clip: ModelFileSelection
    video_vae: ModelFileSelection
    audio_vae: ModelFileSelection


class RuntimeSettings(StrictModel):
    # The ComfyUI address is not a setting: the plugin embeds Director in one
    # ComfyUI process and create_app injects that instance's loopback URL.
    client_id: Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")]
    # Pure Standard segment prompts repeat stable loader ids/inputs so ComfyUI
    # may retain its endpoint-local cache between prompts. RayLight uses an
    # explicit cleanup boundary and is not covered by this best-effort name.
    memory_policy: Literal["keep_resident"] = "keep_resident"
    # Ray worker CUDA allocations live outside ComfyUI's driver process.
    # ``keep_until_switch`` retains one exact runtime key and lets Director
    # replace that pool explicitly before an incompatible RayLight or Standard
    # unit runs. ``release_after_sampling`` remains available for endpoints
    # that should give the worker CUDA memory back after every sample.
    raylight_residency_policy: Literal[
        "release_after_sampling",
        "keep_until_switch",
    ] = "keep_until_switch"
    # Multi-GPU inference (RayLight) is opt-in. Enabling installs the optional
    # RayLight python dependencies and takes effect after a ComfyUI restart;
    # while disabled, a >=2 GPU pool fails closed instead of compiling RayLight
    # graphs. Capability (platform/deps/registered nodes) is always probed at
    # runtime, never persisted.
    multi_gpu_enabled: bool = False
    models: SettingsModels


# ``RuntimeSettings`` is the frozen pre-v5 contract used only by historical
# snapshots and the v4 compatibility compiler.  Give it an explicit name so
# no new code has to infer which settings generation it is parsing.
RuntimeSettingsV1 = RuntimeSettings


class RuntimeDiffusionPlacement(StrictModel):
    """Host placement for one diffusion family, with no creative filename."""

    device: DeviceTarget = "default"
    raylight: RayLightProfile = Field(default_factory=RayLightProfile)


class RuntimePlacementV2(StrictModel):
    """The complete runtime-only placement authority after the v5 migration."""

    fl2va: RuntimeDiffusionPlacement = Field(
        default_factory=RuntimeDiffusionPlacement
    )
    ref2va: RuntimeDiffusionPlacement = Field(
        default_factory=RuntimeDiffusionPlacement
    )
    clip_device: DeviceTarget = "default"
    video_vae_device: VaeDeviceTarget = "default"
    audio_vae_device: VaeDeviceTarget = "default"


class LegacyStandardLoraOverrideEvidence(StrictModel):
    """One exact v1 override retained temporarily as host compatibility data."""

    family: TimelineMode
    model_filename: Annotated[str, Field(min_length=1, max_length=1024)]
    lora_filename: Annotated[str, Field(min_length=1, max_length=1024)]
    loader: StandardLoraLoader


class LegacyLoraResolutionCompat(StrictModel):
    """Bounded Stage-6 bridge to the frozen Standard LoRA resolver."""

    schema_version: Literal[1] = 1
    auto_resolution_strategy_version: Literal[
        "v4-known-filename-or-safetensors-metadata-v1"
    ] = "v4-known-filename-or-safetensors-metadata-v1"
    explicit_overrides: Annotated[
        list[LegacyStandardLoraOverrideEvidence], Field(max_length=2)
    ] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_sorted_families(self) -> "LegacyLoraResolutionCompat":
        families = [record.family for record in self.explicit_overrides]
        if len(families) != len(set(families)):
            raise ValueError("legacy LoRA override families must be unique")
        if families != sorted(families):
            raise ValueError("legacy LoRA overrides must use stable family order")
        return self


class RuntimeSettingsV2(StrictModel):
    """Runtime/host authority after creative model choices move into projects."""

    schema_version: Literal[2]
    client_id: Annotated[
        str,
        Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$"),
    ]
    memory_policy: Literal["keep_resident"] = "keep_resident"
    raylight_residency_policy: Literal[
        "release_after_sampling", "keep_until_switch"
    ] = "keep_until_switch"
    multi_gpu_enabled: bool = False
    placement: RuntimePlacementV2 = Field(default_factory=RuntimePlacementV2)
    legacy_lora_resolution_compat: LegacyLoraResolutionCompat = Field(
        default_factory=LegacyLoraResolutionCompat
    )


class LoraLoaderOverrideRecord(StrictModel):
    """One exact LoRA-path mapping retained as user-owned runtime data.

    This model deliberately validates only the durable wire structure. Product
    support policy is mutable across DirectorDeck releases, so consulting the
    current product configuration here would make a retired mapping impossible
    to read, preserve, or delete. New and changed records are validated at the
    settings PUT boundary; active records are validated again when a Standard
    LoRA is resolved for one project/job.
    """

    lora_filename: Annotated[str, Field(min_length=1, max_length=1024)]
    adapter_id: Annotated[
        str,
        Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]{0,63}$"),
    ]
    options: Annotated[dict[str, bool], Field(max_length=16)] = Field(
        default_factory=dict
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_record(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        normalized.pop("family", None)
        normalized.pop("model_filename", None)
        adapter_id = normalized.get("adapter_id")
        if adapter_id == "dedicated":
            normalized["adapter_id"] = "minimax_h3_turbo"
        elif adapter_id == "bypass_model_only":
            normalized["adapter_id"] = "model_only"
        normalized.setdefault("options", {})
        return normalized

    def binding_tuple(self) -> tuple[str]:
        return (self.lora_filename,)


def lora_loader_binding_utf16_sort_key(
    record: LoraLoaderOverrideRecord,
) -> tuple[bytes]:
    """Match ECMAScript path ordering without case/path normalization."""

    return (utf16_sort_key(record.lora_filename),)


class RuntimeSettingsV3(StrictModel):
    """Runtime authority with one optional mapping per exact LoRA path."""

    schema_version: Literal[3]
    client_id: Annotated[
        str,
        Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$"),
    ]
    memory_policy: Literal["keep_resident"] = "keep_resident"
    raylight_residency_policy: Literal[
        "release_after_sampling", "keep_until_switch"
    ] = "keep_until_switch"
    multi_gpu_enabled: bool = False
    placement: RuntimePlacementV2 = Field(default_factory=RuntimePlacementV2)
    lora_loader_overrides: Annotated[
        list[LoraLoaderOverrideRecord], Field(max_length=256)
    ] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def migrate_exact_triple_mappings(cls, value: Any) -> Any:
        """Collapse the staged exact-triple shape into LoRA-only mappings.

        Existing dedicated mappings are more specific than the two historical
        model-only variants when several old triples collapse onto one LoRA.
        This migration is deterministic and is persisted canonically by the
        database startup normalizer.
        """

        if not isinstance(value, dict):
            return value
        records = value.get("lora_loader_overrides")
        if not isinstance(records, list) or not any(
            isinstance(record, dict)
            and ("family" in record or "model_filename" in record)
            for record in records
        ):
            return value
        collapsed: dict[str, dict[str, Any]] = {}
        for raw in records:
            if not isinstance(raw, dict):
                continue
            lora_filename = raw.get("lora_filename")
            if not isinstance(lora_filename, str):
                continue
            adapter_id = raw.get("adapter_id")
            if adapter_id == "dedicated":
                adapter_id = "minimax_h3_turbo"
            elif adapter_id == "bypass_model_only":
                adapter_id = "model_only"
            candidate = {
                "lora_filename": lora_filename,
                "adapter_id": adapter_id,
                "options": raw.get("options", {}),
            }
            previous = collapsed.get(lora_filename)
            if previous is None or (
                candidate["adapter_id"] == "minimax_h3_turbo"
                and previous.get("adapter_id") != "minimax_h3_turbo"
            ):
                collapsed[lora_filename] = candidate
        normalized = dict(value)
        normalized["lora_loader_overrides"] = list(collapsed.values())
        return normalized

    @field_validator("lora_loader_overrides")
    @classmethod
    def exact_unique_stable_overrides(
        cls, value: list[LoraLoaderOverrideRecord]
    ) -> list[LoraLoaderOverrideRecord]:
        seen: set[tuple[str]] = set()
        for record in value:
            binding = record.binding_tuple()
            if binding in seen:
                raise ValueError("LoRA loader override paths must be unique")
            seen.add(binding)
        return sorted(value, key=lora_loader_binding_utf16_sort_key)


class RuntimeSettingsMigrationNotice(StrictModel):
    """Typed, durable action required after removing legacy auto resolution."""

    schema_version: Literal[1] = 1
    id: Annotated[
        str,
        Field(min_length=1, max_length=128, pattern=r"^[a-z0-9._-]+$"),
    ]
    code: Literal["legacy_lora_resolution_review_required"]
    severity: Literal["warning"] = "warning"
    action: Literal["review_lora_loader_mappings"]
    legacy_strategy_version: Literal[
        "v4-known-filename-or-safetensors-metadata-v1"
    ]
    message: Annotated[str, Field(min_length=1, max_length=1024)]
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("migration notice timestamp must include a timezone")
        return value


class RuntimeSettingsMigrationNoticeListRead(StrictModel):
    notices: Annotated[list[RuntimeSettingsMigrationNotice], Field(max_length=256)]


class RuntimeSettingsAuthorityV2Read(StrictModel):
    settings: RuntimeSettingsV2
    authority_token: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class RuntimeSettingsAuthorityV1WriteRequest(StrictModel):
    """Frozen stale-writer shape retained only for a structured tombstone."""

    document: RuntimeSettingsV1
    expected_authority_token: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    schema_version: Literal[1]


class RuntimeSettingsAuthorityV2WriteRequest(StrictModel):
    document: RuntimeSettingsV2
    expected_authority_token: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    schema_version: Literal[2]


class RuntimeSettingsAuthorityV3Read(StrictModel):
    settings: RuntimeSettingsV3
    authority_token: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class RuntimeSettingsAuthorityV3WriteRequest(StrictModel):
    document: RuntimeSettingsV3
    expected_authority_token: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    schema_version: Literal[3]


class RuntimeSettingsAuthorityRead(StrictModel):
    settings: RuntimeSettings
    authority_token: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def canonicalize_live_runtime_settings(settings: RuntimeSettings) -> RuntimeSettings:
    """Remove obsolete user routing choices from the live settings document.

    ``backend`` and the legacy ``lora_loader`` remain accepted in the
    wire/schema shape so an older browser or database can upgrade without
    becoming unreadable. They are no longer authorities: the execution backend
    comes from the logical GPU pool, while Standard LoRA selection requires the
    user's exact visible ``standard_lora_loader_override`` mapping. Historical job
    snapshots are deliberately not passed through this helper, preserving their
    audit payload verbatim.
    """

    normalized: dict[str, DiffusionModelBinding] = {}
    for family in ("fl2va", "ref2va"):
        binding = getattr(settings.models, family)
        normalized[family] = binding.model_copy(
            update={
                "backend": "auto",
                "lora_loader": "auto",
                "lora_low_vram": False,
                "standard_lora_loader_override": (
                    binding.standard_lora_loader_override
                    if binding.lora_name is not None
                    and len(binding.raylight.gpu_select) == 1
                    else None
                ),
                # RayLight owns placement through its logical GPU pool.  A
                # stale Standard device must not survive a 1 -> N GPU edit and
                # become a second, contradictory placement authority.
                **(
                    {"device": "default"}
                    if len(binding.raylight.gpu_select) >= 2
                    else {}
                ),
            }
        )
    return settings.model_copy(
        update={"models": settings.models.model_copy(update=normalized)}
    )


def default_settings() -> RuntimeSettings:
    return RuntimeSettings.model_validate(
        {
            "client_id": "directordeck",
            "memory_policy": "keep_resident",
            "raylight_residency_policy": "keep_until_switch",
            "models": {
                "fl2va": {
                    "filename": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                    "device": "default",
                },
                "ref2va": {
                    "filename": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                    "device": "default",
                },
                "clip": {
                    "filename": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                    "device": "default",
                },
                "video_vae": {
                    "filename": "minimax_h3_video_vae_fp16.safetensors",
                    "device": "default",
                },
                "audio_vae": {
                    "filename": "minimax_h3_audio_vae_fp32.safetensors",
                    "device": "default",
                },
            },
        }
    )


def default_model_stack() -> ModelStack:
    """Creative model defaults used only for the fresh singleton project."""

    models = default_settings().models
    return ModelStack.model_validate(
        {
            role: {"filename": getattr(models, role).filename}
            for role in ("fl2va", "ref2va", "clip", "video_vae", "audio_vae")
        }
    )


def empty_model_stack() -> ModelStack:
    """Return the explicit five-slot incomplete creative editing state."""

    return ModelStack.model_validate(
        {
            role: {"filename": None}
            for role in ("fl2va", "ref2va", "clip", "video_vae", "audio_vae")
        }
    )


def default_runtime_settings_v2() -> RuntimeSettingsV2:
    """Return runtime-only defaults; no creative model file is smuggled in."""

    return RuntimeSettingsV2(
        schema_version=2,
        client_id="directordeck",
        memory_policy="keep_resident",
        raylight_residency_policy="keep_until_switch",
        multi_gpu_enabled=False,
        placement=RuntimePlacementV2(),
        legacy_lora_resolution_compat=LegacyLoraResolutionCompat(),
    )


def default_runtime_settings_v3() -> RuntimeSettingsV3:
    """Return current runtime defaults without inferred LoRA mappings."""

    return RuntimeSettingsV3(
        schema_version=3,
        client_id="directordeck",
        memory_policy="keep_resident",
        raylight_residency_policy="keep_until_switch",
        multi_gpu_enabled=False,
        placement=RuntimePlacementV2(),
        lora_loader_overrides=[],
    )


class VideoMetadata(StrictModel):
    """Server-probed video facts persisted with an uploaded asset."""

    duration: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    native_fps: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    frame_count: Annotated[int, Field(gt=0)]
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    probe_method: Annotated[str, Field(min_length=1, max_length=128)]
    # Old persisted uploads did not record stream presence. Treating the
    # missing fact as silent is the only fail-closed migration for optional
    # source-soundtrack conditioning.
    has_audio: bool = False


class AssetReference(StrictModel):
    """Stable ComfyUI input identity returned by the upload endpoint.

    The first four fields are the frontend contract.  The remaining fields are
    accepted when that same response object is embedded in a saved draft.
    """

    name: Annotated[str, Field(min_length=1, max_length=1024)]
    subfolder: Annotated[str, Field(max_length=1024)] = ""
    type: Literal["input", "output", "temp"] = "input"
    kind: AssetKind
    id: Annotated[str, Field(min_length=1, max_length=128)]
    filename: str | None = None
    path: str | None = None
    preview_url: str | None = None
    content_hash: Annotated[
        str | None, Field(pattern=r"^sha256:[0-9a-f]{64}$")
    ] = None
    metadata: VideoMetadata | None = None

    @model_validator(mode="after")
    def validate_kind_and_path(self) -> "AssetReference":
        if self.type != "input":
            raise ValueError("Director input assets must have type='input'")
        for component in (self.name, self.subfolder):
            normalized = component.replace("\\", "/")
            if normalized.startswith("/") or ".." in normalized.split("/"):
                raise ValueError("asset paths must stay inside the ComfyUI input directory")
        if self.kind == "video" and self.metadata is None:
            raise ValueError("video assets require server-probed metadata")
        if self.kind != "video" and self.metadata is not None:
            raise ValueError("metadata is allowed only for video assets")
        return self

    @property
    def comfy_path(self) -> str:
        return f"{self.subfolder.strip('/')}/{self.name}".strip("/")


class SlottedAssetReference(AssetReference):
    """An uploaded asset assigned to a stable, modality-local reference slot."""

    slot: Annotated[int, Field(ge=0)]


def _require_reference_capacity(
    value: Any, *, label: str, limit: int, unit: str
) -> Any:
    if isinstance(value, list) and len(value) > limit:
        raise ValueError(f"MiniMax H3 {label}最多 {limit} {unit}")
    return value


def _validate_reference_image_capacity(value: Any) -> Any:
    return _require_reference_capacity(
        value,
        label="参考图片",
        limit=H3_REFERENCE_LIMITS.reference_images,
        unit="张",
    )


def _validate_reference_video_capacity(value: Any) -> Any:
    return _require_reference_capacity(
        value,
        label="参考视频通道",
        limit=H3_REFERENCE_LIMITS.reference_video_channels,
        unit="路",
    )


def _validate_reference_audio_capacity(value: Any) -> Any:
    return _require_reference_capacity(
        value,
        label="独立参考音频",
        limit=H3_REFERENCE_LIMITS.standalone_reference_audios,
        unit="条",
    )


H3ReferenceImageList: TypeAlias = Annotated[
    list[SlottedAssetReference],
    BeforeValidator(_validate_reference_image_capacity),
    Field(max_length=H3_REFERENCE_LIMITS.reference_images),
]
H3ReferenceVideoList: TypeAlias = Annotated[
    list[SlottedAssetReference],
    BeforeValidator(_validate_reference_video_capacity),
    Field(max_length=H3_REFERENCE_LIMITS.reference_video_channels),
]
H3StandaloneReferenceAudioList: TypeAlias = Annotated[
    list[SlottedAssetReference],
    BeforeValidator(_validate_reference_audio_capacity),
    Field(max_length=H3_REFERENCE_LIMITS.standalone_reference_audios),
]


class RenderConfig(StrictModel):
    width: Annotated[int, Field(ge=32, le=8192, multiple_of=32)] = 864
    height: Annotated[int, Field(ge=32, le=8192, multiple_of=32)] = 480
    fps: Annotated[float, Field(ge=1, le=240)] = 24.0


class SamplingConfig(StrictModel):
    steps: Annotated[int, Field(ge=1, le=200)] = 25
    # Seeds are always concrete and JSON-safe. ``random_seed`` is an editor
    # intent: the browser rolls a new safe integer immediately before submit,
    # updates the visible disabled input, and submits that exact value. The
    # server never draws a second hidden seed, so compile reports, job
    # snapshots and ComfyUI prompts all agree.
    seed: Annotated[int, Field(ge=0, le=2**53 - 1)] = 0
    random_seed: bool = True
    sampler: Literal["res_multistep", "euler", "dpmpp_2m"] = "res_multistep"
    # Keep this as a deliberately small, audited subset of ComfyUI's
    # scheduler registry.  Both stock BasicScheduler and RayLight's
    # DirectorDeckRayBasicScheduler consumes these exact wire values.
    scheduler: SchedulerName = "simple"
    shift: Annotated[float, Field(ge=0.01, le=100)] = 12.0
    audio_shift: Annotated[float, Field(ge=0.01, le=100)] = 3.0

    @model_validator(mode="before")
    @classmethod
    def migrate_obsolete_cfg_and_seed_sentinel(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        # MiniMax H3 uses BasicGuider and has no CFG/negative branch. Accept
        # and discard this old wire field so persisted drafts and six-mode
        # clients upgrade without making the meaningless control survive in
        # normalized responses.
        migrated.pop("cfg", None)
        seed = migrated.get("seed")
        if seed == -1:
            migrated["seed"] = secrets.randbelow(2**53)
            migrated["random_seed"] = True
        elif "seed" in migrated and "random_seed" not in migrated:
            # A non-negative seed in the former schema explicitly meant fixed.
            migrated["random_seed"] = False
        return migrated


class TimelineSamplingConfig(StrictModel):
    """Independent sampler controls for the two MiniMax H3 model families."""

    fl2va: SamplingConfig = Field(default_factory=SamplingConfig)
    ref2va: SamplingConfig = Field(default_factory=SamplingConfig)


class ShotBase(StrictModel):
    id: Annotated[str, Field(min_length=1, max_length=128)]
    title: Annotated[str, Field(max_length=256)] = ""
    prompt: Annotated[str, Field(max_length=MINIMAX_H3_PROMPT_MAX_CHARACTERS)] = ""
    # The Director generation-group UI caps a segment near the 512-frame
    # execution boundary. At the maximum supported 240 fps, 2 seconds stays
    # within that bound; lower frame rates can still reach the common 5–15 s
    # H3 durations. The canonical compiler performs the final frame check.
    duration_seconds: Annotated[float, Field(gt=0, le=120)] = 5.0
    enabled: bool = True


class T2VShot(ShotBase):
    pass


class I2VShot(ShotBase):
    first_image: AssetReference | None = None

    @field_validator("first_image")
    @classmethod
    def image_only(cls, value: AssetReference | None) -> AssetReference | None:
        if value is not None and value.kind != "image":
            raise ValueError("first_image must reference an image")
        return value


class FL2VShot(ShotBase):
    duration_seconds: Annotated[float, Field(ge=0.1, le=120)] = 5.0
    first_image: AssetReference | None = None
    last_image: AssetReference | None = None

    @field_validator("first_image", "last_image")
    @classmethod
    def images_only(cls, value: AssetReference | None) -> AssetReference | None:
        if value is not None and value.kind != "image":
            raise ValueError("first_image and last_image must reference images")
        return value


class R2VShot(ShotBase):
    reference_images: H3ReferenceImageList = Field(default_factory=list)
    reference_audios: H3StandaloneReferenceAudioList = Field(default_factory=list)
    reference_videos: H3ReferenceVideoList = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_reference_kinds(self) -> "R2VShot":
        _require_slotted_assets(
            self.reference_images,
            "image",
            "reference_images",
            max_slot=H3_REFERENCE_LIMITS.reference_images - 1,
        )
        _require_slotted_assets(
            self.reference_audios,
            "audio",
            "reference_audios",
            max_slot=H3_REFERENCE_LIMITS.standalone_reference_audios - 1,
        )
        _require_slotted_assets(
            self.reference_videos,
            "video",
            "reference_videos",
            max_slot=H3_REFERENCE_LIMITS.reference_video_channels - 1,
        )
        return self


class V2VShot(ShotBase):
    source_video: AssetReference | None = None
    source_start_seconds: Annotated[float, Field(ge=0, le=86_400)] = 0.0
    source_duration_seconds: Annotated[float, Field(gt=0, le=86_400)] = 5.0

    @model_validator(mode="after")
    def validate_source_video(self) -> "V2VShot":
        if self.source_video is not None and self.source_video.kind != "video":
            raise ValueError("source_video must reference a video")
        if self.source_video is not None:
            assert self.source_video.metadata is not None
            source_end = self.source_start_seconds + self.source_duration_seconds
            if source_end > self.source_video.metadata.duration + 1e-6:
                raise ValueError(
                    "source_start_seconds + source_duration_seconds exceeds "
                    "source_video.metadata.duration"
                )
        return self


class RV2VShot(V2VShot):
    reference_images: H3ReferenceImageList = Field(default_factory=list)
    reference_audios: H3StandaloneReferenceAudioList = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_reference_kinds(self) -> "RV2VShot":
        _require_slotted_assets(
            self.reference_images,
            "image",
            "reference_images",
            max_slot=H3_REFERENCE_LIMITS.reference_images - 1,
        )
        _require_slotted_assets(
            self.reference_audios,
            "audio",
            "reference_audios",
            max_slot=H3_REFERENCE_LIMITS.standalone_reference_audios - 1,
        )
        return self


def _require_asset_kind(values: list[AssetReference], kind: AssetKind, field: str) -> None:
    if any(value.kind != kind for value in values):
        raise ValueError(f"{field} may contain only {kind} assets")


def _require_slotted_assets(
    values: list[SlottedAssetReference],
    kind: AssetKind,
    field: str,
    *,
    max_slot: int,
) -> None:
    _require_asset_kind(values, kind, field)
    slots = [value.slot for value in values]
    asset_ids = [value.id for value in values]
    invalid = [slot for slot in slots if slot > max_slot]
    if invalid:
        raise ValueError(f"{field} slots must be between 0 and {max_slot}")
    if len(set(slots)) != len(slots):
        raise ValueError(f"{field} slots must be unique")
    if len(set(asset_ids)) != len(asset_ids):
        raise ValueError(f"{field} asset ids must be unique")


class DraftBase(StrictModel):
    prompt: Annotated[str, Field(max_length=MINIMAX_H3_PROMPT_MAX_CHARACTERS)] = ""
    ref_image_size: Literal["match", "max"] = "match"
    render: RenderConfig = Field(default_factory=RenderConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)


class T2VDraft(DraftBase):
    mode: Literal["t2v"]
    shots: Annotated[list[T2VShot], Field(min_length=1, max_length=128)]


class I2VDraft(DraftBase):
    mode: Literal["i2v"]
    shots: Annotated[list[I2VShot], Field(min_length=1, max_length=128)]


class FL2VDraft(DraftBase):
    mode: Literal["fl2v"]
    shots: Annotated[list[FL2VShot], Field(min_length=1, max_length=128)]


class R2VDraft(DraftBase):
    mode: Literal["r2v"]
    shots: Annotated[list[R2VShot], Field(min_length=1, max_length=128)]


class V2VDraft(DraftBase):
    mode: Literal["v2v"]
    shots: Annotated[list[V2VShot], Field(min_length=1, max_length=128)]


class RV2VDraft(DraftBase):
    mode: Literal["rv2v"]
    shots: Annotated[list[RV2VShot], Field(min_length=1, max_length=128)]


# These six classes are retained only as the strict v1 migration boundary.
# They reject fields that never belonged to the declared legacy recipe before
# the document is normalized into the two editable v2 model-family shapes.
class UnifiedT2VSegment(T2VShot):
    mode: Literal["t2v"]


class UnifiedI2VSegment(I2VShot):
    mode: Literal["i2v"]


class UnifiedFL2VSegment(FL2VShot):
    mode: Literal["fl2v"]


class UnifiedR2VSegment(R2VShot):
    mode: Literal["r2v"]


class UnifiedV2VSegment(V2VShot):
    mode: Literal["v2v"]
    source_audio_as_reference: bool = False


class UnifiedRV2VSegment(RV2VShot):
    mode: Literal["rv2v"]
    source_audio_as_reference: bool = False


_LEGACY_TIMELINE_SEGMENT_MODELS: dict[
    GenerationMode, type[ShotBase]
] = {
    "t2v": UnifiedT2VSegment,
    "i2v": UnifiedI2VSegment,
    "fl2v": UnifiedFL2VSegment,
    "r2v": UnifiedR2VSegment,
    "v2v": UnifiedV2VSegment,
    "rv2v": UnifiedRV2VSegment,
}


class TimelineContinuity(StrictModel):
    """Incoming continuation boundary authored on one timeline segment."""

    enabled: bool = False
    overlap_frames: Literal[5, 22, 39, 56] = 22


class UnifiedFL2VASegment(ShotBase):
    """Editable FL2VA segment; anchors deterministically select its recipe."""

    mode: Literal["fl2va"]
    continuity: TimelineContinuity = Field(default_factory=TimelineContinuity)
    ref_image_size: Literal["match", "max"]
    audio_mode: Literal["generate", "source", "mute"]
    first_image: AssetReference | None = None
    last_image: AssetReference | None = None

    @field_validator("first_image", "last_image")
    @classmethod
    def anchors_are_images(
        cls, value: AssetReference | None
    ) -> AssetReference | None:
        if value is not None and value.kind != "image":
            raise ValueError("FL2VA anchors must reference images")
        return value


class UnifiedRef2VASegment(ShotBase):
    """Editable Ref2VA segment with explicit source and reference-video slots."""

    mode: Literal["ref2va"]
    continuity: TimelineContinuity = Field(default_factory=TimelineContinuity)
    ref_image_size: Literal["match", "max"]
    audio_mode: Literal["generate", "source", "mute"]
    # The editor first lays a source video onto the timeline at its complete
    # duration. Such a source segment is a valid saved editing state even when
    # it must still be split before the native H3 512-frame compile boundary.
    duration_seconds: Annotated[float, Field(gt=0, le=86_400)] = 5.0
    source_video: AssetReference | None = None
    source_start_seconds: Annotated[float, Field(ge=0, le=86_400)] = 0.0
    source_duration_seconds: Annotated[float, Field(gt=0, le=86_400)] = 5.0
    source_audio_as_reference: bool = False
    reference_images: H3ReferenceImageList = Field(default_factory=list)
    reference_audios: H3StandaloneReferenceAudioList = Field(default_factory=list)
    reference_videos: H3ReferenceVideoList = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_media_contract(self) -> "UnifiedRef2VASegment":
        _require_slotted_assets(
            self.reference_images,
            "image",
            "reference_images",
            max_slot=H3_REFERENCE_LIMITS.reference_images - 1,
        )
        _require_slotted_assets(
            self.reference_audios,
            "audio",
            "reference_audios",
            max_slot=H3_REFERENCE_LIMITS.standalone_reference_audios - 1,
        )
        # The stock node exposes one shared ref_videos pool. A source occupies
        # ref_video_0; independent references follow it densely.
        independent_video_capacity = (
            H3_REFERENCE_LIMITS.independent_reference_video_capacity(
                has_source_video=self.source_video is not None
            )
        )
        if len(self.reference_videos) > independent_video_capacity:
            raise ValueError(
                "MiniMax H3 包括源视频在内最多支持 "
                f"{H3_REFERENCE_LIMITS.reference_video_channels} 路参考视频；"
                f"源视频占用 {H3_REFERENCE_LIMITS.source_videos} 路后，"
                f"最多可再添加 {independent_video_capacity} 路独立参考视频"
            )
        _require_slotted_assets(
            self.reference_videos,
            "video",
            "reference_videos",
            max_slot=independent_video_capacity - 1,
        )
        if self.source_video is not None and self.source_video.kind != "video":
            raise ValueError("source_video must reference a video")
        if self.source_video is not None:
            assert self.source_video.metadata is not None
            source_end = self.source_start_seconds + self.source_duration_seconds
            if source_end > self.source_video.metadata.duration + 1e-6:
                raise ValueError(
                    "source_start_seconds + source_duration_seconds exceeds "
                    "source_video.metadata.duration"
                )
        if self.source_video is not None and any(
            asset.id == self.source_video.id for asset in self.reference_videos
        ):
            raise ValueError(
                "source_video cannot also occupy an independent reference video slot"
            )
        return self


UnifiedTimelineSegment: TypeAlias = UnifiedFL2VASegment | UnifiedRef2VASegment


def timeline_segment_recipe(segment: UnifiedTimelineSegment) -> GenerationMode:
    """Derive one native six-recipe template from family and typed media."""

    if isinstance(segment, UnifiedFL2VASegment):
        if segment.last_image is not None:
            return "fl2v"
        if segment.first_image is not None:
            return "i2v"
        return "t2v"
    has_independent_references = bool(
        segment.reference_images
        or segment.reference_audios
        or segment.reference_videos
    )
    if segment.source_video is not None:
        return "rv2v" if has_independent_references else "v2v"
    return "r2v"


class UnifiedTimelineDraft(StrictModel):
    """Canonical editable document for one long-video project.

    Rendering/export are project-wide. Sampling is family-wide because FL2VA and
    Ref2VA are distinct diffusion models and may need different schedules.
    Audio output and reference-image sizing follow each segment. ``mode`` selects only FL2VA or
    Ref2VA; the server derives the exact six-recipe native template from the
    typed asset shape at compile/submit time.
    """

    version: Literal[4] = 4
    title: Annotated[str, Field(min_length=1, max_length=256)] = "未命名长视频"
    render: RenderConfig = Field(default_factory=RenderConfig)
    sampling: TimelineSamplingConfig = Field(default_factory=TimelineSamplingConfig)
    export_mode: Literal["all", "segments"] = "all"
    segments: Annotated[
        list[
            Annotated[
                UnifiedFL2VASegment | UnifiedRef2VASegment,
                Field(discriminator="mode"),
            ]
        ],
        Field(min_length=1, max_length=128),
    ]

    @model_validator(mode="before")
    @classmethod
    def migrate_shared_prompt_sampling_and_continuity(cls, value: Any) -> Any:
        """Normalize strict v1-v3 timeline contracts into canonical v4.

        A shared prompt was only a fallback for an empty segment prompt, so
        materializing it into those segments preserves the exact effective
        text without retaining a second source of truth. The former single
        sampler is copied to both model families. Version 2's project-wide
        continuation switch is copied onto every segment. This preserves its
        behavior not only for the current enabled order, but also if a formerly
        disabled segment is enabled or the timeline is reordered later.

        Version 1 recipe objects and version 2 family objects retain separate,
        strict migration boundaries. In particular, a document claiming v2
        cannot smuggle in the v3 per-segment field and have it silently become
        meaningful.
        """

        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        raw_version = migrated.get("version", 1)
        legacy_document = raw_version is None or (
            isinstance(raw_version, int)
            and not isinstance(raw_version, bool)
            and raw_version == 1
        )
        v2_document = (
            isinstance(raw_version, int)
            and not isinstance(raw_version, bool)
            and raw_version == 2
        )
        v3_document = (
            isinstance(raw_version, int)
            and not isinstance(raw_version, bool)
            and raw_version == 3
        )
        if not legacy_document and not v2_document and not v3_document:
            return migrated
        legacy_ref_image_size = migrated.pop("ref_image_size", "match")
        legacy_audio_mode = migrated.pop("audio_mode", "generate")
        raw_continuity = migrated.pop("continuity", {})
        continuity = TimelineContinuity.model_validate(raw_continuity)
        raw_segments = migrated.get("segments")
        if legacy_document:
            shared_prompt = migrated.pop("prompt", "")
            if isinstance(shared_prompt, str) and shared_prompt:
                if isinstance(raw_segments, list):
                    segments_with_prompts: list[Any] = []
                    for raw_segment in raw_segments:
                        if isinstance(raw_segment, dict):
                            segment = dict(raw_segment)
                            if not str(segment.get("prompt") or "").strip():
                                segment["prompt"] = shared_prompt
                            segments_with_prompts.append(segment)
                        else:
                            segments_with_prompts.append(raw_segment)
                    raw_segments = segments_with_prompts
                    migrated["segments"] = raw_segments
            sampling = migrated.get("sampling")
            if isinstance(sampling, dict) and not (
                "fl2va" in sampling or "ref2va" in sampling
            ):
                shared_sampling = dict(sampling)
                # A legacy flat ``-1`` represented one shared random choice.
                # Draw it once before cloning so migration does not unexpectedly
                # give FL2VA and Ref2VA two unrelated values.
                if shared_sampling.get("seed") == -1:
                    shared_sampling["seed"] = secrets.randbelow(2**53)
                    shared_sampling["random_seed"] = True
                migrated["sampling"] = {
                    "fl2va": dict(shared_sampling),
                    "ref2va": dict(shared_sampling),
                }
        if isinstance(raw_segments, list):
            normalized_segments: list[Any] = []
            for raw_segment in raw_segments:
                if not isinstance(raw_segment, dict):
                    normalized_segments.append(raw_segment)
                    continue
                if v2_document:
                    if "continuity" in raw_segment:
                        raise ValueError(
                            "timeline version 2 stores continuity only at the project level"
                        )
                    segment = dict(raw_segment)
                    segment["continuity"] = continuity.model_dump(mode="json")
                    if "ref_image_size" in segment or "audio_mode" in segment:
                        raise ValueError(
                            "timeline version 2 stores media policies only at the project level"
                        )
                    segment["ref_image_size"] = legacy_ref_image_size
                    segment["audio_mode"] = legacy_audio_mode
                    normalized_segments.append(segment)
                    continue
                if v3_document:
                    if "ref_image_size" in raw_segment or "audio_mode" in raw_segment:
                        raise ValueError(
                            "timeline version 3 stores media policies only at the project level"
                        )
                    segment = dict(raw_segment)
                    segment["ref_image_size"] = legacy_ref_image_size
                    segment["audio_mode"] = legacy_audio_mode
                    normalized_segments.append(segment)
                    continue
                raw_mode = raw_segment.get("mode")
                if isinstance(raw_mode, str) and raw_mode in {"fl2va", "ref2va"}:
                    raise ValueError(
                        "timeline version 1 accepts only the six legacy recipe modes"
                    )
                if (
                    not isinstance(raw_mode, str)
                    or raw_mode not in _LEGACY_TIMELINE_SEGMENT_MODELS
                ):
                    normalized_segments.append(raw_segment)
                    continue
                # Validate the exact v1 recipe before widening it into a v2
                # family shape; illegal hidden fields never become meaningful
                # merely because the new editor is more expressive.
                legacy = _LEGACY_TIMELINE_SEGMENT_MODELS[raw_mode].model_validate(
                    raw_segment
                )
                segment = legacy.model_dump(mode="json")
                segment["mode"] = MODEL_ROUTE[raw_mode]
                if raw_mode in {"t2v", "i2v", "fl2v"}:
                    segment.setdefault("first_image", None)
                    segment.setdefault("last_image", None)
                else:
                    segment.setdefault("source_video", None)
                    segment.setdefault("source_start_seconds", 0.0)
                    segment.setdefault("source_duration_seconds", 5.0)
                    segment.setdefault("source_audio_as_reference", False)
                    segment.setdefault("reference_images", [])
                    segment.setdefault("reference_audios", [])
                    segment.setdefault("reference_videos", [])
                segment["continuity"] = continuity.model_dump(mode="json")
                segment["ref_image_size"] = legacy_ref_image_size
                segment["audio_mode"] = legacy_audio_mode
                normalized_segments.append(segment)
            migrated["segments"] = normalized_segments
        migrated["version"] = 4
        return migrated

    @model_validator(mode="after")
    def unique_segment_ids(self) -> "UnifiedTimelineDraft":
        ids = [segment.id for segment in self.segments]
        duplicates = sorted({segment_id for segment_id in ids if ids.count(segment_id) > 1})
        if duplicates:
            raise ValueError(f"segment ids must be unique; duplicates: {', '.join(duplicates)}")
        return self


# Frozen historical/compatibility name. ``UnifiedTimelineDraft`` remains an
# alias for this model during the Stage-6 integration window so the v4 compiler
# and immutable fixture builders cannot accidentally start accepting v5.
UnifiedTimelineDraftV4 = UnifiedTimelineDraft


FeatureIdentifier: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=64, pattern=r"^[A-Za-z][A-Za-z0-9._-]{0,63}$"),
]


def _validate_feature_json_value(value: Any, *, depth: int, path: str) -> None:
    if depth > MAX_FEATURE_PARAMETER_DEPTH:
        raise ValueError(
            f"feature params exceed maximum nesting depth at {path}"
        )
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        if len(value) > MAX_FEATURE_STRING_LENGTH:
            raise ValueError(f"feature string is too long at {path}")
        return
    if isinstance(value, int):
        if abs(value) > 2**53 - 1:
            raise ValueError(f"feature integer is outside JSON-safe range at {path}")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"feature number must be finite at {path}")
        return
    if isinstance(value, list):
        if len(value) > MAX_FEATURE_ARRAY_ITEMS:
            raise ValueError(f"feature array is too large at {path}")
        for index, item in enumerate(value):
            _validate_feature_json_value(
                item,
                depth=depth + 1,
                path=f"{path}[{index}]",
            )
        return
    if isinstance(value, dict):
        if len(value) > MAX_FEATURE_OBJECT_KEYS:
            raise ValueError(f"feature object has too many keys at {path}")
        for key, item in value.items():
            if not isinstance(key, str) or _FEATURE_ID_PATTERN.fullmatch(key) is None:
                raise ValueError(f"feature object has an invalid key at {path}")
            _validate_feature_json_value(
                item,
                depth=depth + 1,
                path=f"{path}.{key}",
            )
        return
    raise ValueError(f"feature params contain a non-JSON value at {path}")


class FeatureSelection(StrictModel):
    enabled: bool = False
    params: dict[FeatureIdentifier, Any] = Field(default_factory=dict)

    @field_validator("params")
    @classmethod
    def bounded_json_params(
        cls, value: dict[FeatureIdentifier, Any]
    ) -> dict[FeatureIdentifier, Any]:
        _validate_feature_json_value(value, depth=0, path="params")
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(encoded) > MAX_FEATURE_PARAMS_BYTES:
            raise ValueError("feature params exceed the encoded byte limit")
        return value


class FeatureConfiguration(StrictModel):
    template_bundle_version: Annotated[int, Field(ge=1, le=2_147_483_647)]
    project: Annotated[
        dict[FeatureIdentifier, FeatureSelection],
        Field(max_length=MAX_FEATURES_PER_SCOPE),
    ] = Field(default_factory=dict)
    by_segment: Annotated[
        dict[
            Annotated[str, Field(min_length=1, max_length=128)],
            Annotated[
                dict[FeatureIdentifier, FeatureSelection],
                Field(max_length=MAX_FEATURES_PER_SCOPE),
            ],
        ],
        Field(max_length=128),
    ] = Field(default_factory=dict)


class LoraFamilyFeatureSelection(StrictModel):
    enabled: bool = False
    filename: Annotated[str, Field(min_length=1, max_length=1024)] | None = None
    strength: Annotated[
        float, Field(ge=-10, le=10, allow_inf_nan=False)
    ] = 1.0


class LoraFeatureParams(StrictModel):
    by_family: dict[TimelineMode, LoraFamilyFeatureSelection]

    @field_validator("by_family")
    @classmethod
    def exact_families(
        cls, value: dict[TimelineMode, LoraFamilyFeatureSelection]
    ) -> dict[TimelineMode, LoraFamilyFeatureSelection]:
        if set(value) != {"fl2va", "ref2va"}:
            raise ValueError("LoRA feature params require both model families")
        return value


class UnifiedTimelineDraftV5(StrictModel):
    """The sole creative authority for all newly saved and submitted projects."""

    version: Literal[5]
    title: Annotated[str, Field(min_length=1, max_length=256)] = "未命名长视频"
    render: RenderConfig = Field(default_factory=RenderConfig)
    sampling: TimelineSamplingConfig = Field(default_factory=TimelineSamplingConfig)
    export_mode: Literal["all", "segments"] = "all"
    model_stack: ModelStack
    features: FeatureConfiguration
    segments: Annotated[
        list[
            Annotated[
                UnifiedFL2VASegment | UnifiedRef2VASegment,
                Field(discriminator="mode"),
            ]
        ],
        Field(min_length=1, max_length=128),
    ]

    @model_validator(mode="after")
    def validate_segment_feature_identity(self) -> "UnifiedTimelineDraftV5":
        ids = [segment.id for segment in self.segments]
        duplicates = sorted(
            {segment_id for segment_id in ids if ids.count(segment_id) > 1}
        )
        if duplicates:
            raise ValueError(
                "segment ids must be unique; duplicates: " + ", ".join(duplicates)
            )
        unknown_override_ids = sorted(set(self.features.by_segment) - set(ids))
        if unknown_override_ids:
            raise ValueError(
                "feature overrides reference unknown segments: "
                + ", ".join(unknown_override_ids)
            )
        lora = self.features.project.get("lora")
        if lora is not None:
            LoraFeatureParams.model_validate(lora.params)
        for segment_id, selections in self.features.by_segment.items():
            segment_lora = selections.get("lora")
            if segment_lora is not None:
                try:
                    LoraFeatureParams.model_validate(segment_lora.params)
                except ValidationError as exc:
                    raise ValueError(
                        f"segment {segment_id} has invalid LoRA feature params"
                    ) from exc
        return self


ModeDraft: TypeAlias = T2VDraft | I2VDraft | FL2VDraft | R2VDraft | V2VDraft | RV2VDraft
ModeDraftAdapter = TypeAdapter(Annotated[ModeDraft, Field(discriminator="mode")])
MODE_MODELS: dict[GenerationMode, type[DraftBase]] = {
    "t2v": T2VDraft,
    "i2v": I2VDraft,
    "fl2v": FL2VDraft,
    "r2v": R2VDraft,
    "v2v": V2VDraft,
    "rv2v": RV2VDraft,
}


def iter_draft_assets(draft: ModeDraft) -> Iterator[tuple[str, AssetReference]]:
    """Yield every media reference with a stable, user-facing field location.

    Disabled shots are intentionally included: persisted drafts must never
    contain untrusted ComfyUI paths that could become active after a later
    toggle-only edit.
    """

    for index, shot in enumerate(draft.shots):
        prefix = f"shots[{index}]({shot.id})"
        if isinstance(draft, I2VDraft):
            if shot.first_image is not None:
                yield f"{prefix}.first_image", shot.first_image
        elif isinstance(draft, FL2VDraft):
            if shot.first_image is not None:
                yield f"{prefix}.first_image", shot.first_image
            if shot.last_image is not None:
                yield f"{prefix}.last_image", shot.last_image
        elif isinstance(draft, R2VDraft):
            for asset_index, asset in enumerate(shot.reference_images):
                yield f"{prefix}.reference_images[{asset_index}]", asset
            for asset_index, asset in enumerate(shot.reference_audios):
                yield f"{prefix}.reference_audios[{asset_index}]", asset
            for asset_index, asset in enumerate(shot.reference_videos):
                yield f"{prefix}.reference_videos[{asset_index}]", asset
        elif isinstance(draft, RV2VDraft):
            if shot.source_video is not None:
                yield f"{prefix}.source_video", shot.source_video
            for asset_index, asset in enumerate(shot.reference_images):
                yield f"{prefix}.reference_images[{asset_index}]", asset
            for asset_index, asset in enumerate(shot.reference_audios):
                yield f"{prefix}.reference_audios[{asset_index}]", asset
        elif isinstance(draft, V2VDraft) and shot.source_video is not None:
            yield f"{prefix}.source_video", shot.source_video


def iter_timeline_assets(
    draft: UnifiedTimelineDraft | UnifiedTimelineDraftV5,
    *,
    segment_ids: set[str] | None = None,
) -> Iterator[tuple[str, AssetReference]]:
    """Yield all media in the unified timeline, including disabled segments."""

    for index, segment in enumerate(draft.segments):
        if segment_ids is not None and segment.id not in segment_ids:
            continue
        prefix = f"segments[{index}]({segment.id})"
        if isinstance(segment, UnifiedFL2VASegment):
            if segment.first_image is not None:
                yield f"{prefix}.first_image", segment.first_image
            if segment.last_image is not None:
                yield f"{prefix}.last_image", segment.last_image
        elif isinstance(segment, UnifiedRef2VASegment):
            if segment.source_video is not None:
                yield f"{prefix}.source_video", segment.source_video
            for asset_index, asset in enumerate(segment.reference_images):
                yield f"{prefix}.reference_images[{asset_index}]", asset
            for asset_index, asset in enumerate(segment.reference_audios):
                yield f"{prefix}.reference_audios[{asset_index}]", asset
            for asset_index, asset in enumerate(segment.reference_videos):
                yield f"{prefix}.reference_videos[{asset_index}]", asset


def validate_mode_draft(mode: GenerationMode, value: Any) -> ModeDraft:
    model = MODE_MODELS[mode]
    draft = model.model_validate(value)
    if draft.mode != mode:
        raise ValueError(f"path mode '{mode}' does not match body mode '{draft.mode}'")
    return draft  # type: ignore[return-value]


def default_draft(mode: GenerationMode) -> ModeDraft:
    shot: dict[str, Any] = {
        "id": f"{mode}-shot-1",
        "title": "镜头 01",
        "prompt": "",
        "duration_seconds": 5.0,
        "enabled": True,
    }
    if mode == "i2v":
        shot["first_image"] = None
    elif mode == "fl2v":
        shot.update(first_image=None, last_image=None)
    elif mode == "r2v":
        shot.update(reference_images=[], reference_audios=[], reference_videos=[])
    elif mode == "v2v":
        shot.update(source_video=None, source_start_seconds=0.0, source_duration_seconds=5.0)
    elif mode == "rv2v":
        shot.update(
            source_video=None,
            source_start_seconds=0.0,
            source_duration_seconds=5.0,
            reference_images=[],
            reference_audios=[],
        )
    return validate_mode_draft(
        mode,
        {
            "mode": mode,
            "prompt": "",
            "ref_image_size": "match",
            "render": {"width": 864, "height": 480, "fps": 24.0},
            "sampling": {
                "steps": 25,
                "seed": 0,
                "random_seed": True,
                "sampler": "res_multistep",
                "scheduler": "simple",
                "shift": 12.0,
                "audio_shift": 3.0,
            },
            "shots": [shot],
        },
    )


def validate_timeline_draft(value: Any) -> UnifiedTimelineDraft:
    return UnifiedTimelineDraft.model_validate(value)


def validate_timeline_draft_v4(value: Any) -> UnifiedTimelineDraftV4:
    """Parse only the frozen v1-v4 migration domain."""

    return UnifiedTimelineDraftV4.model_validate(value)


def validate_timeline_draft_v5(value: Any) -> UnifiedTimelineDraftV5:
    """Parse only the current v5 creative authority; never upgrades v4."""

    return UnifiedTimelineDraftV5.model_validate(value)


def validate_timeline_snapshot(
    value: Any,
) -> UnifiedTimelineDraftV4 | UnifiedTimelineDraftV5:
    """Explicitly dispatch immutable job snapshots without latest-schema guessing."""

    if not isinstance(value, dict):
        raise TypeError("timeline snapshot must be an object")
    if value.get("version") == 5:
        return validate_timeline_draft_v5(value)
    return validate_timeline_draft_v4(value)


def default_timeline_draft() -> UnifiedTimelineDraft:
    return UnifiedTimelineDraft.model_validate(
        {
            "version": 4,
            "title": "未命名长视频",
            "render": {"width": 864, "height": 480, "fps": 24.0},
            "sampling": {
                "fl2va": {
                    "steps": 25,
                    "seed": 0,
                    "random_seed": True,
                    "sampler": "res_multistep",
                    "scheduler": "simple",
                    "shift": 12.0,
                    "audio_shift": 3.0,
                },
                "ref2va": {
                    "steps": 25,
                    "seed": 0,
                    "random_seed": True,
                    "sampler": "res_multistep",
                    "scheduler": "simple",
                    "shift": 12.0,
                    "audio_shift": 3.0,
                },
            },
            "export_mode": "all",
            "segments": [
                {
                    "id": "timeline-segment-1",
                    "title": "片段 01",
                    "mode": "fl2va",
                    "prompt": "",
                    "duration_seconds": 5.0,
                    "enabled": True,
                    "ref_image_size": "match",
                    "audio_mode": "generate",
                }
            ],
        }
    )


def default_timeline_draft_v5(
    initial_model_stack: ModelStack | None = None,
) -> UnifiedTimelineDraftV5:
    """Create the frozen Bundle-5 project default for compatibility paths."""

    legacy = default_timeline_draft()
    return UnifiedTimelineDraftV5.model_validate(
        {
            **legacy.model_dump(mode="json", exclude={"version"}),
            "version": 5,
            "model_stack": (
                initial_model_stack.model_dump(mode="json")
                if initial_model_stack is not None
                else empty_model_stack().model_dump(mode="json")
            ),
            "features": {
                "template_bundle_version": 5,
                "project": {
                    "lora": {
                        "enabled": False,
                        "params": {
                            "by_family": {
                                "fl2va": {
                                    "enabled": False,
                                    "filename": None,
                                    "strength": 1.0,
                                },
                                "ref2va": {
                                    "enabled": False,
                                    "filename": None,
                                    "strength": 1.0,
                                },
                            }
                        },
                    }
                },
                "by_segment": {},
            },
        }
    )


def default_timeline_draft_v6(
    initial_model_stack: ModelStack | None = None,
) -> UnifiedTimelineDraftV5:
    """Create a timeline-schema-5 project using the Bundle-6 feature default."""

    legacy = default_timeline_draft_v5(initial_model_stack)
    return legacy.model_copy(
        update={
            "features": FeatureConfiguration(
                template_bundle_version=6,
                project={
                    "lora": legacy.features.project["lora"].model_copy(deep=True),
                    "comfy_kitchen_attention": FeatureSelection(
                        enabled=False,
                        params={},
                    ),
                },
                by_segment={},
            )
        },
        deep=True,
    )


def migrate_mode_drafts_to_timeline(
    drafts: list[ModeDraft],
) -> UnifiedTimelineDraft:
    """Import one unambiguous customized legacy draft into the new timeline.

    Legacy shared prompts are materialized into each segment.  Multiple
    customized documents have no cross-document order and may have conflicting
    render/sampler settings, so callers keep them in the legacy table and
    expose a migration notice rather than inventing a destructive merge.
    """

    customized = [
        draft
        for draft in drafts
        if not mode_draft_is_default(draft)
    ]
    # Separate legacy mode documents do not imply an ordering between them.
    # Importing exactly one customized document is deterministic; concatenating
    # two or more would silently invent a story order and shared sampler.
    if len(customized) != 1:
        return default_timeline_draft()
    return mode_draft_to_timeline(customized[0], title="从旧版六模式迁移")


def mode_draft_is_default(draft: ModeDraft) -> bool:
    """Compare defaults without treating a visible random seed as creative data."""

    current = draft.model_dump(mode="json")
    baseline = default_draft(draft.mode).model_dump(mode="json")
    for document in (current, baseline):
        sampling = document.get("sampling")
        if isinstance(sampling, dict) and sampling.get("random_seed") is True:
            sampling["seed"] = 0
    return current == baseline


def mode_draft_to_timeline(
    source: ModeDraft,
    *,
    title: str = "旧版模式任务",
) -> UnifiedTimelineDraft:
    """Losslessly wrap one legacy mode document in the native timeline model.

    This helper is also the compatibility boundary for the legacy ``/api/jobs``
    request.  Old clients may keep sending one mode document, but execution is
    compiled by the same server-owned native segment graph as the new timeline
    API; it never falls back to the former Director custom node.
    """

    segments: list[dict[str, Any]] = []
    for shot in source.shots:
        segment = shot.model_dump(mode="json")
        segment["mode"] = source.mode
        segment["prompt"] = shot.prompt or source.prompt
        segments.append(segment)
    return UnifiedTimelineDraft.model_validate(
        {
            # Route this genuine six-mode compatibility source through the
            # same strict v1 migration boundary as persisted old timelines.
            "version": 1,
            "title": title,
            "ref_image_size": source.ref_image_size,
            "render": source.render.model_dump(mode="json"),
            "sampling": {
                "fl2va": source.sampling.model_dump(mode="json"),
                "ref2va": source.sampling.model_dump(mode="json"),
            },
            "continuity": {"enabled": False, "overlap_frames": 22},
            "audio_mode": "generate",
            "export_mode": "all",
            "segments": segments,
        }
    )


class DetectShotsRequest(StrictModel):
    """Safe web contract for RV2V source-video shot detection."""

    asset_id: Annotated[str, Field(min_length=1, max_length=128)]
    frame_rate: Annotated[float, Field(ge=1, le=240, allow_inf_nan=False)]
    sensitivity: Literal["low", "medium", "high"]
    min_shot_frames: Annotated[int, Field(ge=4, le=100_000)]


class DetectShotsResponse(StrictModel):
    cut_frames: list[Annotated[int, Field(ge=0)]]
    shot_count: Annotated[int, Field(ge=0)]
    warnings: list[str] = Field(default_factory=list)


class CreateJobRequest(StrictModel):
    mode: GenerationMode
    config: dict[str, Any] | None = None


class TimelineJobRequest(StrictModel):
    # Stage 6 accepts only an explicit v5 creative snapshot for every new job.
    # Loading mutable project/settings state after request validation would
    # reintroduce the split authority this migration removes.
    config: UnifiedTimelineDraftV5
    segment_ids: Annotated[list[Annotated[str, Field(min_length=1, max_length=128)]], Field(min_length=1, max_length=128)] | None = None

    @field_validator("segment_ids")
    @classmethod
    def unique_segment_selection(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("segment_ids must be unique")
        return value


class FeaturePreflightRequest(StrictModel):
    """Read-only preflight input for the current v5 creative authority."""

    config: UnifiedTimelineDraftV5 | None = None
    segment_ids: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=128)]],
        Field(min_length=1, max_length=128),
    ] | None = None
    project_id: Annotated[str, Field(min_length=1, max_length=128)] | None = None

    @field_validator("segment_ids")
    @classmethod
    def unique_segment_selection(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("segment_ids must be unique")
        return value


class TimelineSegmentPlanRead(StrictModel):
    segment_id: str
    mode: TimelineMode
    recipe: GenerationMode
    model_family: TimelineMode
    backend: Literal["standard", "raylight"]
    frame_count: Annotated[int, Field(ge=5)]
    # ``frame_count`` remains the public visible-take length. Continuity can
    # sample a longer internal AV latent, then remove the predecessor prefix
    # and H3 alignment tail before SaveVideo.
    visible_frame_count: Annotated[int, Field(ge=5)]
    sample_frame_count: Annotated[int, Field(ge=5, le=512)]
    continuity_context_frames: Literal[0, 5, 22, 39, 56]
    alignment_tail_frame_count: Annotated[int, Field(ge=0, le=16)]
    predecessor_segment_id: str | None = None
    continuity_source: Literal["same_run", "historical_take"] | None = None
    historical_take_id: str | None = None
    anchor_reset: bool
    seed_mode: Literal["fixed", "random"]
    seed: Annotated[int, Field(ge=0, le=2**53 - 1)]
    conditioning_node: Literal[
        "MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo"
    ]
    node_classes: list[str]

    @model_validator(mode="after")
    def validate_continuity_source(self) -> "TimelineSegmentPlanRead":
        if self.predecessor_segment_id is None:
            if self.continuity_source is not None or self.historical_take_id is not None:
                raise ValueError(
                    "continuity source and historical take require a predecessor"
                )
            return self
        if self.continuity_source is None:
            raise ValueError("a predecessor requires an explicit continuity source")
        if self.continuity_source == "same_run":
            if self.historical_take_id is not None:
                raise ValueError("same-run continuity cannot name a historical take")
        elif not self.historical_take_id:
            raise ValueError("historical continuity requires a take id")
        return self


class TimelineNodePolicyRead(StrictModel):
    graph_source: Literal["server"] = "server"
    accepts_client_workflow: Literal[False] = False
    allowed_nodes: list[str]
    custom_nodes: list[str] = Field(default_factory=list)
    provenance: dict[
        str,
        Literal[
            "comfy-core",
            "comfy-core-official-minimax-h3",
            "comfy-extras",
            "raylight",
            "lora-custom",
        ],
    ]


class TimelineCompileDigestRead(StrictModel):
    algorithm: Literal["sha256-canonical-json-v1"]
    value: Annotated[str, Field(pattern=r"^sha256-[0-9a-f]{64}$")]


class TimelineCompileFeatureResolutionRead(StrictModel):
    segment_id: Annotated[str, Field(min_length=1, max_length=128)]
    unit_id: Annotated[str, Field(min_length=1, max_length=256)]
    feature_id: Annotated[str, Field(min_length=1, max_length=128)]
    version: Annotated[int, Field(ge=1, le=2_147_483_647)]
    backend: Literal["standard", "raylight"]
    family: TimelineMode
    template_id: Annotated[str, Field(min_length=1, max_length=128)]
    resolution: FeatureResolution
    adapter_fingerprint: Annotated[
        str, Field(pattern=r"^sha256:[0-9a-f]{64}$")
    ]
    capability: CompiledCapabilityEvaluation


class TimelineCompileEffectiveFeatureRead(StrictModel):
    id: Annotated[str, Field(min_length=1, max_length=128)]
    version: Annotated[int, Field(ge=1, le=2_147_483_647)]
    state: Literal["active", "noop"]
    adapter_fingerprint: Annotated[
        str, Field(pattern=r"^sha256:[0-9a-f]{64}$")
    ]
    capability: CompiledCapabilityEvaluation


class TimelineCompileEffectiveSegmentRead(StrictModel):
    unit_id: Annotated[str, Field(min_length=1, max_length=256)]
    backend: Literal["standard", "raylight"]
    family: TimelineMode
    template_id: Annotated[str, Field(min_length=1, max_length=128)]
    features: Annotated[
        list[TimelineCompileEffectiveFeatureRead],
        Field(min_length=1, max_length=64),
    ]

    @model_validator(mode="after")
    def validate_feature_identities(self) -> "TimelineCompileEffectiveSegmentRead":
        identities = [(item.id, item.version) for item in self.features]
        if len(identities) != len(set(identities)):
            raise ValueError("effective compile features must be unique")
        return self


class TimelineCompileFeaturesRead(StrictModel):
    requested: FeatureConfiguration
    effective_by_segment: Annotated[
        dict[
            Annotated[str, Field(min_length=1, max_length=128)],
            TimelineCompileEffectiveSegmentRead,
        ],
        Field(max_length=128),
    ] = Field(default_factory=dict)
    resolutions: Annotated[
        list[TimelineCompileFeatureResolutionRead],
        Field(max_length=8_192),
    ] = Field(default_factory=list)
    uses: Annotated[list[CompiledFeatureUseV3], Field(max_length=8_192)] = Field(
        default_factory=list
    )
    notices: Annotated[
        list[CompiledFeatureNotice],
        Field(max_length=256),
    ] = Field(default_factory=list)
    advisories: Annotated[
        list[CompiledCapabilityReason],
        Field(max_length=256),
    ] = Field(default_factory=list)


class TimelineCompileRead(StrictModel):
    template_bundle_version: Annotated[int, Field(ge=1, le=2_147_483_647)]
    host_capability_revision: Annotated[
        str, Field(pattern=r"^sha256:[0-9a-f]{64}$")
    ]
    execution_strategy: Literal["native_segment_graph_v1"] = "native_segment_graph_v1"
    model_families: list[Literal["fl2va", "ref2va"]]
    plans: list[TimelineSegmentPlanRead]
    node_policy: TimelineNodePolicyRead
    features: TimelineCompileFeaturesRead
    effective_execution_digest: TimelineCompileDigestRead

    @model_validator(mode="after")
    def validate_compile_report_authorities(self) -> "TimelineCompileRead":
        if self.features.requested.template_bundle_version != self.template_bundle_version:
            raise ValueError("requested feature bundle differs from compiled bundle")
        plan_ids = [plan.segment_id for plan in self.plans]
        if len(plan_ids) != len(set(plan_ids)):
            raise ValueError("compile report plan segment ids must be unique")
        plans_by_segment = {plan.segment_id: plan for plan in self.plans}
        if self.template_bundle_version == 6:
            if self.features.effective_by_segment or self.features.resolutions:
                raise ValueError("Bundle 6 cannot carry legacy feature resolutions")
            uses_by_segment = {segment_id: [] for segment_id in plan_ids}
            for use in self.features.uses:
                plan = plans_by_segment.get(use.segment_id)
                if plan is None:
                    raise ValueError("feature use references an uncompiled segment")
                if use.backend != plan.backend or use.family != plan.model_family:
                    raise ValueError("feature use route differs from its plan")
                uses_by_segment[use.segment_id].append(use)
            if any(not uses for uses in uses_by_segment.values()):
                raise ValueError("compiled segment has no Bundle-6 feature uses")
            use_notice_keys = {
                (item.segment_id, item.unit_id, item.feature_id)
                for item in self.features.uses
            }
            if any(
                (notice.segment_id, notice.unit_id, notice.feature_id)
                not in use_notice_keys
                for notice in self.features.notices
            ):
                raise ValueError("compile notice has no matching feature use")
        else:
            if self.features.uses or self.features.advisories:
                raise ValueError("legacy compile report cannot carry Bundle-6 evidence")
            self._validate_legacy_compile_features(plans_by_segment)
        expected_families = [
            family
            for family in ("fl2va", "ref2va")
            if any(plan.model_family == family for plan in self.plans)
        ]
        if self.model_families != expected_families:
            raise ValueError("compile report family summary differs from plans")
        return self

    def _validate_legacy_compile_features(
        self,
        plans_by_segment: dict[str, TimelineSegmentPlanRead],
    ) -> None:
        plan_ids = list(plans_by_segment)
        if set(self.features.effective_by_segment) != set(plan_ids):
            raise ValueError("effective feature scopes must exactly cover compiled plans")
        resolutions_by_segment: dict[
            str, list[TimelineCompileFeatureResolutionRead]
        ] = {segment_id: [] for segment_id in plan_ids}
        for item in self.features.resolutions:
            if item.segment_id not in plans_by_segment:
                raise ValueError("feature resolution references an uncompiled segment")
            resolutions_by_segment[item.segment_id].append(item)
        for segment_id, effective in self.features.effective_by_segment.items():
            plan = plans_by_segment[segment_id]
            if effective.backend != plan.backend or effective.family != plan.model_family:
                raise ValueError("effective feature route differs from its plan")
            actual = resolutions_by_segment[segment_id]
            if not actual:
                raise ValueError("compiled segment has no feature resolutions")
            if any(
                item.unit_id != effective.unit_id
                or item.backend != effective.backend
                or item.family != effective.family
                or item.template_id != effective.template_id
                for item in actual
            ):
                raise ValueError("feature resolution route differs from effective scope")
            expected_features = [
                (
                    item.id,
                    item.version,
                    item.state,
                    item.adapter_fingerprint,
                    item.capability,
                )
                for item in effective.features
            ]
            actual_features = [
                (
                    item.feature_id,
                    item.version,
                    item.resolution.state,
                    item.adapter_fingerprint,
                    item.capability,
                )
                for item in actual
            ]
            if expected_features != actual_features:
                raise ValueError("effective feature summary differs from resolutions")
        resolution_notice_keys = {
            (item.segment_id, item.unit_id, item.feature_id)
            for item in self.features.resolutions
        }
        if any(
            (notice.segment_id, notice.unit_id, notice.feature_id)
            not in resolution_notice_keys
            for notice in self.features.notices
        ):
            raise ValueError("compile notice has no matching feature resolution")


class AssetListRead(StrictModel):
    assets: list[AssetReference]
    outputs_preserved: Literal[True] = True


class AssetDeleteRead(StrictModel):
    deleted_asset_id: str
    outputs_preserved: Literal[True] = True
    unbound_usages: list[str] = Field(default_factory=list)


class AssetTrashRequest(StrictModel):
    asset_ids: Annotated[list[str], Field(min_length=1, max_length=128)]
    cascade: bool = False

    @field_validator("asset_ids")
    @classmethod
    def validate_asset_ids(cls, value: list[str]) -> list[str]:
        if any(not asset_id for asset_id in value):
            raise ValueError("asset ids must be non-empty")
        if len(set(value)) != len(value):
            raise ValueError("asset ids must be unique")
        return value


class AssetTrashBatchRead(StrictModel):
    batch_id: str
    asset_ids: list[str]
    assets: list[AssetReference]
    cascade: bool
    unbound_usages: list[str] = Field(default_factory=list)
    unbound_usages_by_asset: dict[str, list[str]] = Field(default_factory=dict)
    created_at: str
    remote_files_preserved: Literal[True] = True


class AssetTrashListRead(StrictModel):
    batches: list[AssetTrashBatchRead]
    remote_files_preserved: Literal[True] = True


class AssetTrashRestoreRequest(StrictModel):
    mode: Literal["registration_only", "with_references"]


class AssetTrashRestoreRead(StrictModel):
    batch_id: str
    restored_asset_ids: list[str]
    restored_references: bool
    mode: Literal["registration_only", "with_references"]
    remote_files_preserved: Literal[True] = True


class AssetTrashPurgeRead(StrictModel):
    batch_id: str
    purged_asset_ids: list[str]
    remote_files_preserved: Literal[True] = True


class JobChildRead(StrictModel):
    id: str
    family: Literal["fl2va", "ref2va"]
    backend: Literal["standard", "raylight"]
    segment_ids: list[str]
    status: JobStatus
    progress: Annotated[float, Field(ge=0, le=1)]
    stage: str | None = None
    prompt_id: str | None = None
    outputs: list[str] = Field(default_factory=list)
    error: str | None = None


class JobSegmentResultRead(StrictModel):
    """One unambiguous generated take mapped back to its stable segment ID."""

    segment_id: str
    child_id: str
    output_url: str
    output_file: str
    # This reports exact complete timeline/runtime equality. Historical takes
    # remain valid versioned outputs and may still be selected by stable
    # segment ID; clients use this flag only to describe exact currentness.
    current_snapshot: bool


class JobRead(StrictModel):
    id: str
    mode: JobMode
    status: JobStatus
    display_name: str
    project_title: str | None = None
    # Stable project ownership when a timeline task was submitted under the
    # multi-project model. Legacy six-mode tasks and tasks predating the
    # project table keep ``None`` and remain visible under the "旧任务" bucket.
    project_id: str | None = None
    # ``current_project`` deliberately compares only the typed timeline. A
    # later model/runtime setting edit must not move a task into a different
    # project's history filter. ``segment_results.current_snapshot`` remains
    # the stricter timeline + runtime eligibility check for the main monitor.
    current_project: bool = False
    progress: Annotated[float, Field(ge=0, le=1)]
    stage: str | None = None
    prompt_id: str | None = None
    outputs: list[str] = Field(default_factory=list)
    output_files: list[str] = Field(default_factory=list)
    error: str | None = None
    preview_url: str | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None
    execution_duration_seconds: Annotated[float, Field(ge=0)] | None = None
    output_count: Annotated[int, Field(ge=0)] = 0
    error_summary: str | None = None
    children: list[JobChildRead] = Field(default_factory=list)
    segment_results: list[JobSegmentResultRead] = Field(default_factory=list)
    live_preview_url: str | None = None


class JobStatusSummaryRead(StrictModel):
    total: Annotated[int, Field(ge=0)] = 0
    active: Annotated[int, Field(ge=0)] = 0
    queued: Annotated[int, Field(ge=0)] = 0
    preparing: Annotated[int, Field(ge=0)] = 0
    running: Annotated[int, Field(ge=0)] = 0
    cancelling: Annotated[int, Field(ge=0)] = 0
    succeeded: Annotated[int, Field(ge=0)] = 0
    failed: Annotated[int, Field(ge=0)] = 0
    cancelled: Annotated[int, Field(ge=0)] = 0


class JobListRead(StrictModel):
    jobs: list[JobRead]
    # ``total`` is the number matching the current list filter, while summary
    # always describes the complete local Director history.
    total: Annotated[int, Field(ge=0)] = 0
    limit: Annotated[int, Field(ge=1, le=256)] = 100
    offset: Annotated[int, Field(ge=0)] = 0
    has_more: bool = False
    summary: JobStatusSummaryRead = Field(default_factory=JobStatusSummaryRead)


class JobBulkCancelRequest(StrictModel):
    job_ids: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=128)]],
        Field(min_length=1, max_length=100),
    ]

    @field_validator("job_ids")
    @classmethod
    def unique_job_ids(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("job_ids must be unique")
        return value


class JobRecoveryConfirmComfyRestartRequest(StrictModel):
    """Deliberate operator certificate for an ambiguous pre-restart submit.

    An empty ComfyUI queue/history response is not proof that an old
    ``POST /prompt`` can never finish validation and enqueue later.  This
    non-boolean literal makes callers explicitly certify the one external fact
    that closes that race: the owning ComfyUI process was restarted.
    """

    confirmation: Literal["comfyui_process_restarted"]


class RayLightRuntimeRecoveryConfirmRequest(StrictModel):
    """Operator certificate for discarding a pre-restart RayLight ledger."""

    confirmation: Literal["comfyui_process_restarted"]
    expected_epoch: Annotated[int, Field(ge=0)]
    expected_recovery_token: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$")
    ]


class RayLightRuntimeStatusRead(StrictModel):
    active: bool
    recovery_required: bool
    epoch: Annotated[int, Field(ge=0)]
    runtime_gpu_indexes: list[Annotated[int, Field(ge=0)]]
    available_gpu_indexes: list[Annotated[int, Field(ge=0)]]
    invalid_gpu_indexes: list[Annotated[int, Field(ge=0)]]
    tainted: bool
    recovery_token: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$")
    ] | None


class JobBulkCancelRead(StrictModel):
    jobs: list[JobRead]
    requested_count: Annotated[int, Field(ge=1, le=100)]
    terminal_count: Annotated[int, Field(ge=0, le=100)]


class JobDiagnosticChildRead(StrictModel):
    id: str
    family: Literal["fl2va", "ref2va"]
    backend: Literal["standard", "raylight"]
    segment_ids: list[str]
    status: JobStatus
    progress: Annotated[float, Field(ge=0, le=1)]
    stage: str | None = None
    output_files: list[str] = Field(default_factory=list)
    error_summary: str | None = None


class JobDiagnosticRead(StrictModel):
    schema_version: Literal[1] = 1
    id: str
    display_name: str
    project_title: str | None = None
    mode: JobMode
    status: JobStatus
    progress: Annotated[float, Field(ge=0, le=1)]
    stage: str | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None
    execution_duration_seconds: Annotated[float, Field(ge=0)] | None = None
    output_files: list[str] = Field(default_factory=list)
    error_summary: str | None = None
    children: list[JobDiagnosticChildRead] = Field(default_factory=list)
    settings_included: Literal[False] = False
    workflow_included: Literal[False] = False


class JobGenerationRenderRead(StrictModel):
    width: int
    height: int
    fps: float
    export_mode: Literal["all", "segments"]
    total_duration_seconds: Annotated[float, Field(gt=0)]


class JobGenerationSamplingRead(StrictModel):
    family: Literal["fl2va", "ref2va"]
    steps: int
    seed: Annotated[int, Field(ge=0, le=2**53 - 1)]
    random_seed: bool
    sampler: Literal["res_multistep", "euler", "dpmpp_2m"]
    scheduler: SchedulerName
    shift: float
    audio_shift: float


class JobGenerationModelRead(StrictModel):
    family: Literal["fl2va", "ref2va"]
    filename: str
    device: str
    lora_name: str | None = None
    lora_strength: float
    backends: list[Literal["standard", "raylight"]]
    # These are ComfyUI/RayLight logical indices, never claims about physical
    # PCI bus numbering. They are relevant only when RayLight was selected.
    logical_gpu_indices: list[int] = Field(default_factory=list)
    ulysses_degree: int | None = None
    ring_degree: int | None = None


class JobGenerationSharedModelRead(StrictModel):
    role: Literal["clip", "video_vae", "audio_vae"]
    filename: str
    device: str


class JobGenerationSegmentRead(StrictModel):
    id: str
    title: str
    family: Literal["fl2va", "ref2va"]
    recipe: GenerationMode
    duration_seconds: Annotated[float, Field(gt=0)]
    prompt: str
    continuity_enabled: bool
    continuity_overlap_frames: Literal[5, 22, 39, 56]
    ref_image_size: Literal["match", "max"]
    audio_mode: Literal["generate", "source", "mute"]
    has_first_image: bool = False
    has_last_image: bool = False
    has_source_video: bool = False
    source_audio_as_reference: bool = False
    reference_image_count: Annotated[int, Field(ge=0)] = 0
    reference_audio_count: Annotated[int, Field(ge=0)] = 0
    reference_video_count: Annotated[int, Field(ge=0)] = 0


class JobGenerationDetailsRead(StrictModel):
    """Typed, human-readable generation parameters for one historical task.

    This deliberately excludes endpoint credentials, client IDs, workflow
    graphs, ComfyUI prompt identifiers, and absolute asset/output paths.
    """

    schema_version: Literal[2] = 2
    job_id: str
    project_title: str
    render: JobGenerationRenderRead
    sampling: list[JobGenerationSamplingRead]
    models: list[JobGenerationModelRead] = Field(default_factory=list)
    shared_models: list[JobGenerationSharedModelRead] = Field(default_factory=list)
    runtime_snapshot_available: bool
    segments: list[JobGenerationSegmentRead]


class JobProjectSnapshotRead(StrictModel):
    job_id: str
    project: UnifiedTimelineDraftV5
    segment_ids: list[str] | None = None


class JobOutputImportRequest(StrictModel):
    output_index: Annotated[int, Field(ge=0)] | None = None
    segment_id: Annotated[str, Field(min_length=1, max_length=128)] | None = None

    @model_validator(mode="after")
    def exactly_one_output_identity(self) -> "JobOutputImportRequest":
        if (self.output_index is None) == (self.segment_id is None):
            raise ValueError("exactly one of output_index or segment_id is required")
        return self


class JobOutputImportRead(StrictModel):
    asset: AssetReference


class JobDeleteRead(StrictModel):
    deleted_job_id: str
    outputs_preserved: Literal[True] = True


class JobClearRead(StrictModel):
    deleted_count: Annotated[int, Field(ge=0)]
    active_count: Annotated[int, Field(ge=0)]
    outputs_preserved: Literal[True] = True


class ProjectSummaryRead(StrictModel):
    """Lightweight project listing row; the timeline document is fetched separately."""

    id: str
    title: str
    created_at: str
    updated_at: str
    # Zero is the bounded sentinel for a preserved project whose raw document
    # cannot be decoded by the current reader. Valid timeline documents still
    # contain at least one segment.
    segment_count: Annotated[int, Field(ge=0)]


class ProjectListRead(StrictModel):
    projects: list[ProjectSummaryRead]


class ProjectCreateRequest(StrictModel):
    title: Annotated[str, Field(min_length=0, max_length=256)] = ""
    initial_model_stack: ModelStack | None = None


class ProjectRenameRequest(StrictModel):
    title: Annotated[str, Field(min_length=1, max_length=256)]


class ProjectImportRequest(StrictModel):
    """Create a project from an existing validated timeline document.

    Segment identities are preserved verbatim; the new project id still scopes
    the durable take ledger, so an imported project starts without reused
    renders.
    """

    title: Annotated[str, Field(min_length=0, max_length=256)] = ""
    document: UnifiedTimelineDraftV5


class TimelineAuthorityRead(StrictModel):
    """One timeline document together with its durable server CAS revision."""

    document: UnifiedTimelineDraftV5
    revision: Annotated[int, Field(ge=0, le=MAX_TIMELINE_REVISION)]


class FeatureBundleMigrationNoticeListRead(StrictModel):
    """Bounded, advisory-only messages owned by one project authority."""

    notices: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=512)]],
        Field(max_length=32),
    ]


class TimelineAuthorityWriteRequest(StrictModel):
    """A conditional timeline replacement based on an exact server revision."""

    document: UnifiedTimelineDraftV5
    expected_revision: Annotated[int, Field(ge=0, le=MAX_TIMELINE_REVISION)]


class TimelineRevisionConflictRead(StrictModel):
    code: Literal["timeline_revision_conflict"] = "timeline_revision_conflict"
    message: str
    project_id: str
    expected_revision: Annotated[int, Field(ge=0, le=MAX_TIMELINE_REVISION)]
    actual_revision: Annotated[int, Field(ge=0, le=MAX_TIMELINE_REVISION)]


class TimelineRevisionExhaustedRead(StrictModel):
    code: Literal["timeline_revision_exhausted"] = "timeline_revision_exhausted"
    message: str
    project_id: str
    revision: Annotated[int, Field(ge=0, le=MAX_TIMELINE_REVISION)]


class ProjectDeleteRead(StrictModel):
    deleted_project_id: str
    # Deleting a project never removes ComfyUI-owned output files.
    outputs_preserved: Literal[True] = True
    # Task records are kept as audit history (they become "旧任务" with a
    # null project_id) unless the caller opts into cascading task deletion.
    orphaned_jobs: Annotated[int, Field(ge=0)] = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
