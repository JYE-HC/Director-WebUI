from __future__ import annotations

"""Pure v4 creative-input resolution for the staged workflow compiler.

The resolver is intentionally separated from prompt construction.  It reads
one already-validated v4 timeline, the settings captured for that same
request, and the two explicit pieces of server evidence supplied by the
caller.  It never reads a database, probes ComfyUI, mutates its inputs, or
falls back to another creative authority.

Stage 2 keeps the production compiler on its legacy path.  These immutable
results let the replacement segment compilers consume exactly the same route
decisions without making ``native_templates`` a dependency of this module (or
creating a future import cycle when that module adopts the resolver).
"""

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Protocol, TypeAlias

from pydantic import Field, model_validator

from ..schemas import (
    DiffusionModelBinding,
    RuntimeSettings,
    UnifiedFL2VASegment,
    UnifiedRef2VASegment,
    UnifiedTimelineDraft,
    UnifiedTimelineSegment,
    timeline_segment_recipe,
)
from .contracts import (
    Backend,
    ContractModel,
    FrozenMap,
    JsonObject,
    JsonValue,
    ModelFamily,
)
from .lora_factory import (
    LoraAdapterResolutionError,
    LoraLoaderBindingKey,
    ResolvedLoraAdapter,
    require_lora_adapter,
    resolve_raylight_lora_adapter,
)


ContinuitySource: TypeAlias = Literal["same_run", "historical_take"]
TemplateId: TypeAlias = Literal["h3_standard_segment", "h3_raylight_segment"]
Recipe: TypeAlias = Literal["t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"]
LoraLoaderNode: TypeAlias = Literal[
    "MiniMaxH3TurboLoRA",
    "LoraLoaderBypassModelOnly",
    "LoraLoaderModelOnly",
    "DirectorDeckRayLoraLoader",
]
# Historical v4/job evidence may still deserialize ``factory_default``; live
# resolution has no factory fallback and emits ``user_override`` instead.
LoraResolutionSource: TypeAlias = Literal[
    "user_override",
    "factory_default",
    "backend_fixed",
]

SegmentId = Annotated[str, Field(min_length=1, max_length=128)]
UnitId = Annotated[str, Field(min_length=1, max_length=256)]


class CreativeCompileInputError(ValueError):
    """The explicit v4 inputs cannot be represented by the native templates."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "creative_configuration_invalid",
        rule: str = "v4_creative_input_resolution",
        remediation: str = (
            "Correct the selected timeline or runtime mapping and run preflight again."
        ),
        feature_id: str | None = None,
        segment_id: str | None = None,
        backend: Backend | None = None,
        safe_details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.rule = rule
        self.public_message = message
        self.remediation = remediation
        self.feature_id = feature_id
        self.segment_id = segment_id
        self.backend = backend
        self.safe_details = dict(safe_details or {})


class HistoricalTakeLike(Protocol):
    """Structural input accepted from the current historical-take resolver."""

    id: str
    segment_id: str
    output: Mapping[str, Any]


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _template_for_backend(backend: Backend) -> TemplateId:
    return (
        "h3_standard_segment"
        if backend == "standard"
        else "h3_raylight_segment"
    )


class V4ResolvedHistoricalTake(ContractModel):
    """The exact persisted predecessor evidence consumed by one target."""

    target_segment_id: SegmentId
    # The legacy NativeHistoricalTake deliberately did not constrain this
    # database identity; preserve that acceptance boundary during stage 2.
    id: str
    segment_id: SegmentId
    output: JsonObject
    annotated_output_path: Annotated[str, Field(min_length=1, max_length=1_050)]

    @model_validator(mode="after")
    def validate_annotated_path(self) -> "V4ResolvedHistoricalTake":
        expected = _annotated_predecessor_output(self.output)
        if self.annotated_output_path != expected:
            raise ValueError("annotated historical output path does not match output")
        return self

    def materialize_output(self) -> dict[str, Any]:
        """Return a fresh ordinary mapping for the legacy late binder."""

        return _thaw_json(self.output)


class V4LoraResolution(ContractModel):
    """The current per-family loader decision, resolved before graph emission."""

    family: ModelFamily
    backend: Backend
    lora_name: Annotated[str, Field(min_length=1, max_length=1_024)]
    model_filename: Annotated[str, Field(min_length=1, max_length=1_024)]
    adapter_id: Annotated[str, Field(min_length=1, max_length=256)]
    binding: LoraLoaderBindingKey | None
    loader_node: LoraLoaderNode
    source: LoraResolutionSource
    options: FrozenMap[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_backend_loader(self) -> "V4LoraResolution":
        if self.backend == "raylight":
            if (
                self.loader_node != "DirectorDeckRayLoraLoader"
                or self.adapter_id != "ray_lora"
                or self.source != "backend_fixed"
                or self.binding is not None
                or self.options
            ):
                raise ValueError("RayLight LoRA resolution must use DirectorDeckRayLoraLoader")
        elif (
            self.loader_node == "DirectorDeckRayLoraLoader"
            or self.adapter_id == "ray_lora"
            or self.source == "backend_fixed"
            or self.binding is None
        ):
            raise ValueError(
                "Standard LoRA resolution must carry one exact mapped binding"
            )
        elif (
            self.binding.family != self.family
            or self.binding.model_filename != self.model_filename
            or self.binding.lora_filename != self.lora_name
        ):
            raise ValueError("Standard LoRA binding does not match the route")
        return self


class V4ResolvedSegmentRoute(ContractModel):
    """One immutable, graph-free route for the existing one-segment unit."""

    timeline_index: Annotated[int, Field(ge=0, le=127)]
    segment_id: SegmentId
    segment_document: JsonObject
    family: ModelFamily
    recipe: Recipe
    backend: Backend
    template_id: TemplateId
    unit_id: UnitId
    predecessor_segment_id: SegmentId | None = None
    continuity_source: ContinuitySource | None = None
    continuity_overlap_frames: Literal[5, 22, 39, 56]
    historical_take: V4ResolvedHistoricalTake | None = None
    anchor_reset: bool
    clear_raylight_vram_after_sampling: bool
    lora_resolution: V4LoraResolution | None = None

    @model_validator(mode="after")
    def validate_route(self) -> "V4ResolvedSegmentRoute":
        if self.template_id != _template_for_backend(self.backend):
            raise ValueError("segment template must match resolved backend")
        expected_unit_id = (
            f"{self.backend}-{self.family}-{self.timeline_index:03d}"
        )
        if self.unit_id != expected_unit_id:
            raise ValueError("unit id must match the legacy routing identity")
        if self.segment_document.get("id") != self.segment_id:
            raise ValueError("segment snapshot id must match route owner")
        if self.segment_document.get("mode") != self.family:
            raise ValueError("segment snapshot family must match route family")
        if self.predecessor_segment_id is None:
            if self.continuity_source is not None or self.historical_take is not None:
                raise ValueError(
                    "continuity source requires an authored predecessor"
                )
        elif self.continuity_source is None:
            raise ValueError("authored predecessor requires a continuity source")
        if self.continuity_source == "historical_take":
            if self.historical_take is None:
                raise ValueError("historical continuity requires take evidence")
            if (
                self.historical_take.target_segment_id != self.segment_id
                or self.historical_take.segment_id
                != self.predecessor_segment_id
            ):
                raise ValueError("historical take does not match continuity route")
        elif self.historical_take is not None:
            raise ValueError("same-run continuity cannot carry a historical take")
        if self.anchor_reset and self.predecessor_segment_id is not None:
            raise ValueError("an anchor reset cannot also consume a predecessor")
        if self.lora_resolution is not None and (
            self.lora_resolution.family != self.family
            or self.lora_resolution.backend != self.backend
        ):
            raise ValueError("LoRA resolution must match route family and backend")
        return self

    def materialize_segment(self) -> UnifiedTimelineSegment:
        """Return a fresh typed segment without exposing mutable source state."""

        document = _thaw_json(self.segment_document)
        if self.family == "fl2va":
            segment = UnifiedFL2VASegment.model_validate(document)
        else:
            segment = UnifiedRef2VASegment.model_validate(document)
        if timeline_segment_recipe(segment) != self.recipe:
            raise AssertionError("resolved segment recipe drifted during round-trip")
        return segment


class V4CreativeCompileInput(ContractModel):
    """Complete immutable authority and route snapshot for a v4 compilation."""

    version: Literal[4] = 4
    creative_authority: Literal[
        "timeline_v4+captured_legacy_settings"
    ] = "timeline_v4+captured_legacy_settings"
    draft_document: JsonObject
    captured_legacy_settings: JsonObject
    requested_segment_ids: tuple[SegmentId, ...] | None
    selected_segment_ids: tuple[SegmentId, ...]
    routes: tuple[V4ResolvedSegmentRoute, ...]
    submission_order: tuple[UnitId, ...]
    families: tuple[ModelFamily, ...]
    has_continuity_edges: bool
    keep_raylight_resident: bool
    clear_raylight_vram_after_sampling: bool

    @model_validator(mode="after")
    def validate_compile_input(self) -> "V4CreativeCompileInput":
        if self.draft_document.get("version") != 4:
            raise ValueError("v4 creative input requires timeline version 4")
        if self.captured_legacy_settings.get("memory_policy") != "keep_resident":
            raise ValueError("captured settings use an unsupported memory policy")
        if not self.routes or not self.selected_segment_ids:
            raise ValueError("v4 creative input must contain at least one route")
        route_segment_ids = tuple(route.segment_id for route in self.routes)
        if len(set(route_segment_ids)) != len(route_segment_ids):
            raise ValueError("resolved segment routes must be unique")
        if set(route_segment_ids) != set(self.selected_segment_ids):
            raise ValueError("selected segment ids must equal resolved route owners")
        if len(set(self.selected_segment_ids)) != len(self.selected_segment_ids):
            raise ValueError("selected segment ids must be unique")
        if self.submission_order != tuple(route.unit_id for route in self.routes):
            raise ValueError("submission order must match route order")
        if len(set(self.submission_order)) != len(self.submission_order):
            raise ValueError("resolved unit ids must be unique")
        expected_families = tuple(
            family
            for family in ("fl2va", "ref2va")
            if any(route.family == family for route in self.routes)
        )
        if self.families != expected_families:
            raise ValueError("families must use the legacy canonical order")
        expected_edges = any(
            route.predecessor_segment_id is not None for route in self.routes
        )
        if self.has_continuity_edges != expected_edges:
            raise ValueError("continuity edge summary does not match routes")
        if self.clear_raylight_vram_after_sampling == self.keep_raylight_resident:
            raise ValueError("RayLight residency flags must be complementary")
        if any(
            route.clear_raylight_vram_after_sampling
            != self.clear_raylight_vram_after_sampling
            for route in self.routes
        ):
            raise ValueError("all routes must share captured RayLight residency")

        expected_routes = (
            tuple(sorted(self.routes, key=lambda route: route.timeline_index))
            if expected_edges
            else tuple(
                sorted(
                    self.routes,
                    key=lambda route: (
                        ("standard", "raylight").index(route.backend),
                        ("fl2va", "ref2va").index(route.family),
                        route.timeline_index,
                    ),
                )
            )
        )
        if self.routes != expected_routes:
            raise ValueError("routes do not follow the legacy submission order")

        draft_segments = self.draft_document.get("segments")
        if not isinstance(draft_segments, tuple):
            raise ValueError("v4 timeline snapshot must contain segment array")
        selected_in_timeline = tuple(
            segment.get("id")
            for segment in draft_segments
            if isinstance(segment, FrozenMap)
            and segment.get("enabled") is True
            and segment.get("id") in set(self.selected_segment_ids)
        )
        if self.selected_segment_ids != selected_in_timeline:
            raise ValueError("selected segment ids must retain timeline order")
        return self

    def requires_timeline_assembly(self) -> bool:
        """Return whether this exact selection will run Director ffmpeg.

        Whole-timeline assembly is a contextual capability rather than a
        property of ``SaveVideo``.  A partial selection, per-segment export, or
        one-segment full run publishes native takes without re-encoding.
        """

        if self.draft_document.get("export_mode") != "all":
            return False
        if len(self.selected_segment_ids) <= 1:
            return False
        segments = self.draft_document.get("segments")
        if not isinstance(segments, (tuple, list)):
            raise AssertionError("validated v4 draft has no segment list")
        enabled_ids = {
            str(segment["id"])
            for segment in segments
            if isinstance(segment, Mapping)
            and segment.get("enabled", True) is True
            and isinstance(segment.get("id"), str)
        }
        return bool(enabled_ids) and set(self.selected_segment_ids) == enabled_ids

    def materialize_draft(self) -> UnifiedTimelineDraft:
        """Return a fresh strict v4 draft reconstructed from the snapshot."""

        return UnifiedTimelineDraft.model_validate(_thaw_json(self.draft_document))

    def materialize_settings(self) -> RuntimeSettings:
        """Return fresh captured settings; no live-settings merge is performed."""

        return RuntimeSettings.model_validate(
            _thaw_json(self.captured_legacy_settings)
        )


def _resolve_lora(
    family: ModelFamily,
    backend: Backend,
    binding: DiffusionModelBinding,
    resolved_adapter: ResolvedLoraAdapter | None,
) -> V4LoraResolution | None:
    if binding.lora_name is None:
        return None
    if binding.lora_strength == 0.0:
        raise CreativeCompileInputError(
            f"{family} LoRA is enabled with zero strength; clear the LoRA "
            "selection to disable it or choose a non-zero strength",
            code="lora_strength_invalid",
            rule="lora_strength",
            remediation="Clear the LoRA selection or choose a finite non-zero strength.",
            feature_id="lora",
            backend=backend,
            safe_details={"family": family},
        )
    if backend == "raylight":
        ray = resolve_raylight_lora_adapter(family)
        return V4LoraResolution(
            family=family,
            backend=backend,
            lora_name=binding.lora_name,
            model_filename=binding.filename,
            adapter_id=ray.adapter.adapter_id,
            binding=None,
            loader_node=ray.adapter.class_type,
            source=ray.source,
            options=dict(ray.options),
        )

    exact_binding = LoraLoaderBindingKey(
        family=family,
        model_filename=binding.filename,
        lora_filename=binding.lora_name,
    )
    if resolved_adapter is None:
        override = binding.standard_lora_loader_override
        if override is not None and (
            override.model_filename == exact_binding.model_filename
            and override.lora_name == exact_binding.lora_filename
        ):
            try:
                adapter = require_lora_adapter(override.loader)
            except LoraAdapterResolutionError as exc:
                raise CreativeCompileInputError(
                    "The selected Standard LoRA adapter is unknown.",
                    code=exc.code,
                    rule="standard_lora_adapter",
                    remediation=(
                        "Choose a supported adapter for this exact model and "
                        "LoRA binding in Settings."
                    ),
                    feature_id="lora",
                    backend=backend,
                    safe_details={
                        "family": family,
                        "adapter_id": override.loader,
                    },
                ) from exc
            resolved_adapter = ResolvedLoraAdapter(
                adapter=adapter,
                binding=exact_binding,
                source="user_override",
                options=(
                    {"low_vram": binding.lora_low_vram}
                    if adapter.input_contract == "dedicated_model"
                    else {}
                ),
            )
    if resolved_adapter is None:
        raise CreativeCompileInputError(
            "The selected Standard LoRA requires an explicit compatible loader mapping.",
            code="lora_loader_mapping_required",
            rule="standard_lora_resolution",
            remediation="Choose the Standard LoRA loader for this exact model and LoRA pair in Settings.",
            feature_id="lora",
            backend=backend,
            safe_details={"family": family},
        )
    if (
        resolved_adapter.adapter.backend != "standard"
        or family not in resolved_adapter.adapter.supported_families
        or resolved_adapter.binding != exact_binding
    ):
        raise CreativeCompileInputError(
            "The resolved LoRA adapter does not match the exact Standard binding.",
            code="lora_adapter_incompatible",
            rule="standard_lora_adapter",
            remediation=(
                "Choose a supported adapter for this exact model and LoRA "
                "binding in Settings."
            ),
            feature_id="lora",
            backend=backend,
            safe_details={
                "family": family,
                "adapter_id": resolved_adapter.adapter.adapter_id,
            },
        )
    return V4LoraResolution(
        family=family,
        backend=backend,
        lora_name=binding.lora_name,
        model_filename=binding.filename,
        adapter_id=resolved_adapter.adapter.adapter_id,
        binding=exact_binding,
        loader_node=resolved_adapter.adapter.class_type,
        source=resolved_adapter.source,
        options=dict(resolved_adapter.options),
    )


def _resolve_backend(binding: DiffusionModelBinding) -> Backend:
    return "raylight" if len(binding.raylight.gpu_select) >= 2 else "standard"


def _selected_segments(
    draft: UnifiedTimelineDraft,
    segment_ids: list[str] | tuple[str, ...] | None,
) -> list[UnifiedTimelineSegment]:
    enabled = [segment for segment in draft.segments if segment.enabled]
    by_id = {segment.id: segment for segment in enabled}
    if segment_ids is None:
        selected = enabled
    else:
        missing = [segment_id for segment_id in segment_ids if segment_id not in by_id]
        if missing:
            raise CreativeCompileInputError(
                "The requested segment selection is empty, disabled, or stale.",
                code="segment_selection_invalid",
                rule="segment_selection",
                remediation="Select only enabled segments from the current project and run preflight again.",
            )
        selected_set = set(segment_ids)
        selected = [segment for segment in enabled if segment.id in selected_set]
    if not selected:
        raise CreativeCompileInputError(
            "at least one enabled timeline segment is required",
            code="segment_selection_invalid",
            rule="segment_selection",
            remediation="Enable and select at least one timeline segment.",
        )
    return selected


def _require_dense_slots(
    values: list[Any], *, field: str, segment_id: str
) -> None:
    slots = sorted(value.slot for value in values)
    expected = list(range(len(values)))
    if slots != expected:
        raise CreativeCompileInputError(
            f"segment '{segment_id}' {field} slots must be dense {expected}; "
            f"got {slots}. Repair the slots and prompt tags explicitly."
        )


def _validate_native_reference_slots(segment: UnifiedTimelineSegment) -> None:
    if isinstance(segment, UnifiedRef2VASegment):
        _require_dense_slots(
            segment.reference_images,
            field="reference_images",
            segment_id=segment.id,
        )
        _require_dense_slots(
            segment.reference_videos,
            field="reference_videos",
            segment_id=segment.id,
        )
        _require_dense_slots(
            segment.reference_audios,
            field="reference_audios",
            segment_id=segment.id,
        )


def _continuity_predecessors(
    draft: UnifiedTimelineDraft,
) -> dict[str, UnifiedTimelineSegment]:
    predecessors: dict[str, UnifiedTimelineSegment] = {}
    previous: UnifiedTimelineSegment | None = None
    for segment in (item for item in draft.segments if item.enabled):
        anchor_reset = (
            isinstance(segment, UnifiedFL2VASegment)
            and segment.first_image is not None
        )
        if segment.continuity.enabled and previous is not None and not anchor_reset:
            predecessors[segment.id] = previous
        previous = segment
    return predecessors


def _annotated_predecessor_output(output: Mapping[str, Any]) -> str:
    filename = output.get("filename")
    subfolder = output.get("subfolder", "")
    output_type = output.get("type")
    if output_type != "output":
        raise CreativeCompileInputError(
            "continuity predecessor must be a persisted ComfyUI output"
        )
    if not isinstance(filename, str) or not filename or len(filename) > 512:
        raise CreativeCompileInputError(
            "continuity predecessor filename is invalid"
        )
    if not isinstance(subfolder, str) or len(subfolder) > 512:
        raise CreativeCompileInputError(
            "continuity predecessor subfolder is invalid"
        )
    if (
        filename != filename.strip()
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or "[" in filename
        or "]" in filename
        or any(
            ord(character) < 32 or ord(character) == 127
            for character in filename
        )
    ):
        raise CreativeCompileInputError(
            "continuity predecessor filename is unsafe"
        )
    if (
        subfolder != subfolder.strip()
        or "\\" in subfolder
        or "[" in subfolder
        or "]" in subfolder
        or any(
            ord(character) < 32 or ord(character) == 127
            for character in subfolder
        )
    ):
        raise CreativeCompileInputError(
            "continuity predecessor subfolder is unsafe"
        )
    folder = PurePosixPath(subfolder)
    if folder.is_absolute() or any(
        part in {"", ".", ".."} for part in folder.parts
    ):
        raise CreativeCompileInputError(
            "continuity predecessor subfolder is unsafe"
        )
    relative = PurePosixPath(filename) if not subfolder else folder / filename
    return f"{relative.as_posix()} [output]"


class CreativeCompileInputResolver:
    """Resolve versioned creative authority without graph or I/O side effects."""

    @staticmethod
    def resolve_v4(
        draft: UnifiedTimelineDraft,
        captured_legacy_settings: RuntimeSettings,
        selected_segment_ids: list[str] | tuple[str, ...] | None,
        historical_takes: Mapping[str, HistoricalTakeLike] | None,
        resolved_lora_adapters: Mapping[
            ModelFamily, ResolvedLoraAdapter
        ]
        | None,
    ) -> V4CreativeCompileInput:
        if draft.version != 4:
            raise CreativeCompileInputError(
                "v4 creative input requires timeline version 4"
            )
        if captured_legacy_settings.memory_policy != "keep_resident":
            raise CreativeCompileInputError(
                "native segment workflows support memory_policy='keep_resident' "
                "only; clear_between_segments has no equivalent stock ComfyUI node"
            )
        if draft.render.fps != 24.0:
            raise CreativeCompileInputError(
                "MiniMax H3 native temporal and reference-video contracts are "
                "fixed at 24 fps; render.fps must equal 24"
            )

        selected = _selected_segments(draft, selected_segment_ids)
        for segment in selected:
            _validate_native_reference_slots(segment)

        timeline_positions = {
            segment.id: index for index, segment in enumerate(draft.segments)
        }
        continuity_predecessors = _continuity_predecessors(draft)
        enabled_segments = [segment for segment in draft.segments if segment.enabled]
        previous_enabled = {
            segment.id: enabled_segments[index - 1] if index else None
            for index, segment in enumerate(enabled_segments)
        }
        selected_ids = {segment.id for segment in selected}
        keep_raylight_resident = (
            captured_legacy_settings.raylight_residency_policy
            == "keep_until_switch"
        )
        clear_raylight_vram_after_sampling = not keep_raylight_resident

        routed: list[V4ResolvedSegmentRoute] = []
        lora_resolutions: dict[ModelFamily, V4LoraResolution | None] = {}
        for segment in selected:
            timeline_index = timeline_positions[segment.id]
            family: ModelFamily = segment.mode
            binding = getattr(captured_legacy_settings.models, family)
            backend = _resolve_backend(binding)
            if backend == "raylight" and not captured_legacy_settings.multi_gpu_enabled:
                raise CreativeCompileInputError(
                    "multi-GPU inference is disabled; enable it in system settings "
                    "(installs the RayLight components and requires a restart) or "
                    f"reduce {family}.raylight.gpu_select to a single GPU"
                )
            if backend == "raylight" and binding.device != "default":
                raise CreativeCompileInputError(
                    f"{family}.device must be 'default' when RayLight is enabled; "
                    "raylight.gpu_select is the authoritative logical GPU pool"
                )

            predecessor_segment_id: str | None = None
            continuity_source: ContinuitySource | None = None
            historical_snapshot: V4ResolvedHistoricalTake | None = None
            explicit_anchor_reset = (
                isinstance(segment, UnifiedFL2VASegment)
                and segment.first_image is not None
            )
            anchor_reset = False
            if segment.continuity.enabled:
                predecessor = continuity_predecessors.get(segment.id)
                anchor_reset = predecessor is None and (
                    previous_enabled[segment.id] is None or explicit_anchor_reset
                )
                if predecessor is not None:
                    predecessor_segment_id = predecessor.id
                    if predecessor.id in selected_ids:
                        continuity_source = "same_run"
                    else:
                        continuity_source = "historical_take"
                        historical_take = (
                            None
                            if historical_takes is None
                            else historical_takes.get(segment.id)
                        )
                        if historical_take is None:
                            raise CreativeCompileInputError(
                                f"continuity segment '{segment.id}' requires a "
                                "server-resolved historical take for predecessor "
                                f"'{predecessor.id}'"
                            )
                        if historical_take.segment_id != predecessor.id:
                            raise CreativeCompileInputError(
                                f"historical take '{historical_take.id}' belongs to "
                                f"segment '{historical_take.segment_id}', not current "
                                f"predecessor '{predecessor.id}'"
                            )
                        annotated_path = _annotated_predecessor_output(
                            historical_take.output
                        )
                        historical_snapshot = V4ResolvedHistoricalTake(
                            target_segment_id=segment.id,
                            id=historical_take.id,
                            segment_id=historical_take.segment_id,
                            output=dict(historical_take.output),
                            annotated_output_path=annotated_path,
                        )

            if family not in lora_resolutions:
                resolved_adapter = (
                    None
                    if resolved_lora_adapters is None
                    else resolved_lora_adapters.get(family)
                )
                lora_resolutions[family] = _resolve_lora(
                    family,
                    backend,
                    binding,
                    resolved_adapter,
                )
            lora_resolution = lora_resolutions[family]
            routed.append(
                V4ResolvedSegmentRoute(
                    timeline_index=timeline_index,
                    segment_id=segment.id,
                    segment_document=segment.model_dump(mode="json"),
                    family=family,
                    recipe=timeline_segment_recipe(segment),
                    backend=backend,
                    template_id=_template_for_backend(backend),
                    unit_id=f"{backend}-{family}-{timeline_index:03d}",
                    predecessor_segment_id=predecessor_segment_id,
                    continuity_source=continuity_source,
                    continuity_overlap_frames=segment.continuity.overlap_frames,
                    historical_take=historical_snapshot,
                    anchor_reset=anchor_reset,
                    clear_raylight_vram_after_sampling=(
                        clear_raylight_vram_after_sampling
                    ),
                    lora_resolution=lora_resolution,
                )
            )

        has_continuity_edges = any(
            route.predecessor_segment_id is not None for route in routed
        )
        ordered_routes = (
            tuple(sorted(routed, key=lambda route: route.timeline_index))
            if has_continuity_edges
            else tuple(
                sorted(
                    routed,
                    key=lambda route: (
                        ("standard", "raylight").index(route.backend),
                        ("fl2va", "ref2va").index(route.family),
                        route.timeline_index,
                    ),
                )
            )
        )
        families = tuple(
            family
            for family in ("fl2va", "ref2va")
            if any(route.family == family for route in ordered_routes)
        )
        return V4CreativeCompileInput(
            draft_document=draft.model_dump(mode="json"),
            captured_legacy_settings=captured_legacy_settings.model_dump(
                mode="json"
            ),
            requested_segment_ids=(
                None
                if selected_segment_ids is None
                else tuple(selected_segment_ids)
            ),
            selected_segment_ids=tuple(segment.id for segment in selected),
            routes=ordered_routes,
            submission_order=tuple(route.unit_id for route in ordered_routes),
            families=families,
            has_continuity_edges=has_continuity_edges,
            keep_raylight_resident=keep_raylight_resident,
            clear_raylight_vram_after_sampling=(
                clear_raylight_vram_after_sampling
            ),
        )


__all__ = [
    "CreativeCompileInputError",
    "CreativeCompileInputResolver",
    "HistoricalTakeLike",
    "V4CreativeCompileInput",
    "V4LoraResolution",
    "V4ResolvedHistoricalTake",
    "V4ResolvedSegmentRoute",
]
