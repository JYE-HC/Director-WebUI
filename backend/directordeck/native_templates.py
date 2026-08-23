from __future__ import annotations

"""Server-owned MiniMax H3 workflow templates.

The browser submits timeline data, never a ComfyUI prompt.  This module is the
only place that turns that data into API-format ``class_type + inputs`` graphs.
Conditioning, media IO and decode/save are stock ComfyUI nodes.  RayLight is an
approved, narrow substitution for the model/sampler path when a model family
is assigned a multi-GPU topology.
"""

import hashlib
import json
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Any, Literal, Protocol

from .schemas import (
    AssetReference,
    DiffusionModelBinding,
    RuntimeSettings,
    UnifiedFL2VASegment,
    UnifiedRef2VASegment,
    UnifiedTimelineDraft,
    UnifiedTimelineSegment,
)
from .workflow.audit import (
    FeatureAuditTrace,
    GraphAuditError,
    ResolvedNodeEmission,
    build_graph_audit_spec,
    validate_bound_graph,
    validate_graph_audit_spec,
)
from .workflow.contracts import (
    FeatureResolution,
    GraphAuditSpec,
    NodeContractEvidence,
    ResolvedImplementationIdentity,
    NodeContractRegistry,
)
from .workflow.compile_report import (
    CompiledFeatureNotice,
    CompiledFeatureResolution,
)
from .workflow.node_contracts import (
    V4_NODE_CONTRACT_REGISTRY,
    native_expected_module_policy,
    native_provenance_policy,
)
from .workflow.lora_factory import ResolvedLoraAdapter
from .workflow.execution import PreviewSpec, ProgressSpec


ModelFamily = Literal["fl2va", "ref2va"]
ExecutionBackend = Literal["standard", "raylight"]
ContinuitySource = Literal["same_run", "historical_take"]
RaylightAttentionMode = Literal["ck_int8", "torch_flash"]
RaylightXFuserAttention = Literal["COMFY_KITCHEN_INT8", "TORCH_FLASH"]

# Director always assigns RayLight an explicit local logical GPU pool.  Keep
# Comfy-managed CLIP/VAE models on every other logical GPU warm by asking the
# installed RayLight initializer to release driver models only on that pool.
# This is deliberately a fixed template input rather than another user-facing
# setting: the RayLight node's missing-input default remains ``legacy_all`` for
# old third-party workflows, while Director must never silently regain that
# endpoint-global cleanup behavior.
_RAYLIGHT_DRIVER_CLEANUP_POLICY = "ray_devices"
# Director's frozen v4 RayLight templates use comfy-kitchen's INT8 attention
# adapter by default. Stage 8 adds one strict semantic enum for v5 while this
# alias preserves the exact v4 prompt bytes. The semantic value is resolved to
# a reviewed host enum before it can enter either a prompt or actor namespace:
# changing attention backends must never reuse a pool initialized with a
# different kernel contract.
_RAYLIGHT_DEFAULT_ATTENTION_MODE: RaylightAttentionMode = "ck_int8"
_RAYLIGHT_ATTENTION_BY_MODE: dict[
    RaylightAttentionMode, RaylightXFuserAttention
] = {
    "ck_int8": "COMFY_KITCHEN_INT8",
    "torch_flash": "TORCH_FLASH",
}
_RAYLIGHT_XFUSER_ATTENTION: RaylightXFuserAttention = (
    _RAYLIGHT_ATTENTION_BY_MODE[_RAYLIGHT_DEFAULT_ATTENTION_MODE]
)
# Per-worker LRU cap for base models offloaded to CPU RAM on a model switch
# (installed RayLight fork feature). 2 models at ~37GB per GPU on 48GB cards
# keeps both resident in RAM across A<->B switches; raise it only after
# accounting for world_size x model_bytes per cached model.
_RAYLIGHT_RAM_CACHE_MAX_MODELS = 2

# DirectorDeck 0.2.7 gave its maintained RayLight fork independent ComfyUI
# class IDs. Runtime descriptors are Director-owned durable orchestration
# data, so descriptors written by an earlier Director build must be translated
# at this boundary before the current RayKill planner replays their loader
# chain. Keep this migration deliberately limited to the three class types
# that can legally occur in ``loader_subgraph``; unknown nodes remain invalid.
_LEGACY_RAYLIGHT_LOADER_CLASS_TYPE_ALIASES = {
    "RayInitializerAdvanced": "DirectorDeckRayInitializerAdvanced",
    "RayLoraLoader": "DirectorDeckRayLoraLoader",
    "RayUNETLoader": "DirectorDeckRayUNETLoader",
}


class NativeTemplateError(ValueError):
    """The timeline cannot be represented by the validated native templates."""


def resolve_raylight_attention_backend(
    mode: RaylightAttentionMode | str,
    *,
    binding: DiffusionModelBinding | None = None,
) -> RaylightXFuserAttention:
    """Resolve Director's strict RayLight enum to the exact host literal.

    The optional binding check belongs at this common boundary so namespace
    calculation and prompt emission cannot disagree about topology support.
    Comfy-kitchen INT8 has only been reviewed for non-ring execution; selecting
    it with a ring degree above one therefore fails before any actor starts.
    """

    if mode == "ck_int8":
        attention: RaylightXFuserAttention = "COMFY_KITCHEN_INT8"
    elif mode == "torch_flash":
        attention = "TORCH_FLASH"
    else:
        raise NativeTemplateError(
            "RayLight attention mode must be 'ck_int8' or 'torch_flash'"
        )
    if (
        binding is not None
        and mode == "ck_int8"
        and binding.raylight.ring_degree > 1
    ):
        raise NativeTemplateError(
            "RayLight ck_int8 attention requires ring_degree=1"
        )
    return attention


_UNBOUND_PREDECESSOR_OUTPUT = (
    "__DIRECTORDECK_UNBOUND_PREDECESSOR_OUTPUT__.mp4 [output]"
)


@dataclass(frozen=True)
class NativeContinuityDependency:
    """One output edge which must be resolved after its predecessor succeeds."""

    predecessor_segment_id: str
    overlap_frames: int
    load_video_node_id: str
    source: ContinuitySource = "same_run"
    historical_take_id: str | None = None
    resolved: bool = False
    bound_file: str | None = None


@dataclass(frozen=True)
class NativeHistoricalTake:
    """Server-resolved persisted output for one authored predecessor."""

    id: str
    segment_id: str
    output: Mapping[str, Any]


@dataclass(frozen=True)
class NativeFeatureIdentityEvidence:
    """Interpreter-owned identity captured from the same active resolution."""

    feature_id: str
    cache_identity: Any
    runtime_pool_identity: Any | None


@dataclass(frozen=True)
class NativeWorkflowUnit:
    """One independently submitted ComfyUI prompt.

    RayLight owns process-global Ray state, therefore the parent orchestrator
    must submit RayLight units serially and must not overlap them with another
    RayLight topology on the same ComfyUI endpoint.
    """

    id: str
    family: ModelFamily
    backend: ExecutionBackend
    segment_ids: tuple[str, ...]
    prompt: dict[str, Any]
    output_nodes: dict[str, str]
    continuity: NativeContinuityDependency | None = None
    graph_audit_spec: GraphAuditSpec | None = None
    graph_audit_traces: tuple[FeatureAuditTrace, ...] = ()
    compile_feature_resolutions: tuple[CompiledFeatureResolution, ...] = ()
    compile_feature_notices: tuple[CompiledFeatureNotice, ...] = ()
    feature_identity_evidence: tuple[NativeFeatureIdentityEvidence, ...] = ()
    progress_spec: ProgressSpec | None = None
    preview_spec: PreviewSpec | None = None
    raylight_runtime_epoch: int | None = None
    raylight_runtime_namespace: str | None = None


@dataclass(frozen=True)
class NativeCompileResult:
    workflows: tuple[NativeWorkflowUnit, ...]
    manifest: dict[str, Any]
    plans: tuple[dict[str, Any], ...]
    families: tuple[ModelFamily, ...]
    node_policy: dict[str, Any]


_PROVENANCE: dict[str, str] = dict(native_provenance_policy())
EXPECTED_NATIVE_NODE_MODULES: dict[str, str] = dict(
    native_expected_module_policy()
)

_RAYLIGHT_REQUIRED = frozenset(
    {
        "DirectorDeckRayInitializerAdvanced",
        "DirectorDeckRayUNETLoader",
        "DirectorDeckRayMiniMaxH3SigmaShift",
        "DirectorDeckRayBasicGuider",
        "DirectorDeckRayBasicScheduler",
        "DirectorDeckRayXFuserSamplerCustomAdvanced",
    }
)


class _NativeNodeEmitter(Protocol):
    """Ordered node append boundary shared by the built-in interpreters."""

    def add(self, class_type: str, **inputs: Any) -> str: ...


def _edge(node_id: str, output: int = 0) -> list[Any]:
    return [node_id, output]


def _align_h3_frame_count(raw_frames: int) -> int:
    frames = max(5, raw_frames)
    frames += (5 - frames % 17) % 17
    if frames > 512:
        raise NativeTemplateError(
            f"segment compiles to {frames} frames; native H3 template limit is 512"
        )
    return frames


def _align_h3_frames(duration_seconds: float, fps: float) -> int:
    raw = max(5, int(round(duration_seconds * fps)))
    return _align_h3_frame_count(raw)


def _annotated_predecessor_output(output: Mapping[str, Any]) -> str:
    """Return a canonical Comfy ``[output]`` path from history metadata."""

    filename = output.get("filename")
    subfolder = output.get("subfolder", "")
    output_type = output.get("type")
    if output_type != "output":
        raise NativeTemplateError(
            "continuity predecessor must be a persisted ComfyUI output"
        )
    if not isinstance(filename, str) or not filename or len(filename) > 512:
        raise NativeTemplateError("continuity predecessor filename is invalid")
    if not isinstance(subfolder, str) or len(subfolder) > 512:
        raise NativeTemplateError("continuity predecessor subfolder is invalid")
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
        raise NativeTemplateError("continuity predecessor filename is unsafe")
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
        raise NativeTemplateError("continuity predecessor subfolder is unsafe")
    folder = PurePosixPath(subfolder)
    if folder.is_absolute() or any(
        part in {"", ".", ".."} for part in folder.parts
    ):
        raise NativeTemplateError("continuity predecessor subfolder is unsafe")
    relative = PurePosixPath(filename) if not subfolder else folder / filename
    return f"{relative.as_posix()} [output]"


def _bound_late_values(unit: NativeWorkflowUnit) -> dict[str, Any]:
    """Return independent evidence for every dependency already materialized."""

    values: dict[str, Any] = {}
    dependency = unit.continuity
    if (
        dependency is not None
        and dependency.resolved
        and dependency.bound_file is not None
    ):
        values[
            f"/{dependency.load_video_node_id}/inputs/file"
        ] = dependency.bound_file
    if unit.raylight_runtime_namespace is not None:
        runtime_pointers = tuple(
            item.input_pointer
            for item in (
                unit.graph_audit_spec.allowed_late_bound_inputs
                if unit.graph_audit_spec is not None
                else ()
            )
            if item.source_kind == "runtime_epoch"
        )
        if len(runtime_pointers) != 1:
            raise NativeTemplateError(
                f"RayLight workflow '{unit.id}' has invalid bound epoch evidence"
            )
        values[runtime_pointers[0]] = unit.raylight_runtime_namespace
    return values


def bind_native_workflow_predecessor_output(
    unit: NativeWorkflowUnit,
    output: Mapping[str, Any],
) -> NativeWorkflowUnit:
    """Purely bind one successor graph to its predecessor's SaveVideo output.

    The caller must first resolve the unique history output belonging to the
    predecessor unit's declared SaveVideo node. This helper deliberately
    accepts only that single descriptor, never a free-form annotated path.
    """

    dependency = unit.continuity
    if dependency is None:
        raise NativeTemplateError(
            f"native workflow '{unit.id}' has no continuity predecessor"
        )
    if dependency.resolved:
        raise NativeTemplateError(
            f"native workflow '{unit.id}' continuity predecessor is already bound"
        )
    prompt = deepcopy(unit.prompt)
    node = prompt.get(dependency.load_video_node_id)
    if not isinstance(node, dict) or node.get("class_type") != "LoadVideo":
        raise NativeTemplateError(
            f"native workflow '{unit.id}' continuity LoadVideo node is invalid"
        )
    inputs = node.get("inputs")
    if (
        not isinstance(inputs, dict)
        or inputs.get("file") != _UNBOUND_PREDECESSOR_OUTPUT
    ):
        raise NativeTemplateError(
            f"native workflow '{unit.id}' continuity placeholder was modified"
        )
    annotated_output = _annotated_predecessor_output(output)
    inputs["file"] = annotated_output
    if unit.graph_audit_spec is not None:
        pointer = f"/{dependency.load_video_node_id}/inputs/file"
        expected_values = _bound_late_values(unit)
        expected_values[pointer] = annotated_output
        try:
            validate_bound_graph(
                prompt_base=unit.prompt,
                bound_prompt=prompt,
                spec=unit.graph_audit_spec,
                node_contract_registry=V4_NODE_CONTRACT_REGISTRY,
                model_family=unit.family,
                backend=unit.backend,
                feature_traces=unit.graph_audit_traces,
                expected_late_bound_values=expected_values,
                enforce_runtime_effects=False,
            )
        except GraphAuditError as exc:
            raise NativeTemplateError(
                f"native workflow '{unit.id}' continuity binding failed graph audit: {exc}"
            ) from exc
    return replace(
        unit,
        prompt=prompt,
        continuity=replace(
            dependency,
            resolved=True,
            bound_file=annotated_output,
        ),
    )


def normalize_native_output_descriptor(
    output: Mapping[str, Any],
) -> dict[str, str]:
    """Validate and normalize one persistent ComfyUI output descriptor."""

    _annotated_predecessor_output(output)
    return {
        "filename": str(output["filename"]),
        "subfolder": str(output.get("subfolder") or ""),
        "type": "output",
    }


def validate_native_workflow_runtime_effects(
    unit: NativeWorkflowUnit,
    *,
    node_contract_registry: NodeContractRegistry = V4_NODE_CONTRACT_REGISTRY,
) -> None:
    """Reject a compiled graph whose output path is not fail-closed.

    This gate intentionally does not require continuity or runtime-epoch late
    binding.  Submission endpoints can therefore run it before creating any
    durable job rows, while :func:`validate_native_workflow_ready` reruns the
    same audit against the exact materialized prompt immediately before POST.
    """

    if unit.graph_audit_spec is None or not unit.graph_audit_traces:
        raise NativeTemplateError(
            f"native workflow '{unit.id}' has no submission-ready graph audit"
        )
    try:
        validate_graph_audit_spec(
            prompt=unit.prompt,
            spec=unit.graph_audit_spec,
            node_contract_registry=node_contract_registry,
            model_family=unit.family,
            backend=unit.backend,
            feature_traces=unit.graph_audit_traces,
        )
    except GraphAuditError as exc:
        raise NativeTemplateError(
            f"native workflow '{unit.id}' failed submission graph audit: {exc}"
        ) from exc


def validate_native_workflow_ready(unit: NativeWorkflowUnit) -> None:
    """Fail closed unless every dynamic predecessor input has been bound."""

    placeholder_nodes = [
        str(node_id)
        for node_id, node in unit.prompt.items()
        if isinstance(node, dict)
        and node.get("class_type") == "LoadVideo"
        and isinstance(node.get("inputs"), dict)
        and node["inputs"].get("file") == _UNBOUND_PREDECESSOR_OUTPUT
    ]
    dependency = unit.continuity
    if dependency is None:
        if placeholder_nodes:
            raise NativeTemplateError(
                f"native workflow '{unit.id}' contains an undeclared continuity input"
            )
    else:
        if not dependency.resolved or dependency.bound_file is None:
            raise NativeTemplateError(
                f"native workflow '{unit.id}' is waiting for predecessor segment "
                f"'{dependency.predecessor_segment_id}'"
            )
        node = unit.prompt.get(dependency.load_video_node_id)
        if (
            not isinstance(node, dict)
            or node.get("class_type") != "LoadVideo"
            or not isinstance(node.get("inputs"), dict)
            or node["inputs"].get("file") != dependency.bound_file
            or placeholder_nodes
        ):
            raise NativeTemplateError(
                f"native workflow '{unit.id}' predecessor input differs from its "
                "exact bound output evidence"
            )

    runtime_epoch_pointers = tuple(
        item.input_pointer
        for item in (
            unit.graph_audit_spec.allowed_late_bound_inputs
            if unit.graph_audit_spec is not None
            else ()
        )
        if item.source_kind == "runtime_epoch"
    )
    if runtime_epoch_pointers:
        if len(runtime_epoch_pointers) != 1 or unit.backend != "raylight":
            raise NativeTemplateError(
                f"native workflow '{unit.id}' has an invalid runtime-epoch dependency"
            )
        descriptor = raylight_runtime_descriptor(unit)
        if descriptor is None:
            raise NativeTemplateError(
                f"RayLight workflow '{unit.id}' has no runtime epoch descriptor"
            )
        compatibility_key = str(descriptor["compatibility_key"])
        runtime_namespace = str(descriptor["runtime_namespace"])
        expected_epoch = unit.raylight_runtime_epoch
        expected_namespace = unit.raylight_runtime_namespace
        if (
            not isinstance(expected_epoch, int)
            or isinstance(expected_epoch, bool)
            or expected_epoch < 1
            or not isinstance(expected_namespace, str)
            or not expected_namespace.startswith("director-")
            or expected_namespace != f"{compatibility_key}-e{expected_epoch}"
            or runtime_namespace != expected_namespace
        ):
            raise NativeTemplateError(
                f"RayLight workflow '{unit.id}' differs from its exact bound "
                "runtime epoch evidence"
            )

    validate_native_workflow_runtime_effects(unit)


def _asset_path(asset: AssetReference) -> str:
    return asset.comfy_path


def _require_h3_video_asset(asset: AssetReference, *, usage: str) -> None:
    metadata = asset.metadata
    if metadata is None:
        raise NativeTemplateError(f"{usage} requires server-probed video metadata")
    if abs(metadata.native_fps - 24.0) > 0.01:
        raise NativeTemplateError(
            f"{usage} '{asset.name}' is {metadata.native_fps:.6g} fps; "
            "MiniMax H3 reference frames require a server-created 24 fps proxy"
        )
    if metadata.frame_count < 5:
        raise NativeTemplateError(
            f"{usage} '{asset.name}' contains {metadata.frame_count} frame(s); "
            "MiniMax H3 reference video requires at least 5 frames"
        )


def _require_h3_source_range(
    asset: AssetReference, *, start: float, duration: float, usage: str
) -> None:
    """Reject trims that stock ReferenceToVideo cannot condition on."""

    _require_h3_video_asset(asset, usage=usage)
    assert asset.metadata is not None
    full_frames = max(1, int(round(asset.metadata.duration * 24.0)))
    source_start = min(full_frames - 1, max(0, int(round(start * 24.0))))
    source_end = min(
        full_frames,
        max(source_start + 1, int(round((start + duration) * 24.0))),
    )
    selected_frames = source_end - source_start
    if selected_frames < 5:
        raise NativeTemplateError(
            f"{usage} '{asset.name}' selects {selected_frames} frame(s); "
            "MiniMax H3 reference video requires at least 5 frames"
        )


def resolve_execution_backend(binding: DiffusionModelBinding) -> ExecutionBackend:
    """Derive the sole execution route from the configured logical GPU pool.

    The legacy ``binding.backend`` value is intentionally ignored.  Keeping a
    hidden explicit value after removing its UI would let an old browser or
    database silently override the visible GPU selection.  Capability probing
    still verifies the exact derived graph before submission and never falls
    back to another backend when a required node is missing.
    """

    return "raylight" if len(binding.raylight.gpu_select) >= 2 else "standard"


def required_nodes_for_backend(
    backend: ExecutionBackend, *, has_lora: bool = False
) -> frozenset[str]:
    nodes = set(_RAYLIGHT_REQUIRED if backend == "raylight" else ())
    if backend == "raylight" and has_lora:
        nodes.add("DirectorDeckRayLoraLoader")
    return frozenset(nodes)


def _shared_core(
    graph: _NativeNodeEmitter,
    settings: RuntimeSettings,
) -> dict[str, list[Any]]:
    clip_loader = graph.add(
        "CLIPLoader",
        clip_name=settings.models.clip.filename,
        type="minimax",
        device="default",
    )
    clip = graph.add(
        "SelectCLIPDevice",
        clip=_edge(clip_loader),
        device=settings.models.clip.device,
    )
    video_vae_loader = graph.add(
        "VAELoader", vae_name=settings.models.video_vae.filename
    )
    video_vae = graph.add(
        "SelectVAEDevice",
        vae=_edge(video_vae_loader),
        device=settings.models.video_vae.device,
    )
    audio_vae_loader = graph.add(
        "VAELoader", vae_name=settings.models.audio_vae.filename
    )
    audio_vae = graph.add(
        "SelectVAEDevice",
        vae=_edge(audio_vae_loader),
        device=settings.models.audio_vae.device,
    )
    return {
        "clip": _edge(clip),
        "video_vae": _edge(video_vae),
        "audio_vae": _edge(audio_vae),
    }


def raylight_runtime_namespace(
    binding: DiffusionModelBinding,
    *,
    attention_mode: RaylightAttentionMode = _RAYLIGHT_DEFAULT_ATTENTION_MODE,
    enforce_attention_topology: bool = True,
) -> str:
    """Return a stable actor namespace for one compatible resident pool.

    A job identifier must not participate in this key: doing so would force
    ``DirectorDeckRayInitializerAdvanced`` to tear down and recreate the same actors for
    every take.  Only the GPU pool and topology participate.  Model file, LoRA
    and family deliberately do NOT participate: the installed worker RAM cache
    moves the outgoing model to CPU RAM on a key change and re-activates it
    from RAM when the key comes back, so a model switch keeps the pool alive
    and never re-reads the checkpoint.  The installed initializer still calls
    ``ray.shutdown()`` whenever this input changes (topology change), giving
    an explicit old-pool teardown for a genuinely incompatible pool.
    """

    profile = binding.raylight
    attention = resolve_raylight_attention_backend(
        attention_mode,
        binding=binding if enforce_attention_topology else None,
    )
    gpu_pool = "-".join(str(index) for index in profile.gpu_select)
    compatibility_document = {
        "backend": "raylight",
        "weight_dtype": "default",
        "gpu_select": profile.gpu_select,
        "driver_cleanup_policy": _RAYLIGHT_DRIVER_CLEANUP_POLICY,
        "ulysses_degree": profile.ulysses_degree,
        "ring_degree": profile.ring_degree,
        "cfg_degree": profile.cfg_degree,
        "dp_degree": profile.dp_degree,
        "fsdp": profile.fsdp,
        "cpu_offload": profile.cpu_offload,
        "attention": attention,
        "use_mmap": False,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            compatibility_document,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:20]
    return (
        f"director-g{gpu_pool}"
        f"-u{profile.ulysses_degree}-r{profile.ring_degree}"
        f"-c{profile.cfg_degree}-d{profile.dp_degree}"
        f"-f{int(profile.fsdp)}-o{int(profile.cpu_offload)}"
        f"-k{fingerprint}"
    )


def _raylight_namespace(
    family: ModelFamily,
    binding: DiffusionModelBinding,
    *,
    attention_mode: RaylightAttentionMode = _RAYLIGHT_DEFAULT_ATTENTION_MODE,
) -> str:
    """Compatibility wrapper retaining the frozen v4 compiler signature.

    ``family`` deliberately does not participate in resident-pool identity;
    the same actors can swap model families through the worker RAM cache.
    """

    del family
    return raylight_runtime_namespace(
        binding,
        attention_mode=attention_mode,
        # This wrapper is part of the frozen v4 compiler boundary.  V4
        # historically emitted ck-int8 with ring parallelism, so the new v5
        # semantic-mode guard must not retroactively reject that graph.
        enforce_attention_topology=False,
    )


def raylight_runtime_descriptor(unit: NativeWorkflowUnit) -> dict[str, Any] | None:
    """Return the exact cached loader chain needed for a later kill.

    ``DirectorDeckRayKill`` consumes ``RAY_ACTORS`` while ``DirectorDeckRayInitializerAdvanced`` only
    produces ``RAY_ACTORS_INIT``.  Persisting just the initializer would make
    the transition graph type-invalid.  The descriptor therefore owns the
    complete, minimal initializer -> optional LoRA -> UNET-loader subgraph.
    Reusing the original node ids and inputs lets ComfyUI return the live
    loader output from cache; if that cache is gone, the same subgraph safely
    reconstructs a pool which the barrier immediately tears down.
    """

    matches = [
        (node_id, node)
        for node_id, node in unit.prompt.items()
        if node.get("class_type") == "DirectorDeckRayInitializerAdvanced"
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise NativeTemplateError(
            f"RayLight unit '{unit.id}' must contain exactly one initializer"
        )
    node_id, node = matches[0]
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise NativeTemplateError(
            f"RayLight unit '{unit.id}' has invalid initializer inputs"
        )
    runtime_namespace = str(inputs.get("ray_cluster_namespace") or "")
    if not runtime_namespace:
        raise NativeTemplateError(
            f"RayLight unit '{unit.id}' has an empty runtime namespace"
        )
    compatibility_key = re.sub(r"-e[1-9][0-9]*$", "", runtime_namespace)

    loader_matches = [
        (candidate_id, candidate)
        for candidate_id, candidate in unit.prompt.items()
        if candidate.get("class_type") == "DirectorDeckRayUNETLoader"
    ]
    if len(loader_matches) != 1:
        raise NativeTemplateError(
            f"RayLight unit '{unit.id}' must contain exactly one UNET loader"
        )
    loader_node_id, loader_node = loader_matches[0]
    loader_inputs = loader_node.get("inputs")
    if not isinstance(loader_inputs, dict):
        raise NativeTemplateError(
            f"RayLight unit '{unit.id}' has invalid UNET loader inputs"
        )
    if loader_inputs.get("ray_actors_init") != _edge(str(node_id)):
        raise NativeTemplateError(
            f"RayLight unit '{unit.id}' loader is not bound to its initializer"
        )

    dependency_ids = {str(node_id), str(loader_node_id)}
    lora_edge = loader_inputs.get("lora")
    if lora_edge is not None:
        if (
            not isinstance(lora_edge, list)
            or len(lora_edge) != 2
            or not isinstance(lora_edge[0], str)
        ):
            raise NativeTemplateError(
                f"RayLight unit '{unit.id}' has an invalid LoRA edge"
            )
        lora_node = unit.prompt.get(lora_edge[0])
        if not isinstance(lora_node, dict) or lora_node.get("class_type") != "DirectorDeckRayLoraLoader":
            raise NativeTemplateError(
                f"RayLight unit '{unit.id}' loader has an invalid LoRA dependency"
            )
        dependency_ids.add(lora_edge[0])

    loader_subgraph = {
        dependency_id: deepcopy(unit.prompt[dependency_id])
        for dependency_id in dependency_ids
    }
    # The runtime key deliberately excludes the loader inputs (model file and
    # LoRA): the workers own a per-worker RAM cache and swap bases in place,
    # so a model switch keeps the pool and its epoch.  The key still crosses
    # the DirectorDeckRayKill/epoch boundary on namespace (topology) changes and on sigma
    # mutations, which the workers cannot absorb safely.
    # DirectorDeckRayMiniMaxH3SigmaShift mutates each worker's ModelPatcher in place. Its
    # cached output is therefore valid only for the exact shift pair: in
    # A(12) -> B(8) -> A(12), ComfyUI could otherwise return A's old output
    # while the shared actors still hold B's mutation. Treat a shift change as
    # an incompatible runtime key and cross the normal DirectorDeckRayKill/epoch boundary.
    sigma_matches = [
        candidate
        for candidate in unit.prompt.values()
        if candidate.get("class_type") == "DirectorDeckRayMiniMaxH3SigmaShift"
    ]
    # Shutdown units intentionally contain only the loader chain plus DirectorDeckRayKill;
    # their descriptor was already persisted from a complete generation unit.
    is_shutdown_unit = any(
        candidate.get("class_type") == "DirectorDeckRayKill"
        for candidate in unit.prompt.values()
    )
    if is_shutdown_unit:
        return None
    if len(sigma_matches) != 1 or not isinstance(sigma_matches[0].get("inputs"), dict):
        raise NativeTemplateError(
            f"RayLight unit '{unit.id}' must contain exactly one sigma-shift node"
        )
    sigma_inputs = sigma_matches[0]["inputs"]
    # Pool-level inputs of the initializer participate (minus the namespace,
    # which is the canonical compatibility_key). This deliberately includes
    # XFuser_attention: a legacy TORCH_FLASH pool must cross DirectorDeckRayKill before a
    # COMFY_KITCHEN_INT8 prompt can start. The model file and LoRA live in the
    # loader node and deliberately do not, so a model switch keeps the same
    # runtime key and reuses the pool via the worker RAM cache.
    runtime_identity = {
        "__initializer_inputs__": {
            key: value
            for key, value in inputs.items()
            if key != "ray_cluster_namespace"
        },
        "__runtime_mutations__": {
            "DirectorDeckRayMiniMaxH3SigmaShift": {
                "shift_video": sigma_inputs.get("shift_video"),
                "shift_audio": sigma_inputs.get("shift_audio"),
            }
        },
    }
    runtime_key = hashlib.sha256(
        json.dumps(
            runtime_identity,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "version": 2,
        "family": unit.family,
        "compatibility_key": compatibility_key,
        "runtime_key": runtime_key,
        "runtime_namespace": runtime_namespace,
        "initializer_node_id": str(node_id),
        "loader_node_id": str(loader_node_id),
        "loader_subgraph": loader_subgraph,
        "clear_vram_after_sampling": bool(
            inputs.get("clear_vram_after_sampling", False)
        ),
    }


def _raylight_logical_gpu_indices(inputs: Mapping[str, Any]) -> tuple[int, ...]:
    """Parse Director's explicit logical GPU pool from one initializer.

    RayLight interprets ``GPU_SELECT`` inside the ComfyUI process CUDA
    namespace.  Persisted shutdown descriptors must keep that exact identity;
    silently dropping, remapping or deduplicating entries could target a
    different actor pool during a safety transition.
    """

    raw = inputs.get("GPU_SELECT")
    if not isinstance(raw, str) or not raw:
        raise NativeTemplateError(
            "persisted RayLight initializer GPU_SELECT is invalid"
        )
    indices: list[int] = []
    for token in raw.split(","):
        if not token or not token.isdecimal():
            raise NativeTemplateError(
                "persisted RayLight initializer GPU_SELECT is invalid"
            )
        index = int(token)
        if index in indices:
            raise NativeTemplateError(
                "persisted RayLight initializer GPU_SELECT contains duplicates"
            )
        indices.append(index)
    declared = inputs.get("GPU")
    if (
        not isinstance(declared, int)
        or isinstance(declared, bool)
        or declared != len(indices)
    ):
        raise NativeTemplateError(
            "persisted RayLight initializer GPU count does not match GPU_SELECT"
        )
    return tuple(indices)


def migrate_legacy_raylight_runtime_descriptor(
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate only Director's former RayLight loader class IDs.

    The persisted descriptor is not a user workflow: Director created it from
    an already-compiled RayLight unit so a later Standard/model/topology switch
    can enqueue an exact cleanup barrier. Renaming the maintained fork's class
    IDs must therefore migrate that internal ledger instead of turning it into
    a permanent runtime authorization failure.
    """

    migrated = deepcopy(dict(descriptor))
    if migrated.get("version") != 2:
        return migrated
    loader_subgraph = migrated.get("loader_subgraph")
    if not isinstance(loader_subgraph, dict):
        return migrated
    for node in loader_subgraph.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        replacement = (
            _LEGACY_RAYLIGHT_LOADER_CLASS_TYPE_ALIASES.get(class_type)
            if isinstance(class_type, str)
            else None
        )
        if replacement is not None:
            node["class_type"] = replacement
    return migrated


def raylight_runtime_logical_gpu_indices(
    descriptor: Mapping[str, Any],
) -> tuple[int, ...]:
    """Return the exact logical GPU pool recorded in a runtime descriptor."""

    initializer_node_id = descriptor.get("initializer_node_id")
    loader_subgraph = descriptor.get("loader_subgraph")
    if (
        descriptor.get("version") != 2
        or not isinstance(initializer_node_id, str)
        or not isinstance(loader_subgraph, Mapping)
    ):
        raise NativeTemplateError("persisted RayLight loader identity is invalid")
    initializer = loader_subgraph.get(initializer_node_id)
    if (
        not isinstance(initializer, Mapping)
        or initializer.get("class_type") != "DirectorDeckRayInitializerAdvanced"
        or not isinstance(initializer.get("inputs"), Mapping)
    ):
        raise NativeTemplateError("persisted RayLight initializer is invalid")
    return _raylight_logical_gpu_indices(initializer["inputs"])


def raylight_workflow_logical_gpu_indices(
    unit: NativeWorkflowUnit,
) -> tuple[int, ...]:
    """Return the explicit logical GPU pool in one RayLight workflow unit."""

    initializers = [
        node
        for node in unit.prompt.values()
        if isinstance(node, Mapping)
        and node.get("class_type") == "DirectorDeckRayInitializerAdvanced"
    ]
    if len(initializers) != 1 or not isinstance(initializers[0].get("inputs"), Mapping):
        raise NativeTemplateError(
            f"RayLight unit '{unit.id}' must contain one valid initializer"
        )
    return _raylight_logical_gpu_indices(initializers[0]["inputs"])


def bind_raylight_runtime_epoch(
    unit: NativeWorkflowUnit, epoch: int
) -> NativeWorkflowUnit:
    """Bind a Ray unit to one persistent endpoint transition epoch."""

    if unit.backend != "raylight" or epoch < 1:
        raise NativeTemplateError("RayLight runtime epoch must be a positive integer")
    descriptor = raylight_runtime_descriptor(unit)
    if descriptor is None:
        raise NativeTemplateError(
            f"RayLight unit '{unit.id}' has no initializer to bind"
        )
    compatibility_key = str(descriptor["compatibility_key"])
    if not compatibility_key.startswith("director-"):
        raise NativeTemplateError(
            f"RayLight workflow '{unit.id}' has a non-Director runtime namespace"
        )
    prompt = deepcopy(unit.prompt)
    initializer_node_id = str(descriptor["initializer_node_id"])
    initializer = prompt[initializer_node_id]
    runtime_namespace = f"{compatibility_key}-e{epoch}"
    initializer["inputs"]["ray_cluster_namespace"] = runtime_namespace
    if unit.graph_audit_spec is not None:
        pointer = f"/{initializer_node_id}/inputs/ray_cluster_namespace"
        expected_values = _bound_late_values(unit)
        expected_values[pointer] = runtime_namespace
        try:
            validate_bound_graph(
                prompt_base=unit.prompt,
                bound_prompt=prompt,
                spec=unit.graph_audit_spec,
                node_contract_registry=V4_NODE_CONTRACT_REGISTRY,
                model_family=unit.family,
                backend=unit.backend,
                feature_traces=unit.graph_audit_traces,
                expected_late_bound_values=expected_values,
                enforce_runtime_effects=False,
            )
        except GraphAuditError as exc:
            raise NativeTemplateError(
                f"RayLight workflow '{unit.id}' epoch binding failed graph audit: {exc}"
            ) from exc
    return replace(
        unit,
        prompt=prompt,
        raylight_runtime_epoch=epoch,
        raylight_runtime_namespace=runtime_namespace,
    )


def _release_node_contract_snapshot(
    prompt: Mapping[str, Any],
) -> dict[str, NodeContractEvidence]:
    snapshot: dict[str, NodeContractEvidence] = {}
    for node_id, node in prompt.items():
        class_type = str(node.get("class_type") or "")
        contract = V4_NODE_CONTRACT_REGISTRY.require(class_type)
        if len(contract.allowed_python_modules) != 1:
            raise NativeTemplateError(
                f"native node contract must select one module: {class_type}"
            )
        snapshot[str(node_id)] = NodeContractEvidence(
            contract_id=contract.contract_id,
            semantic_version=contract.semantic_version,
            class_type=contract.class_type,
            python_module=contract.allowed_python_modules[0],
            runtime_fingerprint=contract.supported_runtime_fingerprints[0],
            execution_terminal_role=contract.execution_terminal_role,
            persistent_artifact_role=contract.persistent_artifact_role,
        )
    return snapshot


def _raylight_control_audit(
    prompt: Mapping[str, Any],
    *,
    family: ModelFamily,
) -> tuple[GraphAuditSpec, tuple[FeatureAuditTrace, ...]]:
    ordered_classes = tuple(
        dict.fromkeys(str(node["class_type"]) for node in prompt.values())
    )
    implementations: list[ResolvedImplementationIdentity] = []
    binding_by_class: dict[str, str] = {}
    for class_type in ordered_classes:
        contract = V4_NODE_CONTRACT_REGISTRY.require(class_type)
        binding_key = "raylight_kill_control." + re.sub(
            r"[^A-Za-z0-9_.:-]", "_", class_type
        )
        binding_by_class[class_type] = binding_key
        implementations.append(
            ResolvedImplementationIdentity(
                role="node",
                class_type=class_type,
                implementation_id=contract.contract_id,
                semantic_version=contract.semantic_version,
                runtime_fingerprint=contract.supported_runtime_fingerprints[0],
                binding_key=binding_key,
            )
        )
    resolution = FeatureResolution(
        state="active",
        implementations=tuple(implementations),
        resolution_details={"source": "persisted_raylight_loader_descriptor"},
    )
    trace = FeatureAuditTrace(
        feature_id="raylight_kill_control",
        resolution=resolution,
        emitted_nodes=tuple(
            ResolvedNodeEmission(
                node_id=str(node_id),
                implementation_binding_key=binding_by_class[str(node["class_type"])],
                output_affecting=False,
            )
            for node_id, node in prompt.items()
        ),
        structural_influence=False,
    )
    spec = build_graph_audit_spec(
        prompt=prompt,
        node_contract_registry=V4_NODE_CONTRACT_REGISTRY,
        node_contract_snapshot=_release_node_contract_snapshot(prompt),
        public_writes=(),
        public_reads=(),
        feature_traces=(trace,),
        model_family=family,
        backend="raylight",
        unit_kind="control",
        control_kind="ray_kill",
    )
    return spec, (trace,)


def build_raylight_shutdown_unit(
    descriptor: dict[str, Any], *, unit_id: str
) -> NativeWorkflowUnit:
    """Build a forced DirectorDeckRayKill barrier from a persisted resident descriptor.

    Reusing the original loader subgraph normally retrieves live ``RAY_ACTORS``
    from ComfyUI's cache. If that cache was cleared, running the initializer is
    still safe: RayLight first shuts down any current pool, creates a temporary
    compatible pool, loads the recorded model, and this prompt immediately
    kills it. The unique kill node id prevents ComfyUI from caching away the
    side effect.
    """

    if descriptor.get("version") != 2:
        raise NativeTemplateError("unsupported persisted RayLight runtime descriptor")
    family = descriptor.get("family")
    if family not in {"fl2va", "ref2va"}:
        raise NativeTemplateError("persisted RayLight runtime family is invalid")
    initializer_node_id = descriptor.get("initializer_node_id")
    loader_node_id = descriptor.get("loader_node_id")
    loader_subgraph = descriptor.get("loader_subgraph")
    if (
        not isinstance(initializer_node_id, str)
        or not initializer_node_id
        or not isinstance(loader_node_id, str)
        or not loader_node_id
        or not isinstance(loader_subgraph, dict)
    ):
        raise NativeTemplateError("persisted RayLight loader identity is invalid")
    node_types = {
        str(node.get("class_type") or "")
        for node in loader_subgraph.values()
        if isinstance(node, dict)
    }
    if (
        set(loader_subgraph) != {
            key for key in loader_subgraph if isinstance(key, str)
        }
        or initializer_node_id not in loader_subgraph
        or loader_node_id not in loader_subgraph
        or not node_types <= {
            "DirectorDeckRayInitializerAdvanced",
            "DirectorDeckRayLoraLoader",
            "DirectorDeckRayUNETLoader",
        }
        or "DirectorDeckRayInitializerAdvanced" not in node_types
        or "DirectorDeckRayUNETLoader" not in node_types
    ):
        raise NativeTemplateError("persisted RayLight loader subgraph is invalid")
    initializer_node = loader_subgraph[initializer_node_id]
    loader_node = loader_subgraph[loader_node_id]
    if (
        not isinstance(initializer_node, dict)
        or initializer_node.get("class_type") != "DirectorDeckRayInitializerAdvanced"
        or not isinstance(initializer_node.get("inputs"), dict)
        or not isinstance(loader_node, dict)
        or loader_node.get("class_type") != "DirectorDeckRayUNETLoader"
        or not isinstance(loader_node.get("inputs"), dict)
        or loader_node["inputs"].get("ray_actors_init")
        != _edge(initializer_node_id)
    ):
        raise NativeTemplateError("persisted RayLight loader chain is invalid")
    kill_node_id = f"ray-kill-{unit_id}"
    if kill_node_id in loader_subgraph:
        raise NativeTemplateError("RayLight shutdown node id collides with loader graph")
    prompt = deepcopy(loader_subgraph)
    prompt[kill_node_id] = {
        "class_type": "DirectorDeckRayKill",
        "inputs": {
            "ray_actors": _edge(loader_node_id),
            "kill_mode": "Kill Entire Cluster",
        },
    }
    graph_audit_spec, graph_audit_traces = _raylight_control_audit(
        prompt,
        family=family,
    )
    return NativeWorkflowUnit(
        id=unit_id,
        family=family,
        backend="raylight",
        segment_ids=(),
        prompt=prompt,
        output_nodes={},
        graph_audit_spec=graph_audit_spec,
        graph_audit_traces=graph_audit_traces,
    )


def _load_image(graph: _NativeNodeEmitter, asset: AssetReference) -> list[Any]:
    return _edge(graph.add("LoadImage", image=_asset_path(asset)))


def _load_video_components(
    graph: _NativeNodeEmitter,
    asset: AssetReference,
    *,
    start: float | None = None,
    duration: float | None = None,
) -> tuple[list[Any], list[Any]]:
    if start is not None and duration is not None:
        _require_h3_source_range(
            asset,
            start=start,
            duration=duration,
            usage="native source-video conditioning",
        )
    else:
        _require_h3_video_asset(asset, usage="native video conditioning")
    video = graph.add("LoadVideo", file=_asset_path(asset))
    current = _edge(video)
    if start is not None and duration is not None:
        sliced = graph.add(
            "Video Slice",
            video=current,
            start_time=start,
            duration=duration,
            strict_duration=True,
        )
        current = _edge(sliced)
    components = graph.add("GetVideoComponents", video=current)
    return _edge(components, 0), _edge(components, 1)


def _load_audio(graph: _NativeNodeEmitter, asset: AssetReference) -> list[Any]:
    return _edge(graph.add("LoadAudio", audio=_asset_path(asset)))


def _require_dense_slots(values: list[Any], *, field: str, segment_id: str) -> None:
    """Stock Autogrow assigns dense ordinals; never silently renumber prompts."""

    slots = sorted(value.slot for value in values)
    expected = list(range(len(values)))
    if slots != expected:
        raise NativeTemplateError(
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


def _conditioning(
    graph: _NativeNodeEmitter,
    segment: UnifiedTimelineSegment,
    draft: UnifiedTimelineDraft,
    shared: dict[str, list[Any]],
    *,
    frames: int,
) -> tuple[list[Any], list[Any], list[Any] | None]:
    prompt = segment.prompt.strip()
    source_audio: list[Any] | None = None
    common: dict[str, Any] = {
        "clip": shared["clip"],
        "vae": shared["video_vae"],
        "prompt": prompt,
        "width": draft.render.width,
        "height": draft.render.height,
        "length": frames,
    }
    if isinstance(segment, UnifiedFL2VASegment):
        if segment.first_image is not None:
            common["first_frame"] = _load_image(graph, segment.first_image)
        if segment.last_image is not None:
            common["last_frame"] = _load_image(graph, segment.last_image)
        node = graph.add("MiniMaxH3ImageToVideo", **common)
        return _edge(node, 0), _edge(node, 1), None

    common.update(
        audio_vae=shared["audio_vae"], ref_image_size=segment.ref_image_size
    )
    if not isinstance(segment, UnifiedRef2VASegment):
        raise NativeTemplateError(f"unsupported segment mode: {segment.mode}")
    if segment.source_video is None and not (
        segment.reference_images
        or segment.reference_audios
        or segment.reference_videos
    ):
        raise NativeTemplateError(
            f"Ref2VA segment '{segment.id}' requires source_video or independent "
            "reference media"
        )
    video_offset = 0
    if segment.source_video is not None:
        images, source_audio = _load_video_components(
            graph,
            segment.source_video,
            start=segment.source_start_seconds,
            duration=segment.source_duration_seconds,
        )
        common["ref_videos.ref_video_0"] = images
        video_offset = 1
        if segment.source_audio_as_reference:
            assert segment.source_video.metadata is not None
            if not segment.source_video.metadata.has_audio:
                raise NativeTemplateError(
                    f"Ref2VA segment '{segment.id}' enables "
                    "source_audio_as_reference, but its server-probed source "
                    "video has no audio stream"
                )
            # Stock MiniMaxH3ReferenceToVideo pairs this soundtrack with the
            # first reference video and assigns it <Audio 1> before any
            # independent ref_audios autogrow inputs.
            common["ref_video_audios.ref_video_audio_0"] = source_audio
    for dense, asset in enumerate(
        sorted(segment.reference_images, key=lambda item: item.slot)
    ):
        common[f"ref_images.ref_image_{dense}"] = _load_image(graph, asset)
    for dense, asset in enumerate(
        sorted(segment.reference_videos, key=lambda item: item.slot),
        start=video_offset,
    ):
        images, _audio = _load_video_components(graph, asset)
        common[f"ref_videos.ref_video_{dense}"] = images
    for dense, asset in enumerate(
        sorted(segment.reference_audios, key=lambda item: item.slot)
    ):
        common[f"ref_audios.ref_audio_{dense}"] = _load_audio(graph, asset)
    node = graph.add("MiniMaxH3ReferenceToVideo", **common)
    return _edge(node, 0), _edge(node, 1), source_audio


def _add_continuity_guides(
    graph: _NativeNodeEmitter,
    *,
    conditioning: list[Any],
    latent: list[Any],
    segment: UnifiedTimelineSegment,
    draft: UnifiedTimelineDraft,
    shared: dict[str, list[Any]],
    visible_frames: int,
    overlap_frames: int,
) -> tuple[list[Any], str]:
    """Anchor the predecessor tail at frame zero of the successor sample."""

    predecessor_video = graph.add(
        "LoadVideo", file=_UNBOUND_PREDECESSOR_OUTPUT
    )
    components = graph.add(
        "GetVideoComponents", video=_edge(predecessor_video)
    )
    tail_images = graph.add(
        "ImageFromBatch",
        image=_edge(components, 0),
        batch_index=-overlap_frames,
        length=overlap_frames,
    )
    guide_inputs: dict[str, Any] = {
        "positive": conditioning,
        "latent": latent,
        "vae": shared["video_vae"],
        "image": _edge(tail_images),
        "frame_idx": 0,
    }
    if segment.audio_mode == "generate":
        tail_audio = graph.add(
            "TrimAudioDuration",
            audio=_edge(components, 1),
            start_index=-(overlap_frames / draft.render.fps),
            duration=overlap_frames / draft.render.fps,
        )
        guide_inputs.update(
            audio_vae=shared["audio_vae"],
            audio=_edge(tail_audio),
        )
    conditioning = _edge(graph.add("MiniMaxH3AddGuide", **guide_inputs))

    # ImageToVideo keeps last_frame in the Qwen presentation so FL2VA prompts
    # can resolve its <Picture N> label. A continuity sample also contains a
    # hidden aligned tail, so repeat the same image as a guide at the final
    # visible output frame; the node's implicit sample-end anchor is cropped.
    if isinstance(segment, UnifiedFL2VASegment) and segment.last_image is not None:
        conditioning = _edge(
            graph.add(
                "MiniMaxH3AddGuide",
                positive=conditioning,
                latent=latent,
                vae=shared["video_vae"],
                image=_load_image(graph, segment.last_image),
                frame_idx=overlap_frames + visible_frames - 1,
            )
        )
    return conditioning, predecessor_video


def compile_native_timeline(
    draft: UnifiedTimelineDraft,
    settings: RuntimeSettings,
    job_id: str,
    segment_ids: list[str] | None = None,
    *,
    historical_takes: Mapping[str, NativeHistoricalTake] | None = None,
    resolved_lora_adapters: Mapping[
        ModelFamily, ResolvedLoraAdapter
    ] | None = None,
) -> NativeCompileResult:
    """Compile v4 timeline data through the validated feature templates.

    The local import is intentional: the v4 compiler reuses the stable native
    result and Ray lifecycle contracts defined above, while this module remains
    the public compatibility entry point.
    """

    from .workflow.v4_compiler import compile_v4_timeline

    return compile_v4_timeline(
        draft,
        settings,
        job_id,
        segment_ids,
        historical_takes=historical_takes,
        resolved_lora_adapters=resolved_lora_adapters,
    )


def validate_native_capabilities(
    result: NativeCompileResult,
    available_nodes: list[str] | set[str],
    node_provenance: dict[str, str] | None = None,
) -> None:
    """Fail before submission only when a required class_type is absent.

    ``node_provenance`` remains in the call contract for compatibility, but is
    advisory and deliberately ignored. ComfyUI decides whether a present node
    implementation can execute the emitted prompt.
    """

    available = set(available_nodes)
    required = set(result.node_policy["allowed_nodes"])
    missing = sorted(required - available)
    if missing:
        raise NativeTemplateError(
            "ComfyUI is missing nodes required by the selected server template: "
            + ", ".join(missing)
        )
    _ = node_provenance
    forbidden = {
        node["class_type"]
        for unit in result.workflows
        for node in unit.prompt.values()
        if node["class_type"] == "MiniMaxH3Director"
    }
    if forbidden:
        raise AssertionError("native template must never emit MiniMaxH3Director")


def validate_native_workflow_unit_capabilities(
    unit: NativeWorkflowUnit,
    available_nodes: list[str] | set[str],
    node_provenance: dict[str, str] | None = None,
) -> None:
    """Require the internal barrier's class_types to exist on the host."""

    node_names = sorted(
        {str(node.get("class_type") or "") for node in unit.prompt.values()}
    )
    unknown = sorted(set(node_names) - set(_PROVENANCE))
    if unknown:
        raise NativeTemplateError(
            "native transition emitted unclassified nodes: " + ", ".join(unknown)
        )
    synthetic = NativeCompileResult(
        workflows=(unit,),
        manifest={},
        plans=(),
        families=(unit.family,),
        node_policy={
            "graph_source": "server",
            "accepts_client_workflow": False,
            "allowed_nodes": node_names,
            "custom_nodes": node_names,
            "provenance": {name: _PROVENANCE[name] for name in node_names},
        },
    )
    validate_native_capabilities(
        synthetic, available_nodes, node_provenance=node_provenance
    )
