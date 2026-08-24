from __future__ import annotations

"""Standard placement and bundled RayLight runtime descriptor implementation."""

from typing import Literal, Mapping

from ...native_templates import raylight_runtime_namespace
from ...schemas import DiffusionModelBinding, StrictModel
from ..contracts import EdgeRef, FeatureEmission, Resource, ScopedGraphBuilderProtocol
from ..feature_config import ExecutionStrategyConfigV1, LoraConfigV1, V6ConfigContext
from ..v6_registry import (
    FeatureResolutionV6,
    ResolvedEarlierFeatures,
    ResolvedFeatureImplementation,
)
from ._emitter import ScopedBuilderEmitter
from .raylight_model_path import emit_raylight_pool_intent
from .standard_model_path import emit_standard_model_device


class RayRuntimeDescriptorV1(StrictModel):
    attention_mode: Literal["TORCH_FLASH", "COMFY_KITCHEN_INT8"]
    namespace: str
    gpu_select: tuple[int, ...]
    ulysses_degree: int
    ring_degree: int
    cfg_degree: int
    dp_degree: int
    fsdp: bool
    cpu_offload: bool
    clear_vram_after_sampling: bool


def diffusion_binding(
    context: V6ConfigContext,
    lora: LoraConfigV1 | None = None,
) -> DiffusionModelBinding:
    model_filename = getattr(context.draft.model_stack, context.family).filename
    if model_filename is None:
        raise ValueError("diffusion model selection is incomplete")
    placement = getattr(context.settings.placement, context.family)
    return DiffusionModelBinding(
        filename=model_filename,
        device=placement.device,
        lora_name=(lora.lora_filename if lora is not None else None),
        lora_strength=(lora.strength if lora is not None else 1.0),
        lora_loader="auto",
        lora_low_vram=bool(lora.options.get("low_vram", False)) if lora else False,
        backend="auto",
        raylight=placement.raylight,
    )


def resolve_execution_strategy(
    *,
    config: ExecutionStrategyConfigV1,
    dependencies: ResolvedEarlierFeatures,
    context: V6ConfigContext,
    implementation_id: str,
    implementation_version: int,
) -> FeatureResolutionV6:
    if config.backend == "standard":
        return FeatureResolutionV6(
            implementation=ResolvedFeatureImplementation(
                implementation_id=implementation_id,
                implementation_version=implementation_version,
                carrier_kind="comfy_node",
                responsibility="host_user",
                class_types=("SelectModelDevice",),
            ),
            details={"backend": "standard", "device": config.device},
        )

    ray = config.raylight
    if ray is None:
        raise TypeError("RayLight execution strategy requires its topology")
    ck = dependencies.optional("comfy_kitchen_attention")
    attention_mode: Literal["TORCH_FLASH", "COMFY_KITCHEN_INT8"] = (
        "COMFY_KITCHEN_INT8" if ck is not None else "TORCH_FLASH"
    )
    semantic_mode = "ck_int8" if ck is not None else "torch_flash"
    binding = diffusion_binding(context)
    descriptor = RayRuntimeDescriptorV1(
        attention_mode=attention_mode,
        namespace=raylight_runtime_namespace(
            binding,
            attention_mode=semantic_mode,
            enforce_attention_topology=True,
        ),
        gpu_select=ray.gpu_select,
        ulysses_degree=ray.ulysses_degree,
        ring_degree=ray.ring_degree,
        cfg_degree=ray.cfg_degree,
        dp_degree=ray.dp_degree,
        fsdp=ray.fsdp,
        cpu_offload=ray.cpu_offload,
        clear_vram_after_sampling=ray.clear_vram_after_sampling,
    )
    return FeatureResolutionV6(
        implementation=ResolvedFeatureImplementation(
            implementation_id=implementation_id,
            implementation_version=implementation_version,
            carrier_kind="director_runtime",
            responsibility="director",
            class_types=("DirectorDeckRayInitializerAdvanced",),
        ),
        details={
            "backend": "raylight",
            "runtime_descriptor": descriptor.model_dump(mode="json"),
        },
    )


def ray_runtime_descriptor(
    resolution: FeatureResolutionV6,
) -> RayRuntimeDescriptorV1:
    raw = resolution.details.get("runtime_descriptor")
    if raw is None:
        raise TypeError("RayLight execution resolution has no runtime descriptor")
    return RayRuntimeDescriptorV1.model_validate(dict(raw))


def emit_execution_strategy(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    config: StrictModel,
    resolution: FeatureResolutionV6,
    context: V6ConfigContext,
) -> FeatureEmission:
    typed = ExecutionStrategyConfigV1.model_validate(config)
    emitter = ScopedBuilderEmitter(builder)
    if typed.backend == "standard":
        model = inputs.get("model")
        if model is None or not isinstance(model.value, EdgeRef):
            raise TypeError("Standard execution strategy requires one MODEL edge")
        emitted = emit_standard_model_device(
            emitter,
            [model.value.node_id, model.value.output_slot],
            diffusion_binding(context),
        )
        return FeatureEmission(outputs={"model": builder.edge(emitted[0], emitted[1])})

    descriptor = ray_runtime_descriptor(resolution)
    emitted = emit_raylight_pool_intent(
        emitter,
        diffusion_binding(context),
        namespace=descriptor.namespace,
        clear_vram_after_sampling=descriptor.clear_vram_after_sampling,
        attention_mode=(
            "ck_int8"
            if descriptor.attention_mode == "COMFY_KITCHEN_INT8"
            else "torch_flash"
        ),
        enforce_attention_topology=True,
    )
    return FeatureEmission(
        outputs={"ray_actors_init": builder.edge(emitted[0], emitted[1])}
    )


__all__ = [
    "RayRuntimeDescriptorV1",
    "diffusion_binding",
    "emit_execution_strategy",
    "ray_runtime_descriptor",
    "resolve_execution_strategy",
]
