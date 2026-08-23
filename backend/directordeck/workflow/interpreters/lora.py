from __future__ import annotations

from collections.abc import Mapping

from ...native_templates import (
    NativeTemplateError,
    resolve_execution_backend,
)
from ...schemas import DiffusionModelBinding
from ..contracts import JsonValue
from ..lora_factory import LoraAdapterResolutionError, require_lora_adapter
from ._emitter import NativeEdge, NativeNodeEmitter, edge


INTERPRETER_ID = "lora"
INTERPRETER_VERSION = 1


def emit_standard_lora(
    emitter: NativeNodeEmitter,
    model: NativeEdge,
    binding: DiffusionModelBinding,
    *,
    adapter_id: str,
    loader_node: str,
    adapter_options: Mapping[str, JsonValue],
) -> NativeEdge:
    if resolve_execution_backend(binding) != "standard":
        raise NativeTemplateError("Standard LoRA interpreter received a RayLight binding")
    if binding.lora_name is None:
        raise NativeTemplateError("disabled Standard LoRA must not invoke its interpreter")
    try:
        adapter = require_lora_adapter(adapter_id)
    except LoraAdapterResolutionError as exc:
        raise NativeTemplateError(
            f"invalid resolved Standard LoRA adapter: {adapter_id!r}"
        ) from exc
    if adapter.backend != "standard" or adapter.class_type != loader_node:
        raise NativeTemplateError("Standard LoRA adapter/node evidence drifted")
    options = dict(adapter_options)
    if adapter.input_contract == "dedicated_model":
        inputs = {
            "model": model,
            "lora_name": binding.lora_name,
            "strength": binding.lora_strength,
            "low_vram": options.get("low_vram", binding.lora_low_vram),
        }
    elif adapter.input_contract in {"model_only", "bypass_model_only"}:
        inputs = {
            "model": model,
            "lora_name": binding.lora_name,
            "strength_model": binding.lora_strength,
        }
    else:
        raise NativeTemplateError(
            f"invalid Standard LoRA input contract: {adapter.input_contract!r}"
        )
    return edge(emitter.add(loader_node, **inputs))


def emit_raylight_lora(
    emitter: NativeNodeEmitter,
    binding: DiffusionModelBinding,
) -> NativeEdge:
    if resolve_execution_backend(binding) != "raylight":
        raise NativeTemplateError("RayLight LoRA interpreter received a Standard binding")
    if binding.lora_name is None:
        raise NativeTemplateError("disabled RayLight LoRA must not invoke its interpreter")
    return edge(
        emitter.add(
            "DirectorDeckRayLoraLoader",
            lora_name=binding.lora_name,
            strength_model=binding.lora_strength,
        )
    )
