from __future__ import annotations

from ...native_templates import NativeTemplateError, resolve_execution_backend
from ...schemas import DiffusionModelBinding, SamplingConfig
from ._emitter import NativeEdge, NativeNodeEmitter, edge


MODEL_LOAD_INTERPRETER_ID = "standard_model_load"
MODEL_DEVICE_INTERPRETER_ID = "standard_model_device"
SIGMA_SHIFT_INTERPRETER_ID = "standard_sigma_shift"
INTERPRETER_VERSION = 1


def _require_standard(binding: DiffusionModelBinding) -> None:
    if resolve_execution_backend(binding) != "standard":
        raise NativeTemplateError("Standard model interpreter received a RayLight binding")


def emit_standard_model_load(
    emitter: NativeNodeEmitter,
    binding: DiffusionModelBinding,
) -> NativeEdge:
    _require_standard(binding)
    return edge(
        emitter.add(
            "UNETLoader",
            unet_name=binding.filename,
            weight_dtype="default",
        )
    )


def emit_standard_model_device(
    emitter: NativeNodeEmitter,
    model: NativeEdge,
    binding: DiffusionModelBinding,
) -> NativeEdge:
    _require_standard(binding)
    return edge(
        emitter.add(
            "SelectModelDevice",
            model=model,
            device=binding.device,
        )
    )


def emit_standard_sigma_shift(
    emitter: NativeNodeEmitter,
    model: NativeEdge,
    sampling: SamplingConfig,
) -> NativeEdge:
    return edge(
        emitter.add(
            "MiniMaxH3SigmaShift",
            model=model,
            shift_video=sampling.shift,
            shift_audio=sampling.audio_shift,
        )
    )
