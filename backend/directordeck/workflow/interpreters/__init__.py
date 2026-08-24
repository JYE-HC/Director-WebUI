"""Stage-2 behavior-preserving native workflow fragment interpreters."""

from ._emitter import NativeEdge, NativeNodeEmitter, ScopedBuilderEmitter
from ._types import (
    AudioOutput,
    ConditioningOutputs,
    ContinuityOutputs,
    SharedModelOutputs,
)
from .audio_output import emit_audio_output
from .builtin import (
    V4BuiltinContext,
    V4BuiltinInterpreter,
    V4BuiltinParams,
    builtin_implementation_identity,
    builtin_required_capability_ids,
    builtin_interpreter_map,
    builtin_interpreters,
    catalog_implementation_alternatives,
)
from .conditioning import (
    emit_family_conditioning,
    emit_fl2va_conditioning,
    emit_ref2va_conditioning,
)
from .continuity import emit_continuity
from .decode_video import emit_decode_video
from .lora import emit_raylight_lora, emit_standard_lora
from .raylight_model_path import (
    emit_raylight_model_load,
    emit_raylight_pool_intent,
    emit_raylight_sigma_shift,
)
from .sampling_raylight import emit_raylight_sampling
from .sampling_standard import emit_standard_sampling
from .save_take import emit_save_take
from .shared_models import emit_auxiliary_models, emit_shared_models
from .standard_model_path import (
    emit_standard_model_device,
    emit_standard_model_load,
    emit_standard_sigma_shift,
)


STANDARD_INTERPRETER_IDS = (
    "shared_models",
    "standard_model_load",
    "standard_model_device",
    "lora",
    "standard_sigma_shift",
    "family_conditioning",
    "continuity",
    "standard_sampling",
    "decode_video",
    "audio_output",
    "save_take",
)

RAYLIGHT_INTERPRETER_IDS = (
    "shared_models",
    "raylight_pool_intent",
    "lora",
    "raylight_model_load",
    "raylight_sigma_shift",
    "family_conditioning",
    "continuity",
    "raylight_sampling",
    "decode_video",
    "audio_output",
    "save_take",
)


__all__ = [
    "AudioOutput",
    "ConditioningOutputs",
    "ContinuityOutputs",
    "NativeEdge",
    "NativeNodeEmitter",
    "RAYLIGHT_INTERPRETER_IDS",
    "STANDARD_INTERPRETER_IDS",
    "ScopedBuilderEmitter",
    "SharedModelOutputs",
    "V4BuiltinContext",
    "V4BuiltinInterpreter",
    "V4BuiltinParams",
    "builtin_implementation_identity",
    "builtin_required_capability_ids",
    "builtin_interpreter_map",
    "builtin_interpreters",
    "catalog_implementation_alternatives",
    "emit_audio_output",
    "emit_auxiliary_models",
    "emit_continuity",
    "emit_decode_video",
    "emit_family_conditioning",
    "emit_fl2va_conditioning",
    "emit_raylight_lora",
    "emit_raylight_model_load",
    "emit_raylight_pool_intent",
    "emit_raylight_sampling",
    "emit_raylight_sigma_shift",
    "emit_ref2va_conditioning",
    "emit_save_take",
    "emit_shared_models",
    "emit_standard_lora",
    "emit_standard_model_device",
    "emit_standard_model_load",
    "emit_standard_sampling",
    "emit_standard_sigma_shift",
]
