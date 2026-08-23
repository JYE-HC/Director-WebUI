from __future__ import annotations

from ...native_templates import _shared_core
from ...schemas import RuntimeSettings
from ._emitter import NativeNodeEmitter
from ._types import SharedModelOutputs


INTERPRETER_ID = "shared_models"
INTERPRETER_VERSION = 1


def emit_shared_models(
    emitter: NativeNodeEmitter,
    settings: RuntimeSettings,
) -> SharedModelOutputs:
    """Emit the legacy CLIP/video-VAE/audio-VAE loader fragment verbatim."""

    shared = _shared_core(emitter, settings)  # type: ignore[arg-type]
    return SharedModelOutputs(
        clip=shared["clip"],
        video_vae=shared["video_vae"],
        audio_vae=shared["audio_vae"],
    )
