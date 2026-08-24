from __future__ import annotations

from ...schemas import RuntimeSettings
from ._emitter import NativeNodeEmitter, edge
from ._types import SharedModelOutputs


INTERPRETER_ID = "shared_models"
INTERPRETER_VERSION = 1


def emit_auxiliary_models(
    emitter: NativeNodeEmitter,
    *,
    clip_filename: str,
    clip_device: str,
    video_vae_filename: str,
    video_vae_device: str,
    audio_vae_filename: str | None,
    audio_vae_device: str | None,
) -> SharedModelOutputs:
    """Emit the shared model fragment from its explicit feature authority."""

    clip_loader = emitter.add(
        "CLIPLoader",
        clip_name=clip_filename,
        type="minimax",
        device="default",
    )
    clip = emitter.add(
        "SelectCLIPDevice",
        clip=edge(clip_loader),
        device=clip_device,
    )
    video_loader = emitter.add("VAELoader", vae_name=video_vae_filename)
    video_vae = emitter.add(
        "SelectVAEDevice",
        vae=edge(video_loader),
        device=video_vae_device,
    )
    audio_vae = None
    if audio_vae_filename is not None and audio_vae_device is not None:
        audio_loader = emitter.add("VAELoader", vae_name=audio_vae_filename)
        audio_vae = edge(
            emitter.add(
                "SelectVAEDevice",
                vae=edge(audio_loader),
                device=audio_vae_device,
            )
        )
    return SharedModelOutputs(
        clip=edge(clip),
        video_vae=edge(video_vae),
        audio_vae=audio_vae,
    )


def emit_shared_models(
    emitter: NativeNodeEmitter,
    settings: RuntimeSettings,
) -> SharedModelOutputs:
    """Emit the legacy CLIP/video-VAE/audio-VAE loader fragment verbatim."""

    return emit_auxiliary_models(
        emitter,
        clip_filename=settings.models.clip.filename,
        clip_device=settings.models.clip.device,
        video_vae_filename=settings.models.video_vae.filename,
        video_vae_device=settings.models.video_vae.device,
        audio_vae_filename=settings.models.audio_vae.filename,
        audio_vae_device=settings.models.audio_vae.device,
    )
