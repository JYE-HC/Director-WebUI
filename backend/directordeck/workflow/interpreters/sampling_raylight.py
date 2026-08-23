from __future__ import annotations

from ...schemas import SamplingConfig
from ._emitter import NativeEdge, NativeNodeEmitter, edge


INTERPRETER_ID = "raylight_sampling"
INTERPRETER_VERSION = 1


def emit_raylight_sampling(
    emitter: NativeNodeEmitter,
    *,
    ray_actors: NativeEdge,
    conditioning: NativeEdge,
    latent: NativeEdge,
    sampling: SamplingConfig,
    seed: int,
) -> NativeEdge:
    guider = emitter.add(
        "DirectorDeckRayBasicGuider",
        ray_actors=ray_actors,
        conditioning=conditioning,
    )
    scheduler = emitter.add(
        "DirectorDeckRayBasicScheduler",
        ray_actors=ray_actors,
        scheduler=sampling.scheduler,
        steps=sampling.steps,
        denoise=1.0,
    )
    sampler = emitter.add("KSamplerSelect", sampler_name=sampling.sampler)
    sampled = emitter.add(
        "DirectorDeckRayXFuserSamplerCustomAdvanced",
        add_noise=True,
        noise_seed=seed,
        guider=edge(guider),
        sampler=edge(sampler),
        sigmas=edge(scheduler),
        latent_image=latent,
    )
    return edge(sampled, 0)
