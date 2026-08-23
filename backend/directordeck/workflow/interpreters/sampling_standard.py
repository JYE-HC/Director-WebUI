from __future__ import annotations

from ...schemas import SamplingConfig
from ._emitter import NativeEdge, NativeNodeEmitter, edge


INTERPRETER_ID = "standard_sampling"
INTERPRETER_VERSION = 1


def emit_standard_sampling(
    emitter: NativeNodeEmitter,
    *,
    model: NativeEdge,
    conditioning: NativeEdge,
    latent: NativeEdge,
    sampling: SamplingConfig,
    seed: int,
) -> NativeEdge:
    guider = emitter.add(
        "BasicGuider",
        model=model,
        conditioning=conditioning,
    )
    scheduler = emitter.add(
        "BasicScheduler",
        model=model,
        scheduler=sampling.scheduler,
        steps=sampling.steps,
        denoise=1.0,
    )
    sampler = emitter.add("KSamplerSelect", sampler_name=sampling.sampler)
    noise = emitter.add("RandomNoise", noise_seed=seed)
    sampled = emitter.add(
        "SamplerCustomAdvanced",
        noise=edge(noise),
        guider=edge(guider),
        sampler=edge(sampler),
        sigmas=edge(scheduler),
        latent_image=latent,
    )
    return edge(sampled, 0)
