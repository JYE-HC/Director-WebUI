# Added for Director Web; see DIRECTOR_MODIFICATIONS.md.
from __future__ import annotations

from collections.abc import Sequence


_NO_VISIBLE_CUDA_DEVICES = {"", "-1", "NoDevFiles"}


def resolve_cuda_visible_devices(
    selected_gpus: Sequence[int] | None,
    parent_cuda_visible_devices: str | None,
    visible_device_count: int | None = None,
) -> str | None:
    """Resolve logical GPU indices through the parent's CUDA visibility mask.

    CUDA renumbers a mask such as ``CUDA_VISIBLE_DEVICES=7,6`` to logical
    devices ``0,1`` inside ComfyUI. Ray must be started with the corresponding
    physical tokens (``7,6``), not the logical indices again, or it escapes the
    parent mask and schedules workers on physical GPUs 0 and 1.
    """

    if selected_gpus is None:
        return parent_cuda_visible_devices

    selected = tuple(selected_gpus)
    if not selected:
        return parent_cuda_visible_devices
    if any(gpu_idx < 0 for gpu_idx in selected):
        raise ValueError(f"GPU_SELECT only supports zero-based GPU indices, got {selected}")
    if len(set(selected)) != len(selected):
        raise ValueError(f"GPU_SELECT contains duplicate GPU indices: {selected}")
    if parent_cuda_visible_devices is None:
        return ",".join(str(gpu_idx) for gpu_idx in selected)

    raw_visibility = parent_cuda_visible_devices.strip()
    if raw_visibility in _NO_VISIBLE_CUDA_DEVICES:
        if selected:
            raise ValueError(
                "GPU_SELECT requested GPUs, but the parent CUDA_VISIBLE_DEVICES exposes none"
            )
        return raw_visibility

    visible_tokens = tuple(token.strip() for token in raw_visibility.split(","))
    if any(not token for token in visible_tokens):
        raise ValueError(
            f"Parent CUDA_VISIBLE_DEVICES is malformed: {parent_cuda_visible_devices!r}"
        )
    normalized_tokens = tuple(
        str(int(token)) if token.isdigit() else token.casefold() for token in visible_tokens
    )
    if len(set(normalized_tokens)) != len(normalized_tokens):
        raise ValueError(
            f"Parent CUDA_VISIBLE_DEVICES contains duplicate devices: {parent_cuda_visible_devices!r}"
        )
    if visible_device_count is not None and len(visible_tokens) != visible_device_count:
        raise ValueError(
            "Parent CUDA_VISIBLE_DEVICES contains "
            f"{len(visible_tokens)} entries, but torch reports {visible_device_count} visible CUDA devices"
        )

    invalid = [gpu_idx for gpu_idx in selected if gpu_idx < 0 or gpu_idx >= len(visible_tokens)]
    if invalid:
        raise ValueError(
            "GPU_SELECT contains logical GPU indices outside the parent "
            f"CUDA_VISIBLE_DEVICES range 0-{len(visible_tokens) - 1}: {invalid}"
        )

    return ",".join(visible_tokens[gpu_idx] for gpu_idx in selected)
