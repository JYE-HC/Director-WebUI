from __future__ import annotations

"""Small live observer for the dedicated CK capability affordance."""

from collections.abc import Mapping
from typing import Any

import httpx

from ..comfy import ComfyClientProtocol, ComfyError
from .comfy_kitchen_attention import (
    ComfyKitchenAttentionHostObservationV1,
    LogicalDeviceObservationV1,
    ObjectInfoChoiceObservationV1,
    RAYLIGHT_CK_CLASS_TYPE,
    STANDARD_CK_CLASS_TYPE,
    observe_comfy_kitchen_attention_object_info,
)


def _logical_devices(
    value: Any,
) -> tuple[str, tuple[LogicalDeviceObservationV1, ...], str | None]:
    if not isinstance(value, Mapping):
        return "unknown", (), None
    devices = value.get("devices")
    if devices is None:
        system = value.get("system")
        devices = system.get("devices") if isinstance(system, Mapping) else None
    if not isinstance(devices, list):
        return "unknown", (), None
    primary_backend: str | None = None
    observed: list[LogicalDeviceObservationV1] = []
    for position, item in enumerate(devices):
        if not isinstance(item, Mapping):
            return "unknown", (), None
        backend = item.get("type")
        if not isinstance(backend, str):
            return "unknown", (), None
        backend = backend.strip().lower()
        if backend not in {"cuda", "cpu", "xpu", "mps"}:
            return "unknown", (), None
        if position == 0:
            # ComfyUI deliberately reports its actual primary/default device
            # first. Preserve that identity instead of guessing from index 0.
            primary_backend = backend
        if backend == "cpu":
            continue
        index = item.get("index")
        if backend == "mps" and index is None:
            continue
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            return "unknown", (), None
        observed.append(
            LogicalDeviceObservationV1(logical_index=index, backend=backend)
        )
    try:
        result = tuple(sorted(observed, key=lambda item: item.logical_index))
        if len({item.logical_index for item in result}) != len(result):
            return "unknown", (), None
        return "observed", result, primary_backend
    except (TypeError, ValueError):
        return "unknown", (), None


async def observe_comfy_kitchen_attention_host(
    client: ComfyClientProtocol,
    *,
    context_revision: str,
    previous: ComfyKitchenAttentionHostObservationV1 | None = None,
) -> ComfyKitchenAttentionHostObservationV1:
    unknown = ObjectInfoChoiceObservationV1(state="unknown")
    standard = previous.standard_attention if previous else unknown
    raylight = previous.raylight_attention if previous else unknown
    connected = previous.host_connected if previous else False
    missing_classes = tuple(
        class_type
        for class_type, observation in (
            (STANDARD_CK_CLASS_TYPE, standard),
            (RAYLIGHT_CK_CLASS_TYPE, raylight),
        )
        if observation.state == "unknown"
    )
    if missing_classes:
        object_info: Any = None
        request_succeeded = False
        try:
            object_info = await client.object_info(missing_classes)
            request_succeeded = True
        except (ComfyError, httpx.HTTPError, OSError, RuntimeError, TypeError, ValueError):
            pass
        connected = connected or request_succeeded
        choices = observe_comfy_kitchen_attention_object_info(
            request_succeeded=request_succeeded,
            object_info=object_info,
        )
        if STANDARD_CK_CLASS_TYPE in missing_classes:
            standard = choices.standard_attention
        if RAYLIGHT_CK_CLASS_TYPE in missing_classes:
            raylight = choices.raylight_attention

    inventory_state = previous.gpu_inventory_state if previous else "unknown"
    inventory = previous.gpu_inventory if previous else ()
    primary_backend = previous.primary_device_backend if previous else None
    if connected and inventory_state == "unknown":
        try:
            inventory_state, inventory, primary_backend = _logical_devices(
                await client.system_stats()
            )
        except (ComfyError, httpx.HTTPError, OSError, RuntimeError, TypeError, ValueError):
            pass
    return ComfyKitchenAttentionHostObservationV1(
        context_revision=context_revision,
        host_connected=connected,
        standard_attention=standard,
        raylight_attention=raylight,
        primary_device_backend=primary_backend,
        gpu_inventory_state=inventory_state,
        gpu_inventory=inventory,
    )


def comfy_kitchen_attention_host_observation_complete(
    observation: ComfyKitchenAttentionHostObservationV1,
) -> bool:
    """Return whether all process-static facts were observed conclusively."""

    return (
        observation.host_connected
        and observation.standard_attention.state != "unknown"
        and observation.raylight_attention.state != "unknown"
        and observation.gpu_inventory_state == "observed"
        and observation.primary_device_backend is not None
    )


__all__ = [
    "comfy_kitchen_attention_host_observation_complete",
    "observe_comfy_kitchen_attention_host",
]
