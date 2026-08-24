from __future__ import annotations

"""Contextual, advisory-only Comfy Kitchen attention capability projection."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from ..schemas import RuntimeSettingsV3, StrictModel
from ..workflow.contracts import Identifier, ModelFamily


CapabilityState = Literal["available", "unavailable", "unknown"]
CapabilityBackend = Literal["standard", "raylight"]
ObjectInfoObservationState = Literal["observed", "absent", "unknown"]
DeviceBackend = Literal["cuda", "cpu", "xpu", "mps"]

STANDARD_CK_CLASS_TYPE = "ModelAttentionBackend"
STANDARD_CK_INPUT = "attention"
STANDARD_CK_CHOICE = "comfy kitchen attention"
RAYLIGHT_CK_CLASS_TYPE = "DirectorDeckRayInitializerAdvanced"
RAYLIGHT_CK_INPUT = "XFuser_attention"
RAYLIGHT_CK_CHOICE = "COMFY_KITCHEN_INT8"


class ContextualCapabilityReasonV1(StrictModel):
    code: Identifier
    message: Annotated[str, Field(min_length=1, max_length=1024)]


class ComfyKitchenAttentionCapabilityV1(StrictModel):
    context_revision: Identifier
    backend: CapabilityBackend | None
    state: CapabilityState
    reasons: Annotated[
        tuple[ContextualCapabilityReasonV1, ...],
        Field(max_length=16),
    ] = ()

    @model_validator(mode="after")
    def validate_reason_state(self) -> "ComfyKitchenAttentionCapabilityV1":
        if self.state == "available" and self.reasons:
            raise ValueError("available capability must not include reasons")
        if self.state != "available" and not self.reasons:
            raise ValueError("non-available capability must include a reason")
        codes = tuple(reason.code for reason in self.reasons)
        if len(codes) != len(set(codes)):
            raise ValueError("capability reason codes must be unique")
        return self


class ObjectInfoChoiceObservationV1(StrictModel):
    """One bounded combo observation, detached from the complete object-info."""

    state: ObjectInfoObservationState
    choices: Annotated[tuple[Annotated[str, Field(max_length=256)], ...], Field(max_length=256)] = ()

    @model_validator(mode="after")
    def validate_choices(self) -> "ObjectInfoChoiceObservationV1":
        if self.state != "observed" and self.choices:
            raise ValueError("only an observed object-info input may expose choices")
        return self


class ComfyKitchenAttentionObjectInfoObservationV1(StrictModel):
    standard_attention: ObjectInfoChoiceObservationV1
    raylight_attention: ObjectInfoChoiceObservationV1


class LogicalDeviceObservationV1(StrictModel):
    logical_index: Annotated[int, Field(ge=0, le=255)]
    backend: Literal["cuda", "xpu", "mps"]


class ComfyKitchenAttentionHostObservationV1(StrictModel):
    """Only the host facts needed by the dedicated CK UI affordance."""

    context_revision: Identifier
    host_connected: bool
    standard_attention: ObjectInfoChoiceObservationV1
    raylight_attention: ObjectInfoChoiceObservationV1
    primary_device_backend: DeviceBackend | None = None
    gpu_inventory_state: Literal["observed", "unknown"] = "unknown"
    gpu_inventory: Annotated[
        tuple[LogicalDeviceObservationV1, ...],
        Field(max_length=256),
    ] = ()

    @model_validator(mode="after")
    def validate_gpu_inventory(self) -> "ComfyKitchenAttentionHostObservationV1":
        indices = tuple(gpu.logical_index for gpu in self.gpu_inventory)
        if len(indices) != len(set(indices)):
            raise ValueError("logical GPU observation indices must be unique")
        if self.gpu_inventory_state == "unknown" and self.gpu_inventory:
            raise ValueError("unknown GPU inventory cannot expose devices")
        if self.gpu_inventory_state == "unknown" and self.primary_device_backend:
            raise ValueError("unknown GPU inventory cannot expose a primary device")
        return self


def observe_object_info_choice(
    *,
    node_observed: bool | None,
    node_info: Any,
    input_name: str,
) -> ObjectInfoChoiceObservationV1:
    """Normalize one relevant required combo without retaining full object-info.

    ``node_observed=None`` means the request or top-level structure could not be
    confirmed. ``False`` means a successful inventory definitively omitted the
    node. A present node with an unrecognised schema is also unknown.
    """

    if node_observed is None:
        return ObjectInfoChoiceObservationV1(state="unknown")
    if node_observed is False:
        return ObjectInfoChoiceObservationV1(state="absent")
    if not isinstance(node_info, dict):
        return ObjectInfoChoiceObservationV1(state="unknown")
    inputs = node_info.get("input")
    if not isinstance(inputs, dict):
        return ObjectInfoChoiceObservationV1(state="unknown")
    required = inputs.get("required")
    if not isinstance(required, dict):
        return ObjectInfoChoiceObservationV1(state="unknown")
    if input_name not in required:
        return ObjectInfoChoiceObservationV1(state="observed")
    entry = required[input_name]
    if not isinstance(entry, (tuple, list)) or not entry:
        return ObjectInfoChoiceObservationV1(state="unknown")
    choices = entry[0]
    if choices == "COMBO":
        metadata = entry[1] if len(entry) > 1 else None
        if not isinstance(metadata, dict):
            return ObjectInfoChoiceObservationV1(state="unknown")
        choices = metadata.get("options")
    if (
        not isinstance(choices, (tuple, list))
        or len(choices) > 256
        or any(
            not isinstance(choice, str) or len(choice) > 256
            for choice in choices
        )
    ):
        return ObjectInfoChoiceObservationV1(state="unknown")
    return ObjectInfoChoiceObservationV1(
        state="observed",
        choices=tuple(choices),
    )


def observe_comfy_kitchen_attention_object_info(
    *,
    request_succeeded: bool,
    object_info: Any,
) -> ComfyKitchenAttentionObjectInfoObservationV1:
    """Read only the two exact CK carriers from one object-info response."""

    if type(request_succeeded) is not bool:
        raise TypeError("request_succeeded must be boolean")
    if not request_succeeded or not isinstance(object_info, Mapping):
        unknown = ObjectInfoChoiceObservationV1(state="unknown")
        return ComfyKitchenAttentionObjectInfoObservationV1(
            standard_attention=unknown,
            raylight_attention=unknown,
        )

    def choice(class_type: str, input_name: str) -> ObjectInfoChoiceObservationV1:
        return observe_object_info_choice(
            node_observed=class_type in object_info,
            node_info=object_info.get(class_type),
            input_name=input_name,
        )

    return ComfyKitchenAttentionObjectInfoObservationV1(
        standard_attention=choice(STANDARD_CK_CLASS_TYPE, STANDARD_CK_INPUT),
        raylight_attention=choice(RAYLIGHT_CK_CLASS_TYPE, RAYLIGHT_CK_INPUT),
    )


def project_comfy_kitchen_attention_capability(
    *,
    settings: RuntimeSettingsV3 | None,
    host: ComfyKitchenAttentionHostObservationV1,
    reachable_families: Sequence[ModelFamily] = ("fl2va", "ref2va"),
) -> ComfyKitchenAttentionCapabilityV1:
    """Project current-path observations into a non-authoritative UI hint."""

    families = tuple(reachable_families)
    if not families or len(families) != len(set(families)) or any(
        family not in ("fl2va", "ref2va") for family in families
    ):
        raise ValueError("reachable families must be a non-empty unique subset")
    context_revision = _projection_revision(settings, host, families)
    if settings is None:
        return _result(
            context_revision,
            backend=None,
            state="unknown",
            code="runtime_settings_unavailable",
            message="Runtime settings are temporarily unavailable.",
        )
    backend: CapabilityBackend = (
        "raylight" if settings.multi_gpu_enabled else "standard"
    )
    if not host.host_connected:
        return _result(
            context_revision,
            backend=backend,
            state="unknown",
            code="host_not_connected",
            message="The current ComfyUI host is not connected.",
        )
    if backend == "standard":
        return _project_standard(settings, host, families, context_revision)
    return _project_raylight(settings, host, families, context_revision)


def _projection_revision(
    settings: RuntimeSettingsV3 | None,
    host: ComfyKitchenAttentionHostObservationV1,
    families: tuple[ModelFamily, ...],
) -> str:
    if settings is None:
        context: object = None
    elif settings.multi_gpu_enabled:
        context = {
            "backend": "raylight",
            "families": [
                {
                    "family": family,
                    "gpu_select": getattr(
                        settings.placement, family
                    ).raylight.gpu_select,
                    "ring_degree": getattr(
                        settings.placement, family
                    ).raylight.ring_degree,
                }
                for family in families
            ],
        }
    else:
        context = {
            "backend": "standard",
            "families": [
                {
                    "family": family,
                    "device": getattr(settings.placement, family).device,
                }
                for family in families
            ],
        }
    encoded = json.dumps(
        {
            "host": host.model_dump(mode="json"),
            "context": context,
        },
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "ck:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _standard_target_backend(
    settings: RuntimeSettingsV3,
    host: ComfyKitchenAttentionHostObservationV1,
    families: tuple[ModelFamily, ...],
) -> DeviceBackend | None:
    by_index = {device.logical_index: device.backend for device in host.gpu_inventory}
    backends: set[DeviceBackend] = set()
    unresolved = False
    for family in families:
        device = getattr(settings.placement, family).device
        if device == "default":
            backend = host.primary_device_backend
        elif device == "cpu":
            backend = "cpu"
        elif device.startswith("gpu:") and host.gpu_inventory_state == "observed":
            try:
                backend = by_index.get(int(device.removeprefix("gpu:")))
            except ValueError:
                backend = None
        else:
            backend = None
        if backend is None:
            unresolved = True
        else:
            backends.add(backend)
    known_non_cuda = next(
        (backend for backend in ("cpu", "xpu", "mps") if backend in backends),
        None,
    )
    if known_non_cuda is not None:
        return known_non_cuda
    if unresolved:
        return None
    return "cuda" if backends == {"cuda"} else None


def _project_standard(
    settings: RuntimeSettingsV3,
    host: ComfyKitchenAttentionHostObservationV1,
    families: tuple[ModelFamily, ...],
    context_revision: str,
) -> ComfyKitchenAttentionCapabilityV1:
    observation = host.standard_attention
    if observation.state == "unknown":
        return _result(
            context_revision,
            backend="standard",
            state="unknown",
            code="standard_ck_node_not_observed",
            message="The official CK attention node could not be confirmed.",
        )
    if observation.state == "absent":
        return _result(
            context_revision,
            backend="standard",
            state="unavailable",
            code="standard_ck_node_not_observed",
            message="The official CK attention node is not available on this host.",
        )
    if STANDARD_CK_CHOICE not in observation.choices:
        return _result(
            context_revision,
            backend="standard",
            state="unavailable",
            code="standard_ck_choice_not_observed",
            message="The official node does not offer Comfy Kitchen attention.",
        )
    target_backend = _standard_target_backend(settings, host, families)
    if target_backend is None:
        return _result(
            context_revision,
            backend="standard",
            state="unknown",
            code="target_device_not_cuda",
            message="The Standard diffusion target could not be confirmed as CUDA.",
        )
    if target_backend != "cuda":
        return _result(
            context_revision,
            backend="standard",
            state="unavailable",
            code="target_device_not_cuda",
            message="The Standard diffusion target is not CUDA.",
        )
    return ComfyKitchenAttentionCapabilityV1(
        context_revision=context_revision,
        backend="standard",
        state="available",
    )


def _project_raylight(
    settings: RuntimeSettingsV3,
    host: ComfyKitchenAttentionHostObservationV1,
    families: tuple[ModelFamily, ...],
    context_revision: str,
) -> ComfyKitchenAttentionCapabilityV1:
    observation = host.raylight_attention
    if observation.state == "unknown":
        return _result(
            context_revision,
            backend="raylight",
            state="unknown",
            code="bundled_raylight_ck_not_observed",
            message="The bundled RayLight CK integration could not be confirmed.",
        )
    if observation.state == "absent":
        return _result(
            context_revision,
            backend="raylight",
            state="unavailable",
            code="bundled_raylight_ck_not_observed",
            message="The bundled RayLight CK integration is not available.",
        )
    if RAYLIGHT_CK_CHOICE not in observation.choices:
        return _result(
            context_revision,
            backend="raylight",
            state="unavailable",
            code="raylight_ck_choice_not_observed",
            message="The bundled RayLight initializer does not offer CK attention.",
        )
    profiles = tuple(getattr(settings.placement, family).raylight for family in families)
    if any(profile.ring_degree != 1 for profile in profiles):
        return _result(
            context_revision,
            backend="raylight",
            state="unavailable",
            code="raylight_ring_degree_incompatible",
            message="Comfy Kitchen attention requires RayLight ring degree 1.",
        )
    if host.gpu_inventory_state == "unknown":
        return _result(
            context_revision,
            backend="raylight",
            state="unknown",
            code="raylight_topology_incompatible",
            message="The selected RayLight CUDA topology could not be confirmed.",
        )
    selected_indices = tuple(
        dict.fromkeys(index for profile in profiles for index in profile.gpu_select)
    )
    inventory = {gpu.logical_index: gpu for gpu in host.gpu_inventory}
    if any(index not in inventory for index in selected_indices):
        return _result(
            context_revision,
            backend="raylight",
            state="unavailable",
            code="raylight_topology_incompatible",
            message="The selected RayLight topology cannot be built from observed GPUs.",
        )
    if not selected_indices or any(
        inventory[index].backend != "cuda" for index in selected_indices
    ):
        return _result(
            context_revision,
            backend="raylight",
            state="unavailable",
            code="target_device_not_cuda",
            message="The selected RayLight topology is not entirely CUDA.",
        )
    return ComfyKitchenAttentionCapabilityV1(
        context_revision=context_revision,
        backend="raylight",
        state="available",
    )


def _result(
    context_revision: str,
    *,
    backend: CapabilityBackend | None,
    state: Literal["unavailable", "unknown"],
    code: str,
    message: str,
) -> ComfyKitchenAttentionCapabilityV1:
    return ComfyKitchenAttentionCapabilityV1(
        context_revision=context_revision,
        backend=backend,
        state=state,
        reasons=(ContextualCapabilityReasonV1(code=code, message=message),),
    )


__all__ = [
    "ComfyKitchenAttentionCapabilityV1",
    "ComfyKitchenAttentionHostObservationV1",
    "ComfyKitchenAttentionObjectInfoObservationV1",
    "ContextualCapabilityReasonV1",
    "LogicalDeviceObservationV1",
    "ObjectInfoChoiceObservationV1",
    "RAYLIGHT_CK_CHOICE",
    "RAYLIGHT_CK_CLASS_TYPE",
    "RAYLIGHT_CK_INPUT",
    "STANDARD_CK_CHOICE",
    "STANDARD_CK_CLASS_TYPE",
    "STANDARD_CK_INPUT",
    "observe_comfy_kitchen_attention_object_info",
    "observe_object_info_choice",
    "project_comfy_kitchen_attention_capability",
]
