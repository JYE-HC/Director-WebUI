from __future__ import annotations

import pytest
from pydantic import ValidationError

from directordeck.capabilities.comfy_kitchen_attention import (
    ComfyKitchenAttentionCapabilityV1,
    ComfyKitchenAttentionHostObservationV1,
    ContextualCapabilityReasonV1,
    LogicalDeviceObservationV1,
    ObjectInfoChoiceObservationV1,
    RAYLIGHT_CK_CLASS_TYPE,
    STANDARD_CK_CLASS_TYPE,
    observe_comfy_kitchen_attention_object_info,
    observe_object_info_choice,
    project_comfy_kitchen_attention_capability,
)
from directordeck.schemas import (
    RuntimeSettingsV3,
    default_runtime_settings_v3,
)


def _settings(*, multi_gpu: bool = False) -> RuntimeSettingsV3:
    document = default_runtime_settings_v3().model_dump(mode="json")
    document["multi_gpu_enabled"] = multi_gpu
    if multi_gpu:
        for family in ("fl2va", "ref2va"):
            document["placement"][family]["raylight"].update(
                {
                    "gpu_select": [0, 1],
                    "ulysses_degree": 2,
                    "ring_degree": 1,
                }
            )
    return RuntimeSettingsV3.model_validate(document)


def _with_ring_two(
    settings: RuntimeSettingsV3,
    *,
    family: str = "ref2va",
) -> RuntimeSettingsV3:
    document = settings.model_dump(mode="json")
    document["placement"][family]["raylight"].update(
        {
            "gpu_select": [0, 1],
            "ulysses_degree": 1,
            "ring_degree": 2,
        }
    )
    return RuntimeSettingsV3.model_validate(document)


def _choice(
    *choices: str,
    state: str = "observed",
) -> ObjectInfoChoiceObservationV1:
    return ObjectInfoChoiceObservationV1(
        state=state,
        choices=tuple(choices),
    )


def _gpu(index: int, backend: str = "cuda") -> LogicalDeviceObservationV1:
    return LogicalDeviceObservationV1(
        logical_index=index,
        backend=backend,
    )


def _host(
    *,
    connected: bool = True,
    standard: ObjectInfoChoiceObservationV1 | None = None,
    raylight: ObjectInfoChoiceObservationV1 | None = None,
    primary_backend: str | None = "cuda",
    inventory_state: str = "observed",
    inventory: tuple[LogicalDeviceObservationV1, ...] | None = None,
) -> ComfyKitchenAttentionHostObservationV1:
    return ComfyKitchenAttentionHostObservationV1(
        context_revision="ctx:stage2",
        host_connected=connected,
        standard_attention=standard or _choice(
            "pytorch attention",
            "comfy kitchen attention",
        ),
        raylight_attention=raylight or _choice(
            "TORCH_FLASH",
            "COMFY_KITCHEN_INT8",
        ),
        primary_device_backend=(
            primary_backend if inventory_state == "observed" else None
        ),
        gpu_inventory_state=inventory_state,
        gpu_inventory=(
            (_gpu(0), _gpu(1))
            if inventory is None and inventory_state == "observed"
            else inventory or ()
        ),
    )


def _reason_code(capability: ComfyKitchenAttentionCapabilityV1) -> str:
    assert len(capability.reasons) == 1
    return capability.reasons[0].code


def test_wire_is_bounded_advisory_state_without_authorization_fields() -> None:
    assert set(ComfyKitchenAttentionCapabilityV1.model_fields) == {
        "context_revision",
        "backend",
        "state",
        "reasons",
    }
    assert set(ComfyKitchenAttentionHostObservationV1.model_fields) == {
        "context_revision",
        "host_connected",
        "standard_attention",
        "raylight_attention",
        "primary_device_backend",
        "gpu_inventory_state",
        "gpu_inventory",
    }
    forbidden = {
        "submission_allowed",
        "source",
        "module",
        "package",
        "version",
        "fingerprint",
        "per_gpu_probe",
    }
    assert forbidden.isdisjoint(ComfyKitchenAttentionCapabilityV1.model_fields)
    assert forbidden.isdisjoint(ComfyKitchenAttentionHostObservationV1.model_fields)

    with pytest.raises(ValidationError, match="must not include reasons"):
        ComfyKitchenAttentionCapabilityV1(
            context_revision="ctx:invalid",
            backend="standard",
            state="available",
            reasons=(
                ContextualCapabilityReasonV1(code="unexpected", message="No."),
            ),
        )
    with pytest.raises(ValidationError, match="must include a reason"):
        ComfyKitchenAttentionCapabilityV1(
            context_revision="ctx:invalid",
            backend="standard",
            state="unknown",
        )


def test_object_info_provider_reads_only_exact_ck_carriers_and_choices() -> None:
    observed = observe_comfy_kitchen_attention_object_info(
        request_succeeded=True,
        object_info={
            STANDARD_CK_CLASS_TYPE: {
                "input": {
                    "required": {
                        "attention": [
                            ["pytorch attention", "comfy kitchen attention"],
                            {"default": "pytorch attention"},
                        ]
                    }
                },
                "python_module": "ignored.external.metadata",
            },
            RAYLIGHT_CK_CLASS_TYPE: {
                "input": {
                    "required": {
                        "XFuser_attention": [
                            "COMBO",
                            {"options": ["TORCH_FLASH", "COMFY_KITCHEN_INT8"]},
                        ]
                    }
                }
            },
            "RayInitializerAdvanced": {
                "input": {
                    "required": {
                        "XFuser_attention": [["COMFY_KITCHEN_INT8"]]
                    }
                }
            },
        },
    )

    assert observed.standard_attention.state == "observed"
    assert observed.standard_attention.choices == (
        "pytorch attention",
        "comfy kitchen attention",
    )
    assert observed.raylight_attention.state == "observed"
    assert observed.raylight_attention.choices == (
        "TORCH_FLASH",
        "COMFY_KITCHEN_INT8",
    )

    failed = observe_comfy_kitchen_attention_object_info(
        request_succeeded=False,
        object_info=None,
    )
    assert failed.standard_attention.state == "unknown"
    assert failed.raylight_attention.state == "unknown"

    external_only = observe_comfy_kitchen_attention_object_info(
        request_succeeded=True,
        object_info={"RayInitializerAdvanced": {}},
    )
    assert external_only.raylight_attention.state == "absent"


@pytest.mark.parametrize(
    ("node_observed", "node_info", "expected"),
    [
        (None, None, "unknown"),
        (False, None, "absent"),
        (True, None, "unknown"),
        (True, {"input": []}, "unknown"),
        (True, {"input": {"required": {"attention": "COMBO"}}}, "unknown"),
    ],
)
def test_object_info_failure_or_unrecognised_structure_is_unknown(
    node_observed: bool | None,
    node_info: object,
    expected: str,
) -> None:
    observation = observe_object_info_choice(
        node_observed=node_observed,
        node_info=node_info,
        input_name="attention",
    )
    assert observation.state == expected


def test_standard_path_available_and_does_not_evaluate_raylight() -> None:
    capability = project_comfy_kitchen_attention_capability(
        settings=_with_ring_two(_settings()),
        host=_host(
            raylight=_choice(state="unknown"),
        ),
    )

    assert capability.context_revision.startswith("ck:")
    assert capability.backend == "standard"
    assert capability.state == "available"
    assert capability.reasons == ()


@pytest.mark.parametrize(
    ("standard", "target", "state", "reason"),
    [
        (
            ObjectInfoChoiceObservationV1(state="unknown"),
            "cuda",
            "unknown",
            "standard_ck_node_not_observed",
        ),
        (
            ObjectInfoChoiceObservationV1(state="absent"),
            "cuda",
            "unavailable",
            "standard_ck_node_not_observed",
        ),
        (
            ObjectInfoChoiceObservationV1(
                state="observed",
                choices=("Comfy Kitchen Attention",),
            ),
            "cuda",
            "unavailable",
            "standard_ck_choice_not_observed",
        ),
        (
            ObjectInfoChoiceObservationV1(
                state="observed",
                choices=("comfy kitchen attention",),
            ),
            None,
            "unknown",
            "target_device_not_cuda",
        ),
        (
            ObjectInfoChoiceObservationV1(
                state="observed",
                choices=("comfy kitchen attention",),
            ),
            "cpu",
            "unavailable",
            "target_device_not_cuda",
        ),
    ],
)
def test_standard_path_distinguishes_unknown_from_definite_unavailable(
    standard: ObjectInfoChoiceObservationV1,
    target: str | None,
    state: str,
    reason: str,
) -> None:
    capability = project_comfy_kitchen_attention_capability(
        settings=_settings(),
        host=_host(standard=standard, primary_backend=target),
    )
    assert capability.backend == "standard"
    assert capability.state == state
    assert _reason_code(capability) == reason


def test_raylight_path_available_and_does_not_evaluate_standard() -> None:
    capability = project_comfy_kitchen_attention_capability(
        settings=_settings(multi_gpu=True),
        host=_host(
            standard=_choice(state="unknown"),
            primary_backend="cpu",
        ),
    )

    assert capability.backend == "raylight"
    assert capability.state == "available"
    assert capability.reasons == ()


def test_raylight_checks_every_reachable_profile_for_exact_ring_one() -> None:
    settings = _with_ring_two(_settings(multi_gpu=True), family="ref2va")
    unavailable = project_comfy_kitchen_attention_capability(
        settings=settings,
        host=_host(),
    )
    fl2va_only = project_comfy_kitchen_attention_capability(
        settings=settings,
        host=_host(),
        reachable_families=("fl2va",),
    )

    assert unavailable.state == "unavailable"
    assert _reason_code(unavailable) == "raylight_ring_degree_incompatible"
    assert fl2va_only.state == "available"


@pytest.mark.parametrize(
    ("raylight", "inventory_state", "inventory", "state", "reason"),
    [
        (
            ObjectInfoChoiceObservationV1(state="unknown"),
            "observed",
            (_gpu(0), _gpu(1)),
            "unknown",
            "bundled_raylight_ck_not_observed",
        ),
        (
            ObjectInfoChoiceObservationV1(state="absent"),
            "observed",
            (_gpu(0), _gpu(1)),
            "unavailable",
            "bundled_raylight_ck_not_observed",
        ),
        (
            ObjectInfoChoiceObservationV1(
                state="observed",
                choices=("TORCH_FLASH",),
            ),
            "observed",
            (_gpu(0), _gpu(1)),
            "unavailable",
            "raylight_ck_choice_not_observed",
        ),
        (
            ObjectInfoChoiceObservationV1(
                state="observed",
                choices=("COMFY_KITCHEN_INT8",),
            ),
            "unknown",
            (),
            "unknown",
            "raylight_topology_incompatible",
        ),
        (
            ObjectInfoChoiceObservationV1(
                state="observed",
                choices=("COMFY_KITCHEN_INT8",),
            ),
            "observed",
            (_gpu(0),),
            "unavailable",
            "raylight_topology_incompatible",
        ),
        (
            ObjectInfoChoiceObservationV1(
                state="observed",
                choices=("COMFY_KITCHEN_INT8",),
            ),
            "observed",
            (_gpu(0), _gpu(1, "xpu")),
            "unavailable",
            "target_device_not_cuda",
        ),
    ],
)
def test_raylight_path_reports_bounded_observation_and_topology_results(
    raylight: ObjectInfoChoiceObservationV1,
    inventory_state: str,
    inventory: tuple[LogicalDeviceObservationV1, ...],
    state: str,
    reason: str,
) -> None:
    capability = project_comfy_kitchen_attention_capability(
        settings=_settings(multi_gpu=True),
        host=_host(
            raylight=raylight,
            inventory_state=inventory_state,
            inventory=inventory,
        ),
    )
    assert capability.backend == "raylight"
    assert capability.state == state
    assert _reason_code(capability) == reason


def test_missing_settings_and_disconnected_host_are_unknown_not_denials() -> None:
    missing_settings = project_comfy_kitchen_attention_capability(
        settings=None,
        host=_host(),
    )
    disconnected = project_comfy_kitchen_attention_capability(
        settings=_settings(multi_gpu=True),
        host=_host(connected=False),
    )

    assert missing_settings.backend is None
    assert missing_settings.state == "unknown"
    assert _reason_code(missing_settings) == "runtime_settings_unavailable"
    assert disconnected.backend == "raylight"
    assert disconnected.state == "unknown"
    assert _reason_code(disconnected) == "host_not_connected"
