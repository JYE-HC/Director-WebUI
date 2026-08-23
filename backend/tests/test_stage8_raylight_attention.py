from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import pytest

from directordeck.native_templates import (
    NativeTemplateError,
    NativeWorkflowUnit,
    RaylightAttentionMode,
    _raylight_namespace,
    build_raylight_shutdown_unit,
    raylight_runtime_descriptor,
    raylight_runtime_namespace,
    resolve_raylight_attention_backend,
)
from directordeck.schemas import DiffusionModelBinding, RuntimeSettings
from directordeck.workflow.interpreters.raylight_model_path import (
    emit_raylight_model_load,
    emit_raylight_pool_intent,
    emit_raylight_sigma_shift,
)

from .extensible_workflow_v0_fixture_builder import _draft, _settings


class _PromptEmitter:
    def __init__(self) -> None:
        self.prompt: dict[str, dict[str, Any]] = {}

    def add(self, class_type: str, **inputs: Any) -> str:
        node_id = str(len(self.prompt) + 1)
        self.prompt[node_id] = {
            "class_type": class_type,
            "inputs": inputs,
        }
        return node_id


def _binding(*, ring_degree: int = 1) -> DiffusionModelBinding:
    raw = _settings("raylight").model_dump(mode="json")
    if ring_degree > 1:
        raw["models"]["fl2va"]["raylight"].update(
            gpu_select=list(range(ring_degree)),
            ulysses_degree=1,
            ring_degree=ring_degree,
        )
    return RuntimeSettings.model_validate(raw).models.fl2va


def _runtime_unit(mode: RaylightAttentionMode) -> NativeWorkflowUnit:
    binding = _binding()
    emitter = _PromptEmitter()
    pool = emit_raylight_pool_intent(
        emitter,
        binding,
        namespace=raylight_runtime_namespace(
            binding,
            attention_mode=mode,
        ),
        clear_vram_after_sampling=False,
        attention_mode=mode,
    )
    model = emit_raylight_model_load(emitter, pool, binding)
    emit_raylight_sigma_shift(
        emitter,
        model,
        _draft("t2v").sampling.fl2va,
    )
    return NativeWorkflowUnit(
        id=f"raylight-attention-{mode}",
        family="fl2va",
        backend="raylight",
        segment_ids=("segment-1",),
        prompt=emitter.prompt,
        output_nodes={},
    )


def _initializer(unit: NativeWorkflowUnit) -> dict[str, Any]:
    matches = [
        node
        for node in unit.prompt.values()
        if node["class_type"] == "DirectorDeckRayInitializerAdvanced"
    ]
    assert len(matches) == 1
    return matches[0]


def test_raylight_attention_modes_map_to_exact_host_enums() -> None:
    assert resolve_raylight_attention_backend("ck_int8") == "COMFY_KITCHEN_INT8"
    assert resolve_raylight_attention_backend("torch_flash") == "TORCH_FLASH"


def test_raylight_attention_changes_only_initializer_contract_and_runtime_identity(
) -> None:
    ck_unit = _runtime_unit("ck_int8")
    flash_unit = _runtime_unit("torch_flash")
    ck_initializer = _initializer(ck_unit)
    flash_initializer = _initializer(flash_unit)

    assert ck_initializer["inputs"]["XFuser_attention"] == "COMFY_KITCHEN_INT8"
    assert flash_initializer["inputs"]["XFuser_attention"] == "TORCH_FLASH"
    assert (
        ck_initializer["inputs"]["ray_cluster_namespace"]
        != flash_initializer["inputs"]["ray_cluster_namespace"]
    )

    ck_without_identity = deepcopy(ck_unit.prompt)
    flash_without_identity = deepcopy(flash_unit.prompt)
    for prompt in (ck_without_identity, flash_without_identity):
        initializer = next(
            node
            for node in prompt.values()
            if node["class_type"] == "DirectorDeckRayInitializerAdvanced"
        )
        initializer["inputs"].pop("XFuser_attention")
        initializer["inputs"].pop("ray_cluster_namespace")
    assert ck_without_identity == flash_without_identity

    ck_descriptor = raylight_runtime_descriptor(ck_unit)
    flash_descriptor = raylight_runtime_descriptor(flash_unit)
    assert ck_descriptor is not None
    assert flash_descriptor is not None
    assert (
        ck_descriptor["compatibility_key"]
        != flash_descriptor["compatibility_key"]
    )
    assert ck_descriptor["runtime_key"] != flash_descriptor["runtime_key"]


def test_raylight_attention_namespace_and_initializer_must_use_same_mode() -> None:
    binding = _binding()
    ck_namespace = raylight_runtime_namespace(
        binding,
        attention_mode="ck_int8",
    )
    emitter = _PromptEmitter()

    with pytest.raises(
        NativeTemplateError,
        match="attention and runtime namespace do not match",
    ):
        emit_raylight_pool_intent(
            emitter,
            binding,
            namespace=ck_namespace,
            clear_vram_after_sampling=False,
            attention_mode="torch_flash",
        )

    assert emitter.prompt == {}


def test_raylight_attention_rejects_unknown_mode_before_graph_emission() -> None:
    invalid_mode = cast(RaylightAttentionMode, "host_default")
    binding = _binding()
    emitter = _PromptEmitter()

    with pytest.raises(NativeTemplateError, match="attention mode must be"):
        resolve_raylight_attention_backend(invalid_mode)
    with pytest.raises(NativeTemplateError, match="attention mode must be"):
        _raylight_namespace(
            "fl2va",
            binding,
            attention_mode=invalid_mode,
        )
    with pytest.raises(NativeTemplateError, match="attention mode must be"):
        emit_raylight_pool_intent(
            emitter,
            binding,
            namespace="director-invalid",
            clear_vram_after_sampling=False,
            attention_mode=invalid_mode,
        )

    assert emitter.prompt == {}


def test_ck_int8_with_ring_parallelism_fails_closed_but_torch_flash_is_valid(
) -> None:
    binding = _binding(ring_degree=2)
    emitter = _PromptEmitter()

    with pytest.raises(NativeTemplateError, match="requires ring_degree=1"):
        raylight_runtime_namespace(binding, attention_mode="ck_int8")
    with pytest.raises(NativeTemplateError, match="requires ring_degree=1"):
        emit_raylight_pool_intent(
            emitter,
            binding,
            namespace="director-invalid-ring",
            clear_vram_after_sampling=False,
            attention_mode="ck_int8",
        )

    flash_namespace = raylight_runtime_namespace(
        binding,
        attention_mode="torch_flash",
    )
    emit_raylight_pool_intent(
        emitter,
        binding,
        namespace=flash_namespace,
        clear_vram_after_sampling=False,
        attention_mode="torch_flash",
    )
    assert _initializer(
        NativeWorkflowUnit(
            id="ring-flash",
            family="fl2va",
            backend="raylight",
            segment_ids=("segment-1",),
            prompt=emitter.prompt,
            output_nodes={},
        )
    )["inputs"]["XFuser_attention"] == "TORCH_FLASH"


def test_shutdown_replays_persisted_attention_value_exactly() -> None:
    unit = _runtime_unit("torch_flash")
    descriptor = raylight_runtime_descriptor(unit)
    assert descriptor is not None

    shutdown = build_raylight_shutdown_unit(
        descriptor,
        unit_id="attention-recovery",
    )
    assert _initializer(shutdown)["inputs"]["XFuser_attention"] == "TORCH_FLASH"


def test_frozen_v4_default_namespace_remains_explicit_ck_int8() -> None:
    binding = _binding()

    assert _raylight_namespace("fl2va", binding) == _raylight_namespace(
        "fl2va",
        binding,
        attention_mode="ck_int8",
    )
    assert _initializer(_runtime_unit("ck_int8"))["inputs"][
        "XFuser_attention"
    ] == "COMFY_KITCHEN_INT8"
