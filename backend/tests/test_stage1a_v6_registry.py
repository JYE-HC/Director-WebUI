from __future__ import annotations

from types import SimpleNamespace

import pytest

from directordeck.schemas import RuntimeSettingsV3, UnifiedTimelineDraftV5
from directordeck.workflow.contracts import ResolvedImplementationIdentity
from directordeck.workflow.feature_config import (
    ComfyKitchenAttentionParamsV1,
    V6_CONFIG_RESOLVERS,
)
from directordeck.workflow.feature_definitions import BUNDLE6_FEATURE_DEFINITIONS
from directordeck.workflow.node_contracts import (
    CURRENT_NODE_CONTRACT_REGISTRY,
    V4_NODE_CONTRACT_REGISTRY,
    V5_NODE_CONTRACT_REGISTRY,
    V6_NODE_CONTRACT_REGISTRY,
    node_contract_registry_for_bundle,
    require_v6_node_contract,
)
from directordeck.workflow.templates_v6 import (
    V6_RAYLIGHT_SEGMENT_TEMPLATE,
    V6_STANDARD_SEGMENT_TEMPLATE,
)
from directordeck.workflow.v6_implementations import V6_FEATURE_REGISTRY
from directordeck.workflow.v6_registry import (
    ResolvedEarlierFeatures,
    V6RegistryError,
)

from .test_workflow_v5_compat import _v4_pair, _v5_pair


def _v6_pair() -> tuple[UnifiedTimelineDraftV5, RuntimeSettingsV3]:
    v4, settings_v1 = _v4_pair()
    draft, settings = _v5_pair(v4, settings_v1)
    document = draft.model_dump(mode="json")
    document["features"]["template_bundle_version"] = 6
    document["features"]["project"]["comfy_kitchen_attention"] = {
        "enabled": True,
        "params": {},
    }
    return UnifiedTimelineDraftV5.model_validate(document), settings


def _context(
    draft: UnifiedTimelineDraftV5,
    settings: RuntimeSettingsV3,
    *,
    backend: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        draft=draft,
        settings=settings,
        segment=draft.segments[0],
        backend=backend,
        family="fl2va",
        job_id="v6-registry",
        visible_frames=120,
        sample_frames=120,
        continuity_prefix_frames=0,
        predecessor_segment_id=None,
        continuity_source=None,
        historical_take_id=None,
        clear_raylight_vram_after_sampling=False,
    )


def _ray_settings(settings: RuntimeSettingsV3) -> RuntimeSettingsV3:
    document = settings.model_dump(mode="json")
    document["multi_gpu_enabled"] = True
    document["placement"]["fl2va"]["raylight"].update(
        {
            "gpu_select": [0, 1],
            "ulysses_degree": 2,
            "ring_degree": 1,
            "cfg_degree": 1,
            "dp_degree": 1,
            "fsdp": False,
            "cpu_offload": False,
        }
    )
    return RuntimeSettingsV3.model_validate(document)


def test_v6_registry_has_exact_feature_and_route_keys() -> None:
    expected_features = tuple(
        (definition.id, definition.version)
        for definition in BUNDLE6_FEATURE_DEFINITIONS
    )
    expected_implementations = tuple(
        (
            definition.id,
            definition.version,
            backend,
            family,
        )
        for definition in BUNDLE6_FEATURE_DEFINITIONS
        for backend in definition.backends
        for family in definition.families
    )

    assert V6_FEATURE_REGISTRY.feature_identities == expected_features
    assert V6_FEATURE_REGISTRY.implementation_identities == expected_implementations
    assert len(expected_features) == 12
    assert len(expected_implementations) == 48
    V6_FEATURE_REGISTRY.validate_template(V6_STANDARD_SEGMENT_TEMPLATE)
    V6_FEATURE_REGISTRY.validate_template(V6_RAYLIGHT_SEGMENT_TEMPLATE)

    with pytest.raises(V6RegistryError, match="unknown exact feature"):
        V6_FEATURE_REGISTRY.require_feature("comfy_kitchen_attention", 2)
    with pytest.raises(V6RegistryError, match="unknown exact feature implementation"):
        V6_FEATURE_REGISTRY.require_implementation(
            "comfy_kitchen_attention",
            2,
            "standard",
            "fl2va",
        )


@pytest.mark.parametrize("family", ("fl2va", "ref2va"))
def test_standard_ck_maps_to_official_comfy_node(family: str) -> None:
    draft, settings = _v6_pair()
    context = _context(draft, settings, backend="standard")
    context.family = family
    implementation = V6_FEATURE_REGISTRY.require_implementation(
        "comfy_kitchen_attention",
        1,
        "standard",
        family,  # type: ignore[arg-type]
    )

    resolution = implementation.resolve(
        ComfyKitchenAttentionParamsV1(),
        ResolvedEarlierFeatures((), {}),
        context,
    )

    assert resolution.implementation.implementation_id == (
        "directordeck.v6.comfy_kitchen_attention.standard"
    )
    assert resolution.implementation.carrier_kind == "comfy_node"
    assert resolution.implementation.responsibility == "host_user"
    assert resolution.implementation.class_types == ("ModelAttentionBackend",)
    assert "runtime_fingerprint" not in type(
        resolution.implementation
    ).model_fields
    assert resolution.details == {
        "backend": "standard",
        "prompt_choice": "comfy kitchen attention",
    }


def test_raylight_ck_is_director_runtime_intent_consumed_by_execution() -> None:
    draft, settings = _v6_pair()
    settings = _ray_settings(settings)
    context = _context(draft, settings, backend="raylight")
    ck_implementation = V6_FEATURE_REGISTRY.require_implementation(
        "comfy_kitchen_attention",
        1,
        "raylight",
        "fl2va",
    )
    ck_resolution = ck_implementation.resolve(
        ComfyKitchenAttentionParamsV1(),
        ResolvedEarlierFeatures((), {}),
        context,
    )
    assert ck_resolution.implementation.carrier_kind == "director_runtime"
    assert ck_resolution.implementation.responsibility == "director"
    assert ck_resolution.implementation.class_types == ()
    assert ck_resolution.details["initializer_attention"] == "COMFY_KITCHEN_INT8"

    execution_use = next(
        entry
        for entry in V6_RAYLIGHT_SEGMENT_TEMPLATE.entries
        if entry.feature_id == "execution_strategy"
    )
    execution_config = V6_CONFIG_RESOLVERS["execution_strategy"].resolve(
        context
    ).config
    assert execution_config is not None
    execution = V6_FEATURE_REGISTRY.require_implementation(
        "execution_strategy",
        1,
        "raylight",
        "fl2va",
    )
    enabled = execution.resolve(
        execution_config,
        ResolvedEarlierFeatures(
            execution_use.dependencies,
            {"comfy_kitchen_attention": ck_resolution},
        ),
        context,
    )
    disabled = execution.resolve(
        execution_config,
        ResolvedEarlierFeatures(
            execution_use.dependencies,
            {"comfy_kitchen_attention": None},
        ),
        context,
    )
    assert enabled.details["runtime_descriptor"]["attention_mode"] == (
        "COMFY_KITCHEN_INT8"
    )
    assert disabled.details["runtime_descriptor"]["attention_mode"] == (
        "TORCH_FLASH"
    )


def test_v6_official_ck_node_contract_is_advisory_not_an_authorization_gate() -> None:
    contract = require_v6_node_contract("ModelAttentionBackend")
    assert contract.contract_id == "directordeck.node.ModelAttentionBackend"
    assert contract.allowed_python_modules == (
        "comfy_extras.nodes_model_advanced",
    )
    assert set(contract.object_info_contract.required_inputs) == {
        "model",
        "attention",
    }
    assert contract.object_info_contract.required_inputs[
        "attention"
    ].enum_values == (
        "pytorch attention",
        "comfy kitchen attention",
    )
    effect = contract.runtime_effect_contract
    assert effect.policy == "identity_allowed"
    assert effect.unsupported_behavior == "fallback"
    assert effect.validation_method == "user_assumed"
    assert effect.verified_model_families == ("fl2va", "ref2va")
    assert effect.verified_backends == ("standard",)
    assert any("advisory" in note for note in effect.notes)

    adapter_identity = ResolvedImplementationIdentity(
        role="feature.comfy_kitchen_attention",
        class_type=contract.class_type,
        implementation_id=contract.contract_id,
        semantic_version=contract.semantic_version,
        runtime_fingerprint=contract.supported_runtime_fingerprints[0],
        binding_key="comfy_kitchen_attention.ModelAttentionBackend",
    )
    assert V6_NODE_CONTRACT_REGISTRY.validate_implementation(
        adapter_identity,
        output_affecting=True,
        model_family="fl2va",
        backend="standard",
    ) == contract


def test_v6_registry_does_not_reinterpret_frozen_v5_contracts() -> None:
    assert CURRENT_NODE_CONTRACT_REGISTRY is V6_NODE_CONTRACT_REGISTRY
    assert node_contract_registry_for_bundle(5) is V5_NODE_CONTRACT_REGISTRY
    assert node_contract_registry_for_bundle(6) is V6_NODE_CONTRACT_REGISTRY
    assert set(V5_NODE_CONTRACT_REGISTRY.contracts) == (
        set(V4_NODE_CONTRACT_REGISTRY.contracts)
        | {
            "DirectorStrictModelAttentionBackend",
            "DirectorStrictH3LowVramSagePatch",
        }
    )
    assert set(V6_NODE_CONTRACT_REGISTRY.contracts) == (
        set(V4_NODE_CONTRACT_REGISTRY.contracts) | {"ModelAttentionBackend"}
    )
    for class_type, frozen in V4_NODE_CONTRACT_REGISTRY.contracts.items():
        assert V5_NODE_CONTRACT_REGISTRY.require(class_type) == frozen
        assert V6_NODE_CONTRACT_REGISTRY.require(class_type) == frozen
    with pytest.raises(KeyError):
        V5_NODE_CONTRACT_REGISTRY.require("ModelAttentionBackend")
    with pytest.raises(KeyError):
        V6_NODE_CONTRACT_REGISTRY.require(
            "DirectorStrictModelAttentionBackend"
        )
