from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import directordeck.workflow.node_contracts as node_contracts
from directordeck.native_templates import (
    EXPECTED_NATIVE_NODE_MODULES,
    _PROVENANCE,
)
from directordeck.workflow.contracts import (
    NodeContractRegistry,
    ResolvedImplementationIdentity,
    RuntimeEffectContract,
)
from directordeck.workflow.canonical import canonical_json_bytes
from directordeck.workflow.node_contracts import (
    CURRENT_NODE_CONTRACT_REGISTRY,
    DIRECTOR_NODE_ADAPTER_CONTENT_DIGESTS,
    DIRECTOR_NODE_ADAPTER_SOURCE_FILES,
    RuntimeFingerprintMaterial,
    V4_NODE_CONTRACT_REGISTRY,
    compute_runtime_fingerprint,
    current_expected_module_policy,
    current_provenance_policy,
    current_release_supported_runtime_fingerprints,
    native_expected_module_policy,
    native_provenance_policy,
    release_supported_runtime_fingerprints,
    require_current_node_contract,
    require_native_node_contract,
)


_GOLDENS = (
    Path(__file__).parent
    / "fixtures"
    / "extensible_workflow_v0"
    / "native_prompt_goldens.json"
)

_STAGE8_STRICT_FEATURE_CLASSES = {
    "DirectorStrictModelAttentionBackend",
    "DirectorStrictH3LowVramSagePatch",
}
_FROZEN_TO_CURRENT_RAYLIGHT_CLASS = {
    "RayInitializerAdvanced": "DirectorDeckRayInitializerAdvanced",
    "RayLoraLoader": "DirectorDeckRayLoraLoader",
    "RayUNETLoader": "DirectorDeckRayUNETLoader",
    "RayMiniMaxH3SigmaShift": "DirectorDeckRayMiniMaxH3SigmaShift",
    "RayBasicGuider": "DirectorDeckRayBasicGuider",
    "RayBasicScheduler": "DirectorDeckRayBasicScheduler",
    "XFuserSamplerCustomAdvanced": "DirectorDeckRayXFuserSamplerCustomAdvanced",
    "RayKill": "DirectorDeckRayKill",
}


def _current_golden_class_type(class_type: str) -> str:
    return _FROZEN_TO_CURRENT_RAYLIGHT_CLASS.get(class_type, class_type)


def _golden_cases() -> list[dict[str, Any]]:
    payload = json.loads(_GOLDENS.read_text(encoding="utf-8"))
    assert payload["fixture_schema"] == 1
    return payload["cases"]


def _prompt_edges(value: Any, prompt_ids: frozenset[str]) -> Iterator[tuple[str, int]]:
    if isinstance(value, list):
        if (
            len(value) == 2
            and isinstance(value[0], str)
            and value[0] in prompt_ids
            and isinstance(value[1], int)
            and not isinstance(value[1], bool)
        ):
            yield value[0], value[1]
            return
        for item in value:
            yield from _prompt_edges(item, prompt_ids)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _prompt_edges(item, prompt_ids)


def _implementation(class_type: str, fingerprint: str) -> ResolvedImplementationIdentity:
    contract = require_native_node_contract(class_type)
    return ResolvedImplementationIdentity(
        role="test.role",
        class_type=class_type,
        implementation_id=contract.contract_id,
        semantic_version=contract.semantic_version,
        runtime_fingerprint=fingerprint,
        binding_key=f"test.binding.{class_type.replace(' ', '_')}",
    )


def test_registry_covers_all_26_stage_zero_prompts_and_exact_slots() -> None:
    cases = _golden_cases()
    assert len(cases) == 26
    emitted_classes: set[str] = set()

    for case in cases:
        for unit in case["units"]:
            prompt = unit["prompt"]
            prompt_ids = frozenset(prompt)
            for node_id, node in prompt.items():
                contract = require_native_node_contract(
                    _current_golden_class_type(node["class_type"])
                )
                emitted_classes.add(contract.class_type)

                known_inputs = {
                    *contract.object_info_contract.required_inputs,
                    *contract.object_info_contract.optional_inputs,
                }
                assert set(node["inputs"]) <= known_inputs, (
                    case["id"],
                    node_id,
                    contract.class_type,
                    sorted(set(node["inputs"]) - known_inputs),
                )
                assert set(node["inputs"]) <= set(
                    contract.object_info_contract.director_supplied_inputs
                )

                for producer_id, output_slot in _prompt_edges(
                    node["inputs"], prompt_ids
                ):
                    producer = require_native_node_contract(
                        _current_golden_class_type(
                            prompt[producer_id]["class_type"]
                        )
                    )
                    assert 0 <= output_slot < len(producer.output_contract.slots), (
                        case["id"],
                        node_id,
                        producer_id,
                        output_slot,
                    )

    assert emitted_classes == set(V4_NODE_CONTRACT_REGISTRY.contracts)


def test_segment_and_raylight_control_terminal_roles_match_goldens() -> None:
    for case in _golden_cases():
        for unit in case["units"]:
            contracts = [
                require_native_node_contract(
                    _current_golden_class_type(node["class_type"])
                )
                for node in unit["prompt"].values()
            ]
            terminals = [
                contract
                for contract in contracts
                if contract.execution_terminal_role is not None
            ]
            artifacts = [
                contract
                for contract in contracts
                if contract.persistent_artifact_role is not None
            ]
            if case["kind"] == "raylight_control":
                assert [contract.class_type for contract in terminals] == ["DirectorDeckRayKill"]
                assert terminals[0].execution_terminal_role == "ray_kill"
                assert artifacts == []
                assert unit["output_nodes"] == {}
            else:
                assert [contract.class_type for contract in terminals] == ["SaveVideo"]
                assert [contract.class_type for contract in artifacts] == ["SaveVideo"]
                assert terminals[0].execution_terminal_role == "take"
                assert artifacts[0].persistent_artifact_role == "take"
                assert set(unit["output_nodes"].values()) == {
                    node_id
                    for node_id, node in unit["prompt"].items()
                    if node["class_type"] == "SaveVideo"
                }


def test_legacy_module_and_provenance_views_are_derived_exactly() -> None:
    assert dict(native_expected_module_policy()) == EXPECTED_NATIVE_NODE_MODULES
    assert dict(native_provenance_policy()) == _PROVENANCE

    selected = ("DirectorDeckRayKill", "UNETLoader", "DirectorDeckRayKill")
    assert dict(native_expected_module_policy(selected)) == {
        "DirectorDeckRayKill": "custom_nodes.DirectorDeck-RayLight",
        "UNETLoader": "nodes",
    }
    assert dict(native_provenance_policy(selected)) == {
        "DirectorDeckRayKill": "raylight",
        "UNETLoader": "comfy-core",
    }


def test_registry_is_frozen_round_trippable_and_has_module_scoped_fingerprints() -> None:
    restored = NodeContractRegistry.model_validate_json(
        V4_NODE_CONTRACT_REGISTRY.model_dump_json()
    )
    assert restored == V4_NODE_CONTRACT_REGISTRY

    with pytest.raises(TypeError):
        V4_NODE_CONTRACT_REGISTRY.contracts[  # type: ignore[index]
            "Other"
        ] = require_native_node_contract("UNETLoader")
    with pytest.raises(ValidationError):
        require_native_node_contract("UNETLoader").semantic_version = "2.0.0"  # type: ignore[misc]

    module_fingerprints: dict[str, str] = {}
    for contract in V4_NODE_CONTRACT_REGISTRY.contracts.values():
        assert len(contract.allowed_python_modules) == 1
        assert len(contract.supported_runtime_fingerprints) == 1
        module = contract.allowed_python_modules[0]
        fingerprint = contract.supported_runtime_fingerprints[0]
        previous = module_fingerprints.setdefault(module, fingerprint)
        assert previous == fingerprint
        assert release_supported_runtime_fingerprints(contract.class_type) == (
            fingerprint,
        )

    assert len(module_fingerprints) == len(set(EXPECTED_NATIVE_NODE_MODULES.values()))
    assert len(set(module_fingerprints.values())) == len(module_fingerprints)


def test_current_registry_extends_v4_without_changing_any_v4_contract() -> None:
    assert set(CURRENT_NODE_CONTRACT_REGISTRY.contracts) == (
        set(V4_NODE_CONTRACT_REGISTRY.contracts) | _STAGE8_STRICT_FEATURE_CLASSES
    )
    for class_type, contract in V4_NODE_CONTRACT_REGISTRY.contracts.items():
        assert CURRENT_NODE_CONTRACT_REGISTRY.require(class_type) == contract
    assert dict(current_expected_module_policy()) == {
        **dict(native_expected_module_policy()),
        "DirectorStrictModelAttentionBackend": (
            "custom_nodes.DirectorDeck-Strict-Attention"
        ),
        "DirectorStrictH3LowVramSagePatch": (
            "custom_nodes.DirectorDeck-Strict-H3"
        ),
    }
    assert dict(current_provenance_policy(_STAGE8_STRICT_FEATURE_CLASSES)) == {
        "DirectorStrictModelAttentionBackend": (
            "director-owned-strict-attention"
        ),
        "DirectorStrictH3LowVramSagePatch": "director-owned-strict-h3",
    }


def test_adapter_content_digests_come_from_reviewed_module_source_closures() -> None:
    directordeck_root = Path(__file__).parents[1] / "directordeck"
    current_modules = {
        contract.allowed_python_modules[0]
        for contract in CURRENT_NODE_CONTRACT_REGISTRY.contracts.values()
    }
    assert set(DIRECTOR_NODE_ADAPTER_SOURCE_FILES) == current_modules
    assert set(DIRECTOR_NODE_ADAPTER_CONTENT_DIGESTS) == current_modules

    for module, relative_files in DIRECTOR_NODE_ADAPTER_SOURCE_FILES.items():
        assert relative_files
        assert tuple(relative_files) == tuple(sorted(set(relative_files)))
        assert all(not Path(relative).is_absolute() for relative in relative_files)
        assert all(".." not in Path(relative).parts for relative in relative_files)
        source_manifest = tuple(
            {
                "module_path": relative,
                "content_digest": "sha256:"
                + hashlib.sha256(
                    (directordeck_root / relative).read_bytes()
                ).hexdigest(),
            }
            for relative in relative_files
        )
        expected = "sha256:" + hashlib.sha256(
            canonical_json_bytes(source_manifest)
        ).hexdigest()
        assert DIRECTOR_NODE_ADAPTER_CONTENT_DIGESTS[module] == expected


def _copy_reviewed_adapter_sources(destination: Path) -> None:
    source_root = Path(__file__).parents[1] / "directordeck"
    for relative in {
        relative
        for files in DIRECTOR_NODE_ADAPTER_SOURCE_FILES.values()
        for relative in files
    }:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / relative, target)


def test_adapter_source_drift_is_scoped_and_unrelated_interpreters_are_ignored(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "directordeck"
    _copy_reviewed_adapter_sources(source_root)
    baseline = {
        module: node_contracts._director_adapter_content_digest(
            module,
            directordeck_root=source_root,
        )
        for module in DIRECTOR_NODE_ADAPTER_SOURCE_FILES
    }

    unrelated = source_root / "workflow" / "interpreters" / "future_default_off.py"
    unrelated.write_text("INTERPRETER_ID = 'future_default_off'\n", encoding="utf-8")
    assert {
        module: node_contracts._director_adapter_content_digest(
            module,
            directordeck_root=source_root,
        )
        for module in DIRECTOR_NODE_ADAPTER_SOURCE_FILES
    } == baseline

    current_only = source_root / "workflow" / "v5_registry.py"
    current_only_bytes = current_only.read_bytes()
    current_only.write_bytes(current_only_bytes + b"\n# reviewed v5 drift\n")
    current_changed = {
        module: node_contracts._director_adapter_content_digest(
            module,
            directordeck_root=source_root,
        )
        for module in DIRECTOR_NODE_ADAPTER_SOURCE_FILES
    }
    assert {
        module
        for module in baseline
        if current_changed[module] != baseline[module]
    } == {
        "custom_nodes.DirectorDeck-Strict-Attention",
        "custom_nodes.DirectorDeck-Strict-H3",
    }
    current_only.write_bytes(current_only_bytes)

    sampling = source_root / "workflow" / "interpreters" / "sampling_standard.py"
    sampling.write_bytes(sampling.read_bytes() + b"\n# reviewed drift\n")
    changed = {
        module: node_contracts._director_adapter_content_digest(
            module,
            directordeck_root=source_root,
        )
        for module in DIRECTOR_NODE_ADAPTER_SOURCE_FILES
    }
    assert {
        module for module in baseline if changed[module] != baseline[module]
    } == {"comfy_extras.nodes_custom_sampler"}


def test_shared_graph_builder_drift_reaches_every_registered_module(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "directordeck"
    _copy_reviewed_adapter_sources(source_root)
    baseline = {
        module: node_contracts._director_adapter_content_digest(
            module,
            directordeck_root=source_root,
        )
        for module in DIRECTOR_NODE_ADAPTER_SOURCE_FILES
    }
    builder = source_root / "workflow" / "builder.py"
    builder.write_bytes(builder.read_bytes() + b"\n# reviewed shared drift\n")
    changed = {
        module: node_contracts._director_adapter_content_digest(
            module,
            directordeck_root=source_root,
        )
        for module in DIRECTOR_NODE_ADAPTER_SOURCE_FILES
    }
    assert all(changed[module] != baseline[module] for module in baseline)


def test_object_info_slice_uses_director_defaults_not_comfy_widget_defaults() -> None:
    image_to_video = require_native_node_contract(
        "MiniMaxH3ImageToVideo"
    ).object_info_contract
    assert image_to_video.required_inputs["width"].director_default == 864
    assert image_to_video.required_inputs["height"].director_default == 480
    assert image_to_video.required_inputs["length"].director_default == 124

    scheduler = require_native_node_contract("BasicScheduler").object_info_contract
    assert scheduler.required_inputs["steps"].director_default == 25
    assert scheduler.required_inputs["scheduler"].director_default == "simple"
    assert scheduler.required_inputs["scheduler"].enum_values == (
        "simple",
        "normal",
        "karras",
        "beta",
    )
    sampler = require_native_node_contract("KSamplerSelect").object_info_contract
    assert sampler.required_inputs["sampler_name"].director_default == (
        "res_multistep"
    )

    create_video = require_native_node_contract("CreateVideo").object_info_contract
    assert create_video.required_inputs["fps"].director_default == 24.0
    save_video = require_native_node_contract("SaveVideo").object_info_contract
    assert not save_video.required_inputs["filename_prefix"].has_director_default

    video_slice = require_native_node_contract("Video Slice").object_info_contract
    assert not video_slice.required_inputs["start_time"].has_director_default
    assert not video_slice.required_inputs["duration"].has_director_default
    assert video_slice.required_inputs["strict_duration"].director_default is True

    ray = require_native_node_contract(
        "DirectorDeckRayInitializerAdvanced"
    ).object_info_contract
    assert not ray.required_inputs["GPU"].has_director_default
    assert not ray.required_inputs["GPU_SELECT"].has_director_default
    assert ray.required_inputs["ulysses_degree"].director_default == 1


@pytest.mark.parametrize(
    "class_type",
    (
        "MiniMaxH3TurboLoRA",
        "LoraLoaderModelOnly",
        "LoraLoaderBypassModelOnly",
    ),
)
def test_user_mapped_lora_is_explicitly_user_assumed(
    class_type: str,
) -> None:
    contract = require_native_node_contract(class_type)
    effect = contract.runtime_effect_contract
    assert effect.policy == "identity_allowed"
    assert effect.unsupported_behavior == "identity"
    assert effect.validation_method == "user_assumed"
    assert any("user" in note.lower() for note in effect.notes)

    implementation = _implementation(
        class_type, contract.supported_runtime_fingerprints[0]
    )
    for output_affecting in (False, True):
        assert (
            V4_NODE_CONTRACT_REGISTRY.validate_implementation(
                implementation,
                output_affecting=output_affecting,
                model_family="fl2va",
                backend="standard",
            )
            == contract
        )


def test_non_lora_identity_contract_cannot_be_output_affecting() -> None:
    contract = require_native_node_contract("SelectModelDevice")
    with pytest.raises(ValueError, match="strict_transform"):
        V4_NODE_CONTRACT_REGISTRY.validate_implementation(
            _implementation(
                contract.class_type,
                contract.supported_runtime_fingerprints[0],
            ),
            output_affecting=True,
            model_family="fl2va",
            backend="standard",
        )


def test_stage8_strict_feature_contracts_are_exact_and_fail_closed() -> None:
    attention = require_current_node_contract(
        "DirectorStrictModelAttentionBackend"
    )
    assert attention.allowed_python_modules == (
        "custom_nodes.DirectorDeck-Strict-Attention",
    )
    assert set(attention.object_info_contract.required_inputs) == {
        "model",
        "mode",
    }
    assert attention.object_info_contract.required_inputs["mode"].enum_values == (
        "pytorch",
        "ck_int8",
    )
    assert tuple(attention.object_info_contract.outputs) == (
        node_contracts._output(0, "MODEL", "model"),
    )

    low_vram = require_current_node_contract(
        "DirectorStrictH3LowVramSagePatch"
    )
    assert low_vram.allowed_python_modules == (
        "custom_nodes.DirectorDeck-Strict-H3",
    )
    assert set(low_vram.object_info_contract.required_inputs) == {"model"}
    assert tuple(low_vram.object_info_contract.outputs) == (
        node_contracts._output(0, "MODEL", "model"),
    )

    expected_validation = {
        attention.class_type: "strict_wrapper",
        low_vram.class_type: "director_owned_implementation",
    }
    for contract in (attention, low_vram):
        effect = contract.runtime_effect_contract
        assert effect.policy == "strict_transform"
        assert effect.unsupported_behavior == "raise"
        assert effect.validation_method == expected_validation[contract.class_type]
        assert effect.verified_model_families == ("fl2va", "ref2va")
        assert effect.verified_backends == ("standard",)
        assert effect.notes
        assert current_release_supported_runtime_fingerprints(
            contract.class_type
        ) == contract.supported_runtime_fingerprints
        implementation = ResolvedImplementationIdentity(
            role=f"feature.{contract.class_type}",
            class_type=contract.class_type,
            implementation_id=contract.contract_id,
            semantic_version=contract.semantic_version,
            runtime_fingerprint=contract.supported_runtime_fingerprints[0],
            binding_key=f"feature.{contract.class_type}",
        )
        assert (
            CURRENT_NODE_CONTRACT_REGISTRY.validate_implementation(
                implementation,
                output_affecting=True,
                model_family="fl2va",
                backend="standard",
            )
            == contract
        )
        with pytest.raises(ValueError, match="not verified for backend"):
            CURRENT_NODE_CONTRACT_REGISTRY.validate_implementation(
                implementation,
                output_affecting=True,
                model_family="fl2va",
                backend="raylight",
            )


def test_identity_device_selectors_cannot_be_reported_as_strict_effects() -> None:
    for class_type in (
        "SelectModelDevice",
        "SelectCLIPDevice",
        "SelectVAEDevice",
    ):
        contract = require_native_node_contract(class_type)
        assert contract.runtime_effect_contract.policy == "identity_allowed"
        with pytest.raises(ValueError, match="strict_transform"):
            V4_NODE_CONTRACT_REGISTRY.validate_implementation(
                _implementation(
                    class_type, contract.supported_runtime_fingerprints[0]
                ),
                output_affecting=True,
                model_family="fl2va",
                backend="standard",
            )


@pytest.mark.parametrize(
    ("class_type", "backend"),
    (
        ("DirectorDeckRayLoraLoader", "raylight"),
        ("DirectorDeckRayUNETLoader", "raylight"),
    ),
)
def test_bundled_lora_chain_has_director_owned_fail_closed_contract(
    class_type: str,
    backend: str,
) -> None:
    contract = require_native_node_contract(class_type)
    effect = contract.runtime_effect_contract
    assert effect.policy == "strict_transform"
    assert effect.unsupported_behavior == "raise"
    assert effect.validation_method == "director_owned_implementation"
    assert effect.verified_backends == (backend,)
    assert effect.notes

    implementation = _implementation(
        class_type, contract.supported_runtime_fingerprints[0]
    )
    assert (
        V4_NODE_CONTRACT_REGISTRY.validate_implementation(
            implementation,
            output_affecting=True,
            model_family="fl2va",
            backend=backend,
        )
        == contract
    )


def test_strict_raylight_attention_contract_rejects_fallback_and_drift() -> None:
    contract = require_native_node_contract("DirectorDeckRayInitializerAdvanced")
    effect = contract.runtime_effect_contract
    assert effect.policy == "strict_transform"
    assert effect.unsupported_behavior == "raise"
    attention = contract.object_info_contract.required_inputs["XFuser_attention"]
    assert attention.enum_values == ("COMFY_KITCHEN_INT8", "TORCH_FLASH")
    assert attention.director_default == "COMFY_KITCHEN_INT8"
    assert any("without attention fallback" in note for note in effect.notes)

    implementation = _implementation(
        contract.class_type, contract.supported_runtime_fingerprints[0]
    )
    assert (
        V4_NODE_CONTRACT_REGISTRY.validate_implementation(
            implementation,
            output_affecting=True,
            model_family="fl2va",
            backend="raylight",
        )
        == contract
    )

    raw = effect.model_dump()
    raw["unsupported_behavior"] = "fallback"
    with pytest.raises(ValidationError, match="must raise"):
        RuntimeEffectContract(**raw)

    with pytest.raises(ValueError, match="adapter identity does not match"):
        V4_NODE_CONTRACT_REGISTRY.validate_implementation(
            _implementation(
                contract.class_type,
                "sha256:" + "f" * 64,
            ),
            output_affecting=True,
            model_family="fl2va",
            backend="raylight",
        )


def test_release_support_fingerprint_api_does_not_construct_host_observation() -> None:
    fingerprint = release_supported_runtime_fingerprints("UNETLoader")[0]
    assert fingerprint.startswith("sha256:")
    assert len(fingerprint) == 71

    # A supported fingerprint is only an allow-list value.  The caller still
    # has to explicitly construct an implementation identity from independently
    # observed host evidence before registry validation can occur.
    with pytest.raises(TypeError):
        _implementation("UNETLoader")  # type: ignore[call-arg]


def test_runtime_fingerprint_material_is_canonical_strict_and_versioned() -> None:
    object_info = require_native_node_contract("UNETLoader").object_info_contract
    material = RuntimeFingerprintMaterial(
        normalized_module_identity="nodes",
        object_info_contract_slice={"UNETLoader": object_info},
        adapter_module_content_digest="sha256:" + "a" * 64,
        package_version="0.33.0",
        director_wrapper_semantic_version="1.0.0",
    )
    restored = RuntimeFingerprintMaterial.model_validate_json(
        material.model_dump_json()
    )
    assert restored == material
    assert compute_runtime_fingerprint(restored) == compute_runtime_fingerprint(
        material
    )

    changed_package = RuntimeFingerprintMaterial(
        **{**material.model_dump(), "package_version": "0.34.0"}
    )
    changed_adapter = RuntimeFingerprintMaterial(
        **{
            **material.model_dump(),
            "director_wrapper_semantic_version": "1.1.0",
        }
    )
    baseline = compute_runtime_fingerprint(material)
    assert compute_runtime_fingerprint(changed_package) != baseline
    assert compute_runtime_fingerprint(changed_adapter) != baseline

    with pytest.raises(TypeError, match="must be validated"):
        compute_runtime_fingerprint(material.model_dump())  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="object-info slice"):
        RuntimeFingerprintMaterial(
            normalized_module_identity="nodes",
            object_info_contract_slice={},
            adapter_module_content_digest="sha256:" + "a" * 64,
            package_version=None,
            director_wrapper_semantic_version="1.0.0",
        )


def test_unknown_class_type_is_rejected_by_all_registry_views() -> None:
    with pytest.raises(KeyError, match="unknown node contract"):
        require_native_node_contract("UnknownNode")
    with pytest.raises(KeyError, match="unknown node contract"):
        native_expected_module_policy(("UnknownNode",))
    with pytest.raises(KeyError, match="unknown node contract"):
        native_provenance_policy(("UnknownNode",))
