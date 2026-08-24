from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import ValidationError

from directordeck.capabilities import (
    CapabilityEvaluator,
    STRICT_ATTENTION_CK_INT8_RUNTIME_PROBE,
    STRICT_ATTENTION_PYTORCH_RUNTIME_PROBE,
    STRICT_H3_SAGE_RUNTIME_PROBE,
    contextual_runtime_capability_id,
    runtime_probe_key,
)
from directordeck.workflow.contracts import (
    CapabilitySet,
    FeatureResolution,
    HostCapabilitySnapshot,
    LogicalGpuCapability,
    RayLightInstallation,
    ResolvedImplementationIdentity,
    RuntimeProbeEvidence,
)
from directordeck.workflow.node_contracts import V5_NODE_CONTRACT_REGISTRY

from .test_plugin_lifecycle import loaded_plugin


@dataclass(frozen=True)
class _Context:
    backend: str = "standard"
    family: str = "fl2va"
    template_bundle_version: int = 5
    binding: object | None = None


def _snapshot(
    evidence: dict[str, RuntimeProbeEvidence] | None = None,
    *,
    schema_version: int = 2,
) -> HostCapabilitySnapshot:
    contract = V5_NODE_CONTRACT_REGISTRY.require(
        "DirectorStrictModelAttentionBackend"
    )
    module = contract.allowed_python_modules[0]
    return HostCapabilitySnapshot(
        schema_version=schema_version,
        generated_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
        node_registry={contract.class_type: module},
        object_info_slices={contract.class_type: contract.object_info_contract},
        module_fingerprints={
            module: contract.supported_runtime_fingerprints[0]
        },
        importable_packages={},
        gpu_inventory=(LogicalGpuCapability(logical_index=0, backend="cuda"),),
        raylight_installation=RayLightInstallation(installed=False),
        media_tool_status={},
        runtime_probe_evidence=evidence or {},
    )


def _resolution() -> FeatureResolution:
    contract = V5_NODE_CONTRACT_REGISTRY.require(
        "DirectorStrictModelAttentionBackend"
    )
    return FeatureResolution(
        state="active",
        implementations=(
            ResolvedImplementationIdentity(
                role="node",
                class_type=contract.class_type,
                implementation_id=contract.contract_id,
                semantic_version=contract.semantic_version,
                runtime_fingerprint=contract.supported_runtime_fingerprints[0],
                binding_key="attention_backend_override.strict",
            ),
        ),
    )


def _evaluate(
    snapshot: HostCapabilitySnapshot,
    runtime_id: str,
):
    class_type = "DirectorStrictModelAttentionBackend"
    return CapabilityEvaluator(V5_NODE_CONTRACT_REGISTRY).evaluate(
        feature_id="attention_backend_override",
        ctx=_Context(binding=SimpleNamespace(device="gpu:0")),
        resolution=_resolution(),
        required_capabilities=CapabilitySet(
            ids=(f"node.{class_type}", runtime_id)
        ),
        snapshot=snapshot,
        readiness=None,
    )


def test_runtime_probe_keys_are_contextual_exact_and_bounded() -> None:
    assert runtime_probe_key(STRICT_ATTENTION_PYTORCH_RUNTIME_PROBE) == (
        "strict_attention.pytorch"
    )
    assert runtime_probe_key(
        STRICT_ATTENTION_CK_INT8_RUNTIME_PROBE,
        device="default",
    ) == "strict_attention.ck_int8.default"
    assert runtime_probe_key(
        STRICT_ATTENTION_CK_INT8_RUNTIME_PROBE,
        device="cpu",
    ) == "strict_attention.ck_int8.cpu"
    assert runtime_probe_key(
        STRICT_H3_SAGE_RUNTIME_PROBE,
        device="gpu:7",
    ) == "strict_h3_sage.gpu_7"
    assert contextual_runtime_capability_id(
        STRICT_H3_SAGE_RUNTIME_PROBE,
        _Context(binding=SimpleNamespace(device="gpu:7")),
    ) == "runtime.strict_h3_sage.gpu_7"
    assert contextual_runtime_capability_id(
        STRICT_H3_SAGE_RUNTIME_PROBE,
        _Context(),
    ) == "runtime.strict_h3_sage.any"

    with pytest.raises(ValueError, match="global runtime probe"):
        runtime_probe_key(
            STRICT_ATTENTION_PYTORCH_RUNTIME_PROBE,
            device="cpu",
        )
    with pytest.raises(ValueError, match="gpu:N"):
        runtime_probe_key(
            STRICT_H3_SAGE_RUNTIME_PROBE,
            device="gpu:256",
        )


def test_schema1_remains_readable_only_with_empty_runtime_evidence() -> None:
    legacy_document = _snapshot(schema_version=1).model_dump(mode="json")
    legacy_document.pop("runtime_probe_evidence")
    restored = HostCapabilitySnapshot.model_validate_json(
        json.dumps(legacy_document)
    )
    assert restored.runtime_probe_evidence == {}

    with pytest.raises(ValidationError, match="schema 1"):
        _snapshot(
            {
                "strict_attention.pytorch": RuntimeProbeEvidence(
                    available=True,
                    code="available",
                )
            },
            schema_version=1,
        )


def test_runtime_evidence_is_strict_private_and_changes_host_revision() -> None:
    available = _snapshot(
        {
            "strict_attention.pytorch": RuntimeProbeEvidence(
                available=True,
                code="available",
            )
        }
    )
    unavailable = _snapshot(
        {
            "strict_attention.pytorch": RuntimeProbeEvidence(
                available=False,
                code="attention_backend_unavailable",
            )
        }
    )
    assert available.host_capability_revision() != (
        unavailable.host_capability_revision()
    )
    with pytest.raises(ValidationError, match="agree"):
        RuntimeProbeEvidence(available=False, code="available")
    with pytest.raises(ValidationError):
        _snapshot(
            {
                "strict_attention.pytorch": RuntimeProbeEvidence(
                    available=False,
                    code="attention_backend_unavailable",
                    architecture="/" + "home/alice/private",
                )
            }
        )


def test_registered_runtime_probe_observations_are_advisory() -> None:
    runtime_id = "runtime.strict_attention.ck_int8.gpu_0"
    available = _evaluate(
        _snapshot(
            {
                "strict_attention.ck_int8.gpu_0": RuntimeProbeEvidence(
                    available=True,
                    code="available",
                    architecture="sm90",
                )
            }
        ),
        runtime_id,
    )
    assert available.available is True

    unavailable = _evaluate(
        _snapshot(
            {
                "strict_attention.ck_int8.gpu_0": RuntimeProbeEvidence(
                    available=False,
                    code="comfy_kitchen_int8_device_unavailable",
                    architecture="sm80",
                )
            }
        ),
        runtime_id,
    )
    assert unavailable.available is True
    assert unavailable.reasons == ()

    missing = _evaluate(_snapshot(), runtime_id)
    assert missing.available is True
    assert missing.reasons == ()
    unknown = _evaluate(
        _snapshot(),
        "runtime.strict_attention.ck_int8.gpu_999",
    )
    assert {reason.code for reason in unknown.reasons} == {"unknown_capability"}


def _probe_module(name: str, source: str) -> ModuleType:
    module = ModuleType(name)
    exec(source, module.__dict__)
    return module


def _install_default_device(
    monkeypatch: pytest.MonkeyPatch,
    *,
    device_type: str,
    device_index: int | None,
    current_cuda_device: int = 0,
) -> None:
    model_management = ModuleType("comfy.model_management")
    model_management.get_torch_device = lambda: SimpleNamespace(
        type=device_type,
        index=device_index,
    )
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        current_device=lambda: current_cuda_device,
    )
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    monkeypatch.setitem(sys.modules, "torch", torch)


def _available_cuda_probe_modules(plugin):
    attention = _probe_module(
        "director_test_default_device_attention",
        """
def director_runtime_capability(mode, device_index=None):
    if mode == 'pytorch':
        return {'available': True, 'code': 'available', 'architecture': None}
    available = type(device_index) is int and device_index >= 0
    return {
        'available': available,
        'code': 'available' if available else 'model_device_not_cuda',
        'architecture': f'gpu{device_index}' if available else None,
    }
""",
    )
    h3 = _probe_module(
        "director_test_default_device_h3",
        """
def director_runtime_capability(device_index=None):
    available = type(device_index) is int and device_index >= 0
    return {
        'available': available,
        'code': 'available' if available else 'model_device_not_cuda',
        'architecture': f'gpu{device_index}' if available else None,
    }
""",
    )
    return {
        plugin._STRICT_ATTENTION_RUNTIME_MODULE: attention,
        plugin._STRICT_H3_RUNTIME_MODULE: h3,
    }


def test_plugin_default_strict_probe_respects_comfy_cpu_placement(
    loaded_plugin,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = loaded_plugin.module
    _install_default_device(
        monkeypatch,
        device_type="cpu",
        device_index=None,
        current_cuda_device=1,
    )
    monkeypatch.setattr(
        plugin,
        "_BUNDLED_NODE_MODULES",
        _available_cuda_probe_modules(plugin),
    )

    evidence = plugin._runtime_probe_evidence(
        (
            SimpleNamespace(logical_index=0, backend="cuda"),
            SimpleNamespace(logical_index=1, backend="cuda"),
        )
    )

    for prefix in ("strict_attention.ck_int8", "strict_h3_sage"):
        assert evidence[f"{prefix}.default"] == {
            "available": False,
            "code": "model_device_not_cuda",
            "architecture": None,
        }
        assert evidence[f"{prefix}.gpu_0"]["available"] is True
        assert evidence[f"{prefix}.gpu_1"]["available"] is True
        assert evidence[f"{prefix}.any"]["available"] is True


@pytest.mark.parametrize(
    ("device_index", "current_cuda_device", "expected_index"),
    ((1, 0, 1), (None, 0, 0)),
)
def test_plugin_default_strict_probe_maps_comfy_cuda_device_exactly(
    loaded_plugin,
    monkeypatch: pytest.MonkeyPatch,
    device_index: int | None,
    current_cuda_device: int,
    expected_index: int,
) -> None:
    plugin = loaded_plugin.module
    _install_default_device(
        monkeypatch,
        device_type="cuda",
        device_index=device_index,
        current_cuda_device=current_cuda_device,
    )
    monkeypatch.setattr(
        plugin,
        "_BUNDLED_NODE_MODULES",
        _available_cuda_probe_modules(plugin),
    )

    evidence = plugin._runtime_probe_evidence(
        (
            SimpleNamespace(logical_index=0, backend="cuda"),
            SimpleNamespace(logical_index=1, backend="cuda"),
        )
    )

    for prefix in ("strict_attention.ck_int8", "strict_h3_sage"):
        assert evidence[f"{prefix}.default"] == {
            "available": True,
            "code": "available",
            "architecture": f"gpu{expected_index}",
        }


def test_plugin_uses_exact_loaded_hooks_and_sanitizes_probe_failures(
    loaded_plugin,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = loaded_plugin.module
    attention = _probe_module(
        "director_test_strict_attention",
        """
CK_DEVICE = 1
def director_runtime_capability(mode, device_index=None):
    if mode == 'pytorch':
        return {'available': True, 'code': 'available', 'architecture': None}
    if device_index == 0:
        raise RuntimeError('/' + 'home/alice/private driver failure')
    available = device_index == CK_DEVICE
    return {
        'available': available,
        'code': 'available' if available else 'ck_device_unavailable',
        'architecture': 'sm90' if available else None,
    }
""",
    )
    h3 = _probe_module(
        "director_test_strict_h3",
        """
def director_runtime_capability(device_index=None):
    available = device_index == 0
    return {
        'available': available,
        'code': 'available' if available else 'sage_device_unavailable',
        'architecture': 'sm80' if available else None,
    }
""",
    )
    monkeypatch.setattr(
        plugin,
        "_BUNDLED_NODE_MODULES",
        {
            plugin._STRICT_ATTENTION_RUNTIME_MODULE: attention,
            plugin._STRICT_H3_RUNTIME_MODULE: h3,
        },
    )

    evidence = plugin._runtime_probe_evidence(
        (
            SimpleNamespace(logical_index=0, backend="cuda"),
            SimpleNamespace(logical_index=1, backend="cuda"),
        )
    )

    assert evidence["strict_attention.pytorch"]["available"] is True
    assert evidence["strict_attention.ck_int8.gpu_0"] == {
        "available": False,
        "code": "runtime_probe_failed",
        "architecture": None,
    }
    assert evidence["strict_attention.ck_int8.gpu_1"]["available"] is True
    assert evidence["strict_attention.ck_int8.any"]["available"] is True
    assert evidence["strict_h3_sage.gpu_0"]["available"] is True
    assert evidence["strict_h3_sage.any"]["available"] is True
    assert "/home/alice" not in str(evidence)

    # A callable merely assigned onto the module is not proof that the loaded
    # bundled module owns the audited hook.
    attention.director_runtime_capability = lambda *_args: {
        "available": True,
        "code": "available",
        "architecture": None,
    }
    rejected = plugin._runtime_probe_evidence(())
    assert rejected["strict_attention.pytorch"]["code"] == (
        "runtime_probe_hook_unavailable"
    )


def test_plugin_snapshot_schema2_binds_runtime_evidence_into_revision(
    loaded_plugin,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from .test_plugin_host_capabilities import (
        _provider,
        _raw_object_info,
        _registry_for_object_info,
        _stabilize_environment,
    )

    plugin = loaded_plugin.module
    _stabilize_environment(plugin, monkeypatch)
    attention = _probe_module(
        "director_test_snapshot_attention",
        """
PYTORCH_AVAILABLE = True
def director_runtime_capability(mode, device_index=None):
    if mode == 'pytorch':
        return {
            'available': PYTORCH_AVAILABLE,
            'code': 'available' if PYTORCH_AVAILABLE else 'pytorch_unavailable',
            'architecture': None,
        }
    return {'available': False, 'code': 'ck_unavailable', 'architecture': None}
""",
    )
    h3 = _probe_module(
        "director_test_snapshot_h3",
        """
def director_runtime_capability(device_index=None):
    return {'available': False, 'code': 'sage_unavailable', 'architecture': None}
""",
    )
    monkeypatch.setattr(
        plugin,
        "_BUNDLED_NODE_MODULES",
        {
            plugin._STRICT_ATTENTION_RUNTIME_MODULE: attention,
            plugin._STRICT_H3_RUNTIME_MODULE: h3,
        },
    )
    object_info = _raw_object_info()
    registry = _registry_for_object_info(object_info)
    provider = _provider(
        plugin,
        object_info=object_info,
        registry=registry,
        generated_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
    )

    first = provider.snapshot()
    assert first.schema_version == 2
    assert first.runtime_probe_evidence[
        "strict_attention.pytorch"
    ].available is True
    serialized = first.model_dump_json()
    assert "/home/" not in serialized
    assert "private" not in serialized

    attention.PYTORCH_AVAILABLE = False
    cached = provider.snapshot()
    assert cached is first
    assert cached.runtime_probe_evidence[
        "strict_attention.pytorch"
    ].code == "available"

    changed = _provider(
        plugin,
        object_info=object_info,
        registry=registry,
        generated_at=datetime(2026, 8, 22, 1, tzinfo=timezone.utc),
    ).snapshot()
    assert changed.runtime_probe_evidence[
        "strict_attention.pytorch"
    ].code == "pytorch_unavailable"
    assert changed.host_capability_revision() != first.host_capability_revision()
