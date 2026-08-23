from __future__ import annotations

import copy
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from directordeck.workflow.node_contracts import (
    CURRENT_NODE_CONTRACT_REGISTRY,
    V4_NODE_CONTRACT_REGISTRY,
)

from . import extensible_workflow_v0_fixture_builder as fixture_builder
from .test_plugin_lifecycle import loaded_plugin


def _raw_input(input_contract: Any) -> list[Any]:
    if input_contract.port_type == "DYNAMIC_COMBO":
        values = input_contract.enum_values or ("auto",)
        return [
            "COMFY_DYNAMICCOMBO_V3",
            {
                "options": [{"key": value, "inputs": {}} for value in values],
                "default": "private-token-that-must-not-be-copied",
            },
        ]
    if input_contract.port_type == "COMBO":
        values = input_contract.enum_values or (
            "/" + "home/alice/private-model.safetensors",
        )
        return ["COMBO", {"options": list(values)}]
    return [
        input_contract.port_type,
        {"default": "/" + "home/alice/private-widget-default"},
    ]


def _raw_object_info(*, current: bool = False) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    registry = (
        CURRENT_NODE_CONTRACT_REGISTRY
        if current
        else V4_NODE_CONTRACT_REGISTRY
    )
    for class_type, contract in registry.contracts.items():
        object_info = contract.object_info_contract
        result[class_type] = {
            "python_module": contract.allowed_python_modules[0],
            "input": {
                "required": {
                    name: _raw_input(value)
                    for name, value in object_info.required_inputs.items()
                },
                "optional": {
                    name: _raw_input(value)
                    for name, value in object_info.optional_inputs.items()
                },
            },
            "output": [item.port_type for item in object_info.outputs],
            "output_name": [
                item.name or item.port_type for item in object_info.outputs
            ],
            "output_is_list": [item.is_list for item in object_info.outputs],
            "output_node": object_info.output_node,
        }
    return result


def _use_real_v3_h3_reference_autogrow_shape(
    object_info: dict[str, dict[str, Any]],
) -> None:
    """Mirror ComfyUI's V3 /object_info shape for H3 reference groups."""

    def autogrow(
        *,
        input_id: str,
        port_type: str,
        prefix: str,
        maximum: int,
    ) -> list[object]:
        return [
            "COMFY_AUTOGROW_V3",
            {
                "template": {
                    "input": {
                        "required": {
                            input_id: [port_type, {"forceInput": True}],
                        }
                    },
                    "prefix": prefix,
                    "min": 0,
                    "max": maximum,
                }
            },
        ]

    object_info["MiniMaxH3ReferenceToVideo"]["input"]["optional"] = {
        "ref_images": autogrow(
            input_id="ref_image",
            port_type="IMAGE",
            prefix="ref_image_",
            maximum=9,
        ),
        "ref_videos": autogrow(
            input_id="ref_video",
            port_type="IMAGE",
            prefix="ref_video_",
            maximum=3,
        ),
        "ref_video_audios": autogrow(
            input_id="ref_video_audio",
            port_type="AUDIO",
            prefix="ref_video_audio_",
            maximum=3,
        ),
        "ref_audios": autogrow(
            input_id="ref_audio",
            port_type="AUDIO",
            prefix="ref_audio_",
            maximum=3,
        ),
    }


def _registry_for_object_info(
    object_info: dict[str, dict[str, Any]],
) -> dict[str, object]:
    registry: dict[str, object] = {}
    for class_type, raw in object_info.items():
        python_module = raw.get("python_module")
        if not isinstance(python_module, str):
            python_module = "host.interface"
        node_class = type(
            f"Host_{class_type}",
            (),
            {"__module__": python_module},
        )
        node_class.RELATIVE_PYTHON_MODULE = python_module
        registry[class_type] = node_class
    return registry


def _provider(
    plugin: Any,
    *,
    object_info: dict[str, dict[str, Any]],
    registry: dict[str, object],
    generated_at: datetime,
) -> Any:
    return plugin._ComfyHostCapabilityProvider(
        comfy_url="http://127.0.0.1:8188",
        object_info_loader=lambda: object_info,
        node_registry_loader=lambda: registry,
        generated_at_factory=lambda: generated_at,
    )


def _stabilize_environment(plugin: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        plugin,
        "_package_capabilities",
        lambda: (
            ("ray", True, "2.49.0"),
            ("static_ffmpeg", False, None),
            ("torch", True, "2.8.0"),
            ("xfuser", True, "0.4.3"),
        ),
    )
    monkeypatch.setattr(
        plugin,
        "_logical_gpu_inventory",
        lambda: (
            {
                "logical_index": 0,
                "backend": "cuda",
                "total_memory_mb": 24_576,
            },
            {
                "logical_index": 1,
                "backend": "cuda",
                "total_memory_mb": 24_576,
            },
        ),
    )
    monkeypatch.setattr(
        plugin,
        "_media_tool_capabilities",
        lambda: (("ffmpeg", True, "7.1"), ("ffprobe", True, "7.1")),
    )
    plugin._state.raylight = "registered"


def test_provider_reports_available_nodes_without_source_fingerprint_authorization(
    loaded_plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = loaded_plugin.module
    _stabilize_environment(plugin, monkeypatch)
    object_info = _raw_object_info()
    registry = _registry_for_object_info(object_info)
    generated_at = datetime(2026, 8, 21, 12, 30, tzinfo=timezone.utc)

    snapshot = _provider(
        plugin,
        object_info=object_info,
        registry=registry,
        generated_at=generated_at,
    ).snapshot()

    assert snapshot.generated_at == generated_at
    assert set(snapshot.node_registry) == set(V4_NODE_CONTRACT_REGISTRY.contracts)
    assert snapshot.module_fingerprints == {}
    assert snapshot.raylight_installation.installed is True
    assert snapshot.raylight_installation.node_contracts_available is True
    assert snapshot.raylight_installation.package_version == "1.8.0+director.1"
    assert tuple(item.logical_index for item in snapshot.gpu_inventory) == (0, 1)

    serialized = snapshot.model_dump_json()
    assert "/home/alice" not in serialized
    assert "private-token" not in serialized
    assert "queue" not in serialized
    later = _provider(
        plugin,
        object_info=object_info,
        registry=registry,
        generated_at=generated_at + timedelta(hours=1),
    ).snapshot()
    assert later.host_capability_revision() == snapshot.host_capability_revision()


def test_provider_caches_process_static_snapshot(
    loaded_plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = loaded_plugin.module
    _stabilize_environment(plugin, monkeypatch)
    object_info = _raw_object_info()
    registry = _registry_for_object_info(object_info)
    calls = {"object_info": 0, "registry": 0, "generated_at": 0}

    def load_object_info() -> dict[str, dict[str, Any]]:
        calls["object_info"] += 1
        return object_info

    def load_registry() -> dict[str, object]:
        calls["registry"] += 1
        return registry

    def generated_at() -> datetime:
        calls["generated_at"] += 1
        return datetime(2026, 8, 21, 12, 30, tzinfo=timezone.utc)

    provider = plugin._ComfyHostCapabilityProvider(
        comfy_url="http://127.0.0.1:8188",
        object_info_loader=load_object_info,
        node_registry_loader=load_registry,
        generated_at_factory=generated_at,
    )

    first = provider.snapshot()
    second = provider.snapshot()

    assert second is first
    assert calls == {"object_info": 1, "registry": 1, "generated_at": 1}


def test_provider_invalidate_recaptures_after_preserving_cached_identity(
    loaded_plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = loaded_plugin.module
    _stabilize_environment(plugin, monkeypatch)
    object_info = _raw_object_info()
    registry = _registry_for_object_info(object_info)
    calls = {"object_info": 0, "registry": 0, "generated_at": 0}
    generated_at_values = iter(
        (
            datetime(2026, 8, 21, 12, 30, tzinfo=timezone.utc),
            datetime(2026, 8, 21, 12, 31, tzinfo=timezone.utc),
        )
    )

    def load_object_info() -> dict[str, dict[str, Any]]:
        calls["object_info"] += 1
        return object_info

    def load_registry() -> dict[str, object]:
        calls["registry"] += 1
        return registry

    def generated_at() -> datetime:
        calls["generated_at"] += 1
        return next(generated_at_values)

    provider = plugin._ComfyHostCapabilityProvider(
        comfy_url="http://127.0.0.1:8188",
        object_info_loader=load_object_info,
        node_registry_loader=load_registry,
        generated_at_factory=generated_at,
    )

    first = provider.snapshot()
    cached = provider.snapshot()
    provider.invalidate()
    refreshed = provider.snapshot()

    assert cached is first
    assert refreshed is not first
    assert refreshed.generated_at == datetime(
        2026,
        8,
        21,
        12,
        31,
        tzinfo=timezone.utc,
    )
    assert provider.snapshot() is refreshed
    assert calls == {"object_info": 2, "registry": 2, "generated_at": 2}


def test_provider_module_fingerprint_does_not_depend_on_optional_node_presence(
    loaded_plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from directordeck.capabilities import (
        build_operational_readiness,
        preflight_v4_timeline,
    )

    from . import extensible_workflow_v0_fixture_builder as fixture_builder

    plugin = loaded_plugin.module
    _stabilize_environment(plugin, monkeypatch)
    object_info = _raw_object_info()
    registry = _registry_for_object_info(object_info)
    object_info.pop("DirectorDeckRayLoraLoader")
    registry.pop("DirectorDeckRayLoraLoader")

    snapshot = _provider(
        plugin,
        object_info=object_info,
        registry=registry,
        generated_at=datetime(2026, 8, 21, tzinfo=timezone.utc),
    ).snapshot()

    assert "DirectorDeckRayLoraLoader" not in snapshot.node_registry
    assert snapshot.module_fingerprints == {}
    report = preflight_v4_timeline(
        draft=fixture_builder._draft("t2v"),
        settings=fixture_builder._settings("raylight"),
        snapshot=snapshot,
        readiness=build_operational_readiness(
            endpoint_online=True,
            available_logical_gpu_count=2,
        ),
    )
    assert report.valid is True


def test_provider_ignores_interface_node_provenance_but_rejects_bad_interface(
    loaded_plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = loaded_plugin.module
    _stabilize_environment(plugin, monkeypatch)
    object_info = _raw_object_info()
    registry = _registry_for_object_info(object_info)

    registry.pop("SaveVideo")
    object_info["ImageFromBatch"]["python_module"] = "custom_nodes.spoofed"
    registry["LoadAudio"].RELATIVE_PYTHON_MODULE = "custom_nodes.spoofed"
    object_info["UNETLoader"]["input"]["required"]["new_required_input"] = [
        "STRING",
        {},
    ]
    snapshot = _provider(
        plugin,
        object_info=object_info,
        registry=registry,
        generated_at=datetime(2026, 8, 21, tzinfo=timezone.utc),
    ).snapshot()

    assert "SaveVideo" not in snapshot.node_registry
    assert snapshot.node_registry["ImageFromBatch"] == "custom_nodes.spoofed"
    assert "LoadAudio" in snapshot.node_registry
    assert "UNETLoader" in snapshot.node_registry
    assert "UNETLoader" not in snapshot.object_info_slices
    assert "comfy_extras.nodes_images" not in snapshot.module_fingerprints
    assert "nodes" not in snapshot.module_fingerprints


def test_provider_accepts_user_mapped_third_party_module_without_source_attestation(
    loaded_plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = loaded_plugin.module
    _stabilize_environment(plugin, monkeypatch)
    object_info = _raw_object_info()
    class_type = "LoraLoaderModelOnly"
    object_info[class_type]["python_module"] = "custom_nodes.user_lora_pack"
    registry = _registry_for_object_info(object_info)
    registry[class_type].__module__ = "arbitrary.rewritten.source_label"
    registry[class_type].RELATIVE_PYTHON_MODULE = "another.user.label"

    snapshot = _provider(
        plugin,
        object_info=object_info,
        registry=registry,
        generated_at=datetime(2026, 8, 21, tzinfo=timezone.utc),
    ).snapshot()

    assert snapshot.node_registry[class_type] == "custom_nodes.user_lora_pack"
    assert class_type in snapshot.object_info_slices
    assert "custom_nodes.user_lora_pack" not in snapshot.module_fingerprints


@pytest.mark.parametrize("raw_module", (None, "../../private/source.py"))
def test_interface_only_nodes_ignore_missing_or_invalid_python_module(
    loaded_plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
    raw_module: object,
) -> None:
    plugin = loaded_plugin.module
    _stabilize_environment(plugin, monkeypatch)
    object_info = _raw_object_info()
    class_type = "LoraLoaderModelOnly"
    if raw_module is None:
        object_info[class_type].pop("python_module")
    else:
        object_info[class_type]["python_module"] = raw_module
    registry = _registry_for_object_info(object_info)

    snapshot = _provider(
        plugin,
        object_info=object_info,
        registry=registry,
        generated_at=datetime(2026, 8, 21, tzinfo=timezone.utc),
    ).snapshot()

    assert snapshot.node_registry[class_type] == "host.interface"
    assert class_type in snapshot.object_info_slices


def test_provider_accepts_real_v3_autogrow_group_shape_for_h3_reference_inputs(
    loaded_plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = loaded_plugin.module
    _stabilize_environment(plugin, monkeypatch)
    object_info = _raw_object_info()
    _use_real_v3_h3_reference_autogrow_shape(object_info)
    registry = _registry_for_object_info(object_info)

    snapshot = _provider(
        plugin,
        object_info=object_info,
        registry=registry,
        generated_at=datetime(2026, 8, 21, tzinfo=timezone.utc),
    ).snapshot()

    class_type = "MiniMaxH3ReferenceToVideo"
    assert class_type in snapshot.node_registry
    assert class_type in snapshot.object_info_slices

    object_info[class_type]["input"]["optional"]["ref_images"][1][
        "template"
    ]["max"] = 8
    incompatible = _provider(
        plugin,
        object_info=object_info,
        registry=registry,
        generated_at=datetime(2026, 8, 21, 1, tzinfo=timezone.utc),
    ).snapshot()
    assert class_type in incompatible.node_registry
    assert class_type not in incompatible.object_info_slices


def test_provider_missing_conditional_raylight_node_preserves_exact_active_contracts(
    loaded_plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from directordeck.capabilities import (
        build_operational_readiness,
        preflight_v4_timeline,
    )

    plugin = loaded_plugin.module
    _stabilize_environment(plugin, monkeypatch)
    object_info = _raw_object_info()
    registry = _registry_for_object_info(object_info)
    object_info.pop("DirectorDeckRayLoraLoader")
    registry.pop("DirectorDeckRayLoraLoader")

    snapshot = _provider(
        plugin,
        object_info=object_info,
        registry=registry,
        generated_at=datetime(2026, 8, 21, tzinfo=timezone.utc),
    ).snapshot()
    assert "DirectorDeckRayLoraLoader" not in snapshot.node_registry
    assert snapshot.raylight_installation.installed is True
    assert snapshot.raylight_installation.node_contracts_available is False
    assert snapshot.module_fingerprints == {}

    readiness = build_operational_readiness(
        endpoint_online=True,
        available_logical_gpu_count=2,
    )
    no_lora = preflight_v4_timeline(
        draft=fixture_builder._draft("t2v"),
        settings=fixture_builder._settings("raylight"),
        snapshot=snapshot,
        readiness=readiness,
    )
    assert no_lora.valid is True

    with_lora = preflight_v4_timeline(
        draft=fixture_builder._draft("t2v"),
        settings=fixture_builder._settings(
            "raylight",
            lora_family="fl2va",
            lora_name="conditional-ray-lora.safetensors",
        ),
        snapshot=snapshot,
        readiness=readiness,
    )
    assert with_lora.valid is False
    assert {
        (error.code, error.feature_id)
        for error in with_lora.errors
    } == {("node_unavailable", "lora")}


def test_provider_never_uses_provenance_or_source_as_capability_authority(
    loaded_plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = loaded_plugin.module
    _stabilize_environment(plugin, monkeypatch)
    object_info = _raw_object_info(current=True)
    registry = _registry_for_object_info(object_info)
    class_type = "DirectorStrictModelAttentionBackend"
    registry[class_type] = type(
        "UserModifiedAttention",
        (),
        {"RELATIVE_PYTHON_MODULE": "custom_nodes.user_modified"},
    )
    object_info[class_type]["python_module"] = "custom_nodes.user_modified"

    snapshot = _provider(
        plugin,
        object_info=object_info,
        registry=registry,
        generated_at=datetime(2026, 8, 21, tzinfo=timezone.utc),
    ).snapshot()

    assert snapshot.node_registry[class_type] == "custom_nodes.user_modified"
    assert class_type in snapshot.object_info_slices
    assert snapshot.module_fingerprints == {}


def test_logical_gpu_inventory_uses_only_logical_indices_and_memory(
    loaded_plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = loaded_plugin.module

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def get_device_properties(index: int) -> Any:
            return SimpleNamespace(
                total_memory=(index + 1) * 8 * 1024 * 1024 * 1024,
                name=f"Private GPU {index}",
                uuid=f"GPU-private-{index}",
            )

    real_import = plugin.importlib.import_module
    monkeypatch.setattr(
        plugin.importlib,
        "import_module",
        lambda name: SimpleNamespace(cuda=FakeCuda())
        if name == "torch"
        else real_import(name),
    )

    inventory = plugin._logical_gpu_inventory()

    assert inventory == (
        {"logical_index": 0, "backend": "cuda", "total_memory_mb": 8192},
        {"logical_index": 1, "backend": "cuda", "total_memory_mb": 16384},
    )
    rendered = json.dumps(inventory)
    assert "Private GPU" not in rendered
    assert "GPU-private" not in rendered


def test_media_capabilities_require_successful_version_and_exact_encoders(
    loaded_plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = loaded_plugin.module
    monkeypatch.setattr(
        plugin.shutil,
        "which",
        lambda name: f"/usr/bin/{name}",
    )
    calls: list[list[str]] = []

    def successful_run(argv: list[str], **_kwargs: Any) -> Any:
        calls.append(argv)
        if argv[-1] == "-version":
            name = argv[0].rsplit("/", 1)[-1]
            return subprocess.CompletedProcess(
                argv, 0, stdout=f"{name} version 7.1-static\n", stderr=""
            )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                " V....D libx264              libx264 H.264 encoder\n"
                " A..... aac                  AAC encoder\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(plugin.subprocess, "run", successful_run)

    assert plugin._media_tool_capabilities() == (
        ("ffmpeg", True, "7.1-static"),
        ("ffprobe", True, "7.1-static"),
    )
    assert calls == [
        ["/usr/bin/ffmpeg", "-version"],
        ["/usr/bin/ffmpeg", "-hide_banner", "-encoders"],
        ["/usr/bin/ffprobe", "-version"],
    ]


def test_media_capabilities_allow_successful_vendor_version_without_parsing(
    loaded_plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = loaded_plugin.module
    monkeypatch.setattr(
        plugin.shutil,
        "which",
        lambda name: f"/usr/bin/{name}",
    )

    def vendor_run(argv: list[str], **_kwargs: Any) -> Any:
        if argv[-1] == "-version":
            return subprocess.CompletedProcess(
                argv, 0, stdout="Vendor media tools build unknown\n", stderr=""
            )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=" V....D libx264\n A..... aac\n",
            stderr="",
        )

    monkeypatch.setattr(plugin.subprocess, "run", vendor_run)

    assert plugin._media_tool_capabilities() == (
        ("ffmpeg", True, None),
        ("ffprobe", True, None),
    )


@pytest.mark.parametrize(
    ("failed_argv_suffix", "returncode", "encoder_listing"),
    (
        (("ffmpeg", "-version"), 1, None),
        (("ffmpeg", "-encoders"), 1, None),
        (("ffmpeg", "-encoders"), 0, " V....D libx264\n"),
        (("ffprobe", "-version"), 1, None),
    ),
    ids=(
        "ffmpeg-version-failed",
        "ffmpeg-encoder-probe-failed",
        "ffmpeg-required-encoder-missing",
        "ffprobe-version-failed",
    ),
)
def test_media_capabilities_fail_closed_on_unusable_binaries(
    loaded_plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
    failed_argv_suffix: tuple[str, str],
    returncode: int,
    encoder_listing: str | None,
) -> None:
    plugin = loaded_plugin.module
    monkeypatch.setattr(
        plugin.shutil,
        "which",
        lambda name: f"/usr/bin/{name}",
    )

    def probe(argv: list[str], **_kwargs: Any) -> Any:
        executable_name = argv[0].rsplit("/", 1)[-1]
        operation = argv[-1]
        failed = (executable_name, operation) == failed_argv_suffix
        if operation == "-version":
            return subprocess.CompletedProcess(
                argv,
                returncode if failed else 0,
                stdout=f"{executable_name} version 7.1\n",
                stderr="failed" if failed else "",
            )
        listing = encoder_listing if failed and encoder_listing is not None else (
            " V....D libx264\n A..... aac\n"
        )
        return subprocess.CompletedProcess(
            argv,
            returncode if failed else 0,
            stdout=listing,
            stderr="failed" if failed else "",
        )

    monkeypatch.setattr(plugin.subprocess, "run", probe)

    capabilities = dict(
        (name, (available, version))
        for name, available, version in plugin._media_tool_capabilities()
    )
    failed_tool = failed_argv_suffix[0]
    available, version = capabilities[failed_tool]
    assert available is False
    if failed_argv_suffix[1] == "-version":
        assert version is None
    else:
        assert version == "7.1"


def test_backend_injection_uses_the_read_only_host_capability_provider(
    loaded_plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    from .test_plugin_lifecycle import install_fake_backend, start_fake_backend

    plugin = loaded_plugin.module
    control = install_fake_backend(monkeypatch, plugin)
    thread = start_fake_backend(plugin, tmp_path / "director.sqlite3")
    assert control.started.wait(timeout=1)
    try:
        assert control.app_kwargs is not None
        provider = control.app_kwargs["host_capability_provider"]
        assert isinstance(provider, plugin._ComfyHostCapabilityProvider)
        assert provider._comfy_url == "http://127.0.0.1:8188"
    finally:
        control.release.set()
        thread.join(timeout=1)
        assert not thread.is_alive()
