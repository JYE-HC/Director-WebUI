"""DirectorDeck plugin entry.

Embeds the Director Web backend (FastAPI) into the ComfyUI process:

- runs the Director backend with uvicorn on a daemon thread bound to
  127.0.0.1 (internal loopback only, never user facing);
- serves the built frontend from ``dist/`` under ``/directordeck/``;
- reverse-proxies ``/directordeck/api/*`` to the internal backend with fully
  streamed request/response bodies (SSE, Range media, large uploads);
- exposes ``/directordeck/status`` for the menu extension and the SPA to learn
  the explicit ``starting``/``ready``/``stopped``/``failed`` backend state
  without touching the proxy;
- injects this ComfyUI instance's loopback address into the backend at
  construction; the ComfyUI address is not a setting and is never persisted;
- registers Director-owned strict attention and H3 implementation nodes, plus
  the bundled Director-maintained RayLight fork behind a
  platform/dependency gate (multi-GPU is opt-in, see docs); Standard
  LoRA loader nodes remain user-installed and are selected by exact mapping.

The Director backend stays a pure HTTP/WS client of ComfyUI; nothing here
reaches into ComfyUI internals beyond the documented plugin surface
(PromptServer routes, folder_paths, cli args).
"""

from __future__ import annotations

import asyncio
import atexit
import importlib
import importlib.metadata
import importlib.util
import inspect
import ipaddress
import json
import logging
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import tomllib
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import ModuleType
from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener

from aiohttp import web

LOGGER = logging.getLogger("DirectorDeck")

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

_PLUGIN_ROOT = Path(__file__).resolve().parent
_BACKEND_PATH = _PLUGIN_ROOT / "backend"
_DIST_DIR = _PLUGIN_ROOT / "dist"
_NODES_DIR = _PLUGIN_ROOT / "nodes"

_DEFAULT_INTERNAL_PORT = 18788
_PORT_SCAN_LIMIT = 20
_ATEXIT_JOIN_TIMEOUT_SECONDS = 10.0
_COMFY_SHUTDOWN_JOIN_TIMEOUT_SECONDS = 8.0
# One identity per loaded ComfyUI process. Restarting only the embedded backend
# thread keeps it stable; a full ComfyUI restart necessarily creates a new one.
_COMFY_BOOT_RUNTIME_INSTANCE_ID = str(uuid.uuid4())
_VIDEO_OUTPUT_EXTENSIONS = frozenset(
    {".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi", ".mpeg", ".mpg"}
)
_HOST_CAPABILITY_SCHEMA_VERSION = 2
_OBJECT_INFO_LIMIT_BYTES = 32 * 1024 * 1024
_HOST_CAPABILITY_HTTP_TIMEOUT_SECONDS = 10.0
_SAFE_VERSION = re.compile(r"^[0-9A-Za-z][0-9A-Za-z.+_-]{0,127}$")
_SAFE_MODULE_IDENTITY = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,255}$")
_SAFE_RUNTIME_PROBE_VALUE = re.compile(
    r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$"
)
_IMPORTABILITY_PACKAGES = (
    ("ray", "ray", "ray"),
    ("static_ffmpeg", "static_ffmpeg", "static-ffmpeg"),
    ("torch", "torch", "torch"),
    ("xfuser", "xfuser", "xfuser"),
)
# Retain loaded modules only so their optional runtime probes can be called.
# Node availability is determined by namespaced class presence, never source or
# Python object identity.
_BUNDLED_NODE_MODULES: dict[str, ModuleType] = {}
_PREEXISTING_HOST_NODE_MAPPINGS: dict[str, object] = {}

_STRICT_ATTENTION_RUNTIME_MODULE = (
    "custom_nodes.DirectorDeck-Strict-Attention"
)
_STRICT_H3_RUNTIME_MODULE = "custom_nodes.DirectorDeck-Strict-H3"
_DIRECTOR_RAYLIGHT_RUNTIME_MODULE = "custom_nodes.DirectorDeck-RayLight"
_DIRECTOR_RAYLIGHT_CLASS_TYPE_ALIASES = {
    "RayInitializerAdvanced": "DirectorDeckRayInitializerAdvanced",
    "RayLoraLoader": "DirectorDeckRayLoraLoader",
    "RayUNETLoader": "DirectorDeckRayUNETLoader",
    "RayMiniMaxH3SigmaShift": "DirectorDeckRayMiniMaxH3SigmaShift",
    "RayBasicGuider": "DirectorDeckRayBasicGuider",
    "RayBasicScheduler": "DirectorDeckRayBasicScheduler",
    "XFuserSamplerCustomAdvanced": "DirectorDeckRayXFuserSamplerCustomAdvanced",
    "RayKill": "DirectorDeckRayKill",
}

_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


class _BackendState:
    def __init__(self) -> None:
        self.status = "starting"
        self.error: str | None = None
        self.port: int | None = None
        self.version: str | None = None
        self.database_path: str | None = None
        self.server = None  # uvicorn.Server once constructed
        self.thread: threading.Thread | None = None
        # RayLight gate outcome: registered | deps_missing | platform_unsupported
        # | pack_missing | load_failed
        self.raylight = "unknown"
        self.raylight_detail: str | None = None
        self.nodes_error: str | None = None


_state = _BackendState()
_proxy_session = None
_proxy_session_lock = threading.Lock()


def _unavailable_runtime_probe(code: str) -> dict[str, object]:
    return {"available": False, "code": code, "architecture": None}


def _call_bundled_runtime_probe(
    logical_module: str,
    *args: object,
) -> dict[str, object]:
    """Call only the exact function owned by an actually loaded bundled pack."""

    module = _BUNDLED_NODE_MODULES.get(logical_module)
    if not isinstance(module, ModuleType):
        return _unavailable_runtime_probe("runtime_probe_module_unavailable")
    hook = getattr(module, "director_runtime_capability", None)
    if (
        not inspect.isfunction(hook)
        or getattr(hook, "__globals__", None) is not module.__dict__
    ):
        return _unavailable_runtime_probe("runtime_probe_hook_unavailable")
    try:
        raw = hook(*args)
    except Exception:
        return _unavailable_runtime_probe("runtime_probe_failed")
    if type(raw) is not dict or set(raw) != {
        "available",
        "code",
        "architecture",
    }:
        return _unavailable_runtime_probe("runtime_probe_invalid")
    available = raw.get("available")
    code = raw.get("code")
    architecture = raw.get("architecture")
    if (
        type(available) is not bool
        or not isinstance(code, str)
        or _SAFE_RUNTIME_PROBE_VALUE.fullmatch(code) is None
        or (
            architecture is not None
            and (
                not isinstance(architecture, str)
                or _SAFE_RUNTIME_PROBE_VALUE.fullmatch(architecture) is None
            )
        )
        or available != (code == "available")
    ):
        return _unavailable_runtime_probe("runtime_probe_invalid")
    return {
        "available": available,
        "code": code,
        "architecture": architecture,
    }


def _aggregate_runtime_probes(
    observations: tuple[dict[str, object], ...],
) -> dict[str, object]:
    if not observations:
        return _unavailable_runtime_probe("runtime_probe_evidence_missing")
    for observation in observations:
        if observation["available"] is True:
            return dict(observation)
    # Deterministic ordering is established by the provider: default, CPU,
    # then logical GPUs.  The aggregate preserves one stable reason without
    # exposing raw host diagnostics.
    return dict(observations[0])


def _comfy_default_cuda_probe_target() -> tuple[int | None, str | None]:
    """Resolve ComfyUI's actual default model device for strict CUDA probes.

    CUDA visibility is not placement authority: ComfyUI may be running with
    ``--cpu`` while ``torch.cuda`` can still enumerate devices.  Only the
    in-process model-management selection can authorize the ``default`` probe.
    Unknown and non-CUDA selections fail closed without exposing host details.
    """

    try:
        model_management = importlib.import_module("comfy.model_management")
        get_torch_device = getattr(model_management, "get_torch_device", None)
        if not callable(get_torch_device):
            return None, "cuda_device_unavailable"
        device = get_torch_device()
        device_type = getattr(device, "type", None)
        if device_type != "cuda":
            return None, "model_device_not_cuda"
        device_index = getattr(device, "index", None)
        if device_index is None:
            torch = importlib.import_module("torch")
            cuda = getattr(torch, "cuda", None)
            current_device = getattr(cuda, "current_device", None)
            if not callable(current_device):
                return None, "cuda_device_unavailable"
            device_index = current_device()
    except Exception:
        return None, "cuda_device_unavailable"

    if type(device_index) is not int or not (0 <= device_index <= 255):
        return None, "cuda_device_unavailable"
    return device_index, None


def _runtime_probe_evidence(
    gpu_inventory: tuple[object, ...],
) -> dict[str, dict[str, object]]:
    """Collect exact strict-runtime evidence for catalog and contextual gates."""

    from directordeck.capabilities.evaluator import (
        STRICT_ATTENTION_CK_INT8_RUNTIME_PROBE,
        STRICT_ATTENTION_PYTORCH_RUNTIME_PROBE,
        STRICT_H3_SAGE_RUNTIME_PROBE,
        runtime_probe_key,
    )

    evidence: dict[str, dict[str, object]] = {}
    evidence[runtime_probe_key(STRICT_ATTENTION_PYTORCH_RUNTIME_PROBE)] = (
        _call_bundled_runtime_probe(
            _STRICT_ATTENTION_RUNTIME_MODULE,
            "pytorch",
            None,
        )
    )

    default_device_index, default_unavailable_code = (
        _comfy_default_cuda_probe_target()
    )
    device_probes: list[tuple[str, int, str | None]] = [
        (
            "default",
            default_device_index if default_device_index is not None else -1,
            default_unavailable_code,
        ),
        ("cpu", -1, None),
    ]
    for item in gpu_inventory:
        logical_index = getattr(item, "logical_index", None)
        if type(logical_index) is int and 0 <= logical_index <= 255:
            device_probes.append(
                (f"gpu:{logical_index}", logical_index, None)
            )

    attention_observations: list[dict[str, object]] = []
    h3_observations: list[dict[str, object]] = []
    for device, device_index, unavailable_code in device_probes:
        if unavailable_code is None:
            attention = _call_bundled_runtime_probe(
                _STRICT_ATTENTION_RUNTIME_MODULE,
                "ck_int8",
                device_index,
            )
            h3 = _call_bundled_runtime_probe(
                _STRICT_H3_RUNTIME_MODULE,
                device_index,
            )
        else:
            attention = _unavailable_runtime_probe(unavailable_code)
            h3 = _unavailable_runtime_probe(unavailable_code)
        evidence[
            runtime_probe_key(
                STRICT_ATTENTION_CK_INT8_RUNTIME_PROBE,
                device=device,
            )
        ] = attention
        evidence[
            runtime_probe_key(
                STRICT_H3_SAGE_RUNTIME_PROBE,
                device=device,
            )
        ] = h3
        attention_observations.append(attention)
        h3_observations.append(h3)

    evidence[
        runtime_probe_key(
            STRICT_ATTENTION_CK_INT8_RUNTIME_PROBE,
            device="any",
        )
    ] = _aggregate_runtime_probes(tuple(attention_observations))
    evidence[
        runtime_probe_key(
            STRICT_H3_SAGE_RUNTIME_PROBE,
            device="any",
        )
    ] = _aggregate_runtime_probes(tuple(h3_observations))
    return evidence


class _ComfyOutputProbeProvider:
    """Safely inspect one persistent ComfyUI output without downloading it."""

    def probe_output(self, descriptor):
        # Imports stay lazy because the backend path is installed by
        # ``_run_backend`` and plugin import must remain side-effect bounded.
        import folder_paths

        from directordeck.host_artifacts import (
            HostOutputProbeError,
            HostOutputProbeResult,
            PermanentHostOutputProbeError,
        )
        from directordeck.media import MediaToolError, probe_video_path

        document = (
            descriptor.model_dump(mode="json")
            if hasattr(descriptor, "model_dump")
            else descriptor
        )
        if not isinstance(document, dict):
            raise PermanentHostOutputProbeError(
                "host output descriptor is invalid"
            )
        filename = document.get("filename")
        subfolder = document.get("subfolder", "")
        if (
            document.get("type") != "output"
            or not isinstance(filename, str)
            or not filename
            or filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
            or any(ord(character) < 32 or ord(character) == 127 for character in filename)
            or not isinstance(subfolder, str)
            or "\\" in subfolder
            or any(ord(character) < 32 or ord(character) == 127 for character in subfolder)
        ):
            raise PermanentHostOutputProbeError(
                "host output descriptor is unsafe"
            )
        folder_parts = PurePosixPath(subfolder).parts if subfolder else ()
        if (
            PurePosixPath(subfolder).is_absolute()
            or any(part in {"", ".", ".."} for part in folder_parts)
            or Path(filename).suffix.lower() not in _VIDEO_OUTPUT_EXTENSIONS
        ):
            raise PermanentHostOutputProbeError(
                "host output descriptor is unsafe"
            )

        try:
            output_root = Path(folder_paths.get_output_directory()).resolve(strict=True)
            candidate = output_root.joinpath(*folder_parts, filename).resolve(strict=True)
            try:
                candidate.relative_to(output_root)
            except ValueError as exc:
                raise PermanentHostOutputProbeError(
                    "host output descriptor escapes the output root"
                ) from exc
            if not candidate.is_file():
                raise PermanentHostOutputProbeError(
                    "host output is not a regular file"
                )
            metadata = probe_video_path(
                candidate,
                probe_method="directordeck_host_ffprobe_v1",
                allow_frame_count_estimate_on_timeout=True,
            )
        except HostOutputProbeError:
            raise
        except (MediaToolError, OSError, RuntimeError, ValueError) as exc:
            # Do not expose the absolute ComfyUI output path through task APIs.
            raise HostOutputProbeError("host output media probe failed") from exc

        return HostOutputProbeResult(
            width=metadata.width,
            height=metadata.height,
            fps=metadata.native_fps,
            frame_count=metadata.frame_count,
            duration_seconds=metadata.duration,
            has_audio=metadata.has_audio,
            media_probe_version=metadata.probe_method,
        )


class _ComfyHostCapabilityProvider:
    """Collect a canonical, privacy-safe view of this ComfyUI process.

    The provider lives in the plugin layer because it is the only layer allowed
    to inspect ComfyUI's in-process node registry.  The embedded Director
    backend receives only the frozen contract returned by :meth:`snapshot` and
    therefore remains a pure HTTP/WS client of ComfyUI internals.  The snapshot
    records class presence and bounded advisory interface observations; it does
    not authenticate node source, package provenance, or Python object identity.
    """

    def __init__(
        self,
        *,
        comfy_url: str,
        tls_certfile: str | Path | None = None,
        object_info_loader: Callable[[], Mapping[str, object]] | None = None,
        node_registry_loader: Callable[[], Mapping[str, object]] | None = None,
        generated_at_factory: Callable[[], datetime] | None = None,
    ) -> None:
        self._comfy_url = comfy_url.rstrip("/")
        self._tls_certfile = Path(tls_certfile) if tls_certfile is not None else None
        self._object_info_loader = object_info_loader or self._load_object_info
        self._node_registry_loader = node_registry_loader or self._load_node_registry
        self._generated_at_factory = generated_at_factory or (
            lambda: datetime.now(timezone.utc)
        )
        self._snapshot_lock = threading.Lock()
        self._snapshot_cache: object | None = None

    def _load_node_registry(self) -> Mapping[str, object]:
        try:
            nodes_module = importlib.import_module("nodes")
            registry = getattr(nodes_module, "NODE_CLASS_MAPPINGS")
        except (AttributeError, ImportError) as exc:
            raise RuntimeError("ComfyUI node registry is unavailable") from exc
        if not isinstance(registry, Mapping):
            raise RuntimeError("ComfyUI node registry is invalid")
        return registry

    def _load_object_info(self) -> Mapping[str, object]:
        context: ssl.SSLContext | None = None
        if self._comfy_url.lower().startswith("https://"):
            context = ssl.create_default_context()
            if self._tls_certfile is not None:
                context.load_verify_locations(cafile=str(self._tls_certfile))
                context.verify_flags |= getattr(
                    ssl,
                    "VERIFY_X509_PARTIAL_CHAIN",
                    0,
                )
        handlers: list[object] = [ProxyHandler({})]
        if context is not None:
            handlers.append(HTTPSHandler(context=context))
        opener = build_opener(*handlers)
        request = Request(
            f"{self._comfy_url}/object_info",
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with opener.open(
                request,
                timeout=_HOST_CAPABILITY_HTTP_TIMEOUT_SECONDS,
            ) as response:
                payload = response.read(_OBJECT_INFO_LIMIT_BYTES + 1)
        except (OSError, TimeoutError, ValueError) as exc:
            raise RuntimeError("ComfyUI object-info probe is unavailable") from exc
        if len(payload) > _OBJECT_INFO_LIMIT_BYTES:
            raise RuntimeError("ComfyUI object-info probe is too large")
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("ComfyUI object-info probe is invalid") from exc
        if not isinstance(document, Mapping):
            raise RuntimeError("ComfyUI object-info probe is invalid")
        return document

    def snapshot(self):
        # Node registration, bounded object-info interfaces, package versions,
        # logical GPUs and media tools are process-static. Dynamic queue, model
        # inventory and Ray-ledger facts are intentionally collected elsewhere.
        # Cache this expensive observation for the lifetime of one ComfyUI
        # process; a full restart constructs a fresh provider automatically.
        with self._snapshot_lock:
            if self._snapshot_cache is None:
                self._snapshot_cache = self._capture_snapshot()
            return self._snapshot_cache

    def _capture_snapshot(self):
        # Imports remain lazy: packaged plugin import happens before the
        # embedded backend path is installed and must not import backend code.
        from directordeck.workflow.contracts import (
            HostCapabilitySnapshot,
            LogicalGpuCapability,
            MediaToolCapability,
            PackageCapability,
            RayLightInstallation,
            RuntimeProbeEvidence,
        )
        from directordeck.workflow.node_contracts import (
            CURRENT_NODE_CONTRACT_REGISTRY,
        )

        live_registry = self._node_registry_loader()
        object_info = self._object_info_loader()
        node_registry: dict[str, str] = {}
        object_info_slices: dict[str, object] = {}

        for class_type in sorted(CURRENT_NODE_CONTRACT_REGISTRY.contracts):
            contract = CURRENT_NODE_CONTRACT_REGISTRY.contracts[class_type]
            raw = object_info.get(class_type)
            live_node = live_registry.get(class_type)
            if live_node is None:
                continue
            raw_module = raw.get("python_module") if isinstance(raw, Mapping) else None
            node_registry[class_type] = _observed_node_module(raw_module)
            interface_matches = isinstance(raw, Mapping) and _object_info_matches_contract(
                raw,
                contract.object_info_contract,
            )
            if not interface_matches:
                # Interface drift remains visible as a missing bounded slice,
                # but ComfyUI execution—not this observer—decides compatibility.
                continue
            # This is a deliberately bounded slice. Dynamic model/file choices,
            # Comfy widget defaults, paths and unrelated optional fields are not
            # copied out of the host response.
            object_info_slices[class_type] = contract.object_info_contract

        importable_packages = {
            capability_name: PackageCapability(
                importable=available,
                version=version,
            )
            for capability_name, available, version in _package_capabilities()
        }
        gpu_inventory = tuple(
            LogicalGpuCapability(**item) for item in _logical_gpu_inventory()
        )
        media_tool_status = {
            name: MediaToolCapability(available=available, version=version)
            for name, available, version in _media_tool_capabilities()
        }
        runtime_probe_evidence = {
            key: RuntimeProbeEvidence(**value)
            for key, value in _runtime_probe_evidence(gpu_inventory).items()
        }

        ray_registered = _state.raylight == "registered"
        ray_nodes_available = ray_registered and all(
            live_registry.get(class_type) is not None
            for class_type in _DIRECTOR_RAYLIGHT_CLASS_TYPE_ALIASES.values()
        )
        ray_version = (
            _bundled_node_package_version(_DIRECTOR_RAYLIGHT_RUNTIME_MODULE)
            if ray_registered
            else None
        )
        ray_reasons: tuple[str, ...] = ()
        if not ray_registered:
            state = _state.raylight
            ray_reasons = (
                state
                if state
                in {
                    "deps_missing",
                    "platform_unsupported",
                    "pack_missing",
                    "load_failed",
                }
                else "raylight_not_registered",
            )
        elif not ray_nodes_available:
            ray_reasons = ("raylight_node_missing",)

        return HostCapabilitySnapshot(
            schema_version=_HOST_CAPABILITY_SCHEMA_VERSION,
            generated_at=self._generated_at_factory(),
            node_registry=node_registry,
            object_info_slices=object_info_slices,
            # Schema compatibility only. Director no longer reads or compares
            # live node source fingerprints.
            module_fingerprints={},
            importable_packages=importable_packages,
            gpu_inventory=gpu_inventory,
            raylight_installation=RayLightInstallation(
                installed=ray_registered,
                package_version=ray_version,
                node_contracts_available=ray_nodes_available,
                reason_codes=ray_reasons,
            ),
            media_tool_status=media_tool_status,
            runtime_probe_evidence=runtime_probe_evidence,
        )


def _raw_input_contract(value: object) -> tuple[str, tuple[object, ...]] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    declaration = value[0]
    metadata = value[1] if len(value) > 1 and isinstance(value[1], Mapping) else {}
    if isinstance(declaration, (list, tuple)):
        return "COMBO", tuple(declaration)
    if not isinstance(declaration, str):
        return None
    if declaration == "COMFY_DYNAMICCOMBO_V3":
        options = metadata.get("options")
        if not isinstance(options, (list, tuple)):
            return None
        values: list[object] = []
        for option in options:
            if not isinstance(option, Mapping) or "key" not in option:
                return None
            values.append(option["key"])
        return "DYNAMIC_COMBO", tuple(values)
    options = metadata.get("options")
    if declaration == "COMBO" and isinstance(options, (list, tuple)):
        return declaration, tuple(options)
    return declaration, ()


def _raw_autogrow_member_contract(
    raw_optional: Mapping[str, object],
    flattened_name: str,
) -> tuple[str, tuple[object, ...]] | None:
    """Resolve one V3 Autogrow member from its ``group.member`` prompt path.

    ComfyUI exposes an Autogrow group once in ``/object_info`` (for example
    ``ref_images``) while prompts address its generated members by flattened
    paths (for example ``ref_images.ref_image_0``).  Validate the actual group
    template, generated name and member port instead of requiring every
    possible flattened path to be repeated in ``/object_info``.
    """

    group_name, separator, member_name = flattened_name.partition(".")
    if not separator or not group_name or not member_name:
        return None
    raw_group = raw_optional.get(group_name)
    if not isinstance(raw_group, (list, tuple)) or len(raw_group) < 2:
        return None
    if raw_group[0] != "COMFY_AUTOGROW_V3" or not isinstance(
        raw_group[1], Mapping
    ):
        return None
    template = raw_group[1].get("template")
    if not isinstance(template, Mapping):
        return None

    prefix = template.get("prefix")
    names = template.get("names")
    if isinstance(prefix, str):
        if not member_name.startswith(prefix):
            return None
        suffix = member_name.removeprefix(prefix)
        maximum = template.get("max")
        if (
            not suffix.isdigit()
            or not isinstance(maximum, int)
            or isinstance(maximum, bool)
            or not 0 <= int(suffix) < maximum
        ):
            return None
    elif isinstance(names, (list, tuple)):
        if member_name not in names:
            return None
    else:
        return None

    template_input = template.get("input")
    if not isinstance(template_input, Mapping):
        return None
    declared_members: list[object] = []
    for input_kind in ("required", "optional"):
        inputs = template_input.get(input_kind, {})
        if not isinstance(inputs, Mapping):
            return None
        declared_members.extend(inputs.values())
    if len(declared_members) != 1:
        return None
    return _raw_input_contract(declared_members[0])


def _object_info_matches_contract(
    raw: Mapping[str, object],
    expected: object,
) -> bool:
    inputs = raw.get("input")
    if not isinstance(inputs, Mapping):
        return False
    raw_required = inputs.get("required")
    raw_optional = inputs.get("optional", {})
    if not isinstance(raw_required, Mapping) or not isinstance(raw_optional, Mapping):
        return False

    expected_required = expected.required_inputs
    expected_optional = expected.optional_inputs
    # This is advisory normalization only. Extra optional widgets and moving a
    # required input to optional are compatible observations; ComfyUI remains
    # the authority even when this bounded matcher returns False.
    if not set(expected_required) <= set(raw_required) | set(raw_optional):
        return False
    if set(raw_required) - set(expected_required):
        return False
    if any(
        name not in raw_optional
        and _raw_autogrow_member_contract(raw_optional, name) is None
        for name in expected_optional
    ):
        return False
    for name, input_contract in (
        *expected_required.items(),
        *expected_optional.items(),
    ):
        if name in raw_required:
            normalized = _raw_input_contract(raw_required[name])
        elif name in raw_optional:
            normalized = _raw_input_contract(raw_optional[name])
        else:
            normalized = _raw_autogrow_member_contract(raw_optional, name)
        if normalized is None or normalized[0] != input_contract.port_type:
            return False
        live_enum = normalized[1]
        if input_contract.enum_values and not all(
            any(candidate == expected_value for candidate in live_enum)
            for expected_value in input_contract.enum_values
        ):
            return False

    outputs = raw.get("output")
    output_names = raw.get("output_name", outputs)
    output_is_list = raw.get("output_is_list")
    if (
        not isinstance(outputs, (list, tuple))
        or not isinstance(output_names, (list, tuple))
        or len(outputs) < len(expected.outputs)
        or len(output_names) < len(expected.outputs)
    ):
        return False
    if output_is_list is None:
        output_is_list = [False] * len(outputs)
    if not isinstance(output_is_list, (list, tuple)) or len(output_is_list) < len(
        expected.outputs
    ):
        return False
    for item in expected.outputs:
        if outputs[item.index] != item.port_type:
            return False
        if bool(output_is_list[item.index]) is not item.is_list:
            return False
    return bool(raw.get("output_node", False)) is expected.output_node


def _observed_node_module(raw_module: object) -> str:
    """Return a privacy-safe advisory label, never an authorization result."""

    return (
        raw_module
        if isinstance(raw_module, str)
        and _SAFE_MODULE_IDENTITY.fullmatch(raw_module) is not None
        else "host.interface"
    )


def _safe_version(value: object) -> str | None:
    rendered = str(value).strip()
    return rendered if _SAFE_VERSION.fullmatch(rendered) else None


def _version_from_package_root(package_root: Path) -> str | None:
    try:
        with (package_root / "pyproject.toml").open("rb") as stream:
            return _safe_version(tomllib.load(stream)["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return None


def _bundled_node_package_version(module: str) -> str | None:
    if not module.startswith("custom_nodes."):
        return None
    package_name = module.removeprefix("custom_nodes.")
    source_package_name = (
        "raylight"
        if module == _DIRECTOR_RAYLIGHT_RUNTIME_MODULE
        else package_name
    )
    candidates = (
        _NODES_DIR / package_name,
        _PLUGIN_ROOT.parent / "custom_nodes" / source_package_name,
    )
    for candidate in candidates:
        version = _version_from_package_root(candidate)
        if version is not None:
            return version
    return None


def _package_capabilities() -> tuple[tuple[str, bool, str | None], ...]:
    result: list[tuple[str, bool, str | None]] = []
    for capability_name, import_name, distribution_name in _IMPORTABILITY_PACKAGES:
        try:
            importable = importlib.util.find_spec(import_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            importable = False
        version: str | None = None
        if importable:
            try:
                version = _safe_version(importlib.metadata.version(distribution_name))
            except importlib.metadata.PackageNotFoundError:
                pass
        result.append((capability_name, importable, version))
    return tuple(result)


def _device_memory_mb(device_properties: object) -> int | None:
    memory = getattr(device_properties, "total_memory", None)
    if not isinstance(memory, int) or isinstance(memory, bool) or memory <= 0:
        return None
    return max(1, memory // (1024 * 1024))


def _logical_gpu_inventory() -> tuple[dict[str, object], ...]:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return ()

    cuda = getattr(torch, "cuda", None)
    try:
        cuda_available = bool(cuda is not None and cuda.is_available())
    except (AttributeError, RuntimeError):
        cuda_available = False
    if cuda_available:
        try:
            count = int(cuda.device_count())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return ()
        result: list[dict[str, object]] = []
        for logical_index in range(max(0, count)):
            try:
                memory_mb = _device_memory_mb(cuda.get_device_properties(logical_index))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                memory_mb = None
            result.append(
                {
                    "logical_index": logical_index,
                    "backend": "cuda",
                    "total_memory_mb": memory_mb,
                }
            )
        return tuple(result)

    xpu = getattr(torch, "xpu", None)
    try:
        xpu_available = bool(xpu is not None and xpu.is_available())
    except (AttributeError, RuntimeError):
        xpu_available = False
    if xpu_available:
        try:
            count = int(xpu.device_count())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return ()
        return tuple(
            {
                "logical_index": logical_index,
                "backend": "xpu",
                "total_memory_mb": _device_memory_mb(
                    xpu.get_device_properties(logical_index)
                ),
            }
            for logical_index in range(max(0, count))
        )

    mps = getattr(getattr(torch, "backends", None), "mps", None)
    try:
        mps_available = bool(mps is not None and mps.is_available())
    except (AttributeError, RuntimeError):
        mps_available = False
    return (
        {"logical_index": 0, "backend": "mps", "total_memory_mb": None},
    ) if mps_available else ()


_MEDIA_PROBE_TIMEOUT_SECONDS = 2.0
_REQUIRED_FFMPEG_ENCODERS = frozenset({"libx264", "aac"})


def _run_media_probe(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=_MEDIA_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed if completed.returncode == 0 else None


def _media_tool_version(
    completed: subprocess.CompletedProcess[str],
    name: str,
) -> str | None:
    first_line = completed.stdout.splitlines()[0] if completed.stdout else ""
    match = re.search(
        rf"\b{re.escape(name)}\s+version\s+([^\s]+)",
        first_line,
        re.IGNORECASE,
    )
    return _safe_version(match.group(1)) if match is not None else None


def _ffmpeg_has_required_encoders(executable: str) -> bool:
    completed = _run_media_probe([executable, "-hide_banner", "-encoders"])
    if completed is None:
        return False
    encoder_names: set[str] = set()
    for line in (completed.stdout + "\n" + completed.stderr).splitlines():
        match = re.match(r"^\s*[A-Z.]{6}\s+(\S+)(?:\s|$)", line)
        if match is not None:
            encoder_names.add(match.group(1))
    return _REQUIRED_FFMPEG_ENCODERS <= encoder_names


def _media_tool_capabilities() -> tuple[tuple[str, bool, str | None], ...]:
    result: list[tuple[str, bool, str | None]] = []
    for name in ("ffmpeg", "ffprobe"):
        executable = shutil.which(name)
        version_probe = (
            _run_media_probe([executable, "-version"])
            if executable is not None
            else None
        )
        available = version_probe is not None
        if available and name == "ffmpeg":
            available = _ffmpeg_has_required_encoders(executable)
        version = (
            _media_tool_version(version_probe, name)
            if version_probe is not None
            else None
        )
        result.append(
            (
                name,
                available,
                version,
            )
        )
    return tuple(result)


def _set_backend_failure(error: str, *, preserve_existing: bool = False) -> None:
    """Record a terminal failure without losing an earlier diagnostic."""
    if preserve_existing and _state.error:
        if error not in _state.error:
            _state.error = f"{_state.error}\n{error}"
    else:
        _state.error = error
    _state.status = "failed"


def _record_shutdown_timeout(thread: threading.Thread, timeout: float) -> None:
    if thread.is_alive():
        _set_backend_failure(
            "TimeoutError: Director backend did not stop within "
            f"{timeout:g} seconds",
            preserve_existing=True,
        )


def _load_node_pack(
    pack_dir: Path,
    unique_name: str,
    *,
    logical_module: str,
    class_type_aliases: Mapping[str, str] | None = None,
) -> int:
    """Exec a bundled node pack's __init__.py and merge its node mappings."""
    if logical_module in _BUNDLED_NODE_MODULES:
        raise RuntimeError(f"bundled logical module already loaded: {logical_module}")
    spec = importlib.util.spec_from_file_location(unique_name, pack_dir / "__init__.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create module spec for bundled pack {pack_dir.name}")
    module = importlib.util.module_from_spec(spec)
    missing = object()
    previous_module = sys.modules.get(unique_name, missing)
    sys.modules[unique_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous_module is missing:
            sys.modules.pop(unique_name, None)
        else:
            sys.modules[unique_name] = previous_module
        raise
    mappings = getattr(module, "NODE_CLASS_MAPPINGS", {})
    displays = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
    if type(mappings) is not dict or not mappings or not all(
        isinstance(class_type, str) and class_type and isinstance(node_class, type)
        for class_type, node_class in mappings.items()
    ):
        if previous_module is missing:
            sys.modules.pop(unique_name, None)
        else:
            sys.modules[unique_name] = previous_module
        raise TypeError(f"bundled pack {pack_dir.name} has invalid node mappings")
    if type(displays) is not dict or not all(
        isinstance(class_type, str) and isinstance(display_name, str)
        for class_type, display_name in displays.items()
    ) or not set(displays).issubset(mappings):
        if previous_module is missing:
            sys.modules.pop(unique_name, None)
        else:
            sys.modules[unique_name] = previous_module
        raise TypeError(f"bundled pack {pack_dir.name} has invalid display mappings")
    if class_type_aliases is not None:
        if (
            not class_type_aliases
            or not all(
                isinstance(source, str)
                and source
                and isinstance(target, str)
                and target
                for source, target in class_type_aliases.items()
            )
            or len(set(class_type_aliases.values())) != len(class_type_aliases)
        ):
            if previous_module is missing:
                sys.modules.pop(unique_name, None)
            else:
                sys.modules[unique_name] = previous_module
            raise TypeError(f"bundled pack {pack_dir.name} has invalid node aliases")
        missing_alias_sources = set(class_type_aliases) - set(mappings)
        if missing_alias_sources:
            if previous_module is missing:
                sys.modules.pop(unique_name, None)
            else:
                sys.modules[unique_name] = previous_module
            raise RuntimeError(
                f"bundled pack {pack_dir.name} is missing aliased nodes: "
                + ", ".join(sorted(missing_alias_sources))
            )
        mappings = {
            target: mappings[source]
            for source, target in class_type_aliases.items()
        }
        displays = {
            target: f"DirectorDeck · {displays[source]}"
            for source, target in class_type_aliases.items()
            if source in displays
        }
    conflicts = set(mappings).intersection(
        {*NODE_CLASS_MAPPINGS, *_PREEXISTING_HOST_NODE_MAPPINGS}
    )
    if conflicts:
        if previous_module is missing:
            sys.modules.pop(unique_name, None)
        else:
            sys.modules[unique_name] = previous_module
        raise RuntimeError(
            f"bundled pack {pack_dir.name} node conflict: "
            + ", ".join(sorted(conflicts))
        )
    NODE_CLASS_MAPPINGS.update(mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(displays)
    _BUNDLED_NODE_MODULES[logical_module] = module
    return len(mappings)


def _raylight_gate() -> str:
    if not sys.platform.startswith("linux"):
        return "platform_unsupported"
    if importlib.util.find_spec("ray") is None or importlib.util.find_spec("xfuser") is None:
        return "deps_missing"
    return "ok"


def _load_bundled_nodes() -> None:
    if not _PREEXISTING_HOST_NODE_MAPPINGS:
        try:
            host_nodes = importlib.import_module("nodes")
            host_mappings = getattr(host_nodes, "NODE_CLASS_MAPPINGS")
        except (AttributeError, ImportError):
            host_mappings = {}
        if isinstance(host_mappings, Mapping):
            _PREEXISTING_HOST_NODE_MAPPINGS.update(host_mappings)
    strict_packs = (
        (
            "DirectorDeck-Strict-Attention",
            "director_deck_strict_attention",
            "custom_nodes.DirectorDeck-Strict-Attention",
            "strict attention",
        ),
        (
            "DirectorDeck-Strict-H3",
            "director_deck_strict_h3",
            "custom_nodes.DirectorDeck-Strict-H3",
            "strict H3",
        ),
    )
    for package_name, unique_name, logical_module, label in strict_packs:
        pack_dir = _NODES_DIR / package_name
        if not pack_dir.is_dir():
            continue
        try:
            count = _load_node_pack(
                pack_dir,
                unique_name,
                logical_module=logical_module,
            )
            LOGGER.info("Director: registered %d %s nodes", count, label)
        except Exception as exc:  # noqa: BLE001 - recorded for /directordeck/status
            diagnostic = f"{package_name}: {type(exc).__name__}: {exc}"
            _state.nodes_error = (
                f"{_state.nodes_error}\n{diagnostic}"
                if _state.nodes_error
                else diagnostic
            )
            LOGGER.exception("Director: failed to load %s nodes", label)
    gate = _raylight_gate()
    if gate != "ok":
        _state.raylight = gate
        if gate == "deps_missing":
            LOGGER.info(
                "Director: RayLight nodes skipped; install requirements-raylight.txt "
                "and restart to enable multi-GPU"
            )
        elif gate == "platform_unsupported":
            LOGGER.info("Director: RayLight nodes skipped (multi-GPU requires Linux)")
        return
    raylight_dir = _NODES_DIR / "DirectorDeck-RayLight"
    if not raylight_dir.is_dir():
        _state.raylight = "pack_missing"
        return
    try:
        count = _load_node_pack(
            raylight_dir,
            "director_deck_raylight",
            logical_module=_DIRECTOR_RAYLIGHT_RUNTIME_MODULE,
            class_type_aliases=_DIRECTOR_RAYLIGHT_CLASS_TYPE_ALIASES,
        )
        _state.raylight = "registered"
        LOGGER.info("Director: registered %d RayLight nodes", count)
    except Exception as exc:  # noqa: BLE001 - recorded for /directordeck/status
        _state.raylight = "load_failed"
        _state.raylight_detail = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("Director: failed to load bundled RayLight nodes")


def _internal_port() -> int:
    override = os.environ.get("DIRECTOR_INTERNAL_PORT", "").strip()
    if override:
        return int(override)
    candidate = _DEFAULT_INTERNAL_PORT
    for _ in range(_PORT_SCAN_LIMIT):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", candidate))
            except OSError:
                candidate += 1
                continue
        return candidate
    raise RuntimeError("DirectorDeck: no free loopback port for the backend")


def _database_location() -> Path:
    """The database lives under ComfyUI's user dir, never in the plugin dir."""
    import folder_paths

    db_dir = Path(folder_paths.get_user_directory()) / "directordeck" / "database"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "directordeck.sqlite3"


def _comfyui_callback_host(listen: object) -> str:
    """Choose an address that this process can reach from ComfyUI's binds."""

    addresses = [
        address.strip()
        for address in str(listen or "").split(",")
        if address.strip()
    ]
    if not addresses:
        addresses = ["0.0.0.0"]
    address = addresses[0].removeprefix("[").removesuffix("]")
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return address
    if parsed.is_unspecified:
        parsed = ipaddress.ip_address("::1" if parsed.version == 6 else "127.0.0.1")
    rendered = str(parsed)
    return f"[{rendered}]" if parsed.version == 6 else rendered


def _comfyui_callback_url() -> str:
    from comfy.cli_args import args as comfy_args

    tls_enabled = bool(
        getattr(comfy_args, "tls_certfile", None)
        and getattr(comfy_args, "tls_keyfile", None)
    )
    scheme = "https" if tls_enabled else "http"
    host = _comfyui_callback_host(getattr(comfy_args, "listen", "127.0.0.1"))
    return f"{scheme}://{host}:{getattr(comfy_args, 'port', 8188)}"


_MIN_COMFYUI_VERSION = (0, 33, 0)


def _comfyui_version_check() -> str | None:
    """Return an advisory public-version warning without pinning source/commit."""

    minimum = ".".join(str(part) for part in _MIN_COMFYUI_VERSION)
    try:
        import comfyui_version

        raw = str(comfyui_version.__version__)
    except Exception:  # noqa: BLE001 - render a bounded advisory
        return (
            f"Director is designed for ComfyUI {minimum} or newer, but the "
            "installed ComfyUI version could not be determined; startup will "
            "continue and incompatible host behavior may fail at execution time"
        )
    match = re.match(r"^\s*v?(\d+)\.(\d+)\.(\d+)(?:\D.*)?$", raw)
    if match is None:
        return (
            f"Director is designed for ComfyUI {minimum} or newer, but version "
            f"{raw!r} could not be parsed; startup will continue and incompatible "
            "host behavior may fail at execution time"
        )
    version = tuple(int(part) for part in match.groups())
    if version < _MIN_COMFYUI_VERSION:
        return (
            f"ComfyUI {raw} is older than Director's recommended compatibility "
            f"baseline {minimum}; startup will continue and incompatible host "
            "behavior may fail at execution time"
        )
    return None


def _plugin_version() -> str | None:
    """Read the plugin package version from the bundled pyproject.toml."""
    try:
        with (_PLUGIN_ROOT / "pyproject.toml").open("rb") as stream:
            return str(tomllib.load(stream)["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return None


class _DirectorHttpxRequestFilter(logging.Filter):
    """Silence routine HTTPX request summaries from Director's backend only."""

    def __init__(self, thread_ident: int) -> None:
        super().__init__()
        self._thread_ident = thread_ident

    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            record.thread == self._thread_ident
            and record.name == "httpx"
            and record.levelno == logging.INFO
            and record.getMessage().startswith("HTTP Request:")
        )


def _run_backend(database_path: Path) -> None:
    # ComfyUI's process-wide logging configuration exposes HTTPX INFO records.
    # Filter only the routine summaries emitted by this dedicated thread; do
    # not change the httpx logger level or hide another plugin's diagnostics.
    httpx_logger = logging.getLogger("httpx")
    request_filter = _DirectorHttpxRequestFilter(threading.get_ident())
    httpx_logger.addFilter(request_filter)
    try:
        _serve_backend(database_path)
    finally:
        httpx_logger.removeFilter(request_filter)


def _serve_backend(database_path: Path) -> None:
    try:
        if str(_BACKEND_PATH) not in sys.path:
            sys.path.insert(0, str(_BACKEND_PATH))
        import uvicorn

        from directordeck.app import create_app
        from directordeck.instance_lock import (
            DirectorInstanceLock,
            DirectorInstanceLockError,
        )
        from comfy.cli_args import args as comfy_args

        # Probe the single-instance lock ourselves: uvicorn converts a
        # lifespan startup failure into SystemExit, which would hide the
        # actionable owner diagnostic from /directordeck/status.
        probe = DirectorInstanceLock(database_path)
        try:
            probe.acquire()
        except DirectorInstanceLockError as exc:
            _set_backend_failure(str(exc))
            LOGGER.error("Director backend cannot start: %s", exc)
            return
        else:
            probe.release()

        comfy_url = _comfyui_callback_url()
        comfy_tls_certfile = (
            getattr(comfy_args, "tls_certfile", None)
            if getattr(comfy_args, "tls_keyfile", None)
            else None
        )
        app = create_app(
            database_path=database_path,
            comfy_url=comfy_url,
            host_output_probe=_ComfyOutputProbeProvider(),
            host_capability_provider=_ComfyHostCapabilityProvider(
                comfy_url=comfy_url,
                tls_certfile=comfy_tls_certfile,
            ),
            comfy_tls_certfile=comfy_tls_certfile,
            public_api_prefix="/directordeck",
            raylight_requirements_path=_PLUGIN_ROOT / "requirements-raylight.txt",
            endpoint_runtime_instance_id=_COMFY_BOOT_RUNTIME_INSTANCE_ID,
        )
        _state.version = _plugin_version()
        port = _internal_port()
        config = uvicorn.Config(
            app=app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        _state.server = server
        _state.port = port
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.serve())
        finally:
            loop.close()
    except BaseException as exc:  # noqa: BLE001 - surfaced via /directordeck/status
        _set_backend_failure(f"{type(exc).__name__}: {exc}")
        LOGGER.exception("Director backend exited with an error")
        return
    # ``uvicorn.Server.started`` remains true after ``serve()`` returns.  Do
    # not leave a backend that was observed as ready stuck in that stale state.
    _state.status = "stopped"
    LOGGER.info("Director backend stopped")


def _shutdown_backend() -> None:
    server = _state.server
    thread = _state.thread
    if server is not None:
        server.should_exit = True
    if thread is not None:
        thread.join(timeout=_ATEXIT_JOIN_TIMEOUT_SECONDS)
        _record_shutdown_timeout(thread, _ATEXIT_JOIN_TIMEOUT_SECONDS)


def _start_backend() -> None:
    _state.status = "starting"
    _state.error = None
    _state.server = None
    _state.port = None
    version_warning = _comfyui_version_check()
    if version_warning is not None:
        LOGGER.warning("Director compatibility warning: %s", version_warning)
    database_path = _database_location()
    _state.database_path = str(database_path)
    thread = threading.Thread(
        target=_run_backend,
        args=(database_path,),
        name="directordeck-backend",
        daemon=True,
    )
    _state.thread = thread
    thread.start()
    atexit.register(_shutdown_backend)


async def _on_comfy_shutdown(_app: web.Application) -> None:
    """Stop the backend before aiohttp drains open connections.

    The backend holds long-lived connections to this ComfyUI instance
    (progress websocket, polling clients). aiohttp fires ``on_shutdown``
    before waiting for connection drain, so closing the backend here keeps
    ComfyUI's shutdown from stalling on those connections.
    """
    global _proxy_session
    server = _state.server
    if server is not None:
        server.should_exit = True
    thread = _state.thread
    if thread is not None:
        await asyncio.get_running_loop().run_in_executor(
            None, thread.join, _COMFY_SHUTDOWN_JOIN_TIMEOUT_SECONDS
        )
        _record_shutdown_timeout(thread, _COMFY_SHUTDOWN_JOIN_TIMEOUT_SECONDS)
    if _proxy_session is not None and not _proxy_session.closed:
        await _proxy_session.close()
    _proxy_session = None


def _register_routes() -> None:
    import server as comfy_server

    prompt_server = comfy_server.PromptServer.instance
    prompt_server.app.on_shutdown.append(_on_comfy_shutdown)
    routes = prompt_server.routes

    @routes.get("/directordeck/status")
    async def _director_status(_request: web.Request) -> web.Response:
        if (
            _state.status == "starting"
            and _state.server is not None
            and getattr(_state.server, "started", False)
        ):
            _state.status = "ready"
        return web.json_response(
            {
                "backend": _state.status,
                "error": _state.error,
                "version": _state.version,
                "database_path": _state.database_path,
                "comfy_url": _comfyui_callback_url(),
                "raylight": {
                    "status": _state.raylight,
                    "detail": _state.raylight_detail,
                },
                "nodes_error": _state.nodes_error,
            }
        )

    @routes.get("/directordeck")
    async def _director_root(_request: web.Request) -> web.Response:
        raise web.HTTPFound("/directordeck/")

    @routes.route("*", "/directordeck/api/{tail:.*}")
    async def _director_api_proxy(request: web.Request) -> web.StreamResponse:
        if _state.status in {"failed", "stopped"}:
            error = (
                "directordeck_backend_failed"
                if _state.status == "failed"
                else "directordeck_backend_stopped"
            )
            return web.json_response(
                {"error": error, "detail": _state.error}, status=503
            )
        server = _state.server
        if server is None or not server.started or _state.port is None:
            return web.json_response(
                {"error": "directordeck_backend_starting"}, status=503
            )
        session = await _get_proxy_session()
        tail = request.match_info["tail"]
        target = f"http://127.0.0.1:{_state.port}/api/{tail}"
        if request.query_string:
            target = f"{target}?{request.query_string}"
        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in _HOP_BY_HOP_HEADERS
            and name.lower() not in {"host", "content-length", "accept-encoding"}
        }
        try:
            upstream = await session.request(
                request.method,
                target,
                headers=headers,
                data=request.content if request.can_read_body else None,
                allow_redirects=False,
            )
        except OSError as exc:
            return web.json_response(
                {"error": "directordeck_backend_unreachable", "detail": str(exc)}, status=502
            )
        response_headers = {
            name: value
            for name, value in upstream.headers.items()
            if name.lower() not in _HOP_BY_HOP_HEADERS
            and name.lower() != "content-length"
        }
        downstream = web.StreamResponse(
            status=upstream.status, reason=upstream.reason, headers=response_headers
        )
        await downstream.prepare(request)
        try:
            async for chunk in upstream.content.iter_any():
                await downstream.write(chunk)
        finally:
            upstream.release()
        await downstream.write_eof()
        return downstream

    if _DIST_DIR.is_dir():
        index_file = _DIST_DIR / "index.html"

        @routes.get("/directordeck/")
        async def _director_index(_request: web.Request) -> web.Response:
            if not index_file.is_file():
                raise web.HTTPNotFound
            return web.FileResponse(index_file)

        # Root-level dist files (favicon, manifest, …). Single-segment only;
        # /directordeck/status, /directordeck/api/* and /directordeck/assets/* are all
        # registered before this fallback and win their matches.
        @routes.get("/directordeck/{filename:[^/]+}")
        async def _director_dist_file(request: web.Request) -> web.Response:
            candidate = (_DIST_DIR / request.match_info["filename"]).resolve()
            if candidate.parent != _DIST_DIR or not candidate.is_file():
                raise web.HTTPNotFound
            return web.FileResponse(candidate)

        assets_dir = _DIST_DIR / "assets"
        if assets_dir.is_dir():
            prompt_server.app.add_routes(
                [web.static("/directordeck/assets/", str(assets_dir), show_index=False)]
            )
    else:
        LOGGER.warning("Director: frontend dist not found at %s", _DIST_DIR)


async def _get_proxy_session():
    global _proxy_session
    if _proxy_session is None or _proxy_session.closed:
        import aiohttp

        with _proxy_session_lock:
            if _proxy_session is None or _proxy_session.closed:
                timeout = aiohttp.ClientTimeout(total=None)
                _proxy_session = aiohttp.ClientSession(
                    timeout=timeout, auto_decompress=False
                )
    return _proxy_session


try:
    _load_bundled_nodes()
    _start_backend()
    _register_routes()
except BaseException:  # noqa: BLE001 - a plugin must not break ComfyUI startup
    _state.status = "failed"
    import traceback

    _state.error = traceback.format_exc(limit=5)
    LOGGER.exception("DirectorDeck plugin initialization failed")
