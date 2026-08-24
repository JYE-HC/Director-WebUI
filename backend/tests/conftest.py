from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable, Mapping
import re
from typing import Any

import httpx
import pytest

from directordeck.config_manager import initialize_directordeck_config


# Production initializes this once in the FastAPI lifespan. Workflow/schema
# unit tests intentionally bypass the application, so initialize the same
# immutable process snapshot at test-session startup.
initialize_directordeck_config()

from directordeck.app import create_app
from directordeck.comfy import ComfyClient, ComfyError
from directordeck.host_artifacts import HostOutputProbeError
from directordeck.media import MediaToolError, VideoProxyResult
from directordeck.migrations.timeline_v4_v5 import (
    legacy_creative_binding_context,
    migrate_runtime_settings_v1_to_v2,
    migrate_timeline_v4_to_v5,
)
from directordeck.schemas import (
    LoraFeatureParams,
    LoraLoaderOverrideRecord,
    RuntimeSettingsV1,
    RuntimeSettingsV2,
    RuntimeSettingsV3,
    UnifiedTimelineDraftV4,
    UnifiedTimelineDraftV5,
    VideoMetadata,
    default_settings,
    mode_draft_to_timeline,
    validate_mode_draft,
)
from directordeck.workflow.contracts import (
    HostCapabilitySnapshot,
    LogicalGpuCapability,
    MediaToolCapability,
    PackageCapability,
    RayLightInstallation,
    canonical_sha256,
)
from directordeck.workflow.effective_features import (
    migrate_timeline_feature_authority_to_v5,
)
from directordeck.workflow.node_contracts import V4_NODE_CONTRACT_REGISTRY
from directordeck.workflow.v6_projection import project_v5_authority_to_v6


VIDEO_METADATA: dict[str, Any] = {
    "duration": 12.0,
    "native_fps": 24.0,
    "frame_count": 288,
    "width": 1920,
    "height": 1080,
    "probe_method": "fake_ffprobe",
    "has_audio": True,
}

# The embedded backend's single ComfyUI endpoint; create_app requires it and
# the FakeComfy factory ignores it.
TEST_COMFY_URL = "http://comfy.test:8188"


class FakeComfy:
    def __init__(self) -> None:
        self.prompts: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []
        self.video_probes: list[dict[str, str]] = []
        self.video_probe_result: dict[str, Any] | None = dict(VIDEO_METADATA)
        self.shot_detection_requests: list[dict[str, Any]] = []
        self.shot_detection_result: Any = {
            "cutFrames": [0, 120, 288],
            "shotCount": 2,
            "warnings": [],
        }
        self.cancelled: list[str] = []
        self.interrupted: list[str] = []
        self.pending_cancelled: list[str] = []
        self.histories: dict[str, dict[str, Any]] = {}
        self.running: list[Any] = []
        self.pending: list[Any] = []
        self.complete_on_cancel: dict[str, Any] | None = None
        self.cancel_error: Exception | None = None
        self.history_response: dict[str, Any] | None = None
        self.queue_error: Exception | None = None
        self.history_requests: list[tuple[str | None, int | None]] = []
        self.queue_requests = 0
        self.progresses: dict[str, dict[str, Any]] = {}
        self.history_started: asyncio.Event | None = None
        self.history_release: asyncio.Event | None = None
        self.preflight_started: asyncio.Event | None = None
        self.preflight_release: asyncio.Event | None = None
        self.submit_started: asyncio.Event | None = None
        self.submit_release: asyncio.Event | None = None
        self.submit_error_after_side_effect: Exception | None = None
        self.view_content = b"fake-video"
        self.auto_complete_ray_kill = True
        # Most orchestration tests are about prompt structure rather than a
        # multi-minute sampler lifecycle. Strict terminal-gate tests disable
        # this explicitly and drive queue/history themselves.
        self.auto_complete_raylight = True
        self.available_nodes = [
            *ComfyClient.STANDARD_REQUIRED_NODES,
            *ComfyClient.RAYLIGHT_REQUIRED_NODES,
            *ComfyClient.RAYLIGHT_LORA_REQUIRED_NODES,
            "MiniMaxH3TurboLoRA",
            "LoraLoaderBypassModelOnly",
            "LoraLoaderModelOnly",
            "ImageFromBatch",
            "MiniMaxH3AddGuide",
            "TrimAudioDuration",
        ]
        from directordeck.native_templates import EXPECTED_NATIVE_NODE_MODULES

        self.node_provenance = {
            node: EXPECTED_NATIVE_NODE_MODULES[node]
            for node in self.available_nodes
        }
        diffusion_models = [
            "generic_h3_diffusion.safetensors",
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        ]
        self.model_inventory = {
            "fl2va": list(diffusion_models),
            "ref2va": list(diffusion_models),
            "clip": ["qwen3vl_32b_minimax_h3_int8_convrot.safetensors"],
            "video_vae": ["minimax_h3_video_vae_fp16.safetensors"],
            "audio_vae": ["minimax_h3_audio_vae_fp32.safetensors"],
            "loras": [
                "minimax_h3_turbo_v4_step600_ema.safetensors",
                "minimax_h3_fl2v_turbo_4step_v1.0_768p_10ErosMax_beta1_pruned_compat_v001_T8.safetensors",
                "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
                "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
                "renamed_generic.safetensors",
                "style.safetensors",
            ],
        }
        self.system_devices = [
            {"index": 0, "type": "cuda", "name": "NVIDIA RTX A6000", "vram_total": 48_000, "vram_free": 40_000},
            {"index": 1, "type": "cuda", "name": "NVIDIA RTX A6000", "vram_total": 48_000, "vram_free": 39_000},
        ]
    async def capabilities(self) -> dict[str, Any]:
        return {
            "connection": "online",
            "supported_modes": ["t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"],
            "supports_cancel": True,
            "native_timeline": {
                "supported": True,
                "modes": ["fl2va", "ref2va"],
                "continuity": False,
            },
            "execution_backends": {
                "standard": {"available": True, "missing_nodes": []},
                "raylight": {"available": True, "missing_nodes": []},
            },
            "available_nodes": list(self.available_nodes),
            "node_provenance": dict(self.node_provenance),
            "missing_nodes": [],
            "latency_ms": 1.5,
        }

    async def models(self) -> dict[str, list[str]]:
        return {key: list(value) for key, value in self.model_inventory.items()}

    async def system_stats(self) -> dict[str, Any]:
        if self.preflight_started is not None:
            self.preflight_started.set()
        if self.preflight_release is not None:
            await self.preflight_release.wait()
        return {"devices": [dict(device) for device in self.system_devices]}

    async def upload(self, filename: str, content: bytes | Path, content_type: str, kind: str) -> dict[str, Any]:
        materialized = content.read_bytes() if isinstance(content, Path) else content
        self.uploads.append(
            {"filename": filename, "content": materialized, "content_type": content_type, "kind": kind}
        )
        return {"name": filename, "subfolder": "" if kind == "video" else "directordeck", "type": "input"}

    async def upload_output(
        self, filename: str, content: bytes, content_type: str, subfolder: str
    ) -> dict[str, Any]:
        self.uploads.append(
            {
                "filename": filename,
                "content": content,
                "content_type": content_type,
                "kind": "output",
                "subfolder": subfolder,
            }
        )
        return {"name": filename, "subfolder": subfolder, "type": "output"}

    async def submit(
        self,
        prompt: dict[str, Any],
        client_id: str,
        prompt_id: str | None = None,
        *,
        on_receipt: Callable[[str | None, str], None] | None = None,
    ) -> dict[str, Any]:
        if self.submit_started is not None:
            self.submit_started.set()
        if self.submit_release is not None:
            await self.submit_release.wait()
        prompt_id = prompt_id or f"prompt-{len(self.prompts) + 1}"
        self.prompts.append({"prompt": prompt, "client_id": client_id, "prompt_id": prompt_id})
        self.pending.append([len(self.pending), prompt_id])
        if self.auto_complete_ray_kill and any(
            isinstance(node, dict) and node.get("class_type") == "DirectorDeckRayKill"
            for node in prompt.values()
        ):
            self.pending = [
                item for item in self.pending if prompt_id not in item
            ]
            self.histories[prompt_id] = {
                "status": {
                    "status_str": "success",
                    "completed": True,
                    "messages": [],
                },
                "outputs": {},
            }
        elif self.auto_complete_raylight and any(
            isinstance(node, dict)
            and node.get("class_type") == "DirectorDeckRayXFuserSamplerCustomAdvanced"
            for node in prompt.values()
        ):
            self.pending = [item for item in self.pending if prompt_id not in item]
            output_nodes = {
                str(node_id): {
                    "videos": [
                        {
                            "filename": f"{prompt_id}-{node_id}.mp4",
                            "subfolder": "segments",
                            "type": "output",
                        }
                    ]
                }
                for node_id, node in prompt.items()
                if isinstance(node, dict) and node.get("class_type") == "SaveVideo"
            }
            self.histories[prompt_id] = {
                "status": {
                    "status_str": "success",
                    "completed": True,
                    "messages": [],
                },
                "outputs": output_nodes,
            }
        if self.submit_error_after_side_effect is not None:
            raise self.submit_error_after_side_effect
        if on_receipt is not None:
            on_receipt(prompt_id, prompt_id)
        return {"prompt_id": prompt_id, "number": len(self.prompts), "node_errors": {}}

    async def history(
        self, prompt_id: str | None = None, *, max_items: int | None = None
    ) -> dict[str, Any]:
        self.history_requests.append((prompt_id, max_items))
        if self.history_response is not None:
            result = dict(self.history_response)
        elif prompt_id is None:
            result = dict(self.histories)
        else:
            result = {prompt_id: self.histories[prompt_id]} if prompt_id in self.histories else {}
        if self.history_started is not None:
            self.history_started.set()
        if self.history_release is not None:
            await self.history_release.wait()
        return result

    async def queue(self) -> dict[str, Any]:
        self.queue_requests += 1
        if self.queue_error is not None:
            raise self.queue_error
        return {"queue_running": self.running, "queue_pending": self.pending}

    async def cancel(self, prompt_id: str) -> bool:
        if self.cancel_error is not None:
            raise self.cancel_error
        if self.complete_on_cancel is not None:
            self.histories[prompt_id] = self.complete_on_cancel
            self.running = [item for item in self.running if prompt_id not in item]
            self.pending = [item for item in self.pending if prompt_id not in item]
            return False
        if any(prompt_id in item for item in self.running):
            self.cancelled.append(prompt_id)
            self.interrupted.append(prompt_id)
            return True
        if any(prompt_id in item for item in self.pending):
            self.cancelled.append(prompt_id)
            self.pending_cancelled.append(prompt_id)
            self.pending = [item for item in self.pending if prompt_id not in item]
            return True
        return False

    async def view(self, params: dict[str, str]) -> httpx.Response:
        filename = params.get("filename", "").lower()
        if filename.endswith(".png"):
            content_type = "image/png"
        elif filename.endswith(".wav"):
            content_type = "audio/wav"
        else:
            content_type = "video/mp4"
        return httpx.Response(
            200,
            content=self.view_content,
            headers={"content-type": content_type},
            request=httpx.Request("GET", "http://comfy.test/view", params=params),
        )

    def _prompt_output_metadata(
        self,
        descriptor: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Derive the fake probe from the exact submitted one-segment graph."""

        def contains_descriptor(value: Any) -> bool:
            if isinstance(value, Mapping):
                if all(value.get(key) == descriptor.get(key) for key in descriptor):
                    return True
                return any(contains_descriptor(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return any(contains_descriptor(item) for item in value)
            return False

        prompt_record: Mapping[str, Any] | None = None
        for prompt_id, history in self.histories.items():
            if contains_descriptor(history):
                prompt_record = next(
                    (
                        item
                        for item in reversed(self.prompts)
                        if item.get("prompt_id") == prompt_id
                    ),
                    None,
                )
                break
        if prompt_record is None:
            stem = Path(str(descriptor.get("filename") or "")).stem
            for item in reversed(self.prompts):
                prompt = item.get("prompt")
                if not isinstance(prompt, Mapping):
                    continue
                if any(
                    isinstance(node, Mapping)
                    and node.get("class_type") == "SaveVideo"
                    and str(node.get("inputs", {}).get("filename_prefix") or "")
                    .endswith(f"_{stem}")
                    for node in prompt.values()
                ):
                    prompt_record = item
                    break
        if prompt_record is None:
            return None
        prompt = prompt_record.get("prompt")
        if not isinstance(prompt, Mapping):
            return None
        geometry: Mapping[str, Any] | None = None
        video: Mapping[str, Any] | None = None
        for node in prompt.values():
            if not isinstance(node, Mapping):
                continue
            inputs = node.get("inputs")
            if not isinstance(inputs, Mapping):
                continue
            if all(key in inputs for key in ("width", "height", "length")):
                geometry = inputs
            if node.get("class_type") == "CreateVideo" and "fps" in inputs:
                video = inputs
        if geometry is None or video is None:
            return None
        try:
            width = int(geometry["width"])
            height = int(geometry["height"])
            frame_count = int(geometry["length"])
            fps = float(video["fps"])
        except (KeyError, TypeError, ValueError):
            return None
        images = video.get("images")
        if (
            isinstance(images, (list, tuple))
            and len(images) == 2
            and isinstance(images[0], str)
        ):
            visible_node = prompt.get(images[0])
            visible_inputs = (
                visible_node.get("inputs")
                if isinstance(visible_node, Mapping)
                else None
            )
            visible_length = (
                visible_inputs.get("length")
                if isinstance(visible_inputs, Mapping)
                else None
            )
            if isinstance(visible_length, int) and not isinstance(
                visible_length,
                bool,
            ):
                frame_count = visible_length
        return {
            "width": width,
            "height": height,
            "native_fps": fps,
            "frame_count": frame_count,
            "duration": frame_count / fps,
            "has_audio": video.get("audio") is not None,
            "probe_method": "fake_exact_prompt_probe_v1",
        }

    def probe_output(self, descriptor) -> dict[str, Any]:
        document = descriptor.model_dump(mode="json")
        self.video_probes.append(document)
        metadata = self.video_probe_result
        if metadata == VIDEO_METADATA:
            metadata = self._prompt_output_metadata(document) or metadata
        if not isinstance(metadata, dict):
            raise HostOutputProbeError("fake host output probe failed")
        return {
            "width": metadata["width"],
            "height": metadata["height"],
            "fps": metadata["native_fps"],
            "frame_count": metadata["frame_count"],
            "duration_seconds": metadata["duration"],
            "has_audio": metadata["has_audio"],
            "media_probe_version": metadata["probe_method"],
        }


class FakeHostCapabilityProvider:
    """Trusted synthetic live evidence for endpoint orchestration tests."""

    def __init__(self, fake_comfy: FakeComfy) -> None:
        self.fake_comfy = fake_comfy

    def snapshot(self) -> HostCapabilitySnapshot:
        node_registry: dict[str, str] = {}
        object_info: dict[str, object] = {}
        module_fingerprints: dict[str, str] = {}
        for contract in V4_NODE_CONTRACT_REGISTRY.contracts.values():
            if contract.class_type not in self.fake_comfy.available_nodes:
                continue
            module = self.fake_comfy.node_provenance.get(
                contract.class_type,
                contract.allowed_python_modules[0],
            )
            node_registry[contract.class_type] = module
            object_info[contract.class_type] = contract.object_info_contract
            fingerprint = (
                contract.supported_runtime_fingerprints[0]
                if module in contract.allowed_python_modules
                else canonical_sha256(
                    {"synthetic_untrusted_module": module}
                )
            )
            observed = module_fingerprints.setdefault(
                module,
                fingerprint,
            )
            assert observed == fingerprint
        gpu_count = sum(
            1
            for device in self.fake_comfy.system_devices
            if str(device.get("type", "")).lower() == "cuda"
        )
        ray_contracts = tuple(
            contract
            for contract in V4_NODE_CONTRACT_REGISTRY.contracts.values()
            if "custom_nodes.DirectorDeck-RayLight" in contract.allowed_python_modules
        )
        ray_available = all(
            contract.class_type in node_registry for contract in ray_contracts
        )
        return HostCapabilitySnapshot(
            schema_version=1,
            generated_at=datetime.now(timezone.utc),
            node_registry=node_registry,
            object_info_slices=object_info,
            module_fingerprints=module_fingerprints,
            importable_packages={
                "ray": PackageCapability(importable=True),
                "xfuser": PackageCapability(importable=True),
            },
            gpu_inventory=tuple(
                LogicalGpuCapability(logical_index=index, backend="cuda")
                for index in range(gpu_count)
            ),
            raylight_installation=RayLightInstallation(
                installed=ray_available,
                node_contracts_available=ray_available,
                reason_codes=() if ray_available else ("not_registered",),
            ),
            media_tool_status={
                "ffmpeg": MediaToolCapability(available=True),
                "ffprobe": MediaToolCapability(available=True),
            },
        )


@pytest.fixture
def fake_comfy() -> FakeComfy:
    return FakeComfy()


async def wait_for_submission_tasks(client, *, timeout: float = 2.0) -> None:
    """Wait until every accepted background job dispatcher has terminated."""

    async def drained() -> None:
        while client.director_app.state.submission_tasks:
            await asyncio.sleep(0)

    await asyncio.wait_for(drained(), timeout=timeout)


async def runtime_authority_headers(client) -> dict[str, str]:
    response = await client.get("/api/settings/authority")
    assert response.status_code == 200, response.text
    return {
        "X-Director-Runtime-Authority": response.json()["authority_token"],
    }


def _response_with_json(response: httpx.Response, document: Any) -> httpx.Response:
    """Keep old orchestration assertions focused on their original payload.

    Stage 6 writes through CAS envelopes.  Most pre-Stage-6 endpoint tests are
    about asset validation, prompt compilation, or job recovery rather than
    the envelope itself, so the compatibility helpers below return the saved
    document while still exercising the real v5/CAS route.  Dedicated CAS and
    migration tests call the raw client routes directly.
    """

    return httpx.Response(
        response.status_code,
        json=document,
        headers=response.headers,
        request=response.request,
    )


def v5_timeline_fixture(
    document: Mapping[str, Any],
    legacy_settings: Mapping[str, Any] | RuntimeSettingsV1 | None = None,
) -> dict[str, Any]:
    """Synchronous v5 conversion for direct-database/lifespan fixtures."""

    if document.get("version") == 5:
        return UnifiedTimelineDraftV5.model_validate(document).model_dump(
            mode="json"
        )
    settings = (
        legacy_settings
        if isinstance(legacy_settings, RuntimeSettingsV1)
        else RuntimeSettingsV1.model_validate(
            legacy_settings
            if legacy_settings is not None
            else default_settings().model_dump(mode="json")
        )
    )
    return migrate_timeline_v4_to_v5(
        UnifiedTimelineDraftV4.model_validate(document),
        settings,
    ).model_dump(mode="json")


def _runtime_settings_v3_from_legacy(
    legacy: RuntimeSettingsV1,
) -> RuntimeSettingsV3:
    runtime_v2 = migrate_runtime_settings_v1_to_v2(legacy)
    return RuntimeSettingsV3(
        schema_version=3,
        client_id=runtime_v2.client_id,
        memory_policy=runtime_v2.memory_policy,
        raylight_residency_policy=runtime_v2.raylight_residency_policy,
        multi_gpu_enabled=runtime_v2.multi_gpu_enabled,
        placement=runtime_v2.placement,
        lora_loader_overrides=[
            LoraLoaderOverrideRecord(
                lora_filename=record.lora_filename,
                adapter_id=(
                    "minimax_h3_turbo"
                    if record.loader == "dedicated"
                    else "model_only"
                ),
                options=(
                    {"low_vram": False}
                    if record.loader == "dedicated"
                    else {}
                ),
            )
            for record in (
                runtime_v2.legacy_lora_resolution_compat.explicit_overrides
            )
        ],
    )


def save_database_legacy_settings(database, settings) -> RuntimeSettingsV3:
    """Apply one frozen combined settings fixture through both DB CAS APIs."""

    legacy = (
        settings
        if isinstance(settings, RuntimeSettingsV1)
        else RuntimeSettingsV1.model_validate(settings)
    )
    context = legacy_creative_binding_context(legacy)
    timeline, revision = database.get_timeline_authority()
    project_features = dict(timeline.features.project)
    project_features["lora"] = context.lora
    database.validate_and_put_timeline_authority(
        timeline.model_copy(
            update={
                "model_stack": context.model_stack,
                "features": timeline.features.model_copy(
                    update={"project": project_features}
                ),
            }
        ),
        expected_revision=revision,
    )
    runtime = _runtime_settings_v3_from_legacy(legacy)
    _current, token = database.get_settings_authority()
    saved, _next_token = database.put_settings_v3_authority(
        runtime,
        expected_authority_token=token,
        schema_version=3,
    )
    return saved


async def legacy_settings_document(
    client,
    *,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Project current v5 creative + v3 runtime authorities into frozen v1.

    This is intentionally test-only.  Production code never exposes a second
    settings authority after Stage 6.
    """

    timeline_path = (
        "/api/timeline/authority"
        if project_id is None
        else f"/api/projects/{project_id}/timeline/authority"
    )
    timeline_response = await client.get(timeline_path)
    assert timeline_response.status_code == 200, timeline_response.text
    runtime_response = await client.get("/api/settings/authority")
    assert runtime_response.status_code == 200, runtime_response.text
    timeline = UnifiedTimelineDraftV5.model_validate(
        timeline_response.json()["document"]
    )
    runtime = RuntimeSettingsV3.model_validate(
        runtime_response.json()["settings"]
    )
    raw = default_settings().model_dump(mode="json")
    raw.update(
        client_id=runtime.client_id,
        memory_policy=runtime.memory_policy,
        raylight_residency_policy=runtime.raylight_residency_policy,
        multi_gpu_enabled=runtime.multi_gpu_enabled,
    )

    lora_selection = timeline.features.project.get("lora")
    lora_params = (
        None
        if lora_selection is None
        else LoraFeatureParams.model_validate(lora_selection.params)
    )
    for family in ("fl2va", "ref2va"):
        filename = getattr(timeline.model_stack, family).filename
        if filename is None:
            raise AssertionError(
                "legacy test settings projection cannot represent an incomplete "
                f"{family} model binding"
            )
        placement = getattr(runtime.placement, family)
        family_lora = (
            None if lora_params is None else lora_params.by_family[family]
        )
        lora_enabled = bool(
            lora_selection is not None
            and lora_selection.enabled
            and family_lora is not None
            and family_lora.enabled
        )
        lora_name = family_lora.filename if lora_enabled else None
        binding = raw["models"][family]
        binding.update(
            filename=filename,
            device=placement.device,
            backend="auto",
            raylight=placement.raylight.model_dump(mode="json"),
            lora_name=lora_name,
            lora_strength=(family_lora.strength if lora_enabled else 1.0),
            lora_loader="auto",
            lora_low_vram=False,
            standard_lora_loader_override=None,
        )
        if lora_name is not None:
            exact_override = next(
                (
                    record
                    for record in runtime.lora_loader_overrides
                    if record.lora_filename == lora_name
                ),
                None,
            )
            adapter_id = exact_override.adapter_id if exact_override else "model_only"
            binding["lora_low_vram"] = bool(
                exact_override.options.get("low_vram", False)
                if exact_override else False
            )
            binding["standard_lora_loader_override"] = {
                "loader": adapter_id,
                "model_filename": filename,
                "lora_name": lora_name,
            }

    for role, device_field in (
        ("clip", "clip_device"),
        ("video_vae", "video_vae_device"),
        ("audio_vae", "audio_vae_device"),
    ):
        filename = getattr(timeline.model_stack, role).filename
        if filename is None:
            raise AssertionError(
                "legacy test settings projection cannot represent an incomplete "
                f"{role} model binding"
            )
        raw["models"][role].update(
            filename=filename,
            device=getattr(runtime.placement, device_field),
        )
    return RuntimeSettingsV1.model_validate(raw).model_dump(mode="json")


async def v5_timeline_document(
    client,
    document: Mapping[str, Any],
    *,
    legacy_settings: Mapping[str, Any] | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Convert a frozen v1-v4 test document to the explicit v5 request shape."""

    if document.get("version") == 5:
        converted = UnifiedTimelineDraftV5.model_validate(document)
    else:
        settings = RuntimeSettingsV1.model_validate(
            legacy_settings
            if legacy_settings is not None
            else await legacy_settings_document(client, project_id=project_id)
        )
        legacy = UnifiedTimelineDraftV4.model_validate(document)
        converted = migrate_timeline_v4_to_v5(legacy, settings)
    if converted.features.template_bundle_version == 4:
        converted = migrate_timeline_feature_authority_to_v5(converted)
    return converted.model_dump(mode="json")


async def save_timeline_document(
    client,
    document: Mapping[str, Any],
    *,
    project_id: str | None = None,
    legacy_settings: Mapping[str, Any] | None = None,
) -> httpx.Response:
    """Save a legacy test timeline through the current revision-CAS authority."""

    authority_path = (
        "/api/timeline/authority"
        if project_id is None
        else f"/api/projects/{project_id}/timeline/authority"
    )
    current = await client.get(authority_path)
    assert current.status_code == 200, current.text
    converted = await v5_timeline_document(
        client,
        document,
        legacy_settings=legacy_settings,
        project_id=project_id,
    )
    current_bundle = (
        current.json()
        .get("document", {})
        .get("features", {})
        .get("template_bundle_version")
    )
    if current_bundle == 6 and converted["features"]["template_bundle_version"] == 5:
        converted = project_v5_authority_to_v6(
            UnifiedTimelineDraftV5.model_validate(converted)
        ).draft.model_dump(mode="json")
    saved = await client.put(
        authority_path,
        json={
            "document": converted,
            "expected_revision": current.json()["revision"],
        },
    )
    if saved.status_code != 200:
        return saved
    return _response_with_json(saved, saved.json()["document"])


async def save_legacy_settings_document(
    client,
    document: Mapping[str, Any],
    *,
    project_id: str | None = None,
) -> httpx.Response:
    """Apply an old combined settings fixture to the two Stage-6 authorities."""

    legacy = RuntimeSettingsV1.model_validate(document)
    context = legacy_creative_binding_context(legacy)
    timeline_path = (
        "/api/timeline/authority"
        if project_id is None
        else f"/api/projects/{project_id}/timeline/authority"
    )
    current_timeline = await client.get(timeline_path)
    assert current_timeline.status_code == 200, current_timeline.text
    timeline = UnifiedTimelineDraftV5.model_validate(
        current_timeline.json()["document"]
    )
    project_features = dict(timeline.features.project)
    project_features["lora"] = context.lora
    timeline = timeline.model_copy(
        update={
            "model_stack": context.model_stack,
            "features": timeline.features.model_copy(
                update={"project": project_features}
            ),
        }
    )
    saved_timeline = await client.put(
        timeline_path,
        json={
            "document": timeline.model_dump(mode="json"),
            "expected_revision": current_timeline.json()["revision"],
        },
    )
    if saved_timeline.status_code != 200:
        return saved_timeline

    current_runtime = await client.get("/api/settings/authority")
    assert current_runtime.status_code == 200, current_runtime.text
    runtime = _runtime_settings_v3_from_legacy(legacy)
    saved_runtime = await client.put(
        "/api/settings/authority",
        json={
            "document": runtime.model_dump(mode="json"),
            "expected_authority_token": current_runtime.json()[
                "authority_token"
            ],
            "schema_version": 3,
        },
    )
    if saved_runtime.status_code != 200:
        return saved_runtime
    # Return the exact combined legacy fixture that was split across the two
    # current authorities.  Re-projecting through the production v5 compiler
    # would incorrectly require an active unknown LoRA to already have a
    # mapping, preventing tests from saving the incomplete state whose
    # fail-closed compile diagnostic they need to exercise.
    return _response_with_json(saved_runtime, legacy.model_dump(mode="json"))


async def submit_timeline_document(
    client,
    document: Mapping[str, Any],
    *,
    segment_ids: list[str] | None = None,
    project_id: str | None = None,
    endpoint: str | None = None,
) -> httpx.Response:
    """Submit a legacy timeline fixture as one explicit immutable v5 snapshot."""

    converted = await v5_timeline_document(
        client,
        document,
        project_id=project_id,
    )
    body: dict[str, Any] = {"config": converted}
    if segment_ids is not None:
        body["segment_ids"] = segment_ids
    target = endpoint or (
        "/api/timeline/jobs"
        if project_id is None
        else f"/api/projects/{project_id}/jobs"
    )
    return await client.post(target, json=body)


async def compile_timeline_document(
    client,
    document: Mapping[str, Any],
    *,
    segment_ids: list[str] | None = None,
    project_id: str | None = None,
) -> httpx.Response:
    converted = await v5_timeline_document(
        client,
        document,
        project_id=project_id,
    )
    body: dict[str, Any] = {"config": converted}
    if segment_ids is not None:
        body["segment_ids"] = segment_ids
    endpoint = (
        "/api/timeline/compile"
        if project_id is None
        else f"/api/projects/{project_id}/compile"
    )
    return await client.post(endpoint, json=body)


async def submit_legacy_mode_job(
    client,
    mode: str,
    config: Mapping[str, Any] | None = None,
) -> httpx.Response:
    """Exercise old six-mode orchestration assertions through the v5 API."""

    if config is None:
        draft_response = await client.get(f"/api/drafts/{mode}")
        assert draft_response.status_code == 200, draft_response.text
        config = draft_response.json()
    legacy_draft = validate_mode_draft(mode, config)
    timeline = mode_draft_to_timeline(legacy_draft)
    return await submit_timeline_document(
        client,
        timeline.model_dump(mode="json"),
        endpoint="/api/jobs",
    )


def adapt_legacy_workflow_requests(client, monkeypatch) -> None:
    """Opt an orchestration-focused test module into v5 request conversion.

    This deliberately is not installed by the shared ``client`` fixture:
    Stage-6 API boundary tests must observe raw 409/410 responses.  Large
    pre-Stage-6 recovery suites can opt in once at module scope and continue
    asserting their actual concern (execution evidence and lifecycle races)
    while every accepted request still reaches production as an explicit v5
    snapshot.
    """

    original_post = client.post

    async def current_document(
        document: Mapping[str, Any],
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        converted = await v5_timeline_document(
            client,
            document,
            project_id=project_id,
        )
        authority_path = (
            "/api/timeline/authority"
            if project_id is None
            else f"/api/projects/{project_id}/timeline/authority"
        )
        authority = await client.get(authority_path)
        assert authority.status_code == 200, authority.text
        current_bundle = (
            authority.json()
            .get("document", {})
            .get("features", {})
            .get("template_bundle_version")
        )
        if (
            current_bundle == 6
            and converted["features"]["template_bundle_version"] == 5
        ):
            return project_v5_authority_to_v6(
                UnifiedTimelineDraftV5.model_validate(converted)
            ).draft.model_dump(mode="json")
        return converted

    async def v5_post(url: str, *args, **kwargs):
        body = kwargs.get("json")
        if not isinstance(body, Mapping):
            return await original_post(url, *args, **kwargs)
        converted_body = dict(body)
        project_id: str | None = None
        project_match = re.fullmatch(
            r"/api/projects/([^/]+)/(?:compile|jobs)",
            url,
        )
        if project_match is not None:
            project_id = project_match.group(1)
        if "mode" in body and url == "/api/jobs":
            mode = body.get("mode")
            config = body.get("config")
            if isinstance(mode, str):
                if config is None:
                    draft_response = await client.get(f"/api/drafts/{mode}")
                    assert draft_response.status_code == 200, draft_response.text
                    config = draft_response.json()
                legacy_draft = validate_mode_draft(mode, config)
                timeline = mode_draft_to_timeline(legacy_draft)
                converted_body = {
                    "config": await current_document(
                        timeline.model_dump(mode="json"),
                    )
                }
        else:
            config = body.get("config")
            if isinstance(config, Mapping) and config.get("version") != 5:
                if url == "/api/features/preflight":
                    converted_body["config"] = await v5_timeline_document(
                        client,
                        config,
                        project_id=project_id,
                    )
                else:
                    converted_body["config"] = await current_document(
                        config,
                        project_id=project_id,
                    )
        rewritten = dict(kwargs)
        rewritten["json"] = converted_body
        return await original_post(url, *args, **rewritten)

    monkeypatch.setattr(client, "post", v5_post)


@pytest.fixture
async def client(tmp_path: Path, fake_comfy: FakeComfy, monkeypatch):
    # This managed sandbox cannot wake asyncio's default thread executor.
    # Endpoint tests run the already-unit-tested media callable inline and use
    # a deterministic proxy; test_media.py exercises real ffmpeg separately.
    async def run_sync(function, *args):
        return function(*args)

    def create_proxy(source: Path, destination: Path) -> VideoProxyResult:
        metadata = fake_comfy.video_probe_result
        if not isinstance(metadata, dict):
            raise MediaToolError("video probe failed")
        try:
            validated = VideoMetadata.model_validate(metadata)
        except ValueError as exc:
            raise MediaToolError("video probe returned invalid metadata") from exc
        destination.write_bytes(source.read_bytes())
        return VideoProxyResult(
            metadata=validated,
            strategy="transcode",
        )

    monkeypatch.setattr("directordeck.app.anyio.to_thread.run_sync", run_sync)
    monkeypatch.setattr("directordeck.app.create_24fps_proxy_file", create_proxy)
    app = create_app(
        database_path=tmp_path / "directordeck.sqlite3",
        comfy_url=TEST_COMFY_URL,
        comfy_factory=lambda _comfy_url: fake_comfy,
        host_capability_provider=FakeHostCapabilityProvider(fake_comfy),
        host_output_probe=fake_comfy,
    )
    # httpx.AsyncClient + ASGITransport is intentionally used instead of
    # Starlette TestClient. Some sandboxed Python builds cannot start its
    # blocking portal even for an empty ASGI app. Initialize persistence
    # explicitly because ASGITransport does not drive lifespan hooks.
    app.state.database.initialize()
    for name, kind in (
        ("first.png", "image"),
        ("last.png", "image"),
        ("reference.png", "image"),
        ("identity.png", "image"),
        ("voice.wav", "audio"),
        ("motion.mp4", "video"),
        ("source.mp4", "video"),
    ):
        document = asset(name, kind)
        app.state.database.put_asset(
            document["id"],
            document,
        )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://testserver",
    ) as test_client:
        test_client.director_app = app  # type: ignore[attr-defined]
        yield test_client


def asset(name: str, kind: str, *, slot: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "name": name,
        "subfolder": "directordeck",
        "type": "input",
        "kind": kind,
        "id": f"fixture-{kind}-{name}",
    }
    if kind == "video":
        value["metadata"] = dict(VIDEO_METADATA)
    if slot is not None:
        value["slot"] = slot
    return value


def runnable_draft(mode: str) -> dict[str, Any]:
    shot: dict[str, Any] = {
        "id": f"{mode}-1",
        "title": "镜头 01",
        "prompt": "A cinematic camera move",
        "duration_seconds": 5.0,
        "enabled": True,
    }
    if mode == "i2v":
        shot["first_image"] = asset("first.png", "image")
    elif mode == "fl2v":
        shot.update(first_image=asset("first.png", "image"), last_image=asset("last.png", "image"))
    elif mode == "r2v":
        shot.update(
            reference_images=[asset("reference.png", "image", slot=0)],
            reference_audios=[asset("voice.wav", "audio", slot=0)],
            reference_videos=[asset("motion.mp4", "video", slot=0)],
        )
    elif mode == "v2v":
        shot.update(
            source_video=asset("source.mp4", "video"),
            source_start_seconds=1.0,
            source_duration_seconds=5.0,
        )
    elif mode == "rv2v":
        shot.update(
            source_video=asset("source.mp4", "video"),
            source_start_seconds=1.0,
            source_duration_seconds=5.0,
            reference_images=[asset("identity.png", "image", slot=0)],
            reference_audios=[asset("voice.wav", "audio", slot=0)],
        )
    return {
        "mode": mode,
        "prompt": "Shared visual direction",
        "ref_image_size": "match",
        "render": {"width": 864, "height": 480, "fps": 24.0},
        "sampling": {
            "steps": 25,
            "cfg": 1.0,
            "seed": 42,
            "sampler": "res_multistep",
            "scheduler": "simple",
            "shift": 12.0,
            "audio_shift": 3.0,
        },
        "shots": [shot],
    }
