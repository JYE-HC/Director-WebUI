from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from director.app import create_app
from director.comfy import ComfyClient, ComfyError
from director.media import MediaToolError, VideoProxyResult
from director.schemas import RuntimeSettings, VideoMetadata, default_settings


VIDEO_METADATA: dict[str, Any] = {
    "duration": 12.0,
    "native_fps": 24.0,
    "frame_count": 288,
    "width": 1920,
    "height": 1080,
    "probe_method": "fake_ffprobe",
    "has_audio": True,
}


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
        self.lora_metadata_requests: list[str] = []
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
        from director.native_templates import EXPECTED_NATIVE_NODE_MODULES

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
        self.lora_metadata_by_name: dict[str, dict[str, str] | None] = {
            "renamed_generic.safetensors": {
                "target_format": "ComfyUI generic LoRA",
                "source_format": "Diffusers PEFT LoRA",
                "base_model": "MiniMax-H3",
            },
            "style.safetensors": None,
        }
        self.system_devices = [
            {"index": 0, "type": "cuda", "name": "NVIDIA RTX A6000", "vram_total": 48_000, "vram_free": 40_000},
            {"index": 1, "type": "cuda", "name": "NVIDIA RTX A6000", "vram_total": 48_000, "vram_free": 39_000},
        ]

    async def capabilities(self) -> dict[str, Any]:
        if self.preflight_started is not None:
            self.preflight_started.set()
        if self.preflight_release is not None:
            await self.preflight_release.wait()
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

    async def lora_metadata(self, filename: str) -> dict[str, str] | None:
        self.lora_metadata_requests.append(filename)
        metadata = self.lora_metadata_by_name.get(filename)
        return dict(metadata) if metadata is not None else None

    async def system_stats(self) -> dict[str, Any]:
        return {"devices": [dict(device) for device in self.system_devices]}

    async def upload(self, filename: str, content: bytes | Path, content_type: str, kind: str) -> dict[str, Any]:
        materialized = content.read_bytes() if isinstance(content, Path) else content
        self.uploads.append(
            {"filename": filename, "content": materialized, "content_type": content_type, "kind": kind}
        )
        return {"name": filename, "subfolder": "" if kind == "video" else "director-web", "type": "input"}

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
    ) -> dict[str, Any]:
        if self.submit_started is not None:
            self.submit_started.set()
        if self.submit_release is not None:
            await self.submit_release.wait()
        prompt_id = prompt_id or f"prompt-{len(self.prompts) + 1}"
        self.prompts.append({"prompt": prompt, "client_id": client_id, "prompt_id": prompt_id})
        self.pending.append([len(self.pending), prompt_id])
        if self.auto_complete_ray_kill and any(
            isinstance(node, dict) and node.get("class_type") == "RayKill"
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
            and node.get("class_type") == "XFuserSamplerCustomAdvanced"
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

    monkeypatch.setattr("director.app.anyio.to_thread.run_sync", run_sync)
    monkeypatch.setattr("director.app.create_24fps_proxy_file", create_proxy)
    app = create_app(database_path=tmp_path / "director.sqlite3", comfy_factory=lambda _settings: fake_comfy)
    # httpx.AsyncClient + ASGITransport is intentionally used instead of
    # Starlette TestClient. Some sandboxed Python builds cannot start its
    # blocking portal even for an empty ASGI app. Initialize persistence
    # explicitly because ASGITransport does not drive lifespan hooks.
    app.state.database.initialize()
    app.state.database.put_settings(default_settings("http://comfy.test:8188"))
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
            comfy_origin="http://comfy.test:8188",
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
        "subfolder": "director-web",
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
