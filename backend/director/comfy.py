from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from .schemas import AssetKind, RuntimeSettings


class ComfyError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 502, detail: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


@dataclass
class ComfyMediaStream:
    """One streaming ComfyUI `/view` response and its owning HTTP client."""

    response: httpx.Response
    client: httpx.AsyncClient
    _closed: bool = field(default=False, init=False, repr=False)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self.response.aclose()
        finally:
            await self.client.aclose()


class ComfyClientProtocol(Protocol):
    async def capabilities(self) -> dict[str, Any]: ...
    async def models(self) -> dict[str, list[str]]: ...
    async def lora_metadata(self, filename: str) -> dict[str, str] | None: ...
    async def system_stats(self) -> dict[str, Any]: ...
    async def upload(self, filename: str, content: bytes | Path, content_type: str, kind: AssetKind) -> dict[str, Any]: ...
    async def upload_output(
        self, filename: str, content: bytes, content_type: str, subfolder: str
    ) -> dict[str, Any]: ...
    async def submit(
        self,
        prompt: dict[str, Any],
        client_id: str,
        prompt_id: str | None = None,
    ) -> dict[str, Any]: ...
    async def history(
        self, prompt_id: str | None = None, *, max_items: int | None = None
    ) -> dict[str, Any]: ...
    async def queue(self) -> dict[str, Any]: ...
    async def cancel(self, prompt_id: str) -> bool: ...
    async def view(self, params: dict[str, str]) -> httpx.Response: ...
    async def view_stream(
        self, params: dict[str, str], *, byte_range: str | None = None
    ) -> ComfyMediaStream: ...


class ComfyClient:
    CONTINUITY_REQUIRED_NODE_MODULES = {
        "MiniMaxH3AddGuide": "comfy_extras.nodes_minimax_h3",
        "ImageFromBatch": "comfy_extras.nodes_images",
        "TrimAudioDuration": "comfy_extras.nodes_audio",
    }
    CONTINUITY_REQUIRED_NODES = tuple(CONTINUITY_REQUIRED_NODE_MODULES)
    STANDARD_REQUIRED_NODES = (
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "SelectModelDevice",
        "SelectCLIPDevice",
        "SelectVAEDevice",
        "MiniMaxH3ImageToVideo",
        "MiniMaxH3ReferenceToVideo",
        "MiniMaxH3SigmaShift",
        "LoadImage",
        "LoadVideo",
        "Video Slice",
        "GetVideoComponents",
        "LoadAudio",
        "RandomNoise",
        "BasicGuider",
        "BasicScheduler",
        "KSamplerSelect",
        "SamplerCustomAdvanced",
        "VAEDecode",
        "VAEDecodeAudio",
        "CreateVideo",
        "SaveVideo",
    )
    RAYLIGHT_REQUIRED_NODES = (
        "CLIPLoader",
        "VAELoader",
        "SelectCLIPDevice",
        "SelectVAEDevice",
        "MiniMaxH3ImageToVideo",
        "MiniMaxH3ReferenceToVideo",
        "LoadImage",
        "LoadVideo",
        "Video Slice",
        "GetVideoComponents",
        "LoadAudio",
        "RayInitializerAdvanced",
        "RayUNETLoader",
        "RayMiniMaxH3SigmaShift",
        "RayBasicGuider",
        "RayBasicScheduler",
        "KSamplerSelect",
        "XFuserSamplerCustomAdvanced",
        "RayKill",
        "VAEDecode",
        "VAEDecodeAudio",
        "CreateVideo",
        "SaveVideo",
    )
    # RayLoraLoader is only emitted when the selected RayLight model binding
    # actually carries a LoRA.  Keeping it outside the base backend contract
    # prevents a LoRA-free RayLight installation from being reported as
    # unavailable; exact compiled-graph preflight still fails closed whenever
    # a selected workflow does emit this conditional node.
    RAYLIGHT_LORA_REQUIRED_NODES = ("RayLoraLoader",)

    @staticmethod
    def raylight_initializer_contract_issues(node_info: Any) -> list[str]:
        """Validate the Director-specific RayLight initializer API schema.

        ComfyUI drops unknown optional prompt inputs.  Looking only at the node
        name would therefore accept stock/older RayLight while silently losing
        Director's scoped cleanup and RAM-cache behavior.
        """

        if not isinstance(node_info, dict):
            return ["RayInitializerAdvanced object_info is missing"]
        inputs = node_info.get("input")
        if not isinstance(inputs, dict):
            return ["RayInitializerAdvanced input schema is missing"]
        required = inputs.get("required")
        optional = inputs.get("optional")
        if not isinstance(required, dict) or not isinstance(optional, dict):
            return ["RayInitializerAdvanced required/optional schema is missing"]

        issues: list[str] = []
        attention = required.get("XFuser_attention")
        if not isinstance(attention, (list, tuple)) or not attention:
            issues.append("XFuser_attention must be a required combo")
        else:
            metadata = (
                attention[1]
                if len(attention) > 1 and isinstance(attention[1], dict)
                else {}
            )
            choices = attention[0]
            if choices == "COMBO":
                choices = metadata.get("options")
            if (
                not isinstance(choices, (list, tuple))
                or "COMFY_KITCHEN_INT8" not in choices
                or "TORCH_FLASH" not in choices
            ):
                issues.append(
                    "XFuser_attention must offer COMFY_KITCHEN_INT8 and TORCH_FLASH"
                )
            if metadata.get("default") != "TORCH_FLASH":
                issues.append("XFuser_attention default must be TORCH_FLASH")

        policy = optional.get("driver_cleanup_policy")
        policy_choices = (
            policy[0]
            if isinstance(policy, (list, tuple)) and policy
            else None
        )
        if (
            not isinstance(policy_choices, (list, tuple))
            or "legacy_all" not in policy_choices
            or "ray_devices" not in policy_choices
        ):
            issues.append(
                "driver_cleanup_policy must offer legacy_all and ray_devices"
            )

        ram_cache = optional.get("ram_cache_max_models")
        ram_metadata = (
            ram_cache[1]
            if isinstance(ram_cache, (list, tuple))
            and len(ram_cache) > 1
            and isinstance(ram_cache[1], dict)
            else {}
        )
        if (
            not isinstance(ram_cache, (list, tuple))
            or not ram_cache
            or ram_cache[0] != "INT"
            or ram_metadata.get("default") != 2
            or ram_metadata.get("min") != 0
        ):
            issues.append(
                "ram_cache_max_models must be optional INT with default 2 and min 0"
            )
        return issues

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport

    @asynccontextmanager
    async def _http(self, *, timeout: float | None = None) -> AsyncIterator[httpx.AsyncClient]:
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout if timeout is not None else self.timeout,
            transport=self.transport,
        ) as client:
            yield client

    @staticmethod
    async def _json(response: httpx.Response) -> Any:
        try:
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail: Any
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise ComfyError(
                f"ComfyUI returned HTTP {response.status_code}",
                status_code=502,
                detail=detail,
            ) from exc
        except ValueError as exc:
            raise ComfyError("ComfyUI returned invalid JSON", detail=response.text[:1000]) from exc

    async def capabilities(self) -> dict[str, Any]:
        started = time.monotonic()
        try:
            async with self._http(timeout=10) as client:
                response = await client.get("/object_info")
                nodes = await self._json(response)
                features_response = await client.get("/features")
                features = await self._json(features_response)
                # Native atomic cancel is idempotent for an unknown UUID, so
                # this probes capability without touching queue state. Older
                # ComfyUI versions return 404.
                cancel_probe = await client.post(
                    f"/api/jobs/{uuid.uuid4()}/cancel"
                )
        except (httpx.HTTPError, ComfyError) as exc:
            raise ComfyError(f"Cannot connect to ComfyUI: {exc}") from exc
        available = sorted(str(key) for key in nodes) if isinstance(nodes, dict) else []
        node_provenance = {
            str(name): str(info.get("python_module", ""))
            for name, info in (nodes.items() if isinstance(nodes, dict) else [])
            if isinstance(info, dict)
        }
        standard_missing = [
            node for node in self.STANDARD_REQUIRED_NODES if node not in available
        ]
        raylight_missing = [
            node for node in self.RAYLIGHT_REQUIRED_NODES if node not in available
        ]
        raylight_lora_missing = [
            node
            for node in self.RAYLIGHT_LORA_REQUIRED_NODES
            if node not in available
        ]
        raylight_contract_issues = (
            self.raylight_initializer_contract_issues(
                nodes.get("RayInitializerAdvanced")
            )
            if isinstance(nodes, dict) and "RayInitializerAdvanced" in nodes
            else []
        )
        continuity_missing = [
            node for node in self.CONTINUITY_REQUIRED_NODES if node not in available
        ]
        continuity_invalid_provenance = [
            node
            for node, expected_module in self.CONTINUITY_REQUIRED_NODE_MODULES.items()
            if node in available and node_provenance.get(node) != expected_module
        ]
        try:
            cancel_payload = cancel_probe.json()
        except ValueError:
            cancel_payload = None
        supports_cancel = (
            cancel_probe.status_code == 200
            and isinstance(cancel_payload, dict)
            and set(cancel_payload) == {"cancelled"}
            and isinstance(cancel_payload.get("cancelled"), bool)
        )
        return {
            "connection": "online",
            "supported_modes": ["t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"] if not standard_missing else [],
            "supports_cancel": supports_cancel,
            "available_nodes": available,
            "node_provenance": node_provenance,
            "missing_nodes": standard_missing,
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "features": features if isinstance(features, dict) else {},
            "native_timeline": {
                "supported": not standard_missing,
                # Public timeline v2 exposes model families. The six recipe
                # names remain under supported_modes for legacy draft APIs and
                # are derived server-side for timeline execution.
                "modes": ["fl2va", "ref2va"],
                "continuity": (
                    not standard_missing
                    and not continuity_missing
                    and not continuity_invalid_provenance
                ),
            },
            "execution_backends": {
                "standard": {
                    "available": not standard_missing,
                    "missing_nodes": standard_missing,
                },
                "raylight": {
                    "available": not raylight_missing and not raylight_contract_issues,
                    "missing_nodes": raylight_missing,
                    "contract_issues": raylight_contract_issues,
                    "conditional_requirements": {
                        "lora": {
                            "available": not raylight_lora_missing,
                            "missing_nodes": raylight_lora_missing,
                        }
                    },
                },
            },
        }

    async def models(self) -> dict[str, list[str]]:
        async with self._http() as client:
            diffusion = await self._json(await client.get("/models/diffusion_models"))
            clips = await self._json(await client.get("/models/text_encoders"))
            vaes = await self._json(await client.get("/models/vae"))
            loras = await self._json(await client.get("/models/loras"))
        diffusion_list = sorted(str(item) for item in diffusion)
        clip_list = sorted(str(item) for item in clips)
        vae_list = sorted(str(item) for item in vaes)
        lora_list = sorted(str(item) for item in loras)
        return {
            # FL2VA and Ref2VA are independent settings slots, not filename
            # classifiers. ComfyUI is the authority for which diffusion models
            # exist, so both selectors must receive its complete inventory.
            "fl2va": list(diffusion_list),
            "ref2va": list(diffusion_list),
            "clip": clip_list,
            "video_vae": [item for item in vae_list if "video" in item.lower()],
            "audio_vae": [item for item in vae_list if "audio" in item.lower()],
            "loras": lora_list,
        }

    async def lora_metadata(self, filename: str) -> dict[str, str] | None:
        """Read one remote LoRA's safetensors metadata without loading it.

        ComfyUI owns the model filesystem and may be on another host. Its
        fixed-folder metadata route performs path validation and reads only the
        bounded safetensors header. A 404 means the file has no inspectable
        metadata contract; other failures remain upstream errors.
        """

        async with self._http() as client:
            response = await client.get(
                "/view_metadata/loras", params={"filename": filename}
            )
        if response.status_code == 404:
            return None
        value = await self._json(response)
        if (
            not isinstance(value, dict)
            or len(value) > 256
            or any(
                not isinstance(key, str)
                or not isinstance(item, str)
                or len(key) > 256
                or len(item) > 8192
                for key, item in value.items()
            )
            or sum(len(key) + len(item) for key, item in value.items()) > 65_536
        ):
            raise ComfyError(
                "ComfyUI /view_metadata/loras returned invalid metadata"
            )
        return dict(value)

    async def system_stats(self) -> dict[str, Any]:
        async with self._http() as client:
            value = await self._json(await client.get("/system_stats"))
        if not isinstance(value, dict):
            raise ComfyError("ComfyUI /system_stats returned an invalid object")
        return value

    async def upload(
        self,
        filename: str,
        content: bytes | Path,
        content_type: str,
        kind: AssetKind,
    ) -> dict[str, Any]:
        async with self._http(timeout=300) as client:
            # ComfyUI's standard upload endpoint stores arbitrary input media;
            # the historical Director chunk route is intentionally not used.
            data = {"type": "input", "subfolder": "director-web"}
            if isinstance(content, Path):
                with content.open("rb") as stream:
                    response = await client.post(
                        "/upload/image",
                        data=data,
                        files={"image": (filename, stream, content_type)},
                    )
            else:
                response = await client.post(
                    "/upload/image", data=data, files={"image": (filename, content, content_type)}
                )
            value = await self._json(response)
        if not isinstance(value, dict) or not value.get("name"):
            raise ComfyError("ComfyUI upload response is missing the asset name", detail=value)
        return value

    async def upload_output(
        self,
        filename: str,
        content: bytes,
        content_type: str,
        subfolder: str,
    ) -> dict[str, Any]:
        async with self._http(timeout=300) as client:
            data = {"type": "output", "subfolder": subfolder}
            files = {"image": (filename, content, content_type)}
            value = await self._json(
                await client.post("/upload/image", data=data, files=files)
            )
        if not isinstance(value, dict) or not value.get("name"):
            raise ComfyError(
                "ComfyUI output upload response is missing the asset name",
                detail=value,
            )
        return value

    async def submit(
        self,
        prompt: dict[str, Any],
        client_id: str,
        prompt_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "client_id": client_id,
            # Server-owned, low-cost video-latent previews. The backend accepts
            # only metadata-bearing event-4 frames from registered sampler
            # nodes; clients cannot choose a decoder or submit extra_data.
            "extra_data": {"preview_method": "latent2rgb"},
        }
        if prompt_id is not None:
            payload["prompt_id"] = prompt_id
        async with self._http(timeout=60) as client:
            value = await self._json(
                await client.post("/prompt", json=payload)
            )
            if not isinstance(value, dict) or not value.get("prompt_id"):
                raise ComfyError("ComfyUI accepted no prompt id", detail=value)
            if prompt_id is None or str(value["prompt_id"]) == prompt_id:
                return value

            actual_prompt_id = str(value["prompt_id"])
            mismatch_detail: dict[str, Any] = {
                "requested_prompt_id": prompt_id,
                "actual_prompt_id": actual_prompt_id,
                "submit_response": value,
            }
            # The incompatible server has already queued a different id. Keep
            # response validation and exact cleanup inside this live client
            # lifetime: calling the generic cancel fallback could add a queue
            # race or target anything other than the id ComfyUI actually
            # minted.
            safe_actual_id = quote(actual_prompt_id, safe="")
            try:
                cleanup_response = await client.post(
                    f"/api/jobs/{safe_actual_id}/cancel"
                )
            except httpx.HTTPError as exc:
                mismatch_detail["cleanup_error"] = str(exc)
                raise ComfyError(
                    "ComfyUI returned a different prompt id; atomic cleanup "
                    "failed and the unexpected job may still be queued or running",
                    detail=mismatch_detail,
                ) from exc

            mismatch_detail["cleanup_status_code"] = cleanup_response.status_code
            if cleanup_response.status_code == 404:
                try:
                    mismatch_detail["cleanup_response"] = cleanup_response.json()
                except ValueError:
                    mismatch_detail["cleanup_response"] = cleanup_response.text[:1000]
                raise ComfyError(
                    "ComfyUI returned a different prompt id but has no atomic "
                    "cleanup endpoint; the unexpected job may still be queued "
                    "or running",
                    detail=mismatch_detail,
                )

            try:
                cleanup_value = await self._json(cleanup_response)
            except ComfyError as exc:
                mismatch_detail["cleanup_error"] = str(exc)
                mismatch_detail["cleanup_response"] = exc.detail
                raise ComfyError(
                    "ComfyUI returned a different prompt id; atomic cleanup "
                    "errored and the unexpected job may still be queued or running",
                    detail=mismatch_detail,
                ) from exc

            mismatch_detail["cleanup_response"] = cleanup_value
            if (
                not isinstance(cleanup_value, dict)
                or set(cleanup_value) != {"cancelled"}
                or not isinstance(cleanup_value.get("cancelled"), bool)
            ):
                raise ComfyError(
                    "ComfyUI returned a different prompt id and an invalid "
                    "atomic cleanup response; the unexpected job may still be "
                    "queued or running",
                    detail=mismatch_detail,
                )
            if cleanup_value["cancelled"] is not True:
                raise ComfyError(
                    "ComfyUI returned a different prompt id and atomic cleanup "
                    "was not confirmed; the unexpected job may still be queued "
                    "or running",
                    detail=mismatch_detail,
                )
            raise ComfyError(
                "ComfyUI returned a different prompt id than requested; the "
                "unexpected job was atomically cancelled",
                detail=mismatch_detail,
            )

    async def history(
        self, prompt_id: str | None = None, *, max_items: int | None = None
    ) -> dict[str, Any]:
        path = f"/history/{prompt_id}" if prompt_id else "/history"
        params = (
            {"max_items": max_items}
            if prompt_id is None and max_items is not None
            else None
        )
        async with self._http() as client:
            value = await self._json(await client.get(path, params=params))
        if not isinstance(value, dict):
            raise ComfyError(
                "ComfyUI history returned an invalid object", detail=value
            )
        return value

    async def queue(self) -> dict[str, Any]:
        async with self._http() as client:
            value = await self._json(await client.get("/queue"))
        if not isinstance(value, dict):
            raise ComfyError(
                "ComfyUI queue returned an invalid object", detail=value
            )
        return value

    @staticmethod
    def _queue_contains(entries: Any, prompt_id: str) -> bool:
        if not isinstance(entries, list):
            return False
        for entry in entries:
            if isinstance(entry, list) and len(entry) > 1 and str(entry[1]) == prompt_id:
                return True
            if isinstance(entry, dict) and str(entry.get("prompt_id", "")) == prompt_id:
                return True
        return False

    async def cancel(self, prompt_id: str) -> bool:
        async with self._http() as client:
            safe_prompt_id = quote(prompt_id, safe="")
            atomic_response = await client.post(f"/api/jobs/{safe_prompt_id}/cancel")
            if atomic_response.status_code != 404:
                value = await self._json(atomic_response)
                if not isinstance(value, dict) or not isinstance(value.get("cancelled"), bool):
                    raise ComfyError(
                        "ComfyUI atomic cancel returned an invalid response", detail=value
                    )
                return value["cancelled"]

            # There is no race-free targeted cancellation in older ComfyUI.
            # Even pending queue deletion can race with that prompt becoming
            # the active global job. Fail closed rather than claim success
            # while GPU work may continue or interrupt an unrelated prompt.
            queue_value = await self._json(await client.get("/queue"))
            if not isinstance(queue_value, dict):
                raise ComfyError("ComfyUI queue returned an invalid object", detail=queue_value)
            if self._queue_contains(queue_value.get("queue_pending"), prompt_id):
                raise ComfyError(
                    "ComfyUI does not support atomic job cancellation; upgrade "
                    "ComfyUI before cancelling a queued prompt",
                    status_code=409,
                )
            if self._queue_contains(queue_value.get("queue_running"), prompt_id):
                # Older ComfyUI releases treated /interrupt as process-global
                # and ignored a target id. A queue snapshot followed by that
                # call can therefore kill the next unrelated prompt. Both
                # running interruption and pending deletion are deliberately
                # fail-closed unless the native atomic jobs endpoint exists.
                raise ComfyError(
                    "ComfyUI does not support atomic running-job cancellation; "
                    "upgrade ComfyUI before cancelling an active prompt",
                    status_code=409,
                )
            return False

    async def view(self, params: dict[str, str]) -> httpx.Response:
        async with self._http(timeout=300) as client:
            response = await client.get("/view", params=params)
            response.raise_for_status()
            return response

    async def view_stream(
        self,
        params: dict[str, str],
        *,
        byte_range: str | None = None,
    ) -> ComfyMediaStream:
        """Open `/view` without buffering the generated media in Director."""

        client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=300,
            transport=self.transport,
        )
        headers = {"Accept-Encoding": "identity"}
        if byte_range is not None:
            headers["Range"] = byte_range
        try:
            request = client.build_request("GET", "/view", params=params, headers=headers)
            response = await client.send(request, stream=True)
            if response.status_code not in {200, 206, 416}:
                await response.aread()
                if response.is_error or response.is_redirect:
                    response.raise_for_status()
                raise ComfyError(
                    "ComfyUI returned an unsupported media response status",
                    detail={"status_code": response.status_code},
                )
            return ComfyMediaStream(response=response, client=client)
        except BaseException:
            await client.aclose()
            raise


def default_comfy_factory(settings: RuntimeSettings) -> ComfyClient:
    return ComfyClient(str(settings.comfy_url))
