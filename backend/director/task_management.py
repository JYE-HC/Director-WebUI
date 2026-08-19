from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import anyio
from pydantic import ValidationError

from .comfy import ComfyClientProtocol
from .media import create_24fps_proxy_bytes
from .public_url import public_api_url
from .schemas import AssetReference, RuntimeSettings, UnifiedTimelineDraft


class AssetRegistry(Protocol):
    """The narrow persistence surface needed by generated-output imports."""

    def put_asset(
        self,
        asset_id: str,
        document: dict[str, Any],
        *,
        comfy_origin: str,
    ) -> None: ...

    def put_asset_if_current_origin(
        self,
        asset_id: str,
        document: dict[str, Any],
        *,
        expected_comfy_origin: str,
    ) -> bool: ...


ComfyFactory = Callable[[RuntimeSettings], ComfyClientProtocol]
SettingsReader = Callable[[], RuntimeSettings]


class TaskManagementError(ValueError):
    """A fail-closed task-management request that is safe to show to clients."""

    def __init__(self, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


_MAX_IMPORTED_VIDEO_BYTES = 512 * 1024 * 1024
_VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi", ".mpeg", ".mpg"}
)
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._\-()\u4e00-\u9fff]+")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


def canonical_comfy_origin(value: Any) -> str:
    origin = str(value).strip().rstrip("/")
    if not origin:
        raise TaskManagementError(
            "ComfyUI 地址尚未配置，无法导入生成结果",
            status_code=409,
        )
    return origin


def _safe_filename(value: str) -> str:
    basename = value.replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = _SAFE_FILENAME.sub("_", basename).strip("._")
    return cleaned or "generated-video.mp4"


def _output_reference(value: Any) -> dict[str, str]:
    """Validate an immutable Comfy history reference without rewriting it.

    The original filename and subfolder are sent back to the job's captured
    ComfyUI origin. Sanitising either field before that read could select a
    different file, so malformed values are rejected instead.
    """

    if not isinstance(value, Mapping):
        raise TaskManagementError("任务输出记录无效", status_code=409)
    filename = value.get("filename")
    subfolder = value.get("subfolder", "")
    output_type = value.get("type", "output")
    if (
        not isinstance(filename, str)
        or not filename
        or len(filename) > 1024
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or _CONTROL_CHARACTERS.search(filename)
    ):
        raise TaskManagementError("任务输出文件名无效", status_code=409)
    if (
        not isinstance(subfolder, str)
        or len(subfolder) > 1024
        or subfolder.startswith(("/", "\\"))
        or "\\" in subfolder
        or _CONTROL_CHARACTERS.search(subfolder)
        or any(component in {"", ".", ".."} for component in subfolder.split("/") if subfolder)
    ):
        raise TaskManagementError("任务输出子目录无效", status_code=409)
    if output_type != "output":
        raise TaskManagementError("只能把持久化的 ComfyUI output 加入素材库", status_code=409)
    if Path(filename).suffix.lower() not in _VIDEO_EXTENSIONS:
        raise TaskManagementError("当前导演台只支持把生成视频加入素材库")
    return {
        "filename": filename,
        "subfolder": subfolder,
        "type": "output",
    }


def _segment_output(job: Mapping[str, Any], segment_id: str) -> dict[str, str]:
    if not segment_id or len(segment_id) > 128:
        raise TaskManagementError("分段 ID 无效")
    snapshot = job.get("config_snapshot")
    if not isinstance(snapshot, Mapping):
        raise TaskManagementError("该历史任务没有有效的项目快照", status_code=409)
    try:
        timeline = UnifiedTimelineDraft.model_validate(snapshot.get("timeline"))
    except ValueError as exc:
        raise TaskManagementError(
            "该历史任务的项目快照不符合当前时间线契约",
            status_code=409,
        ) from exc
    known_ids = {segment.id for segment in timeline.segments}
    if segment_id not in known_ids:
        raise TaskManagementError("任务快照中不存在该分段", status_code=404)

    matches: list[dict[str, str]] = []
    children = job.get("children")
    if not isinstance(children, list):
        raise TaskManagementError("任务没有可用的分段输出记录", status_code=404)
    for child in children:
        if not isinstance(child, Mapping):
            continue
        declared = child.get("segment_ids")
        output_nodes = child.get("output_nodes")
        outputs = child.get("outputs")
        if (
            not isinstance(declared, list)
            or segment_id not in declared
            or not isinstance(output_nodes, Mapping)
            or not isinstance(outputs, list)
        ):
            continue
        node_id = output_nodes.get(segment_id)
        if not isinstance(node_id, str) or not node_id:
            continue
        for output in outputs:
            if (
                isinstance(output, Mapping)
                and str(output.get("node_id") or "") == node_id
            ):
                matches.append(_output_reference(output))
    if not matches:
        raise TaskManagementError("未找到该分段的生成结果", status_code=404)
    if len(matches) != 1:
        raise TaskManagementError("该分段的生成结果不唯一，拒绝猜测", status_code=409)
    return matches[0]


def resolve_job_output(
    job: Mapping[str, Any],
    *,
    output_index: int | None = None,
    segment_id: str | None = None,
) -> dict[str, str]:
    """Resolve exactly one persisted output; never accept a path or URL."""

    if (output_index is None) == (segment_id is None):
        raise TaskManagementError("必须且只能选择一个任务输出或分段输出")
    if segment_id is not None:
        return _segment_output(job, segment_id)
    if (
        not isinstance(output_index, int)
        or isinstance(output_index, bool)
        or output_index < 0
    ):
        raise TaskManagementError("任务输出序号无效")
    outputs = job.get("outputs")
    if not isinstance(outputs, list) or output_index >= len(outputs):
        raise TaskManagementError("任务输出不存在", status_code=404)
    return _output_reference(outputs[output_index])


async def import_job_output_as_asset(
    *,
    registry: AssetRegistry,
    comfy_factory: ComfyFactory,
    job: Mapping[str, Any],
    target_settings: RuntimeSettings,
    current_settings: SettingsReader | None = None,
    output_index: int | None = None,
    segment_id: str | None = None,
) -> AssetReference:
    """Copy one job-owned output into the active ComfyUI input library.

    Source reads are bound to the immutable job settings snapshot. The target
    is the current live settings origin. Even when both origins match, bytes
    pass through the normal 24fps proxy and input-upload contract; an output
    path is never re-labelled as a trusted input asset.
    """

    output = resolve_job_output(
        job,
        output_index=output_index,
        segment_id=segment_id,
    )
    try:
        source_settings = RuntimeSettings.model_validate(job.get("settings_snapshot"))
    except ValueError as exc:
        raise TaskManagementError(
            "任务缺少有效的 ComfyUI 设置快照，无法读取生成结果",
            status_code=409,
        ) from exc
    canonical_comfy_origin(source_settings.comfy_url)
    target_origin = canonical_comfy_origin(target_settings.comfy_url)

    upstream = await comfy_factory(source_settings).view(output)
    content = upstream.content
    if not content:
        raise TaskManagementError("生成结果为空，无法加入素材库", status_code=502)
    if len(content) > _MAX_IMPORTED_VIDEO_BYTES:
        raise TaskManagementError(
            "生成结果超过素材库 512 MiB 视频上限",
            status_code=413,
        )

    proxy = await anyio.to_thread.run_sync(
        partial(
            create_24fps_proxy_bytes,
            content,
            Path(output["filename"]).suffix or ".mp4",
        )
    )
    if len(proxy.content) > _MAX_IMPORTED_VIDEO_BYTES:
        raise TaskManagementError(
            "标准化后的生成视频超过素材库 512 MiB 上限",
            status_code=413,
        )
    upload_name = f"{Path(_safe_filename(output['filename'])).stem}_24fps.mp4"
    uploaded = await comfy_factory(target_settings).upload(
        upload_name,
        proxy.content,
        "video/mp4",
        "video",
    )
    if not isinstance(uploaded, Mapping):
        raise TaskManagementError("ComfyUI 返回了无效的素材上传结果", status_code=502)
    name = uploaded.get("name")
    subfolder = uploaded.get("subfolder", "")
    uploaded_type = uploaded.get("type")
    if (
        not isinstance(name, str)
        or not name
        or len(name) > 1024
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or _CONTROL_CHARACTERS.search(name)
        or not isinstance(subfolder, str)
        or len(subfolder) > 1024
        or subfolder.startswith(("/", "\\"))
        or "\\" in subfolder
        or _CONTROL_CHARACTERS.search(subfolder)
        or any(component in {"", ".", ".."} for component in subfolder.split("/") if subfolder)
        or uploaded_type != "input"
    ):
        raise TaskManagementError("ComfyUI 返回了无效的素材路径", status_code=502)
    asset_id = str(uuid.uuid4())
    path = f"{subfolder.strip('/')}/{name}".strip("/")
    try:
        asset = AssetReference.model_validate(
            {
                "name": name,
                "subfolder": subfolder,
                "type": "input",
                "kind": "video",
                "id": asset_id,
                "filename": name,
                "path": path,
                "preview_url": public_api_url(f"/api/assets/{asset_id}/preview"),
                "content_hash": f"sha256:{hashlib.sha256(proxy.content).hexdigest()}",
                "metadata": proxy.metadata.model_dump(mode="json"),
            }
        )
    except ValidationError as exc:
        raise TaskManagementError(
            "ComfyUI 返回了不符合素材契约的上传结果",
            status_code=502,
        ) from exc
    document = asset.model_dump(mode="json")
    if current_settings is not None:
        current_origin = canonical_comfy_origin(current_settings().comfy_url)
        if current_origin != target_origin:
            # The old target may now contain an unregistered upload, but it
            # must not be returned as a usable asset after authority changed.
            raise TaskManagementError(
                "导入期间 ComfyUI 地址已变更，结果未加入当前素材库，请重试",
                status_code=409,
            )
    if not registry.put_asset_if_current_origin(
        asset_id,
        document,
        expected_comfy_origin=target_origin,
    ):
        raise TaskManagementError(
            "导入期间 ComfyUI 地址已变更，结果未加入当前素材库，请重试",
            status_code=409,
        )
    return asset
