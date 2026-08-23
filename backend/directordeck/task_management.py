from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import anyio
from pydantic import ValidationError

from .comfy import ComfyClientProtocol
from .media import create_24fps_proxy_bytes
from .public_url import public_api_url
from .schemas import (
    AssetReference,
    UnifiedTimelineDraft,
    UnifiedTimelineDraftV5,
)
from .workflow.execution import (
    CompiledExecutionPlan,
    LockedSubmissionPlan,
    ObservedArtifactSpec,
    ObservedAssemblyArtifactSpec,
    compiled_execution_plan_digest,
    ordered_compiled_segment_units,
    sha256_document_digest,
)


class AssetRegistry(Protocol):
    """The narrow persistence surface needed by generated-output imports."""

    def put_asset(
        self,
        asset_id: str,
        document: dict[str, Any],
    ) -> None: ...


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
_TYPED_PARENT_OUTPUT_AUTHORITY = "_typed_parent_output_authority"
_COMPILED_EXECUTION_PLAN_AUTHORITY = "_compiled_execution_plan_authority"
_OBSERVED_ASSEMBLY_AUTHORITY = "_observed_assembly_authority"


def attach_parent_output_authority(
    job: Mapping[str, Any],
    *,
    typed: bool,
    compiled_plan: CompiledExecutionPlan | None,
    observed_assembly: ObservedAssemblyArtifactSpec | None,
) -> dict[str, Any]:
    """Attach trusted, process-local contracts without mutating durable columns."""

    enriched = dict(job)
    enriched[_TYPED_PARENT_OUTPUT_AUTHORITY] = typed
    enriched[_COMPILED_EXECUTION_PLAN_AUTHORITY] = compiled_plan
    enriched[_OBSERVED_ASSEMBLY_AUTHORITY] = observed_assembly
    return enriched


def attached_compiled_execution_plan(
    job: Mapping[str, Any],
) -> CompiledExecutionPlan | None:
    """Read the process-local typed plan attached by the app projector.

    The private key never crosses a response or persistence boundary. Keeping
    this accessor beside the attachment helper prevents other modules from
    duplicating its sentinel name or accepting an unvalidated mapping.
    """

    plan = job.get(_COMPILED_EXECUTION_PLAN_AUTHORITY)
    return plan if isinstance(plan, CompiledExecutionPlan) else None


def ordered_observed_artifacts(
    job: Mapping[str, Any],
) -> tuple[ObservedArtifactSpec, ...]:
    """Resolve every typed segment in captured timeline order, or fail closed."""

    plan = job.get(_COMPILED_EXECUTION_PLAN_AUTHORITY)
    if not isinstance(plan, CompiledExecutionPlan):
        raise TaskManagementError(
            "任务没有可信的编译执行计划",
            status_code=409,
        )
    children = job.get("children")
    if not isinstance(children, list):
        raise TaskManagementError("任务没有可信的分段执行记录", status_code=409)
    config_snapshot = job.get("config_snapshot")
    if not isinstance(config_snapshot, Mapping):
        raise TaskManagementError("任务没有可信的时间线快照", status_code=409)
    try:
        ordered_units = ordered_compiled_segment_units(plan, config_snapshot)
    except ValueError as exc:
        raise TaskManagementError(
            "任务的时间线顺序与编译执行计划不一致",
            status_code=409,
        ) from exc
    plan_digest = compiled_execution_plan_digest(plan)
    ordered: list[ObservedArtifactSpec] = []
    for unit in ordered_units:
        matches: list[ObservedArtifactSpec] = []
        for child in children:
            if not isinstance(child, Mapping):
                continue
            evidence = child.get("execution_evidence")
            if not isinstance(evidence, Mapping):
                continue
            locked_plan = evidence.get("locked_submission_plan")
            snapshot = evidence.get("exact_prompt_snapshot")
            expected = (
                snapshot.expected_output_spec
                if hasattr(snapshot, "expected_output_spec")
                else None
            )
            artifact = child.get("observed_artifact")
            if (
                not isinstance(locked_plan, LockedSubmissionPlan)
                or snapshot is None
                or expected is None
                or not isinstance(artifact, ObservedArtifactSpec)
                or str(evidence.get("child_id") or "")
                != str(child.get("id") or "")
                or str(child.get("job_id") or "") != str(job.get("id") or "")
                or child.get("status") != "succeeded"
                or child.get("segment_ids") != [unit.owner_segment_id]
                or locked_plan.source_compiled_plan_digest != plan_digest
                or locked_plan.source_unit_id != unit.id
                or snapshot.unit_id != unit.id
                or snapshot.owner_segment_id != unit.owner_segment_id
                or expected != unit.expected_output_spec
                or artifact.child_id != str(child.get("id") or "")
                or artifact.segment_id != unit.owner_segment_id
            ):
                continue
            matches.append(artifact)
        if len(matches) != 1:
            raise TaskManagementError(
                "任务分段缺少唯一的 Expected+Observed 输出证据",
                status_code=409,
            )
        ordered.append(matches[0])
    return tuple(ordered)


def _is_full_timeline_selection(snapshot: Mapping[str, Any]) -> bool:
    segment_ids = snapshot.get("segment_ids")
    if segment_ids is None:
        return True
    timeline = snapshot.get("timeline")
    if not isinstance(segment_ids, list) or not isinstance(timeline, Mapping):
        return False
    segments = timeline.get("segments")
    if not isinstance(segments, list):
        return False
    enabled_ids = {
        str(segment["id"])
        for segment in segments
        if isinstance(segment, Mapping)
        and isinstance(segment.get("id"), str)
        and segment.get("enabled", True) is True
    }
    return bool(enabled_ids) and {str(item) for item in segment_ids} == enabled_ids


def authoritative_parent_outputs(
    job: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Return the only parent output projection accepted by every API path."""

    if not bool(job.get(_TYPED_PARENT_OUTPUT_AUTHORITY)):
        outputs = job.get("outputs")
        if not isinstance(outputs, list):
            return []
        return [dict(output) for output in outputs if isinstance(output, Mapping)]

    if job.get("status") != "succeeded":
        return []

    plan = job.get(_COMPILED_EXECUTION_PLAN_AUTHORITY)
    if not isinstance(plan, CompiledExecutionPlan):
        return []
    config_snapshot = job.get("config_snapshot")
    if not isinstance(config_snapshot, Mapping):
        return []
    try:
        ordered_units = ordered_compiled_segment_units(plan, config_snapshot)
        artifacts = ordered_observed_artifacts(job)
    except (TaskManagementError, ValueError):
        return []
    outputs = [
        {
            "node_id": unit.expected_output_spec.node_id,
            **artifact.output_descriptor.model_dump(mode="json"),
        }
        for unit, artifact in zip(ordered_units, artifacts, strict=True)
    ]

    timeline = (
        config_snapshot.get("timeline")
        if isinstance(config_snapshot, Mapping)
        else None
    )
    if not isinstance(timeline, Mapping):
        return []
    assemble_full_timeline = (
        timeline.get("export_mode") == "all"
        and _is_full_timeline_selection(config_snapshot)
    )
    if not assemble_full_timeline or len(artifacts) == 1:
        return outputs

    assembly = job.get(_OBSERVED_ASSEMBLY_AUTHORITY)
    if not isinstance(assembly, ObservedAssemblyArtifactSpec):
        return []
    expected_plan_digest = compiled_execution_plan_digest(plan)
    if (
        assembly.job_id != str(job.get("id") or "")
        or assembly.source_compiled_plan_digest != expected_plan_digest
        or len(assembly.source_artifacts) != len(artifacts)
    ):
        return []
    for source, unit, artifact in zip(
        assembly.source_artifacts,
        ordered_units,
        artifacts,
        strict=True,
    ):
        if (
            source.segment_id != unit.owner_segment_id
            or source.child_id != artifact.child_id
            or source.observed_artifact_digest
            != sha256_document_digest(artifact.model_dump(mode="json"))
        ):
            return []
    return [
        {
            "node_id": "assembly",
            **assembly.output_descriptor.model_dump(mode="json"),
        }
    ]


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
    raw_timeline = snapshot.get("timeline")
    try:
        if not isinstance(raw_timeline, Mapping):
            raise ValueError("timeline snapshot is not an object")
        if raw_timeline.get("version") == 5:
            timeline = UnifiedTimelineDraftV5.model_validate(raw_timeline)
        else:
            # Historical v4 jobs remain readable, but future/unknown schema
            # versions never get projected by guessing.
            timeline = UnifiedTimelineDraft.model_validate(raw_timeline)
    except ValueError as exc:
        raise TaskManagementError(
            "该历史任务的项目快照不符合当前时间线契约",
            status_code=409,
        ) from exc
    known_ids = {segment.id for segment in timeline.segments}
    if segment_id not in known_ids:
        raise TaskManagementError("任务快照中不存在该分段", status_code=404)

    matches: list[dict[str, str]] = []
    typed_segment = False
    children = job.get("children")
    if not isinstance(children, list):
        raise TaskManagementError("任务没有可用的分段输出记录", status_code=404)
    for child in children:
        if not isinstance(child, Mapping):
            continue
        execution_evidence = child.get("execution_evidence")
        if isinstance(execution_evidence, Mapping):
            if segment_id in child.get("segment_ids", []):
                # Mutable membership is used only to report an unavailable
                # typed result, never to locate or expose a file.
                typed_segment = True
            snapshot = execution_evidence.get("exact_prompt_snapshot")
            expected = (
                snapshot.expected_output_spec
                if hasattr(snapshot, "expected_output_spec")
                else None
            )
            if expected is None or expected.segment_id != segment_id:
                continue
            typed_segment = True
            artifact = child.get("observed_artifact")
            if (
                isinstance(artifact, ObservedArtifactSpec)
                and child.get("status") == "succeeded"
                and artifact.child_id == str(child.get("id") or "")
                and artifact.segment_id == expected.segment_id
            ):
                matches.append(
                    _output_reference(
                        artifact.output_descriptor.model_dump(mode="json")
                    )
                )
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
    if typed_segment and not matches:
        raise TaskManagementError(
            "该分段尚无可信的实际媒体探测结果",
            status_code=409,
        )
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
    outputs = authoritative_parent_outputs(job)
    if output_index >= len(outputs):
        raise TaskManagementError("任务输出不存在", status_code=404)
    return _output_reference(outputs[output_index])


async def import_job_output_as_asset(
    *,
    registry: AssetRegistry,
    client: ComfyClientProtocol,
    job: Mapping[str, Any],
    output_index: int | None = None,
    segment_id: str | None = None,
) -> AssetReference:
    """Copy one job-owned output into the host ComfyUI input library.

    The embedded backend talks to exactly one ComfyUI instance, so the source
    read and the target upload share that single client. Even so, bytes pass
    through the normal 24fps proxy and input-upload contract; an output path
    is never re-labelled as a trusted input asset.
    """

    output = resolve_job_output(
        job,
        output_index=output_index,
        segment_id=segment_id,
    )

    upstream = await client.view(output)
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
    uploaded = await client.upload(
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
    registry.put_asset(asset_id, document)
    return asset
