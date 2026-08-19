from __future__ import annotations

import asyncio
import json
import hashlib
import logging
import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Awaitable, Callable, Literal
from urllib.parse import quote

import httpx
import anyio
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Path as ApiPath,
    Query,
    Request,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from .comfy import ComfyClient, ComfyClientProtocol, ComfyError, default_comfy_factory
from .compiler import (
    DraftNotRunnable,
    timeline_segment_take_fingerprint,
    unified_continuity_predecessors,
    validate_runnable,
    validate_unified_runnable,
)
from .database import (
    AssetTrashInUse,
    AssetTrashOriginConflict,
    AssetTrashRestoreConflict,
    Database,
    TimelineComfyOriginConflict,
    TimelineRevisionConflict,
    TimelineRevisionExhausted,
)
from .instance_lock import DirectorInstanceLock
from .public_url import public_api_url, set_public_api_prefix
from .raylight_setup import (
    RayLightInstallConflict,
    RayLightInstallManager,
    RayLightInstallUnavailable,
    default_requirements_path as raylight_default_requirements_path,
    dependencies_installed as raylight_dependencies_installed,
    platform_supported as raylight_platform_supported,
)
from .media import (
    MediaToolError,
    assemble_video_bytes,
    create_24fps_proxy_file,
    detect_shots_bytes,
)
from .media_setup import FFmpegInstallManager, ensure_media_tools_on_path, media_tools_status
from .native_templates import (
    ModelFamily,
    NativeCompileResult,
    NativeHistoricalTake,
    NativeTemplateError,
    NativeWorkflowUnit,
    bind_native_workflow_predecessor_output,
    bind_raylight_runtime_epoch,
    build_raylight_shutdown_unit,
    compile_native_timeline,
    raylight_runtime_descriptor,
    raylight_runtime_logical_gpu_indices,
    raylight_workflow_logical_gpu_indices,
    resolve_execution_backend,
    standard_lora_metadata_requests,
    validate_native_capabilities,
    validate_native_workflow_ready,
    validate_native_workflow_unit_capabilities,
)
from .progress import (
    ComfyExecutionEvent,
    ComfyProgressEvent,
    ComfyPreviewEvent,
    ComfyReconcileHint,
    LivePreviewCache,
    NativeProgressManager,
    child_execution_snapshot,
    child_progress_snapshot,
    sampler_segment_for_node,
)
from .schemas import (
    AssetKind,
    AssetDeleteRead,
    AssetListRead,
    AssetReference,
    AssetTrashBatchRead,
    AssetTrashListRead,
    AssetTrashPurgeRead,
    AssetTrashRequest,
    AssetTrashRestoreRead,
    AssetTrashRestoreRequest,
    ComfyURLRequest,
    CreateJobRequest,
    DetectShotsRequest,
    DetectShotsResponse,
    GenerationMode,
    JobBulkCancelRead,
    JobBulkCancelRequest,
    JobClearRead,
    JobDeleteRead,
    JobDiagnosticRead,
    JobGenerationDetailsRead,
    JobListRead,
    JobOutputImportRead,
    JobOutputImportRequest,
    JobProjectSnapshotRead,
    JobRecoveryConfirmComfyRestartRequest,
    JobRead,
    JobStatus,
    ModeDraft,
    ProjectCreateRequest,
    ProjectDeleteRead,
    ProjectImportRequest,
    ProjectListRead,
    ProjectRenameRequest,
    ProjectSummaryRead,
    RayLightRuntimeRecoveryConfirmRequest,
    RayLightRuntimeStatusRead,
    RuntimeSettings,
    RuntimeSettingsAuthorityRead,
    StorageConfigureRequest,
    StorageMigrateRequest,
    StorageMigrationRead,
    StorageStatusRead,
    TimelineAuthorityRead,
    TimelineAuthorityWriteRequest,
    TimelineComfyOriginConflictRead,
    TimelineCompileRead,
    TimelineJobRequest,
    TimelineRevisionConflictRead,
    TimelineRevisionExhaustedRead,
    UnifiedFL2VASegment,
    UnifiedTimelineDraft,
    VideoMetadata,
    mode_draft_to_timeline,
    timeline_segment_recipe,
    utc_now,
    validate_mode_draft,
    validate_timeline_draft,
)
from .storage import (
    StorageConfigurationError,
    StorageConflictError,
    StorageController,
    StorageOperationError,
    StoragePathError,
    StorageStatus,
    StorageValidationError,
)
from .task_management import TaskManagementError, import_job_output_as_asset


ComfyFactory = Callable[[RuntimeSettings], ComfyClientProtocol]
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._\-()\u4e00-\u9fff]+")
_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
_RAYLIGHT_GENERATION_POLL_SECONDS = 1.0
_TERMINAL_STATUS_ORDER = ("succeeded", "failed", "cancelled")
_SUBMISSION_OWNERSHIP_STAGES = {
    "submitting",
    "cancelling_during_submit",
}
_RECOVERY_OWNERSHIP_STAGES = {
    "submission_cancel_pending",
    "submission_cancel_failed",
    "submission_cancel_unconfirmed",
    "restart_cancel_pending",
    "restart_cancel_unconfirmed",
    "restart_cancel_failed",
}
_PROCESS_OWNERSHIP_STAGES = (
    _SUBMISSION_OWNERSHIP_STAGES | _RECOVERY_OWNERSHIP_STAGES
)
_EXTERNAL_CANCEL_STAGES = {
    "ComfyUI 端已中断",
    "ComfyUI 端任务已移除",
}
_SUBMISSION_CLEANUP_CANCEL_STAGES = {
    "cancelled_after_submission_failure",
    "not_submitted",
}
_DIRECTOR_CANCEL_STAGES = _SUBMISSION_CLEANUP_CANCEL_STAGES | {
    "cancelled",
    "cancelled_after_restart",
    "restart_cancelled_not_submitted",
    "cancelled_after_confirmed_comfy_restart",
}
_COMFY_NOT_CONFIGURED = "ComfyUI 地址尚未配置，请先在系统设置中保存地址"
_UPLOAD_READ_CHUNK = 1024 * 1024
_UPLOAD_PROGRESS_TTL_SECONDS = 15 * 60
_UPLOAD_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
logger = logging.getLogger(__name__)
_UPLOAD_LIMITS: dict[AssetKind, int] = {
    "image": 32 * 1024 * 1024,
    "audio": 128 * 1024 * 1024,
    "video": 512 * 1024 * 1024,
}
_UPLOAD_MIME_BY_EXTENSION: dict[AssetKind, dict[str, frozenset[str]]] = {
    "image": {
        ".png": frozenset({"image/png"}),
        ".jpg": frozenset({"image/jpeg", "image/pjpeg"}),
        ".jpeg": frozenset({"image/jpeg", "image/pjpeg"}),
        ".webp": frozenset({"image/webp"}),
    },
    "audio": {
        ".wav": frozenset({"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}),
        ".mp3": frozenset({"audio/mpeg"}),
        ".flac": frozenset({"audio/flac", "audio/x-flac"}),
        ".ogg": frozenset({"audio/ogg", "application/ogg"}),
        ".oga": frozenset({"audio/ogg", "application/ogg"}),
        ".m4a": frozenset({"audio/mp4", "audio/x-m4a"}),
        ".aac": frozenset({"audio/aac", "audio/x-aac"}),
    },
    "video": {
        ".mp4": frozenset({"video/mp4"}),
        ".m4v": frozenset({"video/mp4", "video/x-m4v"}),
        ".mov": frozenset({"video/quicktime"}),
        ".webm": frozenset({"video/webm"}),
        ".mkv": frozenset({"video/x-matroska", "video/mkv"}),
        ".avi": frozenset({"video/x-msvideo", "video/avi"}),
        ".mpeg": frozenset({"video/mpeg"}),
        ".mpg": frozenset({"video/mpeg"}),
    },
}
_INLINE_MEDIA_PREFIX_BY_EXTENSION: dict[str, str] = {
    extension: f"{kind}/"
    for kind, extensions in _UPLOAD_MIME_BY_EXTENSION.items()
    for extension in extensions
}


class PromptTerminalEvents:
    """Per-prompt websocket terminal hints for dispatcher wait-gates.

    A gate registers one asyncio.Event per prompt and waits for either
    that event (ComfyUI delivered execution_success / execution_error /
    execution_interrupted for the prompt) or the poll timeout. The event
    is only an acceleration hint: the gate always re-reads exact history as
    the authoritative terminal certificate, so a dropped or racing set()
    can never corrupt lifecycle state.
    """

    def __init__(self) -> None:
        self._events: dict[str, asyncio.Event] = {}

    def register(self, prompt_id: str) -> asyncio.Event:
        event = self._events.get(prompt_id)
        if event is None:
            event = asyncio.Event()
            self._events[prompt_id] = event
        return event

    def unregister(self, prompt_id: str) -> None:
        self._events.pop(prompt_id, None)

    def notify(self, prompt_id: str) -> None:
        event = self._events.get(prompt_id)
        if event is not None:
            event.set()


def _safe_filename(value: str | None) -> str:
    basename = os.path.basename((value or "asset.bin").replace("\\", "/"))
    cleaned = _SAFE_FILENAME.sub("_", basename).strip("._")
    return cleaned or "asset.bin"


def _validate_upload_metadata(kind: AssetKind, filename: str, content_type: str | None) -> str:
    extension = Path(filename).suffix.lower()
    allowed_types = _UPLOAD_MIME_BY_EXTENSION[kind].get(extension)
    if allowed_types is None:
        allowed_extensions = ", ".join(sorted(_UPLOAD_MIME_BY_EXTENSION[kind]))
        raise HTTPException(
            status_code=422,
            detail=f"unsupported {kind} extension '{extension or '(none)'}'; allowed: {allowed_extensions}",
        )
    normalized_type = (content_type or "").split(";", 1)[0].strip().lower()
    if normalized_type not in allowed_types:
        raise HTTPException(
            status_code=422,
            detail=(
                f"content type '{normalized_type or '(missing)'}' does not match "
                f"{kind} extension '{extension}'"
            ),
        )
    return normalized_type


async def _read_upload_limited(file: UploadFile, max_bytes: int) -> bytes:
    content = bytearray()
    while True:
        remaining_with_sentinel = max_bytes - len(content) + 1
        chunk = await file.read(min(_UPLOAD_READ_CHUNK, remaining_with_sentinel))
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"uploaded file exceeds the {max_bytes}-byte limit",
            )
    if not content:
        raise HTTPException(status_code=422, detail="uploaded file is empty")
    return bytes(content)


async def _spool_upload_limited(
    file: UploadFile, destination: Path, max_bytes: int
) -> int:
    """Copy an upload to a bounded temporary file without retaining it in RAM."""

    size = 0
    with destination.open("wb") as output:
        while True:
            chunk = await file.read(_UPLOAD_READ_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"uploaded file exceeds the {max_bytes}-byte limit",
                )
            output.write(chunk)
    if size == 0:
        raise HTTPException(status_code=422, detail="uploaded file is empty")
    return size


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_UPLOAD_READ_CHUNK):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _upload_progress(app: FastAPI, upload_id: str | None, **values: Any) -> None:
    if upload_id is None:
        return
    now = time.monotonic()
    progress: dict[str, dict[str, Any]] = app.state.upload_progress
    for key, entry in list(progress.items()):
        if now - float(entry["touched_at"]) > _UPLOAD_PROGRESS_TTL_SECONDS:
            progress.pop(key, None)
    current = progress.get(upload_id, {})
    progress[upload_id] = {**current, **values, "touched_at": now}


def _valid_media_signature(extension: str, content: bytes) -> bool:
    """Cheaply reject disguised uploads before they reach ComfyUI.

    This is deliberately a signature check rather than a full decoder.  It
    covers every extension accepted by this API and complements, rather than
    trusts, the browser-provided MIME value.
    """

    extension = extension.lower()
    if extension == ".png":
        return content.startswith(b"\x89PNG\r\n\x1a\n")
    if extension in {".jpg", ".jpeg"}:
        return content.startswith(b"\xff\xd8\xff")
    if extension == ".webp":
        return len(content) >= 12 and content.startswith(b"RIFF") and content[8:12] == b"WEBP"
    if extension == ".wav":
        return len(content) >= 12 and content.startswith(b"RIFF") and content[8:12] == b"WAVE"
    if extension == ".flac":
        return content.startswith(b"fLaC")
    if extension in {".ogg", ".oga"}:
        return content.startswith(b"OggS")
    if extension == ".mp3":
        return content.startswith(b"ID3") or (
            len(content) >= 2 and content[0] == 0xFF and content[1] & 0xE0 == 0xE0
        )
    if extension == ".aac":
        return len(content) >= 2 and content[0] == 0xFF and content[1] & 0xF0 == 0xF0
    if extension in {".mp4", ".m4v", ".mov", ".m4a"}:
        marker = content.find(b"ftyp", 4, min(len(content), 64))
        return marker >= 4
    if extension in {".webm", ".mkv"}:
        return content.startswith(b"\x1a\x45\xdf\xa3")
    if extension == ".avi":
        return len(content) >= 12 and content.startswith(b"RIFF") and content[8:12] == b"AVI "
    if extension in {".mpeg", ".mpg"}:
        return content.startswith((b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3"))
    return False


def _validate_upload_signature(filename: str, content: bytes) -> None:
    extension = Path(filename).suffix.lower()
    if not _valid_media_signature(extension, content):
        raise HTTPException(
            status_code=422,
            detail=f"uploaded bytes do not match the declared '{extension}' media format",
        )


def _origin_settings(settings: RuntimeSettings, comfy_origin: str) -> RuntimeSettings:
    document = settings.model_dump(mode="json")
    document["comfy_url"] = comfy_origin
    return RuntimeSettings.model_validate(document)


def _media_response(
    upstream: httpx.Response,
    *,
    filename: str,
    cache_control: str | None = None,
    byte_range: str | None = None,
) -> Response:
    media_type, headers = _media_response_headers(
        upstream,
        filename=filename,
        cache_control=cache_control,
    )
    content = upstream.content
    status_code = 200
    if byte_range is not None:
        match = re.fullmatch(r"bytes=(\d{0,20})-(\d{0,20})", byte_range.strip())
        size = len(content)
        if match is None:
            return _range_not_satisfiable(headers, size)
        start_text, end_text = match.groups()
        if not start_text and not end_text:
            return _range_not_satisfiable(headers, size)
        try:
            if start_text:
                start = int(start_text)
                end = int(end_text) if end_text else size - 1
            else:
                suffix = int(end_text)
                if suffix <= 0:
                    return _range_not_satisfiable(headers, size)
                start = max(0, size - suffix)
                end = size - 1
        except (ValueError, OverflowError):
            return _range_not_satisfiable(headers, size)
        if size == 0 or start >= size or start > end:
            return _range_not_satisfiable(headers, size)
        end = min(end, size - 1)
        content = content[start : end + 1]
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        status_code = 206
    return Response(
        content=content,
        status_code=status_code,
        media_type=media_type,
        headers=headers,
    )


def _media_response_headers(
    upstream: httpx.Response,
    *,
    filename: str,
    cache_control: str | None = None,
) -> tuple[str, dict[str, str]]:
    extension = Path(filename).suffix.lower()
    media_type = (upstream.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
    expected_prefix = _INLINE_MEDIA_PREFIX_BY_EXTENSION.get(extension)
    inline_safe = bool(expected_prefix and media_type.startswith(expected_prefix))
    if not inline_safe:
        media_type = "application/octet-stream"
    cleaned_filename = re.sub(r'[\x00-\x1f\x7f"\\/]', "_", filename)
    ascii_filename = cleaned_filename.encode("ascii", "ignore").decode("ascii") or "media.bin"
    encoded_filename = quote(cleaned_filename, safe="")
    disposition = "inline" if inline_safe else "attachment"
    headers = {
        "X-Content-Type-Options": "nosniff",
        "Accept-Ranges": "bytes",
        "Content-Disposition": (
            f'{disposition}; filename="{ascii_filename}"; '
            f"filename*=UTF-8''{encoded_filename}"
        ),
    }
    if cache_control is not None:
        headers["Cache-Control"] = cache_control
    return media_type, headers


def _range_not_satisfiable(headers: dict[str, str], size: int | None) -> Response:
    total = "*" if size is None else str(size)
    return Response(
        status_code=416,
        headers={**headers, "Content-Range": f"bytes */{total}"},
    )


def _valid_forward_range(value: str | None) -> bool:
    if value is None:
        return True
    match = re.fullmatch(r"bytes=(\d{0,20})-(\d{0,20})", value.strip())
    return bool(match and any(match.groups()))


class _ClosingStreamingResponse(StreamingResponse):
    """Close the upstream HTTP stream even when the downstream disconnects.

    A generator ``finally`` is not sufficient here: ASGI send failures can
    happen after a yielded chunk, outside the generator itself. Owning the
    close at the response boundary prevents abandoned ComfyUI connections
    when a browser cancels a video seek.
    """

    def __init__(
        self,
        *args: Any,
        close_stream: Callable[[], Awaitable[None]],
        **kwargs: Any,
    ) -> None:
        self._close_stream = close_stream
        super().__init__(*args, **kwargs)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._close_stream()


async def _proxy_comfy_media(
    client: ComfyClientProtocol,
    params: dict[str, str],
    *,
    filename: str,
    byte_range: str | None,
    cache_control: str | None = None,
) -> Response:
    """Stream media from ComfyUI and preserve its native Range semantics."""

    if not _valid_forward_range(byte_range):
        placeholder = httpx.Response(200, headers={"content-type": "application/octet-stream"})
        _media_type, headers = _media_response_headers(
            placeholder,
            filename=filename,
            cache_control=cache_control,
        )
        return _range_not_satisfiable(headers, None)

    open_stream = getattr(client, "view_stream", None)
    if not callable(open_stream):
        # Test doubles and older embedders retain the eager compatibility path.
        upstream = await client.view(params)
        return _media_response(
            upstream,
            filename=filename,
            cache_control=cache_control,
            byte_range=byte_range,
        )

    stream = await open_stream(params, byte_range=byte_range)
    upstream = stream.response
    media_type, headers = _media_response_headers(
        upstream,
        filename=filename,
        cache_control=cache_control,
    )
    if upstream.status_code == 416:
        content_range = upstream.headers.get("content-range")
        if content_range:
            headers["Content-Range"] = content_range
        await stream.aclose()
        return Response(status_code=416, headers=headers)
    if upstream.status_code == 206:
        content_range = upstream.headers.get("content-range")
        if not content_range:
            await stream.aclose()
            raise HTTPException(
                status_code=502,
                detail="ComfyUI returned a partial media response without Content-Range",
            )
        headers["Content-Range"] = content_range
    content_length = upstream.headers.get("content-length")
    if content_length and not upstream.headers.get("content-encoding"):
        headers["Content-Length"] = content_length

    async def body():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await stream.aclose()

    return _ClosingStreamingResponse(
        body(),
        close_stream=stream.aclose,
        status_code=upstream.status_code,
        media_type=media_type,
        headers=headers,
    )


def _db(request: Request) -> Database:
    return request.app.state.database


def _storage_status_read(status: StorageStatus) -> StorageStatusRead:
    return StorageStatusRead(
        active_database_path=str(status.active_database_path),
        active_database_identity=status.active_database_identity,
        configured_database_path=str(status.configured_database_path),
        recommended_database_path=str(status.recommended_database_path),
        source=status.source,
        restart_required=status.restart_required,
    )


def _storage_http_error(
    exc: StorageConfigurationError
    | StoragePathError
    | StorageValidationError
    | StorageConflictError
    | StorageOperationError,
) -> HTTPException:
    if isinstance(exc, (StoragePathError, StorageValidationError)):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, StorageConflictError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, StorageConfigurationError):
        return HTTPException(status_code=500, detail="storage configuration is invalid")
    return HTTPException(status_code=500, detail="database storage operation failed")


def _comfy(request: Request, settings: RuntimeSettings | None = None) -> ComfyClientProtocol:
    settings = settings or _db(request).get_settings()
    if not str(settings.comfy_url).strip():
        raise HTTPException(status_code=409, detail=_COMFY_NOT_CONFIGURED)
    return request.app.state.comfy_factory(settings)


def _runtime_authority_changed() -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={
            "code": "runtime_authority_changed",
            "message": "runtime settings changed while resources were being read",
        },
    )


def _runtime_authority_snapshot(
    request: Request,
) -> tuple[RuntimeSettings, str]:
    settings, authority = _db(request).get_settings_authority()
    expected = request.headers.get("X-Director-Runtime-Authority")
    # These endpoints form one App-owned authority batch. A request without a
    # token cannot be safely combined with its siblings and must fail closed.
    if expected is None or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise HTTPException(
            status_code=428,
            detail={
                "code": "runtime_authority_required",
                "message": "a valid runtime settings authority token is required",
            },
        )
    if expected != authority:
        raise _runtime_authority_changed()
    return settings, authority


def _assert_runtime_authority(request: Request, expected: str) -> None:
    _settings, current = _db(request).get_settings_authority()
    if current != expected:
        raise _runtime_authority_changed()


def _validation_error(exc: ValidationError | ValueError) -> RequestValidationError:
    if isinstance(exc, ValidationError):
        return RequestValidationError(exc.errors())
    return RequestValidationError(
        [{"type": "value_error", "loc": ("body",), "msg": str(exc), "input": None}]
    )


def _timeline_revision_conflict(exc: TimelineRevisionConflict) -> HTTPException:
    detail = TimelineRevisionConflictRead(
        message=(
            "timeline changed on the server; fetch the current authority "
            "before retrying"
        ),
        project_id=exc.project_id,
        expected_revision=exc.expected_revision,
        actual_revision=exc.actual_revision,
    )
    return HTTPException(status_code=409, detail=detail.model_dump(mode="json"))


def _timeline_revision_exhausted(exc: TimelineRevisionExhausted) -> HTTPException:
    detail = TimelineRevisionExhaustedRead(
        message=(
            "timeline revision space is exhausted; create or import a new "
            "project before editing further"
        ),
        project_id=exc.project_id,
        revision=exc.revision,
    )
    return HTTPException(status_code=409, detail=detail.model_dump(mode="json"))


def _timeline_comfy_origin_conflict(
    exc: TimelineComfyOriginConflict,
) -> HTTPException:
    detail = TimelineComfyOriginConflictRead(
        message=(
            "ComfyUI endpoint changed before the timeline write acquired its "
            "transaction; fetch current settings and timeline authorities "
            "before retrying"
        ),
        project_id=exc.project_id,
    )
    return HTTPException(status_code=409, detail=detail.model_dump(mode="json"))


def _normalize_shot_detection(value: Any, *, total_frames: int) -> DetectShotsResponse:
    """Validate the Director response and make the two timeline bounds explicit."""

    if not isinstance(value, dict):
        raise ValueError("shot detection response must be an object")
    raw_cuts = value.get("cutFrames")
    raw_count = value.get("shotCount")
    raw_warnings = value.get("warnings", [])
    if not isinstance(raw_cuts, list):
        raise ValueError("shot detection cutFrames must be an array")
    if not isinstance(raw_count, int) or isinstance(raw_count, bool) or raw_count < 0:
        raise ValueError("shot detection shotCount must be a non-negative integer")
    if not isinstance(raw_warnings, list) or any(
        not isinstance(warning, str) for warning in raw_warnings
    ):
        raise ValueError("shot detection warnings must be an array of strings")

    cuts: list[int] = []
    for cut in raw_cuts:
        if not isinstance(cut, int) or isinstance(cut, bool) or not 0 <= cut <= total_frames:
            raise ValueError(
                f"shot detection cutFrames entries must be integers between 0 and {total_frames}"
            )
        cuts.append(cut)
    normalized = sorted({0, total_frames, *cuts})
    shot_count = max(0, len(normalized) - 1)
    if raw_count != shot_count:
        raise ValueError("shot detection shotCount does not match cutFrames")
    return DetectShotsResponse(
        cut_frames=normalized,
        shot_count=shot_count,
        warnings=list(raw_warnings),
    )


def _segment_output_candidates(
    job: dict[str, Any],
) -> dict[str, list[tuple[dict[str, Any], dict[str, str]]]]:
    """Resolve child media only through the immutable SaveVideo node map.

    ComfyUI history ordering and filenames are not identities. A result is
    exposed only when its ``node_id`` maps to a segment declared by that child;
    duplicate candidates remain ambiguous and are deliberately not surfaced.
    """

    candidates: dict[str, list[tuple[dict[str, Any], dict[str, str]]]] = {}
    for child in job.get("children", []):
        declared_segments = {str(item) for item in child.get("segment_ids", [])}
        node_to_segment = {
            str(node_id): str(segment_id)
            for segment_id, node_id in (child.get("output_nodes") or {}).items()
            if str(segment_id) in declared_segments
        }
        for output in child.get("outputs", []):
            segment_id = node_to_segment.get(str(output.get("node_id") or ""))
            if segment_id is None:
                continue
            candidates.setdefault(segment_id, []).append((child, output))
    return candidates


def _segment_results(
    job: dict[str, Any], *, current_snapshot: bool = False
) -> list[dict[str, Any]]:
    candidates = _segment_output_candidates(job)
    snapshot = job.get("config_snapshot") or {}
    timeline = snapshot.get("timeline") if isinstance(snapshot, dict) else None
    segment_ids = snapshot.get("segment_ids") if isinstance(snapshot, dict) else None
    selected = set(segment_ids) if isinstance(segment_ids, list) else None
    ordered_ids = [
        str(segment.get("id"))
        for segment in (timeline or {}).get("segments", [])
        if isinstance(segment, dict)
        and segment.get("enabled", True)
        and (selected is None or segment.get("id") in selected)
    ]
    results: list[dict[str, Any]] = []
    for segment_id in ordered_ids:
        matches = candidates.get(segment_id, [])
        if len(matches) != 1:
            continue
        child, output = matches[0]
        results.append(
            {
                "segment_id": segment_id,
                "child_id": str(child["id"]),
                "output_url": public_api_url(
                    f"/api/jobs/{quote(str(job['id']), safe='')}/segment-output"
                    f"?segment_id={quote(segment_id, safe='')}"
                ),
                "output_file": _output_file_location(output),
                "current_snapshot": current_snapshot,
            }
        )
    return results


def _job_timeline_snapshot(job: dict[str, Any]) -> UnifiedTimelineDraft | None:
    snapshot = job.get("config_snapshot")
    if not isinstance(snapshot, dict):
        return None
    try:
        return UnifiedTimelineDraft.model_validate(snapshot.get("timeline"))
    except (TypeError, ValidationError, ValueError):
        return None


def _snapshot_filename(value: str) -> str:
    """Keep a useful model basename without exposing a historical path."""

    return re.split(r"[\\/]", value)[-1]


def _job_generation_details(
    job: dict[str, Any], children: list[dict[str, Any]]
) -> JobGenerationDetailsRead | None:
    timeline = _job_timeline_snapshot(job)
    if timeline is None:
        return None

    snapshot = job.get("config_snapshot")
    raw_segment_ids = (
        snapshot.get("segment_ids") if isinstance(snapshot, dict) else None
    )
    selected_ids = (
        {segment_id for segment_id in raw_segment_ids if isinstance(segment_id, str)}
        if isinstance(raw_segment_ids, list)
        else None
    )
    segments = [
        segment
        for segment in timeline.segments
        if segment.enabled and (selected_ids is None or segment.id in selected_ids)
    ]
    if not segments:
        return None

    families = [
        family
        for family in ("fl2va", "ref2va")
        if any(segment.mode == family for segment in segments)
    ]
    sampling = []
    for family in families:
        config = getattr(timeline.sampling, family)
        sampling.append(
            {
                "family": family,
                "steps": config.steps,
                "seed": config.seed,
                "random_seed": config.random_seed,
                "sampler": config.sampler,
                "scheduler": config.scheduler,
                "shift": config.shift,
                "audio_shift": config.audio_shift,
            }
        )

    models: list[dict[str, Any]] = []
    shared_models: list[dict[str, Any]] = []
    runtime_snapshot_available = False
    try:
        settings = RuntimeSettings.model_validate(job.get("settings_snapshot"))
    except (TypeError, ValidationError, ValueError):
        settings = None
    if settings is not None:
        runtime_snapshot_available = True
        for family in families:
            binding = getattr(settings.models, family)
            child_backends = {
                child.get("backend")
                for child in children
                if child.get("family") == family
                and child.get("backend") in {"standard", "raylight"}
            }
            backends = [
                backend
                for backend in ("standard", "raylight")
                if backend in child_backends
            ]
            if not backends:
                backends = [resolve_execution_backend(binding)]
            uses_raylight = "raylight" in backends
            models.append(
                {
                    "family": family,
                    "filename": _snapshot_filename(binding.filename),
                    "device": binding.device,
                    "lora_name": (
                        _snapshot_filename(binding.lora_name)
                        if binding.lora_name is not None
                        else None
                    ),
                    "lora_strength": binding.lora_strength,
                    "backends": backends,
                    "logical_gpu_indices": (
                        list(binding.raylight.gpu_select) if uses_raylight else []
                    ),
                    "ulysses_degree": (
                        binding.raylight.ulysses_degree if uses_raylight else None
                    ),
                    "ring_degree": (
                        binding.raylight.ring_degree if uses_raylight else None
                    ),
                }
            )
        for role in ("clip", "video_vae", "audio_vae"):
            binding = getattr(settings.models, role)
            shared_models.append(
                {
                    "role": role,
                    "filename": _snapshot_filename(binding.filename),
                    "device": binding.device,
                }
            )

    segment_details = []
    for segment in segments:
        is_fl2va = isinstance(segment, UnifiedFL2VASegment)
        segment_details.append(
            {
                "id": segment.id,
                "title": segment.title,
                "family": segment.mode,
                "recipe": timeline_segment_recipe(segment),
                "duration_seconds": segment.duration_seconds,
                "prompt": segment.prompt,
                "continuity_enabled": segment.continuity.enabled,
                "continuity_overlap_frames": segment.continuity.overlap_frames,
                "ref_image_size": segment.ref_image_size,
                "audio_mode": segment.audio_mode,
                "has_first_image": bool(is_fl2va and segment.first_image is not None),
                "has_last_image": bool(is_fl2va and segment.last_image is not None),
                "has_source_video": bool(
                    not is_fl2va and segment.source_video is not None
                ),
                "source_audio_as_reference": bool(
                    not is_fl2va and segment.source_audio_as_reference
                ),
                "reference_image_count": (
                    0 if is_fl2va else len(segment.reference_images)
                ),
                "reference_audio_count": (
                    0 if is_fl2va else len(segment.reference_audios)
                ),
                "reference_video_count": (
                    0 if is_fl2va else len(segment.reference_videos)
                ),
            }
        )

    return JobGenerationDetailsRead(
        job_id=str(job["id"]),
        project_title=timeline.title,
        render={
            "width": timeline.render.width,
            "height": timeline.render.height,
            "fps": timeline.render.fps,
            "export_mode": timeline.export_mode,
            "total_duration_seconds": sum(
                segment.duration_seconds for segment in segments
            ),
        },
        sampling=sampling,
        models=models,
        shared_models=shared_models,
        runtime_snapshot_available=runtime_snapshot_available,
        segments=segment_details,
    )


def _job_identity(job: dict[str, Any]) -> tuple[str, str | None]:
    """Derive a human task title without returning the raw config snapshot."""

    snapshot = job.get("config_snapshot")
    timeline = snapshot.get("timeline") if isinstance(snapshot, dict) else None
    raw_title = timeline.get("title") if isinstance(timeline, dict) else None
    project_title = (
        raw_title.strip()
        if isinstance(raw_title, str) and raw_title.strip()
        else None
    )
    if project_title is not None:
        return project_title, project_title
    mode = str(job.get("mode") or "").strip()
    if mode == "timeline":
        return "长视频生成任务", None
    if mode:
        return f"{mode.upper()} 生成任务", None
    return "生成任务", None


def _parse_job_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _job_execution_duration(job: dict[str, Any]) -> float | None:
    started_at = _parse_job_timestamp(job.get("started_at"))
    if started_at is None:
        return None
    completed_at = _parse_job_timestamp(job.get("completed_at"))
    end = completed_at or datetime.now(timezone.utc)
    return max(0.0, (end - started_at).total_seconds())


_DIAGNOSTIC_SECRET = re.compile(
    r"(?i)[\"']?\b((?:access|refresh)[_-]?token|client[_-]?secret|api[_-]?key|"
    r"token|secret|password)\b[\"']?\s*[:=]\s*"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_DIAGNOSTIC_AUTHORIZATION = re.compile(
    r"(?i)[\"']?\bauthorization\b[\"']?\s*[:=]\s*"
    r"(?:\"[^\"]*\"|'[^']*'|(?:(?:bearer|basic)\s+)?[^\s,;]+)"
)
_URL_USERINFO = re.compile(r"(?i)(https?://)[^/@\s]+@")


def _error_summary(value: Any, *, limit: int = 320) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = " ".join(value.split())
    normalized = _URL_USERINFO.sub(r"\1[已隐藏]@", normalized)
    normalized = _DIAGNOSTIC_AUTHORIZATION.sub("authorization=[已隐藏]", normalized)
    normalized = _DIAGNOSTIC_SECRET.sub(lambda match: f"{match.group(1)}=[已隐藏]", normalized)
    if len(normalized) > limit:
        return normalized[: limit - 1].rstrip() + "…"
    return normalized


def _job_error_summary(job: dict[str, Any]) -> str | None:
    own_error = _error_summary(job.get("error"))
    if own_error is not None:
        return own_error
    for child in job.get("children", []):
        child_error = _error_summary(child.get("error"))
        if child_error is not None:
            return child_error
    return None


def _job_read(
    job: dict[str, Any], *, live_preview_available: bool = False,
    current_snapshot: bool = False,
    current_project: bool = False,
) -> JobRead:
    outputs = [
        public_api_url(f"/api/jobs/{job['id']}/outputs/{index}")
        for index, _ in enumerate(job["outputs"])
    ]
    output_files = [_output_file_location(output) for output in job["outputs"]]
    segment_results = _segment_results(job, current_snapshot=current_snapshot)
    visible_output_files = set(output_files)
    visible_output_files.update(result["output_file"] for result in segment_results)
    display_name, project_title = _job_identity(job)
    return JobRead.model_validate(
        {
            "id": job["id"],
            "mode": job["mode"],
            "status": job["status"],
            "display_name": display_name,
            "project_title": project_title,
            "project_id": job.get("project_id"),
            "current_project": current_project,
            "progress": job["progress"],
            "stage": job["stage"],
            "prompt_id": job["prompt_id"],
            "outputs": outputs,
            "output_files": output_files,
            "error": job["error"],
            "preview_url": outputs[0] if outputs else None,
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "started_at": job["started_at"],
            "completed_at": job["completed_at"],
            "execution_duration_seconds": _job_execution_duration(job),
            "output_count": len(visible_output_files),
            "error_summary": _job_error_summary(job),
            "segment_results": segment_results,
            "live_preview_url": (
                public_api_url(f"/api/jobs/{quote(str(job['id']), safe='')}/live-preview")
                if live_preview_available and job["status"] not in _TERMINAL_STATUSES
                else None
            ),
            "children": [
                {
                    "id": child["id"],
                    "family": child["family"],
                    "backend": child["backend"],
                    "segment_ids": child["segment_ids"],
                    "status": child["status"],
                    "progress": child["progress"],
                    "stage": child["stage"],
                    "prompt_id": child["prompt_id"],
                    "outputs": [
                        _output_file_location(output)
                        for output in child["outputs"]
                    ],
                    "error": child["error"],
                }
                for child in job.get("children", [])
                # Internal RayKill barriers are durable control prompts, not
                # user timeline segments. Keep them available to cancellation
                # and restart recovery without exposing a phantom empty child
                # in the task drawer.
                if child.get("segment_ids")
            ],
        }
    )


def _live_preview_for_job(request: Request, job: dict[str, Any]):
    """Return a cache hit only while its exact native child sampler is live."""

    cache: LivePreviewCache = request.app.state.live_preview_cache
    if job["status"] in _TERMINAL_STATUSES:
        cache.evict(str(job["id"]))
        return None
    preview = cache.get(str(job["id"]))
    if preview is None:
        return None
    child = _db(request).get_job_child(preview.child_id)
    if (
        child is None
        or child["job_id"] != str(job["id"])
        or child["status"] not in {"queued", "running"}
        or child.get("prompt_id") != preview.prompt_id
        or sampler_segment_for_node(child, preview.node_id) != preview.segment_id
    ):
        # Do not tombstone: another child of this nonterminal parent may still
        # produce the next legitimate frame.
        cache.evict(str(job["id"]))
        return None
    return preview


def _job_read_for_request(
    request: Request,
    job: dict[str, Any],
    *,
    current_timeline: UnifiedTimelineDraft | None = None,
    current_settings: RuntimeSettings | None = None,
) -> JobRead:
    """Annotate one job with strict currentness flags.

    ``current_project`` deliberately compares only the typed timeline (scoped
    by the caller to the active project's timeline); ``current_snapshot`` adds
    the stricter runtime-settings equality for the main monitor. Loose project
    membership is expressed by the separate ``project_id`` field.
    """

    current_snapshot = False
    current_project = False
    snapshot_timeline = _job_timeline_snapshot(job)
    if snapshot_timeline is not None:
        if current_timeline is None:
            current_timeline = _db(request).get_timeline()
        current_project = (
            snapshot_timeline.model_dump(mode="json")
            == current_timeline.model_dump(mode="json")
        )
    if job.get("mode") == "timeline" and snapshot_timeline is not None:
        try:
            snapshot_settings = RuntimeSettings.model_validate(
                job.get("settings_snapshot")
            )
            if current_settings is None:
                current_settings = _db(request).get_settings()
        except (TypeError, ValidationError, ValueError):
            # Historical or partially migrated jobs remain inspectable, but
            # can never be described as an exact current snapshot.
            pass
        else:
            current_snapshot = (
                current_project
                and snapshot_settings.model_dump(mode="json")
                == current_settings.model_dump(mode="json")
            )
    return _job_read(
        job,
        live_preview_available=_live_preview_for_job(request, job) is not None,
        current_snapshot=current_snapshot,
        current_project=current_project,
    )


def _output_file_location(output: dict[str, Any]) -> str:
    """Return a safe ComfyUI-root-relative label for an output reference."""

    output_type = str(output.get("type") or "output").strip().lower()
    if output_type not in {"input", "output", "temp"}:
        output_type = "output"
    subfolders: list[str] = []
    for component in str(output.get("subfolder") or "").replace("\\", "/").split("/"):
        if component in {"", ".", ".."}:
            continue
        safe_component = _SAFE_FILENAME.sub("_", component).strip("._")
        if safe_component:
            subfolders.append(safe_component)
    return "/".join(
        [output_type, *subfolders, _safe_filename(str(output.get("filename") or "media.bin"))]
    )


def _contains_prompt(items: Any, prompt_id: str) -> bool:
    if isinstance(items, dict):
        return any(_contains_prompt(value, prompt_id) for value in items.values())
    if isinstance(items, (list, tuple)):
        return any(value == prompt_id or _contains_prompt(value, prompt_id) for value in items)
    return False


def _collect_output_files(value: Any, *, node_id: str = "") -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, dict):
        if isinstance(value.get("filename"), str):
            found.append(
                {
                    "node_id": node_id,
                    "filename": value["filename"],
                    "subfolder": str(value.get("subfolder") or ""),
                    "type": str(value.get("type") or "output"),
                }
            )
        else:
            for key, child in value.items():
                found.extend(_collect_output_files(child, node_id=node_id or str(key)))
    elif isinstance(value, list):
        for child in value:
            found.extend(_collect_output_files(child, node_id=node_id))
    deduplicated: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in found:
        identity = (item["filename"], item["subfolder"], item["type"])
        if identity not in seen:
            seen.add(identity)
            deduplicated.append(item)
    return deduplicated


def _history_error(entry: dict[str, Any]) -> str | None:
    status = entry.get("status")
    if not isinstance(status, dict):
        return None
    messages = status.get("messages")
    if not isinstance(messages, list):
        return None
    explicit_errors: list[Any] = []
    for message in messages:
        if not isinstance(message, (list, tuple)) or not message:
            continue
        event_name = str(message[0]).lower()
        if event_name in {"execution_error", "error", "execution_interrupted"}:
            explicit_errors.append(message)
    return json.dumps(explicit_errors, ensure_ascii=False)[:20_000] if explicit_errors else None


def _history_has_event(entry: dict[str, Any], event: str) -> bool:
    status = entry.get("status")
    messages = status.get("messages") if isinstance(status, dict) else None
    if not isinstance(messages, list):
        return False
    expected = event.lower()
    return any(
        isinstance(message, (list, tuple))
        and bool(message)
        and str(message[0]).lower() == expected
        for message in messages
    )


def _exact_history_result(
    value: Any, prompt_id: str
) -> tuple[dict[str, Any] | None, bool]:
    """Parse one exact history lookup without mistaking bad data for absence."""

    if not isinstance(value, dict):
        return None, False
    if not value:
        return None, True
    if set(value) != {prompt_id}:
        return None, False
    entry = value.get(prompt_id)
    return (entry, False) if isinstance(entry, dict) else (None, False)


def _raylight_recovery_history_state(
    entry: dict[str, Any],
) -> Literal["terminal", "nonterminal", "invalid"]:
    """Classify an exact history entry without trusting contradictory flags."""

    status = entry.get("status")
    if not isinstance(status, dict):
        return "invalid"
    raw_status_value = status.get("status_str")
    completed = status.get("completed")
    messages = status.get("messages")
    if (
        not isinstance(raw_status_value, str)
        or not isinstance(completed, bool)
        or not isinstance(messages, list)
    ):
        return "invalid"
    status_value = str(status.get("status_str") or "").strip().lower()
    has_failure_evidence = bool(_history_error(entry))
    if status_value == "success":
        return "terminal" if completed and not has_failure_evidence else "invalid"
    if status_value in {"error", "failed"}:
        # Official ComfyUI reports execution errors with completed=false. Some
        # compatible servers historically emitted true, so the explicit error
        # status remains authoritative, but it must carry a concrete error or
        # interruption event rather than a bare label.
        return "terminal" if has_failure_evidence else "invalid"
    if status_value in {"pending", "queued", "running"}:
        return "invalid" if completed or has_failure_evidence else "nonterminal"
    return "invalid"


def _raylight_recovery_history_is_terminal(entry: dict[str, Any]) -> bool:
    return _raylight_recovery_history_state(entry) == "terminal"


def _raylight_child_has_terminal_history_certificate(
    child: dict[str, Any],
) -> bool:
    """Recognize only server-owned stages written from exact history."""

    status = child.get("status")
    stage = child.get("stage")
    return (
        status == "succeeded"
        and stage in {"completed", "RayLight 已安全释放"}
    ) or (
        status == "failed"
        and stage in {"failed", "RayLight 安全切换失败"}
    ) or (
        status == "cancelled" and stage == "ComfyUI 端已中断"
    )


def _raylight_runtime_has_terminal_certificate(
    state: dict[str, Any],
) -> bool:
    certificate = state.get("tail_terminal_certificate")
    return (
        isinstance(certificate, dict)
        and certificate.get("prompt_id") == state.get("tail_prompt_id")
        and certificate.get("action") == state.get("tail_action")
        and isinstance(certificate.get("succeeded"), bool)
    )


def _cas_job_update(database: Database, snapshot: dict[str, Any], **updates: Any) -> dict[str, Any]:
    updated = database.update_job_if_status(
        snapshot["id"], snapshot["status"], **updates
    )
    if updated is not None:
        return updated
    latest = database.get_job(snapshot["id"])
    if latest is None:
        raise HTTPException(status_code=404, detail="job disappeared during status sync")
    return latest


async def _cas_active_child_update(
    database: Database, snapshot: dict[str, Any], **updates: Any
) -> tuple[dict[str, Any], bool]:
    """Write from one child row version without ever reviving a terminal row.

    Keeping this lifecycle boundary awaitable also lets concurrency tests stop
    precisely after a row was listed but before its CAS, matching the network
    races between user cancellation and restart recovery.
    """

    if snapshot["status"] in _TERMINAL_STATUSES:
        return snapshot, False
    updated = database.update_job_child_if_snapshot(
        snapshot["id"],
        expected_status=snapshot["status"],
        expected_updated_at=snapshot["updated_at"],
        **updates,
    )
    if updated is not None:
        return updated, True
    latest = database.get_job_child(snapshot["id"])
    if latest is None:
        raise KeyError(snapshot["id"])
    return latest, False


async def _sync_job(
    request: Request,
    job: dict[str, Any],
    *,
    allow_timeline_assembly: bool = False,
) -> dict[str, Any]:
    if _db(request).list_job_children(job["id"]):
        return await _sync_timeline_job(
            request, job, allow_assembly=allow_timeline_assembly
        )
    terminal = job["status"] in _TERMINAL_STATUSES
    if terminal or not job.get("prompt_id"):
        return job
    client = _comfy(request, RuntimeSettings.model_validate(job["settings_snapshot"]))

    def apply_history(entry: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(entry, dict):
            return None
        raw_status = entry.get("status")
        status_block = raw_status if isinstance(raw_status, dict) else {}
        status_value = str(status_block.get("status_str") or "").lower()
        outputs = _collect_output_files(entry.get("outputs") or {})
        history_error = _history_error(entry)
        interrupted = _history_has_event(entry, "execution_interrupted")
        if interrupted:
            return _cas_job_update(
                _db(request),
                job,
                status="cancelled",
                progress=1.0,
                stage=(
                    "cancelled"
                    if job["status"] in {"cancelling", "cancelled"}
                    else "ComfyUI 端已中断"
                ),
                outputs=[],
                error=None,
                completed_at=utc_now(),
            )
        failed = status_value in {"error", "failed"} or bool(history_error)
        succeeded = status_value == "success" or (
            bool(status_block.get("completed")) and not failed
        )
        if not failed and not succeeded:
            return job
        return _cas_job_update(
            _db(request),
            job,
            status="failed" if failed else "succeeded",
            progress=1.0,
            stage="failed" if failed else "completed",
            outputs=outputs,
            error=history_error if failed else None,
            completed_at=utc_now(),
        )

    try:
        history = await client.history(job["prompt_id"])
        entry, _ = _exact_history_result(history, str(job["prompt_id"]))
        history_update = apply_history(entry)
        if history_update is not None:
            return history_update
        queue = await client.queue()
        queue_available = (
            isinstance(queue, dict)
            and isinstance(queue.get("queue_running"), list)
            and isinstance(queue.get("queue_pending"), list)
        )
        if not queue_available:
            return job
        running = _contains_prompt(queue["queue_running"], job["prompt_id"])
        pending = _contains_prompt(queue["queue_pending"], job["prompt_id"])
        # Once cancellation has started, queue snapshots are informational only:
        # a stale pending/running response must never revive the local job.
        if job["status"] in {"cancelling", "cancelled"}:
            if running or pending:
                return job
        if running:
            # Historical rows use standard queue/history only. Native
            # parent/child jobs are handled above and never call the removed
            # Director-specific progress endpoint.
            progress = max(float(job["progress"]), 0.01)
            stage = job.get("stage") if job["status"] == "running" else "sampling"
            updated = _db(request).update_job_progress_monotonic(
                job["id"],
                job["status"],
                progress,
                stage=stage,
                started_at=job.get("started_at") or utc_now(),
                expected_updated_at=job["updated_at"],
            )
            if updated is not None:
                return updated
            latest = _db(request).get_job(job["id"])
            if latest is None:
                raise HTTPException(status_code=404, detail="job disappeared during status sync")
            return latest
        if (
            job["status"] in {"preparing", "queued"}
            and pending
        ):
            return _cas_job_update(
                _db(request), job, status="queued", progress=0.0, stage="queued"
            )
        if pending:
            return job

        # The first history read precedes the queue snapshot, so it alone
        # cannot close the execution-to-history handoff race. Repeat the exact
        # lookup after observing the prompt off both queues. Only two valid,
        # successful upstream snapshots may classify a submitted legacy prompt
        # as externally removed.
        confirmation = await client.history(job["prompt_id"])
        confirmed_entry, confirmed_absent = _exact_history_result(
            confirmation, str(job["prompt_id"])
        )
        history_update = apply_history(confirmed_entry)
        if history_update is not None:
            return history_update
        if confirmed_absent and job["status"] in {"preparing", "queued", "cancelling"}:
            return _cas_job_update(
                _db(request),
                job,
                status="cancelled",
                progress=1.0,
                stage=(
                    "cancelled"
                    if job["status"] in {"cancelling", "cancelled"}
                    else "ComfyUI 端任务已移除"
                ),
                outputs=[],
                error=None,
                completed_at=utc_now(),
            )
    except (ComfyError, httpx.HTTPError):
        # A status read must not turn a durable job into a failure merely because
        # ComfyUI is temporarily unreachable.
        return job
    return job


async def _sync_timeline_child(
    request: Request,
    child: dict[str, Any],
    *,
    parent_cancelling: bool,
    history_entry: dict[str, Any] | None,
    running: bool,
    pending: bool,
    confirmed_absent: bool,
    respect_process_ownership: bool = True,
) -> dict[str, Any]:
    """Apply one parent-level queue/history snapshot to a child row."""

    if child["status"] in _TERMINAL_STATUSES or not child.get("prompt_id"):
        return child
    # While POST /prompt is in flight, the submission coroutine is the only
    # owner allowed to bind this caller-assigned prompt id. Queue/history may
    # already expose the side effect before the HTTP response returns; letting
    # a GET or the background reconciler advance this row would make the
    # submit owner's compare-and-set fail and misreport a successful enqueue.
    # Lifespan recovery clears this marker after a hard process restart.
    if (
        respect_process_ownership
        and child.get("stage") in _PROCESS_OWNERSHIP_STAGES
    ):
        return child
    database = _db(request)

    def commit(**updates: Any) -> dict[str, Any]:
        updated = database.update_job_child_if_snapshot(
            child["id"],
            expected_status=child["status"],
            expected_updated_at=child["updated_at"],
            **updates,
        )
        if updated is not None:
            return updated
        latest = database.get_job_child(child["id"])
        if latest is None:
            raise KeyError(child["id"])
        return latest

    if isinstance(history_entry, dict):
        status_block = (
            history_entry.get("status")
            if isinstance(history_entry.get("status"), dict)
            else {}
        )
        status_value = str(status_block.get("status_str") or "").lower()
        error = _history_error(history_entry)
        interrupted = _history_has_event(history_entry, "execution_interrupted")
        if child.get("backend") == "raylight":
            history_state = _raylight_recovery_history_state(history_entry)
            if history_state == "invalid":
                raise ComfyError(
                    "RayLight history status is contradictory; refusing to certify the runtime tail"
                )
            failed = history_state == "terminal" and status_value in {
                "error",
                "failed",
            }
            succeeded = history_state == "terminal" and status_value == "success"
        else:
            failed = status_value in {"error", "failed"} or bool(error)
            succeeded = status_value == "success" or (
                bool(status_block.get("completed")) and not failed
            )
        if interrupted:
            return commit(
                status="cancelled",
                progress=1.0,
                stage=("cancelled" if parent_cancelling else "ComfyUI 端已中断"),
                outputs=[],
                error=None,
                completed_at=utc_now(),
            )
        if failed or succeeded:
            return commit(
                status="failed" if failed else "succeeded",
                progress=1.0,
                stage="failed" if failed else "completed",
                outputs=(
                    []
                    if failed
                    else _collect_output_files(history_entry.get("outputs") or {})
                ),
                error=error if failed else None,
                completed_at=utc_now(),
            )
    if running:
        # A standard websocket event may already have persisted exact
        # segment/step detail. Queue polling owns lifecycle only and must
        # not replace that richer snapshot with a generic label.
        detailed_stage = (
            child.get("stage")
            if child["status"] == "running" and child.get("stage")
            else "sampling"
        )
        return commit(
            status="cancelling" if parent_cancelling else "running",
            progress=max(float(child["progress"]), 0.01),
            stage=(
                child.get("stage") or "cancelling"
                if parent_cancelling
                else detailed_stage
            ),
            started_at=child.get("started_at") or utc_now(),
        )
    if pending:
        return commit(
            status="cancelling" if parent_cancelling else "queued",
            stage=(
                child.get("stage") or "cancelling"
                if parent_cancelling
                else "queued"
            ),
        )
    if confirmed_absent and child.get("stage") not in _PROCESS_OWNERSHIP_STAGES:
        was_running = child["status"] == "running" or child.get("started_at") is not None
        return commit(
            status=("failed" if was_running and not parent_cancelling else "cancelled"),
            progress=1.0,
            stage=(
                "cancelled"
                if parent_cancelling
                else "ComfyUI 端任务状态丢失"
                if was_running
                else "ComfyUI 端任务已移除"
            ),
            outputs=[],
            error=(
                "ComfyUI 中的运行任务已不在队列或历史记录中，无法确认生成结果"
                if was_running and not parent_cancelling
                else None
            ),
            completed_at=utc_now(),
        )
    return child


async def _sync_timeline_children_batch(
    request: Request,
    job: dict[str, Any],
    children: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reconcile all segment children with one queue and bounded history reads."""

    active = [
        child
        for child in children
        if child["status"] not in _TERMINAL_STATUSES and child.get("prompt_id")
        and child.get("stage") not in _RECOVERY_OWNERSHIP_STAGES
    ]
    if not active:
        return children
    database = _db(request)
    client = _comfy(
        request, RuntimeSettings.model_validate(job["settings_snapshot"])
    )
    queue: dict[str, Any] = {}
    queue_available = False
    try:
        queue = await client.queue()
        # Only a structurally complete queue response can establish absence.
        # Treating an HTTP 200 with a malformed/partial body as two empty
        # queues would turn an upstream contract failure into a false cancel.
        queue_available = (
            isinstance(queue, dict)
            and isinstance(queue.get("queue_running"), list)
            and isinstance(queue.get("queue_pending"), list)
        )
        if not queue_available:
            queue = {}
    except (ComfyError, httpx.HTTPError):
        # History can still advance terminal rows while absence remains
        # untrusted until a later successful queue snapshot.
        pass
    running_entries = queue.get("queue_running", [])
    pending_entries = queue.get("queue_pending", [])
    placement: dict[str, tuple[bool, bool]] = {
        str(child["prompt_id"]): (
            _contains_prompt(running_entries, str(child["prompt_id"])),
            _contains_prompt(pending_entries, str(child["prompt_id"])),
        )
        for child in active
    }
    off_queue = [
        child
        for child in active
        if not any(placement[str(child["prompt_id"])])
    ]
    history_entries: dict[str, dict[str, Any]] = {}
    exact_absent_prompt_ids: set[str] = set()
    rotated_rows: dict[str, dict[str, Any]] = {}
    if off_queue:
        try:
            # The public ceiling is 128 segments. A 2x window tolerates
            # unrelated endpoint traffic without pulling unbounded history.
            bulk = await client.history(
                max_items=min(256, max(128, len(active) * 2))
            )
        except (ComfyError, httpx.HTTPError):
            bulk = {}
        if isinstance(bulk, dict):
            history_entries.update(
                {
                    str(prompt_id): entry
                    for prompt_id, entry in bulk.items()
                    if isinstance(entry, dict)
                }
            )
        # Exact fallback closes the queue/history handoff race and recovers a
        # job whose entries fell outside the bounded bulk window.
        if queue_available:
            # Bound exact fallback work per HTTP poll. Missing old history
            # entries rotate by child row age over later polls without creating
            # a 128-request burst against a degraded endpoint.
            missing_candidates = sorted(
                (
                    child
                    for child in off_queue
                    if str(child["prompt_id"]) not in history_entries
                ),
                key=lambda child: (
                    str(child.get("updated_at") or ""),
                    int(child.get("group_index") or 0),
                ),
            )
            for child in missing_candidates[:16]:
                prompt_id = str(child["prompt_id"])
                try:
                    exact = await client.history(prompt_id)
                except (ComfyError, httpx.HTTPError):
                    exact = {}
                    exact_read_succeeded = False
                else:
                    exact_read_succeeded = True
                entry, exact_absent = _exact_history_result(exact, prompt_id)
                if isinstance(entry, dict):
                    history_entries[prompt_id] = entry
                    continue
                if exact_read_succeeded and exact_absent:
                    exact_absent_prompt_ids.add(prompt_id)
                # Rotate an unresolved prompt behind older siblings. The CAS
                # cannot overwrite a simultaneous WebSocket/history update.
                rotated = database.update_job_child_if_snapshot(
                    child["id"],
                    expected_status=child["status"],
                    expected_updated_at=child["updated_at"],
                    updated_at=utc_now(),
                )
                if rotated is not None:
                    rotated_rows[child["id"]] = rotated

    for original_child in active:
        child = rotated_rows.get(original_child["id"], original_child)
        prompt_id = str(child["prompt_id"])
        running, pending = placement[prompt_id]
        await _sync_timeline_child(
            request,
            child,
            parent_cancelling=job["status"] == "cancelling",
            history_entry=history_entries.get(prompt_id),
            running=running,
            pending=pending,
            # A bounded bulk response cannot prove that an older prompt never
            # completed. Only an exact successful lookup may close a
            # cancelling child as absent without losing a completed take.
            confirmed_absent=(
                queue_available
                and not running
                and not pending
                and prompt_id in exact_absent_prompt_ids
            ),
        )
    # WebSocket progress and HTTP reconciliation can commit concurrently.
    # Aggregate the parent from a fresh authoritative snapshot rather than a
    # stale pre-await child row that could momentarily rewind visible progress.
    return database.list_job_children(job["id"])


def _ordered_timeline_outputs(job: dict[str, Any], children: list[dict[str, Any]]) -> list[dict[str, str]]:
    snapshot = job.get("config_snapshot") or {}
    timeline = snapshot.get("timeline") if isinstance(snapshot, dict) else None
    segment_ids = snapshot.get("segment_ids") if isinstance(snapshot, dict) else None
    selected = set(segment_ids) if isinstance(segment_ids, list) else None
    ordered_ids = [
        str(segment.get("id"))
        for segment in (timeline or {}).get("segments", [])
        if isinstance(segment, dict)
        and segment.get("enabled", True)
        and (selected is None or segment.get("id") in selected)
    ]
    output_by_segment: dict[str, dict[str, str]] = {}
    unknown_nodes: list[str] = []
    for child in children:
        node_to_segment = {
            str(node_id): str(segment_id)
            for segment_id, node_id in child["output_nodes"].items()
        }
        for output in child["outputs"]:
            segment_id = node_to_segment.get(str(output.get("node_id") or ""))
            if segment_id is None:
                unknown_nodes.append(str(output.get("node_id") or ""))
                continue
            if segment_id in output_by_segment:
                raise ValueError(
                    f"native workflow returned duplicate output for segment '{segment_id}'"
                )
            output_by_segment[segment_id] = output
    expected = set(ordered_ids)
    actual = set(output_by_segment)
    if unknown_nodes or actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "native workflow output set does not match the requested timeline: "
            f"missing={missing}, unexpected={unexpected}, unknown_nodes={sorted(unknown_nodes)}"
        )
    return [output_by_segment[item] for item in ordered_ids]


def _is_full_timeline_selection(snapshot: dict[str, Any]) -> bool:
    """Return whether the requested set covers every enabled segment.

    ``segment_ids=None`` remains a backwards-compatible shorthand.  New
    clients always send the checkbox selection explicitly, so full-run and
    assembly semantics must come from stable segment IDs rather than from a
    separate request shape.
    """

    segment_ids = snapshot.get("segment_ids")
    if segment_ids is None:
        return True
    timeline = snapshot.get("timeline")
    if not isinstance(segment_ids, list) or not isinstance(timeline, dict):
        return False
    segments = timeline.get("segments")
    if not isinstance(segments, list):
        return False
    enabled_ids = {
        str(segment["id"])
        for segment in segments
        if isinstance(segment, dict)
        and isinstance(segment.get("id"), str)
        and segment.get("enabled", True) is True
    }
    return bool(enabled_ids) and {str(item) for item in segment_ids} == enabled_ids


async def _assemble_timeline_output(
    request: Request,
    job: dict[str, Any],
    segment_outputs: list[dict[str, str]],
) -> dict[str, str]:
    """Download exact child takes, assemble them, then register one Comfy output."""

    settings = RuntimeSettings.model_validate(job["settings_snapshot"])
    client = _comfy(request, settings)
    segment_bytes: list[bytes] = []
    for output in segment_outputs:
        response = await client.view(
            {
                "filename": output["filename"],
                "subfolder": output.get("subfolder", ""),
                "type": output.get("type", "output"),
            }
        )
        if not response.content:
            raise MediaToolError(
                f"generated segment '{output['filename']}' is empty"
            )
        segment_bytes.append(response.content)
    timeline = job["config_snapshot"]["timeline"]
    proxy = await anyio.to_thread.run_sync(
        partial(
            assemble_video_bytes,
            segment_bytes,
            fps=float(timeline["render"]["fps"]),
            width=int(timeline["render"]["width"]),
            height=int(timeline["render"]["height"]),
        )
    )
    filename = f"Director_timeline_{job['id'][:8]}_full.mp4"
    uploaded = await client.upload_output(
        filename,
        proxy.content,
        "video/mp4",
        "director-web/timelines",
    )
    return {
        "node_id": "assembly",
        "filename": str(uploaded["name"]),
        "subfolder": str(uploaded.get("subfolder") or ""),
        "type": str(uploaded.get("type") or "output"),
    }


async def _sync_timeline_job_once(
    request: Request,
    job: dict[str, Any],
    *,
    allow_assembly: bool = True,
    honor_cancel_intent: bool = True,
) -> dict[str, Any]:
    database = _db(request)
    children = database.list_job_children(job["id"])
    if not children:
        job["children"] = []
        return job
    if job["status"] not in _TERMINAL_STATUSES:
        if job.get("stage") == "assembling" and allow_assembly:
            job["children"] = children
            return job
        children = await _sync_timeline_children_batch(request, job, children)
        # Queue/history I/O above can overlap an explicit cancel request. Use
        # the latest durable parent intent for control/sibling classification;
        # the snapshot supplied by the caller may predate that request.
        latest_job = database.get_job(job["id"])
        if latest_job is None:
            raise KeyError(job["id"])
        if latest_job["status"] in _TERMINAL_STATUSES:
            latest_job["children"] = database.list_job_children(job["id"])
            return latest_job
        job = latest_job
        # A failed control barrier stops the strict dispatcher before any
        # following unbound generation can cross the network. Close those
        # provably unsubmitted siblings now so the orchestration failure can
        # become a coherent terminal parent even when a cancel click races the
        # first history reconciliation.
        terminal_controls = [
            child
            for child in children
            if not child["segment_ids"]
            and child["status"] in {"failed", "cancelled"}
        ]
        if terminal_controls:
            explicit_control_cancel = (
                bool(job.get("cancel_requested"))
                and not any(child["status"] == "failed" for child in terminal_controls)
                and all(
                    child.get("stage") in _DIRECTOR_CANCEL_STAGES
                    for child in terminal_controls
                )
            )
            for child in children:
                if (
                    child["status"] not in _TERMINAL_STATUSES
                    and not child.get("prompt_id")
                ):
                    database.update_job_child_if_snapshot(
                        child["id"],
                        expected_status=child["status"],
                        expected_updated_at=child["updated_at"],
                        status="cancelled" if explicit_control_cancel else "failed",
                        progress=1.0,
                        stage=(
                            "cancelled"
                            if explicit_control_cancel
                            else "raylight_switch_failed"
                        ),
                        error=(
                            None
                            if explicit_control_cancel
                            else "RayLight runtime switch did not complete"
                        ),
                        completed_at=utc_now(),
                    )
            children = database.list_job_children(job["id"])
        # Ray runtime reuse is certified only by the exact queue-tail child.
        # Older terminal rows may be reconciled after a newer submission has
        # already advanced the endpoint ledger, so Database performs the
        # prompt-id match and update atomically. Control/barrier children have
        # no segment ids and manage their state synchronously at the barrier.
        settings_snapshot = RuntimeSettings.model_validate(job["settings_snapshot"])
        comfy_origin = Database.canonical_comfy_origin(
            settings_snapshot.comfy_url
        )
        for child in children:
            if (
                child["backend"] == "raylight"
                and child["status"] in _TERMINAL_STATUSES
                and child.get("prompt_id")
            ):
                database.settle_raylight_runtime_prompt(
                    comfy_origin,
                    str(child["prompt_id"]),
                    succeeded=child["status"] == "succeeded",
                    terminal_history_certified=(
                        _raylight_child_has_terminal_history_certificate(child)
                    ),
                )
        # RayKill barriers are persisted as control children so cancellation,
        # restart recovery and audit snapshots can still target their exact
        # prompt id. They carry no segment and must not inflate user-visible
        # segment progress or completion counts.
        segment_children = [child for child in children if child["segment_ids"]]
        control_children = [child for child in children if not child["segment_ids"]]
        segment_statuses = {child["status"] for child in segment_children}
        durable_all_terminal = all(
            child["status"] in _TERMINAL_STATUSES for child in children
        )
        failed_children = [
            child for child in segment_children if child["status"] == "failed"
        ]
        cancelled_children = [
            child for child in segment_children if child["status"] == "cancelled"
        ]
        failed_controls = [
            child
            for child in control_children
            if child["status"] == "failed"
        ]
        cancelled_controls = [
            child
            for child in control_children
            if child["status"] == "cancelled"
        ]
        total_segments = sum(len(child["segment_ids"]) for child in segment_children)
        progress = sum(
            float(child["progress"]) * len(child["segment_ids"])
            for child in segment_children
        ) / max(1, total_segments)
        updates: dict[str, Any] = {"progress": min(1.0, max(0.0, progress))}
        # Only an explicit operator request owns the terminal cancellation
        # result. Internal fail-closed submission cleanup also uses the live
        # ``cancelling`` state, but must not hide a failed RayKill barrier.
        explicit_cancel = honor_cancel_intent and bool(job.get("cancel_requested"))
        if durable_all_terminal and failed_controls:
            updates.update(
                status="failed",
                progress=1.0,
                stage="raylight_switch_failed",
                error=next(
                    (child["error"] for child in failed_controls if child["error"]),
                    "RayLight runtime switch did not complete",
                ),
                completed_at=utc_now(),
            )
        elif explicit_cancel and durable_all_terminal:
            updates.update(
                status="cancelled",
                progress=1.0,
                stage="cancelled",
                outputs=[],
                error=None,
                completed_at=utc_now(),
            )
        elif durable_all_terminal and cancelled_controls:
            director_cancelled = bool(job.get("cancel_requested")) and all(
                child.get("stage") in _DIRECTOR_CANCEL_STAGES
                for child in cancelled_controls
            )
            if director_cancelled:
                # During the first cancel probe, the live dispatcher can see
                # the parent claim and close its barrier before this coroutine
                # returns from ComfyUI. Preserve the cancellation lifecycle;
                # the following cancel pass (or retry) publishes the terminal
                # parent instead of misclassifying RayKill as a switch fault.
                updates.update(
                    status="cancelling",
                    stage="cancelling",
                    error=None,
                    completed_at=None,
                )
            else:
                updates.update(
                    status="failed",
                    progress=1.0,
                    stage="raylight_switch_failed",
                    error=next(
                        (
                            child["error"]
                            for child in cancelled_controls
                            if child["error"]
                        ),
                        "RayLight runtime switch did not complete",
                    ),
                    completed_at=utc_now(),
                )
        elif (
            honor_cancel_intent
            and job["status"] == "cancelling"
            and durable_all_terminal
        ):
            updates.update(
                status="cancelled",
                progress=1.0,
                stage="cancelled",
                outputs=[],
                error=None,
                completed_at=utc_now(),
            )
        elif durable_all_terminal and failed_children:
            updates.update(
                status="failed",
                progress=1.0,
                stage="segments_failed",
                error=next(
                    (child["error"] for child in failed_children if child["error"]),
                    f"{len(failed_children)} timeline segment(s) failed",
                ),
                completed_at=utc_now(),
            )
        elif (
            durable_all_terminal
            and cancelled_children
            and not honor_cancel_intent
            and bool(job.get("cancel_requested"))
            and all(
                child.get("stage") in _DIRECTOR_CANCEL_STAGES
                for child in cancelled_children
            )
        ):
            # These terminal child rows were produced by the still-live
            # submission cleanup, not by pre-click Comfy history. Keep the
            # parent active so the explicit cancel intent can win the cleanup
            # finalizer without mistaking its own exact cancellations for a
            # pre-existing partial generation failure.
            updates.update(
                status="cancelling",
                stage="cancelling",
                error=None,
                completed_at=None,
            )
        elif durable_all_terminal and cancelled_children:
            externally_cancelled = [
                child
                for child in cancelled_children
                if child.get("stage") in _EXTERNAL_CANCEL_STAGES
            ]
            if len(externally_cancelled) == len(segment_children):
                # ComfyUI's pending dequeue/queue-clear path has no durable
                # tombstone. Once every not-yet-started child is authoritatively
                # absent, make that neutral upstream-removal state visible
                # rather than leaving the Director task queued forever.
                updates.update(
                    status="cancelled",
                    progress=1.0,
                    stage="ComfyUI 端任务已移除",
                    outputs=[],
                    error=None,
                    completed_at=utc_now(),
                )
            else:
                # A child cannot disappear silently: a partial external cancel
                # makes the requested long-form result incomplete, while every
                # succeeded segment take remains available for inspection.
                updates.update(
                    status="failed",
                    progress=1.0,
                    stage="segments_cancelled",
                    error=(
                        f"{len(cancelled_children)} timeline segment(s) were cancelled "
                        "before the parent completed"
                    ),
                    completed_at=utc_now(),
                )
        elif durable_all_terminal and all(
            status == "succeeded" for status in segment_statuses
        ):
            try:
                ordered_outputs = _ordered_timeline_outputs(job, children)
            except ValueError as exc:
                updates.update(
                    status="failed",
                    progress=1.0,
                    stage="output_missing",
                    outputs=[],
                    error=str(exc),
                    completed_at=utc_now(),
                )
            else:
                snapshot = job.get("config_snapshot") or {}
                timeline = snapshot.get("timeline") or {}
                full_run = _is_full_timeline_selection(snapshot)
                if full_run and timeline.get("export_mode") == "all":
                    # A one-segment timeline is already the exact final movie;
                    # avoid a lossy and unnecessary ffmpeg round-trip.
                    if len(ordered_outputs) == 1:
                        updates.update(
                            status="succeeded",
                            progress=1.0,
                            stage="completed",
                            outputs=ordered_outputs,
                            error=None,
                            completed_at=utc_now(),
                        )
                        current_job_id = job["id"]
                        committed = database.update_job_if_snapshot(
                            current_job_id,
                            expected_status=job["status"],
                            expected_stage=job.get("stage"),
                            expected_updated_at=job["updated_at"],
                            **updates,
                        )
                        job = committed or database.get_job(current_job_id)
                        if job is None:
                            raise KeyError(current_job_id)
                        job["children"] = database.list_job_children(job["id"])
                        return job
                    if not allow_assembly:
                        updates.update(
                            status="running",
                            progress=1.0,
                            stage="segments_ready",
                            outputs=[],
                            error=None,
                            completed_at=None,
                        )
                        # The cancellation path calls this mode to reconcile
                        # child history without starting a new ffmpeg process.
                        # It will transition the parent to cancelling below.
                        pass
                    else:
                        claimed = database.claim_job_stage(
                            job["id"],
                            expected_status=job["status"],
                            expected_updated_at=job["updated_at"],
                            status="running",
                            stage="assembling",
                        )
                        if claimed is None:
                            latest = database.get_job(job["id"])
                            if latest is None:
                                raise KeyError(job["id"])
                            latest["children"] = database.list_job_children(job["id"])
                            return latest
                        try:
                            assembled = await _assemble_timeline_output(
                                request, claimed, ordered_outputs
                            )
                        except asyncio.CancelledError:
                            # A disconnected HTTP client or server shutdown can
                            # cancel the coroutine while ffmpeg/download/upload
                            # is in flight without terminating the backend
                            # process. Release this process-local claim so the
                            # next status reconciliation can retry assembly.
                            database.update_job_if_snapshot(
                                job["id"],
                                expected_status="running",
                                expected_stage="assembling",
                                expected_updated_at=claimed["updated_at"],
                                status="running",
                                stage="assembly_retry",
                                completed_at=None,
                            )
                            raise
                        except (MediaToolError, ComfyError, httpx.HTTPError, ValueError) as exc:
                            terminal_updates = dict(
                                status="failed",
                                progress=1.0,
                                stage="assembly_failed",
                                outputs=[],
                                error=str(exc),
                                completed_at=utc_now(),
                            )
                        except Exception:
                            # A programming/integration exception must not
                            # strand the durable process-local claim at
                            # ``assembling``.  Release it for one later,
                            # single-flight reconciler attempt and propagate
                            # the unexpected exception to normal observability.
                            database.update_job_if_snapshot(
                                job["id"],
                                expected_status="running",
                                expected_stage="assembling",
                                expected_updated_at=claimed["updated_at"],
                                status="running",
                                stage="assembly_retry",
                                completed_at=None,
                            )
                            raise
                        else:
                            terminal_updates = dict(
                                status="succeeded",
                                progress=1.0,
                                stage="completed",
                                outputs=[assembled],
                                error=None,
                                completed_at=utc_now(),
                            )
                        # Cancellation can win while ffmpeg or the output upload is
                        # in flight. Commit only against the exact assembly claim;
                        # a lost CAS leaves the current cancelling/cancelled state
                        # authoritative and never revives it.
                        committed = database.update_job_if_snapshot(
                            job["id"],
                            expected_status="running",
                            expected_stage="assembling",
                            expected_updated_at=claimed["updated_at"],
                            **terminal_updates,
                        )
                        if committed is None:
                            committed = database.get_job(job["id"])
                            if committed is None:
                                raise KeyError(job["id"])
                        committed["children"] = database.list_job_children(job["id"])
                        return committed
                else:
                    updates.update(
                        status="succeeded",
                        progress=1.0,
                        stage="segments_completed",
                        outputs=ordered_outputs,
                        error=None,
                        completed_at=utc_now(),
                    )
        else:
            updates.update(
                status=(
                    "cancelling"
                    if job["status"] == "cancelling"
                    else "preparing"
                    if segment_statuses & {"preparing"}
                    else "running"
                    if segment_statuses & {"running", "succeeded", "failed"}
                    else "queued"
                ),
                stage=(
                    job.get("stage") or "cancelling"
                    if job["status"] == "cancelling"
                    else job.get("stage") or "preflight"
                    if segment_statuses & {"preparing"}
                    else (
                        f"片段失败 {len(failed_children)} · "
                        f"已终态 {sum(child['status'] in _TERMINAL_STATUSES for child in segment_children)}/"
                        f"{len(segment_children)}"
                    )
                    if failed_children
                    else (
                        f"ComfyUI 端已移除 {len(cancelled_children)}/"
                        f"{len(segment_children)} · 其余排队中"
                    )
                    if cancelled_children and segment_statuses <= {"queued", "cancelled"}
                    else next(
                        (
                            child.get("stage")
                            for child in segment_children
                            if child["status"] == "running" and child.get("stage")
                        ),
                        None,
                    )
                    or (
                        "native segments "
                        f"{sum(child['status'] == 'succeeded' for child in segment_children)}/"
                        f"{len(segment_children)}"
                    )
                ),
                started_at=(
                    job.get("started_at")
                    or (
                        utc_now()
                        if segment_statuses & {"running", "succeeded"}
                        else None
                    )
                ),
            )
        committed = database.update_job_if_snapshot(
            job["id"],
            expected_status=job["status"],
            expected_stage=job.get("stage"),
            expected_updated_at=job["updated_at"],
            **updates,
        )
        if committed is None:
            committed = database.get_job(job["id"])
            if committed is None:
                raise KeyError(job["id"])
        job = committed
    job["children"] = database.list_job_children(job["id"])
    return job


async def _sync_timeline_job(
    request: Request,
    job: dict[str, Any],
    *,
    allow_assembly: bool = False,
) -> dict[str, Any]:
    """Coalesce every process-local background sync for one timeline parent.

    The reconciler and explicit cancellation recovery may converge on the same
    parent. Only the first caller owns ComfyUI queue/history reads and possible
    assembly; later background callers await that process-owned task. The
    runner deliberately re-reads SQLite after registration so a stale scan
    snapshot cannot revive work that changed before it ran. Public list/detail
    GET handlers do not call this function; they are SQLite-only.

    ``asyncio.shield`` detaches the work from any one HTTP connection.  The
    lifespan shutdown path remains the owner that can cancel these tasks and,
    for an in-flight assembly, release the durable assembly claim.
    """

    database = _db(request)
    job_id = str(job["id"])
    tasks: dict[str, asyncio.Task[dict[str, Any]]] = (
        request.app.state.timeline_sync_tasks
    )
    lock: asyncio.Lock = request.app.state.timeline_sync_lock

    async with lock:
        task = tasks.get(job_id)
        bypass_assembly_flight = bool(
            task is not None
            and not task.done()
            and not allow_assembly
            and bool(getattr(task, "_director_allow_assembly", True))
        )
        if not bypass_assembly_flight and (task is None or task.done()):

            async def run() -> dict[str, Any]:
                latest = database.get_job(job_id)
                if latest is None:
                    raise KeyError(job_id)
                return await _sync_timeline_job_once(
                    request, latest, allow_assembly=allow_assembly
                )

            task = _register_timeline_sync_task(
                request,
                job_id,
                run(),
                allow_assembly=allow_assembly,
                honor_cancel_intent=True,
            )

    if bypass_assembly_flight:
        # HTTP reads must never queue behind multi-minute download/ffmpeg/upload
        # work owned by the background reconciler. The assembly claim and any
        # already-persisted child takes are immediately observable in SQLite.
        latest = database.get_job(job_id)
        if latest is None:
            raise KeyError(job_id)
        latest["children"] = database.list_job_children(job_id)
        return latest

    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # A priority cancellation request may preempt this managed flight to
        # stop an in-process assembly.  Preserve the original HTTP waiter: it
        # should observe the replacement no-assembly flight's authoritative
        # parent result, not inherit the internal preemption as a disconnect.
        replacement = getattr(task, "_director_replacement", None)
        if replacement is None:
            async with lock:
                replacement = tasks.get(job_id)
        if replacement is None or replacement is task:
            raise
        reconciled = await asyncio.shield(replacement)
        if (
            reconciled["status"] not in _TERMINAL_STATUSES
            and bool(reconciled.get("cancel_requested"))
        ):
            # The replacement's first pass deliberately ignores the just-set
            # cancel flag so history that predates the click can win. That
            # pass may therefore publish an intermediate ``segments_ready``
            # snapshot after preempting assembly. Existing GET/reconciler
            # waiters must not return that stale pre-cancel view after the
            # cancel request has already taken ownership; one ordinary pass
            # applies the durable intent without starting assembly again.
            latest = database.get_job(job_id)
            if latest is None:
                raise KeyError(job_id)
            return await _sync_timeline_job(
                request,
                latest,
                allow_assembly=False,
            )
        return reconciled


def _register_timeline_sync_task(
    request: Request,
    job_id: str,
    coroutine: Any,
    *,
    allow_assembly: bool,
    honor_cancel_intent: bool,
) -> asyncio.Task[dict[str, Any]]:
    """Register one managed flight; caller holds ``timeline_sync_lock``."""

    tasks: dict[str, asyncio.Task[dict[str, Any]]] = (
        request.app.state.timeline_sync_tasks
    )
    all_tasks: set[asyncio.Task[dict[str, Any]]] = (
        request.app.state.timeline_sync_all_tasks
    )
    task = asyncio.create_task(coroutine, name=f"timeline-sync:{job_id}")
    setattr(task, "_director_allow_assembly", allow_assembly)
    setattr(task, "_director_honor_cancel_intent", honor_cancel_intent)
    tasks[job_id] = task
    all_tasks.add(task)

    def consume(done: asyncio.Task[dict[str, Any]]) -> None:
        # Identity protects a priority replacement if callback scheduling is
        # delayed until after cancellation acquired the short registry lock.
        if tasks.get(job_id) is done:
            tasks.pop(job_id, None)
        all_tasks.discard(done)
        if not done.cancelled():
            done.exception()

    task.add_done_callback(consume)
    return task


async def _sync_timeline_before_cancel(
    request: Request,
    job: dict[str, Any],
    *,
    honor_cancel_intent: bool,
) -> dict[str, Any]:
    """Reserve a no-assembly flight, preempting active post-processing.

    With no active flight this preserves the historical rule that a prompt
    already completed upstream is reconciled before cancellation is claimed.
    If assembly/status sync is active, cancellation first wins the durable
    parent status, then cancels and replaces that flight so an ffmpeg result
    cannot commit success after the user's request.
    """

    database = _db(request)
    job_id = str(job["id"])
    tasks: dict[str, asyncio.Task[dict[str, Any]]] = (
        request.app.state.timeline_sync_tasks
    )
    lock: asyncio.Lock = request.app.state.timeline_sync_lock
    async with lock:
        active = tasks.get(job_id)
        dispatcher = request.app.state.submission_jobs.get(job_id)
        if dispatcher is not None and not dispatcher.done():
            # A live submit/cleanup coroutine still owns the parent. Claim
            # cancellation before reconciliation so a partially-cleaned child
            # cannot make the parent failed in the narrow interval before the
            # dispatcher atomically publishes its own terminal result.
            latest = database.get_job(job_id)
            if latest is None:
                raise KeyError(job_id)
            if latest["status"] not in _TERMINAL_STATUSES and latest["status"] != "cancelling":
                latest = database.update_job_if_status(
                    job_id,
                    latest["status"],
                    status="cancelling",
                    stage="cancelling",
                    error=None,
                    completed_at=None,
                ) or database.get_job(job_id)
                if latest is None:
                    raise KeyError(job_id)
        if (
            active is not None
            and not active.done()
            and not bool(getattr(active, "_director_allow_assembly", True))
            and (
                not bool(getattr(active, "_director_honor_cancel_intent", True))
                or honor_cancel_intent
            )
        ):
            task = active
        else:
            latest = database.get_job(job_id)
            if latest is None:
                raise KeyError(job_id)
            if (
                active is not None
                and not active.done()
                and latest["status"] not in _TERMINAL_STATUSES
            ):
                latest = database.update_job_if_status(
                    job_id,
                    latest["status"],
                    status="cancelling",
                    stage="cancelling",
                    error=None,
                    completed_at=None,
                ) or database.get_job(job_id)
                if latest is None:
                    raise KeyError(job_id)
            async def run() -> dict[str, Any]:
                latest = database.get_job(job_id)
                if latest is None:
                    raise KeyError(job_id)
                return await _sync_timeline_job_once(
                    request,
                    latest,
                    allow_assembly=False,
                    # The first pass of an explicit cancel request must honor
                    # terminal history that already existed before the click.
                    # The durable intent still wins anything arriving later
                    # during upstream cancellation dispatch.
                    honor_cancel_intent=honor_cancel_intent,
                )

            task = _register_timeline_sync_task(
                request,
                job_id,
                run(),
                allow_assembly=False,
                honor_cancel_intent=honor_cancel_intent,
            )
            if active is not None and not active.done():
                # Attach before cancellation. The replacement may finish and leave
                # the registry before existing waiters handle CancelledError; the
                # old task then carries their exact hand-off without a global leak.
                setattr(active, "_director_replacement", task)
                active.cancel()
    return await asyncio.shield(task)


async def _sync_existing_job(
    request: Request,
    job: dict[str, Any],
    *,
    allow_timeline_assembly: bool = False,
) -> dict[str, Any]:
    """Map a row disappearing during an upstream await to the public 404 contract."""

    try:
        return await _sync_job(
            request,
            job,
            allow_timeline_assembly=allow_timeline_assembly,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc


async def _quiesce_cancelled_submission_dispatcher(
    request: Request, result: dict[str, Any]
) -> dict[str, Any]:
    """Stop a cancelled job's dispatcher before publishing the API result.

    A retry can enter cancellation after another owner has already committed
    the terminal parent and children.  That retry still owns the responsibility
    to interrupt a dispatcher waiting behind an endpoint ticket; otherwise the
    terminal row remains undeletable until an unrelated predecessor finishes.
    Succeeded, failed, and still-cancelling results may retain ambiguous submit
    cleanup, so they deliberately keep the normal live-dispatcher delete guard.
    """

    if result["status"] != "cancelled":
        return result
    job_id = str(result["id"])
    dispatcher = request.app.state.submission_jobs.get(job_id)
    current = asyncio.current_task()
    if dispatcher is None or dispatcher is current:
        return result
    if not dispatcher.done():
        dispatcher.cancel()
        done, _ = await asyncio.wait({dispatcher}, timeout=2.0)
        if dispatcher not in done:
            return result
    if not dispatcher.cancelled():
        try:
            dispatcher.result()
        except HTTPException:
            pass
    # The task's registered callback normally removes this entry first.  Make
    # the ownership handoff synchronous with the cancel response as well, so
    # an immediate following DELETE cannot observe a completed stale owner.
    if request.app.state.submission_jobs.get(job_id) is dispatcher:
        request.app.state.submission_jobs.pop(job_id, None)
    return result


async def _cancel_timeline_job(
    request: Request,
    job: dict[str, Any],
    *,
    initial_cancel_claimed: bool = False,
) -> dict[str, Any]:
    """Cancel every unfinished native workflow unit owned by one parent job.

    Calling this again while the parent is already ``cancelling`` deliberately
    re-dispatches cancellation for every unfinished bound prompt.  This is the
    operator-controlled retry path after a transient ``cancel_failed`` result.
    """

    database = _db(request)
    # First reconcile an upstream prompt that has already reached history.
    # Cancellation must not overwrite a real success merely because the
    # browser's last polled parent snapshot was still queued/running.
    job = await _sync_timeline_before_cancel(
        request,
        job,
        # Only the request that atomically persisted the first cancel intent
        # gets the pre-click truth probe. Retries must honor terminal children
        # produced by that earlier request instead of reclassifying them.
        honor_cancel_intent=not initial_cancel_claimed,
    )
    if job["status"] in _TERMINAL_STATUSES:
        return await _quiesce_cancelled_submission_dispatcher(request, job)
    while job["status"] != "cancelling":
        transitioned = database.update_job_if_status(
            job["id"],
            job["status"],
            status="cancelling",
            stage="cancelling",
            error=None,
            completed_at=None,
        )
        if transitioned is not None:
            job = transitioned
            break
        latest = database.get_job(job["id"])
        if latest is None:
            raise HTTPException(status_code=404, detail="job not found")
        if latest["status"] in _TERMINAL_STATUSES:
            result = await _sync_timeline_job(request, latest)
            return await _quiesce_cancelled_submission_dispatcher(
                request, result
            )
        job = latest

    client = _comfy(
        request, RuntimeSettings.model_validate(job["settings_snapshot"])
    )
    dispatch_errors: list[str] = []
    recovery_error_prefixes: set[str] = set()
    for child in database.list_job_children(job["id"]):
        if child["status"] in _TERMINAL_STATUSES:
            continue
        if not child.get("prompt_id"):
            await _cas_active_child_update(
                database,
                child,
                status="cancelled",
                progress=1.0,
                stage="cancelled",
                completed_at=utc_now(),
            )
            continue
        # The restart worker and a user retry can list the same child before
        # either awaits ComfyUI. Claim the exact row version; if the other path
        # won, reload and never turn its terminal result back into cancelling.
        while child["status"] not in _TERMINAL_STATUSES:
            live_submission_owned = (
                child.get("stage") in _SUBMISSION_OWNERSHIP_STAGES
            )
            recovery_owned = child.get("stage") in _RECOVERY_OWNERSHIP_STAGES
            recovery_prefix = (
                "restart"
                if str(child.get("stage") or "").startswith("restart_")
                else "submission"
            )
            child, claimed = await _cas_active_child_update(
                database,
                child,
                status="cancelling",
                stage=(
                    "cancelling_during_submit"
                    if live_submission_owned
                    else f"{recovery_prefix}_cancel_pending"
                    if recovery_owned
                    else "cancelling"
                ),
            )
            if claimed:
                break
        if child["status"] in _TERMINAL_STATUSES:
            continue
        prompt_id = str(child["prompt_id"])
        try:
            dispatched = await client.cancel(prompt_id)
        except (ComfyError, httpx.HTTPError) as exc:
            dispatch_errors.append(f"{child['id']}: {exc}")
            if recovery_owned:
                recovery_error_prefixes.add(recovery_prefix)
            await _cas_active_child_update(
                database,
                child,
                stage=(
                    "cancelling_during_submit"
                    if live_submission_owned
                    else f"{recovery_prefix}_cancel_failed"
                    if recovery_owned
                    else "cancel_failed"
                ),
                error=str(exc),
            )
            continue
        if dispatched and not live_submission_owned:
            # A successful directed cancellation is safe to persist against a
            # newer nonterminal cancellation marker, but only while it still
            # names the exact prompt we cancelled.
            confirmation = child
            while (
                confirmation["status"] not in _TERMINAL_STATUSES
                and str(confirmation.get("prompt_id") or "") == prompt_id
            ):
                confirmation, committed = await _cas_active_child_update(
                    database,
                    confirmation,
                    status="cancelled",
                    progress=1.0,
                    stage="cancelled",
                    error=None,
                    completed_at=utc_now(),
                )
                if committed:
                    break
    if dispatch_errors:
        latest = database.get_job(job["id"])
        if latest is None:
            raise HTTPException(status_code=404, detail="job not found")
        if latest["status"] not in _TERMINAL_STATUSES:
            job = database.update_job_if_snapshot(
                latest["id"],
                expected_status=latest["status"],
                expected_stage=latest.get("stage"),
                expected_updated_at=latest["updated_at"],
                status="cancelling",
                stage=(
                    "restart_cancel_failed"
                    if "restart" in recovery_error_prefixes
                    else "submission_cancel_failed"
                    if "submission" in recovery_error_prefixes
                    else "cancel_failed"
                ),
                error="; ".join(dispatch_errors)[:20_000],
            ) or database.get_job(latest["id"])
            if job is None:
                raise HTTPException(status_code=404, detail="job not found")
        else:
            job = latest
    else:
        latest = database.get_job(job["id"])
        if latest is None:
            raise HTTPException(status_code=404, detail="job not found")
        job = latest
    result = await _sync_timeline_job(request, job)
    return await _quiesce_cancelled_submission_dispatcher(request, result)


def _invalid_comfy_payload(code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=502,
        detail={"code": code, "message": message},
    )


def _raylight_recovery_in_flight(message: str) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={
            "code": "raylight_recovery_in_flight",
            "message": message,
        },
    )


def _cuda_devices(stats: dict[str, Any]) -> list[dict[str, Any]]:
    """Return CUDA entries keyed by their reported Comfy logical identity."""

    if not isinstance(stats, dict):
        raise _invalid_comfy_payload(
            "comfy_system_stats_invalid",
            "ComfyUI /system_stats response must be an object",
        )
    if "devices" in stats:
        devices = stats["devices"]
    else:
        system = stats.get("system")
        devices = system.get("devices") if isinstance(system, dict) else None
    if not isinstance(devices, list):
        raise _invalid_comfy_payload(
            "comfy_system_stats_invalid",
            "ComfyUI /system_stats response must contain a devices list",
        )

    cuda_devices: list[dict[str, Any]] = []
    for device in devices:
        if not isinstance(device, dict):
            raise _invalid_comfy_payload(
                "comfy_system_stats_invalid",
                "ComfyUI /system_stats contains a malformed device entry",
            )
        device_type = device.get("type")
        if not isinstance(device_type, str) or not device_type.strip():
            raise _invalid_comfy_payload(
                "comfy_system_stats_invalid",
                "ComfyUI /system_stats device type is invalid",
            )
        if device_type.strip().lower() != "cuda":
            continue
        index = device.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise _invalid_comfy_payload(
                "comfy_system_stats_invalid",
                "ComfyUI /system_stats CUDA device index is invalid",
            )
        for field in ("vram_total", "vram_free"):
            value = device.get(field, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise _invalid_comfy_payload(
                    "comfy_system_stats_invalid",
                    f"ComfyUI /system_stats CUDA device {field} is invalid",
                )
        cuda_devices.append(device)

    cuda_devices.sort(key=lambda device: int(device["index"]))
    reported = [int(device["index"]) for device in cuda_devices]
    if reported != list(range(len(cuda_devices))):
        raise _invalid_comfy_payload(
            "comfy_system_stats_invalid",
            "ComfyUI /system_stats CUDA indexes must be unique and dense from zero",
        )
    return cuda_devices


def _device_names(stats: dict[str, Any]) -> set[str]:
    result = {"default", "cpu"}
    for device in _cuda_devices(stats):
        result.add(f"gpu:{device['index']}")
    return result


def _visible_logical_gpu_indexes(stats: dict[str, Any]) -> tuple[int, ...]:
    """Mirror RayLight's zero-based ``torch.cuda.device_count`` namespace.

    ``/system_stats`` is validated as the same dense logical namespace before
    this range is constructed. Device array order is not identity: ComfyUI may
    place its primary CUDA device first.
    """

    return tuple(int(device["index"]) for device in _cuda_devices(stats))


def _raylight_runtime_recovery_token(
    comfy_origin: str,
    state: dict[str, Any],
    visible: tuple[int, ...],
) -> str:
    payload = {
        "version": 1,
        "comfy_origin": Database.canonical_comfy_origin(comfy_origin),
        "runtime_state": state,
        "available_gpu_indexes": list(visible),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _raylight_runtime_status(
    database: Database,
    comfy_origin: str,
    stats: dict[str, Any],
    *,
    state: dict[str, Any] | None = None,
) -> RayLightRuntimeStatusRead:
    try:
        origin = Database.canonical_comfy_origin(comfy_origin)
        runtime = (
            state
            if state is not None
            else database.get_raylight_runtime_state(origin)
        )
        visible = _visible_logical_gpu_indexes(stats)
        if runtime is None:
            return RayLightRuntimeStatusRead(
                active=False,
                recovery_required=False,
                epoch=0,
                runtime_gpu_indexes=[],
                available_gpu_indexes=list(visible),
                invalid_gpu_indexes=[],
                tainted=False,
                recovery_token=None,
            )
        current = runtime.get("current")
        recorded = (
            raylight_runtime_logical_gpu_indices(current)
            if isinstance(current, dict)
            else ()
        )
        invalid = tuple(index for index in recorded if index >= len(visible))
        return RayLightRuntimeStatusRead(
            active=isinstance(current, dict),
            recovery_required=bool(invalid),
            epoch=int(runtime["epoch"]),
            runtime_gpu_indexes=list(recorded),
            available_gpu_indexes=list(visible),
            invalid_gpu_indexes=list(invalid),
            tainted=bool(runtime.get("tainted")),
            recovery_token=(
                _raylight_runtime_recovery_token(origin, runtime, visible)
                if invalid
                else None
            ),
        )
    except (KeyError, NativeTemplateError, TypeError, ValidationError, ValueError) as exc:
        raise NativeTemplateError(
            "persisted RayLight runtime state is invalid"
        ) from exc


def _raylight_runtime_recovery_detail(
    status: RayLightRuntimeStatusRead,
) -> dict[str, Any]:
    invalid = ", ".join(str(index) for index in status.invalid_gpu_indexes)
    return {
        "code": "raylight_runtime_restart_confirmation_required",
        "message": (
            "旧 RayLight 运行状态引用了当前不可见的 ComfyUI 逻辑 GPU "
            f"{invalid}；请在系统设置中确认 ComfyUI 已重启并恢复 RayLight"
        ),
        "runtime_gpu_indexes": status.runtime_gpu_indexes,
        "available_gpu_indexes": status.available_gpu_indexes,
        "invalid_gpu_indexes": status.invalid_gpu_indexes,
        "expected_epoch": status.epoch,
        "recovery_token": status.recovery_token,
    }


async def _preflight_timeline(
    client: ComfyClientProtocol,
    settings: RuntimeSettings,
    compiled: NativeCompileResult,
    database: Database,
) -> None:
    """Verify exactly the server-selected native templates and model inventory."""

    capabilities = await client.capabilities()
    raylight_contract_issues = (
        (capabilities.get("execution_backends") or {})
        .get("raylight", {})
        .get("contract_issues", [])
    )
    if (
        "RayInitializerAdvanced" in compiled.node_policy["allowed_nodes"]
        and raylight_contract_issues
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "raylight_contract_mismatch",
                "message": (
                    "installed RayLight is not the Director-compatible build: "
                    + "; ".join(str(item) for item in raylight_contract_issues)
                ),
                "issues": list(raylight_contract_issues),
            },
        )
    try:
        validate_native_capabilities(
            compiled,
            set(capabilities.get("available_nodes") or []),
            {
                str(name): str(source)
                for name, source in (
                    capabilities.get("node_provenance") or {}
                ).items()
            }
            if isinstance(capabilities.get("node_provenance"), dict)
            else None,
        )
    except NativeTemplateError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "message": str(exc),
                "nodes": sorted(
                    set(compiled.node_policy["allowed_nodes"])
                    - set(capabilities.get("available_nodes") or [])
                ),
            },
        ) from exc
    inventory = await client.models()
    expected: dict[str, str] = {
        "clip": settings.models.clip.filename,
        "video_vae": settings.models.video_vae.filename,
        "audio_vae": settings.models.audio_vae.filename,
    }
    placements = {
        settings.models.clip.device,
        settings.models.video_vae.device,
        settings.models.audio_vae.device,
    }
    for family in compiled.families:
        if family not in {"fl2va", "ref2va"}:
            raise HTTPException(status_code=422, detail=f"unknown model family: {family}")
        binding = getattr(settings.models, family)
        expected[family] = binding.filename
        placements.add(binding.device)
        if binding.lora_name is not None:
            assert binding.lora_name is not None
            expected[f"loras:{family}"] = binding.lora_name
        backend = next(
            unit.backend for unit in compiled.workflows if unit.family == family
        )
        if backend == "raylight":
            placements.update(f"gpu:{index}" for index in binding.raylight.gpu_select)
    absent: list[str] = []
    for role, filename in expected.items():
        inventory_role = "loras" if role.startswith("loras:") else role
        if filename not in inventory.get(inventory_role, []):
            absent.append(f"{role}:{filename}")
    if absent:
        raise HTTPException(
            status_code=409,
            detail={"message": "configured model files are unavailable", "models": absent},
        )
    stats = await client.system_stats()
    available_devices = _device_names(stats)
    invalid_devices = sorted(
        device for device in placements if device not in available_devices
    )
    if invalid_devices:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "configured logical GPU is unavailable",
                "devices": invalid_devices,
            },
        )
    try:
        runtime_status = _raylight_runtime_status(
            database, str(settings.comfy_url), stats
        )
    except NativeTemplateError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "raylight_runtime_state_invalid",
                "message": str(exc),
            },
        ) from exc
    if runtime_status.recovery_required:
        raise HTTPException(
            status_code=409,
            detail=_raylight_runtime_recovery_detail(runtime_status),
        )


async def _preflight_raylight_transition(
    client: ComfyClientProtocol,
    unit: NativeWorkflowUnit,
    database: Database,
    comfy_origin: str,
) -> None:
    """Verify the server-owned RayKill barrier before it enters ComfyUI."""

    capabilities = await client.capabilities()
    raylight_contract_issues = (
        (capabilities.get("execution_backends") or {})
        .get("raylight", {})
        .get("contract_issues", [])
    )
    if raylight_contract_issues:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "raylight_contract_mismatch",
                "message": (
                    "installed RayLight is not the Director-compatible build: "
                    + "; ".join(str(item) for item in raylight_contract_issues)
                ),
                "issues": list(raylight_contract_issues),
            },
        )
    available = set(capabilities.get("available_nodes") or [])
    provenance = (
        {
            str(name): str(source)
            for name, source in capabilities["node_provenance"].items()
        }
        if isinstance(capabilities.get("node_provenance"), dict)
        else None
    )
    try:
        validate_native_workflow_unit_capabilities(
            unit, available, node_provenance=provenance
        )
    except NativeTemplateError as exc:
        required = {
            str(node.get("class_type") or "")
            for node in unit.prompt.values()
        }
        raise HTTPException(
            status_code=409,
            detail={
                "message": str(exc),
                "nodes": sorted(required - available),
            },
        ) from exc
    stats = await client.system_stats()
    try:
        runtime_status = _raylight_runtime_status(database, comfy_origin, stats)
    except NativeTemplateError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "raylight_runtime_state_invalid",
                "message": str(exc),
            },
        ) from exc
    visible = tuple(runtime_status.available_gpu_indexes)
    recorded = raylight_workflow_logical_gpu_indices(unit)
    invalid = [index for index in recorded if index >= len(visible)]
    if invalid:
        status = (
            runtime_status
            if runtime_status.recovery_required
            and tuple(runtime_status.runtime_gpu_indexes) == recorded
            else RayLightRuntimeStatusRead(
                active=True,
                recovery_required=True,
                epoch=runtime_status.epoch,
                runtime_gpu_indexes=list(recorded),
                available_gpu_indexes=list(visible),
                invalid_gpu_indexes=invalid,
                tainted=True,
                recovery_token=None,
            )
        )
        raise HTTPException(
            status_code=409,
            detail=_raylight_runtime_recovery_detail(status),
        )


async def _await_raylight_transition(
    client: ComfyClientProtocol,
    prompt_id: str,
    *,
    timeout_seconds: float = 300.0,
    database: Database | None = None,
    job_id: str | None = None,
    child_id: str | None = None,
) -> bool:
    """Wait for positive RayKill history without timing out in queue backlog.

    ``timeout_seconds`` limits only time during which the barrier is no longer
    visibly queued/running. A long video ahead of it may legitimately keep it
    pending for much longer, so queue presence resets that ambiguity deadline.
    """

    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while True:
        if database is not None and job_id is not None:
            parent = database.get_job(job_id)
            if parent is None:
                raise KeyError(job_id)
            child = database.get_job_child(child_id) if child_id is not None else None
            if parent["status"] in {"cancelling", "cancelled"} or (
                child is not None and child["status"] in {"cancelling", "cancelled"}
            ):
                return False
        history = await client.history(prompt_id)
        entry, exact_absent = _exact_history_result(history, prompt_id)
        if entry is not None:
            history_state = _raylight_recovery_history_state(entry)
            if history_state == "invalid":
                raise ComfyError(
                    "RayLight safe-switch history status is contradictory; "
                    "Standard workflow was not submitted"
                )
            status = entry["status"]
            assert isinstance(status, dict)
            status_value = str(status["status_str"]).strip().lower()
            error = _history_error(entry)
            interrupted = _history_has_event(entry, "execution_interrupted")
            failed = history_state == "terminal" and status_value in {
                "error",
                "failed",
            }
            succeeded = history_state == "terminal" and status_value == "success"
            if failed:
                if database is not None and child_id is not None:
                    child = database.get_job_child(child_id)
                    if child is not None and child["status"] not in _TERMINAL_STATUSES:
                        database.update_job_child_if_snapshot(
                            child_id,
                            expected_status=child["status"],
                            expected_updated_at=child["updated_at"],
                            status="failed",
                            progress=1.0,
                            stage="RayLight 安全切换失败",
                            outputs=[],
                            error=error or "RayLight safe-switch barrier failed",
                            completed_at=utc_now(),
                        )
                raise ComfyError(
                    "RayLight safe-switch barrier failed; Standard workflow was not submitted"
                    + (f": {error}" if error else "")
                )
            if succeeded:
                return True
        elif not exact_absent:
            raise ComfyError(
                "RayLight safe-switch history response is malformed; "
                "Standard workflow was not submitted"
            )
        queue = await client.queue()
        queue_valid = (
            isinstance(queue, dict)
            and isinstance(queue.get("queue_running"), list)
            and isinstance(queue.get("queue_pending"), list)
        )
        if not queue_valid:
            raise ComfyError(
                "RayLight safe-switch queue response is malformed; target workflow was not submitted"
            )
        visible = _contains_prompt(
            queue["queue_running"], prompt_id
        ) or _contains_prompt(queue["queue_pending"], prompt_id)
        now = asyncio.get_running_loop().time()
        if visible:
            # A queued/running prompt is not hung merely because an earlier
            # generation is slow. Once it disappears, exact history still has
            # a full bounded hand-off window in which to become visible.
            deadline = now + timeout_seconds
        elif now >= deadline:
            raise ComfyError(
                "RayLight safe-switch barrier timed out; Standard workflow was not submitted"
            )
        await asyncio.sleep(0.25)


async def _await_timeline_generation(
    client: ComfyClientProtocol,
    database: Database,
    job_id: str,
    child_id: str,
    prompt_id: str,
    *,
    stop_on_parent_cancel: bool = True,
    dispatch_job_id: str | None = None,
    running_stage: str = "sampling",
    error_context: str = "timeline generation",
    terminal_events: PromptTerminalEvents | None = None,
) -> dict[str, Any]:
    """Terminal-gate one generation before a dependent successor is submitted.

    Exact history is the positive terminal authority for both Standard
    continuity predecessors and Ray generations. Two complete queue/history
    absence observations cover external pending removal without confusing the
    normal queue-to-history hand-off window.
    """

    absent_observations = 0
    while True:
        parent = database.get_job(job_id)
        if parent is None:
            raise KeyError(job_id)
        child = database.get_job_child(child_id)
        if child is None:
            raise KeyError(child_id)
        if child["status"] in _TERMINAL_STATUSES:
            return child
        if dispatch_job_id is not None and dispatch_job_id != job_id:
            dispatch_parent = database.get_job(dispatch_job_id)
            if dispatch_parent is None:
                raise KeyError(dispatch_job_id)
            if dispatch_parent["status"] in {"cancelling", "cancelled"}:
                raise asyncio.CancelledError
        was_running = child["status"] == "running" or child.get("started_at") is not None
        # The explicit Director cancellation path owns upstream interruption
        # and closes every unsubmitted sibling. Stop this gate immediately so
        # the endpoint lock cannot later submit a successor for that parent.
        if (
            stop_on_parent_cancel
            and parent["status"] in {"cancelling", "cancelled"}
        ):
            return child
        history = await client.history(prompt_id)
        entry, exact_absent = _exact_history_result(history, prompt_id)
        if entry is not None:
            raw_status = entry.get("status")
            status = raw_status if isinstance(raw_status, dict) else {}
            status_value = str(status.get("status_str") or "").lower()
            error = _history_error(entry)
            interrupted = _history_has_event(entry, "execution_interrupted")
            if child.get("backend") == "raylight":
                history_state = _raylight_recovery_history_state(entry)
                if history_state == "invalid":
                    raise ComfyError(
                        f"{error_context} history status is contradictory; "
                        "successor was not submitted"
                    )
                failed = history_state == "terminal" and status_value in {
                    "error",
                    "failed",
                }
                succeeded = (
                    history_state == "terminal" and status_value == "success"
                )
            else:
                failed = status_value in {"error", "failed"} or bool(error)
                succeeded = status_value == "success" or (
                    bool(status.get("completed"))
                    and not failed
                    and not interrupted
                )
            if interrupted or failed or succeeded:
                updates = {
                    "status": (
                        "cancelled" if interrupted else "failed" if failed else "succeeded"
                    ),
                    "progress": 1.0,
                    "stage": (
                        "ComfyUI 端已中断"
                        if interrupted
                        else "failed"
                        if failed
                        else "completed"
                    ),
                    "outputs": (
                        []
                        if interrupted or failed
                        else _collect_output_files(entry.get("outputs") or {})
                    ),
                    "error": error if failed else None,
                    "completed_at": utc_now(),
                }
                committed = database.update_job_child_if_snapshot(
                    child_id,
                    expected_status=child["status"],
                    expected_updated_at=child["updated_at"],
                    **updates,
                )
                if committed is not None:
                    return committed
                continue
        elif not exact_absent:
            raise ComfyError(
                f"{error_context} history response is malformed; successor was not submitted"
            )

        queue = await client.queue()
        if (
            not isinstance(queue, dict)
            or not isinstance(queue.get("queue_running"), list)
            or not isinstance(queue.get("queue_pending"), list)
        ):
            raise ComfyError(
                f"{error_context} queue response is malformed; successor was not submitted"
            )
        running = _contains_prompt(queue["queue_running"], prompt_id)
        visible = running or _contains_prompt(queue["queue_pending"], prompt_id)
        if running and child["status"] == "queued":
            advanced = database.update_job_child_if_snapshot(
                child_id,
                expected_status=child["status"],
                expected_updated_at=child["updated_at"],
                status="running",
                progress=max(float(child["progress"]), 0.01),
                stage=child.get("stage") or running_stage,
                started_at=child.get("started_at") or utc_now(),
            )
            if advanced is not None:
                child = advanced
                was_running = True
        if visible:
            absent_observations = 0
        elif exact_absent:
            if child.get("stage") in _PROCESS_OWNERSHIP_STAGES:
                # An ambiguous/lost submit response can precede a delayed
                # Comfy queue.put. Only the recovery owner may close it by
                # directed cancellation or exact terminal history; a newer
                # parent must remain behind this endpoint gate meanwhile.
                absent_observations = 0
                await asyncio.sleep(_RAYLIGHT_GENERATION_POLL_SECONDS)
                continue
            absent_observations += 1
            if absent_observations >= 2:
                committed = database.update_job_child_if_snapshot(
                    child_id,
                    expected_status=child["status"],
                    expected_updated_at=child["updated_at"],
                    status=(
                        "failed" if was_running else "cancelled"
                    ),
                    progress=1.0,
                    stage=(
                        "ComfyUI 端运行任务丢失"
                        if was_running
                        else "ComfyUI 端任务已移除"
                    ),
                    outputs=[],
                    error=(
                        f"ComfyUI restarted or lost a running {error_context} prompt"
                        if was_running
                        else None
                    ),
                    completed_at=utc_now(),
                )
                if committed is not None:
                    return committed
        if terminal_events is not None:
            gate_event = terminal_events.register(prompt_id)
            try:
                gate_event.clear()
                try:
                    await asyncio.wait_for(
                        gate_event.wait(),
                        timeout=_RAYLIGHT_GENERATION_POLL_SECONDS,
                    )
                except asyncio.TimeoutError:
                    pass
            finally:
                terminal_events.unregister(prompt_id)
        else:
            await asyncio.sleep(_RAYLIGHT_GENERATION_POLL_SECONDS)


async def _await_raylight_generation(
    client: ComfyClientProtocol,
    database: Database,
    job_id: str,
    child_id: str,
    prompt_id: str,
    *,
    stop_on_parent_cancel: bool = True,
    dispatch_job_id: str | None = None,
    terminal_events: PromptTerminalEvents | None = None,
) -> dict[str, Any]:
    """Preserve the Ray-specific gate while sharing exact terminal logic."""

    return await _await_timeline_generation(
        client,
        database,
        job_id,
        child_id,
        prompt_id,
        stop_on_parent_cancel=stop_on_parent_cancel,
        dispatch_job_id=dispatch_job_id,
        running_stage="RayLight 采样中",
        error_context="RayLight generation",
        terminal_events=terminal_events,
    )


async def _refresh_raylight_runtime_tail(
    client: ComfyClientProtocol,
    database: Database,
    comfy_origin: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Refresh the persisted Ray queue tail before planning another batch.

    The ledger is intentionally conservative across cancellation, transport
    ambiguity and backend restarts. Positive exact history certifies reuse;
    a failed/interrupted tail or a prompt proven absent from both a structurally
    valid queue and a second exact-history read taints the cached actor chain.
    A still queued/running tail remains valid because the endpoint submission
    lock appends the next Director unit behind it.
    """

    current = state.get("current")
    if not isinstance(current, dict):
        return state
    tail_prompt_id = state.get("tail_prompt_id")
    if not isinstance(tail_prompt_id, str) or not tail_prompt_id:
        state = dict(state)
        state["tainted"] = True
        database.put_raylight_runtime_state(comfy_origin, state)
        return state
    tail_children = database.find_any_job_children_by_prompt_id(tail_prompt_id)
    ambiguous_tail = (
        len(tail_children) == 1
        and tail_children[0]["status"] not in _TERMINAL_STATUSES
        and tail_children[0].get("stage") in _PROCESS_OWNERSHIP_STAGES
    )

    def terminal_result(entry: dict[str, Any]) -> bool | None:
        history_state = _raylight_recovery_history_state(entry)
        if history_state == "invalid":
            raise ComfyError(
                "RayLight runtime tail history is contradictory; refusing to reuse cached actors"
            )
        if history_state == "nonterminal":
            return None
        status = entry["status"]
        assert isinstance(status, dict)
        return str(status["status_str"]).strip().lower() == "success"

    def persist_terminal(
        result: bool,
        *,
        terminal_history_certified: bool,
    ) -> dict[str, Any]:
        if not result and state.get("tail_action") == "shutdown":
            # A failed/cancelled/lost old barrier is no longer waitable. Keep
            # the prior descriptor tainted but clear this dead tail so the
            # current dispatcher emits and owns a replacement RayKill child.
            recovered = dict(
                state,
                tail_prompt_id=None,
                tail_action=None,
                tainted=True,
            )
            recovered.pop("tail_terminal_certificate", None)
            database.put_raylight_runtime_state(comfy_origin, recovered)
            return recovered
        database.settle_raylight_runtime_prompt(
            comfy_origin,
            tail_prompt_id,
            succeeded=result,
            terminal_history_certified=terminal_history_certified,
        )
        return database.get_raylight_runtime_state(comfy_origin) or state

    def keep_ambiguous_tail() -> dict[str, Any]:
        guarded = dict(state, tainted=True)
        database.put_raylight_runtime_state(comfy_origin, guarded)
        return guarded

    history = await client.history(tail_prompt_id)
    entry, absent = _exact_history_result(history, tail_prompt_id)
    if entry is not None:
        result = terminal_result(entry)
        if result is not None:
            return persist_terminal(
                result,
                terminal_history_certified=True,
            )
    elif not absent:
        raise ComfyError(
            "RayLight runtime tail history response is malformed; refusing to reuse cached actors"
        )

    queue = await client.queue()
    if (
        not isinstance(queue, dict)
        or not isinstance(queue.get("queue_running"), list)
        or not isinstance(queue.get("queue_pending"), list)
    ):
        raise ComfyError(
            "RayLight runtime tail queue response is malformed; refusing to reuse cached actors"
        )
    if _contains_prompt(queue["queue_running"], tail_prompt_id) or _contains_prompt(
        queue["queue_pending"], tail_prompt_id
    ):
        # Older Director versions acknowledged POST /prompt by clearing this
        # flag before execution completed. Normalize a still-visible tail back
        # to the strict terminal-gated contract on first observation.
        if not state.get("tainted"):
            state = dict(state, tainted=True)
            database.put_raylight_runtime_state(comfy_origin, state)
        return state

    # The first history lookup preceded the queue snapshot. Close the normal
    # execution->history handoff race before declaring the cached state tainted.
    confirmation = await client.history(tail_prompt_id)
    confirmed_entry, confirmed_absent = _exact_history_result(
        confirmation, tail_prompt_id
    )
    if confirmed_entry is not None:
        result = terminal_result(confirmed_entry)
        if result is not None:
            return persist_terminal(
                result,
                terminal_history_certified=True,
            )
    elif not confirmed_absent:
        raise ComfyError(
            "RayLight runtime tail confirmation is malformed; refusing to reuse cached actors"
        )

    return (
        keep_ambiguous_tail()
        if ambiguous_tail
        else persist_terminal(
            False,
            terminal_history_certified=False,
        )
    )


def _endpoint_recovery_blockers(
    database: Database,
    comfy_origin: str,
    *,
    dispatch_job_id: str,
) -> list[dict[str, Any]]:
    """Return old ambiguous submissions that can still enqueue on an endpoint."""

    blockers: list[dict[str, Any]] = []
    for parent in database.list_interrupted_preparing_jobs():
        if str(parent["id"]) == dispatch_job_id:
            continue
        try:
            settings = RuntimeSettings.model_validate(parent["settings_snapshot"])
        except ValidationError as exc:
            raise ComfyError(
                "an interrupted submission has an invalid endpoint snapshot; "
                "refusing to enqueue newer work"
            ) from exc
        if Database.canonical_comfy_origin(settings.comfy_url) != comfy_origin:
            continue
        for child in database.list_job_children(str(parent["id"])):
            if (
                child["status"] not in _TERMINAL_STATUSES
                and child.get("prompt_id")
                and child.get("stage") in _RECOVERY_OWNERSHIP_STAGES
            ):
                blockers.append(child)
    return blockers


async def _await_endpoint_submission_recovery(
    request: Request,
    database: Database,
    comfy_origin: str,
    *,
    dispatch_job_id: str,
) -> None:
    """Keep newer Director prompts behind every ambiguous old POST.

    The startup recovery worker owns directed cancellation. A lost Standard
    ``POST /prompt`` can reach ``queue.put`` after a newer Ray submission just
    as a lost Ray prompt can; the Ray runtime ledger alone cannot represent
    that Standard tail. Waiting on the durable recovery marker under the same
    endpoint submission lock closes that cross-backend restart race.
    """

    while _endpoint_recovery_blockers(
        database, comfy_origin, dispatch_job_id=dispatch_job_id
    ):
        parent = database.get_job(dispatch_job_id)
        if parent is None:
            raise KeyError(dispatch_job_id)
        if parent["status"] in {"cancelling", "cancelled"}:
            raise asyncio.CancelledError
        await asyncio.sleep(_RAYLIGHT_GENERATION_POLL_SECONDS)


async def _cleanup_failed_timeline_submission(
    request: Request,
    *,
    job_id: str,
    client: ComfyClientProtocol,
    error: Exception,
    possibly_submitted: dict[str, str],
    inline_cancelled: set[str] | None = None,
) -> None:
    """Fail closed after a prompt submission may have reached ComfyUI.

    A caller-supplied prompt id is durable before ``POST /prompt``.  A
    transport/protocol error therefore cannot prove that ComfyUI rejected the
    side effect.  Every nonterminal bound child, plus the exact in-flight child
    even if a concurrent local cancellation already marked it terminal, must
    be targeted with the native atomic cancel endpoint.  Only an explicit
    ``True`` acknowledgement permits a local cancelled state; an exception or
    ``False`` remains ``cancelling`` for queue/history or restart recovery.
    """

    database = _db(request)
    inline_cancelled = inline_cancelled or set()
    durable_children = {
        str(child["id"]): child
        for child in database.list_job_children(job_id)
    }
    single_prompt = (
        next(iter(possibly_submitted.values()))
        if len(possibly_submitted) == 1
        else None
    )
    # ComfyClient validates this acknowledgement inside the same HTTP client
    # lifetime that observed the mismatched actual id. Repeating cancellation
    # would commonly return false because the prompt is already gone, turning
    # a confirmed cleanup back into an artificial recovery marker.
    for child_id in inline_cancelled:
        child = durable_children.get(child_id)
        if child is not None and child["status"] not in _TERMINAL_STATUSES:
            database.update_job_child_if_snapshot(
                child_id,
                expected_status=child["status"],
                expected_updated_at=child["updated_at"],
                status="cancelled",
                progress=1.0,
                stage="cancelled_after_submission_failure",
                error=None,
                completed_at=utc_now(),
            )
    durable_children = {
        str(child["id"]): child
        for child in database.list_job_children(job_id)
    }
    targets: dict[str, str] = {}
    for child_id, prompt_id in possibly_submitted.items():
        child = durable_children.get(child_id)
        if (
            child_id not in inline_cancelled
            and (child is None or child["status"] not in _TERMINAL_STATUSES)
        ):
            targets[child_id] = prompt_id
    for child_id, child in durable_children.items():
        if (
            child["status"] not in _TERMINAL_STATUSES
            and child.get("prompt_id")
        ):
            targets[child_id] = str(child["prompt_id"])

    uncertain: list[str] = []
    for child_id, prompt_id in targets.items():
        child = database.get_job_child(child_id)
        if child is not None and child["status"] not in _TERMINAL_STATUSES:
            before_cleanup_claim = getattr(
                request.app.state, "before_cleanup_cancel_claim", None
            )
            if before_cleanup_claim is not None:
                await before_cleanup_claim(job_id, child_id)
            claimed = database.update_job_child_if_snapshot(
                child_id,
                expected_status=child["status"],
                expected_updated_at=child["updated_at"],
                status="cancelling",
                stage="submission_cancel_pending",
                prompt_id=prompt_id,
                error=str(error),
                completed_at=None,
            )
            if claimed is not None:
                child = claimed
            before_cleanup_cancel = getattr(
                request.app.state, "before_cleanup_cancel_request", None
            )
            if before_cleanup_cancel is not None:
                await before_cleanup_cancel(job_id, child_id)
        try:
            confirmed = await client.cancel(prompt_id)
        except Exception as cancel_exc:
            uncertain.append(f"{child_id}: {cancel_exc}")
            child = database.get_job_child(child_id)
            if child is not None and child["status"] == "cancelling":
                database.update_job_child_if_snapshot(
                    child_id,
                    expected_status=child["status"],
                    expected_updated_at=child["updated_at"],
                    stage="submission_cancel_failed",
                    error=str(cancel_exc),
                    completed_at=None,
                )
            continue
        if not confirmed:
            uncertain.append(f"{child_id}: cancellation was not confirmed")
            child = database.get_job_child(child_id)
            if child is not None and child["status"] == "cancelling":
                database.update_job_child_if_snapshot(
                    child_id,
                    expected_status=child["status"],
                    expected_updated_at=child["updated_at"],
                    stage="submission_cancel_unconfirmed",
                    error=str(error),
                    completed_at=None,
                )
            continue
        child = database.get_job_child(child_id)
        if child is not None and child["status"] == "cancelling":
            database.update_job_child_if_snapshot(
                child_id,
                expected_status=child["status"],
                expected_updated_at=child["updated_at"],
                status="cancelled",
                progress=1.0,
                stage="cancelled_after_submission_failure",
                error=None,
                completed_at=utc_now(),
            )

    parent = database.get_job(job_id)
    if parent is None:
        return
    remaining = database.list_job_children(job_id)
    if uncertain:
        # Children without a prompt id provably never crossed the network and
        # can be closed immediately. Bound/possibly-bound children remain live.
        for child in remaining:
            if (
                child["status"] not in _TERMINAL_STATUSES
                and not child.get("prompt_id")
            ):
                database.update_job_child_if_snapshot(
                    child["id"],
                    expected_status=child["status"],
                    expected_updated_at=child["updated_at"],
                    status="cancelled",
                    progress=1.0,
                    stage="not_submitted",
                    error=None,
                    completed_at=utc_now(),
                )
        latest_parent = database.get_job(job_id)
        if latest_parent is None or latest_parent["status"] in _TERMINAL_STATUSES:
            return
        database.update_job_if_snapshot(
            job_id,
            expected_status=latest_parent["status"],
            expected_stage=latest_parent.get("stage"),
            expected_updated_at=latest_parent["updated_at"],
            status="cancelling",
            stage="submission_cancel_pending",
            prompt_id=single_prompt,
            error=(
                f"native submission failed: {error}; cleanup pending: "
                + "; ".join(uncertain)
            )[:20_000],
            completed_at=None,
        )
        return

    # Every possible upstream side effect was explicitly cancelled. Only now
    # may this request become terminal locally. Re-read and close the parent
    # plus all remaining children in one transaction so a user cancellation
    # that won while cleanup awaited ComfyUI remains the monotonic winner.
    before_cleanup_finalize = getattr(
        request.app.state, "before_cleanup_finalize", None
    )
    if before_cleanup_finalize is not None:
        await before_cleanup_finalize(job_id)
    failed_control = any(
        not child["segment_ids"] and child["status"] == "failed"
        for child in database.list_job_children(job_id)
    )
    database.finalize_timeline_submission_failure(
        job_id,
        error=str(error),
        prompt_id=single_prompt,
        failure_stage=(
            "raylight_switch_failed" if failed_control else "submission_failed"
        ),
    )


def _bind_actual_prompt_id_from_submit_error(
    database: Database,
    error: Exception,
    possibly_submitted: dict[str, str],
) -> set[str]:
    """Persist a mismatched Comfy prompt id before failure cleanup.

    Only ``ComfyError`` has the submit-client detail contract.  Even then, the
    reported requested id must identify exactly one in-flight child and still
    equal that child's durable prompt id.  This prevents unrelated exception
    detail from redirecting cancellation to an arbitrary upstream job.
    """

    if not isinstance(error, ComfyError) or not isinstance(error.detail, dict):
        return set()
    requested = error.detail.get("requested_prompt_id")
    actual = error.detail.get("actual_prompt_id")
    if not isinstance(requested, str) or not isinstance(actual, str):
        return set()
    if (
        not actual
        or actual == requested
        or actual != actual.strip()
        or len(actual) > 512
        or any(
            ord(character) < 0x20 or ord(character) == 0x7F
            for character in actual
        )
    ):
        return set()
    candidates = [
        child_id
        for child_id, durable_prompt_id in possibly_submitted.items()
        if durable_prompt_id == requested
    ]
    if len(candidates) != 1:
        return set()
    child_id = candidates[0]
    try:
        rebound = database.replace_job_child_prompt_id_if_current(
            child_id,
            expected_prompt_id=requested,
            prompt_id=actual,
        )
    except KeyError:
        return set()
    if rebound is not None:
        possibly_submitted[child_id] = actual
        cleanup_response = error.detail.get("cleanup_response")
        if (
            isinstance(cleanup_response, dict)
            and set(cleanup_response) == {"cancelled"}
            and cleanup_response.get("cancelled") is True
        ):
            return {child_id}
    return set()


def _resolve_historical_continuity_takes(
    database: Database,
    draft: UnifiedTimelineDraft,
    *,
    segment_ids: list[str] | None,
    comfy_origin: str,
    project_id: str | None = None,
) -> dict[str, NativeHistoricalTake]:
    """Resolve every omitted direct predecessor using server-owned state."""

    enabled = [segment for segment in draft.segments if segment.enabled]
    selected_ids = (
        set(segment_ids) if segment_ids is not None else {item.id for item in enabled}
    )
    predecessors = unified_continuity_predecessors(draft)
    resolved: dict[str, NativeHistoricalTake] = {}
    for segment in enabled:
        if segment.id not in selected_ids:
            continue
        predecessor = predecessors.get(segment.id)
        if predecessor is None or predecessor.id in selected_ids:
            continue
        fingerprint = timeline_segment_take_fingerprint(draft, predecessor)
        require_audio = segment.audio_mode == "generate"
        try:
            take = database.find_latest_segment_take(
                predecessor.id,
                fingerprint,
                comfy_origin=comfy_origin,
                require_audio=require_audio,
                project_id=project_id,
            )
        except (NativeTemplateError, TypeError, ValueError) as exc:
            raise DraftNotRunnable(
                f"片段 '{segment.id}' 的直接前驱 '{predecessor.id}' "
                "已有历史成片记录无效，无法用于接续"
            ) from exc
        if take is None:
            if require_audio and database.has_segment_take(
                predecessor.id,
                comfy_origin=comfy_origin,
                content_fingerprint=fingerprint,
                project_id=project_id,
            ):
                raise DraftNotRunnable(
                    f"片段 '{segment.id}' 的直接前驱 '{predecessor.id}' "
                    "有输出规格匹配的历史成功成片，但不含生成音频接续所需的音轨"
                )
            if database.has_segment_take(
                predecessor.id,
                comfy_origin=comfy_origin,
                project_id=project_id,
            ):
                raise DraftNotRunnable(
                    f"片段 '{segment.id}' 的直接前驱 '{predecessor.id}' "
                    "存在历史成功成片，但分辨率、帧率或可见帧数与当前分段不一致"
                )
            raise DraftNotRunnable(
                f"片段 '{segment.id}' 的直接前驱 '{predecessor.id}' "
                "在当前 ComfyUI 地址下没有可用的历史成功成片"
            )
        resolved[segment.id] = NativeHistoricalTake(
            id=str(take["id"]),
            segment_id=predecessor.id,
            output=dict(take["output"]),
        )
    return resolved


def _project_summary(project: dict[str, Any]) -> ProjectSummaryRead:
    """Project a project row into its public summary without leaking the document."""

    timeline = validate_timeline_draft(project["document"])
    return ProjectSummaryRead(
        id=project["id"],
        title=project["title"],
        created_at=project["created_at"],
        updated_at=project["updated_at"],
        segment_count=len(timeline.segments),
    )


async def _standard_lora_metadata_for_timeline(
    client: ComfyClientProtocol,
    draft: UnifiedTimelineDraft,
    settings: RuntimeSettings,
    segment_ids: list[str] | None,
) -> dict[ModelFamily, dict[str, str] | None]:
    """Ask the owning ComfyUI endpoint about unknown auto Standard LoRAs."""

    requests = standard_lora_metadata_requests(
        draft, settings, segment_ids=segment_ids
    )
    if not requests:
        return {}
    families = list(requests)
    values = await asyncio.gather(
        *(client.lora_metadata(requests[family]) for family in families)
    )
    return dict(zip(families, values, strict=True))


async def _compile_timeline_report(
    request: Request,
    body: TimelineJobRequest,
    *,
    project_id: str | None,
) -> TimelineCompileRead:
    """Shared compile/preflight path for the legacy and project-scoped routes."""

    database = _db(request)
    settings = database.get_settings()
    owner_project_id = project_id or database.LEGACY_DEFAULT_PROJECT_ID
    draft = body.config or database.get_project_timeline(owner_project_id)
    try:
        database.validate_timeline_assets(
            draft,
            comfy_origin=str(settings.comfy_url),
            segment_ids=body.segment_ids,
        )
        validate_unified_runnable(draft, segment_ids=body.segment_ids)
        historical_takes = _resolve_historical_continuity_takes(
            database,
            draft,
            segment_ids=body.segment_ids,
            comfy_origin=str(settings.comfy_url),
            project_id=owner_project_id,
        )
        client = _comfy(request, settings)
        standard_lora_metadata = await _standard_lora_metadata_for_timeline(
            client,
            draft,
            settings,
            body.segment_ids,
        )
        compiled = compile_native_timeline(
            draft,
            settings,
            f"preview-{uuid.uuid4()}",
            body.segment_ids,
            historical_takes=historical_takes,
            standard_lora_metadata=standard_lora_metadata,
        )
        # Compile reports are also the user's explicit preflight action.
        # Run the same side-effect-free registry, provenance, model and
        # logical-device checks used immediately before job submission;
        # never present a locally compilable plan as runtime-ready when
        # the selected ComfyUI endpoint cannot execute it.
        await _preflight_timeline(
            client, settings, compiled, database
        )
    except (NativeTemplateError, DraftNotRunnable, ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ComfyError, httpx.HTTPError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot inspect LoRA metadata from ComfyUI: {exc}",
        ) from exc
    return TimelineCompileRead(
        model_families=list(compiled.families),
        plans=list(compiled.plans),
        node_policy=compiled.node_policy,
    )


def _native_continuity_graph(
    workflows: tuple[NativeWorkflowUnit, ...],
) -> tuple[
    dict[str, NativeWorkflowUnit],
    dict[str, tuple[str, ...]],
]:
    """Validate and index the server-owned one-segment dependency graph."""

    units_by_segment: dict[str, NativeWorkflowUnit] = {}
    for unit in workflows:
        if len(unit.segment_ids) != 1:
            raise NativeTemplateError(
                f"native workflow '{unit.id}' must own exactly one segment"
            )
        segment_id = unit.segment_ids[0]
        if segment_id in units_by_segment:
            raise NativeTemplateError(
                f"native continuity segment '{segment_id}' has multiple workflows"
            )
        output_node_id = unit.output_nodes.get(segment_id)
        output_node = (
            unit.prompt.get(output_node_id)
            if isinstance(output_node_id, str)
            else None
        )
        if (
            set(unit.output_nodes) != {segment_id}
            or not isinstance(output_node, dict)
            or output_node.get("class_type") != "SaveVideo"
        ):
            raise NativeTemplateError(
                f"native workflow '{unit.id}' must declare its unique SaveVideo "
                f"output for segment '{segment_id}'"
            )
        units_by_segment[segment_id] = unit

    mutable_dependents: dict[str, list[str]] = {}
    predecessor_by_segment: dict[str, str] = {}
    for segment_id, unit in units_by_segment.items():
        dependency = unit.continuity
        if dependency is None:
            continue
        predecessor_id = dependency.predecessor_segment_id
        if dependency.source == "historical_take":
            if (
                not dependency.resolved
                or not dependency.historical_take_id
                or predecessor_id in units_by_segment
            ):
                raise NativeTemplateError(
                    f"continuity segment '{segment_id}' has an invalid "
                    "historical-take dependency"
                )
            continue
        if dependency.source != "same_run":
            raise NativeTemplateError(
                f"continuity segment '{segment_id}' has an unknown dependency source"
            )
        if predecessor_id not in units_by_segment:
            raise NativeTemplateError(
                f"continuity segment '{segment_id}' requires unselected predecessor "
                f"'{predecessor_id}'"
            )
        if dependency.resolved or dependency.historical_take_id is not None:
            raise NativeTemplateError(
                f"continuity segment '{segment_id}' has an invalid same-run dependency"
            )
        if predecessor_id == segment_id:
            raise NativeTemplateError(
                f"continuity segment '{segment_id}' cannot depend on itself"
            )
        predecessor_by_segment[segment_id] = predecessor_id
        mutable_dependents.setdefault(predecessor_id, []).append(segment_id)

    # The current compiler emits a chain, but validate the generic graph here
    # before persistence so a future template change cannot create a dispatcher
    # deadlock or a prompt that is impossible to bind.
    for segment_id in predecessor_by_segment:
        seen: set[str] = set()
        cursor = segment_id
        while cursor in predecessor_by_segment:
            if cursor in seen:
                raise NativeTemplateError(
                    f"native continuity dependency cycle includes '{cursor}'"
                )
            seen.add(cursor)
            cursor = predecessor_by_segment[cursor]

    return units_by_segment, {
        predecessor_id: tuple(successor_ids)
        for predecessor_id, successor_ids in mutable_dependents.items()
    }


def _continuity_output_descriptor(
    child: dict[str, Any], predecessor_segment_id: str
) -> dict[str, str]:
    """Resolve exactly one persistent SaveVideo result for a predecessor."""

    output_nodes = child.get("output_nodes")
    output_node_id = (
        output_nodes.get(predecessor_segment_id)
        if isinstance(output_nodes, dict)
        else None
    )
    if not isinstance(output_node_id, str) or not output_node_id:
        raise NativeTemplateError(
            f"continuity predecessor '{predecessor_segment_id}' has no declared "
            "SaveVideo output node"
        )
    candidates = [
        output
        for output in child.get("outputs") or []
        if isinstance(output, dict)
        and str(output.get("node_id") or "") == output_node_id
        and output.get("type") == "output"
        and isinstance(output.get("filename"), str)
        and bool(output["filename"])
    ]
    if len(candidates) != 1:
        raise NativeTemplateError(
            f"continuity predecessor '{predecessor_segment_id}' must publish exactly "
            f"one persistent SaveVideo output from node '{output_node_id}'; "
            f"received {len(candidates)}"
        )
    output = candidates[0]
    return {
        "filename": str(output["filename"]),
        "subfolder": str(output.get("subfolder") or ""),
        "type": "output",
    }


def _fail_continuity_descendants(
    database: Database,
    *,
    job_id: str,
    child_ids_by_segment: dict[str, str],
    dependents: dict[str, tuple[str, ...]],
    predecessor_segment_id: str,
    reason: str,
) -> set[str]:
    """Fail every provably unsubmitted transitive successor in one chain."""

    failed_segments: set[str] = set()
    pending = list(dependents.get(predecessor_segment_id, ()))
    while pending:
        segment_id = pending.pop(0)
        if segment_id in failed_segments:
            continue
        failed_segments.add(segment_id)
        pending.extend(dependents.get(segment_id, ()))
        child_id = child_ids_by_segment.get(segment_id)
        if child_id is None:
            continue
        child = database.get_job_child(child_id)
        if (
            child is None
            or child["status"] != "preparing"
            or child.get("prompt_id") is not None
        ):
            continue
        database.fail_job_child_dependency_if_dispatching(
            job_id,
            child_id,
            expected_updated_at=child["updated_at"],
            error=(
                f"continuity dependency failed after predecessor "
                f"'{predecessor_segment_id}': {reason}"
            )[:20_000],
        )
    return failed_segments


async def _create_timeline_job_impl(
    request: Request,
    body: TimelineJobRequest,
    *,
    parent_mode: str = "timeline",
    job_id: str | None = None,
    accepted: asyncio.Event | None = None,
    accepted_release: asyncio.Event | None = None,
    project_id: str | None = None,
) -> JobRead:
    database = _db(request)
    settings = database.get_settings()
    client = _comfy(request, settings)
    owner_project_id = project_id or database.LEGACY_DEFAULT_PROJECT_ID
    draft = body.config or database.get_project_timeline(owner_project_id)
    try:
        database.validate_timeline_assets(
            draft,
            comfy_origin=str(settings.comfy_url),
            segment_ids=body.segment_ids,
        )
    except (ValidationError, ValueError) as exc:
        raise _validation_error(exc) from exc
    job_id = job_id or str(uuid.uuid4())
    try:
        validate_unified_runnable(draft, segment_ids=body.segment_ids)
        historical_takes = _resolve_historical_continuity_takes(
            database,
            draft,
            segment_ids=body.segment_ids,
            comfy_origin=str(settings.comfy_url),
            project_id=owner_project_id,
        )
        standard_lora_metadata = await _standard_lora_metadata_for_timeline(
            client,
            draft,
            settings,
            body.segment_ids,
        )
        # The browser never submits a workflow. This is the only compilation
        # boundary and returns family/backend workflow units held server-side.
        compiled = compile_native_timeline(
            draft,
            settings,
            job_id,
            body.segment_ids,
            historical_takes=historical_takes,
            standard_lora_metadata=standard_lora_metadata,
        )
        _, continuity_dependents = _native_continuity_graph(
            compiled.workflows
        )
    except (NativeTemplateError, DraftNotRunnable, ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ComfyError, httpx.HTTPError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot inspect LoRA metadata from ComfyUI: {exc}",
        ) from exc
    database = _db(request)
    now = utc_now()
    database.create_job(
        {
            "id": job_id,
            "mode": parent_mode,
            "status": "preparing",
            "progress": 0.0,
            "stage": "preflight",
            # A native timeline has multiple prompt IDs; never overload this
            # legacy column with a made-up representative child.
            "prompt_id": None,
            # Timeline jobs carry their owning project. Legacy six-mode
            # submissions stay project-less and appear under "旧任务".
            "project_id": owner_project_id if parent_mode == "timeline" else None,
            "outputs": [],
            "error": None,
            "config_snapshot": {
                "timeline": draft.model_dump(mode="json"),
                "segment_ids": body.segment_ids,
            },
            "settings_snapshot": settings.model_dump(mode="json"),
            "prompt_snapshot": compiled.manifest,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )
    child_ids: dict[str, str] = {}
    child_ids_by_segment: dict[str, str] = {}
    for index, unit in enumerate(compiled.workflows):
        child_id = str(uuid.uuid4())
        child_ids[unit.id] = child_id
        child_ids_by_segment[unit.segment_ids[0]] = child_id
        database.create_job_child(
            {
                "id": child_id,
                "job_id": job_id,
                # Leave the preceding even slot available for a dynamically
                # planned RayKill barrier once the endpoint queue-tail state is
                # read under its submission lock.
                "group_index": index * 2 + 1,
                "family": unit.family,
                "backend": unit.backend,
                "segment_ids": list(unit.segment_ids),
                "output_nodes": dict(unit.output_nodes),
                "status": "preparing",
                "progress": 0.0,
                "stage": "preflight",
                "prompt_id": None,
                "outputs": [],
                "error": None,
                "prompt_snapshot": unit.prompt,
                "created_at": now,
                "updated_at": now,
                "started_at": None,
                "completed_at": None,
            }
        )
    submitted_children: list[tuple[str, str]] = []
    # Insert before awaiting POST /prompt: a thrown response can still follow
    # an accepted upstream side effect.
    possibly_submitted: dict[str, str] = {}
    submission_lock: anyio.Lock | None = None
    lock_acquired = False
    endpoint_key: str | None = None
    submission_ticket: asyncio.Future[None] | None = None
    predecessor: asyncio.Future[None] | None = None
    transition_unit_ids: set[str] = set()
    ray_state_before_submit: dict[str, dict[str, Any]] = {}
    continuity_outputs: dict[str, dict[str, str]] = {}
    dependency_failed_segments: set[str] = set()
    try:
        await _preflight_timeline(client, settings, compiled, database)
        gate = database.update_job_if_status(
            job_id, "preparing", stage="submitting"
        )
        if gate is None:
            latest = database.get_job(job_id)
            if latest is None:
                raise HTTPException(status_code=404, detail="job disappeared during submission")
            latest["children"] = database.list_job_children(job_id)
            return _job_read_for_request(request, latest)
        endpoint_key = Database.canonical_comfy_origin(settings.comfy_url)
        loop = asyncio.get_running_loop()
        submission_ticket = loop.create_future()
        async with request.app.state.submission_ticket_lock:
            predecessor = request.app.state.submission_tails.get(endpoint_key)
            request.app.state.submission_tails[endpoint_key] = submission_ticket
        # Register endpoint order before acknowledging a Ray request. Thus a
        # second HTTP request can return quickly without racing its background
        # task ahead of the first one at the process-local AnyIO lock.
        if accepted is not None:
            accepted.set()
            if accepted_release is not None:
                await accepted_release.wait()
        if predecessor is not None:
            await asyncio.shield(predecessor)
        # Subscribe before POST /prompt and briefly await the initial socket
        # handshake so a newly configured endpoint does not lose its first
        # fast node event. The wait is bounded; queue/history remain the
        # lifecycle fallback if the optional websocket is unavailable.
        if isinstance(client, ComfyClient):
            await request.app.state.progress_manager.ensure_ready(
                str(settings.comfy_url),
                settings.client_id,
                timeout_seconds=1.0,
            )
        submission_lock = request.app.state.submission_locks.setdefault(
            endpoint_key, anyio.Lock()
        )
        await submission_lock.acquire()
        lock_acquired = True
        await _await_endpoint_submission_recovery(
            request,
            database,
            endpoint_key,
            dispatch_job_id=job_id,
        )
        # This is a conservative queue-tail ledger, not a claim that CUDA
        # weights are already physically resident. State is advanced before a
        # Ray POST because a lost response may still have accepted the prompt;
        # cancellation/failure therefore leaves a conservative key that forces
        # a later incompatible request through another barrier. Epoch never
        # resets, including across Standard, so A -> B -> A cannot hit A's old
        # cached initializer output containing actor handles killed by B.
        runtime_state = database.get_raylight_runtime_state(endpoint_key) or {
            "version": 2,
            "epoch": 0,
            "current": None,
            "tail_prompt_id": None,
            "tail_action": None,
            "tainted": False,
        }
        # The initial timeline preflight can precede an endpoint-ticket wait.
        # Re-read visibility while holding the submission lock and before even
        # refreshing the old queue tail, so a ComfyUI restart/GPU change in
        # that interval cannot mutate or replay an incompatible ledger.
        try:
            locked_runtime_status = _raylight_runtime_status(
                database,
                endpoint_key,
                await client.system_stats(),
                state=runtime_state,
            )
        except NativeTemplateError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "raylight_runtime_state_invalid",
                    "message": str(exc),
                },
            ) from exc
        if locked_runtime_status.recovery_required:
            raise HTTPException(
                status_code=409,
                detail=_raylight_runtime_recovery_detail(locked_runtime_status),
            )
        runtime_state = await _refresh_raylight_runtime_tail(
            client, database, endpoint_key, runtime_state
        )
        if runtime_state.get("legacy_unknown") and any(
            unit.backend == "standard" for unit in compiled.workflows
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "检测到旧版 RayLight 运行状态，无法证明旧 GPU actor 已释放；"
                    "为避免与 Standard 争用显存，本任务已阻止。请先提交一次 RayLight "
                    "任务以显式重建运行池，再提交 Standard 任务。"
                ),
            )
        epoch = int(runtime_state["epoch"])
        current_descriptor = runtime_state.get("current")
        current_tainted = bool(runtime_state.get("tainted"))
        tail_prompt_id = runtime_state.get("tail_prompt_id")
        tail_action = runtime_state.get("tail_action")
        if (
            isinstance(tail_prompt_id, str)
            and tail_action == "ray_unit"
            and current_tainted
        ):
            # A process restart can leave a durable Ray prompt executing after
            # its submission coroutine disappeared. Never append behind that
            # uncertified generation. If its child cannot be resolved, retain
            # a tainted descriptor so the next unit first queues RayKill.
            tail_children = database.find_any_job_children_by_prompt_id(
                tail_prompt_id
            )
            if len(tail_children) == 1:
                tail_child = tail_children[0]
                terminal = await _await_raylight_generation(
                    client,
                    database,
                    str(tail_child["job_id"]),
                    str(tail_child["id"]),
                    tail_prompt_id,
                    stop_on_parent_cancel=False,
                    dispatch_job_id=job_id,
                    terminal_events=request.app.state.prompt_terminal_events,
                )
                succeeded = terminal["status"] == "succeeded"
                database.settle_raylight_runtime_prompt(
                    endpoint_key,
                    tail_prompt_id,
                    succeeded=succeeded,
                    terminal_history_certified=(
                        _raylight_child_has_terminal_history_certificate(terminal)
                    ),
                )
                current_tainted = not succeeded
            else:
                current_tainted = True
                runtime_state = dict(runtime_state, tainted=True)
                database.put_raylight_runtime_state(endpoint_key, runtime_state)
        elif isinstance(tail_prompt_id, str) and tail_action == "shutdown":
            # A durable queued barrier is itself the serialization point. Wait
            # for positive exact history before treating the previous pool as
            # gone, even when its control child belongs to a restarted task.
            tail_children = database.find_any_job_children_by_prompt_id(
                tail_prompt_id
            )
            tail_child = tail_children[0] if len(tail_children) == 1 else None
            while (
                tail_child is not None
                and tail_child["status"] not in _TERMINAL_STATUSES
                and tail_child.get("stage") in _PROCESS_OWNERSHIP_STAGES
            ):
                # A lost RayKill POST response can be followed by delayed
                # queue.put. Its recovery owner must first provide an exact
                # terminal/cancel certificate; otherwise a newer barrier could
                # run first and the delayed old RayKill would kill the new pool.
                dispatch_parent = database.get_job(job_id)
                if dispatch_parent is None:
                    raise KeyError(job_id)
                if dispatch_parent["status"] in {"cancelling", "cancelled"}:
                    raise asyncio.CancelledError
                await asyncio.sleep(_RAYLIGHT_GENERATION_POLL_SECONDS)
                tail_child = database.get_job_child(str(tail_child["id"]))
                if tail_child is None:
                    raise KeyError(tail_prompt_id)
            if tail_child is not None and tail_child["status"] != "succeeded":
                current_tainted = True
            else:
                await _await_raylight_transition(client, tail_prompt_id)
                database.settle_raylight_runtime_prompt(
                    endpoint_key,
                    tail_prompt_id,
                    succeeded=True,
                    terminal_history_certified=True,
                )
                current_descriptor = None
                current_tainted = False

        planned_units: list[NativeWorkflowUnit] = []
        compiled_positions = {
            unit.id: index for index, unit in enumerate(compiled.workflows)
        }
        barrier_serial = 0

        def persist_planned_manifest() -> None:
            planned_manifest = dict(compiled.manifest)
            planned_manifest["submission_order"] = [
                unit.id for unit in planned_units
            ]
            planned_manifest["runtime_epoch"] = epoch
            planned_manifest["runtime_transitions"] = [
                unit.id for unit in planned_units
                if unit.id in transition_unit_ids
            ]
            planned_manifest["units"] = [
                {
                    "id": unit.id,
                    "family": unit.family,
                    "backend": unit.backend,
                    "segment_ids": list(unit.segment_ids),
                    "output_nodes": dict(unit.output_nodes),
                    "continuity": (
                        {
                            "predecessor_segment_id": (
                                unit.continuity.predecessor_segment_id
                            ),
                            "overlap_frames": unit.continuity.overlap_frames,
                            "load_video_node_id": (
                                unit.continuity.load_video_node_id
                            ),
                            "source": unit.continuity.source,
                            "historical_take_id": (
                                unit.continuity.historical_take_id
                            ),
                            "resolved": unit.continuity.resolved,
                        }
                        if unit.continuity is not None
                        else None
                    ),
                    "runtime_namespace": (
                        None
                        if unit.id in transition_unit_ids
                        else descriptor["runtime_namespace"]
                        if (descriptor := raylight_runtime_descriptor(unit)) is not None
                        else None
                    ),
                }
                for unit in planned_units
            ]
            database.update_job(job_id, prompt_snapshot=planned_manifest)

        def dynamically_planned_units():
            """Yield only the next safe unit; resume after its terminal gate."""

            nonlocal barrier_serial, current_descriptor, current_tainted, epoch
            for original_unit in compiled.workflows:
                segment_id = original_unit.segment_ids[0]
                if segment_id in dependency_failed_segments:
                    continue
                dependency = original_unit.continuity
                if (
                    dependency is not None
                    and dependency.source == "same_run"
                ):
                    predecessor_output = continuity_outputs.get(
                        dependency.predecessor_segment_id
                    )
                    if predecessor_output is None:
                        newly_failed = _fail_continuity_descendants(
                            database,
                            job_id=job_id,
                            child_ids_by_segment=child_ids_by_segment,
                            dependents=continuity_dependents,
                            predecessor_segment_id=(
                                dependency.predecessor_segment_id
                            ),
                            reason="predecessor did not produce a certified output",
                        )
                        dependency_failed_segments.update(newly_failed)
                        continue
                    try:
                        original_unit = bind_native_workflow_predecessor_output(
                            original_unit, predecessor_output
                        )
                    except NativeTemplateError as exc:
                        newly_failed = _fail_continuity_descendants(
                            database,
                            job_id=job_id,
                            child_ids_by_segment=child_ids_by_segment,
                            dependents=continuity_dependents,
                            predecessor_segment_id=(
                                dependency.predecessor_segment_id
                            ),
                            reason=str(exc),
                        )
                        dependency_failed_segments.update(newly_failed)
                        continue
                elif dependency is not None and (
                    dependency.source != "historical_take"
                    or not dependency.resolved
                    or not dependency.historical_take_id
                ):
                    raise NativeTemplateError(
                        f"continuity segment '{segment_id}' has an unresolved "
                        "historical take"
                    )
                target_descriptor = raylight_runtime_descriptor(original_unit)
                target_key = (
                    str(target_descriptor["runtime_key"])
                    if target_descriptor is not None
                    else None
                )
                current_key = (
                    str(current_descriptor.get("runtime_key") or "")
                    if isinstance(current_descriptor, dict)
                    else None
                )
                incompatible = current_descriptor is not None and (
                    current_tainted
                    or original_unit.backend == "standard"
                    or current_key != target_key
                )
                if incompatible:
                    barrier_serial += 1
                    transition_unit = build_raylight_shutdown_unit(
                        current_descriptor,
                        unit_id=(
                            f"switch-{compiled_positions[original_unit.id]:03d}-"
                            f"{barrier_serial}-{job_id}"
                        ),
                    )
                    transition_unit_ids.add(transition_unit.id)
                    child_id = str(uuid.uuid4())
                    child_ids[transition_unit.id] = child_id
                    transition_now = utc_now()
                    database.create_job_child(
                        {
                            "id": child_id,
                            "job_id": job_id,
                            "group_index": compiled_positions[original_unit.id] * 2,
                            "family": transition_unit.family,
                            "backend": transition_unit.backend,
                            "segment_ids": [],
                            "output_nodes": {},
                            "status": "preparing",
                            "progress": 0.0,
                            "stage": "RayLight 安全切换",
                            "prompt_id": None,
                            "outputs": [],
                            "error": None,
                            "prompt_snapshot": transition_unit.prompt,
                            "created_at": transition_now,
                            "updated_at": transition_now,
                            "started_at": None,
                            "completed_at": None,
                        }
                    )
                    parent_after_create = database.get_job(job_id)
                    if (
                        parent_after_create is None
                        or parent_after_create["status"] != "preparing"
                    ):
                        database.update_job_child_if_status(
                            child_id,
                            "preparing",
                            status="cancelled",
                            progress=1.0,
                            stage="not_submitted",
                            error=None,
                            completed_at=utc_now(),
                        )
                        return
                    ray_state_before_submit[transition_unit.id] = {
                        "version": 2,
                        "epoch": epoch,
                        "current": current_descriptor,
                        "tail_prompt_id": child_id,
                        "tail_action": "shutdown",
                        "tainted": True,
                    }
                    planned_units.append(transition_unit)
                    persist_planned_manifest()
                    yield transition_unit
                    # The generator resumes only after the outer loop has
                    # positively certified RayKill history.
                    current_descriptor = None
                    current_tainted = False
                if original_unit.backend == "raylight":
                    if current_descriptor is None:
                        epoch += 1
                    bound_unit = bind_raylight_runtime_epoch(original_unit, epoch)
                    validate_native_workflow_ready(bound_unit)
                    current_descriptor = raylight_runtime_descriptor(bound_unit)
                    if current_descriptor is None:
                        raise NativeTemplateError(
                            f"RayLight unit '{original_unit.id}' has no runtime descriptor"
                        )
                    ray_state_before_submit[bound_unit.id] = {
                        "version": 2,
                        "epoch": epoch,
                        "current": current_descriptor,
                        "tail_prompt_id": child_ids[bound_unit.id],
                        "tail_action": "ray_unit",
                        "tainted": True,
                    }
                    planned_units.append(bound_unit)
                    persist_planned_manifest()
                    yield bound_unit
                else:
                    validate_native_workflow_ready(original_unit)
                    planned_units.append(original_unit)
                    persist_planned_manifest()
                    yield original_unit

        workflows_to_submit = dynamically_planned_units()
        # ComfyUI's normal queue serializes these one-segment prompts. Stable
        # loader ids/inputs permit endpoint-local cache reuse without putting
        # 128 independent sampling/decode branches in one failure domain.
        for unit in workflows_to_submit:
            segment_id = unit.segment_ids[0] if unit.segment_ids else None
            if (
                segment_id is not None
                and segment_id in dependency_failed_segments
            ):
                continue
            if unit.id in transition_unit_ids:
                await _preflight_raylight_transition(
                    client, unit, database, endpoint_key
                )
            current = database.get_job(job_id)
            if current is None:
                raise HTTPException(
                    status_code=404, detail="job disappeared during submission"
                )
            if current["status"] != "preparing":
                # Cancellation may arrive between family submissions. The
                # parent cancellation path owns all already-bound prompts and
                # marks every not-yet-submitted child terminal.
                submission_lock.release()
                lock_acquired = False
                return _job_read_for_request(
                    request, await _cancel_timeline_job(request, current)
                )
            child_id = child_ids[unit.id]
            planned_ray_state = ray_state_before_submit.get(unit.id)
            validate_native_workflow_ready(unit)
            if planned_ray_state is not None:
                database.put_raylight_runtime_state(
                    endpoint_key, planned_ray_state
                )
            before_claim = getattr(
                request.app.state, "before_submission_claim", None
            )
            if before_claim is not None:
                await before_claim(job_id, child_id)
            claimed_child = database.claim_job_child_submission(
                job_id,
                child_id,
                prompt_id=child_id,
                # Replace the compile-time base namespace with the exact
                # persistent epoch that was chosen under the endpoint lock.
                prompt_snapshot=unit.prompt,
            )
            if claimed_child is None:
                latest_parent = database.get_job(job_id)
                if latest_parent is None:
                    raise HTTPException(
                        status_code=404, detail="job disappeared before submission"
                    )
                if (
                    latest_parent["status"] in {"cancelling", "cancelled"}
                    or bool(latest_parent.get("cancel_requested"))
                ):
                    # ``mark_job_cancel_requested`` deliberately persists the
                    # operator's intent before its pre-cancel history probe.
                    # The atomic child claim also checks that bit, so losing
                    # the claim in this narrow preparing-state window belongs
                    # to cancellation—not to preflight failure.
                    submission_lock.release()
                    lock_acquired = False
                    return _job_read_for_request(
                        request, await _cancel_timeline_job(request, latest_parent)
                    )
                raise HTTPException(
                    status_code=409,
                    detail="job child changed state before submission",
                )
            possibly_submitted[child_id] = child_id
            submitted = await client.submit(
                unit.prompt, settings.client_id, prompt_id=child_id
            )
            prompt_id = str(submitted["prompt_id"])
            submitted_children.append((child_id, prompt_id))
            bound = database.update_job_child_if_status(
                child_id,
                "preparing",
                status="queued",
                progress=0.0,
                stage="queued",
                prompt_id=prompt_id,
            )
            if bound is None:
                # ComfyUI may begin a caller-ID prompt and emit ``executing``
                # before POST /prompt returns. The websocket sink can already
                # have advanced this exact child to running. Treat only the
                # returned, durably matching prompt id as successful early
                # execution; every other lost preparing CAS still belongs to
                # the cancellation/error path below.
                latest_child = database.get_job_child(child_id)
                if (
                    latest_child is not None
                    and latest_child["status"] in {
                        "running",
                        "succeeded",
                        "failed",
                    }
                    and latest_child.get("prompt_id") == prompt_id
                ):
                    bound = latest_child
            if bound is None:
                # Cancellation won while POST /prompt was in flight.  The
                # returned caller id may have become visible to websocket and
                # cancellation before the HTTP response arrived. Claim only
                # current nonterminal snapshots; a confirmed cancellation (or
                # any other terminal result) must never be revived merely to
                # record this late response.
                latest_child = database.get_job_child(child_id)
                if latest_child is None:
                    raise HTTPException(
                        status_code=404, detail="job child disappeared after submission"
                    )
                if latest_child.get("prompt_id") != prompt_id:
                    raise HTTPException(
                        status_code=409,
                        detail="job child prompt changed during submission",
                    )
                if latest_child["status"] not in _TERMINAL_STATUSES:
                    claimed_cancel = database.update_job_child_if_snapshot(
                        child_id,
                        expected_status=latest_child["status"],
                        expected_updated_at=latest_child["updated_at"],
                        status="cancelling",
                        stage="cancelling_after_submit",
                        prompt_id=prompt_id,
                        completed_at=None,
                    )
                    latest_child = claimed_cancel or database.get_job_child(child_id)
                    if latest_child is None:
                        raise HTTPException(
                            status_code=404,
                            detail="job child disappeared after submission",
                        )
                latest_parent = database.get_job(job_id)
                if latest_parent is None:
                    raise HTTPException(
                        status_code=404, detail="job disappeared after submission"
                    )
                if (
                    latest_parent["status"] not in _TERMINAL_STATUSES
                    and latest_parent["status"] != "cancelling"
                ):
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "job changed state unexpectedly during submission: "
                            f"{latest_parent['status']}"
                        ),
                    )
                if latest_parent["status"] not in _TERMINAL_STATUSES:
                    claimed_parent = database.update_job_if_snapshot(
                        job_id,
                        expected_status=latest_parent["status"],
                        expected_stage=latest_parent.get("stage"),
                        expected_updated_at=latest_parent["updated_at"],
                        status="cancelling",
                        stage="cancelling_after_submit",
                        prompt_id=(
                            prompt_id
                            if len(compiled.workflows) == 1
                            and not transition_unit_ids
                            else None
                        ),
                        completed_at=None,
                    )
                    latest_parent = claimed_parent or database.get_job(job_id)
                    if latest_parent is None:
                        raise HTTPException(
                            status_code=404,
                            detail="job disappeared after submission",
                        )
                try:
                    dispatched = await client.cancel(prompt_id)
                except (ComfyError, httpx.HTTPError) as exc:
                    latest_child = database.get_job_child(child_id)
                    latest_parent = database.get_job(job_id)
                    if latest_child is None or latest_parent is None:
                        raise HTTPException(
                            status_code=404,
                            detail="job disappeared after submission cancellation",
                        ) from exc
                    if (
                        latest_child["status"] in _TERMINAL_STATUSES
                        or latest_parent["status"] in _TERMINAL_STATUSES
                    ):
                        # A previous exact directed cancel already published a
                        # durable terminal result. A redundant cleanup retry
                        # failing after the late POST response is not authority
                        # to reopen either lifecycle row.
                        submission_lock.release()
                        lock_acquired = False
                        return _job_read_for_request(
                            request,
                            await _sync_timeline_job(request, latest_parent),
                        )
                    database.update_job_child_if_snapshot(
                        child_id,
                        expected_status=latest_child["status"],
                        expected_updated_at=latest_child["updated_at"],
                        status="cancelling",
                        stage="cancel_failed",
                        error=str(exc),
                    )
                    database.update_job_if_snapshot(
                        job_id,
                        expected_status=latest_parent["status"],
                        expected_stage=latest_parent.get("stage"),
                        expected_updated_at=latest_parent["updated_at"],
                        status="cancelling",
                        stage="cancel_failed",
                        error=f"upstream cancel after submit failed: {exc}",
                    )
                    raise HTTPException(status_code=502, detail=str(exc)) from exc
                latest_parent = database.get_job(job_id)
                if latest_parent is None:
                    raise HTTPException(
                        status_code=404, detail="job disappeared after submission"
                    )
                if not dispatched:
                    reconciled = await _sync_timeline_job(request, latest_parent)
                    if reconciled["status"] in {"succeeded", "failed"}:
                        submission_lock.release()
                        lock_acquired = False
                        return _job_read_for_request(request, reconciled)
                else:
                    latest_child = database.get_job_child(child_id)
                    if (
                        latest_child is not None
                        and latest_child["status"] not in _TERMINAL_STATUSES
                    ):
                        database.update_job_child_if_snapshot(
                            child_id,
                            expected_status=latest_child["status"],
                            expected_updated_at=latest_child["updated_at"],
                            status="cancelled",
                            progress=1.0,
                            stage="cancelled",
                            error=None,
                            completed_at=utc_now(),
                        )
                cancelled_parent = database.get_job(job_id)
                if cancelled_parent is None:
                    raise HTTPException(
                        status_code=404, detail="job disappeared after cancellation"
                    )
                submission_lock.release()
                lock_acquired = False
                return _job_read_for_request(
                    request,
                    await _sync_timeline_job(request, cancelled_parent),
                )
            if unit.id in transition_unit_ids:
                # Queue order alone is insufficient: ComfyUI continues with
                # later prompts after a failed output node. Require positive
                # history evidence from RayKill before submitting Standard.
                transitioned = await _await_raylight_transition(
                    client,
                    prompt_id,
                    database=database,
                    job_id=job_id,
                    child_id=child_id,
                )
                if not transitioned:
                    database.settle_raylight_runtime_prompt(
                        endpoint_key, prompt_id, succeeded=False
                    )
                    latest_parent = database.get_job(job_id)
                    if latest_parent is None:
                        raise HTTPException(
                            status_code=404,
                            detail="job disappeared during RayLight barrier",
                        )
                    submission_lock.release()
                    lock_acquired = False
                    return _job_read_for_request(
                        request, await _cancel_timeline_job(request, latest_parent)
                    )
                for expected_status in ("queued", "running"):
                    if database.update_job_child_if_status(
                        child_id,
                        expected_status,
                        status="succeeded",
                        progress=1.0,
                        stage="RayLight 已安全释放",
                        outputs=[],
                        error=None,
                        completed_at=utc_now(),
                    ) is not None:
                        break
                database.settle_raylight_runtime_prompt(
                    endpoint_key,
                    prompt_id,
                    succeeded=True,
                    terminal_history_certified=True,
                )
            elif planned_ray_state is not None:
                # Ray actors are mutable and failures are not covered by a
                # method-level finally in the installed sampler. Do not even
                # enqueue the next Director Ray generation until exact history
                # certifies this one. A failed or externally removed prompt
                # leaves the descriptor tainted, causing the generator to emit
                # RayKill + a fresh epoch before its next segment.
                terminal = await _await_raylight_generation(
                    client,
                    database,
                    job_id,
                    child_id,
                    prompt_id,
                    terminal_events=request.app.state.prompt_terminal_events,
                )
                succeeded = terminal["status"] == "succeeded"
                database.settle_raylight_runtime_prompt(
                    endpoint_key,
                    prompt_id,
                    succeeded=succeeded,
                    terminal_history_certified=(
                        _raylight_child_has_terminal_history_certificate(terminal)
                    ),
                )
                current_tainted = not succeeded
                latest_parent = database.get_job(job_id)
                if latest_parent is None:
                    raise HTTPException(
                        status_code=404, detail="job disappeared during RayLight gate"
                    )
                if latest_parent["status"] in {"cancelling", "cancelled"}:
                    submission_lock.release()
                    lock_acquired = False
                    return _job_read_for_request(
                        request, await _cancel_timeline_job(request, latest_parent)
                    )
            if segment_id is not None and segment_id in continuity_dependents:
                terminal_child = database.get_job_child(child_id)
                if terminal_child is None:
                    raise HTTPException(
                        status_code=404,
                        detail="job child disappeared during continuity gate",
                    )
                if terminal_child["status"] not in _TERMINAL_STATUSES:
                    terminal_child = await _await_timeline_generation(
                        client,
                        database,
                        job_id,
                        child_id,
                        prompt_id,
                        running_stage="sampling",
                        error_context="continuity predecessor generation",
                        terminal_events=request.app.state.prompt_terminal_events,
                    )
                latest_parent = database.get_job(job_id)
                if latest_parent is None:
                    raise HTTPException(
                        status_code=404,
                        detail="job disappeared during continuity gate",
                    )
                if latest_parent["status"] in {"cancelling", "cancelled"}:
                    submission_lock.release()
                    lock_acquired = False
                    return _job_read_for_request(
                        request,
                        await _cancel_timeline_job(request, latest_parent),
                    )
                if terminal_child["status"] == "succeeded":
                    try:
                        continuity_outputs[segment_id] = (
                            _continuity_output_descriptor(
                                terminal_child, segment_id
                            )
                        )
                    except NativeTemplateError as exc:
                        dependency_failed_segments.update(
                            _fail_continuity_descendants(
                                database,
                                job_id=job_id,
                                child_ids_by_segment=child_ids_by_segment,
                                dependents=continuity_dependents,
                                predecessor_segment_id=segment_id,
                                reason=str(exc),
                            )
                        )
                else:
                    dependency_failed_segments.update(
                        _fail_continuity_descendants(
                            database,
                            job_id=job_id,
                            child_ids_by_segment=child_ids_by_segment,
                            dependents=continuity_dependents,
                            predecessor_segment_id=segment_id,
                            reason=(
                                f"predecessor ended with status "
                                f"'{terminal_child['status']}'"
                            ),
                        )
                    )
        submission_lock.release()
        lock_acquired = False
    except HTTPException as exc:
        if lock_acquired and submission_lock is not None:
            submission_lock.release()
            lock_acquired = False
        failed_parent = database.update_job_if_status(
            job_id,
            "preparing",
            status="failed",
            progress=1.0,
            stage="preflight_failed",
            error=json.dumps(exc.detail, ensure_ascii=False),
            completed_at=utc_now(),
        )
        if failed_parent is not None:
            for child in database.list_job_children(job_id):
                if child["status"] not in _TERMINAL_STATUSES:
                    database.update_job_child_if_snapshot(
                        child["id"],
                        expected_status=child["status"],
                        expected_updated_at=child["updated_at"],
                        status="failed",
                        progress=1.0,
                        stage="preflight_failed",
                        error=json.dumps(exc.detail, ensure_ascii=False),
                        completed_at=utc_now(),
                    )
        raise
    except (ComfyError, httpx.HTTPError, KeyError) as exc:
        # An incompatible ComfyUI may ignore our caller-assigned prompt id and
        # queue another one. ComfyClient reports both ids after attempting an
        # inline exact cancellation. Bind only a detail value authenticated by
        # the current durable requested id, so outer cleanup and restart
        # recovery target the actual upstream side effect if inline cleanup was
        # false or raised.
        inline_cancelled = _bind_actual_prompt_id_from_submit_error(
            database, exc, possibly_submitted
        )
        if lock_acquired and submission_lock is not None:
            submission_lock.release()
            lock_acquired = False
        await _cleanup_failed_timeline_submission(
            request,
            job_id=job_id,
            client=client,
            error=exc,
            possibly_submitted=possibly_submitted,
            inline_cancelled=inline_cancelled,
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except asyncio.CancelledError:
        if lock_acquired and submission_lock is not None:
            submission_lock.release()
            lock_acquired = False
        current = database.get_job(job_id)
        if current is not None and current["status"] in {"cancelling", "cancelled"}:
            current["children"] = database.list_job_children(job_id)
            return _job_read_for_request(request, current)
        raise
    except BaseException:
        # A graceful server shutdown can cancel the shielded submission task.
        # Persist a finite recovery state before releasing the endpoint lock;
        # caller-assigned prompt ids make every possibly accepted side effect
        # targetable after restart.
        if lock_acquired and submission_lock is not None:
            submission_lock.release()
            lock_acquired = False
        current = database.get_job(job_id)
        if current is not None and current["status"] not in _TERMINAL_STATUSES:
            has_bound_prompt = False
            for child in database.list_job_children(job_id):
                if child["status"] in _TERMINAL_STATUSES:
                    continue
                if child.get("prompt_id"):
                    has_bound_prompt = True
                    database.update_job_child_if_snapshot(
                        child["id"],
                        expected_status=child["status"],
                        expected_updated_at=child["updated_at"],
                        status="cancelling",
                        stage="submission_interrupted",
                        completed_at=None,
                    )
                else:
                    database.update_job_child_if_snapshot(
                        child["id"],
                        expected_status=child["status"],
                        expected_updated_at=child["updated_at"],
                        status="cancelled",
                        progress=1.0,
                        stage="not_submitted",
                        completed_at=utc_now(),
                    )
            latest = database.get_job(job_id)
            if latest is not None and latest["status"] not in _TERMINAL_STATUSES:
                database.update_job_if_snapshot(
                    job_id,
                    expected_status=latest["status"],
                    expected_stage=latest.get("stage"),
                    expected_updated_at=latest["updated_at"],
                    status="cancelling" if has_bound_prompt else "cancelled",
                    progress=1.0 if not has_bound_prompt else current["progress"],
                    stage=(
                        "submission_interrupted"
                        if has_bound_prompt
                        else "submission_cancelled"
                    ),
                    error="native segment submission was interrupted",
                    completed_at=None if has_bound_prompt else utc_now(),
                )
        raise
    finally:
        if submission_ticket is not None:
            ticket = submission_ticket

            def release_submission_ticket(
                _completed_predecessor: asyncio.Future[None] | None = None,
            ) -> None:
                if not ticket.done():
                    ticket.set_result(None)
                if (
                    endpoint_key is not None
                    and request.app.state.submission_tails.get(endpoint_key)
                    is ticket
                ):
                    request.app.state.submission_tails.pop(endpoint_key, None)

            if predecessor is not None and not predecessor.done():
                # A cancelled waiter may end before the task ahead of it. Its
                # dispatcher ownership can be released immediately (making
                # the terminal record deletable), but its endpoint ticket must
                # keep proxying that predecessor. Otherwise a task created
                # after the cancellation could jump across the unfinished
                # ticket chain before either task has acquired the AnyIO lock.
                predecessor.add_done_callback(release_submission_ticket)
            else:
                release_submission_ticket()
    published_children = database.list_job_children(job_id)
    published_segment_children = [
        child for child in published_children if child["segment_ids"]
    ]
    started_children = [
        child
        for child in published_segment_children
        if child["status"] in {"running", "succeeded", "failed"}
    ]
    published_progress = (
        sum(
            float(child["progress"]) * len(child["segment_ids"])
            for child in published_segment_children
        )
        / sum(len(child["segment_ids"]) for child in published_segment_children)
        if published_segment_children
        else 0.0
    )
    job = database.update_job_if_status(
        job_id,
        "preparing",
        status="running" if started_children else "queued",
        progress=published_progress,
        stage=(
            (started_children[0].get("stage") or "开始执行")
            if started_children
            else "queued"
        ),
        started_at=utc_now() if started_children else None,
        # Keep the old single-prompt field useful for legacy clients while
        # child rows remain the execution authority. Mixed-family parents
        # deliberately leave it null.
        prompt_id=(submitted_children[0][1] if len(submitted_children) == 1 else None),
    )
    if job is None:
        latest = database.get_job(job_id)
        if latest is None:
            raise HTTPException(status_code=404, detail="job disappeared after submission")
        if latest["status"] in {"cancelling", "cancelled"}:
            job = await _cancel_timeline_job(request, latest)
        else:
            # The background reconciler may legitimately advance a terminal-
            # gated parent while this dispatcher still owns later siblings, or
            # already own a single-flight assembly claim. Losing the preparing
            # CAS is neither a cancellation request nor authority to rewrite
            # that claim; the reconciler remains the lifecycle owner.
            job = latest
    job["children"] = database.list_job_children(job_id)
    return _job_read_for_request(request, job)


async def _create_timeline_job(
    request: Request,
    body: TimelineJobRequest,
    *,
    parent_mode: str = "timeline",
    project_id: str | None = None,
) -> JobRead:
    """Complete the server-owned submission batch despite client disconnects.

    Once persistence starts, cancellation of the ASGI request must not strand
    a partially submitted parent. User cancellation remains a separate API
    operation and can still win each child/parent CAS while this shielded task
    runs. A hard process exit is reconciled by lifespan startup recovery.
    """

    accepted = asyncio.Event()
    accepted_release = asyncio.Event()
    job_id = str(uuid.uuid4())
    task = asyncio.create_task(
        _create_timeline_job_impl(
            request,
            body,
            parent_mode=parent_mode,
            job_id=job_id,
            accepted=accepted,
            accepted_release=accepted_release,
            project_id=project_id,
        ),
        name="timeline-submit",
    )
    tasks: set[asyncio.Task[JobRead]] = request.app.state.submission_tasks
    tasks.add(task)
    # Keep a strong durable-job mapping for explicit cancellation. The public
    # task set alone cannot target one dispatcher without inspecting private
    # coroutine frames.
    request.app.state.submission_jobs[job_id] = task

    def consume(done: asyncio.Task[JobRead]) -> None:
        tasks.discard(done)
        if request.app.state.submission_jobs.get(job_id) is done:
            request.app.state.submission_jobs.pop(job_id, None)
        if not done.cancelled():
            # Retrieve a detached task exception after client disconnect; a
            # normal awaiting caller still observes the same stored exception.
            done.exception()

    task.add_done_callback(consume)
    accepted_wait = asyncio.create_task(accepted.wait())
    try:
        done, _ = await asyncio.wait(
            {task, accepted_wait}, return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            return await asyncio.shield(task)
        accepted_release.set()
        job = _db(request).get_job(job_id)
        if job is None:
            return await asyncio.shield(task)
        job["children"] = _db(request).list_job_children(job_id)
        return _job_read_for_request(request, job)
    finally:
        accepted_release.set()
        accepted_wait.cancel()


async def _run_timeline_reconciler(request: Request) -> None:
    """Continuously reconcile a bounded batch without requiring browser GETs."""

    database = _db(request)
    wake_event: asyncio.Event = request.app.state.reconcile_wake_event
    while True:
        snapshots = database.list_active_timeline_jobs(
            limit=int(request.app.state.reconcile_batch_size)
        )
        if snapshots:
            # Any active row may change during this pass. Wake the SSE task
            # stream so the browser refreshes without waiting for its fallback
            # timer. A tombstoneless ComfyUI restart is surfaced here too.
            request.app.state.task_change_event.set()
        for snapshot in snapshots:
            try:
                await _sync_job(
                    request, snapshot, allow_timeline_assembly=True
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                # Per-job transport, validation, deletion, or assembly errors
                # must not kill the process-owned monitor. Durable state and
                # the next bounded pass remain the recovery mechanism. Touch
                # the ordering version so four poison rows cannot permanently
                # starve every newer active parent from this bounded batch.
                database.touch_active_timeline_job(
                    str(snapshot["id"]),
                    expected_updated_at=str(snapshot["updated_at"]),
                )
                continue
        try:
            await asyncio.wait_for(
                wake_event.wait(),
                timeout=float(request.app.state.reconcile_interval_seconds),
            )
        except asyncio.TimeoutError:
            continue
        # Queue transitions commonly emit several adjacent status/execution
        # messages.  Merge that burst before the next bounded authority read.
        wake_event.clear()
        await asyncio.sleep(0.05)
        wake_event.clear()


async def _recover_interrupted_submission(
    request: Request, snapshot: dict[str, Any]
) -> None:
    """Target every prompt id whose submit owner died in another process.

    Startup persistence marks these children ``restart_cancel_pending``.  The
    ordinary reconciler deliberately ignores that process-ownership stage so
    it cannot infer local cancellation from queue absence before this worker
    has attempted ComfyUI's prompt-id-specific cancel operation.
    """

    database = _db(request)
    job = database.get_job(str(snapshot["id"]))
    if job is None or job["status"] in _TERMINAL_STATUSES:
        return
    client = _comfy(
        request, RuntimeSettings.model_validate(job["settings_snapshot"])
    )
    dispatch_errors: list[str] = []
    has_unconfirmed = False
    for child in database.list_job_children(job["id"]):
        if child["status"] in _TERMINAL_STATUSES:
            continue
        prompt_id = child.get("prompt_id")
        if not prompt_id:
            await _cas_active_child_update(
                database,
                child,
                status="cancelled",
                progress=1.0,
                stage="restart_cancelled_not_submitted",
                error=None,
                completed_at=utc_now(),
            )
            continue
        # Reassert ownership in case a previous bounded pass recorded a
        # transient failure.  This closes the race with the normal reconciler
        # before the exact directed cancellation is retried.
        while child["status"] not in _TERMINAL_STATUSES:
            child, claimed = await _cas_active_child_update(
                database,
                child,
                status="cancelling",
                stage="restart_cancel_pending",
                error=None,
                completed_at=None,
            )
            if claimed:
                break
        if child["status"] in _TERMINAL_STATUSES:
            continue
        prompt_id = str(child["prompt_id"])
        try:
            dispatched = await client.cancel(prompt_id)
        except asyncio.CancelledError:
            raise
        except (ComfyError, httpx.HTTPError) as exc:
            dispatch_errors.append(f"{child['id']}: {exc}")
            await _cas_active_child_update(
                database,
                child,
                status="cancelling",
                stage="restart_cancel_failed",
                error=str(exc),
                completed_at=None,
            )
            continue
        if dispatched:
            confirmation = child
            while (
                confirmation["status"] not in _TERMINAL_STATUSES
                and str(confirmation.get("prompt_id") or "") == prompt_id
            ):
                confirmation, committed = await _cas_active_child_update(
                    database,
                    confirmation,
                    status="cancelled",
                    progress=1.0,
                    stage="cancelled_after_restart",
                    error=None,
                    completed_at=utc_now(),
                )
                if committed:
                    break
        else:
            unconfirmed, _ = await _cas_active_child_update(
                database,
                child,
                status="cancelling",
                stage="restart_cancel_unconfirmed",
                error="ComfyUI could not confirm directed restart cancellation",
                completed_at=None,
            )
            # ``False`` is not proof that the caller-assigned prompt id can
            # never appear: the old /prompt handler may still be validating
            # before its eventual queue.put.  Accept a positive exact history
            # record, but never turn temporary queue/history absence into a
            # local terminal state.  The recovery-owned marker remains and the
            # next bounded pass retries this same directed cancel.
            try:
                exact_history = await client.history(str(prompt_id))
            except asyncio.CancelledError:
                raise
            except (ComfyError, httpx.HTTPError):
                exact_history = {}
            history_entry = (
                exact_history.get(str(prompt_id))
                if isinstance(exact_history, dict)
                else None
            )
            if isinstance(history_entry, dict):
                unconfirmed = await _sync_timeline_child(
                    request,
                    unconfirmed,
                    parent_cancelling=True,
                    history_entry=history_entry,
                    running=False,
                    pending=False,
                    confirmed_absent=False,
                    respect_process_ownership=False,
                )
            if unconfirmed["status"] not in _TERMINAL_STATUSES:
                has_unconfirmed = True

    latest = database.get_job(job["id"])
    if latest is None or latest["status"] in _TERMINAL_STATUSES:
        return
    if dispatch_errors:
        database.update_job_if_snapshot(
            latest["id"],
            expected_status=latest["status"],
            expected_stage=latest.get("stage"),
            expected_updated_at=latest["updated_at"],
            status="cancelling",
            stage="restart_cancel_failed",
            error="; ".join(dispatch_errors)[:20_000],
            completed_at=None,
        )
        # Do not let an unavailable queue/history endpoint delay the next
        # targeted-cancel retry.  The bounded worker rotates this row using
        # its freshly updated row version.
        return
    if has_unconfirmed:
        database.update_job_if_snapshot(
            latest["id"],
            expected_status=latest["status"],
            expected_stage=latest.get("stage"),
            expected_updated_at=latest["updated_at"],
            status="cancelling",
            stage="restart_cancel_unconfirmed",
            error="one or more directed restart cancellations were unconfirmed",
            completed_at=None,
        )
        return
    if latest is not None:
        await _sync_timeline_job(request, latest, allow_assembly=False)


async def _run_interrupted_submission_recovery(request: Request) -> None:
    """Retry a bounded, cancellable batch after the ASGI app is available."""

    database = _db(request)
    while True:
        snapshots = database.list_interrupted_preparing_jobs(
            limit=int(request.app.state.recovery_batch_size)
        )
        if snapshots:
            # Separate parents can point at different ComfyUI endpoints.  A
            # bounded gather prevents one black-hole endpoint from stopping
            # every other restart cancellation while limiting fan-out.
            async def recover(snapshot: dict[str, Any]) -> None:
                try:
                    await _recover_interrupted_submission(request, snapshot)
                except asyncio.CancelledError:
                    raise
                except (ComfyError, httpx.HTTPError, HTTPException, KeyError, ValidationError):
                    latest = database.get_job(str(snapshot["id"]))
                    if latest is not None and latest["status"] not in _TERMINAL_STATUSES:
                        database.update_job_if_status(
                            str(snapshot["id"]),
                            latest["status"],
                            status="cancelling",
                            stage="restart_cancel_failed",
                            error="restart submission recovery failed",
                            completed_at=None,
                        )

            await asyncio.gather(*(recover(snapshot) for snapshot in snapshots))
        await asyncio.sleep(float(request.app.state.reconcile_interval_seconds))


def create_app(
    *,
    database_path: str | Path | None = None,
    comfy_factory: ComfyFactory | None = None,
    storage_config_path: str | Path | None = None,
    legacy_database_path: str | Path | None = None,
    public_api_prefix: str = "",
    raylight_requirements_path: str | Path | None = None,
) -> FastAPI:
    set_public_api_prefix(public_api_prefix)
    storage = StorageController.resolve(
        database_path,
        storage_config_path=storage_config_path,
        legacy_database_path=legacy_database_path,
    )
    database = Database(storage.active_database_path)
    instance_lock = DirectorInstanceLock(storage.active_database_path)
    live_preview_cache = LivePreviewCache()
    reconcile_wake_event = asyncio.Event()
    prompt_terminal_events = PromptTerminalEvents()
    task_change_event = asyncio.Event()
    submission_tasks: set[asyncio.Task[JobRead]] = set()
    submission_jobs: dict[str, asyncio.Task[JobRead]] = {}
    timeline_sync_tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}
    timeline_sync_all_tasks: set[asyncio.Task[dict[str, Any]]] = set()
    timeline_sync_lock = asyncio.Lock()
    storage_operation_lock = asyncio.Lock()
    storage_gate = asyncio.Condition()
    submission_ticket_lock = asyncio.Lock()
    submission_tails: dict[str, asyncio.Future[None]] = {}

    async def persist_native_progress(
        comfy_origin: str, event: ComfyProgressEvent | ComfyExecutionEvent
    ) -> None:
        expected_origin = Database.canonical_comfy_origin(comfy_origin)
        for child in database.find_job_children_by_prompt_id(event.prompt_id):
            parent = database.get_job(child["job_id"])
            if parent is None or parent["status"] in _TERMINAL_STATUSES:
                continue
            try:
                snapshot_settings = RuntimeSettings.model_validate(
                    parent["settings_snapshot"]
                )
            except ValidationError:
                continue
            if (
                Database.canonical_comfy_origin(snapshot_settings.comfy_url)
                != expected_origin
            ):
                continue
            snapshot = (
                child_progress_snapshot(child, event)
                if isinstance(event, ComfyProgressEvent)
                else child_execution_snapshot(child, event)
            )
            if snapshot is None:
                continue
            try:
                database.update_job_child_progress_monotonic(
                    child["id"],
                    progress=snapshot.progress,
                    stage=snapshot.stage,
                    expected_updated_at=child["updated_at"],
                )
                # Live stage/step detail changed; the SSE stream coalesces
                # these high-frequency websocket frames into at most one
                # browser refresh per second.
                task_change_event.set()
            except KeyError:
                # A terminal task can be forgotten while a late websocket
                # frame is in flight.  Deletion wins and must not turn the
                # reconnecting monitor into an error loop.
                continue

    async def persist_native_preview(
        comfy_origin: str, event: ComfyPreviewEvent
    ) -> None:
        expected_origin = Database.canonical_comfy_origin(comfy_origin)
        for child in database.find_job_children_by_prompt_id(event.prompt_id):
            parent = database.get_job(child["job_id"])
            if parent is None or parent["status"] in _TERMINAL_STATUSES:
                continue
            try:
                snapshot_settings = RuntimeSettings.model_validate(
                    parent["settings_snapshot"]
                )
            except ValidationError:
                continue
            if (
                Database.canonical_comfy_origin(snapshot_settings.comfy_url)
                != expected_origin
            ):
                continue
            segment_id = sampler_segment_for_node(child, event.node_id)
            if segment_id is None:
                continue
            # Re-read both rows at the final cache boundary. DELETE cascades
            # child rows; the cache tombstone closes the inverse ordering where
            # deletion wins immediately before this put.
            latest_parent = database.get_job(parent["id"])
            latest_child = database.get_job_child(child["id"])
            if (
                latest_parent is None
                or latest_parent["status"] in _TERMINAL_STATUSES
                or latest_child is None
                or latest_child["status"] not in {"queued", "running"}
                or latest_child.get("prompt_id") != event.prompt_id
                or sampler_segment_for_node(latest_child, event.node_id)
                != segment_id
            ):
                continue
            live_preview_cache.put(
                job_id=parent["id"],
                child_id=child["id"],
                segment_id=segment_id,
                event=event,
            )

    async def wake_native_reconcile(
        _comfy_origin: str, event: ComfyReconcileHint
    ) -> None:
        # WebSocket delivery is optional and lossy.  It may only shorten the
        # wait before the existing queue/history reconciliation pass; it never
        # writes lifecycle state or replaces the periodic timeout fallback.
        reconcile_wake_event.set()
        task_change_event.set()
        # Terminal frames also wake any dispatcher wait-gate blocked on this
        # exact prompt. The gate still re-reads history for its terminal
        # certificate, so this is a pure latency hint, never a state write.
        if event.prompt_id is not None and event.event_type in {
            "execution_success",
            "execution_error",
            "execution_interrupted",
        }:
            prompt_terminal_events.notify(event.prompt_id)

    progress_manager = NativeProgressManager(
        persist_native_progress, persist_native_preview, wake_native_reconcile
    )

    @asynccontextmanager
    async def managed_lifespan(app: FastAPI):
        reconciler_task: asyncio.Task[None] | None = None
        recovery_task: asyncio.Task[None] | None = None
        recovery_request = Request(
            {
                "type": "http",
                "app": app,
                "method": "POST",
                "path": "/internal/recover-submissions",
                "root_path": "",
                "scheme": "http",
                "query_string": b"",
                "headers": [],
                "client": ("127.0.0.1", 0),
                "server": ("127.0.0.1", 0),
            }
        )
        try:
            reconcile_wake_event.clear()
            database.initialize()
            database.recover_interrupted_assemblies()
            # Startup is intentionally local-only.  The previous process
            # cannot still own these rows, so replace its transient ownership
            # markers in one SQLite transaction.  Every ComfyUI
            # queue/history/cancel request is deferred to managed tasks
            # created below, after the app can yield.
            database.prepare_interrupted_submissions_for_recovery()
            settings = database.get_settings()
            if str(settings.comfy_url).strip():
                progress_manager.ensure(str(settings.comfy_url), settings.client_id)
            for snapshot in database.list_active_job_settings():
                try:
                    active_settings = RuntimeSettings.model_validate(snapshot)
                except ValidationError:
                    continue
                if str(active_settings.comfy_url).strip():
                    progress_manager.ensure(
                        str(active_settings.comfy_url), active_settings.client_id
                    )
            reconciler_task = asyncio.create_task(
                _run_timeline_reconciler(recovery_request),
                name="timeline-reconciler",
            )
            recovery_task = asyncio.create_task(
                _run_interrupted_submission_recovery(recovery_request),
                name="interrupted-submission-recovery",
            )
            yield
        finally:
            managed_background = [
                task
                for task in (recovery_task, reconciler_task)
                if task is not None
            ]
            for task in managed_background:
                task.cancel()
            if managed_background:
                await asyncio.gather(*managed_background, return_exceptions=True)
            pending_submissions = list(submission_tasks)
            for task in pending_submissions:
                task.cancel()
            if pending_submissions:
                await asyncio.gather(*pending_submissions, return_exceptions=True)
            submission_jobs.clear()
            submission_tails.clear()
            pending_syncs = list(timeline_sync_all_tasks)
            for task in pending_syncs:
                task.cancel()
            if pending_syncs:
                await asyncio.gather(*pending_syncs, return_exceptions=True)
            timeline_sync_tasks.clear()
            timeline_sync_all_tasks.clear()
            await progress_manager.close()
            live_preview_cache.clear()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # This must be the first persistence/lifecycle action.  A losing
        # process must not migrate SQLite, claim recovery rows, or contact
        # ComfyUI before startup fails with an actionable owner diagnostic.
        instance_lock.acquire()
        ensure_media_tools_on_path()
        try:
            async with managed_lifespan(app):
                yield
        finally:
            # managed_lifespan drains recovery/reconcile/submission/sync tasks
            # and closes progress sockets before the database may be handed to
            # another Director process.
            instance_lock.release()

    app = FastAPI(title="Director Web API", version="0.1.0", lifespan=lifespan)
    app.state.database = database
    app.state.instance_lock = instance_lock
    app.state.storage = storage
    app.state.storage_operation_lock = storage_operation_lock
    app.state.storage_gate = storage_gate
    app.state.storage_write_frozen = False
    app.state.storage_transitioning = False
    app.state.storage_inflight_mutations = 0
    app.state.comfy_factory = comfy_factory or default_comfy_factory
    app.state.progress_manager = progress_manager
    app.state.live_preview_cache = live_preview_cache
    app.state.raylight_install_manager = RayLightInstallManager()
    app.state.ffmpeg_install_manager = FFmpegInstallManager()
    app.state.raylight_requirements_path = (
        Path(raylight_requirements_path)
        if raylight_requirements_path is not None
        else raylight_default_requirements_path()
    )
    app.state.reconcile_wake_event = reconcile_wake_event
    app.state.prompt_terminal_events = prompt_terminal_events
    app.state.task_change_event = task_change_event
    app.state.submission_tasks = submission_tasks
    app.state.submission_jobs = submission_jobs
    app.state.timeline_sync_tasks = timeline_sync_tasks
    app.state.timeline_sync_all_tasks = timeline_sync_all_tasks
    app.state.timeline_sync_lock = timeline_sync_lock
    app.state.submission_ticket_lock = submission_ticket_lock
    app.state.submission_tails = submission_tails
    app.state.upload_progress = {}
    app.state.reconcile_batch_size = 4
    app.state.recovery_batch_size = 4
    # The reconciler's normal terminal path is event-driven: ComfyUI's
    # execution_success/error/interrupted websocket frames set
    # reconcile_wake_event and wake the pass immediately. This periodic
    # timeout is only the lossy-websocket / restart fallback, so it can be
    # deliberately coarse instead of a tight 2-second poll. The interrupted-
    # submission recovery worker shares the same value for its idempotent
    # directed-cancel retry; a 10s retry cadence is still bounded by the
    # reconciler's eventual consistency.
    app.state.reconcile_interval_seconds = 10.0
    # EventSource reconnects automatically after a clean EOF. Bound every SSE
    # response so a reverse proxy that misses a downstream disconnect cannot
    # keep an orphaned upstream socket for the lifetime of the process.
    app.state.task_events_max_lifetime_seconds = 300.0
    # Serializes parent prompt batches per ComfyUI endpoint so segment units
    # from concurrent HTTP requests cannot interleave in the global queue.
    app.state.submission_locks = {}

    async def begin_storage_transition() -> bool:
        """Stop admitting new writes and drain already-admitted requests."""

        entered = False
        previous_frozen = False
        try:
            async with storage_gate:
                previous_frozen = bool(app.state.storage_write_frozen)
                app.state.storage_transitioning = True
                entered = True
                while app.state.storage_inflight_mutations:
                    await storage_gate.wait()
                return previous_frozen
        except BaseException:
            if entered:
                await finish_storage_transition(frozen=previous_frozen)
            raise

    async def _finish_storage_transition(*, frozen: bool) -> None:
        async with storage_gate:
            app.state.storage_write_frozen = frozen
            app.state.storage_transitioning = False
            storage_gate.notify_all()

    async def _complete_storage_cleanup(operation: Awaitable[None]) -> None:
        """Finish gate bookkeeping even if this request is being cancelled."""

        cleanup = asyncio.ensure_future(operation)
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            # shield keeps the cleanup alive. Awaiting it after consuming the
            # cancellation makes the state transition observable before the
            # request exits. Do not turn a response already produced by
            # call_next into CancelledError merely because cancellation landed
            # on this final bookkeeping await. If call_next itself was
            # cancelled, its original exception still propagates after the
            # surrounding finally block completes.
            await cleanup

    async def finish_storage_transition(*, frozen: bool) -> None:
        await _complete_storage_cleanup(
            _finish_storage_transition(frozen=frozen)
        )

    async def release_storage_mutation() -> None:
        async def release() -> None:
            async with storage_gate:
                app.state.storage_inflight_mutations -= 1
                if app.state.storage_inflight_mutations == 0:
                    storage_gate.notify_all()

        await _complete_storage_cleanup(release())

    async def storage_has_active_owner() -> bool:
        has_active_work = await anyio.to_thread.run_sync(database.has_active_work)
        has_background_owner = any(
            not task.done() for task in (*submission_tasks, *timeline_sync_all_tasks)
        )
        return has_active_work or has_background_owner

    class StorageRestartWriteGateMiddleware:
        """Pure ASGI write gate that preserves the caller's asyncio task.

        Starlette's decorator-style HTTP middleware dispatches the route in a
        child task. Several lifecycle paths intentionally use task ownership
        while coordinating cancellation, so this gate stays at the ASGI layer
        and never adds a task boundary around the endpoint.
        """

        def __init__(self, asgi_app: Any) -> None:
            self.asgi_app = asgi_app

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            if scope.get("type") != "http":
                await self.asgi_app(scope, receive, send)
                return
            method = str(scope.get("method") or "").upper()
            path = str(scope.get("path") or "")
            is_user_mutation = method in {"POST", "PUT", "PATCH", "DELETE"}
            is_storage_operation = path == "/api/storage" or path.startswith(
                "/api/storage/"
            )
            if not is_user_mutation:
                await self.asgi_app(scope, receive, send)
                return

            presented_identity: str | None = None
            for raw_name, raw_value in scope.get("headers") or ():
                if raw_name.lower() == b"x-director-database-identity":
                    presented_identity = raw_value.decode("latin-1")
                    break
            if (
                presented_identity is not None
                and presented_identity != storage.active_database_identity
            ):
                response = JSONResponse(
                    status_code=409,
                    content={
                        "code": "stale_database_identity",
                        "detail": (
                            "the browser is editing a different Director database; "
                            "refresh before making changes"
                        ),
                    },
                )
                await response(scope, receive, send)
                return

            # Storage operations remain available while an earlier selection
            # has frozen ordinary writes, but they are not exempt from stale
            # browser identity. This prevents an old session from cancelling
            # a newer session's pending B→C selection by submitting PUT B.
            if is_storage_operation:
                await self.asgi_app(scope, receive, send)
                return

            blocked_by_transition = False
            async with storage_gate:
                blocked_by_transition = bool(
                    app.state.storage_transitioning
                    or app.state.storage_write_frozen
                )
                if not blocked_by_transition:
                    app.state.storage_inflight_mutations += 1
            if blocked_by_transition:
                response = JSONResponse(
                    status_code=409,
                    content={
                        "detail": (
                            "a database storage change is pending; restart "
                            "Director before making further changes"
                        )
                    },
                )
                await response(scope, receive, send)
                return
            try:
                await self.asgi_app(scope, receive, send)
            finally:
                await release_storage_mutation()

    app.add_middleware(StorageRestartWriteGateMiddleware)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/storage", response_model=StorageStatusRead)
    async def get_storage() -> StorageStatusRead:
        try:
            status = storage.status()
        except StorageConfigurationError as exc:
            raise _storage_http_error(exc) from exc
        return _storage_status_read(status)

    @app.put("/api/storage", response_model=StorageStatusRead)
    async def configure_storage(body: StorageConfigureRequest) -> StorageStatusRead:
        async with storage_operation_lock:
            if not instance_lock.acquired:
                raise HTTPException(
                    status_code=503,
                    detail="Director storage ownership is not active",
                )
            try:
                targets_active = storage.targets_active_database(
                    body.database_path
                )
            except StoragePathError as exc:
                raise _storage_http_error(exc) from exc
            # Do this before closing the write gate. A job can be waiting in a
            # cancellable ComfyUI preflight while its create request is still
            # in flight; transitioning first would reject the cancel request
            # and wait forever for that create request to finish.
            if not targets_active and await storage_has_active_owner():
                raise HTTPException(
                    status_code=409,
                    detail="database selection requires all jobs to be terminal",
                )
            previous_frozen = await begin_storage_transition()
            next_frozen = previous_frozen
            try:
                try:
                    targets_active = storage.targets_active_database(
                        body.database_path
                    )
                    if not targets_active and await storage_has_active_owner():
                        raise HTTPException(
                            status_code=409,
                            detail=(
                                "database selection requires all jobs to be terminal"
                            ),
                        )
                    status = await anyio.to_thread.run_sync(
                        partial(storage.configure_existing, body.database_path)
                    )
                except (
                    StorageConfigurationError,
                    StoragePathError,
                    StorageValidationError,
                    StorageConflictError,
                    StorageOperationError,
                ) as exc:
                    raise _storage_http_error(exc) from exc
                # Selecting a different authority is a restart boundary. The
                # active path explicitly cancels a pending switch and unfreezes.
                next_frozen = status.restart_required
            finally:
                await finish_storage_transition(frozen=next_frozen)
        return _storage_status_read(status)

    @app.post("/api/storage/migrate", response_model=StorageMigrationRead)
    async def migrate_storage(body: StorageMigrateRequest) -> StorageMigrationRead:
        async with storage_operation_lock:
            if not instance_lock.acquired:
                raise HTTPException(
                    status_code=503,
                    detail="Director storage ownership is not active",
                )
            try:
                # Validate user-controlled expansion/resolution before doing
                # either the idle preflight or changing admission state.
                storage.targets_active_database(body.target_path)
            except StoragePathError as exc:
                raise _storage_http_error(exc) from exc
            if await storage_has_active_owner():
                raise HTTPException(
                    status_code=409,
                    detail="database migration requires all jobs to be terminal",
                )
            previous_frozen = await begin_storage_transition()
            next_frozen = previous_frozen
            try:
                if previous_frozen:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "a database storage change is already pending; "
                            "restart Director or select the active database"
                        ),
                    )
                if await storage_has_active_owner():
                    raise HTTPException(
                        status_code=409,
                        detail="database migration requires all jobs to be terminal",
                    )
                try:
                    status, migrated_from, migrated_to = (
                        await anyio.to_thread.run_sync(
                            partial(storage.migrate, body.target_path)
                        )
                    )
                except (
                    StorageConfigurationError,
                    StoragePathError,
                    StorageValidationError,
                    StorageConflictError,
                    StorageOperationError,
                ) as exc:
                    raise _storage_http_error(exc) from exc
                # The copied database and bootstrap selection now form the next
                # instance's authority. Keep this process read-only so the
                # source cannot diverge after its final consistent snapshot.
                next_frozen = True
            finally:
                await finish_storage_transition(frozen=next_frozen)
        current = _storage_status_read(status)
        return StorageMigrationRead(
            **current.model_dump(),
            migrated_from=str(migrated_from),
            migrated_to=str(migrated_to),
        )

    @app.get("/api/settings", response_model=RuntimeSettings)
    async def get_settings(request: Request) -> RuntimeSettings:
        return _db(request).get_settings()

    @app.get(
        "/api/settings/authority",
        response_model=RuntimeSettingsAuthorityRead,
    )
    async def get_settings_authority(
        request: Request,
    ) -> RuntimeSettingsAuthorityRead:
        settings, authority = _db(request).get_settings_authority()
        return RuntimeSettingsAuthorityRead(
            settings=settings,
            authority_token=authority,
        )

    @app.put("/api/settings", response_model=RuntimeSettings)
    async def put_settings(request: Request, settings: RuntimeSettings) -> RuntimeSettings:
        return _db(request).put_settings(settings)

    @app.get("/api/capabilities")
    async def get_capabilities(request: Request) -> dict[str, Any]:
        settings, authority = _runtime_authority_snapshot(request)
        if not str(settings.comfy_url).strip():
            report = {
                "connection": "offline",
                "supported_modes": [],
                "supports_cancel": False,
                "available_nodes": [],
                "missing_nodes": list(ComfyClient.STANDARD_REQUIRED_NODES),
                "native_timeline": {
                    "supported": False,
                    "modes": [],
                    "continuity": False,
                },
                "execution_backends": {
                    "standard": {
                        "available": False,
                        "missing_nodes": list(ComfyClient.STANDARD_REQUIRED_NODES),
                    },
                    "raylight": {
                        "available": False,
                        "missing_nodes": list(ComfyClient.RAYLIGHT_REQUIRED_NODES),
                        "conditional_requirements": {
                            "lora": {
                                "available": False,
                                "missing_nodes": list(
                                    ComfyClient.RAYLIGHT_LORA_REQUIRED_NODES
                                ),
                            }
                        },
                    },
                },
                "message": _COMFY_NOT_CONFIGURED,
            }
            _assert_runtime_authority(request, authority)
            return report
        try:
            report = await _comfy(request, settings).capabilities()
            report.setdefault("message", "ComfyUI connection is ready")
        except (ComfyError, httpx.HTTPError) as exc:
            report = {
                "connection": "offline",
                "supported_modes": [],
                "supports_cancel": False,
                "available_nodes": [],
                "missing_nodes": list(ComfyClient.STANDARD_REQUIRED_NODES),
                "native_timeline": {
                    "supported": False,
                    "modes": [],
                    "continuity": False,
                },
                "execution_backends": {
                    "standard": {
                        "available": False,
                        "missing_nodes": list(ComfyClient.STANDARD_REQUIRED_NODES),
                    },
                    "raylight": {
                        "available": False,
                        "missing_nodes": list(ComfyClient.RAYLIGHT_REQUIRED_NODES),
                        "conditional_requirements": {
                            "lora": {
                                "available": False,
                                "missing_nodes": list(
                                    ComfyClient.RAYLIGHT_LORA_REQUIRED_NODES
                                ),
                            }
                        },
                    },
                },
                "message": str(exc),
            }
        _assert_runtime_authority(request, authority)
        return report

    @app.post("/api/capabilities")
    async def test_capabilities(request: Request, body: ComfyURLRequest) -> dict[str, Any]:
        started = __import__("time").monotonic()
        try:
            probe_settings = _origin_settings(
                _db(request).get_settings(), str(body.comfy_url)
            )
            report = await _comfy(request, probe_settings).capabilities()
            missing = report.get("missing_nodes") or []
            return {
                # Reaching the capability endpoint is a successful connection
                # test. Missing execution nodes affect readiness, not network
                # connectivity, and are reported separately in the message.
                "ok": True,
                "latency_ms": report.get("latency_ms") or round((__import__("time").monotonic() - started) * 1000, 1),
                "message": "连接成功" if not missing else f"连接成功，但缺少节点: {', '.join(missing)}",
            }
        except (ComfyError, httpx.HTTPError) as exc:
            return {"ok": False, "message": str(exc)}

    @app.get("/api/models")
    async def get_models(request: Request) -> dict[str, list[str]]:
        settings, authority = _runtime_authority_snapshot(request)
        try:
            models = await _comfy(request, settings).models()
        except ComfyError as exc:
            detail = exc.detail if isinstance(exc.detail, str) and exc.detail.strip() else str(exc)
            raise HTTPException(status_code=502, detail=detail) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        _assert_runtime_authority(request, authority)
        return models

    @app.get("/api/media/setup")
    async def get_media_setup(request: Request) -> dict[str, Any]:
        status = media_tools_status()
        status["install"] = request.app.state.ffmpeg_install_manager.snapshot()
        return status

    @app.post("/api/media/ffmpeg/install")
    async def post_media_ffmpeg_install(request: Request) -> dict[str, Any]:
        manager = request.app.state.ffmpeg_install_manager
        try:
            await manager.start_install()
        except RayLightInstallUnavailable as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RayLightInstallConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return manager.snapshot()

    @app.post("/api/media/ffmpeg/cancel")
    async def post_media_ffmpeg_cancel(request: Request) -> dict[str, Any]:
        manager = request.app.state.ffmpeg_install_manager
        await manager.cancel()
        return manager.snapshot()

    @app.get("/api/raylight/setup")
    async def get_raylight_setup(request: Request) -> dict[str, Any]:
        requirements_path = request.app.state.raylight_requirements_path
        settings = _db(request).get_settings()
        return {
            "enabled": settings.multi_gpu_enabled,
            "platform_supported": raylight_platform_supported(),
            "dependencies_installed": raylight_dependencies_installed(),
            "requirements_available": (
                requirements_path is not None and requirements_path.is_file()
            ),
            "install": request.app.state.raylight_install_manager.snapshot(),
        }

    @app.post("/api/raylight/setup/install")
    async def post_raylight_setup_install(request: Request) -> dict[str, Any]:
        if not raylight_platform_supported():
            raise HTTPException(
                status_code=400,
                detail="multi-GPU inference requires Linux on this release",
            )
        requirements_path = request.app.state.raylight_requirements_path
        if requirements_path is None or not requirements_path.is_file():
            raise HTTPException(
                status_code=400,
                detail="RayLight requirements file is not available in this installation",
            )
        manager = request.app.state.raylight_install_manager
        try:
            await manager.start(requirements_path)
        except RayLightInstallUnavailable as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RayLightInstallConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return manager.snapshot()

    @app.post("/api/raylight/setup/cancel")
    async def post_raylight_setup_cancel(request: Request) -> dict[str, Any]:
        manager = request.app.state.raylight_install_manager
        await manager.cancel()
        return manager.snapshot()

    @app.get(
        "/api/raylight/runtime",
        response_model=RayLightRuntimeStatusRead,
    )
    async def get_raylight_runtime_status(
        request: Request,
    ) -> RayLightRuntimeStatusRead:
        database = _db(request)
        settings, authority = _runtime_authority_snapshot(request)
        try:
            stats = await _comfy(request, settings).system_stats()
            status = _raylight_runtime_status(
                database, str(settings.comfy_url), stats
            )
        except NativeTemplateError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "raylight_runtime_state_invalid",
                    "message": str(exc),
                },
            ) from exc
        except (ComfyError, httpx.HTTPError) as exc:
            raise _invalid_comfy_payload(
                "comfy_system_stats_unavailable",
                str(exc),
            ) from exc
        _assert_runtime_authority(request, authority)
        return status

    @app.post(
        "/api/raylight/runtime/recovery/confirm-comfy-restart",
        response_model=RayLightRuntimeStatusRead,
    )
    async def confirm_raylight_runtime_restart(
        request: Request,
        body: RayLightRuntimeRecoveryConfirmRequest,
    ) -> RayLightRuntimeStatusRead:
        """Discard an incompatible pre-restart actor ledger by certificate.

        A changed GPU inventory or empty queue is never enough on its own.
        The strict request literal is the operator's assertion that the old
        ComfyUI process really exited; endpoint serialization and SQLite CAS
        then ensure no current Director work can be crossed by the reset.
        """

        assert body.confirmation == "comfyui_process_restarted"
        database = _db(request)
        settings = database.get_settings()
        origin = Database.canonical_comfy_origin(settings.comfy_url)
        expected_origin = Database.canonical_comfy_origin(
            body.expected_comfy_origin
        )
        if not origin or expected_origin != origin:
            raise HTTPException(
                status_code=409,
                detail="ComfyUI endpoint changed before RayLight recovery started",
            )
        if any(
            not task.done() for task in request.app.state.submission_tasks
        ):
            raise _raylight_recovery_in_flight(
                "RayLight recovery requires all Director submissions to finish"
            )
        async with request.app.state.submission_ticket_lock:
            tail = request.app.state.submission_tails.get(origin)
            if tail is not None and not tail.done():
                raise _raylight_recovery_in_flight(
                    "RayLight recovery is waiting for an endpoint submission"
                )
        submission_lock = request.app.state.submission_locks.setdefault(
            origin, anyio.Lock()
        )
        if submission_lock.locked():
            raise _raylight_recovery_in_flight(
                "RayLight recovery is waiting for the endpoint submission lock"
            )
        await submission_lock.acquire()
        try:
            current_settings = database.get_settings()
            if (
                Database.canonical_comfy_origin(current_settings.comfy_url)
                != expected_origin
            ):
                raise HTTPException(
                    status_code=409,
                    detail="ComfyUI endpoint changed before RayLight recovery completed",
                )
            client = _comfy(request, current_settings)
            try:
                runtime_state = database.get_raylight_runtime_state(
                    expected_origin
                )
                stats = await client.system_stats()
                status = _raylight_runtime_status(
                    database,
                    expected_origin,
                    stats,
                    state=runtime_state,
                )
            except (NativeTemplateError, ValidationError, ValueError) as exc:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "raylight_runtime_state_invalid",
                        "message": str(exc),
                    },
                ) from exc
            except (ComfyError, httpx.HTTPError) as exc:
                raise _invalid_comfy_payload(
                    "comfy_system_stats_unavailable",
                    str(exc),
                ) from exc
            if status.epoch != body.expected_epoch:
                raise HTTPException(
                    status_code=409,
                    detail="RayLight runtime epoch changed; refresh settings and retry",
                )
            if not status.recovery_required:
                if not status.active:
                    return status
                raise HTTPException(
                    status_code=409,
                    detail="RayLight runtime no longer requires GPU restart recovery",
                )
            if (
                status.recovery_token is None
                or body.expected_recovery_token != status.recovery_token
            ):
                raise HTTPException(
                    status_code=409,
                    detail="RayLight runtime changed; refresh settings and retry",
                )
            if runtime_state is None:
                raise HTTPException(
                    status_code=409,
                    detail="RayLight runtime state no longer exists",
                )
            tail_prompt_id = runtime_state.get("tail_prompt_id")
            tail_action = runtime_state.get("tail_action")
            if (
                not isinstance(tail_prompt_id, str)
                or not tail_prompt_id
                or tail_action not in {"ray_unit", "shutdown"}
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "RayLight recovery requires a recorded runtime tail with "
                        "exact terminal history"
                    ),
                )
            # Official ComfyUI history is process-local and disappears on the
            # very restart being certified. Prefer the exact terminal result
            # Director persisted when it originally reconciled this same tail;
            # otherwise require a fresh exact history result now. The durable
            # certificate is covered by the UI token and final full-state CAS.
            if not _raylight_runtime_has_terminal_certificate(runtime_state):
                try:
                    history = await client.history(tail_prompt_id)
                except (ComfyError, httpx.HTTPError) as exc:
                    raise _invalid_comfy_payload(
                        "comfy_history_unavailable",
                        str(exc),
                    ) from exc
                history_entry, history_absent = _exact_history_result(
                    history, tail_prompt_id
                )
                if history_entry is None and not history_absent:
                    raise _invalid_comfy_payload(
                        "comfy_history_invalid",
                        "ComfyUI exact history response is malformed during RayLight recovery",
                    )
                history_state = (
                    _raylight_recovery_history_state(history_entry)
                    if history_entry is not None
                    else "nonterminal"
                )
                if history_state == "invalid":
                    raise _invalid_comfy_payload(
                        "comfy_history_invalid",
                        "ComfyUI exact history status is contradictory during RayLight recovery",
                    )
                if history_state != "terminal":
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "RayLight recovery requires exact terminal history for "
                            "the recorded runtime tail"
                        ),
                    )
            try:
                queue = await client.queue()
            except (ComfyError, httpx.HTTPError) as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            if (
                not isinstance(queue, dict)
                or not isinstance(queue.get("queue_running"), list)
                or not isinstance(queue.get("queue_pending"), list)
            ):
                raise HTTPException(
                    status_code=502,
                    detail="ComfyUI queue response is invalid during RayLight recovery",
                )
            if queue["queue_running"] or queue["queue_pending"]:
                raise HTTPException(
                    status_code=409,
                    detail="RayLight recovery requires an empty ComfyUI queue",
                )
            try:
                recovered, backup_path = database.confirm_raylight_runtime_restart(
                    expected_origin,
                    expected_epoch=body.expected_epoch,
                    expected_runtime_state=runtime_state,
                    visible_gpu_count=len(status.available_gpu_indexes),
                )
            except (NativeTemplateError, ValidationError, ValueError) as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            logger.info(
                "RayLight runtime recovery backup created at %s",
                backup_path,
            )
            return _raylight_runtime_status(
                database,
                expected_origin,
                stats,
                state=recovered,
            )
        finally:
            submission_lock.release()

    @app.get("/api/gpus")
    async def get_gpus(request: Request) -> dict[str, Any]:
        settings, authority = _runtime_authority_snapshot(request)
        try:
            stats = await _comfy(request, settings).system_stats()
        except (ComfyError, httpx.HTTPError) as exc:
            raise _invalid_comfy_payload(
                "comfy_system_stats_unavailable",
                str(exc),
            ) from exc
        gpus: list[dict[str, Any]] = []
        for device in _cuda_devices(stats):
            index = int(device["index"])
            gpus.append(
                {
                    "index": index,
                    "name": str(device.get("name") or device.get("type") or f"GPU {index}"),
                    "vram_total": int(device.get("vram_total") or 0),
                    "vram_free": int(device.get("vram_free") or 0),
                    "visible": True,
                }
            )
        _assert_runtime_authority(request, authority)
        return {"gpus": gpus}

    @app.get("/api/system_stats")
    async def get_system_stats(request: Request) -> dict[str, Any]:
        """Transparent ComfyUI system-stats proxy for diagnostics."""
        try:
            stats = await _comfy(request).system_stats()
        except (ComfyError, httpx.HTTPError) as exc:
            raise _invalid_comfy_payload(
                "comfy_system_stats_unavailable",
                str(exc),
            ) from exc
        _cuda_devices(stats)
        return stats

    @app.get("/api/drafts/{mode}")
    async def get_draft(request: Request, mode: GenerationMode) -> Any:
        return _db(request).get_draft(mode)

    @app.put("/api/drafts/{mode}")
    async def put_draft(request: Request, mode: GenerationMode, body: dict[str, Any]) -> Any:
        database = _db(request)
        try:
            draft = validate_mode_draft(mode, body)
            return database.validate_and_put_draft(
                mode,
                draft,
                comfy_origin=str(database.get_settings().comfy_url),
            )
        except (ValidationError, ValueError) as exc:
            raise _validation_error(exc) from exc

    @app.get("/api/timeline", response_model=UnifiedTimelineDraft)
    async def get_timeline(request: Request) -> UnifiedTimelineDraft:
        return _db(request).get_timeline()

    @app.put("/api/timeline", response_model=UnifiedTimelineDraft)
    async def put_timeline(
        request: Request, body: UnifiedTimelineDraft
    ) -> UnifiedTimelineDraft:
        database = _db(request)
        try:
            return database.validate_and_put_timeline(
                body,
                comfy_origin=str(database.get_settings().comfy_url),
            )
        except (ValidationError, ValueError) as exc:
            raise _validation_error(exc) from exc

    @app.get("/api/timeline/authority", response_model=TimelineAuthorityRead)
    async def get_timeline_authority(request: Request) -> TimelineAuthorityRead:
        document, revision = _db(request).get_timeline_authority()
        return TimelineAuthorityRead(document=document, revision=revision)

    @app.put("/api/timeline/authority", response_model=TimelineAuthorityRead)
    async def put_timeline_authority(
        request: Request,
        body: TimelineAuthorityWriteRequest,
    ) -> TimelineAuthorityRead:
        database = _db(request)
        try:
            document, revision = database.validate_and_put_timeline_authority(
                body.document,
                expected_revision=body.expected_revision,
                comfy_origin=str(database.get_settings().comfy_url),
            )
        except TimelineRevisionConflict as exc:
            raise _timeline_revision_conflict(exc) from None
        except TimelineRevisionExhausted as exc:
            raise _timeline_revision_exhausted(exc) from None
        except TimelineComfyOriginConflict as exc:
            raise _timeline_comfy_origin_conflict(exc) from None
        except (ValidationError, ValueError) as exc:
            raise _validation_error(exc) from exc
        return TimelineAuthorityRead(document=document, revision=revision)

    @app.post("/api/timeline/compile", response_model=TimelineCompileRead)
    async def compile_timeline(
        request: Request, body: TimelineJobRequest
    ) -> TimelineCompileRead:
        return await _compile_timeline_report(request, body, project_id=None)

    @app.post("/api/timeline/jobs", response_model=JobRead)
    async def create_timeline_job(
        request: Request, body: TimelineJobRequest
    ) -> JobRead:
        return await _create_timeline_job(request, body)

    # --- Project management (multi-project) ---

    @app.get("/api/projects", response_model=ProjectListRead)
    async def list_projects(request: Request) -> ProjectListRead:
        database = _db(request)
        storage = request.app.state.storage
        active_database_identity = storage.active_database_identity
        projects = database.list_projects()
        if storage.active_database_identity != active_database_identity:
            raise HTTPException(
                status_code=409,
                detail="the active Director database changed while listing projects",
            )
        return ProjectListRead(
            projects=projects,
            active_database_identity=active_database_identity,
        )

    @app.post("/api/projects", response_model=ProjectSummaryRead)
    async def create_project(
        request: Request, body: ProjectCreateRequest
    ) -> ProjectSummaryRead:
        project = _db(request).create_project(body.title)
        return _project_summary(project)

    @app.post("/api/projects/import", response_model=ProjectSummaryRead)
    async def import_project(
        request: Request, body: ProjectImportRequest
    ) -> ProjectSummaryRead:
        project = _db(request).import_project(body.title, body.document)
        return _project_summary(project)

    @app.get("/api/projects/{project_id}", response_model=ProjectSummaryRead)
    async def get_project(request: Request, project_id: str) -> ProjectSummaryRead:
        project = _db(request).get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        return _project_summary(project)

    @app.patch("/api/projects/{project_id}", response_model=ProjectSummaryRead)
    async def rename_project(
        request: Request, project_id: str, body: ProjectRenameRequest
    ) -> ProjectSummaryRead:
        try:
            project = _db(request).rename_project(project_id, body.title)
        except KeyError:
            raise HTTPException(status_code=404, detail="project not found") from None
        except ValueError as exc:
            raise _validation_error(exc) from exc
        return _project_summary(project)

    @app.delete("/api/projects/{project_id}", response_model=ProjectDeleteRead)
    async def delete_project(
        request: Request, project_id: str
    ) -> ProjectDeleteRead:
        database = _db(request)
        remaining = database.list_projects()
        if len(remaining) <= 1 and any(
            project["id"] == project_id for project in remaining
        ):
            raise HTTPException(
                status_code=409, detail="cannot delete the last project"
            )
        try:
            orphaned = database.delete_project(project_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="project not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return ProjectDeleteRead(
            deleted_project_id=project_id, orphaned_jobs=orphaned
        )

    @app.get(
        "/api/projects/{project_id}/timeline",
        response_model=UnifiedTimelineDraft,
    )
    async def get_project_timeline(
        request: Request, project_id: str
    ) -> UnifiedTimelineDraft:
        try:
            return _db(request).get_project_timeline(project_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="project not found") from None

    @app.put(
        "/api/projects/{project_id}/timeline",
        response_model=UnifiedTimelineDraft,
    )
    async def put_project_timeline(
        request: Request, project_id: str, body: UnifiedTimelineDraft
    ) -> UnifiedTimelineDraft:
        database = _db(request)
        try:
            return database.validate_and_put_project_timeline(
                project_id,
                body,
                comfy_origin=str(database.get_settings().comfy_url),
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="project not found") from None
        except (ValidationError, ValueError) as exc:
            raise _validation_error(exc) from exc

    @app.get(
        "/api/projects/{project_id}/timeline/authority",
        response_model=TimelineAuthorityRead,
    )
    async def get_project_timeline_authority(
        request: Request, project_id: str
    ) -> TimelineAuthorityRead:
        try:
            document, revision = _db(request).get_project_timeline_authority(
                project_id
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="project not found") from None
        return TimelineAuthorityRead(document=document, revision=revision)

    @app.put(
        "/api/projects/{project_id}/timeline/authority",
        response_model=TimelineAuthorityRead,
    )
    async def put_project_timeline_authority(
        request: Request,
        project_id: str,
        body: TimelineAuthorityWriteRequest,
    ) -> TimelineAuthorityRead:
        database = _db(request)
        try:
            document, revision = (
                database.validate_and_put_project_timeline_authority(
                    project_id,
                    body.document,
                    expected_revision=body.expected_revision,
                    comfy_origin=str(database.get_settings().comfy_url),
                )
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="project not found") from None
        except TimelineRevisionConflict as exc:
            raise _timeline_revision_conflict(exc) from None
        except TimelineRevisionExhausted as exc:
            raise _timeline_revision_exhausted(exc) from None
        except TimelineComfyOriginConflict as exc:
            raise _timeline_comfy_origin_conflict(exc) from None
        except (ValidationError, ValueError) as exc:
            raise _validation_error(exc) from exc
        return TimelineAuthorityRead(document=document, revision=revision)

    @app.post(
        "/api/projects/{project_id}/compile",
        response_model=TimelineCompileRead,
    )
    async def compile_project_timeline(
        request: Request, project_id: str, body: TimelineJobRequest
    ) -> TimelineCompileRead:
        return await _compile_timeline_report(
            request, body, project_id=project_id
        )

    @app.post(
        "/api/projects/{project_id}/jobs",
        response_model=JobRead,
    )
    async def create_project_job(
        request: Request, project_id: str, body: TimelineJobRequest
    ) -> JobRead:
        if _db(request).get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        return await _create_timeline_job(request, body, project_id=project_id)

    @app.get("/api/assets", response_model=AssetListRead)
    async def list_assets(
        request: Request, kind: AssetKind | None = None
    ) -> AssetListRead:
        database = _db(request)
        settings = database.get_settings()
        if not str(settings.comfy_url).strip():
            raise HTTPException(status_code=409, detail=_COMFY_NOT_CONFIGURED)
        comfy_origin = Database.canonical_comfy_origin(settings.comfy_url)
        active_database_identity = request.app.state.storage.active_database_identity
        try:
            assets = database.list_assets(
                comfy_origin=comfy_origin, kind=kind
            )
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return AssetListRead(
            assets=assets,
            active_database_identity=active_database_identity,
            comfy_origin=comfy_origin,
        )

    @app.get("/api/asset-trash", response_model=AssetTrashListRead)
    async def list_asset_trash(request: Request) -> AssetTrashListRead:
        database = _db(request)
        settings = database.get_settings()
        if not str(settings.comfy_url).strip():
            raise HTTPException(status_code=409, detail=_COMFY_NOT_CONFIGURED)
        comfy_origin = Database.canonical_comfy_origin(settings.comfy_url)
        active_database_identity = request.app.state.storage.active_database_identity
        try:
            batches = database.list_asset_trash_batches(
                comfy_origin=comfy_origin
            )
        except AssetTrashOriginConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "asset_trash_origin_conflict",
                    "message": str(exc),
                    "remote_files_preserved": True,
                },
            ) from exc
        return AssetTrashListRead(
            batches=[AssetTrashBatchRead.model_validate(item) for item in batches],
            active_database_identity=active_database_identity,
            comfy_origin=comfy_origin,
        )

    @app.post("/api/asset-trash", response_model=AssetTrashBatchRead)
    async def trash_assets(
        request: Request, body: AssetTrashRequest
    ) -> AssetTrashBatchRead:
        database = _db(request)
        settings = database.get_settings()
        if not str(settings.comfy_url).strip():
            raise HTTPException(status_code=409, detail=_COMFY_NOT_CONFIGURED)
        try:
            result = database.trash_assets(
                body.asset_ids,
                cascade=body.cascade,
                expected_comfy_origin=str(settings.comfy_url),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="asset not found") from exc
        except AssetTrashInUse as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "assets_in_use",
                    "message": "one or more assets are still referenced by saved drafts",
                    "usages": exc.usages,
                    "usages_by_asset": exc.usages_by_asset,
                    "remote_files_preserved": True,
                },
            ) from exc
        except AssetTrashOriginConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "asset_trash_origin_conflict",
                    "message": str(exc),
                    "remote_files_preserved": True,
                },
            ) from exc
        except OverflowError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return AssetTrashBatchRead.model_validate(result)

    @app.post(
        "/api/asset-trash/{batch_id}/restore",
        response_model=AssetTrashRestoreRead,
    )
    async def restore_asset_trash(
        request: Request,
        batch_id: str,
        body: AssetTrashRestoreRequest,
    ) -> AssetTrashRestoreRead:
        database = _db(request)
        settings = database.get_settings()
        if not str(settings.comfy_url).strip():
            raise HTTPException(status_code=409, detail=_COMFY_NOT_CONFIGURED)
        try:
            result = database.restore_asset_trash_batch(
                batch_id,
                mode=body.mode,
                expected_comfy_origin=str(settings.comfy_url),
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="asset trash batch not found"
            ) from exc
        except AssetTrashOriginConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "asset_trash_origin_conflict",
                    "message": str(exc),
                    "remote_files_preserved": True,
                },
            ) from exc
        except AssetTrashRestoreConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "asset_trash_restore_conflict",
                    "message": str(exc),
                    "conflicts": exc.conflicts,
                    "remote_files_preserved": True,
                },
            ) from exc
        return AssetTrashRestoreRead.model_validate(result)

    @app.delete(
        "/api/asset-trash/{batch_id}", response_model=AssetTrashPurgeRead
    )
    async def purge_asset_trash(
        request: Request, batch_id: str
    ) -> AssetTrashPurgeRead:
        database = _db(request)
        settings = database.get_settings()
        if not str(settings.comfy_url).strip():
            raise HTTPException(status_code=409, detail=_COMFY_NOT_CONFIGURED)
        try:
            result = database.purge_asset_trash_batch(
                batch_id, expected_comfy_origin=str(settings.comfy_url)
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="asset trash batch not found"
            ) from exc
        except AssetTrashOriginConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "asset_trash_origin_conflict",
                    "message": str(exc),
                    "remote_files_preserved": True,
                },
            ) from exc
        except AssetTrashRestoreConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "asset_trash_purge_conflict",
                    "message": str(exc),
                    "conflicts": exc.conflicts,
                    "remote_files_preserved": True,
                },
            ) from exc
        return AssetTrashPurgeRead.model_validate(result)

    @app.get("/api/uploads/{upload_id}")
    async def get_upload_progress(request: Request, upload_id: str) -> dict[str, Any]:
        normalized_id = upload_id.strip().lower()
        if not _UPLOAD_ID.fullmatch(normalized_id):
            raise HTTPException(status_code=404, detail="upload progress not found")
        if normalized_id not in request.app.state.upload_progress:
            raise HTTPException(status_code=404, detail="upload progress not found")
        _upload_progress(request.app, normalized_id)
        snapshot = request.app.state.upload_progress.get(normalized_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="upload progress not found")
        return {key: value for key, value in snapshot.items() if key != "touched_at"}

    @app.post("/api/assets")
    async def upload_asset(
        request: Request,
        file: Annotated[UploadFile, File()],
        kind: Annotated[AssetKind, Form()],
        upload_id: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        filename = _safe_filename(file.filename)
        normalized_upload_id = upload_id.strip().lower() if upload_id else None
        if normalized_upload_id is not None and not _UPLOAD_ID.fullmatch(
            normalized_upload_id
        ):
            raise HTTPException(status_code=422, detail="upload_id must be a UUIDv4")
        started_at = time.monotonic()
        phase_started = started_at
        timings: dict[str, float] = {}
        input_bytes = 0
        output_bytes = 0
        strategy = "passthrough"
        _upload_progress(
            request.app,
            normalized_upload_id,
            stage="processing",
            input_bytes=0,
            output_bytes=0,
        )
        try:
            settings = _db(request).get_settings()
            comfy = _comfy(request, settings)
            content_type = _validate_upload_metadata(kind, filename, file.content_type)
            with tempfile.TemporaryDirectory(prefix="director-upload-") as directory:
                source_path = Path(directory) / f"source{Path(filename).suffix or '.bin'}"
                input_bytes = await _spool_upload_limited(
                    file, source_path, _UPLOAD_LIMITS[kind]
                )
                with source_path.open("rb") as stream:
                    _validate_upload_signature(filename, stream.read(64))
                timings["receive"] = time.monotonic() - phase_started
                phase_started = time.monotonic()
                metadata: VideoMetadata | None = None
                upload_path = source_path
                _upload_progress(
                    request.app,
                    normalized_upload_id,
                    stage="analyzing" if kind == "video" else "forwarding",
                    input_bytes=input_bytes,
                )
                if kind == "video":
                    proxy_path = Path(directory) / "proxy.mp4"
                    try:
                        proxy = await anyio.to_thread.run_sync(
                            partial(create_24fps_proxy_file, source_path, proxy_path)
                        )
                    except MediaToolError as exc:
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                "video could not be normalized to the required "
                                f"24fps proxy: {exc}"
                            ),
                        ) from exc
                    strategy = proxy.strategy
                    timings["normalize"] = time.monotonic() - phase_started
                    phase_started = time.monotonic()
                    filename = f"{Path(filename).stem}_24fps.mp4"
                    upload_path = proxy_path
                    content_type = "video/mp4"
                    metadata = proxy.metadata
                output_bytes = upload_path.stat().st_size
                _upload_progress(
                    request.app,
                    normalized_upload_id,
                    stage="forwarding",
                    strategy=strategy,
                    output_bytes=output_bytes,
                )
                try:
                    uploaded = await comfy.upload(
                        filename, upload_path, content_type, kind
                    )
                except (ComfyError, httpx.HTTPError) as exc:
                    raise HTTPException(status_code=502, detail=str(exc)) from exc
                timings["forward"] = time.monotonic() - phase_started
                phase_started = time.monotonic()
                content_hash = await anyio.to_thread.run_sync(_hash_file, upload_path)
                name = str(uploaded["name"])
                subfolder = str(uploaded.get("subfolder") or "")
                path = f"{subfolder.strip('/')}/{name}".strip("/")
                asset_id = str(uuid.uuid4())
                document = {
                    "name": name,
                    "subfolder": subfolder,
                    "type": "input",
                    "kind": kind,
                    "id": asset_id,
                    "filename": name,
                    "path": path,
                    "preview_url": public_api_url(f"/api/assets/{asset_id}/preview"),
                    "content_hash": content_hash,
                    "metadata": (
                        metadata.model_dump(mode="json")
                        if metadata is not None
                        else None
                    ),
                }
                asset = AssetReference.model_validate(document).model_dump(mode="json")
                if not _db(request).put_asset_if_current_origin(
                    asset_id,
                    asset,
                    expected_comfy_origin=Database.canonical_comfy_origin(
                        settings.comfy_url
                    ),
                ):
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "code": "asset_upload_origin_changed",
                            "message": (
                                "ComfyUI endpoint changed while the upload was in "
                                "progress; the remote file was preserved but was not "
                                "registered in Director"
                            ),
                            "remote_files_preserved": True,
                        },
                    )
                timings["persist"] = time.monotonic() - phase_started
                elapsed = time.monotonic() - started_at
                _upload_progress(
                    request.app,
                    normalized_upload_id,
                    stage="complete",
                    strategy=strategy,
                    input_bytes=input_bytes,
                    output_bytes=output_bytes,
                    elapsed_seconds=round(elapsed, 3),
                )
                logger.info(
                    "asset_upload_complete kind=%s input_bytes=%d output_bytes=%d "
                    "strategy=%s receive_seconds=%.3f normalize_seconds=%.3f "
                    "forward_seconds=%.3f persist_seconds=%.3f total_seconds=%.3f",
                    kind,
                    input_bytes,
                    output_bytes,
                    strategy,
                    timings.get("receive", 0.0),
                    timings.get("normalize", 0.0),
                    timings.get("forward", 0.0),
                    timings.get("persist", 0.0),
                    elapsed,
                )
                return {"asset": asset}
        except HTTPException as exc:
            elapsed = time.monotonic() - started_at
            _upload_progress(
                request.app,
                normalized_upload_id,
                stage="failed",
                strategy=strategy,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                elapsed_seconds=round(elapsed, 3),
            )
            logger.warning(
                "asset_upload_rejected kind=%s status=%d input_bytes=%d "
                "output_bytes=%d strategy=%s total_seconds=%.3f",
                kind,
                exc.status_code,
                input_bytes,
                output_bytes,
                strategy,
                elapsed,
            )
            raise
        except Exception:
            elapsed = time.monotonic() - started_at
            _upload_progress(
                request.app,
                normalized_upload_id,
                stage="failed",
                strategy=strategy,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                elapsed_seconds=round(elapsed, 3),
            )
            logger.exception(
                "asset_upload_failed kind=%s input_bytes=%d output_bytes=%d "
                "strategy=%s total_seconds=%.3f",
                kind,
                input_bytes,
                output_bytes,
                strategy,
                elapsed,
            )
            raise
        finally:
            await file.close()

    @app.delete("/api/assets/{asset_id}", response_model=AssetDeleteRead)
    async def delete_asset(
        request: Request, asset_id: str, cascade: bool = False
    ) -> AssetDeleteRead:
        database = _db(request)
        settings = database.get_settings()
        if not str(settings.comfy_url).strip():
            raise HTTPException(status_code=409, detail=_COMFY_NOT_CONFIGURED)
        try:
            record = database.get_asset_record(asset_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if record is None:
            raise HTTPException(status_code=404, detail="asset not found")
        _document, asset_origin = record
        active_origin = Database.canonical_comfy_origin(settings.comfy_url)
        if asset_origin != active_origin:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"asset id '{asset_id}' belongs to ComfyUI '{asset_origin}', "
                    f"not the active endpoint '{active_origin}'"
                ),
            )
        try:
            usages = database.delete_asset_if_unused(
                asset_id,
                cascade=cascade,
                expected_comfy_origin=active_origin,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="asset not found") from exc
        except AssetTrashOriginConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "asset_trash_origin_conflict",
                    "message": str(exc),
                    "outputs_preserved": True,
                    "remote_files_preserved": True,
                },
            ) from exc
        if usages and not cascade:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "asset is still referenced by saved drafts",
                    "usages": usages,
                    "outputs_preserved": True,
                },
            )
        return AssetDeleteRead(
            deleted_asset_id=asset_id,
            unbound_usages=usages if cascade else [],
        )

    @app.get("/api/assets/{asset_id}/preview")
    async def preview_asset(request: Request, asset_id: str) -> Response:
        database = _db(request)
        record = database.get_asset_record(asset_id)
        if record is None:
            raise HTTPException(status_code=404, detail="asset not found")
        asset, comfy_origin = record
        origin_settings = _origin_settings(database.get_settings(), comfy_origin)
        try:
            return await _proxy_comfy_media(
                _comfy(request, origin_settings),
                {"filename": asset["name"], "subfolder": asset["subfolder"], "type": asset["type"]},
                filename=asset["name"],
                cache_control="private, max-age=60",
                byte_range=request.headers.get("range"),
            )
        except (ComfyError, httpx.HTTPError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/rv2v/detect-shots", response_model=DetectShotsResponse)
    async def detect_rv2v_shots(
        request: Request, body: DetectShotsRequest
    ) -> DetectShotsResponse:
        """Proxy smart splitting without accepting a client-supplied ComfyUI path."""

        database = _db(request)
        settings = database.get_settings()
        # Constructing the active client first preserves the same explicit
        # first-run guard as uploads and jobs.
        comfy = _comfy(request, settings)
        try:
            record = database.get_asset_record(body.asset_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if record is None:
            raise HTTPException(status_code=404, detail="asset not found")
        document, asset_origin = record
        active_origin = Database.canonical_comfy_origin(settings.comfy_url)
        if asset_origin != active_origin:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"asset id '{body.asset_id}' belongs to ComfyUI '{asset_origin}', "
                    f"not the active endpoint '{active_origin}'"
                ),
            )
        try:
            asset = AssetReference.model_validate(document)
        except ValidationError as exc:
            raise HTTPException(status_code=409, detail="stored asset is invalid") from exc
        if asset.kind != "video":
            raise HTTPException(status_code=422, detail="shot detection requires a video asset")
        assert asset.metadata is not None
        total_frames = max(1, int(round(asset.metadata.duration * body.frame_rate)))
        try:
            upstream = await comfy.view(
                {
                    "filename": asset.name,
                    "subfolder": asset.subfolder,
                    "type": asset.type,
                }
            )
            return await anyio.to_thread.run_sync(
                partial(
                    detect_shots_bytes,
                    upstream.content,
                    suffix=Path(asset.name).suffix or ".mp4",
                    frame_rate=body.frame_rate,
                    total_frames=total_frames,
                    sensitivity=body.sensitivity,
                    min_shot_frames=body.min_shot_frames,
                )
            )
        except (ComfyError, httpx.HTTPError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except (MediaToolError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/jobs", response_model=JobRead)
    async def create_job(
        request: Request, body: CreateJobRequest | TimelineJobRequest
    ) -> JobRead:
        # Generic job creation accepts the new unified contract as well as the
        # legacy six-mode body.  The explicit /api/timeline/jobs route remains
        # the clearest client API, while this keeps automation on /api/jobs.
        if isinstance(body, TimelineJobRequest):
            return await _create_timeline_job(request, body)
        database = _db(request)
        settings = database.get_settings()
        # Guard before compiling or inserting a local job so first-run
        # configuration mistakes never leave orphaned ``preparing`` rows.
        client = _comfy(request, settings)
        try:
            draft = (
                validate_mode_draft(body.mode, body.config)
                if body.config is not None
                else database.get_draft(body.mode)
            )
            # Preserve the declared six-recipe semantics at the legacy API
            # boundary. Converting an unfinished I2V/FL2V/R2V/V2V draft into
            # the v2 family shape first could otherwise make missing media
            # derive a different runnable recipe (for example I2V -> T2V).
            validate_runnable(draft)
            database.validate_draft_assets(draft, comfy_origin=str(settings.comfy_url))
        except (DraftNotRunnable, ValidationError, ValueError) as exc:
            raise _validation_error(exc) from exc
        # Legacy clients retain their request/response mode, but the hidden
        # execution path is the same native timeline compiler and child-job
        # orchestrator. No endpoint can submit a Director workflow anymore.
        timeline = mode_draft_to_timeline(draft, title=f"旧版 {body.mode} 任务")
        return await _create_timeline_job(
            request,
            TimelineJobRequest(config=timeline),
            parent_mode=body.mode,
        )

    @app.get("/api/tasks/events")
    async def task_events(request: Request) -> StreamingResponse:
        """Server-sent events that invalidate the browser's task list.

        The frontend replaces its fixed 2.5s poll with this long-lived stream:
        each refresh event is only a hint to perform its existing single-flight
        listTasks read (SQLite-only). The stream wakes on websocket progress/
        terminal frames, the reconciler's active-job pass, and the terminal
        wait-gate hints, then coalesces bursts to at most one refresh per
        second. A heartbeat keeps idle proxies from closing the connection;
        the client keeps a coarse fallback timer for reconnects.
        """

        async def stream():
            loop = asyncio.get_running_loop()
            min_interval = 1.0
            heartbeat_interval = 15.0
            max_lifetime = max(
                0.01,
                float(request.app.state.task_events_max_lifetime_seconds),
            )
            deadline = loop.time() + max_lifetime
            last_emit = 0.0
            while True:
                if await request.is_disconnected():
                    return
                lifetime_remaining = deadline - loop.time()
                if lifetime_remaining <= 0:
                    return
                try:
                    await asyncio.wait_for(
                        request.app.state.task_change_event.wait(),
                        timeout=min(heartbeat_interval, lifetime_remaining),
                    )
                except asyncio.TimeoutError:
                    if loop.time() >= deadline:
                        return
                    yield ": heartbeat\n\n"
                    continue
                request.app.state.task_change_event.clear()
                now = loop.time()
                remaining = min_interval - (now - last_emit)
                if remaining > 0:
                    await asyncio.sleep(remaining)
                request.app.state.task_change_event.clear()
                last_emit = loop.time()
                yield "event: refresh\ndata: {}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/jobs", response_model=JobListRead)
    async def list_jobs(
        request: Request,
        limit: Annotated[int, Query(ge=1, le=256)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
        status: Annotated[list[JobStatus] | None, Query()] = None,
        q: Annotated[str | None, Query(max_length=256)] = None,
        project_id: Annotated[str | None, Query(max_length=128)] = None,
        sort_by: Literal["created_at", "execution_duration"] = "created_at",
        sort_order: Literal["asc", "desc"] = "desc",
    ) -> JobListRead:
        database = _db(request)
        page, total = database.list_jobs_page(
            limit=limit,
            offset=offset,
            statuses=tuple(status or ()),
            search=q,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        children_by_job = database.list_job_children_for_jobs(
            [str(snapshot["id"]) for snapshot in page]
        )
        jobs: list[dict[str, Any]] = []
        for snapshot in page:
            # GET is deliberately SQLite-only.  A black-hole ComfyUI endpoint
            # must never freeze the task panel; the managed reconciler owns all
            # queue/history I/O and persists the next observable snapshot.
            snapshot["children"] = children_by_job[str(snapshot["id"])]
            jobs.append(snapshot)
        active_project_id = project_id or database.LEGACY_DEFAULT_PROJECT_ID
        try:
            current_timeline = database.get_project_timeline(active_project_id)
        except KeyError:
            current_timeline = None
        current_settings = database.get_settings()
        return JobListRead(
            jobs=[
                _job_read_for_request(
                    request,
                    job,
                    current_timeline=current_timeline,
                    current_settings=current_settings,
                )
                for job in jobs
            ],
            total=total,
            limit=limit,
            offset=offset,
            has_more=offset + len(jobs) < total,
            summary=database.job_status_summary(),
        )

    @app.get("/api/jobs/{job_id}", response_model=JobRead)
    async def get_job(request: Request, job_id: str) -> JobRead:
        database = _db(request)
        job = database.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        job["children"] = database.list_job_children(job_id)
        return _job_read_for_request(request, job)

    @app.get("/api/jobs/{job_id}/diagnostic", response_model=JobDiagnosticRead)
    async def get_job_diagnostic(
        request: Request, job_id: str
    ) -> JobDiagnosticRead:
        """Export a local, deliberately redacted task diagnostic.

        The response schema has no settings snapshot, typed creative config,
        server prompt, workflow graph, or ComfyUI prompt identifiers. It is
        therefore safe for the task menu's "export diagnostic" action without
        turning a browser history view into a workflow exfiltration endpoint.
        """

        database = _db(request)
        job = database.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        job["children"] = database.list_job_children(job_id)
        public = _job_read_for_request(request, job)
        output_files = list(public.output_files)
        output_files.extend(
            result.output_file
            for result in public.segment_results
            if result.output_file not in output_files
        )
        return JobDiagnosticRead(
            id=public.id,
            display_name=public.display_name,
            project_title=public.project_title,
            mode=public.mode,
            status=public.status,
            progress=public.progress,
            stage=public.stage,
            created_at=public.created_at,
            updated_at=public.updated_at,
            started_at=public.started_at,
            completed_at=public.completed_at,
            execution_duration_seconds=public.execution_duration_seconds,
            output_files=output_files,
            error_summary=public.error_summary,
            children=[
                {
                    "id": child.id,
                    "family": child.family,
                    "backend": child.backend,
                    "segment_ids": child.segment_ids,
                    "status": child.status,
                    "progress": child.progress,
                    "stage": child.stage,
                    "output_files": child.outputs,
                    "error_summary": _error_summary(child.error),
                }
                for child in public.children
            ],
        )

    @app.get(
        "/api/jobs/{job_id}/generation-details",
        response_model=JobGenerationDetailsRead,
    )
    async def get_job_generation_details(
        request: Request, job_id: str
    ) -> JobGenerationDetailsRead:
        """Return a typed, path-safe view of the immutable generation snapshot.

        This is intentionally separate from the job-list response so task
        history stays lightweight. It never returns the ComfyUI endpoint,
        client ID, workflow graph, prompt IDs, or asset/output paths.
        """

        database = _db(request)
        job = database.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        details = _job_generation_details(
            job,
            database.list_job_children(job_id),
        )
        if details is None:
            raise HTTPException(
                status_code=409,
                detail="this historical task has no compatible generation snapshot",
            )
        return details

    @app.get("/api/jobs/{job_id}/project", response_model=JobProjectSnapshotRead)
    async def get_job_project(
        request: Request, job_id: str
    ) -> JobProjectSnapshotRead:
        """Return only the typed source project, never runtime or workflow data."""

        job = _db(request).get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        project = _job_timeline_snapshot(job)
        if project is None:
            raise HTTPException(
                status_code=409,
                detail="this historical task has no compatible project snapshot",
            )
        snapshot = job.get("config_snapshot")
        raw_segment_ids = (
            snapshot.get("segment_ids") if isinstance(snapshot, dict) else None
        )
        segment_ids = (
            [str(segment_id) for segment_id in raw_segment_ids]
            if isinstance(raw_segment_ids, list)
            else None
        )
        return JobProjectSnapshotRead(
            job_id=job_id,
            project=project,
            segment_ids=segment_ids,
        )

    @app.post(
        "/api/jobs/{job_id}/import-output",
        response_model=JobOutputImportRead,
    )
    async def import_job_output(
        request: Request,
        job_id: str,
        body: JobOutputImportRequest,
    ) -> JobOutputImportRead:
        """Copy one persisted generated video into the active input library.

        The browser supplies only a parent-local output index or a stable
        timeline segment ID. Source paths are resolved from durable Director
        rows and read through the job's immutable ComfyUI origin; arbitrary
        client paths and URLs never cross this boundary.
        """

        database = _db(request)
        job = database.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        job["children"] = database.list_job_children(job_id)
        target_settings = database.get_settings()
        # Preserve the normal explicit first-run guard before doing any source
        # download or potentially expensive 24fps normalization.
        _comfy(request, target_settings)
        try:
            asset = await import_job_output_as_asset(
                registry=database,
                comfy_factory=request.app.state.comfy_factory,
                job=job,
                target_settings=target_settings,
                current_settings=database.get_settings,
                output_index=body.output_index,
                segment_id=body.segment_id,
            )
        except TaskManagementError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except (MediaToolError, ValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (ComfyError, httpx.HTTPError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return JobOutputImportRead(asset=asset)

    @app.delete("/api/jobs/{job_id}", response_model=JobDeleteRead)
    async def delete_job(request: Request, job_id: str) -> JobDeleteRead:
        """Forget one terminal Director job without deleting ComfyUI media.

        Generated files belong to the ComfyUI endpoint captured by the job's
        immutable settings snapshot.  ComfyUI's standard HTTP API has no safe,
        portable output-file delete operation, so this endpoint intentionally
        removes only the Director SQLite record and its proxy URLs.
        """

        database = _db(request)
        job = database.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        dispatcher = request.app.state.submission_jobs.get(job_id)
        if dispatcher is not None and not dispatcher.done():
            raise HTTPException(
                status_code=409,
                detail="job submission is still shutting down; retry shortly",
            )
        if job["status"] not in _TERMINAL_STATUSES:
            raise HTTPException(
                status_code=409,
                detail="only terminal jobs can be deleted; cancel the job first",
            )
        try:
            deleted = database.delete_job_if_status(job_id, job["status"])
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        if not deleted:
            raise HTTPException(
                status_code=409,
                detail="job status changed while it was being deleted; retry after refreshing",
            )
        request.app.state.live_preview_cache.discard(job_id)
        return JobDeleteRead(deleted_job_id=job_id)

    @app.delete("/api/jobs", response_model=JobClearRead)
    async def clear_terminal_jobs(request: Request) -> JobClearRead:
        """Clear all locally known terminal jobs and leave active jobs intact."""

        live_submission_job_ids = {
            job_id
            for job_id, task in request.app.state.submission_jobs.items()
            if not task.done()
        }
        deleted_count, active_count = _db(request).delete_terminal_jobs(
            _TERMINAL_STATUS_ORDER,
            excluded_job_ids=live_submission_job_ids,
        )
        return JobClearRead(
            deleted_count=deleted_count,
            active_count=active_count,
        )

    @app.post(
        "/api/jobs/{job_id}/recovery/confirm-comfy-restart",
        response_model=JobRead,
    )
    async def confirm_job_comfy_restart(
        request: Request,
        job_id: str,
        body: JobRecoveryConfirmComfyRestartRequest,
    ) -> JobRead:
        """End only an ambiguous restart-owned submission after operator proof.

        ``body.confirmation`` is intentionally a strict literal rather than a
        checkbox boolean.  The endpoint never contacts ComfyUI and never uses
        queue/history absence as evidence: the operator is certifying that the
        previous ComfyUI process, which could still have completed an old
        ``POST /prompt``, no longer exists.
        """

        # Keep the parsed body named and visible here.  Pydantic has already
        # enforced the one accepted literal and StrictModel rejects extras.
        assert body.confirmation == "comfyui_process_restarted"
        dispatcher = request.app.state.submission_jobs.get(job_id)
        if dispatcher is not None and not dispatcher.done():
            raise HTTPException(
                status_code=409,
                detail=(
                    "job submission is still owned by this Director process; "
                    "wait for it to stop before confirming a ComfyUI restart"
                ),
            )
        try:
            settled = _db(request).confirm_comfy_restart_recovery(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        request.app.state.live_preview_cache.discard(job_id)
        settled["children"] = _db(request).list_job_children(job_id)
        return _job_read_for_request(request, settled)

    @app.post("/api/jobs/{job_id}/cancel", response_model=JobRead)
    async def cancel_job(request: Request, job_id: str) -> JobRead:
        database = _db(request)
        job = database.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if database.list_job_children(job_id):
            initial_cancel_claimed = False
            if job["status"] not in _TERMINAL_STATUSES:
                marked, initial_cancel_claimed = database.mark_job_cancel_requested(
                    job_id
                )
                if marked is not None:
                    job = marked
                else:
                    latest = database.get_job(job_id)
                    if latest is None:
                        raise HTTPException(status_code=404, detail="job not found")
                    job = latest
            return _job_read_for_request(
                request,
                await _cancel_timeline_job(
                    request,
                    job,
                    initial_cancel_claimed=initial_cancel_claimed,
                ),
            )
        if job["status"] in _TERMINAL_STATUSES:
            return _job_read_for_request(request, job)
        job = await _sync_existing_job(request, job)
        if job["status"] in _TERMINAL_STATUSES:
            return _job_read_for_request(request, job)
        if job["status"] == "preparing" and not job.get("prompt_id"):
            job = database.update_job_if_status(
                job_id,
                "preparing",
                status="cancelled",
                progress=1.0,
                stage="cancelled",
                completed_at=utc_now(),
            )
            if job is not None:
                return _job_read_for_request(request, job)
            latest = database.get_job(job_id)
            if latest is None:
                raise HTTPException(status_code=404, detail="job not found")
            job = latest
            if job["status"] in _TERMINAL_STATUSES:
                return _job_read_for_request(request, job)
        while job["status"] != "cancelling":
            if job["status"] in _TERMINAL_STATUSES:
                return _job_read_for_request(request, job)
            transitioned = database.update_job_if_status(
                job_id,
                job["status"],
                status="cancelling",
                stage="cancelling",
                error=None,
                completed_at=None,
            )
            if transitioned is not None:
                job = transitioned
                break
            latest = database.get_job(job_id)
            if latest is None:
                raise HTTPException(status_code=404, detail="job not found")
            job = latest
        if job.get("prompt_id"):
            try:
                client = _comfy(request, RuntimeSettings.model_validate(job["settings_snapshot"]))
                # The local ComfyUI endpoint atomically classifies this prompt
                # as running/pending/terminal and performs exactly the matching
                # action. ComfyClient retains a two-call fallback only for an
                # older server that returns 404 for the endpoint itself.
                cancel_dispatched = await client.cancel(job["prompt_id"])
                if not cancel_dispatched:
                    # The prompt may have completed after the pre-cancel sync
                    # but before ComfyUI classified it. Reconcile once more so
                    # a successful job is not overwritten as cancelled.
                    latest = database.get_job(job_id)
                    if latest is not None:
                        job = await _sync_existing_job(request, latest)
                        if job["status"] in _TERMINAL_STATUSES:
                            return _job_read_for_request(request, job)
                    job = database.update_job_if_status(
                        job_id,
                        "cancelling",
                        stage="cancel_unconfirmed",
                    ) or job
                    return _job_read_for_request(request, job)
            except (ComfyError, httpx.HTTPError) as exc:
                job = database.update_job_if_status(
                    job_id,
                    "cancelling",
                    stage="cancel_failed",
                    error=str(exc),
                )
                if job is None:
                    job = database.get_job(job_id)
                    if job is None:
                        raise HTTPException(status_code=404, detail="job not found")
                return _job_read_for_request(request, job)
        job = database.update_job_if_status(
            job_id,
            "cancelling",
            status="cancelled",
            progress=1.0,
            stage="cancelled",
            error=None,
            completed_at=utc_now(),
        )
        if job is None:
            job = database.get_job(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="job not found")
        return _job_read_for_request(request, job)

    @app.post("/api/jobs/cancel", response_model=JobBulkCancelRead)
    async def cancel_jobs(
        request: Request, body: JobBulkCancelRequest
    ) -> JobBulkCancelRead:
        """Cancel an explicit set of locally-owned parent tasks.

        Every ID is validated before the first upstream side effect. Each task
        then uses the same exact-ID cancellation path as the single-task API;
        this endpoint never invokes ComfyUI's global queue clear or interrupt.
        """

        database = _db(request)
        missing = [job_id for job_id in body.job_ids if database.get_job(job_id) is None]
        if missing:
            raise HTTPException(
                status_code=404,
                detail="job not found: " + ", ".join(missing),
            )
        results: list[JobRead] = []
        for job_id in body.job_ids:
            results.append(await cancel_job(request, job_id))
        return JobBulkCancelRead(
            jobs=results,
            requested_count=len(body.job_ids),
            terminal_count=sum(
                result.status in _TERMINAL_STATUSES for result in results
            ),
        )

    @app.get("/api/jobs/{job_id}/live-preview")
    async def get_job_live_preview(request: Request, job_id: str) -> Response:
        """Return the latest authenticated sampler preview from process memory."""

        database = _db(request)
        job = database.get_job(job_id)
        if job is None or job["status"] in _TERMINAL_STATUSES:
            raise HTTPException(status_code=404, detail="live preview not found")
        preview = _live_preview_for_job(request, job)
        if preview is None:
            raise HTTPException(status_code=404, detail="live preview not found")
        return Response(
            content=preview.content,
            media_type=preview.mime_type,
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/api/jobs/{job_id}/segment-output")
    async def get_job_segment_output(
        request: Request,
        job_id: str,
        segment_id: Annotated[str, Query(min_length=1, max_length=128)],
    ) -> Response:
        """Proxy one generated take selected by stable timeline segment ID."""

        database = _db(request)
        job = database.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        # Segment takes are durable child-row outputs. Do not trigger queue,
        # history, or whole-timeline assembly merely to proxy an existing take.
        job["children"] = database.list_job_children(job_id)
        matches = _segment_output_candidates(job).get(segment_id, [])
        if not matches:
            raise HTTPException(status_code=404, detail="segment output not found")
        if len(matches) != 1:
            raise HTTPException(
                status_code=409,
                detail="segment output is ambiguous; refresh or rerun the segment",
            )
        _child, output = matches[0]
        try:
            return await _proxy_comfy_media(
                _comfy(request, RuntimeSettings.model_validate(job["settings_snapshot"])),
                {
                    "filename": output["filename"],
                    "subfolder": output.get("subfolder", ""),
                    "type": output.get("type", "output"),
                },
                filename=str(output["filename"]),
                byte_range=request.headers.get("range"),
            )
        except (ComfyError, httpx.HTTPError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/outputs/{index}")
    async def get_job_output(
        request: Request,
        job_id: str,
        index: Annotated[int, ApiPath(ge=0)],
    ) -> Response:
        database = _db(request)
        job = database.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if index >= len(job["outputs"]):
            raise HTTPException(status_code=404, detail="job output not found")
        output = job["outputs"][index]
        try:
            return await _proxy_comfy_media(
                _comfy(request, RuntimeSettings.model_validate(job["settings_snapshot"])),
                {
                    "filename": output["filename"],
                    "subfolder": output.get("subfolder", ""),
                    "type": output.get("type", "output"),
                },
                filename=str(output["filename"]),
                byte_range=request.headers.get("range"),
            )
        except (ComfyError, httpx.HTTPError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


app = create_app()
