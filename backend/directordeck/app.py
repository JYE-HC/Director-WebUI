from __future__ import annotations

import asyncio
from collections.abc import Mapping
import json
import hashlib
import logging
import os
import re
import ssl
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Awaitable, Callable, Literal
from urllib.parse import quote, urlsplit

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
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from .comfy import (
    ComfyClient,
    ComfyClientProtocol,
    ComfyError,
    ComfyPromptRejected,
    default_comfy_factory,
)
from .capabilities import (
    CapabilityReason,
    FeaturePreflightReport,
    build_feature_catalog,
    build_operational_readiness,
    capture_host_capabilities,
    feature_catalog_etag,
    preflight_projected_v5_timeline,
    quote_feature_catalog_etag,
)
from .capabilities.comfy_kitchen_attention import (
    ComfyKitchenAttentionCapabilityV1,
    ComfyKitchenAttentionHostObservationV1,
    ObjectInfoChoiceObservationV1,
    project_comfy_kitchen_attention_capability,
)
from .capabilities.comfy_kitchen_attention_observer import (
    comfy_kitchen_attention_host_observation_complete,
    observe_comfy_kitchen_attention_host,
)
from .compiler import (
    DraftNotRunnable,
    timeline_segment_take_fingerprint,
    unified_continuity_predecessors,
    validate_unified_runnable,
)
from .config_manager import (
    get_directordeck_config,
    get_directordeck_config_diagnostics,
    get_lora_loader_policy,
    initialize_directordeck_config,
)
from .database import (
    AssetTrashInUse,
    AssetTrashRestoreConflict,
    Database,
    ExecutionEvidenceConflict,
    RayRuntimeIntentConflict,
    SettingsAuthorityConflict,
    TimelineRevisionConflict,
    TimelineRevisionExhausted,
    TimelineTemplateBundleConflict,
)
from .execution.submission import LockedSubmissionPlanner, SubmissionPlanningError
from .host_artifacts import (
    HostOutputProbeError,
    HostOutputProbeProvider,
    HostOutputProbeResult,
    PermanentHostOutputProbeError,
)
from .instance_lock import DirectorInstanceLock
from .public_url import public_api_url, set_public_api_prefix
from .raylight_setup import (
    RayLightInstallConflict,
    RayLightInstallManager,
    RayLightInstallUnavailable,
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
from .migration_api import (
    HistoricalCreativeInputError,
    HistoricalSaveAsProjectRequest,
    ProjectImportCommitRequest,
    ProjectImportCoordinator,
    ProjectImportError,
    ProjectImportPreflightRead,
    ProjectImportPreflightRequest,
    prepare_project_import,
    resolve_historical_creative_input,
)
from .migrations import (
    ProjectMigrationReceipt,
    RuntimeSettingsSchemaMigrated,
    TimelineSchemaMigrated,
    migrate_timeline_v4_to_v5,
)
from .native_templates import (
    ModelFamily,
    NativeHistoricalTake,
    NativeTemplateError,
    raylight_runtime_logical_gpu_indices,
    resolve_execution_backend,
)
from .progress import (
    ComfyExecutionEvent,
    ComfyProgressEvent,
    ComfyPreviewEvent,
    ComfyReconcileHint,
    LivePreviewCache,
    NativeProgressManager,
    child_execution_start_snapshot,
    child_execution_snapshot,
    child_progress_snapshot,
    durable_preview_phase_watermark,
    preview_phase_index_for_event,
    preview_source_for_node,
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
    DetectShotsRequest,
    DetectShotsResponse,
    FeatureBundleMigrationNoticeListRead,
    FeaturePreflightRequest,
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
    LoraFeatureParams,
    ModeDraft,
    ProjectCreateRequest,
    ProjectDeleteRead,
    ProjectListRead,
    ProjectSummaryRead,
    RayLightRuntimeRecoveryConfirmRequest,
    RayLightRuntimeStatusRead,
    RuntimeSettings,
    RuntimeSettingsAuthorityV1WriteRequest,
    RuntimeSettingsAuthorityV2WriteRequest,
    RuntimeSettingsAuthorityV3Read,
    RuntimeSettingsAuthorityV3WriteRequest,
    RuntimeSettingsMigrationNoticeListRead,
    RuntimeSettingsV2,
    RuntimeSettingsV3,
    StorageStatusRead,
    TimelineAuthorityRead,
    TimelineAuthorityWriteRequest,
    TimelineCompileRead,
    TimelineJobRequest,
    TimelineRevisionConflictRead,
    TimelineRevisionExhaustedRead,
    UnifiedFL2VASegment,
    UnifiedTimelineDraft,
    UnifiedTimelineDraftV5,
    VideoMetadata,
    timeline_segment_recipe,
    utc_now,
    validate_mode_draft,
    validate_timeline_draft,
    validate_timeline_draft_v5,
)
from .storage import StorageController
from .task_management import (
    TaskManagementError,
    attached_compiled_execution_plan,
    attach_parent_output_authority,
    authoritative_parent_outputs,
    import_job_output_as_asset,
    ordered_observed_artifacts,
)
from .workflow.execution import (
    AssemblySourceArtifactRef,
    CompiledExecutionPlan,
    ContinuityLateBindingEvidence,
    DocumentDigest,
    EndpointIdentity,
    ExactCancelConfirmedEvidence,
    HistoryTerminalEvidence,
    LockedSegmentUnit,
    LockedSubmissionPlan,
    LockedSubmissionUnit,
    ObservedArtifactSpec,
    ObservedAssemblyArtifactSpec,
    OutputObservationReceipt,
    OutputDescriptor,
    PromptOwnership,
    PreparedControlUnit,
    PreparedSegmentUnit,
    RuntimeEpochLateBindingEvidence,
    compiled_execution_plan_digest,
    sha256_document_digest,
)
from .workflow.feature_config import V6FeatureConfigurationError
from .workflow.compile_report import (
    CompiledExecutionReportV2,
    CompiledExecutionReportV3,
)
from .workflow.v5_compat import (
    V5CreativeAuthorityError,
    project_v5_compile_authority,
    project_v5_contextual_host_authority,
)
from .workflow.project_compiler import (
    ProjectCompilerBundleError,
    compile_project_execution_plan,
)
from .workflow.runtime_snapshot import (
    JobRuntimeSnapshotV1,
    build_job_runtime_snapshot,
    validate_job_runtime_snapshot_creative_binding,
)
from .workflow.effective_features import (
    migrate_timeline_feature_authority_to_v5,
)
from .workflow.contracts import (
    HostCapabilityProvider,
    HostCapabilitySnapshot,
    OperationalReadiness,
    canonical_sha256,
)
from .workflow.templates import CURRENT_TEMPLATE_BUNDLE, V5_TEMPLATE_BUNDLE
from .workflow.v6_projection import (
    V5V6ProjectionError,
)
from .workflow.v4_resolver import CreativeCompileInputError


ComfyFactory = Callable[[str], ComfyClientProtocol]
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._\-()\u4e00-\u9fff]+")
_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
_RAYLIGHT_GENERATION_POLL_SECONDS = 1.0
_LEGACY_COMPILE_OBSERVATION_REVISION = canonical_sha256(
    {
        "schema_version": 1,
        "authority": "compiled_execution_plan",
        "host_observation": "advisory_only",
    }
)
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
    "restart_certificate_required",
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
_UPLOAD_READ_CHUNK = 1024 * 1024
# The embedded backend serves exactly one ComfyUI instance (the host process),
# so endpoint-scoped submission serialization uses one fixed key.
_EMBEDDED_ENDPOINT_KEY = "embedded"
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


def _comfy(request: Request) -> ComfyClientProtocol:
    return request.app.state.comfy_factory(request.app.state.comfy_url)


def _ck_host_context_revision(request: Request) -> str:
    return "ck:" + hashlib.sha256(
        request.app.state.endpoint_identity.runtime_instance_id.encode("utf-8")
    ).hexdigest()


async def _comfy_kitchen_attention_capability(
    request: Request,
    *,
    reachable_families: tuple[ModelFamily, ...],
) -> ComfyKitchenAttentionCapabilityV1:
    try:
        settings = _db(request).get_settings()
    except (RuntimeError, TypeError, ValidationError, ValueError):
        unknown = ObjectInfoChoiceObservationV1(state="unknown")
        host = ComfyKitchenAttentionHostObservationV1(
            context_revision=_ck_host_context_revision(request),
            host_connected=False,
            standard_attention=unknown,
            raylight_attention=unknown,
        )
        return project_comfy_kitchen_attention_capability(settings=None, host=host)

    host: ComfyKitchenAttentionHostObservationV1 | None = request.app.state.ck_host_observation
    if host is None or not comfy_kitchen_attention_host_observation_complete(host):
        observed_generation = request.app.state.ck_host_observation_generation
        async with request.app.state.ck_host_observation_lock:
            host = request.app.state.ck_host_observation
            if host is None or not comfy_kitchen_attention_host_observation_complete(host):
                if (
                    request.app.state.ck_host_observation_generation
                    == observed_generation
                ):
                    host = await observe_comfy_kitchen_attention_host(
                        _comfy(request),
                        context_revision=_ck_host_context_revision(request),
                        previous=host,
                    )
                    request.app.state.ck_host_observation = host
                    request.app.state.ck_host_observation_generation += 1
            assert host is not None
    return project_comfy_kitchen_attention_capability(
        settings=settings,
        host=host,
        reachable_families=reachable_families,
    )


async def _host_capability_snapshot(request: Request) -> HostCapabilitySnapshot:
    """Capture one immutable, provider-validated host observation.

    The backend never imports ComfyUI internals.  Absence or failure of the
    plugin-owned provider is a configuration/readiness failure. Director never
    substitutes its compiler identities for observations of the live host.
    """

    provider: HostCapabilityProvider | None = (
        request.app.state.host_capability_provider
    )
    if provider is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "host_capability_provider_unavailable",
                "message": "Host capability observation is unavailable.",
            },
        )
    try:
        captured = await anyio.to_thread.run_sync(
            capture_host_capabilities,
            provider,
        )
        return captured.snapshot
    except (OSError, RuntimeError, TypeError, ValidationError, ValueError) as exc:
        logger.warning("Host capability capture failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail={
                "code": "host_capability_snapshot_unavailable",
                "message": "Host capability observation could not be validated.",
            },
        ) from exc


def _snapshot_has_node_class(
    snapshot: HostCapabilitySnapshot,
    class_type: str,
) -> bool:
    """Return whether ComfyUI objectively exposes the required class_type."""

    return class_type in snapshot.node_registry


def _snapshot_supports_raylight_cleanup(
    snapshot: HostCapabilitySnapshot,
) -> bool:
    """Return whether a persisted descriptor can queue a RayKill control."""

    return _snapshot_has_node_class(snapshot, "DirectorDeckRayKill")


def _snapshot_cuda_gpu_count(snapshot: HostCapabilitySnapshot) -> int:
    """Return the dense logical namespace owned by RayLight's CUDA ledger."""

    return sum(item.backend == "cuda" for item in snapshot.gpu_inventory)


def _host_operational_readiness(
    request: Request,
    snapshot: HostCapabilitySnapshot,
) -> OperationalReadiness:
    """Read transient runtime facts without contacting or mutating ComfyUI."""

    try:
        state = _db(request).get_raylight_runtime_state()
        if state is None:
            return build_operational_readiness(
                endpoint_online=True,
                available_logical_gpu_count=_snapshot_cuda_gpu_count(snapshot),
            )
        current = state.get("current")
        runtime_gpu_indices = (
            raylight_runtime_logical_gpu_indices(current)
            if isinstance(current, dict)
            else ()
        )
        legacy_unknown = bool(state.get("legacy_unknown"))
        ray_tainted = bool(state.get("tainted"))
        cleanup_contract_available = _snapshot_supports_raylight_cleanup(snapshot)
        blocking_reasons = (
            ("ray_cleanup_unavailable",)
            if isinstance(current, dict) and not cleanup_contract_available
            else ()
        )
        return build_operational_readiness(
            endpoint_online=True,
            ray_recovery_required=legacy_unknown,
            ray_tainted=ray_tainted,
            # A full descriptor lets the Stage-4 locked planner place an exact
            # RayKill barrier.  Taint without that evidence is not repairable.
            ray_cleanup_available=(
                ray_tainted
                and isinstance(current, dict)
                and cleanup_contract_available
            ),
            runtime_gpu_indices=runtime_gpu_indices,
            available_logical_gpu_count=_snapshot_cuda_gpu_count(snapshot),
            blocking_reason_codes=blocking_reasons,
        )
    except (KeyError, NativeTemplateError, TypeError, ValidationError, ValueError):
        return build_operational_readiness(
            endpoint_online=True,
            ray_recovery_required=True,
            ray_tainted=True,
            ray_cleanup_available=False,
            available_logical_gpu_count=_snapshot_cuda_gpu_count(snapshot),
            blocking_reason_codes=("ray_runtime_state_invalid",),
        )


def _project_not_found_reason() -> CapabilityReason:
    """Return one privacy-safe project-scope failure for every entrypoint."""

    return CapabilityReason(
        code="project_not_found",
        feature_id=None,
        segment_id=None,
        unit_id=None,
        backend=None,
        rule="project_scope",
        message="The selected project no longer exists.",
        remediation="Reload the project list and select an existing project.",
        safe_details={},
    )


def _project_not_found_http_error() -> HTTPException:
    reason = _project_not_found_reason()
    return HTTPException(
        status_code=404,
        detail={
            "code": reason.code,
            "message": reason.message,
            "reasons": [reason.model_dump(mode="json")],
        },
    )


def _feature_preflight_report_from_reason(
    *,
    snapshot: HostCapabilitySnapshot,
    readiness: OperationalReadiness,
    reason: CapabilityReason,
) -> FeaturePreflightReport:
    return FeaturePreflightReport(
        template_bundle_version=CURRENT_TEMPLATE_BUNDLE.version,
        host_capability_revision=snapshot.host_capability_revision(),
        operational_readiness=readiness,
        valid=False,
        errors=(reason,),
        effective_by_segment={},
    )


def _host_context_observation_reason() -> CapabilityReason:
    """Return one safe failure contract for every host-context entrypoint."""

    return CapabilityReason(
        code="host_context_unavailable",
        feature_id=None,
        segment_id=None,
        unit_id=None,
        backend=None,
        rule="host_context_observation",
        message="The current ComfyUI model or runtime context could not be observed.",
        remediation="Check the embedded ComfyUI connection and run preflight again.",
        safe_details={},
    )


def _host_context_observation_http_error() -> HTTPException:
    reason = _host_context_observation_reason()
    return HTTPException(
        status_code=502,
        detail={
            "code": reason.code,
            "message": reason.message,
            "reasons": [reason.model_dump(mode="json")],
        },
    )


def _creative_input_reason(error: BaseException) -> CapabilityReason:
    """Project resolver/legacy validation failures into one safe wire shape."""

    chain: list[BaseException] = []
    candidate: BaseException | None = error
    while candidate is not None and len(chain) < 4:
        chain.append(candidate)
        candidate = candidate.__cause__
    candidate = next(
        (item for item in chain if isinstance(item, CreativeCompileInputError)),
        None,
    )
    if isinstance(candidate, CreativeCompileInputError):
        return CapabilityReason(
            code=candidate.code,
            feature_id=candidate.feature_id,
            segment_id=candidate.segment_id,
            unit_id=None,
            backend=candidate.backend,
            rule=candidate.rule,
            message=candidate.public_message,
            remediation=candidate.remediation,
            safe_details=candidate.safe_details,
        )
    draft_error = next(
        (item for item in chain if isinstance(item, DraftNotRunnable)),
        None,
    )
    if isinstance(draft_error, DraftNotRunnable):
        draft_message = str(draft_error)
        if (
            draft_message.startswith(
                "segment_ids must name enabled timeline segments"
            )
            or draft_message == "at least one enabled timeline segment is required"
        ):
            return _segment_selection_reason()
    v6_error = next(
        (
            item
            for item in chain
            if isinstance(item, (V5V6ProjectionError, V6FeatureConfigurationError))
        ),
        None,
    )
    if isinstance(v6_error, (V5V6ProjectionError, V6FeatureConfigurationError)):
        return CapabilityReason(
            code=v6_error.code,
            feature_id=v6_error.feature_id,
            segment_id=v6_error.segment_id,
            unit_id=None,
            backend=None,
            rule="bundle6_project_configuration",
            message="The selected project workflow configuration is invalid.",
            remediation="Correct this project configuration and retry the action.",
            safe_details=v6_error.safe_details,
        )
    rendered = str(error)
    if isinstance(error, NativeTemplateError) and re.fullmatch(
        r"segment compiles to \d+ frames; native H3 template limit is 512",
        rendered,
    ):
        return CapabilityReason(
            code="segment_frame_limit_exceeded",
            feature_id=None,
            segment_id=None,
            unit_id=None,
            backend=None,
            rule="native_h3_frame_limit",
            message="A selected segment exceeds the 512-frame MiniMax H3 limit.",
            remediation=(
                "Split the segment into shorter segments, then run preflight again."
            ),
            safe_details={"max_frames": 512},
        )
    if isinstance(error, DraftNotRunnable):
        if rendered.startswith(
            "Ref2VA segments need source_video or independent reference media: "
        ):
            return CapabilityReason(
                code="ref2va_input_required",
                feature_id=None,
                segment_id=None,
                unit_id=None,
                backend=None,
                rule="ref2va_conditioning_input",
                message=(
                    "Ref2VA segments need a source video or independent reference media."
                ),
                remediation=(
                    "Add a source video or at least one reference image, video, "
                    "or audio, then run preflight again."
                ),
                safe_details={},
            )
        historical_failure: tuple[str, str, str] | None = None
        for suffix, code, message in (
            (
                "已有历史成片记录无效，无法用于接续",
                "historical_take_invalid",
                "The selected historical take has invalid execution evidence.",
            ),
            (
                "有输出规格匹配的历史成功成片，但不含生成音频接续所需的音轨",
                "historical_take_audio_required",
                "The selected historical take does not contain the required audio track.",
            ),
            (
                "存在历史成功成片，但分辨率、帧率或可见帧数与当前分段不一致",
                "historical_take_geometry_mismatch",
                "The selected historical take does not match the current segment geometry.",
            ),
            (
                "只有旧任务输出定位记录，实际媒体规格与音轨信息不可用；请重新生成前驱",
                "historical_take_observation_unavailable",
                "The selected legacy take has no verified media observation.",
            ),
            (
                "没有可用的历史成功成片",
                "historical_take_required",
                "No compatible historical take is available for continuity.",
            ),
        ):
            if rendered.endswith(suffix):
                historical_failure = (code, message, suffix)
                break
        if historical_failure is not None:
            code, message, _ = historical_failure
            return CapabilityReason(
                code=code,
                feature_id="continuity",
                segment_id=None,
                unit_id=None,
                backend=None,
                rule="historical_take_compatibility",
                message=message,
                remediation=(
                    "Regenerate the predecessor with the current settings or disable "
                    "continuity, then run preflight again."
                ),
                safe_details={},
            )
        if (
            rendered.startswith("segment_ids must name enabled timeline segments")
            or rendered == "at least one enabled timeline segment is required"
        ):
            return _segment_selection_reason()
    if (
        isinstance(error, ValueError)
        and ": asset id '" in rendered
        and rendered.endswith("' is not registered")
    ):
        return CapabilityReason(
            code="asset_unavailable",
            feature_id=None,
            segment_id=None,
            unit_id=None,
            backend=None,
            rule="asset_registry",
            message=(
                "A selected segment references an asset that is no longer available."
            ),
            remediation=(
                "Reload the asset library and replace or remove the missing asset, "
                "then run preflight again."
            ),
            safe_details={},
        )
    return CapabilityReason(
        code="creative_configuration_invalid",
        feature_id=None,
        segment_id=None,
        unit_id=None,
        backend=None,
        rule="timeline_validation",
        message="The selected timeline is not runnable.",
        remediation="Correct the timeline, assets, or segment selection and run preflight again.",
        safe_details={},
    )


def _capability_reasons_http_error(
    reasons: tuple[CapabilityReason, ...],
) -> HTTPException:
    if not reasons:
        reasons = (
            CapabilityReason(
                code="capability_unavailable",
                feature_id=None,
                segment_id=None,
                unit_id=None,
                backend=None,
                rule="capability_evaluation",
                message="The selected feature is unavailable.",
                remediation="Run preflight again after checking the current host capabilities.",
                safe_details={},
            ),
        )
    first = reasons[0]
    return HTTPException(
        status_code=422,
        detail={
            "code": first.code,
            "message": first.message,
            "reasons": [reason.model_dump(mode="json") for reason in reasons],
        },
    )


def _creative_input_http_error(
    reason: CapabilityReason,
    *additional_reasons: CapabilityReason,
) -> HTTPException:
    return _capability_reasons_http_error((reason, *additional_reasons))


def _v5_creative_authority_reason(
    exc: V5CreativeAuthorityError,
) -> CapabilityReason:
    if exc.code == "segment_selection_invalid":
        return _segment_selection_reason()
    return CapabilityReason(
        code=exc.code,
        feature_id=exc.feature_id,
        segment_id=exc.segment_id,
        unit_id=None,
        backend=None,
        rule="v5_creative_authority",
        message=str(exc),
        remediation=(
            "Complete or correct the project-owned model and feature "
            "configuration, then run preflight again."
        ),
        safe_details=exc.safe_details,
    )


def _segment_selection_reason() -> CapabilityReason:
    return CapabilityReason(
        code="segment_selection_invalid",
        feature_id=None,
        segment_id=None,
        unit_id=None,
        backend=None,
        rule="segment_selection",
        message="The requested segment selection is empty, disabled, or stale.",
        remediation=(
            "Select only enabled segments from the current project and run "
            "preflight again."
        ),
        safe_details={},
    )


def _runtime_authority_changed() -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={
            "code": "runtime_authority_changed",
            "message": "runtime settings changed while resources were being read",
        },
    )


def _execution_plan_invariant_http_error() -> HTTPException:
    return HTTPException(
        status_code=500,
        detail={
            "code": "execution_plan_invariant_failed",
            "message": "The compiled execution evidence is internally inconsistent.",
        },
    )


def _runtime_authority_snapshot(
    request: Request,
) -> tuple[RuntimeSettingsV3, str]:
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


def _timeline_template_bundle_conflict(
    exc: TimelineTemplateBundleConflict,
) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={
            "code": "timeline_template_bundle_conflict",
            "message": (
                "The project workflow bundle changed on the server; "
                "fetch the current timeline authority before retrying."
            ),
            "project_id": exc.project_id,
            "submitted_template_bundle": exc.submitted,
            "current_template_bundle": exc.current,
        },
    )


def _legacy_generation_api_retired() -> HTTPException:
    """Versioned tombstone for the six pre-timeline write APIs.

    Reads remain available for historical display and offline migration, but a
    v5 server must never manufacture a new creative snapshot by combining an
    old mode draft with live runtime settings.
    """

    return HTTPException(
        status_code=410,
        detail={
            "code": "legacy_generation_api_retired",
            "message": (
                "Legacy six-mode generation writes are retired; refresh the "
                "client and submit a v5 timeline snapshot."
            ),
            "required_schema": 5,
        },
    )


def _project_import_http_error(exc: ProjectImportError) -> HTTPException:
    detail: dict[str, Any] = {
        "code": exc.code,
        "message": exc.message,
    }
    if exc.details:
        detail["details"] = exc.details
    return HTTPException(status_code=exc.status_code, detail=detail)


def _historical_creative_http_error(
    exc: HistoricalCreativeInputError,
) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={"code": exc.code, "message": exc.message},
    )


def _runtime_settings_schema_migrated() -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={
            "code": "runtime_settings_schema_migrated",
            "message": (
                "Runtime settings migrated to schema 3; refresh before saving."
            ),
            "current_schema": 3,
        },
    )


def _settings_authority_conflict() -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={
            "code": "runtime_settings_authority_conflict",
            "message": (
                "Runtime settings changed on the server; fetch the current "
                "authority before retrying."
            ),
        },
    )


def _lora_product_config_unavailable() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={
            "code": "lora_product_config_unavailable",
            "message": "DirectorDeck's LoRA loader configuration is unavailable.",
        },
    )


def _lora_loader_options_invalid(
    lora_filename: str,
    adapter_id: str,
) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={
            "code": "lora_loader_options_invalid",
            "message": "The selected LoRA loader configuration is invalid.",
            "lora_filename": lora_filename,
            "adapter_id": adapter_id,
        },
    )


def _validated_lora_loader_mapping_update(
    current: RuntimeSettingsV3,
    candidate: RuntimeSettingsV3,
) -> RuntimeSettingsV3:
    """Validate only new/changed mappings and preserve retired records."""

    current_by_filename = {
        record.lora_filename: record for record in current.lora_loader_overrides
    }
    config = None
    normalized_records = []
    for record in candidate.lora_loader_overrides:
        previous = current_by_filename.get(record.lora_filename)
        if previous is not None and previous == record:
            normalized_records.append(record)
            continue
        try:
            if config is None:
                config = get_directordeck_config()
            policy = get_lora_loader_policy(record.lora_filename)
        except RuntimeError as exc:
            raise _lora_product_config_unavailable() from exc
        if record.adapter_id not in policy.loader_ids:
            raise _lora_loader_mapping_not_allowed(
                (record.lora_filename, record.adapter_id, policy.loader_ids)
            )
        try:
            options = config.normalize_lora_loader_options(
                record.adapter_id,
                dict(record.options),
            )
        except (KeyError, ValueError) as exc:
            raise _lora_loader_options_invalid(
                record.lora_filename,
                record.adapter_id,
            ) from exc
        normalized_records.append(record.model_copy(update={"options": options}))
    return candidate.model_copy(
        update={"lora_loader_overrides": normalized_records}
    )


def _lora_loader_mapping_not_allowed(
    issue: tuple[str, str, tuple[str, ...]],
) -> HTTPException:
    lora_filename, adapter_id, allowed_loader_ids = issue
    return HTTPException(
        status_code=422,
        detail={
            "code": "lora_loader_not_allowed_for_file",
            "message": "The selected LoRA loader is not allowed for this LoRA file.",
            "lora_filename": lora_filename,
            "adapter_id": adapter_id,
            "allowed_loader_ids": list(allowed_loader_ids),
        },
    )


def _timeline_schema_migrated(
    database: Database, project_id: str
) -> HTTPException:
    receipt = database.get_latest_project_migration_receipt(
        project_id,
        from_schema=4,
        to_schema=5,
    )
    return HTTPException(
        status_code=409,
        detail={
            "code": "timeline_schema_migrated",
            "message": "Timeline schema migrated to v5; refresh before saving.",
            "project_id": project_id,
            "current_schema": 5,
            "migration_id": (
                receipt.migration_id if receipt is not None else None
            ),
        },
    )


def _timeline_authority_required(project_id: str) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={
            "code": "timeline_authority_required",
            "message": "Timeline writes require an expected server revision.",
            "project_id": project_id,
        },
    )


def _project_document_unreadable(project_id: str) -> HTTPException:
    """Confine a damaged creative authority to its own project."""

    return HTTPException(
        status_code=409,
        detail={
            "code": "project_document_unreadable",
            "message": (
                "The stored project document is unreadable; "
                "export or delete this project, or choose another project."
            ),
            "project_id": project_id,
        },
    )


def _project_rename_api_retired(project_id: str) -> HTTPException:
    """Retire title writes that bypass timeline history, WAL, and CAS."""

    return HTTPException(
        status_code=410,
        detail={
            "code": "project_rename_api_retired",
            "message": (
                "Project titles are part of timeline creative authority; "
                "update document.title through timeline authority CAS."
            ),
            "required_endpoint": (
                f"/api/projects/{project_id}/timeline/authority"
            ),
            "required_schema": 5,
        },
    )


def _project_import_preflight_required() -> HTTPException:
    return HTTPException(
        status_code=410,
        detail={
            "code": "project_import_preflight_required",
            "message": (
                "Direct project import is retired; use import preflight and "
                "commit with the returned short-lived token."
            ),
        },
    )


def _parse_v5_timeline_authority_write(
    database: Database,
    project_id: str,
    body: Mapping[str, Any],
) -> TimelineAuthorityWriteRequest:
    document = body.get("document")
    if isinstance(document, Mapping) and document.get("version") == 4:
        raise _timeline_schema_migrated(database, project_id)
    try:
        return TimelineAuthorityWriteRequest.model_validate(body)
    except ValidationError as exc:
        raise _validation_error(exc) from exc


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


def _child_output_node_mapping(child: dict[str, Any]) -> dict[str, str]:
    evidence = child.get("execution_evidence")
    if isinstance(evidence, dict) and "exact_prompt_snapshot" in evidence:
        snapshot = evidence["exact_prompt_snapshot"]
        expected = (
            snapshot.expected_output_spec
            if hasattr(snapshot, "expected_output_spec")
            else snapshot.get("expected_output_spec")
            if isinstance(snapshot, dict)
            else None
        )
        if expected is None:
            return {}
        segment_id = (
            expected.segment_id
            if hasattr(expected, "segment_id")
            else expected.get("segment_id")
            if isinstance(expected, dict)
            else None
        )
        node_id = (
            expected.node_id
            if hasattr(expected, "node_id")
            else expected.get("node_id")
            if isinstance(expected, dict)
            else None
        )
        if (
            isinstance(segment_id, str)
            and segment_id in child.get("segment_ids", [])
            and isinstance(node_id, str)
        ):
            return {node_id: segment_id}
        return {}
    declared_segments = {str(item) for item in child.get("segment_ids", [])}
    return {
        str(node_id): str(segment_id)
        for segment_id, node_id in (child.get("output_nodes") or {}).items()
        if str(segment_id) in declared_segments
    }


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
        evidence = child.get("execution_evidence")
        if isinstance(evidence, dict):
            snapshot = evidence.get("exact_prompt_snapshot")
            expected = (
                snapshot.expected_output_spec
                if hasattr(snapshot, "expected_output_spec")
                else None
            )
            artifact = child.get("observed_artifact")
            if (
                expected is None
                or not isinstance(artifact, ObservedArtifactSpec)
                or child.get("status") != "succeeded"
                or artifact.child_id != str(child.get("id") or "")
                or artifact.segment_id != expected.segment_id
                or expected.segment_id not in child.get("segment_ids", [])
            ):
                # A typed child can never fall back to mutable legacy columns.
                continue
            output = {
                "node_id": expected.node_id,
                **artifact.output_descriptor.model_dump(mode="json"),
            }
            candidates.setdefault(expected.segment_id, []).append(
                (child, output)
            )
            continue
        node_to_segment = _child_output_node_mapping(child)
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


def _job_legacy_creative_projection(
    job: dict[str, Any],
) -> tuple[UnifiedTimelineDraft, RuntimeSettings | None] | None:
    """Project either historical schema into the frozen display contract."""

    snapshot = job.get("config_snapshot")
    if not isinstance(snapshot, dict):
        return None
    raw_timeline = snapshot.get("timeline")
    if not isinstance(raw_timeline, Mapping):
        return None
    try:
        if raw_timeline.get("version") == 5:
            timeline_v5 = UnifiedTimelineDraftV5.model_validate(raw_timeline)
            legacy_document = timeline_v5.model_dump(
                mode="json",
                exclude={"model_stack", "features"},
            )
            legacy_document["version"] = 4
            return UnifiedTimelineDraft.model_validate(legacy_document), None
        timeline_v4 = UnifiedTimelineDraft.model_validate(raw_timeline)
        try:
            settings_v1 = RuntimeSettings.model_validate(
                job.get("settings_snapshot")
            )
        except (TypeError, ValidationError, ValueError):
            settings_v1 = None
        return timeline_v4, settings_v1
    except (
        TypeError,
        ValidationError,
        ValueError,
    ):
        return None


def _job_timeline_snapshot(job: dict[str, Any]) -> UnifiedTimelineDraft | None:
    projection = _job_legacy_creative_projection(job)
    if projection is None:
        return None
    return projection[0]


def _job_v5_creative_snapshot(
    job: Mapping[str, Any],
) -> UnifiedTimelineDraftV5 | None:
    try:
        return resolve_historical_creative_input(
            job,
            migrate_v4_to_v5=migrate_timeline_v4_to_v5,
        )
    except (HistoricalCreativeInputError, ProjectImportError):
        return None


def _snapshot_filename(value: str) -> str:
    """Keep a useful model basename without exposing a historical path."""

    return re.split(r"[\\/]", value)[-1]


def _v5_generation_lora(
    draft: UnifiedTimelineDraftV5,
    family: ModelFamily,
) -> tuple[str | None, float]:
    selection = draft.features.project.get("lora")
    if selection is None or not selection.enabled:
        return None, 1.0
    params = LoraFeatureParams.model_validate(selection.params)
    family_selection = params.by_family[family]
    if not family_selection.enabled or family_selection.filename is None:
        return None, family_selection.strength
    return _snapshot_filename(family_selection.filename), family_selection.strength


def _v5_generation_runtime_details(
    job: Mapping[str, Any],
    draft: UnifiedTimelineDraftV5,
    families: list[ModelFamily],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Read bounded and historical full runtime snapshots without live state."""

    config_snapshot = job.get("config_snapshot")
    raw_segment_ids = (
        config_snapshot.get("segment_ids")
        if isinstance(config_snapshot, Mapping)
        else None
    )
    if raw_segment_ids is not None and not (
        isinstance(raw_segment_ids, list)
        and all(isinstance(item, str) for item in raw_segment_ids)
    ):
        return None
    segment_ids = raw_segment_ids if isinstance(raw_segment_ids, list) else None
    raw_runtime = job.get("settings_snapshot")
    if not isinstance(raw_runtime, Mapping):
        return None

    family_runtime: dict[str, tuple[str, str, Any | None]] = {}
    shared_devices: dict[str, str | None] = {}
    if "snapshot_schema_version" in raw_runtime:
        try:
            bounded = JobRuntimeSnapshotV1.model_validate(raw_runtime)
            validate_job_runtime_snapshot_creative_binding(
                bounded,
                draft,
                segment_ids,
            )
        except (TypeError, ValidationError, ValueError):
            return None
        for family, placement in bounded.family_map().items():
            family_runtime[family] = (
                placement.backend,
                placement.device,
                placement.raylight_profile,
            )
        runtime = bounded.runtime_projection
        shared_devices = {
            "clip": runtime.clip_device,
            "video_vae": runtime.video_vae_device,
            "audio_vae": runtime.audio_vae_device,
        }
    else:
        schema_version = raw_runtime.get("schema_version")
        try:
            if schema_version == 3:
                historical = RuntimeSettingsV3.model_validate(raw_runtime)
            elif schema_version == 2:
                historical = RuntimeSettingsV2.model_validate(raw_runtime)
            else:
                return None
        except (TypeError, ValidationError, ValueError):
            return None
        for family in families:
            placement = getattr(historical.placement, family)
            backend = (
                "raylight"
                if len(placement.raylight.gpu_select) >= 2
                else "standard"
            )
            family_runtime[family] = (
                backend,
                placement.device,
                placement.raylight if backend == "raylight" else None,
            )
        shared_devices = {
            "clip": historical.placement.clip_device,
            "video_vae": historical.placement.video_vae_device,
            "audio_vae": historical.placement.audio_vae_device,
        }

    if set(family_runtime) != set(families):
        return None
    models: list[dict[str, Any]] = []
    for family in families:
        model_filename = getattr(draft.model_stack, family).filename
        if model_filename is None:
            return None
        backend, device, raylight = family_runtime[family]
        lora_name, lora_strength = _v5_generation_lora(draft, family)
        models.append(
            {
                "family": family,
                "filename": _snapshot_filename(model_filename),
                "device": device,
                "lora_name": lora_name,
                "lora_strength": lora_strength,
                "backends": [backend],
                "logical_gpu_indices": (
                    list(raylight.gpu_select) if raylight is not None else []
                ),
                "ulysses_degree": (
                    raylight.ulysses_degree if raylight is not None else None
                ),
                "ring_degree": (
                    raylight.ring_degree if raylight is not None else None
                ),
            }
        )

    shared_models: list[dict[str, Any]] = []
    for role in ("clip", "video_vae", "audio_vae"):
        filename = getattr(draft.model_stack, role).filename
        device = shared_devices[role]
        # The bounded snapshot intentionally omits an unused audio VAE route.
        if role == "audio_vae" and device is None:
            continue
        if filename is None or device is None:
            return None
        shared_models.append(
            {
                "role": role,
                "filename": _snapshot_filename(filename),
                "device": device,
            }
        )
    return models, shared_models


def _job_generation_details(
    job: dict[str, Any], children: list[dict[str, Any]]
) -> JobGenerationDetailsRead | None:
    projection = _job_legacy_creative_projection(job)
    if projection is None:
        return None
    timeline, settings = projection

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
    captured_v5 = None
    raw_timeline = snapshot.get("timeline") if isinstance(snapshot, Mapping) else None
    if isinstance(raw_timeline, Mapping) and raw_timeline.get("version") == 5:
        try:
            captured_v5 = UnifiedTimelineDraftV5.model_validate(raw_timeline)
        except (TypeError, ValidationError, ValueError):
            captured_v5 = None
    if captured_v5 is not None:
        v5_runtime = _v5_generation_runtime_details(
            job,
            captured_v5,
            families,
        )
        if v5_runtime is not None:
            models, shared_models = v5_runtime
            runtime_snapshot_available = True
    elif settings is not None:
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


def _child_public_outputs(child: dict[str, Any]) -> list[str]:
    """Project typed child media only from immutable expected/observed facts."""

    evidence = child.get("execution_evidence")
    if isinstance(evidence, dict):
        snapshot = evidence.get("exact_prompt_snapshot")
        expected = (
            snapshot.expected_output_spec
            if hasattr(snapshot, "expected_output_spec")
            else None
        )
        artifact = child.get("observed_artifact")
        if (
            expected is None
            or not isinstance(artifact, ObservedArtifactSpec)
            or child.get("status") != "succeeded"
            or artifact.child_id != str(child.get("id") or "")
            or artifact.segment_id != expected.segment_id
            or expected.segment_id not in child.get("segment_ids", [])
        ):
            # Merely having mutable compatibility outputs can never make an
            # incomplete or corrupt typed evidence chain public.
            return []
        return [
            _output_file_location(
                {
                    "node_id": expected.node_id,
                    **artifact.output_descriptor.model_dump(mode="json"),
                }
            )
        ]
    return [_output_file_location(output) for output in child.get("outputs", [])]


def _job_read(
    job: dict[str, Any], *, live_preview_available: bool = False,
    current_snapshot: bool = False,
    current_project: bool = False,
) -> JobRead:
    parent_outputs = authoritative_parent_outputs(job)
    outputs = [
        public_api_url(f"/api/jobs/{job['id']}/outputs/{index}")
        for index, _ in enumerate(parent_outputs)
    ]
    output_files = [_output_file_location(output) for output in parent_outputs]
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
                    "outputs": _child_public_outputs(child),
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


def _child_with_execution_evidence(
    database: Database,
    child: dict[str, Any],
) -> dict[str, Any]:
    try:
        evidence = database.get_job_child_execution_evidence(str(child["id"]))
    except (ExecutionEvidenceConflict, RuntimeError, ValidationError, ValueError):
        # Preserve the fact that this is a typed child so every downstream
        # projection fails closed instead of reinterpreting mutable legacy
        # columns after immutable evidence corruption.
        enriched = dict(child)
        enriched["execution_evidence"] = {"invalid": True}
        enriched["observed_artifact"] = None
        return enriched
    if evidence is None:
        try:
            has_typed_marker = database.has_job_child_execution_marker(
                str(child["id"])
            )
        except RuntimeError:
            has_typed_marker = True
        if has_typed_marker:
            enriched = dict(child)
            enriched["execution_evidence"] = {"invalid": True}
            enriched["observed_artifact"] = None
            return enriched
        return child
    enriched = dict(child)
    enriched["execution_evidence"] = evidence
    try:
        enriched["observed_artifact"] = database.get_observed_artifact(
            str(child["id"])
        )
    except (ExecutionEvidenceConflict, RuntimeError, ValidationError, ValueError):
        enriched["observed_artifact"] = None
    return enriched


def _job_with_parent_output_authority(
    database: Database,
    job: dict[str, Any],
) -> dict[str, Any]:
    """Attach the typed parent-output facts used by every public consumer."""

    enriched = dict(job)
    raw_children = enriched.get("children")
    if not isinstance(raw_children, list):
        raw_children = database.list_job_children(str(enriched["id"]))
    children = [
        child
        if isinstance(child, dict) and "execution_evidence" in child
        else _child_with_execution_evidence(database, child)
        for child in raw_children
        if isinstance(child, dict)
    ]
    enriched["children"] = children
    try:
        typed = database.has_job_execution_marker(str(enriched["id"]))
    except RuntimeError:
        typed = True
    typed = typed or any(
        isinstance(child.get("execution_evidence"), dict) for child in children
    )
    try:
        plan = database.get_job_execution_plan(str(enriched["id"]))
    except (ExecutionEvidenceConflict, RuntimeError, ValidationError, ValueError):
        plan = None
        typed = True
    else:
        typed = typed or plan is not None

    assembly = None
    config_snapshot = enriched.get("config_snapshot")
    timeline = (
        config_snapshot.get("timeline")
        if isinstance(config_snapshot, dict)
        else None
    )
    needs_assembly = (
        plan is not None
        and len(plan.segment_units) > 1
        and isinstance(timeline, dict)
        and timeline.get("export_mode") == "all"
        and _is_full_timeline_selection(config_snapshot)
    )
    if needs_assembly:
        try:
            assembly = database.get_observed_assembly_artifact(str(enriched["id"]))
        except (ExecutionEvidenceConflict, RuntimeError, ValidationError, ValueError):
            typed = True
            assembly = None
    return attach_parent_output_authority(
        enriched,
        typed=typed,
        compiled_plan=plan,
        observed_assembly=assembly,
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
    database = _db(request)
    child = database.get_job_child(preview.child_id)
    if child is not None:
        child = _child_with_execution_evidence(database, child)
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


class _UnsetCurrentTimeline:
    """Distinguish the legacy default scope from an explicitly missing scope."""


_UNSET_CURRENT_TIMELINE = _UnsetCurrentTimeline()


def _job_read_context_for_project(
    request: Request,
    project_id: str | None,
) -> tuple[str, UnifiedTimelineDraftV5 | None, RuntimeSettingsV3]:
    """Resolve one request's active-project comparison authority exactly once."""

    database = _db(request)
    active_project_id = project_id or database.LEGACY_DEFAULT_PROJECT_ID
    try:
        current_timeline = database.get_project_timeline(active_project_id)
    except KeyError:
        # A stale browser tab may cancel a still-owned task after its active
        # project was deleted. The mutation remains valid, but currentness must
        # fail closed rather than silently comparing against the default.
        current_timeline = None
    return active_project_id, current_timeline, database.get_settings()


def _current_effective_execution_evidence(
    request: Request,
    *,
    draft: UnifiedTimelineDraftV5,
    settings: RuntimeSettingsV3,
    segment_ids: list[str] | None,
    project_id: str,
    job_id: str,
    cache: dict[str, Any] | None,
) -> tuple[DocumentDigest, JobRuntimeSnapshotV1]:
    """Recompile with the current project's own frozen bundle compiler.

    Host observations remain absent.  Bundle 5 and Bundle 6 consume only their
    captured project/settings inputs, exact LoRA policy result and historical
    take ledger; neither may fall back to the process-current compiler.
    """

    historical_takes = _resolve_historical_continuity_takes(
        _db(request),
        draft,
        segment_ids=segment_ids,
        project_id=project_id,
    )
    cache_key = sha256_document_digest(
        {
            "project_id": project_id,
            "timeline": draft.model_dump(mode="json"),
            "segment_ids": segment_ids,
            # Extra settings may invalidate this performance cache, but never
            # enter the resulting execution digest unless the selected bundle
            # compiler actually consumes them.
            "settings": settings.model_dump(mode="json"),
            "historical_takes": [
                {
                    "target_segment_id": target_segment_id,
                    "take_id": take.id,
                    "source_segment_id": take.segment_id,
                    "output": dict(take.output),
                }
                for target_segment_id, take in sorted(historical_takes.items())
            ],
        }
    ).value
    if cache is not None and cache_key in cache:
        cached = cache[cache_key]
        if (
            isinstance(cached, tuple)
            and len(cached) == 2
            and isinstance(cached[0], DocumentDigest)
            and isinstance(cached[1], JobRuntimeSnapshotV1)
        ):
            return cached
    plan = compile_project_execution_plan(
        draft,
        settings,
        job_id,
        segment_ids,
        historical_takes=historical_takes,
    )
    evidence = (
        plan.effective_execution_digest,
        build_job_runtime_snapshot(draft, segment_ids, settings, plan),
    )
    if cache is not None:
        cache[cache_key] = evidence
    return evidence


def _job_read_for_request(
    request: Request,
    job: dict[str, Any],
    *,
    current_timeline: (
        UnifiedTimelineDraftV5 | None | _UnsetCurrentTimeline
    ) = _UNSET_CURRENT_TIMELINE,
    current_settings: RuntimeSettingsV3 | None = None,
    current_project_id: str | None = None,
    current_execution_digest_cache: dict[str, Any] | None = None,
) -> JobRead:
    """Annotate one job with strict currentness flags.

    ``current_project`` deliberately compares only the typed timeline (scoped
    by the caller to the active project's timeline); ``current_snapshot`` adds
    equality of the execution-relevant runtime projection and a freshly
    resolved effective-execution digest. Loose project membership is expressed
    by the separate ``project_id`` field.
    """

    job = _job_with_parent_output_authority(_db(request), job)

    current_snapshot = False
    current_project = False
    snapshot_timeline = _job_v5_creative_snapshot(job)
    if snapshot_timeline is not None:
        if isinstance(current_timeline, _UnsetCurrentTimeline):
            current_project_id = _db(request).LEGACY_DEFAULT_PROJECT_ID
            current_timeline = _db(request).get_timeline()
        elif current_project_id is None:
            current_project_id = str(
                job.get("project_id")
                or _db(request).LEGACY_DEFAULT_PROJECT_ID
            )
        if current_timeline is not None:
            current_project = (
                snapshot_timeline.model_dump(mode="json")
                == current_timeline.model_dump(mode="json")
            )
    raw_job_snapshot = job.get("config_snapshot")
    raw_job_timeline = (
        raw_job_snapshot.get("timeline")
        if isinstance(raw_job_snapshot, Mapping)
        else None
    )
    # A historical v4 projection remains inspectable but cannot mint modern
    # runtime currentness. Bundle 5 and 6 both use timeline schema 5 here.
    has_captured_timeline_authority = (
        isinstance(raw_job_timeline, Mapping)
        and raw_job_timeline.get("version") == 5
    )
    if (
        job.get("mode") == "timeline"
        and snapshot_timeline is not None
        and has_captured_timeline_authority
    ):
        try:
            snapshot_runtime = JobRuntimeSnapshotV1.model_validate(
                job.get("settings_snapshot")
            )
            if current_settings is None:
                current_settings = _db(request).get_settings()
            if "segment_ids" not in raw_job_snapshot:
                raise ValueError("captured segment selection is missing")
            raw_segment_ids = raw_job_snapshot.get("segment_ids")
            segment_ids = (
                raw_segment_ids if isinstance(raw_segment_ids, list) else None
            )
            if raw_segment_ids is not None and segment_ids is None:
                raise ValueError("invalid captured segment selection")
            if current_timeline is None:
                raise ValueError("current project timeline is unavailable")
        except (TypeError, ValidationError, ValueError):
            # Historical or partially migrated jobs remain inspectable, but
            # can never be described as an exact current snapshot.
            pass
        else:
            try:
                stored_plan = attached_compiled_execution_plan(job)
                if stored_plan is None or current_project_id is None:
                    raise ValueError("job has no typed execution plan")
                (
                    current_digest,
                    current_runtime_snapshot,
                ) = _current_effective_execution_evidence(
                    request,
                    draft=current_timeline,
                    settings=current_settings,
                    segment_ids=segment_ids,
                    project_id=current_project_id,
                    job_id=str(job.get("id") or "currentness"),
                    cache=current_execution_digest_cache,
                )
                current_snapshot = (
                    snapshot_runtime.has_same_execution_identity(
                        current_runtime_snapshot
                    )
                    and current_digest
                    == stored_plan.effective_execution_digest
                )
            except (
                DraftNotRunnable,
                NativeTemplateError,
                ProjectCompilerBundleError,
                TypeError,
                V5CreativeAuthorityError,
                V5V6ProjectionError,
                ValidationError,
                ValueError,
            ):
                # Missing current historical takes or any graph/adapter
                # identity drift is not evidence of currentness.
                current_snapshot = False
    return _job_read(
        job,
        live_preview_available=_live_preview_for_job(request, job) is not None,
        current_snapshot=current_snapshot,
        current_project=current_project,
    )


def _job_read_for_project_scope(
    request: Request,
    job: dict[str, Any],
    *,
    project_id: str | None,
) -> JobRead:
    """Project one job against the caller's explicit active-project scope."""

    active_project_id, current_timeline, current_settings = _job_read_context_for_project(
        request,
        project_id,
    )
    return _job_read_for_request(
        request,
        job,
        current_timeline=current_timeline,
        current_settings=current_settings,
        current_project_id=active_project_id,
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


def _trusted_history_output_descriptor(
    history_entry: Mapping[str, Any],
    *,
    node_id: str,
) -> OutputDescriptor:
    """Extract exactly one safe persistent descriptor from the expected node."""

    outputs = history_entry.get("outputs")
    if not isinstance(outputs, Mapping) or node_id not in outputs:
        raise ExecutionEvidenceConflict(
            "successful history is missing the expected output node"
        )
    candidates: list[OutputDescriptor] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            if "filename" in value:
                if value.get("type") != "output":
                    raise ExecutionEvidenceConflict(
                        "expected output node returned a non-persistent descriptor"
                    )
                try:
                    candidates.append(OutputDescriptor.model_validate(value))
                except ValidationError as exc:
                    raise ExecutionEvidenceConflict(
                        "expected output node returned an unsafe descriptor"
                    ) from exc
                return
            for nested in value.values():
                visit(nested)
            return
        if isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(outputs[node_id])
    if len(candidates) != 1:
        raise ExecutionEvidenceConflict(
            "successful history must contain exactly one expected take descriptor"
        )
    return candidates[0]


async def _observe_output_receipt(
    request: Request,
    child: dict[str, Any],
    receipt: OutputObservationReceipt,
) -> dict[str, Any]:
    """Finish the path-free host probe and atomic observed-artifact publish."""

    database = _db(request)
    existing = database.get_observed_artifact(str(child["id"]))
    if existing is not None:
        latest = database.get_job_child(str(child["id"]))
        if latest is None:
            raise KeyError(child["id"])
        return latest
    provider: HostOutputProbeProvider | None = request.app.state.host_output_probe
    if provider is None:
        return child
    try:
        raw_probe = await anyio.to_thread.run_sync(
            provider.probe_output,
            receipt.output_descriptor,
        )
        probe = HostOutputProbeResult.model_validate(raw_probe)
    except PermanentHostOutputProbeError:
        # A structurally unsafe descriptor, unsupported media type, or path
        # escape cannot heal on a later reconciliation.  Preserve successful
        # history ownership but close local artifact verification explicitly
        # instead of leaving the child in an infinite retry loop.
        return database.fail_output_observation(
            str(child["id"]),
            error="host output cannot be observed safely",
            updated_at=datetime.now(timezone.utc),
        )
    except HostOutputProbeError:
        # The immutable receipt is the retry authority.  A missing media tool,
        # temporarily unavailable file, or bounded probe timeout must not
        # discard exact terminal history or trigger a resubmission.
        latest = database.get_job_child(str(child["id"]))
        return latest if latest is not None else child
    except ValidationError:
        return database.fail_output_observation(
            str(child["id"]),
            error="host output probe returned invalid metadata",
            updated_at=datetime.now(timezone.utc),
        )
    artifact = ObservedArtifactSpec(
        segment_id=receipt.segment_id,
        child_id=receipt.child_id,
        output_descriptor=receipt.output_descriptor,
        width=probe.width,
        height=probe.height,
        fps=probe.fps,
        frame_count=probe.frame_count,
        duration_seconds=probe.duration_seconds,
        has_audio=probe.has_audio,
        media_probe_version=probe.media_probe_version,
        content_hash=None,
    )
    return database.finalize_observed_artifact(
        str(child["id"]),
        artifact=artifact,
        updated_at=datetime.now(timezone.utc),
    )[0]


async def _record_and_observe_typed_success(
    request: Request,
    child: dict[str, Any],
    history_entry: Mapping[str, Any],
    ownership: PromptOwnership,
) -> dict[str, Any]:
    """Bridge exact successful history to an ObservedArtifactSpec."""

    database = _db(request)
    evidence = database.get_job_child_execution_evidence(str(child["id"]))
    if evidence is None:
        raise ExecutionEvidenceConflict(
            "typed prompt ownership has no exact execution evidence"
        )
    snapshot = evidence["exact_prompt_snapshot"]
    expected = snapshot.expected_output_spec
    if snapshot.unit_kind != "segment" or expected is None:
        raise ExecutionEvidenceConflict(
            "only segment prompts can publish observed artifacts"
        )
    observed_at = datetime.now(timezone.utc)
    history_evidence = HistoryTerminalEvidence(
        prompt_id=ownership.effective_prompt_id,
        terminal_status="succeeded",
        history_digest=sha256_document_digest(history_entry),
        observed_at=observed_at,
    )
    try:
        descriptor = _trusted_history_output_descriptor(
            history_entry,
            node_id=expected.node_id,
        )
    except ExecutionEvidenceConflict as exc:
        failed = database.fail_successful_history_artifact(
            str(child["id"]),
            expected_revision=ownership.ownership_revision,
            evidence=history_evidence,
            error=str(exc),
            updated_at=observed_at,
        )
        if failed is not None:
            return failed[0]
        latest = database.get_job_child(str(child["id"]))
        if latest is None:
            raise KeyError(child["id"])
        return latest
    receipt = OutputObservationReceipt(
        child_id=str(child["id"]),
        segment_id=expected.segment_id,
        node_id=expected.node_id,
        output_descriptor=descriptor,
        exact_prompt_snapshot_digest=sha256_document_digest(
            snapshot.model_dump(mode="json")
        ),
        expected_output_spec_digest=sha256_document_digest(
            expected.model_dump(mode="json")
        ),
        history_evidence=history_evidence,
    )
    recorded = database.record_output_observation_receipt(
        str(child["id"]),
        expected_revision=ownership.ownership_revision,
        receipt=receipt,
        updated_at=observed_at,
    )
    if recorded is None:
        latest = database.get_job_child(str(child["id"]))
        durable_receipt = database.get_output_observation_receipt(
            str(child["id"])
        )
        if latest is None:
            raise KeyError(child["id"])
        if durable_receipt is None:
            return latest
        return await _observe_output_receipt(request, latest, durable_receipt)
    return await _observe_output_receipt(request, recorded[0], recorded[2])


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
    client = _comfy(request)

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
    database = _db(request)
    pending_receipt = database.get_output_observation_receipt(str(child["id"]))
    if pending_receipt is not None:
        return await _observe_output_receipt(request, child, pending_receipt)
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

    def commit_history_terminal(
        *,
        status: str,
        stage: str,
        outputs: list[dict[str, Any]],
        error: str | None,
    ) -> dict[str, Any]:
        ownership = _typed_prompt_ownership_for_child(database, child)
        if ownership is None:
            return commit(
                status=status,
                progress=1.0,
                stage=stage,
                outputs=outputs,
                error=error,
                completed_at=utc_now(),
            )
        if ownership.effective_prompt_id != str(child["prompt_id"]):
            raise ExecutionEvidenceConflict(
                "history prompt id does not match durable ownership"
            )
        observed_at = datetime.now(timezone.utc)
        confirmed = database.confirm_prompt_terminal(
            str(child["id"]),
            expected_revision=ownership.ownership_revision,
            evidence=HistoryTerminalEvidence(
                prompt_id=ownership.effective_prompt_id,
                terminal_status=status,
                history_digest=sha256_document_digest(history_entry),
                observed_at=observed_at,
            ),
            outputs=outputs,
            stage=stage,
            error=error,
            updated_at=observed_at,
        )
        if confirmed is not None:
            return confirmed[0]
        latest = database.get_job_child(str(child["id"]))
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
        history_state = _raylight_recovery_history_state(history_entry)
        if history_state == "invalid":
            raise ComfyError(
                "ComfyUI history status is contradictory; refusing to certify prompt ownership"
            )
        failed = history_state == "terminal" and status_value in {
            "error",
            "failed",
        }
        succeeded = history_state == "terminal" and status_value == "success"
        if interrupted:
            return commit_history_terminal(
                status="cancelled",
                stage=("cancelled" if parent_cancelling else "ComfyUI 端已中断"),
                outputs=[],
                error=None,
            )
        if failed or succeeded:
            if succeeded:
                ownership = _typed_prompt_ownership_for_child(database, child)
                if ownership is not None:
                    execution_evidence = (
                        database.get_job_child_execution_evidence(
                            str(child["id"])
                        )
                    )
                    if (
                        execution_evidence is not None
                        and execution_evidence[
                            "exact_prompt_snapshot"
                        ].unit_kind
                        == "segment"
                    ):
                        return await _record_and_observe_typed_success(
                            request,
                            child,
                            history_entry,
                            ownership,
                        )
            return commit_history_terminal(
                status="failed" if failed else "succeeded",
                stage="failed" if failed else "completed",
                outputs=(
                    []
                    if failed
                    else _collect_output_files(history_entry.get("outputs") or {})
                ),
                error=error if failed else None,
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
        ownership = _typed_prompt_ownership_for_child(database, child)
        if ownership is not None:
            if ownership.state not in {
                "cleanup_confirmed",
                "terminal_confirmed",
                "unconfirmed",
            }:
                database.compare_and_set_prompt_ownership(
                    str(child["id"]),
                    expected_revision=ownership.ownership_revision,
                    state="unconfirmed",
                    updated_at=datetime.now(timezone.utc),
                )
            # One exact absence is not release evidence for Stage-4 jobs.
            latest = database.get_job_child(str(child["id"]))
            return latest or child
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
    client = _comfy(request)
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
        evidence = child.get("execution_evidence")
        if isinstance(evidence, dict):
            snapshot = evidence.get("exact_prompt_snapshot")
            expected = (
                snapshot.expected_output_spec
                if hasattr(snapshot, "expected_output_spec")
                else None
            )
            artifact = child.get("observed_artifact")
            if (
                expected is None
                or not isinstance(artifact, ObservedArtifactSpec)
                or artifact.child_id != str(child["id"])
                or artifact.segment_id != expected.segment_id
            ):
                # Typed results are complete only when the immutable expected
                # output and the actual media observation agree.  The mutable
                # child output projection is never recovery authority.
                continue
            if expected.segment_id in output_by_segment:
                raise ValueError(
                    "native workflow returned duplicate output for segment "
                    f"'{expected.segment_id}'"
                )
            output_by_segment[expected.segment_id] = {
                "node_id": expected.node_id,
                **artifact.output_descriptor.model_dump(mode="json"),
            }
            continue
        node_to_segment = _child_output_node_mapping(child)
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


async def _assemble_output_descriptors(
    request: Request,
    job: dict[str, Any],
    outputs: tuple[OutputDescriptor, ...],
) -> tuple[OutputDescriptor, VideoMetadata, str]:
    client = _comfy(request)
    segment_bytes: list[bytes] = []
    for output in outputs:
        response = await client.view(output.model_dump(mode="json"))
        if not response.content:
            raise MediaToolError(
                f"generated segment '{output.filename}' is empty"
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
    filename = f"DirectorDeck_timeline_{job['id'][:8]}_full.mp4"
    uploaded = await client.upload_output(
        filename,
        proxy.content,
        "video/mp4",
        "directordeck/timelines",
    )
    descriptor = OutputDescriptor(
        filename=str(uploaded["name"]),
        subfolder=str(uploaded.get("subfolder") or ""),
        type=str(uploaded.get("type") or "output"),
    )
    return (
        descriptor,
        proxy.metadata,
        f"sha256:{hashlib.sha256(proxy.content).hexdigest()}",
    )


async def _assemble_timeline_output(
    request: Request,
    job: dict[str, Any],
    segment_artifacts: tuple[ObservedArtifactSpec, ...],
) -> tuple[OutputDescriptor, VideoMetadata, str]:
    """Assemble only ordered, host-observed child artifacts."""

    return await _assemble_output_descriptors(
        request,
        job,
        tuple(artifact.output_descriptor for artifact in segment_artifacts),
    )


async def _assemble_legacy_timeline_output(
    request: Request,
    job: dict[str, Any],
    outputs: list[dict[str, str]],
) -> tuple[OutputDescriptor, VideoMetadata, str]:
    """Keep pre-typed in-flight jobs recoverable without upgrading authority."""

    return await _assemble_output_descriptors(
        request,
        job,
        tuple(
            OutputDescriptor.model_validate(
                {
                    "filename": output.get("filename"),
                    "subfolder": output.get("subfolder", ""),
                    "type": output.get("type", "output"),
                }
            )
            for output in outputs
        ),
    )


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
        for child in children:
            if (
                child["backend"] == "raylight"
                and child["status"] in _TERMINAL_STATUSES
                and child.get("prompt_id")
            ):
                database.settle_raylight_runtime_prompt(
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
            execution_plan: CompiledExecutionPlan | None = None
            ordered_artifacts: tuple[ObservedArtifactSpec, ...] = ()
            try:
                enriched_children = [
                    _child_with_execution_evidence(database, child)
                    for child in children
                ]
                execution_plan = database.get_job_execution_plan(str(job["id"]))
                if execution_plan is None:
                    if any(
                        isinstance(child.get("execution_evidence"), dict)
                        for child in enriched_children
                    ):
                        raise TaskManagementError(
                            "任务的 typed 执行计划缺失或损坏",
                            status_code=409,
                        )
                    ordered_outputs = _ordered_timeline_outputs(
                        job,
                        enriched_children,
                    )
                else:
                    authority_job = attach_parent_output_authority(
                        {**job, "children": enriched_children},
                        typed=True,
                        compiled_plan=execution_plan,
                        observed_assembly=None,
                    )
                    ordered_artifacts = ordered_observed_artifacts(authority_job)
                    ordered_outputs = [
                        {
                            "node_id": unit.expected_output_spec.node_id,
                            **artifact.output_descriptor.model_dump(mode="json"),
                        }
                        for unit, artifact in zip(
                            execution_plan.segment_units,
                            ordered_artifacts,
                            strict=True,
                        )
                    ]
            except (
                ExecutionEvidenceConflict,
                RuntimeError,
                TaskManagementError,
                ValidationError,
                ValueError,
            ) as exc:
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
                            if execution_plan is not None:
                                descriptor, metadata, content_hash = (
                                    await _assemble_timeline_output(
                                        request,
                                        claimed,
                                        ordered_artifacts,
                                    )
                                )
                                assembly = ObservedAssemblyArtifactSpec(
                                    job_id=str(job["id"]),
                                    source_compiled_plan_digest=(
                                        compiled_execution_plan_digest(
                                            execution_plan
                                        )
                                    ),
                                    source_artifacts=tuple(
                                        AssemblySourceArtifactRef(
                                            segment_id=artifact.segment_id,
                                            child_id=artifact.child_id,
                                            observed_artifact_digest=(
                                                sha256_document_digest(
                                                    artifact.model_dump(mode="json")
                                                )
                                            ),
                                        )
                                        for artifact in ordered_artifacts
                                    ),
                                    output_descriptor=descriptor,
                                    width=metadata.width,
                                    height=metadata.height,
                                    fps=metadata.native_fps,
                                    frame_count=metadata.frame_count,
                                    duration_seconds=metadata.duration,
                                    has_audio=metadata.has_audio,
                                    media_probe_version=metadata.probe_method,
                                    content_hash=content_hash,
                                )
                                committed = (
                                    database.finalize_observed_assembly_artifact(
                                        str(job["id"]),
                                        expected_updated_at=str(
                                            claimed["updated_at"]
                                        ),
                                        artifact=assembly,
                                        updated_at=datetime.now(timezone.utc),
                                    )
                                )
                                if committed is None:
                                    committed = database.get_job(str(job["id"]))
                                    if committed is None:
                                        raise KeyError(job["id"])
                                committed["children"] = (
                                    database.list_job_children(str(job["id"]))
                                )
                                return committed
                            descriptor, _metadata, _content_hash = (
                                await _assemble_legacy_timeline_output(
                                    request,
                                    claimed,
                                    ordered_outputs,
                                )
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
                        except (
                            ComfyError,
                            ExecutionEvidenceConflict,
                            MediaToolError,
                            TaskManagementError,
                            ValidationError,
                            ValueError,
                            httpx.HTTPError,
                        ) as exc:
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
                                outputs=[
                                    {
                                        "node_id": "assembly",
                                        **descriptor.model_dump(mode="json"),
                                    }
                                ],
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


def _typed_prompt_ownership_for_child(
    database: Database,
    child: dict[str, Any],
) -> PromptOwnership | None:
    """Return the immutable prompt authority for a Stage-4 child.

    A typed ownership row without its immutable exact snapshot, or a mutable
    child projection that no longer names the effective prompt, is corruption.
    Cancellation/recovery must fail closed instead of silently falling back to
    the legacy lifecycle fields in either case.
    """

    child_id = str(child["id"])
    ownership = database.get_prompt_ownership(child_id)
    evidence = database.get_job_child_execution_evidence(child_id)
    if ownership is None and evidence is None:
        markerless_legacy_control = (
            child.get("status") in _TERMINAL_STATUSES
            and not child.get("segment_ids")
            and not child.get("prompt_snapshot")
        )
        # A typed parent pre-creates every child before any prompt is claimed.
        # Such an unsubmitted child legitimately has neither exact evidence nor
        # ownership.  A mutable prompt id, however, proves that submission was
        # projected at some point; if every immutable child marker has vanished
        # we must not reinterpret that submitted typed child as legacy.  The
        # sole migration exception is an already-terminal pre-Stage-4 RayKill
        # row: those controls had no segment and stored an empty prompt snapshot.
        # They are safe to ignore during idempotent cancellation; any later
        # segment continuation independently requires the complete typed control
        # certificate and therefore still fails closed.
        if (
            child.get("prompt_id")
            and not markerless_legacy_control
            and database.has_job_child_execution_marker(child_id)
        ):
            raise ExecutionEvidenceConflict(
                f"child {child_id} lost its exact prompt ownership evidence"
            )
        return None
    if ownership is None or evidence is None:
        raise ExecutionEvidenceConflict(
            f"child {child_id} has incomplete exact prompt ownership evidence"
        )
    if str(child.get("prompt_id") or "") != ownership.effective_prompt_id:
        raise ExecutionEvidenceConflict(
            f"child {child_id} prompt id differs from durable ownership"
        )
    return ownership


def _claim_typed_prompt_cancel(
    database: Database,
    child: dict[str, Any],
) -> PromptOwnership | None:
    """Monotonically claim cancellation while tolerating a receipt race."""

    ownership = _typed_prompt_ownership_for_child(database, child)
    if ownership is None:
        return None
    for _ in range(8):
        if ownership.state in {
            "cancel_pending",
            "cleanup_confirmed",
            "terminal_confirmed",
        }:
            return ownership
        claimed = database.compare_and_set_prompt_ownership(
            str(child["id"]),
            expected_revision=ownership.ownership_revision,
            state="cancel_pending",
            updated_at=datetime.now(timezone.utc),
        )
        if claimed is not None:
            return claimed
        latest = database.get_prompt_ownership(str(child["id"]))
        if latest is None:
            raise ExecutionEvidenceConflict(
                f"prompt ownership for child {child['id']} disappeared"
            )
        ownership = latest
    raise ExecutionEvidenceConflict(
        f"prompt ownership for child {child['id']} did not converge"
    )


def _mark_typed_prompt_unconfirmed(
    database: Database,
    child_id: str,
    ownership: PromptOwnership | None,
) -> PromptOwnership | None:
    """Persist ambiguity without ever clearing an ownership record."""

    if ownership is None or ownership.state in {
        "cleanup_confirmed",
        "terminal_confirmed",
    }:
        return ownership
    if ownership.state == "unconfirmed":
        return ownership
    changed = database.compare_and_set_prompt_ownership(
        child_id,
        expected_revision=ownership.ownership_revision,
        state="unconfirmed",
        updated_at=datetime.now(timezone.utc),
    )
    return changed or database.get_prompt_ownership(child_id)


def _confirm_typed_exact_cancel(
    database: Database,
    child_id: str,
    ownership: PromptOwnership,
    *,
    stage: str,
) -> dict[str, Any] | None:
    """Atomically settle ownership, child lifecycle, and a matching Ray tail."""

    if ownership.state == "terminal_confirmed":
        return database.get_job_child(child_id)
    if ownership.state == "cleanup_confirmed":
        return database.get_job_child(child_id)
    confirmed_at = datetime.now(timezone.utc)
    released = database.confirm_prompt_cleanup(
        child_id,
        expected_revision=ownership.ownership_revision,
        evidence=ExactCancelConfirmedEvidence(
            prompt_id=ownership.effective_prompt_id,
            confirmation_id=f"director-exact-cancel:{ownership.effective_prompt_id}",
            confirmed_at=confirmed_at,
        ),
        stage=stage,
        updated_at=confirmed_at,
    )
    if released is not None:
        return released[0]
    latest_ownership = database.get_prompt_ownership(child_id)
    if latest_ownership is not None and latest_ownership.state in {
        "cleanup_confirmed",
        "terminal_confirmed",
    }:
        return database.get_job_child(child_id)
    return None


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
    current_endpoint = request.app.state.endpoint_identity
    for endpoint_child in database.list_job_children(job["id"]):
        endpoint_ownership = _typed_prompt_ownership_for_child(
            database, endpoint_child
        )
        if endpoint_child["status"] in _TERMINAL_STATUSES:
            if endpoint_ownership is not None and endpoint_ownership.state not in {
                "cleanup_confirmed",
                "terminal_confirmed",
            }:
                raise ExecutionEvidenceConflict(
                    f"terminal child {endpoint_child['id']} retains prompt ownership"
                )
            continue
        if not endpoint_child.get("prompt_id"):
            continue
        if endpoint_ownership is None:
            continue
        exact_evidence = database.get_job_child_execution_evidence(
            str(endpoint_child["id"])
        )
        if exact_evidence is None:  # Guarded by the typed helper above.
            raise ExecutionEvidenceConflict(
                f"child {endpoint_child['id']} lost its exact prompt evidence"
            )
        frozen_endpoint = exact_evidence[
            "exact_prompt_snapshot"
        ].endpoint_identity
        if frozen_endpoint.endpoint_key != current_endpoint.endpoint_key:
            raise HTTPException(
                status_code=409,
                detail="job prompt belongs to a different ComfyUI endpoint",
            )
        if (
            frozen_endpoint.runtime_instance_id
            != current_endpoint.runtime_instance_id
        ):
            _mark_typed_prompt_unconfirmed(
                database,
                str(endpoint_child["id"]),
                endpoint_ownership,
            )
            await _cas_active_child_update(
                database,
                endpoint_child,
                status="cancelling",
                stage="restart_certificate_required",
                error=(
                    "ComfyUI runtime instance changed; explicit restart "
                    "confirmation is required"
                ),
                completed_at=None,
            )
            latest = database.get_job(job["id"])
            if latest is not None and latest["status"] not in _TERMINAL_STATUSES:
                database.update_job_if_snapshot(
                    latest["id"],
                    expected_status=latest["status"],
                    expected_stage=latest.get("stage"),
                    expected_updated_at=latest["updated_at"],
                    status="cancelling",
                    stage="restart_certificate_required",
                    error=(
                        "ComfyUI runtime instance changed; confirm restart "
                        "before retrying"
                    ),
                    completed_at=None,
                )
            raise HTTPException(
                status_code=409,
                detail=(
                    "ComfyUI runtime instance changed; explicit restart "
                    "confirmation is required"
                ),
            )
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

    client = _comfy(request)
    dispatch_errors: list[str] = []
    recovery_error_prefixes: set[str] = set()
    restart_certificate_required = False
    for child in database.list_job_children(job["id"]):
        ownership = _typed_prompt_ownership_for_child(database, child)
        if child["status"] in _TERMINAL_STATUSES:
            if ownership is not None and ownership.state not in {
                "cleanup_confirmed",
                "terminal_confirmed",
            }:
                raise ExecutionEvidenceConflict(
                    f"terminal child {child['id']} retains prompt ownership"
                )
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
        if ownership is not None and ownership.state in {
            "cleanup_confirmed",
            "terminal_confirmed",
        }:
            continue
        if ownership is not None:
            exact_evidence = database.get_job_child_execution_evidence(
                str(child["id"])
            )
            if exact_evidence is None:  # Guarded by the typed helper above.
                raise ExecutionEvidenceConflict(
                    f"child {child['id']} lost its exact prompt evidence"
                )
            frozen_endpoint = exact_evidence[
                "exact_prompt_snapshot"
            ].endpoint_identity
            current_endpoint = request.app.state.endpoint_identity
            if frozen_endpoint.endpoint_key != current_endpoint.endpoint_key:
                raise HTTPException(
                    status_code=409,
                    detail="job prompt belongs to a different ComfyUI endpoint",
                )
            if (
                frozen_endpoint.runtime_instance_id
                != current_endpoint.runtime_instance_id
            ):
                ownership = _mark_typed_prompt_unconfirmed(
                    database, str(child["id"]), ownership
                )
                await _cas_active_child_update(
                    database,
                    child,
                    status="cancelling",
                    stage="restart_certificate_required",
                    error=(
                        "ComfyUI runtime instance changed; explicit restart "
                        "confirmation is required"
                    ),
                    completed_at=None,
                )
                restart_certificate_required = True
                continue
        live_submission_owned = child.get("stage") in _SUBMISSION_OWNERSHIP_STAGES
        if live_submission_owned:
            # The caller-assigned id is only provisional until the in-flight
            # POST returns its authenticated receipt.  Persist the user's
            # parent/child cancellation intent, but let the dispatcher that
            # owns that HTTP request bind the actual id before performing the
            # exact directed cancel.  Cancelling the provisional id here can
            # otherwise make the parent terminal while an incompatible server
            # is about to reveal a different actual id.
            while child["status"] not in _TERMINAL_STATUSES:
                child, claimed = await _cas_active_child_update(
                    database,
                    child,
                    status="cancelling",
                    stage="cancelling_during_submit",
                    completed_at=None,
                )
                if claimed:
                    break
            continue
        ownership = _claim_typed_prompt_cancel(database, child)
        # The restart worker and a user retry can list the same child before
        # either awaits ComfyUI. Claim the exact row version; if the other path
        # won, reload and never turn its terminal result back into cancelling.
        while child["status"] not in _TERMINAL_STATUSES:
            live_submission_owned = False
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
        prompt_id = (
            ownership.effective_prompt_id
            if ownership is not None
            else str(child["prompt_id"])
        )
        try:
            dispatched = await client.cancel(prompt_id)
        except (ComfyError, httpx.HTTPError) as exc:
            dispatch_errors.append(f"{child['id']}: {exc}")
            if recovery_owned:
                recovery_error_prefixes.add(recovery_prefix)
            ownership = _mark_typed_prompt_unconfirmed(
                database, str(child["id"]), ownership
            )
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
        if dispatched and ownership is not None:
            _confirm_typed_exact_cancel(
                database,
                str(child["id"]),
                ownership,
                stage="cancelled",
            )
        elif dispatched and not live_submission_owned:
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
        elif not dispatched:
            _mark_typed_prompt_unconfirmed(
                database, str(child["id"]), ownership
            )
    if restart_certificate_required:
        latest = database.get_job(job["id"])
        if latest is None:
            raise HTTPException(status_code=404, detail="job not found")
        if latest["status"] not in _TERMINAL_STATUSES:
            database.update_job_if_snapshot(
                latest["id"],
                expected_status=latest["status"],
                expected_stage=latest.get("stage"),
                expected_updated_at=latest["updated_at"],
                status="cancelling",
                stage="restart_certificate_required",
                error=(
                    "ComfyUI runtime instance changed; confirm restart before retrying"
                ),
                completed_at=None,
            )
        raise HTTPException(
            status_code=409,
            detail=(
                "ComfyUI runtime instance changed; explicit restart confirmation "
                "is required"
            ),
        )
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
    state: dict[str, Any],
    visible: tuple[int, ...],
) -> str:
    payload = {
        "version": 1,
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
    stats: dict[str, Any],
    *,
    state: dict[str, Any] | None = None,
) -> RayLightRuntimeStatusRead:
    try:
        runtime = (
            state
            if state is not None
            else database.get_raylight_runtime_state()
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
        legacy_unknown = bool(runtime.get("legacy_unknown"))
        recorded = (
            raylight_runtime_logical_gpu_indices(current)
            if isinstance(current, dict)
            else ()
        )
        invalid = tuple(index for index in recorded if index >= len(visible))
        recovery_required = bool(invalid) or legacy_unknown
        return RayLightRuntimeStatusRead(
            active=isinstance(current, dict),
            recovery_required=recovery_required,
            epoch=int(runtime["epoch"]),
            runtime_gpu_indexes=list(recorded),
            available_gpu_indexes=list(visible),
            invalid_gpu_indexes=list(invalid),
            tainted=bool(runtime.get("tainted")),
            recovery_token=(
                _raylight_runtime_recovery_token(runtime, visible)
                if recovery_required
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
    if status.invalid_gpu_indexes:
        message = (
            "旧 RayLight 运行状态引用了当前不可见的 ComfyUI 逻辑 GPU "
            f"{invalid}；请在系统设置中确认 ComfyUI 已重启并恢复 RayLight"
        )
    else:
        message = (
            "检测到无法验证 ownership 的旧版 RayLight 运行状态；请先确认 "
            "ComfyUI 已完整重启并清空旧任务，再恢复 RayLight"
        )
    return {
        "code": "raylight_runtime_restart_confirmation_required",
        "message": message,
        "runtime_gpu_indexes": status.runtime_gpu_indexes,
        "available_gpu_indexes": status.available_gpu_indexes,
        "invalid_gpu_indexes": status.invalid_gpu_indexes,
        "expected_epoch": status.epoch,
        "recovery_token": status.recovery_token,
    }


async def _preflight_execution_plan(
    client: ComfyClientProtocol,
    plan: CompiledExecutionPlan,
    database: Database,
) -> None:
    """Verify only Director-owned Ray ledger facts for the current job.

    Standard plans deliberately skip this check. Host node, media, package,
    model and provenance observations are advisory and are left to ComfyUI's
    actual prompt validation/execution path.
    """

    if not any(unit.backend == "raylight" for unit in plan.segment_units):
        return
    stats = await client.system_stats()
    try:
        runtime_status = _raylight_runtime_status(database, stats)
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
    request: Request,
    client: ComfyClientProtocol,
    unit: PreparedControlUnit,
    database: Database,
) -> None:
    """Verify only Director-owned Ray runtime state before its control prompt.

    Host class observations remain available through catalog/preflight, but
    they do not authorize this production submission. ComfyUI validates the
    actual control prompt and any failure is persisted on this job.
    """

    del request, unit
    stats = await client.system_stats()
    try:
        runtime_status = _raylight_runtime_status(database, stats)
    except NativeTemplateError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "raylight_runtime_state_invalid",
                "message": str(exc),
            },
        ) from exc
    visible = tuple(runtime_status.available_gpu_indexes)
    recorded = tuple(runtime_status.runtime_gpu_indexes)
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
                        ownership = _typed_prompt_ownership_for_child(
                            database, child
                        )
                        if ownership is None:
                            database.update_job_child_if_snapshot(
                                child_id,
                                expected_status=child["status"],
                                expected_updated_at=child["updated_at"],
                                status="failed",
                                progress=1.0,
                                stage="RayLight 安全切换失败",
                                outputs=[],
                                error=(
                                    error
                                    or "RayLight safe-switch barrier failed"
                                ),
                                completed_at=utc_now(),
                            )
                        else:
                            observed_at = datetime.now(timezone.utc)
                            database.confirm_prompt_terminal(
                                child_id,
                                expected_revision=(
                                    ownership.ownership_revision
                                ),
                                evidence=HistoryTerminalEvidence(
                                    prompt_id=ownership.effective_prompt_id,
                                    terminal_status="failed",
                                    history_digest=sha256_document_digest(entry),
                                    observed_at=observed_at,
                                ),
                                outputs=[],
                                stage="RayLight 安全切换失败",
                                error=(
                                    error
                                    or "RayLight safe-switch barrier failed"
                                ),
                                updated_at=observed_at,
                            )
                raise ComfyError(
                    "RayLight safe-switch barrier failed; Standard workflow was not submitted"
                    + (f": {error}" if error else "")
                )
            if succeeded:
                if database is not None and child_id is not None:
                    child = database.get_job_child(child_id)
                    if child is not None and child["status"] not in _TERMINAL_STATUSES:
                        ownership = _typed_prompt_ownership_for_child(
                            database, child
                        )
                        if ownership is not None:
                            observed_at = datetime.now(timezone.utc)
                            database.confirm_prompt_terminal(
                                child_id,
                                expected_revision=(
                                    ownership.ownership_revision
                                ),
                                evidence=HistoryTerminalEvidence(
                                    prompt_id=ownership.effective_prompt_id,
                                    terminal_status="succeeded",
                                    history_digest=sha256_document_digest(entry),
                                    observed_at=observed_at,
                                ),
                                outputs=[],
                                stage="RayLight 已安全释放",
                                error=None,
                                updated_at=observed_at,
                            )
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
    request: Request,
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
        pending_receipt = database.get_output_observation_receipt(child_id)
        if pending_receipt is not None:
            child = await _observe_output_receipt(
                request,
                child,
                pending_receipt,
            )
            if child["status"] in _TERMINAL_STATUSES:
                return child
            await asyncio.sleep(_RAYLIGHT_GENERATION_POLL_SECONDS)
            continue
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
            # Exact terminal history has one classifier for every backend.
            # Accepting a bare ``success`` string or an unrelated completed
            # flag here would let a contradictory Standard entry publish an
            # ObservedArtifactSpec and unlock dependent submissions even
            # though the normal reconciliation path correctly rejects it.
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
            if interrupted or failed or succeeded:
                terminal_status = (
                    "cancelled"
                    if interrupted
                    else "failed"
                    if failed
                    else "succeeded"
                )
                updates = {
                    "status": terminal_status,
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
                ownership = _typed_prompt_ownership_for_child(database, child)
                if ownership is not None:
                    execution_evidence = (
                        database.get_job_child_execution_evidence(child_id)
                    )
                    if (
                        terminal_status == "succeeded"
                        and execution_evidence is not None
                        and execution_evidence[
                            "exact_prompt_snapshot"
                        ].unit_kind
                        == "segment"
                    ):
                        observed = await _record_and_observe_typed_success(
                            request,
                            child,
                            entry,
                            ownership,
                        )
                        if observed["status"] in _TERMINAL_STATUSES:
                            return observed
                        # Exact successful history is already frozen in the
                        # output receipt.  A temporarily unavailable host
                        # probe remains retryable and must keep this endpoint
                        # gate closed; returning a verifying child would make
                        # the continuity dispatcher permanently fail every
                        # descendant and would misclassify a Ray success as a
                        # runtime failure.
                        await asyncio.sleep(_RAYLIGHT_GENERATION_POLL_SECONDS)
                        continue
                    observed_at = datetime.now(timezone.utc)
                    confirmed = database.confirm_prompt_terminal(
                        child_id,
                        expected_revision=ownership.ownership_revision,
                        evidence=HistoryTerminalEvidence(
                            prompt_id=ownership.effective_prompt_id,
                            terminal_status=terminal_status,
                            history_digest=sha256_document_digest(entry),
                            observed_at=observed_at,
                        ),
                        outputs=updates["outputs"],
                        stage=str(updates["stage"]),
                        error=updates["error"],
                        updated_at=observed_at,
                    )
                    if confirmed is not None:
                        return confirmed[0]
                    continue
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
            ownership = _typed_prompt_ownership_for_child(database, child)
            if ownership is not None:
                _mark_typed_prompt_unconfirmed(
                    database, child_id, ownership
                )
                absent_observations = 0
                await asyncio.sleep(_RAYLIGHT_GENERATION_POLL_SECONDS)
                continue
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
    request: Request,
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
        request=request,
        stop_on_parent_cancel=stop_on_parent_cancel,
        dispatch_job_id=dispatch_job_id,
        running_stage="RayLight 采样中",
        error_context="RayLight generation",
        terminal_events=terminal_events,
    )


async def _refresh_raylight_runtime_tail(
    client: ComfyClientProtocol,
    database: Database,
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
        database.put_raylight_runtime_state(state)
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
            database.put_raylight_runtime_state(recovered)
            return recovered
        database.settle_raylight_runtime_prompt(
            tail_prompt_id,
            succeeded=result,
            terminal_history_certified=terminal_history_certified,
        )
        return database.get_raylight_runtime_state() or state

    def keep_ambiguous_tail() -> dict[str, Any]:
        guarded = dict(state, tainted=True)
        database.put_raylight_runtime_state(guarded)
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
            database.put_raylight_runtime_state(state)
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
        ownership = (
            _typed_prompt_ownership_for_child(database, child)
            if child is not None
            else None
        )
        if ownership is not None and ownership.state in {
            "cleanup_confirmed",
            "terminal_confirmed",
        }:
            continue
        if ownership is not None and ownership.state != "cancel_pending":
            claimed_ownership = database.compare_and_set_prompt_ownership(
                child_id,
                expected_revision=ownership.ownership_revision,
                state="cancel_pending",
                updated_at=datetime.now(timezone.utc),
            )
            if claimed_ownership is None:
                uncertain.append(f"{child_id}: ownership changed before cancel")
                continue
            ownership = claimed_ownership
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
            if ownership is not None:
                database.compare_and_set_prompt_ownership(
                    child_id,
                    expected_revision=ownership.ownership_revision,
                    state="unconfirmed",
                    updated_at=datetime.now(timezone.utc),
                )
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
            if ownership is not None:
                database.compare_and_set_prompt_ownership(
                    child_id,
                    expected_revision=ownership.ownership_revision,
                    state="unconfirmed",
                    updated_at=datetime.now(timezone.utc),
                )
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
        if ownership is not None:
            confirmed_at = datetime.now(timezone.utc)
            released = database.confirm_prompt_cleanup(
                child_id,
                expected_revision=ownership.ownership_revision,
                evidence=ExactCancelConfirmedEvidence(
                    prompt_id=ownership.effective_prompt_id,
                    confirmation_id=(
                        f"director-exact-cancel:{ownership.effective_prompt_id}"
                    ),
                    confirmed_at=confirmed_at,
                ),
                stage="cancelled_after_submission_failure",
                updated_at=confirmed_at,
            )
            if released is None:
                uncertain.append(
                    f"{child_id}: ownership changed before cancel confirmation"
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


def _mark_timeline_submission_interrupted(
    database: Database,
    job_id: str,
) -> None:
    """Persist a finite recovery owner after any unexpected submit exit."""

    current = database.get_job(job_id)
    if current is None or current["status"] in _TERMINAL_STATUSES:
        return
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
    if latest is None or latest["status"] in _TERMINAL_STATUSES:
        return
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
        or actual != actual.strip()
        or len(actual) > 512
        or any(
            ord(character) < 0x20 or ord(character) == 0x7F
            for character in actual
        )
    ):
        return set()
    candidates: list[str] = []
    for child_id, durable_prompt_id in possibly_submitted.items():
        candidate_child = database.get_job_child(child_id)
        candidate_ownership = (
            _typed_prompt_ownership_for_child(database, candidate_child)
            if candidate_child is not None
            else None
        )
        if candidate_ownership is not None:
            if (
                candidate_ownership.requested_prompt_id == requested
                and candidate_ownership.effective_prompt_id
                in {requested, actual}
            ):
                candidates.append(child_id)
        elif durable_prompt_id == requested:
            candidates.append(child_id)
    if len(candidates) != 1:
        return set()
    child_id = candidates[0]
    ownership_child = database.get_job_child(child_id)
    ownership = (
        _typed_prompt_ownership_for_child(database, ownership_child)
        if ownership_child is not None
        else None
    )
    cleanup_response = error.detail.get("cleanup_response")
    cleanup_confirmed = (
        isinstance(cleanup_response, dict)
        and set(cleanup_response) == {"cancelled"}
        and cleanup_response.get("cancelled") is True
    )
    if (
        ownership is not None
        and ownership.requested_prompt_id == requested
    ):
        if actual == requested:
            # A receipt hook may fail before it advances ownership even when
            # ComfyUI honored the caller-assigned id.  Preserve the inline
            # cleanup acknowledgement instead of issuing a second cancel;
            # when cleanup was not confirmed, the normal outer path retains
            # and targets the already-durable requested id.
            if cleanup_confirmed:
                confirmed_at = datetime.now(timezone.utc)
                released = database.confirm_prompt_cleanup(
                    child_id,
                    expected_revision=ownership.ownership_revision,
                    updated_at=confirmed_at,
                    evidence=ExactCancelConfirmedEvidence(
                        prompt_id=actual,
                        confirmation_id=f"comfy-atomic-cancel:{actual}",
                        confirmed_at=confirmed_at,
                    ),
                    stage="cancelled_after_submission_failure",
                )
                if released is not None:
                    return {child_id}
            return set()
        if ownership.effective_prompt_id == requested:
            rebound = database.record_prompt_submission_receipt(
                child_id,
                expected_revision=ownership.ownership_revision,
                actual_prompt_id=actual,
                state="owned_actual_id",
                updated_at=datetime.now(timezone.utc),
            )
            if rebound is None:
                return set()
            ownership = rebound
        elif ownership.effective_prompt_id != actual:
            return set()
        possibly_submitted[child_id] = actual
        if cleanup_confirmed:
            confirmed_at = datetime.now(timezone.utc)
            released = database.confirm_prompt_cleanup(
                child_id,
                expected_revision=ownership.ownership_revision,
                updated_at=confirmed_at,
                evidence=ExactCancelConfirmedEvidence(
                    prompt_id=actual,
                    confirmation_id=f"comfy-atomic-cancel:{actual}",
                    confirmed_at=confirmed_at,
                ),
                stage="cancelled_after_submission_failure",
            )
            if released is not None:
                return {child_id}
        return set()

    # Legacy submissions have no typed ownership row. Preserve the old
    # best-effort projection until their historical lifecycle is exhausted.
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
        if cleanup_confirmed:
            return {child_id}
    return set()


def _resolve_historical_continuity_takes(
    database: Database,
    draft: UnifiedTimelineDraft,
    *,
    segment_ids: list[str] | None,
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
            take = database.find_latest_observed_segment_take(
                predecessor.id,
                fingerprint,
                require_audio=require_audio,
                project_id=project_id,
            )
            has_exact_observed = (
                take is None
                and database.has_observed_segment_take(
                    predecessor.id,
                    content_fingerprint=fingerprint,
                    project_id=project_id,
                )
            )
            has_any_observed = (
                take is None
                and not has_exact_observed
                and database.has_observed_segment_take(
                    predecessor.id,
                    project_id=project_id,
                )
            )
        except (
            ExecutionEvidenceConflict,
            NativeTemplateError,
            TypeError,
            ValueError,
        ) as exc:
            raise DraftNotRunnable(
                f"片段 '{segment.id}' 的直接前驱 '{predecessor.id}' "
                "已有历史成片记录无效，无法用于接续"
            ) from exc
        if take is None:
            if require_audio and has_exact_observed:
                raise DraftNotRunnable(
                    f"片段 '{segment.id}' 的直接前驱 '{predecessor.id}' "
                    "有输出规格匹配的历史成功成片，但不含生成音频接续所需的音轨"
                )
            if has_exact_observed or has_any_observed:
                raise DraftNotRunnable(
                    f"片段 '{segment.id}' 的直接前驱 '{predecessor.id}' "
                    "存在历史成功成片，但分辨率、帧率或可见帧数与当前分段不一致"
                )
            if database.has_segment_take(
                predecessor.id,
                project_id=project_id,
            ):
                raise DraftNotRunnable(
                    f"片段 '{segment.id}' 的直接前驱 '{predecessor.id}' "
                    "只有旧任务输出定位记录，实际媒体规格与音轨信息不可用；请重新生成前驱"
                )
            raise DraftNotRunnable(
                f"片段 '{segment.id}' 的直接前驱 '{predecessor.id}' "
                "没有可用的历史成功成片"
            )
        resolved[segment.id] = NativeHistoricalTake(
            id=str(take["id"]),
            segment_id=predecessor.id,
            output=dict(take["output"]),
        )
    return resolved


def _project_summary(project: dict[str, Any]) -> ProjectSummaryRead:
    """Project a project row into its public summary without leaking the document."""

    try:
        timeline = validate_timeline_draft_v5(project["document"])
        segment_count = len(timeline.segments)
    except (TypeError, ValueError, ValidationError):
        segment_count = 0
    return ProjectSummaryRead(
        id=project["id"],
        title=project["title"],
        created_at=project["created_at"],
        updated_at=project["updated_at"],
        segment_count=segment_count,
    )


def _contextual_host_reason(
    *,
    code: str,
    rule: str,
    message: str,
    remediation: str,
    safe_details: dict[str, Any] | None = None,
) -> CapabilityReason:
    return CapabilityReason(
        code=code,
        feature_id=None,
        segment_id=None,
        unit_id=None,
        backend=None,
        rule=rule,
        message=message,
        remediation=remediation,
        safe_details=safe_details or {},
    )


@dataclass(frozen=True, slots=True)
class _ContextualHostRequirements:
    """Exact non-node host resources used by the selected v4 routes."""

    model_bindings: tuple[tuple[str, str, str], ...]
    standard_placements: tuple[str, ...]


_MODEL_INVENTORY_CATEGORIES = (
    "fl2va",
    "ref2va",
    "clip",
    "video_vae",
    "audio_vae",
    "loras",
)


def _validated_model_inventory(value: object) -> dict[str, frozenset[str]]:
    """Normalize ComfyUI model inventory without trusting its JSON shape."""

    if not isinstance(value, Mapping):
        raise ComfyError("invalid model inventory")
    normalized: dict[str, frozenset[str]] = {}
    for category in _MODEL_INVENTORY_CATEGORIES:
        items = value.get(category)
        if (
            not isinstance(items, list)
            or len(items) > 100_000
            or any(
                not isinstance(item, str) or not item or len(item) > 4_096
                for item in items
            )
        ):
            raise ComfyError("invalid model inventory")
        normalized[category] = frozenset(items)
    return normalized


def _contextual_host_requirements(
    draft: UnifiedTimelineDraft,
    settings: RuntimeSettings,
    segment_ids: list[str] | None,
) -> _ContextualHostRequirements:
    """Resolve model files/devices without emitting a workflow graph.

    The projection mirrors the v4 shared-model and model-path interpreters.
    In particular the audio VAE is reachable only for Ref2VA or generated
    audio, and RayLight diffusion placement is evaluated by the capability
    evaluator against its CUDA-only logical inventory.
    """

    selected_ids = (
        set(segment_ids)
        if segment_ids is not None
        else {segment.id for segment in draft.segments if segment.enabled}
    )
    segments = tuple(
        segment
        for segment in draft.segments
        if segment.enabled and segment.id in selected_ids
    )
    families = tuple(dict.fromkeys(segment.mode for segment in segments))
    needs_audio_vae = any(
        segment.mode == "ref2va" or segment.audio_mode == "generate"
        for segment in segments
    )

    bindings: list[tuple[str, str, str]] = [
        ("clip", "clip", settings.models.clip.filename),
        ("video_vae", "video_vae", settings.models.video_vae.filename),
    ]
    placements = {
        settings.models.clip.device,
        settings.models.video_vae.device,
    }
    if needs_audio_vae:
        bindings.append(
            ("audio_vae", "audio_vae", settings.models.audio_vae.filename)
        )
        placements.add(settings.models.audio_vae.device)

    for family in families:
        binding = getattr(settings.models, family)
        bindings.append((family, family, binding.filename))
        if resolve_execution_backend(binding) == "standard":
            placements.add(binding.device)
        if binding.lora_name is not None:
            bindings.append(("loras", f"loras:{family}", binding.lora_name))

    return _ContextualHostRequirements(
        model_bindings=tuple(bindings),
        standard_placements=tuple(sorted(placements)),
    )


async def _contextual_host_errors(
    client: ComfyClientProtocol,
    *,
    draft: UnifiedTimelineDraft,
    settings: RuntimeSettings,
    segment_ids: list[str] | None,
    snapshot: HostCapabilitySnapshot,
) -> tuple[CapabilityReason, ...]:
    """Read-only exact model and Standard-placement checks for v4 routes."""

    requirements = _contextual_host_requirements(draft, settings, segment_ids)
    inventory = _validated_model_inventory(await client.models())

    errors: list[CapabilityReason] = []
    absent = sorted(
        role
        for category, role, filename in requirements.model_bindings
        if filename
        not in inventory.get(category, [])
    )
    if absent:
        errors.append(
            _contextual_host_reason(
                code="model_binding_unavailable",
                rule="host_model_inventory",
                message="One or more selected model files are unavailable.",
                remediation="Select files reported by the current ComfyUI host and run preflight again.",
                safe_details={"bindings": absent},
            )
        )

    available_devices = {"default", "cpu"} | {
        f"gpu:{item.logical_index}" for item in snapshot.gpu_inventory
    }
    invalid_devices = sorted(
        device
        for device in requirements.standard_placements
        if device not in available_devices
    )
    if invalid_devices:
        errors.append(
            _contextual_host_reason(
                code="runtime_placement_unavailable",
                rule="host_logical_gpu_inventory",
                message="One or more selected runtime devices are unavailable.",
                remediation="Select logical devices reported by the current ComfyUI host and run preflight again.",
                safe_details={"devices": invalid_devices},
            )
        )

    return tuple(errors)


async def _project_import_capability_issues(
    request: Request,
    draft_v5: UnifiedTimelineDraftV5,
) -> list[dict[str, Any]]:
    """Observe current host compatibility without making import contingent on it."""

    if draft_v5.features.template_bundle_version == 6:
        # Bundle-6 import validity is established by its pure project
        # contract. Live host observations are advisory and not an import
        # token; CK has its own bounded diagnostic endpoint.
        return []
    database = _db(request)
    captured_settings = database.get_settings()
    try:
        projection = project_v5_compile_authority(
            draft_v5,
            captured_settings,
        )
    except V5CreativeAuthorityError as exc:
        return [_v5_creative_authority_reason(exc).model_dump(mode="json")]
    try:
        snapshot = await _host_capability_snapshot(request)
        readiness = _host_operational_readiness(request, snapshot)
        client = _comfy(request)
        contextual = await _contextual_host_errors(
            client,
            draft=projection.draft,
            settings=projection.settings,
            segment_ids=None,
            snapshot=snapshot,
        )
        feature_report = preflight_projected_v5_timeline(
            draft=projection.draft,
            settings=projection.settings,
            effective_features=projection.effective_features,
            snapshot=snapshot,
            readiness=readiness,
            segment_ids=None,
            historical_takes={},
            resolved_lora_adapters=projection.lora_adapter_map(),
        )
    except (ComfyError, httpx.HTTPError, ValidationError, ValueError):
        return [_host_context_observation_reason().model_dump(mode="json")]
    return [
        reason.model_dump(mode="json")
        for reason in (*feature_report.errors, *contextual)
    ]


def _compile_report_for_preview(
    execution_plan: CompiledExecutionPlan,
) -> CompiledExecutionReportV2 | CompiledExecutionReportV3:
    raw = execution_plan.model_dump(mode="json")["compile_report"]
    encoded = json.dumps(
        raw,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    try:
        if execution_plan.version == 2:
            return CompiledExecutionReportV2.model_validate_json(encoded)
        if execution_plan.version == 3:
            return CompiledExecutionReportV3.model_validate_json(encoded)
    except ValidationError:
        logger.exception("compiled execution report failed its typed invariant")
        raise _execution_plan_invariant_http_error() from None
    raise _execution_plan_invariant_http_error()


async def _compile_timeline_report(
    request: Request,
    body: TimelineJobRequest,
    *,
    project_id: str | None,
) -> TimelineCompileRead:
    """Compile through the same immutable plan authority as submission."""

    database = _db(request)
    captured_settings = database.get_settings()
    owner_project_id = project_id or database.LEGACY_DEFAULT_PROJECT_ID
    if not database.project_exists(owner_project_id):
        raise _project_not_found_http_error()
    # ``config`` is required by the timeline-schema-5 request. Project lookup above
    # establishes ownership scope only; compilation never reloads its mutable
    # document as a fallback.
    draft_v5 = body.config
    try:
        database.validate_timeline_assets(
            draft_v5,
            segment_ids=body.segment_ids,
        )
        validate_unified_runnable(draft_v5, segment_ids=body.segment_ids)
        historical_takes = _resolve_historical_continuity_takes(
            database,
            draft_v5,
            segment_ids=body.segment_ids,
            project_id=owner_project_id,
        )
        execution_plan = compile_project_execution_plan(
            draft_v5,
            captured_settings,
            f"preview-{uuid.uuid4()}",
            body.segment_ids,
            historical_takes=historical_takes,
        )
    except V5CreativeAuthorityError as exc:
        raise _creative_input_http_error(
            _v5_creative_authority_reason(exc),
        ) from exc
    except (
        ProjectCompilerBundleError,
        V5V6ProjectionError,
        NativeTemplateError,
        DraftNotRunnable,
        ValidationError,
        ValueError,
    ) as exc:
        reason = _creative_input_reason(exc)
        raise _creative_input_http_error(reason) from exc
    plan_document = execution_plan.model_dump(mode="json")
    report = _compile_report_for_preview(execution_plan)
    # Bundle 5 keeps this legacy response field for wire compatibility only.
    # Compile and submission no longer capture a live host observation or use
    # one as authorization, so the stable value explicitly identifies the
    # compiled-plan authority instead of pretending to be a host token.
    host_revision = (
        getattr(report, "host_capability_revision", None)
        or _LEGACY_COMPILE_OBSERVATION_REVISION
    )
    if isinstance(report, CompiledExecutionReportV2):
        feature_resolutions = report.feature_resolutions
        resolutions_by_segment: dict[str, list[Any]] = {}
        for resolution in feature_resolutions:
            resolutions_by_segment.setdefault(resolution.segment_id, []).append(
                resolution
            )
        effective_by_segment: dict[str, dict[str, Any]] = {}
        for segment_id, actual in resolutions_by_segment.items():
            if not actual:
                raise _execution_plan_invariant_http_error()
            first = actual[0]
            if any(
                feature.unit_id != first.unit_id
                or feature.backend != first.backend
                or feature.family != first.family
                or feature.template_id != first.template_id
                for feature in actual
            ):
                logger.error(
                    "compiled feature evidence contains inconsistent route identity",
                    extra={"segment_id": segment_id},
                )
                raise _execution_plan_invariant_http_error()
            effective_by_segment[segment_id] = {
                "unit_id": first.unit_id,
                "backend": first.backend,
                "family": first.family,
                "template_id": first.template_id,
                "features": [
                    {
                        "id": feature.feature_id,
                        "version": feature.version,
                        "state": feature.resolution.state,
                        "adapter_fingerprint": feature.adapter_fingerprint,
                        "capability": feature.capability,
                    }
                    for feature in actual
                ],
            }
        features_payload: dict[str, Any] = {
            "requested": draft_v5.features.model_dump(mode="json"),
            "effective_by_segment": effective_by_segment,
            "resolutions": [
                {
                    "segment_id": resolution.segment_id,
                    "unit_id": resolution.unit_id,
                    "feature_id": resolution.feature_id,
                    "version": resolution.version,
                    "backend": resolution.backend,
                    "family": resolution.family,
                    "template_id": resolution.template_id,
                    "resolution": resolution.resolution,
                    "adapter_fingerprint": resolution.adapter_fingerprint,
                    "capability": resolution.capability,
                }
                for resolution in feature_resolutions
            ],
            "uses": [],
            "notices": list(report.notices),
            "advisories": [],
        }
    else:
        features_payload = {
            "requested": draft_v5.features.model_dump(mode="json"),
            "effective_by_segment": {},
            "resolutions": [],
            "uses": list(report.feature_resolutions),
            "notices": list(report.notices),
            "advisories": list(report.advisories),
        }
    return TimelineCompileRead(
        template_bundle_version=execution_plan.template_bundle_version,
        host_capability_revision=host_revision,
        model_families=list(report.families),
        plans=plan_document["compile_report"]["plans"],
        node_policy=plan_document["node_policy"],
        features=features_payload,
        effective_execution_digest=(
            execution_plan.effective_execution_digest.model_dump(mode="json")
        ),
    )


def _execution_continuity_graph(
    plan: CompiledExecutionPlan,
) -> dict[str, tuple[str, ...]]:
    """Validate and index dependencies from the sole compiled-plan authority."""

    units_by_segment: dict[str, PreparedSegmentUnit] = {}
    for unit in plan.segment_units:
        segment_id = unit.owner_segment_id
        if segment_id in units_by_segment:
            raise NativeTemplateError(
                f"continuity segment '{segment_id}' has multiple prepared units"
            )
        output = unit.expected_output_spec
        output_node = unit.prompt_base.get(output.node_id)
        if not isinstance(output_node, Mapping) or output_node.get(
            "class_type"
        ) != "SaveVideo":
            raise NativeTemplateError(
                f"prepared unit '{unit.id}' must declare its unique SaveVideo "
                f"output for segment '{segment_id}'"
            )
        units_by_segment[segment_id] = unit

    mutable_dependents: dict[str, list[str]] = {}
    predecessor_by_segment: dict[str, str] = {}
    for segment_id, unit in units_by_segment.items():
        dependency = unit.continuity_dependency
        if dependency is None:
            continue
        predecessor_id = str(dependency.get("predecessor_segment_id") or "")
        source = dependency.get("source")
        if source == "historical_take":
            if (
                dependency.get("resolved") is not True
                or not dependency.get("historical_take_id")
                or predecessor_id in units_by_segment
            ):
                raise NativeTemplateError(
                    f"continuity segment '{segment_id}' has an invalid "
                    "historical-take dependency"
                )
            continue
        if source != "same_run":
            raise NativeTemplateError(
                f"continuity segment '{segment_id}' has an unknown dependency source"
            )
        if predecessor_id not in units_by_segment:
            raise NativeTemplateError(
                f"continuity segment '{segment_id}' requires unselected predecessor "
                f"'{predecessor_id}'"
            )
        if dependency.get("resolved") or dependency.get("historical_take_id") is not None:
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

    return {
        predecessor_id: tuple(successor_ids)
        for predecessor_id, successor_ids in mutable_dependents.items()
    }


def _continuity_output_descriptor(
    database: Database,
    child: dict[str, Any],
    predecessor_segment_id: str,
) -> dict[str, str]:
    """Resolve exactly one persistent SaveVideo result for a predecessor."""

    execution_evidence = database.get_job_child_execution_evidence(
        str(child["id"])
    )
    if execution_evidence is not None:
        expected = execution_evidence[
            "exact_prompt_snapshot"
        ].expected_output_spec
        artifact = database.get_observed_artifact(str(child["id"]))
        if (
            expected is None
            or expected.segment_id != predecessor_segment_id
            or artifact is None
            or artifact.segment_id != predecessor_segment_id
            or artifact.child_id != str(child["id"])
        ):
            raise NativeTemplateError(
                f"continuity predecessor '{predecessor_segment_id}' has no "
                "trusted observed artifact"
            )
        return artifact.output_descriptor.model_dump(mode="json")

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


def _fail_admitted_timeline_job(
    database: Database,
    job_id: str,
    *,
    detail: Any,
    stage: str = "compile_failed",
) -> dict[str, Any]:
    """Persist a local failure after the public job row already exists."""

    error = (
        detail
        if isinstance(detail, str)
        else json.dumps(detail, ensure_ascii=False)
    )
    failed = database.update_job_if_status(
        job_id,
        "preparing",
        status="failed",
        progress=1.0,
        stage=stage,
        error=error,
        completed_at=utc_now(),
    )
    if failed is not None:
        for child in database.list_job_children(job_id):
            if child["status"] in _TERMINAL_STATUSES:
                continue
            database.update_job_child_if_snapshot(
                child["id"],
                expected_status=child["status"],
                expected_updated_at=child["updated_at"],
                status="failed",
                progress=1.0,
                stage=stage,
                error=error,
                completed_at=utc_now(),
            )
        return failed
    latest = database.get_job(job_id)
    if latest is None:
        raise KeyError(job_id)
    return latest


async def _create_timeline_job_impl(
    request: Request,
    body: TimelineJobRequest,
    *,
    parent_mode: str = "timeline",
    job_id: str | None = None,
    accepted: asyncio.Event | None = None,
    accepted_release: asyncio.Event | None = None,
    captured_settings_future: asyncio.Future[RuntimeSettingsV3] | None = None,
    project_id: str | None = None,
) -> JobRead:
    database = _db(request)
    captured_settings = database.get_settings()
    if captured_settings_future is not None and not captured_settings_future.done():
        captured_settings_future.set_result(captured_settings)
    client = _comfy(request)
    owner_project_id = project_id or database.LEGACY_DEFAULT_PROJECT_ID

    def project_job_read(snapshot: dict[str, Any]) -> JobRead:
        # The accepted response is derived from the two snapshots captured at
        # request admission.  Reading a mutable live project/settings pair
        # here would reintroduce split creative authority into job creation.
        return _job_read_for_request(
            request,
            snapshot,
            current_timeline=draft_v5,
            current_settings=captured_settings,
        )

    if not database.project_exists(owner_project_id):
        raise _project_not_found_http_error()
    # The request is the immutable creative snapshot.  This scope check must
    # not become a second document read or a live-settings creative fallback.
    draft_v5 = body.config
    try:
        if draft_v5.features.template_bundle_version not in {5, 6}:
            raise ProjectCompilerBundleError(
                draft_v5.features.template_bundle_version
            )
        # Exact feature/adapter projection belongs to the one post-admission
        # compiler call. This admission pass checks only cheap creative scope
        # so any local compiler failure is persisted on the accepted job.
        draft = draft_v5
        settings = captured_settings
        database.validate_timeline_assets(
            draft,
            segment_ids=body.segment_ids,
        )
        validate_unified_runnable(draft, segment_ids=body.segment_ids)
    except V5CreativeAuthorityError as exc:
        raise _creative_input_http_error(
            _v5_creative_authority_reason(exc)
        ) from exc
    except (
        ProjectCompilerBundleError,
        V5V6ProjectionError,
        ValidationError,
        ValueError,
    ) as exc:
        raise _creative_input_http_error(_creative_input_reason(exc)) from exc
    job_id = job_id or str(uuid.uuid4())
    now = utc_now()
    database.create_job(
        {
            "id": job_id,
            "mode": parent_mode,
            "status": "preparing",
            "progress": 0.0,
            "stage": "compiling",
            "prompt_id": None,
            "project_id": owner_project_id if parent_mode == "timeline" else None,
            "outputs": [],
            "error": None,
            "config_snapshot": {
                "timeline": draft_v5.model_dump(mode="json"),
                "segment_ids": body.segment_ids,
            },
            # The bounded runtime and prompt snapshots are filled from the one
            # immutable compiler result below. This empty value exists only
            # while the admitted background task is in its local compile stage.
            "settings_snapshot": {},
            "prompt_snapshot": None,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )
    # Preserve HTTP admission order without making compilation part of the
    # public response.  Later jobs may compile in parallel, but their ComfyUI
    # side effects remain chained behind this durable parent.
    endpoint_key = _EMBEDDED_ENDPOINT_KEY
    submission_ticket = asyncio.get_running_loop().create_future()
    predecessor: asyncio.Future[None] | None
    async with request.app.state.submission_ticket_lock:
        predecessor = request.app.state.submission_tails.get(endpoint_key)
        request.app.state.submission_tails[endpoint_key] = submission_ticket
    ticket_settlement_requested = False

    def release_submission_ticket(
        _completed_predecessor: asyncio.Future[None] | None = None,
    ) -> None:
        if not submission_ticket.done():
            submission_ticket.set_result(None)
        if request.app.state.submission_tails.get(endpoint_key) is submission_ticket:
            request.app.state.submission_tails.pop(endpoint_key, None)

    def settle_submission_ticket() -> None:
        nonlocal ticket_settlement_requested
        if ticket_settlement_requested:
            return
        ticket_settlement_requested = True
        if predecessor is not None and not predecessor.done():
            # A failed/cancelled job still proxies the unfinished predecessor,
            # so a third admission cannot jump across the original order.
            predecessor.add_done_callback(release_submission_ticket)
        else:
            release_submission_ticket()

    if accepted is not None:
        accepted.set()
        if accepted_release is not None:
            try:
                await accepted_release.wait()
            except BaseException:
                settle_submission_ticket()
                raise

    try:
        before_timeline_compile = getattr(
            request.app.state, "before_timeline_compile", None
        )
        if before_timeline_compile is not None:
            await before_timeline_compile(job_id)
        historical_takes = _resolve_historical_continuity_takes(
            database,
            draft,
            segment_ids=body.segment_ids,
            project_id=owner_project_id,
        )
        execution_plan = compile_project_execution_plan(
            draft_v5,
            captured_settings,
            job_id,
            body.segment_ids,
            historical_takes=historical_takes,
        )
        execution_plan_digest = compiled_execution_plan_digest(execution_plan)
        continuity_dependents = _execution_continuity_graph(execution_plan)
        job_runtime_snapshot = build_job_runtime_snapshot(
            draft_v5,
            body.segment_ids,
            captured_settings,
            execution_plan,
        )
        compile_report = execution_plan.model_dump(mode="json")["compile_report"]
        compiled_manifest = compile_report.get("manifest")
        if not isinstance(compiled_manifest, dict):
            raise _execution_plan_invariant_http_error()
        prepared_parent = database.update_job_if_status(
            job_id,
            "preparing",
            stage="preflight",
            settings_snapshot=job_runtime_snapshot.model_dump(mode="json"),
            prompt_snapshot=compiled_manifest,
        )
        if prepared_parent is None:
            latest = database.get_job(job_id)
            if latest is None:
                raise KeyError(job_id)
            latest["children"] = database.list_job_children(job_id)
            settle_submission_ticket()
            return project_job_read(latest)
        database.create_job_execution_plan(job_id, execution_plan)
    except V5CreativeAuthorityError as exc:
        failure = _creative_input_http_error(
            _v5_creative_authority_reason(exc)
        )
        failed = _fail_admitted_timeline_job(
            database, job_id, detail=failure.detail
        )
        failed["children"] = database.list_job_children(job_id)
        settle_submission_ticket()
        return project_job_read(failed)
    except (
        ProjectCompilerBundleError,
        V5V6ProjectionError,
        NativeTemplateError,
        DraftNotRunnable,
        ValidationError,
        ValueError,
    ) as exc:
        failure = _creative_input_http_error(_creative_input_reason(exc))
        failed = _fail_admitted_timeline_job(
            database, job_id, detail=failure.detail
        )
        failed["children"] = database.list_job_children(job_id)
        settle_submission_ticket()
        return project_job_read(failed)
    except HTTPException as exc:
        failed = _fail_admitted_timeline_job(
            database, job_id, detail=exc.detail
        )
        failed["children"] = database.list_job_children(job_id)
        settle_submission_ticket()
        return project_job_read(failed)
    except Exception:
        logger.exception("timeline compilation failed after job admission")
        failure = _execution_plan_invariant_http_error()
        failed = _fail_admitted_timeline_job(
            database, job_id, detail=failure.detail
        )
        failed["children"] = database.list_job_children(job_id)
        settle_submission_ticket()
        return project_job_read(failed)
    except BaseException:
        settle_submission_ticket()
        raise

    try:
        now = utc_now()
        child_ids: dict[str, str] = {}
        child_ids_by_segment: dict[str, str] = {}
        for index, unit in enumerate(execution_plan.segment_units):
            child_id = str(uuid.uuid4())
            child_ids[unit.id] = child_id
            child_ids_by_segment[unit.owner_segment_id] = child_id
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
                    "segment_ids": [unit.owner_segment_id],
                    "output_nodes": {
                        unit.owner_segment_id: unit.expected_output_spec.node_id
                    },
                    "status": "preparing",
                    "progress": 0.0,
                    "stage": "preflight",
                    "prompt_id": None,
                    "outputs": [],
                    "error": None,
                    "prompt_snapshot": unit.model_dump(mode="json")["prompt_base"],
                    "created_at": now,
                    "updated_at": now,
                    "started_at": None,
                    "completed_at": None,
                }
            )
    except Exception:
        logger.exception("timeline child materialization failed after job admission")
        failure = _execution_plan_invariant_http_error()
        failed = _fail_admitted_timeline_job(
            database,
            job_id,
            detail=failure.detail,
            stage="preflight_failed",
        )
        failed["children"] = database.list_job_children(job_id)
        settle_submission_ticket()
        return project_job_read(failed)
    except BaseException:
        settle_submission_ticket()
        raise
    submitted_children: list[tuple[str, str]] = []
    # Insert before awaiting POST /prompt: a thrown response can still follow
    # an accepted upstream side effect.
    possibly_submitted: dict[str, str] = {}
    submission_lock: anyio.Lock | None = None
    lock_acquired = False
    continuity_outputs: dict[str, dict[str, str]] = {}
    dependency_failed_segments: set[str] = set()
    try:
        await _preflight_execution_plan(client, execution_plan, database)
        gate = database.update_job_if_status(
            job_id, "preparing", stage="submitting"
        )
        if gate is None:
            latest = database.get_job(job_id)
            if latest is None:
                raise HTTPException(status_code=404, detail="job disappeared during submission")
            latest["children"] = database.list_job_children(job_id)
            return project_job_read(latest)
        if predecessor is not None:
            await asyncio.shield(predecessor)
        # Subscribe before POST /prompt and briefly await the initial socket
        # handshake so a newly configured endpoint does not lose its first
        # fast node event. The wait is bounded; queue/history remain the
        # lifecycle fallback if the optional websocket is unavailable.
        if isinstance(client, ComfyClient):
            await request.app.state.progress_manager.ensure_ready(
                request.app.state.comfy_url,
                settings.client_id,
                timeout_seconds=1.0,
            )
        submission_lock = request.app.state.submission_locks.setdefault(
            endpoint_key, anyio.Lock()
        )
        await submission_lock.acquire()
        lock_acquired = True
        empty_runtime_state = {
            "version": 2,
            "epoch": 0,
            "current": None,
            "tail_prompt_id": None,
            "tail_action": None,
            "tainted": False,
        }
        plan_uses_raylight = any(
            unit.backend == "raylight" for unit in execution_plan.segment_units
        )
        standard_tracks_raylight_runtime = True
        try:
            stored_runtime_state = database.get_raylight_runtime_state()
        except (NativeTemplateError, TypeError, ValueError):
            if plan_uses_raylight:
                raise
            # A damaged Ray-only ledger is not authority over a Standard plan.
            # Preserve it for explicit Ray recovery and submit Standard without
            # reading or rewriting that unrelated state.
            stored_runtime_state = None
            standard_tracks_raylight_runtime = False
        runtime_state = stored_runtime_state or empty_runtime_state
        if not plan_uses_raylight:
            # A Standard job still shuts down a settled Director-owned pool,
            # but it must not inherit an old ambiguous Ray prompt as global
            # submission authority.  That old job keeps converging through its
            # own recovery worker while this Standard plan uses no Ray ledger.
            if bool(runtime_state.get("tainted")):
                runtime_state = empty_runtime_state
                standard_tracks_raylight_runtime = False
        if plan_uses_raylight:
            # Only a selected Ray path consumes Director's bundled runtime
            # ledger. Standard work must not inherit unrelated Ray recovery as
            # a global host gate.
            try:
                locked_runtime_status = _raylight_runtime_status(
                    database,
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
                client, database, runtime_state
            )
        tail_prompt_id = runtime_state.get("tail_prompt_id")
        tail_action = runtime_state.get("tail_action")
        if (
            isinstance(tail_prompt_id, str)
            and tail_action == "ray_unit"
            and bool(runtime_state.get("tainted"))
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
                if (
                    tail_child["status"] not in _TERMINAL_STATUSES
                    and tail_child.get("stage") in _PROCESS_OWNERSHIP_STAGES
                ):
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "code": "raylight_transition_recovery_pending",
                            "message": (
                                "A previous Director RayLight submission is still "
                                "being reconciled."
                            ),
                        },
                    )
                terminal = await _await_raylight_generation(
                    client,
                    database,
                    str(tail_child["job_id"]),
                    str(tail_child["id"]),
                    tail_prompt_id,
                    request=request,
                    stop_on_parent_cancel=False,
                    dispatch_job_id=job_id,
                    terminal_events=request.app.state.prompt_terminal_events,
                )
                succeeded = terminal["status"] == "succeeded"
                database.settle_raylight_runtime_prompt(
                    tail_prompt_id,
                    succeeded=succeeded,
                    terminal_history_certified=(
                        _raylight_child_has_terminal_history_certificate(terminal)
                    ),
                )
            else:
                runtime_state = dict(runtime_state, tainted=True)
                database.put_raylight_runtime_state(runtime_state)
        elif isinstance(tail_prompt_id, str) and tail_action == "shutdown":
            # A durable queued barrier is itself the serialization point. Wait
            # for positive exact history before treating the previous pool as
            # gone, even when its control child belongs to a restarted task.
            tail_children = database.find_any_job_children_by_prompt_id(
                tail_prompt_id
            )
            tail_child = tail_children[0] if len(tail_children) == 1 else None
            if (
                tail_child is not None
                and tail_child["status"] not in _TERMINAL_STATUSES
                and tail_child.get("stage") in _PROCESS_OWNERSHIP_STAGES
            ):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "raylight_transition_recovery_pending",
                        "message": (
                            "A previous Director RayLight shutdown is still "
                            "being reconciled."
                        ),
                    },
                )
            if tail_child is None or tail_child["status"] == "succeeded":
                await _await_raylight_transition(client, tail_prompt_id)
                database.settle_raylight_runtime_prompt(
                    tail_prompt_id,
                    succeeded=True,
                    terminal_history_certified=True,
                )

        planned_units: list[LockedSubmissionUnit] = []
        transition_unit_ids: set[str] = set()
        submission_planner = LockedSubmissionPlanner(
            request.app.state.endpoint_identity
        )

        def planned_continuity_manifest(
            unit: LockedSubmissionUnit,
        ) -> dict[str, Any] | None:
            if not isinstance(unit, LockedSegmentUnit):
                return None
            dependency = unit.continuity_dependency
            if dependency is None:
                return None
            pointer = str(dependency.get("input_pointer") or "")
            pointer_parts = pointer.removeprefix("/").split("/")
            if len(pointer_parts) != 3 or pointer_parts[1:] != ["inputs", "file"]:
                raise NativeTemplateError(
                    f"workflow '{unit.id}' has an invalid continuity pointer"
                )
            return {
                "predecessor_segment_id": dependency.get(
                    "predecessor_segment_id"
                ),
                "overlap_frames": dependency.get("overlap_frames"),
                "load_video_node_id": pointer_parts[0],
                "source": dependency.get("source"),
                "historical_take_id": dependency.get("historical_take_id"),
                "resolved": dependency.get("resolved"),
            }

        def persist_planned_manifest() -> None:
            planned_manifest = dict(compiled_manifest)
            planned_manifest["submission_order"] = [
                unit.id for unit in planned_units
            ]
            current_ray_ledger = (
                database.get_raylight_runtime_state()
                if plan_uses_raylight or standard_tracks_raylight_runtime
                else None
            )
            planned_manifest["runtime_epoch"] = (
                int(current_ray_ledger["epoch"])
                if current_ray_ledger is not None
                else 0
            )
            planned_manifest["runtime_transitions"] = [
                unit.id for unit in planned_units
                if isinstance(unit, PreparedControlUnit)
            ]
            planned_manifest["units"] = [
                {
                    "id": unit.id,
                    "family": unit.family,
                    "backend": unit.backend,
                    "segment_ids": (
                        [unit.owner_segment_id]
                        if isinstance(unit, LockedSegmentUnit)
                        else []
                    ),
                    "output_nodes": (
                        {
                            unit.owner_segment_id: (
                                unit.expected_output_spec.node_id
                            )
                        }
                        if isinstance(unit, LockedSegmentUnit)
                        else {}
                    ),
                    "continuity": planned_continuity_manifest(unit),
                    "runtime_namespace": (
                        next(
                            (
                                unit.late_bound_values[evidence.input_pointer]
                                for evidence in unit.late_binding_evidence
                                if isinstance(
                                    evidence, RuntimeEpochLateBindingEvidence
                                )
                            ),
                            None,
                        )
                        if isinstance(unit, LockedSegmentUnit)
                        else None
                    ),
                }
                for unit in planned_units
            ]
            database.update_job(job_id, prompt_snapshot=planned_manifest)

        def continuity_evidence_for(
            prepared: PreparedSegmentUnit,
        ) -> tuple[ContinuityLateBindingEvidence, ...]:
            """Read output authority only; the planner materializes the graph."""

            dependency = prepared.continuity_dependency
            if dependency is not None:
                predecessor_id = str(
                    dependency.get("predecessor_segment_id") or ""
                )
                source = dependency.get("source")
                if source == "same_run":
                    raw_output = continuity_outputs.get(
                        predecessor_id
                    )
                else:
                    historical = historical_takes.get(
                        prepared.owner_segment_id
                    )
                    raw_output = (
                        historical.output if historical is not None else None
                    )
                if not isinstance(raw_output, Mapping):
                    raise NativeTemplateError(
                        f"workflow '{prepared.id}' has no continuity output evidence"
                    )
                output = OutputDescriptor.model_validate(
                    {
                        "filename": raw_output.get("filename"),
                        "subfolder": raw_output.get("subfolder") or "",
                        "type": raw_output.get("type") or "output",
                    }
                )
                return (
                    ContinuityLateBindingEvidence(
                        input_pointer=str(dependency.get("input_pointer") or ""),
                        predecessor_segment_id=predecessor_id,
                        dependency_source=source,
                        historical_take_id=dependency.get("historical_take_id"),
                        output=output,
                    ),
                )
            return ()

        def dynamically_planned_units():
            """Yield only the next safe unit; resume after its terminal gate."""

            for ordinal, prepared in enumerate(execution_plan.segment_units):
                segment_id = prepared.owner_segment_id
                if segment_id in dependency_failed_segments:
                    continue
                dependency = prepared.continuity_dependency
                if (
                    dependency is not None
                    and dependency.get("source") == "same_run"
                ):
                    predecessor_id = str(
                        dependency.get("predecessor_segment_id") or ""
                    )
                    predecessor_output = continuity_outputs.get(
                        predecessor_id
                    )
                    if predecessor_output is None:
                        newly_failed = _fail_continuity_descendants(
                            database,
                            job_id=job_id,
                            child_ids_by_segment=child_ids_by_segment,
                            dependents=continuity_dependents,
                            predecessor_segment_id=predecessor_id,
                            reason="predecessor did not produce a certified output",
                        )
                        dependency_failed_segments.update(newly_failed)
                        continue
                elif dependency is not None and (
                    dependency.get("source") != "historical_take"
                    or dependency.get("resolved") is not True
                    or not dependency.get("historical_take_id")
                ):
                    raise NativeTemplateError(
                        f"continuity segment '{segment_id}' has an unresolved "
                        "historical take"
                    )
                wave = submission_planner.build_wave(
                    execution_plan,
                    source_unit_ordinal=ordinal,
                    segment_child_id=child_ids[prepared.id],
                    continuity_evidence=continuity_evidence_for(prepared),
                    ray_ledger_before=(
                        database.get_raylight_runtime_state()
                        if (
                            prepared.backend == "raylight"
                            or standard_tracks_raylight_runtime
                        )
                        else None
                    ),
                    source_compiled_plan_digest=execution_plan_digest,
                )
                segment_plan = wave
                if len(wave.units) == 2:
                    transition_unit = wave.units[0]
                    assert isinstance(transition_unit, PreparedControlUnit)
                    transition_unit_ids.add(transition_unit.id)
                    planned_units.append(transition_unit)
                    persist_planned_manifest()
                    yield transition_unit, wave
                    # The generator resumes only after the outer loop has
                    # positively certified RayKill history.
                    segment_plan = submission_planner.segment_continuation(
                        wave,
                        ray_ledger_before=(
                            database.get_raylight_runtime_state()
                        ),
                    )
                segment_unit = segment_plan.units[0]
                assert isinstance(segment_unit, LockedSegmentUnit)
                planned_units.append(segment_unit)
                persist_planned_manifest()
                yield segment_unit, segment_plan

        workflows_to_submit = dynamically_planned_units()
        # ComfyUI's normal queue serializes these one-segment prompts. Stable
        # loader ids/inputs permit endpoint-local cache reuse without putting
        # 128 independent sampling/decode branches in one failure domain.
        for unit, locked_plan in workflows_to_submit:
            segment_id = (
                unit.owner_segment_id
                if isinstance(unit, LockedSegmentUnit)
                else None
            )
            if (
                segment_id is not None
                and segment_id in dependency_failed_segments
            ):
                continue
            if unit.id in transition_unit_ids:
                await _preflight_raylight_transition(
                    request, client, unit, database
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
                return project_job_read(
                    await _cancel_timeline_job(request, current)
                )
            child_id = unit.child_id
            locked_unit = unit
            before_claim = getattr(
                request.app.state, "before_submission_claim", None
            )
            if before_claim is not None:
                await before_claim(job_id, child_id)
            exact_snapshot = submission_planner.exact_snapshot(
                locked_plan,
                locked_unit,
            )
            try:
                claimed_child, initial_ownership = (
                    database.persist_job_child_submission_intent(
                        job_id,
                        locked_plan=locked_plan,
                        exact_snapshot=exact_snapshot,
                    )
                )
            except (ExecutionEvidenceConflict, RayRuntimeIntentConflict):
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
                    return project_job_read(
                        await _cancel_timeline_job(request, latest_parent)
                    )
                raise HTTPException(
                    status_code=409,
                    detail="job child changed state before submission",
                )

            possibly_submitted[child_id] = initial_ownership.effective_prompt_id
            after_submission_intent = getattr(
                request.app.state, "after_submission_intent", None
            )
            if after_submission_intent is not None:
                # Crash-window test seam: intent/exact snapshot/ownership are
                # durable, while no network side effect has started yet.
                await after_submission_intent(job_id, child_id, exact_snapshot)

            def persist_submit_receipt(
                requested_prompt_id: str | None,
                actual_prompt_id: str,
            ) -> None:
                if requested_prompt_id != locked_unit.requested_prompt_id:
                    raise ExecutionEvidenceConflict(
                        "ComfyUI receipt does not match the locked requested id"
                    )
                receipt = database.record_prompt_submission_receipt(
                    child_id,
                    expected_revision=initial_ownership.ownership_revision,
                    actual_prompt_id=actual_prompt_id,
                    state=(
                        "owned_requested_id"
                        if actual_prompt_id == requested_prompt_id
                        else "owned_actual_id"
                    ),
                    updated_at=datetime.now(timezone.utc),
                )
                if receipt is None:
                    raise ExecutionEvidenceConflict(
                        "prompt ownership changed before receipt persistence"
                    )
                possibly_submitted[child_id] = receipt.effective_prompt_id

            exact_prompt_document = exact_snapshot.model_dump(mode="json")[
                "exact_prompt"
            ]
            if not isinstance(exact_prompt_document, dict):
                raise ExecutionEvidenceConflict(
                    "exact prompt snapshot did not materialize a JSON object"
                )
            submitted = await client.submit(
                exact_prompt_document,
                settings.client_id,
                prompt_id=locked_unit.requested_prompt_id,
                on_receipt=persist_submit_receipt,
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
                            if len(execution_plan.segment_units) == 1
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
                latest_child = database.get_job_child(child_id)
                if latest_child is None:
                    raise HTTPException(
                        status_code=404,
                        detail="job child disappeared after submission",
                    )
                cancel_ownership = _claim_typed_prompt_cancel(
                    database, latest_child
                )
                if cancel_ownership is not None and cancel_ownership.state in {
                    "cleanup_confirmed",
                    "terminal_confirmed",
                }:
                    submission_lock.release()
                    lock_acquired = False
                    return project_job_read(
                        await _sync_timeline_job(request, latest_parent)
                    )
                try:
                    dispatched = await client.cancel(prompt_id)
                except (ComfyError, httpx.HTTPError) as exc:
                    _mark_typed_prompt_unconfirmed(
                        database, child_id, cancel_ownership
                    )
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
                        return project_job_read(
                            await _sync_timeline_job(request, latest_parent)
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
                    _mark_typed_prompt_unconfirmed(
                        database, child_id, cancel_ownership
                    )
                    reconciled = await _sync_timeline_job(request, latest_parent)
                    if reconciled["status"] in {"succeeded", "failed"}:
                        submission_lock.release()
                        lock_acquired = False
                        return project_job_read(reconciled)
                else:
                    if cancel_ownership is not None:
                        _confirm_typed_exact_cancel(
                            database,
                            child_id,
                            cancel_ownership,
                            stage="cancelled",
                        )
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
                return project_job_read(
                    await _sync_timeline_job(request, cancelled_parent)
                )
            if isinstance(unit, PreparedControlUnit):
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
                        prompt_id, succeeded=False
                    )
                    latest_parent = database.get_job(job_id)
                    if latest_parent is None:
                        raise HTTPException(
                            status_code=404,
                            detail="job disappeared during RayLight barrier",
                        )
                    submission_lock.release()
                    lock_acquired = False
                    return project_job_read(
                        await _cancel_timeline_job(request, latest_parent)
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
                    prompt_id,
                    succeeded=True,
                    terminal_history_certified=True,
                )
            elif isinstance(unit, LockedSegmentUnit) and unit.backend == "raylight":
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
                    request=request,
                    terminal_events=request.app.state.prompt_terminal_events,
                )
                succeeded = terminal["status"] == "succeeded"
                database.settle_raylight_runtime_prompt(
                    prompt_id,
                    succeeded=succeeded,
                    terminal_history_certified=(
                        _raylight_child_has_terminal_history_certificate(terminal)
                    ),
                )
                latest_parent = database.get_job(job_id)
                if latest_parent is None:
                    raise HTTPException(
                        status_code=404, detail="job disappeared during RayLight gate"
                    )
                if latest_parent["status"] in {"cancelling", "cancelled"}:
                    submission_lock.release()
                    lock_acquired = False
                    return project_job_read(
                        await _cancel_timeline_job(request, latest_parent)
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
                        request=request,
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
                    return project_job_read(
                        await _cancel_timeline_job(request, latest_parent)
                    )
                if terminal_child["status"] == "succeeded":
                    try:
                        continuity_outputs[segment_id] = (
                            _continuity_output_descriptor(
                                database,
                                terminal_child,
                                segment_id,
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
        latest_after_http_error = database.get_job(job_id)
        cleanup_already_owned = (
            latest_after_http_error is not None
            and latest_after_http_error["status"] == "cancelling"
            and latest_after_http_error.get("stage") == "cancel_failed"
        )
        if possibly_submitted and not cleanup_already_owned:
            # HTTPException is not limited to the initial read-only preflight:
            # a later unit can fail its dependency/claim gate after an earlier
            # unit already owns a real ComfyUI prompt.  That state must use the
            # same exact-id cleanup protocol as a transport failure.  Marking
            # it locally failed would orphan the upstream prompt and erase the
            # only recovery owner while releasing the endpoint lock.
            try:
                settle_submission_ticket()
                await _cleanup_failed_timeline_submission(
                    request,
                    job_id=job_id,
                    client=client,
                    error=exc,
                    possibly_submitted=possibly_submitted,
                )
            except BaseException:
                _mark_timeline_submission_interrupted(database, job_id)
                raise
            latest = database.get_job(job_id)
            if latest is None:
                raise KeyError(job_id)
            latest["children"] = database.list_job_children(job_id)
            return project_job_read(latest)
        if cleanup_already_owned:
            # The late-submit cancellation branch already attempted the exact
            # prompt id and persisted an unconfirmed recovery owner before it
            # raised this HTTPException.  A second cleanup pass would overwrite
            # the more precise ``cancel_failed`` state.
            latest = database.get_job(job_id)
            if latest is None:
                raise KeyError(job_id)
            latest["children"] = database.list_job_children(job_id)
            return project_job_read(latest)
        failed_parent = _fail_admitted_timeline_job(
            database,
            job_id,
            detail=exc.detail,
            stage="preflight_failed",
        )
        failed_parent["children"] = database.list_job_children(job_id)
        return project_job_read(failed_parent)
    except (
        ExecutionEvidenceConflict,
        NativeTemplateError,
        SubmissionPlanningError,
        ValidationError,
        ValueError,
    ) as exc:
        if lock_acquired and submission_lock is not None:
            submission_lock.release()
            lock_acquired = False
        if possibly_submitted:
            try:
                settle_submission_ticket()
                await _cleanup_failed_timeline_submission(
                    request,
                    job_id=job_id,
                    client=client,
                    error=exc,
                    possibly_submitted=possibly_submitted,
                )
            except BaseException:
                _mark_timeline_submission_interrupted(database, job_id)
                raise
            latest = database.get_job(job_id)
            if latest is None:
                raise KeyError(job_id)
            latest["children"] = database.list_job_children(job_id)
            return project_job_read(latest)
        failed_parent = _fail_admitted_timeline_job(
            database,
            job_id,
            detail={
                "code": "submission_plan_invalid",
                "message": str(exc),
            },
            stage="preflight_failed",
        )
        failed_parent["children"] = database.list_job_children(job_id)
        return project_job_read(failed_parent)
    except ComfyPromptRejected as exc:
        if lock_acquired and submission_lock is not None:
            submission_lock.release()
            lock_acquired = False
        rejection_detail = {
            "code": "comfy_prompt_rejected",
            "message": "ComfyUI rejected the generated prompt before queue admission.",
            "status_code": exc.status_code,
            "detail": exc.detail,
        }
        rejection_error = json.dumps(rejection_detail, ensure_ascii=False)
        try:
            rejected_child = database.fail_rejected_job_child_submission(
                child_id,
                expected_revision=initial_ownership.ownership_revision,
                error=rejection_error,
                updated_at=datetime.now(timezone.utc),
            )
            if rejected_child is None:
                # A concurrent lifecycle owner changed the intent. Preserve the
                # existing conservative cleanup path instead of overwriting it.
                settle_submission_ticket()
                await _cleanup_failed_timeline_submission(
                    request,
                    job_id=job_id,
                    client=client,
                    error=exc,
                    possibly_submitted=possibly_submitted,
                )
            else:
                possibly_submitted.pop(child_id, None)
                if possibly_submitted:
                    # Earlier units may already own real prompts; only the current
                    # rejected intent is side-effect free.
                    settle_submission_ticket()
                    await _cleanup_failed_timeline_submission(
                        request,
                        job_id=job_id,
                        client=client,
                        error=exc,
                        possibly_submitted=possibly_submitted,
                    )
                else:
                    _fail_admitted_timeline_job(
                        database,
                        job_id,
                        detail=rejection_detail,
                        stage="submission_failed",
                    )
        except BaseException:
            _mark_timeline_submission_interrupted(database, job_id)
            raise
        latest = database.get_job(job_id)
        if latest is None:
            raise KeyError(job_id)
        latest["children"] = database.list_job_children(job_id)
        return project_job_read(latest)
    except (ComfyError, httpx.HTTPError, KeyError) as exc:
        # An incompatible ComfyUI may ignore our caller-assigned prompt id and
        # queue another one. ComfyClient reports both ids after attempting an
        # inline exact cancellation. Bind only a detail value authenticated by
        # the current durable requested id, so outer cleanup and restart
        # recovery target the actual upstream side effect if inline cleanup was
        # false or raised.
        try:
            inline_cancelled = _bind_actual_prompt_id_from_submit_error(
                database, exc, possibly_submitted
            )
            if lock_acquired and submission_lock is not None:
                submission_lock.release()
                lock_acquired = False
            settle_submission_ticket()
            await _cleanup_failed_timeline_submission(
                request,
                job_id=job_id,
                client=client,
                error=exc,
                possibly_submitted=possibly_submitted,
                inline_cancelled=inline_cancelled,
            )
        except BaseException:
            if lock_acquired and submission_lock is not None:
                submission_lock.release()
                lock_acquired = False
            _mark_timeline_submission_interrupted(database, job_id)
            raise
        latest = database.get_job(job_id)
        if latest is None:
            raise KeyError(job_id)
        latest["children"] = database.list_job_children(job_id)
        return project_job_read(latest)
    except asyncio.CancelledError:
        if lock_acquired and submission_lock is not None:
            submission_lock.release()
            lock_acquired = False
        current = database.get_job(job_id)
        if current is not None and current["status"] in {"cancelling", "cancelled"}:
            current["children"] = database.list_job_children(job_id)
            return project_job_read(current)
        raise
    except BaseException:
        logger.exception("timeline submission interrupted unexpectedly")
        # A graceful server shutdown can cancel the shielded submission task.
        # Persist a finite recovery state before releasing the endpoint lock;
        # caller-assigned prompt ids make every possibly accepted side effect
        # targetable after restart.
        if lock_acquired and submission_lock is not None:
            submission_lock.release()
            lock_acquired = False
        _mark_timeline_submission_interrupted(database, job_id)
        raise
    finally:
        settle_submission_ticket()
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
    return project_job_read(job)


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

    if body.config.features.template_bundle_version == 4:
        try:
            body = body.model_copy(
                update={
                    "config": migrate_timeline_feature_authority_to_v5(
                        body.config
                    )
                },
                deep=True,
            )
        except (ValidationError, ValueError) as exc:
            raise _creative_input_http_error(_creative_input_reason(exc)) from exc

    accepted = asyncio.Event()
    accepted_release = asyncio.Event()
    captured_settings_future: asyncio.Future[RuntimeSettingsV3] = (
        asyncio.get_running_loop().create_future()
    )
    job_id = str(uuid.uuid4())
    task = asyncio.create_task(
        _create_timeline_job_impl(
            request,
            body,
            parent_mode=parent_mode,
            job_id=job_id,
            accepted=accepted,
            accepted_release=accepted_release,
            captured_settings_future=captured_settings_future,
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
            error = done.exception()
            if error is not None and accepted.is_set():
                logger.error(
                    "detached timeline submission failed for job %s",
                    job_id,
                    exc_info=error,
                )

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
        return _job_read_for_request(
            request,
            job,
            current_timeline=body.config,
            current_settings=await asyncio.shield(captured_settings_future),
        )
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
    client = _comfy(request)
    dispatch_errors: list[str] = []
    has_unconfirmed = False
    for child in database.list_job_children(job["id"]):
        ownership = _typed_prompt_ownership_for_child(database, child)
        if child["status"] in _TERMINAL_STATUSES:
            if ownership is not None and ownership.state not in {
                "cleanup_confirmed",
                "terminal_confirmed",
            }:
                raise ExecutionEvidenceConflict(
                    f"terminal child {child['id']} retains prompt ownership"
                )
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
        if ownership is not None and ownership.state in {
            "cleanup_confirmed",
            "terminal_confirmed",
        }:
            continue
        if ownership is not None:
            exact_evidence = database.get_job_child_execution_evidence(
                str(child["id"])
            )
            if exact_evidence is None:  # Guarded by the typed helper above.
                raise ExecutionEvidenceConflict(
                    f"child {child['id']} lost its exact prompt evidence"
                )
            frozen_endpoint = exact_evidence[
                "exact_prompt_snapshot"
            ].endpoint_identity
            current_endpoint = request.app.state.endpoint_identity
            if frozen_endpoint.endpoint_key != current_endpoint.endpoint_key:
                raise ExecutionEvidenceConflict(
                    "recovery endpoint does not match the exact prompt endpoint"
                )
            if (
                frozen_endpoint.runtime_instance_id
                != current_endpoint.runtime_instance_id
            ):
                ownership = _mark_typed_prompt_unconfirmed(
                    database, str(child["id"]), ownership
                )
                await _cas_active_child_update(
                    database,
                    child,
                    status="cancelling",
                    stage="restart_certificate_required",
                    error=(
                        "ComfyUI runtime instance changed; explicit restart "
                        "confirmation is required"
                    ),
                    completed_at=None,
                )
                has_unconfirmed = True
                continue
        ownership = _claim_typed_prompt_cancel(database, child)
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
        prompt_id = (
            ownership.effective_prompt_id
            if ownership is not None
            else str(child["prompt_id"])
        )
        try:
            dispatched = await client.cancel(prompt_id)
        except asyncio.CancelledError:
            raise
        except (ComfyError, httpx.HTTPError) as exc:
            dispatch_errors.append(f"{child['id']}: {exc}")
            ownership = _mark_typed_prompt_unconfirmed(
                database, str(child["id"]), ownership
            )
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
            if ownership is not None:
                released = _confirm_typed_exact_cancel(
                    database,
                    str(child["id"]),
                    ownership,
                    stage="cancelled_after_restart",
                )
                if released is None:
                    has_unconfirmed = True
                continue
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
            ownership = _mark_typed_prompt_unconfirmed(
                database, str(child["id"]), ownership
            )
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
                except (
                    ComfyError,
                    ExecutionEvidenceConflict,
                    httpx.HTTPError,
                    HTTPException,
                    KeyError,
                    ValidationError,
                ):
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
    database_path: str | Path,
    comfy_url: str,
    comfy_factory: ComfyFactory | None = None,
    host_capability_provider: HostCapabilityProvider | None = None,
    host_output_probe: HostOutputProbeProvider | None = None,
    comfy_tls_certfile: str | Path | None = None,
    public_api_prefix: str = "",
    raylight_requirements_path: str | Path | None = None,
    endpoint_runtime_instance_id: str | None = None,
) -> FastAPI:
    set_public_api_prefix(public_api_prefix)
    comfy_url = comfy_url.rstrip("/")
    if not comfy_url:
        raise ValueError("create_app requires the host ComfyUI callback URL")
    comfy_tls_context: ssl.SSLContext | None = None
    if comfy_tls_certfile is not None:
        if urlsplit(comfy_url).scheme.lower() != "https":
            raise ValueError("comfy_tls_certfile requires an https ComfyUI callback URL")
        comfy_tls_context = ssl.create_default_context()
        comfy_tls_context.load_verify_locations(cafile=str(comfy_tls_certfile))
        # ComfyUI commonly receives a leaf/full-chain PEM instead of a
        # separately installed root CA. Trust that explicit local chain while
        # retaining normal hostname/SAN validation.
        comfy_tls_context.verify_flags |= getattr(
            ssl,
            "VERIFY_X509_PARTIAL_CHAIN",
            0,
        )
    storage = StorageController.resolve(database_path)
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
    submission_ticket_lock = asyncio.Lock()
    submission_tails: dict[str, asyncio.Future[None]] = {}

    async def persist_native_progress(
        _comfy_origin: str, event: ComfyProgressEvent | ComfyExecutionEvent
    ) -> None:
        for child in database.find_job_children_by_prompt_id(event.prompt_id):
            child = _child_with_execution_evidence(database, child)
            parent_status = database.get_job_status(child["job_id"])
            if parent_status is None or parent_status in _TERMINAL_STATUSES:
                continue
            phase_index = preview_phase_index_for_event(child, event)
            if (
                phase_index is not None
                and child.get("status") in {"preparing", "queued", "running"}
            ):
                # Node/progress events are phase-start evidence even when their
                # numeric snapshot loses a later monotonic database race.
                live_preview_cache.advance_phase(
                    job_id=str(child["job_id"]),
                    child_id=str(child["id"]),
                    prompt_id=event.prompt_id,
                    phase_index=phase_index,
                )
            snapshot = (
                child_progress_snapshot(child, event)
                if isinstance(event, ComfyProgressEvent)
                else child_execution_snapshot(child, event)
            )
            if snapshot is None and isinstance(event, ComfyExecutionEvent):
                snapshot = child_execution_start_snapshot(child, event)
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
        _comfy_origin: str, event: ComfyPreviewEvent
    ) -> None:
        for child in database.find_job_children_by_prompt_id(event.prompt_id):
            child = _child_with_execution_evidence(database, child)
            parent_job_id = str(child["job_id"])
            parent_status = database.get_job_status(parent_job_id)
            if parent_status is None or parent_status in _TERMINAL_STATUSES:
                continue
            preview_source = preview_source_for_node(child, event.node_id)
            if preview_source is None:
                continue
            segment_id = preview_source.segment_id
            # Re-read both rows at the final cache boundary. DELETE cascades
            # child rows; the cache tombstone closes the inverse ordering where
            # deletion wins immediately before this put.
            latest_parent_status = database.get_job_status(parent_job_id)
            latest_child = database.get_job_child(child["id"])
            if latest_child is not None:
                latest_child = _child_with_execution_evidence(
                    database, latest_child
                )
            if (
                latest_parent_status is None
                or latest_parent_status in _TERMINAL_STATUSES
                or latest_child is None
                or latest_child["status"] not in {"queued", "running"}
                or latest_child.get("prompt_id") != event.prompt_id
            ):
                continue
            latest_preview_source = preview_source_for_node(
                latest_child, event.node_id
            )
            if (
                latest_preview_source is None
                or latest_preview_source.segment_id != segment_id
            ):
                continue
            live_preview_cache.put(
                job_id=parent_job_id,
                child_id=child["id"],
                segment_id=segment_id,
                event=event,
                source=latest_preview_source,
                minimum_phase_index=durable_preview_phase_watermark(
                    latest_child
                ),
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
        persist_native_progress,
        persist_native_preview,
        wake_native_reconcile,
        ssl_context=comfy_tls_context,
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
            # Product-owned support configuration is parsed exactly once per
            # DirectorDeck process. A broken Director-owned file is diagnosed
            # locally: persistence and the API still start, while only a
            # request that needs the unavailable LoRA policy fails explicitly.
            try:
                initialize_directordeck_config()
            except Exception as exc:
                app.state.product_config_initialization_failed = True
                logger.error(
                    "DirectorDeck product configuration failed to initialize: %s",
                    type(exc).__name__,
                )
            else:
                app.state.product_config_initialization_failed = False
            database.initialize()
            database.recover_interrupted_assemblies()
            # Startup is intentionally local-only.  The previous process
            # cannot still own these rows, so replace its transient ownership
            # markers in one SQLite transaction.  Every ComfyUI
            # queue/history/cancel request is deferred to managed tasks
            # created below, after the app can yield.
            database.prepare_interrupted_submissions_for_recovery()
            settings = database.get_settings()
            progress_manager.ensure(app.state.comfy_url, settings.client_id)
            for snapshot in database.list_active_job_settings():
                try:
                    if (
                        isinstance(snapshot, Mapping)
                        and snapshot.get("snapshot_schema_version") == 1
                    ):
                        bounded_snapshot = JobRuntimeSnapshotV1.model_validate(
                            snapshot
                        )
                        # Older bounded V1 rows did not capture control-plane
                        # routing. They remain readable, but startup must not
                        # guess their original client from mutable settings.
                        if bounded_snapshot.control_evidence is None:
                            continue
                        active_client_id = (
                            bounded_snapshot.control_evidence.progress_client_id
                        )
                    else:
                        active_settings = (
                            RuntimeSettingsV3.model_validate(snapshot)
                            if isinstance(snapshot, Mapping)
                            and snapshot.get("schema_version") == 3
                            else RuntimeSettingsV2.model_validate(snapshot)
                            if isinstance(snapshot, Mapping)
                            and snapshot.get("schema_version") == 2
                            else RuntimeSettings.model_validate(snapshot)
                        )
                        active_client_id = active_settings.client_id
                except ValidationError:
                    continue
                progress_manager.ensure(
                    app.state.comfy_url, active_client_id
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
            # Explicitly stop installer subprocesses before uvicorn closes its
            # event loop. Pending pip/download children must not survive a
            # ComfyUI restart, especially on Windows Desktop.
            await asyncio.gather(
                app.state.raylight_install_manager.close(),
                app.state.ffmpeg_install_manager.close(),
                return_exceptions=True,
            )
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
    app.state.product_config_initialization_failed = False
    app.state.instance_lock = instance_lock
    app.state.storage = storage
    app.state.comfy_url = comfy_url
    app.state.endpoint_identity = EndpointIdentity(
        schema_version=1,
        endpoint_key=_EMBEDDED_ENDPOINT_KEY,
        runtime_instance_id=(
            endpoint_runtime_instance_id or str(uuid.uuid4())
        ),
    )
    app.state.comfy_factory = comfy_factory or (
        partial(ComfyClient, verify=comfy_tls_context)
        if comfy_tls_context is not None
        else default_comfy_factory
    )
    app.state.ck_host_observation = None
    app.state.ck_host_observation_generation = 0
    app.state.ck_host_observation_lock = asyncio.Lock()
    app.state.host_capability_provider = host_capability_provider
    app.state.host_output_probe = host_output_probe
    app.state.comfy_tls_context = comfy_tls_context
    app.state.progress_manager = progress_manager
    app.state.live_preview_cache = live_preview_cache
    app.state.raylight_install_manager = RayLightInstallManager()
    host_capability_invalidator = getattr(
        host_capability_provider,
        "invalidate",
        None,
    )
    app.state.ffmpeg_install_manager = FFmpegInstallManager(
        on_ready=(
            host_capability_invalidator
            if callable(host_capability_invalidator)
            else None
        )
    )
    app.state.project_import_coordinator = ProjectImportCoordinator()
    # The plugin always passes its bundled requirements file; an absent path
    # only means RayLight installation is unavailable for this build.
    app.state.raylight_requirements_path = (
        Path(raylight_requirements_path) if raylight_requirements_path is not None else None
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

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/storage", response_model=StorageStatusRead)
    async def get_storage() -> StorageStatusRead:
        # The embedded plugin fixes the database below the host ComfyUI user
        # directory. There is no runtime selection or migration; this read
        # only reports the fixed location.
        return StorageStatusRead(
            active_database_path=str(storage.active_database_path)
        )

    @app.get("/api/settings", response_model=RuntimeSettingsV3)
    async def get_settings(request: Request) -> RuntimeSettingsV3:
        return _db(request).get_settings()

    @app.get(
        "/api/settings/authority",
        response_model=RuntimeSettingsAuthorityV3Read,
    )
    async def get_settings_authority(
        request: Request,
    ) -> RuntimeSettingsAuthorityV3Read:
        settings, authority = _db(request).get_settings_authority()
        return RuntimeSettingsAuthorityV3Read(
            settings=settings,
            authority_token=authority,
        )

    @app.get(
        "/api/settings/migration-notices",
        response_model=RuntimeSettingsMigrationNoticeListRead,
    )
    async def get_runtime_settings_migration_notices(
        request: Request,
    ) -> RuntimeSettingsMigrationNoticeListRead:
        return RuntimeSettingsMigrationNoticeListRead(
            notices=_db(request).list_runtime_settings_migration_notices()
        )

    @app.put("/api/settings")
    async def put_legacy_settings(
        request: Request, body: dict[str, Any]
    ) -> Any:
        # No non-CAS v3 write exists.  Keeping this route as a structured
        # tombstone also lets stale RuntimeSettingsV1 tabs recover their WAL
        # instead of receiving an ambiguous validation error.
        del request, body
        raise _runtime_settings_schema_migrated()

    @app.put(
        "/api/settings/authority",
        response_model=RuntimeSettingsAuthorityV3Read,
    )
    async def put_settings_authority(
        request: Request,
        body: (
            RuntimeSettingsAuthorityV1WriteRequest
            | RuntimeSettingsAuthorityV2WriteRequest
            | RuntimeSettingsAuthorityV3WriteRequest
        ),
    ) -> RuntimeSettingsAuthorityV3Read:
        if body.schema_version != 3:
            raise _runtime_settings_schema_migrated()
        database = _db(request)
        current, current_authority = database.get_settings_authority()
        if current_authority != body.expected_authority_token:
            raise _settings_authority_conflict()
        validated_document = _validated_lora_loader_mapping_update(
            current,
            body.document,
        )
        try:
            settings, authority = database.put_settings_v3_authority(
                validated_document,
                expected_authority_token=body.expected_authority_token,
                schema_version=body.schema_version,
            )
        except RuntimeSettingsSchemaMigrated:
            raise _runtime_settings_schema_migrated() from None
        except SettingsAuthorityConflict:
            raise _settings_authority_conflict() from None
        return RuntimeSettingsAuthorityV3Read(
            settings=settings,
            authority_token=authority,
        )

    @app.get("/api/features/catalog")
    async def get_feature_catalog(request: Request) -> Response:
        snapshot = await _host_capability_snapshot(request)
        catalog = build_feature_catalog(
            snapshot,
            # This endpoint is a frozen Bundle-5 compatibility/debug wire.
            # Bundle 6 uses its native compiler and dedicated product controls.
            template_bundle=V5_TEMPLATE_BUNDLE,
        )
        etag = quote_feature_catalog_etag(
            feature_catalog_etag(
                template_bundle_version=catalog.template_bundle_version,
                host_capability_revision=catalog.host_capability_revision,
            )
        )
        if_none_match = request.headers.get("if-none-match")
        candidates = (
            {item.strip() for item in if_none_match.split(",")}
            if if_none_match
            else set()
        )
        if "*" in candidates or etag in candidates:
            return Response(status_code=304, headers={"ETag": etag})
        return Response(
            content=catalog.model_dump_json(),
            media_type="application/json",
            headers={"ETag": etag},
        )

    @app.get("/api/config")
    async def get_product_config(request: Request) -> Response:
        """Return the immutable product configuration loaded at startup."""

        try:
            if request.app.state.product_config_initialization_failed:
                raise RuntimeError("product configuration unavailable")
            config = get_directordeck_config()
        except RuntimeError:
            payload: dict[str, Any] = {
                "schema_version": 1,
                "lora": {
                    "loaders": [],
                    "fallback_policy": None,
                    "loader_policies": [],
                },
                "diagnostics": [
                    {
                        "code": "lora_product_config_unavailable",
                        "message": (
                            "DirectorDeck's LoRA loader configuration is unavailable."
                        ),
                    }
                ],
            }
        else:
            payload = config.model_dump(mode="json")
            payload["diagnostics"] = [
                diagnostic.model_dump(mode="json")
                for diagnostic in get_directordeck_config_diagnostics()
            ]
        return Response(
            content=json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ),
            media_type="application/json",
        )

    @app.post(
        "/api/features/preflight",
        response_model=FeaturePreflightReport,
    )
    async def preflight_features(
        request: Request,
        body: FeaturePreflightRequest,
    ) -> FeaturePreflightReport:
        # This route is deliberately advisory and read-only. Production
        # compile/submission do not consume its capability evidence as an
        # authorization token.
        snapshot = await _host_capability_snapshot(request)
        readiness = _host_operational_readiness(request, snapshot)
        database = _db(request)
        project_id = body.project_id or database.LEGACY_DEFAULT_PROJECT_ID
        if not database.project_exists(project_id):
            return _feature_preflight_report_from_reason(
                snapshot=snapshot,
                readiness=readiness,
                reason=_project_not_found_reason(),
            )
        try:
            draft_v5 = migrate_timeline_feature_authority_to_v5(
                body.config or database.get_project_timeline(project_id)
            )
        except KeyError:
            # The project may have been deleted between the scope check and read.
            return _feature_preflight_report_from_reason(
                snapshot=snapshot,
                readiness=readiness,
                reason=_project_not_found_reason(),
            )
        except (ValidationError, ValueError) as exc:
            return _feature_preflight_report_from_reason(
                snapshot=snapshot,
                readiness=readiness,
                reason=_creative_input_reason(exc),
            )
        captured_settings = database.get_settings()
        try:
            host_projection = project_v5_contextual_host_authority(
                draft_v5,
                captured_settings,
                body.segment_ids,
            )
            draft = host_projection.draft
            settings = host_projection.settings
            database.validate_timeline_assets(
                draft,
                segment_ids=body.segment_ids,
            )
            validate_unified_runnable(draft, segment_ids=body.segment_ids)
            historical_takes = _resolve_historical_continuity_takes(
                database,
                draft,
                segment_ids=body.segment_ids,
                project_id=project_id,
            )
        except V5CreativeAuthorityError as exc:
            return _feature_preflight_report_from_reason(
                snapshot=snapshot,
                readiness=readiness,
                reason=_v5_creative_authority_reason(exc),
            )
        except (DraftNotRunnable, NativeTemplateError, ValidationError, ValueError) as exc:
            reason = _creative_input_reason(exc)
            return _feature_preflight_report_from_reason(
                snapshot=snapshot,
                readiness=readiness,
                reason=reason,
            )
        try:
            client = _comfy(request)
            contextual_errors = await _contextual_host_errors(
                client,
                draft=draft,
                settings=settings,
                segment_ids=body.segment_ids,
                snapshot=snapshot,
            )
        except (ComfyError, httpx.HTTPError):
            return _feature_preflight_report_from_reason(
                snapshot=snapshot,
                readiness=readiness,
                reason=_host_context_observation_reason(),
            )
        try:
            projection = project_v5_compile_authority(
                draft_v5,
                captured_settings,
                body.segment_ids,
            )
            draft = projection.draft
            settings = projection.settings
        except V5CreativeAuthorityError as exc:
            return FeaturePreflightReport(
                template_bundle_version=CURRENT_TEMPLATE_BUNDLE.version,
                host_capability_revision=snapshot.host_capability_revision(),
                operational_readiness=readiness,
                valid=False,
                errors=(
                    _v5_creative_authority_reason(exc),
                    *contextual_errors,
                ),
                effective_by_segment={},
            )
        try:
            feature_report = preflight_projected_v5_timeline(
                draft=draft,
                settings=settings,
                effective_features=projection.effective_features,
                snapshot=snapshot,
                readiness=readiness,
                segment_ids=body.segment_ids,
                historical_takes=historical_takes,
                resolved_lora_adapters=projection.lora_adapter_map(),
            )
        except (
            NativeTemplateError,
            DraftNotRunnable,
            ValidationError,
            ValueError,
        ) as exc:
            reason = _creative_input_reason(exc)
            return FeaturePreflightReport(
                template_bundle_version=CURRENT_TEMPLATE_BUNDLE.version,
                host_capability_revision=snapshot.host_capability_revision(),
                operational_readiness=readiness,
                valid=False,
                errors=(reason, *contextual_errors),
                effective_by_segment={},
            )
        errors = (*feature_report.errors, *contextual_errors)
        return FeaturePreflightReport(
            template_bundle_version=feature_report.template_bundle_version,
            host_capability_revision=feature_report.host_capability_revision,
            operational_readiness=feature_report.operational_readiness,
            valid=not errors,
            errors=errors,
            effective_by_segment=dict(feature_report.effective_by_segment),
        )

    @app.get("/api/capabilities")
    async def get_capabilities(request: Request) -> dict[str, Any]:
        _settings, authority = _runtime_authority_snapshot(request)
        try:
            report = await _comfy(request).capabilities()
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

    @app.get(
        "/api/capabilities/comfy-kitchen-attention",
        response_model=ComfyKitchenAttentionCapabilityV1,
    )
    async def get_comfy_kitchen_attention_capability(
        request: Request,
        family: Annotated[
            list[Literal["fl2va", "ref2va"]] | None,
            Query(),
        ] = None,
    ) -> ComfyKitchenAttentionCapabilityV1:
        """Return a cached UI hint; it never authorizes compilation/submission."""

        reachable_families = tuple(dict.fromkeys(family or ("fl2va", "ref2va")))
        return await _comfy_kitchen_attention_capability(
            request,
            reachable_families=reachable_families,
        )

    @app.post("/api/capabilities")
    async def test_capabilities(request: Request) -> dict[str, Any]:
        """Re-probe the embedded host ComfyUI instance (the only endpoint)."""

        started = time.monotonic()
        try:
            report = await _comfy(request).capabilities()
            missing = report.get("missing_nodes") or []
            return {
                # Reaching the capability endpoint is a successful connection
                # test. Missing execution nodes affect readiness, not network
                # connectivity, and are reported separately in the message.
                "ok": True,
                "latency_ms": report.get("latency_ms")
                or round((time.monotonic() - started) * 1000, 1),
                "message": "连接成功" if not missing else f"连接成功，但缺少节点: {', '.join(missing)}",
            }
        except (ComfyError, httpx.HTTPError) as exc:
            return {"ok": False, "message": str(exc)}

    @app.get("/api/models")
    async def get_models(request: Request) -> dict[str, list[str]]:
        _settings, authority = _runtime_authority_snapshot(request)
        try:
            models = await _comfy(request).models()
        except ComfyError as exc:
            detail = exc.detail if isinstance(exc.detail, str) and exc.detail.strip() else str(exc)
            raise HTTPException(status_code=502, detail=detail) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        _assert_runtime_authority(request, authority)
        return models

    @app.get("/api/media/setup")
    async def get_media_setup(request: Request) -> dict[str, Any]:
        manager = request.app.state.ffmpeg_install_manager
        # Progress polling is a lightweight in-memory read. Executable startup
        # and encoder enumeration may be slow on Windows, so only idle/failed
        # states perform the real probe in a worker. Re-check the manager after
        # that await to avoid a mixed `ready:false + install:ready` response.
        status = manager.media_status_snapshot()
        if status is None:
            observed = await anyio.to_thread.run_sync(media_tools_status)
            status = manager.media_status_snapshot() or observed
        status["install"] = manager.snapshot()
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
        _settings, authority = _runtime_authority_snapshot(request)
        try:
            stats = await _comfy(request).system_stats()
            status = _raylight_runtime_status(database, stats)
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
        endpoint_key = _EMBEDDED_ENDPOINT_KEY
        if any(
            not task.done() for task in request.app.state.submission_tasks
        ):
            raise _raylight_recovery_in_flight(
                "RayLight recovery requires all Director submissions to finish"
            )
        async with request.app.state.submission_ticket_lock:
            tail = request.app.state.submission_tails.get(endpoint_key)
            if tail is not None and not tail.done():
                raise _raylight_recovery_in_flight(
                    "RayLight recovery is waiting for an endpoint submission"
                )
        submission_lock = request.app.state.submission_locks.setdefault(
            endpoint_key, anyio.Lock()
        )
        if submission_lock.locked():
            raise _raylight_recovery_in_flight(
                "RayLight recovery is waiting for the endpoint submission lock"
            )
        await submission_lock.acquire()
        try:
            client = _comfy(request)
            try:
                runtime_state = database.get_raylight_runtime_state()
                stats = await client.system_stats()
                status = _raylight_runtime_status(
                    database,
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
            legacy_unknown = bool(runtime_state.get("legacy_unknown"))
            tail_prompt_id = runtime_state.get("tail_prompt_id")
            tail_action = runtime_state.get("tail_action")
            if not legacy_unknown and (
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
            if (
                not legacy_unknown
                and not _raylight_runtime_has_terminal_certificate(runtime_state)
            ):
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
                stats,
                state=recovered,
            )
        finally:
            submission_lock.release()

    @app.get("/api/gpus")
    async def get_gpus(request: Request) -> dict[str, Any]:
        _settings, authority = _runtime_authority_snapshot(request)
        try:
            stats = await _comfy(request).system_stats()
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
    async def put_draft(
        request: Request, mode: GenerationMode, body: dict[str, Any]
    ) -> Any:
        # The models and read endpoint intentionally remain for historical
        # display/migration.  Accepting writes would recreate a second
        # creative authority after the v5 cut-over.
        del request, mode, body
        raise _legacy_generation_api_retired()

    @app.get("/api/timeline", response_model=UnifiedTimelineDraftV5)
    async def get_timeline(request: Request) -> UnifiedTimelineDraftV5:
        return _db(request).get_timeline()

    @app.put("/api/timeline")
    async def put_timeline(
        request: Request, body: dict[str, Any]
    ) -> Any:
        database = _db(request)
        project_id = database.LEGACY_DEFAULT_PROJECT_ID
        if body.get("version") == 4:
            raise _timeline_schema_migrated(database, project_id)
        raise _timeline_authority_required(project_id)

    @app.get("/api/timeline/authority", response_model=TimelineAuthorityRead)
    async def get_timeline_authority(request: Request) -> TimelineAuthorityRead:
        document, revision = _db(request).get_timeline_authority()
        return TimelineAuthorityRead(document=document, revision=revision)

    def feature_bundle_migration_notices(
        database: Database,
        project_id: str,
    ) -> FeatureBundleMigrationNoticeListRead:
        if not database.project_exists(project_id):
            raise HTTPException(status_code=404, detail="project not found")
        return FeatureBundleMigrationNoticeListRead(
            notices=database.list_feature_bundle_migration_notices(project_id)
        )

    @app.get(
        "/api/timeline/migration-notices",
        response_model=FeatureBundleMigrationNoticeListRead,
    )
    async def get_timeline_migration_notices(
        request: Request,
    ) -> FeatureBundleMigrationNoticeListRead:
        database = _db(request)
        return feature_bundle_migration_notices(
            database,
            database.LEGACY_DEFAULT_PROJECT_ID,
        )

    @app.put("/api/timeline/authority", response_model=TimelineAuthorityRead)
    async def put_timeline_authority(
        request: Request,
        body: dict[str, Any],
    ) -> TimelineAuthorityRead:
        database = _db(request)
        parsed = _parse_v5_timeline_authority_write(
            database,
            database.LEGACY_DEFAULT_PROJECT_ID,
            body,
        )
        try:
            document, revision = database.validate_and_put_timeline_authority(
                parsed.document,
                expected_revision=parsed.expected_revision,
            )
        except TimelineSchemaMigrated:
            raise _timeline_schema_migrated(
                database, database.LEGACY_DEFAULT_PROJECT_ID
            ) from None
        except TimelineRevisionConflict as exc:
            raise _timeline_revision_conflict(exc) from None
        except TimelineRevisionExhausted as exc:
            raise _timeline_revision_exhausted(exc) from None
        except TimelineTemplateBundleConflict as exc:
            raise _timeline_template_bundle_conflict(exc) from None
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
        return ProjectListRead(projects=database.list_projects())

    @app.post("/api/projects", response_model=ProjectSummaryRead)
    async def create_project(
        request: Request, body: ProjectCreateRequest
    ) -> ProjectSummaryRead:
        project = _db(request).create_project(
            body.title,
            initial_model_stack=body.initial_model_stack,
        )
        return _project_summary(project)

    @app.post("/api/projects/import")
    async def import_project_legacy(
        request: Request, body: dict[str, Any]
    ) -> Any:
        del request, body
        raise _project_import_preflight_required()

    @app.post(
        "/api/projects/import/preflight",
        response_model=ProjectImportPreflightRead,
    )
    async def preflight_project_import(
        request: Request,
        body: ProjectImportPreflightRequest,
    ) -> ProjectImportPreflightRead:
        try:
            digest, proposed, missing_context, missing_models = (
                prepare_project_import(
                    body,
                    migrate_v4_to_v5=migrate_timeline_v4_to_v5,
                )
            )
        except ProjectImportError as exc:
            raise _project_import_http_error(exc) from exc
        if proposed is None:
            return ProjectImportPreflightRead(
                status="needs_input",
                input_digest=digest,
                missing_context=missing_context,
                missing_model_bindings=missing_models,
            )
        try:
            _db(request).validate_timeline_assets(proposed)
        except (ValidationError, ValueError) as exc:
            reason = _creative_input_reason(exc)
            return ProjectImportPreflightRead(
                status="needs_input",
                input_digest=digest,
                proposed_document=proposed,
                missing_context=["assets"],
                missing_model_bindings=missing_models,
                capability_issues=[reason.model_dump(mode="json")],
            )
        try:
            capability_issues = await _project_import_capability_issues(
                request,
                proposed,
            )
            return request.app.state.project_import_coordinator.issue(
                title=body.title,
                input_digest=digest,
                proposed_document=proposed,
                missing_model_bindings=missing_models,
                capability_issues=capability_issues,
            )
        except ProjectImportError as exc:
            raise _project_import_http_error(exc) from exc

    @app.post(
        "/api/projects/import/commit",
        response_model=ProjectSummaryRead,
    )
    async def commit_project_import(
        request: Request,
        body: ProjectImportCommitRequest,
    ) -> ProjectSummaryRead:
        try:
            title, proposed = (
                request.app.state.project_import_coordinator.consume(body)
            )
            database = _db(request)
            # ``import_project`` revalidates assets and inserts under one
            # BEGIN IMMEDIATE transaction, closing the preflight/delete race.
            project = database.import_project(title, proposed)
        except ProjectImportError as exc:
            raise _project_import_http_error(exc) from exc
        except (ValidationError, ValueError) as exc:
            raise _creative_input_http_error(_creative_input_reason(exc)) from exc
        return _project_summary(project)

    @app.get(
        "/api/projects/{project_id}/migration-receipts/latest",
        response_model=ProjectMigrationReceipt,
    )
    async def get_latest_project_migration_receipt(
        request: Request,
        project_id: str,
        from_schema: Annotated[int, Query(alias="from", ge=1)] = 4,
        to_schema: Annotated[int, Query(alias="to", ge=2)] = 5,
    ) -> ProjectMigrationReceipt:
        receipt = _db(request).get_latest_project_migration_receipt(
            project_id,
            from_schema=from_schema,
            to_schema=to_schema,
        )
        if receipt is None:
            raise HTTPException(status_code=404, detail="migration receipt not found")
        return receipt

    @app.get(
        "/api/projects/{project_id}/migration-receipts/{migration_id}",
        response_model=ProjectMigrationReceipt,
    )
    async def get_project_migration_receipt(
        request: Request,
        project_id: str,
        migration_id: str,
    ) -> ProjectMigrationReceipt:
        receipt = _db(request).get_project_migration_receipt(
            project_id,
            migration_id,
        )
        if receipt is None:
            raise HTTPException(status_code=404, detail="migration receipt not found")
        return receipt

    @app.get("/api/projects/{project_id}", response_model=ProjectSummaryRead)
    async def get_project(request: Request, project_id: str) -> ProjectSummaryRead:
        project = _db(request).get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        return _project_summary(project)

    @app.get("/api/projects/{project_id}/export")
    async def export_project(request: Request, project_id: str) -> Response:
        raw_document = _db(request).get_project_raw_document(project_id)
        if raw_document is None:
            raise HTTPException(status_code=404, detail="project not found")
        filename = quote(f"{project_id}.directordeck-project.json", safe="")
        return Response(
            content=raw_document.encode("utf-8"),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{filename}"
            },
        )

    @app.patch("/api/projects/{project_id}")
    async def rename_project(project_id: str) -> None:
        raise _project_rename_api_retired(project_id)

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
        response_model=UnifiedTimelineDraftV5,
    )
    async def get_project_timeline(
        request: Request, project_id: str
    ) -> UnifiedTimelineDraftV5:
        try:
            return _db(request).get_project_timeline(project_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="project not found") from None
        except (TypeError, ValueError, ValidationError):
            raise _project_document_unreadable(project_id) from None

    @app.put("/api/projects/{project_id}/timeline")
    async def put_project_timeline(
        request: Request, project_id: str, body: dict[str, Any]
    ) -> Any:
        database = _db(request)
        if database.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="project not found") from None
        if body.get("version") == 4:
            raise _timeline_schema_migrated(database, project_id)
        raise _timeline_authority_required(project_id)

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
        except (TypeError, ValueError, ValidationError):
            raise _project_document_unreadable(project_id) from None
        return TimelineAuthorityRead(document=document, revision=revision)

    @app.get(
        "/api/projects/{project_id}/migration-notices",
        response_model=FeatureBundleMigrationNoticeListRead,
    )
    async def get_project_migration_notices(
        request: Request,
        project_id: str,
    ) -> FeatureBundleMigrationNoticeListRead:
        return feature_bundle_migration_notices(_db(request), project_id)

    @app.put(
        "/api/projects/{project_id}/timeline/authority",
        response_model=TimelineAuthorityRead,
    )
    async def put_project_timeline_authority(
        request: Request,
        project_id: str,
        body: dict[str, Any],
    ) -> TimelineAuthorityRead:
        database = _db(request)
        if database.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        parsed = _parse_v5_timeline_authority_write(
            database,
            project_id,
            body,
        )
        try:
            document, revision = (
                database.validate_and_put_project_timeline_authority(
                    project_id,
                    parsed.document,
                    expected_revision=parsed.expected_revision,
                )
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="project not found") from None
        except TimelineSchemaMigrated:
            raise _timeline_schema_migrated(database, project_id) from None
        except TimelineRevisionConflict as exc:
            raise _timeline_revision_conflict(exc) from None
        except TimelineRevisionExhausted as exc:
            raise _timeline_revision_exhausted(exc) from None
        except TimelineTemplateBundleConflict as exc:
            raise _timeline_template_bundle_conflict(exc) from None
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
        return await _create_timeline_job(request, body, project_id=project_id)

    @app.get("/api/assets", response_model=AssetListRead)
    async def list_assets(
        request: Request, kind: AssetKind | None = None
    ) -> AssetListRead:
        database = _db(request)
        try:
            assets = database.list_assets(kind=kind)
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return AssetListRead(assets=assets)

    @app.get("/api/asset-trash", response_model=AssetTrashListRead)
    async def list_asset_trash(request: Request) -> AssetTrashListRead:
        database = _db(request)
        batches = database.list_asset_trash_batches()
        return AssetTrashListRead(
            batches=[AssetTrashBatchRead.model_validate(item) for item in batches],
        )

    @app.post("/api/asset-trash", response_model=AssetTrashBatchRead)
    async def trash_assets(
        request: Request, body: AssetTrashRequest
    ) -> AssetTrashBatchRead:
        database = _db(request)
        try:
            result = database.trash_assets(
                body.asset_ids,
                cascade=body.cascade,
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
        try:
            result = database.restore_asset_trash_batch(
                batch_id,
                mode=body.mode,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="asset trash batch not found"
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
        try:
            result = database.purge_asset_trash_batch(batch_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="asset trash batch not found"
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
            comfy = _comfy(request)
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
                _db(request).put_asset(asset_id, asset)
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
        try:
            usages = database.delete_asset_if_unused(
                asset_id,
                cascade=cascade,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="asset not found") from exc
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
        asset = database.get_asset_record(asset_id)
        if asset is None:
            raise HTTPException(status_code=404, detail="asset not found")
        try:
            return await _proxy_comfy_media(
                _comfy(request),
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
        comfy = _comfy(request)
        document = database.get_asset_record(body.asset_id)
        if document is None:
            raise HTTPException(status_code=404, detail="asset not found")
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
        request: Request, body: dict[str, Any]
    ) -> JobRead:
        # Generic job creation accepts the new unified contract as well as the
        # legacy six-mode body.  The explicit /api/timeline/jobs route remains
        # the clearest client API, while this keeps automation on /api/jobs.
        if "mode" in body:
            raise _legacy_generation_api_retired()
        try:
            timeline_request = TimelineJobRequest.model_validate(body)
        except ValidationError as exc:
            raise _validation_error(exc) from exc
        return await _create_timeline_job(request, timeline_request)

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
        active_project_id, current_timeline, current_settings = _job_read_context_for_project(
            request,
            project_id,
        )
        execution_digest_cache: dict[str, Any] = {}
        return JobListRead(
            jobs=[
                _job_read_for_request(
                    request,
                    job,
                    current_timeline=current_timeline,
                    current_settings=current_settings,
                    current_project_id=active_project_id,
                    current_execution_digest_cache=execution_digest_cache,
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
    async def get_job(
        request: Request,
        job_id: str,
        project_id: Annotated[str | None, Query(max_length=128)] = None,
    ) -> JobRead:
        database = _db(request)
        job = database.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        job["children"] = database.list_job_children(job_id)
        job = _job_with_parent_output_authority(database, job)
        return _job_read_for_project_scope(
            request,
            job,
            project_id=project_id,
        )

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
        """Return a v5 creative view resolved only from immutable job evidence."""

        job = _db(request).get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        try:
            project = resolve_historical_creative_input(
                job,
                migrate_v4_to_v5=migrate_timeline_v4_to_v5,
            )
        except (HistoricalCreativeInputError, ProjectImportError) as exc:
            if isinstance(exc, ProjectImportError):
                historical = HistoricalCreativeInputError(exc.code, exc.message)
            else:
                historical = exc
            raise _historical_creative_http_error(historical) from exc
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
        "/api/jobs/{job_id}/save-as-project",
        response_model=ProjectSummaryRead,
    )
    async def save_historical_job_as_project(
        request: Request,
        job_id: str,
        body: HistoricalSaveAsProjectRequest,
    ) -> ProjectSummaryRead:
        """Create a fresh project scope without copying any historical ledger."""

        database = _db(request)
        job = database.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        try:
            project = resolve_historical_creative_input(
                job,
                migrate_v4_to_v5=migrate_timeline_v4_to_v5,
            )
            if body.title.strip():
                project = project.model_copy(
                    update={"title": body.title.strip()}
                )
            saved = database.import_project(body.title, project)
        except (HistoricalCreativeInputError, ProjectImportError) as exc:
            if isinstance(exc, ProjectImportError):
                historical = HistoricalCreativeInputError(exc.code, exc.message)
            else:
                historical = exc
            raise _historical_creative_http_error(historical) from exc
        except (ValidationError, ValueError) as exc:
            raise _creative_input_http_error(_creative_input_reason(exc)) from exc
        return _project_summary(saved)

    @app.post(
        "/api/jobs/{job_id}/import-output",
        response_model=JobOutputImportRead,
    )
    async def import_job_output(
        request: Request,
        job_id: str,
        body: JobOutputImportRequest,
    ) -> JobOutputImportRead:
        """Copy one persisted generated video into the host's input library.

        The browser supplies only a parent-local output index or a stable
        timeline segment ID. Source paths are resolved from durable Director
        rows and read through the embedded ComfyUI connection; arbitrary
        client paths and URLs never cross this boundary.
        """

        database = _db(request)
        job = database.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        job["children"] = database.list_job_children(job_id)
        job = _job_with_parent_output_authority(database, job)
        try:
            asset = await import_job_output_as_asset(
                registry=database,
                client=_comfy(request),
                job=job,
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

    @app.post("/api/jobs/{job_id}/retry", response_model=JobRead)
    async def retry_job(request: Request, job_id: str) -> JobRead:
        """Create a fresh job from historical creative config only.

        No compiled, exact-prompt, child, prompt-id, or runtime ledger state is
        copied.  The normal creation path therefore performs current schema
        validation, model inspection, capability preflight, compilation, and
        creates an entirely new execution-evidence lineage.
        """

        database = _db(request)
        source = database.get_job(job_id)
        if source is None:
            raise HTTPException(status_code=404, detail="job not found")
        dispatcher = request.app.state.submission_jobs.get(job_id)
        if dispatcher is not None and not dispatcher.done():
            raise HTTPException(
                status_code=409,
                detail="job submission still owns prompt side effects",
            )

        children = database.list_job_children(job_id)
        unreleased: list[str] = []
        for child in children:
            try:
                ownership = _typed_prompt_ownership_for_child(database, child)
            except ExecutionEvidenceConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            if ownership is not None:
                if ownership.state not in {
                    "cleanup_confirmed",
                    "terminal_confirmed",
                }:
                    unreleased.append(str(child["id"]))
            elif (
                source["status"] not in _TERMINAL_STATUSES
                and child.get("prompt_id")
            ):
                # Legacy jobs have no structured ownership evidence.  Their
                # active prompt projection remains the conservative gate.
                unreleased.append(str(child["id"]))
        if unreleased:
            raise HTTPException(
                status_code=409,
                detail=(
                    "job still owns one or more ComfyUI prompts: "
                    + ", ".join(unreleased)
                ),
            )

        # Retry is a new write and therefore must start from the strict v5
        # creative view. Historical v4 jobs are upgraded only through their
        # own immutable settings snapshot by the historical resolver; the
        # display-only v4 projection is never submitted back to the v5 API.
        timeline = _job_v5_creative_snapshot(source)
        if timeline is None:
            raise HTTPException(
                status_code=409,
                detail="historical job config cannot migrate to the current schema",
            )
        config_snapshot = source.get("config_snapshot")
        raw_segment_ids = (
            config_snapshot.get("segment_ids")
            if isinstance(config_snapshot, dict)
            else None
        )
        try:
            segment_ids = (
                [str(value) for value in raw_segment_ids]
                if isinstance(raw_segment_ids, list)
                else None
            )
            retry_body = TimelineJobRequest(
                config=timeline,
                segment_ids=segment_ids,
            )
        except (ValidationError, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail="historical job selection cannot migrate to the current schema",
            ) from exc
        return await _create_timeline_job(
            request,
            retry_body,
            parent_mode=str(source["mode"]),
            project_id=(
                str(source["project_id"])
                if source.get("project_id") is not None
                else None
            ),
        )

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
        except ExecutionEvidenceConflict as exc:
            raise HTTPException(
                status_code=409,
                detail=(
                    "job still owns unresolved ComfyUI prompt evidence; "
                    "complete cancellation or recovery before deleting it"
                ),
            ) from exc
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
        project_id: Annotated[str | None, Query(max_length=128)] = None,
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
            settled = _db(request).confirm_comfy_restart_recovery(
                job_id,
                current_endpoint_identity=request.app.state.endpoint_identity,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        except (ExecutionEvidenceConflict, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        request.app.state.live_preview_cache.discard(job_id)
        settled["children"] = _db(request).list_job_children(job_id)
        return _job_read_for_project_scope(
            request,
            settled,
            project_id=project_id,
        )

    async def _cancel_job_with_read_context(
        request: Request,
        job_id: str,
        *,
        current_project_id: str,
        current_timeline: UnifiedTimelineDraftV5 | None,
        current_settings: RuntimeSettingsV3,
    ) -> JobRead:
        def read_job(snapshot: dict[str, Any]) -> JobRead:
            return _job_read_for_request(
                request,
                snapshot,
                current_timeline=current_timeline,
                current_settings=current_settings,
                current_project_id=current_project_id,
            )

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
            return read_job(
                await _cancel_timeline_job(
                    request,
                    job,
                    initial_cancel_claimed=initial_cancel_claimed,
                ),
            )
        if job["status"] in _TERMINAL_STATUSES:
            return read_job(job)
        job = await _sync_existing_job(request, job)
        if job["status"] in _TERMINAL_STATUSES:
            return read_job(job)
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
                return read_job(
                    await _quiesce_cancelled_submission_dispatcher(
                        request, job
                    )
                )
            latest = database.get_job(job_id)
            if latest is None:
                raise HTTPException(status_code=404, detail="job not found")
            job = latest
            if job["status"] in _TERMINAL_STATUSES:
                return read_job(job)
        while job["status"] != "cancelling":
            if job["status"] in _TERMINAL_STATUSES:
                return read_job(job)
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
                client = _comfy(request)
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
                            return read_job(job)
                    job = database.update_job_if_status(
                        job_id,
                        "cancelling",
                        stage="cancel_unconfirmed",
                    ) or job
                    return read_job(job)
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
                return read_job(job)
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
        return read_job(job)

    @app.post("/api/jobs/{job_id}/cancel", response_model=JobRead)
    async def cancel_job(
        request: Request,
        job_id: str,
        project_id: Annotated[str | None, Query(max_length=128)] = None,
    ) -> JobRead:
        active_project_id, current_timeline, current_settings = _job_read_context_for_project(
            request,
            project_id,
        )
        try:
            return await _cancel_job_with_read_context(
                request,
                job_id,
                current_project_id=active_project_id,
                current_timeline=current_timeline,
                current_settings=current_settings,
            )
        except ExecutionEvidenceConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/jobs/cancel", response_model=JobBulkCancelRead)
    async def cancel_jobs(
        request: Request,
        body: JobBulkCancelRequest,
        project_id: Annotated[str | None, Query(max_length=128)] = None,
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
        active_project_id, current_timeline, current_settings = _job_read_context_for_project(
            request,
            project_id,
        )
        results: list[JobRead] = []
        for job_id in body.job_ids:
            try:
                results.append(
                    await _cancel_job_with_read_context(
                        request,
                        job_id,
                        current_project_id=active_project_id,
                        current_timeline=current_timeline,
                        current_settings=current_settings,
                    )
                )
            except ExecutionEvidenceConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
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
        job["children"] = [
            _child_with_execution_evidence(database, child)
            for child in database.list_job_children(job_id)
        ]
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
                _comfy(request),
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
        job["children"] = database.list_job_children(job_id)
        job = _job_with_parent_output_authority(database, job)
        outputs = authoritative_parent_outputs(job)
        if index >= len(outputs):
            raise HTTPException(status_code=404, detail="job output not found")
        output = outputs[index]
        try:
            return await _proxy_comfy_media(
                _comfy(request),
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
