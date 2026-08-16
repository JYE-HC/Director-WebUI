from __future__ import annotations

from typing import Any

import httpx
import pytest

from director.media import VideoProxy
from director.schemas import (
    RuntimeSettings,
    VideoMetadata,
    default_settings,
    mode_draft_to_timeline,
    validate_mode_draft,
)
from director.task_management import (
    TaskManagementError,
    import_job_output_as_asset,
    resolve_job_output,
)

from .conftest import VIDEO_METADATA, runnable_draft


def _timeline() -> dict[str, Any]:
    draft = validate_mode_draft("t2v", runnable_draft("t2v"))
    return mode_draft_to_timeline(draft, title="来源项目").model_dump(mode="json")


def _job(*, source_url: str = "http://source.test:8188") -> dict[str, Any]:
    timeline = _timeline()
    segment_id = timeline["segments"][0]["id"]
    return {
        "id": "job-1",
        "mode": "timeline",
        "status": "succeeded",
        "progress": 1.0,
        "stage": "completed",
        "prompt_id": None,
        "outputs": [
            {
                "node_id": "assembly",
                "filename": "Director_full.mp4",
                "subfolder": "director-web/timelines",
                "type": "output",
            }
        ],
        "error": None,
        "config_snapshot": {"timeline": timeline, "segment_ids": [segment_id]},
        "settings_snapshot": default_settings(source_url).model_dump(mode="json"),
        "prompt_snapshot": {"do_not": "leak"},
        "created_at": "2026-08-13T00:00:00+00:00",
        "updated_at": "2026-08-13T00:01:00+00:00",
        "started_at": "2026-08-13T00:00:02+00:00",
        "completed_at": "2026-08-13T00:01:00+00:00",
        "children": [
            {
                "id": "child-1",
                "family": "fl2va",
                "backend": "raylight",
                "segment_ids": [segment_id],
                "output_nodes": {segment_id: "save-1"},
                "status": "succeeded",
                "progress": 1.0,
                "stage": "completed",
                "prompt_id": "prompt-1",
                "outputs": [
                    {
                        "node_id": "save-1",
                        "filename": "segment-1.mp4",
                        "subfolder": "director-web/segments",
                        "type": "output",
                    }
                ],
                "error": None,
            }
        ],
    }


def test_output_resolution_uses_only_persisted_index_or_stable_segment_map() -> None:
    job = _job()
    segment_id = job["children"][0]["segment_ids"][0]

    assert resolve_job_output(job, output_index=0)["filename"] == "Director_full.mp4"
    assert resolve_job_output(job, segment_id=segment_id)["filename"] == "segment-1.mp4"

    with pytest.raises(TaskManagementError, match="必须且只能"):
        resolve_job_output(job)
    with pytest.raises(TaskManagementError, match="必须且只能"):
        resolve_job_output(job, output_index=0, segment_id=segment_id)

    forged = _job()
    forged["outputs"][0]["subfolder"] = "../secrets"
    with pytest.raises(TaskManagementError, match="子目录无效"):
        resolve_job_output(forged, output_index=0)

    ambiguous = _job()
    ambiguous["children"].append(dict(ambiguous["children"][0], id="child-2"))
    with pytest.raises(TaskManagementError, match="不唯一"):
        resolve_job_output(ambiguous, segment_id=segment_id)


class _Registry:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, Any], str]] = []
        self.accept_current_origin = True

    def put_asset(
        self,
        asset_id: str,
        document: dict[str, Any],
        *,
        comfy_origin: str,
    ) -> None:
        self.records.append((asset_id, document, comfy_origin))

    def put_asset_if_current_origin(
        self,
        asset_id: str,
        document: dict[str, Any],
        *,
        expected_comfy_origin: str,
    ) -> bool:
        if not self.accept_current_origin:
            return False
        self.records.append((asset_id, document, expected_comfy_origin))
        return True


class _Comfy:
    def __init__(self, *, content: bytes = b"generated-output") -> None:
        self.content = content
        self.views: list[dict[str, str]] = []
        self.uploads: list[dict[str, Any]] = []

    async def view(self, params: dict[str, str]) -> httpx.Response:
        self.views.append(dict(params))
        return httpx.Response(
            200,
            content=self.content,
            headers={"content-type": "video/mp4"},
            request=httpx.Request("GET", "http://comfy.test/view"),
        )

    async def upload(
        self,
        filename: str,
        content: bytes,
        content_type: str,
        kind: str,
    ) -> dict[str, Any]:
        self.uploads.append(
            {
                "filename": filename,
                "content": content,
                "content_type": content_type,
                "kind": kind,
            }
        )
        return {"name": filename, "subfolder": "director-web", "type": "input"}


async def test_import_output_reads_snapshot_origin_and_registers_at_live_origin(
    monkeypatch,
) -> None:
    source = _Comfy()
    target = _Comfy()
    clients = {
        "http://source.test:8188": source,
        "http://target.test:8288": target,
    }
    registry = _Registry()

    def proxy(content: bytes, _suffix: str) -> VideoProxy:
        assert content == b"generated-output"
        return VideoProxy(
            content=b"normalized-24fps",
            filename_suffix=".mp4",
            metadata=VideoMetadata.model_validate(VIDEO_METADATA),
        )

    async def run_sync(function):
        return function()

    monkeypatch.setattr("director.task_management.anyio.to_thread.run_sync", run_sync)
    monkeypatch.setattr("director.task_management.create_24fps_proxy_bytes", proxy)

    def factory(settings: RuntimeSettings):
        return clients[str(settings.comfy_url).rstrip("/")]

    asset = await import_job_output_as_asset(
        registry=registry,
        comfy_factory=factory,
        job=_job(),
        target_settings=default_settings("http://target.test:8288"),
        output_index=0,
    )

    assert source.views == [
        {
            "filename": "Director_full.mp4",
            "subfolder": "director-web/timelines",
            "type": "output",
        }
    ]
    assert target.uploads == [
        {
            "filename": "Director_full_24fps.mp4",
            "content": b"normalized-24fps",
            "content_type": "video/mp4",
            "kind": "video",
        }
    ]
    assert asset.type == "input"
    assert asset.kind == "video"
    assert asset.metadata is not None and asset.metadata.native_fps == 24
    assert registry.records[0][0] == asset.id
    assert registry.records[0][2] == "http://target.test:8288"


async def test_same_origin_import_still_reads_normalizes_and_uploads(monkeypatch) -> None:
    comfy = _Comfy()
    registry = _Registry()
    async def run_sync(function):
        return function()

    monkeypatch.setattr("director.task_management.anyio.to_thread.run_sync", run_sync)
    monkeypatch.setattr(
        "director.task_management.create_24fps_proxy_bytes",
        lambda content, _suffix: VideoProxy(
            content=b"normalized",
            filename_suffix=".mp4",
            metadata=VideoMetadata.model_validate(VIDEO_METADATA),
        ),
    )

    await import_job_output_as_asset(
        registry=registry,
        comfy_factory=lambda _settings: comfy,
        job=_job(),
        target_settings=default_settings("http://source.test:8188"),
        output_index=0,
    )

    assert len(comfy.views) == 1
    assert len(comfy.uploads) == 1


async def test_import_does_not_register_if_live_target_changes(monkeypatch) -> None:
    source = _Comfy()
    target = _Comfy()
    registry = _Registry()

    async def run_sync(function):
        return function()

    monkeypatch.setattr("director.task_management.anyio.to_thread.run_sync", run_sync)
    monkeypatch.setattr(
        "director.task_management.create_24fps_proxy_bytes",
        lambda content, _suffix: VideoProxy(
            content=b"normalized",
            filename_suffix=".mp4",
            metadata=VideoMetadata.model_validate(VIDEO_METADATA),
        ),
    )

    def factory(settings: RuntimeSettings):
        origin = str(settings.comfy_url).rstrip("/")
        return source if origin == "http://source.test:8188" else target

    with pytest.raises(TaskManagementError, match="地址已变更"):
        await import_job_output_as_asset(
            registry=registry,
            comfy_factory=factory,
            job=_job(),
            target_settings=default_settings("http://target.test:8288"),
            current_settings=lambda: default_settings("http://new.test:8388"),
            output_index=0,
        )

    assert len(source.views) == 1
    assert len(target.uploads) == 1
    assert registry.records == []


async def test_import_atomically_rejects_endpoint_change_at_asset_insert(
    monkeypatch,
) -> None:
    comfy = _Comfy()
    registry = _Registry()
    registry.accept_current_origin = False

    async def run_sync(function):
        return function()

    monkeypatch.setattr("director.task_management.anyio.to_thread.run_sync", run_sync)
    monkeypatch.setattr(
        "director.task_management.create_24fps_proxy_bytes",
        lambda content, _suffix: VideoProxy(
            content=b"normalized",
            filename_suffix=".mp4",
            metadata=VideoMetadata.model_validate(VIDEO_METADATA),
        ),
    )

    with pytest.raises(TaskManagementError, match="地址已变更"):
        await import_job_output_as_asset(
            registry=registry,
            comfy_factory=lambda _settings: comfy,
            job=_job(),
            target_settings=default_settings("http://source.test:8188"),
            current_settings=lambda: default_settings("http://source.test:8188"),
            output_index=0,
        )

    assert len(comfy.uploads) == 1
    assert registry.records == []


@pytest.mark.parametrize(
    "uploaded",
    [
        {"name": "take.mp4", "subfolder": "", "type": "output"},
        {"name": "../take.mp4", "subfolder": "", "type": "input"},
        {"name": "take.mp4", "subfolder": "../private", "type": "input"},
    ],
)
async def test_import_rejects_invalid_comfy_upload_contract(
    monkeypatch, uploaded: dict[str, str]
) -> None:
    comfy = _Comfy()

    async def bad_upload(*_args, **_kwargs):
        return uploaded

    async def run_sync(function):
        return function()

    comfy.upload = bad_upload  # type: ignore[method-assign]
    monkeypatch.setattr("director.task_management.anyio.to_thread.run_sync", run_sync)
    monkeypatch.setattr(
        "director.task_management.create_24fps_proxy_bytes",
        lambda content, _suffix: VideoProxy(
            content=b"normalized",
            filename_suffix=".mp4",
            metadata=VideoMetadata.model_validate(VIDEO_METADATA),
        ),
    )

    with pytest.raises(TaskManagementError) as raised:
        await import_job_output_as_asset(
            registry=_Registry(),
            comfy_factory=lambda _settings: comfy,
            job=_job(),
            target_settings=default_settings("http://source.test:8188"),
            output_index=0,
        )
    assert raised.value.status_code == 502
