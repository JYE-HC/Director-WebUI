from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from directordeck.app import _segment_output_candidates
from directordeck.media import VideoProxy
from directordeck.migrations.timeline_v4_v5 import migrate_timeline_v4_to_v5
from directordeck.schemas import (
    UnifiedTimelineDraftV4,
    VideoMetadata,
    default_settings,
    mode_draft_to_timeline,
    validate_mode_draft,
)
from directordeck.task_management import (
    TaskManagementError,
    import_job_output_as_asset,
    resolve_job_output,
)
from directordeck.workflow.execution import ObservedArtifactSpec, OutputDescriptor

from .conftest import VIDEO_METADATA, runnable_draft


def _timeline() -> dict[str, Any]:
    draft = validate_mode_draft("t2v", runnable_draft("t2v"))
    return mode_draft_to_timeline(draft, title="来源项目").model_dump(mode="json")


def _job() -> dict[str, Any]:
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
                "subfolder": "directordeck/timelines",
                "type": "output",
            }
        ],
        "error": None,
        "config_snapshot": {"timeline": timeline, "segment_ids": [segment_id]},
        "settings_snapshot": default_settings().model_dump(mode="json"),
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
                        "subfolder": "directordeck/segments",
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


def test_segment_output_resolution_accepts_strict_v5_job_snapshot() -> None:
    job = _job()
    job["config_snapshot"]["timeline"] = migrate_timeline_v4_to_v5(
        UnifiedTimelineDraftV4.model_validate(job["config_snapshot"]["timeline"]),
        default_settings(),
    ).model_dump(mode="json")
    segment_id = job["children"][0]["segment_ids"][0]

    assert resolve_job_output(job, segment_id=segment_id)["filename"] == (
        "segment-1.mp4"
    )


def test_typed_output_resolution_never_falls_back_to_mutable_child_columns() -> None:
    job = _job()
    child = job["children"][0]
    segment_id = child["segment_ids"][0]
    expected = SimpleNamespace(segment_id=segment_id, node_id="trusted-save")
    child["execution_evidence"] = {
        "exact_prompt_snapshot": SimpleNamespace(expected_output_spec=expected)
    }
    child["observed_artifact"] = ObservedArtifactSpec(
        segment_id=segment_id,
        child_id=child["id"],
        output_descriptor=OutputDescriptor(
            filename="observed.mp4",
            subfolder="directordeck/observed",
            type="output",
        ),
        width=864,
        height=480,
        fps=24,
        frame_count=25,
        duration_seconds=25 / 24,
        has_audio=True,
        media_probe_version="test-ffprobe-v1",
    )

    assert resolve_job_output(job, segment_id=segment_id)["filename"] == "observed.mp4"
    assert _segment_output_candidates(job)[segment_id][0][1]["filename"] == (
        "observed.mp4"
    )

    child["observed_artifact"] = child["observed_artifact"].model_copy(
        update={"child_id": "forged-child"}
    )
    with pytest.raises(TaskManagementError, match="可信的实际媒体"):
        resolve_job_output(job, segment_id=segment_id)
    assert _segment_output_candidates(job) == {}

    child["execution_evidence"] = {"invalid": True}
    child["observed_artifact"] = None
    with pytest.raises(TaskManagementError, match="可信的实际媒体"):
        resolve_job_output(job, segment_id=segment_id)


class _Registry:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, Any]]] = []

    def put_asset(
        self,
        asset_id: str,
        document: dict[str, Any],
    ) -> None:
        self.records.append((asset_id, document))


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
        return {"name": filename, "subfolder": "directordeck", "type": "input"}


async def test_import_output_reads_normalizes_uploads_and_registers(
    monkeypatch,
) -> None:
    comfy = _Comfy()
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

    monkeypatch.setattr("directordeck.task_management.anyio.to_thread.run_sync", run_sync)
    monkeypatch.setattr("directordeck.task_management.create_24fps_proxy_bytes", proxy)

    asset = await import_job_output_as_asset(
        registry=registry,
        client=comfy,
        job=_job(),
        output_index=0,
    )

    assert comfy.views == [
        {
            "filename": "Director_full.mp4",
            "subfolder": "directordeck/timelines",
            "type": "output",
        }
    ]
    assert comfy.uploads == [
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
    monkeypatch.setattr("directordeck.task_management.anyio.to_thread.run_sync", run_sync)
    monkeypatch.setattr(
        "directordeck.task_management.create_24fps_proxy_bytes",
        lambda content, _suffix: VideoProxy(
            content=b"normalized",
            filename_suffix=".mp4",
            metadata=VideoMetadata.model_validate(VIDEO_METADATA),
        ),
    )

    with pytest.raises(TaskManagementError) as raised:
        await import_job_output_as_asset(
            registry=_Registry(),
            client=comfy,
            job=_job(),
            output_index=0,
        )
    assert raised.value.status_code == 502
