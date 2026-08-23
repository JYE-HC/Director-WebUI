from __future__ import annotations

import uuid

from directordeck.compiler import timeline_segment_take_fingerprint
from directordeck.schemas import (
    UnifiedTimelineDraft,
    default_settings,
    default_timeline_draft,
)

from .conftest import asset, save_timeline_document


async def test_project_list_create_rename_delete(client) -> None:
    database = client.director_app.state.database

    listed = (await client.get("/api/projects")).json()
    ids = [project["id"] for project in listed["projects"]]
    assert database.LEGACY_DEFAULT_PROJECT_ID in ids

    created = (await client.post("/api/projects", json={"title": "第二部影片"})).json()
    assert created["id"] != database.LEGACY_DEFAULT_PROJECT_ID
    assert created["title"] == "第二部影片"
    assert created["segment_count"] == 1

    listed = (await client.get("/api/projects")).json()
    assert any(project["id"] == created["id"] for project in listed["projects"])

    # A fresh project gets a fresh stable segment id, never the legacy seed.
    timeline = (await client.get(f"/api/projects/{created['id']}/timeline")).json()
    assert timeline["segments"][0]["id"] != "timeline-segment-1"

    timeline["title"] = "改名后"
    renamed = await save_timeline_document(
        client,
        timeline,
        project_id=created["id"],
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "改名后"
    # CAS timeline writes keep the list title and document title in sync.
    summary = await client.get(f"/api/projects/{created['id']}")
    assert summary.status_code == 200
    assert summary.json()["title"] == "改名后"
    timeline = (await client.get(f"/api/projects/{created['id']}/timeline")).json()
    assert timeline["title"] == "改名后"

    deleted = await client.delete(f"/api/projects/{created['id']}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted_project_id"] == created["id"]
    assert deleted.json()["outputs_preserved"] is True

    refused = await client.delete(f"/api/projects/{database.LEGACY_DEFAULT_PROJECT_ID}")
    assert refused.status_code == 409


async def test_project_timeline_roundtrip_isolated_from_default(client) -> None:
    created = (await client.post("/api/projects", json={"title": "独立项目"})).json()
    document = (
        await client.get(f"/api/projects/{created['id']}/timeline")
    ).json()
    document["title"] = "独立项目"
    document["segments"][0]["prompt"] = "独立项目的第一段"

    put = await save_timeline_document(
        client,
        document,
        project_id=created["id"],
    )
    assert put.status_code == 200

    default_timeline = (await client.get("/api/timeline")).json()
    assert default_timeline["segments"][0]["prompt"] != "独立项目的第一段"

    fetched = (await client.get(f"/api/projects/{created['id']}/timeline")).json()
    assert fetched["segments"][0]["prompt"] == "独立项目的第一段"

    # The project summary stays in sync with the saved document title.
    listed = (await client.get("/api/projects")).json()
    entry = next(project for project in listed["projects"] if project["id"] == created["id"])
    assert entry["title"] == "独立项目"


async def test_cascade_delete_unbinds_across_all_projects(client) -> None:
    database = client.director_app.state.database
    first = asset("first.png", "image")

    default_document = default_timeline_draft().model_dump(mode="json")
    default_document["segments"][0]["first_image"] = first
    await save_timeline_document(client, default_document)

    created = (await client.post("/api/projects", json={"title": "级联测试"})).json()
    second_document = default_timeline_draft().model_dump(mode="json")
    second_document["title"] = "级联测试"
    second_document["segments"][0]["first_image"] = first
    await save_timeline_document(
        client,
        second_document,
        project_id=created["id"],
        legacy_settings=default_settings().model_dump(mode="json"),
    )

    response = await client.delete(f"/api/assets/{first['id']}?cascade=true")
    assert response.status_code == 200
    usages = response.json()["unbound_usages"]
    assert any(usage.startswith("timeline.") for usage in usages)
    assert any(
        usage.startswith(f"project.{created['id']}.") for usage in usages
    )

    default_after = (await client.get("/api/timeline")).json()
    assert default_after["segments"][0]["first_image"] is None
    second_after = (await client.get(f"/api/projects/{created['id']}/timeline")).json()
    assert second_after["segments"][0]["first_image"] is None


async def test_segment_take_lookup_scoped_by_project(client) -> None:
    database = client.director_app.state.database
    created_a = (await client.post("/api/projects", json={"title": "A"})).json()
    created_b = (await client.post("/api/projects", json={"title": "B"})).json()

    document = default_timeline_draft().model_dump(mode="json")
    document["segments"][0]["id"] = "shared-segment"
    for project_id in (created_a["id"], created_b["id"]):
        saved = await save_timeline_document(
            client,
            document,
            project_id=project_id,
            legacy_settings=default_settings().model_dump(mode="json"),
        )
        assert saved.status_code == 200, saved.text

    timeline = UnifiedTimelineDraft.model_validate(document)
    fingerprint = timeline_segment_take_fingerprint(timeline, timeline.segments[0])

    job_id = str(uuid.uuid4())
    child_id = str(uuid.uuid4())
    settings = default_settings()
    now = "2026-08-12T00:00:00+00:00"
    database.create_job(
        {
            "id": job_id,
            "mode": "timeline",
            "status": "succeeded",
            "progress": 1.0,
            "stage": "completed",
            "prompt_id": None,
            "project_id": created_a["id"],
            "outputs": [],
            "error": None,
            "config_snapshot": {"timeline": document, "segment_ids": ["shared-segment"]},
            "settings_snapshot": settings.model_dump(mode="json"),
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": now,
            "completed_at": now,
        }
    )
    database.create_job_child(
        {
            "id": child_id,
            "job_id": job_id,
            "group_index": 1,
            "family": "fl2va",
            "backend": "standard",
            "segment_ids": ["shared-segment"],
            "output_nodes": {"shared-segment": "save"},
            "status": "succeeded",
            "progress": 1.0,
            "stage": "completed",
            "prompt_id": child_id,
            "outputs": [
                {
                    "node_id": "save",
                    "filename": "shared.mp4",
                    "subfolder": "segments",
                    "type": "output",
                }
            ],
            "error": None,
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": now,
            "completed_at": now,
        }
    )

    assert database.find_latest_segment_take(
        "shared-segment", fingerprint, project_id=created_a["id"]
    ) is not None
    assert database.find_latest_segment_take(
        "shared-segment", fingerprint, project_id=created_b["id"]
    ) is None


async def test_import_project_preserves_document_verbatim(client) -> None:
    document = (await client.get("/api/timeline")).json()
    document["title"] = "历史来源"
    document["segments"][0]["id"] = "historical-segment"
    document["segments"][0]["prompt"] = "从历史任务恢复"

    preflight = await client.post(
        "/api/projects/import/preflight",
        json={"title": "历史来源", "document": document},
    )
    assert preflight.status_code == 200, preflight.text
    proposal = preflight.json()
    assert proposal["status"] == "ready"
    committed = await client.post(
        "/api/projects/import/commit",
        json={
            "commit_token": proposal["commit_token"],
            "input_digest": proposal["input_digest"],
        },
    )
    assert committed.status_code == 200, committed.text
    imported = committed.json()
    assert imported["title"] == "历史来源"
    assert imported["segment_count"] == 1

    fetched = (await client.get(f"/api/projects/{imported['id']}/timeline")).json()
    assert fetched["segments"][0]["id"] == "historical-segment"
    assert fetched["segments"][0]["prompt"] == "从历史任务恢复"

    listed = (await client.get("/api/projects")).json()
    assert any(project["id"] == imported["id"] for project in listed["projects"])
