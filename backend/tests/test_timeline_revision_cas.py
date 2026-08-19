from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from directordeck.database import Database, TimelineRevisionConflict
from directordeck.schemas import (
    MAX_TIMELINE_REVISION,
    UnifiedTimelineDraft,
    default_settings,
    default_timeline_draft,
)

from .conftest import asset


def _titled_timeline(title: str) -> UnifiedTimelineDraft:
    return default_timeline_draft().model_copy(update={"title": title}, deep=True)


async def test_default_authority_cas_success_and_stale_conflict_are_exact(
    client,
) -> None:
    initial = await client.get("/api/timeline/authority")
    assert initial.status_code == 200, initial.text
    assert initial.json()["revision"] == 0

    first = initial.json()["document"]
    first["title"] = "CAS 胜者"
    saved = await client.put(
        "/api/timeline/authority",
        json={"document": first, "expected_revision": 0},
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["revision"] == 1
    assert saved.json()["document"]["title"] == "CAS 胜者"

    stale = initial.json()["document"]
    stale["title"] = "过期写入"
    refused = await client.put(
        "/api/timeline/authority",
        json={"document": stale, "expected_revision": 0},
    )
    assert refused.status_code == 409, refused.text
    assert refused.json() == {
        "detail": {
            "code": "timeline_revision_conflict",
            "message": (
                "timeline changed on the server; fetch the current authority "
                "before retrying"
            ),
            "project_id": "default",
            "expected_revision": 0,
            "actual_revision": 1,
        }
    }

    current = (await client.get("/api/timeline/authority")).json()
    assert current["revision"] == 1
    assert current["document"]["title"] == "CAS 胜者"
    # The additive authority API does not change the legacy raw response.
    assert (await client.get("/api/timeline")).json() == current["document"]


async def test_authority_request_schema_is_strict_and_revision_is_json_safe(
    client,
) -> None:
    document = (await client.get("/api/timeline")).json()

    missing = await client.put(
        "/api/timeline/authority",
        json={"document": document},
    )
    extra = await client.put(
        "/api/timeline/authority",
        json={"document": document, "expected_revision": 0, "extra": True},
    )
    unsafe = await client.put(
        "/api/timeline/authority",
        json={"document": document, "expected_revision": 2**53},
    )

    assert missing.status_code == 422
    assert extra.status_code == 422
    assert unsafe.status_code == 422
    assert (await client.get("/api/timeline/authority")).json()["revision"] == 0


async def test_project_authority_is_isolated_and_default_alias_delegates(
    client,
) -> None:
    created = (await client.post("/api/projects", json={"title": "项目 CAS"})).json()
    project_id = created["id"]
    initial = (
        await client.get(f"/api/projects/{project_id}/timeline/authority")
    ).json()
    assert initial["revision"] == 0

    document = initial["document"]
    document["segments"][0]["prompt"] = "项目独立 revision"
    saved = await client.put(
        f"/api/projects/{project_id}/timeline/authority",
        json={"document": document, "expected_revision": 0},
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["revision"] == 1

    legacy_document = saved.json()["document"]
    legacy_document["title"] = "项目裸 PUT"
    legacy = await client.put(
        f"/api/projects/{project_id}/timeline", json=legacy_document
    )
    assert legacy.status_code == 200, legacy.text
    assert (
        await client.get(f"/api/projects/{project_id}/timeline/authority")
    ).json()["revision"] == 2
    stale = await client.put(
        f"/api/projects/{project_id}/timeline/authority",
        json={"document": document, "expected_revision": 1},
    )
    assert stale.status_code == 409
    assert stale.json()["detail"] == {
        "code": "timeline_revision_conflict",
        "message": (
            "timeline changed on the server; fetch the current authority "
            "before retrying"
        ),
        "project_id": project_id,
        "expected_revision": 1,
        "actual_revision": 2,
    }

    assert (await client.get("/api/timeline/authority")).json()["revision"] == 0
    default_alias = (
        await client.get("/api/projects/default/timeline/authority")
    ).json()
    assert default_alias == (await client.get("/api/timeline/authority")).json()

    missing = await client.get("/api/projects/missing/timeline/authority")
    assert missing.status_code == 404
    missing_put = await client.put(
        "/api/projects/missing/timeline/authority",
        json={"document": document, "expected_revision": 0},
    )
    assert missing_put.status_code == 404


async def test_legacy_put_and_project_rename_advance_revision(client) -> None:
    default_before = (await client.get("/api/timeline/authority")).json()
    legacy_document = default_before["document"]
    legacy_document["title"] = "裸 PUT 仍兼容"
    legacy = await client.put("/api/timeline", json=legacy_document)
    assert legacy.status_code == 200, legacy.text
    default_after = (await client.get("/api/timeline/authority")).json()
    assert default_after["revision"] == default_before["revision"] + 1

    stale = await client.put(
        "/api/timeline/authority",
        json={
            "document": default_before["document"],
            "expected_revision": default_before["revision"],
        },
    )
    assert stale.status_code == 409

    created = (await client.post("/api/projects", json={"title": "改名前"})).json()
    project_id = created["id"]
    assert (
        await client.get(f"/api/projects/{project_id}/timeline/authority")
    ).json()["revision"] == 0
    renamed = await client.patch(
        f"/api/projects/{project_id}", json={"title": "改名后"}
    )
    assert renamed.status_code == 200, renamed.text
    renamed_authority = (
        await client.get(f"/api/projects/{project_id}/timeline/authority")
    ).json()
    assert renamed_authority["revision"] == 1
    assert renamed_authority["document"]["title"] == "改名后"


async def test_revision_exhaustion_is_an_explicit_409_for_default_and_project(
    client,
) -> None:
    database = client.director_app.state.database
    created = (await client.post("/api/projects", json={"title": "耗尽项目"})).json()
    project_id = created["id"]
    with database.connect() as connection:
        connection.execute(
            "UPDATE unified_timeline SET revision = ? WHERE singleton = 1",
            (MAX_TIMELINE_REVISION,),
        )
        connection.execute(
            "UPDATE projects SET revision = ? WHERE id = ?",
            (MAX_TIMELINE_REVISION, project_id),
        )

    cases = [
        ("/api/timeline/authority", "default"),
        (f"/api/projects/{project_id}/timeline/authority", project_id),
    ]
    for endpoint, expected_project_id in cases:
        before = (await client.get(endpoint)).json()
        edited = json.loads(json.dumps(before["document"]))
        edited["segments"][0]["prompt"] = "不得越过 revision 上限"

        refused = await client.put(
            endpoint,
            json={
                "document": edited,
                "expected_revision": MAX_TIMELINE_REVISION,
            },
        )

        assert refused.status_code == 409, refused.text
        assert refused.json() == {
            "detail": {
                "code": "timeline_revision_exhausted",
                "message": (
                    "timeline revision space is exhausted; create or import a new "
                    "project before editing further"
                ),
                "project_id": expected_project_id,
                "revision": MAX_TIMELINE_REVISION,
            }
        }
        assert (await client.get(endpoint)).json() == before


async def test_asset_cascade_advances_only_changed_project_revisions(client) -> None:
    first = asset("first.png", "image")
    default_document = (await client.get("/api/timeline")).json()
    default_document["segments"][0]["first_image"] = first
    assert (await client.put("/api/timeline", json=default_document)).status_code == 200

    changed = (await client.post("/api/projects", json={"title": "受影响"})).json()
    changed_document = (
        await client.get(f"/api/projects/{changed['id']}/timeline")
    ).json()
    changed_document["segments"][0]["first_image"] = first
    assert (
        await client.put(
            f"/api/projects/{changed['id']}/timeline", json=changed_document
        )
    ).status_code == 200

    untouched = (await client.post("/api/projects", json={"title": "不受影响"})).json()
    before_default = (await client.get("/api/timeline/authority")).json()["revision"]
    before_changed = (
        await client.get(f"/api/projects/{changed['id']}/timeline/authority")
    ).json()["revision"]
    before_untouched = (
        await client.get(f"/api/projects/{untouched['id']}/timeline/authority")
    ).json()["revision"]

    deleted = await client.delete(f"/api/assets/{first['id']}?cascade=true")
    assert deleted.status_code == 200, deleted.text

    default_after = (await client.get("/api/timeline/authority")).json()
    changed_after = (
        await client.get(f"/api/projects/{changed['id']}/timeline/authority")
    ).json()
    untouched_after = (
        await client.get(f"/api/projects/{untouched['id']}/timeline/authority")
    ).json()
    assert default_after["revision"] == before_default + 1
    assert changed_after["revision"] == before_changed + 1
    assert untouched_after["revision"] == before_untouched
    assert default_after["document"]["segments"][0]["first_image"] is None
    assert changed_after["document"]["segments"][0]["first_image"] is None


def test_initialize_adds_revision_columns_without_rewriting_current_documents(
    tmp_path,
) -> None:
    path = tmp_path / "legacy-without-timeline-revisions.sqlite3"
    default = default_timeline_draft().model_dump(mode="json")
    project = {**default, "title": "旧项目"}
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE unified_timeline (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                document TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE projects (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                document TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO unified_timeline(singleton, document, updated_at) "
            "VALUES(1, ?, ?)",
            (json.dumps(default), "2026-08-16T00:00:00+00:00"),
        )
        connection.execute(
            "INSERT INTO projects(id, title, document, created_at, updated_at) "
            "VALUES(?, ?, ?, ?, ?)",
            (
                "legacy-project",
                "旧项目",
                json.dumps(project),
                "2026-08-16T00:00:00+00:00",
                "2026-08-16T00:00:00+00:00",
            ),
        )

    database = Database(path)
    database.initialize()

    assert database.get_timeline_authority()[1] == 0
    assert database.get_project_timeline_authority("legacy-project")[1] == 0
    with sqlite3.connect(path) as connection:
        timeline_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(unified_timeline)")
        }
        project_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(projects)")
        }
    assert "revision" in timeline_columns
    assert "revision" in project_columns

    # Repeated startup is idempotent when canonical storage did not change.
    database.initialize()
    assert database.get_timeline_authority()[1] == 0
    assert database.get_project_timeline_authority("legacy-project")[1] == 0


def test_startup_canonicalization_advances_revision_once(tmp_path) -> None:
    path = tmp_path / "timeline-canonicalization-revision.sqlite3"
    database = Database(path)
    database.initialize()
    legacy = default_timeline_draft().model_dump(mode="json")
    legacy["version"] = 3
    legacy["ref_image_size"] = "max"
    legacy["audio_mode"] = "mute"
    for segment in legacy["segments"]:
        segment.pop("ref_image_size")
        segment.pop("audio_mode")
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE unified_timeline SET document = ?, revision = 7 WHERE singleton = 1",
            (json.dumps(legacy),),
        )

    database.initialize()
    document, revision = database.get_timeline_authority()
    assert revision == 8
    assert document.version == 4
    assert document.segments[0].ref_image_size == "max"
    assert document.segments[0].audio_mode == "mute"

    database.initialize()
    assert database.get_timeline_authority()[1] == 8


def test_two_concurrent_cas_writers_have_exactly_one_winner(tmp_path) -> None:
    database = Database(tmp_path / "concurrent-cas.sqlite3")
    database.initialize()
    barrier = threading.Barrier(2)

    def write(title: str):
        barrier.wait(timeout=2)
        try:
            return database.validate_and_put_timeline_authority(
                _titled_timeline(title),
                expected_revision=0,
            )
        except TimelineRevisionConflict as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(write, ("并发 A", "并发 B")))

    winners = [result for result in results if isinstance(result, tuple)]
    conflicts = [
        result for result in results if isinstance(result, TimelineRevisionConflict)
    ]
    assert len(winners) == 1
    assert len(conflicts) == 1
    assert winners[0][1] == 1
    assert conflicts[0].expected_revision == 0
    assert conflicts[0].actual_revision == 1
    document, revision = database.get_timeline_authority()
    assert revision == 1
    assert document.title in {"并发 A", "并发 B"}
