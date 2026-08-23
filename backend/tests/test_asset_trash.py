from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import directordeck.database as database_module
from directordeck.database import Database
from directordeck.migrations.timeline_v4_v5 import migrate_timeline_v4_to_v5
from directordeck.schemas import default_settings, default_timeline_draft

from .conftest import asset, runnable_draft, save_timeline_document


def timeline_with_anchors(*, first: bool = True, last: bool = True) -> dict[str, Any]:
    document = default_timeline_draft().model_dump(mode="json")
    segment = document["segments"][0]
    segment["id"] = "trash-segment"
    segment["first_image"] = asset("first.png", "image") if first else None
    segment["last_image"] = asset("last.png", "image") if last else None
    return document


async def test_asset_read_envelopes(client) -> None:
    assets_response = await client.get("/api/assets?kind=image")
    assert assets_response.status_code == 200, assets_response.text
    assets_payload = assets_response.json()
    assert set(assets_payload) == {"assets", "outputs_preserved"}
    assert assets_payload["outputs_preserved"] is True
    assert assets_payload["assets"]

    trash_response = await client.get("/api/asset-trash")
    assert trash_response.status_code == 200, trash_response.text
    assert trash_response.json() == {
        "batches": [],
        "remote_files_preserved": True,
    }


async def test_multi_asset_cascade_and_exact_inverse_restore_advance_each_revision_once(
    client,
) -> None:
    first = asset("first.png", "image")
    last = asset("last.png", "image")
    default_saved = await save_timeline_document(client, timeline_with_anchors())
    assert default_saved.status_code == 200, default_saved.text

    created = (await client.post("/api/projects", json={"title": "回收站项目"})).json()
    project_document = timeline_with_anchors()
    project_document["title"] = "回收站项目"
    project_document["segments"][0]["id"] = "project-trash-segment"
    project_saved = await save_timeline_document(
        client,
        project_document,
        project_id=created["id"],
        legacy_settings=default_settings().model_dump(mode="json"),
    )
    assert project_saved.status_code == 200, project_saved.text

    retired = await client.put("/api/drafts/i2v", json=runnable_draft("i2v"))
    assert retired.status_code == 410, retired.text
    draft_saved = client.director_app.state.database.validate_and_put_draft(
        "i2v",
        database_module.validate_mode_draft("i2v", runnable_draft("i2v")),
    )
    default_before = (await client.get("/api/timeline/authority")).json()
    project_before = (
        await client.get(f"/api/projects/{created['id']}/timeline/authority")
    ).json()

    trashed = await client.post(
        "/api/asset-trash",
        json={"asset_ids": [first["id"], last["id"]], "cascade": True},
    )

    assert trashed.status_code == 200, trashed.text
    batch = trashed.json()
    assert batch["asset_ids"] == [first["id"], last["id"]]
    assert batch["remote_files_preserved"] is True
    assert set(batch["unbound_usages_by_asset"]) == {first["id"], last["id"]}
    assert len(batch["assets"]) == 2
    after_default = (await client.get("/api/timeline/authority")).json()
    after_project = (
        await client.get(f"/api/projects/{created['id']}/timeline/authority")
    ).json()
    assert after_default["revision"] == default_before["revision"] + 1
    assert after_project["revision"] == project_before["revision"] + 1
    assert after_default["document"]["segments"][0]["first_image"] is None
    assert after_default["document"]["segments"][0]["last_image"] is None
    assert after_project["document"]["segments"][0]["first_image"] is None
    assert after_project["document"]["segments"][0]["last_image"] is None

    listed_ids = {
        item["id"] for item in (await client.get("/api/assets")).json()["assets"]
    }
    assert first["id"] not in listed_ids and last["id"] not in listed_ids
    assert (await client.get(f"/api/assets/{first['id']}/preview")).status_code == 404
    refused_reference = await client.put(
        "/api/timeline/authority",
        json={
            "document": default_before["document"],
            "expected_revision": after_default["revision"],
        },
    )
    assert refused_reference.status_code == 422
    trash_list = (await client.get("/api/asset-trash")).json()
    assert [item["batch_id"] for item in trash_list["batches"]] == [
        batch["batch_id"]
    ]
    assert trash_list["remote_files_preserved"] is True

    restored = await client.post(
        f"/api/asset-trash/{batch['batch_id']}/restore",
        json={"mode": "with_references"},
    )

    assert restored.status_code == 200, restored.text
    assert restored.json() == {
        "batch_id": batch["batch_id"],
        "restored_asset_ids": [first["id"], last["id"]],
        "restored_references": True,
        "mode": "with_references",
        "remote_files_preserved": True,
    }
    restored_default = (await client.get("/api/timeline/authority")).json()
    restored_project = (
        await client.get(f"/api/projects/{created['id']}/timeline/authority")
    ).json()
    assert restored_default["document"] == default_before["document"]
    assert restored_project["document"] == project_before["document"]
    assert restored_default["revision"] == after_default["revision"] + 1
    assert restored_project["revision"] == after_project["revision"] + 1
    assert (await client.get("/api/drafts/i2v")).json() == draft_saved.model_dump(
        mode="json"
    )
    assert (await client.get("/api/asset-trash")).json()["batches"] == []


async def test_batch_without_cascade_refuses_all_assets_without_partial_tombstones(
    client,
) -> None:
    first = asset("first.png", "image")
    last = asset("last.png", "image")
    saved = await save_timeline_document(client, timeline_with_anchors())
    assert saved.status_code == 200
    before = (await client.get("/api/timeline/authority")).json()

    refused = await client.post(
        "/api/asset-trash",
        json={"asset_ids": [first["id"], last["id"]], "cascade": False},
    )

    assert refused.status_code == 409
    detail = refused.json()["detail"]
    assert detail["code"] == "assets_in_use"
    assert detail["remote_files_preserved"] is True
    assert detail["usages_by_asset"][first["id"]]
    assert detail["usages_by_asset"][last["id"]]
    database = client.director_app.state.database
    assert database.get_asset(first["id"]) is not None
    assert database.get_asset(last["id"]) is not None
    assert (await client.get("/api/timeline/authority")).json() == before
    assert (await client.get("/api/asset-trash")).json()["batches"] == []


async def test_restore_conflict_is_atomic_and_registration_only_is_safe_fallback(
    client,
) -> None:
    first = asset("first.png", "image")
    saved = await save_timeline_document(
        client, timeline_with_anchors(first=True, last=False)
    )
    assert saved.status_code == 200
    trashed = await client.post(
        "/api/asset-trash",
        json={"asset_ids": [first["id"]], "cascade": True},
    )
    assert trashed.status_code == 200
    batch_id = trashed.json()["batch_id"]
    after_trash = (await client.get("/api/timeline/authority")).json()
    edited = json.loads(json.dumps(after_trash["document"]))
    edited["segments"][0]["prompt"] = "edited after trash"
    changed = await client.put(
        "/api/timeline/authority",
        json={
            "document": edited,
            "expected_revision": after_trash["revision"],
        },
    )
    assert changed.status_code == 200

    refused = await client.post(
        f"/api/asset-trash/{batch_id}/restore",
        json={"mode": "with_references"},
    )

    assert refused.status_code == 409
    detail = refused.json()["detail"]
    assert detail["code"] == "asset_trash_restore_conflict"
    assert any(
        item["owner_kind"] == "timeline"
        and "revision_changed" in item["reason"]
        for item in detail["conflicts"]
    )
    database = client.director_app.state.database
    assert database.get_asset(first["id"]) is None
    assert database.get_asset_record(first["id"], include_trashed=True) is not None
    assert (await client.get("/api/timeline/authority")).json() == changed.json()

    fallback = await client.post(
        f"/api/asset-trash/{batch_id}/restore",
        json={"mode": "registration_only"},
    )
    assert fallback.status_code == 200, fallback.text
    assert fallback.json()["restored_references"] is False
    assert fallback.json()["remote_files_preserved"] is True
    assert database.get_asset(first["id"]) is not None
    current = (await client.get("/api/timeline/authority")).json()
    assert current == changed.json()
    assert current["document"]["segments"][0]["first_image"] is None


async def test_restore_checks_exact_document_digest_even_if_revision_is_unchanged(
    client,
) -> None:
    first = asset("first.png", "image")
    await save_timeline_document(
        client, timeline_with_anchors(first=True, last=False)
    )
    trashed = (
        await client.post(
            "/api/asset-trash",
            json={"asset_ids": [first["id"]], "cascade": True},
        )
    ).json()
    authority = (await client.get("/api/timeline/authority")).json()
    tampered = json.loads(json.dumps(authority["document"]))
    tampered["segments"][0]["prompt"] = "out-of-band same-revision edit"
    with client.director_app.state.database.connect() as connection:
        connection.execute(
            "UPDATE unified_timeline SET document = ? WHERE singleton = 1",
            (json.dumps(tampered, ensure_ascii=False),),
        )

    refused = await client.post(
        f"/api/asset-trash/{trashed['batch_id']}/restore",
        json={"mode": "with_references"},
    )

    assert refused.status_code == 409
    conflict = refused.json()["detail"]["conflicts"][0]
    assert "document_changed" in conflict["reason"]
    assert "revision_changed" not in conflict["reason"]
    assert client.director_app.state.database.get_asset(first["id"]) is None


async def test_conflict_in_one_project_prevents_partial_restore_everywhere(
    client,
) -> None:
    first = asset("first.png", "image")
    await save_timeline_document(
        client, timeline_with_anchors(first=True, last=False)
    )
    created = (await client.post("/api/projects", json={"title": "冲突项目"})).json()
    project_document = timeline_with_anchors(first=True, last=False)
    project_document["title"] = "冲突项目"
    await save_timeline_document(
        client,
        project_document,
        project_id=created["id"],
        legacy_settings=default_settings().model_dump(mode="json"),
    )
    trashed = (
        await client.post(
            "/api/asset-trash",
            json={"asset_ids": [first["id"]], "cascade": True},
        )
    ).json()
    default_after = (await client.get("/api/timeline/authority")).json()
    project_after = (
        await client.get(f"/api/projects/{created['id']}/timeline/authority")
    ).json()
    changed_project = json.loads(json.dumps(project_after["document"]))
    changed_project["segments"][0]["prompt"] = "project changed"
    updated = await client.put(
        f"/api/projects/{created['id']}/timeline/authority",
        json={
            "document": changed_project,
            "expected_revision": project_after["revision"],
        },
    )
    assert updated.status_code == 200

    refused = await client.post(
        f"/api/asset-trash/{trashed['batch_id']}/restore",
        json={"mode": "with_references"},
    )

    assert refused.status_code == 409
    assert (await client.get("/api/timeline/authority")).json() == default_after
    assert (
        await client.get(f"/api/projects/{created['id']}/timeline/authority")
    ).json() == updated.json()
    assert client.director_app.state.database.get_asset(first["id"]) is None


def test_multi_asset_cascade_validation_failure_rolls_back_batch_and_documents(
    client, monkeypatch
) -> None:
    database = client.director_app.state.database
    document = default_timeline_draft()
    raw = timeline_with_anchors()
    current_revision = database.get_timeline_authority()[1]
    database.validate_and_put_timeline_authority(
        migrate_timeline_v4_to_v5(
            document.model_validate(raw),
            default_settings(),
        ),
        expected_revision=current_revision,
    )
    before, before_revision = database.get_timeline_authority()
    original = database_module.validate_timeline_draft_v5

    def fail_after_both_anchors_are_unbound(value: Any):
        if isinstance(value, dict):
            segments = value.get("segments")
            if (
                isinstance(segments, list)
                and segments
                and segments[0].get("first_image") is None
                and segments[0].get("last_image") is None
            ):
                raise ValueError("forced multi-asset cascade failure")
        return original(value)

    monkeypatch.setattr(
        database_module,
        "validate_timeline_draft_v5",
        fail_after_both_anchors_are_unbound,
    )
    with pytest.raises(ValueError, match="forced multi-asset cascade failure"):
        database.trash_assets(
            [asset("first.png", "image")["id"], asset("last.png", "image")["id"]],
            cascade=True,
        )

    after, after_revision = database.get_timeline_authority()
    assert after == before
    assert after_revision == before_revision
    assert database.get_asset(asset("first.png", "image")["id"]) is not None
    assert database.get_asset(asset("last.png", "image")["id"]) is not None
    with database.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM asset_trash_batches"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM asset_trash_document_changes"
        ).fetchone()[0] == 0


async def test_purge_removes_only_director_registration_and_recovery_bundle(
    client, fake_comfy
) -> None:
    reference = asset("reference.png", "image")
    trashed = await client.post(
        "/api/asset-trash",
        json={"asset_ids": [reference["id"]], "cascade": False},
    )
    assert trashed.status_code == 200, trashed.text
    batch_id = trashed.json()["batch_id"]
    uploads_before = list(fake_comfy.uploads)

    purged = await client.delete(f"/api/asset-trash/{batch_id}")

    assert purged.status_code == 200, purged.text
    assert purged.json() == {
        "batch_id": batch_id,
        "purged_asset_ids": [reference["id"]],
        "remote_files_preserved": True,
    }
    assert fake_comfy.uploads == uploads_before
    database = client.director_app.state.database
    assert database.get_asset_record(reference["id"], include_trashed=True) is None
    with database.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM asset_trash_batches WHERE id = ?", (batch_id,)
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM asset_trash_document_changes WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()[0] == 0
    assert (await client.delete(f"/api/asset-trash/{batch_id}")).status_code == 404


def test_asset_trash_schema_migrates_legacy_assets_as_live(tmp_path: Path) -> None:
    path = tmp_path / "legacy-trash.sqlite3"
    settings = default_settings()
    legacy_asset = asset("legacy.png", "image")
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE settings(
                singleton INTEGER PRIMARY KEY,
                document TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE assets(
                id TEXT PRIMARY KEY,
                document TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO settings VALUES(1, ?, 'then')",
            (settings.model_dump_json(),),
        )
        connection.execute(
            "INSERT INTO assets VALUES(?, ?, 'then')",
            (legacy_asset["id"], json.dumps(legacy_asset)),
        )

    database = Database(path)
    database.initialize()

    with database.connect() as connection:
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(assets)")
        }
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        row = connection.execute(
            "SELECT trashed_at, trash_batch_id FROM assets WHERE id = ?",
            (legacy_asset["id"],),
        ).fetchone()
    assert {"trashed_at", "trash_batch_id"} <= columns
    assert {"asset_trash_batches", "asset_trash_document_changes"} <= tables
    assert tuple(row) == (None, None)
    listed = database.list_assets()
    assert [item["id"] for item in listed] == [legacy_asset["id"]]
