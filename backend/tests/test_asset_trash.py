from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import director.database as database_module
from director.database import Database
from director.schemas import default_settings, default_timeline_draft

from .conftest import asset, runnable_draft
from .test_assets_and_jobs import media_bytes


def timeline_with_anchors(*, first: bool = True, last: bool = True) -> dict[str, Any]:
    document = default_timeline_draft().model_dump(mode="json")
    segment = document["segments"][0]
    segment["id"] = "trash-segment"
    segment["first_image"] = asset("first.png", "image") if first else None
    segment["last_image"] = asset("last.png", "image") if last else None
    return document


async def test_asset_read_envelopes_expose_exact_database_and_origin_scope(
    client,
) -> None:
    expected_scope = {
        "active_database_identity": (
            client.director_app.state.storage.active_database_identity
        ),
        "comfy_origin": "http://comfy.test:8188",
    }

    assets_response = await client.get("/api/assets?kind=image")
    assert assets_response.status_code == 200, assets_response.text
    assets_payload = assets_response.json()
    assert set(assets_payload) == {
        "assets",
        "outputs_preserved",
        "active_database_identity",
        "comfy_origin",
    }
    assert {
        key: assets_payload[key]
        for key in ("outputs_preserved", "active_database_identity", "comfy_origin")
    } == {"outputs_preserved": True, **expected_scope}
    assert assets_payload["assets"]

    trash_response = await client.get("/api/asset-trash")
    assert trash_response.status_code == 200, trash_response.text
    assert trash_response.json() == {
        "batches": [],
        "remote_files_preserved": True,
        **expected_scope,
    }


async def test_multi_asset_cascade_and_exact_inverse_restore_advance_each_revision_once(
    client,
) -> None:
    first = asset("first.png", "image")
    last = asset("last.png", "image")
    default_saved = await client.put("/api/timeline", json=timeline_with_anchors())
    assert default_saved.status_code == 200, default_saved.text

    created = (await client.post("/api/projects", json={"title": "回收站项目"})).json()
    project_document = timeline_with_anchors()
    project_document["title"] = "回收站项目"
    project_document["segments"][0]["id"] = "project-trash-segment"
    project_saved = await client.put(
        f"/api/projects/{created['id']}/timeline", json=project_document
    )
    assert project_saved.status_code == 200, project_saved.text

    draft_saved = await client.put("/api/drafts/i2v", json=runnable_draft("i2v"))
    assert draft_saved.status_code == 200, draft_saved.text
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
    refused_reference = await client.put("/api/timeline", json=default_before["document"])
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
    assert (await client.get("/api/drafts/i2v")).json() == draft_saved.json()
    assert (await client.get("/api/asset-trash")).json()["batches"] == []


async def test_batch_without_cascade_refuses_all_assets_without_partial_tombstones(
    client,
) -> None:
    first = asset("first.png", "image")
    last = asset("last.png", "image")
    saved = await client.put("/api/timeline", json=timeline_with_anchors())
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
    saved = await client.put(
        "/api/timeline", json=timeline_with_anchors(first=True, last=False)
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
    await client.put(
        "/api/timeline", json=timeline_with_anchors(first=True, last=False)
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
    await client.put(
        "/api/timeline", json=timeline_with_anchors(first=True, last=False)
    )
    created = (await client.post("/api/projects", json={"title": "冲突项目"})).json()
    project_document = timeline_with_anchors(first=True, last=False)
    project_document["title"] = "冲突项目"
    await client.put(
        f"/api/projects/{created['id']}/timeline", json=project_document
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
    database.validate_and_put_timeline(
        document.model_validate(raw), comfy_origin="http://comfy.test:8188"
    )
    before, before_revision = database.get_timeline_authority()
    original = database_module.validate_timeline_draft

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
        database_module, "validate_timeline_draft", fail_after_both_anchors_are_unbound
    )
    with pytest.raises(ValueError, match="forced multi-asset cascade failure"):
        database.trash_assets(
            [asset("first.png", "image")["id"], asset("last.png", "image")["id"]],
            cascade=True,
            expected_comfy_origin="http://comfy.test:8188",
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


async def test_trash_batches_are_isolated_by_active_comfy_origin(client) -> None:
    reference = asset("reference.png", "image")
    trashed = (
        await client.post(
            "/api/asset-trash",
            json={"asset_ids": [reference["id"]], "cascade": False},
        )
    ).json()
    settings = (await client.get("/api/settings")).json()
    settings["comfy_url"] = "http://other-comfy.test:8188"
    assert (await client.put("/api/settings", json=settings)).status_code == 200

    assert (await client.get("/api/asset-trash")).json()["batches"] == []
    restore = await client.post(
        f"/api/asset-trash/{trashed['batch_id']}/restore",
        json={"mode": "registration_only"},
    )
    purge = await client.delete(f"/api/asset-trash/{trashed['batch_id']}")
    assert restore.status_code == 409
    assert purge.status_code == 409
    assert restore.json()["detail"]["code"] == "asset_trash_origin_conflict"
    assert purge.json()["detail"]["remote_files_preserved"] is True

    settings["comfy_url"] = "http://comfy.test:8188"
    await client.put("/api/settings", json=settings)
    restored = await client.post(
        f"/api/asset-trash/{trashed['batch_id']}/restore",
        json={"mode": "registration_only"},
    )
    assert restored.status_code == 200


def test_asset_trash_schema_migrates_legacy_assets_as_live(tmp_path: Path) -> None:
    path = tmp_path / "legacy-trash.sqlite3"
    settings = default_settings("http://legacy-comfy.test:8188")
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
            "SELECT comfy_origin, trashed_at, trash_batch_id FROM assets WHERE id = ?",
            (legacy_asset["id"],),
        ).fetchone()
    assert {"comfy_origin", "trashed_at", "trash_batch_id"} <= columns
    assert {"asset_trash_batches", "asset_trash_document_changes"} <= tables
    assert tuple(row) == ("http://legacy-comfy.test:8188", None, None)
    listed = database.list_assets(
        comfy_origin="http://legacy-comfy.test:8188"
    )
    assert [item["id"] for item in listed] == [legacy_asset["id"]]


async def test_upload_endpoint_switch_after_remote_write_does_not_register_asset(
    client, fake_comfy, monkeypatch
) -> None:
    upload_started = asyncio.Event()
    release_upload = asyncio.Event()
    original_upload = fake_comfy.upload

    async def blocked_upload(
        filename: str,
        content: bytes | Path,
        content_type: str,
        kind: str,
    ) -> dict[str, Any]:
        upload_started.set()
        await release_upload.wait()
        return await original_upload(filename, content, content_type, kind)

    monkeypatch.setattr(fake_comfy, "upload", blocked_upload)
    request = asyncio.create_task(
        client.post(
            "/api/assets",
            data={"kind": "image"},
            files={
                "file": (
                    "origin-race.png",
                    media_bytes("origin-race.png"),
                    "image/png",
                )
            },
        )
    )
    await asyncio.wait_for(upload_started.wait(), timeout=2)
    settings = (await client.get("/api/settings")).json()
    settings["comfy_url"] = "http://switched-comfy.test:8188"
    switched = await client.put("/api/settings", json=settings)
    assert switched.status_code == 200
    release_upload.set()
    response = await asyncio.wait_for(request, timeout=2)

    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {
        "code": "asset_upload_origin_changed",
        "message": (
            "ComfyUI endpoint changed while the upload was in progress; the remote "
            "file was preserved but was not registered in Director"
        ),
        "remote_files_preserved": True,
    }
    assert any(item["filename"] == "origin-race.png" for item in fake_comfy.uploads)
    with client.director_app.state.database.connect() as connection:
        rows = connection.execute("SELECT document FROM assets").fetchall()
    assert all("origin-race.png" not in str(row["document"]) for row in rows)
