from __future__ import annotations

import json
from pathlib import Path

import pytest

from directordeck.database import (
    Database,
    TimelineTemplateBundleConflict,
)
from directordeck.migrations.feature_bundle_v5_v6 import (
    migrate_feature_bundle_v5_authorities_to_v6,
)
from directordeck.schemas import (
    FeatureConfiguration,
    RuntimeSettingsV3,
    UnifiedTimelineDraftV5,
    default_model_stack,
    default_runtime_settings_v3,
    default_timeline_draft_v5,
    default_timeline_draft_v6,
)
from directordeck.workflow.v6_projection import project_v5_authority_to_v6


def _bundle5(*, attention: str | None = None) -> UnifiedTimelineDraftV5:
    document = default_timeline_draft_v5(default_model_stack()).model_dump(
        mode="json"
    )
    if attention is not None:
        document["features"]["project"]["attention_backend_override"] = {
            "enabled": True,
            "params": {"mode": attention},
        }
    return UnifiedTimelineDraftV5.model_validate(document)


def _write_project(
    database: Database,
    project_id: str,
    raw_document: str,
    revision: int,
) -> None:
    with database.connect() as db:
        db.execute(
            "UPDATE projects SET document = ?, revision = ? WHERE id = ?",
            (raw_document, revision, project_id),
        )


def test_fresh_default_and_created_projects_use_bundle6(tmp_path: Path) -> None:
    database = Database(tmp_path / "fresh-v6.sqlite3")

    database.initialize()

    default, revision = database.get_timeline_authority()
    created = database.create_project("Bundle 6")
    assert revision == 0
    assert default.features.template_bundle_version == 6
    assert default.features.project["comfy_kitchen_attention"].enabled is False
    assert default.features.project["comfy_kitchen_attention"].params == {}
    assert created["document"]["features"]["template_bundle_version"] == 6


def test_each_migrated_authority_advances_once_and_is_idempotent(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "authority-cas.sqlite3")
    database.initialize()
    first = database.create_project("first")["id"]
    second = database.create_project("second")["id"]
    _write_project(
        database,
        first,
        _bundle5(attention="ck_int8").model_dump_json(),
        7,
    )
    _write_project(database, second, _bundle5().model_dump_json(), 11)

    with database.connect() as db:
        outcomes = migrate_feature_bundle_v5_authorities_to_v6(
            db,
            created_at="2026-08-24T00:00:00+00:00",
        )

    by_id = {outcome.project_id: outcome for outcome in outcomes}
    assert by_id[first].status == "migrated"
    assert by_id[first].new_revision == 8
    assert by_id[first].source_digest != by_id[first].destination_digest
    assert by_id[second].new_revision == 12
    assert database.get_project_timeline_authority(first)[1] == 8
    assert database.get_project_timeline_authority(second)[1] == 12
    assert database.list_feature_bundle_migration_notices(first) == [
        "Standard CK now uses ComfyUI's official ModelAttentionBackend carrier."
    ]
    assert database.list_feature_bundle_migration_notices(second) == []

    with database.connect() as db:
        repeated = migrate_feature_bundle_v5_authorities_to_v6(
            db,
            created_at="2026-08-24T00:01:00+00:00",
        )
    repeated_by_id = {outcome.project_id: outcome for outcome in repeated}
    assert repeated_by_id[first].status == "already_v6"
    assert database.get_project_timeline_authority(first)[1] == 8


def test_conflicting_project_stays_exact_v5_while_other_project_migrates(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "local-conflict.sqlite3")
    database.initialize()
    conflict_id = database.create_project("conflict")["id"]
    safe_id = database.create_project("safe")["id"]
    conflict_raw = json.dumps(
        _bundle5(attention="pytorch").model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )
    _write_project(database, conflict_id, conflict_raw, 3)
    _write_project(database, safe_id, _bundle5().model_dump_json(), 5)

    with database.connect() as db:
        outcomes = migrate_feature_bundle_v5_authorities_to_v6(
            db,
            created_at="2026-08-24T00:00:00+00:00",
        )

    by_id = {outcome.project_id: outcome for outcome in outcomes}
    assert by_id[conflict_id].status == "retained_v5"
    assert by_id[conflict_id].reason_code == "attention_migration_conflict"
    assert by_id[safe_id].status == "migrated"
    with database.connect() as db:
        conflict = db.execute(
            "SELECT document, revision FROM projects WHERE id = ?", (conflict_id,)
        ).fetchone()
    assert str(conflict["document"]) == conflict_raw
    assert int(conflict["revision"]) == 3
    assert database.get_project_timeline_authority(safe_id)[0].features.template_bundle_version == 6
    assert database.list_feature_bundle_migration_notices(conflict_id)


def test_conflicting_default_is_backed_up_exactly_before_new_v6_default(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "default-conflict.sqlite3")
    database.initialize()
    raw = json.dumps(
        _bundle5(attention="pytorch").model_dump(mode="json"),
        ensure_ascii=False,
        separators=(", ", ": "),
    )
    with database.connect() as db:
        db.execute(
            "UPDATE unified_timeline SET document = ?, revision = 9 "
            "WHERE singleton = 1",
            (raw,),
        )

    database.initialize()

    current, revision = database.get_timeline_authority()
    assert current.features.template_bundle_version == 6
    assert revision == 10
    with database.connect() as db:
        backups = db.execute(
            "SELECT id, document FROM projects "
            "WHERE id LIKE 'bundle5-default-%'"
        ).fetchall()
    assert len(backups) == 1
    assert str(backups[0]["document"]) == raw
    backup_id = str(backups[0]["id"])
    assert backup_id in {project["id"] for project in database.list_projects()}
    assert database.list_feature_bundle_migration_notices(backup_id)

    database.initialize()
    with database.connect() as db:
        assert db.execute(
            "SELECT COUNT(*) FROM projects WHERE id LIKE 'bundle5-default-%'"
        ).fetchone()[0] == 1
    assert database.get_timeline_authority()[1] == 10


def test_damaged_project_is_preserved_and_does_not_block_startup(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "damaged-project.sqlite3")
    database.initialize()
    damaged_id = database.create_project("damaged")["id"]
    safe_id = database.create_project("safe")["id"]
    _write_project(database, damaged_id, "{not-json", 4)
    _write_project(database, safe_id, _bundle5().model_dump_json(), 6)

    database.initialize()

    with database.connect() as db:
        damaged = db.execute(
            "SELECT document, revision FROM projects WHERE id = ?", (damaged_id,)
        ).fetchone()
    assert str(damaged["document"]) == "{not-json"
    assert int(damaged["revision"]) == 4
    assert database.get_project_timeline_authority(safe_id)[0].features.template_bundle_version == 6
    assert database.list_feature_bundle_migration_notices(damaged_id)


async def test_damaged_project_remains_listable_exportable_and_deletable(
    client,
) -> None:
    database = client.director_app.state.database
    damaged_id = database.create_project("damaged")["id"]
    safe_id = database.create_project("safe")["id"]
    raw = "{not-json"
    _write_project(database, damaged_id, raw, 4)
    with database.connect() as db:
        migrate_feature_bundle_v5_authorities_to_v6(
            db,
            created_at="2026-08-24T00:00:00+00:00",
        )

    listed = await client.get("/api/projects")
    summary = await client.get(f"/api/projects/{damaged_id}")
    exported = await client.get(f"/api/projects/{damaged_id}/export")
    timeline = await client.get(f"/api/projects/{damaged_id}/timeline/authority")
    safe = await client.get(f"/api/projects/{safe_id}/timeline/authority")

    assert listed.status_code == 200, listed.text
    damaged_summary = next(
        project
        for project in listed.json()["projects"]
        if project["id"] == damaged_id
    )
    assert damaged_summary["segment_count"] == 0
    assert summary.status_code == 200, summary.text
    assert summary.json()["segment_count"] == 0
    assert exported.status_code == 200, exported.text
    assert exported.content == raw.encode("utf-8")
    assert exported.headers["content-type"] == "application/octet-stream"
    assert "attachment;" in exported.headers["content-disposition"]
    assert timeline.status_code == 409
    assert timeline.json()["detail"]["code"] == "project_document_unreadable"
    assert safe.status_code == 200, safe.text

    deleted = await client.delete(f"/api/projects/{damaged_id}")
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["deleted_project_id"] == damaged_id
    assert database.get_project_raw_document(damaged_id) is None


def test_one_transaction_failure_rolls_back_only_its_project(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "local-transaction.sqlite3")
    database.initialize()
    blocked_id = database.create_project("blocked")["id"]
    safe_id = database.create_project("safe")["id"]
    _write_project(database, blocked_id, _bundle5().model_dump_json(), 2)
    _write_project(database, safe_id, _bundle5().model_dump_json(), 4)
    with database.connect() as db:
        db.execute(
            f"""
            CREATE TRIGGER block_one_bundle_migration
            BEFORE UPDATE OF document ON projects
            WHEN OLD.id = '{blocked_id}'
            BEGIN
                SELECT RAISE(ABORT, 'blocked for test');
            END
            """
        )

    with database.connect() as db:
        outcomes = migrate_feature_bundle_v5_authorities_to_v6(
            db,
            created_at="2026-08-24T00:00:00+00:00",
        )

    by_id = {outcome.project_id: outcome for outcome in outcomes}
    assert by_id[blocked_id].reason_code == "migration_transaction_failed"
    assert database.get_project_timeline_authority(blocked_id)[0].features.template_bundle_version == 5
    assert database.get_project_timeline_authority(blocked_id)[1] == 2
    assert database.get_project_timeline_authority(safe_id)[0].features.template_bundle_version == 6
    assert database.list_feature_bundle_migration_notices(blocked_id)


def test_bundle_cas_rejects_downgrade_and_allows_explicit_upgrade(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "bundle-cas.sqlite3")
    database.initialize()
    current, revision = database.get_timeline_authority()

    with pytest.raises(TimelineTemplateBundleConflict):
        database.validate_and_put_timeline_authority(
            _bundle5(),
            expected_revision=revision,
        )
    assert database.get_timeline_authority() == (current, revision)

    source = _bundle5()
    with database.connect() as db:
        db.execute(
            "UPDATE unified_timeline SET document = ?, revision = 7 "
            "WHERE singleton = 1",
            (source.model_dump_json(),),
        )
    upgraded = project_v5_authority_to_v6(source).draft
    saved, saved_revision = database.validate_and_put_timeline_authority(
        upgraded,
        expected_revision=7,
    )
    assert saved.features.template_bundle_version == 6
    assert saved_revision == 8


async def test_bundle_downgrade_put_returns_refreshable_conflict(client) -> None:
    authority = (await client.get("/api/timeline/authority")).json()

    response = await client.put(
        "/api/timeline/authority",
        json={
            "document": _bundle5().model_dump(mode="json"),
            "expected_revision": authority["revision"],
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "timeline_template_bundle_conflict",
        "message": (
            "The project workflow bundle changed on the server; "
            "fetch the current timeline authority before retrying."
        ),
        "project_id": "default",
        "submitted_template_bundle": 5,
        "current_template_bundle": 6,
    }


async def test_feature_bundle_migration_notice_api_is_project_scoped(client) -> None:
    database = client.director_app.state.database
    carrier_id = database.create_project("carrier change")["id"]
    quiet_id = database.create_project("quiet migration")["id"]
    _write_project(
        database,
        carrier_id,
        _bundle5(attention="ck_int8").model_dump_json(),
        3,
    )
    _write_project(database, quiet_id, _bundle5().model_dump_json(), 4)
    with database.connect() as db:
        migrate_feature_bundle_v5_authorities_to_v6(
            db,
            created_at="2026-08-24T00:00:00+00:00",
        )

    carrier = await client.get(
        f"/api/projects/{carrier_id}/migration-notices"
    )
    quiet = await client.get(f"/api/projects/{quiet_id}/migration-notices")
    default = await client.get("/api/timeline/migration-notices")
    missing = await client.get("/api/projects/missing/migration-notices")

    assert carrier.status_code == 200, carrier.text
    assert carrier.json() == {
        "notices": [
            "Standard CK now uses ComfyUI's official ModelAttentionBackend carrier."
        ]
    }
    assert quiet.json() == {"notices": []}
    assert default.json() == {"notices": []}
    assert missing.status_code == 404


def test_project_compiler_dispatches_exact_bundle_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from directordeck.workflow import project_compiler

    settings: RuntimeSettingsV3 = default_runtime_settings_v3()
    v5 = _bundle5()
    v6 = default_timeline_draft_v6(default_model_stack())
    calls: list[int] = []

    def compile_v5(*_args: object, **_kwargs: object) -> str:
        calls.append(5)
        return "v5"

    def compile_v6(*_args: object, **_kwargs: object) -> str:
        calls.append(6)
        return "v6"

    monkeypatch.setattr(project_compiler, "compile_v5_execution_plan", compile_v5)
    monkeypatch.setattr(project_compiler, "compile_v6_execution_plan", compile_v6)
    assert project_compiler.compile_project_execution_plan(
        v5, settings, "job-v5"
    ) == "v5"
    assert project_compiler.compile_project_execution_plan(
        v6, settings, "job-v6"
    ) == "v6"
    assert calls == [5, 6]

    unsupported = v6.model_copy(
        update={
            "features": FeatureConfiguration(
                template_bundle_version=7,
                project=v6.features.project,
                by_segment={},
            )
        },
        deep=True,
    )
    with pytest.raises(project_compiler.ProjectCompilerBundleError):
        project_compiler.compile_project_execution_plan(
            unsupported, settings, "unsupported"
        )
