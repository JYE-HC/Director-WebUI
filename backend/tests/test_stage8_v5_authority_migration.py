from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from directordeck.database import (
    Database,
    TimelineRevisionConflict,
    TimelineTemplateBundleConflict,
)
from directordeck.migration_api import (
    ProjectImportPreflightRequest,
    prepare_project_import,
)
from directordeck.migrations import (
    migrate_feature_bundle_v4_authorities_to_v5,
)
from directordeck.schemas import UnifiedTimelineDraftV5
from directordeck.workflow.effective_features import (
    migrate_timeline_feature_authority_to_v5,
)
from directordeck.workflow.v5_compat import project_v5_compile_authority

from .test_workflow_v5_compat import _v4_pair, _v5_pair


def _bundle4(document: dict) -> dict:
    detached = copy.deepcopy(document)
    detached["features"]["template_bundle_version"] = 4
    lora = detached["features"]["project"].get("lora")
    detached["features"]["project"] = ({"lora": lora} if lora is not None else {})
    detached["features"]["by_segment"] = {
        segment_id: ({"lora": selections["lora"]} if "lora" in selections else {})
        for segment_id, selections in detached["features"]["by_segment"].items()
        if "lora" in selections
    }
    return detached


def test_startup_migrates_default_and_projects_once_and_wal_replay_recovers(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "feature-authorities.sqlite3")
    database.initialize()
    created = database.create_project("bundle-four-project")
    project_id = created["id"]
    default, _default_revision = database.get_timeline_authority()
    project, _project_revision = database.get_project_timeline_authority(project_id)
    with database.connect() as db:
        db.execute(
            "UPDATE unified_timeline SET document = ?, revision = 7, "
            "updated_at = ? WHERE singleton = 1",
            (json.dumps(_bundle4(default.model_dump(mode="json"))), "before-v5"),
        )
        db.execute(
            "UPDATE projects SET document = ?, revision = 12, updated_at = ? "
            "WHERE id = ?",
            (
                json.dumps(_bundle4(project.model_dump(mode="json"))),
                "before-v5",
                project_id,
            ),
        )

    database.initialize()

    current_default, default_revision = database.get_timeline_authority()
    current_project, project_revision = database.get_project_timeline_authority(
        project_id
    )
    assert current_default.features.template_bundle_version == 6
    assert current_project.features.template_bundle_version == 6
    assert default_revision == 9
    assert project_revision == 14
    with database.connect() as db:
        before_restart = tuple(
            tuple(row)
            for row in db.execute(
                "SELECT 'default', document, revision, updated_at "
                "FROM unified_timeline WHERE singleton = 1 "
                "UNION ALL SELECT id, document, revision, updated_at "
                "FROM projects ORDER BY 1"
            ).fetchall()
        )

    database.initialize()

    with database.connect() as db:
        after_restart = tuple(
            tuple(row)
            for row in db.execute(
                "SELECT 'default', document, revision, updated_at "
                "FROM unified_timeline WHERE singleton = 1 "
                "UNION ALL SELECT id, document, revision, updated_at "
                "FROM projects ORDER BY 1"
            ).fetchall()
        )
    assert after_restart == before_restart

    # A pending Bundle-4 WAL first observes the startup CAS bump. Even if an old
    # tab retries against the latest revision, it may not downgrade Bundle 6.
    replay = _bundle4(current_default.model_dump(mode="json"))
    replay["title"] = "replayed after refresh"
    with pytest.raises(TimelineRevisionConflict) as caught:
        database.validate_and_put_timeline_authority(
            UnifiedTimelineDraftV5.model_validate(replay),
            expected_revision=7,
        )
    assert caught.value.actual_revision == 9
    with pytest.raises(TimelineTemplateBundleConflict):
        database.validate_and_put_timeline_authority(
            UnifiedTimelineDraftV5.model_validate(replay),
            expected_revision=default_revision,
        )
    assert database.get_timeline_authority() == (current_default, default_revision)


def test_feature_authority_migration_isolates_an_invalid_project(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "feature-authorities-rollback.sqlite3")
    database.initialize()
    created = database.create_project("invalid-bundle-four")
    project_id = created["id"]
    default, _ = database.get_timeline_authority()
    invalid = _bundle4(created["document"])
    invalid["features"]["project"]["future"] = {
        "enabled": False,
        "params": {},
    }
    with database.connect() as db:
        db.execute(
            "UPDATE unified_timeline SET document = ?, revision = 2 "
            "WHERE singleton = 1",
            (json.dumps(_bundle4(default.model_dump(mode="json"))),),
        )
        db.execute(
            "UPDATE projects SET document = ?, revision = 3 WHERE id = ?",
            (json.dumps(invalid), project_id),
        )

    with database.connect() as db:
        migrated = migrate_feature_bundle_v4_authorities_to_v5(
            db,
            created_at="2026-08-22T12:00:00+00:00",
        )
    assert migrated == ("default",)
    with database.connect() as db:
        default_row = db.execute(
            "SELECT document, revision FROM unified_timeline WHERE singleton = 1"
        ).fetchone()
        project_row = db.execute(
            "SELECT document, revision FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
    assert json.loads(default_row["document"])["features"][
        "template_bundle_version"
    ] == 5
    assert int(default_row["revision"]) == 3
    assert json.loads(project_row["document"])["features"][
        "template_bundle_version"
    ] == 4
    assert int(project_row["revision"]) == 3


def test_feature_authority_migration_skips_corrupt_current_authority(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "feature-authorities-corrupt-current.sqlite3")
    database.initialize()
    created = database.create_project("bundle-four-after-corrupt-current")
    project_id = created["id"]
    with database.connect() as db:
        db.execute(
            "UPDATE unified_timeline SET document = ? WHERE singleton = 1",
            (json.dumps({"features": {"template_bundle_version": 6}}),),
        )
        db.execute(
            "UPDATE projects SET document = ?, revision = 4 WHERE id = ?",
            (json.dumps(_bundle4(created["document"])), project_id),
        )

    with database.connect() as db:
        migrated = migrate_feature_bundle_v4_authorities_to_v5(
            db,
            created_at="2026-08-22T12:00:00+00:00",
        )

    assert migrated == (project_id,)
    with database.connect() as db:
        project_row = db.execute(
            "SELECT document, revision FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
    assert json.loads(project_row["document"])["features"][
        "template_bundle_version"
    ] == 5
    assert int(project_row["revision"]) == 5


def test_import_gate_upgrades_unambiguous_v5_and_retains_v5_conflicts() -> None:
    v4, settings_v1 = _v4_pair()
    legacy, settings_v3 = _v5_pair(v4, settings_v1)
    source = migrate_timeline_feature_authority_to_v5(legacy)

    _digest, migrated, _missing_context, _missing_models = prepare_project_import(
        ProjectImportPreflightRequest(document=source.model_dump(mode="json")),
        migrate_v4_to_v5=lambda *_args: pytest.fail("schema migration not expected"),
    )
    assert migrated is not None
    assert migrated.features.template_bundle_version == 6

    current = source.model_dump(mode="json")
    current["features"]["project"]["attention_backend_override"] = {
        "enabled": False,
        "params": {"mode": "ck_int8"},
    }
    _digest, accepted, _missing_context, _missing_models = prepare_project_import(
        ProjectImportPreflightRequest(document=current),
        migrate_v4_to_v5=lambda *_args: pytest.fail("schema migration not expected"),
    )
    assert accepted is not None
    assert accepted.features.template_bundle_version == 6

    conflict = source.model_dump(mode="json")
    conflict["features"]["project"]["attention_backend_override"] = {
        "enabled": True,
        "params": {"mode": "pytorch"},
    }
    _digest, retained, _missing_context, _missing_models = prepare_project_import(
        ProjectImportPreflightRequest(document=conflict),
        migrate_v4_to_v5=lambda *_args: pytest.fail("schema migration not expected"),
    )
    assert retained is not None
    assert retained.features.template_bundle_version == 5
    assert project_v5_compile_authority(retained, settings_v3).effective_features


async def test_existing_bundle4_authority_opens_edits_and_submits_on_bundle6(
    client,
) -> None:
    database = client.director_app.state.database
    current, _ = database.get_timeline_authority()
    with database.connect() as db:
        db.execute(
            "UPDATE unified_timeline SET document = ?, revision = 20 "
            "WHERE singleton = 1",
            (json.dumps(_bundle4(current.model_dump(mode="json"))),),
        )

    database.initialize()

    opened = await client.get("/api/timeline/authority")
    assert opened.status_code == 200, opened.text
    authority = opened.json()
    assert authority["document"]["features"]["template_bundle_version"] == 6
    assert authority["revision"] == 22
    edited = authority["document"]
    edited["title"] = "opened and edited on bundle 6"
    edited["segments"][0]["prompt"] = "A current bundle migration test"
    saved = await client.put(
        "/api/timeline/authority",
        json={"document": edited, "expected_revision": authority["revision"]},
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["document"]["features"]["template_bundle_version"] == 6

    compiled = await client.post(
        "/api/timeline/compile",
        json={"config": saved.json()["document"]},
    )
    assert compiled.status_code == 200, compiled.text
    compile_report = compiled.json()
    assert compile_report["template_bundle_version"] == 6
    assert compile_report["features"]["effective_by_segment"] == {}
    assert compile_report["features"]["resolutions"] == []
    assert compile_report["features"]["uses"]
    assert compile_report["features"]["advisories"] == []
    assert "runtime_fingerprint" not in json.dumps(
        compile_report["features"]["uses"],
        sort_keys=True,
    )
    submitted = await client.post(
        "/api/timeline/jobs",
        json={"config": saved.json()["document"]},
    )
    assert submitted.status_code == 200, submitted.text
    assert database.get_job(submitted.json()["id"]) is not None
