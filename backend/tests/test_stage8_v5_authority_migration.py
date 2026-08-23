from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from directordeck.database import Database, TimelineRevisionConflict
from directordeck.migration_api import (
    ProjectImportPreflightRequest,
    prepare_project_import,
)
from directordeck.migrations import (
    FeatureBundleMigrationConflict,
    migrate_feature_bundle_v4_authorities_to_v5,
)
from directordeck.schemas import UnifiedTimelineDraftV5
from directordeck.workflow.v5_compat import project_v5_compile_authority

from .test_workflow_v5_compat import _v4_pair, _v5_pair


def _bundle4(document: dict) -> dict:
    detached = copy.deepcopy(document)
    detached["features"]["template_bundle_version"] = 4
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
    assert current_default.features.template_bundle_version == 5
    assert current_project.features.template_bundle_version == 5
    assert default_revision == 8
    assert project_revision == 13
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

    # A pending full-state WAL entry still carrying bundle 4 first observes the
    # startup CAS bump, then can be replayed against the current revision. The
    # write boundary upgrades it and can never persist bundle 4 again.
    replay = _bundle4(current_default.model_dump(mode="json"))
    replay["title"] = "replayed after refresh"
    with pytest.raises(TimelineRevisionConflict) as caught:
        database.validate_and_put_timeline_authority(
            UnifiedTimelineDraftV5.model_validate(replay),
            expected_revision=7,
        )
    assert caught.value.actual_revision == 8
    saved, saved_revision = database.validate_and_put_timeline_authority(
        UnifiedTimelineDraftV5.model_validate(replay),
        expected_revision=default_revision,
    )
    assert saved.title == "replayed after refresh"
    assert saved.features.template_bundle_version == 5
    assert saved_revision == 9


def test_feature_authority_migration_rolls_back_all_rows_on_invalid_project(
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
        with pytest.raises(FeatureBundleMigrationConflict) as caught:
            migrate_feature_bundle_v4_authorities_to_v5(
                db,
                created_at="2026-08-22T12:00:00+00:00",
            )
    assert caught.value.project_id == project_id
    with database.connect() as db:
        default_row = db.execute(
            "SELECT document, revision FROM unified_timeline WHERE singleton = 1"
        ).fetchone()
        project_row = db.execute(
            "SELECT document, revision FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
    assert json.loads(default_row["document"])["features"][
        "template_bundle_version"
    ] == 4
    assert int(default_row["revision"]) == 2
    assert json.loads(project_row["document"])["features"][
        "template_bundle_version"
    ] == 4
    assert int(project_row["revision"]) == 3


def test_import_gate_upgrades_bundle4_and_accepts_strict_bundle5_features() -> None:
    v4, settings_v1 = _v4_pair()
    source, settings_v3 = _v5_pair(v4, settings_v1)

    _digest, migrated, _missing_context, _missing_models = prepare_project_import(
        ProjectImportPreflightRequest(document=source.model_dump(mode="json")),
        migrate_v4_to_v5=lambda *_args: pytest.fail("schema migration not expected"),
    )
    assert migrated is not None
    assert migrated.features.template_bundle_version == 5

    current = migrated.model_dump(mode="json")
    current["features"]["project"]["attention_backend_override"] = {
        "enabled": False,
        "params": {"mode": "ck_int8"},
    }
    _digest, accepted, _missing_context, _missing_models = prepare_project_import(
        ProjectImportPreflightRequest(document=current),
        migrate_v4_to_v5=lambda *_args: pytest.fail("schema migration not expected"),
    )
    assert accepted is not None
    assert accepted.features.template_bundle_version == 5
    # The same current resolver used by compile consumes the imported result.
    assert project_v5_compile_authority(accepted, settings_v3).effective_features


async def test_existing_bundle4_authority_opens_edits_and_submits_on_bundle5(
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
    assert authority["document"]["features"]["template_bundle_version"] == 5
    assert authority["revision"] == 21
    edited = authority["document"]
    edited["title"] = "opened and edited on bundle 5"
    edited["segments"][0]["prompt"] = "A current bundle migration test"
    saved = await client.put(
        "/api/timeline/authority",
        json={"document": edited, "expected_revision": authority["revision"]},
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["document"]["features"]["template_bundle_version"] == 5

    compiled = await client.post(
        "/api/timeline/compile",
        json={"config": saved.json()["document"]},
    )
    assert compiled.status_code == 200, compiled.text
    assert compiled.json()["template_bundle_version"] == 5
    submitted = await client.post(
        "/api/timeline/jobs",
        json={"config": saved.json()["document"]},
    )
    assert submitted.status_code == 200, submitted.text
    assert database.get_job(submitted.json()["id"]) is not None
