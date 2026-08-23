from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from directordeck.database import (
    Database,
    SettingsAuthorityConflict,
    SettingsAuthorityRequired,
)
from directordeck.migrations import (
    RuntimeSettingsSchemaMigrated,
    TimelineSchemaMigrated,
    WorkflowMigrationConflict,
    legacy_client_timeline_v4_projection,
    legacy_client_timeline_v5_projection,
    legacy_creative_binding_context,
    migrate_runtime_settings_v1_to_v2,
    migrate_timeline_v4_to_v5,
    migrate_v4_authorities_to_v5,
)
from directordeck.schemas import (
    FeatureConfiguration,
    FeatureSelection,
    ModelStack,
    RuntimeSettingsV1,
    RuntimeSettingsV2,
    UnifiedTimelineDraftV4,
    UnifiedTimelineDraftV5,
    default_settings,
    default_timeline_draft,
    default_timeline_draft_v5,
)
from directordeck.workflow.execution import (
    legacy_fnv1a32_document_digest,
    sha256_document_digest,
)


_CLIENT_FIXTURES = (
    Path(__file__).resolve().parents[2]
    / "frontend"
    / "src"
    / "test"
    / "fixtures"
    / "extensible-workflow-v0"
)


def _legacy_settings() -> RuntimeSettingsV1:
    raw = default_settings().model_dump(mode="json")
    raw["client_id"] = "stage6-migration"
    raw["multi_gpu_enabled"] = True
    raw["models"]["fl2va"].update(
        filename="diffusion/fl2va-stage6.safetensors",
        device="gpu:1",
        lora_name="loras/fl2va-stage6.safetensors",
        lora_strength=0.625,
        standard_lora_loader_override={
            "loader": "model_only",
            "model_filename": "diffusion/fl2va-stage6.safetensors",
            "lora_name": "loras/fl2va-stage6.safetensors",
        },
    )
    raw["models"]["ref2va"].update(
        filename="diffusion/ref2va-stage6.safetensors",
        device="default",
        lora_name="loras/ref2va-stage6.safetensors",
        lora_strength=0.875,
        raylight={
            "gpu_select": [0, 1],
            "ulysses_degree": 2,
            "ring_degree": 1,
            "cfg_degree": 1,
            "dp_degree": 1,
            "fsdp": False,
            "cpu_offload": False,
        },
    )
    raw["models"]["clip"].update(
        filename="text/clip-stage6.safetensors",
        device="gpu:0",
    )
    raw["models"]["video_vae"].update(
        filename="vae/video-stage6.safetensors",
        device="gpu:0",
    )
    raw["models"]["audio_vae"].update(
        filename="vae/audio-stage6.safetensors",
        device="default",
    )
    return RuntimeSettingsV1.model_validate(raw)


def _legacy_timeline(title: str):
    return default_timeline_draft().model_copy(update={"title": title}, deep=True)


def test_receipt_client_digest_projection_matches_frozen_typescript_golden() -> None:
    legacy_raw = json.loads(
        (_CLIENT_FIXTURES / "timeline-project-v4.json").read_text(encoding="utf-8")
    )
    backend_v5_raw = json.loads(
        (
            _CLIENT_FIXTURES / "timeline-project-v5-backend-model-dump.json"
        ).read_text(encoding="utf-8")
    )
    client_v5_raw = json.loads(
        (
            _CLIENT_FIXTURES / "timeline-project-v5-client-digest.json"
        ).read_text(encoding="utf-8")
    )
    legacy = UnifiedTimelineDraftV4.model_validate(legacy_raw)
    current = UnifiedTimelineDraftV5.model_validate(backend_v5_raw)

    legacy_projection = legacy_client_timeline_v4_projection(
        legacy,
        source_raw=legacy_raw,
    )
    current_projection = legacy_client_timeline_v5_projection(current)

    assert legacy_projection == legacy_raw
    assert current_projection == client_v5_raw
    assert legacy_fnv1a32_document_digest(legacy_projection).value == (
        "fnv1a-93bba5c8"
    )
    assert legacy_fnv1a32_document_digest(current_projection).value == (
        "fnv1a-0189db3f"
    )


def test_database_migration_receipt_matches_cross_runtime_golden(
    tmp_path: Path,
) -> None:
    """Exercise the golden digests through the real SQLite migration path."""

    legacy_raw = json.loads(
        (_CLIENT_FIXTURES / "timeline-project-v4.json").read_text(encoding="utf-8")
    )
    expected_v5_raw = json.loads(
        (
            _CLIENT_FIXTURES / "timeline-project-v5-backend-model-dump.json"
        ).read_text(encoding="utf-8")
    )
    client_v5_raw = json.loads(
        (
            _CLIENT_FIXTURES / "timeline-project-v5-client-digest.json"
        ).read_text(encoding="utf-8")
    )
    legacy_settings_raw = default_settings().model_dump(mode="json")
    for role in ("fl2va", "ref2va", "clip", "video_vae", "audio_vae"):
        legacy_settings_raw["models"][role]["filename"] = expected_v5_raw[
            "model_stack"
        ][role]["filename"]
    for family in ("fl2va", "ref2va"):
        family_lora = expected_v5_raw["features"]["project"]["lora"]["params"][
            "by_family"
        ][family]
        legacy_settings_raw["models"][family].update(
            lora_name=family_lora["filename"],
            lora_strength=family_lora["strength"],
            standard_lora_loader_override=None,
        )
    legacy_settings = RuntimeSettingsV1.model_validate(legacy_settings_raw)

    database = Database(tmp_path / "cross-runtime-receipt.sqlite3")
    database.initialize()
    with database.connect() as db:
        db.execute("DROP TRIGGER IF EXISTS project_migration_receipts_immutable_update")
        db.execute("DROP TRIGGER IF EXISTS project_migration_receipts_immutable_delete")
        db.execute("DELETE FROM project_migration_receipts")
        db.execute(
            "UPDATE settings SET document = ?, revision = 7 WHERE singleton = 1",
            (legacy_settings.model_dump_json(),),
        )
        db.execute(
            "UPDATE unified_timeline SET document = ?, revision = 4 "
            "WHERE singleton = 1",
            (json.dumps(legacy_raw, ensure_ascii=False),),
        )

    database.initialize()

    migrated, revision = database.get_timeline_authority()
    receipt = database.get_latest_project_migration_receipt("default")
    expected_current = json.loads(json.dumps(expected_v5_raw))
    expected_current["features"]["template_bundle_version"] = 5
    assert migrated.model_dump(mode="json") == expected_current
    # The frozen schema receipt lands at revision 5/bundle 4. The independent
    # feature-authority migration then advances exactly once to bundle 5.
    assert revision == 6
    assert receipt is not None
    assert receipt.old_client_digest.value == "fnv1a-93bba5c8"
    assert receipt.new_client_digest.value == "fnv1a-0189db3f"
    assert receipt.old_server_digest == sha256_document_digest(legacy_raw)
    assert receipt.new_server_digest == sha256_document_digest(expected_v5_raw)
    assert legacy_client_timeline_v5_projection(
        UnifiedTimelineDraftV5.model_validate(expected_v5_raw)
    ) == client_v5_raw


def _restore_pre_stage6_authorities(
    database: Database,
    *,
    projects: dict[str, tuple[dict, int]],
    default_revision: int = 4,
    settings_revision: int = 7,
) -> RuntimeSettingsV1:
    settings = _legacy_settings()
    default = _legacy_timeline("default-v4")
    with database.connect() as db:
        db.execute("DROP TRIGGER IF EXISTS project_migration_receipts_immutable_update")
        db.execute("DROP TRIGGER IF EXISTS project_migration_receipts_immutable_delete")
        db.execute("DELETE FROM project_migration_receipts")
        db.execute("DELETE FROM projects")
        db.execute(
            "UPDATE settings SET document = ?, revision = ?, updated_at = ? "
            "WHERE singleton = 1",
            (settings.model_dump_json(), settings_revision, "before-stage6"),
        )
        db.execute(
            "UPDATE unified_timeline SET document = ?, revision = ?, updated_at = ? "
            "WHERE singleton = 1",
            (default.model_dump_json(), default_revision, "before-stage6"),
        )
        for project_id, (document, revision) in projects.items():
            db.execute(
                "INSERT INTO projects(id, title, document, created_at, updated_at, "
                "revision) VALUES(?, ?, ?, ?, ?, ?)",
                (
                    project_id,
                    str(document.get("title") or project_id),
                    json.dumps(document, ensure_ascii=False),
                    "before-stage6",
                    "before-stage6",
                    revision,
                ),
            )
    return settings


def _migrate_one_legacy_default(database: Database) -> None:
    _restore_pre_stage6_authorities(database, projects={})
    database.initialize()


def test_v5_contract_allows_incomplete_models_and_bounds_feature_json() -> None:
    incomplete = default_timeline_draft_v5()
    assert incomplete.version == 5
    assert all(
        getattr(incomplete.model_stack, role).filename is None
        for role in ("fl2va", "ref2va", "clip", "video_vae", "audio_vae")
    )
    with pytest.raises(ValueError):
        ModelStack.model_validate({})
    missing_version = incomplete.model_dump(mode="json")
    missing_version.pop("version")
    with pytest.raises(ValueError):
        UnifiedTimelineDraftV5.model_validate(missing_version)

    raw = incomplete.model_dump(mode="json")
    raw["features"]["by_segment"]["missing-segment"] = {}
    with pytest.raises(ValueError, match="unknown segments"):
        UnifiedTimelineDraftV5.model_validate(raw)

    nested: object = True
    for index in range(10):
        nested = {f"level{index}": nested}
    with pytest.raises(ValueError, match="nesting depth"):
        FeatureSelection(enabled=True, params={"root": nested})

    with pytest.raises(ValueError):
        FeatureConfiguration(
            template_bundle_version=4,
            project={
                f"feature{index}": FeatureSelection()
                for index in range(65)
            },
        )


def test_pure_migration_moves_every_creative_and_runtime_authority_exactly() -> None:
    settings = _legacy_settings()
    timeline = _legacy_timeline("all-bindings")

    migrated = migrate_timeline_v4_to_v5(timeline, settings)
    runtime = migrate_runtime_settings_v1_to_v2(settings)
    context = legacy_creative_binding_context(settings)

    assert migrated.model_stack.model_dump(mode="json") == {
        "fl2va": {"filename": "diffusion/fl2va-stage6.safetensors"},
        "ref2va": {"filename": "diffusion/ref2va-stage6.safetensors"},
        "clip": {"filename": "text/clip-stage6.safetensors"},
        "video_vae": {"filename": "vae/video-stage6.safetensors"},
        "audio_vae": {"filename": "vae/audio-stage6.safetensors"},
    }
    assert migrated.features.project["lora"].model_dump(mode="json") == {
        "enabled": True,
        "params": {
            "by_family": {
                "fl2va": {
                    "enabled": True,
                    "filename": "loras/fl2va-stage6.safetensors",
                    "strength": 0.625,
                },
                "ref2va": {
                    "enabled": True,
                    "filename": "loras/ref2va-stage6.safetensors",
                    "strength": 0.875,
                },
            }
        },
    }
    assert migrated.segments == timeline.segments
    assert migrated.render == timeline.render
    assert migrated.sampling == timeline.sampling
    assert migrated.export_mode == timeline.export_mode

    assert runtime.schema_version == 2
    assert runtime.placement.fl2va.device == "gpu:1"
    assert runtime.placement.ref2va.raylight.gpu_select == [0, 1]
    assert runtime.placement.clip_device == "gpu:0"
    assert runtime.placement.video_vae_device == "gpu:0"
    assert runtime.placement.audio_vae_device == "default"
    assert runtime.legacy_lora_resolution_compat.explicit_overrides == (
        context.explicit_standard_lora_overrides
    )
    serialized_runtime = runtime.model_dump(mode="json")
    assert "models" not in serialized_runtime
    assert "features" not in serialized_runtime


def test_fresh_database_starts_at_v5_without_a_fictitious_receipt(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "fresh.sqlite3")
    database.initialize()

    timeline, revision = database.get_timeline_authority()
    settings, token = database.get_settings_authority()
    assert timeline.version == 5
    assert revision == 0
    assert all(
        getattr(timeline.model_stack, role).filename is not None
        for role in ("fl2va", "ref2va", "clip", "video_vae", "audio_vae")
    )
    assert settings.schema_version == 3
    assert len(token) == 64
    assert database.get_latest_project_migration_receipt("default") is None

    database.initialize()
    assert database.get_timeline_authority() == (timeline, revision)
    assert database.get_settings_authority() == (settings, token)

    incomplete = database.create_project("incomplete")
    assert all(
        value["filename"] is None
        for value in incomplete["document"]["model_stack"].values()
    )
    explicit_stack = ModelStack.model_validate(timeline.model_stack.model_dump())
    complete = database.create_project(
        "complete",
        initial_model_stack=explicit_stack,
    )
    assert complete["document"]["model_stack"] == explicit_stack.model_dump(
        mode="json"
    )


def test_database_migration_is_atomic_across_default_projects_and_settings(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "atomic.sqlite3")
    database.initialize()
    _restore_pre_stage6_authorities(
        database,
        projects={
            "project-a": (_legacy_timeline("project-a").model_dump(mode="json"), 2),
            "project-b": (_legacy_timeline("project-b").model_dump(mode="json"), 9),
        },
    )

    with database.connect() as db:
        receipts = migrate_v4_authorities_to_v5(
            db, created_at="2026-08-21T20:00:00+00:00"
        )

    assert {receipt.project_id for receipt in receipts} == {
        "default",
        "project-a",
        "project-b",
    }
    with database.connect() as db:
        raw_settings = db.execute(
            "SELECT document FROM settings WHERE singleton = 1"
        ).fetchone()
    assert raw_settings is not None
    assert RuntimeSettingsV2.model_validate_json(
        raw_settings["document"]
    ).schema_version == 2
    assert database.get_timeline_authority()[1] == 5
    assert database.get_project_timeline_authority("project-a")[1] == 3
    assert database.get_project_timeline_authority("project-b")[1] == 10
    for project_id in ("default", "project-a", "project-b"):
        timeline, revision = database.get_project_timeline_authority(project_id)
        receipt = database.get_latest_project_migration_receipt(project_id)
        assert timeline.version == 5
        assert receipt is not None
        assert receipt.new_revision == revision
        assert receipt.new_server_digest == sha256_document_digest(
            timeline.model_dump(mode="json")
        )
        assert receipt.legacy_creative_binding_context.model_stack == (
            timeline.model_stack
        )


def test_v5_import_validates_assets_and_inserts_in_one_write_transaction(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "import-atomic.sqlite3")
    database.initialize()
    raw = default_timeline_draft_v5().model_dump(mode="json")
    raw["segments"][0]["first_image"] = {
        "id": "missing-import-asset",
        "name": "missing.png",
        "subfolder": "",
        "type": "input",
        "kind": "image",
    }
    proposed = UnifiedTimelineDraftV5.model_validate(raw)
    before = database.list_projects()

    with pytest.raises(ValueError):
        database.import_project("must-roll-back", proposed)

    assert database.list_projects() == before


def test_migration_rolls_back_every_authority_when_one_project_is_invalid(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "rollback.sqlite3")
    database.initialize()
    valid = _legacy_timeline("valid").model_dump(mode="json")
    invalid = _legacy_timeline("invalid").model_dump(mode="json")
    invalid["version"] = 99
    _restore_pre_stage6_authorities(
        database,
        projects={"a-valid": (valid, 3), "z-invalid": (invalid, 6)},
    )

    with database.connect() as db:
        with pytest.raises(WorkflowMigrationConflict, match="valid v4"):
            migrate_v4_authorities_to_v5(
                db, created_at="2026-08-21T20:01:00+00:00"
            )

    with database.connect() as db:
        settings = json.loads(
            db.execute(
                "SELECT document FROM settings WHERE singleton = 1"
            ).fetchone()["document"]
        )
        default = db.execute(
            "SELECT document, revision FROM unified_timeline WHERE singleton = 1"
        ).fetchone()
        rows = db.execute(
            "SELECT id, document, revision FROM projects ORDER BY id"
        ).fetchall()
        receipt_count = db.execute(
            "SELECT COUNT(*) FROM project_migration_receipts"
        ).fetchone()[0]
    assert "schema_version" not in settings
    assert json.loads(default["document"])["version"] == 4
    assert int(default["revision"]) == 4
    assert [(row["id"], json.loads(row["document"])["version"], row["revision"]) for row in rows] == [
        ("a-valid", 4, 3),
        ("z-invalid", 99, 6),
    ]
    assert receipt_count == 0


def test_restart_is_idempotent_and_edited_migrated_project_is_never_overwritten(
    tmp_path: Path,
) -> None:
    path = tmp_path / "idempotent.sqlite3"
    database = Database(path)
    database.initialize()
    _migrate_one_legacy_default(database)
    created = database.create_project("post-migration")
    project_id = created["id"]
    timeline, revision = database.get_project_timeline_authority(project_id)
    edited = timeline.model_copy(update={"title": "edited-after-migration"}, deep=True)
    database.validate_and_put_project_timeline_authority(
        project_id,
        edited,
        expected_revision=revision,
    )
    default_receipt = database.get_latest_project_migration_receipt("default")
    assert default_receipt is not None
    default, default_revision = database.get_timeline_authority()
    edited_default = default.model_copy(
        update={"title": "receipt-project-edited-after-migration"}, deep=True
    )
    database.validate_and_put_timeline_authority(
        edited_default,
        expected_revision=default_revision,
    )
    before_settings, before_token = database.get_settings_authority()

    database.initialize()
    database.initialize()

    current, current_revision = database.get_project_timeline_authority(project_id)
    after_settings, after_token = database.get_settings_authority()
    assert current.title == "edited-after-migration"
    assert current_revision == revision + 1
    assert database.get_latest_project_migration_receipt(project_id) is None
    assert database.get_latest_project_migration_receipt("default") == default_receipt
    current_default, current_default_revision = database.get_timeline_authority()
    assert current_default.title == "receipt-project-edited-after-migration"
    assert current_default_revision == default_revision + 1
    assert after_settings == before_settings
    assert after_token == before_token


def test_receipt_digest_mismatch_fails_closed_on_read_and_restart(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "receipt-corrupt.sqlite3")
    database.initialize()
    _migrate_one_legacy_default(database)
    receipt = database.get_latest_project_migration_receipt("default")
    assert receipt is not None
    with database.connect() as db:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute(
                "DELETE FROM project_migration_receipts WHERE migration_id = ?",
                (receipt.migration_id,),
            )
    with database.connect() as db:
        db.execute("DROP TRIGGER project_migration_receipts_immutable_update")
        db.execute(
            "UPDATE project_migration_receipts SET receipt_digest = ? "
            "WHERE migration_id = ?",
            ("sha256-" + "0" * 64, receipt.migration_id),
        )
    with pytest.raises(WorkflowMigrationConflict, match="digest"):
        database.get_latest_project_migration_receipt("default")
    with pytest.raises(WorkflowMigrationConflict, match="digest"):
        database.initialize()


def test_settings_v3_cas_detects_aba_and_old_schema_writes_are_rejected(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "cas.sqlite3")
    database.initialize()
    original, token_a = database.get_settings_authority()

    with pytest.raises(RuntimeSettingsSchemaMigrated):
        database.put_settings(default_settings())
    with pytest.raises(SettingsAuthorityRequired):
        database.put_settings(original)

    changed = original.model_copy(update={"client_id": "stage6-b"})
    _saved_b, token_b = database.put_settings_v3_authority(
        changed,
        expected_authority_token=token_a,
        schema_version=3,
    )
    saved_a, token_a2 = database.put_settings_v3_authority(
        original,
        expected_authority_token=token_b,
        schema_version=3,
    )
    assert saved_a == original
    assert token_a2 != token_a
    with pytest.raises(SettingsAuthorityConflict):
        database.put_settings_v3_authority(
            changed,
            expected_authority_token=token_a,
            schema_version=3,
        )
    with pytest.raises(RuntimeSettingsSchemaMigrated):
        database.put_settings_v3_authority(
            original,
            expected_authority_token=token_a2,
            schema_version=2,
        )
    with pytest.raises(RuntimeSettingsSchemaMigrated):
        database.put_settings_v2_authority(
            migrate_runtime_settings_v1_to_v2(default_settings()),
            expected_authority_token=token_a2,
            schema_version=2,
        )


def test_timeline_v4_stale_write_reports_the_exact_migration_receipt(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "timeline-stale.sqlite3")
    database.initialize()
    _migrate_one_legacy_default(database)
    receipt = database.get_latest_project_migration_receipt("default")
    assert receipt is not None

    with pytest.raises(TimelineSchemaMigrated) as caught:
        database.validate_and_put_timeline_authority(
            default_timeline_draft(), expected_revision=receipt.new_revision
        )

    assert caught.value.project_id == "default"
    assert caught.value.migration_id == receipt.migration_id
    current, revision = database.get_timeline_authority()
    assert current.version == 5
    assert current.features.template_bundle_version == 5
    assert revision == receipt.new_revision + 1


def test_receipt_exists_but_authority_digest_changed_at_migration_revision_fails(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "authority-corrupt.sqlite3")
    database.initialize()
    _migrate_one_legacy_default(database)
    receipt = database.get_latest_project_migration_receipt("default")
    assert receipt is not None
    with database.connect() as db:
        raw = json.loads(
            db.execute(
                "SELECT document FROM unified_timeline WHERE singleton = 1"
            ).fetchone()["document"]
        )
        raw["title"] = "tampered-without-revision"
        raw["features"]["template_bundle_version"] = 4
        db.execute(
            "UPDATE unified_timeline SET document = ?, revision = ? "
            "WHERE singleton = 1",
            (json.dumps(raw, ensure_ascii=False), receipt.new_revision),
        )

    with pytest.raises(WorkflowMigrationConflict, match="does not match"):
        database.initialize()
