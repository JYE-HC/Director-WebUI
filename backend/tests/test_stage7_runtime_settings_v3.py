from __future__ import annotations

import sqlite3

import pytest
from pydantic import ValidationError

from directordeck.database import Database
from directordeck.migrations import (
    RuntimeSettingsSchemaMigrated,
    migrate_runtime_settings_v2_to_v3,
)
from directordeck.schemas import (
    LegacyLoraResolutionCompat,
    LegacyStandardLoraOverrideEvidence,
    RuntimeSettingsV2,
    RuntimeSettingsV3,
    default_runtime_settings_v2,
    default_runtime_settings_v3,
    default_settings,
)


def _v3(records: list[dict[str, object]]) -> RuntimeSettingsV3:
    raw = default_runtime_settings_v3().model_dump(mode="json")
    raw["lora_loader_overrides"] = records
    return RuntimeSettingsV3.model_validate(raw)


def _record(
    lora_filename: str,
    *,
    adapter_id: str = "model_only",
    options: dict[str, bool] | None = None,
) -> dict[str, object]:
    return {
        "lora_filename": lora_filename,
        "adapter_id": adapter_id,
        "options": options or {},
    }


def _legacy_v2() -> RuntimeSettingsV2:
    raw = default_runtime_settings_v2().model_dump(mode="json")
    raw["client_id"] = "stage7-migration"
    raw["legacy_lora_resolution_compat"] = LegacyLoraResolutionCompat(
        explicit_overrides=[
            LegacyStandardLoraOverrideEvidence(
                family="fl2va",
                model_filename="Models/FL2VA/Main.safetensors",
                lora_filename="LoRAs/H3/Style.safetensors",
                loader="dedicated",
            ),
            LegacyStandardLoraOverrideEvidence(
                family="ref2va",
                model_filename="models/ref2va/main.safetensors",
                lora_filename="loras/ref2va/style.safetensors",
                loader="bypass_model_only",
            ),
        ]
    ).model_dump(mode="json")
    return RuntimeSettingsV2.model_validate(raw)


def _seed_v2_authority(
    database: Database,
    settings: RuntimeSettingsV2,
    *,
    revision: int = 17,
) -> None:
    with database.connect() as db:
        db.execute(
            "UPDATE settings SET document = ?, revision = ? WHERE singleton = 1",
            (settings.model_dump_json(), revision),
        )
        db.execute("DELETE FROM runtime_settings_migration_notices")


def test_v3_override_identity_is_lora_path_case_sensitive_and_preserving() -> None:
    settings = _v3(
        [
            _record("LoRAs/Style.safetensors"),
            _record("loras/Style.safetensors"),
            _record("other/Style.safetensors"),
        ]
    )

    bindings = [record.binding_tuple() for record in settings.lora_loader_overrides]
    assert len(bindings) == 3
    assert ("LoRAs/Style.safetensors",) in bindings
    assert ("loras/Style.safetensors",) in bindings
    assert ("other/Style.safetensors",) in bindings


def test_v3_override_order_uses_ecmascript_utf16_tuple_order() -> None:
    # Python code-point order puts U+E000 before U+1F600. ECMAScript compares
    # UTF-16 code units, where the emoji's D83D lead surrogate sorts first.
    settings = _v3(
        [
            _record("loras/\ue000.safetensors"),
            _record("loras/😀.safetensors"),
        ]
    )

    assert [
        record.lora_filename for record in settings.lora_loader_overrides
    ] == ["loras/😀.safetensors", "loras/\ue000.safetensors"]


def test_v3_override_paths_are_unique_and_bounded() -> None:
    duplicate = _record("loras/style.safetensors")
    with pytest.raises(ValidationError, match="override paths must be unique"):
        _v3(
            [
                duplicate,
                {
                    **duplicate,
                    "adapter_id": "minimax_h3_turbo",
                    "options": {"low_vram": False},
                },
            ]
        )

    with pytest.raises(ValidationError):
        _v3(
            [
                _record(
                    f"loras/style-{index}.safetensors",
                )
                for index in range(257)
            ]
        )
    with pytest.raises(ValidationError):
        _v3([_record("m" * 1025)])


def test_v3_override_schema_keeps_retired_or_unknown_loader_records_readable() -> None:
    settings = _v3(
        [
            _record(
                "loras/retired.safetensors",
                adapter_id="future_uninstalled_adapter",
                options={"retired_mode": True},
            ),
            _record(
                "loras/ray.safetensors",
                adapter_id="ray_lora",
            ),
        ]
    )

    assert [
        record.model_dump(mode="json")
        for record in settings.lora_loader_overrides
    ] == [
        {
            "lora_filename": "loras/ray.safetensors",
            "adapter_id": "ray_lora",
            "options": {},
        },
        {
            "lora_filename": "loras/retired.safetensors",
            "adapter_id": "future_uninstalled_adapter",
            "options": {"retired_mode": True},
        },
    ]


def test_v2_to_v3_migration_is_atomic_exact_and_idempotent(tmp_path) -> None:
    database = Database(tmp_path / "runtime-v3.sqlite3")
    database.initialize()
    legacy = _legacy_v2()
    _seed_v2_authority(database, legacy)
    historical_snapshot = legacy.model_dump(mode="json")
    now = "2026-08-22T12:00:00+00:00"
    database.create_job(
        {
            "id": "historical-v2-settings",
            "mode": "timeline",
            "status": "succeeded",
            "progress": 1.0,
            "stage": "completed",
            "prompt_id": None,
            "project_id": "default",
            "outputs": [],
            "error": None,
            "config_snapshot": {},
            "settings_snapshot": historical_snapshot,
            "prompt_snapshot": None,
            "created_at": now,
            "updated_at": now,
            "started_at": now,
            "completed_at": now,
        }
    )

    with database.connect() as db:
        migrated, created_notices = migrate_runtime_settings_v2_to_v3(
            db,
            created_at=now,
        )

    assert migrated.schema_version == 3
    assert migrated.client_id == legacy.client_id
    assert [
        record.model_dump(mode="json")
        for record in migrated.lora_loader_overrides
    ] == [
        {
            "lora_filename": "LoRAs/H3/Style.safetensors",
            "adapter_id": "minimax_h3_turbo",
            # Historical data stays structural; the current product default is
            # applied only if this mapping becomes active or is changed.
            "options": {},
        },
        {
            "lora_filename": "loras/ref2va/style.safetensors",
            "adapter_id": "model_only",
            "options": {},
        },
    ]
    assert "legacy_lora_resolution_compat" not in migrated.model_dump(
        mode="json"
    )
    assert len(created_notices) == 1
    notice = created_notices[0]
    assert notice.code == "legacy_lora_resolution_review_required"
    assert notice.action == "review_lora_loader_mappings"
    assert "metadata-based" in notice.message

    with database.connect() as db:
        row_after_first = db.execute(
            "SELECT document, revision FROM settings WHERE singleton = 1"
        ).fetchone()
        notice_count_after_first = db.execute(
            "SELECT COUNT(*) FROM runtime_settings_migration_notices"
        ).fetchone()[0]
    assert row_after_first["revision"] == 18
    assert notice_count_after_first == 1
    assert database.get_job("historical-v2-settings")["settings_snapshot"] == (
        historical_snapshot
    )

    with database.connect() as db:
        migrated_again, existing_notices = migrate_runtime_settings_v2_to_v3(
            db,
            created_at="2026-08-22T13:00:00+00:00",
        )
    with database.connect() as db:
        row_after_second = db.execute(
            "SELECT document, revision FROM settings WHERE singleton = 1"
        ).fetchone()
        notice_count_after_second = db.execute(
            "SELECT COUNT(*) FROM runtime_settings_migration_notices"
        ).fetchone()[0]
    assert migrated_again == migrated
    assert existing_notices == [notice]
    assert dict(row_after_second) == dict(row_after_first)
    assert notice_count_after_second == 1


def test_v2_to_v3_notice_failure_rolls_back_authority(tmp_path) -> None:
    database = Database(tmp_path / "runtime-v3-rollback.sqlite3")
    database.initialize()
    legacy = _legacy_v2()
    _seed_v2_authority(database, legacy, revision=9)
    with database.connect() as db:
        before = db.execute(
            "SELECT document, revision, updated_at FROM settings WHERE singleton = 1"
        ).fetchone()
        db.execute(
            "CREATE TRIGGER reject_runtime_settings_notice "
            "BEFORE INSERT ON runtime_settings_migration_notices "
            "BEGIN SELECT RAISE(ABORT, 'notice rejected'); END"
        )

    with database.connect() as db:
        with pytest.raises(sqlite3.IntegrityError, match="notice rejected"):
            migrate_runtime_settings_v2_to_v3(
                db,
                created_at="2026-08-22T12:00:00+00:00",
            )

    with database.connect() as db:
        after = db.execute(
            "SELECT document, revision, updated_at FROM settings WHERE singleton = 1"
        ).fetchone()
        notice_count = db.execute(
            "SELECT COUNT(*) FROM runtime_settings_migration_notices"
        ).fetchone()[0]
    assert dict(after) == dict(before)
    assert notice_count == 0


def test_fresh_database_starts_at_v3_without_legacy_notice(tmp_path) -> None:
    database = Database(tmp_path / "fresh-v3.sqlite3")
    database.initialize()
    first, first_token = database.get_settings_authority()
    database.initialize()
    second, second_token = database.get_settings_authority()

    assert first == second == default_runtime_settings_v3()
    assert first_token == second_token
    assert database.list_runtime_settings_migration_notices() == []


async def test_v3_settings_api_rejects_stale_writers_and_keeps_cas(client) -> None:
    authority_response = await client.get("/api/settings/authority")
    assert authority_response.status_code == 200, authority_response.text
    authority = authority_response.json()
    assert authority["settings"]["schema_version"] == 3

    stale_documents = (
        (1, default_settings().model_dump(mode="json")),
        (2, default_runtime_settings_v2().model_dump(mode="json")),
    )
    for schema_version, document in stale_documents:
        stale = await client.put(
            "/api/settings/authority",
            json={
                "schema_version": schema_version,
                "document": document,
                "expected_authority_token": authority["authority_token"],
            },
        )
        assert stale.status_code == 409, stale.text
        assert stale.json()["detail"]["code"] == (
            "runtime_settings_schema_migrated"
        )
        assert stale.json()["detail"]["current_schema"] == 3

    malformed = dict(authority["settings"])
    malformed["legacy_lora_resolution_compat"] = {
        "schema_version": 1,
        "auto_resolution_strategy_version": (
            "v4-known-filename-or-safetensors-metadata-v1"
        ),
        "explicit_overrides": [],
    }
    rejected = await client.put(
        "/api/settings/authority",
        json={
            "schema_version": 3,
            "document": malformed,
            "expected_authority_token": authority["authority_token"],
        },
    )
    assert rejected.status_code == 422, rejected.text

    document = dict(authority["settings"])
    document["client_id"] = "stage7-cas"
    saved = await client.put(
        "/api/settings/authority",
        json={
            "schema_version": 3,
            "document": document,
            "expected_authority_token": authority["authority_token"],
        },
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["settings"]["client_id"] == "stage7-cas"
    assert saved.json()["authority_token"] != authority["authority_token"]

    conflict = await client.put(
        "/api/settings/authority",
        json={
            "schema_version": 3,
            "document": document,
            "expected_authority_token": authority["authority_token"],
        },
    )
    assert conflict.status_code == 409, conflict.text
    assert conflict.json()["detail"]["code"] == (
        "runtime_settings_authority_conflict"
    )

    notices = await client.get("/api/settings/migration-notices")
    assert notices.status_code == 200, notices.text
    assert notices.json() == {"notices": []}


async def test_settings_api_rejects_a_new_mapping_outside_the_effective_policy(
    client,
) -> None:
    authority = (await client.get("/api/settings/authority")).json()
    document = authority["settings"]
    document["lora_loader_overrides"] = [{
        "lora_filename": "loras/style.safetensors",
        "adapter_id": "minimax_h3_turbo",
        "options": {"low_vram": False},
    }]

    rejected = await client.put(
        "/api/settings/authority",
        json={
            "schema_version": 3,
            "document": document,
            "expected_authority_token": authority["authority_token"],
        },
    )

    assert rejected.status_code == 422, rejected.text
    detail = rejected.json()["detail"]
    assert detail["code"] == "lora_loader_not_allowed_for_file"
    assert detail["allowed_loader_ids"] == ["model_only"]

    document["lora_loader_overrides"] = [{
        "lora_filename": "loras/minimax_h3_turbo_v4.safetensors",
        "adapter_id": "minimax_h3_turbo",
        "options": {"low_vram": True},
    }]
    allowed = await client.put(
        "/api/settings/authority",
        json={
            "schema_version": 3,
            "document": document,
            "expected_authority_token": authority["authority_token"],
        },
    )
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["settings"]["lora_loader_overrides"] == [
        document["lora_loader_overrides"][0]
    ]


async def test_settings_api_rejects_invalid_options_on_a_new_mapping(client) -> None:
    authority = (await client.get("/api/settings/authority")).json()
    document = authority["settings"]
    document["lora_loader_overrides"] = [{
        "lora_filename": "loras/minimax_h3_turbo_v4.safetensors",
        "adapter_id": "minimax_h3_turbo",
        "options": {"removed_option": True},
    }]

    rejected = await client.put(
        "/api/settings/authority",
        json={
            "schema_version": 3,
            "document": document,
            "expected_authority_token": authority["authority_token"],
        },
    )

    assert rejected.status_code == 422, rejected.text
    assert rejected.json()["detail"] == {
        "code": "lora_loader_options_invalid",
        "message": "The selected LoRA loader configuration is invalid.",
        "lora_filename": "loras/minimax_h3_turbo_v4.safetensors",
        "adapter_id": "minimax_h3_turbo",
    }


async def test_settings_api_reports_config_failure_only_for_a_new_mapping(
    client,
    monkeypatch,
) -> None:
    authority = (await client.get("/api/settings/authority")).json()
    document = authority["settings"]
    document["lora_loader_overrides"] = [{
        "lora_filename": "loras/style.safetensors",
        "adapter_id": "model_only",
        "options": {},
    }]

    def unavailable():
        raise RuntimeError("private product configuration failure")

    monkeypatch.setattr(
        "directordeck.app.get_directordeck_config",
        unavailable,
    )
    rejected = await client.put(
        "/api/settings/authority",
        json={
            "schema_version": 3,
            "document": document,
            "expected_authority_token": authority["authority_token"],
        },
    )

    assert rejected.status_code == 503, rejected.text
    assert rejected.json()["detail"]["code"] == (
        "lora_product_config_unavailable"
    )
    assert "private product configuration failure" not in rejected.text


async def test_settings_api_preserves_and_allows_removing_an_old_invalid_mapping(
    client,
    monkeypatch,
) -> None:
    database = client.director_app.state.database
    settings, authority = database.get_settings_authority()
    raw = settings.model_dump(mode="json")
    raw["lora_loader_overrides"] = [{
        "lora_filename": "loras/style.safetensors",
        "adapter_id": "retired_loader",
        "options": {"retired_mode": True},
    }]
    historical = RuntimeSettingsV3.model_validate(raw)
    database.put_settings_v3_authority(
        historical,
        expected_authority_token=authority,
        schema_version=3,
    )
    current = (await client.get("/api/settings/authority")).json()

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("product configuration unavailable")

    monkeypatch.setattr(
        "directordeck.app.get_directordeck_config",
        unavailable,
    )
    monkeypatch.setattr(
        "directordeck.app.get_lora_loader_policy",
        unavailable,
    )

    preserved_document = current["settings"]
    preserved_document["client_id"] = "preserve-historical-invalid-mapping"
    preserved = await client.put(
        "/api/settings/authority",
        json={
            "schema_version": 3,
            "document": preserved_document,
            "expected_authority_token": current["authority_token"],
        },
    )
    assert preserved.status_code == 200, preserved.text
    assert len(preserved.json()["settings"]["lora_loader_overrides"]) == 1

    restored_document = preserved.json()["settings"]
    restored_document["lora_loader_overrides"] = []
    restored = await client.put(
        "/api/settings/authority",
        json={
            "schema_version": 3,
            "document": restored_document,
            "expected_authority_token": preserved.json()["authority_token"],
        },
    )
    assert restored.status_code == 200, restored.text
    assert restored.json()["settings"]["lora_loader_overrides"] == []


async def test_migrated_notice_api_returns_typed_action(client) -> None:
    database = client.director_app.state.database
    _seed_v2_authority(database, _legacy_v2(), revision=4)
    with database.connect() as db:
        migrate_runtime_settings_v2_to_v3(
            db,
            created_at="2026-08-22T12:00:00+00:00",
        )

    response = await client.get("/api/settings/migration-notices")

    assert response.status_code == 200, response.text
    assert response.json() == {
        "notices": [
            {
                "schema_version": 1,
                "id": "runtime-settings-v2-v3-lora-resolution-review",
                "code": "legacy_lora_resolution_review_required",
                "severity": "warning",
                "action": "review_lora_loader_mappings",
                "legacy_strategy_version": (
                    "v4-known-filename-or-safetensors-metadata-v1"
                ),
                "message": (
                    "Legacy incomplete, filename-based, or metadata-based "
                    "Standard LoRA resolution was retired. Review LoRA loader "
                    "mappings; unmapped LoRAs now use the configured default "
                    "model-only loader."
                ),
                "created_at": "2026-08-22T12:00:00Z",
            }
        ]
    }


def test_direct_stale_v2_database_write_is_tombstoned(tmp_path) -> None:
    database = Database(tmp_path / "stale-v2.sqlite3")
    database.initialize()
    _settings, token = database.get_settings_authority()

    with pytest.raises(RuntimeSettingsSchemaMigrated) as caught:
        database.put_settings_v2_authority(
            default_runtime_settings_v2(),
            expected_authority_token=token,
            schema_version=2,
        )

    assert caught.value.current_schema == 3
