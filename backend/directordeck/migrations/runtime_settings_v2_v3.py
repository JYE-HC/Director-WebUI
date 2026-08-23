from __future__ import annotations

"""Atomic, idempotent RuntimeSettingsV2 to mapping-only V3 migration."""

from datetime import datetime
import json
import sqlite3

from pydantic import ValidationError

from ..schemas import (
    LoraLoaderOverrideRecord,
    RuntimeSettingsMigrationNotice,
    RuntimeSettingsV2,
    RuntimeSettingsV3,
)
from .timeline_v4_v5 import WorkflowMigrationConflict


RUNTIME_SETTINGS_V2_V3_MIGRATION_VERSION = "runtime-settings-v2-v3@1"
_AUTO_RESOLUTION_NOTICE_ID = "runtime-settings-v2-v3-lora-resolution-review"
_LEGACY_LOADER_ADAPTER_IDS = {
    "dedicated": "minimax_h3_turbo",
    "bypass_model_only": "model_only",
    "model_only": "model_only",
}


def ensure_runtime_settings_migration_notice_schema(
    db: sqlite3.Connection,
) -> None:
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS runtime_settings_migration_notices (
            id TEXT PRIMARY KEY,
            document TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )


def _notice_created_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WorkflowMigrationConflict(
            "runtime settings migration timestamp is invalid"
        ) from exc
    if parsed.tzinfo is None:
        raise WorkflowMigrationConflict(
            "runtime settings migration timestamp must include a timezone"
        )
    return parsed


def _decode_notices(
    db: sqlite3.Connection,
) -> list[RuntimeSettingsMigrationNotice]:
    notices: list[RuntimeSettingsMigrationNotice] = []
    for row in db.execute(
        "SELECT document FROM runtime_settings_migration_notices "
        "ORDER BY created_at, id"
    ).fetchall():
        try:
            notices.append(
                RuntimeSettingsMigrationNotice.model_validate_json(
                    str(row["document"])
                )
            )
        except ValidationError as exc:
            raise WorkflowMigrationConflict(
                "runtime settings migration notice is invalid"
            ) from exc
    return notices


def migrate_runtime_settings_v2_to_v3(
    db: sqlite3.Connection,
    *,
    created_at: str,
) -> tuple[RuntimeSettingsV3, list[RuntimeSettingsMigrationNotice]]:
    """Advance the settings authority exactly once with its notices.

    Only complete explicit V2 overrides are facts strong enough to become
    exact mapping records.  The legacy automatic/metadata strategy becomes an
    actionable notice; it never manufactures a broad mapping or persists a
    factory default.
    """

    if db.in_transaction:
        raise RuntimeError(
            "v2 to v3 settings migration requires an idle connection"
        )
    db.execute("BEGIN IMMEDIATE")
    try:
        ensure_runtime_settings_migration_notice_schema(db)
        row = db.execute(
            "SELECT document, revision FROM settings WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise WorkflowMigrationConflict("runtime settings authority is missing")
        try:
            raw = json.loads(str(row["document"]))
        except (json.JSONDecodeError, TypeError) as exc:
            raise WorkflowMigrationConflict(
                "runtime settings document is malformed"
            ) from exc
        if not isinstance(raw, dict):
            raise WorkflowMigrationConflict(
                "runtime settings document must be an object"
            )

        if raw.get("schema_version") == 3:
            try:
                current = RuntimeSettingsV3.model_validate(raw)
            except ValidationError as exc:
                raise WorkflowMigrationConflict(
                    "runtime settings v3 document is invalid"
                ) from exc
            notices = _decode_notices(db)
            db.commit()
            return current, notices

        if raw.get("schema_version") != 2:
            raise WorkflowMigrationConflict(
                "runtime settings v2 authority is required before v3 migration"
            )
        try:
            legacy = RuntimeSettingsV2.model_validate(raw)
        except ValidationError as exc:
            raise WorkflowMigrationConflict(
                "runtime settings v2 document is invalid"
            ) from exc

        collapsed_overrides: dict[str, LoraLoaderOverrideRecord] = {}
        for record in legacy.legacy_lora_resolution_compat.explicit_overrides:
            candidate = LoraLoaderOverrideRecord(
                lora_filename=record.lora_filename,
                adapter_id=_LEGACY_LOADER_ADAPTER_IDS[record.loader],
                options={},
            )
            previous = collapsed_overrides.get(record.lora_filename)
            if previous is None or (
                candidate.adapter_id == "minimax_h3_turbo"
                and previous.adapter_id != "minimax_h3_turbo"
            ):
                collapsed_overrides[record.lora_filename] = candidate
        overrides = list(collapsed_overrides.values())
        migrated = RuntimeSettingsV3(
            schema_version=3,
            client_id=legacy.client_id,
            memory_policy=legacy.memory_policy,
            raylight_residency_policy=legacy.raylight_residency_policy,
            multi_gpu_enabled=legacy.multi_gpu_enabled,
            placement=legacy.placement,
            lora_loader_overrides=overrides,
        )
        notice = RuntimeSettingsMigrationNotice(
            id=_AUTO_RESOLUTION_NOTICE_ID,
            code="legacy_lora_resolution_review_required",
            action="review_lora_loader_mappings",
            legacy_strategy_version=(
                legacy.legacy_lora_resolution_compat.auto_resolution_strategy_version
            ),
            message=(
                "Legacy incomplete, filename-based, or metadata-based Standard "
                "LoRA resolution was retired. Review LoRA loader mappings; "
                "unmapped LoRAs now use the configured default model-only loader."
            ),
            created_at=_notice_created_at(created_at),
        )
        revision = int(row["revision"])
        serialized = migrated.model_dump_json()
        cursor = db.execute(
            "UPDATE settings SET document = ?, updated_at = ?, "
            "revision = revision + 1 WHERE singleton = 1 AND revision = ?",
            (serialized, created_at, revision),
        )
        if cursor.rowcount != 1:
            raise WorkflowMigrationConflict(
                "runtime settings authority changed during v3 migration"
            )
        db.execute(
            "INSERT INTO runtime_settings_migration_notices"
            "(id, document, created_at) VALUES(?, ?, ?)",
            (notice.id, notice.model_dump_json(), created_at),
        )
        db.commit()
        return migrated, [notice]
    except Exception:
        db.rollback()
        raise


__all__ = [
    "RUNTIME_SETTINGS_V2_V3_MIGRATION_VERSION",
    "ensure_runtime_settings_migration_notice_schema",
    "migrate_runtime_settings_v2_to_v3",
]
