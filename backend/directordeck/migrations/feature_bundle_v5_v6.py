from __future__ import annotations

"""Per-authority Bundle-5 to Bundle-6 migration.

Every authority owns one short transaction.  Conflicting projects remain
byte-for-byte Bundle 5; they never prevent another project or the default from
moving forward.
"""

from dataclasses import dataclass
import hashlib
import json
import sqlite3
from typing import Literal

from pydantic import ValidationError

from ..schemas import (
    MAX_TIMELINE_REVISION,
    UnifiedTimelineDraftV5,
    default_model_stack,
    default_timeline_draft_v6,
)
from ..workflow.v6_projection import (
    V5V6ProjectionError,
    project_v5_authority_to_v6,
)


FEATURE_BUNDLE_V5_V6_MIGRATION_VERSION = 1
_NOTICE_PREFIX = "feature-bundle-v5-v6"


@dataclass(frozen=True, slots=True)
class FeatureBundleV5V6MigrationOutcome:
    project_id: str
    status: Literal[
        "migrated",
        "already_v6",
        "retained_v5",
        "default_replaced",
        "damaged",
    ]
    old_revision: int
    new_revision: int
    source_digest: str
    destination_digest: str | None = None
    backup_project_id: str | None = None
    reason_code: str | None = None


def _digest(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _notice_id(project_id: str, code: str) -> str:
    return f"{feature_bundle_migration_notice_prefix(project_id)}:{code}"


def feature_bundle_migration_notice_prefix(project_id: str) -> str:
    owner = hashlib.sha256(project_id.encode("utf-8")).hexdigest()[:24]
    return f"{_NOTICE_PREFIX}:{owner}"


def _record_notice(
    db: sqlite3.Connection,
    *,
    project_id: str,
    code: str,
    message: str,
    created_at: str,
) -> None:
    db.execute(
        "INSERT OR IGNORE INTO migration_notices(id, message, created_at) "
        "VALUES(?, ?, ?)",
        (_notice_id(project_id, code), message[:512], created_at),
    )


def _row(
    db: sqlite3.Connection,
    kind: Literal["timeline", "project"],
    project_id: str,
) -> sqlite3.Row | None:
    if kind == "timeline":
        return db.execute(
            "SELECT document, revision, updated_at FROM unified_timeline "
            "WHERE singleton = 1"
        ).fetchone()
    return db.execute(
        "SELECT document, revision, updated_at FROM projects WHERE id = ?",
        (project_id,),
    ).fetchone()


def _update(
    db: sqlite3.Connection,
    *,
    kind: Literal["timeline", "project"],
    project_id: str,
    document: UnifiedTimelineDraftV5,
    revision: int,
    created_at: str,
) -> None:
    if kind == "timeline":
        cursor = db.execute(
            "UPDATE unified_timeline SET document = ?, updated_at = ?, "
            "revision = revision + 1 WHERE singleton = 1 AND revision = ?",
            (document.model_dump_json(), created_at, revision),
        )
    else:
        cursor = db.execute(
            "UPDATE projects SET document = ?, title = ?, updated_at = ?, "
            "revision = revision + 1 WHERE id = ? AND revision = ?",
            (
                document.model_dump_json(),
                document.title,
                created_at,
                project_id,
                revision,
            ),
        )
    if cursor.rowcount != 1:
        raise sqlite3.IntegrityError("feature bundle migration CAS failed")


def _replace_default(
    db: sqlite3.Connection,
    *,
    source: UnifiedTimelineDraftV5 | None,
    raw_document: str,
    revision: int,
    source_updated_at: str,
    created_at: str,
    reason_code: str,
) -> tuple[UnifiedTimelineDraftV5, str]:
    source_digest = _digest(raw_document)
    backup_id = f"bundle5-default-{source_digest[:32]}"
    title = (
        f"{source.title}（Bundle 5 迁移保留）"
        if source is not None
        else "旧默认项目（迁移保留）"
    )
    db.execute(
        "INSERT OR IGNORE INTO projects(id, title, document, created_at, updated_at) "
        "VALUES(?, ?, ?, ?, ?)",
        (backup_id, title[:256], raw_document, source_updated_at, source_updated_at),
    )
    backup = db.execute(
        "SELECT document FROM projects WHERE id = ?", (backup_id,)
    ).fetchone()
    if backup is None or str(backup["document"]) != raw_document:
        raise sqlite3.IntegrityError("default migration backup is inconsistent")

    fresh = default_timeline_draft_v6(
        source.model_stack if source is not None else default_model_stack()
    )
    fresh_segment = fresh.segments[0].model_copy(
        update={"id": f"timeline-segment-{source_digest[:32]}"}
    )
    fresh = fresh.model_copy(update={"segments": [fresh_segment]}, deep=True)
    _update(
        db,
        kind="timeline",
        project_id="default",
        document=fresh,
        revision=revision,
        created_at=created_at,
    )
    _record_notice(
        db,
        project_id=backup_id,
        code=reason_code,
        message=(
            "该项目保留了无法无歧义迁移的 Bundle 5 默认配置；"
            "其他项目与 Bundle 6 默认不受影响。"
        ),
        created_at=created_at,
    )
    return fresh, backup_id


def _migrate_one(
    db: sqlite3.Connection,
    *,
    kind: Literal["timeline", "project"],
    project_id: str,
    created_at: str,
) -> FeatureBundleV5V6MigrationOutcome | None:
    db.execute("BEGIN IMMEDIATE")
    try:
        row = _row(db, kind, project_id)
        if row is None:
            db.rollback()
            return None
        raw = str(row["document"])
        revision = int(row["revision"])
        source_digest = _digest(raw)
        try:
            source = UnifiedTimelineDraftV5.model_validate(json.loads(raw))
        except (json.JSONDecodeError, TypeError, ValidationError):
            source = None

        if source is not None and source.features.template_bundle_version == 6:
            try:
                project_v5_authority_to_v6(source)
            except V5V6ProjectionError as exc:
                reason_code = exc.code
            else:
                db.commit()
                return FeatureBundleV5V6MigrationOutcome(
                    project_id=project_id,
                    status="already_v6",
                    old_revision=revision,
                    new_revision=revision,
                    source_digest=source_digest,
                    destination_digest=source_digest,
                )
        elif source is not None and source.features.template_bundle_version == 5:
            try:
                projection = project_v5_authority_to_v6(source)
            except V5V6ProjectionError as exc:
                reason_code = exc.code
            else:
                if revision < MAX_TIMELINE_REVISION:
                    _update(
                        db,
                        kind=kind,
                        project_id=project_id,
                        document=projection.draft,
                        revision=revision,
                        created_at=created_at,
                    )
                    for index, notice in enumerate(projection.notices):
                        _record_notice(
                            db,
                            project_id=project_id,
                            code=f"behavior-change-{index + 1}",
                            message=notice,
                            created_at=created_at,
                        )
                    db.commit()
                    destination = projection.draft.model_dump_json()
                    return FeatureBundleV5V6MigrationOutcome(
                        project_id=project_id,
                        status="migrated",
                        old_revision=revision,
                        new_revision=revision + 1,
                        source_digest=source_digest,
                        destination_digest=_digest(destination),
                    )
                reason_code = "timeline_revision_exhausted"
        else:
            reason_code = "authority_invalid"

        if kind == "timeline" and revision < MAX_TIMELINE_REVISION:
            replacement, backup_id = _replace_default(
                db,
                source=source,
                raw_document=raw,
                revision=revision,
                source_updated_at=str(row["updated_at"]),
                created_at=created_at,
                reason_code=reason_code,
            )
            db.commit()
            return FeatureBundleV5V6MigrationOutcome(
                project_id=project_id,
                status="default_replaced",
                old_revision=revision,
                new_revision=revision + 1,
                source_digest=source_digest,
                destination_digest=_digest(replacement.model_dump_json()),
                backup_project_id=backup_id,
                reason_code=reason_code,
            )

        _record_notice(
            db,
            project_id=project_id,
            code=reason_code,
            message=(
                "该项目的 Bundle 5 配置无法无歧义迁移，原始数据已保留；"
                "仅该项目继续使用兼容路径。"
                if source is not None
                else "该项目配置无法自动迁移，原始数据已保留；仅该项目需要处理。"
            ),
            created_at=created_at,
        )
        db.commit()
        return FeatureBundleV5V6MigrationOutcome(
            project_id=project_id,
            status="retained_v5" if source is not None else "damaged",
            old_revision=revision,
            new_revision=revision,
            source_digest=source_digest,
            reason_code=reason_code,
        )
    except Exception:
        db.rollback()
        raise


def migrate_feature_bundle_v5_authorities_to_v6(
    db: sqlite3.Connection,
    *,
    created_at: str,
) -> tuple[FeatureBundleV5V6MigrationOutcome, ...]:
    """Migrate each live authority independently; never create a global gate."""

    if db.in_transaction:
        raise RuntimeError("feature bundle migration requires an idle connection")
    project_ids = tuple(
        str(row["id"])
        for row in db.execute("SELECT id FROM projects ORDER BY id").fetchall()
    )
    outcomes: list[FeatureBundleV5V6MigrationOutcome] = []
    for kind, project_id in (
        ("timeline", "default"),
        *(("project", value) for value in project_ids),
    ):
        try:
            outcome = _migrate_one(
                db,
                kind=kind,  # type: ignore[arg-type]
                project_id=project_id,
                created_at=created_at,
            )
        except (sqlite3.Error, RuntimeError, ValueError):
            # A row-local failure is retried on the next startup.  The raw row
            # was rolled back and must not prevent subsequent authorities.
            try:
                db.execute("BEGIN IMMEDIATE")
                row = _row(db, kind, project_id)  # type: ignore[arg-type]
                if row is None:
                    db.rollback()
                    continue
                raw = str(row["document"])
                revision = int(row["revision"])
                _record_notice(
                    db,
                    project_id=project_id,
                    code="migration-transaction-failed",
                    message=(
                        "该项目迁移事务未完成，原始数据已回滚并保留；"
                        "其他项目不受影响，后续启动会重试。"
                    ),
                    created_at=created_at,
                )
                db.commit()
                outcome = FeatureBundleV5V6MigrationOutcome(
                    project_id=project_id,
                    status="damaged",
                    old_revision=revision,
                    new_revision=revision,
                    source_digest=_digest(raw),
                    reason_code="migration_transaction_failed",
                )
            except (sqlite3.Error, RuntimeError, ValueError):
                db.rollback()
                continue
        if outcome is not None:
            outcomes.append(outcome)
    return tuple(outcomes)


__all__ = [
    "FEATURE_BUNDLE_V5_V6_MIGRATION_VERSION",
    "FeatureBundleV5V6MigrationOutcome",
    "feature_bundle_migration_notice_prefix",
    "migrate_feature_bundle_v5_authorities_to_v6",
]
