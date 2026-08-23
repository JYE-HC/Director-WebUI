from __future__ import annotations

"""Atomic persisted-authority upgrade from feature bundle 4 to bundle 5.

This migration is deliberately separate from the frozen timeline schema-4 to
schema-5 receipt migration.  The latter preserves its historical destination
bytes and receipt digests; this pass upgrades only schema-5 feature authority
and advances each affected authority's CAS revision exactly once.
"""

import json
import sqlite3
from typing import Any

from pydantic import ValidationError

from ..schemas import MAX_TIMELINE_REVISION, UnifiedTimelineDraftV5
from ..workflow.effective_features import (
    V5FeatureConfigurationError,
    migrate_timeline_feature_authority_to_v5,
)
from ..workflow.templates import V5_TEMPLATE_BUNDLE


FEATURE_BUNDLE_V4_V5_MIGRATION_VERSION = 1


class FeatureBundleMigrationConflict(RuntimeError):
    """One persisted creative authority cannot be upgraded safely."""

    def __init__(
        self,
        message: str,
        *,
        project_id: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"feature bundle migration failed for {project_id}: {message}")
        self.project_id = project_id
        self.details = dict(details or {})


def _authorities(
    db: sqlite3.Connection,
) -> list[tuple[str, str, str, int]]:
    rows: list[tuple[str, str, str, int]] = []
    default = db.execute(
        "SELECT document, revision FROM unified_timeline WHERE singleton = 1"
    ).fetchone()
    if default is None:
        raise FeatureBundleMigrationConflict(
            "the default authority is missing",
            project_id="default",
        )
    rows.append(("timeline", "default", str(default["document"]), int(default["revision"])))
    rows.extend(
        (
            "project",
            str(row["id"]),
            str(row["document"]),
            int(row["revision"]),
        )
        for row in db.execute(
            "SELECT id, document, revision FROM projects ORDER BY id"
        ).fetchall()
    )
    return rows


def migrate_feature_bundle_v4_authorities_to_v5(
    db: sqlite3.Connection,
    *,
    created_at: str,
) -> tuple[str, ...]:
    """Upgrade all schema-v5 bundle-4 authorities in one transaction.

    Bundle-5 authorities are still validated against the current resolver but
    remain byte/revision untouched.  Any invalid authority rolls back every
    bundle change, leaving a deterministic retry path on the next initialize.
    """

    migrated: list[str] = []
    try:
        db.execute("BEGIN IMMEDIATE")
        for kind, project_id, raw_document, revision in _authorities(db):
            try:
                decoded = json.loads(raw_document)
                document = UnifiedTimelineDraftV5.model_validate(decoded)
            except (json.JSONDecodeError, TypeError, ValidationError) as exc:
                raise FeatureBundleMigrationConflict(
                    "the authority is not a valid timeline schema 5 document",
                    project_id=project_id,
                ) from exc
            try:
                current = migrate_timeline_feature_authority_to_v5(document)
            except V5FeatureConfigurationError as exc:
                raise FeatureBundleMigrationConflict(
                    "the feature configuration has no deterministic current migration",
                    project_id=project_id,
                    details={
                        "code": exc.code,
                        "feature_id": exc.feature_id,
                        **exc.safe_details,
                    },
                ) from exc
            source_bundle = document.features.template_bundle_version
            if source_bundle == V5_TEMPLATE_BUNDLE.version:
                continue
            if source_bundle != 4:  # pragma: no cover - guarded by resolver.
                raise FeatureBundleMigrationConflict(
                    "the feature bundle version is unsupported",
                    project_id=project_id,
                    details={"template_bundle_version": source_bundle},
                )
            if revision >= MAX_TIMELINE_REVISION:
                raise FeatureBundleMigrationConflict(
                    "the authority revision is exhausted",
                    project_id=project_id,
                    details={"revision": revision},
                )
            if kind == "timeline":
                cursor = db.execute(
                    "UPDATE unified_timeline SET document = ?, updated_at = ?, "
                    "revision = revision + 1 WHERE singleton = 1 AND revision = ?",
                    (current.model_dump_json(), created_at, revision),
                )
            else:
                cursor = db.execute(
                    "UPDATE projects SET document = ?, title = ?, updated_at = ?, "
                    "revision = revision + 1 WHERE id = ? AND revision = ?",
                    (
                        current.model_dump_json(),
                        current.title,
                        created_at,
                        project_id,
                        revision,
                    ),
                )
            if cursor.rowcount != 1:
                raise FeatureBundleMigrationConflict(
                    "the authority CAS update did not affect exactly one row",
                    project_id=project_id,
                )
            migrated.append(project_id)
        db.commit()
        return tuple(migrated)
    except BaseException:
        db.rollback()
        raise


__all__ = [
    "FEATURE_BUNDLE_V4_V5_MIGRATION_VERSION",
    "FeatureBundleMigrationConflict",
    "migrate_feature_bundle_v4_authorities_to_v5",
]
