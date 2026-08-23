from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .compiler import (
    segment_take_geometry_fingerprint,
    timeline_segment_take_fingerprint,
    unified_continuity_predecessors,
)
from .execution.submission import (
    SubmissionPlanningError,
    validate_locked_submission_transition,
)
from .native_templates import (
    NativeTemplateError,
    migrate_legacy_raylight_runtime_descriptor,
    normalize_native_output_descriptor,
    raylight_runtime_logical_gpu_indices,
)
from .migrations.timeline_v4_v5 import (
    ProjectMigrationReceipt,
    RuntimeSettingsSchemaMigrated,
    TimelineSchemaMigrated,
    decode_project_migration_receipt,
    migrate_v4_authorities_to_v5,
)
from .migrations.runtime_settings_v2_v3 import (
    migrate_runtime_settings_v2_to_v3,
)
from .migrations.feature_bundle_v4_v5 import (
    migrate_feature_bundle_v4_authorities_to_v5,
)
from .schemas import (
    AssetReference,
    GenerationMode,
    MAX_TIMELINE_REVISION,
    MODE_ORDER,
    ModelStack,
    ModeDraft,
    RuntimeSettings,
    RuntimeSettingsMigrationNotice,
    RuntimeSettingsV2,
    RuntimeSettingsV3,
    UnifiedTimelineDraft,
    UnifiedTimelineDraftV5,
    UnifiedTimelineSegment,
    canonicalize_live_runtime_settings,
    default_draft,
    default_model_stack,
    default_runtime_settings_v3,
    default_settings,
    default_timeline_draft,
    default_timeline_draft_v5,
    iter_draft_assets,
    iter_timeline_assets,
    migrate_mode_drafts_to_timeline,
    mode_draft_is_default,
    utc_now,
    validate_mode_draft,
    validate_timeline_draft,
    validate_timeline_snapshot,
    validate_timeline_draft_v5,
)
from .workflow.audit import GraphAuditError
from .workflow.execution import (
    CompiledExecutionPlan,
    ContinuityLateBindingEvidence,
    EndpointIdentity,
    EndpointRestartCertificate,
    ExactCancelConfirmedEvidence,
    ExactPromptSnapshot,
    HistoryTerminalEvidence,
    LockedSegmentUnit,
    LockedSubmissionPlan,
    ObservedArtifactSpec,
    ObservedAssemblyArtifactSpec,
    OutputObservationReceipt,
    PreparedControlUnit,
    PreparedSegmentUnit,
    PromptOwnership,
    PromptOwnershipState,
    PromptReleaseEvidence,
    RuntimeEpochLateBindingEvidence,
    canonical_json,
    canonical_values_equal,
    compiled_execution_plan_digest,
    compiled_execution_plan_digest_from_canonical_json,
    ordered_compiled_segment_units,
    sha256_document_digest,
    transition_prompt_ownership,
)
from .workflow.node_contracts import (
    CURRENT_NODE_CONTRACT_REGISTRY,
    V4_NODE_CONTRACT_REGISTRY,
)
from .workflow.templates import CURRENT_TEMPLATE_BUNDLE, V4_TEMPLATE_BUNDLE
from .workflow.effective_features import (
    migrate_timeline_feature_authority_to_v5,
)


# The task list deliberately excludes internal prompt/workflow snapshots. The
# public projection still needs the immutable timeline/runtime snapshots for
# display identity and currentness, plus child output-node mappings for stable
# segment results, but never exposes or interprets the executable prompt graph.
_JOB_LIST_COLUMNS = ", ".join(
    (
        "id",
        "mode",
        "status",
        "progress",
        "stage",
        "prompt_id",
        "project_id",
        "outputs",
        "error",
        "config_snapshot",
        "settings_snapshot",
        "created_at",
        "updated_at",
        "started_at",
        "completed_at",
    )
)
_JOB_CHILD_LIST_COLUMNS = ", ".join(
    (
        "id",
        "job_id",
        "family",
        "backend",
        "segment_ids",
        "output_nodes",
        "status",
        "progress",
        "stage",
        "prompt_id",
        "outputs",
        "error",
    )
)


class TimelineRevisionConflict(RuntimeError):
    """A conditional timeline write was based on an obsolete server revision."""

    def __init__(
        self,
        project_id: str,
        expected_revision: int,
        actual_revision: int,
    ) -> None:
        super().__init__(
            f"timeline revision conflict for {project_id}: "
            f"expected {expected_revision}, actual {actual_revision}"
        )
        self.project_id = project_id
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision


class TimelineRevisionExhausted(OverflowError):
    """A timeline can no longer advance its JSON-safe durable revision."""

    def __init__(self, project_id: str, revision: int) -> None:
        super().__init__(
            f"timeline revision space is exhausted for {project_id} at {revision}"
        )
        self.project_id = project_id
        self.revision = revision


class SettingsAuthorityConflict(RuntimeError):
    """A conditional RuntimeSettingsV3 write used an obsolete authority."""

    code = "runtime_settings_authority_conflict"

    def __init__(self, expected_authority_token: str) -> None:
        super().__init__("runtime settings authority changed on the server")
        self.expected_authority_token = expected_authority_token


class SettingsAuthorityRequired(RuntimeError):
    """RuntimeSettingsV3 may not be persisted with latest-wins semantics."""

    code = "runtime_settings_authority_required"


class AssetTrashInUse(RuntimeError):
    """One or more assets are still referenced and cascade was not requested."""

    def __init__(self, usages_by_asset: dict[str, list[str]]) -> None:
        super().__init__("assets are still referenced by saved drafts")
        self.usages_by_asset = usages_by_asset

    @property
    def usages(self) -> list[str]:
        return [
            usage
            for usages in self.usages_by_asset.values()
            for usage in usages
        ]


class AssetTrashRestoreConflict(RuntimeError):
    """An inverse bundle no longer matches the post-trash authority."""

    def __init__(self, conflicts: list[dict[str, Any]]) -> None:
        super().__init__("asset references changed after the trash operation")
        self.conflicts = conflicts


class ExecutionEvidenceConflict(RuntimeError):
    """Persisted execution evidence no longer matches a submission intent."""


class RayRuntimeIntentConflict(ExecutionEvidenceConflict):
    """The Ray runtime ledger changed before an intent could be committed."""


_DATABASE_UNSET = object()


def _copy_sqlite_database(
    source: sqlite3.Connection, destination: sqlite3.Connection
) -> None:
    source.backup(destination)


def _sqlite_quick_check_is_ok(connection: sqlite3.Connection) -> bool:
    check = connection.execute("PRAGMA quick_check").fetchone()
    return check is not None and check[0] == "ok"


def _fsync_file(path: Path) -> None:
    # Windows os.fsync (_commit) requires a writable handle; a read-only
    # descriptor fails with EBADF.  POSIX fsync accepts either, so open
    # read-write on every platform.
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        # Windows cannot fsync a directory (os.open on a directory fails
        # outright), so the post-rename metadata flush is POSIX-only.  The
        # file payload itself was fsynced before the atomic replace.
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _commit_database(connection: sqlite3.Connection) -> None:
    connection.commit()


def _remove_sqlite_artifacts(path: Path) -> None:
    first_error: OSError | None = None
    for suffix in ("-wal", "-shm", "-journal", ""):
        try:
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        except OSError as exc:
            # Still attempt the main file if deleting one transient sidecar
            # fails; a sidecar error must not preserve a final-looking backup.
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _create_crash_safe_sqlite_backup(source_path: Path, backup_path: Path) -> None:
    """Publish one validated backup durably, without exposing a partial file."""

    temporary_descriptor: int | None = None
    temporary_path: Path | None = None
    source_connection: sqlite3.Connection | None = None
    destination_connection: sqlite3.Connection | None = None
    published = False
    try:
        temporary_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{backup_path.name}.",
            suffix=".tmp",
            dir=backup_path.parent,
        )
        temporary_path = Path(temporary_name)
        os.close(temporary_descriptor)
        temporary_descriptor = None

        source_uri = f"{source_path.resolve().as_uri()}?mode=ro"
        source_connection = sqlite3.connect(
            source_uri,
            uri=True,
            timeout=30,
        )
        destination_connection = sqlite3.connect(
            temporary_path,
            timeout=30,
        )
        _copy_sqlite_database(source_connection, destination_connection)
        destination_connection.commit()
        if not _sqlite_quick_check_is_ok(destination_connection):
            raise sqlite3.DatabaseError(
                "RayLight recovery backup failed SQLite quick_check"
            )
        destination_connection.close()
        destination_connection = None
        source_connection.close()
        source_connection = None

        os.chmod(temporary_path, 0o600)
        _fsync_file(temporary_path)
        _atomic_replace(temporary_path, backup_path)
        temporary_path = None
        published = True
        _fsync_directory(backup_path.parent)
    except (OSError, sqlite3.Error) as exc:
        if destination_connection is not None:
            try:
                destination_connection.close()
            except sqlite3.Error:
                pass
        if source_connection is not None:
            try:
                source_connection.close()
            except sqlite3.Error:
                pass
        if temporary_descriptor is not None:
            try:
                os.close(temporary_descriptor)
            except OSError:
                pass
        if temporary_path is not None:
            try:
                _remove_sqlite_artifacts(temporary_path)
            except OSError:
                pass
        if published:
            try:
                _remove_sqlite_artifacts(backup_path)
            except OSError:
                pass
            try:
                _fsync_directory(backup_path.parent)
            except OSError:
                pass
        raise RuntimeError(
            "Director could not create the RayLight recovery backup"
        ) from exc


class _ClosingConnection(sqlite3.Connection):
    """Commit/rollback like sqlite3's context manager, then always close.

    ``sqlite3.Connection.__exit__`` deliberately leaves the descriptor open.
    Every Database operation uses ``with self.connect()``, so using the stock
    connection there leaked one file descriptor per request until process exit.
    """

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        try:
            return bool(super().__exit__(exc_type, exc, traceback))
        finally:
            self.close()


class Database:
    _RAYLIGHT_RESIDENCY_MIGRATION_ID = (
        "raylight-residency-keyed-switch-v2"
    )
    # Stable id of the first project created from the pre-multi-project
    # singleton timeline. Legacy endpoints delegate to it; new projects get
    # server-generated UUIDs and fresh segment ids.
    LEGACY_DEFAULT_PROJECT_ID = "default"
    _REFERENCE_TAG = re.compile(
        r"<\s*(Picture|Audio|Video)\s+([0-9]+)\s*>", re.IGNORECASE
    )
    _REFERENCE_FIELDS = {
        "reference_images": "Picture",
        "reference_audios": "Audio",
        "reference_videos": "Video",
    }
    _TERMINAL_JOB_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
    _RESTART_RECOVERY_STAGES = frozenset(
        {
            "restart_cancel_pending",
            "restart_cancel_failed",
            "restart_cancel_unconfirmed",
            "restart_certificate_required",
        }
    )
    _CONFIRMED_COMFY_RESTART_STAGE = "cancelled_after_confirmed_comfy_restart"
    _EXECUTION_EVIDENCE_TABLES = frozenset(
        {
            "job_execution_plans",
            "job_child_execution_evidence",
            "prompt_ownership",
        }
    )
    _TYPED_EXECUTION_CONTRACT_VERSION = 1
    _TYPED_EXECUTION_MARKER_COLUMN = "execution_contract_version"
    _OBSERVED_SEGMENT_TAKE_SELECT = (
        "SELECT t.*, "
        "a.take_id AS observation_take_id, "
        "a.source_child_id AS observation_source_child_id, "
        "a.schema_version AS observation_schema_version, "
        "a.observed_artifact, "
        "a.observed_artifact_digest, "
        "a.receipt_digest AS observation_receipt_digest, "
        "c.id AS live_child_id, "
        "c.job_id AS live_child_job_id, "
        "c.status AS live_child_status, "
        "j.id AS live_source_job_id, "
        "r.child_id AS receipt_child_id, "
        "r.schema_version AS receipt_schema_version, "
        "r.receipt, "
        "r.receipt_digest AS durable_receipt_digest, "
        "e.child_id AS exact_child_id, "
        "e.schema_version AS exact_schema_version, "
        "e.unit_id AS exact_unit_id, "
        "e.unit_kind AS exact_unit_kind, "
        "e.endpoint_key AS exact_endpoint_key, "
        "e.endpoint_runtime_instance_id AS exact_runtime_instance_id, "
        "e.exact_prompt_snapshot, "
        "e.exact_prompt_snapshot_digest, "
        "o.requested_prompt_id AS observed_ownership_requested_id, "
        "o.actual_prompt_id AS observed_ownership_actual_id, "
        "o.state AS observed_ownership_state, "
        "o.ownership_revision AS observed_ownership_revision, "
        "o.cleanup_certificate AS observed_ownership_certificate, "
        "o.updated_at AS observed_ownership_updated_at"
    )

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=30,
            factory=_ClosingConnection,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @classmethod
    def _execution_evidence_schema_exists(
        cls, db: sqlite3.Connection
    ) -> bool:
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name IN ('job_execution_plans', "
            "'job_child_execution_evidence', 'prompt_ownership')"
        ).fetchall()
        present = {str(row["name"]) for row in rows}
        if not present:
            return False
        if present != cls._EXECUTION_EVIDENCE_TABLES:
            missing = sorted(cls._EXECUTION_EVIDENCE_TABLES - present)
            raise RuntimeError(
                "Director execution evidence schema is incomplete: "
                + ", ".join(missing)
            )
        return True

    @classmethod
    def _typed_execution_marker_column_exists(
        cls, db: sqlite3.Connection
    ) -> bool:
        return any(
            str(row["name"]) == cls._TYPED_EXECUTION_MARKER_COLUMN
            for row in db.execute("PRAGMA table_info(jobs)").fetchall()
        )

    @staticmethod
    def _ensure_execution_evidence_schema(db: sqlite3.Connection) -> None:
        """Create Stage-4-only tables inside the caller's write transaction.

        These objects intentionally do not belong to ``initialize()``.  The
        immutable phase-0 database fixture must still project to its frozen
        schema after ordinary startup; the first real Stage-4 plan write both
        creates this additive schema and persists its first row atomically.
        Individual ``execute`` calls are used instead of ``executescript``
        because the latter would implicitly commit an active transaction.
        """

        db.execute(
            """
            CREATE TABLE IF NOT EXISTS job_execution_plans (
                job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
                schema_version INTEGER NOT NULL CHECK (schema_version >= 1),
                compiled_plan TEXT NOT NULL,
                compiled_plan_digest TEXT NOT NULL,
                unit_index_version INTEGER NOT NULL DEFAULT 1
                    CHECK (unit_index_version IN (0, 1)),
                created_at TEXT NOT NULL
            )
            """
        )
        plan_columns = {
            str(row["name"])
            for row in db.execute(
                "PRAGMA table_info(job_execution_plans)"
            ).fetchall()
        }
        if "unit_index_version" not in plan_columns:
            # Upgrade the pre-index Stage-4 development schema. The one
            # permitted 0 -> 1 transition below occurs only after every unit
            # has been derived from authenticated immutable plan bytes.
            db.execute("DROP TRIGGER IF EXISTS job_execution_plans_immutable")
            db.execute(
                "ALTER TABLE job_execution_plans ADD COLUMN "
                "unit_index_version INTEGER NOT NULL DEFAULT 0 "
                "CHECK (unit_index_version IN (0, 1))"
            )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS job_execution_plan_units (
                job_id TEXT NOT NULL
                    REFERENCES job_execution_plans(job_id) ON DELETE CASCADE,
                unit_ordinal INTEGER NOT NULL CHECK (unit_ordinal >= 0),
                unit_id TEXT NOT NULL,
                schema_version INTEGER NOT NULL CHECK (schema_version >= 1),
                source_compiled_plan_digest TEXT NOT NULL,
                prepared_unit TEXT NOT NULL,
                prepared_unit_digest TEXT NOT NULL,
                PRIMARY KEY (job_id, unit_ordinal),
                UNIQUE (job_id, unit_id)
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS job_child_execution_evidence (
                child_id TEXT PRIMARY KEY
                    REFERENCES job_children(id) ON DELETE CASCADE,
                schema_version INTEGER NOT NULL CHECK (schema_version >= 1),
                unit_id TEXT NOT NULL,
                unit_kind TEXT NOT NULL
                    CHECK (unit_kind IN ('segment', 'control')),
                endpoint_key TEXT NOT NULL,
                endpoint_runtime_instance_id TEXT NOT NULL,
                source_compiled_plan_digest TEXT NOT NULL,
                locked_submission_plan TEXT NOT NULL,
                locked_submission_plan_digest TEXT NOT NULL,
                exact_prompt_snapshot TEXT NOT NULL,
                exact_prompt_snapshot_digest TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS job_child_execution_unit_idx "
            "ON job_child_execution_evidence(unit_id)"
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_ownership (
                child_id TEXT PRIMARY KEY
                    REFERENCES job_children(id) ON DELETE CASCADE,
                requested_prompt_id TEXT NOT NULL,
                actual_prompt_id TEXT,
                state TEXT NOT NULL CHECK (state IN (
                    'prepared', 'submitting', 'owned_requested_id',
                    'owned_actual_id', 'cancel_pending', 'cleanup_confirmed',
                    'terminal_confirmed', 'unconfirmed'
                )),
                ownership_revision INTEGER NOT NULL
                    CHECK (ownership_revision >= 0),
                cleanup_certificate TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS prompt_ownership_requested_idx "
            "ON prompt_ownership(requested_prompt_id)"
        )
        db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS prompt_ownership_actual_idx "
            "ON prompt_ownership(actual_prompt_id) "
            "WHERE actual_prompt_id IS NOT NULL"
        )
        db.execute(
            """
            CREATE TRIGGER IF NOT EXISTS job_execution_plans_immutable
            BEFORE UPDATE ON job_execution_plans
            WHEN NOT (
                OLD.unit_index_version = 0
                AND NEW.unit_index_version = 1
                AND NEW.job_id IS OLD.job_id
                AND NEW.schema_version IS OLD.schema_version
                AND NEW.compiled_plan IS OLD.compiled_plan
                AND NEW.compiled_plan_digest IS OLD.compiled_plan_digest
                AND NEW.created_at IS OLD.created_at
            )
            BEGIN
                SELECT RAISE(ABORT, 'job execution plan is immutable');
            END
            """
        )
        db.execute(
            """
            CREATE TRIGGER IF NOT EXISTS job_execution_plan_units_immutable
            BEFORE UPDATE ON job_execution_plan_units
            BEGIN
                SELECT RAISE(ABORT, 'job execution plan unit is immutable');
            END
            """
        )
        db.execute(
            """
            CREATE TRIGGER IF NOT EXISTS job_child_execution_evidence_immutable
            BEFORE UPDATE ON job_child_execution_evidence
            BEGIN
                SELECT RAISE(ABORT, 'job child execution evidence is immutable');
            END
            """
        )
        job_columns = {
            str(row["name"])
            for row in db.execute("PRAGMA table_info(jobs)").fetchall()
        }
        if Database._TYPED_EXECUTION_MARKER_COLUMN not in job_columns:
            db.execute(
                "ALTER TABLE jobs ADD COLUMN execution_contract_version INTEGER "
                "CHECK (execution_contract_version = 1)"
            )
        # Upgrade databases produced by an earlier Stage-4 development build.
        # The marker lives on the parent row itself, so losing every auxiliary
        # evidence row can never make that parent look like a legacy task.
        db.execute(
            "UPDATE jobs SET execution_contract_version = 1 "
            "WHERE execution_contract_version IS NULL AND id IN "
            "(SELECT job_id FROM job_execution_plans)"
        )
        db.execute(
            """
            CREATE TRIGGER IF NOT EXISTS jobs_execution_contract_marker_immutable
            BEFORE UPDATE OF execution_contract_version ON jobs
            WHEN OLD.execution_contract_version IS NOT NULL
                 AND NEW.execution_contract_version IS NOT OLD.execution_contract_version
            BEGIN
                SELECT RAISE(ABORT, 'job execution contract marker is immutable');
            END
            """
        )

    @staticmethod
    def _artifact_observation_schema_exists(db: sqlite3.Connection) -> bool:
        expected = {
            "job_child_output_receipts",
            "segment_take_observed_artifacts",
        }
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name IN ('job_child_output_receipts', "
            "'segment_take_observed_artifacts')"
        ).fetchall()
        present = {str(row["name"]) for row in rows}
        if not present:
            return False
        if present != expected:
            raise RuntimeError(
                "Director artifact observation schema is incomplete: "
                + ", ".join(sorted(expected - present))
            )
        return True

    @staticmethod
    def _ensure_artifact_observation_schema(db: sqlite3.Connection) -> None:
        """Create output-observation evidence only on the first typed receipt.

        Like the execution-evidence tables, these objects stay out of ordinary
        ``initialize()`` so the immutable Phase-0 database fixture is not
        silently migrated merely by starting DirectorDeck.
        """

        db.execute(
            """
            CREATE TABLE IF NOT EXISTS job_child_output_receipts (
                child_id TEXT PRIMARY KEY
                    REFERENCES job_children(id) ON DELETE CASCADE,
                schema_version INTEGER NOT NULL CHECK (schema_version = 1),
                receipt TEXT NOT NULL,
                receipt_digest TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS segment_take_observed_artifacts (
                take_id TEXT PRIMARY KEY
                    REFERENCES segment_takes(id) ON DELETE CASCADE,
                source_child_id TEXT NOT NULL UNIQUE,
                schema_version INTEGER NOT NULL CHECK (schema_version = 1),
                observed_artifact TEXT NOT NULL,
                observed_artifact_digest TEXT NOT NULL,
                receipt_digest TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        db.execute(
            """
            CREATE TRIGGER IF NOT EXISTS job_child_output_receipts_immutable
            BEFORE UPDATE ON job_child_output_receipts
            BEGIN
                SELECT RAISE(ABORT, 'job child output receipt is immutable');
            END
            """
        )
        db.execute(
            """
            CREATE TRIGGER IF NOT EXISTS segment_take_observed_artifacts_immutable
            BEFORE UPDATE ON segment_take_observed_artifacts
            BEGIN
                SELECT RAISE(ABORT, 'observed artifact is immutable');
            END
            """
        )

    @staticmethod
    def _assembly_artifact_schema_exists(db: sqlite3.Connection) -> bool:
        row = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'job_observed_assembly_artifacts'"
        ).fetchone()
        return row is not None

    @staticmethod
    def _ensure_assembly_artifact_schema(db: sqlite3.Connection) -> None:
        """Create parent assembly authority inside the caller's transaction."""

        db.execute(
            """
            CREATE TABLE IF NOT EXISTS job_observed_assembly_artifacts (
                job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
                schema_version INTEGER NOT NULL CHECK (schema_version = 1),
                source_compiled_plan_digest TEXT NOT NULL,
                observed_assembly_artifact TEXT NOT NULL,
                observed_assembly_artifact_digest TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        db.execute(
            """
            CREATE TRIGGER IF NOT EXISTS job_observed_assembly_artifacts_immutable
            BEFORE UPDATE ON job_observed_assembly_artifacts
            BEGIN
                SELECT RAISE(ABORT, 'observed assembly artifact is immutable');
            END
            """
        )

    def initialize(self) -> None:
        # The default database can be the first file created below the
        # project-local .data/database directory.
        # Request owner-only permissions for a newly-created directory and
        # pre-create a missing SQLite file without following a final symlink.
        # Existing directories/files deliberately retain their permissions.
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            descriptor = os.open(
                self.path,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        with self.connect() as db:
            # Distinguish a genuine upgrade from a fresh database.  On upgrade
            # the six legacy documents are folded into one timeline exactly
            # once; a fresh workspace starts with one focused T2V segment.
            existing_tables = {
                str(row["name"])
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            }
            fresh_database = not existing_tables
            had_mode_drafts_table = "mode_drafts" in existing_tables
            had_unified_timeline_table = "unified_timeline" in existing_tables
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    document TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0)
                );
                CREATE TABLE IF NOT EXISTS mode_drafts (
                    mode TEXT PRIMARY KEY CHECK (mode IN ('t2v','i2v','fl2v','r2v','v2v','rv2v')),
                    document TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS unified_timeline (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    document TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0
                        CHECK (revision >= 0 AND revision <= 9007199254740991)
                );
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    document TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0
                        CHECK (revision >= 0 AND revision <= 9007199254740991)
                );
                CREATE TABLE IF NOT EXISTS migration_notices (
                    id TEXT PRIMARY KEY,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runtime_settings_migration_notices (
                    id TEXT PRIMARY KEY,
                    document TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY,
                    document TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    trashed_at TEXT,
                    trash_batch_id TEXT
                );
                CREATE TABLE IF NOT EXISTS asset_trash_batches (
                    id TEXT PRIMARY KEY,
                    asset_ids TEXT NOT NULL,
                    cascade INTEGER NOT NULL CHECK (cascade IN (0, 1)),
                    unbound_usages TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS asset_trash_document_changes (
                    batch_id TEXT NOT NULL REFERENCES asset_trash_batches(id)
                        ON DELETE CASCADE,
                    owner_kind TEXT NOT NULL
                        CHECK (owner_kind IN ('timeline', 'project', 'draft')),
                    owner_id TEXT NOT NULL,
                    before_document TEXT NOT NULL,
                    after_digest TEXT NOT NULL,
                    after_revision INTEGER,
                    PRIMARY KEY(batch_id, owner_kind, owner_id)
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0
                        CHECK (cancel_requested IN (0, 1)),
                    progress REAL NOT NULL,
                    stage TEXT,
                    prompt_id TEXT,
                    project_id TEXT,
                    outputs TEXT NOT NULL DEFAULT '[]',
                    error TEXT,
                    config_snapshot TEXT NOT NULL,
                    settings_snapshot TEXT NOT NULL,
                    prompt_snapshot TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS jobs_prompt_id_idx ON jobs(prompt_id);
                CREATE INDEX IF NOT EXISTS jobs_created_at_idx ON jobs(created_at DESC);
                CREATE TABLE IF NOT EXISTS job_children (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    group_index INTEGER NOT NULL,
                    family TEXT NOT NULL CHECK (family IN ('fl2va','ref2va')),
                    backend TEXT NOT NULL CHECK (backend IN ('standard','raylight')),
                    segment_ids TEXT NOT NULL,
                    output_nodes TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    stage TEXT,
                    prompt_id TEXT,
                    outputs TEXT NOT NULL DEFAULT '[]',
                    error TEXT,
                    prompt_snapshot TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    UNIQUE(job_id, group_index)
                );
                CREATE INDEX IF NOT EXISTS job_children_job_idx
                    ON job_children(job_id, group_index);
                CREATE UNIQUE INDEX IF NOT EXISTS job_children_prompt_idx
                    ON job_children(prompt_id) WHERE prompt_id IS NOT NULL;
                CREATE TABLE IF NOT EXISTS segment_takes (
                    id TEXT PRIMARY KEY,
                    segment_id TEXT NOT NULL,
                    content_fingerprint TEXT NOT NULL,
                    project_id TEXT,
                    output_descriptor TEXT NOT NULL,
                    has_audio INTEGER NOT NULL CHECK (has_audio IN (0, 1)),
                    source_job_id TEXT NOT NULL,
                    source_child_id TEXT NOT NULL UNIQUE,
                    completed_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS segment_takes_lookup_idx
                    ON segment_takes(
                        segment_id,
                        content_fingerprint,
                        has_audio,
                        completed_at DESC,
                        id DESC
                    );
                CREATE TABLE IF NOT EXISTS raylight_runtime_state (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    descriptor TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            now = utc_now()
            timeline_columns = {
                str(row["name"])
                for row in db.execute(
                    "PRAGMA table_info(unified_timeline)"
                ).fetchall()
            }
            if "revision" not in timeline_columns:
                db.execute(
                    "ALTER TABLE unified_timeline ADD COLUMN revision INTEGER "
                    "NOT NULL DEFAULT 0 CHECK (revision >= 0 AND "
                    "revision <= 9007199254740991)"
                )
            project_columns = {
                str(row["name"])
                for row in db.execute("PRAGMA table_info(projects)").fetchall()
            }
            if "revision" not in project_columns:
                db.execute(
                    "ALTER TABLE projects ADD COLUMN revision INTEGER "
                    "NOT NULL DEFAULT 0 CHECK (revision >= 0 AND "
                    "revision <= 9007199254740991)"
                )
            job_columns = {
                str(row["name"])
                for row in db.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "cancel_requested" not in job_columns:
                db.execute(
                    "ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER "
                    "NOT NULL DEFAULT 0 CHECK (cancel_requested IN (0, 1))"
                )
                # Before this column existed, the plain cancelling/cancelling
                # pair was written only by the explicit cancel route. Internal
                # recovery and submission cleanup used distinct stage markers.
                db.execute(
                    "UPDATE jobs SET cancel_requested = 1 "
                    "WHERE status = 'cancelling' AND stage = 'cancelling'"
                )
            if "project_id" not in job_columns:
                db.execute("ALTER TABLE jobs ADD COLUMN project_id TEXT")
            if self._execution_evidence_schema_exists(db):
                # Stage-4 schemas are lazy so opening the frozen legacy fixture
                # remains byte/schema preserving.  Only databases that already
                # contain typed evidence receive additive contract-marker
                # upgrades here.
                self._ensure_execution_evidence_schema(db)
            take_columns = {
                str(row["name"])
                for row in db.execute("PRAGMA table_info(segment_takes)").fetchall()
            }
            if "project_id" not in take_columns:
                db.execute("ALTER TABLE segment_takes ADD COLUMN project_id TEXT")
            settings_columns = {
                str(row["name"])
                for row in db.execute("PRAGMA table_info(settings)").fetchall()
            }
            if "revision" not in settings_columns:
                # A wall-clock timestamp is not a CAS version: it can repeat or
                # move backwards. Every settings write advances this durable,
                # transactionally serialized revision so A -> B -> A remains
                # observable even when ``utc_now`` is frozen.
                db.execute(
                    "ALTER TABLE settings ADD COLUMN revision INTEGER "
                    "NOT NULL DEFAULT 0 CHECK (revision >= 0)"
                )
            settings: RuntimeSettings | RuntimeSettingsV2 | RuntimeSettingsV3 = (
                default_runtime_settings_v3()
                if fresh_database
                else default_settings()
            )
            db.execute(
                "INSERT OR IGNORE INTO settings(singleton, document, updated_at) VALUES(1, ?, ?)",
                (settings.model_dump_json(), now),
            )
            asset_columns = {
                str(row["name"])
                for row in db.execute("PRAGMA table_info(assets)").fetchall()
            }
            if "trashed_at" not in asset_columns:
                db.execute("ALTER TABLE assets ADD COLUMN trashed_at TEXT")
            if "trash_batch_id" not in asset_columns:
                db.execute("ALTER TABLE assets ADD COLUMN trash_batch_id TEXT")
            db.execute(
                "CREATE INDEX IF NOT EXISTS assets_live_idx "
                "ON assets(trashed_at, created_at DESC, id DESC)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS assets_trash_batch_idx "
                "ON assets(trash_batch_id) WHERE trash_batch_id IS NOT NULL"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS asset_trash_batches_created_idx "
                "ON asset_trash_batches(created_at DESC, id DESC)"
            )
            settings_row = db.execute(
                "SELECT document FROM settings WHERE singleton = 1"
            ).fetchone()
            if settings_row is None:
                raise RuntimeError("settings row is missing during asset migration")
            raw_settings = json.loads(settings_row["document"])
            try:
                if raw_settings.get("schema_version") == 3:
                    persisted_settings = RuntimeSettingsV3.model_validate(raw_settings)
                elif raw_settings.get("schema_version") == 2:
                    persisted_settings = RuntimeSettingsV2.model_validate(raw_settings)
                else:
                    # ``clear_between_segments`` belonged to the former custom-node
                    # executor. Native per-segment prompts rely on ComfyUI's stable
                    # loader cache between prompts; there is no stock per-segment
                    # unload contract, so upgrades normalize this obsolete value once.
                    if raw_settings.get("memory_policy") == "clear_between_segments":
                        raw_settings["memory_policy"] = "keep_resident"
                    # The former persistent policies pinned the endpoint forever.
                    if raw_settings.get("raylight_residency_policy") in {
                        "dedicated_keep_fl2va",
                        "dedicated_keep_ref2va",
                    }:
                        raw_settings["raylight_residency_policy"] = "keep_until_switch"
                    # Normalize retired Ray switches before the frozen v1 parser.
                    for family in ("fl2va", "ref2va"):
                        binding = (raw_settings.get("models") or {}).get(family)
                        raylight = (
                            binding.get("raylight")
                            if isinstance(binding, dict)
                            else None
                        )
                        if isinstance(raylight, dict):
                            raylight["fsdp"] = False
                            raylight["cpu_offload"] = False
                    persisted_settings = canonicalize_live_runtime_settings(
                        RuntimeSettings.model_validate(raw_settings)
                    )
            except ValidationError as exc:
                # The plugin era deliberately keeps no compatibility reader for
                # pre-plugin documents. Fail with an actionable message rather
                # than exposing a raw schema dump.
                raise RuntimeError(
                    f"Director database at {self.path} uses an obsolete "
                    "settings document; delete the database (or the whole "
                    "ComfyUI user/directordeck directory) and restart ComfyUI"
                ) from exc
            normalized_settings = persisted_settings.model_dump(mode="json")
            if json.loads(settings_row["document"]) != normalized_settings:
                db.execute(
                    "UPDATE settings SET document = ?, updated_at = ?, "
                    "revision = revision + 1 WHERE singleton = 1",
                    (json.dumps(normalized_settings, ensure_ascii=False), now),
                )
            for mode in MODE_ORDER:
                db.execute(
                    "INSERT OR IGNORE INTO mode_drafts(mode, document, updated_at) VALUES(?, ?, ?)",
                    (mode, default_draft(mode).model_dump_json(), now),
                )
            # MiniMax H3 has no negative-conditioning input.  Persisted drafts
            # from the original web schema contained compatibility-only
            # ``negative_prompt`` fields; strip them before strict validation
            # so upgrades do not strand existing user drafts.  Re-validating
            # also materializes newly added defaults such as ref_image_size.
            draft_rows = db.execute(
                "SELECT mode, document FROM mode_drafts"
            ).fetchall()
            for row in draft_rows:
                mode = str(row["mode"])
                document = json.loads(row["document"])
                document.pop("negative_prompt", None)
                sampling = document.get("sampling")
                if (
                    isinstance(sampling, dict)
                    and isinstance(sampling.get("seed"), int)
                    and not isinstance(sampling.get("seed"), bool)
                    and sampling["seed"] > 2**53 - 1
                ):
                    # Older schemas admitted uint64 fixed seeds even though a
                    # browser could not round-trip them exactly. ``-1`` is only
                    # a temporary legacy marker here; schema validation below
                    # immediately resolves it to one concrete JS-safe seed and
                    # records ``random_seed=true`` in canonical storage.
                    sampling["seed"] = -1
                shots = document.get("shots")
                if isinstance(shots, list):
                    for shot in shots:
                        if isinstance(shot, dict):
                            shot.pop("negative_prompt", None)
                normalized_draft = validate_mode_draft(mode, document)
                normalized_document = normalized_draft.model_dump(mode="json")
                if json.loads(row["document"]) != normalized_document:
                    db.execute(
                        "UPDATE mode_drafts SET document = ?, updated_at = ? WHERE mode = ?",
                        (json.dumps(normalized_document, ensure_ascii=False), now, mode),
                    )

            timeline_row = db.execute(
                "SELECT document FROM unified_timeline WHERE singleton = 1"
            ).fetchone()
            if timeline_row is None:
                if had_mode_drafts_table and not had_unified_timeline_table:
                    legacy_drafts = [
                        validate_mode_draft(
                            str(row["mode"]), json.loads(row["document"])
                        )
                        for row in db.execute(
                            "SELECT mode, document FROM mode_drafts ORDER BY "
                            "CASE mode "
                            "WHEN 't2v' THEN 1 WHEN 'i2v' THEN 2 "
                            "WHEN 'fl2v' THEN 3 WHEN 'r2v' THEN 4 "
                            "WHEN 'v2v' THEN 5 WHEN 'rv2v' THEN 6 END"
                        ).fetchall()
                    ]
                    migrated = migrate_mode_drafts_to_timeline(legacy_drafts)
                    customized_modes = [
                        draft.mode
                        for draft in legacy_drafts
                        if not mode_draft_is_default(draft)
                    ]
                    if len(customized_modes) > 1:
                        db.execute(
                            "INSERT OR IGNORE INTO migration_notices(id, message, created_at) "
                            "VALUES(?, ?, ?)",
                            (
                                "multiple-legacy-mode-drafts",
                                "未自动合并多份旧模式草稿（"
                                + ", ".join(customized_modes)
                                + "）：旧草稿仍保留在 mode_drafts，请在统一时间线中明确排序后手动导入。",
                                now,
                            ),
                        )
                elif fresh_database:
                    migrated = default_timeline_draft_v5(default_model_stack())
                else:
                    migrated = default_timeline_draft()
                db.execute(
                    "INSERT INTO unified_timeline(singleton, document, updated_at) VALUES(1, ?, ?)",
                    (migrated.model_dump_json(), now),
                )
            else:
                raw_timeline = json.loads(timeline_row["document"])
                legacy_timeline_compatibility_changed = False
                timeline_sampling = raw_timeline.get("sampling")
                if isinstance(timeline_sampling, dict):
                    sampling_documents = (
                        [timeline_sampling]
                        if not (
                            "fl2va" in timeline_sampling
                            or "ref2va" in timeline_sampling
                        )
                        else [
                            candidate
                            for family in ("fl2va", "ref2va")
                            if isinstance(
                                candidate := timeline_sampling.get(family), dict
                            )
                        ]
                    )
                    for sampling_document in sampling_documents:
                        seed = sampling_document.get("seed")
                        if (
                            isinstance(seed, int)
                            and not isinstance(seed, bool)
                            and seed > 2**53 - 1
                        ):
                            sampling_document["seed"] = -1
                            legacy_timeline_compatibility_changed = True
                is_v5_timeline = raw_timeline.get("version") == 5
                is_frozen_v4_timeline = raw_timeline.get("version") == 4
                normalized_timeline = (
                    validate_timeline_draft_v5(raw_timeline).model_dump(mode="json")
                    if is_v5_timeline
                    else validate_timeline_draft(raw_timeline).model_dump(mode="json")
                )
                # A valid v4 authority must reach the receipt migration with
                # the exact browser-authored optional-key shape intact. Merely
                # round-tripping it through Pydantic materializes absent asset
                # fields as null, advances the revision, and changes the
                # JSON.stringify/FNV digest used to reconcile legacy WAL. V5
                # may still materialize current defaults; v4 is rewritten only
                # for the explicit pre-existing unsafe-seed compatibility fix.
                should_normalize_timeline = (
                    json.loads(timeline_row["document"]) != normalized_timeline
                    and (
                        is_v5_timeline
                        or not is_frozen_v4_timeline
                        or legacy_timeline_compatibility_changed
                    )
                )
                if should_normalize_timeline:
                    db.execute(
                        "UPDATE unified_timeline SET document = ?, updated_at = ?, "
                        "revision = revision + 1 WHERE singleton = 1",
                        (json.dumps(normalized_timeline, ensure_ascii=False), now),
                    )

            # The pre-multi-project singleton timeline is the authority for the
            # legacy/default project (backed by the singleton row itself). Its
            # historical tasks and the durable take ledger are claimed by that
            # project id so a later scoped lookup never silently crosses
            # project boundaries. New projects live in the projects table and
            # are created only through the project API.
            db.execute(
                "UPDATE jobs SET project_id = ? "
                "WHERE project_id IS NULL AND mode = 'timeline'",
                (self.LEGACY_DEFAULT_PROJECT_ID,),
            )
            db.execute(
                "UPDATE segment_takes SET project_id = ? WHERE project_id IS NULL",
                (self.LEGACY_DEFAULT_PROJECT_ID,),
            )

            # Historical snapshots remain readable after the public settings
            # schema drops obsolete memory/FSDP options. Their immutable model
            # selections and task results are otherwise preserved.
            for row in db.execute(
                "SELECT id, settings_snapshot FROM jobs"
            ).fetchall():
                snapshot = json.loads(row["settings_snapshot"])
                # Bounded execution/control evidence is a different typed
                # contract stored in this historical wire column. Legacy
                # settings normalization must never inject settings fields
                # into that strict snapshot during a later restart.
                if "snapshot_schema_version" in snapshot:
                    continue
                changed = False
                if snapshot.get("memory_policy") == "clear_between_segments":
                    snapshot["memory_policy"] = "keep_resident"
                    changed = True
                # Before the explicit family-pinned policy existed, every Ray
                # sampler was compiled with cleanup enabled. Preserve that
                # historical execution fact instead of letting an old audit
                # snapshot look like it requested the new dedicated mode.
                if "raylight_residency_policy" not in snapshot:
                    snapshot["raylight_residency_policy"] = (
                        "release_after_sampling"
                    )
                    changed = True
                elif snapshot.get("raylight_residency_policy") in {
                    "dedicated_keep_fl2va",
                    "dedicated_keep_ref2va",
                }:
                    # The immutable prompt still had cleanup disabled. The new
                    # label preserves that execution fact without claiming the
                    # old permanent single-family restriction still applies.
                    snapshot["raylight_residency_policy"] = "keep_until_switch"
                    changed = True
                for family in ("fl2va", "ref2va"):
                    binding = (snapshot.get("models") or {}).get(family)
                    raylight = binding.get("raylight") if isinstance(binding, dict) else None
                    if isinstance(raylight, dict) and (
                        raylight.get("fsdp") is not False
                        or raylight.get("cpu_offload") is not False
                    ):
                        raylight["fsdp"] = False
                        raylight["cpu_offload"] = False
                        changed = True
                if changed:
                    db.execute(
                        "UPDATE jobs SET settings_snapshot = ? WHERE id = ?",
                        (json.dumps(snapshot, ensure_ascii=False), str(row["id"])),
                    )

            # Record the keyed-switch rollout exactly once. Legacy dedicated
            # values were normalized before strict settings validation above;
            # an explicit release choice remains release and is never silently
            # re-enabled on a later restart.
            residency_marker = db.execute(
                "SELECT 1 FROM migration_notices WHERE id = ?",
                (self._RAYLIGHT_RESIDENCY_MIGRATION_ID,),
            ).fetchone()
            if residency_marker is None:
                db.execute(
                    "INSERT INTO migration_notices(id, message, created_at) "
                    "VALUES(?, ?, ?)",
                    (
                        self._RAYLIGHT_RESIDENCY_MIGRATION_ID,
                        "RayLight 已升级为按完整配置 key 常驻并在不兼容任务前显式切池；不再永久锁定 FL2VA 或 Ref2VA。",
                        now,
                    ),
                )

            # The take ledger intentionally has no foreign key to jobs or
            # children: clearing task history must not erase reusable renders.
            # Re-running this pass is idempotent through source_child_id.
            self._backfill_segment_takes_in_connection(db)

        # Schema/bootstrap normalization above intentionally completes first.
        # The authority migration itself then owns a fresh connection and one
        # explicit BEGIN IMMEDIATE spanning every project, the default timeline,
        # every receipt, and the Stage-6 RuntimeSettingsV2 bridge.
        with self.connect() as migration_db:
            migrate_v4_authorities_to_v5(migration_db, created_at=utc_now())
        # Timeline schema-4 receipts remain frozen at their historical
        # schema-5/bundle-4 destination.  A distinct deterministic pass then
        # advances every affected creative authority's CAS revision exactly
        # once and makes bundle 5 durable before any client can open it.
        with self.connect() as migration_db:
            migrate_feature_bundle_v4_authorities_to_v5(
                migration_db,
                created_at=utc_now(),
            )
        # The mapping-only cut-over advances settings exactly once and writes
        # its actionable notice in the same explicit transaction. Fresh v3
        # databases and already-migrated databases are no-ops.
        with self.connect() as migration_db:
            migrate_runtime_settings_v2_to_v3(
                migration_db,
                created_at=utc_now(),
            )

    @staticmethod
    def _exact_segment_take_output(child: dict[str, Any]) -> dict[str, str] | None:
        segment_ids = child.get("segment_ids")
        output_nodes = child.get("output_nodes")
        if (
            child.get("status") != "succeeded"
            or not isinstance(segment_ids, list)
            or len(segment_ids) != 1
            or not isinstance(segment_ids[0], str)
            or not isinstance(output_nodes, dict)
            or set(output_nodes) != {segment_ids[0]}
        ):
            return None
        output_node_id = output_nodes.get(segment_ids[0])
        if not isinstance(output_node_id, str) or not output_node_id:
            return None
        candidates = [
            output
            for output in child.get("outputs") or []
            if isinstance(output, dict)
            and str(output.get("node_id") or "") == output_node_id
            and output.get("type") == "output"
        ]
        if len(candidates) != 1:
            return None
        try:
            return normalize_native_output_descriptor(candidates[0])
        except NativeTemplateError:
            return None

    def _register_segment_take_from_child_in_connection(
        self, db: sqlite3.Connection, child_id: str
    ) -> None:
        if self._execution_evidence_schema_exists(db):
            typed_row = db.execute(
                "SELECT 1 FROM job_child_execution_evidence WHERE child_id = ?",
                (child_id,),
            ).fetchone()
            if typed_row is not None:
                # Typed children are published only by
                # ``finalize_observed_artifact``.  Never let a generic legacy
                # child update infer a take (or its audio capability) from the
                # mutable compatibility columns.
                return
        row = db.execute(
            "SELECT job_children.*, jobs.config_snapshot AS parent_config_snapshot, "
            "jobs.project_id AS parent_project_id "
            "FROM job_children JOIN jobs ON jobs.id = job_children.job_id "
            "WHERE job_children.id = ? AND job_children.status = 'succeeded'",
            (child_id,),
        ).fetchone()
        if row is None:
            return
        try:
            child = self._job_child_row(row)
            output = self._exact_segment_take_output(child)
            if output is None:
                return
            config_snapshot = json.loads(row["parent_config_snapshot"])
            timeline_document = (
                config_snapshot.get("timeline")
                if isinstance(config_snapshot, dict)
                else None
            )
            if not isinstance(timeline_document, dict):
                return
            timeline = validate_timeline_snapshot(timeline_document)
            segment_id = str(child["segment_ids"][0])
            matches = [
                segment for segment in timeline.segments if segment.id == segment_id
            ]
            if len(matches) != 1:
                return
            content_fingerprint = timeline_segment_take_fingerprint(
                timeline, matches[0]
            )
        except (KeyError, TypeError, ValueError, ValidationError, json.JSONDecodeError):
            # Historical audit rows can predate the unified timeline contract.
            # They remain readable as jobs but cannot be promoted into a take
            # without a complete typed content identity.
            return
        completed_at = str(
            child.get("completed_at") or child.get("updated_at") or utc_now()
        )
        db.execute(
            "INSERT OR IGNORE INTO segment_takes("
            "id, segment_id, content_fingerprint, project_id, "
            "output_descriptor, has_audio, source_job_id, source_child_id, "
            "completed_at, created_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source_child_id) DO UPDATE SET "
            "segment_id = excluded.segment_id, "
            "content_fingerprint = excluded.content_fingerprint, "
            "project_id = excluded.project_id, "
            "output_descriptor = excluded.output_descriptor, "
            "has_audio = excluded.has_audio, "
            "source_job_id = excluded.source_job_id, "
            "completed_at = excluded.completed_at",
            (
                str(uuid.uuid4()),
                segment_id,
                content_fingerprint,
                row["parent_project_id"],
                json.dumps(output, ensure_ascii=False, sort_keys=True),
                int(matches[0].audio_mode != "mute"),
                str(child["job_id"]),
                str(child["id"]),
                completed_at,
                utc_now(),
            ),
        )

    def _backfill_segment_takes_in_connection(self, db: sqlite3.Connection) -> None:
        rows = db.execute(
            "SELECT job_children.id FROM job_children "
            "LEFT JOIN segment_takes ON "
            "segment_takes.source_child_id = job_children.id "
            "WHERE job_children.status = 'succeeded' "
            "AND (segment_takes.source_child_id IS NULL OR "
            "segment_takes.content_fingerprint NOT LIKE 'take-geometry-v1:%') "
            "ORDER BY job_children.completed_at, job_children.updated_at, "
            "job_children.id"
        ).fetchall()
        for row in rows:
            self._register_segment_take_from_child_in_connection(db, str(row["id"]))

    @staticmethod
    def _segment_take_row(row: Mapping[str, Any]) -> dict[str, Any]:
        take = dict(row)
        output = json.loads(take["output_descriptor"])
        if not isinstance(output, dict):
            raise ValueError("persisted segment take output is invalid")
        take["output"] = normalize_native_output_descriptor(output)
        take.pop("output_descriptor", None)
        take["has_audio"] = bool(take["has_audio"])
        return take

    def find_latest_segment_take(
        self,
        segment_id: str,
        content_fingerprint: str,
        *,
        require_audio: bool = False,
        project_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the newest exact reusable take for the segment.

        A non-null ``project_id`` scopes the lookup to one project so two
        projects sharing a segment identity can never reuse each other's
        render. Callers that omit it (legacy tests/automation) keep the
        unscoped behavior.
        """

        audio_clause = " AND has_audio = 1" if require_audio else ""
        project_clause = ""
        parameters: tuple[Any, ...] = (segment_id, content_fingerprint)
        if project_id is not None:
            # NULL-project_id rows predate project scoping (legacy six-mode
            # submissions) and are treated as belonging to every project's
            # lookup; their segment ids never collide with timeline projects.
            project_clause = " AND (project_id = ? OR project_id IS NULL)"
            parameters = (*parameters, project_id)
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM segment_takes WHERE segment_id = ? "
                "AND content_fingerprint = ?"
                f"{audio_clause}{project_clause} ORDER BY completed_at DESC, id DESC "
                "LIMIT 1",
                parameters,
            ).fetchone()
        return self._segment_take_row(row) if row is not None else None

    def find_latest_observed_segment_take(
        self,
        segment_id: str,
        content_fingerprint: str,
        *,
        require_audio: bool = False,
        project_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Return only takes backed by a typed ObservedArtifactSpec."""

        audio_clause = " AND t.has_audio = 1" if require_audio else ""
        project_clause = ""
        parameters: tuple[Any, ...] = (segment_id, content_fingerprint)
        if project_id is not None:
            project_clause = " AND t.project_id = ?"
            parameters = (*parameters, project_id)
        with self.connect() as db:
            if not self._artifact_observation_schema_exists(db):
                return None
            if not self._execution_evidence_schema_exists(db):
                raise ExecutionEvidenceConflict(
                    "observed take has no execution evidence schema"
                )
            row = db.execute(
                self._OBSERVED_SEGMENT_TAKE_SELECT
                + " FROM segment_takes t "
                "JOIN segment_take_observed_artifacts a ON a.take_id = t.id "
                "LEFT JOIN job_children c ON c.id = t.source_child_id "
                "LEFT JOIN jobs j ON j.id = t.source_job_id "
                "LEFT JOIN job_child_output_receipts r "
                "ON r.child_id = a.source_child_id "
                "LEFT JOIN job_child_execution_evidence e "
                "ON e.child_id = a.source_child_id "
                "LEFT JOIN prompt_ownership o "
                "ON o.child_id = a.source_child_id "
                "WHERE t.segment_id = ? AND t.content_fingerprint = ?"
                f"{audio_clause}{project_clause} "
                "ORDER BY t.completed_at DESC, t.id DESC LIMIT 1",
                parameters,
            ).fetchone()
        if row is None:
            return None
        take, artifact = self._observed_segment_take_row(row)
        take["observed_artifact"] = artifact
        return take

    def has_observed_segment_take(
        self,
        segment_id: str,
        *,
        content_fingerprint: str | None = None,
        project_id: str | None = None,
    ) -> bool:
        """Return whether a reusable take has actual typed media evidence."""

        fingerprint_clause = (
            " AND t.content_fingerprint = ?"
            if content_fingerprint is not None
            else ""
        )
        project_clause = " AND t.project_id = ?" if project_id is not None else ""
        parameters: tuple[Any, ...] = (segment_id,)
        if content_fingerprint is not None:
            parameters = (*parameters, content_fingerprint)
        if project_id is not None:
            parameters = (*parameters, project_id)
        with self.connect() as db:
            if not self._artifact_observation_schema_exists(db):
                return False
            if not self._execution_evidence_schema_exists(db):
                raise ExecutionEvidenceConflict(
                    "observed take has no execution evidence schema"
                )
            row = db.execute(
                self._OBSERVED_SEGMENT_TAKE_SELECT
                + " FROM segment_takes t "
                "JOIN segment_take_observed_artifacts a ON a.take_id = t.id "
                "LEFT JOIN job_children c ON c.id = t.source_child_id "
                "LEFT JOIN jobs j ON j.id = t.source_job_id "
                "LEFT JOIN job_child_output_receipts r "
                "ON r.child_id = a.source_child_id "
                "LEFT JOIN job_child_execution_evidence e "
                "ON e.child_id = a.source_child_id "
                "LEFT JOIN prompt_ownership o "
                "ON o.child_id = a.source_child_id "
                "WHERE t.segment_id = ?"
                f"{fingerprint_clause}{project_clause} "
                "ORDER BY t.completed_at DESC, t.id DESC LIMIT 1",
                parameters,
            ).fetchone()
        if row is None:
            return False
        self._observed_segment_take_row(row)
        return True

    def has_segment_take(
        self,
        segment_id: str,
        *,
        content_fingerprint: str | None = None,
        project_id: str | None = None,
    ) -> bool:
        fingerprint_clause = (
            " AND content_fingerprint = ?"
            if content_fingerprint is not None
            else ""
        )
        project_clause = (
            " AND (project_id = ? OR project_id IS NULL)"
            if project_id is not None
            else ""
        )
        parameters: tuple[Any, ...] = (segment_id,)
        if content_fingerprint is not None:
            parameters = (*parameters, content_fingerprint)
        if project_id is not None:
            parameters = (*parameters, project_id)
        with self.connect() as db:
            row = db.execute(
                "SELECT 1 FROM segment_takes WHERE segment_id = ?"
                f"{fingerprint_clause}{project_clause} LIMIT 1",
                parameters,
            ).fetchone()
        return row is not None

    @staticmethod
    def _settings_authority_token(*, revision: int, document: str) -> str:
        return hashlib.sha256(
            (str(revision) + "\0" + document).encode("utf-8")
        ).hexdigest()

    def get_settings_v3_authority(self) -> tuple[RuntimeSettingsV3, str]:
        with self.connect() as db:
            row = db.execute(
                "SELECT document, revision FROM settings WHERE singleton = 1"
            ).fetchone()
        if row is None:
            raise RuntimeError("settings row is missing")
        settings = RuntimeSettingsV3.model_validate_json(row["document"])
        authority = self._settings_authority_token(
            revision=int(row["revision"]),
            document=str(row["document"]),
        )
        return settings, authority

    def get_settings_authority(self) -> tuple[RuntimeSettingsV3, str]:
        return self.get_settings_v3_authority()

    def get_settings(self) -> RuntimeSettingsV3:
        return self.get_settings_authority()[0]

    @staticmethod
    def _require_runtime_settings_v3(
        settings: RuntimeSettingsV3 | RuntimeSettingsV2 | RuntimeSettings,
    ) -> RuntimeSettingsV3:
        if isinstance(settings, (RuntimeSettings, RuntimeSettingsV2)):
            raise RuntimeSettingsSchemaMigrated()
        if not isinstance(settings, RuntimeSettingsV3):
            raise TypeError("settings write requires RuntimeSettingsV3")
        return settings

    def put_settings(
        self,
        settings: RuntimeSettingsV3 | RuntimeSettingsV2 | RuntimeSettings,
        *,
        expected_authority_token: str | None = None,
        schema_version: int | None = None,
    ) -> RuntimeSettingsV3:
        settings = self._require_runtime_settings_v3(settings)
        if expected_authority_token is None or schema_version is None:
            raise SettingsAuthorityRequired(
                "RuntimeSettingsV3 writes require an authority token"
            )
        saved, _next_token = self.put_settings_v3_authority(
            settings,
            expected_authority_token=expected_authority_token,
            schema_version=schema_version,
        )
        return saved

    def put_settings_v3_authority(
        self,
        settings: RuntimeSettingsV3 | RuntimeSettingsV2 | RuntimeSettings,
        *,
        expected_authority_token: str,
        schema_version: int,
    ) -> tuple[RuntimeSettingsV3, str]:
        if schema_version != 3:
            raise RuntimeSettingsSchemaMigrated()
        settings = self._require_runtime_settings_v3(settings)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT document, revision FROM settings WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise RuntimeError("settings row is missing")
            revision = int(row["revision"])
            current_token = self._settings_authority_token(
                revision=revision,
                document=str(row["document"]),
            )
            if current_token != expected_authority_token:
                raise SettingsAuthorityConflict(expected_authority_token)
            serialized = settings.model_dump_json()
            cursor = db.execute(
                "UPDATE settings SET document = ?, updated_at = ?, "
                "revision = revision + 1 WHERE singleton = 1 AND revision = ?",
                (serialized, utc_now(), revision),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("settings CAS update did not affect one row")
        next_token = self._settings_authority_token(
            revision=revision + 1,
            document=serialized,
        )
        return settings, next_token

    def put_settings_v2_authority(
        self,
        settings: RuntimeSettingsV2 | RuntimeSettings,
        *,
        expected_authority_token: str,
        schema_version: int,
    ) -> tuple[RuntimeSettingsV3, str]:
        del settings, expected_authority_token, schema_version
        raise RuntimeSettingsSchemaMigrated()

    def put_settings_authority(
        self,
        settings: RuntimeSettingsV3 | RuntimeSettingsV2 | RuntimeSettings,
        *,
        expected_authority_token: str,
        schema_version: int,
    ) -> tuple[RuntimeSettingsV3, str]:
        return self.put_settings_v3_authority(
            settings,
            expected_authority_token=expected_authority_token,
            schema_version=schema_version,
        )

    def list_runtime_settings_migration_notices(
        self,
    ) -> list[RuntimeSettingsMigrationNotice]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT document FROM runtime_settings_migration_notices "
                "ORDER BY created_at, id"
            ).fetchall()
        try:
            return [
                RuntimeSettingsMigrationNotice.model_validate_json(row["document"])
                for row in rows
            ]
        except ValidationError as exc:
            raise RuntimeError(
                "stored runtime settings migration notice is invalid"
            ) from exc

    def get_raylight_runtime_state(self) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT descriptor FROM raylight_runtime_state "
                "WHERE singleton = 1"
            ).fetchone()
        if row is None:
            return None
        return self._decode_raylight_runtime_state(row["descriptor"])

    @staticmethod
    def _decode_raylight_runtime_state(document: str) -> dict[str, Any]:
        descriptor = json.loads(document)
        if not isinstance(descriptor, dict):
            raise ValueError("persisted RayLight runtime descriptor is invalid")
        # A pre-release keyed-switch build briefly wrote the descriptor without
        # an envelope. It cannot satisfy the v2 full-loader-chain barrier
        # contract, so discard its unverifiable actor handle while retaining a
        # monotonic starting epoch. Pretending it were v2 would make every next
        # transition fail permanently in build_raylight_shutdown_unit().
        if "epoch" not in descriptor or "current" not in descriptor:
            legacy_epoch = 1
            legacy_namespace = descriptor.get("runtime_namespace")
            if isinstance(legacy_namespace, str):
                epoch_match = re.search(r"-e([1-9][0-9]*)$", legacy_namespace)
                if epoch_match is not None:
                    legacy_epoch = int(epoch_match.group(1))
            return {
                "version": 2,
                "epoch": legacy_epoch,
                "current": None,
                "tail_prompt_id": None,
                "tail_action": None,
                "tainted": True,
                "legacy_unknown": True,
            }
        if descriptor.get("version") == 1:
            legacy_epoch = descriptor.get("epoch")
            if (
                not isinstance(legacy_epoch, int)
                or isinstance(legacy_epoch, bool)
                or legacy_epoch < 0
            ):
                raise ValueError("persisted RayLight runtime state is invalid")
            return {
                "version": 2,
                "epoch": legacy_epoch,
                "current": None,
                "tail_prompt_id": None,
                "tail_action": None,
                "tainted": True,
                "legacy_unknown": True,
            }
        epoch = descriptor.get("epoch")
        current = descriptor.get("current")
        if (
            descriptor.get("version") != 2
            or not isinstance(epoch, int)
            or isinstance(epoch, bool)
            or epoch < 0
            or (current is not None and not isinstance(current, dict))
        ):
            raise ValueError("persisted RayLight runtime state is invalid")
        tail_prompt_id = descriptor.get("tail_prompt_id")
        if tail_prompt_id is not None and (
            not isinstance(tail_prompt_id, str) or not tail_prompt_id
        ):
            raise ValueError("persisted RayLight runtime tail prompt is invalid")
        tail_action = descriptor.get("tail_action")
        if tail_action not in {None, "ray_unit", "shutdown"}:
            raise ValueError("persisted RayLight runtime tail action is invalid")
        if tail_action is not None and tail_prompt_id is None:
            raise ValueError("persisted RayLight runtime tail action has no prompt")
        migrated_current = (
            migrate_legacy_raylight_runtime_descriptor(current)
            if isinstance(current, dict)
            else current
        )
        result = {
            "version": 2,
            "epoch": epoch,
            "current": migrated_current,
            "tail_prompt_id": tail_prompt_id,
            "tail_action": tail_action,
            "tainted": bool(descriptor.get("tainted")),
        }
        terminal_certificate = descriptor.get("tail_terminal_certificate")
        if terminal_certificate is not None:
            if (
                not isinstance(terminal_certificate, dict)
                or set(terminal_certificate)
                != {"prompt_id", "action", "succeeded"}
            ):
                raise ValueError(
                    "persisted RayLight terminal certificate is invalid"
                )
            certificate_prompt_id = terminal_certificate.get("prompt_id")
            certificate_action = terminal_certificate.get("action")
            certificate_succeeded = terminal_certificate.get("succeeded")
            if (
                not isinstance(certificate_prompt_id, str)
                or not certificate_prompt_id
                or certificate_action not in {"ray_unit", "shutdown"}
                or not isinstance(certificate_succeeded, bool)
                or certificate_prompt_id != tail_prompt_id
                or certificate_action != tail_action
            ):
                raise ValueError(
                    "persisted RayLight terminal certificate does not match its tail"
                )
            result["tail_terminal_certificate"] = {
                "prompt_id": certificate_prompt_id,
                "action": certificate_action,
                "succeeded": certificate_succeeded,
            }
        legacy_unknown = descriptor.get("legacy_unknown", False)
        if not isinstance(legacy_unknown, bool):
            raise ValueError("persisted RayLight legacy runtime flag is invalid")
        if legacy_unknown:
            result["legacy_unknown"] = True
        return result

    def put_raylight_runtime_state(self, state: dict[str, Any]) -> None:
        normalized = self._decode_raylight_runtime_state(
            json.dumps(state, ensure_ascii=False, sort_keys=True)
        )
        with self.connect() as db:
            db.execute(
                "INSERT INTO raylight_runtime_state"
                "(singleton, descriptor, updated_at) VALUES(1, ?, ?) "
                "ON CONFLICT(singleton) DO UPDATE SET "
                "descriptor = excluded.descriptor, updated_at = excluded.updated_at",
                (
                    json.dumps(normalized, ensure_ascii=False, sort_keys=True),
                    utc_now(),
                ),
            )

    @classmethod
    def _settle_raylight_runtime_prompt_in_connection(
        cls,
        db: sqlite3.Connection,
        prompt_id: str,
        *,
        succeeded: bool,
        terminal_history_certified: bool = False,
        updated_at: str | None = None,
    ) -> bool:
        if not prompt_id:
            return False
        row = db.execute(
            "SELECT descriptor FROM raylight_runtime_state WHERE singleton = 1"
        ).fetchone()
        if row is None:
            return False
        state = cls._decode_raylight_runtime_state(row["descriptor"])
        if state.get("tail_prompt_id") != prompt_id:
            return False
        tail_action = state.get("tail_action")
        if succeeded:
            if tail_action == "shutdown":
                state.update(
                    current=None,
                    tail_prompt_id=None,
                    tail_action=None,
                    tainted=False,
                )
                state.pop("tail_terminal_certificate", None)
            else:
                state.update(tainted=False)
        else:
            state["tainted"] = True
        if (
            terminal_history_certified
            and state.get("tail_prompt_id") == prompt_id
            and tail_action in {"ray_unit", "shutdown"}
        ):
            state["tail_terminal_certificate"] = {
                "prompt_id": prompt_id,
                "action": tail_action,
                "succeeded": succeeded,
            }
        db.execute(
            "UPDATE raylight_runtime_state SET descriptor = ?, updated_at = ? "
            "WHERE singleton = 1",
            (
                json.dumps(state, ensure_ascii=False, sort_keys=True),
                updated_at or utc_now(),
            ),
        )
        return True

    @classmethod
    def _cleanup_raylight_runtime_prompt_in_connection(
        cls,
        db: sqlite3.Connection,
        prompt_id: str,
        *,
        updated_at: str | None = None,
    ) -> bool:
        """Release a cancelled/restarted tail without claiming a clean pool.

        Exact cancel and endpoint restart evidence prove that this prompt no
        longer owns the queue frontier.  They do not prove that the mutable
        Ray actor state is reusable, so preserve the monotonic epoch/current
        descriptor, clear only the matching tail and leave the pool tainted.
        """

        if not prompt_id:
            return False
        row = db.execute(
            "SELECT descriptor FROM raylight_runtime_state WHERE singleton = 1"
        ).fetchone()
        if row is None:
            return False
        state = cls._decode_raylight_runtime_state(row["descriptor"])
        if state.get("tail_prompt_id") != prompt_id:
            return False
        state.update(
            tail_prompt_id=None,
            tail_action=None,
            tainted=True,
        )
        state.pop("tail_terminal_certificate", None)
        db.execute(
            "UPDATE raylight_runtime_state SET descriptor = ?, updated_at = ? "
            "WHERE singleton = 1",
            (
                json.dumps(state, ensure_ascii=False, sort_keys=True),
                updated_at or utc_now(),
            ),
        )
        return True

    def settle_raylight_runtime_prompt(
        self,
        prompt_id: str,
        *,
        succeeded: bool,
        terminal_history_certified: bool = False,
    ) -> bool:
        """Settle only the currently recorded Ray queue tail.

        Reconciliation races later submissions.  Matching ``tail_prompt_id``
        inside one ``BEGIN IMMEDIATE`` transaction prevents a late success or
        failure from an older child from overwriting the newer endpoint tail.
        A successful release-policy sampler has cleared its worker model, but
        not the Ray cluster or ComfyUI's cached actor handles. Both policies
        therefore retain ``current`` and its epoch; the policy controls only
        whether the next compatible prompt reloads weights. Failures and
        cancellations stay tainted until an explicit barrier succeeds.
        """

        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            return self._settle_raylight_runtime_prompt_in_connection(
                db,
                prompt_id,
                succeeded=succeeded,
                terminal_history_certified=terminal_history_certified,
            )

    def confirm_comfy_restart_recovery(
        self,
        job_id: str,
        *,
        current_endpoint_identity: EndpointIdentity | None = None,
    ) -> dict[str, Any]:
        """Release a restarted endpoint through typed evidence when present.

        Historical jobs have no Stage-4 tables/rows and retain the original
        lifecycle-only operator recovery.  Once either exact evidence or
        ownership exists for this job, partial/legacy fallback is forbidden.
        """

        with self.connect() as db:
            typed = self._job_has_typed_execution_marker_in_connection(
                db,
                job_id,
            )
        if not typed:
            return self._confirm_legacy_comfy_restart_recovery(job_id)
        if current_endpoint_identity is None:
            raise ValueError(
                "typed restart recovery requires the current endpoint identity"
            )
        return self._confirm_typed_comfy_restart_recovery(
            job_id,
            current_endpoint_identity=current_endpoint_identity,
        )

    def _confirm_typed_comfy_restart_recovery(
        self,
        job_id: str,
        *,
        current_endpoint_identity: EndpointIdentity,
    ) -> dict[str, Any]:
        """Atomically certify one replacement boot for every owned child."""

        now = utc_now()
        confirmed_at = datetime.fromisoformat(now).astimezone(timezone.utc)
        recovery_stages = sorted(self._RESTART_RECOVERY_STAGES)
        placeholders = ",".join("?" for _ in recovery_stages)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not self._execution_evidence_schema_exists(db):
                raise ExecutionEvidenceConflict(
                    "typed restart evidence schema disappeared"
                )
            parent = db.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if parent is None:
                raise KeyError(job_id)
            rows = db.execute(
                "SELECT c.*, "
                "e.exact_prompt_snapshot AS evidence_snapshot, "
                "e.exact_prompt_snapshot_digest AS evidence_digest, "
                "e.endpoint_key AS evidence_endpoint_key, "
                "e.endpoint_runtime_instance_id AS evidence_runtime_id, "
                "o.requested_prompt_id AS ownership_requested_id, "
                "o.actual_prompt_id AS ownership_actual_id, "
                "o.state AS ownership_state, "
                "o.ownership_revision AS ownership_revision, "
                "o.cleanup_certificate AS ownership_certificate, "
                "o.updated_at AS ownership_updated_at "
                "FROM job_children c "
                "LEFT JOIN job_child_execution_evidence e ON e.child_id = c.id "
                "LEFT JOIN prompt_ownership o ON o.child_id = c.id "
                "WHERE c.job_id = ? ORDER BY c.group_index",
                (job_id,),
            ).fetchall()

            typed_rows = [
                row
                for row in rows
                if row["evidence_snapshot"] is not None
                or row["ownership_requested_id"] is not None
            ]
            if not typed_rows:
                raise ExecutionEvidenceConflict(
                    "typed restart recovery lost all execution evidence"
                )
            for row in typed_rows:
                if (
                    row["evidence_snapshot"] is None
                    or row["ownership_requested_id"] is None
                ):
                    raise ExecutionEvidenceConflict(
                        "restart recovery has incomplete exact ownership evidence"
                    )

            parsed: list[
                tuple[sqlite3.Row, ExactPromptSnapshot, PromptOwnership]
            ] = []
            for row in rows:
                has_evidence = row["evidence_snapshot"] is not None
                has_ownership = row["ownership_requested_id"] is not None
                has_prompt = isinstance(row["prompt_id"], str) and bool(
                    row["prompt_id"]
                )
                if has_evidence != has_ownership:
                    raise ExecutionEvidenceConflict(
                        "restart recovery has incomplete exact ownership evidence"
                    )
                if has_prompt != has_evidence:
                    raise ExecutionEvidenceConflict(
                        "typed restart recovery cannot mix legacy prompt owners"
                    )
                if not has_evidence:
                    continue
                snapshot = ExactPromptSnapshot.model_validate_json(
                    row["evidence_snapshot"]
                )
                if (
                    self._execution_document_digest(snapshot)
                    != row["evidence_digest"]
                    or snapshot.endpoint_identity.endpoint_key
                    != row["evidence_endpoint_key"]
                    or snapshot.endpoint_identity.runtime_instance_id
                    != row["evidence_runtime_id"]
                ):
                    raise ExecutionEvidenceConflict(
                        "restart exact prompt evidence digest/index mismatch"
                    )
                ownership = self._prompt_ownership_row_from_join(row)
                if ownership.effective_prompt_id != str(row["prompt_id"]):
                    raise ExecutionEvidenceConflict(
                        "restart child prompt differs from durable ownership"
                    )
                parsed.append((row, snapshot, ownership))

            if (
                parent["status"] == "cancelled"
                and parent["stage"] == self._CONFIRMED_COMFY_RESTART_STAGE
            ):
                if any(
                    row["status"] not in self._TERMINAL_JOB_STATUSES
                    for row in rows
                ):
                    raise ValueError(
                        "confirmed restart task still has a nonterminal child"
                    )
                restart_certificates = []
                for _, snapshot, ownership in parsed:
                    if ownership.state not in {
                        "cleanup_confirmed",
                        "terminal_confirmed",
                    }:
                        raise ExecutionEvidenceConflict(
                            "confirmed restart task retains unreleased ownership"
                        )
                    certificate = ownership.cleanup_certificate
                    if isinstance(certificate, EndpointRestartCertificate):
                        if (
                            certificate.prompt_id
                            != ownership.effective_prompt_id
                            or certificate.endpoint_identity
                            != snapshot.endpoint_identity
                        ):
                            raise ExecutionEvidenceConflict(
                                "persisted restart certificate does not match exact evidence"
                            )
                        restart_certificates.append(certificate)
                if not restart_certificates:
                    raise ExecutionEvidenceConflict(
                        "confirmed restart task has no endpoint restart certificate"
                    )
                if any(
                    certificate.endpoint_identity.endpoint_key
                    != current_endpoint_identity.endpoint_key
                    or certificate.restart_id
                    != current_endpoint_identity.runtime_instance_id
                    for certificate in restart_certificates
                ):
                    raise ExecutionEvidenceConflict(
                        "restart certificate belongs to a different replacement boot"
                    )
                return self._job_row(parent)

            if (
                parent["status"] != "cancelling"
                or not bool(parent["cancel_requested"])
                or parent["stage"] not in recovery_stages
            ):
                raise ValueError(
                    "job is not owned by restart cancellation recovery"
                )

            unreleased: list[
                tuple[sqlite3.Row, ExactPromptSnapshot, PromptOwnership]
            ] = []
            old_endpoint: EndpointIdentity | None = None
            for row, snapshot, ownership in parsed:
                if row["status"] in self._TERMINAL_JOB_STATUSES:
                    if ownership.state not in {
                        "cleanup_confirmed",
                        "terminal_confirmed",
                    }:
                        raise ExecutionEvidenceConflict(
                            "terminal child retains unreleased prompt ownership"
                        )
                    continue
                if row["stage"] not in recovery_stages:
                    raise ValueError(
                        f"job has an invalid restart-recovery active child: {row['id']}"
                    )
                if ownership.state in {
                    "cleanup_confirmed",
                    "terminal_confirmed",
                }:
                    raise ExecutionEvidenceConflict(
                        "active child already carries terminal ownership evidence"
                    )
                if old_endpoint is None:
                    old_endpoint = snapshot.endpoint_identity
                elif old_endpoint != snapshot.endpoint_identity:
                    raise ExecutionEvidenceConflict(
                        "one job cannot own prompts from multiple endpoint boots"
                    )
                unreleased.append((row, snapshot, ownership))

            if old_endpoint is None:
                raise ExecutionEvidenceConflict(
                    "typed restart task has no unreleased prompt ownership"
                )
            if old_endpoint.endpoint_key != current_endpoint_identity.endpoint_key:
                raise ExecutionEvidenceConflict(
                    "replacement endpoint key does not match the old prompt endpoint"
                )
            if (
                old_endpoint.runtime_instance_id
                == current_endpoint_identity.runtime_instance_id
            ):
                raise ExecutionEvidenceConflict(
                    "ComfyUI runtime instance has not changed"
                )

            # A host clock may move backwards across the very restart being
            # certified.  Ownership timestamps are monotonic evidence, so the
            # certificate must never predate the newest row it releases.
            confirmed_at = max(
                confirmed_at,
                *(ownership.updated_at for _, _, ownership in unreleased),
            )
            now = confirmed_at.isoformat()

            for row, snapshot, ownership in unreleased:
                certificate = EndpointRestartCertificate(
                    certificate_version=1,
                    prompt_id=ownership.effective_prompt_id,
                    endpoint_identity=snapshot.endpoint_identity,
                    restart_id=current_endpoint_identity.runtime_instance_id,
                    queue_and_history_cleared=True,
                    confirmed_at=confirmed_at,
                )
                next_ownership = self._transition_prompt_ownership_in_connection(
                    db,
                    str(row["id"]),
                    expected_revision=ownership.ownership_revision,
                    state="cleanup_confirmed",
                    updated_at=confirmed_at,
                    cleanup_certificate=certificate,
                )
                if next_ownership is None:
                    raise RuntimeError(
                        "restart ownership changed under the write transaction"
                    )
                cursor = db.execute(
                    "UPDATE job_children SET status = 'cancelled', progress = 1.0, "
                    "stage = ?, outputs = '[]', error = NULL, updated_at = ?, "
                    "completed_at = ? WHERE id = ? "
                    "AND status NOT IN ('succeeded', 'failed', 'cancelled') "
                    "AND prompt_id = ?",
                    (
                        self._CONFIRMED_COMFY_RESTART_STAGE,
                        now,
                        now,
                        row["id"],
                        ownership.effective_prompt_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        "restart child changed under the write transaction"
                    )
                if row["backend"] == "raylight":
                    self._cleanup_raylight_runtime_prompt_in_connection(
                        db,
                        ownership.effective_prompt_id,
                        updated_at=now,
                    )

            # Children without prompt ownership provably never crossed the
            # network; close them in the same parent transaction.
            db.execute(
                "UPDATE job_children SET status = 'cancelled', progress = 1.0, "
                "stage = ?, outputs = '[]', error = NULL, updated_at = ?, "
                "completed_at = ? WHERE job_id = ? "
                "AND status NOT IN ('succeeded', 'failed', 'cancelled') "
                "AND prompt_id IS NULL",
                (
                    self._CONFIRMED_COMFY_RESTART_STAGE,
                    now,
                    now,
                    job_id,
                ),
            )
            cursor = db.execute(
                "UPDATE jobs SET status = 'cancelled', progress = 1.0, "
                "stage = ?, outputs = '[]', error = NULL, updated_at = ?, "
                "completed_at = ? WHERE id = ? AND status = 'cancelling' "
                "AND cancel_requested = 1 "
                f"AND stage IN ({placeholders})",
                (
                    self._CONFIRMED_COMFY_RESTART_STAGE,
                    now,
                    now,
                    job_id,
                    *recovery_stages,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    "restart recovery parent changed during confirmation"
                )
            settled = db.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if settled is None:
            raise KeyError(job_id)
        return self._job_row(settled)

    @staticmethod
    def _prompt_ownership_row_from_join(row: sqlite3.Row) -> PromptOwnership:
        return Database._prompt_ownership_row(
            {
                "requested_prompt_id": row["ownership_requested_id"],
                "actual_prompt_id": row["ownership_actual_id"],
                "state": row["ownership_state"],
                "ownership_revision": row["ownership_revision"],
                "cleanup_certificate": row["ownership_certificate"],
                "updated_at": row["ownership_updated_at"],
            }
        )

    def _confirm_legacy_comfy_restart_recovery(
        self, job_id: str
    ) -> dict[str, Any]:
        """Atomically close only a restart-owned ambiguous submission.

        This operation deliberately performs no ComfyUI I/O.  Its caller must
        carry the explicit operator certificate that the old ComfyUI process
        was restarted, which is the only fact that proves an in-flight old
        ``POST /prompt`` can no longer enqueue after a directed cancel returned
        false.  Every lifecycle predicate is rechecked under ``BEGIN
        IMMEDIATE`` so a recovery worker or terminal history update always wins
        cleanly instead of being overwritten.
        """

        now = utc_now()
        recovery_stages = sorted(self._RESTART_RECOVERY_STAGES)
        recovery_placeholders = ",".join("?" for _ in recovery_stages)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if self._job_has_typed_execution_marker_in_connection(db, job_id):
                raise ExecutionEvidenceConflict(
                    "legacy restart recovery cannot consume typed execution evidence"
                )
            parent = db.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if parent is None:
                raise KeyError(job_id)

            children = db.execute(
                "SELECT * FROM job_children WHERE job_id = ? ORDER BY group_index",
                (job_id,),
            ).fetchall()

            # Retrying the exact confirmation is harmless.  Do not broaden
            # idempotency to arbitrary terminal jobs: a succeeded/failed task
            # must never be relabelled by this exceptional recovery action.
            if (
                parent["status"] == "cancelled"
                and parent["stage"] == self._CONFIRMED_COMFY_RESTART_STAGE
            ):
                if any(
                    child["status"] not in self._TERMINAL_JOB_STATUSES
                    for child in children
                ):
                    raise ValueError(
                        "confirmed restart task still has a nonterminal child"
                    )
                return self._job_row(parent)

            if (
                parent["status"] != "cancelling"
                or not bool(parent["cancel_requested"])
                or parent["stage"] not in self._RESTART_RECOVERY_STAGES
            ):
                raise ValueError(
                    "job is not owned by restart cancellation recovery"
                )

            active_children = [
                child
                for child in children
                if child["status"] not in self._TERMINAL_JOB_STATUSES
            ]
            invalid_children = [
                str(child["id"])
                for child in active_children
                if child["stage"] not in self._RESTART_RECOVERY_STAGES
                or not isinstance(child["prompt_id"], str)
                or not child["prompt_id"]
            ]
            if invalid_children:
                raise ValueError(
                    "job has an invalid restart-recovery active child: "
                    + ", ".join(invalid_children)
                )

            target_prompt_ids = {
                str(child["prompt_id"])
                for child in active_children
                if isinstance(child["prompt_id"], str) and child["prompt_id"]
            }

            if active_children:
                cursor = db.execute(
                    "UPDATE job_children SET status = 'cancelled', progress = 1.0, "
                    "stage = ?, error = NULL, updated_at = ?, completed_at = ? "
                    "WHERE job_id = ? "
                    "AND status NOT IN ('succeeded', 'failed', 'cancelled') "
                    f"AND stage IN ({recovery_placeholders})",
                    (
                        self._CONFIRMED_COMFY_RESTART_STAGE,
                        now,
                        now,
                        job_id,
                        *recovery_stages,
                    ),
                )
                if cursor.rowcount != len(active_children):
                    # ``BEGIN IMMEDIATE`` should make this unreachable, but
                    # keep a row-count assertion as a fail-closed guard if the
                    # query predicates are changed later.
                    raise RuntimeError(
                        "restart recovery children changed during confirmation"
                    )

            cursor = db.execute(
                "UPDATE jobs SET status = 'cancelled', progress = 1.0, "
                "stage = ?, outputs = '[]', error = NULL, updated_at = ?, completed_at = ? "
                "WHERE id = ? AND status = 'cancelling' AND cancel_requested = 1 "
                f"AND stage IN ({recovery_placeholders})",
                (
                    self._CONFIRMED_COMFY_RESTART_STAGE,
                    now,
                    now,
                    job_id,
                    *recovery_stages,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    "restart recovery parent changed during confirmation"
                )

            # If this ambiguous prompt was the durable Ray queue tail, retain
            # the last known actor descriptor and monotonic epoch, but remove
            # the no-longer-observable tail and taint the runtime.  The next
            # dispatcher must therefore issue the normal safety barrier before
            # submitting compatible work.
            if target_prompt_ids:
                runtime_row = db.execute(
                    "SELECT descriptor FROM raylight_runtime_state "
                    "WHERE singleton = 1"
                ).fetchone()
                if runtime_row is not None:
                    runtime_state = self._decode_raylight_runtime_state(
                        runtime_row["descriptor"]
                    )
                    if runtime_state.get("tail_prompt_id") in target_prompt_ids:
                        runtime_state.update(
                            tail_prompt_id=None,
                            tail_action=None,
                            tainted=True,
                        )
                        runtime_state.pop("tail_terminal_certificate", None)
                        db.execute(
                            "UPDATE raylight_runtime_state SET descriptor = ?, "
                            "updated_at = ? WHERE singleton = 1",
                            (
                                json.dumps(
                                    runtime_state,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                ),
                                now,
                            ),
                        )

            settled = db.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if settled is None:
            raise KeyError(job_id)
        return self._job_row(settled)

    def confirm_raylight_runtime_restart(
        self,
        *,
        expected_epoch: int,
        expected_runtime_state: dict[str, Any],
        visible_gpu_count: int,
    ) -> tuple[dict[str, Any], Path]:
        """Atomically discard only a confirmed pre-restart RayLight ledger.

        The caller owns the endpoint submission lock and carries the
        operator's explicit ComfyUI-restart certificate.  This transaction
        closes races with settings changes, active Director jobs and runtime
        tail settlement before clearing actor identity.  Epoch is deliberately
        monotonic so the next pool cannot reuse killed ComfyUI cache entries.
        """

        if expected_epoch < 0 or visible_gpu_count < 0:
            raise ValueError("RayLight runtime recovery inputs are invalid")
        expected_state = self._decode_raylight_runtime_state(
            json.dumps(
                expected_runtime_state,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            active_parent = db.execute(
                "SELECT id FROM jobs "
                "WHERE status NOT IN ('succeeded', 'failed', 'cancelled') LIMIT 1"
            ).fetchone()
            active_child = db.execute(
                "SELECT id FROM job_children "
                "WHERE status NOT IN ('succeeded', 'failed', 'cancelled') LIMIT 1"
            ).fetchone()
            if active_parent is not None or active_child is not None:
                raise ValueError(
                    "RayLight runtime recovery requires all Director jobs to be terminal"
                )

            row = db.execute(
                "SELECT descriptor FROM raylight_runtime_state "
                "WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise ValueError("RayLight runtime state no longer exists")
            state = self._decode_raylight_runtime_state(row["descriptor"])
            if state != expected_state:
                raise ValueError(
                    "RayLight runtime changed before recovery completed"
                )
            if int(state["epoch"]) != expected_epoch:
                raise ValueError(
                    "RayLight runtime epoch changed before recovery completed"
                )
            current = state.get("current")
            legacy_unknown = bool(state.get("legacy_unknown"))
            if legacy_unknown:
                # Pre-v2 rows carry neither an authenticated actor descriptor
                # nor an exact tail prompt.  They may be cleared only through
                # the same endpoint lock + explicit ComfyUI restart
                # certificate used for stale GPU topology.  Requiring a
                # fictitious history record would make this safe migration
                # state permanently unrecoverable.
                if current is not None or not bool(state.get("tainted")):
                    raise ValueError(
                        "legacy RayLight recovery state is contradictory"
                    )
            else:
                if current is None:
                    raise ValueError(
                        "RayLight runtime was already cleared before recovery completed"
                    )
                recorded = raylight_runtime_logical_gpu_indices(current)
                invalid = tuple(
                    index for index in recorded if index >= visible_gpu_count
                )
                if not invalid:
                    raise ValueError(
                        "RayLight runtime no longer references unavailable logical GPUs"
                    )
            recovered = {
                "version": 2,
                "epoch": expected_epoch,
                "current": None,
                "tail_prompt_id": None,
                "tail_action": None,
                "tainted": False,
            }
            backup_path = self.path.with_name(
                f"{self.path.stem}.before-raylight-recovery-"
                f"e{expected_epoch}-{uuid.uuid4().hex}.sqlite3"
            )
            backup_published = False
            commit_attempted = False
            try:
                _create_crash_safe_sqlite_backup(self.path, backup_path)
                backup_published = True
                db.execute(
                    "UPDATE raylight_runtime_state SET descriptor = ?, updated_at = ? "
                    "WHERE singleton = 1",
                    (
                        json.dumps(recovered, ensure_ascii=False, sort_keys=True),
                        utc_now(),
                    ),
                )
                # Once COMMIT begins its outcome is intrinsically ambiguous:
                # SQLite may have made the ledger durable before an I/O wrapper
                # reports failure. Never retract the already-valid backup past
                # this boundary, or an uncertain successful clear could be left
                # without its recovery copy.
                commit_attempted = True
                _commit_database(db)
            except BaseException:
                if backup_published and not commit_attempted:
                    try:
                        _remove_sqlite_artifacts(backup_path)
                    except OSError:
                        pass
                    try:
                        _fsync_directory(backup_path.parent)
                    except OSError:
                        pass
                raise
        return recovered, backup_path

    def get_draft(self, mode: GenerationMode) -> ModeDraft:
        with self.connect() as db:
            row = db.execute("SELECT document FROM mode_drafts WHERE mode = ?", (mode,)).fetchone()
        if row is None:
            raise RuntimeError(f"draft row for {mode} is missing")
        return validate_mode_draft(mode, json.loads(row["document"]))

    def put_draft(self, mode: GenerationMode, draft: ModeDraft) -> ModeDraft:
        with self.connect() as db:
            db.execute(
                "UPDATE mode_drafts SET document = ?, updated_at = ? WHERE mode = ?",
                (draft.model_dump_json(), utc_now(), mode),
            )
        return draft

    def validate_and_put_draft(
        self,
        mode: GenerationMode,
        draft: ModeDraft,
    ) -> ModeDraft:
        """Validate every asset and save the legacy draft in one write transaction.

        ``BEGIN IMMEDIATE`` serializes this operation with cascade deletion.
        Without that shared lock, an asset could disappear after validation
        but before the document write, resurrecting a dangling reference.
        """

        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._validate_asset_iterator_in_connection(
                db,
                iter_draft_assets(draft),
            )
            db.execute(
                "UPDATE mode_drafts SET document = ?, updated_at = ? WHERE mode = ?",
                (draft.model_dump_json(), utc_now(), mode),
            )
        return draft

    # The pre-multi-project singleton timeline is the authority for the
    # legacy/default project (id ``LEGACY_DEFAULT_PROJECT_ID``). Only projects
    # created through ``create_project`` live in the ``projects`` table, so the
    # singleton and that table can never hold two diverging copies.
    def _is_legacy_project_id(self, project_id: str) -> bool:
        return project_id == self.LEGACY_DEFAULT_PROJECT_ID

    def get_project_migration_receipt(
        self,
        project_id: str,
        migration_id: str,
    ) -> ProjectMigrationReceipt | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT migration_id, project_id, from_schema, to_schema, "
                "receipt, receipt_digest, created_at "
                "FROM project_migration_receipts "
                "WHERE project_id = ? AND migration_id = ?",
                (project_id, migration_id),
            ).fetchone()
        return None if row is None else decode_project_migration_receipt(row)

    def get_latest_project_migration_receipt(
        self,
        project_id: str,
        *,
        from_schema: int = 4,
        to_schema: int = 5,
    ) -> ProjectMigrationReceipt | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT migration_id, project_id, from_schema, to_schema, "
                "receipt, receipt_digest, created_at "
                "FROM project_migration_receipts "
                "WHERE project_id = ? AND from_schema = ? AND to_schema = ? "
                "ORDER BY created_at DESC, migration_id DESC LIMIT 1",
                (project_id, from_schema, to_schema),
            ).fetchone()
        return None if row is None else decode_project_migration_receipt(row)

    def _require_v5_timeline_write(
        self,
        project_id: str,
        timeline: UnifiedTimelineDraftV5 | UnifiedTimelineDraft,
    ) -> UnifiedTimelineDraftV5:
        if isinstance(timeline, UnifiedTimelineDraft):
            receipt = self.get_latest_project_migration_receipt(project_id)
            raise TimelineSchemaMigrated(
                project_id,
                None if receipt is None else receipt.migration_id,
            )
        if not isinstance(timeline, UnifiedTimelineDraftV5):
            raise TypeError("timeline write requires UnifiedTimelineDraftV5")
        # Full-state browser WAL replay may still carry a valid schema-5,
        # bundle-4 document.  Upgrade it at the CAS boundary so an old payload
        # can never write bundle 4 back after startup migration.  Invalid or
        # future feature authority fails closed through the shared resolver.
        return migrate_timeline_feature_authority_to_v5(timeline)

    @staticmethod
    def _assert_expected_timeline_revision(
        *,
        project_id: str,
        expected_revision: int,
        actual_revision: int,
    ) -> None:
        if expected_revision != actual_revision:
            raise TimelineRevisionConflict(
                project_id,
                expected_revision,
                actual_revision,
            )
        if actual_revision >= MAX_TIMELINE_REVISION:
            raise TimelineRevisionExhausted(project_id, actual_revision)

    def get_timeline_authority(self) -> tuple[UnifiedTimelineDraftV5, int]:
        """Return the default v5 project document and durable CAS revision."""

        with self.connect() as db:
            row = db.execute(
                "SELECT document, revision FROM unified_timeline WHERE singleton = 1"
            ).fetchone()
        if row is None:
            raise RuntimeError("unified timeline row is missing")
        return (
            validate_timeline_draft_v5(json.loads(row["document"])),
            int(row["revision"]),
        )

    def get_timeline(self) -> UnifiedTimelineDraftV5:
        return self.get_timeline_authority()[0]

    def validate_and_put_timeline_authority(
        self,
        timeline: UnifiedTimelineDraftV5 | UnifiedTimelineDraft,
        *,
        expected_revision: int,
    ) -> tuple[UnifiedTimelineDraftV5, int]:
        """CAS-replace the default timeline under the asset-validation lock."""

        timeline = self._require_v5_timeline_write(
            self.LEGACY_DEFAULT_PROJECT_ID, timeline
        )
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT revision FROM unified_timeline WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise RuntimeError("unified timeline row is missing")
            current_revision = int(row["revision"])
            self._assert_expected_timeline_revision(
                project_id=self.LEGACY_DEFAULT_PROJECT_ID,
                expected_revision=expected_revision,
                actual_revision=current_revision,
            )
            self._validate_asset_iterator_in_connection(
                db,
                iter_timeline_assets(timeline),
            )
            cursor = db.execute(
                "UPDATE unified_timeline SET document = ?, updated_at = ?, "
                "revision = revision + 1 WHERE singleton = 1 AND revision = ?",
                (timeline.model_dump_json(), utc_now(), current_revision),
            )
            if cursor.rowcount != 1:
                # BEGIN IMMEDIATE serializes writers, so this can only indicate
                # database corruption or a trigger that violated the CAS row.
                raise RuntimeError("timeline CAS update did not affect one row")
        return timeline, current_revision + 1

    @staticmethod
    def _project_row_summary(row: sqlite3.Row) -> dict[str, Any]:
        timeline = validate_timeline_draft_v5(json.loads(row["document"]))
        return {
            "id": str(row["id"]),
            "title": str(row["title"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "segment_count": len(timeline.segments),
        }

    def _legacy_project_row(self, db: sqlite3.Connection) -> dict[str, Any] | None:
        row = db.execute(
            "SELECT document, updated_at FROM unified_timeline WHERE singleton = 1"
        ).fetchone()
        if row is None:
            return None
        timeline = validate_timeline_draft_v5(json.loads(row["document"]))
        updated_at = str(row["updated_at"])
        return {
            "id": self.LEGACY_DEFAULT_PROJECT_ID,
            "title": timeline.title,
            "document": timeline.model_dump(mode="json"),
            "created_at": updated_at,
            "updated_at": updated_at,
        }

    def list_projects(self) -> list[dict[str, Any]]:
        """Return summaries for the legacy project plus every created project."""

        with self.connect() as db:
            legacy = self._legacy_project_row(db)
            rows = db.execute(
                "SELECT id, title, document, created_at, updated_at "
                "FROM projects ORDER BY updated_at DESC, id"
            ).fetchall()
        summaries: list[dict[str, Any]] = []
        if legacy is not None:
            timeline = validate_timeline_draft_v5(legacy["document"])
            summaries.append(
                {
                    "id": legacy["id"],
                    "title": legacy["title"],
                    "created_at": legacy["created_at"],
                    "updated_at": legacy["updated_at"],
                    "segment_count": len(timeline.segments),
                }
            )
        for row in rows:
            try:
                summaries.append(self._project_row_summary(row))
            except (TypeError, ValueError, ValidationError, json.JSONDecodeError):
                # A corrupt project row must not hide the whole list; expose it
                # so the operator can still delete it explicitly.
                continue
        return summaries

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        if self._is_legacy_project_id(project_id):
            with self.connect() as db:
                return self._legacy_project_row(db)
        with self.connect() as db:
            row = db.execute(
                "SELECT id, title, document, created_at, updated_at "
                "FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": str(row["id"]),
            "title": str(row["title"]),
            "document": json.loads(row["document"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def project_exists(self, project_id: str) -> bool:
        """Check project scope without reading its mutable creative document."""

        with self.connect() as db:
            if self._is_legacy_project_id(project_id):
                row = db.execute(
                    "SELECT 1 FROM unified_timeline WHERE singleton = 1"
                ).fetchone()
            else:
                row = db.execute(
                    "SELECT 1 FROM projects WHERE id = ?", (project_id,)
                ).fetchone()
        return row is not None

    def create_project(
        self,
        title: str | None = None,
        initial_model_stack: ModelStack | None = None,
    ) -> dict[str, Any]:
        """Create a fresh project with a new stable segment identity."""

        project_id = str(uuid.uuid4())
        timeline = default_timeline_draft_v5(initial_model_stack)
        # A fresh stable segment id guarantees the durable take ledger can
        # never match this project's segment against another project's render.
        fresh_segment = timeline.segments[0].model_copy(
            update={"id": f"timeline-segment-{uuid.uuid4().hex}"}
        )
        normalized_title = (title or "").strip() or "未命名长视频"
        timeline = timeline.model_copy(
            update={"title": normalized_title, "segments": [fresh_segment]}
        )
        now = utc_now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._validate_asset_iterator_in_connection(
                db,
                iter_timeline_assets(timeline),
            )
            db.execute(
                "INSERT INTO projects(id, title, document, created_at, updated_at) "
                "VALUES(?, ?, ?, ?, ?)",
                (project_id, timeline.title, timeline.model_dump_json(), now, now),
            )
        project = self.get_project(project_id)
        if project is None:
            raise RuntimeError("created project disappeared")
        return project

    def import_project(
        self, title: str, timeline: UnifiedTimelineDraftV5 | UnifiedTimelineDraft
    ) -> dict[str, Any]:
        """Create a project from an existing validated timeline document.

        Segment identities are preserved so a restored historical project keeps
        its stable structure. The take ledger remains scoped by this new
        project id, so imported segments start without reused renders.
        """

        if isinstance(timeline, UnifiedTimelineDraft):
            raise TimelineSchemaMigrated("import", None)
        if not isinstance(timeline, UnifiedTimelineDraftV5):
            raise TypeError("project import requires UnifiedTimelineDraftV5")
        timeline = migrate_timeline_feature_authority_to_v5(timeline)
        project_id = str(uuid.uuid4())
        normalized_title = (
            title.strip() or timeline.title.strip() or "未命名长视频"
        )
        # Title is part of the v5 creative authority. Keep the denormalized
        # list/index column in lockstep with that document at the one allowed
        # creation boundary; otherwise import would recreate the retired
        # project-rename side channel before the first CAS edit.
        timeline = timeline.model_copy(update={"title": normalized_title}, deep=True)
        now = utc_now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._validate_asset_iterator_in_connection(
                db,
                iter_timeline_assets(timeline),
            )
            db.execute(
                "INSERT INTO projects(id, title, document, created_at, updated_at) "
                "VALUES(?, ?, ?, ?, ?)",
                (project_id, normalized_title, timeline.model_dump_json(), now, now),
            )
        project = self.get_project(project_id)
        if project is None:
            raise RuntimeError("imported project disappeared")
        return project

    def delete_project(self, project_id: str) -> int:
        """Delete one project and orphan its task history for audit.

        Task rows keep their outputs but lose project ownership; the take
        ledger rows remain harmless orphans because a scoped lookup always
        names a concrete project id. The legacy/default project cannot be
        deleted because it is the singleton timeline itself.
        """

        if self._is_legacy_project_id(project_id):
            raise ValueError("cannot delete the default project")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cursor = db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            if cursor.rowcount != 1:
                raise KeyError(project_id)
            orphaned = db.execute(
                "UPDATE jobs SET project_id = NULL WHERE project_id = ?",
                (project_id,),
            ).rowcount
        return int(orphaned)

    def get_project_timeline(self, project_id: str) -> UnifiedTimelineDraftV5:
        # Keep the legacy/default read flowing through get_timeline(). Besides
        # preserving the established public delegation contract, job-list
        # snapshot comparison can continue to prefetch that authority once.
        if self._is_legacy_project_id(project_id):
            return self.get_timeline()
        return self.get_project_timeline_authority(project_id)[0]

    def get_project_timeline_authority(
        self, project_id: str
    ) -> tuple[UnifiedTimelineDraftV5, int]:
        if self._is_legacy_project_id(project_id):
            return self.get_timeline_authority()
        with self.connect() as db:
            row = db.execute(
                "SELECT document, revision FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
        if row is None:
            raise KeyError(project_id)
        return (
            validate_timeline_draft_v5(json.loads(row["document"])),
            int(row["revision"]),
        )

    def validate_and_put_project_timeline_authority(
        self,
        project_id: str,
        timeline: UnifiedTimelineDraftV5 | UnifiedTimelineDraft,
        *,
        expected_revision: int,
    ) -> tuple[UnifiedTimelineDraftV5, int]:
        """CAS-replace one project timeline under the asset-validation lock."""

        if self._is_legacy_project_id(project_id):
            return self.validate_and_put_timeline_authority(
                timeline,
                expected_revision=expected_revision,
            )
        timeline = self._require_v5_timeline_write(project_id, timeline)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT revision FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
            if row is None:
                raise KeyError(project_id)
            current_revision = int(row["revision"])
            self._assert_expected_timeline_revision(
                project_id=project_id,
                expected_revision=expected_revision,
                actual_revision=current_revision,
            )
            self._validate_asset_iterator_in_connection(
                db,
                iter_timeline_assets(timeline),
            )
            cursor = db.execute(
                "UPDATE projects SET document = ?, title = ?, updated_at = ?, "
                "revision = revision + 1 WHERE id = ? AND revision = ?",
                (
                    timeline.model_dump_json(),
                    timeline.title,
                    utc_now(),
                    project_id,
                    current_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("project timeline CAS update did not affect one row")
        return timeline, current_revision + 1

    @staticmethod
    def _document_digest(document: str) -> str:
        return hashlib.sha256(document.encode("utf-8")).hexdigest()

    def put_asset(
        self,
        asset_id: str,
        document: dict[str, Any],
    ) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO assets(id, document, created_at) VALUES(?, ?, ?)",
                (asset_id, json.dumps(document, ensure_ascii=False), utc_now()),
            )

    def get_asset(self, asset_id: str) -> dict[str, Any] | None:
        record = self.get_asset_record(asset_id)
        return record if record is not None else None

    def get_asset_record(
        self,
        asset_id: str,
        *,
        include_trashed: bool = False,
    ) -> dict[str, Any] | None:
        live_clause = "" if include_trashed else " AND trashed_at IS NULL"
        with self.connect() as db:
            row = db.execute(
                "SELECT document FROM assets WHERE id = ?"
                + live_clause,
                (asset_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["document"])

    def list_assets(
        self,
        *,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT document FROM assets WHERE trashed_at IS NULL "
                "ORDER BY created_at DESC, id DESC",
            ).fetchall()
        assets: list[dict[str, Any]] = []
        for row in rows:
            asset = AssetReference.model_validate_json(row["document"])
            if kind is None or asset.kind == kind:
                assets.append(asset.model_dump(mode="json"))
        return assets

    @staticmethod
    def _decode_trash_batch_row(row: sqlite3.Row) -> dict[str, Any]:
        asset_ids = json.loads(row["asset_ids"])
        usages_by_asset = json.loads(row["unbound_usages"])
        if (
            not isinstance(asset_ids, list)
            or not all(isinstance(item, str) for item in asset_ids)
            or not isinstance(usages_by_asset, dict)
            or set(usages_by_asset) != set(asset_ids)
            or not all(
                isinstance(values, list)
                and all(isinstance(value, str) for value in values)
                for values in usages_by_asset.values()
            )
        ):
            raise RuntimeError(f"asset trash batch '{row['id']}' is invalid")
        return {
            "batch_id": str(row["id"]),
            "asset_ids": list(asset_ids),
            "cascade": bool(row["cascade"]),
            "unbound_usages_by_asset": {
                asset_id: list(usages_by_asset[asset_id])
                for asset_id in asset_ids
            },
            "unbound_usages": [
                usage
                for asset_id in asset_ids
                for usage in usages_by_asset[asset_id]
            ],
            "created_at": str(row["created_at"]),
        }

    def _trash_batch_read_in_connection(
        self,
        db: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> dict[str, Any]:
        batch = self._decode_trash_batch_row(row)
        asset_rows = db.execute(
            "SELECT id, document FROM assets WHERE trash_batch_id = ? "
            "AND trashed_at IS NOT NULL",
            (batch["batch_id"],),
        ).fetchall()
        documents = {str(item["id"]): str(item["document"]) for item in asset_rows}
        if set(documents) != set(batch["asset_ids"]):
            raise RuntimeError(
                f"asset trash batch '{batch['batch_id']}' has inconsistent registrations"
            )
        batch["assets"] = [
            AssetReference.model_validate_json(documents[asset_id]).model_dump(
                mode="json"
            )
            for asset_id in batch["asset_ids"]
        ]
        return batch

    def list_asset_trash_batches(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            db.execute("BEGIN")
            rows = db.execute(
                "SELECT id, asset_ids, cascade, unbound_usages, "
                "created_at FROM asset_trash_batches "
                "ORDER BY created_at DESC, id DESC",
            ).fetchall()
            return [self._trash_batch_read_in_connection(db, row) for row in rows]

    @staticmethod
    def _dense_unbind_references(
        item: dict[str, Any],
        field: str,
        asset_id: str,
        *,
        label_offset: int = 0,
    ) -> tuple[bool, dict[int, int | None]]:
        """Remove one registered reference and preserve H3's dense tag identity."""

        references = item.get(field)
        if not isinstance(references, list):
            return False, {}
        ordered = sorted(
            (reference for reference in references if isinstance(reference, dict)),
            key=lambda reference: int(reference.get("slot", 0)),
        )
        if not any(reference.get("id") == asset_id for reference in ordered):
            return False, {}
        retained: list[dict[str, Any]] = []
        label_map: dict[int, int | None] = {}
        for reference in ordered:
            old_label = int(reference["slot"]) + 1 + label_offset
            if reference.get("id") == asset_id:
                label_map[old_label] = None
                continue
            new_slot = len(retained)
            label_map[old_label] = new_slot + 1 + label_offset
            reference["slot"] = new_slot
            retained.append(reference)
        item[field] = retained
        return True, label_map

    @classmethod
    def _rewrite_reference_prompt(
        cls,
        prompt: str,
        label_maps: dict[str, dict[int, int | None]],
        *,
        clear_kinds: set[str] | None = None,
    ) -> str:
        cleared = clear_kinds or set()

        def replace(match: re.Match[str]) -> str:
            kind = match.group(1).title()
            old_label = int(match.group(2))
            if kind in cleared:
                return ""
            mapping = label_maps.get(kind)
            if mapping is None or old_label not in mapping:
                return match.group(0)
            new_label = mapping[old_label]
            return "" if new_label is None else f"<{kind} {new_label}>"

        # Prompt formatting is authored content. Rewriting a typed reference
        # may change only that token; whitespace (including blank lines and
        # indentation) must survive asset cascade operations byte-for-byte.
        return cls._REFERENCE_TAG.sub(replace, prompt)

    @classmethod
    def _unbind_asset_document(
        cls, document: dict[str, Any], asset_id: str
    ) -> bool:
        """Unbind typed media while retaining a runnable canonical timeline.

        Stock H3 reference autogrow inputs are positional. Removing slot zero
        while retaining slot one therefore requires both dense renumbering and
        prompt-tag rewriting. For inherited prompts, the affected segment/shot
        first materializes the shared prompt so other segments keep their own
        label meaning.
        """

        changed = False
        collections = document.get("segments")
        unified = isinstance(collections, list)
        if not unified:
            collections = document.get("shots")
        if not isinstance(collections, list):
            return False
        for item in collections:
            if not isinstance(item, dict):
                continue
            item_changed = False
            source_removed = False
            removed_scalars: set[str] = set()
            item_prompt = str(item.get("prompt") or "")
            root_prompt = str(document.get("prompt") or "")
            # Use trim only to choose between local and inherited text. Once
            # chosen, preserve the authored bytes for the semantic tag rewrite.
            effective_prompt = item_prompt if item_prompt.strip() else root_prompt
            mode = str(item.get("mode") if unified else document.get("mode") or "")
            anchor_fields = (
                ("first_image", "last_image")
                if (unified and mode == "fl2va")
                or (not unified and mode == "fl2v")
                else (("first_image",) if not unified and mode == "i2v" else ())
            )
            old_anchor_labels = {
                field: index + 1
                for index, field in enumerate(
                    field
                    for field in anchor_fields
                    if isinstance(item.get(field), dict)
                )
            }
            label_maps: dict[str, dict[int, int | None]] = {}
            for field in ("first_image", "last_image", "source_video"):
                reference = item.get(field)
                if isinstance(reference, dict) and reference.get("id") == asset_id:
                    item[field] = None
                    removed_scalars.add(field)
                    source_removed = source_removed or field == "source_video"
                    item_changed = True
                    changed = True
            if removed_scalars.intersection(anchor_fields):
                new_anchor_labels = {
                    field: index + 1
                    for index, field in enumerate(
                        field
                        for field in anchor_fields
                        if isinstance(item.get(field), dict)
                    )
                }
                label_maps["Picture"] = {
                    old_label: new_anchor_labels.get(field)
                    for field, old_label in old_anchor_labels.items()
                }
            for field, kind in cls._REFERENCE_FIELDS.items():
                label_offset = (
                    1
                    if unified
                    and mode == "ref2va"
                    and (
                        (
                            field == "reference_audios"
                            and item.get("source_audio_as_reference") is True
                        )
                        or (
                            field == "reference_videos"
                            and item.get("source_video") is not None
                        )
                    )
                    else 0
                )
                unbound, label_map = cls._dense_unbind_references(
                    item,
                    field,
                    asset_id,
                    label_offset=label_offset,
                )
                if unbound:
                    label_maps[kind] = label_map
                    item_changed = True
                    changed = True

            clear_kinds: set[str] = set()
            if unified:
                if mode == "ref2va" and source_removed:
                    # The source owns <Video 1>; independent videos move from
                    # 2..N to 1..N-1 after it is unbound. Its optional paired
                    # soundtrack similarly owns <Audio 1>.
                    video_map = {1: None}
                    video_map.update(
                        {
                            int(reference["slot"]) + 2:
                            int(reference["slot"]) + 1
                            for reference in item.get("reference_videos") or []
                            if isinstance(reference, dict)
                        }
                    )
                    label_maps["Video"] = video_map
                    if item.get("source_audio_as_reference") is True:
                        audio_map = {1: None}
                        audio_map.update(
                            {
                                int(reference["slot"]) + 2:
                                int(reference["slot"]) + 1
                                for reference in item.get("reference_audios") or []
                                if isinstance(reference, dict)
                            }
                        )
                        label_maps["Audio"] = audio_map
                    item["source_audio_as_reference"] = False
                    item["source_start_seconds"] = 0.0
                    item["source_duration_seconds"] = 5.0
                    item_changed = True
            elif source_removed and mode in {"v2v", "rv2v"}:
                # Legacy mode drafts cannot change their discriminated mode.
                # They remain editable but must not retain a tag for a source
                # that this transaction removed.
                clear_kinds = {"Video"}

            if label_maps or clear_kinds:
                # Only materialize inheritance when this exact item needs a
                # semantic rewrite; untouched items keep sharing the root text.
                rewritten = cls._rewrite_reference_prompt(
                    effective_prompt,
                    label_maps,
                    clear_kinds=clear_kinds,
                )
                if item.get("prompt") != rewritten:
                    item["prompt"] = rewritten
                    item_changed = True
                    changed = True
            if unified and source_removed and item.get("enabled", True):
                if item.get("audio_mode") == "source":
                    item["audio_mode"] = "generate"
                    item_changed = True

            # Keep the local flag explicit: it documents that scalar-only
            # changes still count even when they need no tag or mode rewrite.
            changed = changed or item_changed
        return changed

    def _saved_asset_document_owners_in_connection(
        self, db: sqlite3.Connection
    ) -> list[dict[str, Any]]:
        owners: list[dict[str, Any]] = []
        singleton = db.execute(
            "SELECT document, revision FROM unified_timeline WHERE singleton = 1"
        ).fetchone()
        if singleton is not None:
            owners.append(
                {
                    "kind": "timeline",
                    "id": self.LEGACY_DEFAULT_PROJECT_ID,
                    "label": "timeline",
                    "document": str(singleton["document"]),
                    "revision": int(singleton["revision"]),
                }
            )
        for row in db.execute(
            "SELECT id, document, revision FROM projects ORDER BY id"
        ).fetchall():
            owner_id = str(row["id"])
            owners.append(
                {
                    "kind": "project",
                    "id": owner_id,
                    "label": f"project.{owner_id}",
                    "document": str(row["document"]),
                    "revision": int(row["revision"]),
                }
            )
        for row in db.execute(
            "SELECT mode, document FROM mode_drafts ORDER BY mode"
        ).fetchall():
            mode = str(row["mode"])
            owners.append(
                {
                    "kind": "draft",
                    "id": mode,
                    "label": f"drafts.{mode}",
                    "document": str(row["document"]),
                    "revision": None,
                }
            )
        return owners

    def trash_assets(
        self,
        asset_ids: list[str],
        *,
        cascade: bool,
    ) -> dict[str, Any]:
        """Atomically tombstone one user-intent batch and save its inverse bundle."""

        normalized_ids: list[str] = []
        seen: set[str] = set()
        for asset_id in asset_ids:
            if not isinstance(asset_id, str) or not asset_id:
                raise ValueError("asset ids must be non-empty strings")
            if asset_id in seen:
                raise ValueError("asset ids must be unique")
            seen.add(asset_id)
            normalized_ids.append(asset_id)
        if not normalized_ids:
            raise ValueError("at least one asset id is required")
        if len(normalized_ids) > 128:
            raise ValueError("at most 128 assets may be trashed together")

        placeholders = ",".join("?" for _ in normalized_ids)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            asset_rows = db.execute(
                "SELECT id, document, trashed_at FROM assets "
                f"WHERE id IN ({placeholders})",
                tuple(normalized_ids),
            ).fetchall()
            rows_by_id = {str(row["id"]): row for row in asset_rows}
            for asset_id in normalized_ids:
                row = rows_by_id.get(asset_id)
                if row is None or row["trashed_at"] is not None:
                    raise KeyError(asset_id)
                AssetReference.model_validate_json(row["document"])

            target_ids = set(normalized_ids)
            usages_by_asset = {asset_id: [] for asset_id in normalized_ids}
            owners = self._saved_asset_document_owners_in_connection(db)
            for owner in owners:
                if owner["kind"] == "draft":
                    validated = validate_mode_draft(
                        owner["id"], json.loads(owner["document"])
                    )
                    references = iter_draft_assets(validated)
                else:
                    validated = validate_timeline_draft_v5(
                        json.loads(owner["document"])
                    )
                    references = iter_timeline_assets(validated)
                for location, asset in references:
                    if asset.id in target_ids:
                        usages_by_asset[asset.id].append(
                            f"{owner['label']}.{location}"
                        )
            if any(usages_by_asset.values()) and not cascade:
                raise AssetTrashInUse(usages_by_asset)

            batch_id = str(uuid.uuid4())
            now = utc_now()
            db.execute(
                "INSERT INTO asset_trash_batches("
                "id, asset_ids, cascade, unbound_usages, created_at"
                ") VALUES(?, ?, ?, ?, ?)",
                (
                    batch_id,
                    json.dumps(normalized_ids, ensure_ascii=False),
                    int(cascade),
                    json.dumps(usages_by_asset, ensure_ascii=False),
                    now,
                ),
            )

            if cascade and any(usages_by_asset.values()):
                for owner in owners:
                    document = json.loads(owner["document"])
                    changed = False
                    for asset_id in normalized_ids:
                        changed = (
                            self._unbind_asset_document(document, asset_id)
                            or changed
                        )
                    if not changed:
                        continue
                    if owner["kind"] == "draft":
                        normalized = validate_mode_draft(owner["id"], document)
                    else:
                        normalized = validate_timeline_draft_v5(document)
                        if owner["revision"] >= MAX_TIMELINE_REVISION:
                            raise OverflowError("timeline revision space is exhausted")
                    after_document = normalized.model_dump_json()
                    after_revision = (
                        None
                        if owner["revision"] is None
                        else int(owner["revision"]) + 1
                    )
                    if owner["kind"] == "timeline":
                        cursor = db.execute(
                            "UPDATE unified_timeline SET document = ?, updated_at = ?, "
                            "revision = revision + 1 WHERE singleton = 1 "
                            "AND revision = ?",
                            (after_document, now, owner["revision"]),
                        )
                    elif owner["kind"] == "project":
                        cursor = db.execute(
                            "UPDATE projects SET document = ?, title = ?, "
                            "updated_at = ?, revision = revision + 1 "
                            "WHERE id = ? AND revision = ?",
                            (
                                after_document,
                                normalized.title,
                                now,
                                owner["id"],
                                owner["revision"],
                            ),
                        )
                    else:
                        cursor = db.execute(
                            "UPDATE mode_drafts SET document = ?, updated_at = ? "
                            "WHERE mode = ?",
                            (after_document, now, owner["id"]),
                        )
                    if cursor.rowcount != 1:
                        raise RuntimeError(
                            "asset cascade document update did not affect one row"
                        )
                    db.execute(
                        "INSERT INTO asset_trash_document_changes("
                        "batch_id, owner_kind, owner_id, before_document, "
                        "after_digest, after_revision) VALUES(?, ?, ?, ?, ?, ?)",
                        (
                            batch_id,
                            owner["kind"],
                            owner["id"],
                            owner["document"],
                            self._document_digest(after_document),
                            after_revision,
                        ),
                    )

            cursor = db.execute(
                "UPDATE assets SET trashed_at = ?, trash_batch_id = ? "
                f"WHERE id IN ({placeholders}) AND trashed_at IS NULL",
                (now, batch_id, *normalized_ids),
            )
            if cursor.rowcount != len(normalized_ids):
                raise RuntimeError(
                    "asset trash update did not affect every requested registration"
                )
            batch_row = db.execute(
                "SELECT id, asset_ids, cascade, unbound_usages, "
                "created_at FROM asset_trash_batches WHERE id = ?",
                (batch_id,),
            ).fetchone()
            if batch_row is None:
                raise RuntimeError("created asset trash batch disappeared")
            return self._trash_batch_read_in_connection(db, batch_row)

    def delete_asset_if_unused(
        self,
        asset_id: str,
        *,
        cascade: bool = False,
    ) -> list[str]:
        """Compatibility wrapper around the recoverable single-asset trash path."""

        if self.get_asset_record(asset_id) is None:
            raise KeyError(asset_id)
        try:
            batch = self.trash_assets(
                [asset_id],
                cascade=cascade,
            )
        except AssetTrashInUse as exc:
            return exc.usages
        return list(batch["unbound_usages"]) if cascade else []

    def _require_asset_trash_batch_in_connection(
        self,
        db: sqlite3.Connection,
        batch_id: str,
    ) -> tuple[sqlite3.Row, dict[str, Any]]:
        row = db.execute(
            "SELECT id, asset_ids, cascade, unbound_usages, "
            "created_at FROM asset_trash_batches WHERE id = ?",
            (batch_id,),
        ).fetchone()
        if row is None:
            raise KeyError(batch_id)
        return row, self._trash_batch_read_in_connection(db, row)

    @staticmethod
    def _trash_owner_state_in_connection(
        db: sqlite3.Connection,
        owner_kind: str,
        owner_id: str,
    ) -> tuple[str, int | None] | None:
        if owner_kind == "timeline":
            row = db.execute(
                "SELECT document, revision FROM unified_timeline WHERE singleton = 1"
            ).fetchone()
            return (
                (str(row["document"]), int(row["revision"]))
                if row is not None
                else None
            )
        if owner_kind == "project":
            row = db.execute(
                "SELECT document, revision FROM projects WHERE id = ?",
                (owner_id,),
            ).fetchone()
            return (
                (str(row["document"]), int(row["revision"]))
                if row is not None
                else None
            )
        if owner_kind == "draft":
            row = db.execute(
                "SELECT document FROM mode_drafts WHERE mode = ?",
                (owner_id,),
            ).fetchone()
            return (str(row["document"]), None) if row is not None else None
        raise RuntimeError(f"unknown asset trash owner kind '{owner_kind}'")

    def restore_asset_trash_batch(
        self,
        batch_id: str,
        *,
        mode: str,
    ) -> dict[str, Any]:
        if mode not in {"registration_only", "with_references"}:
            raise ValueError("unknown asset trash restore mode")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _row, batch = self._require_asset_trash_batch_in_connection(
                db, batch_id
            )
            changes = db.execute(
                "SELECT owner_kind, owner_id, before_document, after_digest, "
                "after_revision FROM asset_trash_document_changes "
                "WHERE batch_id = ? ORDER BY owner_kind, owner_id",
                (batch_id,),
            ).fetchall()

            conflicts: list[dict[str, Any]] = []
            current_states: dict[tuple[str, str], tuple[str, int | None]] = {}
            if mode == "with_references":
                for change in changes:
                    owner_kind = str(change["owner_kind"])
                    owner_id = str(change["owner_id"])
                    state = self._trash_owner_state_in_connection(
                        db, owner_kind, owner_id
                    )
                    if state is None:
                        conflicts.append(
                            {
                                "owner_kind": owner_kind,
                                "owner_id": owner_id,
                                "reason": "owner_missing",
                            }
                        )
                        continue
                    current_document, current_revision = state
                    current_states[(owner_kind, owner_id)] = state
                    expected_revision = change["after_revision"]
                    reasons: list[str] = []
                    if self._document_digest(current_document) != str(
                        change["after_digest"]
                    ):
                        reasons.append("document_changed")
                    if owner_kind in {"timeline", "project"}:
                        if current_revision != int(expected_revision):
                            reasons.append("revision_changed")
                        if current_revision is not None and current_revision >= MAX_TIMELINE_REVISION:
                            reasons.append("revision_exhausted")
                    if reasons:
                        conflicts.append(
                            {
                                "owner_kind": owner_kind,
                                "owner_id": owner_id,
                                "reason": ",".join(reasons),
                                "expected_revision": (
                                    int(expected_revision)
                                    if expected_revision is not None
                                    else None
                                ),
                                "actual_revision": current_revision,
                            }
                        )
            if conflicts:
                raise AssetTrashRestoreConflict(conflicts)

            placeholders = ",".join("?" for _ in batch["asset_ids"])
            rows = db.execute(
                "SELECT id, trashed_at, trash_batch_id FROM assets "
                f"WHERE id IN ({placeholders})",
                tuple(batch["asset_ids"]),
            ).fetchall()
            rows_by_id = {str(row["id"]): row for row in rows}
            for asset_id in batch["asset_ids"]:
                asset_row = rows_by_id.get(asset_id)
                if (
                    asset_row is None
                    or asset_row["trashed_at"] is None
                    or asset_row["trash_batch_id"] != batch_id
                ):
                    conflicts.append(
                        {
                            "owner_kind": "asset",
                            "owner_id": asset_id,
                            "reason": "registration_changed",
                        }
                    )
            if conflicts:
                raise AssetTrashRestoreConflict(conflicts)

            cursor = db.execute(
                "UPDATE assets SET trashed_at = NULL, trash_batch_id = NULL "
                "WHERE trash_batch_id = ? AND trashed_at IS NOT NULL",
                (batch_id,),
            )
            if cursor.rowcount != len(batch["asset_ids"]):
                raise RuntimeError(
                    "asset restore did not affect every trashed registration"
                )

            if mode == "with_references":
                now = utc_now()
                for change in changes:
                    owner_kind = str(change["owner_kind"])
                    owner_id = str(change["owner_id"])
                    before_document = str(change["before_document"])
                    try:
                        if owner_kind == "draft":
                            restored = validate_mode_draft(
                                owner_id, json.loads(before_document)
                            )
                            self._validate_asset_iterator_in_connection(
                                db,
                                iter_draft_assets(restored),
                            )
                        else:
                            restored = validate_timeline_draft_v5(
                                json.loads(before_document)
                            )
                            self._validate_asset_iterator_in_connection(
                                db,
                                iter_timeline_assets(restored),
                            )
                    except (ValidationError, ValueError, RuntimeError) as exc:
                        raise AssetTrashRestoreConflict(
                            [
                                {
                                    "owner_kind": owner_kind,
                                    "owner_id": owner_id,
                                    "reason": "inverse_document_unavailable",
                                    "message": str(exc),
                                }
                            ]
                        ) from exc

                    if owner_kind == "timeline":
                        _current_document, revision = current_states[
                            (owner_kind, owner_id)
                        ]
                        cursor = db.execute(
                            "UPDATE unified_timeline SET document = ?, updated_at = ?, "
                            "revision = revision + 1 WHERE singleton = 1 "
                            "AND revision = ?",
                            (before_document, now, revision),
                        )
                    elif owner_kind == "project":
                        _current_document, revision = current_states[
                            (owner_kind, owner_id)
                        ]
                        cursor = db.execute(
                            "UPDATE projects SET document = ?, title = ?, "
                            "updated_at = ?, revision = revision + 1 "
                            "WHERE id = ? AND revision = ?",
                            (
                                before_document,
                                restored.title,
                                now,
                                owner_id,
                                revision,
                            ),
                        )
                    else:
                        cursor = db.execute(
                            "UPDATE mode_drafts SET document = ?, updated_at = ? "
                            "WHERE mode = ?",
                            (before_document, now, owner_id),
                        )
                    if cursor.rowcount != 1:
                        raise RuntimeError(
                            "asset restore document update did not affect one row"
                        )

            db.execute("DELETE FROM asset_trash_batches WHERE id = ?", (batch_id,))
            return {
                "batch_id": batch_id,
                "restored_asset_ids": list(batch["asset_ids"]),
                "restored_references": mode == "with_references" and bool(changes),
                "mode": mode,
            }

    def purge_asset_trash_batch(
        self,
        batch_id: str,
    ) -> dict[str, Any]:
        """Forget Director registrations and recovery data, never remote files."""

        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _row, batch = self._require_asset_trash_batch_in_connection(
                db, batch_id
            )
            cursor = db.execute(
                "DELETE FROM assets WHERE trash_batch_id = ? "
                "AND trashed_at IS NOT NULL",
                (batch_id,),
            )
            if cursor.rowcount != len(batch["asset_ids"]):
                raise AssetTrashRestoreConflict(
                    [
                        {
                            "owner_kind": "batch",
                            "owner_id": batch_id,
                            "reason": "registration_changed",
                        }
                    ]
                )
            deleted = db.execute(
                "DELETE FROM asset_trash_batches WHERE id = ?", (batch_id,)
            )
            if deleted.rowcount != 1:
                raise RuntimeError("asset trash purge did not remove its recovery bundle")
            return {
                "batch_id": batch_id,
                "purged_asset_ids": list(batch["asset_ids"]),
            }

    def validate_draft_assets(self, draft: ModeDraft) -> None:
        """Verify draft media against immutable upload records.

        Timeline compilation derives its ComfyUI path only from ``name`` and
        ``subfolder``. Those fields, the media kind/type, and any supplied
        convenience path fields must therefore come from the record created by
        ``POST /api/assets`` rather than from client-controlled JSON.
        """

        self._validate_asset_iterator(iter_draft_assets(draft))

    def validate_timeline_assets(
        self,
        draft: UnifiedTimelineDraft | UnifiedTimelineDraftV5,
        *,
        segment_ids: list[str] | None = None,
    ) -> None:
        self._validate_asset_iterator(
            iter_timeline_assets(
                draft,
                segment_ids=set(segment_ids) if segment_ids is not None else None,
            ),
        )

    def _validate_asset_iterator(
        self,
        references: Any,
    ) -> None:
        with self.connect() as db:
            self._validate_asset_iterator_in_connection(db, references)

    def _validate_asset_iterator_in_connection(
        self,
        db: sqlite3.Connection,
        references: Any,
    ) -> None:
        """Validate references using the caller's transaction and connection."""

        for location, reference in references:
            row = db.execute(
                "SELECT document FROM assets WHERE id = ? "
                "AND trashed_at IS NULL",
                (reference.id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"{location}: asset id '{reference.id}' is not registered")
            try:
                stored = AssetReference.model_validate_json(row["document"])
            except (ValidationError, ValueError) as exc:
                raise RuntimeError(
                    f"stored asset '{reference.id}' is invalid"
                ) from exc
            if stored.id != reference.id:
                raise RuntimeError(
                    f"stored asset '{reference.id}' has an inconsistent document id"
                )
            mismatched = [
                field
                for field in ("name", "subfolder", "type", "kind", "metadata")
                if getattr(reference, field) != getattr(stored, field)
            ]
            # These response convenience fields may be omitted by clients,
            # but if echoed back they cannot be repurposed as forged paths.
            for field in ("filename", "path", "preview_url", "content_hash"):
                supplied = getattr(reference, field)
                if supplied is not None and supplied != getattr(stored, field):
                    mismatched.append(field)
            if mismatched:
                fields = ", ".join(mismatched)
                raise ValueError(
                    f"{location}: asset id '{reference.id}' does not match its registered {fields}"
                )

    @staticmethod
    def _prompt_ownership_row(row: sqlite3.Row) -> PromptOwnership:
        certificate = row["cleanup_certificate"]
        # Strict contracts intentionally reject Python coercion, while their
        # JSON representation uses RFC-3339 datetime strings.  Re-enter through
        # the JSON validator so persisted timestamps retain that strict domain.
        return PromptOwnership.model_validate_json(
            canonical_json(
                {
                "requested_prompt_id": row["requested_prompt_id"],
                "actual_prompt_id": row["actual_prompt_id"],
                "state": row["state"],
                "ownership_revision": row["ownership_revision"],
                "cleanup_certificate": (
                    json.loads(certificate) if certificate is not None else None
                ),
                "updated_at": row["updated_at"],
                }
            )
        )

    @staticmethod
    def _contract_json(value: Any) -> str:
        document = (
            value.model_dump(mode="json")
            if hasattr(value, "model_dump")
            else value
        )
        return canonical_json(document)

    @staticmethod
    def _execution_document_digest(value: Any) -> str:
        document = (
            value.model_dump(mode="json")
            if hasattr(value, "model_dump")
            else value
        )
        return sha256_document_digest(document).value

    @staticmethod
    def _canonical_json_payload_digest(payload: str) -> str:
        if not isinstance(payload, str):
            raise TypeError("canonical JSON payload must be a string")
        return "sha256-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def _prepared_unit_from_compiled_plan_index(
        cls,
        db: sqlite3.Connection,
        *,
        job_id: str,
        plan_row: Mapping[str, Any],
        schema_version: int,
        source_unit_ordinal: int,
    ) -> PreparedSegmentUnit:
        """Decode one authenticated unit without reparsing all sibling units."""

        index_version = int(plan_row["unit_index_version"])
        if index_version == 0:
            plan = CompiledExecutionPlan.model_validate_json(
                plan_row["compiled_plan"]
            )
            if plan.version != schema_version:
                raise ValueError("compiled execution plan schema version drifted")
            if compiled_execution_plan_digest(plan).value != str(
                plan_row["compiled_plan_digest"]
            ):
                raise ValueError("compiled execution plan is not canonical")
            for ordinal, candidate in enumerate(plan.segment_units):
                unit_json = cls._contract_json(candidate)
                db.execute(
                    "INSERT INTO job_execution_plan_units("
                    "job_id, unit_ordinal, unit_id, schema_version, "
                    "source_compiled_plan_digest, prepared_unit, "
                    "prepared_unit_digest) VALUES(?, ?, ?, ?, ?, ?, ?)",
                    (
                        job_id,
                        ordinal,
                        candidate.id,
                        schema_version,
                        str(plan_row["compiled_plan_digest"]),
                        unit_json,
                        cls._canonical_json_payload_digest(unit_json),
                    ),
                )
            updated = db.execute(
                "UPDATE job_execution_plans SET unit_index_version = 1 "
                "WHERE job_id = ? AND unit_index_version = 0",
                (job_id,),
            )
            if updated.rowcount != 1:
                raise ValueError("compiled execution plan unit index raced")
        elif index_version != 1:
            raise ValueError("compiled execution plan unit index is unsupported")

        row = db.execute(
            "SELECT unit_id, schema_version, source_compiled_plan_digest, "
            "prepared_unit, prepared_unit_digest "
            "FROM job_execution_plan_units "
            "WHERE job_id = ? AND unit_ordinal = ?",
            (job_id, source_unit_ordinal),
        ).fetchone()
        if row is None:
            raise ValueError("compiled execution plan unit index is incomplete")
        if (
            int(row["schema_version"]) != schema_version
            or str(row["source_compiled_plan_digest"])
            != str(plan_row["compiled_plan_digest"])
            or cls._canonical_json_payload_digest(str(row["prepared_unit"]))
            != str(row["prepared_unit_digest"])
        ):
            raise ValueError("compiled execution plan unit index is invalid")
        prepared = PreparedSegmentUnit.model_validate_json(row["prepared_unit"])
        if prepared.id != str(row["unit_id"]):
            raise ValueError("compiled execution plan unit identity drifted")
        return prepared

    def create_job_execution_plan(
        self,
        job_id: str,
        plan: CompiledExecutionPlan,
    ) -> CompiledExecutionPlan:
        """Persist one immutable compiler result without touching legacy startup.

        The schema and row are committed together.  A duplicate job id is an
        integrity error even when the bytes are identical: callers must never
        reinterpret an already-created job through a second compiler result.
        """

        plan_json = self._contract_json(plan)
        plan_digest = compiled_execution_plan_digest_from_canonical_json(
            plan_json
        ).value
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._ensure_execution_evidence_schema(db)
            if db.execute(
                "SELECT 1 FROM jobs WHERE id = ?", (job_id,)
            ).fetchone() is None:
                raise KeyError(job_id)
            marker = db.execute(
                "UPDATE jobs SET execution_contract_version = ? "
                "WHERE id = ? AND execution_contract_version IS NULL",
                (self._TYPED_EXECUTION_CONTRACT_VERSION, job_id),
            )
            if marker.rowcount != 1:
                raise sqlite3.IntegrityError(
                    "job already has a typed execution contract"
                )
            db.execute(
                "INSERT INTO job_execution_plans("
                "job_id, schema_version, compiled_plan, "
                "compiled_plan_digest, unit_index_version, created_at) "
                "VALUES(?, ?, ?, ?, 1, ?)",
                (job_id, plan.version, plan_json, plan_digest, utc_now()),
            )
            for ordinal, unit in enumerate(plan.segment_units):
                unit_json = self._contract_json(unit)
                db.execute(
                    "INSERT INTO job_execution_plan_units("
                    "job_id, unit_ordinal, unit_id, schema_version, "
                    "source_compiled_plan_digest, prepared_unit, "
                    "prepared_unit_digest) VALUES(?, ?, ?, ?, ?, ?, ?)",
                    (
                        job_id,
                        ordinal,
                        unit.id,
                        plan.version,
                        plan_digest,
                        unit_json,
                        self._canonical_json_payload_digest(unit_json),
                    ),
                )
        return plan

    def get_job_execution_plan(
        self, job_id: str
    ) -> CompiledExecutionPlan | None:
        """Return ``None`` for legacy jobs without creating Stage-4 tables."""

        with self.connect() as db:
            if not self._execution_evidence_schema_exists(db):
                return None
            row = db.execute(
                "SELECT schema_version, compiled_plan, compiled_plan_digest "
                "FROM job_execution_plans WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        raw_digest = compiled_execution_plan_digest_from_canonical_json(
            row["compiled_plan"]
        ).value
        if raw_digest != row["compiled_plan_digest"]:
            raise ExecutionEvidenceConflict(
                f"compiled execution plan digest mismatch for job {job_id}"
            )
        plan = CompiledExecutionPlan.model_validate_json(row["compiled_plan"])
        if plan.version != row["schema_version"]:
            raise ExecutionEvidenceConflict(
                f"compiled execution plan schema mismatch for job {job_id}"
            )
        return plan

    @classmethod
    def _exact_prompt_snapshot_from_evidence_row(
        cls,
        row: Mapping[str, Any],
        *,
        child_id: str,
    ) -> ExactPromptSnapshot:
        """Validate immutable exact evidence at the transaction boundary."""

        try:
            raw_digest = cls._canonical_json_payload_digest(
                row["exact_prompt_snapshot"]
            )
        except TypeError as exc:
            raise ExecutionEvidenceConflict(
                f"exact prompt evidence is invalid for child {child_id}"
            ) from exc
        if raw_digest != row["exact_prompt_snapshot_digest"]:
            raise ExecutionEvidenceConflict(
                f"exact prompt evidence digest/index mismatch for child {child_id}"
            )
        try:
            snapshot = ExactPromptSnapshot.model_validate_json(
                row["exact_prompt_snapshot"]
            )
        except (TypeError, ValueError) as exc:
            raise ExecutionEvidenceConflict(
                f"exact prompt evidence is invalid for child {child_id}"
            ) from exc
        if (
            snapshot.schema_version != row["schema_version"]
            or snapshot.unit_id != row["unit_id"]
            or snapshot.unit_kind != row["unit_kind"]
            or snapshot.endpoint_identity.endpoint_key != row["endpoint_key"]
            or snapshot.endpoint_identity.runtime_instance_id
            != row["endpoint_runtime_instance_id"]
        ):
            raise ExecutionEvidenceConflict(
                f"exact prompt evidence digest/index mismatch for child {child_id}"
            )
        return snapshot

    def get_job_child_execution_evidence(
        self, child_id: str
    ) -> dict[str, Any] | None:
        """Load immutable exact evidence, or ``None`` for a legacy child."""

        with self.connect() as db:
            if not self._execution_evidence_schema_exists(db):
                return None
            row = db.execute(
                "SELECT * FROM job_child_execution_evidence WHERE child_id = ?",
                (child_id,),
            ).fetchone()
        if row is None:
            return None
        locked_plan, exact_snapshot, _unit = self._execution_evidence_from_row(
            row,
            child_id=child_id,
        )
        return {
            "child_id": child_id,
            "locked_submission_plan": locked_plan,
            "exact_prompt_snapshot": exact_snapshot,
            "created_at": str(row["created_at"]),
        }

    @classmethod
    def _job_has_typed_execution_marker_in_connection(
        cls,
        db: sqlite3.Connection,
        job_id: str,
    ) -> bool:
        """Classify a parent as typed from any surviving durable marker."""

        if cls._typed_execution_marker_column_exists(db):
            marker = db.execute(
                "SELECT execution_contract_version FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if (
                marker is not None
                and marker["execution_contract_version"]
                == cls._TYPED_EXECUTION_CONTRACT_VERSION
            ):
                return True

        if cls._execution_evidence_schema_exists(db):
            marker = db.execute(
                "SELECT "
                "EXISTS(SELECT 1 FROM job_execution_plans WHERE job_id = ?) OR "
                "EXISTS(SELECT 1 FROM job_child_execution_evidence e "
                "JOIN job_children c ON c.id = e.child_id WHERE c.job_id = ?) OR "
                "EXISTS(SELECT 1 FROM prompt_ownership o "
                "JOIN job_children c ON c.id = o.child_id WHERE c.job_id = ?) "
                "AS present",
                (job_id, job_id, job_id),
            ).fetchone()
            if marker is not None and bool(marker["present"]):
                return True
        if cls._artifact_observation_schema_exists(db):
            marker = db.execute(
                "SELECT "
                "EXISTS(SELECT 1 FROM job_child_output_receipts r "
                "JOIN job_children c ON c.id = r.child_id WHERE c.job_id = ?) OR "
                "EXISTS(SELECT 1 FROM segment_take_observed_artifacts a "
                "JOIN segment_takes t ON t.id = a.take_id "
                "WHERE t.source_job_id = ?) AS present",
                (job_id, job_id),
            ).fetchone()
            if marker is not None and bool(marker["present"]):
                return True
        if cls._assembly_artifact_schema_exists(db):
            marker = db.execute(
                "SELECT 1 FROM job_observed_assembly_artifacts WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if marker is not None:
                return True
        return False

    def has_job_child_execution_marker(self, child_id: str) -> bool:
        """Return whether a child has any typed-execution persistence marker.

        This is intentionally an existence probe, not an evidence reader.  A
        partially deleted or corrupt typed chain must still be classified as
        typed so public projections cannot fall back to mutable legacy fields.
        """

        with self.connect() as db:
            child = db.execute(
                "SELECT job_id FROM job_children WHERE id = ?",
                (child_id,),
            ).fetchone()
            if child is not None and self._job_has_typed_execution_marker_in_connection(
                db,
                str(child["job_id"]),
            ):
                return True
            if self._artifact_observation_schema_exists(db):
                marker = db.execute(
                    "SELECT 1 FROM segment_take_observed_artifacts "
                    "WHERE source_child_id = ? LIMIT 1",
                    (child_id,),
                ).fetchone()
                if marker is not None:
                    return True
        return False

    def has_job_execution_marker(self, job_id: str) -> bool:
        """Return whether the parent is permanently classified as typed."""

        with self.connect() as db:
            return self._job_has_typed_execution_marker_in_connection(db, job_id)

    def get_prompt_ownership(self, child_id: str) -> PromptOwnership | None:
        """Load typed ownership, or ``None`` for legacy/unclaimed children."""

        with self.connect() as db:
            if not self._execution_evidence_schema_exists(db):
                return None
            row = db.execute(
                "SELECT * FROM prompt_ownership WHERE child_id = ?", (child_id,)
            ).fetchone()
        return self._prompt_ownership_row(row) if row is not None else None

    @staticmethod
    def _locked_unit_for_snapshot(
        locked_plan: LockedSubmissionPlan,
        exact_snapshot: ExactPromptSnapshot,
    ) -> LockedSegmentUnit | PreparedControlUnit:
        matches = tuple(
            unit for unit in locked_plan.units if unit.id == exact_snapshot.unit_id
        )
        if len(matches) != 1:
            raise ExecutionEvidenceConflict(
                "exact prompt unit is not unique in its locked submission plan"
            )
        unit = matches[0]
        unit_document = unit.model_dump(mode="json")
        prompt = (
            unit_document["exact_prompt"]
            if isinstance(unit, LockedSegmentUnit)
            else unit_document["prompt_base"]
        )
        expected = {
            "unit_id": unit.id,
            "unit_kind": unit.kind,
            "owner_segment_id": unit.owner_segment_id,
            "control_kind": (
                unit.control_kind
                if isinstance(unit, PreparedControlUnit)
                else None
            ),
            "family": unit.family,
            "backend": unit.backend,
            "template_id": unit.template_id,
            "template_revision": unit.template_revision,
            "endpoint_identity": locked_plan.endpoint_identity.model_dump(mode="json"),
            "exact_prompt": prompt,
            "graph_audit_spec": unit.graph_audit_spec.model_dump(mode="json"),
            "expected_output_spec": (
                unit.expected_output_spec.model_dump(mode="json")
                if unit.expected_output_spec is not None
                else None
            ),
            "progress_spec": (
                unit.progress_spec.model_dump(mode="json")
                if unit.progress_spec is not None
                else None
            ),
            "preview_spec": (
                unit.preview_spec.model_dump(mode="json")
                if unit.preview_spec is not None
                else None
            ),
            "effective_execution_digest": unit.effective_execution_digest.model_dump(
                mode="json"
            ),
        }
        actual = exact_snapshot.model_dump(
            mode="json", include=set(expected)
        )
        if not canonical_values_equal(expected, actual):
            raise ExecutionEvidenceConflict(
                "exact prompt snapshot does not match its locked unit"
            )
        return unit

    @classmethod
    def _execution_evidence_from_row(
        cls,
        row: Mapping[str, Any],
        *,
        child_id: str,
    ) -> tuple[
        LockedSubmissionPlan,
        ExactPromptSnapshot,
        LockedSegmentUnit | PreparedControlUnit,
    ]:
        """Authenticate a persisted locked plan and its exact child snapshot."""

        try:
            raw_digest = cls._canonical_json_payload_digest(
                row["locked_submission_plan"]
            )
        except TypeError as exc:
            raise ExecutionEvidenceConflict(
                f"locked submission evidence is invalid for child {child_id}"
            ) from exc
        if raw_digest != row["locked_submission_plan_digest"]:
            raise ExecutionEvidenceConflict(
                f"execution evidence digest/index mismatch for child {child_id}"
            )
        try:
            locked_plan = LockedSubmissionPlan.model_validate_json(
                row["locked_submission_plan"]
            )
        except (TypeError, ValueError) as exc:
            raise ExecutionEvidenceConflict(
                f"locked submission evidence is invalid for child {child_id}"
            ) from exc
        exact_snapshot = cls._exact_prompt_snapshot_from_evidence_row(
            row,
            child_id=child_id,
        )
        if (
            str(row["child_id"]) != child_id
            or locked_plan.endpoint_identity.endpoint_key != row["endpoint_key"]
            or locked_plan.endpoint_identity.runtime_instance_id
            != row["endpoint_runtime_instance_id"]
            or locked_plan.source_compiled_plan_digest.value
            != row["source_compiled_plan_digest"]
        ):
            raise ExecutionEvidenceConflict(
                f"execution evidence digest/index mismatch for child {child_id}"
            )
        unit = cls._locked_unit_for_snapshot(locked_plan, exact_snapshot)
        if unit.child_id != child_id:
            raise ExecutionEvidenceConflict(
                f"execution evidence unit owner mismatch for child {child_id}"
            )
        return locked_plan, exact_snapshot, unit

    def _assert_completed_control_continuation(
        self,
        db: sqlite3.Connection,
        *,
        job_id: str,
        locked_plan: LockedSubmissionPlan,
        segment: LockedSegmentUnit,
    ) -> None:
        """Bind a segment continuation to its same-job RayKill evidence."""

        dependency = locked_plan.control_dependency
        if dependency is None:
            return
        if locked_plan.units != (segment,):
            raise ExecutionEvidenceConflict(
                "control continuation must persist exactly its target segment"
            )
        control_row = db.execute(
            "SELECT * FROM job_children WHERE id = ? AND job_id = ? "
            "AND group_index = ?",
            (
                dependency.control_child_id,
                job_id,
                dependency.control_group_index,
            ),
        ).fetchone()
        if control_row is None:
            raise ExecutionEvidenceConflict(
                "segment continuation is missing its required control child"
            )
        control = self._job_child_row(control_row)
        evidence_row = db.execute(
            "SELECT * FROM job_child_execution_evidence "
            "WHERE child_id = ?",
            (control["id"],),
        ).fetchone()
        ownership = self._prompt_ownership_in_connection(db, control["id"])
        if evidence_row is None or ownership is None:
            raise ExecutionEvidenceConflict(
                "segment continuation has incomplete control evidence"
            )
        pair, control_snapshot, persisted_control = (
            self._execution_evidence_from_row(
                evidence_row,
                child_id=str(control["id"]),
            )
        )
        certificate = ownership.cleanup_certificate
        if (
            sha256_document_digest(pair.model_dump(mode="json"))
            != dependency.original_locked_plan_digest
            or sha256_document_digest(control_snapshot.model_dump(mode="json"))
            != dependency.control_exact_prompt_snapshot_digest
            or len(pair.units) != 2
            or not isinstance(pair.units[0], PreparedControlUnit)
            or persisted_control != pair.units[0]
            or persisted_control.child_id != dependency.control_child_id
            or persisted_control.id != dependency.control_unit_id
            or persisted_control.requested_prompt_id
            != dependency.control_requested_prompt_id
            or persisted_control.group_index != dependency.control_group_index
            or pair.units[1] != segment
            or control["segment_ids"] != []
            or control["prompt_id"] != ownership.effective_prompt_id
            or ownership.requested_prompt_id
            != dependency.control_requested_prompt_id
            or control["status"] != "succeeded"
            or ownership.state != "terminal_confirmed"
            or not isinstance(certificate, HistoryTerminalEvidence)
            or certificate.terminal_status != "succeeded"
        ):
            raise ExecutionEvidenceConflict(
                "segment continuation control is not terminal-confirmed"
            )

    @staticmethod
    def _raylight_runtime_state_in_connection(
        db: sqlite3.Connection,
    ) -> dict[str, Any] | None:
        row = db.execute(
            "SELECT descriptor FROM raylight_runtime_state WHERE singleton = 1"
        ).fetchone()
        return (
            Database._decode_raylight_runtime_state(row["descriptor"])
            if row is not None
            else None
        )

    @staticmethod
    def _normalized_raylight_runtime_state(
        state: Any,
    ) -> dict[str, Any]:
        return Database._decode_raylight_runtime_state(canonical_json(state))

    @classmethod
    def _persist_raylight_intent_in_connection(
        cls,
        db: sqlite3.Connection,
        *,
        before: Any,
        after: Any,
        updated_at: str,
    ) -> None:
        # Every POST owns one exact ledger frontier, including a Standard
        # submission that expects no Ray row.  Treating ``None -> None`` as a
        # no-op would let a concurrently-created resident pool bypass the
        # locked plan's compare-and-set authority.
        current = cls._raylight_runtime_state_in_connection(db)
        expected = (
            None
            if before is None
            else cls._normalized_raylight_runtime_state(before)
        )
        if current != expected:
            raise RayRuntimeIntentConflict(
                "RayLight runtime ledger changed before submission intent"
            )
        if after is None:
            db.execute("DELETE FROM raylight_runtime_state WHERE singleton = 1")
            return
        normalized_after = cls._normalized_raylight_runtime_state(after)
        db.execute(
            "INSERT INTO raylight_runtime_state"
            "(singleton, descriptor, updated_at) VALUES(1, ?, ?) "
            "ON CONFLICT(singleton) DO UPDATE SET "
            "descriptor = excluded.descriptor, updated_at = excluded.updated_at",
            (
                json.dumps(
                    normalized_after, ensure_ascii=False, sort_keys=True
                ),
                updated_at,
            ),
        )

    @classmethod
    def _rebind_raylight_tail_in_connection(
        cls,
        db: sqlite3.Connection,
        *,
        requested_prompt_id: str,
        actual_prompt_id: str,
        updated_at: str,
    ) -> bool:
        state = cls._raylight_runtime_state_in_connection(db)
        if state is None or state.get("tail_prompt_id") != requested_prompt_id:
            return False
        state["tail_prompt_id"] = actual_prompt_id
        certificate = state.get("tail_terminal_certificate")
        if isinstance(certificate, dict):
            certificate["prompt_id"] = actual_prompt_id
        db.execute(
            "UPDATE raylight_runtime_state SET descriptor = ?, updated_at = ? "
            "WHERE singleton = 1",
            (
                json.dumps(state, ensure_ascii=False, sort_keys=True),
                updated_at,
            ),
        )
        return True

    @staticmethod
    def _derived_child_identity(
        unit: LockedSegmentUnit | PreparedControlUnit,
    ) -> tuple[list[str], dict[str, str]]:
        if isinstance(unit, PreparedControlUnit):
            return [], {}
        return [unit.owner_segment_id], {
            unit.owner_segment_id: unit.expected_output_spec.node_id
        }

    @classmethod
    def _assert_available_prompt_identity(
        cls,
        db: sqlite3.Connection,
        *,
        child_id: str,
        prompt_id: str,
    ) -> None:
        collision = db.execute(
            "SELECT child_id FROM prompt_ownership "
            "WHERE child_id != ? AND "
            "(requested_prompt_id = ? OR actual_prompt_id = ?) LIMIT 1",
            (child_id, prompt_id, prompt_id),
        ).fetchone()
        if collision is not None:
            raise ExecutionEvidenceConflict(
                f"prompt id already belongs to child {collision['child_id']}"
            )

    @staticmethod
    def _assert_complete_late_binding_evidence(
        unit: LockedSegmentUnit,
    ) -> ContinuityLateBindingEvidence | None:
        """Re-prove typed late bindings after crossing the persistence boundary.

        ``model_copy(update=...)`` and equivalent deserialization shortcuts do
        not run Pydantic model validators.  The database therefore treats the
        locked unit as an untrusted transport object and independently requires
        exact coverage of every non-resource declaration before committing an
        externally visible submission intent.
        """

        declarations = {
            declaration.input_pointer: declaration
            for declaration in unit.graph_audit_spec.allowed_late_bound_inputs
            if declaration.source_kind != "resource"
        }
        if len(declarations) != sum(
            declaration.source_kind != "resource"
            for declaration in unit.graph_audit_spec.allowed_late_bound_inputs
        ):
            raise ExecutionEvidenceConflict(
                "non-resource late-binding declarations must be unique"
            )
        evidence = tuple(unit.late_binding_evidence)
        if any(
            not isinstance(
                item,
                (
                    ContinuityLateBindingEvidence,
                    RuntimeEpochLateBindingEvidence,
                ),
            )
            for item in evidence
        ):
            raise ExecutionEvidenceConflict(
                "late-binding evidence must use registered typed contracts"
            )
        evidence_by_pointer = {item.input_pointer: item for item in evidence}
        if len(evidence_by_pointer) != len(evidence):
            raise ExecutionEvidenceConflict(
                "late-binding evidence pointers must be unique"
            )
        if set(evidence_by_pointer) != set(declarations):
            raise ExecutionEvidenceConflict(
                "late-binding evidence must exactly cover non-resource pointers"
            )

        expected_values: dict[str, Any] = {}
        continuity: ContinuityLateBindingEvidence | None = None
        for pointer, declaration in declarations.items():
            item = evidence_by_pointer[pointer]
            if item.source_kind != declaration.source_kind:
                raise ExecutionEvidenceConflict(
                    f"late-binding evidence source differs at {pointer!r}"
                )
            if declaration.value_kind != "string":
                raise ExecutionEvidenceConflict(
                    "continuity and runtime late bindings must target strings"
                )
            if isinstance(item, ContinuityLateBindingEvidence):
                if continuity is not None:
                    raise ExecutionEvidenceConflict(
                        "one segment cannot consume multiple continuity bindings"
                    )
                continuity = item
                expected_values[pointer] = item.bound_value
                continue
            compatibility_key = (
                unit.runtime_requirements.ray_compatibility_key
            )
            if unit.backend != "raylight" or compatibility_key is None:
                raise ExecutionEvidenceConflict(
                    "runtime epoch evidence requires a RayLight unit"
                )
            expected_values[pointer] = f"{compatibility_key}-e{item.epoch}"

        dependency = unit.continuity_dependency
        if dependency is None:
            if continuity is not None:
                raise ExecutionEvidenceConflict(
                    "continuity evidence has no compiled dependency"
                )
        else:
            if not isinstance(dependency, Mapping) or continuity is None:
                raise ExecutionEvidenceConflict(
                    "compiled continuity dependency has no typed evidence"
                )
            source = continuity.dependency_source
            expected_resolved = source == "historical_take"
            expected_bound = (
                continuity.bound_value if expected_resolved else None
            )
            if (
                dependency.get("input_pointer") != continuity.input_pointer
                or dependency.get("predecessor_segment_id")
                != continuity.predecessor_segment_id
                or dependency.get("source") != source
                or dependency.get("historical_take_id")
                != continuity.historical_take_id
                or dependency.get("resolved") is not expected_resolved
                or dependency.get("bound_file") != expected_bound
            ):
                raise ExecutionEvidenceConflict(
                    "continuity evidence differs from the compiled dependency"
                )
        if canonical_json(expected_values) != canonical_json(
            unit.late_bound_values
        ):
            raise ExecutionEvidenceConflict(
                "late-bound values are not derived from typed evidence"
            )
        return continuity

    @staticmethod
    def _timeline_continuity_context(
        parent: sqlite3.Row,
        unit: LockedSegmentUnit,
        evidence: ContinuityLateBindingEvidence,
    ) -> tuple[
        UnifiedTimelineDraft,
        UnifiedTimelineSegment,
        UnifiedTimelineSegment,
    ]:
        try:
            config_snapshot = json.loads(parent["config_snapshot"])
            timeline_document = (
                config_snapshot.get("timeline")
                if isinstance(config_snapshot, dict)
                else None
            )
            if not isinstance(timeline_document, dict):
                raise ValueError("timeline snapshot is missing")
            draft = validate_timeline_snapshot(timeline_document)
            target_matches = [
                segment
                for segment in draft.segments
                if segment.id == unit.owner_segment_id
            ]
            predecessor_matches = [
                segment
                for segment in draft.segments
                if segment.id == evidence.predecessor_segment_id
            ]
            if len(target_matches) != 1 or len(predecessor_matches) != 1:
                raise ValueError("continuity segment identity is ambiguous")
            target = target_matches[0]
            predecessor = predecessor_matches[0]
            authored = unified_continuity_predecessors(draft).get(target.id)
            if authored is None or authored.id != predecessor.id:
                raise ValueError("continuity predecessor differs from timeline")
        except (
            KeyError,
            TypeError,
            ValueError,
            ValidationError,
            json.JSONDecodeError,
        ) as exc:
            raise ExecutionEvidenceConflict(
                "continuity evidence differs from the captured timeline"
            ) from exc
        return draft, target, predecessor

    @staticmethod
    def _observed_artifact_geometry_fingerprint(
        artifact: ObservedArtifactSpec,
    ) -> str:
        return segment_take_geometry_fingerprint(
            width=artifact.width,
            height=artifact.height,
            fps=artifact.fps,
            visible_frame_count=artifact.frame_count,
        )

    def _assert_same_run_continuity_authority(
        self,
        db: sqlite3.Connection,
        *,
        job_id: str,
        unit: LockedSegmentUnit,
        evidence: ContinuityLateBindingEvidence,
        require_audio: bool,
    ) -> None:
        rows = db.execute(
            "SELECT * FROM job_children WHERE job_id = ? ORDER BY group_index",
            (job_id,),
        ).fetchall()
        children = [self._job_child_row(row) for row in rows]
        predecessors = [
            child
            for child in children
            if child["segment_ids"] == [evidence.predecessor_segment_id]
        ]
        if len(predecessors) != 1:
            raise ExecutionEvidenceConflict(
                "same-run continuity requires one unique predecessor child"
            )
        predecessor = predecessors[0]
        ownership = self._prompt_ownership_in_connection(
            db, str(predecessor["id"])
        )
        certificate = (
            ownership.cleanup_certificate if ownership is not None else None
        )
        if (
            predecessor["status"] != "succeeded"
            or predecessor["group_index"] >= unit.group_index
            or ownership is None
            or ownership.state != "terminal_confirmed"
            or predecessor.get("prompt_id") != ownership.effective_prompt_id
            or not isinstance(certificate, HistoryTerminalEvidence)
            or certificate.terminal_status != "succeeded"
            or certificate.prompt_id != ownership.effective_prompt_id
        ):
            raise ExecutionEvidenceConflict(
                "same-run predecessor is not terminal-confirmed succeeded"
            )

        exact_row = db.execute(
            "SELECT exact_prompt_snapshot, exact_prompt_snapshot_digest "
            "FROM job_child_execution_evidence WHERE child_id = ?",
            (predecessor["id"],),
        ).fetchone()
        if exact_row is None:
            raise ExecutionEvidenceConflict(
                "same-run predecessor has no exact execution evidence"
            )
        try:
            exact = ExactPromptSnapshot.model_validate_json(
                exact_row["exact_prompt_snapshot"]
            )
        except (TypeError, ValueError) as exc:
            raise ExecutionEvidenceConflict(
                "same-run predecessor exact execution evidence is invalid"
            ) from exc
        if (
            self._execution_document_digest(exact)
            != exact_row["exact_prompt_snapshot_digest"]
            or exact.unit_kind != "segment"
            or exact.owner_segment_id != evidence.predecessor_segment_id
            or exact.expected_output_spec is None
            or exact.expected_output_spec.segment_id
            != evidence.predecessor_segment_id
        ):
            raise ExecutionEvidenceConflict(
                "same-run predecessor exact output identity is invalid"
            )
        if not self._artifact_observation_schema_exists(db):
            raise ExecutionEvidenceConflict(
                "same-run predecessor has no observed artifact schema"
            )
        observed_take = self._observed_segment_take_in_connection(
            db,
            source_child_id=str(predecessor["id"]),
        )
        if observed_take is None:
            raise ExecutionEvidenceConflict(
                "same-run predecessor has no observed artifact"
            )
        _take, artifact = observed_take
        expected_geometry = exact.expected_output_spec
        expected_geometry_fingerprint = segment_take_geometry_fingerprint(
            width=expected_geometry.width,
            height=expected_geometry.height,
            fps=expected_geometry.fps,
            visible_frame_count=expected_geometry.visible_frame_count,
        )
        if (
            artifact.segment_id != evidence.predecessor_segment_id
            or artifact.child_id != predecessor["id"]
            or artifact.output_descriptor != evidence.output
            or self._observed_artifact_geometry_fingerprint(artifact)
            != expected_geometry_fingerprint
            or require_audio
            and not artifact.has_audio
        ):
            raise ExecutionEvidenceConflict(
                "same-run continuity output differs from terminal evidence"
            )

    def _assert_historical_continuity_authority(
        self,
        db: sqlite3.Connection,
        *,
        parent: sqlite3.Row,
        evidence: ContinuityLateBindingEvidence,
        draft: UnifiedTimelineDraft,
        target: UnifiedTimelineSegment,
        predecessor: UnifiedTimelineSegment,
    ) -> None:
        if not self._artifact_observation_schema_exists(db):
            raise ExecutionEvidenceConflict(
                "historical continuity requires observed artifact evidence"
            )
        try:
            observed_take = self._observed_segment_take_in_connection(
                db,
                take_id=evidence.historical_take_id,
            )
        except ExecutionEvidenceConflict as exc:
            raise ExecutionEvidenceConflict(
                "historical continuity take evidence is invalid"
            ) from exc
        if observed_take is None:
            raise ExecutionEvidenceConflict(
                "historical continuity take id is not present"
            )
        take, artifact = observed_take
        try:
            fingerprint = timeline_segment_take_fingerprint(
                draft, predecessor
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            NativeTemplateError,
            json.JSONDecodeError,
        ) as exc:
            raise ExecutionEvidenceConflict(
                "historical continuity take evidence is invalid"
            ) from exc
        if (
            take["segment_id"] != evidence.predecessor_segment_id
            or take["project_id"] != parent["project_id"]
            or take["content_fingerprint"] != fingerprint
            or self._observed_artifact_geometry_fingerprint(artifact) != fingerprint
            or artifact.segment_id != evidence.predecessor_segment_id
            or artifact.child_id != take["source_child_id"]
            or (
                target.audio_mode == "generate"
                and not artifact.has_audio
            )
            or artifact.output_descriptor != evidence.output
        ):
            raise ExecutionEvidenceConflict(
                "historical continuity take does not match exact captured authority"
            )

    def _assert_continuity_submission_authority(
        self,
        db: sqlite3.Connection,
        *,
        job_id: str,
        parent: sqlite3.Row,
        unit: LockedSegmentUnit,
        evidence: ContinuityLateBindingEvidence | None,
    ) -> None:
        if evidence is None:
            return
        draft, target, predecessor = self._timeline_continuity_context(
            parent, unit, evidence
        )
        if evidence.dependency_source == "same_run":
            self._assert_same_run_continuity_authority(
                db,
                job_id=job_id,
                unit=unit,
                evidence=evidence,
                require_audio=target.audio_mode == "generate",
            )
            return
        self._assert_historical_continuity_authority(
            db,
            parent=parent,
            evidence=evidence,
            draft=draft,
            target=target,
            predecessor=predecessor,
        )

    def persist_job_child_submission_intent(
        self,
        job_id: str,
        *,
        locked_plan: LockedSubmissionPlan,
        exact_snapshot: ExactPromptSnapshot,
    ) -> tuple[dict[str, Any], PromptOwnership]:
        """Atomically establish all durable authority before one ``/prompt``.

        A missing child is derived from the locked unit and inserted in this
        same transaction, which is the normal path for a dynamically planned
        RayKill control.  Existing prepared segment children are accepted only
        when their stable identity still matches exactly.
        """

        unit = self._locked_unit_for_snapshot(locked_plan, exact_snapshot)
        try:
            validate_locked_submission_transition(locked_plan, unit)
        except SubmissionPlanningError as exc:
            raise ExecutionEvidenceConflict(
                f"locked submission Ray transition is invalid: {exc}"
            ) from exc
        child_id = unit.child_id
        requested_prompt_id = unit.requested_prompt_id
        now = utc_now()
        ownership = PromptOwnership(
            requested_prompt_id=requested_prompt_id,
            state="submitting",
            ownership_revision=0,
            updated_at=datetime.fromisoformat(now).astimezone(timezone.utc),
        )
        expected_segment_ids, expected_output_nodes = self._derived_child_identity(
            unit
        )
        locked_json = self._contract_json(locked_plan)
        exact_json = self._contract_json(exact_snapshot)

        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._ensure_execution_evidence_schema(db)
            parent = db.execute(
                "SELECT status, cancel_requested, project_id, config_snapshot "
                "FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if parent is None:
                raise KeyError(job_id)
            if parent["status"] != "preparing" or bool(parent["cancel_requested"]):
                raise ExecutionEvidenceConflict(
                    "job is no longer eligible for submission"
                )

            plan_row = db.execute(
                "SELECT schema_version, compiled_plan, compiled_plan_digest, "
                "unit_index_version "
                "FROM job_execution_plans WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if plan_row is None:
                raise ExecutionEvidenceConflict(
                    "submission intent has no persisted compiled plan"
                )
            try:
                persisted_digest = (
                    compiled_execution_plan_digest_from_canonical_json(
                        plan_row["compiled_plan"]
                    )
                )
                if (
                    persisted_digest.value != plan_row["compiled_plan_digest"]
                    or persisted_digest
                    != locked_plan.source_compiled_plan_digest
                ):
                    raise ValueError(
                        "locked submission source digest does not match"
                    )
                persisted_plan_document = json.loads(plan_row["compiled_plan"])
                if not isinstance(persisted_plan_document, dict):
                    raise ValueError("persisted compiled plan is not an object")
                template_bundle_version = persisted_plan_document.get(
                    "template_bundle_version"
                )
                if template_bundle_version == V4_TEMPLATE_BUNDLE.version:
                    node_contract_registry = V4_NODE_CONTRACT_REGISTRY
                elif template_bundle_version == CURRENT_TEMPLATE_BUNDLE.version:
                    node_contract_registry = CURRENT_NODE_CONTRACT_REGISTRY
                else:
                    raise ValueError(
                        "persisted compiled plan uses an unsupported template bundle"
                    )
                prepared = self._prepared_unit_from_compiled_plan_index(
                    db,
                    job_id=job_id,
                    plan_row=plan_row,
                    schema_version=plan_row["schema_version"],
                    source_unit_ordinal=locked_plan.source_unit_ordinal,
                )
                locked_plan.validate_source_prepared_unit(
                    prepared,
                    verified_source_compiled_plan_digest=persisted_digest,
                )
            except (
                AssertionError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                UnicodeEncodeError,
                ValidationError,
                ValueError,
            ) as exc:
                raise ExecutionEvidenceConflict(
                    "locked submission does not match the persisted plan"
                ) from exc
            if isinstance(unit, LockedSegmentUnit):
                continuity_evidence = (
                    self._assert_complete_late_binding_evidence(unit)
                )
                try:
                    unit.validate_materialized_prompt(
                        node_contract_registry=node_contract_registry
                    )
                except (
                    AssertionError,
                    GraphAuditError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as exc:
                    raise ExecutionEvidenceConflict(
                        "exact prompt failed materialized graph validation"
                    ) from exc
                self._assert_continuity_submission_authority(
                    db,
                    job_id=job_id,
                    parent=parent,
                    unit=unit,
                    evidence=continuity_evidence,
                )
                self._assert_completed_control_continuation(
                    db,
                    job_id=job_id,
                    locked_plan=locked_plan,
                    segment=unit,
                )
            if (
                exact_snapshot.effective_execution_digest
                != unit.effective_execution_digest
            ):
                raise ExecutionEvidenceConflict(
                    "exact prompt execution digest does not match its locked unit"
                )

            child_row = db.execute(
                "SELECT * FROM job_children WHERE id = ?", (child_id,)
            ).fetchone()
            if child_row is None:
                db.execute(
                    "INSERT INTO job_children("
                    "id, job_id, group_index, family, backend, segment_ids, "
                    "output_nodes, status, progress, stage, prompt_id, outputs, "
                    "error, prompt_snapshot, created_at, updated_at, started_at, "
                    "completed_at) VALUES(?, ?, ?, ?, ?, ?, ?, 'preparing', "
                    "0.0, 'preflight', NULL, '[]', NULL, ?, ?, ?, NULL, NULL)",
                    (
                        child_id,
                        job_id,
                        unit.group_index,
                        unit.family,
                        unit.backend,
                        json.dumps(expected_segment_ids, ensure_ascii=False),
                        json.dumps(expected_output_nodes, ensure_ascii=False),
                        exact_json,
                        now,
                        now,
                    ),
                )
                child_row = db.execute(
                    "SELECT * FROM job_children WHERE id = ?", (child_id,)
                ).fetchone()
            if child_row is None:
                raise RuntimeError("submission child disappeared during intent")
            child = self._job_child_row(child_row)
            if (
                child["job_id"] != job_id
                or child["group_index"] != unit.group_index
                or child["family"] != unit.family
                or child["backend"] != unit.backend
                or child["segment_ids"] != expected_segment_ids
                or child["output_nodes"] != expected_output_nodes
                or child["status"] != "preparing"
                or child.get("prompt_id") is not None
            ):
                raise ExecutionEvidenceConflict(
                    "submission child no longer matches its locked unit"
                )

            self._assert_available_prompt_identity(
                db,
                child_id=child_id,
                prompt_id=requested_prompt_id,
            )
            db.execute(
                "INSERT INTO job_child_execution_evidence("
                "child_id, schema_version, unit_id, unit_kind, endpoint_key, "
                "endpoint_runtime_instance_id, source_compiled_plan_digest, "
                "locked_submission_plan, locked_submission_plan_digest, "
                "exact_prompt_snapshot, exact_prompt_snapshot_digest, created_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    child_id,
                    exact_snapshot.schema_version,
                    exact_snapshot.unit_id,
                    exact_snapshot.unit_kind,
                    locked_plan.endpoint_identity.endpoint_key,
                    locked_plan.endpoint_identity.runtime_instance_id,
                    locked_plan.source_compiled_plan_digest.value,
                    locked_json,
                    self._canonical_json_payload_digest(locked_json),
                    exact_json,
                    self._canonical_json_payload_digest(exact_json),
                    now,
                ),
            )
            db.execute(
                "INSERT INTO prompt_ownership("
                "child_id, requested_prompt_id, actual_prompt_id, state, "
                "ownership_revision, cleanup_certificate, updated_at) "
                "VALUES(?, ?, NULL, 'submitting', 0, NULL, ?)",
                (child_id, requested_prompt_id, now),
            )
            claimed = db.execute(
                "UPDATE job_children SET stage = 'submitting', prompt_id = ?, "
                "prompt_snapshot = ?, updated_at = ? "
                "WHERE id = ? AND job_id = ? AND status = 'preparing' "
                "AND prompt_id IS NULL AND EXISTS ("
                "SELECT 1 FROM jobs WHERE id = ? AND status = 'preparing' "
                "AND cancel_requested = 0)",
                (
                    requested_prompt_id,
                    canonical_json(exact_snapshot.exact_prompt),
                    now,
                    child_id,
                    job_id,
                    job_id,
                ),
            )
            if claimed.rowcount != 1:
                raise ExecutionEvidenceConflict(
                    "submission child changed while persisting intent"
                )
            self._persist_raylight_intent_in_connection(
                db,
                before=locked_plan.ray_ledger_before,
                after=locked_plan.ray_ledger_after_intent,
                updated_at=now,
            )
            committed_row = db.execute(
                "SELECT * FROM job_children WHERE id = ?", (child_id,)
            ).fetchone()
        if committed_row is None:
            raise RuntimeError("committed submission child disappeared")
        return self._job_child_row(committed_row), ownership

    def _prompt_ownership_in_connection(
        self,
        db: sqlite3.Connection,
        child_id: str,
    ) -> PromptOwnership | None:
        row = db.execute(
            "SELECT * FROM prompt_ownership WHERE child_id = ?", (child_id,)
        ).fetchone()
        return self._prompt_ownership_row(row) if row is not None else None

    def _transition_prompt_ownership_in_connection(
        self,
        db: sqlite3.Connection,
        child_id: str,
        *,
        expected_revision: int,
        state: PromptOwnershipState,
        updated_at: datetime,
        actual_prompt_id: str | None | object = _DATABASE_UNSET,
        cleanup_certificate: (
            PromptReleaseEvidence | dict[str, Any] | None | object
        ) = _DATABASE_UNSET,
    ) -> PromptOwnership | None:
        current = self._prompt_ownership_in_connection(db, child_id)
        if current is None or current.ownership_revision != expected_revision:
            return None
        exact_row = db.execute(
            "SELECT * FROM job_child_execution_evidence WHERE child_id = ?",
            (child_id,),
        ).fetchone()
        if exact_row is None:
            raise ExecutionEvidenceConflict(
                "prompt ownership has no immutable exact execution evidence"
            )
        _locked_plan, _exact_snapshot, locked_unit = (
            self._execution_evidence_from_row(
                exact_row,
                child_id=child_id,
            )
        )
        if locked_unit.requested_prompt_id != current.requested_prompt_id:
            raise ExecutionEvidenceConflict(
                "prompt ownership requested id differs from exact execution evidence"
            )
        transition_arguments: dict[str, Any] = {
            "expected_revision": expected_revision,
            "state": state,
            "updated_at": updated_at,
        }
        if actual_prompt_id is not _DATABASE_UNSET:
            transition_arguments["actual_prompt_id"] = actual_prompt_id
        if cleanup_certificate is not _DATABASE_UNSET:
            transition_arguments["cleanup_certificate"] = cleanup_certificate
        next_ownership = transition_prompt_ownership(
            current, **transition_arguments
        )
        payload = next_ownership.model_dump(mode="json")
        certificate = payload["cleanup_certificate"]
        cursor = db.execute(
            "UPDATE prompt_ownership SET actual_prompt_id = ?, state = ?, "
            "ownership_revision = ?, cleanup_certificate = ?, updated_at = ? "
            "WHERE child_id = ? AND ownership_revision = ?",
            (
                payload["actual_prompt_id"],
                payload["state"],
                payload["ownership_revision"],
                canonical_json(certificate) if certificate is not None else None,
                payload["updated_at"],
                child_id,
                expected_revision,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("prompt ownership CAS changed under write lock")
        return next_ownership

    def compare_and_set_prompt_ownership(
        self,
        child_id: str,
        *,
        expected_revision: int,
        state: PromptOwnershipState,
        updated_at: datetime,
        cleanup_certificate: (
            PromptReleaseEvidence | dict[str, Any] | None | object
        ) = _DATABASE_UNSET,
    ) -> PromptOwnership | None:
        """Apply one typed monotonic ownership transition by revision CAS."""

        if state in {"cleanup_confirmed", "terminal_confirmed"}:
            raise ValueError(
                "confirmed ownership must use the atomic prompt release methods"
            )
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not self._execution_evidence_schema_exists(db):
                return None
            return self._transition_prompt_ownership_in_connection(
                db,
                child_id,
                expected_revision=expected_revision,
                state=state,
                updated_at=updated_at,
                cleanup_certificate=cleanup_certificate,
            )

    def record_prompt_submission_receipt(
        self,
        child_id: str,
        *,
        expected_revision: int,
        actual_prompt_id: str,
        state: PromptOwnershipState,
        updated_at: datetime,
        cleanup_certificate: (
            PromptReleaseEvidence | dict[str, Any] | None | object
        ) = _DATABASE_UNSET,
    ) -> PromptOwnership | None:
        """Atomically bind ComfyUI's receipt and migrate a matching Ray tail."""

        if state not in {
            "owned_requested_id",
            "owned_actual_id",
            "unconfirmed",
        }:
            raise ValueError("submission receipt has an invalid ownership state")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not self._execution_evidence_schema_exists(db):
                return None
            current = self._prompt_ownership_in_connection(db, child_id)
            if current is None or current.ownership_revision != expected_revision:
                return None
            normalized_actual = (
                None
                if actual_prompt_id == current.requested_prompt_id
                else actual_prompt_id
            )
            effective_actual = normalized_actual or current.requested_prompt_id
            self._assert_available_prompt_identity(
                db, child_id=child_id, prompt_id=effective_actual
            )
            child = db.execute(
                "SELECT prompt_id FROM job_children WHERE id = ?", (child_id,)
            ).fetchone()
            if child is None:
                raise KeyError(child_id)
            if child["prompt_id"] != current.effective_prompt_id:
                return None
            next_ownership = self._transition_prompt_ownership_in_connection(
                db,
                child_id,
                expected_revision=expected_revision,
                state=state,
                updated_at=updated_at,
                actual_prompt_id=normalized_actual,
                cleanup_certificate=cleanup_certificate,
            )
            if next_ownership is None:
                return None
            cursor = db.execute(
                "UPDATE job_children SET prompt_id = ?, updated_at = ? "
                "WHERE id = ? AND prompt_id = ?",
                (
                    next_ownership.effective_prompt_id,
                    next_ownership.updated_at.isoformat(),
                    child_id,
                    current.effective_prompt_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("submission receipt child CAS changed under write lock")
            if next_ownership.effective_prompt_id != current.requested_prompt_id:
                self._rebind_raylight_tail_in_connection(
                    db,
                    requested_prompt_id=current.requested_prompt_id,
                    actual_prompt_id=next_ownership.effective_prompt_id,
                    updated_at=next_ownership.updated_at.isoformat(),
                )
            return next_ownership

    @classmethod
    def _output_observation_receipt_row(
        cls, row: Mapping[str, Any]
    ) -> OutputObservationReceipt:
        try:
            receipt = OutputObservationReceipt.model_validate_json(row["receipt"])
        except (TypeError, ValueError) as exc:
            raise ExecutionEvidenceConflict(
                "persisted output observation receipt is invalid"
            ) from exc
        if cls._execution_document_digest(receipt) != row["receipt_digest"]:
            raise ExecutionEvidenceConflict(
                "persisted output observation receipt digest is invalid"
            )
        if (
            int(row["schema_version"]) != 1
            or receipt.child_id != str(row["child_id"])
        ):
            raise ExecutionEvidenceConflict(
                "persisted output observation receipt index is invalid"
            )
        return receipt

    @classmethod
    def _observed_artifact_row(
        cls, row: Mapping[str, Any]
    ) -> ObservedArtifactSpec:
        try:
            artifact = ObservedArtifactSpec.model_validate_json(
                row["observed_artifact"]
            )
        except (TypeError, ValueError) as exc:
            raise ExecutionEvidenceConflict(
                "persisted observed artifact is invalid"
            ) from exc
        if (
            cls._execution_document_digest(artifact)
            != row["observed_artifact_digest"]
        ):
            raise ExecutionEvidenceConflict(
                "persisted observed artifact digest is invalid"
            )
        return artifact

    @classmethod
    def _validate_output_receipt_against_exact_row(
        cls,
        *,
        child_id: str,
        receipt: OutputObservationReceipt,
        exact_row: Mapping[str, Any],
    ) -> ExactPromptSnapshot:
        exact_snapshot = cls._exact_prompt_snapshot_from_evidence_row(
            exact_row,
            child_id=child_id,
        )
        expected = exact_snapshot.expected_output_spec
        if (
            receipt.child_id != child_id
            or receipt.exact_prompt_snapshot_digest.value
            != exact_row["exact_prompt_snapshot_digest"]
            or exact_snapshot.unit_kind != "segment"
            or exact_snapshot.owner_segment_id != receipt.segment_id
            or expected is None
            or receipt.segment_id != expected.segment_id
            or receipt.node_id != expected.node_id
            or receipt.expected_output_spec_digest.value
            != cls._execution_document_digest(expected)
        ):
            raise ExecutionEvidenceConflict(
                "output receipt does not match the exact expected output"
            )
        return exact_snapshot

    def _validate_output_receipt_exact_in_connection(
        self,
        db: sqlite3.Connection,
        *,
        child_id: str,
        receipt: OutputObservationReceipt,
    ) -> ExactPromptSnapshot:
        """Authenticate a receipt against exact evidence under the caller's lock."""

        if not self._execution_evidence_schema_exists(db):
            raise ExecutionEvidenceConflict(
                "output receipt has no exact prompt evidence schema"
            )
        exact_row = db.execute(
            "SELECT * FROM job_child_execution_evidence WHERE child_id = ?",
            (child_id,),
        ).fetchone()
        if exact_row is None:
            raise ExecutionEvidenceConflict(
                "output receipt has no exact prompt snapshot"
            )
        return self._validate_output_receipt_against_exact_row(
            child_id=child_id,
            receipt=receipt,
            exact_row=exact_row,
        )

    @classmethod
    def _observed_segment_take_row(
        cls,
        row: sqlite3.Row,
        *,
        expected_child_id: str | None = None,
        expected_take_id: str | None = None,
    ) -> tuple[dict[str, Any], ObservedArtifactSpec]:
        """Validate one typed take as a single immutable evidence chain.

        ``segment_takes`` intentionally survives deletion of its source job.
        While that child still exists, its immutable output receipt must also
        exist and agree with the observation.  Once job deletion has cascaded
        the child and receipt away, the immutable artifact plus take row remain
        sufficient historical continuity authority.
        """

        try:
            artifact = cls._observed_artifact_row(row)
            take = cls._segment_take_row(
                {
                    key: row[key]
                    for key in (
                        "id",
                        "segment_id",
                        "content_fingerprint",
                        "project_id",
                        "output_descriptor",
                        "has_audio",
                        "source_job_id",
                        "source_child_id",
                        "completed_at",
                        "created_at",
                    )
                }
            )
        except ExecutionEvidenceConflict:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ExecutionEvidenceConflict(
                "persisted observed segment take is invalid"
            ) from exc

        take_id = str(take["id"])
        source_child_id = str(take["source_child_id"])
        observation_take_id = str(row["observation_take_id"])
        observation_source_child_id = str(
            row["observation_source_child_id"]
        )
        if (
            int(row["observation_schema_version"]) != 1
            or observation_take_id != take_id
            or observation_source_child_id != source_child_id
            or artifact.child_id != source_child_id
            or artifact.segment_id != str(take["segment_id"])
            or artifact.output_descriptor.model_dump(mode="json")
            != take["output"]
            or take["content_fingerprint"]
            != cls._observed_artifact_geometry_fingerprint(artifact)
            or artifact.has_audio is not bool(take["has_audio"])
            or (
                expected_child_id is not None
                and source_child_id != expected_child_id
            )
            or expected_take_id is not None
            and take_id != expected_take_id
        ):
            raise ExecutionEvidenceConflict(
                "persisted observed artifact differs from its segment take"
            )

        live_child_id = row["live_child_id"]
        live_source_job_id = row["live_source_job_id"]
        receipt_child_id = row["receipt_child_id"]
        exact_child_id = row["exact_child_id"]
        ownership_requested_id = row["observed_ownership_requested_id"]
        if live_child_id is not None:
            if (
                str(live_child_id) != source_child_id
                or live_source_job_id is None
                or str(row["live_child_job_id"]) != str(take["source_job_id"])
                or str(row["live_child_status"]) != "succeeded"
                or receipt_child_id is None
                or exact_child_id is None
                or ownership_requested_id is None
            ):
                raise ExecutionEvidenceConflict(
                    "live observed take is missing its immutable receipt chain"
                )
        elif (
            live_source_job_id is not None
            or receipt_child_id is not None
            or exact_child_id is not None
            or ownership_requested_id is not None
        ):
            raise ExecutionEvidenceConflict(
                "partial source deletion conflicts with historical take authority"
            )

        if receipt_child_id is not None:
            try:
                receipt = cls._output_observation_receipt_row(
                    {
                        "child_id": receipt_child_id,
                        "schema_version": row["receipt_schema_version"],
                        "receipt": row["receipt"],
                        "receipt_digest": row["durable_receipt_digest"],
                    }
                )
            except ExecutionEvidenceConflict:
                raise
            if (
                str(receipt_child_id) != source_child_id
                or str(row["observation_receipt_digest"])
                != str(row["durable_receipt_digest"])
                or receipt.child_id != artifact.child_id
                or receipt.segment_id != artifact.segment_id
                or receipt.output_descriptor != artifact.output_descriptor
            ):
                raise ExecutionEvidenceConflict(
                    "observed artifact differs from its immutable output receipt"
                )
            cls._validate_output_receipt_against_exact_row(
                child_id=source_child_id,
                receipt=receipt,
                exact_row={
                    "schema_version": row["exact_schema_version"],
                    "unit_id": row["exact_unit_id"],
                    "unit_kind": row["exact_unit_kind"],
                    "endpoint_key": row["exact_endpoint_key"],
                    "endpoint_runtime_instance_id": row[
                        "exact_runtime_instance_id"
                    ],
                    "exact_prompt_snapshot": row["exact_prompt_snapshot"],
                    "exact_prompt_snapshot_digest": row[
                        "exact_prompt_snapshot_digest"
                    ],
                },
            )
            try:
                ownership = cls._prompt_ownership_row(
                    {
                        "requested_prompt_id": ownership_requested_id,
                        "actual_prompt_id": row[
                            "observed_ownership_actual_id"
                        ],
                        "state": row["observed_ownership_state"],
                        "ownership_revision": row[
                            "observed_ownership_revision"
                        ],
                        "cleanup_certificate": row[
                            "observed_ownership_certificate"
                        ],
                        "updated_at": row[
                            "observed_ownership_updated_at"
                        ],
                    }
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ExecutionEvidenceConflict(
                    "persisted observed ownership is invalid"
                ) from exc
            if (
                ownership.state != "terminal_confirmed"
                or ownership.effective_prompt_id
                != receipt.history_evidence.prompt_id
                or ownership.cleanup_certificate != receipt.history_evidence
            ):
                raise ExecutionEvidenceConflict(
                    "observed artifact lacks matching terminal ownership"
                )

        return take, artifact

    def _observed_segment_take_in_connection(
        self,
        db: sqlite3.Connection,
        *,
        source_child_id: str | None = None,
        take_id: str | None = None,
    ) -> tuple[dict[str, Any], ObservedArtifactSpec] | None:
        if (source_child_id is None) == (take_id is None):
            raise ValueError("exactly one observed take identity is required")
        if not self._execution_evidence_schema_exists(db):
            raise ExecutionEvidenceConflict(
                "observed take has no execution evidence schema"
            )
        column = "a.source_child_id" if source_child_id is not None else "a.take_id"
        identity = source_child_id if source_child_id is not None else take_id
        row = db.execute(
            self._OBSERVED_SEGMENT_TAKE_SELECT
            + " FROM segment_take_observed_artifacts a "
            "LEFT JOIN segment_takes t ON t.id = a.take_id "
            "LEFT JOIN job_children c ON c.id = t.source_child_id "
            "LEFT JOIN jobs j ON j.id = t.source_job_id "
            "LEFT JOIN job_child_output_receipts r "
            "ON r.child_id = a.source_child_id "
            "LEFT JOIN job_child_execution_evidence e "
            "ON e.child_id = a.source_child_id "
            "LEFT JOIN prompt_ownership o "
            "ON o.child_id = a.source_child_id "
            f"WHERE {column} = ?",
            (identity,),
        ).fetchone()
        if row is None:
            return None
        return self._observed_segment_take_row(
            row,
            expected_child_id=source_child_id,
            expected_take_id=take_id,
        )

    @classmethod
    def _observed_assembly_artifact_row(
        cls, row: Mapping[str, Any]
    ) -> ObservedAssemblyArtifactSpec:
        try:
            artifact = ObservedAssemblyArtifactSpec.model_validate_json(
                row["observed_assembly_artifact"]
            )
        except (TypeError, ValueError) as exc:
            raise ExecutionEvidenceConflict(
                "persisted observed assembly artifact is invalid"
            ) from exc
        if (
            cls._execution_document_digest(artifact)
            != row["observed_assembly_artifact_digest"]
            or int(row["schema_version"]) != artifact.schema_version
            or str(row["job_id"]) != artifact.job_id
            or str(row["source_compiled_plan_digest"])
            != artifact.source_compiled_plan_digest.value
        ):
            raise ExecutionEvidenceConflict(
                "persisted observed assembly artifact digest/index is invalid"
            )
        return artifact

    def _validate_assembly_sources_in_connection(
        self,
        db: sqlite3.Connection,
        *,
        job_id: str,
        artifact: ObservedAssemblyArtifactSpec,
    ) -> CompiledExecutionPlan:
        """Authenticate every assembly input against its immutable source chain."""

        if artifact.job_id != job_id:
            raise ExecutionEvidenceConflict(
                "observed assembly artifact job id does not match its row"
            )
        if not self._execution_evidence_schema_exists(db):
            raise ExecutionEvidenceConflict(
                "observed assembly artifact has no execution plan evidence"
            )
        if not self._artifact_observation_schema_exists(db):
            raise ExecutionEvidenceConflict(
                "observed assembly artifact has no source observation evidence"
            )
        plan_row = db.execute(
            "SELECT p.*, j.config_snapshot AS parent_config_snapshot "
            "FROM job_execution_plans p "
            "JOIN jobs j ON j.id = p.job_id "
            "WHERE p.job_id = ?",
            (job_id,),
        ).fetchone()
        if plan_row is None:
            raise ExecutionEvidenceConflict(
                "observed assembly artifact has no compiled execution plan"
            )
        try:
            plan = CompiledExecutionPlan.model_validate_json(
                plan_row["compiled_plan"]
            )
        except (TypeError, ValueError) as exc:
            raise ExecutionEvidenceConflict(
                "observed assembly artifact has an invalid compiled plan"
            ) from exc
        plan_digest = compiled_execution_plan_digest(plan).value
        if (
            plan.version != int(plan_row["schema_version"])
            or plan_digest != str(plan_row["compiled_plan_digest"])
            or plan_digest != artifact.source_compiled_plan_digest.value
        ):
            raise ExecutionEvidenceConflict(
                "observed assembly artifact compiled plan digest/index is invalid"
            )

        try:
            config_snapshot = json.loads(plan_row["parent_config_snapshot"])
            if not isinstance(config_snapshot, Mapping):
                raise ValueError("job config snapshot is not an object")
            ordered_units = ordered_compiled_segment_units(
                plan,
                config_snapshot,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ExecutionEvidenceConflict(
                "observed assembly artifact has no trustworthy timeline order"
            ) from exc
        expected_segment_ids = tuple(
            unit.owner_segment_id for unit in ordered_units
        )
        source_segment_ids = tuple(
            source.segment_id for source in artifact.source_artifacts
        )
        if source_segment_ids != expected_segment_ids:
            raise ExecutionEvidenceConflict(
                "observed assembly sources do not exactly follow captured timeline order"
            )

        for source, unit in zip(
            artifact.source_artifacts, ordered_units, strict=True
        ):
            resolved = self._observed_segment_take_in_connection(
                db,
                source_child_id=source.child_id,
            )
            if resolved is None:
                raise ExecutionEvidenceConflict(
                    "observed assembly source has no durable observed take"
                )
            take, observed = resolved
            child_row = db.execute(
                "SELECT c.*, "
                "e.unit_id AS evidence_unit_id, "
                "e.source_compiled_plan_digest "
                "AS evidence_source_compiled_plan_digest "
                "FROM job_children c "
                "JOIN job_child_execution_evidence e ON e.child_id = c.id "
                "WHERE c.id = ?",
                (source.child_id,),
            ).fetchone()
            if child_row is None:
                raise ExecutionEvidenceConflict(
                    "observed assembly source has no live child execution evidence"
                )
            try:
                child = self._job_child_row(child_row)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ExecutionEvidenceConflict(
                    "observed assembly source child is invalid"
                ) from exc
            if (
                str(take["source_job_id"]) != job_id
                or str(take["source_child_id"]) != source.child_id
                or str(child["job_id"]) != job_id
                or child["status"] != "succeeded"
                or child["segment_ids"] != [source.segment_id]
                or observed.segment_id != source.segment_id
                or observed.child_id != source.child_id
                or str(child_row["evidence_unit_id"]) != unit.id
                or str(child_row["evidence_source_compiled_plan_digest"])
                != plan_digest
                or self._execution_document_digest(observed)
                != source.observed_artifact_digest.value
            ):
                raise ExecutionEvidenceConflict(
                    "observed assembly source differs from compiled observed evidence"
                )
        return plan

    def get_observed_assembly_artifact(
        self, job_id: str
    ) -> ObservedAssemblyArtifactSpec | None:
        """Read parent output authority without materializing its lazy schema."""

        with self.connect() as db:
            db.execute("BEGIN")
            if not self._assembly_artifact_schema_exists(db):
                return None
            row = db.execute(
                "SELECT * FROM job_observed_assembly_artifacts WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            artifact = self._observed_assembly_artifact_row(row)
            self._validate_assembly_sources_in_connection(
                db,
                job_id=job_id,
                artifact=artifact,
            )
            return artifact

    def get_output_observation_receipt(
        self, child_id: str
    ) -> OutputObservationReceipt | None:
        with self.connect() as db:
            if not self._artifact_observation_schema_exists(db):
                return None
            row = db.execute(
                "SELECT * FROM job_child_output_receipts WHERE child_id = ?",
                (child_id,),
            ).fetchone()
            if row is None:
                return None
            receipt = self._output_observation_receipt_row(row)
            self._validate_output_receipt_exact_in_connection(
                db,
                child_id=child_id,
                receipt=receipt,
            )
            return receipt

    def get_observed_artifact(
        self, child_id: str
    ) -> ObservedArtifactSpec | None:
        with self.connect() as db:
            if not self._artifact_observation_schema_exists(db):
                return None
            resolved = self._observed_segment_take_in_connection(
                db,
                source_child_id=child_id,
            )
        return resolved[1] if resolved is not None else None

    def list_pending_output_observations(
        self, *, limit: int = 16
    ) -> list[tuple[dict[str, Any], OutputObservationReceipt]]:
        if limit < 1:
            return []
        with self.connect() as db:
            if not self._artifact_observation_schema_exists(db):
                return []
            rows = db.execute(
                "SELECT c.*, r.child_id, r.schema_version, "
                "r.receipt, r.receipt_digest, "
                "r.created_at AS receipt_created_at "
                "FROM job_child_output_receipts r "
                "JOIN job_children c ON c.id = r.child_id "
                "JOIN prompt_ownership o ON o.child_id = c.id "
                "LEFT JOIN segment_take_observed_artifacts a "
                "ON a.source_child_id = c.id "
                "WHERE a.source_child_id IS NULL "
                "AND o.state = 'terminal_confirmed' "
                "AND c.status NOT IN ('succeeded','failed','cancelled') "
                "ORDER BY r.created_at, c.id LIMIT ?",
                (min(limit, 256),),
            ).fetchall()
            pending: list[tuple[dict[str, Any], OutputObservationReceipt]] = []
            for row in rows:
                receipt = self._output_observation_receipt_row(row)
                self._validate_output_receipt_exact_in_connection(
                    db,
                    child_id=str(row["child_id"]),
                    receipt=receipt,
                )
                pending.append((self._job_child_row(row), receipt))
            return pending

    def record_output_observation_receipt(
        self,
        child_id: str,
        *,
        expected_revision: int,
        receipt: OutputObservationReceipt,
        updated_at: datetime,
    ) -> tuple[dict[str, Any], PromptOwnership, OutputObservationReceipt] | None:
        """Release prompt ownership and durably queue local media probing."""

        if receipt.child_id != child_id:
            raise ValueError("output receipt child id does not match its row")
        receipt_json = self._contract_json(receipt)
        receipt_digest = self._canonical_json_payload_digest(receipt_json)
        now = updated_at.astimezone(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not self._execution_evidence_schema_exists(db):
                return None
            self._ensure_artifact_observation_schema(db)
            child_row = db.execute(
                "SELECT * FROM job_children WHERE id = ?", (child_id,)
            ).fetchone()
            if child_row is None:
                raise KeyError(child_id)
            child = self._job_child_row(child_row)
            current = self._prompt_ownership_in_connection(db, child_id)
            if current is None:
                return None
            self._validate_output_receipt_exact_in_connection(
                db,
                child_id=child_id,
                receipt=receipt,
            )
            existing_row = db.execute(
                "SELECT * FROM job_child_output_receipts WHERE child_id = ?",
                (child_id,),
            ).fetchone()
            if existing_row is not None:
                existing = self._output_observation_receipt_row(existing_row)
                if self._contract_json(existing) != receipt_json:
                    raise ExecutionEvidenceConflict(
                        "output observation receipt conflicts with durable evidence"
                    )
                if (
                    current.state != "terminal_confirmed"
                    or current.cleanup_certificate != receipt.history_evidence
                ):
                    raise ExecutionEvidenceConflict(
                        "output receipt exists without matching terminal ownership"
                    )
                return child, current, existing
            if current.ownership_revision != expected_revision:
                return None
            if child["status"] in self._TERMINAL_JOB_STATUSES:
                raise ExecutionEvidenceConflict(
                    "terminal child cannot begin output observation"
                )
            if child.get("prompt_id") != current.effective_prompt_id:
                return None
            if receipt.history_evidence.prompt_id != current.effective_prompt_id:
                raise ExecutionEvidenceConflict(
                    "output receipt history does not match prompt ownership"
                )
            db.execute(
                "INSERT INTO job_child_output_receipts("
                "child_id, schema_version, receipt, receipt_digest, created_at) "
                "VALUES(?, 1, ?, ?, ?)",
                (child_id, receipt_json, receipt_digest, now),
            )
            next_ownership = self._transition_prompt_ownership_in_connection(
                db,
                child_id,
                expected_revision=expected_revision,
                state="terminal_confirmed",
                updated_at=updated_at,
                cleanup_certificate=receipt.history_evidence,
            )
            if next_ownership is None:
                return None
            cursor = db.execute(
                "UPDATE job_children SET status = 'running', "
                "progress = CASE WHEN progress < 0.99 THEN 0.99 ELSE progress END, "
                "stage = 'verifying_output', outputs = '[]', error = NULL, "
                "updated_at = ?, started_at = COALESCE(started_at, ?) "
                "WHERE id = ? AND prompt_id = ? "
                "AND status NOT IN ('succeeded','failed','cancelled')",
                (now, now, child_id, current.effective_prompt_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    "output receipt child changed under the write transaction"
                )
            if child["backend"] == "raylight":
                self._settle_raylight_runtime_prompt_in_connection(
                    db,
                    current.effective_prompt_id,
                    succeeded=True,
                    terminal_history_certified=True,
                    updated_at=now,
                )
            settled_row = db.execute(
                "SELECT * FROM job_children WHERE id = ?", (child_id,)
            ).fetchone()
            if settled_row is None:
                raise RuntimeError("output observation child disappeared")
            return self._job_child_row(settled_row), next_ownership, receipt

    def finalize_observed_artifact(
        self,
        child_id: str,
        *,
        artifact: ObservedArtifactSpec,
        updated_at: datetime,
    ) -> tuple[dict[str, Any], dict[str, Any], ObservedArtifactSpec]:
        """Publish observed media, reusable take, and child success atomically."""

        if artifact.child_id != child_id:
            raise ValueError("observed artifact child id does not match its row")
        artifact_json = self._contract_json(artifact)
        artifact_digest = self._canonical_json_payload_digest(artifact_json)
        now = updated_at.astimezone(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not self._artifact_observation_schema_exists(db):
                raise ExecutionEvidenceConflict(
                    "observed artifact has no durable output receipt"
                )
            receipt_row = db.execute(
                "SELECT * FROM job_child_output_receipts WHERE child_id = ?",
                (child_id,),
            ).fetchone()
            if receipt_row is None:
                raise ExecutionEvidenceConflict(
                    "observed artifact has no durable output receipt"
                )
            receipt = self._output_observation_receipt_row(receipt_row)
            self._validate_output_receipt_exact_in_connection(
                db,
                child_id=child_id,
                receipt=receipt,
            )
            if (
                artifact.segment_id != receipt.segment_id
                or artifact.output_descriptor != receipt.output_descriptor
            ):
                raise ExecutionEvidenceConflict(
                    "observed artifact differs from its trusted history descriptor"
                )
            ownership = self._prompt_ownership_in_connection(db, child_id)
            if (
                ownership is None
                or ownership.state != "terminal_confirmed"
                or ownership.cleanup_certificate != receipt.history_evidence
            ):
                raise ExecutionEvidenceConflict(
                    "observed artifact lacks matching terminal ownership"
                )
            existing_artifact_row = db.execute(
                "SELECT 1 FROM segment_take_observed_artifacts "
                "WHERE source_child_id = ?",
                (child_id,),
            ).fetchone()
            if existing_artifact_row is not None:
                resolved = self._observed_segment_take_in_connection(
                    db,
                    source_child_id=child_id,
                )
                if resolved is None:
                    raise ExecutionEvidenceConflict(
                        "observed artifact transaction is incomplete"
                    )
                take, existing = resolved
                if self._contract_json(existing) != artifact_json:
                    raise ExecutionEvidenceConflict(
                        "a different observed artifact is already durable"
                    )
                child_row = db.execute(
                    "SELECT * FROM job_children WHERE id = ?", (child_id,)
                ).fetchone()
                if child_row is None:
                    raise ExecutionEvidenceConflict(
                        "observed artifact transaction is incomplete"
                    )
                return (
                    self._job_child_row(child_row),
                    take,
                    existing,
                )
            child_row = db.execute(
                "SELECT job_children.*, "
                "jobs.config_snapshot AS parent_config_snapshot, "
                "jobs.project_id AS parent_project_id "
                "FROM job_children JOIN jobs ON jobs.id = job_children.job_id "
                "WHERE job_children.id = ?",
                (child_id,),
            ).fetchone()
            if child_row is None:
                raise KeyError(child_id)
            child = self._job_child_row(child_row)
            if child["status"] in self._TERMINAL_JOB_STATUSES:
                raise ExecutionEvidenceConflict(
                    "terminal child conflicts with pending output observation"
                )
            existing_take = db.execute(
                "SELECT id FROM segment_takes WHERE source_child_id = ?",
                (child_id,),
            ).fetchone()
            if existing_take is not None:
                raise ExecutionEvidenceConflict(
                    "typed output cannot overwrite a legacy segment take"
                )
            try:
                config_snapshot = json.loads(child_row["parent_config_snapshot"])
                timeline_document = config_snapshot.get("timeline")
                timeline = validate_timeline_snapshot(timeline_document)
                matches = [
                    segment
                    for segment in timeline.segments
                    if segment.id == artifact.segment_id
                ]
                if len(matches) != 1:
                    raise ValueError("observed segment is absent from its timeline")
                content_fingerprint = segment_take_geometry_fingerprint(
                    width=artifact.width,
                    height=artifact.height,
                    fps=artifact.fps,
                    visible_frame_count=artifact.frame_count,
                )
            except (TypeError, ValueError, ValidationError, json.JSONDecodeError) as exc:
                raise ExecutionEvidenceConflict(
                    "observed artifact has no valid timeline content identity"
                ) from exc
            take_id = str(uuid.uuid4())
            db.execute(
                "INSERT INTO segment_takes("
                "id, segment_id, content_fingerprint, project_id, "
                "output_descriptor, has_audio, source_job_id, source_child_id, "
                "completed_at, created_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    take_id,
                    artifact.segment_id,
                    content_fingerprint,
                    child_row["parent_project_id"],
                    self._contract_json(artifact.output_descriptor),
                    int(artifact.has_audio),
                    str(child["job_id"]),
                    child_id,
                    now,
                    now,
                ),
            )
            db.execute(
                "INSERT INTO segment_take_observed_artifacts("
                "take_id, source_child_id, schema_version, observed_artifact, "
                "observed_artifact_digest, receipt_digest, created_at) "
                "VALUES(?, ?, 1, ?, ?, ?, ?)",
                (
                    take_id,
                    child_id,
                    artifact_json,
                    artifact_digest,
                    str(receipt_row["receipt_digest"]),
                    now,
                ),
            )
            compatibility_output = {
                "node_id": receipt.node_id,
                **artifact.output_descriptor.model_dump(mode="json"),
            }
            cursor = db.execute(
                "UPDATE job_children SET status = 'succeeded', progress = 1.0, "
                "stage = 'completed', outputs = ?, error = NULL, updated_at = ?, "
                "completed_at = ? WHERE id = ? "
                "AND status NOT IN ('succeeded','failed','cancelled')",
                (
                    json.dumps([compatibility_output], ensure_ascii=False),
                    now,
                    now,
                    child_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    "observed artifact child changed under the write transaction"
                )
            settled_child_row = db.execute(
                "SELECT * FROM job_children WHERE id = ?", (child_id,)
            ).fetchone()
            settled_take_row = db.execute(
                "SELECT * FROM segment_takes WHERE id = ?", (take_id,)
            ).fetchone()
            if settled_child_row is None or settled_take_row is None:
                raise RuntimeError("observed artifact publication disappeared")
            return (
                self._job_child_row(settled_child_row),
                self._segment_take_row(settled_take_row),
                artifact,
            )

    def finalize_observed_assembly_artifact(
        self,
        job_id: str,
        *,
        expected_updated_at: str,
        artifact: ObservedAssemblyArtifactSpec,
        updated_at: datetime,
    ) -> dict[str, Any] | None:
        """Publish parent assembly evidence and parent success atomically."""

        if artifact.job_id != job_id:
            raise ValueError(
                "observed assembly artifact job id does not match its row"
            )
        if updated_at.tzinfo is None or updated_at.utcoffset() is None:
            raise ValueError("assembly artifact timestamp must be timezone-aware")
        artifact_json = self._contract_json(artifact)
        artifact_digest = self._canonical_json_payload_digest(artifact_json)
        now = updated_at.astimezone(timezone.utc).isoformat()
        compatibility_output = {
            "node_id": "assembly",
            **artifact.output_descriptor.model_dump(mode="json"),
        }

        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            parent_row = db.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if parent_row is None:
                raise KeyError(job_id)
            try:
                parent = self._job_row(parent_row)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ExecutionEvidenceConflict(
                    "observed assembly parent job is invalid"
                ) from exc

            existing_row = None
            if self._assembly_artifact_schema_exists(db):
                existing_row = db.execute(
                    "SELECT * FROM job_observed_assembly_artifacts "
                    "WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
            if existing_row is not None:
                existing = self._observed_assembly_artifact_row(existing_row)
                self._validate_assembly_sources_in_connection(
                    db,
                    job_id=job_id,
                    artifact=existing,
                )
                if self._contract_json(existing) != artifact_json:
                    raise ExecutionEvidenceConflict(
                        "a different observed assembly artifact is already durable"
                    )
                if (
                    parent["status"] != "succeeded"
                    or parent["stage"] != "completed"
                    or bool(parent["cancel_requested"])
                    or float(parent["progress"]) != 1.0
                    or parent["outputs"] != [compatibility_output]
                    or parent["error"] is not None
                    or parent["completed_at"] is None
                ):
                    raise ExecutionEvidenceConflict(
                        "observed assembly artifact conflicts with parent terminal state"
                    )
                return parent

            if (
                parent["status"] != "running"
                or parent["stage"] != "assembling"
                or str(parent["updated_at"]) != expected_updated_at
                or bool(parent["cancel_requested"])
            ):
                return None

            self._validate_assembly_sources_in_connection(
                db,
                job_id=job_id,
                artifact=artifact,
            )
            self._ensure_assembly_artifact_schema(db)
            db.execute(
                "INSERT INTO job_observed_assembly_artifacts("
                "job_id, schema_version, source_compiled_plan_digest, "
                "observed_assembly_artifact, "
                "observed_assembly_artifact_digest, created_at) "
                "VALUES(?, 1, ?, ?, ?, ?)",
                (
                    job_id,
                    artifact.source_compiled_plan_digest.value,
                    artifact_json,
                    artifact_digest,
                    now,
                ),
            )
            cursor = db.execute(
                "UPDATE jobs SET status = 'succeeded', progress = 1.0, "
                "stage = 'completed', outputs = ?, error = NULL, "
                "updated_at = ?, completed_at = ? "
                "WHERE id = ? AND status = 'running' AND stage = 'assembling' "
                "AND updated_at = ? AND cancel_requested = 0",
                (
                    json.dumps([compatibility_output], ensure_ascii=False),
                    now,
                    now,
                    job_id,
                    expected_updated_at,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    "assembly parent changed under the write transaction"
                )
            settled_row = db.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if settled_row is None:
                raise RuntimeError("observed assembly parent disappeared")
            return self._job_row(settled_row)

    def fail_output_observation(
        self,
        child_id: str,
        *,
        error: str,
        updated_at: datetime,
    ) -> dict[str, Any]:
        """Fail Director verification without falsifying successful history."""

        if not error.strip():
            raise ValueError("output observation failure requires an error")
        now = updated_at.astimezone(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not self._artifact_observation_schema_exists(db):
                raise ExecutionEvidenceConflict(
                    "output observation failure has no durable receipt"
                )
            receipt_row = db.execute(
                "SELECT * FROM job_child_output_receipts WHERE child_id = ?",
                (child_id,),
            ).fetchone()
            observed_row = db.execute(
                "SELECT 1 FROM segment_take_observed_artifacts "
                "WHERE source_child_id = ?",
                (child_id,),
            ).fetchone()
            ownership = self._prompt_ownership_in_connection(db, child_id)
            receipt = (
                self._output_observation_receipt_row(receipt_row)
                if receipt_row is not None
                else None
            )
            if receipt is not None:
                self._validate_output_receipt_exact_in_connection(
                    db,
                    child_id=child_id,
                    receipt=receipt,
                )
            if (
                receipt is None
                or observed_row is not None
                or ownership is None
                or ownership.state != "terminal_confirmed"
                or ownership.cleanup_certificate != receipt.history_evidence
            ):
                raise ExecutionEvidenceConflict(
                    "output observation failure conflicts with durable evidence"
                )
            cursor = db.execute(
                "UPDATE job_children SET status = 'failed', progress = 1.0, "
                "stage = 'artifact_verification_failed', outputs = '[]', "
                "error = ?, updated_at = ?, completed_at = ? WHERE id = ? "
                "AND status NOT IN ('succeeded','failed','cancelled')",
                (error[:20_000], now, now, child_id),
            )
            if cursor.rowcount != 1:
                row = db.execute(
                    "SELECT * FROM job_children WHERE id = ?", (child_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(child_id)
                child = self._job_child_row(row)
                if child["status"] != "failed":
                    raise ExecutionEvidenceConflict(
                        "output observation failure lost its child CAS"
                    )
                return child
            row = db.execute(
                "SELECT * FROM job_children WHERE id = ?", (child_id,)
            ).fetchone()
            if row is None:
                raise RuntimeError("failed output observation child disappeared")
            return self._job_child_row(row)

    def fail_successful_history_artifact(
        self,
        child_id: str,
        *,
        expected_revision: int,
        evidence: HistoryTerminalEvidence,
        error: str,
        updated_at: datetime,
    ) -> tuple[dict[str, Any], PromptOwnership] | None:
        """Release a successful prompt whose output descriptor is unusable."""

        if evidence.terminal_status != "succeeded":
            raise ValueError("artifact failure requires successful history evidence")
        if not error.strip():
            raise ValueError("artifact failure requires an error")
        now = updated_at.astimezone(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not self._execution_evidence_schema_exists(db):
                return None
            current = self._prompt_ownership_in_connection(db, child_id)
            if current is None:
                return None
            child_row = db.execute(
                "SELECT * FROM job_children WHERE id = ?", (child_id,)
            ).fetchone()
            if child_row is None:
                raise KeyError(child_id)
            child = self._job_child_row(child_row)
            if current.state == "terminal_confirmed":
                if (
                    current.cleanup_certificate == evidence
                    and child["status"] == "failed"
                    and child.get("stage") == "artifact_verification_failed"
                ):
                    return child, current
                raise ExecutionEvidenceConflict(
                    "artifact failure conflicts with terminal ownership"
                )
            if current.ownership_revision != expected_revision:
                return None
            if (
                child["status"] in self._TERMINAL_JOB_STATUSES
                or child.get("prompt_id") != current.effective_prompt_id
                or evidence.prompt_id != current.effective_prompt_id
            ):
                raise ExecutionEvidenceConflict(
                    "artifact failure does not match the active prompt"
                )
            exact_row = db.execute(
                "SELECT * FROM job_child_execution_evidence WHERE child_id = ?",
                (child_id,),
            ).fetchone()
            if exact_row is None:
                raise ExecutionEvidenceConflict(
                    "artifact failure has no exact prompt snapshot"
                )
            exact = self._exact_prompt_snapshot_from_evidence_row(
                exact_row,
                child_id=child_id,
            )
            if exact.unit_kind != "segment" or exact.expected_output_spec is None:
                raise ExecutionEvidenceConflict(
                    "artifact failure requires a typed segment output"
                )
            next_ownership = self._transition_prompt_ownership_in_connection(
                db,
                child_id,
                expected_revision=expected_revision,
                state="terminal_confirmed",
                updated_at=updated_at,
                cleanup_certificate=evidence,
            )
            if next_ownership is None:
                return None
            cursor = db.execute(
                "UPDATE job_children SET status = 'failed', progress = 1.0, "
                "stage = 'artifact_verification_failed', outputs = '[]', "
                "error = ?, updated_at = ?, completed_at = ? WHERE id = ? "
                "AND status NOT IN ('succeeded','failed','cancelled')",
                (error[:20_000], now, now, child_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("artifact failure child changed under write lock")
            if child["backend"] == "raylight":
                self._settle_raylight_runtime_prompt_in_connection(
                    db,
                    current.effective_prompt_id,
                    succeeded=True,
                    terminal_history_certified=True,
                    updated_at=now,
                )
            settled_row = db.execute(
                "SELECT * FROM job_children WHERE id = ?", (child_id,)
            ).fetchone()
            if settled_row is None:
                raise RuntimeError("artifact failure child disappeared")
            return self._job_child_row(settled_row), next_ownership

    def _confirm_prompt_release(
        self,
        child_id: str,
        *,
        expected_revision: int,
        state: PromptOwnershipState,
        evidence: PromptReleaseEvidence,
        status: str,
        stage: str,
        outputs: list[dict[str, Any]],
        error: str | None,
        updated_at: datetime,
        completed_at: str | None,
    ) -> tuple[dict[str, Any], PromptOwnership] | None:
        if status not in self._TERMINAL_JOB_STATUSES:
            raise ValueError("prompt release must publish a terminal child status")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not self._execution_evidence_schema_exists(db):
                return None
            current = self._prompt_ownership_in_connection(db, child_id)
            if current is None or current.ownership_revision != expected_revision:
                return None
            child_row = db.execute(
                "SELECT * FROM job_children WHERE id = ?", (child_id,)
            ).fetchone()
            if child_row is None:
                raise KeyError(child_id)
            child = self._job_child_row(child_row)
            if child.get("prompt_id") != current.effective_prompt_id:
                return None
            exact_row = db.execute(
                "SELECT * FROM job_child_execution_evidence WHERE child_id = ?",
                (child_id,),
            ).fetchone()
            if exact_row is None:
                raise ExecutionEvidenceConflict(
                    "prompt ownership has no immutable exact execution evidence"
                )
            _locked_plan, exact_snapshot, exact_unit = (
                self._execution_evidence_from_row(
                    exact_row,
                    child_id=child_id,
                )
            )
            expected_segment_ids, expected_output_nodes = self._derived_child_identity(
                exact_unit
            )
            if (
                child["group_index"] != exact_unit.group_index
                or child["family"] != exact_unit.family
                or child["backend"] != exact_unit.backend
                or child["segment_ids"] != expected_segment_ids
                or child["output_nodes"] != expected_output_nodes
            ):
                raise ExecutionEvidenceConflict(
                    "prompt release child differs from immutable execution evidence"
                )
            if status == "succeeded" and isinstance(exact_unit, LockedSegmentUnit):
                raise ExecutionEvidenceConflict(
                    "typed segment success requires an observed artifact transaction"
                )
            if (
                child["status"] in self._TERMINAL_JOB_STATUSES
                and child["status"] != status
            ):
                raise ExecutionEvidenceConflict(
                    "terminal child status conflicts with prompt release evidence"
                )

            if isinstance(evidence, HistoryTerminalEvidence):
                if state != "terminal_confirmed" or status != evidence.terminal_status:
                    raise ValueError(
                        "history evidence must match terminal ownership and child status"
                    )
            elif isinstance(
                evidence,
                (ExactCancelConfirmedEvidence, EndpointRestartCertificate),
            ):
                if state != "cleanup_confirmed" or status != "cancelled":
                    raise ValueError(
                        "cleanup evidence must publish cancelled cleanup ownership"
                    )
            else:
                raise TypeError("unsupported prompt release evidence")

            if isinstance(evidence, EndpointRestartCertificate):
                if evidence.endpoint_identity != exact_snapshot.endpoint_identity:
                    raise ExecutionEvidenceConflict(
                        "restart certificate endpoint does not match exact prompt"
                    )

            next_ownership = self._transition_prompt_ownership_in_connection(
                db,
                child_id,
                expected_revision=expected_revision,
                state=state,
                updated_at=updated_at,
                cleanup_certificate=evidence,
            )
            if next_ownership is None:
                return None
            finished_at = completed_at or updated_at.isoformat()
            cursor = db.execute(
                "UPDATE job_children SET status = ?, progress = 1.0, stage = ?, "
                "prompt_id = ?, outputs = ?, error = ?, updated_at = ?, "
                "completed_at = ? WHERE id = ? AND prompt_id = ?",
                (
                    status,
                    stage,
                    next_ownership.effective_prompt_id,
                    json.dumps(outputs, ensure_ascii=False),
                    error,
                    updated_at.isoformat(),
                    finished_at,
                    child_id,
                    current.effective_prompt_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("prompt release child CAS changed under write lock")
            if child["backend"] == "raylight":
                if isinstance(evidence, HistoryTerminalEvidence):
                    self._settle_raylight_runtime_prompt_in_connection(
                        db,
                        next_ownership.effective_prompt_id,
                        succeeded=status == "succeeded",
                        terminal_history_certified=True,
                        updated_at=updated_at.isoformat(),
                    )
                else:
                    self._cleanup_raylight_runtime_prompt_in_connection(
                        db,
                        next_ownership.effective_prompt_id,
                        updated_at=updated_at.isoformat(),
                    )
            settled_row = db.execute(
                "SELECT * FROM job_children WHERE id = ?", (child_id,)
            ).fetchone()
            if settled_row is None:
                raise RuntimeError("released submission child disappeared")
            return self._job_child_row(settled_row), next_ownership

    def confirm_prompt_terminal(
        self,
        child_id: str,
        *,
        expected_revision: int,
        evidence: HistoryTerminalEvidence,
        outputs: list[dict[str, Any]],
        stage: str,
        error: str | None,
        updated_at: datetime,
        completed_at: str | None = None,
    ) -> tuple[dict[str, Any], PromptOwnership] | None:
        """Confirm exact history and settle ownership, child and Ray atomically."""

        return self._confirm_prompt_release(
            child_id,
            expected_revision=expected_revision,
            state="terminal_confirmed",
            evidence=evidence,
            status=evidence.terminal_status,
            stage=stage,
            outputs=outputs,
            error=error,
            updated_at=updated_at,
            completed_at=completed_at,
        )

    def confirm_prompt_cleanup(
        self,
        child_id: str,
        *,
        expected_revision: int,
        evidence: ExactCancelConfirmedEvidence | EndpointRestartCertificate,
        stage: str,
        updated_at: datetime,
        completed_at: str | None = None,
    ) -> tuple[dict[str, Any], PromptOwnership] | None:
        """Confirm exact cancel/restart cleanup in the same durable ledger."""

        return self._confirm_prompt_release(
            child_id,
            expected_revision=expected_revision,
            state="cleanup_confirmed",
            evidence=evidence,
            status="cancelled",
            stage=stage,
            outputs=[],
            error=None,
            updated_at=updated_at,
            completed_at=completed_at,
        )

    def create_job(self, values: dict[str, Any]) -> dict[str, Any]:
        columns = (
            "id", "mode", "status", "progress", "stage", "prompt_id", "project_id",
            "outputs", "error",
            "config_snapshot", "settings_snapshot", "prompt_snapshot", "created_at", "updated_at",
            "started_at", "completed_at",
        )
        row = {column: values.get(column) for column in columns}
        row["outputs"] = json.dumps(row.get("outputs") or [], ensure_ascii=False)
        row["config_snapshot"] = json.dumps(row["config_snapshot"], ensure_ascii=False)
        row["settings_snapshot"] = json.dumps(row["settings_snapshot"], ensure_ascii=False)
        if row["prompt_snapshot"] is not None:
            row["prompt_snapshot"] = json.dumps(row["prompt_snapshot"], ensure_ascii=False)
        placeholders = ",".join("?" for _ in columns)
        with self.connect() as db:
            db.execute(
                f"INSERT INTO jobs({','.join(columns)}) VALUES({placeholders})",
                tuple(row[column] for column in columns),
            )
        return self.get_job(values["id"])  # type: ignore[return-value]

    def create_job_child(self, values: dict[str, Any]) -> dict[str, Any]:
        columns = (
            "id",
            "job_id",
            "group_index",
            "family",
            "backend",
            "segment_ids",
            "output_nodes",
            "status",
            "progress",
            "stage",
            "prompt_id",
            "outputs",
            "error",
            "prompt_snapshot",
            "created_at",
            "updated_at",
            "started_at",
            "completed_at",
        )
        row = {column: values.get(column) for column in columns}
        for field in ("segment_ids", "output_nodes", "outputs", "prompt_snapshot"):
            row[field] = json.dumps(row.get(field) or ([] if field != "output_nodes" else {}), ensure_ascii=False)
        placeholders = ",".join("?" for _ in columns)
        with self.connect() as db:
            db.execute(
                f"INSERT INTO job_children({','.join(columns)}) VALUES({placeholders})",
                tuple(row[column] for column in columns),
            )
            self._register_segment_take_from_child_in_connection(
                db, str(values["id"])
            )
        child = self.get_job_child(str(values["id"]))
        if child is None:
            raise RuntimeError("created job child disappeared")
        return child

    def get_job_child(self, child_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM job_children WHERE id = ?", (child_id,)
            ).fetchone()
        return self._job_child_row(row) if row is not None else None

    def list_job_children(self, job_id: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM job_children WHERE job_id = ? ORDER BY group_index",
                (job_id,),
            ).fetchall()
        return [self._job_child_row(row) for row in rows]

    def list_job_children_for_jobs(
        self, job_ids: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Load only public-list child fields without one query per parent row."""

        children = {job_id: [] for job_id in job_ids}
        if not job_ids:
            return children
        placeholders = ",".join("?" for _ in job_ids)
        with self.connect() as db:
            rows = db.execute(
                f"SELECT {_JOB_CHILD_LIST_COLUMNS} FROM job_children "
                f"WHERE job_id IN ({placeholders}) ORDER BY job_id, group_index",
                tuple(job_ids),
            ).fetchall()
        for row in rows:
            child = self._job_child_list_row(row)
            children[str(child["job_id"])].append(child)
        return children

    def find_job_children_by_prompt_id(self, prompt_id: str) -> list[dict[str, Any]]:
        """Resolve a ComfyUI progress event without trusting a browser job id.

        Prompt ids are supplied by ComfyUI.  Returning all matches lets the
        progress sink additionally verify the parent job's endpoint snapshot,
        which matters when two configured servers happen to issue the same id.
        Terminal rows are excluded because late websocket frames must not
        revive completed or cancelled children. A ``preparing/submitting``
        row is eligible because ComfyUI can start a caller-ID prompt and emit
        its first node event before ``POST /prompt`` returns.
        """

        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM job_children "
                "WHERE prompt_id = ? AND (status IN ('queued', 'running') "
                "OR (status = 'preparing' AND stage = 'submitting')) "
                "ORDER BY created_at DESC",
                (prompt_id,),
            ).fetchall()
        return [self._job_child_row(row) for row in rows]

    def find_any_job_children_by_prompt_id(
        self, prompt_id: str
    ) -> list[dict[str, Any]]:
        """Resolve a durable Ray tail even while cancellation owns its row."""

        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM job_children WHERE prompt_id = ? "
                "ORDER BY created_at DESC",
                (prompt_id,),
            ).fetchall()
        return [self._job_child_row(row) for row in rows]

    def update_job_child(self, child_id: str, **updates: Any) -> dict[str, Any]:
        updates = self._serialize_job_child_updates(updates)
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self.connect() as db:
            cursor = db.execute(
                f"UPDATE job_children SET {assignments} WHERE id = ?",
                (*updates.values(), child_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(child_id)
            self._register_segment_take_from_child_in_connection(db, child_id)
        child = self.get_job_child(child_id)
        if child is None:
            raise KeyError(child_id)
        return child

    def update_job_child_if_status(
        self, child_id: str, expected_status: str, **updates: Any
    ) -> dict[str, Any] | None:
        updates = self._serialize_job_child_updates(updates)
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self.connect() as db:
            cursor = db.execute(
                f"UPDATE job_children SET {assignments} WHERE id = ? AND status = ?",
                (*updates.values(), child_id, expected_status),
            )
            if cursor.rowcount == 0:
                exists = db.execute(
                    "SELECT 1 FROM job_children WHERE id = ?", (child_id,)
                ).fetchone()
                if exists is None:
                    raise KeyError(child_id)
                return None
            self._register_segment_take_from_child_in_connection(db, child_id)
        return self.get_job_child(child_id)

    def replace_job_child_prompt_id_if_current(
        self,
        child_id: str,
        *,
        expected_prompt_id: str,
        prompt_id: str,
    ) -> dict[str, Any] | None:
        """CAS a caller-assigned prompt id to the id returned by ComfyUI.

        This deliberately does not predicate on lifecycle status: cancellation
        may move a live submission from ``preparing`` to ``cancelling`` while
        ``POST /prompt`` is in flight.  The durable prompt id is the ownership
        token that must still match before an unexpected upstream id is trusted.
        """

        now = utc_now()
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE job_children SET prompt_id = ?, updated_at = ? "
                "WHERE id = ? AND prompt_id = ?",
                (prompt_id, now, child_id, expected_prompt_id),
            )
            if cursor.rowcount == 0:
                exists = db.execute(
                    "SELECT 1 FROM job_children WHERE id = ?", (child_id,)
                ).fetchone()
                if exists is None:
                    raise KeyError(child_id)
                return None
        return self.get_job_child(child_id)

    def update_job_child_if_snapshot(
        self,
        child_id: str,
        *,
        expected_status: str,
        expected_updated_at: str,
        **updates: Any,
    ) -> dict[str, Any] | None:
        """Commit a status poll only while its child snapshot is current."""

        updates = self._serialize_job_child_updates(updates)
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self.connect() as db:
            cursor = db.execute(
                f"UPDATE job_children SET {assignments} "
                "WHERE id = ? AND status = ? AND updated_at = ?",
                (
                    *updates.values(),
                    child_id,
                    expected_status,
                    expected_updated_at,
                ),
            )
            if cursor.rowcount == 0:
                exists = db.execute(
                    "SELECT 1 FROM job_children WHERE id = ?", (child_id,)
                ).fetchone()
                if exists is None:
                    raise KeyError(child_id)
                return None
            self._register_segment_take_from_child_in_connection(db, child_id)
        return self.get_job_child(child_id)

    def claim_job_child_submission(
        self,
        job_id: str,
        child_id: str,
        *,
        prompt_id: str,
        prompt_snapshot: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Atomically claim one unsubmitted child while its parent is preparing.

        Cancellation closes both rows in one transaction. Predicating this
        claim on both lifecycle states *and* the durable cancel intent prevents
        the submitter from reviving a terminal child—or claiming the next
        continuity successor during the short interval before the cancel owner
        advances the parent from ``preparing`` to ``cancelling``.
        """

        now = utc_now()
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE job_children SET stage = 'submitting', prompt_id = ?, "
                "prompt_snapshot = ?, updated_at = ? "
                "WHERE id = ? AND job_id = ? AND status = 'preparing' "
                "AND EXISTS (SELECT 1 FROM jobs WHERE id = ? "
                "AND status = 'preparing' AND cancel_requested = 0)",
                (
                    prompt_id,
                    json.dumps(prompt_snapshot, ensure_ascii=False),
                    now,
                    child_id,
                    job_id,
                    job_id,
                ),
            )
            if cursor.rowcount == 0:
                exists = db.execute(
                    "SELECT 1 FROM job_children WHERE id = ?", (child_id,)
                ).fetchone()
                if exists is None:
                    raise KeyError(child_id)
                return None
        return self.get_job_child(child_id)

    def fail_job_child_dependency_if_dispatching(
        self,
        job_id: str,
        child_id: str,
        *,
        expected_updated_at: str,
        error: str,
    ) -> dict[str, Any] | None:
        """Fail one unsubmitted child only while cancellation has not won.

        Dependency convergence races the explicit cancel endpoint after an
        awaited predecessor history read. Keep the parent intent and child
        transition in one SQL predicate so an operator cancellation can never
        leave a provably unsubmitted child labelled as a generation failure.
        """

        now = utc_now()
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE job_children SET status = 'failed', progress = 1.0, "
                "stage = 'dependency_failed', error = ?, updated_at = ?, "
                "completed_at = ? WHERE id = ? AND job_id = ? "
                "AND status = 'preparing' AND prompt_id IS NULL "
                "AND updated_at = ? AND EXISTS (SELECT 1 FROM jobs "
                "WHERE id = ? AND status = 'preparing' AND cancel_requested = 0)",
                (
                    error,
                    now,
                    now,
                    child_id,
                    job_id,
                    expected_updated_at,
                    job_id,
                ),
            )
            if cursor.rowcount == 0:
                exists = db.execute(
                    "SELECT 1 FROM job_children WHERE id = ?", (child_id,)
                ).fetchone()
                if exists is None:
                    raise KeyError(child_id)
                return None
        return self.get_job_child(child_id)

    def finalize_timeline_submission_failure(
        self,
        job_id: str,
        *,
        error: str,
        prompt_id: str | None,
        failure_stage: str = "submission_failed",
    ) -> dict[str, Any] | None:
        """Close a fully-cleaned submission without racing user cancellation.

        The cleanup coroutine can spend time awaiting ComfyUI after its first
        parent snapshot.  A user cancellation may claim the parent meanwhile.
        Reading the live parent and closing its remaining children in the same
        write transaction makes that cancellation intent monotonic: once the
        parent is ``cancelling``, stale submission failure cleanup can only
        finish it as ``cancelled`` and can never revive it as ``failed``.
        """

        now = utc_now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            parent = db.execute(
                "SELECT status, stage, cancel_requested FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if parent is None:
                return None
            if parent["status"] in {"succeeded", "failed", "cancelled"}:
                return self._job_row(
                    db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                )

            user_cancel_won = bool(parent["cancel_requested"])
            child_status = "cancelled" if user_cancel_won else "failed"
            child_stage = "not_submitted" if user_cancel_won else "submission_failed"
            child_error = None if user_cancel_won else error
            db.execute(
                "UPDATE job_children SET status = ?, progress = 1.0, stage = ?, "
                "error = ?, updated_at = ?, completed_at = ? "
                "WHERE job_id = ? AND status NOT IN ('succeeded', 'failed', 'cancelled')",
                (
                    child_status,
                    child_stage,
                    child_error,
                    now,
                    now,
                    job_id,
                ),
            )
            parent_status = "cancelled" if user_cancel_won else "failed"
            parent_stage = "cancelled" if user_cancel_won else failure_stage
            parent_error = None if user_cancel_won else error
            db.execute(
                "UPDATE jobs SET status = ?, progress = 1.0, stage = ?, "
                "prompt_id = ?, error = ?, updated_at = ?, completed_at = ? "
                "WHERE id = ? AND status NOT IN ('succeeded', 'failed', 'cancelled')",
                (
                    parent_status,
                    parent_stage,
                    prompt_id,
                    parent_error,
                    now,
                    now,
                    job_id,
                ),
            )
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._job_row(row) if row is not None else None

    def mark_job_cancel_requested(
        self, job_id: str
    ) -> tuple[dict[str, Any] | None, bool]:
        """Atomically claim the first operator cancel request for one job."""

        now = utc_now()
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE jobs SET cancel_requested = 1, updated_at = ? "
                "WHERE id = ? AND cancel_requested = 0 "
                "AND status NOT IN ('succeeded', 'failed', 'cancelled')",
                (now, job_id),
            )
            if cursor.rowcount == 0:
                exists = db.execute(
                    "SELECT 1 FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
                if exists is None:
                    raise KeyError(job_id)
                return self.get_job(job_id), False
        return self.get_job(job_id), True

    def update_job_child_progress_monotonic(
        self,
        child_id: str,
        *,
        progress: float,
        stage: str,
        expected_updated_at: str,
    ) -> dict[str, Any] | None:
        """Persist a standard websocket step event without allowing rewind.

        ``updated_at`` is the row version.  It prevents an older equal-progress
        phase from overwriting a newer stage, while the numeric predicate
        rejects a delayed event from an earlier sampler/segment.
        """

        now = utc_now()
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE job_children SET status = 'running', progress = ?, "
                "stage = ?, started_at = COALESCE(started_at, ?), updated_at = ? "
                "WHERE id = ? AND (status IN ('queued', 'running') "
                "OR (status = 'preparing' AND stage = 'submitting')) "
                "AND progress <= ? AND updated_at = ?",
                (
                    progress,
                    stage,
                    now,
                    now,
                    child_id,
                    progress,
                    expected_updated_at,
                ),
            )
            if cursor.rowcount == 0:
                exists = db.execute(
                    "SELECT 1 FROM job_children WHERE id = ?", (child_id,)
                ).fetchone()
                if exists is None:
                    raise KeyError(child_id)
                return None
        return self.get_job_child(child_id)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._job_row(row) if row else None

    def get_job_status(self, job_id: str) -> str | None:
        """Read lifecycle state without decoding immutable job snapshots."""

        with self.connect() as db:
            row = db.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return str(row["status"]) if row is not None else None

    def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._job_row(row) for row in rows]

    def list_jobs_page(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        statuses: tuple[str, ...] = (),
        search: str | None = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> tuple[list[dict[str, Any]], int]:
        """Return one filtered page from local task history.

        The query intentionally touches SQLite only. Lifecycle reconciliation
        belongs to the managed background worker; browser filtering must never
        turn into queue/history traffic to a slow or unavailable ComfyUI.
        """

        if not 1 <= limit <= 256:
            raise ValueError("job page limit must be between 1 and 256")
        if offset < 0:
            raise ValueError("job page offset must be non-negative")
        allowed_statuses = {
            "queued",
            "preparing",
            "running",
            "succeeded",
            "failed",
            "cancelling",
            "cancelled",
        }
        if set(statuses) - allowed_statuses:
            raise ValueError("unsupported job status filter")
        if sort_by not in {"created_at", "execution_duration"}:
            raise ValueError("unsupported job sort field")
        if sort_order not in {"asc", "desc"}:
            raise ValueError("unsupported job sort order")

        conditions: list[str] = []
        parameters: list[Any] = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            conditions.append(f"status IN ({placeholders})")
            parameters.extend(statuses)
        normalized_search = (search or "").strip()
        if normalized_search:
            # Escape LIKE metacharacters so a title containing '%' or '_' is
            # searched literally rather than broadening the task query.
            escaped = (
                normalized_search.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            pattern = f"%{escaped}%"
            conditions.append(
                "(id LIKE ? ESCAPE '\\' OR mode LIKE ? ESCAPE '\\' "
                "OR COALESCE(stage, '') LIKE ? ESCAPE '\\' "
                "OR COALESCE(error, '') LIKE ? ESCAPE '\\' "
                "OR COALESCE(json_extract(config_snapshot, '$.timeline.title'), '') "
                "LIKE ? ESCAPE '\\')"
            )
            parameters.extend([pattern] * 5)

        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        if sort_by == "execution_duration":
            duration = (
                "MAX((julianday(COALESCE(completed_at, updated_at)) - "
                "julianday(started_at)) * 86400.0, 0.0)"
            )
            order = (
                f" ORDER BY (started_at IS NULL), {duration} {sort_order.upper()}, "
                "created_at DESC, id"
            )
        else:
            order = f" ORDER BY created_at {sort_order.upper()}, id"

        with self.connect() as db:
            total = int(
                db.execute(
                    "SELECT COUNT(*) FROM jobs" + where,
                    tuple(parameters),
                ).fetchone()[0]
            )
            rows = db.execute(
                f"SELECT {_JOB_LIST_COLUMNS} FROM jobs"
                + where
                + order
                + " LIMIT ? OFFSET ?",
                (*parameters, limit, offset),
            ).fetchall()
        return [self._job_row(row) for row in rows], total

    def job_status_summary(self) -> dict[str, int]:
        """Count the complete local history independently of list filters."""

        counts = {
            "queued": 0,
            "preparing": 0,
            "running": 0,
            "succeeded": 0,
            "failed": 0,
            "cancelling": 0,
            "cancelled": 0,
        }
        with self.connect() as db:
            rows = db.execute(
                "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
            ).fetchall()
        for row in rows:
            status = str(row["status"])
            if status in counts:
                counts[status] = int(row["count"])
        counts["total"] = sum(counts.values())
        counts["active"] = sum(
            counts[status]
            for status in ("queued", "preparing", "running", "cancelling")
        )
        return counts

    def has_active_work(self) -> bool:
        """Return whether any parent or child can still mutate lifecycle state."""

        with self.connect() as db:
            row = db.execute(
                "SELECT EXISTS("
                "SELECT 1 FROM jobs "
                "WHERE status NOT IN ('succeeded', 'failed', 'cancelled')"
                ") OR EXISTS("
                "SELECT 1 FROM job_children "
                "WHERE status NOT IN ('succeeded', 'failed', 'cancelled')"
                ")"
            ).fetchone()
        return bool(row[0])

    def list_active_timeline_jobs(self, limit: int = 4) -> list[dict[str, Any]]:
        """Return a bounded, fair batch for process-owned reconciliation.

        Oldest ``updated_at`` comes first. Every successful reconciliation
        advances that row version, naturally rotating a large active set
        without an in-memory cursor that would be lost on restart.  Historical
        pre-parent/child rows are included too: HTTP reads are SQLite-only, so
        the background owner must remain able to converge those legacy jobs.
        """

        if limit < 1:
            raise ValueError("active timeline job limit must be positive")
        with self.connect() as db:
            rows = db.execute(
                "SELECT jobs.* FROM jobs "
                "WHERE jobs.status NOT IN ('succeeded', 'failed', 'cancelled') "
                "ORDER BY jobs.updated_at, jobs.created_at, jobs.id LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._job_row(row) for row in rows]

    def touch_active_timeline_job(
        self, job_id: str, *, expected_updated_at: str
    ) -> bool:
        """Move one attempted active parent behind its reconciliation peers.

        A malformed or otherwise exceptional row must not permanently occupy
        one of the reconciler's bounded slots.  This write changes no lifecycle
        state; it only advances the ordering version after an attempted pass.
        """

        with self.connect() as db:
            cursor = db.execute(
                "UPDATE jobs SET updated_at = ? WHERE id = ? "
                "AND updated_at = ? "
                "AND status NOT IN ('succeeded', 'failed', 'cancelled')",
                (utc_now(), job_id, expected_updated_at),
            )
        return cursor.rowcount == 1

    def list_active_job_settings(self) -> list[dict[str, Any]]:
        """Return endpoint snapshots that may still emit native progress.

        This is used only during process startup to reconnect websocket
        monitors after a backend restart.  The immutable job snapshot, rather
        than the current global setting, remains authoritative when the user
        changed ComfyUI endpoints while a historical job was active.
        """

        with self.connect() as db:
            rows = db.execute(
                "SELECT settings_snapshot FROM jobs "
                "WHERE status NOT IN ('succeeded', 'failed', 'cancelled')"
            ).fetchall()
        return [json.loads(row["settings_snapshot"]) for row in rows]

    def list_interrupted_preparing_jobs(
        self, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Return only parents that explicitly handed work to recovery.

        The one-time startup transaction below may treat ``preparing`` and
        process-local child markers as leftovers because no coroutine from the
        previous process can still own them.  This query is also used by the
        long-running worker after startup, where those shapes may belong to a
        live preflight or ``POST /prompt``.  Requiring a parent recovery stage
        prevents that worker from cancelling an active current-process submit.
        """

        if limit is not None and limit < 1:
            raise ValueError("interrupted submission limit must be positive")
        limit_sql = " LIMIT ?" if limit is not None else ""
        parameters: tuple[int, ...] = (limit,) if limit is not None else ()
        with self.connect() as db:
            rows = db.execute(
                "SELECT jobs.* FROM jobs "
                "WHERE jobs.status = 'cancelling' AND jobs.stage IN "
                "('submission_interrupted', 'submission_cancel_pending', "
                " 'submission_cancel_failed', 'submission_cancel_unconfirmed', "
                " 'restart_cancel_pending', 'restart_cancel_unconfirmed', "
                " 'restart_cancel_failed') "
                "ORDER BY jobs.updated_at, jobs.created_at, jobs.id" + limit_sql,
                parameters,
            ).fetchall()
        return [self._job_row(row) for row in rows]

    def prepare_interrupted_submissions_for_recovery(self) -> int:
        """Atomically replace dead process ownership with restart ownership.

        This method performs SQLite writes only.  It is safe to call before an
        ASGI application becomes available: no queue, history, cancellation,
        websocket, or other ComfyUI operation is reachable from here.  A
        bound caller-assigned prompt id retains a restart marker until the
        managed recovery worker has attempted the exact upstream cancel.
        """

        now = utc_now()
        with self.connect() as db:
            rows = db.execute(
                "SELECT jobs.id FROM jobs WHERE "
                "(jobs.status NOT IN ('succeeded', 'failed', 'cancelled') "
                " AND jobs.cancel_requested = 1) OR "
                "jobs.status = 'preparing' OR "
                "(jobs.status = 'cancelling' AND jobs.stage IN "
                "('submission_interrupted', 'submission_cancel_pending', "
                "'submission_cancel_failed', 'submission_cancel_unconfirmed', "
                "'restart_cancel_pending', 'restart_cancel_unconfirmed', "
                "'restart_cancel_failed')) OR "
                "(jobs.status NOT IN ('succeeded', 'failed', 'cancelled') AND "
                " EXISTS (SELECT 1 FROM job_children "
                "         WHERE job_children.job_id = jobs.id "
                "         AND job_children.status NOT IN "
                "             ('succeeded', 'failed', 'cancelled') "
                "         AND job_children.stage IN "
                "             ('submitting', 'cancelling_during_submit', "
                "              'submission_cancel_pending', "
                "              'submission_cancel_failed', "
                "              'submission_cancel_unconfirmed', "
                "              'restart_cancel_pending', "
                "              'restart_cancel_unconfirmed', "
                "              'restart_cancel_failed')))"
            ).fetchall()
            for row in rows:
                job_id = str(row["id"])
                children = db.execute(
                    "SELECT id, status, prompt_id FROM job_children "
                    "WHERE job_id = ? ORDER BY group_index",
                    (job_id,),
                ).fetchall()
                if not children:
                    db.execute(
                        "UPDATE jobs SET status = 'cancelled', progress = 1.0, "
                        "stage = 'restart_cancelled_empty_submission', "
                        "error = 'submission stopped before native children were persisted', "
                        "updated_at = ?, completed_at = ? "
                        "WHERE id = ? AND status NOT IN "
                        "('succeeded', 'failed', 'cancelled')",
                        (now, now, job_id),
                    )
                    continue
                for child in children:
                    if child["status"] in {"succeeded", "failed", "cancelled"}:
                        continue
                    if child["prompt_id"]:
                        db.execute(
                            "UPDATE job_children SET status = 'cancelling', "
                            "stage = 'restart_cancel_pending', error = NULL, "
                            "updated_at = ?, completed_at = NULL WHERE id = ?",
                            (now, child["id"]),
                        )
                    else:
                        db.execute(
                            "UPDATE job_children SET status = 'cancelled', "
                            "progress = 1.0, stage = 'restart_cancelled_not_submitted', "
                            "error = NULL, updated_at = ?, completed_at = ? "
                            "WHERE id = ?",
                            (now, now, child["id"]),
                        )
                db.execute(
                    "UPDATE jobs SET status = 'cancelling', cancel_requested = 1, "
                    "stage = 'restart_cancel_pending', updated_at = ?, "
                    "completed_at = NULL WHERE id = ? AND status NOT IN "
                    "('succeeded', 'failed', 'cancelled')",
                    (now, job_id),
                )
        return len(rows)

    def recover_interrupted_assemblies(self) -> int:
        """Make process-local ffmpeg claims retryable after a restart.

        An ``assembling`` row is owned by a coroutine in the previous backend
        process.  No such coroutine survives startup, so retaining the claim
        would strand the parent forever.  Retrying may leave an unreferenced
        remote file if the old process died after upload but before its final
        CAS; the deterministic parent result remains correct and visible.
        """

        now = utc_now()
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE jobs SET stage = 'assembly_retry', updated_at = ? "
                "WHERE status = 'running' AND stage = 'assembling'",
                (now,),
            )
            return cursor.rowcount

    def _assert_job_prompt_ownership_released_in_connection(
        self,
        db: sqlite3.Connection,
        job_id: str,
    ) -> None:
        """Prove a typed terminal job no longer owns an upstream prompt.

        Deletion is the last local recovery boundary.  A terminal projection is
        insufficient: every submitted typed child must retain its exact prompt
        evidence and a positive terminal/cancel/restart certificate until the
        parent row is atomically removed.  Pre-created children with no prompt
        id never crossed the network and require no ownership row.
        """

        if not self._job_has_typed_execution_marker_in_connection(db, job_id):
            return
        evidence_schema = self._execution_evidence_schema_exists(db)
        if not evidence_schema:
            raise ExecutionEvidenceConflict(
                f"typed job {job_id} lost its execution evidence schema"
            )
        plan_row = db.execute(
            "SELECT schema_version, compiled_plan, compiled_plan_digest "
            "FROM job_execution_plans WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if plan_row is None:
            raise ExecutionEvidenceConflict(
                f"typed job {job_id} lost its compiled execution plan"
            )
        try:
            plan_digest = compiled_execution_plan_digest_from_canonical_json(
                plan_row["compiled_plan"]
            ).value
            plan = CompiledExecutionPlan.model_validate_json(
                plan_row["compiled_plan"]
            )
        except (TypeError, ValueError, ValidationError, json.JSONDecodeError) as exc:
            raise ExecutionEvidenceConflict(
                f"typed job {job_id} compiled execution plan is invalid"
            ) from exc
        if (
            plan_digest != plan_row["compiled_plan_digest"]
            or plan.version != plan_row["schema_version"]
        ):
            raise ExecutionEvidenceConflict(
                f"typed job {job_id} compiled execution plan is invalid"
            )
        rows = db.execute(
            "SELECT * FROM job_children WHERE job_id = ? ORDER BY group_index",
            (job_id,),
        ).fetchall()
        for row in rows:
            child = self._job_child_row(row)
            child_id = str(child["id"])
            if child["status"] not in self._TERMINAL_JOB_STATUSES:
                raise ExecutionEvidenceConflict(
                    f"typed job {job_id} retains nonterminal child {child_id}"
                )
            evidence_row = (
                db.execute(
                    "SELECT * FROM job_child_execution_evidence WHERE child_id = ?",
                    (child_id,),
                ).fetchone()
                if evidence_schema
                else None
            )
            ownership_row = (
                db.execute(
                    "SELECT * FROM prompt_ownership WHERE child_id = ?",
                    (child_id,),
                ).fetchone()
                if evidence_schema
                else None
            )
            prompt_id = child.get("prompt_id")
            if not prompt_id:
                if evidence_row is not None or ownership_row is not None:
                    raise ExecutionEvidenceConflict(
                        f"typed child {child_id} has evidence without a prompt id"
                    )
                continue

            markerless_legacy_control = (
                not child.get("segment_ids")
                and not child.get("prompt_snapshot")
                and evidence_row is None
                and ownership_row is None
            )
            if markerless_legacy_control:
                # A pre-Stage-4 terminal RayKill row has no typed certificate.
                # It carries no user output and cannot authorize a new segment;
                # continuation validation independently requires full evidence.
                continue
            if evidence_row is None or ownership_row is None:
                raise ExecutionEvidenceConflict(
                    f"typed child {child_id} has incomplete prompt release evidence"
                )
            try:
                _locked, _snapshot, unit = self._execution_evidence_from_row(
                    evidence_row,
                    child_id=child_id,
                )
                ownership = self._prompt_ownership_row(ownership_row)
            except (
                KeyError,
                TypeError,
                ValueError,
                ValidationError,
                json.JSONDecodeError,
            ) as exc:
                raise ExecutionEvidenceConflict(
                    f"typed child {child_id} prompt release evidence is invalid"
                ) from exc
            if (
                ownership.requested_prompt_id != unit.requested_prompt_id
                or ownership.effective_prompt_id != str(prompt_id)
                or ownership.state
                not in {"cleanup_confirmed", "terminal_confirmed"}
            ):
                raise ExecutionEvidenceConflict(
                    f"typed child {child_id} still owns its ComfyUI prompt"
                )

    def delete_job_if_status(self, job_id: str, expected_status: str) -> bool:
        """Delete one job only when its lifecycle state has not changed.

        The status predicate is the deletion serialization point.  It prevents
        a stale browser view from removing a job that another request has
        transitioned back into an active cancellation/submission workflow.
        ``False`` means the row still exists but no longer has the expected
        status; a missing row is reported separately as ``KeyError``.
        """

        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["status"] != expected_status:
                return False
            self._assert_job_prompt_ownership_released_in_connection(db, job_id)
            cursor = db.execute(
                "DELETE FROM jobs WHERE id = ? AND status = ?",
                (job_id, expected_status),
            )
            if cursor.rowcount == 1:
                return True
            return False

    def delete_terminal_jobs(
        self,
        terminal_statuses: tuple[str, ...],
        *,
        excluded_job_ids: set[str] | None = None,
    ) -> tuple[int, int]:
        """Atomically clear terminal local rows while retaining active jobs."""

        if not terminal_statuses:
            raise ValueError("at least one terminal status is required")
        placeholders = ",".join("?" for _ in terminal_statuses)
        excluded = sorted(excluded_job_ids or set())
        exclusion_clause = ""
        parameters: tuple[Any, ...] = terminal_statuses
        if excluded:
            exclusion_clause = (
                " AND id NOT IN (" + ",".join("?" for _ in excluded) + ")"
            )
            parameters = (*terminal_statuses, *excluded)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            candidates = db.execute(
                f"SELECT id FROM jobs WHERE status IN ({placeholders})"
                f"{exclusion_clause} ORDER BY id",
                parameters,
            ).fetchall()
            deleted_count = 0
            for candidate in candidates:
                job_id = str(candidate["id"])
                try:
                    self._assert_job_prompt_ownership_released_in_connection(
                        db, job_id
                    )
                except ExecutionEvidenceConflict:
                    # Bulk clear is best-effort: retain recovery-owned terminal
                    # jobs and report them through the remaining active count.
                    continue
                deleted_count += db.execute(
                    f"DELETE FROM jobs WHERE id = ? "
                    f"AND status IN ({placeholders})",
                    (job_id, *terminal_statuses),
                ).rowcount
            active_count = int(
                db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            )
        return deleted_count, active_count

    def update_job(self, job_id: str, **updates: Any) -> dict[str, Any]:
        updates = self._serialize_job_updates(updates)
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self.connect() as db:
            cursor = db.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ?",
                (*updates.values(), job_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(job_id)
        return self.get_job(job_id)  # type: ignore[return-value]

    def update_job_if_status(
        self,
        job_id: str,
        expected_status: str,
        **updates: Any,
    ) -> dict[str, Any] | None:
        """Compare-and-set a job row, returning ``None`` when status changed.

        This is the submission/cancellation serialization point: a cancelled
        ``preparing`` job cannot later be unconditionally revived as queued.
        """

        updates = self._serialize_job_updates(updates)
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self.connect() as db:
            cursor = db.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ? AND status = ?",
                (*updates.values(), job_id, expected_status),
            )
            if cursor.rowcount == 0:
                exists = db.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if exists is None:
                    raise KeyError(job_id)
                return None
        return self.get_job(job_id)

    def update_job_progress_monotonic(
        self,
        job_id: str,
        expected_status: str,
        progress: float,
        *,
        stage: str,
        started_at: str,
        expected_updated_at: str,
    ) -> dict[str, Any] | None:
        """Advance running progress without allowing a concurrent stale poll to rewind it."""

        now = utc_now()
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE jobs SET status = 'running', progress = ?, stage = ?, "
                "started_at = COALESCE(started_at, ?), updated_at = ? "
                "WHERE id = ? AND status = ? AND progress <= ? AND updated_at = ?",
                (
                    progress,
                    stage,
                    started_at,
                    now,
                    job_id,
                    expected_status,
                    progress,
                    expected_updated_at,
                ),
            )
            if cursor.rowcount == 0:
                exists = db.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if exists is None:
                    raise KeyError(job_id)
                return None
        return self.get_job(job_id)

    def claim_job_stage(
        self,
        job_id: str,
        *,
        expected_status: str,
        expected_updated_at: str,
        status: str,
        stage: str,
    ) -> dict[str, Any] | None:
        """Acquire a one-writer post-processing stage using the row version."""

        now = utc_now()
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE jobs SET status = ?, stage = ?, updated_at = ? "
                "WHERE id = ? AND status = ? AND updated_at = ?",
                (
                    status,
                    stage,
                    now,
                    job_id,
                    expected_status,
                    expected_updated_at,
                ),
            )
            if cursor.rowcount == 0:
                exists = db.execute(
                    "SELECT 1 FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
                if exists is None:
                    raise KeyError(job_id)
                return None
        return self.get_job(job_id)

    def update_job_if_snapshot(
        self,
        job_id: str,
        *,
        expected_status: str,
        expected_stage: str | None,
        expected_updated_at: str,
        **updates: Any,
    ) -> dict[str, Any] | None:
        """Commit post-processing only if cancellation did not win meanwhile."""

        updates = self._serialize_job_updates(updates)
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self.connect() as db:
            cursor = db.execute(
                f"UPDATE jobs SET {assignments} "
                "WHERE id = ? AND status = ? AND stage IS ? AND updated_at = ?",
                (
                    *updates.values(),
                    job_id,
                    expected_status,
                    expected_stage,
                    expected_updated_at,
                ),
            )
            if cursor.rowcount == 0:
                exists = db.execute(
                    "SELECT 1 FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
                if exists is None:
                    raise KeyError(job_id)
                return None
        return self.get_job(job_id)

    @staticmethod
    def _serialize_job_updates(updates: dict[str, Any]) -> dict[str, Any]:
        updates = dict(updates)
        allowed = {
            "status", "cancel_requested", "progress", "stage", "prompt_id", "outputs", "error", "prompt_snapshot",
            "updated_at", "started_at", "completed_at",
        }
        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(f"unsupported job columns: {sorted(unknown)}")
        updates.setdefault("updated_at", utc_now())
        if "outputs" in updates:
            updates["outputs"] = json.dumps(updates["outputs"], ensure_ascii=False)
        if "prompt_snapshot" in updates and updates["prompt_snapshot"] is not None:
            updates["prompt_snapshot"] = json.dumps(updates["prompt_snapshot"], ensure_ascii=False)
        return updates

    @staticmethod
    def _serialize_job_child_updates(updates: dict[str, Any]) -> dict[str, Any]:
        updates = dict(updates)
        allowed = {
            "status",
            "progress",
            "stage",
            "prompt_id",
            "prompt_snapshot",
            "outputs",
            "error",
            "updated_at",
            "started_at",
            "completed_at",
        }
        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(f"unsupported job child columns: {sorted(unknown)}")
        updates.setdefault("updated_at", utc_now())
        if "outputs" in updates:
            updates["outputs"] = json.dumps(updates["outputs"], ensure_ascii=False)
        if "prompt_snapshot" in updates:
            updates["prompt_snapshot"] = json.dumps(
                updates["prompt_snapshot"], ensure_ascii=False
            )
        return updates

    @staticmethod
    def _job_row(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value.pop(Database._TYPED_EXECUTION_MARKER_COLUMN, None)
        for field in ("outputs", "config_snapshot", "settings_snapshot", "prompt_snapshot"):
            if value.get(field) is not None:
                value[field] = json.loads(value[field])
        return value

    @staticmethod
    def _job_child_row(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        for field in ("segment_ids", "output_nodes", "outputs", "prompt_snapshot"):
            value[field] = json.loads(value[field])
        return value

    @staticmethod
    def _job_child_list_row(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        for field in ("segment_ids", "output_nodes", "outputs"):
            value[field] = json.loads(value[field])
        return value
