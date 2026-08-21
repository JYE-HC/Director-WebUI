from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import uuid
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .compiler import timeline_segment_take_fingerprint
from .native_templates import (
    NativeTemplateError,
    normalize_native_output_descriptor,
    raylight_runtime_logical_gpu_indices,
)
from .schemas import (
    AssetReference,
    GenerationMode,
    MAX_TIMELINE_REVISION,
    MODE_ORDER,
    ModeDraft,
    RuntimeSettings,
    UnifiedTimelineDraft,
    canonicalize_live_runtime_settings,
    default_draft,
    default_settings,
    default_timeline_draft,
    iter_draft_assets,
    iter_timeline_assets,
    migrate_mode_drafts_to_timeline,
    mode_draft_is_default,
    utc_now,
    validate_mode_draft,
    validate_timeline_draft,
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
        }
    )
    _CONFIRMED_COMFY_RESTART_STAGE = "cancelled_after_confirmed_comfy_restart"

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
            had_mode_drafts_table = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'mode_drafts'"
            ).fetchone() is not None
            had_unified_timeline_table = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'unified_timeline'"
            ).fetchone() is not None
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
            settings = default_settings()
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
            # ``clear_between_segments`` belonged to the former custom-node
            # executor. Native per-segment prompts rely on ComfyUI's stable
            # loader cache between prompts; there is no stock per-segment
            # unload contract, so upgrades normalize this obsolete value once.
            if raw_settings.get("memory_policy") == "clear_between_segments":
                raw_settings["memory_policy"] = "keep_resident"
            # The former persistent policies pinned the endpoint forever to a
            # single model family.  The keyed-switch policy keeps the same
            # CUDA residency benefit while allowing Director to explicitly
            # replace the Ray pool before another family/backend runs.
            if raw_settings.get("raylight_residency_policy") in {
                "dedicated_keep_fl2va",
                "dedicated_keep_ref2va",
            }:
                raw_settings["raylight_residency_policy"] = "keep_until_switch"
            # Native timeline v1 deliberately closes the RayLight FSDP path
            # until actor CUDA cleanup is verified. Older web builds exposed
            # these booleans, so normalize them before strict Literal[False]
            # validation rather than making an existing database unbootable.
            for family in ("fl2va", "ref2va"):
                binding = (raw_settings.get("models") or {}).get(family)
                raylight = binding.get("raylight") if isinstance(binding, dict) else None
                if isinstance(raylight, dict):
                    raylight["fsdp"] = False
                    raylight["cpu_offload"] = False
            try:
                persisted_settings = canonicalize_live_runtime_settings(
                    RuntimeSettings.model_validate(raw_settings)
                )
            except ValidationError as exc:
                # The plugin era deliberately keeps no compatibility reader for
                # pre-plugin documents (e.g. a stored ``comfy_url`` fails
                # extra=forbid). Fail with an actionable message instead of a
                # bare schema dump.
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
                else:
                    migrated = default_timeline_draft()
                db.execute(
                    "INSERT INTO unified_timeline(singleton, document, updated_at) VALUES(1, ?, ?)",
                    (migrated.model_dump_json(), now),
                )
            else:
                raw_timeline = json.loads(timeline_row["document"])
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
                normalized_timeline = validate_timeline_draft(
                    raw_timeline
                ).model_dump(mode="json")
                if json.loads(timeline_row["document"]) != normalized_timeline:
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
            timeline = validate_timeline_draft(timeline_document)
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
    def _segment_take_row(row: sqlite3.Row) -> dict[str, Any]:
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

    def get_settings_authority(self) -> tuple[RuntimeSettings, str]:
        with self.connect() as db:
            row = db.execute(
                "SELECT document, revision FROM settings WHERE singleton = 1"
            ).fetchone()
        if row is None:
            raise RuntimeError("settings row is missing")
        settings = canonicalize_live_runtime_settings(
            RuntimeSettings.model_validate_json(row["document"])
        )
        authority = hashlib.sha256(
            (
                str(int(row["revision"]))
                + "\0"
                + str(row["document"])
            ).encode("utf-8")
        ).hexdigest()
        return settings, authority

    def get_settings(self) -> RuntimeSettings:
        return self.get_settings_authority()[0]

    def put_settings(self, settings: RuntimeSettings) -> RuntimeSettings:
        settings = canonicalize_live_runtime_settings(settings)
        with self.connect() as db:
            db.execute(
                "UPDATE settings SET document = ?, updated_at = ?, "
                "revision = revision + 1 WHERE singleton = 1",
                (settings.model_dump_json(), utc_now()),
            )
        return settings

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
        result = {
            "version": 2,
            "epoch": epoch,
            "current": current,
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

        if not prompt_id:
            return False
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT descriptor FROM raylight_runtime_state "
                "WHERE singleton = 1"
            ).fetchone()
            if row is None:
                return False
            state = self._decode_raylight_runtime_state(row["descriptor"])
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
                    utc_now(),
                ),
            )
        return True

    def confirm_comfy_restart_recovery(self, job_id: str) -> dict[str, Any]:
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
            if current is None:
                raise ValueError(
                    "RayLight runtime was already cleared before recovery completed"
                )
            recorded = raylight_runtime_logical_gpu_indices(current)
            invalid = tuple(index for index in recorded if index >= visible_gpu_count)
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

    def get_timeline_authority(self) -> tuple[UnifiedTimelineDraft, int]:
        """Return the legacy/default project document and durable CAS revision."""

        with self.connect() as db:
            row = db.execute(
                "SELECT document, revision FROM unified_timeline WHERE singleton = 1"
            ).fetchone()
        if row is None:
            raise RuntimeError("unified timeline row is missing")
        return (
            validate_timeline_draft(json.loads(row["document"])),
            int(row["revision"]),
        )

    def get_timeline(self) -> UnifiedTimelineDraft:
        return self.get_timeline_authority()[0]

    def put_timeline(self, timeline: UnifiedTimelineDraft) -> UnifiedTimelineDraft:
        with self.connect() as db:
            db.execute(
                "UPDATE unified_timeline SET document = ?, updated_at = ?, "
                "revision = revision + 1 WHERE singleton = 1",
                (timeline.model_dump_json(), utc_now()),
            )
        return timeline

    def validate_and_put_timeline(
        self,
        timeline: UnifiedTimelineDraft,
    ) -> UnifiedTimelineDraft:
        """Validate assets and save the unified timeline under one write lock."""

        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._validate_asset_iterator_in_connection(
                db,
                iter_timeline_assets(timeline),
            )
            db.execute(
                "UPDATE unified_timeline SET document = ?, updated_at = ?, "
                "revision = revision + 1 WHERE singleton = 1",
                (timeline.model_dump_json(), utc_now()),
            )
        return timeline

    def validate_and_put_timeline_authority(
        self,
        timeline: UnifiedTimelineDraft,
        *,
        expected_revision: int,
    ) -> tuple[UnifiedTimelineDraft, int]:
        """CAS-replace the default timeline under the asset-validation lock."""

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
        timeline = validate_timeline_draft(json.loads(row["document"]))
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
        timeline = validate_timeline_draft(json.loads(row["document"]))
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
            timeline = validate_timeline_draft(legacy["document"])
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

    def create_project(self, title: str | None = None) -> dict[str, Any]:
        """Create a fresh project with a new stable segment identity."""

        project_id = str(uuid.uuid4())
        timeline = default_timeline_draft()
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
        self, title: str, timeline: UnifiedTimelineDraft
    ) -> dict[str, Any]:
        """Create a project from an existing validated timeline document.

        Segment identities are preserved so a restored historical project keeps
        its stable structure. The take ledger remains scoped by this new
        project id, so imported segments start without reused renders.
        """

        project_id = str(uuid.uuid4())
        normalized_title = (
            title.strip() or timeline.title.strip() or "未命名长视频"
        )
        now = utc_now()
        with self.connect() as db:
            db.execute(
                "INSERT INTO projects(id, title, document, created_at, updated_at) "
                "VALUES(?, ?, ?, ?, ?)",
                (project_id, normalized_title, timeline.model_dump_json(), now, now),
            )
        project = self.get_project(project_id)
        if project is None:
            raise RuntimeError("imported project disappeared")
        return project

    def rename_project(self, project_id: str, title: str) -> dict[str, Any]:
        normalized = title.strip()
        if not normalized:
            raise ValueError("project title must not be empty")
        if self._is_legacy_project_id(project_id):
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT document FROM unified_timeline WHERE singleton = 1"
                ).fetchone()
                if row is None:
                    raise KeyError(project_id)
                timeline = validate_timeline_draft(json.loads(row["document"]))
                timeline = timeline.model_copy(update={"title": normalized})
                db.execute(
                    "UPDATE unified_timeline SET document = ?, updated_at = ?, "
                    "revision = revision + 1 "
                    "WHERE singleton = 1",
                    (timeline.model_dump_json(), utc_now()),
                )
            project = self.get_project(project_id)
            if project is None:
                raise RuntimeError("renamed project disappeared")
            return project
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT document FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
            if row is None:
                raise KeyError(project_id)
            timeline = validate_timeline_draft(json.loads(row["document"]))
            timeline = timeline.model_copy(update={"title": normalized})
            db.execute(
                "UPDATE projects SET title = ?, document = ?, updated_at = ?, "
                "revision = revision + 1 WHERE id = ?",
                (normalized, timeline.model_dump_json(), utc_now(), project_id),
            )
        project = self.get_project(project_id)
        if project is None:
            raise RuntimeError("renamed project disappeared")
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

    def get_project_timeline(self, project_id: str) -> UnifiedTimelineDraft:
        # Keep the legacy/default read flowing through get_timeline(). Besides
        # preserving the established public delegation contract, job-list
        # snapshot comparison can continue to prefetch that authority once.
        if self._is_legacy_project_id(project_id):
            return self.get_timeline()
        return self.get_project_timeline_authority(project_id)[0]

    def get_project_timeline_authority(
        self, project_id: str
    ) -> tuple[UnifiedTimelineDraft, int]:
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
            validate_timeline_draft(json.loads(row["document"])),
            int(row["revision"]),
        )

    def put_project_timeline(
        self, project_id: str, timeline: UnifiedTimelineDraft
    ) -> UnifiedTimelineDraft:
        if self._is_legacy_project_id(project_id):
            return self.put_timeline(timeline)
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE projects SET document = ?, title = ?, updated_at = ?, "
                "revision = revision + 1 WHERE id = ?",
                (timeline.model_dump_json(), timeline.title, utc_now(), project_id),
            )
        if cursor.rowcount != 1:
            raise KeyError(project_id)
        return timeline

    def validate_and_put_project_timeline(
        self,
        project_id: str,
        timeline: UnifiedTimelineDraft,
    ) -> UnifiedTimelineDraft:
        """Validate assets and save one project's timeline under one write lock."""

        if self._is_legacy_project_id(project_id):
            return self.validate_and_put_timeline(timeline)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if (
                db.execute(
                    "SELECT 1 FROM projects WHERE id = ?", (project_id,)
                ).fetchone()
                is None
            ):
                raise KeyError(project_id)
            self._validate_asset_iterator_in_connection(
                db,
                iter_timeline_assets(timeline),
            )
            db.execute(
                "UPDATE projects SET document = ?, title = ?, updated_at = ?, "
                "revision = revision + 1 WHERE id = ?",
                (timeline.model_dump_json(), timeline.title, utc_now(), project_id),
            )
        return timeline

    def validate_and_put_project_timeline_authority(
        self,
        project_id: str,
        timeline: UnifiedTimelineDraft,
        *,
        expected_revision: int,
    ) -> tuple[UnifiedTimelineDraft, int]:
        """CAS-replace one project timeline under the asset-validation lock."""

        if self._is_legacy_project_id(project_id):
            return self.validate_and_put_timeline_authority(
                timeline,
                expected_revision=expected_revision,
            )
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
                    validated = validate_timeline_draft(
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
                        normalized = validate_timeline_draft(document)
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
                            restored = validate_timeline_draft(
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
        draft: UnifiedTimelineDraft,
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
                "SELECT jobs.id FROM jobs WHERE jobs.status = 'preparing' OR "
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
                    "UPDATE jobs SET status = 'cancelling', "
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

    def delete_job_if_status(self, job_id: str, expected_status: str) -> bool:
        """Delete one job only when its lifecycle state has not changed.

        The status predicate is the deletion serialization point.  It prevents
        a stale browser view from removing a job that another request has
        transitioned back into an active cancellation/submission workflow.
        ``False`` means the row still exists but no longer has the expected
        status; a missing row is reported separately as ``KeyError``.
        """

        with self.connect() as db:
            cursor = db.execute(
                "DELETE FROM jobs WHERE id = ? AND status = ?",
                (job_id, expected_status),
            )
            if cursor.rowcount == 1:
                return True
            exists = db.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if exists is None:
                raise KeyError(job_id)
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
            cursor = db.execute(
                f"DELETE FROM jobs WHERE status IN ({placeholders}){exclusion_clause}",
                parameters,
            )
            active_count = int(
                db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            )
        return cursor.rowcount, active_count

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
