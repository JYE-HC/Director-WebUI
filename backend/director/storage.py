from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, TypeAlias

from .instance_lock import DirectorInstanceLock, DirectorInstanceLockError


StorageSource: TypeAlias = Literal[
    "explicit",
    "environment",
    "bootstrap",
    "legacy",
    "default",
]

_BOOTSTRAP_VERSION = 1
_CORE_DIRECTOR_TABLES = {
    "settings": {"singleton", "document", "updated_at"},
    "mode_drafts": {"mode", "document", "updated_at"},
    "assets": {"id", "document", "created_at"},
    "jobs": {"id", "status", "config_snapshot", "settings_snapshot"},
}


class StorageConfigurationError(RuntimeError):
    """The external bootstrap configuration is missing or malformed."""


class StoragePathError(ValueError):
    """A caller supplied an unsafe or unsupported database path."""


class StorageValidationError(ValueError):
    """A selected file is not a usable Director SQLite database."""


class StorageConflictError(RuntimeError):
    """A safe storage operation cannot proceed in the current state."""


class StorageOperationError(RuntimeError):
    """A filesystem or SQLite operation failed without a safe partial result."""


@dataclass(frozen=True)
class StorageSelection:
    database_path: Path
    source: StorageSource


@dataclass(frozen=True)
class StorageStatus:
    active_database_path: Path
    configured_database_path: Path
    recommended_database_path: Path
    source: StorageSource

    @property
    def active_database_identity(self) -> str:
        return database_identity(self.active_database_path)

    @property
    def restart_required(self) -> bool:
        return self.active_database_path != self.configured_database_path


def _canonical_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def database_identity(path: str | Path) -> str:
    """Return a stable browser epoch that changes when a DB file is replaced."""

    canonical = _canonical_path(path)
    try:
        metadata = canonical.stat()
    except FileNotFoundError:
        payload = f"missing\0{canonical}"
    except OSError:
        payload = f"unavailable\0{canonical}"
    else:
        # Device + inode identifies the existing file authority. Hard-link
        # aliases therefore share an epoch, while an atomic replacement at the
        # same path invalidates browsers that loaded the previous database.
        payload = f"existing\0{metadata.st_dev}\0{metadata.st_ino}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _same_existing_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _database_related_paths(path: Path) -> tuple[Path, ...]:
    """Files SQLite or Director may own for one main database path."""

    return (
        path,
        Path(f"{path}.instance.lock"),
        Path(f"{path}-wal"),
        Path(f"{path}-shm"),
        Path(f"{path}-journal"),
    )


def _has_nonempty_recovery_sidecar(path: Path) -> bool:
    for suffix in ("-wal", "-journal"):
        try:
            if Path(f"{path}{suffix}").lstat().st_size > 0:
                return True
        except FileNotFoundError:
            continue
        except OSError:
            # An unreadable recovery artifact is not evidence that a fresh
            # empty workspace is safe.
            return True
    return False


def _is_incomplete_fresh_default(path: Path) -> bool:
    """Recognize only our safe-create crash marker, never a possible WAL DB."""

    try:
        if path.stat().st_size != 0:
            return False
        return not _has_nonempty_recovery_sidecar(path)
    except OSError:
        return False


def _requested_absolute_path(value: str | Path) -> Path:
    supplied = str(value)
    if (
        not supplied
        or len(supplied) > 4096
        or any(
            ord(character) < 32 or ord(character) == 127
            for character in supplied
        )
    ):
        raise StoragePathError("database path is invalid")
    raw = supplied.strip()
    if not raw:
        raise StoragePathError("database path is invalid")
    try:
        expanded = Path(raw).expanduser()
        if not expanded.is_absolute():
            raise StoragePathError(
                "database path must be absolute (a leading ~ is supported)"
            )
        try:
            expanded.stat()
        except FileNotFoundError:
            # A missing final path (or parent) is legitimate for migration.
            pass
        return expanded.resolve(strict=False)
    except StoragePathError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise StoragePathError("database path is invalid") from exc


def _sqlite_read_only_uri(path: Path) -> str:
    return f"{path.as_uri()}?mode=ro"


def _read_bootstrap(config_path: Path) -> Path | None:
    if not config_path.exists():
        return None
    if not config_path.is_file():
        raise StorageConfigurationError("storage bootstrap configuration is invalid")
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StorageConfigurationError(
            "storage bootstrap configuration is invalid"
        ) from exc
    if (
        not isinstance(document, dict)
        or set(document) != {"version", "database_path"}
        or document.get("version") != _BOOTSTRAP_VERSION
        or not isinstance(document.get("database_path"), str)
    ):
        raise StorageConfigurationError("storage bootstrap configuration is invalid")
    try:
        database_path = _requested_absolute_path(document["database_path"])
    except StoragePathError as exc:
        raise StorageConfigurationError(
            "storage bootstrap configuration is invalid"
        ) from exc
    if not database_path.is_file():
        # Bootstrap paths are written only after selecting an existing Director
        # database or publishing a complete backup. Silently creating a new file
        # here would turn a missing mount/file into an apparently empty workspace.
        raise StorageConfigurationError(
            "configured Director database is unavailable"
        )
    return database_path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_temporary_database(path: Path) -> None:
    """Best-effort cleanup for one exact mkstemp SQLite authority.

    SQLite may materialize WAL/SHM or rollback-journal sidecars while the
    copied database is validated. Remove only names derived from the private
    random temporary main path; the separately published target hard link is
    deliberately outside this set.
    """

    for suffix in ("-wal", "-shm", "-journal", ""):
        try:
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        except OSError:
            # Cleanup must not turn a completed, consistently-published
            # migration into an API failure or mask its original exception.
            pass


def _write_bootstrap(config_path: Path, database_path: Path) -> None:
    temporary_path: Path | None = None
    descriptor: int | None = None
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{config_path.name}.",
            suffix=".tmp",
            dir=config_path.parent,
        )
        temporary_path = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        payload = json.dumps(
            {
                "version": _BOOTSTRAP_VERSION,
                "database_path": str(database_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        ) + "\n"
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, config_path)
        temporary_path = None
        # os.replace is the publication point. A later directory-fsync failure
        # must not be reported as if the old bootstrap were still authoritative:
        # migration cleanup could then remove the already-configured target and
        # leave a bootstrap path dangling. The file contents were fsynced before
        # replace; directory fsync remains a best-effort durability enhancement.
        try:
            _fsync_directory(config_path.parent)
        except OSError:
            pass
    except OSError as exc:
        raise StorageOperationError("storage configuration could not be saved") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def validate_director_database(path: Path) -> None:
    if not path.is_file():
        raise StorageValidationError("selected database does not exist")
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            _sqlite_read_only_uri(path),
            uri=True,
            timeout=5,
        )
        connection.execute("PRAGMA query_only = ON")
        check_rows = connection.execute("PRAGMA quick_check").fetchall()
        if check_rows != [("ok",)]:
            raise StorageValidationError(
                "selected file is not a valid Director database"
            )
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if not set(_CORE_DIRECTOR_TABLES).issubset(tables):
            raise StorageValidationError(
                "selected file is not a valid Director database"
            )
        for table, required_columns in _CORE_DIRECTOR_TABLES.items():
            columns = {
                str(row[1])
                for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if not required_columns.issubset(columns):
                raise StorageValidationError(
                    "selected file is not a valid Director database"
                )
        settings_row = connection.execute(
            "SELECT document FROM settings WHERE singleton = 1"
        ).fetchone()
        if settings_row is None or not isinstance(json.loads(settings_row[0]), dict):
            raise StorageValidationError(
                "selected file is not a valid Director database"
            )
    except StorageValidationError:
        raise
    except (sqlite3.Error, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise StorageValidationError(
            "selected file is not a valid Director database"
        ) from exc
    finally:
        if connection is not None:
            connection.close()


class StorageController:
    """Resolve, report, configure, and safely copy Director database storage."""

    def __init__(
        self,
        *,
        selection: StorageSelection,
        config_path: Path,
        recommended_database_path: Path,
    ) -> None:
        self.selection = selection
        self.config_path = config_path
        self.recommended_database_path = recommended_database_path
        self._validate_bootstrap_does_not_collide_with_databases(
            selection.database_path
        )

    @classmethod
    def resolve(
        cls,
        explicit_database_path: str | Path | None = None,
        *,
        storage_config_path: str | Path | None = None,
        legacy_database_path: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
        project_root: str | Path | None = None,
    ) -> "StorageController":
        environment = os.environ if environ is None else environ
        root_path = _canonical_path(
            Path(__file__).resolve().parents[2]
            if project_root is None
            else project_root
        )
        recommended_database_path = (
            root_path / ".data" / "database" / "director.sqlite3"
        ).resolve(strict=False)
        configured_config_path = storage_config_path or environment.get(
            "DIRECTOR_STORAGE_CONFIG_PATH"
        )
        config_path = _canonical_path(
            configured_config_path
            if configured_config_path
            else root_path / ".data" / "database" / "storage.json"
        )

        if explicit_database_path is not None:
            selection = StorageSelection(
                _canonical_path(explicit_database_path), "explicit"
            )
            return cls(
                selection=selection,
                config_path=config_path,
                recommended_database_path=recommended_database_path,
            )

        environment_database_path = environment.get("DIRECTOR_DATABASE_PATH")
        if environment_database_path:
            selection = StorageSelection(
                _canonical_path(environment_database_path), "environment"
            )
            return cls(
                selection=selection,
                config_path=config_path,
                recommended_database_path=recommended_database_path,
            )

        bootstrap_path = _read_bootstrap(config_path)
        if bootstrap_path is not None:
            validate_director_database(bootstrap_path)
            return cls(
                selection=StorageSelection(bootstrap_path, "bootstrap"),
                config_path=config_path,
                recommended_database_path=recommended_database_path,
            )

        default_path = recommended_database_path
        orphaned_recovery_state = False
        incomplete_fresh_default = False
        if default_path.exists():
            incomplete_fresh_default = _is_incomplete_fresh_default(default_path)
            if not incomplete_fresh_default:
                validate_director_database(default_path)
                return cls(
                    selection=StorageSelection(default_path, "default"),
                    config_path=config_path,
                    recommended_database_path=recommended_database_path,
                )
        else:
            orphaned_recovery_state = _has_nonempty_recovery_sidecar(default_path)

        legacy_path = _canonical_path(
            legacy_database_path
            if legacy_database_path is not None
            else root_path / "data" / "director.sqlite3"
        )
        if legacy_path.is_file():
            validate_director_database(legacy_path)
            return cls(
                selection=StorageSelection(legacy_path, "legacy"),
                config_path=config_path,
                recommended_database_path=recommended_database_path,
            )
        if orphaned_recovery_state:
            raise StorageValidationError(
                "default Director database recovery files exist without the main database"
            )
        # Database.initialize can safely finish a zero-byte file left between
        # O_EXCL creation and SQLite schema initialization. Legacy still wins
        # above, preserving first-upgrade compatibility.
        return cls(
            selection=StorageSelection(default_path, "default"),
            config_path=config_path,
            recommended_database_path=recommended_database_path,
        )

    @property
    def active_database_path(self) -> Path:
        return self.selection.database_path

    @property
    def active_database_identity(self) -> str:
        return database_identity(self.active_database_path)

    def _targets_active_database(self, target: Path) -> bool:
        return target == self.active_database_path or _same_existing_file(
            target, self.active_database_path
        )

    def targets_active_database(self, value: str | Path) -> bool:
        """Validate one requested path and compare it with this process's DB."""

        return self._targets_active_database(_requested_absolute_path(value))

    def _validate_bootstrap_does_not_collide_with_databases(
        self, *database_paths: Path
    ) -> None:
        related_paths = {
            candidate
            for database_path in database_paths
            for candidate in _database_related_paths(database_path)
        }
        if self.config_path in related_paths or any(
            _same_existing_file(self.config_path, candidate)
            for candidate in related_paths
        ):
            raise StorageConflictError(
                "database files conflict with the storage bootstrap configuration"
            )

    def status(self) -> StorageStatus:
        # An explicit constructor argument or environment override will win on
        # every equivalent restart. Do not claim a dormant bootstrap path is the
        # effective next database while that override remains active.
        configured = (
            None
            if self.selection.source in {"explicit", "environment"}
            else _read_bootstrap(self.config_path)
        )
        return StorageStatus(
            active_database_path=self.active_database_path,
            configured_database_path=configured or self.active_database_path,
            recommended_database_path=self.recommended_database_path,
            source=self.selection.source,
        )

    def configure_existing(self, value: str | Path) -> StorageStatus:
        if self.selection.source in {"explicit", "environment"}:
            raise StorageConflictError(
                "database path is controlled by an explicit startup override"
            )
        target = _requested_absolute_path(value)
        self._validate_bootstrap_does_not_collide_with_databases(
            self.active_database_path, target
        )
        if self._targets_active_database(target):
            # A hard-link alias is the same open database, not a restart
            # boundary. Persist the active spelling so status remains false.
            _write_bootstrap(self.config_path, self.active_database_path)
            return self.status()
        if not target.exists():
            raise StorageValidationError(
                "configured database must already exist; use migration to create it"
            )
        target_lock = DirectorInstanceLock(target)
        try:
            target_lock.acquire()
        except DirectorInstanceLockError as exc:
            raise StorageConflictError(
                "selected database is currently in use by another Director instance"
            ) from exc
        try:
            validate_director_database(target)
            _write_bootstrap(self.config_path, target)
        finally:
            target_lock.release()
        return self.status()

    def migrate(self, value: str | Path) -> tuple[StorageStatus, Path, Path]:
        if self.selection.source in {"explicit", "environment"}:
            raise StorageConflictError(
                "database path is controlled by an explicit startup override"
            )
        source = self.active_database_path
        target = _requested_absolute_path(value)
        self._validate_bootstrap_does_not_collide_with_databases(source, target)
        if target == source:
            raise StorageConflictError(
                "migration target must differ from the active database"
            )
        if target.exists() or target.is_symlink():
            raise StorageConflictError("migration target already exists")

        try:
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError as exc:
            raise StorageOperationError("migration target cannot be created") from exc

        target_lock = DirectorInstanceLock(target)
        try:
            target_lock.acquire()
        except DirectorInstanceLockError as exc:
            raise StorageConflictError(
                "migration target is currently in use by another Director instance"
            ) from exc

        temporary_path: Path | None = None
        published = False
        lock_connection: sqlite3.Connection | None = None
        source_connection: sqlite3.Connection | None = None
        destination_connection: sqlite3.Connection | None = None
        try:
            if target.exists() or target.is_symlink():
                raise StorageConflictError("migration target already exists")
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".migrating",
                dir=target.parent,
            )
            os.close(descriptor)
            temporary_path = Path(temporary_name)

            # BEGIN IMMEDIATE lives on a separate connection: it prevents every
            # other SQLite writer while the read-only backup source holds a
            # stable snapshot. Calling backup() on the transaction owner itself
            # deadlocks in Python's sqlite3 wrapper.
            lock_connection = sqlite3.connect(source, timeout=30)
            lock_connection.execute("BEGIN IMMEDIATE")
            source_connection = sqlite3.connect(
                _sqlite_read_only_uri(source), uri=True, timeout=30
            )
            destination_connection = sqlite3.connect(temporary_path, timeout=30)
            source_connection.backup(destination_connection)
            destination_connection.commit()
            destination_connection.close()
            destination_connection = None
            source_connection.close()
            source_connection = None

            validate_director_database(temporary_path)
            with temporary_path.open("rb") as stream:
                os.fsync(stream.fileno())
            # A same-directory hard link publishes without ever replacing a file
            # another process may have created after the preflight check.
            os.link(temporary_path, target)
            published = True
            _fsync_directory(target.parent)
            try:
                _write_bootstrap(self.config_path, target)
            except BaseException:
                try:
                    bootstrap_published = (
                        _read_bootstrap(self.config_path) == target
                    )
                except StorageConfigurationError:
                    bootstrap_published = False
                if not bootstrap_published:
                    target.unlink(missing_ok=True)
                    published = False
                    _fsync_directory(target.parent)
                raise
            return self.status(), source, target
        except (StorageConflictError, StorageValidationError, StorageOperationError):
            raise
        except (OSError, sqlite3.Error) as exc:
            if published:
                try:
                    target.unlink(missing_ok=True)
                    _fsync_directory(target.parent)
                except OSError:
                    pass
            raise StorageOperationError("database migration failed") from exc
        finally:
            if destination_connection is not None:
                destination_connection.close()
            if source_connection is not None:
                source_connection.close()
            if lock_connection is not None:
                try:
                    lock_connection.rollback()
                finally:
                    lock_connection.close()
            if temporary_path is not None:
                _cleanup_temporary_database(temporary_path)
            target_lock.release()
