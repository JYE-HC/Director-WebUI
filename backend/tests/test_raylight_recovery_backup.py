from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import director.database as database_module
from director.database import Database
from director.schemas import default_settings


_COMFY_ORIGIN = "http://comfy.test:8188"


def _stale_runtime_state() -> dict:
    return {
        "version": 2,
        "epoch": 7,
        "current": {
            "version": 2,
            "family": "fl2va",
            "compatibility_key": "director-g0-1-2-3",
            "runtime_key": "old-four-gpu-runtime",
            "runtime_namespace": "director-g0-1-2-3-e7",
            "initializer_node_id": "initializer",
            "loader_node_id": "loader",
            "loader_subgraph": {
                "initializer": {
                    "class_type": "RayInitializerAdvanced",
                    "inputs": {
                        "GPU": 4,
                        "GPU_SELECT": "0,1,2,3",
                    },
                },
                "loader": {
                    "class_type": "RayUNETLoader",
                    "inputs": {
                        "ray_actors_init": ["initializer", 0],
                    },
                },
            },
            "clear_vram_after_sampling": False,
        },
        "tail_prompt_id": "old-four-gpu-tail",
        "tail_action": "shutdown",
        "tainted": True,
    }


def _database_with_stale_runtime(tmp_path: Path) -> tuple[Database, str, dict]:
    database = Database(tmp_path / "director.sqlite3")
    database.initialize()
    database.put_settings(default_settings(_COMFY_ORIGIN))
    origin = database.canonical_comfy_origin(_COMFY_ORIGIN)
    database.put_raylight_runtime_state(origin, _stale_runtime_state())
    state = database.get_raylight_runtime_state(origin)
    assert state is not None
    return database, origin, state


def _recovery_artifacts(database: Database) -> list[Path]:
    return sorted(
        path
        for path in database.path.parent.iterdir()
        if "before-raylight-recovery" in path.name
    )


@pytest.mark.parametrize(
    "failure_point",
    ["backup", "quick_check", "file_fsync", "replace", "directory_fsync"],
)
def test_raylight_recovery_backup_stage_failure_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    database, origin, before = _database_with_stale_runtime(tmp_path)

    if failure_point == "backup":
        def fail_backup(*_args, **_kwargs) -> None:
            raise sqlite3.OperationalError("injected backup failure")

        monkeypatch.setattr(database_module, "_copy_sqlite_database", fail_backup)
    elif failure_point == "quick_check":
        monkeypatch.setattr(
            database_module,
            "_sqlite_quick_check_is_ok",
            lambda _connection: False,
        )
    elif failure_point == "file_fsync":
        def fail_file_fsync(_path: Path) -> None:
            raise OSError("injected file fsync failure")

        monkeypatch.setattr(database_module, "_fsync_file", fail_file_fsync)
    elif failure_point == "replace":
        def fail_replace(_source: Path, _destination: Path) -> None:
            raise OSError("injected atomic replace failure")

        monkeypatch.setattr(database_module, "_atomic_replace", fail_replace)
    else:
        def fail_directory_fsync(_path: Path) -> None:
            raise OSError("injected directory fsync failure")

        monkeypatch.setattr(
            database_module,
            "_fsync_directory",
            fail_directory_fsync,
        )

    with pytest.raises(
        RuntimeError,
        match="could not create the RayLight recovery backup",
    ):
        database.confirm_raylight_runtime_restart(
            origin,
            expected_epoch=7,
            expected_runtime_state=before,
            visible_gpu_count=2,
        )

    assert database.get_raylight_runtime_state(origin) == before
    assert _recovery_artifacts(database) == []
    with database.connect() as connection:
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"


def test_raylight_recovery_update_failure_retracts_published_backup(
    tmp_path: Path,
) -> None:
    database, origin, before = _database_with_stale_runtime(tmp_path)
    with database.connect() as connection:
        connection.execute(
            "CREATE TRIGGER reject_raylight_recovery_update "
            "BEFORE UPDATE ON raylight_runtime_state "
            "BEGIN SELECT RAISE(ABORT, 'injected runtime update failure'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="injected runtime update failure"):
        database.confirm_raylight_runtime_restart(
            origin,
            expected_epoch=7,
            expected_runtime_state=before,
            visible_gpu_count=2,
        )

    assert database.get_raylight_runtime_state(origin) == before
    assert _recovery_artifacts(database) == []


def test_raylight_recovery_keeps_backup_after_ambiguous_commit_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, origin, before = _database_with_stale_runtime(tmp_path)

    def commit_then_fail(connection: sqlite3.Connection) -> None:
        connection.commit()
        raise sqlite3.OperationalError("injected post-commit transport failure")

    monkeypatch.setattr(database_module, "_commit_database", commit_then_fail)
    with pytest.raises(
        sqlite3.OperationalError,
        match="injected post-commit transport failure",
    ):
        database.confirm_raylight_runtime_restart(
            origin,
            expected_epoch=7,
            expected_runtime_state=before,
            visible_gpu_count=2,
        )

    recovered = database.get_raylight_runtime_state(origin)
    assert recovered is not None
    assert recovered["current"] is None
    artifacts = _recovery_artifacts(database)
    assert len(artifacts) == 1
    assert Database(artifacts[0]).get_raylight_runtime_state(origin) == before


def test_raylight_recovery_publishes_only_one_valid_final_backup(
    tmp_path: Path,
) -> None:
    database, origin, before = _database_with_stale_runtime(tmp_path)

    recovered, backup_path = database.confirm_raylight_runtime_restart(
        origin,
        expected_epoch=7,
        expected_runtime_state=before,
        visible_gpu_count=2,
    )

    assert _recovery_artifacts(database) == [backup_path]
    assert backup_path.name.endswith(".sqlite3")
    assert backup_path.stat().st_mode & 0o777 == 0o600
    with sqlite3.connect(backup_path) as backup_connection:
        assert backup_connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    assert Database(backup_path).get_raylight_runtime_state(origin) == before
    assert database.get_raylight_runtime_state(origin) == recovered
