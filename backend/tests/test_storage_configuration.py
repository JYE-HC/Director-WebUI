from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import stat
from pathlib import Path

import httpx
import pytest

import director.storage as storage_module
from director.app import create_app
from director.database import Database
from director.schemas import default_settings
from director.storage import (
    StorageConflictError,
    StorageController,
    StoragePathError,
    StorageValidationError,
    database_identity,
    validate_director_database,
)


def _bootstrap(config_path: Path, database_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {"version": 1, "database_path": str(database_path.resolve())}
        ),
        encoding="utf-8",
    )


def _director_database(path: Path, *, title: str | None = None) -> Database:
    database = Database(path)
    database.initialize()
    if title is not None:
        timeline = database.get_timeline().model_copy(update={"title": title})
        database.put_timeline(timeline)
    return database


@pytest.fixture
def inline_storage_threads(monkeypatch):
    # The managed sandbox cannot wake asyncio's default worker threads. This is
    # the same endpoint-test adaptation used by the shared client fixture; the
    # storage copy itself remains a synchronous, independently tested operation.
    async def run_sync(function):
        return function()

    monkeypatch.setattr("director.app.anyio.to_thread.run_sync", run_sync)


def test_fresh_default_is_repo_data_database_and_does_not_depend_on_cwd(
    tmp_path,
) -> None:
    project_root = tmp_path / "checkout"
    unrelated_cwd = tmp_path / "somewhere" / "else"
    unrelated_cwd.mkdir(parents=True)
    previous_cwd = Path.cwd()
    try:
        os.chdir(unrelated_cwd)
        storage = StorageController.resolve(
            environ={},
            project_root=project_root,
            legacy_database_path=tmp_path / "missing-legacy.sqlite3",
        )
    finally:
        os.chdir(previous_cwd)

    assert storage.selection.source == "default"
    expected = (
        project_root / ".data" / "database" / "director.sqlite3"
    ).resolve()
    assert storage.active_database_path == expected
    assert storage.recommended_database_path == expected
    assert storage.config_path == (
        project_root / ".data" / "database" / "storage.json"
    ).resolve()


def test_repo_bootstrap_default_and_environment_override(tmp_path) -> None:
    project_root = tmp_path / "checkout"
    default_database = tmp_path / "default-selected.sqlite3"
    override_database = tmp_path / "override-selected.sqlite3"
    default_config = project_root / ".data" / "database" / "storage.json"
    override_config = tmp_path / "custom" / "storage.json"
    _director_database(default_database)
    _director_database(override_database)
    _bootstrap(default_config, default_database)
    _bootstrap(override_config, override_database)

    default_selection = StorageController.resolve(
        environ={}, project_root=project_root
    )
    assert default_selection.selection.source == "bootstrap"
    assert default_selection.active_database_path == default_database.resolve()
    assert default_selection.config_path == default_config.resolve()

    overridden = StorageController.resolve(
        environ={"DIRECTOR_STORAGE_CONFIG_PATH": str(override_config)},
        project_root=project_root,
    )
    assert overridden.selection.source == "bootstrap"
    assert overridden.active_database_path == override_database.resolve()
    assert overridden.config_path == override_config.resolve()
    assert overridden.recommended_database_path == (
        project_root / ".data" / "database" / "director.sqlite3"
    ).resolve()


def test_first_upgrade_prefers_valid_legacy_database_when_new_default_is_absent(
    tmp_path,
) -> None:
    project_root = tmp_path / "checkout"
    legacy = tmp_path / "checkout" / "data" / "director.sqlite3"
    _director_database(legacy)

    storage = StorageController.resolve(
        environ={}, project_root=project_root, legacy_database_path=legacy
    )

    assert storage.selection.source == "legacy"
    assert storage.active_database_path == legacy.resolve()


def test_existing_new_default_precedes_legacy_and_must_be_director_database(
    tmp_path,
) -> None:
    project_root = tmp_path / "checkout"
    default_path = project_root / ".data" / "database" / "director.sqlite3"
    legacy = tmp_path / "checkout" / "data" / "director.sqlite3"
    _director_database(default_path)
    _director_database(legacy)

    selected = StorageController.resolve(
        environ={}, project_root=project_root, legacy_database_path=legacy
    )
    assert selected.selection.source == "default"
    assert selected.active_database_path == default_path.resolve()

    default_path.unlink()
    default_path.write_bytes(b"not sqlite")
    with pytest.raises(StorageValidationError, match="valid Director"):
        StorageController.resolve(
            environ={}, project_root=project_root, legacy_database_path=legacy
        )


def test_zero_byte_default_crash_marker_recovers_without_shadowing_legacy(
    tmp_path,
) -> None:
    project_root = tmp_path / "checkout"
    default_path = project_root / ".data" / "database" / "director.sqlite3"
    default_path.parent.mkdir(parents=True)
    default_path.touch()
    legacy = tmp_path / "checkout" / "data" / "director.sqlite3"
    _director_database(legacy)

    with_legacy = StorageController.resolve(
        environ={}, project_root=project_root, legacy_database_path=legacy
    )
    assert with_legacy.selection.source == "legacy"

    without_legacy = StorageController.resolve(
        environ={},
        project_root=project_root,
        legacy_database_path=tmp_path / "missing.sqlite3",
    )
    assert without_legacy.selection.source == "default"
    assert without_legacy.active_database_path == default_path.resolve()
    Database(without_legacy.active_database_path).initialize()
    validate_director_database(default_path)


def test_zero_byte_default_with_nonempty_wal_fails_closed(tmp_path) -> None:
    project_root = tmp_path / "checkout"
    default_path = project_root / ".data" / "database" / "director.sqlite3"
    default_path.parent.mkdir(parents=True)
    default_path.touch()
    Path(f"{default_path}-wal").write_bytes(b"possible database state")

    with pytest.raises(StorageValidationError, match="valid Director"):
        StorageController.resolve(
            environ={},
            project_root=project_root,
            legacy_database_path=tmp_path / "missing.sqlite3",
        )


@pytest.mark.parametrize("sidecar_suffix", ["-wal", "-journal"])
def test_missing_default_with_nonempty_recovery_sidecar_prefers_legacy_or_fails_closed(
    tmp_path, sidecar_suffix
) -> None:
    project_root = tmp_path / "checkout"
    default_path = project_root / ".data" / "database" / "director.sqlite3"
    default_path.parent.mkdir(parents=True)
    Path(f"{default_path}{sidecar_suffix}").write_bytes(b"possible recovery state")
    legacy = tmp_path / "checkout" / "data" / "director.sqlite3"
    _director_database(legacy)

    selected = StorageController.resolve(
        environ={}, project_root=project_root, legacy_database_path=legacy
    )
    assert selected.selection.source == "legacy"

    with pytest.raises(StorageValidationError, match="recovery files"):
        StorageController.resolve(
            environ={},
            project_root=project_root,
            legacy_database_path=tmp_path / "missing.sqlite3",
        )


def test_explicit_then_environment_then_bootstrap_priority(tmp_path) -> None:
    project_root = tmp_path / "checkout"
    config = tmp_path / "storage.json"
    bootstrap_database = tmp_path / "bootstrap.sqlite3"
    environment_database = tmp_path / "environment.sqlite3"
    explicit_database = tmp_path / "explicit.sqlite3"
    _director_database(bootstrap_database)
    _bootstrap(config, bootstrap_database)

    bootstrap = StorageController.resolve(
        environ={},
        project_root=project_root,
        storage_config_path=config,
        legacy_database_path=tmp_path / "missing.sqlite3",
    )
    assert bootstrap.selection.source == "bootstrap"

    environment = StorageController.resolve(
        environ={"DIRECTOR_DATABASE_PATH": str(environment_database)},
        project_root=project_root,
        storage_config_path=config,
    )
    assert environment.selection.source == "environment"
    assert environment.active_database_path == environment_database.resolve()
    assert environment.status().configured_database_path == environment_database.resolve()

    explicit = StorageController.resolve(
        explicit_database,
        environ={"DIRECTOR_DATABASE_PATH": str(environment_database)},
        project_root=project_root,
        storage_config_path=config,
    )
    assert explicit.selection.source == "explicit"
    assert explicit.active_database_path == explicit_database.resolve()


def test_database_context_manager_closes_connection_and_repeated_reads_do_not_leak_fds(
    tmp_path,
) -> None:
    database = _director_database(tmp_path / "director.sqlite3")
    connection = database.connect()
    with connection as opened:
        assert opened.execute("SELECT 1").fetchone()[0] == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")

    descriptor_directory = Path("/proc/self/fd")
    if descriptor_directory.is_dir():
        before = len(list(descriptor_directory.iterdir()))
        for _ in range(100):
            database.get_timeline()
        after = len(list(descriptor_directory.iterdir()))
        assert after <= before + 2


def test_new_database_and_parent_are_private_without_changing_existing_mode(
    tmp_path,
) -> None:
    database_path = tmp_path / "private" / "director.sqlite3"
    previous_umask = os.umask(0o002)
    try:
        Database(database_path).initialize()
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(database_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(database_path.stat().st_mode) == 0o600

    os.chmod(database_path, 0o640)
    Database(database_path).initialize()
    assert stat.S_IMODE(database_path.stat().st_mode) == 0o640


def test_storage_paths_reject_control_characters_and_overlong_values(tmp_path) -> None:
    source = tmp_path / "source.sqlite3"
    _director_database(source)
    storage = StorageController.resolve(
        source,
        environ={},
        storage_config_path=tmp_path / "storage.json",
    )

    with pytest.raises(StoragePathError, match="invalid"):
        storage.targets_active_database(f"{source}\n")
    with pytest.raises(StoragePathError, match="invalid"):
        storage.targets_active_database("/" + ("a" * 4096))
    with pytest.raises(StoragePathError, match="invalid"):
        storage.targets_active_database(
            "~director_user_that_must_not_exist_8e51e4/director.sqlite3"
        )

    first_loop = tmp_path / "first-loop"
    second_loop = tmp_path / "second-loop"
    first_loop.symlink_to(second_loop)
    second_loop.symlink_to(first_loop)
    with pytest.raises(StoragePathError, match="invalid"):
        storage.targets_active_database(first_loop)


def test_storage_target_must_not_collide_with_bootstrap_or_database_sidecars(
    tmp_path,
) -> None:
    source = tmp_path / "legacy" / "director.sqlite3"
    _director_database(source)

    config_as_target = tmp_path / "config-as-database.sqlite3"
    storage = StorageController.resolve(
        environ={},
        project_root=tmp_path / "checkout-one",
        storage_config_path=config_as_target,
        legacy_database_path=source,
    )
    with pytest.raises(StorageConflictError, match="bootstrap"):
        storage.configure_existing(config_as_target)
    with pytest.raises(StorageConflictError, match="bootstrap"):
        storage.migrate(config_as_target)
    assert not config_as_target.exists()

    existing_target = tmp_path / "selected.sqlite3"
    _director_database(existing_target)
    for index, suffix in enumerate(
        (".instance.lock", "-wal", "-shm", "-journal")
    ):
        config_as_sidecar = Path(f"{existing_target}{suffix}")
        storage = StorageController.resolve(
            environ={},
            project_root=tmp_path / f"checkout-sidecar-{index}",
            storage_config_path=config_as_sidecar,
            legacy_database_path=source,
        )
        with pytest.raises(StorageConflictError, match="bootstrap"):
            storage.configure_existing(existing_target)
        with pytest.raises(StorageConflictError, match="bootstrap"):
            storage.migrate(existing_target)
        assert not config_as_sidecar.exists()


@pytest.mark.parametrize(
    "sidecar_suffix", [".instance.lock", "-wal", "-shm", "-journal"]
)
def test_bootstrap_must_not_be_an_active_database_sidecar(
    tmp_path, sidecar_suffix
) -> None:
    source = tmp_path / "legacy" / "director.sqlite3"
    _director_database(source)
    config_as_active_sidecar = Path(f"{source}{sidecar_suffix}")

    with pytest.raises(StorageConflictError, match="bootstrap"):
        StorageController.resolve(
            environ={},
            project_root=tmp_path / "checkout",
            storage_config_path=config_as_active_sidecar,
            legacy_database_path=source,
        )

    validate_director_database(source)


def test_bootstrap_hardlink_alias_of_active_sidecar_fails_before_mutation(
    tmp_path,
) -> None:
    source = tmp_path / "source.sqlite3"
    target = tmp_path / "target.sqlite3"
    config = tmp_path / "storage.json"
    _director_database(source)
    _director_database(target)
    _bootstrap(config, source)
    storage = StorageController.resolve(
        environ={},
        project_root=tmp_path / "checkout",
        storage_config_path=config,
    )
    active_sidecar_alias = Path(f"{source}-journal")
    assert not active_sidecar_alias.exists()
    os.link(config, active_sidecar_alias)
    original_config = config.read_bytes()

    with pytest.raises(StorageConflictError, match="bootstrap"):
        storage.configure_existing(target)
    with pytest.raises(StorageConflictError, match="bootstrap"):
        storage.migrate(tmp_path / "migrated.sqlite3")

    assert config.read_bytes() == original_config
    assert active_sidecar_alias.read_bytes() == original_config


def test_post_replace_directory_fsync_failure_never_leaves_dangling_bootstrap(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source" / "director.sqlite3"
    target = tmp_path / "target" / "director.sqlite3"
    config = tmp_path / "config" / "storage.json"
    _director_database(source, title="durable target")
    _bootstrap(config, source)
    storage = StorageController.resolve(
        environ={},
        storage_config_path=config,
        legacy_database_path=tmp_path / "missing.sqlite3",
    )
    real_fsync_directory = storage_module._fsync_directory

    def fail_config_directory_fsync(path: Path) -> None:
        if Path(path) == config.parent:
            raise OSError("injected config directory fsync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(
        storage_module,
        "_fsync_directory",
        fail_config_directory_fsync,
    )

    status, migrated_from, migrated_to = storage.migrate(target)

    assert migrated_from == source.resolve()
    assert migrated_to == target.resolve()
    assert status.configured_database_path == target.resolve()
    assert target.exists()
    validate_director_database(target)
    assert json.loads(config.read_text(encoding="utf-8"))["database_path"] == str(
        target.resolve()
    )
    assert list(target.parent.glob(f".{target.name}.*.migrating*")) == []


def test_migration_failure_cleans_temporary_database_and_all_sidecars(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source" / "director.sqlite3"
    target = tmp_path / "target" / "director.sqlite3"
    config = tmp_path / "config" / "storage.json"
    _director_database(source)
    _bootstrap(config, source)
    storage = StorageController.resolve(
        environ={},
        storage_config_path=config,
        legacy_database_path=tmp_path / "missing.sqlite3",
    )
    original_config = config.read_bytes()
    real_validate = storage_module.validate_director_database

    def reject_temporary_copy(path: Path) -> None:
        if path.name.endswith(".migrating"):
            for suffix in ("-wal", "-shm", "-journal"):
                Path(f"{path}{suffix}").write_bytes(b"injected sidecar")
            raise StorageValidationError("injected temporary validation failure")
        real_validate(path)

    monkeypatch.setattr(
        storage_module,
        "validate_director_database",
        reject_temporary_copy,
    )

    with pytest.raises(StorageValidationError, match="injected"):
        storage.migrate(target)

    assert not target.exists()
    assert config.read_bytes() == original_config
    assert list(target.parent.glob(f".{target.name}.*.migrating*")) == []


def test_hardlink_alias_is_treated_as_the_active_database(tmp_path) -> None:
    source = tmp_path / "source.sqlite3"
    alias = tmp_path / "source-alias.sqlite3"
    config = tmp_path / "storage.json"
    _director_database(source)
    os.link(source, alias)
    _bootstrap(config, source)
    storage = StorageController.resolve(
        environ={},
        storage_config_path=config,
        legacy_database_path=tmp_path / "missing.sqlite3",
    )

    assert storage.targets_active_database(alias) is True
    assert database_identity(alias) == database_identity(source)
    status = storage.configure_existing(alias)

    assert status.restart_required is False
    assert status.configured_database_path == source.resolve()
    assert json.loads(config.read_text(encoding="utf-8"))["database_path"] == str(
        source.resolve()
    )


def test_database_identity_changes_when_file_is_replaced_at_same_path(
    tmp_path,
) -> None:
    active_path = tmp_path / "director.sqlite3"
    replacement = tmp_path / "replacement.sqlite3"
    _director_database(active_path)
    _director_database(replacement)
    before = database_identity(active_path)

    os.replace(replacement, active_path)

    assert database_identity(active_path) != before


@pytest.mark.asyncio
async def test_storage_get_and_put_select_only_existing_valid_database(
    tmp_path, monkeypatch, inline_storage_threads
) -> None:
    monkeypatch.delenv("DIRECTOR_DATABASE_PATH", raising=False)
    source = tmp_path / "source.sqlite3"
    target = tmp_path / "target.sqlite3"
    invalid = tmp_path / "invalid.sqlite3"
    config = tmp_path / "config" / "storage.json"
    _director_database(source)
    _director_database(target)
    with sqlite3.connect(invalid) as connection:
        connection.execute("CREATE TABLE unrelated(value TEXT)")
    _bootstrap(config, source)
    app = create_app(
        storage_config_path=config,
        legacy_database_path=tmp_path / "missing.sqlite3",
    )
    app.state.instance_lock.acquire()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
            base_url="http://testserver",
        ) as client:
            initial = await client.get("/api/storage")
            assert initial.status_code == 200
            active_identity = database_identity(source)
            assert initial.json() == {
                "active_database_path": str(source.resolve()),
                "active_database_identity": active_identity,
                "configured_database_path": str(source.resolve()),
                "recommended_database_path": str(
                    app.state.storage.recommended_database_path
                ),
                "source": "bootstrap",
                "restart_required": False,
            }

            stale_browser = await client.put(
                "/api/settings",
                headers={"X-Director-Database-Identity": "0" * 64},
                json=default_settings("http://stale.test:8188").model_dump(
                    mode="json"
                ),
            )
            assert stale_browser.status_code == 409
            assert stale_browser.json()["code"] == "stale_database_identity"
            assert "different Director database" in stale_browser.json()["detail"]
            current_browser = await client.put(
                "/api/settings",
                headers={"X-Director-Database-Identity": active_identity},
                json=default_settings("http://current.test:8188").model_dump(
                    mode="json"
                ),
            )
            assert current_browser.status_code == 200

            stale_storage_selection = await client.put(
                "/api/storage",
                headers={"X-Director-Database-Identity": "0" * 64},
                json={"database_path": str(source)},
            )
            stale_storage_migration = await client.post(
                "/api/storage/migrate",
                headers={"X-Director-Database-Identity": "0" * 64},
                json={"target_path": str(tmp_path / "stale-target.sqlite3")},
            )
            assert stale_storage_selection.status_code == 409
            assert stale_storage_migration.status_code == 409
            assert stale_storage_selection.json()["code"] == (
                "stale_database_identity"
            )
            assert stale_storage_migration.json()["code"] == (
                "stale_database_identity"
            )
            assert not (tmp_path / "stale-target.sqlite3").exists()

            missing = await client.put(
                "/api/storage",
                json={"database_path": str(tmp_path / "missing.sqlite3")},
            )
            assert missing.status_code == 422
            relative = await client.put(
                "/api/storage",
                json={"database_path": "relative.sqlite3"},
            )
            assert relative.status_code == 422
            rejected = await client.put(
                "/api/storage", json={"database_path": str(invalid)}
            )
            assert rejected.status_code == 422

            selected = await client.put(
                "/api/storage", json={"database_path": str(target)}
            )
            assert selected.status_code == 200
            assert selected.json() == {
                "active_database_path": str(source.resolve()),
                "active_database_identity": active_identity,
                "configured_database_path": str(target.resolve()),
                "recommended_database_path": str(
                    app.state.storage.recommended_database_path
                ),
                "source": "bootstrap",
                "restart_required": True,
            }
            assert app.state.database.path == source.resolve()
            blocked = await client.put(
                "/api/settings",
                json=default_settings("http://blocked.test:8188").model_dump(
                    mode="json"
                ),
            )
            assert blocked.status_code == 409

            cancelled = await client.put(
                "/api/storage", json={"database_path": str(source)}
            )
            assert cancelled.status_code == 200
            assert cancelled.json()["restart_required"] is False
            resumed = await client.put(
                "/api/settings",
                json=default_settings("http://resumed.test:8188").model_dump(
                    mode="json"
                ),
            )
            assert resumed.status_code == 200
            reselected = await client.put(
                "/api/storage", json={"database_path": str(target)}
            )
            assert reselected.status_code == 200
            assert reselected.json()["restart_required"] is True
    finally:
        app.state.instance_lock.release()

    restarted = create_app(
        storage_config_path=config,
        legacy_database_path=tmp_path / "missing.sqlite3",
    )
    assert restarted.state.database.path == target.resolve()
    assert restarted.state.storage.selection.source == "bootstrap"


@pytest.mark.asyncio
async def test_migrate_copies_consistent_database_updates_bootstrap_and_freezes_writes(
    tmp_path, monkeypatch, inline_storage_threads
) -> None:
    monkeypatch.delenv("DIRECTOR_DATABASE_PATH", raising=False)
    source = tmp_path / "source.sqlite3"
    target = tmp_path / "migrated" / "director.sqlite3"
    config = tmp_path / "config" / "storage.json"
    database = _director_database(source, title="迁移前项目")
    database.put_settings(default_settings("http://comfy.test:8188"))
    _bootstrap(config, source)
    app = create_app(
        storage_config_path=config,
        legacy_database_path=tmp_path / "missing.sqlite3",
    )
    app.state.instance_lock.acquire()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
            base_url="http://testserver",
        ) as client:
            migrated = await client.post(
                "/api/storage/migrate", json={"target_path": str(target)}
            )
            assert migrated.status_code == 200, migrated.text
            assert migrated.json() == {
                "active_database_path": str(source.resolve()),
                "active_database_identity": database_identity(source),
                "configured_database_path": str(target.resolve()),
                "recommended_database_path": str(
                    app.state.storage.recommended_database_path
                ),
                "source": "bootstrap",
                "restart_required": True,
                "migrated_from": str(source.resolve()),
                "migrated_to": str(target.resolve()),
            }
            assert app.state.database.path == source.resolve()
            assert app.state.storage_write_frozen is True

            blocked = await client.put(
                "/api/settings",
                json=default_settings("http://other.test:8188").model_dump(
                    mode="json"
                ),
            )
            assert blocked.status_code == 409
            assert "restart" in blocked.json()["detail"]
            repeated = await client.post(
                "/api/storage/migrate",
                json={"target_path": str(tmp_path / "second.sqlite3")},
            )
            assert repeated.status_code == 409
    finally:
        app.state.instance_lock.release()

    copied = Database(target)
    assert copied.get_timeline().title == "迁移前项目"
    assert str(copied.get_settings().comfy_url) == "http://comfy.test:8188/"
    assert Database(source).get_timeline().title == "迁移前项目"
    assert json.loads(config.read_text(encoding="utf-8"))["database_path"] == str(
        target.resolve()
    )
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700


@pytest.mark.asyncio
async def test_storage_transition_drains_admitted_mutation_and_rejects_new_mutations(
    tmp_path, monkeypatch, inline_storage_threads
) -> None:
    monkeypatch.delenv("DIRECTOR_DATABASE_PATH", raising=False)
    source = tmp_path / "source.sqlite3"
    target = tmp_path / "target.sqlite3"
    config = tmp_path / "storage.json"
    _director_database(source)
    _director_database(target)
    _bootstrap(config, source)
    app = create_app(storage_config_path=config)
    admitted = asyncio.Event()
    release = asyncio.Event()

    @app.post("/api/test/hold-mutation")
    async def hold_mutation() -> dict[str, bool]:
        admitted.set()
        await release.wait()
        return {"released": True}

    app.state.instance_lock.acquire()
    held_request: asyncio.Task[httpx.Response] | None = None
    storage_request: asyncio.Task[httpx.Response] | None = None
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
            base_url="http://testserver",
        ) as client:
            held_request = asyncio.create_task(
                client.post("/api/test/hold-mutation")
            )
            await asyncio.wait_for(admitted.wait(), timeout=1)
            assert app.state.storage_inflight_mutations == 1

            storage_request = asyncio.create_task(
                client.put(
                    "/api/storage", json={"database_path": str(target)}
                )
            )

            async def wait_for_transition() -> None:
                while not app.state.storage_transitioning:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_for_transition(), timeout=1)
            assert storage_request.done() is False
            assert app.state.storage_inflight_mutations == 1

            late_mutation = await client.put(
                "/api/settings",
                json=default_settings("http://late.test:8188").model_dump(
                    mode="json"
                ),
            )
            assert late_mutation.status_code == 409
            assert app.state.storage_inflight_mutations == 1

            release.set()
            held_response, configured = await asyncio.gather(
                held_request, storage_request
            )
            assert held_response.status_code == 200
            assert configured.status_code == 200
            assert configured.json()["restart_required"] is True
            assert app.state.storage_inflight_mutations == 0
            assert app.state.storage_transitioning is False
            assert app.state.storage_write_frozen is True
    finally:
        release.set()
        pending = [
            task
            for task in (held_request, storage_request)
            if task is not None and not task.done()
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        app.state.instance_lock.release()


@pytest.mark.asyncio
async def test_active_preflight_rejects_storage_change_without_blocking_cancel(
    tmp_path, monkeypatch, inline_storage_threads
) -> None:
    monkeypatch.delenv("DIRECTOR_DATABASE_PATH", raising=False)
    source = tmp_path / "source.sqlite3"
    target = tmp_path / "target.sqlite3"
    migration_target = tmp_path / "migrated.sqlite3"
    config = tmp_path / "storage.json"
    _director_database(source)
    _director_database(target)
    _bootstrap(config, source)
    app = create_app(storage_config_path=config)
    preflight_started = asyncio.Event()
    preflight_release = asyncio.Event()

    @app.post("/api/test/preflight")
    async def preflight() -> dict[str, bool]:
        preflight_started.set()
        await preflight_release.wait()
        return {"cancelled": True}

    @app.post("/api/test/preflight/cancel")
    async def cancel_preflight() -> dict[str, bool]:
        preflight_release.set()
        return {"cancelled": True}

    monkeypatch.setattr(
        app.state.database,
        "has_active_work",
        lambda: preflight_started.is_set() and not preflight_release.is_set(),
    )
    app.state.instance_lock.acquire()
    preflight_request: asyncio.Task[httpx.Response] | None = None
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
            base_url="http://testserver",
        ) as client:
            preflight_request = asyncio.create_task(
                client.post("/api/test/preflight")
            )
            await asyncio.wait_for(preflight_started.wait(), timeout=1)

            configured = await asyncio.wait_for(
                client.put(
                    "/api/storage", json={"database_path": str(target)}
                ),
                timeout=1,
            )
            migrated = await asyncio.wait_for(
                client.post(
                    "/api/storage/migrate",
                    json={"target_path": str(migration_target)},
                ),
                timeout=1,
            )
            assert configured.status_code == 409
            assert migrated.status_code == 409
            assert app.state.storage_transitioning is False
            assert app.state.storage_write_frozen is False
            assert not migration_target.exists()

            cancelled = await asyncio.wait_for(
                client.post("/api/test/preflight/cancel"), timeout=1
            )
            assert cancelled.status_code == 200
            completed = await asyncio.wait_for(preflight_request, timeout=1)
            assert completed.status_code == 200
            assert app.state.storage_inflight_mutations == 0
    finally:
        preflight_release.set()
        if preflight_request is not None and not preflight_request.done():
            await asyncio.gather(preflight_request, return_exceptions=True)
        app.state.instance_lock.release()


@pytest.mark.asyncio
async def test_migration_rejects_active_work_existing_target_and_startup_override(
    tmp_path, monkeypatch, inline_storage_threads
) -> None:
    monkeypatch.delenv("DIRECTOR_DATABASE_PATH", raising=False)
    source = tmp_path / "source.sqlite3"
    existing = tmp_path / "existing.sqlite3"
    config = tmp_path / "storage.json"
    _director_database(source)
    _director_database(existing)
    _bootstrap(config, source)
    app = create_app(storage_config_path=config)
    app.state.instance_lock.acquire()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
            base_url="http://testserver",
        ) as client:
            same = await client.post(
                "/api/storage/migrate", json={"target_path": str(source)}
            )
            assert same.status_code == 409
            occupied = await client.post(
                "/api/storage/migrate", json={"target_path": str(existing)}
            )
            assert occupied.status_code == 409

            background_release = asyncio.Event()
            background_owner = asyncio.create_task(background_release.wait())
            app.state.submission_tasks.add(background_owner)
            try:
                live_owner_selection = await client.put(
                    "/api/storage", json={"database_path": str(existing)}
                )
                assert live_owner_selection.status_code == 409
                assert app.state.storage_write_frozen is False
            finally:
                background_release.set()
                await background_owner
                app.state.submission_tasks.discard(background_owner)

            monkeypatch.setattr(app.state.database, "has_active_work", lambda: True)
            active_selection = await client.put(
                "/api/storage", json={"database_path": str(existing)}
            )
            assert active_selection.status_code == 409
            assert app.state.storage_write_frozen is False
            active_target = tmp_path / "active.sqlite3"
            active = await client.post(
                "/api/storage/migrate", json={"target_path": str(active_target)}
            )
            assert active.status_code == 409
            assert not active_target.exists()
    finally:
        app.state.instance_lock.release()

    overridden = create_app(
        database_path=source,
        storage_config_path=config,
    )
    overridden.state.instance_lock.acquire()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=overridden, raise_app_exceptions=True),
            base_url="http://testserver",
        ) as client:
            response = await client.put(
                "/api/storage", json={"database_path": str(existing)}
            )
            assert response.status_code == 409
            status = await client.get("/api/storage")
            assert status.json()["configured_database_path"] == str(source.resolve())
            assert status.json()["restart_required"] is False
    finally:
        overridden.state.instance_lock.release()
