from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from director.app import create_app
from director.instance_lock import DirectorInstanceLock, DirectorInstanceLockError
from director.schemas import default_settings


def _silence_progress_manager(app) -> None:
    app.state.progress_manager.ensure = Mock()
    app.state.progress_manager.close = AsyncMock()


def test_instance_lock_rejects_symlink_without_touching_its_target(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "director.sqlite3"
    diagnostic_target = tmp_path / "must-not-be-truncated.txt"
    diagnostic_target.write_text("keep this content\n", encoding="utf-8")
    instance_lock = DirectorInstanceLock(database_path)
    instance_lock.path.symlink_to(diagnostic_target)

    with pytest.raises(DirectorInstanceLockError, match="symlink"):
        instance_lock.acquire()

    assert instance_lock.acquired is False
    assert diagnostic_target.read_text(encoding="utf-8") == "keep this content\n"


def test_instance_lock_rejects_hardlink_without_truncating_shared_file(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "director.sqlite3"
    shared_target = tmp_path / "must-not-be-truncated.txt"
    shared_target.write_text("keep hard-linked content\n", encoding="utf-8")
    instance_lock = DirectorInstanceLock(database_path)
    os.link(shared_target, instance_lock.path)

    with pytest.raises(DirectorInstanceLockError, match="hard links"):
        instance_lock.acquire()

    assert instance_lock.acquired is False
    assert shared_target.read_text(encoding="utf-8") == "keep hard-linked content\n"
    assert instance_lock.path.read_text(encoding="utf-8") == (
        "keep hard-linked content\n"
    )


def test_instance_lock_creates_private_parent_and_file(tmp_path: Path) -> None:
    instance_lock = DirectorInstanceLock(
        tmp_path / "private-storage" / "director.sqlite3"
    )
    previous_umask = os.umask(0o002)
    try:
        instance_lock.acquire()
    finally:
        os.umask(previous_umask)
        instance_lock.release()

    assert (instance_lock.path.parent.stat().st_mode & 0o777) == 0o700
    assert (instance_lock.path.stat().st_mode & 0o777) == 0o600


async def test_second_lifespan_for_same_database_fails_before_database_or_comfy_io(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "shared.sqlite3"
    first = create_app(database_path=database_path)
    first.state.database.initialize()
    first.state.database.put_settings(default_settings("http://comfy.invalid:8188"))
    _silence_progress_manager(first)

    second = create_app(database_path=database_path)
    original_initialize = second.state.database.initialize
    second.state.database.initialize = Mock(wraps=original_initialize)
    _silence_progress_manager(second)

    async with first.router.lifespan_context(first):
        assert first.state.instance_lock.acquired is True
        with pytest.raises(DirectorInstanceLockError) as raised:
            async with second.router.lifespan_context(second):
                pytest.fail("the second Director instance unexpectedly started")

        assert str(database_path.resolve()) in str(raised.value)
        assert str(first.state.instance_lock.path) in str(raised.value)
        assert '"pid":' in str(raised.value)
        second.state.database.initialize.assert_not_called()
        second.state.progress_manager.ensure.assert_not_called()
        assert second.state.instance_lock.acquired is False

    assert first.state.instance_lock.acquired is False

    async with second.router.lifespan_context(second):
        assert second.state.instance_lock.acquired is True

    second.state.database.initialize.assert_called_once_with()
    second.state.progress_manager.ensure.assert_called_once()
    assert second.state.instance_lock.acquired is False


async def test_different_database_lifespans_can_run_together(tmp_path: Path) -> None:
    first = create_app(database_path=tmp_path / "first.sqlite3")
    second = create_app(database_path=tmp_path / "second.sqlite3")
    _silence_progress_manager(first)
    _silence_progress_manager(second)

    async with first.router.lifespan_context(first):
        async with second.router.lifespan_context(second):
            assert first.state.instance_lock.acquired is True
            assert second.state.instance_lock.acquired is True
            assert first.state.instance_lock.path != second.state.instance_lock.path


async def test_residual_lock_file_does_not_block_startup(tmp_path: Path) -> None:
    database_path = tmp_path / "director.sqlite3"
    lock_path = Path(f"{database_path.resolve()}.instance.lock")
    lock_path.write_text("stale owner from a crashed process\n", encoding="utf-8")
    app = create_app(database_path=database_path)
    _silence_progress_manager(app)

    async with app.router.lifespan_context(app):
        assert app.state.instance_lock.acquired is True
        diagnostic = json.loads(lock_path.read_text(encoding="utf-8"))
        assert diagnostic["pid"] == os.getpid()
        assert diagnostic["database"] == str(database_path.resolve())

    assert lock_path.exists()
    assert app.state.instance_lock.acquired is False


async def test_instance_lock_is_released_after_managed_shutdown(tmp_path: Path) -> None:
    app = create_app(database_path=tmp_path / "director.sqlite3")
    app.state.progress_manager.ensure = Mock()

    async def close_progress() -> None:
        assert app.state.instance_lock.acquired is True

    app.state.progress_manager.close = AsyncMock(side_effect=close_progress)

    async with app.router.lifespan_context(app):
        assert app.state.instance_lock.acquired is True

    app.state.progress_manager.close.assert_awaited_once_with()
    assert app.state.instance_lock.acquired is False
