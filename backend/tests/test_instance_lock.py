from __future__ import annotations

import errno
import json
import os
import socket
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

import director.instance_lock as instance_lock_module
from director.app import create_app
from director.instance_lock import DirectorInstanceLock, DirectorInstanceLockError
from director.schemas import default_settings


def _silence_progress_manager(app) -> None:
    app.state.progress_manager.ensure = Mock()
    app.state.progress_manager.close = AsyncMock()


@pytest.mark.skipif(
    os.name == "nt", reason="Windows symlink creation needs elevated privilege"
)
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


@pytest.mark.skipif(
    os.name == "nt", reason="Windows st_mode does not carry Unix permission bits"
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


def test_instance_lock_rejects_second_acquire_in_same_process(
    tmp_path: Path,
) -> None:
    instance_lock = DirectorInstanceLock(tmp_path / "director.sqlite3")
    instance_lock.acquire()
    try:
        with pytest.raises(DirectorInstanceLockError, match="already held"):
            instance_lock.acquire()
    finally:
        instance_lock.release()


def test_independent_locks_on_one_path_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "director.sqlite3"
    first = DirectorInstanceLock(database_path)
    second = DirectorInstanceLock(database_path)
    first.acquire()
    try:
        with pytest.raises(DirectorInstanceLockError) as raised:
            second.acquire()
    finally:
        first.release()

    message = str(raised.value)
    assert "another process owns database" in message
    assert str(first.path) in message
    owner = json.loads(message.rsplit("current owner: ", 1)[1])
    assert owner["pid"] == os.getpid()
    assert owner["database"] == str(database_path.resolve())
    assert second.acquired is False


def test_instance_lock_can_be_reacquired_after_release(tmp_path: Path) -> None:
    database_path = tmp_path / "director.sqlite3"
    first = DirectorInstanceLock(database_path)
    first.acquire()
    first.release()

    second = DirectorInstanceLock(database_path)
    second.acquire()
    try:
        assert second.acquired is True
    finally:
        second.release()
    assert second.acquired is False


def test_instance_lock_writes_owner_diagnostic_json(tmp_path: Path) -> None:
    database_path = tmp_path / "director.sqlite3"
    instance_lock = DirectorInstanceLock(database_path)
    instance_lock.acquire()
    try:
        diagnostic = json.loads(instance_lock.path.read_text(encoding="utf-8"))
    finally:
        instance_lock.release()

    assert diagnostic["pid"] == os.getpid()
    assert diagnostic["hostname"] == socket.gethostname()
    assert diagnostic["database"] == str(database_path.resolve())
    datetime.fromisoformat(diagnostic["acquired_at"])


class _FakeMsvcrt:
    """In-memory msvcrt.locking stand-in so the Windows branch runs anywhere.

    Byte-range ownership is keyed on the underlying file, so two descriptors
    for one lock path conflict like real per-process Windows byte locks.
    """

    LK_NBLCK = 1
    LK_UNLCK = 2

    def __init__(self) -> None:
        self.owners: dict[tuple[int, int], int] = {}
        self.unlock_offsets: list[int] = []

    def locking(self, descriptor: int, mode: int, nbytes: int) -> None:
        key = self._file_key(descriptor)
        if mode == self.LK_NBLCK:
            owner = self.owners.get(key)
            if owner is not None and owner != descriptor:
                raise OSError(errno.EACCES, "Permission denied")
            self.owners[key] = descriptor
        elif mode == self.LK_UNLCK:
            self.unlock_offsets.append(os.lseek(descriptor, 0, os.SEEK_CUR))
            if self.owners.get(key) == descriptor:
                del self.owners[key]
        else:
            raise ValueError(f"unsupported locking mode: {mode}")

    @staticmethod
    def _file_key(descriptor: int) -> tuple[int, int]:
        metadata = os.fstat(descriptor)
        return (metadata.st_dev, metadata.st_ino)


@pytest.fixture
def fake_windows_locking(monkeypatch: pytest.MonkeyPatch) -> _FakeMsvcrt:
    fake = _FakeMsvcrt()
    monkeypatch.setattr(instance_lock_module, "_IS_WINDOWS", True)
    monkeypatch.setattr(instance_lock_module, "msvcrt", fake, raising=False)
    return fake


def test_windows_branch_excludes_second_owner_and_recovers(
    tmp_path: Path,
    fake_windows_locking: _FakeMsvcrt,
) -> None:
    first = DirectorInstanceLock(tmp_path / "director.sqlite3")
    second = DirectorInstanceLock(first.database_path)
    first.acquire()
    try:
        with pytest.raises(DirectorInstanceLockError) as raised:
            second.acquire()
        assert second.acquired is False
        assert "another process owns database" in str(raised.value)
        assert f'"pid": {os.getpid()}' in str(raised.value)
    finally:
        first.release()

    # LK_UNLCK ran after seeking back to the locked byte past the diagnostic
    # payload even though the diagnostic write had moved the descriptor offset.
    lock_offset = instance_lock_module._WINDOWS_LOCK_OFFSET
    assert fake_windows_locking.unlock_offsets == [lock_offset]

    second.acquire()
    second.release()
    assert fake_windows_locking.unlock_offsets == [lock_offset, lock_offset]


@pytest.mark.parametrize("conflict_errno", [errno.EACCES, errno.EDEADLK])
def test_windows_lock_conflict_maps_to_owner_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    conflict_errno: int,
) -> None:
    def reject_locking(descriptor: int, mode: int, nbytes: int) -> None:
        raise OSError(conflict_errno, "lock conflict")

    fake = Mock(LK_NBLCK=1, LK_UNLCK=2, locking=reject_locking)
    monkeypatch.setattr(instance_lock_module, "_IS_WINDOWS", True)
    monkeypatch.setattr(instance_lock_module, "msvcrt", fake, raising=False)
    instance_lock = DirectorInstanceLock(tmp_path / "director.sqlite3")
    instance_lock.path.write_text(
        json.dumps({"pid": 4242, "hostname": "other-host"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DirectorInstanceLockError) as raised:
        instance_lock.acquire()

    message = str(raised.value)
    assert "another process owns database" in message
    assert str(instance_lock.path) in message
    assert '"pid": 4242' in message
    assert instance_lock.acquired is False


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
