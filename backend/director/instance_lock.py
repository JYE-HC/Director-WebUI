from __future__ import annotations

import errno
import json
import os
import socket
import stat
from datetime import datetime, timezone
from pathlib import Path

_IS_WINDOWS = os.name == "nt"

if _IS_WINDOWS:
    import msvcrt
else:
    import fcntl

# flock reports a conflicting owner as EACCES/EAGAIN, msvcrt.locking as
# EACCES/EDEADLK; all of them take the same owner-diagnostic failure path.
_LOCK_CONFLICT_ERRNOS = frozenset({errno.EACCES, errno.EAGAIN, errno.EDEADLK})


def _lock_exclusive_nb(descriptor: int) -> None:
    """Take the non-blocking exclusive kernel lock on the lock file.

    POSIX locks the whole file with flock; Windows locks just the first byte,
    which is enough because every Director process locks the same range.  The
    OS drops either lock when the owning process exits, so the kernel lock —
    never the leftover file — represents ownership.
    """
    if _IS_WINDOWS:
        # msvcrt locks a byte range starting at the current file offset.
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(descriptor: int) -> None:
    if _IS_WINDOWS:
        # LK_UNLCK must start at the same offset the lock was taken at.
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(descriptor, fcntl.LOCK_UN)


class DirectorInstanceLockError(RuntimeError):
    """Raised when another Director process already owns the database."""


class DirectorInstanceLock:
    """Process lifetime lock for one canonical SQLite database path.

    The lock file is deliberately persistent.  Ownership is represented by a
    kernel lock (``flock(2)`` on POSIX, an ``msvcrt`` byte-range lock on
    Windows), not by the file's existence, so a crash cannot leave a stale
    marker that prevents the next process from starting. The key is the
    canonical path spelling; hard-link or bind-mount aliases are not claimed
    to share this sidecar lock.
    """

    def __init__(self, database_path: str | Path) -> None:
        canonical_database_path = (
            Path(database_path).expanduser().resolve(strict=False)
        )
        self.database_path = canonical_database_path
        self.path = Path(f"{canonical_database_path}.instance.lock")
        self._descriptor: int | None = None

    @property
    def acquired(self) -> bool:
        return self._descriptor is not None

    def acquire(self) -> None:
        if self._descriptor is not None:
            raise DirectorInstanceLockError(
                f"Director instance lock is already held by this app: {self.path}"
            )

        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise DirectorInstanceLockError(
                    f"Director instance lock path must not be a symlink: {self.path}"
                ) from exc
            raise DirectorInstanceLockError(
                f"Director instance lock could not be opened: {self.path}"
            ) from exc
        try:
            descriptor_metadata = os.fstat(descriptor)
        except OSError as exc:
            os.close(descriptor)
            raise DirectorInstanceLockError(
                f"Director instance lock could not be inspected: {self.path}"
            ) from exc
        if not stat.S_ISREG(descriptor_metadata.st_mode):
            os.close(descriptor)
            raise DirectorInstanceLockError(
                f"Director instance lock path is not a regular file: {self.path}"
            )
        if descriptor_metadata.st_nlink != 1:
            os.close(descriptor)
            raise DirectorInstanceLockError(
                f"Director instance lock path must not have hard links: {self.path}"
            )
        try:
            os.set_inheritable(descriptor, False)
        except OSError as exc:
            os.close(descriptor)
            raise DirectorInstanceLockError(
                f"Director instance lock could not be secured: {self.path}"
            ) from exc
        try:
            _lock_exclusive_nb(descriptor)
        except OSError as exc:
            if exc.errno not in _LOCK_CONFLICT_ERRNOS:
                os.close(descriptor)
                raise
            owner = self._read_diagnostic(descriptor)
            os.close(descriptor)
            detail = f"; current owner: {owner}" if owner else ""
            raise DirectorInstanceLockError(
                "Director cannot start because another process owns database "
                f"{self.database_path}. Lock file: {self.path}{detail}"
            ) from exc

        try:
            diagnostic = {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "acquired_at": datetime.now(timezone.utc).isoformat(),
                "database": str(self.database_path),
            }
            payload = (json.dumps(diagnostic, sort_keys=True) + "\n").encode("utf-8")
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, payload)
            os.fsync(descriptor)
        except BaseException:
            _unlock(descriptor)
            os.close(descriptor)
            raise

        self._descriptor = descriptor

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            _unlock(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _read_diagnostic(descriptor: int) -> str:
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            payload = os.read(descriptor, 4096).decode("utf-8", errors="replace")
        except OSError:
            return ""
        return " ".join(payload.strip().split())
