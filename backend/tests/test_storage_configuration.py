"""Database file hygiene tests.

The storage selection/migration suite was removed with the embedded plugin
cutover: the database lives at one fixed path below the ComfyUI user
directory, so there is no bootstrap chain, target validation, or database
identity to exercise anymore.
"""

from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path

import pytest

from directordeck.database import Database


def _director_database(path: Path, *, title: str | None = None) -> Database:
    database = Database(path)
    database.initialize()
    if title is not None:
        timeline, revision = database.get_timeline_authority()
        database.validate_and_put_timeline_authority(
            timeline.model_copy(update={"title": title}),
            expected_revision=revision,
        )
    return database


def test_database_context_manager_closes_connection_and_repeated_reads_do_not_leak_fds(
    tmp_path,
) -> None:
    database = _director_database(tmp_path / "directordeck.sqlite3")
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


@pytest.mark.skipif(
    os.name == "nt", reason="Windows st_mode does not carry Unix permission bits"
)
def test_new_database_and_parent_are_private_without_changing_existing_mode(
    tmp_path,
) -> None:
    database_path = tmp_path / "private" / "directordeck.sqlite3"
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
