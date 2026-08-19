from __future__ import annotations

from pathlib import Path


def _canonical_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


class StorageController:
    """The single, process-fixed Director database location.

    Director runs embedded in one ComfyUI process; the plugin always passes
    the explicit database path below the host's user directory. There is no
    runtime database selection, migration, or identity tracking: the database
    file can never change while this process lives.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.active_database_path = _canonical_path(database_path)

    @classmethod
    def resolve(cls, database_path: str | Path) -> "StorageController":
        return cls(database_path)
