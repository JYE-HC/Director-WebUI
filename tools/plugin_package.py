"""Shared deterministic package/provenance helpers for DirectorDeck releases."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable


PROVENANCE_NAME = "PROVENANCE.json"
PROVENANCE_SCHEMA_VERSION = 1
LOCKFILES = ("frontend/package-lock.json", "uv.lock")


class PackageError(RuntimeError):
    """Raised when a package or its provenance cannot be trusted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            raise PackageError(f"symbolic link is not allowed: {relative.as_posix()}")
        if path.is_file():
            yield path


def artifact_records(root: Path, *, excluded: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    """Return stable hashes for every regular file below *root*.

    Paths are POSIX-normalized and sorted. The provenance file is deliberately
    excluded by its caller so the manifest never attempts to hash itself.
    """

    records: list[dict[str, Any]] = []
    for path in _package_files(root):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        records.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    return records


def artifact_tree_sha256(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            f"{record['sha256']} {record['size']} {record['path']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def command_output(command: list[str], *, cwd: Path) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise PackageError(f"cannot execute {command[0]}: {exc}") from exc
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise PackageError(f"{' '.join(command)} failed: {detail}")
    output = result.stdout.strip()
    if not output:
        raise PackageError(f"{' '.join(command)} returned no version/value")
    return output


def source_identity(project_root: Path) -> dict[str, Any]:
    commit = command_output(["git", "rev-parse", "HEAD"], cwd=project_root)
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise PackageError(f"git returned an invalid source commit: {commit!r}")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=project_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if status.returncode:
        detail = status.stderr.strip() or "unknown error"
        raise PackageError(f"git status failed: {detail}")
    return {"commit": commit, "dirty": bool(status.stdout.strip())}


def lockfile_records(project_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in LOCKFILES:
        path = project_root / relative
        if not path.is_file():
            raise PackageError(f"required lockfile is missing: {relative}")
        records.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    return records


def toolchain(project_root: Path) -> dict[str, str]:
    return {
        "node": command_output(["node", "--version"], cwd=project_root),
        "npm": command_output(["npm", "--version"], cwd=project_root),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def write_provenance(
    package_root: Path,
    *,
    project_root: Path,
    package_name: str,
    package_version: str,
    source: dict[str, Any],
) -> Path:
    records = artifact_records(package_root, excluded=frozenset({PROVENANCE_NAME}))
    provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "generated_by": "tools/build_plugin.py",
        "package": {"name": package_name, "version": package_version},
        "source": source,
        "lockfiles": lockfile_records(project_root),
        "toolchain": toolchain(project_root),
        "artifact": {
            "algorithm": "sha256",
            "scope": f"all generated package files except {PROVENANCE_NAME}",
            "tree_sha256": artifact_tree_sha256(records),
            "files": records,
        },
    }
    destination = package_root / PROVENANCE_NAME
    destination.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination


def load_provenance(package_root: Path) -> dict[str, Any]:
    path = package_root / PROVENANCE_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PackageError(f"required provenance is missing: {PROVENANCE_NAME}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"cannot parse {PROVENANCE_NAME}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PackageError(f"{PROVENANCE_NAME} root must be an object")
    return payload
