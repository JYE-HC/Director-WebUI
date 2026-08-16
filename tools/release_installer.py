#!/usr/bin/env python3
"""Small, dependency-free helpers used by install.sh."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}


def _tree_entries(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"not a normal directory: {root}")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"symbolic link is not allowed: {relative.as_posix()}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"unsupported filesystem entry: {relative.as_posix()}")
        if path.suffix == ".pyc":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        mode = path.stat().st_mode
        entries.append(
            {
                "path": relative.as_posix(),
                "sha256": digest.hexdigest(),
                "executable": bool(mode & stat.S_IXUSR),
            }
        )
    return entries


def tree_digest(root: Path) -> str:
    encoded = json.dumps(
        _tree_entries(root), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest() -> dict[str, Any]:
    with (ROOT / "release-manifest.json").open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("release-manifest.json must contain an object")
    return value


def _nodes() -> list[dict[str, Any]]:
    nodes = _manifest().get("custom_nodes")
    if not isinstance(nodes, list):
        raise ValueError("release manifest custom_nodes must be a list")
    return [node for node in nodes if isinstance(node, dict)]


def expected_node(name: str) -> dict[str, Any]:
    for node in _nodes():
        if node.get("name") == name:
            return node
    raise ValueError(f"unknown bundled custom node: {name}")


def payload_check() -> int:
    failures: list[str] = []
    for node in _nodes():
        name = str(node.get("name") or "")
        expected = str(node.get("tree_sha256") or "")
        root = ROOT / "custom_nodes" / name
        try:
            actual = tree_digest(root)
        except (OSError, ValueError) as exc:
            failures.append(f"{name}: {exc}")
            continue
        if not (root / "LICENSE").is_file():
            failures.append(f"{name}: LICENSE is missing")
        if actual != expected:
            failures.append(
                f"{name}: payload digest mismatch (expected {expected}, got {actual})"
            )
        else:
            print(f"[PASS] bundled node {name}: {actual[:12]}")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}", file=sys.stderr)
        return 2
    return 0


def node_status(name: str, target: Path) -> int:
    expected = str(expected_node(name).get("tree_sha256") or "")
    if target.is_symlink():
        print("symlink")
        return 0
    if not target.exists():
        print("absent")
        return 0
    if not target.is_dir():
        print("conflict")
        return 0
    try:
        actual = tree_digest(target)
    except (OSError, ValueError):
        print("conflict")
        return 0
    print("same" if actual == expected else "conflict")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("payload-check")
    digest_parser = subparsers.add_parser("tree-digest")
    digest_parser.add_argument("path", type=Path)
    status_parser = subparsers.add_parser("node-status")
    status_parser.add_argument("name")
    status_parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "payload-check":
            return payload_check()
        if args.command == "tree-digest":
            print(tree_digest(args.path.resolve()))
            return 0
        if args.command == "node-status":
            return node_status(args.name, args.path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    return 64


if __name__ == "__main__":
    raise SystemExit(main())
