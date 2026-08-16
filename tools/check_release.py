#!/usr/bin/env python3
"""Fail CI when private/runtime files or common secret material enter release."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DENIED_NAMES = {"AGENTS.md", "踩坑.md", ".dsh", ".data", ".venv", "node_modules"}
DENIED_SUFFIXES = {".sqlite", ".sqlite3", ".db", ".log", ".pem", ".key", ".p12", ".pfx"}
TEXT_SUFFIXES = {
    "", ".css", ".env", ".html", ".ini", ".js", ".json", ".md", ".py",
    ".sh", ".toml", ".ts", ".tsx", ".txt", ".yaml", ".yml",
}
PRIVATE_PATTERNS = (
    re.compile(r"/(?:home|Users)/[^/\s]+/"),
    re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\s]+\\\\"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"https?://[^/@\s:]+:[^/@\s]+@"),
)
ALLOWED_EXACT_TEST_FIXTURES = {"https://alice:private-password@"}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        return [ROOT / part.decode("utf-8") for part in result.stdout.split(b"\0") if part]
    return [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]


def main() -> int:
    failures: list[str] = []
    files = tracked_files()
    for path in files:
        relative = path.relative_to(ROOT)
        parts = set(relative.parts)
        if any(part in DENIED_NAMES or part.startswith("踩坑_") for part in parts):
            failures.append(f"denied release path: {relative}")
            continue
        if relative.parts and relative.parts[0] == "data":
            failures.append(f"runtime data path: {relative}")
        if path.suffix.lower() in DENIED_SUFFIXES:
            failures.append(f"private/runtime file type: {relative}")
        if path.name == ".env":
            failures.append(f"real environment file: {relative}")
        if path.is_symlink():
            failures.append(f"symbolic link: {relative}")
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            failures.append(f"cannot stat {relative}: {exc}")
            continue
        if size > 10 * 1024 * 1024:
            failures.append(f"unexpected file larger than 10 MiB: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES or size > 2 * 1024 * 1024:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for pattern in PRIVATE_PATTERNS:
            match = pattern.search(content)
            if match and match.group(0) not in ALLOWED_EXACT_TEST_FIXTURES:
                failures.append(f"private path or credential pattern in {relative}")
                break

    staged = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--stage"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if staged.returncode == 0 and any(line.startswith("160000 ") for line in staged.stdout.splitlines()):
        failures.append("gitlink/submodule detected")

    payload = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "release_installer.py"), "payload-check"],
        check=False,
    )
    if payload.returncode:
        failures.append("bundled custom-node manifest check failed")

    if not (ROOT / "LICENSE").is_file():
        failures.append("root LICENSE is missing")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print(f"[PASS] release privacy/path scan checked {len(files)} tracked files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
