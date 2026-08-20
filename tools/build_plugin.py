#!/usr/bin/env python3
"""Assemble the DirectorDeck plugin package into build/DirectorDeck/.

The package layout produced here is what gets linked into
``ComfyUI/custom_nodes/`` for local testing and (later) published to the
plugin repository:

    DirectorDeck/
    ├── __init__.py        # plugin entry (from plugin/)
    ├── pyproject.toml     # registry manifest (from plugin/)
    ├── web/               # WEB_DIRECTORY sidebar extension (from plugin/)
    ├── backend/directordeck/  # Director backend package (from backend/)
    └── dist/              # built frontend (from frontend/dist, run npm build first)
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

from plugin_package import PackageError, source_identity, write_provenance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = PROJECT_ROOT / "build" / "DirectorDeck"

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".git")

BUNDLED_NODE_PACKS = ("raylight", "ComfyUI-MiniMax-H3-Turbo")


def _package_identity() -> tuple[str, str]:
    manifest = (PROJECT_ROOT / "plugin" / "pyproject.toml").read_text(encoding="utf-8")
    name_match = re.search(r'^name\s*=\s*"([^"]+)"\s*$', manifest, flags=re.MULTILINE)
    version_match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', manifest, flags=re.MULTILINE)
    if name_match is None or version_match is None:
        raise SystemExit("plugin/pyproject.toml is missing project name or version")
    return name_match.group(1), version_match.group(1)


def _copy_payload(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root_files = ("SECURITY.md", "LICENSE", "THIRD_PARTY_NOTICES.md")
    plugin_files = ("__init__.py", "pyproject.toml", "requirements.txt", "README.md")
    for filename in plugin_files:
        source = PROJECT_ROOT / "plugin" / filename
        if not source.is_file():
            raise SystemExit(f"required plugin package file missing: {source}")
        shutil.copy2(source, destination / filename)
    for filename in root_files:
        source = PROJECT_ROOT / filename
        if not source.is_file():
            raise SystemExit(f"required release policy/license file missing: {source}")
        shutil.copy2(source, destination / filename)

    shutil.copytree(PROJECT_ROOT / "plugin" / "web", destination / "web", ignore=IGNORE)
    shutil.copytree(
        PROJECT_ROOT / "backend" / "directordeck",
        destination / "backend" / "directordeck",
        ignore=IGNORE,
    )
    shutil.copytree(PROJECT_ROOT / "frontend" / "dist", destination / "dist", ignore=IGNORE)

    for pack in BUNDLED_NODE_PACKS:
        pack_src = PROJECT_ROOT / "custom_nodes" / pack
        if not pack_src.is_dir():
            raise SystemExit(f"bundled node pack missing: {pack_src}")
        shutil.copytree(pack_src, destination / "nodes" / pack, ignore=IGNORE)
    raylight_requirements = (
        PROJECT_ROOT / "custom_nodes" / "raylight" / "requirements.txt"
    )
    if not raylight_requirements.is_file():
        raise SystemExit(f"bundled RayLight requirements missing: {raylight_requirements}")
    shutil.copy2(raylight_requirements, destination / "requirements-raylight.txt")


def assemble(*, require_clean: bool = False) -> None:
    dist_src = PROJECT_ROOT / "frontend" / "dist"
    if not dist_src.is_dir():
        raise SystemExit(
            "frontend/dist missing; run `npm run build` in frontend/ first"
        )
    try:
        source = source_identity(PROJECT_ROOT)
    except PackageError as exc:
        raise SystemExit(str(exc)) from exc
    if require_clean and source["dirty"]:
        raise SystemExit("release build requires a clean git source tree")

    package_name, package_version = _package_identity()
    BUILD_DIR.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".DirectorDeck-build-", dir=BUILD_DIR.parent))
    try:
        _copy_payload(staging)
        write_provenance(
            staging,
            project_root=PROJECT_ROOT,
            package_name=package_name,
            package_version=package_version,
            source=source,
        )
        if BUILD_DIR.is_symlink():
            raise SystemExit(
                f"refusing to replace symbolic-link build directory: {BUILD_DIR}"
            )
        if BUILD_DIR.exists():
            shutil.rmtree(BUILD_DIR)
        staging.replace(BUILD_DIR)
    except PackageError as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"assembled plugin package at {BUILD_DIR}")


def link(comfyui_root: Path) -> None:
    custom_nodes = comfyui_root / "custom_nodes"
    if not custom_nodes.is_dir():
        raise SystemExit(f"custom_nodes not found under {comfyui_root}")
    target = custom_nodes / "DirectorDeck"
    if target.is_symlink() or target.exists():
        if target.is_symlink() and target.resolve() == BUILD_DIR.resolve():
            print(f"link already in place: {target}")
            return
        raise SystemExit(
            f"{target} already exists and is not this build; remove it first"
        )
    target.symlink_to(BUILD_DIR.resolve())
    print(f"linked {target} -> {BUILD_DIR.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--link",
        metavar="COMFYUI_ROOT",
        type=Path,
        default=None,
        help="symlink the assembled package into COMFYUI_ROOT/custom_nodes",
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="fail unless the source checkout is clean (required by release CI)",
    )
    args = parser.parse_args()
    assemble(require_clean=args.require_clean)
    if args.link is not None:
        link(args.link)


if __name__ == "__main__":
    sys.exit(main())
