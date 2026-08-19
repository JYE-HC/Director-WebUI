#!/usr/bin/env python3
"""Assemble the DirectorDeck plugin package into build/DirectorDeck/.

The package layout produced here is what gets linked into
``ComfyUI/custom_nodes/`` for local testing and (later) published to the
plugin repository:

    DirectorDeck/
    ├── __init__.py        # plugin entry (from plugin/)
    ├── pyproject.toml     # registry manifest (from plugin/)
    ├── web/               # WEB_DIRECTORY sidebar extension (from plugin/)
    ├── backend/director/  # Director backend package (from backend/)
    └── dist/              # built frontend (from frontend/dist, run npm build first)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = PROJECT_ROOT / "build" / "DirectorDeck"

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".git")

BUNDLED_NODE_PACKS = ("raylight", "ComfyUI-MiniMax-H3-Turbo")


def assemble() -> None:
    dist_src = PROJECT_ROOT / "frontend" / "dist"
    if not dist_src.is_dir():
        raise SystemExit(
            "frontend/dist missing; run `npm run build` in frontend/ first"
        )
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)

    shutil.copy2(PROJECT_ROOT / "plugin" / "__init__.py", BUILD_DIR / "__init__.py")
    shutil.copy2(PROJECT_ROOT / "plugin" / "pyproject.toml", BUILD_DIR / "pyproject.toml")
    shutil.copy2(
        PROJECT_ROOT / "plugin" / "requirements.txt", BUILD_DIR / "requirements.txt"
    )
    shutil.copytree(PROJECT_ROOT / "plugin" / "web", BUILD_DIR / "web", ignore=IGNORE)
    shutil.copytree(
        PROJECT_ROOT / "backend" / "director", BUILD_DIR / "backend" / "director",
        ignore=IGNORE,
    )
    shutil.copytree(dist_src, BUILD_DIR / "dist", ignore=IGNORE)

    for pack in BUNDLED_NODE_PACKS:
        pack_src = PROJECT_ROOT / "custom_nodes" / pack
        if not pack_src.is_dir():
            raise SystemExit(f"bundled node pack missing: {pack_src}")
        shutil.copytree(pack_src, BUILD_DIR / "nodes" / pack, ignore=IGNORE)
    raylight_requirements = (
        PROJECT_ROOT / "custom_nodes" / "raylight" / "requirements.txt"
    )
    if raylight_requirements.is_file():
        shutil.copy2(raylight_requirements, BUILD_DIR / "requirements-raylight.txt")
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
    args = parser.parse_args()
    assemble()
    if args.link is not None:
        link(args.link)


if __name__ == "__main__":
    sys.exit(main())
