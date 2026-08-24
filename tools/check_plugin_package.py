#!/usr/bin/env python3
"""Validate a generated or final-repository DirectorDeck package.

The default mode checks the exact tree produced by ``build_plugin.py``. Pass
``--final-repository`` after synchronizing that tree into the public plugin
repository; only the explicitly owned repository metadata files may exist in
addition to the provenance-covered generated payload.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - DirectorDeck requires Python 3.12+
    try:
        import tomli as tomllib  # type: ignore[no-redef,import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("Python 3.11+ (or tomli) is required for package validation") from exc

from plugin_package import (
    PROVENANCE_NAME,
    PROVENANCE_SCHEMA_VERSION,
    PackageError,
    artifact_records,
    artifact_tree_sha256,
    load_provenance,
    lockfile_records,
    source_identity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_DIR = PROJECT_ROOT / "build" / "DirectorDeck"
PACKAGE_NAME = "director-deck"
REPOSITORY_URL = "https://github.com/JYE-HC/DirectorDeck"
PUBLISHER_REQUIREMENTS = ".github/comfy-cli-requirements.txt"
PUBLISHER_PACKAGE = "comfy-cli"
PUBLISHER_VERSION = "1.10.3"
PUBLISHER_WHEEL_SHA256 = (
    "5e5336bb9f820142a94a1e1e01ee2da52fd7d5ce8f537af162fbb15b404e6f60"
)

REQUIRED_PATHS = {
    "__init__.py",
    "backend/directordeck/__init__.py",
    "backend/directordeck/config/directordeck.json",
    "dist/index.html",
    "LICENSE",
    "pyproject.toml",
    PROVENANCE_NAME,
    "README.md",
    "requirements-raylight.txt",
    "requirements.txt",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "web/directordeck-menu.js",
}
REQUIRED_DIRECTORIES = {
    "backend/directordeck",
    "dist/assets",
    "nodes/DirectorDeck-Strict-Attention",
    "nodes/DirectorDeck-Strict-H3",
    "nodes/DirectorDeck-RayLight",
    "nodes/DirectorDeck-RayLight/src/directordeck_raylight",
    "web",
}
FORBIDDEN_BUNDLED_NODE_DIRECTORIES = {
    "nodes/ComfyUI-MiniMax-H3-Turbo",
    "nodes/DirectorDeck-Strict-LoRA",
    "nodes/raylight",
    "nodes/DirectorDeck-RayLight/src/raylight",
    "nodes/DirectorDeck-RayLight/src/_ray_runtime_env",
    "nodes/DirectorDeck-RayLight/docs",
    "nodes/DirectorDeck-RayLight/example_workflows",
    "nodes/DirectorDeck-RayLight/tests",
}
FINAL_REPOSITORY_EXTRAS = {
    PUBLISHER_REQUIREMENTS,
    ".github/workflows/publish.yml",
    ".gitignore",
    "banner.png",
    "icon.png",
}
DENIED_NAMES = {"AGENTS.md", "踩坑.md", ".dsh", ".data", ".venv", "node_modules"}
DENIED_TOP_LEVEL = {
    "assets", "build", "data", "doc", "frontend", "input", "output", "plugin", "tools", "user",
}
DENIED_SUFFIXES = {".db", ".key", ".log", ".p12", ".pem", ".pfx", ".sqlite", ".sqlite3"}
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


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PackageError(f"cannot parse {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PackageError(f"{path.name} root must be a table")
    return payload


def _requirements(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise PackageError(f"cannot read {path.name}: {exc}") from exc
    requirements = [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if any(line.startswith(("-", "http://", "https://", "git+")) for line in requirements):
        raise PackageError("requirements.txt must contain only direct package constraints")
    if len(requirements) != len(set(requirements)):
        raise PackageError("requirements.txt contains duplicate constraints")
    return requirements


def _validate_manifest(package_root: Path, *, source_root: Path) -> str:
    manifest = _load_toml(package_root / "pyproject.toml")
    project = manifest.get("project")
    comfy = manifest.get("tool", {}).get("comfy")
    if not isinstance(project, dict) or not isinstance(comfy, dict):
        raise PackageError("pyproject.toml is missing [project] or [tool.comfy]")
    version = project.get("version")
    if project.get("name") != PACKAGE_NAME:
        raise PackageError(f"pyproject project.name must be {PACKAGE_NAME!r}")
    version_pattern = r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?"
    if not isinstance(version, str) or not re.fullmatch(version_pattern, version):
        raise PackageError("pyproject project.version is not a valid release version")
    if project.get("readme") != "README.md":
        raise PackageError("pyproject project.readme must be README.md")
    if project.get("license") != "GPL-3.0-only":
        raise PackageError("pyproject project.license must be GPL-3.0-only")
    if project.get("requires-python") != ">=3.12":
        raise PackageError("pyproject requires-python must be >=3.12")
    if project.get("urls", {}).get("Repository") != REPOSITORY_URL:
        raise PackageError("pyproject repository URL is incorrect")
    expected_comfy = {
        "PublisherId": "jye-hc",
        "DisplayName": "DirectorDeck",
        "Icon": "https://raw.githubusercontent.com/JYE-HC/DirectorDeck/main/icon.png",
        "Banner": "https://raw.githubusercontent.com/JYE-HC/DirectorDeck/main/banner.png",
    }
    if any(comfy.get(key) != value for key, value in expected_comfy.items()):
        raise PackageError("Comfy Registry publisher/display metadata is incorrect")

    dependencies = project.get("dependencies")
    requirements = _requirements(package_root / "requirements.txt")
    if not isinstance(dependencies, list) or not all(
        isinstance(item, str) for item in dependencies
    ):
        raise PackageError("pyproject project.dependencies must be a string list")
    if dependencies != requirements:
        raise PackageError(
            "pyproject dependencies and requirements.txt must be exactly equivalent and ordered"
        )

    source_manifest = source_root / "pyproject.toml"
    if source_manifest.is_file():
        source_dependencies = _load_toml(source_manifest).get("project", {}).get("dependencies")
        if source_dependencies != dependencies:
            raise PackageError("root, plugin and requirements dependency sets have drifted")
    return version


def _validate_docs(package_root: Path) -> None:
    readme = (package_root / "README.md").read_text(encoding="utf-8")
    security = (package_root / "SECURITY.md").read_text(encoding="utf-8")
    license_text = (package_root / "LICENSE").read_text(encoding="utf-8")
    notices = (package_root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    if "/directordeck/" not in readme or "user/directordeck/database/" not in readme:
        raise PackageError("README is missing the canonical route or database location")
    if "/director/" in readme or "user/director/" in readme:
        raise PackageError("README still contains a legacy Director route or data path")
    normalized_security = security.replace("\r\n", "\n")
    if (
        re.search(r"\bno\s+built-in login\b", normalized_security, flags=re.IGNORECASE) is None
        or "TLS" not in normalized_security
        or "public Internet" not in normalized_security
        or "/directordeck/" not in normalized_security
    ):
        raise PackageError("SECURITY.md does not state the required trust and exposure boundary")
    if "GNU GENERAL PUBLIC LICENSE" not in license_text or "Version 3" not in license_text:
        raise PackageError("LICENSE is not the expected GPL v3 text")
    if "RayLight" not in notices:
        raise PackageError("THIRD_PARTY_NOTICES.md is missing the bundled RayLight notice")


def _validate_product_config(package_root: Path) -> None:
    backend_root = package_root / "backend"
    sys.path.insert(0, str(backend_root))
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from directordeck.config_manager import DirectorDeckConfig

        raw = json.loads(
            (backend_root / "directordeck/config/directordeck.json").read_text(
                encoding="utf-8"
            )
        )
        DirectorDeckConfig.model_validate(raw)
    except Exception as exc:
        raise PackageError("DirectorDeck product configuration is invalid") from exc
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
        try:
            sys.path.remove(str(backend_root))
        except ValueError:  # pragma: no cover - defensive path hygiene.
            pass


def _validate_privacy_and_paths(package_root: Path) -> None:
    for path in package_root.rglob("*"):
        relative = path.relative_to(package_root)
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            raise PackageError(f"symbolic link is not allowed: {relative.as_posix()}")
        if not path.is_file():
            continue
        if relative.parts and relative.parts[0] in DENIED_TOP_LEVEL:
            raise PackageError(f"denied top-level release path: {relative.as_posix()}")
        if any(part in DENIED_NAMES or part.startswith("踩坑_") for part in relative.parts):
            raise PackageError(f"denied release path: {relative.as_posix()}")
        if path.name == ".env" or path.suffix.lower() in DENIED_SUFFIXES:
            raise PackageError(f"private/runtime file type: {relative.as_posix()}")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise PackageError(f"cannot stat {relative.as_posix()}: {exc}") from exc
        if size > 10 * 1024 * 1024:
            raise PackageError(f"unexpected file larger than 10 MiB: {relative.as_posix()}")
        if path.suffix.lower() not in TEXT_SUFFIXES or size > 2 * 1024 * 1024:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for pattern in PRIVATE_PATTERNS:
            match = pattern.search(content)
            if match and match.group(0) not in ALLOWED_EXACT_TEST_FIXTURES:
                raise PackageError(f"private path or credential pattern in {relative.as_posix()}")


def _validate_final_repository_policy(package_root: Path) -> None:
    workflow = (package_root / ".github/workflows/publish.yml").read_text(
        encoding="utf-8"
    )
    action_refs = re.findall(r"^\s*-?\s*uses:\s*([^\s#]+)", workflow, re.MULTILINE)
    if not action_refs or any(
        re.fullmatch(r"[^/@\s]+/[^/@\s]+@[0-9a-f]{40}", action_ref) is None
        for action_ref in action_refs
    ):
        raise PackageError("publish workflow actions must use full commit SHAs")
    required_workflow_tokens = (
        "workflow_dispatch:",
        "contents: read",
        'test "$GITHUB_REF" = "refs/heads/main"',
        "--final-repository",
        "--require-clean-source",
        "--only-binary=:all:",
        "--require-hashes",
        "pip check",
        "importlib.metadata.version",
    )
    if any(token not in workflow for token in required_workflow_tokens):
        raise PackageError("publish workflow is missing a release safety gate")
    if "publish-node-action" in workflow:
        raise PackageError("publish workflow must not call an action with nested movable refs")

    lock_text = (package_root / PUBLISHER_REQUIREMENTS).read_text(encoding="utf-8")
    packages: dict[str, tuple[str, set[str]]] = {}
    current_name: str | None = None
    current_version: str | None = None
    current_hashes: set[str] = set()

    def finish_requirement() -> None:
        nonlocal current_name, current_version, current_hashes
        if current_name is None or current_version is None:
            return
        if not current_hashes:
            raise PackageError(f"publisher requirement {current_name} has no hashes")
        normalized_name = current_name.lower().replace("_", "-")
        if normalized_name in packages:
            raise PackageError(f"duplicate publisher requirement: {normalized_name}")
        packages[normalized_name] = (current_version, set(current_hashes))
        current_name = None
        current_version = None
        current_hashes = set()

    for raw_line in lock_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirement_match = re.fullmatch(
            r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s\\]+)\s*\\",
            line,
        )
        if requirement_match is not None:
            finish_requirement()
            current_name, current_version = requirement_match.groups()
            continue
        hash_match = re.fullmatch(r"--hash=sha256:([0-9a-f]{64})(?:\s*\\)?", line)
        if hash_match is None or current_name is None:
            raise PackageError("publisher requirements are not fully pinned and hash-locked")
        current_hashes.add(hash_match.group(1))
    finish_requirement()

    publisher = packages.get(PUBLISHER_PACKAGE)
    if publisher is None or publisher[0] != PUBLISHER_VERSION:
        raise PackageError(
            f"publisher requirements must pin {PUBLISHER_PACKAGE}=={PUBLISHER_VERSION}"
        )
    if PUBLISHER_WHEEL_SHA256 not in publisher[1]:
        raise PackageError("publisher requirements are missing the audited comfy-cli wheel hash")


def _validate_provenance(
    package_root: Path,
    *,
    source_root: Path,
    version: str,
    final_repository: bool,
    require_clean_source: bool,
) -> int:
    provenance = load_provenance(package_root)
    required_provenance_keys = {
        "artifact", "generated_by", "lockfiles", "package", "schema_version", "source", "toolchain",
    }
    if set(provenance) != required_provenance_keys:
        raise PackageError("provenance top-level fields are malformed")
    if provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise PackageError("unsupported provenance schema_version")
    if provenance.get("generated_by") != "tools/build_plugin.py":
        raise PackageError("provenance generated_by is invalid")
    if provenance.get("package") != {"name": PACKAGE_NAME, "version": version}:
        raise PackageError("provenance package identity does not match pyproject.toml")

    source = provenance.get("source")
    if (
        not isinstance(source, dict)
        or set(source) != {"commit", "dirty"}
        or not isinstance(source.get("commit"), str)
        or not isinstance(source.get("dirty"), bool)
    ):
        raise PackageError("provenance source identity is malformed")
    if require_clean_source and source["dirty"]:
        raise PackageError("release provenance records a dirty source tree")
    current_source = source_identity(source_root)
    if current_source != source:
        raise PackageError("provenance source identity does not match the source checkout")
    if require_clean_source and current_source["dirty"]:
        raise PackageError("source checkout is dirty")
    if provenance.get("lockfiles") != lockfile_records(source_root):
        raise PackageError("provenance lockfile hashes do not match the source checkout")

    toolchain = provenance.get("toolchain")
    required_toolchain = {"node", "npm", "python", "python_implementation"}
    if not isinstance(toolchain, dict) or set(toolchain) != required_toolchain or not all(
        isinstance(toolchain[key], str) and toolchain[key] for key in required_toolchain
    ):
        raise PackageError("provenance toolchain record is incomplete")
    python_match = re.fullmatch(r"([0-9]+)\.([0-9]+)(?:\.[0-9]+)?", toolchain["python"])
    node_match = re.fullmatch(r"v?([0-9]+)\.([0-9]+)(?:\.[0-9]+)?", toolchain["node"])
    if python_match is None or tuple(map(int, python_match.groups())) < (3, 12):
        raise PackageError("release provenance requires Python 3.12 or newer")
    if node_match is None:
        raise PackageError("release provenance contains an invalid Node version")
    node_version = tuple(map(int, node_match.groups()))
    if not (node_version >= (24, 0) or (22, 13) <= node_version < (23, 0)):
        raise PackageError("release provenance Node version violates frontend/package.json engines")

    artifact = provenance.get("artifact")
    required_artifact_keys = {"algorithm", "files", "scope", "tree_sha256"}
    if (
        not isinstance(artifact, dict)
        or set(artifact) != required_artifact_keys
        or artifact.get("algorithm") != "sha256"
        or artifact.get("scope") != f"all generated package files except {PROVENANCE_NAME}"
        or not isinstance(artifact.get("tree_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", artifact["tree_sha256"]) is None
    ):
        raise PackageError("provenance artifact metadata is malformed")
    expected_records = artifact.get("files")
    if not isinstance(expected_records, list):
        raise PackageError("provenance artifact file list is malformed")
    expected_paths_in_order: list[str] = []
    for record in expected_records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size"}:
            raise PackageError("provenance artifact file record is malformed")
        path = record.get("path")
        digest = record.get("sha256")
        size = record.get("size")
        if (
            not isinstance(path, str)
            or not path
            or path == PROVENANCE_NAME
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise PackageError("provenance artifact list is malformed or attempts a self-hash")
        expected_paths_in_order.append(path)
    if expected_paths_in_order != sorted(set(expected_paths_in_order)):
        raise PackageError("provenance artifact paths must be unique and sorted")

    all_records = artifact_records(package_root, excluded=frozenset({PROVENANCE_NAME}))
    expected_paths = set(expected_paths_in_order)
    current_by_path = {record["path"]: record for record in all_records}
    current_generated = [
        current_by_path[path]
        for path in sorted(expected_paths)
        if path in current_by_path
    ]
    if current_generated != expected_records:
        raise PackageError("generated package file hashes, sizes or paths do not match provenance")
    if artifact.get("tree_sha256") != artifact_tree_sha256(expected_records):
        raise PackageError("provenance tree_sha256 does not match its artifact records")

    extras = set(current_by_path) - expected_paths
    allowed_extras = FINAL_REPOSITORY_EXTRAS if final_repository else set()
    if extras != allowed_extras:
        missing = sorted(allowed_extras - extras)
        unexpected = sorted(extras - allowed_extras)
        details = []
        if missing:
            details.append(f"missing final-repository metadata: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected files outside provenance: {', '.join(unexpected)}")
        raise PackageError("; ".join(details) or "package tree differs from provenance")
    return len(expected_records)


def check_package(
    package_root: Path,
    *,
    source_root: Path,
    final_repository: bool,
    require_clean_source: bool,
) -> int:
    if not package_root.is_dir():
        raise PackageError(f"package directory is missing: {package_root}")
    for relative in sorted(REQUIRED_PATHS):
        if not (package_root / relative).is_file():
            raise PackageError(f"required package file is missing: {relative}")
    for relative in sorted(REQUIRED_DIRECTORIES):
        if not (package_root / relative).is_dir():
            raise PackageError(f"required package directory is missing: {relative}")
    for relative in sorted(FORBIDDEN_BUNDLED_NODE_DIRECTORIES):
        if (package_root / relative).exists():
            raise PackageError(f"non-isolated external node pack path must not be bundled: {relative}")
    _validate_privacy_and_paths(package_root)
    if final_repository:
        _validate_final_repository_policy(package_root)
    version = _validate_manifest(package_root, source_root=source_root)
    _validate_docs(package_root)
    _validate_product_config(package_root)
    return _validate_provenance(
        package_root,
        source_root=source_root,
        version=version,
        final_repository=final_repository,
        require_clean_source=require_clean_source,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--final-repository",
        action="store_true",
        help="allow and require only the public repository-owned workflow/icons/gitignore",
    )
    parser.add_argument(
        "--require-clean-source",
        action="store_true",
        help="fail unless both the recorded and current source tree are clean",
    )
    args = parser.parse_args()
    try:
        package_dir = args.package_dir.resolve()
        count = check_package(
            package_dir,
            source_root=args.source_root.resolve(),
            final_repository=args.final_repository,
            # A final public repository is publishable by definition; never
            # permit callers to forget the clean-source requirement.
            require_clean_source=(args.require_clean_source or args.final_repository),
        )
        distribution_records = artifact_records(package_dir)
        distribution_sha256 = artifact_tree_sha256(distribution_records)
    except (OSError, UnicodeDecodeError, PackageError) as exc:
        print(f"[FAIL] {exc}")
        return 1
    mode = "final repository" if args.final_repository else "generated package"
    print(
        f"[PASS] {mode} gate verified {count} provenance-covered files; "
        f"distribution_tree_sha256={distribution_sha256}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
