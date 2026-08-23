from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import check_plugin_package as gate
import build_plugin
from plugin_package import (
    PROVENANCE_NAME,
    PackageError,
    artifact_records,
    artifact_tree_sha256,
)


DEPENDENCIES = [
    "anyio>=4.6,<4.10",
    "fastapi>=0.116,<0.117",
    "httpx>=0.28,<1",
    "pydantic>=2.11,<3",
    "python-multipart>=0.0.20,<1",
    "uvicorn[standard]>=0.35,<1",
    "websockets>=16,<18",
]


@pytest.mark.parametrize(
    "package_name",
    (
        "DirectorDeck-Strict-Attention",
        "DirectorDeck-Strict-H3",
    ),
)
def test_director_strict_packs_are_built_and_required(package_name: str) -> None:
    assert package_name in build_plugin.BUNDLED_NODE_PACKS
    assert f"nodes/{package_name}" in gate.REQUIRED_DIRECTORIES


def test_bundled_raylight_uses_directordeck_package_identity() -> None:
    assert "raylight" in build_plugin.BUNDLED_NODE_PACKS
    assert (
        build_plugin.BUNDLED_NODE_PACK_DESTINATIONS["raylight"]
        == "DirectorDeck-RayLight"
    )
    assert "nodes/DirectorDeck-RayLight" in gate.REQUIRED_DIRECTORIES
    assert (
        "nodes/DirectorDeck-RayLight/src/directordeck_raylight"
        in gate.REQUIRED_DIRECTORIES
    )
    assert "nodes/raylight" in gate.FORBIDDEN_BUNDLED_NODE_DIRECTORIES
    assert (
        "nodes/DirectorDeck-RayLight/src/raylight"
        in gate.FORBIDDEN_BUNDLED_NODE_DIRECTORIES
    )
    assert (
        "nodes/DirectorDeck-RayLight/src/_ray_runtime_env"
        in gate.FORBIDDEN_BUNDLED_NODE_DIRECTORIES
    )
    assert build_plugin.BUNDLED_RAYLIGHT_PYTHON_PACKAGE == (
        "directordeck_raylight"
    )
    assert set(build_plugin.BUNDLED_RAYLIGHT_EXCLUDED_DIRECTORIES) == {
        "docs",
        "example_workflows",
        "tests",
    }
    for directory in build_plugin.BUNDLED_RAYLIGHT_EXCLUDED_DIRECTORIES:
        assert (
            f"nodes/DirectorDeck-RayLight/{directory}"
            in gate.FORBIDDEN_BUNDLED_NODE_DIRECTORIES
        )


def test_bundled_raylight_uses_private_python_import_namespace() -> None:
    source_root = Path(__file__).resolve().parents[2] / "custom_nodes" / "raylight"
    private_package = source_root / "src" / "directordeck_raylight"
    assert private_package.is_dir()
    assert not (source_root / "src" / "raylight").exists()

    entrypoint = (source_root / "__init__.py").read_text(encoding="utf-8")
    assert "from directordeck_raylight.nodes import" in entrypoint
    metadata = (source_root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "directordeck-raylight"' in metadata
    assert 'version = "1.8.0+director.1"' in metadata
    assert "[tool.comfy]" not in metadata

    legacy_import = re.compile(r"(?m)^\s*(?:from|import)\s+raylight(?:\b|\.)")
    legacy_module_lookup = re.compile(
        r"(?:sys\.modules\[|ModuleType\(|import_module\(|__import__\()"
        r"\s*[\"']raylight(?:[\"']|\.)"
    )
    for path in (source_root / "__init__.py", *private_package.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert legacy_import.search(source) is None, path
        assert legacy_module_lookup.search(source) is None, path


@pytest.mark.parametrize(
    "package_name",
    (
        "ComfyUI-MiniMax-H3-Turbo",
        "DirectorDeck-Strict-LoRA",
    ),
)
def test_externally_owned_lora_packs_are_not_bundled(package_name: str) -> None:
    relative = f"nodes/{package_name}"
    assert package_name not in build_plugin.BUNDLED_NODE_PACKS
    assert relative not in gate.REQUIRED_DIRECTORIES
    assert relative in gate.FORBIDDEN_BUNDLED_NODE_DIRECTORIES
    assert not any(path.startswith(f"{relative}/") for path in gate.REQUIRED_PATHS)


def test_release_manifest_uses_advisory_comfyui_floor_and_only_owned_node_fork() -> None:
    manifest = json.loads(
        (Path(__file__).resolve().parents[2] / "release-manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["comfyui"] == {
        "recommended_minimum_version": "0.33.0",
        "enforcement": "warning_only",
    }
    assert [node["name"] for node in manifest["custom_nodes"]] == ["raylight"]
    assert manifest["custom_nodes"][0]["packaged_path"] == (
        "nodes/DirectorDeck-RayLight"
    )
    assert manifest["custom_nodes"][0]["python_package"] == (
        "directordeck_raylight"
    )
    assert manifest["custom_nodes"][0]["version"] == (
        "1.8.0+director.1"
    )


def _manifest(dependencies: list[str]) -> str:
    rendered = "\n".join(f'    "{dependency}",' for dependency in dependencies)
    return f'''[project]
name = "director-deck"
version = "0.2.5"
readme = "README.md"
license = "GPL-3.0-only"
requires-python = ">=3.12"
dependencies = [
{rendered}
]

[project.urls]
Repository = "https://github.com/JYE-HC/DirectorDeck"

[tool.comfy]
PublisherId = "jye-hc"
DisplayName = "DirectorDeck"
Icon = "https://raw.githubusercontent.com/JYE-HC/DirectorDeck/main/icon.png"
Banner = "https://raw.githubusercontent.com/JYE-HC/DirectorDeck/main/banner.png"
'''


def test_manifest_requires_exact_dependency_equivalence(tmp_path: Path) -> None:
    package = tmp_path / "package"
    source = tmp_path / "source"
    package.mkdir()
    source.mkdir()
    (package / "pyproject.toml").write_text(_manifest(DEPENDENCIES), encoding="utf-8")
    (source / "pyproject.toml").write_text(_manifest(DEPENDENCIES), encoding="utf-8")
    (package / "requirements.txt").write_text("\n".join(DEPENDENCIES) + "\n", encoding="utf-8")

    assert gate._validate_manifest(package, source_root=source) == "0.2.5"

    (package / "requirements.txt").write_text(
        "\n".join([*DEPENDENCIES[:-1], "websockets>=16,<19"]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PackageError, match="exactly equivalent"):
        gate._validate_manifest(package, source_root=source)


def test_docs_reject_legacy_route_and_accept_release_policy(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "Open `/directordeck/`; DB: `user/directordeck/database/directordeck.sqlite3`.",
        encoding="utf-8",
    )
    (tmp_path / "SECURITY.md").write_text(
        "There is no\nbuilt-in login. Use TLS; never expose to the public Internet. "
        "Route /directordeck/.",
        encoding="utf-8",
    )
    (tmp_path / "LICENSE").write_text("GNU GENERAL PUBLIC LICENSE\nVersion 3\n", encoding="utf-8")
    (tmp_path / "THIRD_PARTY_NOTICES.md").write_text(
        "RayLight\n", encoding="utf-8"
    )
    gate._validate_docs(tmp_path)

    (tmp_path / "README.md").write_text(
        "Legacy `/director/`; DB: `user/director/database/`.", encoding="utf-8"
    )
    with pytest.raises(PackageError, match="legacy|canonical"):
        gate._validate_docs(tmp_path)

def test_privacy_gate_rejects_private_paths_and_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_note = tmp_path / "AGENTS.md"
    private_note.write_text("internal instructions\n", encoding="utf-8")
    with pytest.raises(PackageError, match="denied release path"):
        gate._validate_privacy_and_paths(tmp_path)

    private_note.unlink()
    monkeypatch.setattr(
        gate,
        "PRIVATE_PATTERNS",
        (re.compile(r"INTERNAL_FIXTURE_PATH"),),
    )
    (tmp_path / "diagnostic.txt").write_text(
        "database: INTERNAL_FIXTURE_PATH\n", encoding="utf-8"
    )
    with pytest.raises(PackageError, match="private path"):
        gate._validate_privacy_and_paths(tmp_path)


def _write_test_provenance(package: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    source = {"commit": "a" * 40, "dirty": False}
    records = artifact_records(package, excluded=frozenset({PROVENANCE_NAME}))
    provenance = {
        "schema_version": 1,
        "generated_by": "tools/build_plugin.py",
        "package": {"name": "director-deck", "version": "0.2.5"},
        "source": source,
        "lockfiles": [],
        "toolchain": {
            "node": "v22.13.1",
            "npm": "10.9.2",
            "python": "3.12.8",
            "python_implementation": "CPython",
        },
        "artifact": {
            "algorithm": "sha256",
            "scope": f"all generated package files except {PROVENANCE_NAME}",
            "tree_sha256": artifact_tree_sha256(records),
            "files": records,
        },
    }
    (package / PROVENANCE_NAME).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return source, records


def test_provenance_detects_tampering_and_never_self_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    payload = package / "payload.js"
    payload.write_text("stable payload\n", encoding="utf-8")
    source, records = _write_test_provenance(package)
    monkeypatch.setattr(gate, "source_identity", lambda _root: source)
    monkeypatch.setattr(gate, "lockfile_records", lambda _root: [])

    assert all(record["path"] != PROVENANCE_NAME for record in records)
    assert gate._validate_provenance(
        package,
        source_root=tmp_path,
        version="0.2.5",
        final_repository=False,
        require_clean_source=True,
    ) == 1

    payload.write_text("tampered payload\n", encoding="utf-8")
    with pytest.raises(PackageError, match="hashes"):
        gate._validate_provenance(
            package,
            source_root=tmp_path,
            version="0.2.5",
            final_repository=False,
            require_clean_source=True,
        )


def test_final_repository_allows_only_declared_metadata_extras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "payload.py").write_text("VALUE = 1\n", encoding="utf-8")
    source, _records = _write_test_provenance(package)
    monkeypatch.setattr(gate, "source_identity", lambda _root: source)
    monkeypatch.setattr(gate, "lockfile_records", lambda _root: [])
    for relative in gate.FINAL_REPOSITORY_EXTRAS:
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"release metadata\n")

    assert gate._validate_provenance(
        package,
        source_root=tmp_path,
        version="0.2.5",
        final_repository=True,
        require_clean_source=True,
    ) == 1

    (package / "private-notes.txt").write_text("must not ship\n", encoding="utf-8")
    with pytest.raises(PackageError, match="unexpected files"):
        gate._validate_provenance(
            package,
            source_root=tmp_path,
            version="0.2.5",
            final_repository=True,
            require_clean_source=True,
        )


def test_final_repository_policy_requires_sha_actions_and_hashed_publisher(
    tmp_path: Path,
) -> None:
    workflow = tmp_path / ".github/workflows/publish.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "on:\n  workflow_dispatch:\npermissions:\n  contents: read\nsteps:\n"
        + "  - uses: actions/checkout@"
        + "a" * 40
        + "\n"
        + "  - run: test \"$GITHUB_REF\" = \"refs/heads/main\"\n"
        + "  - run: python check_plugin_package.py --final-repository --require-clean-source\n"
        + "  - run: python -m pip install --only-binary=:all: --require-hashes -r lock.txt\n"
        + "  - run: python -m pip check\n"
        + "  - run: python -c \"import importlib.metadata; importlib.metadata.version('comfy-cli')\"\n",
        encoding="utf-8",
    )
    requirements = tmp_path / gate.PUBLISHER_REQUIREMENTS
    requirements.write_text(
        f"{gate.PUBLISHER_PACKAGE}=={gate.PUBLISHER_VERSION} {chr(92)}\n"
        f"    --hash=sha256:{gate.PUBLISHER_WHEEL_SHA256}\n",
        encoding="utf-8",
    )

    gate._validate_final_repository_policy(tmp_path)

    workflow.write_text(
        workflow.read_text(encoding="utf-8").replace(
            "actions/checkout@" + "a" * 40,
            "actions/checkout@main",
        ),
        encoding="utf-8",
    )
    with pytest.raises(PackageError, match="full commit SHAs"):
        gate._validate_final_repository_policy(tmp_path)

    workflow.write_text(
        workflow.read_text(encoding="utf-8").replace(
            "actions/checkout@main",
            "actions/checkout@" + "a" * 40,
        ),
        encoding="utf-8",
    )
    requirements.write_text(
        f"{gate.PUBLISHER_PACKAGE}=={gate.PUBLISHER_VERSION} {chr(92)}\n"
        f"    --hash=sha256:{'b' * 64}\n",
        encoding="utf-8",
    )
    with pytest.raises(PackageError, match="audited comfy-cli wheel hash"):
        gate._validate_final_repository_policy(tmp_path)
