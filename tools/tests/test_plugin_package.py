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
        "RayLight\nComfyUI-MiniMax-H3-Turbo\n", encoding="utf-8"
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
