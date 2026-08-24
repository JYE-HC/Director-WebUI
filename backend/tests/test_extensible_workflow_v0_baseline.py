from __future__ import annotations

import gzip
import json
from pathlib import Path
from shutil import copyfileobj

from directordeck.database import Database

from .extensible_workflow_v0_fixture_builder import (
    FIXTURE_DIR,
    PUBLIC_BASELINE_COMMIT,
    database_projection,
    sha256_file,
)


def _manifest() -> dict:
    return json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))


def test_native_prompt_goldens_remain_an_immutable_v0_baseline() -> None:
    """Validate the frozen pre-refactor evidence without regenerating it.

    Stage 7 deliberately replaces identity-permitting stock LoRA nodes with
    Director-owned strict adapters, so the current compiler must no longer be
    byte-equal to this historical v0 prompt set.  Stage-specific contract and
    prompt tests prove that bounded delta; this fixture remains immutable
    evidence for reviewing it.
    """

    path = FIXTURE_DIR / "native_prompt_goldens.json"
    expected_bytes = path.read_bytes()
    payload = json.loads(expected_bytes)
    case_ids = [case["id"] for case in payload["cases"]]
    assert len(case_ids) == len(set(case_ids)) == 26
    assert {
        f"{backend}-{recipe}"
        for backend in ("standard", "raylight")
        for recipe in ("t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v")
    } <= set(case_ids)
    assert {
        "standard-lora-dedicated",
        "standard-lora-model-only",
        "standard-lora-bypass-model-only",
        "raylight-lora",
        "standard-continuity-same-run",
        "standard-continuity-historical",
        "raylight-continuity-same-run",
        "raylight-continuity-historical",
        "audio-generate",
        "audio-source",
        "audio-mute",
        "raykill-without-lora",
        "raykill-with-lora",
        "standard-maximum-reference-slots",
    } <= set(case_ids)
    by_id = {case["id"]: case for case in payload["cases"]}
    for case in payload["cases"]:
        for unit in case["units"]:
            assert unit["node_ids"] == list(unit["prompt"])
            assert unit["output_nodes"] is not None
            assert "plan" in unit
            assert "ray_runtime_descriptor" in unit

    def node_types(case_id: str) -> set[str]:
        return {
            node["class_type"]
            for unit in by_id[case_id]["units"]
            for node in unit["prompt"].values()
        }

    assert "MiniMaxH3TurboLoRA" in node_types("standard-lora-dedicated")
    assert "LoraLoaderModelOnly" in node_types("standard-lora-model-only")
    assert "LoraLoaderBypassModelOnly" in node_types(
        "standard-lora-bypass-model-only"
    )
    assert "RayLoraLoader" in node_types("raylight-lora")
    assert node_types("raykill-without-lora") == {
        "RayInitializerAdvanced",
        "RayUNETLoader",
        "RayKill",
    }
    assert "RayLoraLoader" in node_types("raykill-with-lora")
    for backend in ("standard", "raylight"):
        assert by_id[f"{backend}-continuity-same-run"]["plans"][1][
            "continuity_source"
        ] == "same_run"
        assert by_id[f"{backend}-continuity-historical"]["plans"][0][
            "continuity_source"
        ] == "historical_take"


def test_fixture_manifest_pins_every_payload_hash() -> None:
    manifest = _manifest()
    assert manifest["public_baseline_commit"] == PUBLIC_BASELINE_COMMIT
    assert manifest["timeline_schema"] == 4
    assert manifest["native_prompt_case_count"] == 26
    expected_payloads = {
        "native_prompt_goldens.json",
        "current_v4.sqlite3.gz",
        "current_v4_expected.json",
    }
    assert set(manifest["files"]) == expected_payloads
    assert {
        path.name
        for path in FIXTURE_DIR.iterdir()
        if path.name not in {"README.md", "manifest.json"}
    } == expected_payloads
    for filename, record in manifest["files"].items():
        path = FIXTURE_DIR / filename
        assert sha256_file(path) == record["sha256"]
        assert path.stat().st_size == record["size_bytes"]


def test_current_v4_database_fixture_is_immutable_input_and_migrates_idempotently(
    tmp_path: Path,
) -> None:
    source = FIXTURE_DIR / "current_v4.sqlite3.gz"
    source_digest = sha256_file(source)
    working = tmp_path / "current_v4.sqlite3"
    with gzip.open(source, "rb") as archived, working.open("wb") as unpacked:
        copyfileobj(archived, unpacked)

    expected_projection = json.loads(
        (FIXTURE_DIR / "current_v4_expected.json").read_text(encoding="utf-8")
    )
    # Phase-0 bytes remain the exact pre-migration evidence. Stage 6 upgrades
    # only a disposable copy and never rewrites the checked-in fixture.
    assert database_projection(working) == expected_projection

    database = Database(working)
    database.initialize()
    migrated_projection = database_projection(working)
    assert migrated_projection != expected_projection
    database.initialize()
    assert database_projection(working) == migrated_projection
    assert sha256_file(source) == source_digest
    assert not (FIXTURE_DIR / "current_v4.sqlite3").exists()
    assert not Path(f"{source}-wal").exists()
    assert not Path(f"{source}-shm").exists()

    timeline, revision = database.get_timeline_authority()
    assert timeline.version == 5
    assert timeline.features.template_bundle_version == 6
    assert timeline.features.project["comfy_kitchen_attention"].enabled is False
    # Schema 4 -> 5 keeps its frozen receipt at revision 2; the independent
    # feature-bundle 4 -> 5 and 5 -> 6 migrations each advance CAS once more.
    assert revision == 4
    receipt = database.get_latest_project_migration_receipt("default")
    assert receipt is not None
    assert receipt.old_revision == 1
    assert receipt.new_revision == 2
    assert receipt.old_server_digest.value.startswith("sha256-")
    assert receipt.new_server_digest.value.startswith("sha256-")
    jobs = database.list_jobs()
    assert len(jobs) == 1
    assert jobs[0]["prompt_id"] == "caller-assigned-prompt-v0"
    children = database.list_job_children("baseline-parent-v0")
    assert len(children) == 1
    assert children[0]["segment_ids"] == ["baseline-db-segment"]
    assert len(expected_projection["mode_drafts"]) == 6
    assert expected_projection["row_counts"]["mode_drafts"] == 6
    take_projection = database_projection(working)["segment_takes"]
    assert len(take_projection) == 1
    assert take_projection[0]["id"] == "baseline-take-v0"
    ray_state = database.get_raylight_runtime_state()
    assert ray_state is not None
    assert ray_state["epoch"] == 7
    assert ray_state["tainted"] is True
    assert ray_state["tail_prompt_id"] == "baseline-ray-tail-v0"
