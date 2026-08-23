from __future__ import annotations

"""Explicit generator for the immutable extensible-workflow phase-0 fixtures.

The regression tests import the pure prompt builder below, but never update the
checked-in files.  Maintainers must run this module deliberately after reviewing
an intentional baseline change.
"""

import argparse
import hashlib
import json
import sqlite3
from copy import deepcopy
from gzip import GzipFile
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

from directordeck.database import Database
from directordeck.native_templates import (
    NativeCompileResult,
    NativeHistoricalTake,
    NativeWorkflowUnit,
    bind_raylight_runtime_epoch,
    build_raylight_shutdown_unit,
    compile_native_timeline,
    raylight_runtime_descriptor,
)
from directordeck.schemas import (
    LoraLoaderOverrideRecord,
    RuntimeSettings,
    UnifiedTimelineDraft,
    default_settings,
    default_timeline_draft,
)
from directordeck.workflow.lora_factory import (
    LoraLoaderBindingKey,
    ResolvedLoraAdapter,
    resolve_standard_lora_adapter,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "extensible_workflow_v0"
PUBLIC_BASELINE_COMMIT = "6b959d6f73435afe2a83acef6c62fe9d812200fa"
FIXED_TIME = "2026-08-21T12:00:00+00:00"
RAY_EPOCH = 7


def render_json(value: Any) -> str:
    """The byte contract used by every textual phase-0 backend fixture."""

    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _asset(kind: str, name: str, *, slot: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": f"baseline-{kind}-{name.replace('.', '-')}",
        "name": name,
        "subfolder": "director-baseline",
        "type": "input",
        "kind": kind,
        "content_hash": "sha256:" + "a" * 64,
    }
    if kind == "video":
        value["metadata"] = {
            "duration": 12.0,
            "native_fps": 24.0,
            "frame_count": 288,
            "width": 1920,
            "height": 1080,
            "probe_method": "extensible_workflow_v0_fixture",
            "has_audio": True,
        }
    if slot is not None:
        value["slot"] = slot
    return value


def _segment(recipe: str, *, segment_id: str | None = None) -> dict[str, Any]:
    identity = segment_id or f"baseline-{recipe}"
    family = "fl2va" if recipe in {"t2v", "i2v", "fl2v"} else "ref2va"
    value: dict[str, Any] = {
        "id": identity,
        "title": f"Baseline {recipe}",
        "prompt": f"Generate the frozen {recipe} baseline.",
        "duration_seconds": 5.0,
        "enabled": True,
        "mode": family,
        "continuity": {"enabled": False, "overlap_frames": 22},
        "ref_image_size": "match",
        "audio_mode": "generate",
    }
    if family == "fl2va":
        value.update(first_image=None, last_image=None)
        if recipe in {"i2v", "fl2v"}:
            value["first_image"] = _asset("image", "first.png")
            value["prompt"] = "Begin at <Picture 1>."
        if recipe == "fl2v":
            value["last_image"] = _asset("image", "last.png")
            value["prompt"] = "Move from <Picture 1> to <Picture 2>."
    else:
        value.update(
            source_video=None,
            source_start_seconds=0.0,
            source_duration_seconds=5.0,
            source_audio_as_reference=False,
            reference_images=[],
            reference_audios=[],
            reference_videos=[],
        )
        if recipe in {"r2v", "rv2v"}:
            value["reference_images"] = [_asset("image", "reference.png", slot=0)]
            value["reference_audios"] = [_asset("audio", "voice.wav", slot=0)]
            value["reference_videos"] = [_asset("video", "motion.mp4", slot=0)]
            value["prompt"] = "Use <Picture 1>, <Video 1>, and <Audio 1>."
        if recipe in {"v2v", "rv2v"}:
            value["source_video"] = _asset("video", "source.mp4")
            value["source_start_seconds"] = 1.0
            value["source_duration_seconds"] = 5.0
            value["prompt"] = "Edit <Video 1>."
        if recipe == "rv2v":
            value["prompt"] = (
                "Edit <Video 1> using <Picture 1>, <Video 2>, and <Audio 1>."
            )
    return value


def _draft(
    recipe: str,
    *,
    segment_id: str | None = None,
    audio_mode: str = "generate",
) -> UnifiedTimelineDraft:
    value = default_timeline_draft().model_dump(mode="json")
    value.update(
        version=4,
        title=f"Extensible workflow v0: {recipe}",
        export_mode="segments",
        segments=[_segment(recipe, segment_id=segment_id)],
    )
    value["segments"][0]["audio_mode"] = audio_mode
    for family in ("fl2va", "ref2va"):
        value["sampling"][family].update(
            steps=20,
            seed=424242,
            random_seed=False,
            sampler="res_multistep",
            scheduler="simple",
            shift=12.0,
            audio_shift=3.0,
        )
    return UnifiedTimelineDraft.model_validate(value)


def _settings(
    backend: str,
    *,
    lora_family: str | None = None,
    lora_name: str | None = None,
    lora_strength: float = 1.0,
    standard_override: str | None = None,
) -> RuntimeSettings:
    value = default_settings().model_dump(mode="json")
    value["multi_gpu_enabled"] = backend == "raylight"
    for family in ("fl2va", "ref2va"):
        binding = value["models"][family]
        binding["backend"] = backend
        binding["raylight"] = {
            "gpu_select": [0, 1] if backend == "raylight" else [0],
            "ulysses_degree": 2 if backend == "raylight" else 1,
            "ring_degree": 1,
            "cfg_degree": 1,
            "dp_degree": 1,
            "fsdp": False,
            "cpu_offload": False,
        }
    if lora_family is not None:
        binding = value["models"][lora_family]
        binding.update(lora_name=lora_name, lora_strength=lora_strength)
        if standard_override is not None:
            binding["standard_lora_loader_override"] = {
                "loader": standard_override,
                "lora_name": lora_name,
                "model_filename": binding["filename"],
            }
    return RuntimeSettings.model_validate(value)


def _continuity_draft() -> UnifiedTimelineDraft:
    value = _draft("t2v").model_dump(mode="json")
    first = _segment("t2v", segment_id="baseline-continuity-first")
    second = _segment("t2v", segment_id="baseline-continuity-second")
    second["continuity"] = {"enabled": True, "overlap_frames": 22}
    value["segments"] = [first, second]
    return UnifiedTimelineDraft.model_validate(value)


def _maximum_reference_draft() -> UnifiedTimelineDraft:
    value = _draft("v2v", segment_id="baseline-maximum-reference").model_dump(
        mode="json"
    )
    segment = value["segments"][0]
    segment.update(
        prompt=(
            "Use <Picture 9>, <Video 3>, source <Audio 1>, and independent "
            "<Audio 4>."
        ),
        source_audio_as_reference=True,
        reference_images=[
            _asset("image", f"reference-{index}.png", slot=index)
            for index in range(9)
        ],
        reference_videos=[
            _asset("video", f"motion-{index}.mp4", slot=index)
            for index in range(2)
        ],
        reference_audios=[
            _asset("audio", f"voice-{index}.wav", slot=index)
            for index in range(3)
        ],
    )
    return UnifiedTimelineDraft.model_validate(value)


def _unit_document(
    unit: NativeWorkflowUnit,
    *,
    plan: dict[str, Any] | None,
) -> dict[str, Any]:
    descriptor = raylight_runtime_descriptor(unit)
    return {
        "id": unit.id,
        "family": unit.family,
        "backend": unit.backend,
        "segment_ids": list(unit.segment_ids),
        "prompt": unit.prompt,
        "node_ids": list(unit.prompt),
        "output_nodes": unit.output_nodes,
        # The phase-0 public projection is immutable.  Stage-3 exact binding
        # evidence is intentionally internal to NativeWorkflowUnit and must
        # not change the legacy compile-report bytes.
        "continuity": (
            {
                "predecessor_segment_id": unit.continuity.predecessor_segment_id,
                "overlap_frames": unit.continuity.overlap_frames,
                "load_video_node_id": unit.continuity.load_video_node_id,
                "source": unit.continuity.source,
                "historical_take_id": unit.continuity.historical_take_id,
                "resolved": unit.continuity.resolved,
            }
            if unit.continuity is not None
            else None
        ),
        "plan": plan,
        "ray_runtime_descriptor": _stable_runtime_descriptor(descriptor),
    }


def _stable_runtime_descriptor(
    descriptor: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Normalize the legacy set-built loader map for fixture serialization.

    The runtime descriptor is semantically a mapping, but the current v0
    implementation constructs ``loader_subgraph`` by iterating a set.  Hash
    randomization must not make the checked-in byte baseline process-specific.
    Node content and IDs remain unchanged; only that mapping's presentation is
    ordered by the numeric graph node IDs.
    """

    if descriptor is None:
        return None
    stable = deepcopy(descriptor)
    loader_subgraph = stable.get("loader_subgraph")
    if isinstance(loader_subgraph, dict):
        stable["loader_subgraph"] = {
            key: loader_subgraph[key]
            for key in sorted(
                loader_subgraph,
                key=lambda item: (0, int(item)) if item.isdecimal() else (1, item),
            )
        }
    return stable


def _compile_case(
    case_id: str,
    description: str,
    draft: UnifiedTimelineDraft,
    settings: RuntimeSettings,
    *,
    segment_ids: list[str] | None = None,
    historical_takes: dict[str, NativeHistoricalTake] | None = None,
) -> dict[str, Any]:
    resolved_lora_adapters: dict[str, ResolvedLoraAdapter] = {}
    for family in ("fl2va", "ref2va"):
        binding = getattr(settings.models, family)
        if binding.lora_name is None or len(binding.raylight.gpu_select) >= 2:
            continue
        exact_binding = LoraLoaderBindingKey(
            family=family,
            model_filename=binding.filename,
            lora_filename=binding.lora_name,
        )
        override = binding.standard_lora_loader_override
        overrides = (
            ()
            if override is None
            else (
                LoraLoaderOverrideRecord(
                    family=family,
                    model_filename=override.model_filename,
                    lora_filename=override.lora_name,
                    adapter_id=override.loader,
                ),
            )
        )
        resolved_lora_adapters[family] = resolve_standard_lora_adapter(
            exact_binding,
            overrides,
        )
    result = compile_native_timeline(
        draft,
        settings,
        f"fixture-{case_id}",
        segment_ids,
        historical_takes=historical_takes,
        resolved_lora_adapters=resolved_lora_adapters,
    )
    units = list(result.workflows)
    if any(unit.backend == "raylight" for unit in units):
        units = [
            bind_raylight_runtime_epoch(unit, RAY_EPOCH)
            if unit.backend == "raylight"
            else unit
            for unit in units
        ]
    return {
        "id": case_id,
        "kind": "segment_compile",
        "description": description,
        "bound_raylight_epoch": (
            RAY_EPOCH if any(unit.backend == "raylight" for unit in units) else None
        ),
        "families": list(result.families),
        "manifest": result.manifest,
        "node_policy": result.node_policy,
        "plans": list(result.plans),
        "units": [
            _unit_document(
                unit,
                plan=(result.plans[index] if index < len(result.plans) else None),
            )
            for index, unit in enumerate(units)
        ],
    }


def _raykill_case(*, with_lora: bool) -> dict[str, Any]:
    suffix = "with-lora" if with_lora else "without-lora"
    settings = _settings(
        "raylight",
        lora_family="fl2va" if with_lora else None,
        lora_name="baseline-ray-style.safetensors" if with_lora else None,
        lora_strength=0.875,
    )
    result = compile_native_timeline(
        _draft("t2v", segment_id=f"baseline-raykill-{suffix}"),
        settings,
        f"fixture-raykill-{suffix}",
    )
    source = bind_raylight_runtime_epoch(result.workflows[0], RAY_EPOCH)
    descriptor = _stable_runtime_descriptor(raylight_runtime_descriptor(source))
    assert descriptor is not None
    barrier = build_raylight_shutdown_unit(
        descriptor,
        unit_id=f"baseline-raykill-{suffix}",
    )
    return {
        "id": f"raykill-{suffix}",
        "kind": "raylight_control",
        "description": f"RayKill control prompt {suffix.replace('-', ' ')}",
        "bound_raylight_epoch": RAY_EPOCH,
        "families": [source.family],
        "manifest": None,
        "node_policy": None,
        "plans": [],
        "source_runtime_descriptor": descriptor,
        "units": [_unit_document(barrier, plan=None)],
    }


def build_native_prompt_goldens(
    *,
    skip_case_ids: Iterable[str] = (),
) -> dict[str, Any]:
    skipped = set(skip_case_ids)
    cases: list[dict[str, Any]] = []
    for backend in ("standard", "raylight"):
        for recipe in ("t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"):
            case_id = f"{backend}-{recipe}"
            if case_id in skipped:
                continue
            cases.append(
                _compile_case(
                    case_id,
                    f"{backend} {recipe} segment prompt",
                    _draft(recipe),
                    _settings(backend),
                )
            )

    lora_cases = (
        (
            "standard-lora-dedicated",
            "minimax_h3_turbo_v4_step600_ema.safetensors",
            "dedicated",
        ),
        (
            "standard-lora-model-only",
            "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
            "model_only",
        ),
        (
            "standard-lora-bypass-model-only",
            "baseline-renamed-generic.safetensors",
            "bypass_model_only",
        ),
    )
    for case_id, lora_name, override in lora_cases:
        if case_id in skipped:
            continue
        cases.append(
            _compile_case(
                case_id,
                f"Standard LoRA adapter: {case_id.removeprefix('standard-lora-')}",
                _draft("t2v", segment_id=f"baseline-{case_id}"),
                _settings(
                    "standard",
                    lora_family="fl2va",
                    lora_name=lora_name,
                    lora_strength=0.75,
                    standard_override=override,
                ),
            )
        )
    if "raylight-lora" not in skipped:
        cases.append(
            _compile_case(
                "raylight-lora",
                "RayLight fixed LoRA adapter",
                _draft("t2v", segment_id="baseline-raylight-lora"),
                _settings(
                    "raylight",
                    lora_family="fl2va",
                    lora_name="baseline-ray-style.safetensors",
                    lora_strength=0.875,
                ),
            )
        )

    historical_take = NativeHistoricalTake(
        id="baseline-historical-take",
        segment_id="baseline-continuity-first",
        output={
            "filename": "baseline-predecessor.mp4",
            "subfolder": "director-baseline",
            "type": "output",
        },
    )
    for backend in ("standard", "raylight"):
        same_run_id = f"{backend}-continuity-same-run"
        if same_run_id not in skipped:
            cases.append(
                _compile_case(
                    same_run_id,
                    f"{backend} same-run continuity",
                    _continuity_draft(),
                    _settings(backend),
                )
            )
        historical_id = f"{backend}-continuity-historical"
        if historical_id not in skipped:
            cases.append(
                _compile_case(
                    historical_id,
                    f"{backend} historical-take continuity",
                    _continuity_draft(),
                    _settings(backend),
                    segment_ids=["baseline-continuity-second"],
                    historical_takes={
                        "baseline-continuity-second": historical_take,
                    },
                )
            )

    for audio_mode, recipe in (
        ("generate", "r2v"),
        ("source", "v2v"),
        ("mute", "t2v"),
    ):
        case_id = f"audio-{audio_mode}"
        if case_id in skipped:
            continue
        cases.append(
            _compile_case(
                case_id,
                f"Standard audio mode {audio_mode}",
                _draft(
                    recipe,
                    segment_id=f"baseline-audio-{audio_mode}",
                    audio_mode=audio_mode,
                ),
                _settings("standard"),
            )
        )

    for with_lora in (False, True):
        suffix = "with-lora" if with_lora else "without-lora"
        if f"raykill-{suffix}" not in skipped:
            cases.append(_raykill_case(with_lora=with_lora))
    if "standard-maximum-reference-slots" not in skipped:
        cases.append(
            _compile_case(
                "standard-maximum-reference-slots",
                "All last legal H3 reference slots, including paired source audio",
                _maximum_reference_draft(),
                _settings("standard"),
            )
        )
    return {
        "fixture_schema": 1,
        "baseline": {
            "public_commit": PUBLIC_BASELINE_COMMIT,
            "timeline_schema": 4,
            "raylight_epoch": RAY_EPOCH,
        },
        "serialization": {
            "encoding": "utf-8",
            "ensure_ascii": False,
            "indent": 2,
            "sort_keys": False,
            "trailing_newline": True,
        },
        "cases": cases,
    }


def _database_settings() -> RuntimeSettings:
    value = _settings("standard").model_dump(mode="json")
    value["multi_gpu_enabled"] = True
    value["models"]["ref2va"].update(
        backend="raylight",
        raylight={
            "gpu_select": [0, 1],
            "ulysses_degree": 2,
            "ring_degree": 1,
            "cfg_degree": 1,
            "dp_degree": 1,
            "fsdp": False,
            "cpu_offload": False,
        },
    )
    return RuntimeSettings.model_validate(value)


def build_database_fixture(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        candidate.unlink(missing_ok=True)
    database = Database(path)
    database.initialize()

    timeline = _draft("t2v", segment_id="baseline-db-segment")
    timeline.title = "Extensible workflow v0 database"
    timeline, revision = database.validate_and_put_timeline_authority(
        timeline,
        expected_revision=0,
    )
    assert revision == 1
    settings = database.put_settings(_database_settings())
    compiled = compile_native_timeline(
        timeline,
        settings,
        "fixture-db-parent",
        ["baseline-db-segment"],
    )
    unit = compiled.workflows[0]
    output_node = unit.output_nodes["baseline-db-segment"]
    prompt_id = "caller-assigned-prompt-v0"
    database.create_job(
        {
            "id": "baseline-parent-v0",
            "mode": "timeline",
            "status": "succeeded",
            "progress": 1.0,
            "stage": "completed",
            "prompt_id": prompt_id,
            "project_id": Database.LEGACY_DEFAULT_PROJECT_ID,
            "outputs": [],
            "error": None,
            "config_snapshot": {
                "timeline": timeline.model_dump(mode="json"),
                "segment_ids": ["baseline-db-segment"],
            },
            "settings_snapshot": settings.model_dump(mode="json"),
            "prompt_snapshot": unit.prompt,
            "created_at": FIXED_TIME,
            "updated_at": FIXED_TIME,
            "started_at": FIXED_TIME,
            "completed_at": FIXED_TIME,
        }
    )
    database.create_job_child(
        {
            "id": "baseline-child-v0",
            "job_id": "baseline-parent-v0",
            "group_index": 1,
            "family": "fl2va",
            "backend": "standard",
            "segment_ids": ["baseline-db-segment"],
            "output_nodes": {"baseline-db-segment": output_node},
            "status": "succeeded",
            "progress": 1.0,
            "stage": "completed",
            "prompt_id": prompt_id,
            "outputs": [
                {
                    "node_id": output_node,
                    "filename": "baseline-db-segment.mp4",
                    "subfolder": "director-baseline",
                    "type": "output",
                }
            ],
            "error": None,
            "prompt_snapshot": unit.prompt,
            "created_at": FIXED_TIME,
            "updated_at": FIXED_TIME,
            "started_at": FIXED_TIME,
            "completed_at": FIXED_TIME,
        }
    )

    ray_result = compile_native_timeline(
        _draft("r2v", segment_id="baseline-db-ray"),
        _settings("raylight"),
        "fixture-db-ray-runtime",
    )
    ray_unit = bind_raylight_runtime_epoch(ray_result.workflows[0], RAY_EPOCH)
    descriptor = _stable_runtime_descriptor(raylight_runtime_descriptor(ray_unit))
    assert descriptor is not None
    database.put_raylight_runtime_state(
        {
            "version": 2,
            "epoch": RAY_EPOCH,
            "current": descriptor,
            "tail_prompt_id": "baseline-ray-tail-v0",
            "tail_action": "ray_unit",
            "tainted": True,
            "tail_terminal_certificate": {
                "prompt_id": "baseline-ray-tail-v0",
                "action": "ray_unit",
                "succeeded": False,
            },
        }
    )

    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE segment_takes SET id = ?, created_at = ?, completed_at = ?",
            ("baseline-take-v0", FIXED_TIME, FIXED_TIME),
        )
        for table in (
            "settings",
            "mode_drafts",
            "unified_timeline",
            "projects",
            "migration_notices",
            "assets",
            "asset_trash_batches",
            "asset_trash_document_changes",
            "jobs",
            "job_children",
            "raylight_runtime_state",
        ):
            columns = {
                str(row[1])
                for row in connection.execute(f"PRAGMA table_info({table})")
            }
            if "updated_at" in columns:
                connection.execute(f"UPDATE {table} SET updated_at = ?", (FIXED_TIME,))
            if "created_at" in columns:
                connection.execute(f"UPDATE {table} SET created_at = ?", (FIXED_TIME,))
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode = DELETE").fetchone()
        connection.execute("VACUUM")


def _decoded_rows(
    connection: sqlite3.Connection,
    query: str,
    *,
    json_fields: Iterable[str] = (),
) -> list[dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    rows: list[dict[str, Any]] = []
    for raw in connection.execute(query).fetchall():
        row = dict(raw)
        for field in json_fields:
            if row.get(field) is not None:
                row[field] = json.loads(row[field])
        rows.append(row)
    return rows


def database_projection(path: Path) -> dict[str, Any]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        schema = _decoded_rows(
            connection,
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name",
        )
        return {
            "fixture_schema": 1,
            "pragma_user_version": connection.execute("PRAGMA user_version").fetchone()[0],
            "schema": schema,
            "settings": _decoded_rows(
                connection,
                "SELECT singleton, document, revision, updated_at FROM settings",
                json_fields=("document",),
            ),
            "unified_timeline": _decoded_rows(
                connection,
                "SELECT singleton, document, revision, updated_at "
                "FROM unified_timeline",
                json_fields=("document",),
            ),
            "mode_drafts": _decoded_rows(
                connection,
                "SELECT mode, document, updated_at FROM mode_drafts ORDER BY mode",
                json_fields=("document",),
            ),
            "jobs": _decoded_rows(
                connection,
                "SELECT * FROM jobs ORDER BY id",
                json_fields=(
                    "outputs",
                    "config_snapshot",
                    "settings_snapshot",
                    "prompt_snapshot",
                ),
            ),
            "job_children": _decoded_rows(
                connection,
                "SELECT * FROM job_children ORDER BY job_id, group_index",
                json_fields=("segment_ids", "output_nodes", "outputs", "prompt_snapshot"),
            ),
            "segment_takes": _decoded_rows(
                connection,
                "SELECT * FROM segment_takes ORDER BY id",
                json_fields=("output_descriptor",),
            ),
            "raylight_runtime_state": _decoded_rows(
                connection,
                "SELECT * FROM raylight_runtime_state ORDER BY singleton",
                json_fields=("descriptor",),
            ),
            "migration_notices": _decoded_rows(
                connection,
                "SELECT * FROM migration_notices ORDER BY id",
            ),
            "row_counts": {
                table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for (table,) in connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            },
        }


def write_fixtures(fixture_dir: Path = FIXTURE_DIR) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    native_path = fixture_dir / "native_prompt_goldens.json"
    database_path = fixture_dir / "current_v4.sqlite3.gz"
    expected_path = fixture_dir / "current_v4_expected.json"
    native_path.write_text(
        render_json(build_native_prompt_goldens()),
        encoding="utf-8",
    )
    with TemporaryDirectory(prefix="directordeck-extensible-workflow-v0-") as temp_dir:
        unpacked_database_path = Path(temp_dir) / "current_v4.sqlite3"
        build_database_fixture(unpacked_database_path)
        compressed = BytesIO()
        with GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as archive:
            archive.write(unpacked_database_path.read_bytes())
        database_path.write_bytes(compressed.getvalue())
        expected_path.write_text(
            render_json(database_projection(unpacked_database_path)),
            encoding="utf-8",
        )
    manifest = {
        "fixture_schema": 1,
        "name": "extensible_workflow_v0",
        "purpose": "Read-only behavior baseline for stages 1 through 11",
        "public_baseline_commit": PUBLIC_BASELINE_COMMIT,
        "timeline_schema": 4,
        "native_prompt_case_count": len(build_native_prompt_goldens()["cases"]),
        "files": {
            native_path.name: {
                "sha256": sha256_file(native_path),
                "size_bytes": native_path.stat().st_size,
                "format": "UTF-8 JSON with significant object order and trailing newline",
            },
            database_path.name: {
                "sha256": sha256_file(database_path),
                "size_bytes": database_path.stat().st_size,
                "format": (
                    "Deterministic gzip archive of a SQLite current-v4 database; "
                    "decompress to a temporary directory before opening"
                ),
            },
            expected_path.name: {
                "sha256": sha256_file(expected_path),
                "size_bytes": expected_path.stat().st_size,
                "format": "UTF-8 JSON semantic projection with trailing newline",
            },
        },
    }
    (fixture_dir / "manifest.json").write_text(render_json(manifest), encoding="utf-8")


def verify_fixtures() -> None:
    """Regenerate in a temporary directory and compare without touching baseline files."""

    with TemporaryDirectory(prefix="directordeck-extensible-workflow-v0-verify-") as temp_dir:
        generated_dir = Path(temp_dir) / "extensible_workflow_v0"
        write_fixtures(generated_dir)
        for generated in sorted(generated_dir.iterdir()):
            checked_in = FIXTURE_DIR / generated.name
            if not checked_in.is_file():
                raise SystemExit(f"missing checked-in fixture: {checked_in}")
            if generated.read_bytes() != checked_in.read_bytes():
                raise SystemExit(
                    f"phase-0 fixture differs: {checked_in}; review the behavior change "
                    "before using --write"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="explicitly replace the checked-in baseline after architecture review",
    )
    args = parser.parse_args()
    if args.write:
        write_fixtures()
    else:
        verify_fixtures()


if __name__ == "__main__":
    main()
