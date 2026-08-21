#!/usr/bin/env python3
"""Compare projected and legacy full-snapshot Director task-list reads.

The fixture is synthetic and contains no user data. It deliberately measures
only the server-side synchronous path (page read, child read, snapshot
currentness and public Pydantic projection, summary and JSON serialization),
not ASGI transport, live-preview lookup, or browser rendering. Results are
observations, not pass/fail performance thresholds.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from directordeck.app import _job_read  # noqa: E402
from directordeck.database import Database  # noqa: E402
from directordeck.schemas import (  # noqa: E402
    RuntimeSettings,
    UnifiedTimelineDraft,
    default_settings,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parent_count(value: str) -> int:
    parsed = positive_int(value)
    if parsed > 256:
        raise argparse.ArgumentTypeError("parent count cannot exceed API page limit 256")
    return parsed


def child_count(value: str) -> int:
    parsed = positive_int(value)
    if parsed > 128:
        raise argparse.ArgumentTypeError(
            "children per parent cannot exceed timeline segment limit 128"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parents",
        nargs="+",
        type=parent_count,
        default=[16, 64, 256],
        help="parent-job scales to benchmark (default: 16 64 256)",
    )
    parser.add_argument(
        "--children-per-parent",
        type=child_count,
        default=10,
        help="child rows per parent (default: 10)",
    )
    parser.add_argument(
        "--prompt-kib",
        type=positive_int,
        default=16,
        help="approximate private prompt snapshot size per row (default: 16 KiB)",
    )
    parser.add_argument(
        "--warmups",
        type=positive_int,
        default=3,
        help="unreported warm-up reads per scale (default: 3)",
    )
    parser.add_argument(
        "--rounds",
        type=positive_int,
        default=7,
        help="reported reads per scale (default: 7)",
    )
    return parser.parse_args()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def synthetic_prompt_snapshot(prompt_kib: int) -> str:
    target_bytes = prompt_kib * 1024
    envelope = {
        "1": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"benchmark_payload": ""},
        }
    }
    encoded_without_payload = compact_json(envelope)
    envelope["1"]["inputs"]["benchmark_payload"] = "x" * max(
        1, target_bytes - len(encoded_without_payload.encode("utf-8"))
    )
    return compact_json(envelope)


def synthetic_timeline(children_per_parent: int) -> UnifiedTimelineDraft:
    """Build the fixture through the same strict model used by production."""

    return UnifiedTimelineDraft.model_validate(
        {
            "version": 4,
            "title": "Benchmark project",
            "segments": [
                {
                    "id": f"segment-{child_index:03d}",
                    "mode": "fl2va",
                    "title": f"Benchmark segment {child_index + 1}",
                    "prompt": "Synthetic benchmark prompt",
                    "ref_image_size": "match",
                    "audio_mode": "generate",
                }
                for child_index in range(children_per_parent)
            ],
        }
    )


def populate_database(
    path: Path,
    *,
    parents: int,
    children_per_parent: int,
    prompt_kib: int,
) -> tuple[Database, int, UnifiedTimelineDraft, RuntimeSettings]:
    database = Database(path)
    database.initialize()
    prompt_snapshot = synthetic_prompt_snapshot(prompt_kib)
    current_timeline = synthetic_timeline(children_per_parent)
    current_settings = default_settings()
    timeline_document = current_timeline.model_dump(mode="json")
    settings_document = current_settings.model_dump(mode="json")
    settings_snapshot = compact_json(settings_document)
    segment_ids = [segment.id for segment in current_timeline.segments]
    config_snapshot = compact_json(
        {
            "timeline": timeline_document,
            "segment_ids": segment_ids,
        }
    )
    job_rows: list[tuple[Any, ...]] = []
    child_rows: list[tuple[Any, ...]] = []

    for parent_index in range(parents):
        job_id = f"benchmark-job-{parent_index:04d}"
        timestamp = f"2026-08-20T00:00:00.{parent_index:06d}+00:00"
        job_rows.append(
            (
                job_id,
                "timeline",
                "succeeded",
                1.0,
                "completed",
                None,
                "default",
                "[]",
                None,
                config_snapshot,
                settings_snapshot,
                prompt_snapshot,
                timestamp,
                timestamp,
                timestamp,
                timestamp,
            )
        )
        for child_index, segment_id in enumerate(segment_ids):
            prompt_id = f"benchmark-prompt-{parent_index:04d}-{child_index:02d}"
            child_rows.append(
                (
                    prompt_id,
                    job_id,
                    child_index,
                    "fl2va",
                    "standard",
                    compact_json([segment_id]),
                    compact_json({segment_id: "99"}),
                    "succeeded",
                    1.0,
                    "completed",
                    prompt_id,
                    "[]",
                    None,
                    prompt_snapshot,
                    timestamp,
                    timestamp,
                    timestamp,
                    timestamp,
                )
            )

    with database.connect() as connection:
        connection.executemany(
            "INSERT INTO jobs("
            "id, mode, status, progress, stage, prompt_id, project_id, outputs, "
            "error, config_snapshot, settings_snapshot, prompt_snapshot, "
            "created_at, updated_at, started_at, completed_at"
            ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            job_rows,
        )
        connection.executemany(
            "INSERT INTO job_children("
            "id, job_id, group_index, family, backend, segment_ids, output_nodes, "
            "status, progress, stage, prompt_id, outputs, error, prompt_snapshot, "
            "created_at, updated_at, started_at, completed_at"
            ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            child_rows,
        )
    with database.connect() as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return (
        database,
        len(prompt_snapshot.encode("utf-8")),
        current_timeline,
        current_settings,
    )


def milliseconds(start: int, end: int) -> float:
    return (end - start) / 1_000_000


PROJECTED_PATH = "projected"
LEGACY_PATH = "legacy_full_snapshot_simulation"


def legacy_list_jobs_page(
    database: Database, *, parents: int
) -> tuple[list[dict[str, Any]], int]:
    """Simulate the pre-projection parent query and JSON decoding."""

    with database.connect() as connection:
        total = int(connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])
        rows = connection.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC, id LIMIT ? OFFSET 0",
            (parents,),
        ).fetchall()
    return [database._job_row(row) for row in rows], total


def legacy_list_job_children_for_jobs(
    database: Database, job_ids: list[str]
) -> dict[str, list[dict[str, Any]]]:
    """Simulate the pre-projection child query and prompt JSON decoding."""

    children = {job_id: [] for job_id in job_ids}
    if not job_ids:
        return children
    placeholders = ",".join("?" for _ in job_ids)
    with database.connect() as connection:
        rows = connection.execute(
            f"SELECT * FROM job_children WHERE job_id IN ({placeholders}) "
            "ORDER BY job_id, group_index",
            tuple(job_ids),
        ).fetchall()
    for row in rows:
        child = database._job_child_row(row)
        children[str(child["job_id"])].append(child)
    return children


def read_once(
    database: Database,
    *,
    parents: int,
    path: str,
    current_timeline: UnifiedTimelineDraft,
    current_settings: RuntimeSettings,
) -> dict[str, float | int]:
    started = time.perf_counter_ns()
    if path == PROJECTED_PATH:
        page, total = database.list_jobs_page(
            limit=parents,
            sort_by="created_at",
            sort_order="desc",
        )
    elif path == LEGACY_PATH:
        page, total = legacy_list_jobs_page(database, parents=parents)
    else:
        raise ValueError(f"unsupported benchmark path: {path}")
    page_finished = time.perf_counter_ns()
    job_ids = [str(job["id"]) for job in page]
    if path == PROJECTED_PATH:
        children_by_job = database.list_job_children_for_jobs(job_ids)
    else:
        children_by_job = legacy_list_job_children_for_jobs(database, job_ids)
    children_finished = time.perf_counter_ns()

    public_jobs = []
    current_jobs = 0
    prompt_snapshot_rows = 0
    for job in page:
        job["children"] = children_by_job[str(job["id"])]
        parent_has_prompt = "prompt_snapshot" in job
        child_prompt_rows = sum(
            "prompt_snapshot" in child for child in job["children"]
        )
        if path == PROJECTED_PATH and (parent_has_prompt or child_prompt_rows):
            raise RuntimeError("projected task-list path loaded prompt_snapshot")
        if path == LEGACY_PATH and (
            not parent_has_prompt or child_prompt_rows != len(job["children"])
        ):
            raise RuntimeError("legacy simulation did not load every prompt_snapshot")
        prompt_snapshot_rows += int(parent_has_prompt) + child_prompt_rows

        config_snapshot = job.get("config_snapshot")
        if not isinstance(config_snapshot, dict):
            raise RuntimeError("benchmark config snapshot is not an object")
        snapshot_timeline = UnifiedTimelineDraft.model_validate(
            config_snapshot.get("timeline")
        )
        snapshot_settings = RuntimeSettings.model_validate(
            job.get("settings_snapshot")
        )
        current_project = (
            snapshot_timeline.model_dump(mode="json")
            == current_timeline.model_dump(mode="json")
        )
        current_snapshot = current_project and (
            snapshot_settings.model_dump(mode="json")
            == current_settings.model_dump(mode="json")
        )
        current_jobs += int(current_snapshot)
        public_jobs.append(
            _job_read(
                job,
                current_snapshot=current_snapshot,
                current_project=current_project,
            ).model_dump(mode="json")
        )
    projection_finished = time.perf_counter_ns()

    summary = database.job_status_summary()
    payload = compact_json(
        {
            "jobs": public_jobs,
            "total": total,
            "limit": parents,
            "offset": 0,
            "has_more": False,
            "summary": summary,
        }
    )
    finished = time.perf_counter_ns()
    child_count = sum(len(children) for children in children_by_job.values())
    if total != parents or len(page) != parents:
        raise RuntimeError("benchmark parent fixture was not read completely")
    if current_jobs != parents:
        raise RuntimeError("valid benchmark snapshots did not compare as current")

    return {
        "page_ms": milliseconds(started, page_finished),
        "children_ms": milliseconds(page_finished, children_finished),
        "projection_ms": milliseconds(children_finished, projection_finished),
        "summary_serialization_ms": milliseconds(projection_finished, finished),
        "total_ms": milliseconds(started, finished),
        "children": child_count,
        "current_jobs": current_jobs,
        "prompt_snapshot_rows": prompt_snapshot_rows,
        "response_bytes": len(payload.encode("utf-8")),
    }


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "median": round(statistics.median(values), 3),
        "p95": round(percentile_nearest_rank(values, 0.95), 3),
        "samples": [round(value, 6) for value in values],
    }


def ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 3) if denominator > 0 else None


def alternating_paths(iteration: int) -> tuple[str, str]:
    if iteration % 2 == 0:
        return PROJECTED_PATH, LEGACY_PATH
    return LEGACY_PATH, PROJECTED_PATH


def benchmark_scale(
    *,
    parents: int,
    children_per_parent: int,
    prompt_kib: int,
    warmups: int,
    rounds: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="director-job-read-benchmark-") as directory:
        database_path = Path(directory) / "benchmark.sqlite3"
        (
            database,
            prompt_snapshot_bytes,
            current_timeline,
            current_settings,
        ) = populate_database(
            database_path,
            parents=parents,
            children_per_parent=children_per_parent,
            prompt_kib=prompt_kib,
        )
        for iteration in range(warmups):
            for path in alternating_paths(iteration):
                gc.collect()
                read_once(
                    database,
                    parents=parents,
                    path=path,
                    current_timeline=current_timeline,
                    current_settings=current_settings,
                )

        observations: dict[str, list[dict[str, float | int]]] = {
            PROJECTED_PATH: [],
            LEGACY_PATH: [],
        }
        for iteration in range(rounds):
            for path in alternating_paths(iteration):
                gc.collect()
                observations[path].append(
                    read_once(
                        database,
                        parents=parents,
                        path=path,
                        current_timeline=current_timeline,
                        current_settings=current_settings,
                    )
                )

        metric_names = (
            "page_ms",
            "children_ms",
            "projection_ms",
            "summary_serialization_ms",
            "total_ms",
        )
        timings = {
            path: {
                name: summarize(
                    [float(observation[name]) for observation in observations[path]]
                )
                for name in metric_names
            }
            for path in (PROJECTED_PATH, LEGACY_PATH)
        }
        projected_total = [
            float(observation["total_ms"])
            for observation in observations[PROJECTED_PATH]
        ]
        legacy_total = [
            float(observation["total_ms"])
            for observation in observations[LEGACY_PATH]
        ]
        response_sizes = {
            int(observation["response_bytes"])
            for path_observations in observations.values()
            for observation in path_observations
        }
        if len(response_sizes) != 1:
            raise RuntimeError("benchmark paths produced different response sizes")
        expected_legacy_prompt_rows = parents * (children_per_parent + 1)
        if any(
            int(observation["prompt_snapshot_rows"]) != 0
            for observation in observations[PROJECTED_PATH]
        ):
            raise RuntimeError("projected path decoded a prompt snapshot")
        if any(
            int(observation["prompt_snapshot_rows"])
            != expected_legacy_prompt_rows
            for observation in observations[LEGACY_PATH]
        ):
            raise RuntimeError("legacy simulation prompt-snapshot count changed")
        return {
            "parents": parents,
            "children_per_parent": children_per_parent,
            "children": observations[PROJECTED_PATH][-1]["children"],
            "prompt_snapshot_bytes_per_row": prompt_snapshot_bytes,
            "database_bytes": database_path.stat().st_size,
            "response_bytes": response_sizes.pop(),
            "fixture": {
                "timeline": "UnifiedTimelineDraft v4",
                "settings": "default_settings()",
                "current_jobs_per_read": parents,
            },
            "decoded_prompt_snapshot_rows_per_read": {
                PROJECTED_PATH: 0,
                LEGACY_PATH: expected_legacy_prompt_rows,
            },
            "timings_ms": timings,
            "speedup_projected_vs_legacy": {
                "basis": "legacy time / projected time; values above 1 favor projection",
                "median_total_x": ratio(
                    statistics.median(legacy_total),
                    statistics.median(projected_total),
                ),
                "p95_total_x": ratio(
                    percentile_nearest_rank(legacy_total, 0.95),
                    percentile_nearest_rank(projected_total, 0.95),
                ),
                "paired_total_samples_x": [
                    ratio(legacy, projected)
                    for projected, legacy in zip(projected_total, legacy_total, strict=True)
                ],
            },
        }


def main() -> int:
    args = parse_args()
    scales = list(dict.fromkeys(args.parents))
    report = {
        "benchmark": "director-task-list-read-path-v2",
        "scope": (
            "SQLite page and child reads, timeline/settings Pydantic currentness, "
            "public projection, status summary and JSON serialization; excludes "
            "Request/ASGI transport, live-preview lookup and browser rendering"
        ),
        "paths": {
            PROJECTED_PATH: (
                "production list projections; excludes parent and child "
                "prompt_snapshot columns"
            ),
            LEGACY_PATH: (
                "benchmark-local SELECT * simulation of the former list path; "
                "decodes parent and child prompt_snapshot JSON"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "parameters": {
            "parent_scales": scales,
            "children_per_parent": args.children_per_parent,
            "prompt_kib": args.prompt_kib,
            "warmups": args.warmups,
            "rounds": args.rounds,
        },
        "results": [
            benchmark_scale(
                parents=parents,
                children_per_parent=args.children_per_parent,
                prompt_kib=args.prompt_kib,
                warmups=args.warmups,
                rounds=args.rounds,
            )
            for parents in scales
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
