#!/usr/bin/env python3
"""Non-mutating online ComfyUI capability probe (cancel uses an unknown UUID)."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


async def check(url: str) -> int:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "backend"))
    from director.comfy import ComfyClient, ComfyError

    try:
        report = await ComfyClient(url).capabilities()
    except ComfyError as exc:
        print(f"[FAIL] online ComfyUI probe failed: {exc}")
        return 5
    print(f"[PASS] ComfyUI is online; latency {report.get('latency_ms')} ms")
    if not report.get("supports_cancel"):
        print("[FAIL] ComfyUI does not expose the required atomic job-cancel contract")
        return 5
    for name in ("standard", "raylight"):
        backend = (report.get("execution_backends") or {}).get(name, {})
        level = "PASS" if backend.get("available") else "WARN"
        print(f"[{level}] {name} backend available={bool(backend.get('available'))}")
        for issue in backend.get("contract_issues") or []:
            print(f"[WARN] {name}: {issue}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    return asyncio.run(check(parser.parse_args().url))


if __name__ == "__main__":
    raise SystemExit(main())
