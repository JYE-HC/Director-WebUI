#!/usr/bin/env python3
"""Load only the bundled nodes and inspect their live ComfyUI schemas."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any


def choices(value: Any) -> Any:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    options = value[0]
    if (
        options == "COMBO"
        and len(value) > 1
        and isinstance(value[1], dict)
    ):
        return value[1].get("options")
    return options


async def verify(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    comfy_root = args.comfyui_root.expanduser().resolve()
    raylight_root = args.raylight_root.expanduser().resolve()
    turbo_root = args.turbo_root.expanduser().resolve()
    sys.path.insert(0, str(project_root / "backend"))
    sys.path.insert(0, str(comfy_root))

    import comfy.cli_args

    comfy.cli_args.args.cpu = True

    import nodes
    import server
    from director.comfy import ComfyClient

    server.PromptServer(asyncio.get_running_loop())
    await nodes.init_public_apis()
    builtin_failures = await nodes.init_builtin_extra_nodes()
    if builtin_failures:
        print(f"[WARN] unrelated built-in nodes failed to import: {builtin_failures}")

    base_nodes = set(nodes.NODE_CLASS_MAPPINGS)
    if not await nodes.load_custom_node(
        str(raylight_root), base_nodes, module_parent="custom_nodes"
    ):
        print("[FAIL] bundled RayLight failed to register")
        return 5
    if not await nodes.load_custom_node(
        str(turbo_root), set(nodes.NODE_CLASS_MAPPINGS), module_parent="custom_nodes"
    ):
        print("[FAIL] bundled MiniMax H3 Turbo failed to register")
        return 5

    failures: list[str] = []
    required = set(
        (*ComfyClient.STANDARD_REQUIRED_NODES, *ComfyClient.CONTINUITY_REQUIRED_NODES)
    )
    required.update(ComfyClient.RAYLIGHT_REQUIRED_NODES)
    required.update(ComfyClient.RAYLIGHT_LORA_REQUIRED_NODES)
    missing = sorted(required - set(nodes.NODE_CLASS_MAPPINGS))
    if missing:
        failures.append("missing required nodes: " + ", ".join(missing))

    initializer = nodes.NODE_CLASS_MAPPINGS.get("RayInitializerAdvanced")
    if initializer is not None:
        try:
            issues = ComfyClient.raylight_initializer_contract_issues(
                {"input": initializer.INPUT_TYPES()}
            )
        except Exception as exc:
            issues = [f"initializer INPUT_TYPES failed: {exc}"]
        failures.extend(f"RayInitializerAdvanced: {issue}" for issue in issues)

    scheduler = nodes.NODE_CLASS_MAPPINGS.get("RayBasicScheduler")
    if scheduler is not None:
        try:
            scheduler_options = choices(scheduler.INPUT_TYPES()["required"]["scheduler"])
        except Exception as exc:
            failures.append(f"RayBasicScheduler INPUT_TYPES failed: {exc}")
        else:
            if not isinstance(scheduler_options, (list, tuple)) or "beta" not in scheduler_options:
                failures.append("RayBasicScheduler.scheduler does not offer beta")

    ray_kill = nodes.NODE_CLASS_MAPPINGS.get("RayKill")
    if ray_kill is not None:
        try:
            kill_required = ray_kill.INPUT_TYPES().get("required", {})
        except Exception as exc:
            failures.append(f"RayKill INPUT_TYPES failed: {exc}")
        else:
            if "RAY_ACTORS" not in str(kill_required.get("ray_actors")):
                failures.append("RayKill.ray_actors is not a RAY_ACTORS input")

    expected_custom = {
        "RayInitializerAdvanced": "custom_nodes.raylight",
        "RayKill": "custom_nodes.raylight",
        "MiniMaxH3TurboLoRA": "custom_nodes.ComfyUI-MiniMax-H3-Turbo",
    }
    for name, expected in expected_custom.items():
        node_class = nodes.NODE_CLASS_MAPPINGS.get(name)
        actual = str(getattr(node_class, "RELATIVE_PYTHON_MODULE", ""))
        if node_class is not None and actual != expected:
            failures.append(f"{name} provenance is {actual or 'unknown'}, expected {expected}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 5
    print(f"[PASS] ComfyUI registry contains {len(required)} required node classes")
    print("[PASS] Director RayLight initializer, scheduler, barrier and provenance contracts")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument("--raylight-root", type=Path, required=True)
    parser.add_argument("--turbo-root", type=Path, required=True)
    return asyncio.run(verify(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
