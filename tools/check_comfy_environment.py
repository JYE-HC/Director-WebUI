#!/usr/bin/env python3
"""Inspect the active ComfyUI Python without loading models or starting Ray."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import re
import sys
from typing import Any


failures = 0


def report(level: str, message: str) -> None:
    global failures
    print(f"[{level}] {message}")
    if level == "FAIL":
        failures += 1


def version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", value)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def distribution(
    name: str,
    minimum: str | None,
    *,
    installable: bool,
    phase: str,
) -> str | None:
    try:
        value = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        level = "WARN" if installable and phase == "preinstall" else "FAIL"
        report(level, f"Python package {name} is missing")
        return None
    if minimum and version_tuple(value) < version_tuple(minimum):
        level = "WARN" if installable and phase == "preinstall" else "FAIL"
        report(level, f"{name} {value} is older than required {minimum}")
    else:
        report("PASS", f"{name} {value}")
    return value


def import_module(name: str, *, installable: bool, phase: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception as exc:
        level = "WARN" if installable and phase == "preinstall" else "FAIL"
        report(level, f"cannot import {name}: {type(exc).__name__}: {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preinstall", "verify"), required=True)
    parser.add_argument("--require-raylight", action="store_true")
    args = parser.parse_args()

    py = sys.version_info
    if py < (3, 10):
        report("FAIL", f"Python {py.major}.{py.minor}.{py.micro} is older than 3.10")
    else:
        report("PASS", f"Python {py.major}.{py.minor}.{py.micro}")
    if py >= (3, 13):
        report("WARN", "Python 3.13+ is not part of the tested release matrix")

    distribution("torch", None, installable=False, phase=args.phase)
    distribution("comfy-kitchen", "0.2.31", installable=False, phase=args.phase)
    distribution("comfy-aimdo", "0.4.13", installable=False, phase=args.phase)
    for name, minimum in (
        ("ray", "2.48.0"),
        ("xfuser", "0.4.4"),
        ("yunchang", "0.6.0"),
        ("kernels", None),
        ("huggingface-hub", None),
        ("hf-transfer", None),
    ):
        distribution(name, minimum, installable=True, phase=args.phase)

    torch = import_module("torch", installable=False, phase=args.phase)
    import_module("ray", installable=True, phase=args.phase)
    import_module("xfuser", installable=True, phase=args.phase)
    import_module("yunchang", installable=True, phase=args.phase)
    kitchen = import_module("comfy_kitchen", installable=False, phase=args.phase)

    if kitchen is not None:
        for name in ("int8_attention", "int8_attention_is_available"):
            if not hasattr(kitchen, name):
                report("FAIL", f"comfy_kitchen does not export {name}")
        if all(
            hasattr(kitchen, name)
            for name in ("int8_attention", "int8_attention_is_available")
        ):
            report("PASS", "comfy-kitchen INT8 attention API is present")

    cuda_count = 0
    if torch is not None:
        try:
            cuda_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        except Exception as exc:
            report("WARN", f"CUDA availability probe failed: {exc}")
        if cuda_count:
            report("PASS", f"Torch sees {cuda_count} logical CUDA device(s)")
            for index in range(cuda_count):
                try:
                    capability = torch.cuda.get_device_capability(index)
                    memory_gib = torch.cuda.get_device_properties(index).total_memory / 2**30
                    report(
                        "PASS",
                        f"gpu:{index} compute capability {capability[0]}.{capability[1]}, "
                        f"{memory_gib:.1f} GiB",
                    )
                except Exception as exc:
                    report("WARN", f"gpu:{index} property probe failed: {exc}")
        else:
            report("WARN", "Torch sees no CUDA device; Standard CPU checks can run, GPU generation cannot")
        distributed = getattr(torch, "distributed", None)
        if distributed is None or not distributed.is_available():
            report("WARN", "torch.distributed is unavailable; RayLight cannot run")
        elif not distributed.is_nccl_available():
            report("WARN", "Torch NCCL backend is unavailable; CUDA RayLight cannot run")
        else:
            report("PASS", "Torch distributed NCCL backend is available")

    raylight_ready = cuda_count >= 2
    if cuda_count < 2:
        report("WARN", "RayLight needs at least two logical CUDA devices; Standard remains available")
    if kitchen is not None and hasattr(kitchen, "int8_attention_is_available") and cuda_count:
        available_devices = []
        for index in range(cuda_count):
            try:
                if bool(kitchen.int8_attention_is_available(index)):
                    available_devices.append(index)
            except Exception as exc:
                report("WARN", f"gpu:{index} comfy-kitchen availability probe failed: {exc}")
        if len(available_devices) >= 2:
            report("PASS", f"comfy-kitchen INT8 attention is available on {len(available_devices)} GPU(s)")
        else:
            raylight_ready = False
            report("WARN", "fewer than two GPUs report comfy-kitchen INT8 attention availability")
    if args.require_raylight and not raylight_ready:
        report("FAIL", "--require-raylight was requested, but the active GPU environment is not ready")

    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
