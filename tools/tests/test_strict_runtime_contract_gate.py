from __future__ import annotations

import subprocess
import sys
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_strict_runtime_contract_gate_executes_all_bundled_node_tests() -> None:
    result = subprocess.run(
        [sys.executable, "tools/run_strict_runtime_contract_tests.py"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    match = re.search(
        r"Strict runtime contract gate passed: (\d+)/(\d+) tests",
        output,
    )
    assert match is not None, output
    passed, collected = (int(value) for value in match.groups())
    assert passed == collected
    assert passed >= 46
