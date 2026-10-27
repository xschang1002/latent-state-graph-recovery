"""Subprocess smoke tests for the fast command-line entry points.

Each invocation runs the repository's own Python interpreter on a tiny budget
and asserts a clean exit code plus the expected output artifacts. Total added
runtime is designed to stay well under a minute.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Repo-local scratch space (gitignored), matching tests/test_real_data_entry.py;
# the OS temp dir is not usable for pytest tmp_path in this environment.
TEST_TMP = ROOT / "experiments" / "logs" / "test_cli_smoke"
SCRIPT_TIMEOUT = 120


def run_script(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=SCRIPT_TIMEOUT,
    )


def test_run_synthetic_sanity_produces_output_file():
    TEST_TMP.mkdir(parents=True, exist_ok=True)
    output = TEST_TMP / "sanity.npz"
    result = run_script(
        "run_synthetic_sanity.py",
        "--n-nodes", "4",
        "--timesteps", "150",
        "--seed", "3",
        "--output", str(output),
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert output.is_file()

    summary = json.loads(result.stdout)
    assert summary["seed"] == 3
    assert summary["x_shape"] == [150, 4]
    assert summary["true_lagged_edges"] > 0
    assert 0.0 <= summary["lagged_correlation_edge_auc"] <= 1.0

    with np.load(output) as archive:
        x_shape = archive["x"].shape
        truth_shape = archive["edge_truth"].shape
    assert x_shape == (150, 4)
    assert truth_shape == (3, 4, 4)


def test_run_synthetic_recovery_help_exits_zero():
    result = run_script("run_synthetic_recovery.py", "--help")

    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "usage" in result.stdout.lower()


def test_run_stcg_ablation_help_exits_zero():
    result = run_script("run_stcg_ablation.py", "--help")

    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "usage" in result.stdout.lower()


def test_run_downstream_forecasting_help_exits_zero():
    result = run_script("run_downstream_forecasting.py", "--help")

    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "usage" in result.stdout.lower()
