"""Run small real fits to check the revision dependencies in a new environment.

The fixed tiny budgets validate installation and score-tensor contracts, not
benchmark quality. Run this after installing requirements-revision.txt.
"""
from __future__ import annotations

import argparse
import contextlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from stcg import cmlp_scores, dynotears_scores, msvar_scores, pcmciplus_scores, rpcmci_scores


def run_smoke() -> dict[str, object]:
    rng = np.random.default_rng(20260913)
    x = rng.normal(0, 0.3, (180, 2))
    for t in range(1, len(x)):
        x[t, 0] += 0.5 * x[t - 1, 0]
        x[t, 1] += (0.8 if t < 90 else -0.8) * x[t - 1, 0] + 0.3 * x[t - 1, 1]
    calls = {
        "PCMCI+": lambda: pcmciplus_scores(x, max_lag=1),
        "MSVAR": lambda: msvar_scores(x, max_lag=1, n_regimes=2, maxiter=100, retry_maxiter=150, search_reps=1, return_info=True),
        "Regime-PCMCI": lambda: rpcmci_scores(x, max_lag=1, num_regimes=2, num_iterations=4, max_anneal=3, max_transitions=4, n_jobs=1, seed=0, return_info=True),
        "cMLP": lambda: cmlp_scores(x, max_lag=1, hidden=2, epochs=2, seed=0),
        "DYNOTEARS": lambda: dynotears_scores(x, max_lag=1, max_iter=2),
    }
    results = {}
    for name, call in calls.items():
        start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            fitted = call()
        scores, info = fitted if isinstance(fitted, tuple) else (fitted, {})
        assert scores.shape == (1, 2, 2), (name, scores.shape)
        assert np.isfinite(scores).all() and (scores >= 0).all(), name
        assert np.all(scores[:, np.arange(2), np.arange(2)] == 0), name
        assert scores.max() > 0, f"{name} returned only zero scores"
        result = {"shape": list(scores.shape), "finite": True, "max_score": float(scores.max()), "seconds": time.perf_counter() - start}
        if name == "MSVAR":
            failed = info["failed_targets"]
            assert len(failed) == 0, info
            result["failed_targets"] = failed
            result["converged"] = info["converged"]
        if name == "Regime-PCMCI":
            assert info["error_free_annealings"] > 0, info
            result["error_free_annealings"] = info["error_free_annealings"]
        results[name] = result
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    inherited = "include-system-site-packages = true" in cfg.read_text().lower() if cfg.is_file() else None
    return {"python": platform.python_version(), "platform": platform.system(),
            "isolated_environment": sys.prefix != sys.base_prefix,
            "include_system_site_packages": inherited,
            "versions": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "tigramite", "statsmodels", "ortools", "torch")},
            "torch_cuda_version": torch.version.cuda, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_smoke()
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
