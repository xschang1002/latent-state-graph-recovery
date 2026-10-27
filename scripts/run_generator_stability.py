"""Empirical stability diagnostics for the switching generator.

Each state is rescaled to spectral radius at most 0.90, but stability of every
state separately does not guarantee stability of an arbitrary switching path.
This script measures the realised trajectories instead: the maximum absolute
value, its ratio to the series standard deviation, the ratio of per-state
variances, and the number of trajectories that exceed a blow-up threshold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stcg.mechanism import MechanismConfig, generate_mechanism  # noqa: E402

OUT = ROOT / "results" / "mechanism"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--output", default=str(OUT / "generator_stability.csv"))
    args = parser.parse_args()

    rows = []
    for seed in range(args.seeds):
        dataset = generate_mechanism(MechanismConfig(seed=seed))
        x = dataset.x
        scale = float(np.std(x)) + 1e-12
        state_variances = [
            float(np.var(x[dataset.regimes == state])) for state in np.unique(dataset.regimes)
        ]
        rows.append(
            {
                "seed": seed,
                "max_abs": float(np.max(np.abs(x))),
                "max_over_std": float(np.max(np.abs(x)) / scale),
                "variance_ratio": float(max(state_variances) / max(min(state_variances), 1e-12)),
                "spectral_radius_max": float(dataset.spectral_radius.max()),
                "blowup": bool(np.max(np.abs(x)) > 8.0 * scale),
            }
        )

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = {
        "seeds": len(frame),
        "max_abs_p50": float(frame.max_abs.median()),
        "max_abs_p95": float(frame.max_abs.quantile(0.95)),
        "max_abs_max": float(frame.max_abs.max()),
        "max_over_std_p95": float(frame.max_over_std.quantile(0.95)),
        "max_over_std_max": float(frame.max_over_std.max()),
        "variance_ratio_p95": float(frame.variance_ratio.quantile(0.95)),
        "blowups": int(frame.blowup.sum()),
    }
    pd.DataFrame([summary]).to_csv(output.with_name("generator_stability_summary.csv"), index=False)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
