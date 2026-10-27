"""Gate link: enumerate every bipartition on the real 7-window scores.

The geometric lower bound (Section ``sec:gate-geometry``) constrains the
*optimal* bipartition, not the partition K-means actually returns. This script
closes that gap on the switching-core operating point: for each of the twenty
seeds it rebuilds the exact clustering features, enumerates all 63 inequivalent
non-empty bipartitions of the seven windows (point 0 pinned to group 0 to remove
the label swap), and compares the optimal relative-inertia improvement
``Delta_optimal`` against the K-means-returned ``Delta_code`` and the 0.12 gate.

Quantities match the implementation in ``self_supervised._infer_state_labels``:
``scaled = (features - mean) / (std + 1e-8)`` and
``Delta = 1 - SSE_2 / SSE_1`` where ``SSE_1`` is the total scatter of ``scaled``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stcg import STCGStateConfig, stcg_v1_diagnostics  # noqa: E402
from stcg.mechanism import MechanismConfig, generate_mechanism  # noqa: E402

OUT = ROOT / "results" / "revision2"
CONFIG = ROOT / "configs" / "aggregation_controls.yaml"


def _scaled_features(features: np.ndarray) -> np.ndarray:
    centred = features - features.mean(axis=0, keepdims=True)
    return centred / (centred.std(axis=0, keepdims=True) + 1e-8)


def _improvement(scaled: np.ndarray, inertia_two: float) -> float:
    inertia_one = float(np.square(scaled - scaled.mean(axis=0, keepdims=True)).sum())
    if inertia_one <= 1e-8:
        return 0.0
    return max(0.0, (inertia_one - inertia_two) / inertia_one)


def _inertia_of_partition(scaled: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for state in np.unique(labels):
        members = scaled[labels == state]
        centroid = members.mean(axis=0)
        total += float(np.square(members - centroid).sum())
    return total


def enumerate_bipartitions(n_windows: int):
    """Yield the 63 inequivalent non-empty bipartitions (point 0 in group 0)."""
    for mask in range(1, 1 << (n_windows - 1)):
        labels = np.zeros(n_windows, dtype=int)
        for point in range(1, n_windows):
            labels[point] = (mask >> (point - 1)) & 1
        yield labels


def run_seed(setting: dict, seed: int, config: dict) -> dict:
    dataset = generate_mechanism(MechanismConfig(**setting, seed=seed))
    state_config = STCGStateConfig(
        n_states=int(config["n_states"]),
        base_model="lasso",
        lasso_alpha=0.005,
        window_size=int(config["window"]["size"]),
        stride=int(config["window"]["stride"]),
        aggregate="max",
        state_aggregate="mean",
        seed=seed,
    )
    diagnostics = stcg_v1_diagnostics(dataset.x, int(setting["max_lag"]), state_config)
    scaled = _scaled_features(diagnostics.features)
    n_windows = scaled.shape[0]

    # K-means returned improvement (the gate's actual input).
    centred = scaled - scaled.mean(axis=0, keepdims=True)
    model = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(centred)
    delta_code = _improvement(scaled, float(model.inertia_))

    # Exhaustive optimum over all inequivalent bipartitions.
    delta_optimal = -1.0
    optimal_labels = None
    for labels in enumerate_bipartitions(n_windows):
        inertia = _inertia_of_partition(scaled, labels)
        delta = _improvement(scaled, inertia)
        if delta > delta_optimal:
            delta_optimal = delta
            optimal_labels = labels.copy()

    return {
        "seed": seed,
        "n_windows": int(n_windows),
        "delta_code": float(diagnostics.state_improvement),
        "delta_code_recomputed": delta_code,
        "delta_optimal": delta_optimal,
        "gap_optimal_minus_code": delta_optimal - diagnostics.state_improvement,
        "gate_threshold": 0.12,
        "lower_bound_1_over_M_minus_1": 1.0 / (n_windows - 1),
        "code_accepted": bool(diagnostics.state_improvement >= 0.12),
        "optimal_accepted": bool(delta_optimal >= 0.12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seeds = args.seeds if args.seeds else list(config["seeds"])
    setting = config["settings"]["switching_core"]

    rows = [run_seed(setting, seed, config) for seed in seeds]
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "gate_enumeration.csv", index=False)

    print(frame.round(4).to_string(index=False))
    print("\nsummary over seeds:")
    print(f"  delta_optimal      mean {frame.delta_optimal.mean():.4f}  "
          f"min {frame.delta_optimal.min():.4f}  max {frame.delta_optimal.max():.4f}")
    print(f"  delta_code        mean {frame.delta_code.mean():.4f}  "
          f"min {frame.delta_code.min():.4f}  max {frame.delta_code.max():.4f}")
    print(f"  gap (opt - code)  mean {frame.gap_optimal_minus_code.mean():.4f}  "
          f"max {frame.gap_optimal_minus_code.max():.4f}")
    print(f"  lower bound 1/(M-1) = {frame.lower_bound_1_over_M_minus_1.iloc[0]:.4f}")
    print(f"  code >= 0.12:      {frame.code_accepted.sum()}/{len(frame)}")
    print(f"  optimal >= 0.12:   {frame.optimal_accepted.sum()}/{len(frame)}")
    print(f"  optimal >= lower bound: {(frame.delta_optimal >= frame.lower_bound_1_over_M_minus_1).sum()}/{len(frame)}")
    print(f"  recomputed code == stored improvement: "
          f"{np.allclose(frame.delta_code_recomputed, frame.delta_code)}")


if __name__ == "__main__":
    main()
