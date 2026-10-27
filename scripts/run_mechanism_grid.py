"""Run the WP2 mechanism grid: controlled switching VAR plus negative controls.

Each row reports the three estimands (union, regime-specific, temporal), the
state-gate diagnostics, the realised generator quantities (spectral radius,
overlap, purity), and two reference predictions computed from the *same* window
scores: a state-free window max and an oracle-label aggregation.  Comparing the
fitted model with those two references separates "state clustering helped" from
"any aggregation helped".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stcg import STCGStateConfig, stcg_v1_diagnostics  # noqa: E402
from stcg.analysis import (  # noqa: E402
    evaluate_diagnostics,
    oracle_label_prediction,
    truth_from_dataset,
)
from stcg.evaluation import GraphPrediction, evaluate_prediction, window_purity  # noqa: E402
from stcg.mechanism import MechanismConfig, generate_mechanism  # noqa: E402

OUT = ROOT / "results" / "mechanism"


def window_size_for(timesteps: int, window: dict) -> int:
    """Fixed window across the grid so timesteps is the only count-changing axis."""

    del timesteps
    return int(window["size"])


def build_configs(grid: dict) -> list[dict]:
    core = dict(grid["core"])
    configs: list[dict] = []

    def add(overrides: dict, axis: str, level: str) -> None:
        config = {**core, **overrides}
        configs.append({"config": config, "axis": axis, "level": level, "control": config.get("control")})

    add({}, "core", "core")
    for axis, values in grid["axes"].items():
        for value in values:
            if axis in core and core[axis] == value:
                continue
            add({axis: value}, axis, str(value))
    for interaction in grid["interactions"]:
        keys = list(interaction)
        for first in interaction[keys[0]]:
            for second in interaction[keys[1]]:
                overrides = {keys[0]: first, keys[1]: second}
                add(overrides, "+".join(keys), f"{first}|{second}")
    for control in grid["controls"]:
        add({"control": control}, "control", control)
    return configs


def one_row(
    entry: dict,
    seed: int,
    grid: dict,
) -> dict:
    config = dict(entry["config"])
    timesteps = int(config["timesteps"])
    window = grid["window"]
    window_size = window_size_for(timesteps, window)
    stride = window_size // 2

    mechanism = MechanismConfig(**config, seed=seed)
    dataset = generate_mechanism(mechanism)
    state_config = STCGStateConfig(
        n_states=int(grid["model"]["n_states"]),
        base_model=grid["model"]["base_model"],
        lasso_alpha=float(grid["model"]["lasso_alpha"]),
        window_size=window_size,
        stride=stride,
        aggregate=grid["model"]["aggregate"],
        state_aggregate=grid["model"]["state_aggregate"],
        state_features=grid["model"]["state_features"],
        seed=seed,
    )
    diagnostics = stcg_v1_diagnostics(dataset.x, int(config["max_lag"]), state_config)
    truth = truth_from_dataset(dataset)
    metrics = evaluate_diagnostics(diagnostics, truth)

    bounds = tuple(diagnostics.bounds)
    purity, _ = window_purity(bounds, dataset.regimes)

    # Reference 1: no state clustering at all.
    state_free = GraphPrediction(
        union_scores=diagnostics.window_scores.max(axis=0),
        window_scores=diagnostics.window_scores,
        window_bounds=bounds,
    )
    state_free_metrics = evaluate_prediction(state_free, truth)

    # Reference 2: aggregate with the true window labels.
    oracle = oracle_label_prediction(
        diagnostics.window_scores, bounds, dataset.regimes
    )
    oracle_metrics = evaluate_prediction(oracle, truth)

    row = {
        "axis": entry["axis"],
        "level": entry["level"],
        "seed": seed,
        "control": config.get("control"),
        "config_json": json.dumps(config, sort_keys=True),
        "window_size": window_size,
        "stride": stride,
        "n_windows": len(bounds),
        "true_union_edges": float(dataset.union_support.sum()),
        "overlap_realised": dataset.overlap_realised,
        "spectral_radius_max": float(dataset.spectral_radius.max()),
        "window_purity_mean": float(np.nanmean(purity)),
        "mixed_window_fraction": float(np.mean(purity < 1.0)),
        "state_improvement": diagnostics.state_improvement,
        "temporal_coherence": diagnostics.temporal_coherence,
        "states_accepted": bool(diagnostics.states_accepted),
        "n_states_found": int(np.unique(diagnostics.labels).size),
    }
    row.update(metrics)
    row.update({f"state_free_{key}": value for key, value in state_free_metrics.items()})
    row.update({f"oracle_{key}": value for key, value in oracle_metrics.items()})
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default=str(ROOT / "configs" / "mechanism_grid.yaml"))
    parser.add_argument("--output", default=str(OUT / "mechanism_grid.csv"))
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    grid = yaml.safe_load(Path(args.grid).read_text(encoding="utf-8"))
    seeds = args.seeds if args.seeds else list(grid["seeds"])
    configs = build_configs(grid)
    if args.limit:
        configs = configs[: args.limit]

    rows = []
    for index, entry in enumerate(configs):
        for seed in seeds:
            rows.append(one_row(entry, seed, grid))
        if (index + 1) % 10 == 0:
            print(f"{index + 1}/{len(configs)} configurations", flush=True)

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    summary = (
        frame.groupby(["axis", "level"])
        .agg(
            n=("union_auroc", "size"),
            union_auroc=("union_auroc", "mean"),
            regime_auprc=("regime_auprc_macro", "mean"),
            state_ari=("state_ari", "mean"),
            accepted=("states_accepted", "mean"),
            purity=("window_purity_mean", "mean"),
            oracle_union=("oracle_union_auroc", "mean"),
            state_free_union=("state_free_union_auroc", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output.with_name(output.stem + "_summary.csv"), index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
