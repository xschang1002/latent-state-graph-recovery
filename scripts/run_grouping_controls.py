"""Exact size-matched grouping control and three-condition state factorisation.

Reuse the original core decomposition and deterministically reconstruct forty
old fits whose intermediate maps/labels were not retained. Subsequent runs read
the numeric cache; --refit reconstructs it with identical configurations/seeds.
No new seeds, estimators or confirmatory significance claims are introduced.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from itertools import combinations
import json
from math import comb
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stcg import STCGStateConfig, stcg_v1_diagnostics
from stcg.evaluation import match_states, temporal_delta_auc, window_regime_labels
from stcg.mechanism import MechanismConfig, generate_mechanism
from run_aggregation_controls import _hard_state_scores, _standardise
from run_state_decomposition import _edge_directions, edge_direction_score, state_discrimination
from run_temporal_grid import mean_interval

FIT_SOURCES = ("configs/mechanism_grid.yaml", "src/stcg/mechanism.py", "src/stcg/self_supervised.py",
               "src/stcg/baselines.py", "src/stcg/evaluation.py", "scripts/run_aggregation_controls.py")
REPLAY_CELLS = {"duration600": (600, 1200), "long300": (300, 3000)}


def matched_partitions(labels: np.ndarray):
    """Each unlabelled partition with these two group sizes exactly once."""
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) != 2:
        raise ValueError("Exactly two nonempty groups are required")
    size = int(counts.min())
    for members in combinations(range(len(labels)), size):
        # Equal halves have two complement representations; choose one.
        if 2 * size == len(labels) and 0 not in members:
            continue
        assignment = np.zeros(len(labels), dtype=int)
        assignment[list(members)] = 1
        yield assignment


def temporal_auc(scores, labels, bounds, regimes, support) -> float:
    return float(temporal_delta_auc(scores, labels, bounds, regimes, support)["temporal_delta_auroc"])


def state_factors(scores, labels, bounds, regimes, support) -> dict:
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Factorisation requires two state rows numbered 0,1")
    truth = window_regime_labels(bounds, regimes)
    if set(np.unique(truth)) != {0, 1}:
        raise ValueError("Both true states must occur in the evaluation windows")
    mapping = {int(k): int(v) for k, v in match_states(labels, truth, n_truth_states=2)["mapping"].items()}
    direction, edges = _edge_directions(support)
    if not len(edges):
        raise ValueError("Factorisation needs changing nonself edges")
    j = state_discrimination(labels, truth, mapping)
    concordance = edge_direction_score(scores, direction, edges, mapping)
    auc = temporal_auc(scores, labels, bounds, regimes, support)
    predicted = 0.5 + 0.5 * j * concordance
    assert abs(auc - predicted) < 1e-14
    return {"J": j, "C_edge": concordance, "J_times_C": j * concordance,
            "hard_temporal_auroc": auc, "identity_predicted": predicted, "identity_gap": abs(auc - predicted)}


def build_cache(grid: dict) -> dict:
    arrays = {}
    for cell, (duration, timesteps) in REPLAY_CELLS.items():
        setting = {**grid["core"], "switch_period": duration, "timesteps": timesteps}
        for seed in grid["seeds"]:
            dataset = generate_mechanism(MechanismConfig(**setting, seed=seed))
            config = STCGStateConfig(**grid["model"], window_size=300, stride=150, seed=seed)
            diagnostic = stcg_v1_diagnostics(dataset.x, int(setting["max_lag"]), config)
            labels = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(_standardise(diagnostic.features))
            if diagnostic.states_accepted:
                np.testing.assert_array_equal(labels, diagnostic.labels)
            prefix = f"{cell}_seed{seed:02d}_"
            arrays.update({prefix + "window_scores": diagnostic.window_scores,
                           prefix + "labels": labels, prefix + "bounds": np.asarray(diagnostic.bounds),
                           prefix + "regimes": dataset.regimes, prefix + "support": dataset.state_support,
                           prefix + "gate": np.asarray(diagnostic.states_accepted)})
        print(f"Reconstructed original {cell}: {len(grid['seeds'])} fits", flush=True)
    return arrays


def analyse(arrays: dict, original: pd.DataFrame, core: pd.DataFrame):
    factors, partition_rows, seed_rows = [], [], []
    for row in core.itertuples(index=False):
        raw = original[(original.axis == "core") & (original.seed == row.seed) & (original.method == "raw_window")].iloc[0]
        factors.append({"condition": "core300", "D": 300, "M": 7, "seed": row.seed,
                        "J": row.J, "C_edge": row.R_edge, "J_times_C": row.J_times_R,
                        "hard_temporal_auroc": row.t_auc_delta, "raw_temporal_auroc": raw.temporal_delta_auroc,
                        "identity_predicted": row.identity_predicted, "identity_gap": row.identity_gap,
                        "gate_accepted": bool(raw.hard_gate_accepted), "execution": "native"})
    for cell, (duration, timesteps) in REPLAY_CELLS.items():
        reference = original[(original.switch_period == duration) & (original.timesteps == timesteps)
                             & (original.axis.isin(["switch_period", "timesteps"]))]
        for seed in range(20):
            prefix = f"{cell}_seed{seed:02d}_"
            scores, labels, bounds, regimes, support = (arrays[prefix + key] for key in
                                                      ("window_scores", "labels", "bounds", "regimes", "support"))
            maps = _hard_state_scores(scores, labels)
            decomposition = state_factors(maps, labels, bounds, regimes, support)
            raw = temporal_auc(scores, np.arange(len(scores)), bounds, regimes, support)
            baseline = reference[(reference.seed == seed) & (reference.method == "raw_window")].iloc[0]
            hard = reference[(reference.seed == seed) & (reference.method == "kmeans_hard")].iloc[0]
            assert np.isclose(raw, baseline.temporal_delta_auroc, atol=1e-15, rtol=0)
            assert np.isclose(decomposition["hard_temporal_auroc"], hard.candidate_temporal_delta_auroc, atol=1e-15, rtol=0)
            assert bool(arrays[prefix + "gate"]) == bool(hard.hard_gate_accepted)
            factors.append({"condition": cell, "D": duration, "M": len(scores), "seed": seed,
                            **decomposition, "raw_temporal_auroc": raw,
                            "gate_accepted": bool(hard.hard_gate_accepted),
                            "execution": "native" if cell == "duration600" else "forced"})
            if cell != "duration600":
                continue
            assert bool(hard.hard_gate_accepted) and len(scores) == 7
            values = []
            size = int(np.bincount(labels).min())
            for index, random_labels in enumerate(matched_partitions(labels)):
                random_maps = _hard_state_scores(scores, random_labels)
                value = temporal_auc(random_maps, random_labels, bounds, regimes, support)
                values.append(value)
                partition_rows.append({"seed": seed, "partition": index, "minority_size": size,
                                       "membership": "".join(map(str, random_labels)), "temporal_auroc": value})
            assert len(values) == comb(len(scores), size)
            random_mean = float(np.mean(values))
            learned = decomposition["hard_temporal_auroc"]
            seed_rows.append({"seed": seed, "D": duration, "M": len(scores), "minority_size": size,
                              "n_partitions": len(values), "raw_temporal_auroc": raw,
                              "hard_temporal_auroc": learned, "matched_random_temporal_auroc": random_mean,
                              "hard_minus_raw": learned - raw, "hard_minus_matched_random": learned - random_mean,
                              "matched_random_minus_raw": random_mean - raw})
    return pd.DataFrame(factors), pd.DataFrame(partition_rows), pd.DataFrame(seed_rows)


def coverage(original: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for entry, frame in original[original.method == "raw_window"].groupby("grid_entry", sort=True):
        first = frame.iloc[0]
        no_change = frame.temporal_delta_edges_total == 0
        no_flip = (~frame.evaluable) & ~no_change
        assert int(frame.evaluable.sum() + no_change.sum() + no_flip.sum()) == len(frame)
        rows.append({"grid_entry": entry, "config_id": first.config_id, "axis": first.axis,
                     "level": first.level, "M": int(first.n_windows), "n_total": len(frame),
                     "n_evaluable": int(frame.evaluable.sum()), "n_no_support_change": int(no_change.sum()),
                     "n_no_observed_flip": int(no_flip.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/revision2")
    parser.add_argument("--refit", action="store_true", help="Reconstruct the same forty existing fits instead of reading the cache")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid = yaml.safe_load((ROOT / "configs/mechanism_grid.yaml").read_text())
    expected_sources = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in FIT_SOURCES}
    cache_path = args.output_dir / "grouping_replay_outputs.npz"
    metadata_path = args.output_dir / "grouping_replay_manifest.json"
    if cache_path.exists() and not args.refit:
        metadata = json.loads(metadata_path.read_text())
        assert metadata["fit_source_sha256"] == expected_sources, "Cached fit sources changed; use a fresh output directory"
        assert metadata["cache_sha256"] == hashlib.sha256(cache_path.read_bytes()).hexdigest()
        with np.load(cache_path, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        print("Read forty original-fit outputs from the numeric cache", flush=True)
    else:
        arrays = build_cache(grid)
        np.savez_compressed(cache_path, **arrays)
        metadata_path.write_text(json.dumps({"fit_source_sha256": expected_sources,
                                            "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                                            "replayed_old_fits": 40, "new_seeds": 0}, indent=2, sort_keys=True) + "\n")
    original = pd.read_csv(ROOT / "results/revision2/temporal_grid_detail.csv")
    core = pd.read_csv(ROOT / "results/revision2/state_decomposition.csv")
    factors, partitions, seeds = analyse(arrays, original, core)
    factors.to_csv(args.output_dir / "grouping_state_factors_detail.csv", index=False)
    partitions.to_csv(args.output_dir / "matched_random_partitions.csv", index=False)
    seeds.to_csv(args.output_dir / "matched_random_seed_summary.csv", index=False)
    contrasts = []
    for metric in ("hard_minus_raw", "hard_minus_matched_random", "matched_random_minus_raw"):
        mean, low, high = mean_interval(seeds[metric].to_numpy())
        contrasts.append({"contrast": metric, "n_seeds": len(seeds), "mean": mean,
                          "ci95_low": low, "ci95_high": high})
    pd.DataFrame(contrasts).to_csv(args.output_dir / "matched_random_contrasts.csv", index=False)
    factors.groupby(["condition", "D", "M", "execution"], sort=False).agg(
        n_seeds=("seed", "size"), J=("J", "mean"), C_edge=("C_edge", "mean"),
        J_times_C=("J_times_C", "mean"), hard_temporal_auroc=("hard_temporal_auroc", "mean"),
        raw_temporal_auroc=("raw_temporal_auroc", "mean"), max_identity_gap=("identity_gap", "max")
    ).reset_index().to_csv(args.output_dir / "grouping_state_factors_summary.csv", index=False)
    coverage(original).to_csv(args.output_dir / "temporal_grid_coverage.csv", index=False)
    protocol = {"role": "exploratory controls on existing seeds, no independent positive-gain confirmation",
                "seeds": list(range(20)), "matched_cell": {"h": 300, "D": 600, "T": 1200, "M": 7},
                "partitions": "enumerate each fixed-size unlabelled bipartition; score before within-seed averaging",
                "independent_unit": "simulation seed, not enumerated partition",
                "bootstrap": {"draws": 10000, "seed": 7, "coverage": "pointwise 95% percentile"},
                "factor_cells": {"core300": [300, 7, "native"], "duration600": [600, 7, "native"],
                                 "long300": [300, 19, "forced"]},
                "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                                   (*FIT_SOURCES, "scripts/run_grouping_controls.py", "scripts/run_state_decomposition.py",
                                    "results/revision2/temporal_grid_detail.csv", "results/revision2/state_decomposition.csv")}}
    (args.output_dir / "grouping_controls_protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(pd.DataFrame(contrasts).to_string(index=False))
    print(factors.groupby("condition")[["J", "C_edge", "J_times_C", "hard_temporal_auroc", "raw_temporal_auroc"]].mean().to_string())


if __name__ == "__main__":
    main()
