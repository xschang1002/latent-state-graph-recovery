"""Paired temporal-localisation extension of the existing mechanism grid.

The 35 labelled entries contain 31 distinct generator configurations. Each
configuration/seed is fitted once; repeated entries reuse identical scores.
Hard aggregation uses the existing gate and static-max fallback. An ungated
hard candidate separates grouping from gate selection. GMM has no native gate:
both continuous responsibility emission and the historical argmax emission are
reported. Accepted/rejected subgroups always refer to the hard STCG gate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stcg import STCGStateConfig, stcg_v1_diagnostics
from stcg.evaluation import temporal_delta_auc, temporal_graph_metrics, window_regime_labels, window_purity
from stcg.mechanism import MechanismConfig, generate_mechanism
from run_aggregation_controls import _hard_state_scores, _standardise, _weighted_state_scores
from run_mechanism_grid import build_configs

METHODS = ("raw_window", "kmeans_hard", "gmm_soft", "oracle_label", "gmm_weighted_argmax")
SOURCE_FILES = ("configs/mechanism_grid.yaml", "scripts/run_temporal_grid.py",
                "scripts/run_mechanism_grid.py", "scripts/run_aggregation_controls.py",
                "src/stcg/mechanism.py", "src/stcg/self_supervised.py", "src/stcg/evaluation.py")


def gmm_window_predictions(window_scores: np.ndarray, features: np.ndarray, seed: int) -> tuple[dict, dict]:
    """Same diagonal GMM as the old panel, with both explicit emission rules."""
    with warnings.catch_warnings(record=True) as caught:
        mixture = GaussianMixture(n_components=2, covariance_type="diag", n_init=3, random_state=seed)
        mixture.fit(_standardise(features))
    responsibilities = mixture.predict_proba(_standardise(features))
    prototypes = _weighted_state_scores(window_scores, responsibilities)
    continuous = np.einsum("wk,klij->wlij", responsibilities, prototypes)
    argmax = prototypes[responsibilities.argmax(axis=1)]
    entropy = -(responsibilities * np.log(np.maximum(responsibilities, np.finfo(float).tiny))).sum(axis=1)
    return {"gmm_soft": continuous, "gmm_weighted_argmax": argmax}, {
        "gmm_converged": bool(mixture.converged_), "gmm_iterations": int(mixture.n_iter_),
        "gmm_warning_count": len(caught),
        "gmm_warnings": json.dumps([str(w.message) for w in caught]),
        "gmm_entropy_normalized": float(entropy.mean() / np.log(2)),
        "gmm_fraction_max_below_099": float((responsibilities.max(axis=1) < 0.99).mean()),
        "gmm_max_emission_difference": float(np.max(np.abs(continuous - argmax))),
    }


def temporal_metrics(predictions: np.ndarray, bounds, dataset) -> dict:
    """Evaluate explicit per-window predictions, including continuous mixtures."""
    labels = np.arange(len(bounds))
    delta = temporal_delta_auc(predictions, labels, bounds, dataset.regimes, dataset.state_support)
    pooled = temporal_graph_metrics(predictions, labels, bounds, dataset.regimes, dataset.state_support)
    return {**delta, "pooled_temporal_auroc": pooled["temporal_graph_auroc"]}


def fit_seed(setting: dict, seed: int, grid: dict) -> list[dict]:
    dataset = generate_mechanism(MechanismConfig(**setting, seed=seed))
    config = STCGStateConfig(**grid["model"], window_size=int(grid["window"]["size"]),
                             stride=int(grid["window"]["stride"]), seed=seed)
    diagnostic = stcg_v1_diagnostics(dataset.x, int(setting["max_lag"]), config)
    scores, bounds = diagnostic.window_scores, diagnostic.bounds
    features = _standardise(diagnostic.features)
    candidate_labels = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(features)
    candidate_hard = _hard_state_scores(scores, candidate_labels)[candidate_labels]
    if diagnostic.states_accepted:
        np.testing.assert_array_equal(candidate_labels, diagnostic.labels)
    lookup = {int(label): index for index, label in enumerate(np.unique(diagnostic.labels))}
    hard = diagnostic.state_scores[[lookup[int(label)] for label in diagnostic.labels]]
    true_labels = window_regime_labels(bounds, dataset.regimes)
    true_index = {int(label): index for index, label in enumerate(np.unique(true_labels))}
    oracle = _hard_state_scores(scores, true_labels)[[true_index[int(label)] for label in true_labels]]
    gmm, gmm_info = gmm_window_predictions(scores, diagnostic.features, seed)
    predictions = {"raw_window": scores, "kmeans_hard": hard, "oracle_label": oracle, **gmm}
    evaluated = {method: temporal_metrics(values, bounds, dataset) for method, values in predictions.items()}
    forced = temporal_metrics(candidate_hard, bounds, dataset)
    raw = evaluated["raw_window"]["temporal_delta_auroc"]
    purity, _ = window_purity(bounds, dataset.regimes)
    rows = []
    for method in METHODS:
        metrics = evaluated[method]
        auc = metrics["temporal_delta_auroc"]
        candidate_auc = forced["temporal_delta_auroc"] if method == "kmeans_hard" else auc
        rows.append({"seed": seed, "method": method, "timesteps": len(dataset.x),
                     "switch_period": setting["switch_period"], "window_size": config.window_size,
                     "stride": config.stride, "h_over_D": config.window_size / setting["switch_period"],
                     "n_windows": len(bounds), "window_purity_mean": float(np.mean(purity)),
                     "hard_gate_accepted": bool(diagnostic.states_accepted),
                     "hard_gate_improvement": diagnostic.state_improvement,
                     "hard_gate_coherence": diagnostic.temporal_coherence,
                     "aggregation_executed": method != "raw_window" and
                     (method != "kmeans_hard" or bool(diagnostic.states_accepted)),
                     "native_gate": "STCG" if method == "kmeans_hard" else "none",
                     "raw_temporal_delta_auroc": raw, "delta_temporal": auc - raw,
                     "candidate_temporal_delta_auroc": candidate_auc,
                     "candidate_delta_temporal": candidate_auc - raw,
                     "candidate_pooled_temporal_auroc": forced["pooled_temporal_auroc"]
                     if method == "kmeans_hard" else metrics["pooled_temporal_auroc"],
                     "evaluable": bool(np.isfinite(auc)), **metrics, **gmm_info})
    return rows


def mean_interval(values: np.ndarray, draws: int = 10000) -> tuple[float, float, float]:
    """Pointwise paired-seed percentile CI; no interval below five finite seeds."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if len(values) < 5:
        return mean, np.nan, np.nan
    rng = np.random.default_rng(7)
    means = values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return mean, float(low), float(high)


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["grid_entry", "config_id", "axis", "level", "method"]
    for identity, all_rows in frame.groupby(keys, sort=False):
        metadata = dict(zip(keys, identity)); first = all_rows.iloc[0]
        for subset in ("all", "hard_gate_accepted", "hard_gate_rejected", "aggregation_executed"):
            group = all_rows if subset == "all" else all_rows[
                ~all_rows.hard_gate_accepted if subset == "hard_gate_rejected" else all_rows[subset]]
            row = {**metadata, "subset": subset, "n_seeds": len(group),
                   "n_evaluable": int(group.evaluable.sum()), "n_total": len(all_rows),
                   "n_hard_gate_accepted": int(all_rows.hard_gate_accepted.sum()),
                   "hard_gate_acceptance": float(all_rows.hard_gate_accepted.mean()),
                   "aggregation_execution_rate": float(all_rows.aggregation_executed.mean()),
                   **{field: first[field] for field in ("h_over_D", "n_windows", "timesteps", "switch_period", "window_size", "stride")}}
            for metric in ("temporal_delta_auroc", "raw_temporal_delta_auroc", "delta_temporal",
                           "candidate_temporal_delta_auroc", "candidate_delta_temporal", "pooled_temporal_auroc"):
                mean, low, high = mean_interval(group[metric].to_numpy())
                row.update({metric + "_mean": mean, metric + "_ci95_low": low, metric + "_ci95_high": high})
            rows.append(row)
    return pd.DataFrame(rows)


def protocol(grid: dict, seeds: list[int]) -> dict:
    return {"analysis": "exploratory temporal-localisation extension, no confirmatory grid tests",
            "grid": grid, "seeds": seeds, "grid_entries": 35, "unique_configurations": 31,
            "methods": list(METHODS), "paired_unit": "generator configuration and simulation seed",
            "hard_policy": "existing STCG gate; rejected output is static window max",
            "gmm": {"components": 2, "covariance_type": "diag", "n_init": 3,
                    "random_state": "simulation seed", "native_gate": False,
                    "continuous_emission": "responsibilities @ responsibility-weighted prototypes",
                    "legacy_emission": "weighted prototypes indexed by responsibility argmax"},
            "subgroups": "hard-gate subgroups use the same seeds for all methods; execution subset uses each method's own execution flag",
            "intervals": {"type": "pointwise percentile bootstrap of paired seed differences",
                          "draws": 10000, "seed": 7, "minimum_n": 5, "simultaneous": False},
            "no_changing_edges": "undefined, never replaced by 0.5; denominators retained",
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_FILES}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=ROOT / "configs/mechanism_grid.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/revision2")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    grid = yaml.safe_load(args.grid.read_text())
    seeds = args.seeds if args.seeds is not None else grid["seeds"]
    if len(set(seeds)) != len(seeds):
        parser.error("Seeds must be unique")
    entries = build_configs(grid)
    if args.limit is not None:
        entries = entries[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen = protocol(grid, seeds)
    frozen["executed_grid_entries"] = len(entries)
    frozen_path = args.output_dir / "temporal_grid_protocol.json"
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen:
        raise RuntimeError("Protocol differs from existing output; use a fresh output directory")
    frozen_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    cache, rows = {}, []
    for index, entry in enumerate(entries):
        canonical = json.dumps(asdict(MechanismConfig(**entry["config"])), sort_keys=True)
        config_id = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        for seed in seeds:
            key = (config_id, seed)
            if key not in cache:
                cache[key] = fit_seed(entry["config"], seed, grid)
            rows.extend({**row, "grid_entry": index, "config_id": config_id, "axis": entry["axis"],
                         "level": entry["level"], "config_json": json.dumps(entry["config"], sort_keys=True)}
                        for row in cache[key])
        print(f"Completed {index + 1}/{len(entries)} entries; {len(cache)} distinct configuration/seed fits", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "temporal_grid_detail.csv", index=False)
    summary = summarise(frame)
    summary.to_csv(args.output_dir / "temporal_grid_summary.csv", index=False)
    print(f"Saved {len(frame)} detailed and {len(summary)} summary rows", flush=True)


if __name__ == "__main__":
    main()
