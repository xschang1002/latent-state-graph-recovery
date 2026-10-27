"""Final confirmation on a seed block that no earlier analysis touched.

Seeds 2000--2049 were used to check the first (defective) PELT adapter and the
corrected one, so they are no longer a clean hold-out. This script uses seeds
3000--3049, fixes the method set and the tests in advance, and reports the joint
temporal-graph metric alongside the union and state layers.

Primary family (Holm over nine tests): STCG-v1 minus window max, minus corrected
PELT+refit, and minus the constrained two-segment refit, on union AUROC, union
AUPRC, and the joint temporal-graph AUROC. Global VAR-Lasso is reported with its
own interval as a descriptive reference, not as part of the family.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stcg import STCGStateConfig, stcg_v1_diagnostics  # noqa: E402
from stcg.analysis import truth_from_dataset  # noqa: E402
from stcg.baselines import lasso_var_scores  # noqa: E402
from stcg.evaluation import (  # noqa: E402
    GraphPrediction,
    evaluate_prediction,
    static_temporal_graph_metrics,
    union_metrics,
)
from stcg.mechanism import MechanismConfig, generate_mechanism  # noqa: E402
from stcg.refit import refit_group_scores  # noqa: E402
from stcg.segmented import constrained_segments, pelt_on_features  # noqa: E402
from stcg.stats import bootstrap_ci, holm  # noqa: E402

OUT = ROOT / "results" / "confirmatory"
AUDIT = ROOT / "audit"


def score_states(scores: np.ndarray, labels: np.ndarray, bounds, truth) -> dict:
    prediction = GraphPrediction(
        union_scores=scores.max(axis=0),
        state_scores=scores,
        window_assignments=labels,
        window_bounds=tuple(tuple(b) for b in bounds),
    )
    return evaluate_prediction(prediction, truth)


def static_metrics(union_scores: np.ndarray, bounds, truth) -> dict:
    """Union metrics plus the joint temporal target for a state-free method.

    A static method predicts the same graph at every window, so it still gets a
    well-defined temporal-graph score against the time-varying truth.
    """

    metrics = union_metrics(union_scores, truth.support, prefix="union_")
    metrics.update(
        static_temporal_graph_metrics(union_scores, bounds, truth.regimes, truth.state_support)
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "aggregation_controls.yaml"))
    parser.add_argument("--seed-start", type=int, default=3000)
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--output", default=str(OUT / "final_confirmation_detail.csv"))
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)

    rows, failures = [], []
    for name, setting in config["settings"].items():
        for seed in seeds:
            try:
                dataset = generate_mechanism(MechanismConfig(**setting, seed=seed))
                truth = truth_from_dataset(dataset)
                state_config = STCGStateConfig(
                    n_states=2,
                    base_model="lasso",
                    lasso_alpha=0.005,
                    window_size=int(config["window"]["size"]),
                    stride=int(config["window"]["stride"]),
                    aggregate="max",
                    state_aggregate="mean",
                    seed=seed,
                )
                start = time.perf_counter()
                diagnostics = stcg_v1_diagnostics(dataset.x, int(setting["max_lag"]), state_config)
                seconds = time.perf_counter() - start
                bounds = tuple(tuple(b) for b in diagnostics.bounds)
                max_lag = int(setting["max_lag"])

                def add(method: str, metrics: dict, elapsed: float, extra=None):
                    row = {"setting": name, "seed": seed, "method": method, "seconds": elapsed}
                    row.update(metrics)
                    if extra:
                        row.update(extra)
                    rows.append(row)

                stcg_metrics = score_states(diagnostics.state_scores, diagnostics.labels, bounds, truth)
                add("stcg_v1", stcg_metrics, seconds)

                window_mean = diagnostics.window_scores.mean(axis=0)
                window_max = diagnostics.window_scores.max(axis=0)
                add("window_mean", static_metrics(window_mean, bounds, truth), seconds)
                add("window_max", static_metrics(window_max, bounds, truth), seconds)
                global_scores = lasso_var_scores(dataset.x, max_lag, 0.005)
                add("global_var_lasso", static_metrics(global_scores, bounds, truth), seconds)

                pelt_start = time.perf_counter()
                pelt = pelt_on_features(diagnostics.features, n_pcs=5)
                pelt_scores = refit_group_scores(dataset.x, bounds, pelt.labels, max_lag)
                pelt_seconds = time.perf_counter() - pelt_start
                add(
                    "pelt_refit",
                    score_states(pelt_scores, pelt.labels, bounds, truth),
                    seconds + pelt_seconds,
                    {"n_segments": int(np.unique(pelt.labels).size)},
                )

                constrained = constrained_segments(diagnostics.features, n_segments=2, min_size=2)
                constrained_scores = refit_group_scores(
                    dataset.x, bounds, constrained.labels, max_lag
                )
                add(
                    "constrained_refit",
                    score_states(constrained_scores, constrained.labels, bounds, truth),
                    seconds,
                    {"n_segments": int(np.unique(constrained.labels).size)},
                )
            except Exception as error:  # pragma: no cover
                failures.append(
                    {"setting": name, "seed": seed, "error": f"{type(error).__name__}: {error}"}
                )
        print(f"done {name}", flush=True)

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    frame.to_csv(output, index=False)
    pd.DataFrame(failures, columns=["setting", "seed", "error"]).to_csv(
        AUDIT / "failure_log_final_confirmation.csv", index=False
    )

    per_seed = []
    for seed in seeds:
        record = {"seed": seed}
        for metric in ("union_auroc", "union_auprc", "temporal_graph_auroc"):
            values = {}
            for method in ("stcg_v1", "window_max", "pelt_refit", "constrained_refit", "global_var_lasso"):
                subset = frame[(frame.seed == seed) & (frame.method == method)]
                if subset.empty or metric not in subset or subset[metric].isna().all():
                    continue
                values[method] = float(subset[metric].mean())
            if "stcg_v1" not in values:
                continue
            for method, value in values.items():
                record[f"{metric}_{method}"] = value
                if method != "stcg_v1":
                    record[f"{metric}_diff_{method}"] = values["stcg_v1"] - value
        per_seed.append(record)
    per_seed_frame = pd.DataFrame(per_seed)
    per_seed_frame.to_csv(OUT / "final_confirmation_per_seed.csv", index=False)

    tests = []
    family = {
        "window_max": "primary",
        "pelt_refit": "primary",
        "constrained_refit": "primary",
        "global_var_lasso": "descriptive",
    }
    for metric in ("union_auroc", "union_auprc", "temporal_graph_auroc"):
        for competitor, kind in family.items():
            column = f"{metric}_diff_{competitor}"
            if column not in per_seed_frame:
                continue
            values = per_seed_frame[column].dropna().to_numpy()
            if values.size < 3:
                continue
            low, high = bootstrap_ci(values, draws=10000, seed=7)
            test = stats.ttest_1samp(values, 0.0)
            tests.append(
                {
                    "split": f"final_{args.seed_start}",
                    "metric": metric,
                    "comparison": f"stcg_v1_minus_{competitor}",
                    "family": kind,
                    "n_seeds": int(values.size),
                    "mean_difference": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "t_stat": float(test.statistic),
                    "p_raw": float(test.pvalue),
                }
            )
    tests_frame = pd.DataFrame(tests)
    primary = tests_frame.family == "primary"
    adjusted = holm(tests_frame.loc[primary, "p_raw"].tolist())
    tests_frame.loc[primary, "p_holm"] = adjusted
    tests_frame.loc[~primary, "p_holm"] = np.nan
    tests_frame["significant_holm"] = tests_frame["p_holm"] < 0.05
    tests_frame.to_csv(OUT / "final_confirmation_tests.csv", index=False)
    print(tests_frame.round(4).to_string(index=False))
    print(f"failures: {len(failures)}")


if __name__ == "__main__":
    main()
