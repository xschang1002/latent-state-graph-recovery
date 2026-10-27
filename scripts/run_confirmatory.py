"""WP7: confirmatory run on seeds that were never used during development.

The analysis follows ``analysis/statistical_plan.md``: the seed is the
replication unit, settings are averaged within seed, confidence intervals come
from a seed-clustered bootstrap, and the four primary tests are Holm-corrected.
Failures are logged rather than dropped silently.
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
from stcg.evaluation import union_metrics  # noqa: E402
from stcg.mechanism import MechanismConfig, generate_mechanism  # noqa: E402
from stcg.baselines import lasso_var_scores  # noqa: E402
from stcg.refit import aggregate_within_groups, refit_group_scores  # noqa: E402
from stcg.segmented import pelt_on_features  # noqa: E402
from stcg.stats import bootstrap_ci, holm  # noqa: E402

OUT = ROOT / "results" / "confirmatory"
AUDIT = ROOT / "audit"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "aggregation_controls.yaml")
    )
    parser.add_argument("--split", choices=["development", "confirmation"], default="confirmation")
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--output", default=str(OUT / "confirmatory_detail.csv"))
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.split == "development":
        seeds = list(range(1000, 1000 + args.n_seeds))
    else:
        seeds = list(range(2000, 2000 + args.n_seeds))
    settings = config["settings"]

    OUT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    failures: list[dict] = []
    for name, setting in settings.items():
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
                stcg_seconds = time.perf_counter() - start
                from stcg.analysis import evaluate_diagnostics

                metrics = evaluate_diagnostics(diagnostics, truth)
                rows.append(
                    {
                        "setting": name,
                        "seed": seed,
                        "method": "stcg_v1",
                        "union_auroc": metrics["union_auroc"],
                        "union_auprc": metrics["union_auprc"],
                        "regime_auprc_macro": metrics["regime_auprc_macro"],
                        "state_ari": metrics["state_ari"],
                        "seconds": stcg_seconds,
                        "status": "ok",
                    }
                )
                rows.append(
                    {
                        "setting": name,
                        "seed": seed,
                        "method": "window_max",
                        "union_auroc": union_metrics(
                            diagnostics.window_scores.max(axis=0), truth.support, prefix="union_"
                        )["union_auroc"],
                        "union_auprc": union_metrics(
                            diagnostics.window_scores.max(axis=0), truth.support, prefix="union_"
                        )["union_auprc"],
                        "regime_auprc_macro": np.nan,
                        "state_ari": np.nan,
                        "seconds": stcg_seconds,
                        "status": "ok",
                    }
                )
                pelt_start = time.perf_counter()
                segmentation = pelt_on_features(diagnostics.features, n_pcs=5)
                pelt_scores = refit_group_scores(
                    dataset.x,
                    diagnostics.bounds,
                    segmentation.labels,
                    int(setting["max_lag"]),
                )
                pelt_seconds = time.perf_counter() - pelt_start
                pelt_metrics = union_metrics(
                    pelt_scores.max(axis=0), truth.support, prefix="union_"
                )
                rows.append(
                    {
                        "setting": name,
                        "seed": seed,
                        "method": "pelt_var_lasso",
                        "union_auroc": pelt_metrics["union_auroc"],
                        "union_auprc": pelt_metrics["union_auprc"],
                        "regime_auprc_macro": np.nan,
                        "state_ari": np.nan,
                        "seconds": stcg_seconds + pelt_seconds,
                        "status": "ok",
                    }
                )
                pelt_plain = aggregate_within_groups(
                    diagnostics.window_scores, segmentation.labels
                )
                pelt_plain_metrics = union_metrics(
                    pelt_plain.max(axis=0), truth.support, prefix="union_"
                )
                rows.append(
                    {
                        "setting": name,
                        "seed": seed,
                        "method": "pelt_no_refit",
                        "union_auroc": pelt_plain_metrics["union_auroc"],
                        "union_auprc": pelt_plain_metrics["union_auprc"],
                        "regime_auprc_macro": np.nan,
                        "state_ari": np.nan,
                        "seconds": stcg_seconds,
                        "status": "ok",
                        "n_segments": int(np.unique(segmentation.labels).size),
                    }
                )
                global_scores = lasso_var_scores(dataset.x, int(setting["max_lag"]), 0.005)
                global_metrics = union_metrics(
                    global_scores, truth.support, prefix="union_"
                )
                rows.append(
                    {
                        "setting": name,
                        "seed": seed,
                        "method": "global_var_lasso",
                        "union_auroc": global_metrics["union_auroc"],
                        "union_auprc": global_metrics["union_auprc"],
                        "regime_auprc_macro": np.nan,
                        "state_ari": np.nan,
                        "seconds": np.nan,
                        "status": "ok",
                    }
                )
            except Exception as error:  # pragma: no cover - defensive logging
                failures.append(
                    {
                        "setting": name,
                        "seed": seed,
                        "method": "all",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        print(f"done {name}", flush=True)

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    AUDIT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=["setting", "seed", "method", "error"]).to_csv(
        AUDIT / f"failure_log_{args.split}.csv", index=False
    )

    wide = frame.pivot_table(
        index=["setting", "seed"], columns="method", values=["union_auroc", "union_auprc"]
    )
    per_seed = []
    for seed in seeds:
        record = {"seed": seed}
        for metric in ("union_auroc", "union_auprc"):
            try:
                stcg = wide[(metric, "stcg_v1")].xs(seed, level="seed")
                base = wide[(metric, "window_max")].xs(seed, level="seed")
                pelt = wide[(metric, "pelt_var_lasso")].xs(seed, level="seed")
                record[f"{metric}_stcg"] = float(stcg.mean())
                record[f"{metric}_window_max"] = float(base.mean())
                record[f"{metric}_pelt"] = float(pelt.mean())
                record[f"{metric}_diff_window_max"] = float((stcg - base).mean())
                record[f"{metric}_diff_pelt"] = float((stcg - pelt).mean())
            except KeyError:
                continue
        per_seed.append(record)
    per_seed_frame = pd.DataFrame(per_seed)
    per_seed_frame.to_csv(OUT / f"confirmatory_per_seed_{args.split}.csv", index=False)

    tests = []
    for metric in ("union_auroc", "union_auprc"):
        for competitor in ("window_max", "pelt"):
            column = f"{metric}_diff_{competitor}"
            values = per_seed_frame[column].dropna().to_numpy()
            if values.size < 3:
                continue
            low, high = bootstrap_ci(values)
            test = stats.ttest_1samp(values, 0.0)
            tests.append(
                {
                    "split": args.split,
                    "metric": metric,
                    "comparison": f"stcg_v1_minus_{competitor}",
                    "n_seeds": int(values.size),
                    "mean_difference": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "t_stat": float(test.statistic),
                    "p_raw": float(test.pvalue),
                }
            )
    tests_frame = pd.DataFrame(tests)
    tests_frame["p_holm"] = holm(tests_frame["p_raw"].tolist())
    tests_frame["significant_holm"] = tests_frame["p_holm"] < 0.05
    tests_frame.to_csv(OUT / f"confirmatory_tests_{args.split}.csv", index=False)
    print(tests_frame.round(4).to_string(index=False))
    print(f"failures: {len(failures)}")


if __name__ == "__main__":
    main()
