"""WP4: matched comparison against a low-cost segmented baseline.

Methods compared on the same generated series and the same maximum lag:

- ``stcg_v1``: the two-stage window-score / K-means estimator.
- ``window_max``: no states at all (the aggregation-free floor).
- ``pelt_var_lasso``: PELT on the window-score sequence plus per-segment refit,
  with the BIC penalty and no knowledge of the true number of changes.
- ``pelt_var_lasso_oracle``: same, but given the true number of segments; this
  is an upper bound for the family and is labelled as such.
- ``regime_pcmci`` (optional, ``--with-rpcmci``): joint regime and graph search.

Timing is end-to-end per fit on one worker; the window pass, the partitioning
step, and the refits are recorded separately where possible.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stcg import STCGStateConfig, stcg_v1_diagnostics  # noqa: E402
from stcg.analysis import evaluate_diagnostics, truth_from_dataset  # noqa: E402
from stcg.evaluation import GraphPrediction, GraphTruth, union_metrics  # noqa: E402
from stcg.mechanism import MechanismConfig, generate_mechanism  # noqa: E402
from stcg.segmented import pelt_on_features, segmented_lasso_scores  # noqa: E402

OUT = ROOT / "results" / "baselines"


def _pelt_row(
    dataset,
    diagnostics,
    truth: GraphTruth,
    max_lag: int,
    label: str,
    n_bkps: int | None,
    seeds: int,
) -> dict:
    start = time.perf_counter()
    segmentation = pelt_on_features(
        diagnostics.features, n_pcs=5, n_bkps=n_bkps
    )
    state_scores, ranges = segmented_lasso_scores(
        dataset.x, diagnostics.bounds, segmentation.labels, max_lag, alpha=0.005
    )
    elapsed = time.perf_counter() - start
    union = state_scores.max(axis=0)
    metrics = union_metrics(union, truth.support, prefix="union_")
    # Segment-level scoring: each segment is compared to the union support of
    # the true states that dominate it. Reported only as a diagnostic because
    # PELT segments are not latent states.
    return {
        "method": label,
        "n_segments": len(ranges),
        "penalty": segmentation.penalty,
        "partition_seconds": elapsed,
        "fit_seconds": elapsed,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "aggregation_controls.yaml")
    )
    parser.add_argument("--output", default=str(OUT / "wp4_comparison.csv"))
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--with-rpcmci", action="store_true")
    parser.add_argument("--settings", nargs="*", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seeds = args.seeds if args.seeds else list(config["seeds"])[:5]
    settings = config["settings"]
    if args.settings:
        settings = {name: settings[name] for name in args.settings}

    rows: list[dict] = []
    for name, setting in settings.items():
        for seed in seeds:
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
            stcg_metrics = evaluate_diagnostics(diagnostics, truth)

            rows.append(
                {
                    "setting": name,
                    "seed": seed,
                    "method": "stcg_v1",
                    "n_segments": int(np.unique(diagnostics.labels).size),
                    "penalty": np.nan,
                    "partition_seconds": np.nan,
                    "fit_seconds": stcg_seconds,
                    "states_accepted": bool(diagnostics.states_accepted),
                    **stcg_metrics,
                }
            )

            window_max = diagnostics.window_scores.max(axis=0)
            rows.append(
                {
                    "setting": name,
                    "seed": seed,
                    "method": "window_max",
                    "n_segments": 1,
                    "penalty": np.nan,
                    "partition_seconds": 0.0,
                    "fit_seconds": stcg_seconds,
                    **union_metrics(window_max, truth.support, prefix="union_"),
                }
            )

            pelt_start = time.perf_counter()
            segmentation = pelt_on_features(diagnostics.features, n_pcs=5)
            state_scores, ranges = segmented_lasso_scores(
                dataset.x, diagnostics.bounds, segmentation.labels, int(setting["max_lag"])
            )
            pelt_seconds = time.perf_counter() - pelt_start
            rows.append(
                {
                    "setting": name,
                    "seed": seed,
                    "method": "pelt_var_lasso",
                    "n_segments": len(ranges),
                    "penalty": segmentation.penalty,
                    "partition_seconds": pelt_seconds,
                    "fit_seconds": stcg_seconds + pelt_seconds,
                    **union_metrics(state_scores.max(axis=0), truth.support, prefix="union_"),
                }
            )

            n_true_segments = int(dataset.regimes.size // int(setting["switch_period"])) + 1
            oracle_seg = pelt_on_features(
                diagnostics.features, n_pcs=5, n_bkps=max(1, n_true_segments - 1)
            )
            oracle_scores, oracle_ranges = segmented_lasso_scores(
                dataset.x, diagnostics.bounds, oracle_seg.labels, int(setting["max_lag"])
            )
            rows.append(
                {
                    "setting": name,
                    "seed": seed,
                    "method": "pelt_var_lasso_oracle",
                    "n_segments": len(oracle_ranges),
                    "penalty": np.nan,
                    "partition_seconds": np.nan,
                    "fit_seconds": np.nan,
                    **union_metrics(
                        oracle_scores.max(axis=0), truth.support, prefix="union_"
                    ),
                }
            )

            if args.with_rpcmci:
                from stcg.baselines import rpcmci_scores

                start = time.perf_counter()
                scores = rpcmci_scores(
                    dataset.x, int(setting["max_lag"]), num_regimes=2, seed=seed, n_jobs=1
                )
                elapsed = time.perf_counter() - start
                rows.append(
                    {
                        "setting": name,
                        "seed": seed,
                        "method": "regime_pcmci",
                        "n_segments": 2,
                        "penalty": np.nan,
                        "partition_seconds": np.nan,
                        "fit_seconds": elapsed,
                        **union_metrics(scores, truth.support, prefix="union_"),
                    }
                )
        print(f"done {name}", flush=True)

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = (
        frame.groupby(["setting", "method"])
        .agg(
            n=("union_auroc", "size"),
            union_auroc=("union_auroc", "mean"),
            union_auprc=("union_auprc", "mean"),
            seconds=("fit_seconds", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output.with_name(output.stem + "_summary.csv"), index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
