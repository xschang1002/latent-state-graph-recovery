"""Fair segmentation comparison: constrain the number of segments.

The BIC-penalised PELT baseline returns one segment per window at the seven-window
operating point, which makes it incomparable with a two-state K-means partition.
This script compares like-for-like partitions:

- K-means with two groups (the two-stage estimator);
- the best two contiguous segments (fixed segment count);
- PELT with the BIC penalty (unconstrained), reported with its realised count;
- the best split of a raw-VAR Chow scan (coefficient change detector);
- the oracle two-segment split.

Each partition is scored with window averaging and with refit. A penalty and
minimum-length scan is written separately so that the PELT behaviour can be read
as a function of its own knobs rather than as a single arbitrary setting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stcg import STCGStateConfig, stcg_v1_diagnostics  # noqa: E402
from stcg.analysis import truth_from_dataset  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from stcg.evaluation import union_metrics, window_regime_labels  # noqa: E402
from stcg.mechanism import MechanismConfig, generate_mechanism  # noqa: E402
from stcg.refit import aggregate_within_groups, refit_group_scores  # noqa: E402
from stcg.segmented import (  # noqa: E402
    constrained_segments,
    pelt_on_features,
    var_change_point,
)

OUT = ROOT / "results" / "baselines"


def split_labels_from_time(
    bounds: tuple[tuple[int, int], ...], split: float
) -> np.ndarray:
    centres = np.array([(start + end - 1) // 2 for start, end in bounds])
    return (centres >= split).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "aggregation_controls.yaml"))
    parser.add_argument("--output", default=str(OUT / "segmentation_comparison.csv"))
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seeds = args.seeds if args.seeds else list(config["seeds"])
    rows = []
    scan_rows = []

    for name, setting in config["settings"].items():
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
            diagnostics = stcg_v1_diagnostics(dataset.x, int(setting["max_lag"]), state_config)
            bounds = tuple(tuple(b) for b in diagnostics.bounds)
            window_truth = window_regime_labels(bounds, dataset.regimes)
            max_lag = int(setting["max_lag"])

            def add(method: str, labels: np.ndarray, refit: bool, extra=None):
                scores = (
                    refit_group_scores(dataset.x, bounds, labels, max_lag)
                    if refit
                    else aggregate_within_groups(diagnostics.window_scores, labels)
                )
                metrics = union_metrics(scores.max(axis=0), truth.support, prefix="union_")
                row = {
                    "setting": name,
                    "seed": seed,
                    "method": method,
                    "refit": refit,
                    "n_segments": int(np.unique(labels).size),
                    "partition_ari": float(adjusted_rand_score(window_truth, labels)),
                    "true_segments": int(np.unique(window_truth).size),
                }
                row.update(metrics)
                if extra:
                    row.update(extra)
                rows.append(row)

            kmeans_labels = np.asarray(diagnostics.labels)
            add("kmeans", kmeans_labels, False)
            add("kmeans_refit", kmeans_labels, True)

            constrained = constrained_segments(diagnostics.features, n_segments=2, min_size=2)
            add("constrained_2seg", constrained.labels, False,
                {"breakpoints": str(constrained.breakpoints)})
            add("constrained_2seg_refit", constrained.labels, True,
                {"breakpoints": str(constrained.breakpoints)})

            pelt = pelt_on_features(diagnostics.features, n_pcs=5)
            add("pelt_bic", pelt.labels, False, {"penalty": pelt.penalty})
            add("pelt_bic_refit", pelt.labels, True, {"penalty": pelt.penalty})

            change = var_change_point(dataset.x, max_lag)
            if np.isfinite(change["split"]):
                raw_labels = split_labels_from_time(bounds, change["split"])
                add("raw_var_split", raw_labels, False,
                    {"breakpoints": str([change["split"]]), "f_stat": change["f_stat"]})
                add("raw_var_split_refit", raw_labels, True,
                    {"breakpoints": str([change["split"]]), "f_stat": change["f_stat"]})

            # Oracle contiguous split at the true change point (two segments).
            switches = np.flatnonzero(dataset.regimes[1:] != dataset.regimes[:-1]) + 1
            if switches.size:
                oracle_labels = split_labels_from_time(bounds, float(switches[0]))
                add("oracle_2seg", oracle_labels, False)

            for factor in (0.25, 0.5, 1.0, 2.0, 4.0):
                for min_size in (1, 2, 3):
                    result = pelt_on_features(
                        diagnostics.features, penalty_factor=factor, n_pcs=5, min_size=min_size
                    )
                    scores = aggregate_within_groups(diagnostics.window_scores, result.labels)
                    scan_rows.append(
                        {
                            "setting": name,
                            "seed": seed,
                            "penalty_factor": factor,
                            "min_size": min_size,
                            "n_segments": int(np.unique(result.labels).size),
                            "union_auroc": union_metrics(
                                scores.max(axis=0), truth.support, prefix="union_"
                            )["union_auroc"],
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
            n=("seed", "size"),
            segments=("n_segments", "mean"),
            partition_ari=("partition_ari", "mean"),
            union_auroc=("union_auroc", "mean"),
            union_auprc=("union_auprc", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output.with_name(output.stem + "_summary.csv"), index=False)
    print(summary.round(3).to_string(index=False))

    scan = pd.DataFrame(scan_rows)
    scan.to_csv(output.with_name("pelt_penalty_scan.csv"), index=False)
    scan_summary = (
        scan.groupby(["penalty_factor", "min_size"])
        .agg(segments=("n_segments", "mean"), union_auroc=("union_auroc", "mean"))
        .reset_index()
    )
    scan_summary.to_csv(output.with_name("pelt_penalty_scan_summary.csv"), index=False)
    print(scan_summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
