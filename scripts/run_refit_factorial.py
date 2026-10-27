"""WP4b: 2x2 factorial separating partitioning from re-estimation.

Cells:

======================  =====================  ==========================
                        window-score average   refit on partition ranges
======================  =====================  ==========================
K-means partition       ``kmeans_no_refit``    ``kmeans_refit``
PELT partition          ``pelt_no_refit``      ``pelt_refit``
======================  =====================  ==========================

References: ``global_var_lasso`` (no partition at all) and ``oracle_refit``
(true changepoints, refit). The factorial answers whether PELT's advantage comes
from a better partition or from re-estimating on longer segments.
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
from stcg.baselines import lasso_var_scores  # noqa: E402
from stcg.evaluation import per_regime_metrics, union_metrics, window_regime_labels  # noqa: E402
from stcg.mechanism import MechanismConfig, generate_mechanism  # noqa: E402
from stcg.refit import aggregate_within_groups, refit_group_scores  # noqa: E402
from stcg.segmented import pelt_on_features  # noqa: E402

OUT = ROOT / "results" / "baselines"


def _union(state_scores: np.ndarray, support: np.ndarray) -> dict[str, float]:
    return union_metrics(state_scores.max(axis=0), support, prefix="union_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "aggregation_controls.yaml"))
    parser.add_argument("--output", default=str(OUT / "refit_factorial.csv"))
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seeds = args.seeds if args.seeds else list(config["seeds"])
    rows: list[dict] = []

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
            labels = np.asarray(diagnostics.labels)
            window_truth = window_regime_labels(bounds, dataset.regimes)

            def add(method: str, scores: np.ndarray, partition: str, refit: bool, extra=None):
                row = {
                    "setting": name,
                    "seed": seed,
                    "method": method,
                    "partition": partition,
                    "refit": refit,
                    "n_groups": int(scores.shape[0]),
                }
                row.update(_union(scores, truth.support))
                if method in ("kmeans_no_refit", "kmeans_refit"):
                    row.update(
                        per_regime_metrics(
                            scores, truth.state_support, labels, window_truth
                        )
                    )
                if extra:
                    row.update(extra)
                rows.append(row)

            # References.
            global_scores = lasso_var_scores(dataset.x, int(setting["max_lag"]), 0.005)
            add("global_var_lasso", global_scores[None, ...], "none", False)

            add("kmeans_no_refit", diagnostics.state_scores, "kmeans", False,
                {"states_accepted": bool(diagnostics.states_accepted)})
            add(
                "kmeans_refit",
                refit_group_scores(dataset.x, bounds, labels, int(setting["max_lag"])),
                "kmeans",
                True,
                {"states_accepted": bool(diagnostics.states_accepted)},
            )

            segmentation = pelt_on_features(diagnostics.features, n_pcs=5)
            pelt_labels = segmentation.labels
            add(
                "pelt_no_refit",
                aggregate_within_groups(diagnostics.window_scores, pelt_labels),
                "pelt",
                False,
                {"n_segments": len(np.unique(pelt_labels)), "penalty": segmentation.penalty},
            )
            add(
                "pelt_refit",
                refit_group_scores(dataset.x, bounds, pelt_labels, int(setting["max_lag"])),
                "pelt",
                True,
                {"n_segments": len(np.unique(pelt_labels)), "penalty": segmentation.penalty},
            )

            add(
                "oracle_refit",
                refit_group_scores(dataset.x, bounds, window_truth, int(setting["max_lag"])),
                "oracle",
                True,
            )
        print(f"done {name}", flush=True)

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    summary = (
        frame.groupby(["setting", "partition", "refit", "method"])
        .agg(
            n=("union_auroc", "size"),
            union_auroc=("union_auroc", "mean"),
            union_auprc=("union_auprc", "mean"),
            regime_auprc=("regime_auprc_macro", "mean"),
            groups=("n_groups", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output.with_name(output.stem + "_summary.csv"), index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
