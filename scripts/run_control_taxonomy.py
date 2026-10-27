"""Control taxonomy: what each control changes, and at which estimand.

The earlier manuscript treated every control as "no real graph change". That is
wrong for the lag-shift control: moving an edge from lag 1 to lag 2 changes the
lag-specific support, which is the estimand of this paper, even though the
pair-level graph is unchanged.

Each control is scored on three layers:

- lag-level union AUROC against the lag-specific union support;
- pair-level union AUROC against the pair support (lags collapsed by maximum);
- regime AUPRC and state ARI.

The taxonomy is fixed in advance:

``structure_change``
    The lag-specific support changes: a genuine change of the target estimand.
``mechanism_change``
    The support is fixed but the coefficients change (magnitude or sign).
``nuisance_change``
    Only mean, variance or trend changes; the graph is untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stcg import STCGStateConfig, stcg_v1_diagnostics  # noqa: E402
from stcg.analysis import evaluate_diagnostics, truth_from_dataset  # noqa: E402
from stcg.mechanism import MechanismConfig, generate_mechanism, structure_counts  # noqa: E402
from stcg.metrics import pair_edge_auc  # noqa: E402

OUT = ROOT / "results" / "mechanism"

TAXONOMY = {
    "core": "structure_change",
    "lag_shift": "structure_change",
    "same_support": "mechanism_change",
    "sign_flip": "mechanism_change",
    "variance_only": "nuisance_change",
    "mean_only": "nuisance_change",
    "trend": "nuisance_change",
    "stationary": "no_change",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUT / "control_taxonomy.csv"))
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else list(range(20))
    rows = []
    for name, category in TAXONOMY.items():
        control = None if name in ("core", "lag_shift") else name
        if name == "lag_shift":
            control = "lag_shift"
        for seed in seeds:
            dataset = generate_mechanism(MechanismConfig(seed=seed, control=control))
            truth = truth_from_dataset(dataset)
            state_config = STCGStateConfig(
                n_states=2,
                base_model="lasso",
                lasso_alpha=0.005,
                window_size=300,
                stride=150,
                aggregate="max",
                state_aggregate="mean",
                seed=seed,
            )
            diagnostics = stcg_v1_diagnostics(dataset.x, 3, state_config)
            metrics = evaluate_diagnostics(diagnostics, truth)
            counts = structure_counts(dataset)
            pair_auc = pair_edge_auc(diagnostics.scores, truth.support)
            rows.append(
                {
                    "control": name,
                    "category": category,
                    "seed": seed,
                    "k_dyn": counts["k_dyn"],
                    "k_graph": counts["k_graph"],
                    "union_auroc_lag": metrics["union_auroc"],
                    "union_auroc_pair": pair_auc,
                    "regime_auprc": metrics["regime_auprc_macro"],
                    "temporal_graph_auroc": metrics["temporal_graph_auroc"],
                    "state_ari": metrics["state_ari"],
                    "accepted": bool(diagnostics.states_accepted),
                    "k_hat": int(np.unique(diagnostics.labels).size),
                }
            )
        print(f"done {name}", flush=True)

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = (
        frame.groupby(["category", "control"])
        .agg(
            n=("seed", "size"),
            union_auroc_lag=("union_auroc_lag", "mean"),
            union_auroc_pair=("union_auroc_pair", "mean"),
            regime_auprc=("regime_auprc", "mean"),
            temporal_graph_auroc=("temporal_graph_auroc", "mean"),
            state_ari=("state_ari", "mean"),
            k_dyn=("k_dyn", "mean"),
            k_graph=("k_graph", "mean"),
            accepted=("accepted", "mean"),
            k_hat=("k_hat", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output.with_name(output.stem + "_summary.csv"), index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
