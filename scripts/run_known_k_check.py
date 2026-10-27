"""Known-K protocol check: separate K_true, K_candidate and K_hat.

Three quantities are easy to conflate:

- ``K_true``: the number of states in the generator.
- ``K_candidate``: the state count handed to the method.
- ``K_hat``: the number of states the method actually returns.

Track I supplies ``K_candidate = K_true``. A gate-stress condition supplies
``K_candidate = 2`` when ``K_true = 1`` on purpose, to measure false splits. This
script runs both and reports them separately, so a stationary result is never
labelled "known K" when the candidate count was deliberately wrong.
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
from stcg.analysis import evaluate_diagnostics, truth_from_dataset  # noqa: E402
from stcg.mechanism import MechanismConfig, generate_mechanism  # noqa: E402

OUT = ROOT / "results" / "mechanism"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "aggregation_controls.yaml"))
    parser.add_argument("--output", default=str(OUT / "known_k_check.csv"))
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seeds = args.seeds if args.seeds else list(config["seeds"])
    rows = []
    plans = [
        ("stationary", 1, "track1_known_k"),
        ("stationary", 2, "gate_stress_wrong_candidate"),
        ("switching_core", 2, "track1_known_k"),
        ("switching_core", 1, "under_specified_candidate"),
    ]
    for setting_name, candidate, protocol in plans:
        setting = config["settings"][setting_name]
        for seed in seeds:
            dataset = generate_mechanism(MechanismConfig(**setting, seed=seed))
            truth = truth_from_dataset(dataset)
            state_config = STCGStateConfig(
                n_states=candidate,
                base_model="lasso",
                lasso_alpha=0.005,
                window_size=int(config["window"]["size"]),
                stride=int(config["window"]["stride"]),
                aggregate="max",
                state_aggregate="mean",
                seed=seed,
            )
            diagnostics = stcg_v1_diagnostics(dataset.x, int(setting["max_lag"]), state_config)
            metrics = evaluate_diagnostics(diagnostics, truth)
            rows.append(
                {
                    "setting": setting_name,
                    "protocol": protocol,
                    "seed": seed,
                    "k_true": int(np.unique(dataset.regimes).size),
                    "k_candidate": candidate,
                    "k_hat": int(np.unique(diagnostics.labels).size),
                    "accepted": bool(diagnostics.states_accepted),
                    "union_auroc": metrics["union_auroc"],
                    "union_auprc": metrics["union_auprc"],
                    "regime_auprc_macro": metrics["regime_auprc_macro"],
                    "state_ari": metrics["state_ari"],
                }
            )
        print(f"done {setting_name} K_candidate={candidate}", flush=True)

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = (
        frame.groupby(["setting", "protocol", "k_true", "k_candidate"])
        .agg(
            n=("seed", "size"),
            k_hat=("k_hat", "mean"),
            accepted=("accepted", "mean"),
            union_auroc=("union_auroc", "mean"),
            regime_auprc=("regime_auprc_macro", "mean"),
            state_ari=("state_ari", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output.with_name(output.stem + "_summary.csv"), index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
