"""Post-run numerical diagnostic for the six distinct temporal focus cells.

The frozen primary grid is unchanged. Check 12-decimal score rounding after
observing saturated GMM responsibilities; do not tune or replace any estimator.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from run_temporal_grid import (ROOT, STCGStateConfig, stcg_v1_diagnostics,
                               MechanismConfig, generate_mechanism, build_configs,
                               gmm_window_predictions, temporal_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/revision2/temporal_grid_gmm_numerics.csv")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/mechanism_grid.yaml").read_text())
    primary = pd.read_csv(ROOT / "results/revision2/temporal_grid_detail.csv")
    rows = []
    for entry_index, entry in enumerate(build_configs(config)):
        if entry["axis"] not in ("core", "switch_period", "timesteps"):
            continue
        setting = entry["config"]
        for seed in config["seeds"]:
            dataset = generate_mechanism(MechanismConfig(**setting, seed=seed))
            model = STCGStateConfig(**config["model"], window_size=300, stride=150, seed=seed)
            diagnostic = stcg_v1_diagnostics(dataset.x, int(setting["max_lag"]), model)
            predictions, info = gmm_window_predictions(diagnostic.window_scores, diagnostic.features, seed)
            predictions["raw_window"] = diagnostic.window_scores
            reference = primary[(primary.grid_entry == entry_index) & (primary.seed == seed)].set_index("method")
            for method, prediction in predictions.items():
                native = temporal_metrics(prediction, diagnostic.bounds, dataset)["temporal_delta_auroc"]
                rounded = temporal_metrics(np.round(prediction, 12), diagnostic.bounds, dataset)["temporal_delta_auroc"]
                assert np.isclose(native, reference.loc[method, "temporal_delta_auroc"], atol=1e-15, rtol=0)
                rows.append({"grid_entry": entry_index, "axis": entry["axis"], "level": entry["level"],
                             "seed": seed, "method": method, "decimals": 12,
                             "native_temporal_delta_auroc": native, "rounded_temporal_delta_auroc": rounded,
                             **info})
        print(f"Checked {entry['axis']}={entry['level']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
