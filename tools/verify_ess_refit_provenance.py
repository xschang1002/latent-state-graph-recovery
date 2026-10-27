"""Read-only reconstruction of the ESS PELT score change and source routing."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_ess_alignment as ess


def main() -> None:
    rolled, adjacency, _ = ess.load_subsystem_series("ffill")
    x = rolled.to_numpy(float)
    config = ess.STCGStateConfig(n_states=2, base_model="lasso", lasso_alpha=0.005,
                                 window_size=300, stride=150, aggregate="max", seed=0)
    diagnostic = ess.stcg_v1_diagnostics(x, 3, config)
    segmentation = ess.pelt_on_features(diagnostic.features, n_pcs=5)
    segmented, ranges = ess.segmented_lasso_scores(x, diagnostic.bounds, segmentation.labels, 3)
    # The former <= adjacency merge collapses all touching half-open ranges.
    # Its single full-series fit is reconstructed directly, without old code.
    global_scores = ess.lasso_var_scores(x, 3, alpha=0.005).max(axis=0)
    corrected_scores = segmented.max(axis=(0, 1))
    expert = adjacency.to_numpy().copy(); np.fill_diagonal(expert, 0)
    old = pd.read_csv(ROOT / "results/revision2/ess_alignment_before_audit_2026-10-23.csv")
    corrected_200 = pd.read_csv(ROOT / "results/revision2/ess_alignment_200_draws_seed0.csv")
    current = pd.read_csv(ROOT / "results/tables/ess_alignment.csv")
    rows = []
    for label, support in (("all_edges", expert > 0), ("strong_only", expert == 2)):
        previous = ess.edge_auc(global_scores[None], support[None])
        observed = ess.edge_auc(corrected_scores[None], support[None])
        select = lambda frame: frame[(frame.method == "pelt_var_lasso") & (frame.imputation == "ffill")
                                      & (frame.binarisation == label)].iloc[0]
        assert np.isclose(previous, select(old).auroc, atol=1e-15, rtol=0)
        assert np.isclose(observed, select(corrected_200).auroc, atol=1e-15, rtol=0)
        assert np.isclose(observed, select(current).auroc, atol=1e-15, rtol=0)
        rows.append({"binarisation": label, "previous_global_fit": previous,
                     "corrected_200_and_5000": observed})
    assert len(ranges) == 23 and all(end == start for (_, end), (start, _) in zip(ranges[:-1], ranges[1:]))
    routes = {}
    for filename in ("run_refit_factorial.py", "run_confirmatory.py", "run_final_confirmation.py",
                     "run_segmentation_comparison.py"):
        text = (ROOT / "scripts" / filename).read_text()
        assert "from stcg.refit import" in text and "refit_group_scores(" in text
        assert "segmented_lasso_scores" not in text
        routes[filename] = "stcg.refit.refit_group_scores"
    builder = (ROOT / "tools/build_revision2_tables.py").read_text()
    assert "wp4_comparison" not in builder
    print(json.dumps({"ess_detected_segments": len(np.unique(segmentation.labels)),
                      "ess_ranges": ranges, "observed_scores": rows,
                      "current_synthetic_routes": routes,
                      "legacy_affected_runner": "scripts/run_baselines_wp4.py",
                      "legacy_wp4_used_by_current_table_builder": False}, indent=2))


if __name__ == "__main__":
    main()
