"""ESS subsystem-level alignment check.

The ESS expert graph is a subsystem-level rolled graph, not a per-PV
lag-specific adjacency.  We therefore roll the 227 process variables up to the
23 subsystems that appear in the expert matrix, score lagged subsystem edges,
collapse lags by maximum, and compare with the expert matrix at the subsystem
pair level.  Both binarisations are reported: all expert edges, and only the
strong (value 2) edges.

A degree-preserving permutation null is used because the expert graph is dense
in the process-flow direction and a raw AUROC is not interpretable on its own.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import average_precision_score  # noqa: E402

from stcg import STCGStateConfig, stcg_v1_diagnostics  # noqa: E402
from stcg.baselines import lasso_var_scores  # noqa: E402
from stcg.metrics import edge_auc, lagged_correlation_scores  # noqa: E402
from stcg.segmented import pelt_on_features, segmented_lasso_scores  # noqa: E402
from stcg.stats import holm  # noqa: E402
from stcg.randomization import directed_degree_null  # noqa: E402

ESS_TOOLS = Path(os.environ.get("ESS_DATA_DIR", ROOT / "data" / "external" / "essdata"))


IMPUTATIONS = {
    # Forward fill with a full-hour column-mean fallback for leading missing
    # values. This is an offline analysis, not a strictly online imputation.
    "ffill": lambda frame: frame.ffill().fillna(frame.mean()),
    # Backward-only fill, which does use future observations; kept as the
    # sensitivity check for that concern.
    "bfill": lambda frame: frame.bfill().fillna(frame.mean()),
    "linear": lambda frame: frame.interpolate(limit_direction="both").fillna(frame.mean()),
    "mean": lambda frame: frame.fillna(frame.mean()),
}


def load_subsystem_series(imputation: str = "ffill") -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    data = pd.read_csv(ESS_TOOLS / "preprocessedData.csv")
    adjacency = pd.read_csv(ESS_TOOLS / "adjMatrix.csv", index_col=0)
    data = data.set_index(data.columns[0])
    data = IMPUTATIONS[imputation](data)

    mapping = pd.read_csv(ESS_TOOLS / "pv_subsystem_map.csv") if (ESS_TOOLS / "pv_subsystem_map.csv").exists() else None
    if mapping is None:
        raise FileNotFoundError(
            "Run scripts/build_ess_mapping.py first to create pv_subsystem_map.csv"
        )
    lookup = dict(zip(mapping["pv"], mapping["subsystem"]))

    subsystems = list(adjacency.columns)
    keep = [column for column in data.columns if lookup.get(column) in subsystems]
    dropped = [column for column in data.columns if column not in keep]

    standardised = (data[keep] - data[keep].mean()) / (data[keep].std() + 1e-8)
    grouped = {}
    for subsystem in subsystems:
        members = [column for column in keep if lookup[column] == subsystem]
        if members:
            grouped[subsystem] = standardised[members].mean(axis=1)
    rolled = pd.DataFrame(grouped)[subsystems]
    return rolled, adjacency, {column: lookup.get(column, "") for column in dropped}


def permutation_null(
    scores: np.ndarray, support: np.ndarray, draws: int, seed: int = 0,
    null_supports: np.ndarray | None = None,
) -> tuple[float, float]:
    """Upper-tail Monte Carlo p-value under fixed labelled in/out degrees."""
    if null_supports is None:
        null_supports = directed_degree_null(support, draws, seed)
    if null_supports.shape != (draws, *support.shape):
        raise ValueError("Null sample count or adjacency shape differs from the requested test")
    if (np.diagonal(null_supports, axis1=1, axis2=2).any()
            or not np.isin(null_supports, [0, 1]).all()
            or not (null_supports.sum(axis=1) == support.sum(axis=0)).all()
            or not (null_supports.sum(axis=2) == support.sum(axis=1)).all()):
        raise ValueError("Null samples must preserve each node's in/out degree without self-loops")
    observed = edge_auc(scores[None, ...], support[None, ...])
    # For fixed scores and a fixed edge count, AUROC is the positive rank sum.
    # This is exactly the same tied-rank calculation as edge_auc, evaluated for
    # all null graphs together rather than sorting the same scores repeatedly.
    mask = ~np.eye(len(support), dtype=bool)
    ranks = rankdata(scores[mask], method="average")
    positives = int(support[mask].sum())
    negatives = int(mask.sum()) - positives
    if not positives or not negatives:
        return float("nan"), float("nan")
    rank_sums = null_supports[:, mask].astype(float) @ ranks
    null = (rank_sums - positives * (positives + 1) / 2) / (positives * negatives)
    p_value = float((1 + np.sum(null >= observed)) / (1 + draws))
    return float(null.mean()), p_value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "results" / "tables" / "ess_alignment.csv"))
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--null-seeds", type=int, nargs="+", default=list(range(5)),
                        help="Null-chain seeds; seed 0 is the reference report")
    parser.add_argument("--imputation", choices=sorted(IMPUTATIONS), default="ffill")
    parser.add_argument("--all-imputations", action="store_true")
    parser.add_argument("--max-lag", type=int, default=3)
    args = parser.parse_args(argv)
    if args.draws < 1:
        parser.error("--draws must be positive")
    if (0 not in args.null_seeds or min(args.null_seeds) < 0
            or len(set(args.null_seeds)) != len(args.null_seeds)):
        parser.error("--null-seeds must be unique nonnegative values and include reference seed 0")
    args.null_seeds.sort()
    return args


def correct_families(frame: pd.DataFrame) -> pd.DataFrame:
    """Correct each seed separately; Monte Carlo seeds are not replicates."""
    frame = frame.copy()
    frame["p_holm"] = np.nan
    frame["p_holm_all_imputations"] = np.nan
    for _, group in frame.groupby("null_seed", sort=True):
        primary = group[(group.imputation == "ffill") & group.p_raw.notna()]
        frame.loc[primary.index, "p_holm"] = holm(primary.p_raw.tolist())
        if group.imputation.nunique() == 4 and len(group) == 32:
            frame.loc[group.index, "p_holm_all_imputations"] = holm(group.p_raw.tolist())
    return frame


def main() -> None:
    args = parse_args()

    strategies = sorted(IMPUTATIONS) if args.all_imputations else [args.imputation]
    all_results = []
    null_cache = {}
    for strategy in strategies:
        results = run_strategy(strategy, args, null_cache)
        all_results.extend(results)
    frame = correct_families(pd.DataFrame(all_results))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    reference = frame[frame.null_seed == 0]
    reference.to_csv(output, index=False)
    sensitivity = output.with_name(output.stem + "_seed_sensitivity.csv")
    frame.to_csv(sensitivity, index=False)
    print(reference.round(4).to_string(index=False))
    print("Forward-fill Holm p-value ranges across null seeds:")
    print(frame[frame.imputation == "ffill"].groupby(["method", "binarisation"]).p_holm.agg(["min", "max"]).to_string())
    print(f"Reference seed 0: {output}; all null seeds: {sensitivity}")


def run_strategy(strategy: str, args, null_cache: dict | None = None) -> list[dict]:
    rolled, adjacency, dropped = load_subsystem_series(strategy)
    names = list(rolled.columns)
    x = rolled.to_numpy(dtype=float)
    print(f"rolled series {x.shape}, dropped PVs {len(dropped)}")

    results = []
    score_sets = {}

    state_config = STCGStateConfig(
        n_states=2,
        base_model="lasso",
        lasso_alpha=0.005,
        window_size=300,
        stride=150,
        aggregate="max",
        seed=0,
    )
    diagnostics = stcg_v1_diagnostics(x, args.max_lag, state_config)
    score_sets["stcg_v1"] = diagnostics.scores
    score_sets["window_max"] = diagnostics.window_scores.max(axis=0)
    segmentation = pelt_on_features(diagnostics.features, n_pcs=5)
    pelt_scores, _ = segmented_lasso_scores(x, diagnostics.bounds, segmentation.labels, args.max_lag)
    score_sets["pelt_var_lasso"] = pelt_scores.max(axis=0)
    score_sets["lag_corr"] = lagged_correlation_scores(x, args.max_lag)

    expert = adjacency.to_numpy()
    np.fill_diagonal(expert, 0)
    supports = {
        "all_edges": (expert > 0).astype(int),
        "strong_only": (expert == 2).astype(int),
    }
    # Use the same null graph realizations for every method on an edge set.
    if null_cache is None:
        null_cache = {}
    null_supports = {}
    for seed in getattr(args, "null_seeds", [0]):
        for label, support in supports.items():
            key = (support.shape, support.tobytes(), args.draws, seed)
            if key not in null_cache:
                print(f"sampling null seed {seed}, {label}, {args.draws} draws", flush=True)
                null_cache[key] = directed_degree_null(support, args.draws, seed=seed)
            null_supports[seed, label] = null_cache[key]

    for method, scores in score_sets.items():
        pair_scores = scores.max(axis=0)
        np.fill_diagonal(pair_scores, 0.0)
        for label, support in supports.items():
            if support.sum() == 0:
                continue
            for seed in getattr(args, "null_seeds", [0]):
                null_mean, p_value = permutation_null(
                    pair_scores, support, args.draws, null_supports=null_supports[seed, label]
                )
                results.append(
                    {
                        "method": method,
                        "imputation": strategy,
                        "binarisation": label,
                        "auroc": edge_auc(pair_scores[None, ...], support[None, ...]),
                        "auprc": float(average_precision_score(support.ravel(), pair_scores.ravel())),
                        "prevalence": float(support.mean()),
                        "null_auroc_mean": null_mean,
                        "permutation_p": p_value,
                        "p_raw": p_value,
                        "n_edges": int(support.sum()),
                        "n_nodes": len(names),
                        "n_timesteps": int(x.shape[0]),
                        "null_model": "labelled_in_out_degree_switch_chain",
                        "null_seed": seed,
                        "null_draws": args.draws,
                        "null_exceedances": int(round(p_value * (args.draws + 1) - 1))
                        if np.isfinite(p_value) else np.nan,
                        "null_burn_in_proposals": 100 * int(support.sum()),
                        "null_thinning_proposals": 10 * int(support.sum()),
                        "n_segments": int(np.unique(segmentation.labels).size)
                        if method == "pelt_var_lasso" else np.nan,
                        "n_refit_ranges": len(pelt_scores) if method == "pelt_var_lasso" else np.nan,
                    }
                )
        print(f"scored {method}", flush=True)

    return results


if __name__ == "__main__":
    main()
