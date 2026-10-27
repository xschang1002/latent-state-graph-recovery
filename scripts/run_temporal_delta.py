"""Evaluation link: per-changing-edge temporal AUROC and the static ceiling.

The pooled temporal-graph AUROC (``temporal_graph_auroc``) mixes long-run edge
occupancy with true temporal localisation, so a constant prediction can score
high simply by ranking frequent edges ahead of rare ones. This script computes,
on the switching-core setting (twenty seeds, identical window scores), the two
quantities that separate those effects:

* ``T-AUC_delta`` -- mean per-edge AUROC over edges in ``E_delta`` (edges whose
  support differs between states) that actually flip in the evaluation windows,
  of the per-window score sequence against the active edge label. A constant
  prediction scores exactly 0.5 here.

* ``AUC_static*`` -- the best pooled temporal AUROC any constant edge score can
  reach, ``1/2 + sum_{e,f}|pi_e - pi_f| / (4 P Q)`` with ``pi_e`` the long-run
  activation frequency and ``P=sum pi_e``, ``Q=sum(1-pi_e)``. Verified against a
  direct occupancy-oracle AUROC computation.

Methods with no state assignment (window mean / max / bagged mean) are constant
predictions and therefore get ``T-AUC_delta = 0.5`` by construction; the value
is still computed (not hard-coded) so the constant-predictor claim is checked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stcg import STCGStateConfig, stcg_v1_diagnostics  # noqa: E402
from stcg.evaluation import (  # noqa: E402
    GraphTruth,
    nonself_mask,
    temporal_delta_auc,
    temporal_graph_metrics,
    window_regime_labels,
)
from stcg.mechanism import MechanismConfig, generate_mechanism  # noqa: E402

OUT = ROOT / "results" / "revision2"
CONFIG = ROOT / "configs" / "aggregation_controls.yaml"


def _hard_state_scores(window_scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    states = np.unique(labels)
    return np.stack([window_scores[labels == state].mean(axis=0) for state in states], axis=0)


def _standardise(features: np.ndarray) -> np.ndarray:
    centred = features - features.mean(axis=0, keepdims=True)
    return centred / (centred.std(axis=0, keepdims=True) + 1e-8)


def _active_matrix(state_support: np.ndarray, bounds, regimes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (A, edge_indices): per-window active labels for non-self edges."""
    lags, nodes, _ = state_support.shape[1:]
    mask = nonself_mask((lags, nodes, nodes))
    edges = np.flatnonzero(mask.reshape(-1))
    n_windows = len(bounds)
    active = np.zeros((n_windows, len(edges)), dtype=float)
    for w, (start, end) in enumerate(bounds):
        centre = (start + end - 1) // 2
        state = int(regimes[centre])
        active[w] = state_support[state].reshape(-1)[edges]
    return active, edges


def static_ceiling(state_support: np.ndarray, bounds, regimes: np.ndarray) -> float:
    """Closed-form best static pooled temporal AUROC (reviewer's bound)."""
    active, _ = _active_matrix(state_support, bounds, regimes)
    pi = active.mean(axis=0)
    total = float(pi.sum())
    total_inactive = float((1.0 - pi).sum())
    if total <= 0.0 or total_inactive <= 0.0:
        return float("nan")
    pairwise = float(np.abs(pi[:, None] - pi[None, :]).sum())
    return 0.5 + pairwise / (4.0 * total * total_inactive)


def static_ceiling_direct(state_support: np.ndarray, bounds, regimes: np.ndarray) -> float:
    """Occupancy-oracle AUROC: score every window by the edge's own frequency."""
    active, _ = _active_matrix(state_support, bounds, regimes)
    pi = active.mean(axis=0)
    n_windows = active.shape[0]
    y = active.reshape(-1)  # window-major
    score = np.tile(pi, n_windows)  # same window-major ordering
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def delta_for_stateful(state_scores: np.ndarray, labels: np.ndarray, bounds, truth: GraphTruth) -> dict:
    return temporal_delta_auc(state_scores, labels, bounds, truth.regimes, truth.state_support)


def delta_for_static(union_scores: np.ndarray, bounds, truth: GraphTruth) -> dict:
    """A constant graph emits the same edge score in every window -> 0.5 per edge."""
    n_states = truth.state_support.shape[0]
    return temporal_delta_auc(
        np.repeat(union_scores[None, ...], n_states, axis=0),
        np.zeros(len(bounds), dtype=int),
        bounds,
        truth.regimes,
        truth.state_support,
    )


def pooled_for_stateful(state_scores: np.ndarray, labels: np.ndarray, bounds, truth: GraphTruth) -> float:
    m = temporal_graph_metrics(state_scores, labels, bounds, truth.regimes, truth.state_support)
    return float(m["temporal_graph_auroc"])


def pooled_for_static(union_scores: np.ndarray, bounds, truth: GraphTruth) -> float:
    from stcg.evaluation import static_temporal_graph_metrics

    m = static_temporal_graph_metrics(union_scores, bounds, truth.regimes, truth.state_support)
    return float(m["temporal_graph_auroc"])


def run_seed(setting: dict, seed: int, config: dict) -> list[dict]:
    rows: list[dict] = []
    rng = np.random.default_rng(seed + 991)
    dataset = generate_mechanism(MechanismConfig(**setting, seed=seed))
    truth = GraphTruth(
        support=dataset.union_support,
        state_support=dataset.state_support,
        regimes=dataset.regimes,
    )
    state_config = STCGStateConfig(
        n_states=int(config["n_states"]),
        base_model="lasso",
        lasso_alpha=0.005,
        window_size=int(config["window"]["size"]),
        stride=int(config["window"]["stride"]),
        aggregate="max",
        state_aggregate="mean",
        seed=seed,
    )
    diagnostics = stcg_v1_diagnostics(dataset.x, int(setting["max_lag"]), state_config)
    window_scores = diagnostics.window_scores
    bounds = tuple(tuple(b) for b in diagnostics.bounds)
    fitted_labels = np.asarray(diagnostics.labels)
    ceiling = static_ceiling(dataset.state_support, bounds, dataset.regimes)
    ceiling_direct = static_ceiling_direct(dataset.state_support, bounds, dataset.regimes)

    def add(method: str, pooled: float, delta: dict, ceiling_value: float | None = None):
        row = {"seed": seed, "method": method}
        row["pooled_temporal_auroc"] = pooled
        row["temporal_delta_auroc"] = delta["temporal_delta_auroc"]
        row["temporal_delta_edges"] = delta["temporal_delta_edges"]
        row["temporal_delta_edges_total"] = delta["temporal_delta_edges_total"]
        row["static_ceiling"] = ceiling_value if ceiling_value is not None else ceiling
        row["static_ceiling_direct"] = ceiling_direct
        rows.append(row)

    # Static constant predictions.
    window_mean = window_scores.mean(axis=0)
    window_max = window_scores.max(axis=0)
    bagged = np.stack(
        [
            window_scores[rng.integers(0, window_scores.shape[0], window_scores.shape[0])].mean(axis=0)
            for _ in range(int(config["bag_draws"]))
        ],
        axis=0,
    ).mean(axis=0)
    add("window_mean", pooled_for_static(window_mean, bounds, truth), delta_for_static(window_mean, bounds, truth))
    add("window_max", pooled_for_static(window_max, bounds, truth), delta_for_static(window_max, bounds, truth))
    add("bagged_mean", pooled_for_static(bagged, bounds, truth), delta_for_static(bagged, bounds, truth))

    # Random balanced partitions, averaged over draws.
    pooled_rand, delta_rand = [], []
    for _ in range(int(config["random_partition_draws"])):
        order = rng.permutation(window_scores.shape[0])
        labels = np.zeros(window_scores.shape[0], dtype=int)
        labels[order[: window_scores.shape[0] // 2]] = 1
        state_scores = _hard_state_scores(window_scores, labels)
        pooled_rand.append(pooled_for_stateful(state_scores, labels, bounds, truth))
        delta_rand.append(delta_for_stateful(state_scores, labels, bounds, truth))
    delta_mean = {
        "temporal_delta_auroc": float(np.nanmean([d["temporal_delta_auroc"] for d in delta_rand])),
        "temporal_delta_edges": float(np.nanmean([d["temporal_delta_edges"] for d in delta_rand])),
        "temporal_delta_edges_total": float(np.nanmean([d["temporal_delta_edges_total"] for d in delta_rand])),
    }
    add("random_balanced", float(np.mean(pooled_rand)), delta_mean)

    # Fitted hard partition (STCG-v1 K-means).
    add(
        "kmeans_hard",
        pooled_for_stateful(diagnostics.state_scores, fitted_labels, bounds, truth),
        delta_for_stateful(diagnostics.state_scores, fitted_labels, bounds, truth),
    )

    # Oracle label aggregation.
    oracle_labels = window_regime_labels(bounds, dataset.regimes)
    oracle_scores = _hard_state_scores(window_scores, oracle_labels)
    add(
        "oracle_label",
        pooled_for_stateful(oracle_scores, oracle_labels, bounds, truth),
        delta_for_stateful(oracle_scores, oracle_labels, bounds, truth),
    )

    # Raw window scores: each window emits its own edge scores, no aggregation.
    # This isolates the local time-localisation signal from the state aggregation.
    raw_labels = np.arange(window_scores.shape[0], dtype=int)
    add(
        "raw_window",
        pooled_for_stateful(window_scores, raw_labels, bounds, truth),
        delta_for_stateful(window_scores, raw_labels, bounds, truth),
    )

    # Contiguous temporal DP.
    n_windows = window_scores.shape[0]
    temporal_labels = np.zeros(n_windows, dtype=int)
    if n_windows >= 2:
        cost = np.full((n_windows + 1, n_windows + 1), np.inf)
        for start in range(n_windows):
            for end in range(start + 1, n_windows + 1):
                segment = _standardise(diagnostics.features)[start:end]
                cost[start, end] = float(((segment - segment.mean(axis=0)) ** 2).sum())
        dp = np.full((3, n_windows + 1), np.inf)
        back = np.zeros((3, n_windows + 1), dtype=int)
        dp[0, 0] = 0.0
        for count in range(1, 3):
            for end in range(1, n_windows + 1):
                for start in range(count - 1, end):
                    value = dp[count - 1, start] + cost[start, end]
                    if value < dp[count, end]:
                        dp[count, end] = value
                        back[count, end] = start
        end = n_windows
        for count in range(2, 0, -1):
            start = back[count, end]
            temporal_labels[start:end] = count - 1
            end = start
    temporal_scores = _hard_state_scores(window_scores, temporal_labels)
    add(
        "temporal_dp",
        pooled_for_stateful(temporal_scores, temporal_labels, bounds, truth),
        delta_for_stateful(temporal_scores, temporal_labels, bounds, truth),
    )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="Use seed_start..seed_start+n_seeds-1 instead of the config seed list.",
    )
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--output", default=str(OUT / "temporal_delta_detail.csv"))
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.seed_start is not None:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    else:
        seeds = args.seeds if args.seeds else list(config["seeds"])
    setting = config["settings"]["switching_core"]

    rows: list[dict] = []
    for seed in seeds:
        rows.extend(run_seed(setting, seed, config))
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    summary = (
        frame.groupby("method")
        .agg(
            n=("seed", "size"),
            pooled_temporal_auroc=("pooled_temporal_auroc", "mean"),
            temporal_delta_auroc=("temporal_delta_auroc", "mean"),
            temporal_delta_edges=("temporal_delta_edges", "mean"),
            temporal_delta_edges_total=("temporal_delta_edges_total", "mean"),
            static_ceiling=("static_ceiling", "mean"),
        )
        .reset_index()
    )
    summary_path = output.with_name(output.stem.replace("detail", "summary") + ".csv")
    summary.to_csv(summary_path, index=False)
    print(summary.round(4).to_string(index=False))

    # Ceiling sanity: closed form vs direct occupancy oracle.
    diff = float(np.nanmax(np.abs(frame.static_ceiling - frame.static_ceiling_direct)))
    print(f"\nmax |closed-form - direct| ceiling = {diff:.2e}")

    _write_statistics_and_table(frame, output.parent, output.stem.replace("detail", "stats"), output.stem.replace("detail", "tab"))


def _write_statistics_and_table(frame: pd.DataFrame, out_dir: Path, stats_name: str, tab_name: str) -> None:
    from scipy import stats as _stats

    from stcg.stats import bootstrap_ci

    def per_seed(method: str) -> np.ndarray:
        return frame[frame.method == method]["temporal_delta_auroc"].dropna().to_numpy()

    stats_rows = []
    for method in ("raw_window", "random_balanced", "kmeans_hard", "temporal_dp", "oracle_label"):
        values = per_seed(method)
        low, high = bootstrap_ci(values, draws=10000, seed=7)
        # One-sample t-test against the constant-predictor null of 0.5.
        test = _stats.ttest_1samp(values, 0.5)
        stats_rows.append(
            {
                "method": method,
                "n_seeds": int(values.size),
                "delta_mean": float(values.mean()),
                "ci95_low": low,
                "ci95_high": high,
                "t_vs_05": float(test.statistic),
                "p_vs_05": float(test.pvalue),
            }
        )
    stats_frame = pd.DataFrame(stats_rows)
    stats_frame.to_csv(out_dir / f"{stats_name}.csv", index=False)
    print("\nOne-sample tests against the constant-predictor null (0.5):")
    print(stats_frame.round(4).to_string(index=False))

    # Paired contrasts that matter for the interpretation.
    km = per_seed("kmeans_hard")
    rnd = per_seed("random_balanced")
    ora = per_seed("oracle_label")
    raw = per_seed("raw_window")
    for name, a, b in (
        ("kmeans_vs_random", km, rnd),
        ("oracle_vs_kmeans", ora, km),
        ("kmeans_vs_raw", km, raw),
        ("raw_vs_static05", raw, np.full_like(raw, 0.5)),
    ):
        diff = a - b
        low, high = bootstrap_ci(diff, draws=10000, seed=7)
        test = _stats.ttest_1samp(diff, 0.0)
        print(
            f"{name}: mean {diff.mean():.4f} [{low:.4f},{high:.4f}] "
            f"p={test.pvalue:.4g} n={diff.size}"
        )

    # Reviewer's table: pooled temporal AUROC vs changing-edge temporal AUROC.
    order = [
        ("window_mean", "Window mean"),
        ("window_max", "Window max"),
        ("bagged_mean", "Bagged mean"),
        ("raw_window", "Raw window (no grouping)"),
        ("random_balanced", "Random balanced"),
        ("kmeans_hard", "K-means hard"),
        ("temporal_dp", "Temporal DP"),
        ("oracle_label", "Oracle label"),
    ]
    n_edges = float(frame.temporal_delta_edges.mean())
    lines = []
    for key, label in order:
        sub = frame[frame.method == key]
        pooled = float(sub.pooled_temporal_auroc.mean())
        delta = float(sub.temporal_delta_auroc.mean())
        lines.append(f"{label} & {pooled:.3f} & {delta:.3f} \\\\")
    static_ceiling = float(frame.static_ceiling.mean())
    tex = (
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "Method & pooled temporal AUROC & $\\mathrm{T\\!-\\!AUC}_{\\Delta}$ \\\\\n"
        "\\midrule\n"
        + "\n".join(lines)
        + "\n\\midrule\n"
        f"Static ceiling $\\mathrm{{AUC}}^*_{{\\mathrm{{static}}}}$ & \\multicolumn{{2}}{{c}}{{{static_ceiling:.3f}}} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    (out_dir / f"{tab_name}.tex").write_text(tex, encoding="utf-8")
    print(f"\nchanging edges per sequence (mean): {n_edges:.1f}")
    print(f"static ceiling AUC*: {static_ceiling:.4f}")
    print(f"wrote {out_dir / f'{tab_name}.tex'}")


if __name__ == "__main__":
    main()
