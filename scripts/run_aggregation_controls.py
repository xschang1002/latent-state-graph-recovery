"""WP3: does the gain come from structure, from averaging, or from max?

Every method in this script sees exactly the same window-level Lasso scores.
The only thing that changes is how windows are grouped and aggregated:

- ``window_mean`` / ``window_max``: no states at all.
- ``random_balanced``: random groups of the same size as the fitted states.
- ``bagged_mean``: bootstrap mean of window scores (state-free bagging).
- ``kmeans_hard``: the STCG-v1 partition (K-means on window scores).
- ``kmeans_soft``: Gaussian-mixture responsibilities, weighted average.
- ``temporal_dp``: contiguous segmentation by dynamic programming.
- ``oracle_label``: aggregate with the true window labels.
- ``oracle_refit``: re-fit Lasso on each true state's contiguous runs.

The misassignment sweep corrupts the fitted labels at rate ``eta`` and reports
union, regime-specific, and state metrics on the same scale, so that a robustness
difference between estimands is visible rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stcg import STCGStateConfig, stcg_v1_diagnostics  # noqa: E402
from stcg.baselines import lasso_var_scores  # noqa: E402
from stcg.evaluation import (  # noqa: E402
    GraphPrediction,
    GraphTruth,
    evaluate_prediction,
    union_metrics,
    window_regime_labels,
)
from stcg.mechanism import MechanismConfig, generate_mechanism, misassign_labels  # noqa: E402

OUT = ROOT / "results" / "aggregation"


def _standardise(features: np.ndarray) -> np.ndarray:
    centred = features - features.mean(axis=0, keepdims=True)
    return centred / (centred.std(axis=0, keepdims=True) + 1e-8)


def _weighted_state_scores(
    window_scores: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """weights: (n_windows, n_states) -> (n_states, lags, nodes, nodes)."""

    totals = weights.sum(axis=0)
    scores = np.einsum("wk,wlij->klij", weights, window_scores)
    return scores / np.maximum(totals, 1e-8)[:, None, None, None]


def _hard_state_scores(window_scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    states = np.unique(labels)
    return np.stack([window_scores[labels == state].mean(axis=0) for state in states], axis=0)


def _temporal_dp(features: np.ndarray, n_states: int) -> np.ndarray:
    """Contiguous segmentation minimising within-segment squared error."""

    n_windows = features.shape[0]
    if n_windows < n_states:
        return np.zeros(n_windows, dtype=int)
    cost = np.full((n_windows + 1, n_windows + 1), np.inf)
    for start in range(n_windows):
        for end in range(start + 1, n_windows + 1):
            segment = features[start:end]
            cost[start, end] = float(((segment - segment.mean(axis=0)) ** 2).sum())
    dp = np.full((n_states + 1, n_windows + 1), np.inf)
    back = np.zeros((n_states + 1, n_windows + 1), dtype=int)
    dp[0, 0] = 0.0
    for count in range(1, n_states + 1):
        for end in range(1, n_windows + 1):
            for start in range(count - 1, end):
                value = dp[count - 1, start] + cost[start, end]
                if value < dp[count, end]:
                    dp[count, end] = value
                    back[count, end] = start
    labels = np.zeros(n_windows, dtype=int)
    end = n_windows
    for count in range(n_states, 0, -1):
        start = back[count, end]
        labels[start:end] = count - 1
        end = start
    return labels


def _oracle_refit_scores(
    x: np.ndarray, regimes: np.ndarray, max_lag: int, alpha: float
) -> np.ndarray:
    """Re-fit Lasso on each true state's contiguous runs, then average."""

    state_ids = np.unique(regimes)
    rows = []
    for state in state_ids:
        mask = regimes == state
        # contiguous runs
        padded = np.concatenate([[False], mask, [False]])
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        runs = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]
        run_scores = []
        for start, end in runs:
            if end - start < 2 * max_lag + 10:
                continue
            run_scores.append(lasso_var_scores(x[start:end], max_lag, alpha))
        if run_scores:
            rows.append(np.mean(run_scores, axis=0))
        else:
            rows.append(np.zeros((max_lag, x.shape[1], x.shape[1])))
    return np.stack(rows, axis=0)


def _evaluate(
    union_scores: np.ndarray,
    truth: GraphTruth,
    state_scores: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    bounds: tuple[tuple[int, int], ...] | None = None,
) -> dict[str, float]:
    prediction = GraphPrediction(
        union_scores=union_scores,
        state_scores=state_scores,
        window_assignments=labels,
        window_bounds=bounds,
    )
    metrics = union_metrics(union_scores, truth.support, prefix="union_")
    if state_scores is not None and labels is not None and bounds is not None:
        window_truth = window_regime_labels(bounds, truth.regimes)
        from stcg.evaluation import (
            per_regime_metrics,
            temporal_graph_metrics,
            temporal_state_metrics,
        )

        metrics.update(
            per_regime_metrics(state_scores, truth.state_support, labels, window_truth)
        )
        metrics.update(temporal_state_metrics(labels, bounds, truth.regimes))
        metrics.update(
            temporal_graph_metrics(
                state_scores, labels, bounds, truth.regimes, truth.state_support
            )
        )
    elif bounds is not None:
        # State-free method: score the constant graph as a time-varying one.
        from stcg.evaluation import static_temporal_graph_metrics

        metrics.update(
            static_temporal_graph_metrics(union_scores, bounds, truth.regimes, truth.state_support)
        )
    return metrics


def run_setting(
    name: str,
    setting: dict,
    seeds: list[int],
    config: dict,
) -> list[dict]:
    rows: list[dict] = []
    window = config["window"]
    n_states = int(config["n_states"])
    for seed in seeds:
        mechanism = MechanismConfig(**setting, seed=seed)
        dataset = generate_mechanism(mechanism)
        state_config = STCGStateConfig(
            n_states=n_states,
            base_model="lasso",
            lasso_alpha=0.005,
            window_size=int(window["size"]),
            stride=int(window["stride"]),
            aggregate="max",
            state_aggregate="mean",
            seed=seed,
        )
        diagnostics = stcg_v1_diagnostics(dataset.x, int(setting["max_lag"]), state_config)
        window_scores = diagnostics.window_scores
        features = _standardise(diagnostics.features)
        bounds = tuple(tuple(b) for b in diagnostics.bounds)
        truth = GraphTruth(
            support=dataset.union_support,
            state_support=dataset.state_support,
            regimes=dataset.regimes,
        )
        fitted_labels = np.asarray(diagnostics.labels)
        rng = np.random.default_rng(seed + 991)

        def add(method: str, eta: float, metrics: dict, extra: dict | None = None) -> None:
            row = {"setting": name, "seed": seed, "method": method, "eta": eta}
            row.update(metrics)
            if extra:
                row.update(extra)
            rows.append(row)

        # No-state references.
        add("window_mean", 0.0, _evaluate(window_scores.mean(axis=0), truth, bounds=bounds))
        add("window_max", 0.0, _evaluate(window_scores.max(axis=0), truth, bounds=bounds))

        # State-free bagging.
        bagged = []
        for _ in range(int(config["bag_draws"])):
            index = rng.integers(0, window_scores.shape[0], window_scores.shape[0])
            bagged.append(window_scores[index].mean(axis=0))
        add("bagged_mean", 0.0, _evaluate(np.mean(bagged, axis=0), truth, bounds=bounds))

        # Random balanced partitions, averaged over draws.
        random_metrics = []
        for _ in range(int(config["random_partition_draws"])):
            order = rng.permutation(window_scores.shape[0])
            labels = np.zeros(window_scores.shape[0], dtype=int)
            labels[order[: window_scores.shape[0] // 2]] = 1
            state_scores = _hard_state_scores(window_scores, labels)
            random_metrics.append(
                _evaluate(state_scores.max(axis=0), truth, state_scores, labels, bounds)
            )
        averaged = pd.DataFrame(random_metrics).mean(numeric_only=True).to_dict()
        add("random_balanced", 0.0, averaged)

        # Fitted hard partition.
        add(
            "kmeans_hard",
            0.0,
            _evaluate(diagnostics.scores, truth, diagnostics.state_scores, fitted_labels, bounds),
        )

        # Soft memberships.
        if features.shape[0] >= n_states + 1:
            mixture = GaussianMixture(
                n_components=n_states, covariance_type="diag", n_init=3, random_state=seed
            )
            mixture.fit(features)
            responsibilities = mixture.predict_proba(features)
            soft_scores = _weighted_state_scores(window_scores, responsibilities)
            soft_labels = responsibilities.argmax(axis=1)
            add(
                "kmeans_soft",
                0.0,
                _evaluate(soft_scores.max(axis=0), truth, soft_scores, soft_labels, bounds),
            )
        else:
            soft_scores = None

        # Contiguous segmentation.
        temporal_labels = _temporal_dp(features, n_states)
        temporal_scores = _hard_state_scores(window_scores, temporal_labels)
        add(
            "temporal_dp",
            0.0,
            _evaluate(temporal_scores.max(axis=0), truth, temporal_scores, temporal_labels, bounds),
        )

        # Oracle label aggregation.
        oracle_labels = window_regime_labels(bounds, dataset.regimes)
        oracle_scores = _hard_state_scores(window_scores, oracle_labels)
        add(
            "oracle_label",
            0.0,
            _evaluate(oracle_scores.max(axis=0), truth, oracle_scores, oracle_labels, bounds),
        )

        # Oracle labels corrupted at rate eta: tests the theory's eta directly,
        # because the corruption acts on true labels rather than on fitted ones.
        for eta in config["misassignment"]:
            corrupted = misassign_labels(oracle_labels, float(eta), seed=seed)
            scores = _hard_state_scores(window_scores, corrupted)
            add(
                "oracle_eta",
                float(eta),
                _evaluate(scores.max(axis=0), truth, scores, corrupted, bounds),
            )

        # Oracle regime refit.
        refit_scores = _oracle_refit_scores(
            dataset.x, dataset.regimes, int(setting["max_lag"]), 0.005
        )
        add("oracle_refit", 0.0, _evaluate(refit_scores.max(axis=0), truth))

        # Misassignment sweep on the fitted partition.
        for eta in config["misassignment"]:
            labels = misassign_labels(fitted_labels, float(eta), seed=seed)
            scores = _hard_state_scores(window_scores, labels)
            add(
                "kmeans_eta",
                float(eta),
                _evaluate(scores.max(axis=0), truth, scores, labels, bounds),
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "aggregation_controls.yaml"))
    parser.add_argument("--output", default=str(OUT / "aggregation_controls.csv"))
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seeds = args.seeds if args.seeds else list(config["seeds"])
    rows: list[dict] = []
    for name, setting in config["settings"].items():
        rows.extend(run_setting(name, setting, seeds, config))
        print(f"done {name}", flush=True)

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    summary = (
        frame.groupby(["setting", "method", "eta"], dropna=False)
        .agg(
            n=("union_auroc", "size"),
            union_auroc=("union_auroc", "mean"),
            union_auprc=("union_auprc", "mean"),
            regime_auprc=("regime_auprc_macro", "mean"),
            regime_auroc=("regime_auroc_macro", "mean"),
            temporal_graph=("temporal_graph_auroc", "mean"),
            regime_f1=("regime_f1_macro", "mean"),
            state_ari=("state_ari", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output.with_name(output.stem + "_summary.csv"), index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
