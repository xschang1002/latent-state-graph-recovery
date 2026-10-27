"""Future-facing transition warning on Hydraulic Systems labels.

For each cycle, features are computed only from previous cycles. The target is
whether a chosen condition label changes within the next horizon cycles.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

# Set before sklearn imports so joblib/loky does not oversubscribe CPU cores
# when KMeans runs on Windows.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from _common import fmt, parse_columns, write_csv
from _state_selection import select_state_candidate_index
from stcg import STCGStateConfig, ridge_var_scores, stcg_v1_diagnostics
from stcg.self_supervised import _min_cluster_fraction, _nonself_flatten

warnings.filterwarnings("ignore", category=ConvergenceWarning)


METHOD_MAJORITY = "Train Majority"
METHOD_SENSOR = "Sensor Logistic"
METHOD_GRAPH = "Graph Ridge Logistic"
METHOD_SENSOR_GRAPH = "Sensor+Graph Logistic"
METHOD_STCG = "STCG-v1 Selected Logistic"
METHOD_SENSOR_STCG = "Sensor+STCG-v1 Logistic"
METRICS = ["balanced_accuracy", "precision", "recall", "f1", "average_precision", "roc_auc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=ROOT / "data" / "raw" / "HydraulicSystems.csv")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=ROOT / "data" / "processed" / "HydraulicSystems_labels.csv",
    )
    parser.add_argument("--dataset-name", type=str, default="HydraulicSystems")
    parser.add_argument(
        "--label-columns",
        nargs="+",
        default=["valve_condition", "stable_flag"],
        help="Cycle-level label columns to warn transitions for.",
    )
    parser.add_argument("--columns", type=str, default="TS1,TS2,TS3,TS4,VS1,CE,CP,SE")
    parser.add_argument("--cycle-column", type=str, default="cycle")
    parser.add_argument("--horizon-cycles", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--lookback-cycles", type=int, default=40)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--stcg-v1-base", choices=["ridge", "lasso"], default="ridge")
    parser.add_argument("--stcg-v1-lasso-alpha", type=float, default=0.005)
    parser.add_argument("--stcg-state-counts", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--stcg-window-size", type=int, default=24)
    parser.add_argument("--stcg-window-stride", type=int, default=8)
    parser.add_argument(
        "--stcg-selection-objective",
        choices=["stable", "transition"],
        default="transition",
        help="Self-supervised objective used to select the STCG state count for warning features.",
    )
    parser.add_argument("--stcg-min-improvement", type=float, default=0.04)
    parser.add_argument("--stcg-min-temporal-coherence", type=float, default=0.55)
    parser.add_argument("--stcg-min-cluster-fraction", type=float, default=0.05)
    parser.add_argument(
        "--stcg-selection-policy", choices=["reject_nonpositive", "legacy"],
        default="reject_nonpositive",
        help="Use the fitted K=1 candidate when no score is positive; legacy preserves historical all-zero ties.",
    )
    parser.add_argument("--stcg-seed", type=int, default=70_000)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument(
        "--threshold-metric",
        choices=["f1", "balanced_accuracy"],
        default="f1",
        help="Validation metric used to calibrate warning thresholds.",
    )
    parser.add_argument("--min-test-positives", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "results" / "figures")
    return parser.parse_args()


def load_cycle_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    sensors = pd.read_csv(args.input_csv)
    labels = pd.read_csv(args.labels_csv)
    if args.cycle_column not in labels.columns:
        raise ValueError(f"{args.labels_csv} is missing cycle column: {args.cycle_column}")
    columns = parse_columns(args.columns)
    missing = [column for column in columns if column not in sensors.columns]
    if missing:
        raise ValueError(f"{args.input_csv} is missing sensor columns: {missing}")
    for label_column in args.label_columns:
        if label_column not in labels.columns:
            raise ValueError(f"{args.labels_csv} is missing label column: {label_column}")
    if len(sensors) != len(labels):
        raise ValueError("Sensor rows and label rows must align")

    # A future observation must never impute a past warning feature. Forward
    # filling is causal; initial gaps cannot be filled without future data.
    frame = sensors[columns].apply(pd.to_numeric, errors="coerce").ffill()
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Sensor data contain initial missing or non-finite values; causal forward fill cannot resolve them")
    cycles = labels[args.cycle_column].astype(int)
    if not cycles.is_monotonic_increasing:
        raise ValueError("Sensor and label rows must be in chronological cycle order")
    frame[args.cycle_column] = cycles.to_numpy()
    cycle_mean = frame.groupby(args.cycle_column, sort=True)[columns].mean()
    cycle_std = frame.groupby(args.cycle_column, sort=True)[columns].std().fillna(0.0)
    label_table = labels.groupby(args.cycle_column, sort=True)[args.label_columns].first()

    common_cycles = cycle_mean.index.intersection(label_table.index)
    cycle_features = pd.concat(
        {
            "mean": cycle_mean.loc[common_cycles],
            "std": cycle_std.loc[common_cycles],
        },
        axis=1,
    )
    cycle_features.columns = [f"{kind}_{column}" for kind, column in cycle_features.columns]
    return cycle_features, label_table.loc[common_cycles], columns


def transition_targets(labels: list[str], horizon: int) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    targets = []
    for idx in range(len(labels) - horizon):
        current = labels[idx]
        future = labels[idx + 1 : idx + horizon + 1]
        targets.append(any(value != current for value in future))
    return np.asarray(targets, dtype=int)


def sensor_lookback_features(cycle_values: np.ndarray, end_idx: int, lookback: int) -> np.ndarray:
    window = cycle_values[end_idx - lookback : end_idx]
    return np.concatenate([window.mean(axis=0), window.std(axis=0), window[-1] - window[0]])


def graph_lookback_features(
    cycle_means: np.ndarray,
    end_idx: int,
    lookback: int,
    max_lag: int,
    ridge_alpha: float,
) -> np.ndarray:
    window = cycle_means[end_idx - lookback : end_idx]
    scores = ridge_var_scores(window, max_lag=max_lag, alpha=ridge_alpha)
    _, n_sources, _ = scores.shape
    mask = np.ones_like(scores, dtype=bool)
    for node in range(n_sources):
        mask[:, node, node] = False
    return scores[mask]


def stcg_selection_score(
    improvement: float,
    temporal_coherence: float,
    min_cluster_fraction: float,
    inferred_state_count: int,
    min_improvement: float = 0.04,
    min_temporal_coherence: float = 0.55,
    min_fraction: float = 0.05,
) -> tuple[float, float]:
    if inferred_state_count <= 1:
        return 0.0, 0.0
    if improvement < min_improvement or temporal_coherence < min_temporal_coherence:
        return 0.0, 0.0
    if min_cluster_fraction < min_fraction:
        return 0.0, 0.0
    balance = min(1.0, float(min_cluster_fraction * inferred_state_count))
    parsimony = math.log2(float(inferred_state_count) + 1.0)
    return float(improvement * temporal_coherence * balance / parsimony), balance


def stcg_transition_selection_score(
    improvement: float,
    temporal_coherence: float,
    min_cluster_fraction: float,
    inferred_state_count: int,
    min_improvement: float = 0.04,
    min_fraction: float = 0.05,
) -> tuple[float, float]:
    if inferred_state_count < 2:
        return 0.0, 0.0
    if improvement < min_improvement or min_cluster_fraction < min_fraction:
        return 0.0, 0.0
    balance = min(1.0, float(min_cluster_fraction * inferred_state_count))
    transition_bonus = 1.0 - min(1.0, max(0.0, float(temporal_coherence)))
    parsimony = math.log2(float(inferred_state_count) + 1.0)
    return float(improvement * balance * (0.5 + 0.5 * transition_bonus) / parsimony), balance


def stcg_warning_selection_score(
    diagnostics: object,
    min_fraction: float,
    inferred_count: int,
    args: argparse.Namespace,
) -> tuple[float, float]:
    objective = getattr(args, "stcg_selection_objective", "transition")
    if objective == "stable":
        return stcg_selection_score(
            float(diagnostics.state_improvement),
            float(diagnostics.temporal_coherence),
            min_fraction,
            inferred_count,
            min_improvement=args.stcg_min_improvement,
            min_temporal_coherence=args.stcg_min_temporal_coherence,
            min_fraction=args.stcg_min_cluster_fraction,
        )
    return stcg_transition_selection_score(
        float(diagnostics.state_improvement),
        float(diagnostics.temporal_coherence),
        min_fraction,
        inferred_count,
        min_improvement=args.stcg_min_improvement,
        min_fraction=args.stcg_min_cluster_fraction,
    )


def stcg_config(args: argparse.Namespace, state_count: int, seed: int) -> STCGStateConfig:
    return STCGStateConfig(
        n_states=state_count,
        base_model=args.stcg_v1_base,
        ridge_alpha=args.ridge_alpha,
        lasso_alpha=args.stcg_v1_lasso_alpha,
        window_size=args.stcg_window_size,
        stride=args.stcg_window_stride,
        aggregate="max",
        state_aggregate="mean",
        auto_states=False,
        contrast_mix=0.0,
        seed=seed,
    )


def stcg_state_dynamics_features(labels: np.ndarray, max_states: int) -> np.ndarray:
    if max_states <= 0:
        raise ValueError("max_states must be positive")
    labels = np.asarray(labels)
    feature_count = max_states * 3 + 6
    if labels.size == 0:
        return np.zeros(feature_count, dtype=float)

    values, counts = np.unique(labels, return_counts=True)
    order = sorted(range(len(values)), key=lambda idx: (-int(counts[idx]), int(values[idx])))
    ordered_counts = counts[order].astype(float)
    take = min(max_states, len(ordered_counts))

    fractions = np.zeros(max_states, dtype=float)
    fractions[:take] = ordered_counts[:take] / float(labels.size)

    rank_by_value = {values[idx]: rank for rank, idx in enumerate(order[:max_states])}
    first_state = np.zeros(max_states, dtype=float)
    last_state = np.zeros(max_states, dtype=float)
    first_rank = rank_by_value.get(labels[0])
    last_rank = rank_by_value.get(labels[-1])
    if first_rank is not None:
        first_state[first_rank] = 1.0
    if last_rank is not None:
        last_state[last_rank] = 1.0

    switches = labels[1:] != labels[:-1]
    transition_rate = float(np.mean(switches)) if switches.size else 0.0
    last_run = 1
    for previous in labels[-2::-1]:
        if previous != labels[-1]:
            break
        last_run += 1
    probabilities = counts.astype(float) / float(labels.size)
    entropy = 0.0
    if probabilities.size > 1:
        entropy = float(-np.sum(probabilities * np.log(probabilities)) / math.log(float(probabilities.size)))
    dominant_fraction = float(probabilities.max())
    minority_fraction = float(probabilities.min())
    active_state_fraction = float(min(1.0, probabilities.size / float(max_states)))

    return np.concatenate(
        [
            fractions,
            last_state,
            first_state,
            np.asarray(
                [
                    transition_rate,
                    float(last_run) / float(labels.size),
                    entropy,
                    dominant_fraction,
                    minority_fraction,
                    active_state_fraction,
                ],
                dtype=float,
            ),
        ]
    )


def stcg_selected_lookback_features(
    cycle_means: np.ndarray,
    end_idx: int,
    lookback: int,
    max_lag: int,
    args: argparse.Namespace,
) -> np.ndarray:
    window = cycle_means[end_idx - lookback : end_idx]
    candidate_keys: list[tuple[float, float, int]] = []
    candidate_feature_rows: list[np.ndarray] = []

    for state_count in args.stcg_state_counts:
        diagnostics = stcg_v1_diagnostics(
            window,
            max_lag=max_lag,
            config=stcg_config(args, state_count, args.stcg_seed + end_idx * 31 + state_count),
        )
        inferred_count = int(np.unique(diagnostics.labels).size)
        min_fraction = _min_cluster_fraction(diagnostics.labels)
        state_dynamics = stcg_state_dynamics_features(
            diagnostics.labels,
            max_states=max(args.stcg_state_counts),
        )
        score, balance = stcg_warning_selection_score(diagnostics, min_fraction, inferred_count, args)
        candidate_features = np.concatenate(
            [
                _nonself_flatten(diagnostics.scores),
                state_dynamics,
                np.asarray(
                    [
                        float(state_count),
                        float(inferred_count),
                        score,
                        float(diagnostics.state_improvement),
                        float(diagnostics.temporal_coherence),
                        min_fraction,
                        balance,
                    ],
                    dtype=float,
                ),
            ]
        )
        candidate_keys.append((score, float(diagnostics.temporal_coherence), state_count))
        candidate_feature_rows.append(candidate_features)

    selected = select_state_candidate_index(
        candidate_keys, policy=getattr(args, "stcg_selection_policy", "reject_nonpositive")
    )
    return candidate_feature_rows[selected]


def build_warning_design(
    cycle_features: pd.DataFrame,
    cycle_labels: pd.Series,
    horizon: int,
    lookback: int,
    max_lag: int,
    ridge_alpha: float,
    args: argparse.Namespace,
    feature_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    if lookback <= max_lag + 2:
        raise ValueError("lookback must be larger than max_lag + 2")
    labels = cycle_labels.astype(str).tolist()
    targets = transition_targets(labels, horizon)
    values = cycle_features.to_numpy(dtype=float)
    mean_columns = [column for column in cycle_features.columns if column.startswith("mean_")]
    cycle_means = cycle_features[mean_columns].to_numpy(dtype=float)

    sensor_rows: list[np.ndarray] = []
    graph_rows: list[np.ndarray] = []
    stcg_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    cycles: list[int] = []
    if feature_cache is None:
        feature_cache = {}
    for idx in range(lookback, len(targets)):
        if idx not in feature_cache:
            feature_cache[idx] = (
                sensor_lookback_features(values, idx, lookback),
                graph_lookback_features(cycle_means, idx, lookback, max_lag, ridge_alpha),
                stcg_selected_lookback_features(cycle_means, idx, lookback, max_lag, args),
            )
        sensor_features, graph_features, stcg_features = feature_cache[idx]
        sensor_rows.append(sensor_features)
        graph_rows.append(graph_features)
        stcg_rows.append(stcg_features)
        y_rows.append(int(targets[idx]))
        cycles.append(int(cycle_features.index[idx]))
    return (
        np.stack(sensor_rows),
        np.stack(graph_rows),
        np.stack(stcg_rows),
        np.asarray(y_rows, dtype=int),
        cycles,
    )


def chronological_split(n_rows: int, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    split = int(round(n_rows * train_fraction))
    split = min(max(1, split), n_rows - 1)
    return np.arange(0, split, dtype=int), np.arange(split, n_rows, dtype=int)


def chronological_train_validation_test_split(
    n_rows: int,
    train_fraction: float,
    validation_fraction: float,
    horizon: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split target start positions, purging labels observed across boundaries.

    A target at position i reads labels through i + horizon (inclusive).
    Keep the original chronological boundaries and discard the last horizon
    targets before validation and test. Earlier features may remain available
    as historical lookback context; no labels at or beyond the next boundary
    can supervise the preceding segment. Use horizon=0 only for point targets.
    """
    if n_rows < 3:
        raise ValueError("At least three target rows are required")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 <= validation_fraction < train_fraction:
        raise ValueError("validation_fraction must be non-negative and smaller than train_fraction")
    train_end = int(round((train_fraction - validation_fraction) * n_rows))
    validation_end = int(round(train_fraction * n_rows))
    train_end = min(max(1, train_end), n_rows - 2)
    validation_end = min(max(train_end + 1, validation_end), n_rows - 1)
    if train_end <= horizon or validation_end - train_end <= horizon:
        raise ValueError("Train and validation segments must each contain more rows than the target horizon")
    return (
        np.arange(0, train_end - horizon, dtype=int),
        np.arange(train_end, validation_end - horizon, dtype=int),
        np.arange(validation_end, n_rows, dtype=int),
    )


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return math.nan
    return float(roc_auc_score(y_true, scores))


def safe_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return math.nan
    return float(average_precision_score(y_true, scores))


def metrics_from_scores(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": safe_average_precision(y_true, y_score),
        "roc_auc": safe_auc(y_true, y_score),
    }


def threshold_metric_value(y_true: np.ndarray, y_score: np.ndarray, threshold: float, metric: str) -> float:
    y_pred = (y_score >= threshold).astype(int)
    if metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    raise ValueError(f"Unknown threshold metric: {metric}")


def calibrate_threshold(
    y_validation: np.ndarray,
    validation_score: np.ndarray,
    metric: str,
) -> tuple[float, float]:
    if validation_score.size == 0:
        return 0.5, math.nan
    candidates = sorted(set(float(value) for value in validation_score))
    candidates = [0.0, *candidates, 1.0]
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        score = threshold_metric_value(y_validation, validation_score, threshold, metric)
        if score > best_score or (score == best_score and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_threshold = threshold
            best_score = score
    return float(best_threshold), float(best_score)


def fit_logistic_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if np.unique(y_train).size < 2:
        validation_scores = np.full(x_validation.shape[0], float(np.mean(y_train)))
        test_scores = np.full(x_test.shape[0], float(np.mean(y_train)))
        return validation_scores, test_scores
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=5000, solver="lbfgs"),
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_validation)[:, 1], model.predict_proba(x_test)[:, 1]


def evaluate_method(
    dataset: str,
    label_column: str,
    horizon: int,
    lookback: int,
    method: str,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
    validation_scores: np.ndarray,
    scores: np.ndarray,
    threshold_metric: str,
) -> dict[str, object]:
    threshold, validation_metric = calibrate_threshold(y_validation, validation_scores, threshold_metric)
    row: dict[str, object] = {
        "dataset": dataset,
        "label_column": label_column,
        "horizon_cycles": horizon,
        "lookback_cycles": lookback,
        "split_policy": "purged_target_horizon",
        "purge_horizon_cycles": horizon,
        "method": method,
        "train_rows": int(y_train.size),
        "validation_rows": int(y_validation.size),
        "test_rows": int(y_test.size),
        "train_positive_rate": float(np.mean(y_train)),
        "validation_positive_rate": float(np.mean(y_validation)),
        "test_positive_rate": float(np.mean(y_test)),
        "train_positives": int(np.sum(y_train)),
        "validation_positives": int(np.sum(y_validation)),
        "test_positives": int(np.sum(y_test)),
        "threshold_metric": threshold_metric,
        "threshold": threshold,
        "validation_threshold_metric": validation_metric,
    }
    row.update(metrics_from_scores(y_test, scores, threshold=threshold))
    return row


def evaluate_label_horizon(
    dataset: str,
    label_column: str,
    horizon: int,
    cycle_features: pd.DataFrame,
    cycle_labels: pd.DataFrame,
    args: argparse.Namespace,
    feature_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[dict[str, object]]:
    sensor_x, graph_x, stcg_x, y, _cycles = build_warning_design(
        cycle_features,
        cycle_labels[label_column],
        horizon=horizon,
        lookback=args.lookback_cycles,
        max_lag=args.max_lag,
        ridge_alpha=args.ridge_alpha,
        args=args,
        feature_cache=feature_cache,
    )
    train_idx, validation_idx, test_idx = chronological_train_validation_test_split(
        len(y),
        args.train_fraction,
        args.validation_fraction,
        horizon=horizon,
    )
    y_train = y[train_idx]
    y_validation = y[validation_idx]
    y_test = y[test_idx]
    if int(np.sum(y_test)) < args.min_test_positives:
        return []

    majority_score = np.full(y_test.shape[0], float(np.mean(y_train)))
    majority_validation_score = np.full(y_validation.shape[0], float(np.mean(y_train)))
    sensor_validation_score, sensor_score = fit_logistic_scores(
        sensor_x[train_idx],
        y_train,
        sensor_x[validation_idx],
        sensor_x[test_idx],
    )
    graph_validation_score, graph_score = fit_logistic_scores(
        graph_x[train_idx],
        y_train,
        graph_x[validation_idx],
        graph_x[test_idx],
    )
    stcg_validation_score, stcg_score = fit_logistic_scores(
        stcg_x[train_idx],
        y_train,
        stcg_x[validation_idx],
        stcg_x[test_idx],
    )
    combined_x = np.concatenate([sensor_x, graph_x], axis=1)
    combined_validation_score, combined_score = fit_logistic_scores(
        combined_x[train_idx],
        y_train,
        combined_x[validation_idx],
        combined_x[test_idx],
    )
    sensor_stcg_x = np.concatenate([sensor_x, stcg_x], axis=1)
    sensor_stcg_validation_score, sensor_stcg_score = fit_logistic_scores(
        sensor_stcg_x[train_idx],
        y_train,
        sensor_stcg_x[validation_idx],
        sensor_stcg_x[test_idx],
    )

    rows = [
        evaluate_method(dataset, label_column, horizon, args.lookback_cycles, METHOD_MAJORITY, y_train, y_validation, y_test, majority_validation_score, majority_score, args.threshold_metric),
        evaluate_method(dataset, label_column, horizon, args.lookback_cycles, METHOD_SENSOR, y_train, y_validation, y_test, sensor_validation_score, sensor_score, args.threshold_metric),
        evaluate_method(dataset, label_column, horizon, args.lookback_cycles, METHOD_GRAPH, y_train, y_validation, y_test, graph_validation_score, graph_score, args.threshold_metric),
        evaluate_method(dataset, label_column, horizon, args.lookback_cycles, METHOD_SENSOR_GRAPH, y_train, y_validation, y_test, combined_validation_score, combined_score, args.threshold_metric),
        evaluate_method(dataset, label_column, horizon, args.lookback_cycles, METHOD_STCG, y_train, y_validation, y_test, stcg_validation_score, stcg_score, args.threshold_metric),
        evaluate_method(dataset, label_column, horizon, args.lookback_cycles, METHOD_SENSOR_STCG, y_train, y_validation, y_test, sensor_stcg_validation_score, sensor_stcg_score, args.threshold_metric),
    ]
    for row in rows:
        row["stcg_selection_policy"] = getattr(args, "stcg_selection_policy", "reject_nonpositive")
    return rows


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Hydraulic Transition Warning",
        "",
        "Features are computed only from past cycles. The target is whether the selected label changes within the next horizon cycles.",
        "Decision thresholds are selected on the chronological validation segment.",
        "The final horizon targets before each train/validation and validation/test boundary are purged, so target labels never cross a split boundary. Missing sensor values use causal forward fill only.",
        f"STCG selection policy: `{rows[0].get('stcg_selection_policy', 'unknown')}`. The default rejects all-nonpositive candidate sets to their fitted K=1 candidate.",
        "",
        "| Label | Horizon | Method | Threshold | Test Pos. Rate | Balanced Acc. | Precision | Recall | F1 | AP | ROC AUC |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {horizon} | {method} | {threshold} | {pos} | {bal} | {precision} | {recall} | {f1} | {ap} | {auc} |".format(
                label=row["label_column"],
                horizon=row["horizon_cycles"],
                method=row["method"],
                threshold=fmt(row["threshold"]),
                pos=fmt(row["test_positive_rate"]),
                bal=fmt(row["balanced_accuracy"]),
                precision=fmt(row["precision"]),
                recall=fmt(row["recall"]),
                f1=fmt(row["f1"]),
                ap=fmt(row["average_precision"]),
                auc=fmt(row["roc_auc"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_best_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["label_column"]), int(row["horizon_cycles"]))].append(row)
    best_rows = []
    for (label, horizon), items in sorted(grouped.items()):
        best = max(items, key=lambda row: (float(row["average_precision"]), float(row["f1"])))
        majority = next(row for row in items if row["method"] == METHOD_MAJORITY)
        best_rows.append((label, horizon, best, majority))

    lines = [
        "# Hydraulic Transition Warning Best Methods",
        "",
        "| Label | Horizon | Best Method | Threshold | AP | AP Gain vs Majority | F1 | ROC AUC |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for label, horizon, best, majority in best_rows:
        ap = float(best["average_precision"])
        majority_ap = float(majority["average_precision"])
        lines.append(
            "| {label} | {horizon} | {method} | {threshold} | {ap} | {gain} | {f1} | {auc} |".format(
                label=label,
                horizon=horizon,
                method=best["method"],
                threshold=fmt(best["threshold"]),
                ap=fmt(ap),
                gain=fmt(ap - majority_ap),
                f1=fmt(best["f1"]),
                auc=fmt(best["roc_auc"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, rows: list[dict[str, object]]) -> None:
    labels = sorted(set(str(row["label_column"]) for row in rows))
    fig, axes = plt.subplots(len(labels), 1, figsize=(8.8, 3.4 * len(labels)), squeeze=False)
    for ax, label in zip(axes[:, 0], labels):
        label_rows = [row for row in rows if str(row["label_column"]) == label]
        methods = [
            METHOD_MAJORITY,
            METHOD_SENSOR,
            METHOD_GRAPH,
            METHOD_SENSOR_GRAPH,
            METHOD_STCG,
            METHOD_SENSOR_STCG,
        ]
        horizons = sorted(set(int(row["horizon_cycles"]) for row in label_rows))
        width = 0.8 / len(methods)
        x = np.arange(len(horizons))
        for method_idx, method in enumerate(methods):
            values = []
            for horizon in horizons:
                row = next(
                    item for item in label_rows if int(item["horizon_cycles"]) == horizon and item["method"] == method
                )
                values.append(float(row["average_precision"]))
            ax.bar(x + (method_idx - (len(methods) - 1) / 2) * width, values, width=width, label=method)
        ax.set_title(label)
        ax.set_xlabel("Warning horizon (cycles)")
        ax.set_ylabel("Average precision")
        ax.set_xticks(x)
        ax.set_xticklabels([str(horizon) for horizon in horizons])
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    cycle_features, cycle_labels, _columns = load_cycle_tables(args)

    rows: list[dict[str, object]] = []
    feature_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label_column in args.label_columns:
        for horizon in args.horizon_cycles:
            rows.extend(
                evaluate_label_horizon(
                    dataset=args.dataset_name,
                    label_column=label_column,
                    horizon=horizon,
                    cycle_features=cycle_features,
                    cycle_labels=cycle_labels,
                    args=args,
                    feature_cache=feature_cache,
                )
            )
            print(f"Completed warning: {label_column}, horizon={horizon}", flush=True)
    if not rows:
        raise ValueError("No transition-warning rows produced; check labels, horizons, and split")

    detail_path = args.output_dir / f"{args.dataset_name}_transition_warning_detail.csv"
    markdown_path = args.output_dir / f"{args.dataset_name}_transition_warning_summary.md"
    best_path = args.output_dir / f"{args.dataset_name}_transition_warning_best.md"
    figure_path = args.figure_dir / f"{args.dataset_name}_transition_warning_average_precision.png"
    fieldnames = [
        "dataset",
        "label_column",
        "horizon_cycles",
        "lookback_cycles",
        "split_policy",
        "purge_horizon_cycles",
        "stcg_selection_policy",
        "method",
        "train_rows",
        "validation_rows",
        "test_rows",
        "train_positive_rate",
        "validation_positive_rate",
        "test_positive_rate",
        "train_positives",
        "validation_positives",
        "test_positives",
        "threshold_metric",
        "threshold",
        "validation_threshold_metric",
        *METRICS,
    ]
    write_csv(detail_path, rows, fieldnames)
    write_markdown(markdown_path, rows)
    write_best_markdown(best_path, rows)
    write_figure(figure_path, rows)

    best = max(rows, key=lambda row: (float(row["average_precision"]), float(row["f1"])))
    print(
        json.dumps(
            {
                "detail_csv": str(detail_path),
                "summary_md": str(markdown_path),
                "best_md": str(best_path),
                "figure": str(figure_path),
                "n_rows": len(rows),
                "best_label": best["label_column"],
                "best_horizon_cycles": best["horizon_cycles"],
                "best_method": best["method"],
                "best_average_precision": best["average_precision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
