"""Adapters that turn method outputs into the three estimand layers.

Keeping the adapters in one place means every experiment scores the same
objects the same way.  Nothing here re-fits a model; it only aggregates window
scores that a method already produced.
"""

from __future__ import annotations

import numpy as np

from .evaluation import (
    GraphPrediction,
    GraphTruth,
    evaluate_prediction,
    window_regime_labels,
)
from .self_supervised import STCGStateDiagnostics

__all__ = [
    "prediction_from_diagnostics",
    "evaluate_diagnostics",
    "aggregate_state_scores",
    "union_scores_from_states",
    "oracle_label_prediction",
    "truth_from_dataset",
]


def truth_from_dataset(dataset) -> GraphTruth:
    """Build a GraphTruth from a generated dataset (works for both generators)."""

    return GraphTruth(
        support=np.asarray(dataset.union_support),
        state_support=np.asarray(dataset.state_support),
        regimes=np.asarray(dataset.regimes),
    )


def prediction_from_diagnostics(diagnostics: STCGStateDiagnostics) -> GraphPrediction:
    return GraphPrediction(
        union_scores=np.asarray(diagnostics.scores),
        window_scores=np.asarray(diagnostics.window_scores),
        window_bounds=tuple(tuple(bound) for bound in diagnostics.bounds),
        state_scores=np.asarray(diagnostics.state_scores),
        window_assignments=np.asarray(diagnostics.labels),
    )


def evaluate_diagnostics(
    diagnostics: STCGStateDiagnostics, truth: GraphTruth
) -> dict[str, float]:
    return evaluate_prediction(prediction_from_diagnostics(diagnostics), truth)


def aggregate_state_scores(
    window_scores: np.ndarray,
    labels: np.ndarray,
    mode: str = "mean",
) -> np.ndarray:
    """Aggregate window scores into one score tensor per predicted state."""

    window_scores = np.asarray(window_scores)
    labels = np.asarray(labels).astype(int)
    if window_scores.shape[0] != labels.size:
        raise ValueError(
            f"Expected one label per window: {window_scores.shape[0]} vs {labels.size}"
        )
    states = np.unique(labels)
    rows = []
    for state in states:
        members = window_scores[labels == state]
        if mode == "mean":
            rows.append(members.mean(axis=0))
        elif mode == "median":
            rows.append(np.median(members, axis=0))
        elif mode == "max":
            rows.append(members.max(axis=0))
        else:
            raise ValueError(f"Unknown aggregate mode: {mode}")
    return np.stack(rows, axis=0)


def union_scores_from_states(state_scores: np.ndarray, mode: str = "max") -> np.ndarray:
    state_scores = np.asarray(state_scores)
    if mode == "max":
        return state_scores.max(axis=0)
    if mode == "mean":
        return state_scores.mean(axis=0)
    raise ValueError(f"Unknown union mode: {mode}")


def oracle_label_prediction(
    window_scores: np.ndarray,
    bounds: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    regimes: np.ndarray,
    mode: str = "mean",
    union_mode: str = "max",
) -> GraphPrediction:
    """Aggregate with the true window labels.

    This is an oracle-label upper bound: it uses ground-truth states for
    aggregation but never re-fits the model on pure segments.  The stronger
    ``oracle-regime refit`` bound is computed separately and is not the same
    quantity.
    """

    labels = window_regime_labels(bounds, regimes)
    state_scores = aggregate_state_scores(window_scores, labels, mode=mode)
    return GraphPrediction(
        union_scores=union_scores_from_states(state_scores, mode=union_mode),
        window_scores=np.asarray(window_scores),
        window_bounds=tuple(tuple(bound) for bound in bounds),
        state_scores=state_scores,
        window_assignments=labels,
    )
