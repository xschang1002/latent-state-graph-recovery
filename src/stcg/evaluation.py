"""Three-layer estimands for lagged graph recovery.

The union graph, the regime-specific graphs, and the temporal state sequence
are different estimands.  A method can recover the union while failing on the
regime-specific graphs, and it can recover both while assigning states to the
wrong time points; a single edge AUC cannot distinguish these cases.  Every
function in this module therefore names the estimand it scores and never
converts one layer into another.

Conventions
-----------
``scores`` and ``support`` have shape ``(n_lags, n_nodes, n_nodes)`` and are
indexed ``[lag - 1, source, target]`` for the directed effect
``source(t - lag) -> target(t)``.  Self-lag diagonal entries are never scored.

``state_scores`` has shape ``(n_states, n_lags, n_nodes, n_nodes)`` and rows
are ordered by ``np.unique(window_assignments)``.

``bounds`` is a sequence of ``(start, end)`` half-open window boundaries over
the observed series, and ``regimes`` is the per-time-point ground-truth state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score

__all__ = [
    "GraphTruth",
    "GraphPrediction",
    "nonself_mask",
    "union_metrics",
    "match_states",
    "per_regime_metrics",
    "window_regime_labels",
    "window_purity",
    "temporal_state_metrics",
    "windowed_edge_metrics",
    "temporal_graph_metrics",
    "static_temporal_graph_metrics",
    "temporal_delta_auc",
    "evaluate_prediction",
]


@dataclass(frozen=True)
class GraphTruth:
    """Ground truth for the three estimands.

    ``support`` is the union support.  ``state_support`` is optional and is
    required only for regime-specific scoring.  ``regimes`` is optional and is
    required only for temporal scoring.
    """

    support: np.ndarray
    state_support: np.ndarray | None = None
    regimes: np.ndarray | None = None


@dataclass(frozen=True)
class GraphPrediction:
    """A method's output, kept at the layer where it was produced."""

    union_scores: np.ndarray
    window_scores: np.ndarray | None = None
    window_bounds: tuple[tuple[int, int], ...] | None = None
    state_scores: np.ndarray | None = None
    window_assignments: np.ndarray | None = None


def nonself_mask(shape: tuple[int, ...]) -> np.ndarray:
    """Boolean mask that drops the self-lag diagonal from a score tensor."""

    if len(shape) != 3:
        raise ValueError(f"Expected a 3-d (lags, nodes, nodes) shape, got {shape}")
    _, n_nodes, _ = shape
    mask = np.ones(shape, dtype=bool)
    diagonal = np.arange(n_nodes)
    mask[:, diagonal, diagonal] = False
    return mask


def _flatten(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if values.shape != mask.shape:
        raise ValueError(f"Shape mismatch: values {values.shape}, mask {mask.shape}")
    return values[mask]


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    if y_true.all() or not y_true.any():
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _safe_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    if y_true.all() or not y_true.any():
        return float("nan")
    return float(average_precision_score(y_true, scores))


def _precision_at_k(scores: np.ndarray, y_true: np.ndarray, k: int) -> float:
    if k <= 0 or scores.size == 0:
        return float("nan")
    k = min(k, scores.size)
    top = np.argpartition(-scores, kth=k - 1)[:k]
    return float(y_true[top].mean())


def _shd_at_k(scores: np.ndarray, y_true: np.ndarray, k: int) -> int:
    k = min(max(k, 0), scores.size)
    selected = np.zeros(scores.size, dtype=bool)
    if k:
        selected[np.argpartition(-scores, kth=k - 1)[:k]] = True
    return int(np.logical_xor(selected, y_true).sum())


def union_metrics(
    scores: np.ndarray,
    support: np.ndarray,
    k: int | None = None,
    prefix: str = "",
) -> dict[str, float]:
    """Score the union graph: was an edge active in at least one state?

    ``k`` defaults to the true number of union edges.  Using the true edge
    count is an offline evaluation choice; it must not be read as a
    label-free operating point.
    """

    if scores.shape != support.shape:
        raise ValueError(f"Shape mismatch: scores {scores.shape}, support {support.shape}")
    mask = nonself_mask(scores.shape)
    flat_scores = _flatten(scores, mask)
    y_true = _flatten(support, mask).astype(bool)
    true_k = int(y_true.sum())
    k = true_k if k is None else int(k)
    prevalence = float(y_true.mean()) if y_true.size else float("nan")
    auprc = _safe_auprc(y_true, flat_scores)
    metrics = {
        "auroc": _safe_auc(y_true, flat_scores),
        "auprc": auprc,
        "prevalence": prevalence,
        "auprc_lift": float(auprc / prevalence) if prevalence and np.isfinite(auprc) else float("nan"),
        "auprc_normalized": float((auprc - prevalence) / (1.0 - prevalence))
        if np.isfinite(auprc) and 0.0 < prevalence < 1.0
        else float("nan"),
        "precision_at_k": _precision_at_k(flat_scores, y_true, k),
        "shd": float(_shd_at_k(flat_scores, y_true, k)),
        "true_edges": float(true_k),
    }
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def match_states(
    predicted: np.ndarray,
    truth: np.ndarray,
    n_truth_states: int | None = None,
) -> dict[str, object]:
    """Match predicted state labels to ground-truth labels by overlap.

    Uses a rectangular Hungarian assignment, so a method may return more or
    fewer states than the truth.  Unmatched predicted states are kept (they are
    false state claims) and missed true states are reported separately.
    """

    predicted = np.asarray(predicted).astype(int)
    truth = np.asarray(truth).astype(int)
    if predicted.shape != truth.shape:
        raise ValueError(f"Shape mismatch: predicted {predicted.shape}, truth {truth.shape}")

    pred_states = np.unique(predicted)
    true_states = np.unique(truth) if n_truth_states is None else np.arange(n_truth_states)
    overlap = np.zeros((pred_states.size, true_states.size), dtype=float)
    for i, pred_state in enumerate(pred_states):
        mask = predicted == pred_state
        for j, true_state in enumerate(true_states):
            overlap[i, j] = float(np.sum(mask & (truth == true_state)))

    mapping: dict[int, int] = {}
    inverse: dict[int, int] = {}
    if overlap.size:
        rows, columns = linear_sum_assignment(-overlap)
        for row, column in zip(rows, columns):
            if overlap[row, column] <= 0:
                continue
            mapping[int(pred_states[row])] = int(true_states[column])
            inverse[int(true_states[column])] = int(pred_states[row])

    unmatched_predicted = [int(s) for s in pred_states if int(s) not in mapping]
    missed_truth = [int(s) for s in true_states if int(s) not in inverse]
    return {
        "mapping": mapping,
        "inverse": inverse,
        "unmatched_predicted": unmatched_predicted,
        "missed_truth": missed_truth,
        "overlap": overlap,
        "pred_states": pred_states,
        "true_states": true_states,
    }


def _state_level_scores(
    scores: np.ndarray, support: np.ndarray, prefix: str
) -> dict[str, float]:
    mask = nonself_mask(scores.shape)
    flat_scores = _flatten(scores, mask)
    y_true = _flatten(support, mask).astype(bool)
    true_k = int(y_true.sum())
    precision = _precision_at_k(flat_scores, y_true, true_k) if true_k else float("nan")
    recall = float("nan")
    if true_k:
        k = min(true_k, flat_scores.size)
        top = np.argpartition(-flat_scores, kth=k - 1)[:k]
        recall = float(y_true[top].sum() / true_k)
    f1 = float("nan")
    if np.isfinite(precision) and np.isfinite(recall):
        f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        f"{prefix}auprc": _safe_auprc(y_true, flat_scores),
        f"{prefix}auroc": _safe_auc(y_true, flat_scores),
        f"{prefix}f1": f1,
        f"{prefix}shd": float(_shd_at_k(flat_scores, y_true, true_k)) if true_k else float("nan"),
    }


def per_regime_metrics(
    state_scores: np.ndarray,
    state_support: np.ndarray,
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
) -> dict[str, float]:
    """Score regime-specific graphs after matching state identities.

    Macro averages run over matched ground-truth states with a non-empty
    support.  ``state_edge_f1`` is a micro average over those states, and
    ``unmatched_predicted_fraction`` counts predicted states that claim a
    regime the truth does not contain.
    """

    state_scores = np.asarray(state_scores)
    state_support = np.asarray(state_support)
    if state_scores.ndim != 4 or state_support.ndim != 4:
        raise ValueError(
            "state_scores and state_support must both be (states, lags, nodes, nodes)"
        )
    if state_scores.shape[1:] != state_support.shape[1:]:
        raise ValueError(
            f"Trailing shape mismatch: state_scores {state_scores.shape}, state_support {state_support.shape}"
        )
    predicted_labels = np.asarray(predicted_labels).astype(int)
    true_labels = np.asarray(true_labels).astype(int)
    if predicted_labels.shape != true_labels.shape:
        raise ValueError(
            "per_regime_metrics compares window-level labels: "
            f"{predicted_labels.shape} vs {true_labels.shape}"
        )
    match = match_states(predicted_labels, true_labels, n_truth_states=state_support.shape[0])
    score_rows = {int(label): row for row, label in enumerate(match["pred_states"])}
    if len(score_rows) > state_scores.shape[0]:
        raise ValueError("Need one score tensor per predicted state, ordered by unique label")

    auprc_values: list[float] = []
    auroc_values: list[float] = []
    f1_values: list[float] = []
    shd_values: list[float] = []
    micro_true = 0
    micro_tp = 0
    micro_pred = 0
    prevalences: list[float] = []
    n_possible = float(nonself_mask(state_support.shape[1:]).sum())

    for true_state, pred_state in match["inverse"].items():
        support = state_support[true_state]
        if int(support.sum()) == 0:
            continue
        scores = state_scores[score_rows[pred_state]]
        values = _state_level_scores(scores, support, prefix="")
        prevalences.append(float(support.sum()) / n_possible)
        auprc_values.append(values["auprc"])
        auroc_values.append(values["auroc"])
        f1_values.append(values["f1"])
        shd_values.append(values["shd"])

        mask = nonself_mask(scores.shape)
        flat_scores = _flatten(scores, mask)
        y_true = _flatten(support, mask).astype(bool)
        true_k = int(y_true.sum())
        top = np.argpartition(-flat_scores, kth=min(true_k, flat_scores.size) - 1)[
            : min(true_k, flat_scores.size)
        ]
        micro_true += true_k
        micro_tp += int(y_true[top].sum())
        micro_pred += len(top)

    n_matched = len(match["inverse"])
    n_pred = len(match["pred_states"])
    micro_precision = float(micro_tp / micro_pred) if micro_pred else float("nan")
    micro_recall = float(micro_tp / micro_true) if micro_true else float("nan")
    micro_f1 = float("nan")
    if np.isfinite(micro_precision) and np.isfinite(micro_recall):
        denominator = micro_precision + micro_recall
        micro_f1 = float(2 * micro_precision * micro_recall / denominator) if denominator else 0.0

    def _nanmean(values: list[float]) -> float:
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        return float(array.mean()) if array.size else float("nan")

    regime_auprc = _nanmean(auprc_values)
    regime_prevalence = _nanmean(prevalences)
    return {
        "regime_auprc_macro": regime_auprc,
        "regime_prevalence": regime_prevalence,
        "regime_auprc_lift": float(regime_auprc / regime_prevalence)
        if regime_prevalence and np.isfinite(regime_auprc)
        else float("nan"),
        "regime_auprc_normalized": float(
            (regime_auprc - regime_prevalence) / (1.0 - regime_prevalence)
        )
        if np.isfinite(regime_auprc) and 0.0 < regime_prevalence < 1.0
        else float("nan"),
        "regime_auroc_macro": _nanmean(auroc_values),
        "regime_f1_macro": _nanmean(f1_values),
        "regime_shd_mean": _nanmean(shd_values),
        "state_edge_f1_micro": micro_f1,
        "matched_states": float(n_matched),
        "unmatched_predicted_fraction": float(
            len(match["unmatched_predicted"]) / n_pred if n_pred else float("nan")
        ),
        "missed_truth_states": float(len(match["missed_truth"])),
    }


def window_regime_labels(
    bounds: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    regimes: np.ndarray,
) -> np.ndarray:
    """Ground-truth regime at each window centre, aligned with assignments."""

    return np.array([int(regimes[(start + end - 1) // 2]) for start, end in bounds], dtype=int)


def window_purity(
    bounds: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    regimes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-window purity and the majority regime of each window.

    Purity is the fraction of time points in the window that carry the window's
    most common regime.  A value below one means the window mixes two or more
    regimes, so its score no longer belongs to a single state graph.
    """

    purity = np.zeros(len(bounds), dtype=float)
    majority = np.zeros(len(bounds), dtype=int)
    for index, (start, end) in enumerate(bounds):
        segment = regimes[start:end]
        if segment.size == 0:
            purity[index] = float("nan")
            majority[index] = -1
            continue
        labels, counts = np.unique(segment, return_counts=True)
        best = int(np.argmax(counts))
        purity[index] = float(counts[best] / segment.size)
        majority[index] = int(labels[best])
    return purity, majority


def _change_points(sequence: np.ndarray) -> list[int]:
    if sequence.size < 2:
        return []
    return [int(i) for i in np.flatnonzero(sequence[1:] != sequence[:-1]) + 1]


def _boundary_f1(
    predicted: np.ndarray, truth: np.ndarray, tolerance: int
) -> tuple[float, float, float]:
    predicted_points = _change_points(predicted)
    true_points = _change_points(truth)
    if not predicted_points and not true_points:
        return 1.0, 1.0, 1.0
    if not predicted_points or not true_points:
        return 0.0, 0.0, 0.0
    used = set()
    hits = 0
    for point in predicted_points:
        candidates = [j for j, target in enumerate(true_points) if abs(point - target) <= tolerance and j not in used]
        if candidates:
            best = min(candidates, key=lambda j: abs(point - true_points[j]))
            used.add(best)
            hits += 1
    precision = hits / len(predicted_points)
    recall = len(used) / len(true_points)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def temporal_state_metrics(
    window_assignments: np.ndarray,
    bounds: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    regimes: np.ndarray,
    tolerance: int = 1,
    mode: str = "center",
) -> dict[str, float]:
    """Score the temporal state sequence against the true regime sequence.

    ``mode="center"`` scores one label per window at the window centre, which
    is the convention used for all reported ARI values.  ``mode="coverage"``
    expands each window label over its covered time points and majority-votes,
    which is reported only as a sensitivity check because it hides mixed
    windows.
    """

    window_assignments = np.asarray(window_assignments).astype(int)
    if len(window_assignments) != len(bounds):
        raise ValueError(
            f"Expected one assignment per window: {len(window_assignments)} vs {len(bounds)} bounds"
        )

    if mode == "center":
        indices = np.array([(start + end - 1) // 2 for start, end in bounds], dtype=int)
        true_window = regimes[indices]
    elif mode == "coverage":
        state_ids, rows = np.unique(window_assignments, return_inverse=True)
        votes = np.zeros((regimes.size, len(state_ids)), dtype=int)
        for (start, end), row in zip(bounds, rows):
            votes[start:end, row] += 1
        covered = votes.sum(axis=1) > 0
        true_window = regimes[covered]
        window_assignments = state_ids[np.argmax(votes[covered], axis=1)]
        indices = np.flatnonzero(covered)
    else:
        raise ValueError(f"Unknown temporal mode: {mode}")

    ari = float(adjusted_rand_score(true_window, window_assignments))
    match = match_states(window_assignments, true_window)
    mapping = match["mapping"]

    # Macro F1 over the union of true and predicted states; unmatched entries
    # count as errors rather than being silently dropped.
    labels = match["true_states"]
    inverse = match["inverse"]
    f1_scores: list[float] = []
    for label in labels:
        true_positive = 0
        for pred_label, true_label in mapping.items():
            if true_label == label:
                true_positive = int(np.sum((window_assignments == pred_label) & (true_window == label)))
                break
        pred_label = inverse.get(label)
        pred_count = int(np.sum(window_assignments == pred_label)) if pred_label is not None else 0
        true_count = int(np.sum(true_window == label))
        precision = true_positive / pred_count if pred_count else 0.0
        recall = true_positive / true_count if true_count else 0.0
        f1_scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    f1_scores.extend([0.0] * len(match["unmatched_predicted"]))

    precision, recall, boundary_f1 = _boundary_f1(window_assignments, true_window, tolerance)
    purity, _ = window_purity(bounds, regimes)
    return {
        "state_ari": ari,
        "state_macro_f1": float(np.mean(f1_scores)) if f1_scores else float("nan"),
        "k_hat_minus_k": float(np.unique(window_assignments).size - np.unique(true_window).size),
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": boundary_f1,
        "window_purity_mean": float(np.nanmean(purity)) if purity.size else float("nan"),
        "mixed_window_fraction": float(np.mean(purity < 1.0)) if purity.size else float("nan"),
        "n_windows": float(len(bounds)),
        "n_mixed_windows": float(np.sum(purity < 1.0)),
    }


def windowed_edge_metrics(
    window_scores: np.ndarray,
    bounds: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    regimes: np.ndarray,
    state_support: np.ndarray,
) -> dict[str, float]:
    """Score each window's edges against the graph of its majority regime.

    This is the temporal edge estimand: it asks whether the score of a window
    matches the graph active in that window, without aggregating over states.
    """

    window_scores = np.asarray(window_scores)
    if window_scores.shape[0] != len(bounds):
        raise ValueError("window_scores and bounds disagree on the number of windows")
    purity, majority = window_purity(bounds, regimes)
    aucs: list[float] = []
    for index in range(window_scores.shape[0]):
        state = majority[index]
        if state < 0 or state >= state_support.shape[0]:
            continue
        support = state_support[state]
        if int(support.sum()) == 0:
            continue
        mask = nonself_mask(window_scores[index].shape)
        value = _safe_auc(
            _flatten(support, mask).astype(bool), _flatten(window_scores[index], mask)
        )
        if np.isfinite(value):
            aucs.append(value)
    return {
        "windowed_edge_auroc": float(np.mean(aucs)) if aucs else float("nan"),
        "windowed_edge_n": float(len(aucs)),
        "window_purity_mean": float(np.nanmean(purity)) if purity.size else float("nan"),
    }


def temporal_graph_metrics(
    state_scores: np.ndarray,
    window_assignments: np.ndarray,
    bounds: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    regimes: np.ndarray,
    state_support: np.ndarray,
) -> dict[str, float]:
    """Score the time-varying graph implied by the state assignment.

    The predicted graph at window ``w`` is the state score of the label assigned
    to that window, and the truth is the support of the regime at the window
    centre. Pooling over windows answers a different question from either layer
    alone: did the method use the right graph at the right time? A method can
    have good state graphs and poor state alignment, or vice versa, and this
    metric separates those cases.
    """

    assignments = np.asarray(window_assignments).astype(int)
    if assignments.size != len(bounds):
        raise ValueError("assignments and bounds disagree on the number of windows")
    labels = np.unique(assignments)
    index = {int(label): position for position, label in enumerate(labels)}

    scores: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    for window, (start, end) in enumerate(bounds):
        centre = (start + end - 1) // 2
        true_state = int(regimes[centre])
        if true_state >= state_support.shape[0]:
            continue
        row = index.get(int(assignments[window]))
        if row is None or row >= state_scores.shape[0]:
            continue
        mask = nonself_mask(state_scores[row].shape)
        scores.append(state_scores[row][mask])
        truths.append(state_support[true_state][mask].astype(bool))
    if not scores:
        return {
            "temporal_graph_auroc": float("nan"),
            "temporal_graph_auprc": float("nan"),
            "temporal_graph_prevalence": float("nan"),
            "temporal_graph_auprc_normalized": float("nan"),
            "temporal_graph_windows": 0.0,
        }

    y_score = np.concatenate(scores)
    y_true = np.concatenate(truths)
    auroc = _safe_auc(y_true, y_score)
    auprc = _safe_auprc(y_true, y_score)
    prevalence = float(y_true.mean())
    return {
        "temporal_graph_auroc": auroc,
        "temporal_graph_auprc": auprc,
        "temporal_graph_prevalence": prevalence,
        "temporal_graph_auprc_normalized": float(
            (auprc - prevalence) / (1.0 - prevalence)
        )
        if np.isfinite(auprc) and 0.0 < prevalence < 1.0
        else float("nan"),
        "temporal_graph_windows": float(len(scores)),
    }


def static_temporal_graph_metrics(
    union_scores: np.ndarray,
    bounds: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    regimes: np.ndarray,
    state_support: np.ndarray,
) -> dict[str, float]:
    """Score a static graph as a constant time-varying prediction.

    A method with no state assignment still predicts a graph at every time
    point: the same edge scores ``s_e`` for all windows. This is exactly the
    baseline a dynamic method must beat to show that state information is used,
    so static methods get the same joint temporal target as stateful ones.
    """

    n_windows = len(bounds)
    return temporal_graph_metrics(
        union_scores[None, ...],
        np.zeros(n_windows, dtype=int),
        bounds,
        regimes,
        state_support,
    )


def temporal_delta_auc(
    state_scores: np.ndarray,
    window_assignments: np.ndarray,
    bounds: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    regimes: np.ndarray,
    state_support: np.ndarray,
) -> dict[str, float]:
    """Per-edge temporal AUC over edges that genuinely change across states.

    Let ``E_delta = {e : exists k,k' with A[k,e] != A[k',e]}``. For each such
    edge, the score sequence ``(S_{w,e})_w`` is compared with the active label
    ``(A[z_w,e])_w``. A method that always emits the same score has AUROC 0.5 on
    a changing edge, so this metric isolates *when* the edge is active from
    *whether* it exists at all, which the pooled joint metric cannot.
    """

    assignments = np.asarray(window_assignments).astype(int)
    if assignments.size != len(bounds):
        raise ValueError("assignments and bounds disagree on the number of windows")
    if state_scores.shape[1:] != state_support.shape[1:]:
        raise ValueError("state_scores and state_support trailing shapes disagree")

    n_states = state_support.shape[0]
    nonself = nonself_mask(state_support.shape[1:])
    # Edges that differ between at least one pair of states.
    delta = np.zeros(state_support.shape[1:], dtype=bool)
    for first in range(n_states):
        for second in range(first + 1, n_states):
            delta |= state_support[first] != state_support[second]
    delta &= nonself

    labels = np.unique(assignments)
    index = {int(label): position for position, label in enumerate(labels)}

    aucs: list[float] = []
    for lag, source, target in zip(*np.nonzero(delta)):
        true_seq: list[int] = []
        score_seq: list[float] = []
        for window, (start, end) in enumerate(bounds):
            centre = (start + end - 1) // 2
            true_state = int(regimes[centre])
            if true_state >= n_states:
                continue
            row = index.get(int(assignments[window]))
            if row is None or row >= state_scores.shape[0]:
                continue
            true_seq.append(int(state_support[true_state, lag, source, target]))
            score_seq.append(float(state_scores[row, lag, source, target]))
        if not true_seq:
            continue
        # Only score edges whose active label actually flips in this window set,
        # otherwise AUROC is undefined and would reward a constant predictor.
        if len(set(true_seq)) < 2:
            continue
        value = _safe_auc(np.asarray(true_seq), np.asarray(score_seq))
        if np.isfinite(value):
            aucs.append(value)

    return {
        "temporal_delta_auroc": float(np.mean(aucs)) if aucs else float("nan"),
        "temporal_delta_edges": float(len(aucs)),
        "temporal_delta_edges_total": float(int(delta.sum())),
    }


def evaluate_prediction(
    prediction: GraphPrediction,
    truth: GraphTruth,
) -> dict[str, float]:
    """Score every layer that both the prediction and the truth provide."""

    metrics = union_metrics(prediction.union_scores, truth.support, prefix="union_")
    if (
        prediction.state_scores is not None
        and prediction.window_assignments is not None
        and truth.state_support is not None
        and truth.regimes is not None
    ):
        metrics.update(
            per_regime_metrics(
                prediction.state_scores,
                truth.state_support,
                prediction.window_assignments,
                window_regime_labels(prediction.window_bounds, truth.regimes),
            )
        )
    if (
        prediction.window_assignments is not None
        and prediction.window_bounds is not None
        and truth.regimes is not None
    ):
        metrics.update(
            temporal_state_metrics(
                prediction.window_assignments, prediction.window_bounds, truth.regimes
            )
        )
    if (
        prediction.state_scores is not None
        and prediction.window_assignments is not None
        and prediction.window_bounds is not None
        and truth.regimes is not None
        and truth.state_support is not None
    ):
        metrics.update(
            temporal_graph_metrics(
                prediction.state_scores,
                prediction.window_assignments,
                prediction.window_bounds,
                truth.regimes,
                truth.state_support,
            )
        )
        metrics.update(
            temporal_delta_auc(
                prediction.state_scores,
                prediction.window_assignments,
                prediction.window_bounds,
                truth.regimes,
                truth.state_support,
            )
        )
    if (
        prediction.window_scores is not None
        and prediction.window_bounds is not None
        and truth.regimes is not None
        and truth.state_support is not None
    ):
        metrics.update(
            windowed_edge_metrics(
                prediction.window_scores,
                prediction.window_bounds,
                truth.regimes,
                truth.state_support,
            )
        )
    return metrics
