"""Variance-reduced score aggregation for STCG-v1.

The ensemble is a post-processing wrapper around :func:`stcg_v1_diagnostics`:
window scoring, the supplied state count, and the acceptance gate are unchanged,
and only the K-means partition is repeated. ``size=1`` returns the primary
diagnostics scores exactly, and a rejected candidate partition falls back to the
one-state scores without resampling.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from .self_supervised import (
    STCGStateConfig,
    STCGStateDiagnostics,
    _aggregate_window_scores,
    _state_aggregate,
)


def ensemble_scores(
    diagnostics: STCGStateDiagnostics,
    config: STCGStateConfig,
    size: int = 20,
    seed: int | None = None,
) -> np.ndarray:
    """Average score tensors over independent K-means partitions.

    The primary ``n_init=10`` solution is one member; the remaining ``size-1``
    members use single restarts with deterministic seeds. Members that collapse
    to one state are skipped. The returned tensor has the same shape as
    ``diagnostics.scores``.
    """

    if size < 1:
        raise ValueError("size must be at least 1")
    if size == 1 or not diagnostics.states_accepted:
        # Respect the one-state fallback: when the gate rejects the candidate
        # partition there is no accepted state structure to resample.
        return diagnostics.scores

    resolved_seed = config.seed if seed is None else seed
    features = diagnostics.features
    centered = features - features.mean(axis=0, keepdims=True)
    scaled = centered / (centered.std(axis=0, keepdims=True) + 1e-8)
    n_states = min(config.n_states, scaled.shape[0])
    members = [diagnostics.scores]
    for member in range(1, size):
        labels = KMeans(
            n_clusters=n_states,
            n_init=1,
            random_state=resolved_seed + 1009 * member,
        ).fit_predict(scaled)
        if np.unique(labels).size <= 1:
            continue
        state_scores = [
            _state_aggregate(diagnostics.window_scores[labels == state], config.state_aggregate)
            for state in sorted(np.unique(labels))
        ]
        members.append(_aggregate_window_scores(state_scores, config.aggregate))
    return np.mean(np.stack(members, axis=0), axis=0)
