"""Label-blind selection of graph-state candidates, including null rejection."""
from __future__ import annotations

import math
from typing import Sequence

SELECTION_POLICIES = ("reject_nonpositive", "legacy")


def select_state_candidate_index(
    candidates: Sequence[tuple[float, float, int]],
    policy: str = "reject_nonpositive",
) -> int:
    """Choose using only (score, coherence, K), with an explicit K=1 fallback.

    Positive-score comparisons preserve the historical lexicographic ordering.
    ``legacy`` also preserves the old all-zero tie behavior for reproducibility.
    The caller must supply the actually fitted K=1 candidate when fallback is
    needed; this function never fabricates its features or an audit result.
    """
    if policy not in SELECTION_POLICIES:
        raise ValueError(f"Unknown selection policy: {policy}")
    if not candidates:
        raise ValueError("No state-count candidates were supplied")
    for score, coherence, count in candidates:
        if not math.isfinite(score) or not math.isfinite(coherence):
            raise ValueError("State-count scores and coherence must be finite")
        if count < 1 or int(count) != count:
            raise ValueError("State-count candidates must be positive integers")
    if len({item[2] for item in candidates}) != len(candidates):
        raise ValueError("State-count candidates must be unique")
    if policy == "reject_nonpositive" and not any(score > 0 for score, _, _ in candidates):
        for index, (_, _, count) in enumerate(candidates):
            if count == 1:
                return index
        raise ValueError("All state-count scores are nonpositive; include a fitted K=1 candidate for rejection, or explicitly request legacy behavior")
    return max(range(len(candidates)), key=lambda index: (candidates[index][0], candidates[index][1], -candidates[index][2]))
