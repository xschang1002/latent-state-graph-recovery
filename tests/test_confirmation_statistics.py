"""Ensure pilot and unpaired observations cannot enter confirmation summaries."""
import importlib.util
from pathlib import Path

import pytest
from scipy import stats

spec = importlib.util.spec_from_file_location(
    "confirmation", Path(__file__).resolve().parents[1] / "tools" / "build_confirmation_statistics.py"
)
confirmation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(confirmation)


def test_postpilot_intersection_excludes_pilot_and_missing_opponent_seed():
    rows = [
        {"generator": "var", "method": m, "seed": str(seed)}
        for m, seeds in [("STCG-v1", [0, 4, 5, 6, 7, 20]), ("cMLP", [0, 4, 5, 7, 20])]
        for seed in seeds
    ]
    kept = confirmation.common_postpilot_rows(rows)
    assert len(kept) == 4
    assert {r["seed"] for r in kept if r["method"] == "STCG-v1"} == {"5", "7"}
    assert {r["seed"] for r in kept if r["method"] == "cMLP"} == {"5", "7"}


def test_two_sided_difference_test_is_symmetric():
    values = [-2, -1, -3, -2, -1, 0.5]
    tp, wp = confirmation.two_sided_p(values)
    assert tp == pytest.approx(stats.ttest_1samp(values, 0, alternative="two-sided").pvalue)
    assert confirmation.two_sided_p([-v for v in values]) == pytest.approx((tp, wp))
    assert confirmation.two_sided_p([0, 0, 0]) == (1.0, 1.0)
