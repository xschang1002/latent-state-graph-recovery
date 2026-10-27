"""Bounded evidence-closure regressions; no fitting or live checkpoint inputs.

These fixtures exercise the structural checks through the first numeric
comparison. Numerical/metric tests have separate coverage; no bootstrap or
model fit runs here.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location(
    "review_evidence_key_audit", ROOT / "tools/verify_review_response_evidence.py"
)
verifier = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(verifier)

METHODS = ["STCG-v1", "Regime-PCMCI", "cMLP", "DynVAR-Lasso", "VAR-Lasso"]
METRICS = ["edge_auc", "pair_auc", "precision_at_k", "shd", "lag_accuracy"]


class KeyAuditComplete(Exception):
    """Stop before downstream numerical work after all closure checks pass."""


@pytest.fixture
def evidence_tables(tmp_path, monkeypatch):
    tables = tmp_path / "results/tables"
    tables.mkdir(parents=True)
    definitions = [
        ("obs_noise", 0., 0), ("obs_noise", .05, 1), ("obs_noise", .1, 2), ("obs_noise", .2, 3),
        ("timesteps", 300, 4), ("timesteps", 600, 5), ("timesteps", 1000, 0), ("timesteps", 1500, 6),
        ("edge_prob", .1, 7), ("edge_prob", .18, 0), ("edge_prob", .3, 8), ("edge_prob", .45, 9),
        ("switch_period", 150, 10), ("switch_period", 300, 11), ("switch_period", 600, 12),
    ]
    cells = [dict(axis=axis, value=value, setting_id=f"a{setting:011x}")
             for axis, value, setting in definitions]
    (tables / "revision_specialist_protocol.json").write_text(
        json.dumps({"specification": {"display_cells": cells}}), encoding="utf-8"
    )
    settings = [("stress", f"a{index:011x}") for index in range(13)] + [("timing", "a00000000000")]
    detail, coverage, pairs, timing, summary = [], [], [], [], []
    for phase, setting in settings:
        for method in METHODS:
            coverage.append(dict(phase=phase, setting_id=setting, method=method,
                                 planned=20, attempted=20, successful=20, failed=0, common_n=20))
            for seed in range(100, 120):
                detail.append(dict(phase=phase, setting_id=setting, method=method, seed=seed,
                                   status="ok", execution_workers=1, exclusive_timing=phase == "timing"))
        for opponent in METHODS[1:]:
            for metric in METRICS:
                pairs.append(dict(phase=phase, setting_id=setting, opponent=opponent, metric=metric))
            for metric in ["fit_seconds", "fit_cpu_seconds"]:
                timing.append(dict(phase=phase, setting_id=setting, opponent=opponent, metric=metric))
    for cell in cells:
        for method in METHODS:
            summary.append(dict(**cell, phase="stress", method=method))
    for name, rows in [("detail", detail), ("coverage", coverage), ("paired", pairs),
                       ("timing", timing), ("summary", summary)]:
        pd.DataFrame(rows).to_csv(tables / f"revision_specialist_{name}.csv", index=False)
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(verifier, "TABLES", tables)
    monkeypatch.setattr(verifier, "CHECKS", [])
    monkeypatch.setattr(verifier, "INPUTS", {})

    def stop_before_numeric_comparison(*args, **kwargs):
        raise KeyAuditComplete

    monkeypatch.setattr(verifier, "equal", stop_before_numeric_comparison)
    return tables


def test_complete_frozen_key_sets_reach_numeric_validation(evidence_tables):
    with pytest.raises(KeyAuditComplete):
        verifier.verify_specialists()


@pytest.mark.parametrize("name,change,message", [
    ("coverage", "drop", "exact coverage key set"),
    ("paired", "drop", "exact paired-test key set"),
    ("timing", "empty", "exact timing-comparison key set"),
    ("summary", "empty", "exact display summary key set"),
    ("paired", "duplicate", "exact paired-test key set"),
    ("summary", "duplicate", "exact display summary key set"),
    ("detail", "duplicate", "all 1400 unique frozen fits"),
    ("detail", "replace_setting", "exact frozen job identities"),
])
def test_missing_duplicate_and_substituted_records_are_rejected(evidence_tables, name, change, message):
    path = evidence_tables / f"revision_specialist_{name}.csv"
    frame = pd.read_csv(path, dtype={"setting_id": str})
    if change == "drop":
        frame = frame.iloc[:-1]
    elif change == "empty":
        frame = frame.iloc[:0]
    elif change == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    else:
        frame.loc[0, "setting_id"] = "ffffffffffff"
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError, match=message):
        verifier.verify_specialists()
