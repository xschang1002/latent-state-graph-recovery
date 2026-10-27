"""Study C adapter and read-only derived-output verification fixtures."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_symmetric_localisation as c
import analyze_temporal_localisation as loc


def test_successful_C_null_error_stays_empty_in_the_public_adapter(monkeypatch):
    import run_symmetric_sensitivity as frozen_c
    support = np.zeros((2, 1, 2, 2), int)
    support[0, 0, 0, 1] = 1; support[1, 0, 1, 0] = 1
    regimes = np.repeat([0, 1, 0, 1], 2)
    bounds = np.array([[start, start + 2] for start in range(0, 8, 2)])
    raw = support[[0, 1, 0, 1]].astype(float)
    arrays = {"window_scores": raw, "bounds": bounds, "regimes": regimes, "state_support": support,
              **{"emission_" + method: raw for method in ["forced_hard", "native_hard", "oracle_label", "static_mean"]}}
    case = {"identity": {"seed": 1, "duration": 300, "cell": "fixture", "h": 2, "stride": 2},
            "arrays": arrays, "gate_accepted": True, "fit_status": "ok", "record": {"diagnostic": {"error": None}}}
    monkeypatch.setattr(frozen_c, "iter_cases", lambda root: iter([case]))
    meta, rows, edges = next(c.retained_cases(ROOT))
    assert meta["error"] == "" and meta["status"] == "ok"
    assert len(rows) == len(loc.METHODS)


def test_C_summary_keeps_window_stride_cells_and_full_positive_missingness():
    rows = []
    for h, stride in [(200, 100), (300, 300)]:
        for seed in [1, 2]:
            for method in ["raw_window", "forced_hard"]:
                rows.append({"study": "C", "scorer": "lasso", "duration": 300, "schedule": "fixed_phase",
                             "cell": f"D300_h{h}_s{stride}", "h": h, "stride": stride, "repeat_seed": seed,
                             "method": method, **{metric: np.nan if metric == "boundary_positive_change_ap" and stride == 300 else .25 for metric in loc.METRICS}})
    summary, paired = c.summarise(pd.DataFrame(rows))
    missing = summary[(summary.stride == 300) & (summary.metric == "boundary_positive_change_ap")]
    assert (missing.n_finite == 0).all() and (missing.n_missing == 2).all()
    assert set(summary[["h", "stride"]].itertuples(index=False, name=None)) == {(200, 100), (300, 300)}
    assert paired.loc[(paired.stride == 100), "n_paired"].eq(2).all()


def test_C_wrapper_roundtrip_is_readonly_and_rejects_resealed_number_corruption(tmp_path, monkeypatch):
    locked = {"frozen_C_input_sha256": {}, "source_sha256": {}, "protected_AB_input_sha256": {}, "protected_AB_source_sha256": {}, "definition_protocol_sha256": "fixture"}
    monkeypatch.setattr(c, "protocol", lambda root: locked)
    monkeypatch.setattr(c, "_preserved", lambda root, value: True)
    meta = {"study": "C", "scorer": "lasso", "repeat_seed": 1, "duration": 300,
            "schedule": "fixed_phase", "cell": "D300_h200_s100", "h": 200, "stride": 100, "status": "ok", "error": ""}
    rows = [{"method": method, "status": "ok", **{metric: .2 for metric in loc.METRICS}} for method in ["raw_window", "forced_hard"]]
    monkeypatch.setattr(c, "retained_cases", lambda root: iter([(meta, rows, {"fixture": np.arange(3)})]))
    output, stage = tmp_path / "out", tmp_path / "stage"
    c.freeze(tmp_path, output, stage)
    c.run(tmp_path, output, stage)
    before = {p.relative_to(output).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()}
    assert c.verify(tmp_path, output)["method_rows_recomputed"] == 2
    assert before == {p.relative_to(output).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="overwrite"):
        c.run(tmp_path, output, stage)
    detail = pd.read_csv(output / "detail.csv")
    detail.loc[0, "boundary_slot_f1"] = .9
    detail.to_csv(output / "detail.csv", index=False)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["sha256"]["detail.csv"] = loc.sha(output / "detail.csv")
    (output / "manifest.json").write_bytes(loc.json_bytes(manifest))
    with pytest.raises(AssertionError):
        c.verify(tmp_path, output)
