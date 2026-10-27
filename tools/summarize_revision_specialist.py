"""Analyze the frozen specialist checkpoints without changing their fit protocol.

Analysis-only amendment, 2026-09-13: Windows process CPU clocks can return zero
for very short fits. Preserve those observations and mark their geometric CPU
ratio unavailable rather than imputing a positive duration, dropping the pair,
or failing the whole summary. Positive wall/CPU ratios and all recovery analyses
use the unchanged frozen runner. No estimator, seed, grid, or fit record changes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location(
    "frozen_specialist_stress_analysis", ROOT / "scripts/run_revision_specialist_stress.py")
runner = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(runner)
ORIGINAL_COST_RATIO = runner.paired_cost_ratio


def resolution_aware_cost_ratio(stcg: list[float], opponent: list[float]) -> dict:
    if len(stcg) != len(opponent) or len(stcg) < 2:
        raise ValueError("At least two paired timings required")
    values = [*stcg, *opponent]
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("Timings must be finite and nonnegative")
    zero_stcg = sum(v == 0 for v in stcg)
    zero_opponent = sum(v == 0 for v in opponent)
    if zero_stcg or zero_opponent:
        return {
            "ratio_available": False,
            "ratio_unavailable_reason": "At least one measured duration is below the clock resolution; no imputation or pair exclusion",
            "zero_clock_stcg_count": zero_stcg, "zero_clock_opponent_count": zero_opponent,
            "geometric_mean_opponent_over_stcg": None, "ratio_ci_low": None, "ratio_ci_high": None,
        }
    return {"ratio_available": True, "ratio_unavailable_reason": "",
            "zero_clock_stcg_count": 0, "zero_clock_opponent_count": 0,
            **ORIGINAL_COST_RATIO(stcg, opponent)}


def summarize(output_dir: Path) -> dict:
    frozen = runner.load_protocol(output_dir)
    runner.paired_cost_ratio = resolution_aware_cost_ratio
    try:
        summary = runner.aggregate_results(output_dir)
    finally:
        runner.paired_cost_ratio = ORIGINAL_COST_RATIO
    provenance = {
        "generated_at_utc": runner.utc_now(), "frozen_protocol_hash": frozen["protocol_hash"],
        "frozen_fit_source_hashes": frozen["specification"]["source_hashes"],
        "analysis_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "analysis_amendment": "Preserve zero-resolution CPU measurements and mark the corresponding geometric cost ratio unavailable; no fit or metric changes.",
        "inputs": "revision_specialist_protocol.json and revision_specialist_parts/*.json",
        "summary": summary,
    }
    amendment_path = output_dir / "revision_specialist_storage_amendment.json"
    if amendment_path.exists():
        provenance["storage_amendment_sha256"] = hashlib.sha256(amendment_path.read_bytes()).hexdigest()
    runner.atomic_json(output_dir / "revision_specialist_analysis_provenance.json", provenance)
    report = output_dir / "revision_specialist_analysis.md"
    with report.open("a", encoding="utf-8") as file:
        file.write("\n\n## CPU-clock resolution amendment\n\n"
                   "Windows process CPU clocks can round very short fits to zero. "
                   "These raw observations are retained. A geometric timing ratio that includes "
                   "a zero duration is explicitly unavailable, with zero-clock counts in the timing CSV. "
                   "No positive duration is imputed and no zero pair is dropped. "
                   "Positive wall-time ratios and all recovery statistics are unchanged. "
                   "Use `tools/summarize_revision_specialist.py` for this analysis-only amendment; "
                   "estimation functions and the original protocol remain unchanged; "
                   "any recording-only runner revision is separately logged.\n")
        if amendment_path.exists():
            file.write("\n## Undefined-metric recording amendment\n\n"
                       "At high edge density some truths contain no negative directed pairs, "
                       "so Pair AUC is mathematically undefined. Successful fits retain status=ok "
                       "and null for that metric. Each metric uses its own finite common seeds across "
                       "all five methods, with defined counts and excluded undefined-metric seeds recorded. "
                       "An undefined Pair AUC never removes an otherwise valid Edge-AUC observation. "
                       "Original serialization failures and interrupted jobs are retained in the "
                       "storage amendment/history; estimator, generator, grid and statistical-test "
                       "functions were checked by AST hash to remain unchanged.\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/tables")
    args = parser.parse_args()
    print(json.dumps(summarize(args.output_dir), indent=2))
