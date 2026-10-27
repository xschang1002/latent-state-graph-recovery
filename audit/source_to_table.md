# Source-to-table map (frozen 2026-10-21)

Every table in `paper/main.tex` must trace to a script and a CSV. This file is
the audit trail for the pre-revision manuscript. Rows marked **hand-built**
were assembled by hand from the listed CSV in an earlier session; they are the
only tables without a generator, and WP8 replaces them with script-generated
output.

| # | `\input` file | Generator | Source data |
|---|---|---|---|
| 1 | `synthetic_all_methods_table.tex` | `scripts/summarize_all_recovery_methods.py` | `synthetic_all_methods_summary.csv` |
| 2 | `synthetic_rpcmci_paired_comparison_table.tex` | `scripts/compare_recovery_methods.py` | `synthetic_rpcmci_paired_comparison_summary.csv` |
| 3 | `synthetic_switching_paired_edge_auc_table.tex` | `tools/build_revision_tables.py` | `synthetic_switching_paired_matrix.csv` |
| 4 | `synthetic_state_oracle_table.tex` | `scripts/run_synthetic_state_oracle.py` | `synthetic_state_oracle_summary.csv` |
| 5 | `synthetic_recovery_paired_comparison_table.tex` | `scripts/compare_recovery_methods.py` | `synthetic_recovery_paired_comparison_summary.csv` |
| 6 | `synthetic_recovery_paired_pcmci_comparison_table.tex` | `scripts/compare_recovery_methods.py` | `synthetic_recovery_paired_pcmci_comparison_summary.csv` |
| 7 | `stcg_ensemble_table.tex` | **hand-built** | `stcg_ensemble_detail.csv`, `stcg_ensemble_stress_detail.csv` |
| 8 | `revision_specialist_paper_table.tex` | `tools/build_review_response_tables.py` | `revision_specialist_summary.csv` |
| 9 | `revision_specialist_timing_table.tex` | `tools/build_review_response_tables.py` | `revision_specialist_timing.csv` |
| 10 | `real_regime_discovery_comparison_table.tex` | `tools/build_revision_tables.py` | `real_regime_discovery_comparison_detail.csv` |
| 11 | `revision_real_paper_table.tex` | `tools/build_review_response_tables.py` | `revision_real_forward_detail.csv`, `revision_real_segment_detail.csv` |
| 12 | `real_benchmark_forecasting_table.tex` | `scripts/summarize_real_benchmarks.py` | `downstream_forecasting_detail.csv` |
| 13 | `paper_evidence_summary_table.tex` | `tools/build_revision_tables.py` | multiple frozen CSVs |
| 14 | `stcg_knob_ablation_table.tex` | `tools/build_revision_tables.py` | `stcg_v1_ablation_detail.csv` |
| 15 | `change_point_gate_table.tex` | **hand-built** | `change_point_gate_development.csv`, `change_point_gate_validation.csv` |
| 16 | `real_change_point_gate_table.tex` | **hand-built** | `real_change_point_gate_hydraulic_summary.csv`, `real_change_point_gate_arem_summary.csv` |
| 17 | `revision_gate_paper_table.tex` | `tools/build_review_response_tables.py` | `revision_gate_stress_detail.csv` |

## Freeze artefacts

- `configs/original_frozen.yaml`: commit, environment, seed sets, panel
  definitions, combined SHA-256 over code, configs, and every result CSV/TEX.
- `results/original_reproduction.csv`: 49 `(generator, method)` pairs checked
  against per-seed detail; 49 match at `1e-9`, 0 drift, 0 missing.
- `scripts/freeze_original.py` regenerates both without re-running experiments.

## Numbers verifier

`tools/verify_paper_numbers.py` recomputes 188 manuscript numbers from the
frozen CSVs. Any change to a table or a claim must keep it at
`checked 188 numbers / ALL NUMBERS MATCH`.

## Known gaps carried into WP8

1. Tables 7, 15, 16 have no generator; their values are correct but not
   script-rebuildable.
2. The frozen panel is descriptive (20 seeds) and has no seed-clustered
   inference for the main table; WP7 adds it.
3. The `config` column in `revision_specialist_detail.csv` is a JSON string and
   must be parsed before filtering on `switch_period` (see
   `scripts/freeze_original.py`).

## Revision tables (2026-10-21)

The revised manuscript uses a separate, script-generated set. Every file below
is produced by `tools/build_revision2_tables.py` from the listed CSV and no
number is typed by hand.

| Table | `\input` file | Source data |
|---|---|---|
| Main confirmatory | `results/revision2/tab_confirmatory.tex` | `results/confirmatory/confirmatory_tests_confirmation.csv` |
| 2x2 factorial | `results/revision2/tab_factorial.tex` | `results/baselines/refit_factorial.csv` |
| Segmentation comparison | `results/revision2/tab_segmentation.tex` | `results/baselines/segmentation_comparison_summary.csv` |
| PELT penalty scan | `results/revision2/tab_penalty_scan.tex` | `results/baselines/pelt_penalty_scan_summary.csv` |
| Final hold-out | `results/revision2/tab_final.tex` | `results/confirmatory/final_confirmation_tests.csv` |
| Metric layers | `results/revision2/tab_metrics_layers.tex` | `results/aggregation/aggregation_controls.csv` |
| Control taxonomy | `results/revision2/tab_controls.tex` | `results/mechanism/control_taxonomy_summary.csv` |
| Known-K protocol | `results/revision2/tab_known_k.tex` | `results/mechanism/known_k_check_summary.csv` |
| Prevalence and lift | `results/revision2/tab_prevalence.tex` | `results/aggregation/aggregation_controls.csv` |
| Aggregation controls | `results/revision2/tab_aggregation.tex` | `results/aggregation/aggregation_controls_summary.csv` |
| Negative controls | `results/revision2/tab_controls.tex` | `results/mechanism/mechanism_grid.csv` |
| Axis scans | `results/revision2/tab_axes.tex` | `results/mechanism/mechanism_grid.csv` |
| Cost | `results/revision2/tab_cost.tex` | `results/confirmatory/confirmatory_detail.csv` |
| ESS alignment | `results/revision2/tab_ess.tex` | `results/tables/ess_alignment.csv` |

Verification: `tools/verify_revision2_numbers.py` checks 140 numbers against
these CSVs and against the manuscript text. The pre-revision tables above remain
frozen for the archived manuscript in `paper/legacy/`.
