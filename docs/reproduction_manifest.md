# Reproduction manifest

Current guide: 28 October 2026, delivery release `v2026.10.28`.
This filename-only delivery preserves the PDFs and scientific content of
`v2026.10.27.1`, whose archive link remains printed in the unchanged PDFs.

The scientific entry is [research_guide.md](research_guide.md): questions,
evidence scope and the current supplement map. Exact executable routes are in
[information_sciences_artifact_readme.md](information_sciences_artifact_readme.md),
and [scripts/README.md](../scripts/README.md) groups entry points by task.
The current PDFs are `Manuscript.pdf`
and `Supplementary_Material.pdf`; package
counterparts are `paper/main.pdf` and `paper/supplement.pdf`.
The complete named ZIP, checksums and PDF assets are in the
[public release](https://github.com/xschang1002/latent-state-graph-recovery/releases/tag/v2026.10.28).
The GitHub release and journal submission are separate actions.

The supplement now follows targets (S1), procedures (S2), derivations (S3),
Studies A–C (S4), emissions/boundary sensitivity (S5), additional comparisons
including PELT (S6), ESS (S7), and implementation/reproduction (S8). Dated evidence
indices below retain the numbering of their historical versions. The current
research guide provides the reading map; no frozen record is rewritten to
imply that the new organisation existed at execution time.

`python -B scripts/reproduce_paper.py --verify` checks retained evidence without
fitting. `--rebuild` reconstructs displays from retained inputs. Analysis-CSV
regeneration and fresh generation/fitting use separate documented routes, with
new output and execution-record paths. Eight selected same-seed A/C pipelines
have been generated and fitted afresh; their 492 exact recorded comparisons
are bounded to those cases and the recorded environment.

A/B original inferential families are unchanged. C is a separately specified
study using 30 new independent repeats. Controls and metric sensitivity reuse
fitted arrays post hoc; C itself is not relabelled as a post hoc analysis.
Gated hard (static-max fallback) has a static-max rejection fallback; forced hard always
executes the candidate, and raw fallback retains raw window scores on rejection.
Scientific outputs and frozen protocols are unchanged by this presentation edit.

The remainder is a historical command record. Its dated test counts, section
numbers, availability statements and claims apply to those snapshots; they are
not current acceptance or publication claims. The same archive filename has
been rebuilt. Check the actual delivered manifest/hash and follow the current
instructions above. Public deposition, persistent identifiers and submission
steps are in the [release checklist](../submission_materials/information_sciences/README.md).

## Earlier command and experiment history

2026-10-20 method update: added STCG-v1e, a label-free score-space ensemble over twenty K-means partitions. Window scoring, the supplied state count, and the acceptance gate are unchanged; a rejected candidate falls back to the one-state scores without resampling, and `--ensemble-size 1` reproduces STCG-v1 exactly. The ensemble raises mean Edge AUC on the three main generators (stationary 0.876->0.884, switching 0.812->0.815, nonlinear 0.970->0.971; the last is not significant) and wins 190 of 260 frozen stress fits (0.797->0.801, cell-level p=1.5e-12, seed-clustered p=2.1e-7); it narrows but does not close the Regime-PCMCI gap. Partition instability measured with seeds disjoint from the ensemble correlates with the gain across conditions (r=0.22, p=4.1e-4) but not within them (r=0.08, p=0.21). State-free bagging reproduces the stationary gain and beats the ensemble there (0.886 versus 0.884), while it degrades switching recovery (0.808); random and feature-permuted partitions also fail on switching. Variance reduction is therefore a plausible interpretation of the switching gain, not a demonstrated mechanism. See `scripts/analyze_stcg_ensemble.py` and `scripts/run_stcg_ensemble_controls.py`. Reproduce with `python scripts\run_stcg_ensemble.py --mode main`, `--mode stress`, and `python scripts\run_stcg_ensemble_controls.py`.

2026-09-16 completion update: the frozen specialist stress extension and dedicated timing phase are complete (1,400 planned fits, 1,400 successful, zero failures or timeouts; seeds 100--119), and the corrective shared-chain joint MS-VAR(3) comparison converged on all 20 seeds and is statistically tied with STCG (Edge AUC 0.8274 versus 0.8165, difference +0.0109 [-0.0174, +0.0463], two-sided t p=0.527, Wilcoxon p=0.430). The paper now includes the stress axis table (`tab:specialist-stress`), the matched timing table (`tab:specialist-timing`), and the honest boundary: Regime-PCMCI is significantly stronger at switch periods 150/300 and T=1500 under Holm correction, while STCG-v1's only positive cell (T=300) does not survive correction. All new quoted numbers are machine-checked against the tracked CSVs by `tools/verify_paper_numbers.py` (188 checks).

This manifest records the commands needed to regenerate the current STCG baseline artifacts from the repository root.

2026-09-13 acceptance-fix update: the main scorer uses a supplied K=2 with a one-state fallback; the real regime comparison supplies annotated K to all methods. The paper now distinguishes these protocols from separate multi-K diagnostics, reports two-sided difference tests without claiming equivalence, and withdraws the unsupported synthetic speedup ratio. Actual self-lag features and horizon-purged warning results have been regenerated. See `docs/acceptance_fixes_2026-09-13.md` for changes and validation.

2026-09-09 revision note: the real-data pipeline was re-run after the B1 leakage fix (per-segment interpolation before the chronological split) on a NaN-free data snapshot; no published number moved because of the fix. Full before/after changelog: `docs/audit/real_data_leakage_fix_rerun_2026-09-09.md`. `scripts/run_real_state_label_prediction.py` is DEPRECATED (kept for archaeology); use `run_real_state_label_prediction_forward.py`.

## Environment

Recommended runtime:

- Python 3.12 recommended (the pinned revision environment uses Windows / Python 3.12 / CPU PyTorch); the broad source requirement is Python 3.10+.
- TeX Live with `acmart`
- For the tested pinned environment: `python -m pip install -r requirements-revision.txt` (includes the CPU PyTorch wheel index). For a compatible but unpinned environment use `requirements.txt` or install the project; both include statsmodels and OR-Tools.
- Do not mix historical partial-result caches with reruns after the MSVAR RNG fix. New fits receive an explicit per-dataset seed; historical MSVAR results retain their original, unrecorded retry RNG provenance.

Quick verification:

```powershell
python -m compileall src scripts tests
python -m pytest tests/ -q --basetemp=experiments/logs/pytest_tmp
```

The `--basetemp` flag keeps pytest's temporary directory inside the repository and avoids a machine-specific temp-permission failure on the pinned Windows environment; the current clarification-stage suite passes 353 tests. Older checkpoint counts below retain their historical scope.

Expected status at this checkpoint:

- `pytest`: all tests should pass (353 in the current clarification stage); earlier per-day acceptance runs and their exact counts are recorded in the dated audits.
- `compileall`: passes for `src`, `scripts`, and `tests`.

## Synthetic Recovery

Main recovery table and figure:

```powershell
python scripts\run_synthetic_recovery.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --n-nodes 8 --timesteps 1000 --max-lag 3 --window-size 300 --window-stride 150 --window-aggregate max
```

This main recovery run includes the strong conditional Granger family: global `Granger-F` and dynamic-window `DynGranger-F`.

Outputs:

- `results/tables/synthetic_recovery_detail.csv`
- `results/tables/synthetic_recovery_summary.csv`
- `results/tables/synthetic_recovery_summary.md`
- `results/tables/synthetic_recovery_table.tex`
- `results/figures/synthetic_recovery_edge_auc.png`

Focused PCMCI-ParCorr baseline comparison:

```powershell
python scripts\run_synthetic_pcmci_baseline.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --n-nodes 8 --timesteps 1000 --max-lag 3 --window-size 300 --window-stride 150 --window-aggregate max
python scripts\compare_recovery_methods.py --detail-csv results\tables\synthetic_pcmci_baseline_detail.csv --generators var nonlinear_var switching_var --method-a STCG-v1 --method-b PCMCI-ParCorr --output-prefix synthetic_recovery_paired_pcmci_comparison --latex-generator switching_var
```

Outputs:

- `results/tables/synthetic_pcmci_baseline_detail.csv`
- `results/tables/synthetic_pcmci_baseline_summary.csv`
- `results/tables/synthetic_pcmci_baseline_summary.md`
- `results/tables/synthetic_pcmci_baseline_table.tex`
- `results/figures/synthetic_pcmci_baseline_edge_auc.png`
- `results/tables/synthetic_recovery_paired_pcmci_comparison_detail.csv`
- `results/tables/synthetic_recovery_paired_pcmci_comparison_summary.csv`
- `results/tables/synthetic_recovery_paired_pcmci_comparison_summary.md`
- `results/tables/synthetic_recovery_paired_pcmci_comparison_table.tex`

Paired STCG-v1 vs DynVAR-Lasso comparison:

```powershell
python scripts\compare_recovery_methods.py --generators var nonlinear_var switching_var --method-a STCG-v1 --method-b DynVAR-Lasso
```

Outputs:

- `results/tables/synthetic_recovery_paired_comparison_detail.csv`
- `results/tables/synthetic_recovery_paired_comparison_summary.csv`
- `results/tables/synthetic_recovery_paired_comparison_summary.md`
- `results/tables/synthetic_recovery_paired_comparison_table.tex`

Paired STCG-v1 vs DynGranger-F comparison:

```powershell
python scripts\compare_recovery_methods.py --generators var nonlinear_var switching_var --method-a STCG-v1 --method-b DynGranger-F --output-prefix synthetic_recovery_paired_granger_comparison --latex-generator switching_var
```

Outputs:

- `results/tables/synthetic_recovery_paired_granger_comparison_detail.csv`
- `results/tables/synthetic_recovery_paired_granger_comparison_summary.csv`
- `results/tables/synthetic_recovery_paired_granger_comparison_summary.md`
- `results/tables/synthetic_recovery_paired_granger_comparison_table.tex`

## Extended Synthetic Baselines (PCMCI+, MSVAR, Regime-PCMCI, cMLP, DYNOTEARS)

Five further baselines complete the roster (20 seeds x 3 generators each, defaults match the main recovery run; each runner checkpoints per-seed partials under the gitignored `experiments/logs/<method>_parts/` and merges the reference methods for its summary):

```powershell
python scripts\run_synthetic_pcmciplus_baseline.py
python scripts\run_synthetic_msvar_baseline.py
python scripts\run_synthetic_rpcmci_baseline.py
python scripts\run_synthetic_neural_granger_baseline.py
python scripts\run_synthetic_dynotears_baseline.py
```

Key non-default budgets: Regime-PCMCI uses `num_regimes=2` (the true switching regime count), `num_iterations=20`, `max_anneal=10`, `pc_alpha=0.05`, `alpha_level=0.01`, `tau_max=3`, `max_transitions=10`; the legacy MSAR-X control uses `n_regimes=2`, `maxiter=200`, retry 500, 3 search reps; cMLP uses hidden 8, penalty 0.15, 300 epochs; DYNOTEARS uses `lambda_w=lambda_a=0.05`. The historical MSAR-X raw files retain the identifier `MSVAR` for traceability. This independent-equation autoregressive-error regression is not a shared-chain VAR(L), and its exogenous coefficients are not direct VAR coefficients; its old superiority interpretation is withdrawn. The historical runtime ratio was unsupported and is not reproduced here.

Outputs per baseline (`<name>` in `pcmciplus`, `msvar`, `rpcmci`, `neural_granger`, `dynotears`):

- `results/tables/synthetic_<name>_baseline_detail.csv`
- `results/tables/synthetic_<name>_baseline_summary.csv` / `.md` / `_table.tex`
- `results/figures/synthetic_<name>_baseline_edge_auc.png`

Paired STCG-v1 comparisons per baseline (switching_var latex tables):

```powershell
python scripts\compare_recovery_methods.py --detail-csv results\tables\synthetic_<name>_baseline_detail.csv --generators var nonlinear_var switching_var --method-a STCG-v1 --method-b <MethodName> --output-prefix synthetic_<name>_paired_comparison --latex-generator switching_var
```

Outputs: `results/tables/synthetic_<name>_paired_comparison_detail.csv`, `synthetic_<name>_paired_comparison_summary.csv` / `.md`, and `synthetic_<name>_paired_comparison_table.tex`.

Caveat: Regime-PCMCI on `nonlinear_var` aggregates to n=19 — the seed-18 fit never produced an error-free annealing (0/10) and its NaN row stays in the gitignored partial; the paper's all-methods table footnotes this.

## All-Methods Consolidation

Merges every tracked synthetic detail CSV into the 16-method x 3-generator summary, the switching_var paired matrix (STCG-v1 vs all 15 opponents x 5 metrics), and the all-methods edge-AUC figure:

```powershell
python scripts\summarize_all_recovery_methods.py
```

Outputs:

- `results/tables/synthetic_all_methods_summary.csv` / `.md` / `_table.tex`
- `results/tables/synthetic_switching_paired_matrix.csv` / `.md`
- `results/figures/synthetic_all_methods_edge_auc.png`

## Real Regime Head-to-Head

Descriptive capped-window regime recovery on the two label-carrying datasets. All methods, including STCG-v1, receive the annotated count; STCG uses a zero acceptance threshold. Labels also determine segment placement. This is not held-out forecasting or automatic selection of an unknown K. Resumable per-fit partials live in the gitignored `experiments/logs/real_regime_parts/`:

```powershell
python scripts\run_real_regime_comparison.py
python scripts\run_real_regime_comparison.py --aggregate-only --aggregate-partial
```

Per-call compute budget was < 295 s: STCG all configs in one call; Regime-PCMCI one config (<= 2 seeds) per call; MSVAR bending with `--msvar-maxiter 100 --msvar-retry-maxiter 200 --msvar-search-reps 1`; cooler MSVAR recorded via `--mark-budget-failure` after two genuine budget-killed attempts. Full protocol notes and failure log: `results/tables/real_regime_discovery_comparison.md` and `docs/audit/real_data_repositioning_2026-09-13.md`.

Outputs:

- `results/tables/real_regime_discovery_comparison.csv` / `.md`
- `results/tables/real_regime_discovery_comparison_detail.csv`

## Knob Ablation

One-factor-at-a-time isolation of the STCG-v1 heuristic knobs (paper Table `tab:knob-ablation`) on switching_var, 20 seeds; the windows group uses a period-300 switcher:

```powershell
python scripts\run_stcg_knob_ablation.py --group all
```

Outputs per group (`gates`, `aggregators`, `features`, `windows`):

- `results/tables/stcg_knob_ablation_<group>_detail.csv`
- `results/tables/stcg_knob_ablation_<group>_summary.csv` / `.md`
- `results/tables/stcg_knob_ablation_summary.md` (combined)
- `results/figures/stcg_knob_ablation_edge_auc.png`

## Paper Table/Number Tooling

The compact paper-facing tables are generated from the tracked CSVs (never hand-typed), and the numbers quoted in `paper/main.tex` are verified against the CSVs:

```powershell
python tools\build_revision_tables.py
python tools\verify_paper_numbers.py
python tools\validate_paper_static.py
```

Outputs:

- `results/tables/synthetic_switching_paired_edge_auc_table.tex`
- `results/tables/real_regime_discovery_comparison_table.tex`
- `results/tables/stcg_knob_ablation_table.tex`

`verify_paper_numbers.py` checks the quoted numbers (synthetic headline and paired stats, oracle/diagnostics/robustness/difficulty, real regime head-to-head, alignment, forecasting, warning, knob ablation) against the tracked CSVs and fails on any mismatch; `validate_paper_static.py` checks cite keys, refs, braces, environments, and recursively included table syntax.

## Two-Sided and Post-Pilot Sensitivity Statistics

```powershell
python tools\build_confirmation_statistics.py
```

This recomputes the original switching STCG/Regime-PCMCI two-sided tests and a retrospective sensitivity analysis excluding cMLP pilot seeds 0--4. Every method uses common seeds 5--19 within each generator; nonlinear seed 18 is excluded from all methods because RPCMCI failed. Outputs are `synthetic_rpcmci_two_sided_summary.csv`, `synthetic_postpilot_all_methods_detail.csv`, `synthetic_postpilot_all_methods_summary.csv`, `synthetic_postpilot_paired_summary.csv`, and `synthetic_postpilot_analysis.md` under `results/tables/`. The 225 paired rows include both superiority and two-sided difference tests. This sensitivity analysis is not a newly collected independent confirmation set.

## STCG-v1 Ablation

```powershell
python scripts\run_stcg_ablation.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --n-nodes 8 --timesteps 1000 --max-lag 3 --window-size 300 --window-stride 150
```

Outputs:

- `results/tables/stcg_v1_ablation_detail.csv`
- `results/tables/stcg_v1_ablation_summary.csv`
- `results/tables/stcg_v1_ablation_summary.md`
- `results/tables/stcg_v1_ablation_table.tex`
- `results/figures/stcg_v1_ablation_edge_auc.png`

## STCG-v1 Oracle-State Mechanism Check

```powershell
python scripts\run_synthetic_state_oracle.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
```

This diagnostic compares STCG-v1 inferred graph-state aggregation with an oracle-state upper bound
that uses the true synthetic regime label only after local window scoring. It is not a supervised
training variant.

Outputs:

- `results/tables/synthetic_state_oracle_detail.csv`
- `results/tables/synthetic_state_oracle_summary.csv`
- `results/tables/synthetic_state_oracle_summary.md`
- `results/tables/synthetic_state_oracle_table.tex`
- `results/figures/synthetic_state_oracle_edge_auc.png`

## Synthetic Switch-Difficulty Sweep

```powershell
python scripts\run_synthetic_switch_difficulty.py --seeds 0 1 2 3 4 5 6 7 8 9 --switch-periods 150 300 600
```

This boundary check varies the regime switch period while keeping the sliding-window setup fixed.
Short periods create mixed windows and are expected to reduce the benefit of state aggregation.

Outputs:

- `results/tables/synthetic_switch_difficulty_detail.csv`
- `results/tables/synthetic_switch_difficulty_summary.csv`
- `results/tables/synthetic_switch_difficulty_summary.md`
- `results/tables/synthetic_switch_difficulty_table.tex`
- `results/figures/synthetic_switch_difficulty_edge_auc.png`

## STCG-v1 Robustness

```powershell
python scripts\run_stcg_robustness.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
```

Outputs:

- `results/tables/stcg_v1_robustness_detail.csv`
- `results/tables/stcg_v1_robustness_summary.csv`
- `results/tables/stcg_v1_robustness_summary.md`
- `results/tables/stcg_v1_robustness_table.tex`
- `results/figures/stcg_v1_robustness_edge_auc.png`

## State Diagnostics

```powershell
python scripts\run_stcg_state_diagnostics.py --seeds 0 1 2 3 4 --plot-seed 0
```

Outputs:

- `results/tables/stcg_v1_state_diagnostics_detail.csv`
- `results/tables/stcg_v1_state_diagnostics_summary.csv`
- `results/tables/stcg_v1_state_diagnostics_summary.md`
- `results/figures/stcg_v1_state_diagnostics_seed0.png`

## Downstream Forecasting Harness

Synthetic switching smoke run:

```powershell
python scripts\run_downstream_forecasting.py --seeds 0 1 2 3 4
```

Outputs:

- `results/tables/downstream_forecasting_detail.csv`
- `results/tables/downstream_forecasting_summary.csv`
- `results/tables/downstream_forecasting_summary.md`
- `results/figures/downstream_forecasting_rmse.png`
- `results/tables/downstream_graph_top_edges.csv`
- `results/tables/downstream_graph_states.csv`
- `results/tables/downstream_graph_diagnostics.md`

Local CSV entry point:

```powershell
python scripts\fetch_ett_benchmarks.py
python scripts\fetch_beijing_air_quality.py
python scripts\fetch_uci_occupancy.py
python scripts\fetch_uci_arem.py
python scripts\fetch_uci_har.py
# Preferred real operating-mode candidate. The full UCI/Zenodo zip can be used when the network allows it.
python scripts\fetch_hydraulic_systems.py --force
# Current 2026-05-15 checkpoint used a GitHub-mirror low-frequency subset zip named hydraulic_systems_lowfreq.zip.
python scripts\fetch_hydraulic_systems.py --zip-name hydraulic_systems_lowfreq.zip --sensors TS1,TS2,TS3,TS4,VS1,CE,CP,SE --source-url https://github.com/DrewAlderfer/hydraulic_pump_monitoring/tree/main/data --skip-md5-check
python scripts\check_real_data_inventory.py
python scripts\run_real_benchmark_from_inventory.py
python scripts\run_downstream_forecasting.py --input-csv data\raw\your_timeseries.csv --time-column timestamp --dataset-name your_dataset
```

Inventory outputs:

- `results/tables/real_data_inventory.csv`
- `results/tables/real_data_inventory.md`
- `results/tables/real_benchmark_run_plan.md`

The CSV runner learns graph scores only from the training segment and evaluates one-step prediction on held-out future samples.
It reports both hard graph-mask ridge and soft graph-weighted ridge; the soft version keeps all lagged features and uses graph scores as diagonal ridge-prior weights.
Use `configs/real_benchmark_template.yaml` to record dataset-specific settings before reporting a real-data result.
`scripts\fetch_hydraulic_systems.py` supports the UCI Hydraulic Systems candidate. The 2026-05-15 checkpoint recovered the data through a public GitHub mirror and prepared the low-frequency/summary sensor subset; see `docs/audit/hydraulic_systems_mode_transfer_2026-05-15.md`.
To run all inventory-approved datasets into ignored per-dataset folders under `experiments/logs/real_benchmarks/`, use:

```powershell
python scripts\run_real_benchmark_from_inventory.py --execute
python scripts\summarize_real_benchmarks.py
```

Path-rendering note (2026-09-09): to regenerate the tracked `real_data_inventory.*` byte-identically, run `python scripts\check_real_data_inventory.py --raw-dir data\raw` (a relative path keeps the `file` column relative). The current `run_real_benchmark_from_inventory.py` renders `--input-csv` in `real_benchmark_run_plan.md` as absolute paths; the tracked plan keeps the older relative rendering, which is cosmetic only.

Real benchmark summary outputs:

- `results/tables/real_benchmark_forecasting_summary.csv`
- `results/tables/real_benchmark_forecasting_summary.md`
- `results/tables/real_benchmark_forecasting_table.tex`
- `results/tables/real_benchmark_top_edges.csv`
- `results/tables/real_benchmark_top_edges.md`
- `results/tables/real_benchmark_state_summary.csv`
- `results/tables/real_benchmark_state_summary.md`

External real-state label alignment for datasets with held-out audit labels:

```powershell
python scripts\compare_real_state_labels.py
python scripts\compare_real_state_labels.py --states-csv experiments\logs\real_benchmarks\AReMActivities\tables\downstream_graph_states.csv --labels-csv data\processed\AReMActivities_labels.csv --dataset-name AReMActivities --label-column activity
python scripts\run_real_state_sensitivity.py
python scripts\run_arem_threshold_forecasting.py
python scripts\run_arem_threshold_forecasting.py --input-csv data\raw\UCIHARWindowSignals.csv --labels-csv data\processed\UCIHARWindowSignals_labels.csv --dataset-name UCIHARWindowSignals --label-column activity --state-counts 2 3 4 --window-size 80 --window-stride 40 --min-improvements 0.0 0.04 0.08 0.12 0.16
python scripts\compare_real_state_labels.py --states-csv experiments\logs\real_benchmarks\HydraulicSystems\tables\downstream_graph_states.csv --labels-csv data\processed\HydraulicSystems_labels.csv --dataset-name HydraulicSystems --label-column cooler_condition
python scripts\run_arem_threshold_forecasting.py --input-csv data\raw\HydraulicSystems.csv --labels-csv data\processed\HydraulicSystems_labels.csv --dataset-name HydraulicSystems_cooler_condition --label-column cooler_condition --columns TS1,TS2,TS3,TS4,VS1,CE,CP,SE --state-counts 2 --min-improvements 0.0 0.08 0.12 0.16 --ridge-alphas 1.0 10.0 100.0 --graph-min-weights 0.01 0.10 --graph-weight-powers 1.0 4.0 --window-size 300 --window-stride 150
python scripts\run_arem_threshold_forecasting.py --input-csv data\raw\HydraulicSystems.csv --labels-csv data\processed\HydraulicSystems_labels.csv --dataset-name HydraulicSystems --label-column stable_flag --columns TS1,TS2,TS3,TS4,VS1,CE,CP,SE --state-counts 2 3 --min-improvements 0.0 0.08 0.12 0.16 --ridge-alphas 1.0 10.0 100.0 --graph-min-weights 0.01 0.10 --graph-weight-powers 1.0 4.0 --window-size 300 --window-stride 150
python scripts\run_arem_threshold_forecasting.py --input-csv data\raw\HydraulicSystems.csv --labels-csv data\processed\HydraulicSystems_labels.csv --dataset-name HydraulicSystems_hydraulic_accumulator --label-column hydraulic_accumulator --columns TS1,TS2,TS3,TS4,VS1,CE,CP,SE --state-counts 2 3 4 --min-improvements 0.0 --ridge-alphas 1.0 --graph-min-weights 0.10 --graph-weight-powers 1.0 --window-size 300 --window-stride 150 --seeds 0
python scripts\run_arem_threshold_forecasting.py --input-csv data\raw\HydraulicSystems.csv --labels-csv data\processed\HydraulicSystems_labels.csv --dataset-name HydraulicSystems_internal_pump_leakage --label-column internal_pump_leakage --columns TS1,TS2,TS3,TS4,VS1,CE,CP,SE --state-counts 2 3 4 --min-improvements 0.0 --ridge-alphas 1.0 --graph-min-weights 0.10 --graph-weight-powers 1.0 --window-size 300 --window-stride 150 --seeds 0
python scripts\run_arem_threshold_forecasting.py --input-csv data\raw\HydraulicSystems.csv --labels-csv data\processed\HydraulicSystems_labels.csv --dataset-name HydraulicSystems_valve_condition --label-column valve_condition --columns TS1,TS2,TS3,TS4,VS1,CE,CP,SE --state-counts 2 3 4 --min-improvements 0.0 --ridge-alphas 1.0 --graph-min-weights 0.10 --graph-weight-powers 1.0 --window-size 300 --window-stride 150 --seeds 0
```

Outputs:

- `results/tables/OccupancySensors_state_label_alignment_detail.csv`
- `results/tables/OccupancySensors_state_label_alignment_summary.csv`
- `results/tables/OccupancySensors_state_label_alignment_summary.md`
- `results/figures/OccupancySensors_state_label_alignment_seed0.png`
- `results/tables/AReMActivities_state_label_alignment_detail.csv`
- `results/tables/AReMActivities_state_label_alignment_summary.csv`
- `results/tables/AReMActivities_state_label_alignment_summary.md`
- `results/tables/AReMActivities_state_sensitivity_summary.csv`
- `results/tables/AReMActivities_state_sensitivity_summary.md`
- `results/tables/AReMActivities_state_threshold_forecasting_detail.csv`
- `results/tables/AReMActivities_state_threshold_forecasting_summary.csv`
- `results/tables/AReMActivities_state_threshold_forecasting_summary.md`
- `results/tables/UCIHARWindowSignals_state_label_alignment_detail.csv`
- `results/tables/UCIHARWindowSignals_state_label_alignment_summary.csv`
- `results/tables/UCIHARWindowSignals_state_label_alignment_summary.md`
- `results/tables/UCIHARWindowSignals_state_threshold_forecasting_detail.csv`
- `results/tables/UCIHARWindowSignals_state_threshold_forecasting_summary.csv`
- `results/tables/UCIHARWindowSignals_state_threshold_forecasting_summary.md`
- `results/tables/HydraulicSystems_state_label_alignment_detail.csv`
- `results/tables/HydraulicSystems_state_label_alignment_summary.csv`
- `results/tables/HydraulicSystems_state_label_alignment_summary.md`
- `results/tables/HydraulicSystems_state_threshold_forecasting_detail.csv`
- `results/tables/HydraulicSystems_state_threshold_forecasting_summary.csv`
- `results/tables/HydraulicSystems_state_threshold_forecasting_summary.md`
- `results/tables/HydraulicSystems_cooler_condition_state_threshold_forecasting_detail.csv`
- `results/tables/HydraulicSystems_cooler_condition_state_threshold_forecasting_summary.csv`
- `results/tables/HydraulicSystems_cooler_condition_state_threshold_forecasting_summary.md`
- `results/tables/HydraulicSystems_hydraulic_accumulator_state_threshold_forecasting_detail.csv`
- `results/tables/HydraulicSystems_hydraulic_accumulator_state_threshold_forecasting_summary.csv`
- `results/tables/HydraulicSystems_hydraulic_accumulator_state_threshold_forecasting_summary.md`
- `results/tables/HydraulicSystems_internal_pump_leakage_state_threshold_forecasting_detail.csv`
- `results/tables/HydraulicSystems_internal_pump_leakage_state_threshold_forecasting_summary.csv`
- `results/tables/HydraulicSystems_internal_pump_leakage_state_threshold_forecasting_summary.md`
- `results/tables/HydraulicSystems_valve_condition_state_threshold_forecasting_detail.csv`
- `results/tables/HydraulicSystems_valve_condition_state_threshold_forecasting_summary.csv`
- `results/tables/HydraulicSystems_valve_condition_state_threshold_forecasting_summary.md`
- `results/tables/AReMActivities_threshold_paired_tests.csv`
- `results/tables/AReMActivities_threshold_paired_tests.md`

The AReM paired-test files are derived artifacts: per-seed STCG-v1 minus DynVAR-Lasso weighted-ridge RMSE deltas for setting `auto_k4_min_0.00` (ridge alpha `100`, graph min weight `0.01`, power `4.0`) from `AReMActivities_state_threshold_forecasting_detail.csv`, with paired t-test and one-sided Wilcoxon statistics (currently `p=0.142` and `p=0.0625`; mean delta `-0.002102`, 4/5 seeds improve).
- `results/figures/UCIHARWindowSignals_state_label_alignment_seed0.png`
- `results/figures/AReMActivities_state_label_alignment_seed0.png`
- `results/figures/HydraulicSystems_state_label_alignment_seed0.png`

Real state-label prediction beyond one-step forecasting:

> DEPRECATED harness: the three `run_real_state_label_prediction.py` commands below are kept for archaeology only. That script fits KMeans centroids on the full series and randomly splits overlapping windows, which is systematically optimistic; the canonical commands are the two `run_real_state_label_prediction_forward.py` invocations further below.

```powershell
python scripts\run_real_state_label_prediction.py --input-csv data\raw\HydraulicSystems.csv --labels-csv data\processed\HydraulicSystems_labels.csv --dataset-name HydraulicSystems --label-column cooler_condition --columns TS1,TS2,TS3,TS4,VS1,CE,CP,SE --window-size 300 --window-stride 150
python scripts\run_real_state_label_prediction.py --input-csv data\raw\HydraulicSystems.csv --labels-csv data\processed\HydraulicSystems_labels.csv --dataset-name HydraulicSystems_cooler_K3 --label-column cooler_condition --columns TS1,TS2,TS3,TS4,VS1,CE,CP,SE --window-size 300 --window-stride 150 --stcg-v1-states 3
python scripts\run_real_state_label_prediction.py --input-csv data\raw\AReMActivities.csv --labels-csv data\processed\AReMActivities_labels.csv --dataset-name AReMActivities_K4 --label-column activity --columns avg_rss12,var_rss12,avg_rss13,var_rss13,avg_rss23,var_rss23 --window-size 300 --window-stride 150 --stcg-v1-states 4
python scripts\run_real_state_count_selection.py --input-csv data\raw\HydraulicSystems.csv --labels-csv data\processed\HydraulicSystems_labels.csv --dataset-name HydraulicSystems_cooler_selectK --label-column cooler_condition --columns TS1,TS2,TS3,TS4,VS1,CE,CP,SE --window-size 300 --window-stride 150 --state-counts 1 2 3 4 5 6
python scripts\run_real_state_count_selection.py --input-csv data\raw\AReMActivities.csv --labels-csv data\processed\AReMActivities_labels.csv --dataset-name AReMActivities_selectK --label-column activity --columns avg_rss12,var_rss12,avg_rss13,var_rss13,avg_rss23,var_rss23 --window-size 300 --window-stride 150 --state-counts 1 2 3 4 5 6
python scripts\run_real_state_label_prediction_forward.py --input-csv data\raw\HydraulicSystems.csv --labels-csv data\processed\HydraulicSystems_labels.csv --dataset-name HydraulicSystems_forward --label-column cooler_condition --columns TS1,TS2,TS3,TS4,VS1,CE,CP,SE --window-size 300 --window-stride 150 --stcg-v1-states 3 --stcg-v1-force-states --train-fraction 0.80 --seeds 0 1 2 3 4
python scripts\run_real_state_label_prediction_forward.py --input-csv data\raw\HydraulicSystems.csv --labels-csv data\processed\HydraulicSystems_labels.csv --dataset-name HydraulicSystems_valve_forward --label-column valve_condition --columns TS1,TS2,TS3,TS4,VS1,CE,CP,SE --window-size 300 --window-stride 150 --stcg-v1-states 4 --stcg-v1-force-states --train-fraction 0.80 --seeds 0 1 2 3 4
python scripts\run_hydraulic_transition_warning.py --input-csv data\raw\HydraulicSystems.csv --labels-csv data\processed\HydraulicSystems_labels.csv --dataset-name HydraulicSystems --label-columns valve_condition stable_flag --columns TS1,TS2,TS3,TS4,VS1,CE,CP,SE --horizon-cycles 3 5 10 --lookback-cycles 40
python scripts\run_hydraulic_warning_ablation.py --input-csv data\raw\HydraulicSystems.csv --labels-csv data\processed\HydraulicSystems_labels.csv --dataset-name HydraulicSystems --label-columns valve_condition stable_flag --columns TS1,TS2,TS3,TS4,VS1,CE,CP,SE --horizon-cycles 3 5 10 --lookback-cycles 30 40 60 --selection-objectives stable transition --stcg-window-sizes 16 24 --stcg-window-strides 8
```

Outputs:

- `results/tables/HydraulicSystems_state_label_prediction_detail.csv`
- `results/tables/HydraulicSystems_state_label_prediction_summary.csv`
- `results/tables/HydraulicSystems_state_label_prediction_summary.md`
- `results/figures/HydraulicSystems_state_label_prediction_balanced_accuracy.png`
- `results/tables/HydraulicSystems_cooler_K3_state_label_prediction_detail.csv`
- `results/tables/HydraulicSystems_cooler_K3_state_label_prediction_summary.csv`
- `results/tables/HydraulicSystems_cooler_K3_state_label_prediction_summary.md`
- `results/figures/HydraulicSystems_cooler_K3_state_label_prediction_balanced_accuracy.png`
- `results/tables/AReMActivities_K4_state_label_prediction_detail.csv`
- `results/tables/AReMActivities_K4_state_label_prediction_summary.csv`
- `results/tables/AReMActivities_K4_state_label_prediction_summary.md`
- `results/figures/AReMActivities_K4_state_label_prediction_balanced_accuracy.png`
- `results/tables/HydraulicSystems_cooler_selectK_state_count_selection_detail.csv`
- `results/tables/HydraulicSystems_cooler_selectK_state_count_selection_summary.csv`
- `results/tables/HydraulicSystems_cooler_selectK_state_count_selection_selected.csv`
- `results/tables/HydraulicSystems_cooler_selectK_state_count_selection_summary.md`
- `results/figures/HydraulicSystems_cooler_selectK_state_count_selection.png`
- `results/tables/AReMActivities_selectK_state_count_selection_detail.csv`
- `results/tables/AReMActivities_selectK_state_count_selection_summary.csv`
- `results/tables/AReMActivities_selectK_state_count_selection_selected.csv`
- `results/tables/AReMActivities_selectK_state_count_selection_summary.md`
- `results/figures/AReMActivities_selectK_state_count_selection.png`
- `results/tables/HydraulicSystems_forward_state_label_prediction_forward_detail.csv`
- `results/tables/HydraulicSystems_forward_state_label_prediction_forward_summary.csv`
- `results/tables/HydraulicSystems_forward_state_label_prediction_forward_summary.md`
- `results/figures/HydraulicSystems_forward_state_label_prediction_forward_balanced_accuracy.png`
- `results/tables/HydraulicSystems_valve_forward_state_label_prediction_forward_detail.csv`
- `results/tables/HydraulicSystems_valve_forward_state_label_prediction_forward_summary.csv`
- `results/tables/HydraulicSystems_valve_forward_state_label_prediction_forward_summary.md`
- `results/figures/HydraulicSystems_valve_forward_state_label_prediction_forward_balanced_accuracy.png`
- `results/tables/HydraulicSystems_transition_warning_detail.csv`
- `results/tables/HydraulicSystems_transition_warning_summary.md`
- `results/tables/HydraulicSystems_transition_warning_best.md`
- `results/figures/HydraulicSystems_transition_warning_average_precision.png`
- `results/tables/HydraulicSystems_warning_ablation_detail.csv`
- `results/tables/HydraulicSystems_warning_ablation_selected.csv`
- `results/tables/HydraulicSystems_warning_ablation_summary.md`
- `results/figures/HydraulicSystems_warning_ablation_average_precision.png`

Hard graph-mask sweep:

```powershell
python scripts\run_real_mask_sweep.py --datasets ETTh1 ETTh2
```

Outputs:

- `results/tables/real_mask_sweep_detail.csv`
- `results/tables/real_mask_sweep_summary.csv`
- `results/tables/real_mask_sweep_summary.md`

Soft graph-prior sweep:

```powershell
python scripts\run_real_weight_sweep.py
```

Outputs:

- `results/tables/real_weight_sweep_detail.csv`
- `results/tables/real_weight_sweep_summary.csv`
- `results/tables/real_weight_sweep_summary.md`

## Additional Frozen Revision Evidence

The 2026-09-13 protocols are distributed as `docs/revision_specialist_protocol_2026-09-13.md`, `docs/revision_real_protocol_2026-09-13.md`, and `docs/revision_gate_protocol_2026-09-13.md`. The original reviews and current item-by-item assessment are included separately. These additional checks do not retroactively change the historical main-panel protocols.

Offline specialist reaggregation, with no dataset download or estimator fitting:

```powershell
python tools\summarize_revision_specialist.py
```

The command consumes `results/tables/revision_specialist_protocol.json`, the explicit storage amendment, and `revision_specialist_parts/*.json`; it verifies the frozen document and estimator-source hashes. It produces detail/coverage/summary CSVs, finite-common-seed paired statistics, exclusive timing summaries when available, analysis notes and provenance. Prefer this analysis tool over the historical runner's direct aggregate entry because it records the zero-resolution CPU-clock amendment. Undefined metrics remain unavailable and use metric-specific common seed sets; they do not turn successful fits into failures.

The export also preserves scheduling/storage amendments, the interrupted-status snapshot and original failure records. Only machine prefixes in diagnostic logs or `error`/`traceback` strings are removed in exported copies. `PACKAGE_MANIFEST.json` contains exported byte hashes and separate original hashes for each changed diagnostic; frozen protocols, source hashes and scientific fields remain intact. Raw sensor arrays, model binaries, ordinary scratch JSON and worker logs outside the explicit history allowlist remain excluded.

An export taken during execution is an explicitly incomplete checkpoint. Its manifest reports expected, included and missing frozen stress/timing job records; the coverage CSV gives method-specific successes and metric availability. To require a complete record set at release:

```powershell
python scripts\build_submission_package.py --no-pdf --require-complete-specialist
```

This requirement counts retained failures/timeouts as completed records, not successful scientific results. Without that flag, a partial checkpoint is allowed and labeled as such. Reaggregation works in the extracted artifact with no Git repository. The frozen source fingerprint must still match; do not alter estimator files before reaggregation.

The release check also requires the saved analysis provenance to have summarized every included checkpoint. An in-progress snapshot can have newer raw parts than its CSV summaries; its manifest reports `stored_summary_fit_count` and `summary_count_matches_checkpoint`. Reaggregate the snapshot before interpreting its specialist summaries.

After all 1,400 frozen stress/timing records are available and summarized, run the independent specialist paired-statistics/interval/Holm/timing checks and generate the full 15-setting tables, timing table and effect figure:

```powershell
python tools\verify_review_response_evidence.py --include-specialist
python tools\build_review_response_tables.py --include-specialist
```

These branches require complete specialist evidence. As of 2026-09-16 both have been run against the complete 1,400-record checkpoint and pass; the generated `revision_specialist_paper_table.tex` and `revision_specialist_timing_table.tex` are included in the paper (Section `sec:exp-specialist-stress`), and `results/figures/revision_specialist_rpcmci_effect.png` is retained in the artifact. `tools/run_revision_specialist_tail.py` is the scheduling-only tail driver that finished the remaining frozen fits without a controller: it re-enumerates pending jobs from the frozen protocol and reuses the fingerprinted runner for all estimation, generation, and metric code. The recent literature scope is documented separately in `docs/related_work_scope_2026-09-13.md`.

The additional forward-diagnosis raw predictions and origin/seed summaries are `results/tables/revision_real_forward_*`. Their four coordinate-selected origins share training history, and all six comparators and negative outcomes are retained; they do not establish STCG utility superiority. Verify the exported CSV evidence and rebuild its paper tables without fitting:

```powershell
python tools\verify_review_response_evidence.py
python tools\build_review_response_tables.py
```

The explicitly exported `experiments/logs/revision_real_validation/executed_manifest.json` and `reporting_manifest.json` record the original data/source signatures and the later reporting-source hash. They contain audit metadata, not sensor data or fitted caches. The real runner's `--aggregate-from-manifest` requires the original data and omitted cache parts, so it cannot reproduce the real experiment offline from this source package alone. For a full new fit after public-data ingestion, run `python scripts\run_revision_real_validation.py --stage all --parts-dir experiments/logs/revision_real_fresh`; this creates fresh source/data provenance and does not claim the old execution hash.

Gate outputs use `revision_gate_*`; `positive_fallback` is a post-finding logic-fix variant, while the frozen `full` selector rows explicitly use the historical tie-break. The gate runner's reporting-only clarification after execution is recorded separately from its execution hash.

## Paper Build

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper paper/main.tex
bibtex paper/main
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper paper/main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper paper/main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper paper/main.tex
```

Output:

- `paper/main.pdf`

The PDF and LaTeX intermediate files are ignored by git to avoid binary and build-output churn.

## Current Evidence Checkpoint

- Main synthetic STCG uses a fixed K=2 candidate and a label-free acceptance gate. Multi-K q_K selection is a separate diagnostic, not the algorithm behind the main table.
- Switching Edge AUC is 0.812 versus Regime-PCMCI 0.816. Two-sided t p=0.668 and Wilcoxon p=0.596 do not establish equivalence; the reported 95% CI permits either a loss or gain. The original descriptive twenty-seed panel favors STCG over the other thirteen reference methods under one-sided tests; the separately identified legacy MSAR-X control is excluded from that count.
- cMLP used seeds 0--4 to set hyperparameters. After excluding them, switching STCG/cMLP Edge AUC is 0.803/0.775 on fifteen common seeds, with one-sided t p=0.0086 and Wilcoxon p=0.0051. Full common-seed comparisons are saved above.
- A dedicated exclusive timing phase now supplies the matched measurement the main panel lacked: on the base switching configuration over seeds 100--119, Regime-PCMCI's geometric-mean wall-clock cost is 5575x STCG-v1's (95% CI [5246, 5984]; process-CPU ratio 6293x), cMLP costs 86.6x [81.8, 92.1], and VAR-Lasso (0.116x) and DynVAR-Lasso (0.458x) are faster than STCG-v1. The earlier roughly 200x speedup statement remains withdrawn; stress-grid times collected under concurrent load are descriptive only.
- The actual-self-lag ablation now retains self coefficients for clustering but excludes self edges from returned graph scores. Edge AUC is 0.813 versus default 0.812, and state ARI 0.730 versus 0.764. Default and added-window-statistics recovery results are unchanged.
- Real capped-segment results use annotated K for every method and a zero STCG acceptance threshold. They show descriptive regime alignment, not label-free recovery of an unknown count.
- Full-series alignment and historical random-overlapping-window label prediction are descriptive and optimistic. The canonical future-prediction entry is the forward harness, which fits only chronological training windows.
- Forecasting remains diagnostic: dense ridge/persistence are stronger, and tuned AReM gains are not statistically confirmed. Existing NaN-free forecasting records were not rerun during this acceptance fix.
- Warning and warning-ablation results have been regenerated with horizon-sized purges before validation/test boundaries. Test anchors remain fixed; the last H training and validation origins are excluded. Raw sensor imputation uses forward fill only and rejects unresolved leading/non-finite values. See the split-policy columns in the regenerated CSVs and the acceptance-fix note for revised metrics.
- The period-150 switch-difficulty result retains two states and nearly matches the plain-window Edge AUC; it is not a gate fallback. Switch difficulty uses ten seeds and state diagnostics five, while the main panel, robustness and knob sweeps use twenty.
- The historical robustness/difficulty sweeps used their documented narrower panels. The frozen specialist panel is now complete (1,400 of 1,400 fits, zero failures or timeouts): the operating-point tie with Regime-PCMCI replicates on fresh seeds 100--119, while the specialist is significantly stronger at switch periods 150/300 and T=1500 under the Holm-corrected primary family, and STCG-v1's only positive cell (T=300) does not survive correction. The corrective shared-chain joint MS-VAR(3) comparison on the base condition (20/20 converged) is statistically tied with STCG (two-sided t p=0.527, Wilcoxon p=0.430).
- Final PDF compilation, page/layout inspection, submission-system actions and remote artifact synchronization remain manual release steps. Historical dated audit notes describe earlier snapshots and are superseded by this manifest and the acceptance-fix note where they conflict.

## Variance-Reduced Ensemble (2026-10-20)

```
python scripts\run_stcg_ensemble.py --mode main
python scripts\run_stcg_ensemble.py --mode stress
python scripts\analyze_stcg_ensemble.py
python scripts\run_stcg_ensemble_controls.py
python scripts\run_stcg_ensemble_sensitivity.py
```

Outputs:

- `results/tables/stcg_ensemble_detail.csv`, `stcg_ensemble_summary.csv`, `stcg_ensemble_paired.csv`
- `results/tables/stcg_ensemble_stress_detail.csv`, `stcg_ensemble_stress_summary.csv`, `stcg_ensemble_stress_paired.csv`
- `results/tables/stcg_ensemble_instability.csv`, `stcg_ensemble_instability_summary.csv`, `stcg_ensemble_instability_stats.csv`
- `results/tables/stcg_ensemble_controls_detail.csv`, `stcg_ensemble_controls_summary.csv`
- `results/tables/stcg_ensemble_b_sensitivity.csv`, `stcg_ensemble_b_sensitivity_summary.csv`
- `results/tables/stcg_ensemble_table.tex`
- `results/figures/stcg_ensemble_instability.png`

The run regenerates the STCG-v1 rows exactly (checked against `revision_specialist_detail.csv`) before computing STCG-v1e. `--ensemble-size 1` is the original STCG-v1 path. The instability restarts use seeds disjoint from the ensemble members; the controls use a state-free bootstrap mean, random balanced partitions, and K-means on column-permuted features. The sensitivity run varies $B\in\{5,10,20,40\}$: the switching mean stays at 0.815 while the paired p ranges from 0.032 (B=20) to 0.081 (B=40), which is why the paper calls the main-panel switching gain borderline.

## Change-Point Calibrated Gate (2026-10-21)

```
python scripts\run_change_point_gate.py --split development
python scripts\run_change_point_gate.py --split validation
```

Outputs:/n/n- `results/tables/change_point_gate_development.csv`, `change_point_gate_development_summary.csv`
- `results/tables/change_point_gate_validation.csv`, `change_point_gate_validation_summary.csv`
- `results/tables/change_point_gate_table.tex`

The rule is frozen in `docs/change_point_gate_protocol_2026-10-21.md`: VAR(3) Chow scan, 99 residual-bootstrap draws, alpha=0.05, scan step 10 for T<=2000 and 50 for longer series. Development seeds are the main panel 0--19 and gate stress 100--119; validation uses fresh seeds 200--219. The gate is optional and production defaults are unchanged.

## Real-Data Boundary Check and Theory (2026-10-21)

```
python scripts\run_real_change_point_gate.py --dataset hydraulic
python scripts\run_real_change_point_gate.py --dataset arem
python scripts\analyze_ensemble_variance.py
python scripts\analyze_aggregation_bias.py
```

Outputs:/n/n- `results/tables/real_change_point_gate_hydraulic.csv`, `real_change_point_gate_hydraulic_summary.csv`
- `results/tables/real_change_point_gate_arem.csv`, `real_change_point_gate_arem_summary.csv`
- `results/tables/real_change_point_gate_table.tex`
- `results/tables/stcg_ensemble_variance.csv`
- `results/tables/aggregation_bias.csv`

The real-data check is a negative transfer result: HydraulicSystems accepts 34.2% of condition boundaries and 30.0% of stable segments; AReM accepts 100% of both. The labels mark operating conditions or activities, not stationary VAR segments. The theoretical checks quantify the max-aggregation bias (toy AUC 0.934 vs 0.901 for shared edges; 0.774 vs 0.803 for state-specific edges) and the ensemble member correlation (0.97).

## Revision 2026-10-21: estimands, mechanism grid, confirmatory run

The manuscript was restructured around three estimands. Run, in order:

```
python scripts/freeze_original.py --check
python scripts/run_mechanism_grid.py
python scripts/run_aggregation_controls.py
python scripts/run_baselines_wp4.py
python scripts/run_confirmatory.py --split development --n-seeds 50
python scripts/run_control_taxonomy.py
python scripts/run_known_k_check.py
python scripts/run_refit_factorial.py
python scripts/run_segmentation_comparison.py
python scripts/run_generator_stability.py
python scripts/run_confirmatory.py --split confirmation --n-seeds 50
python scripts/run_final_confirmation.py
python scripts/run_theory_check.py
python scripts/run_ess_alignment.py --all-imputations --draws 5000 --null-seeds 0 1 2 3 4
python tools/build_revision2_tables.py
python tools/verify_revision2_numbers.py
```

Outputs:

- Historical `configs/original_frozen.yaml`, `results/original_reproduction.csv`,
  and `audit/source_to_table.md` are retained. `freeze_original.py --check` verifies
  the 49 summary/detail pairs without writing. An explicit `--snapshot-dir` must
  name a new directory and never replaces the historical freeze.
- `results/mechanism/mechanism_grid.csv` and `_summary.csv` (35 configurations
  x 5 seeds, seven negative controls).
- `results/aggregation/aggregation_controls.csv` and `_summary.csv`
  (shared window scores, nine aggregation rules, label-error sweep).
- `results/baselines/wp4_comparison.csv` and `_summary.csv`.
- `results/confirmatory/confirmatory_detail.csv`,
  `confirmatory_tests_{development,confirmation}.csv`,
  `confirmatory_per_seed_{development,confirmation}.csv`,
  `audit/failure_log_{development,confirmation}.csv`.
- `results/theory/theory_mse_grid.csv`, `theory_mse_simulation.csv`,
  `theory_mse_comparison.csv`, `theory_max_bias.csv`.
- `results/tables/ess_alignment.csv`.
- `results/mechanism/control_taxonomy.csv` and `_summary.csv`.
- `results/mechanism/known_k_check.csv` and `_summary.csv`.
- `results/baselines/refit_factorial.csv` and `_summary.csv`.
- `results/baselines/segmentation_comparison.csv`, `_summary.csv`,
  `pelt_penalty_scan.csv`, `pelt_penalty_scan_summary.csv`.
- `results/mechanism/generator_stability.csv` and `_summary.csv`.
- `results/confirmatory/final_confirmation_detail.csv`,
  `final_confirmation_tests.csv`, `final_confirmation_per_seed.csv`.
- `results/revision2/tab_*.tex` and `results/figures/rev2_*.png`.

The confirmatory seeds are 2000--2049 and were generated after
`analysis/statistical_plan.md` was written. Development seeds are 1000--1049.
Seeds 0--19, 100--119 and 200--219 are development data.

ESS data: `data/external/essdata/` contains `preprocessedData.csv`,
`adjMatrix.csv`, and the MIT licence from `github.com/soerenwengel/essdata`;
`pv_subsystem_map.csv` is derived from the reduced Zenodo archive layout by
`scripts/build_ess_mapping.py`. Set `ESS_DATA_DIR` to override the location.

Verification: `tools/verify_revision2_numbers.py` checks 184 manuscript numbers
against these CSVs; `tools/verify_paper_numbers.py` now verifies the archived
pre-revision manuscript in `paper/legacy/`.

### Correction pass (same date)

Two issues found by review changed the results and are recorded here.

1. **Control taxonomy.** The lag-shift condition was previously grouped with
   negative controls. It changes the lag-specific support, which is the target
   estimand, so it is a structure change. `scripts/run_control_taxonomy.py`
   fixes the taxonomy (structure / mechanism / nuisance) and reports lag-level
   and pair-level AUROC separately. No generator setting changed.
2. **PELT adapter.** `segmented_lasso_scores` built refit ranges from window
   bounds, so adjacent ranges overlapped by one stride and reused 300-row
   windows even when a segment held one window. The 2x2 factorial
   (`scripts/run_refit_factorial.py`) showed PELT returns one segment per window
   in 200/200 confirmatory fits; the adapter was replaced by
   `src/stcg/refit.py`, which partitions at window-centre midpoints. The
   confirmatory run was repeated on the same seeds and the manuscript reports
   the corrected numbers. Both runs are documented in
   `analysis/statistical_plan.md`.
3. **Comparable metrics.** `per_regime_metrics` now returns
   `regime_prevalence` and `regime_auprc_lift`, and `union_metrics` returns
   `auprc_lift`, so the two layers are compared with AUROC and with AUPRC
   relative to prevalence rather than with raw AUPRC alone.

### Third pass: targets, segments, and a clean hold-out

1. **Two structural counts.** `structure_counts` in `src/stcg/mechanism.py`
   reports `k_dyn` (dynamical mechanisms) and `k_graph` (binary supports). The
   mechanism and nuisance families have `k_dyn=2` with `k_graph=1`, so binary
   regime metrics are not a two-graph problem there.
2. **Joint temporal target.** `temporal_graph_metrics` scores
   `G[z_hat(t)]` against `A[z(t)]` window by window. It is reported in the
   aggregation, control and final-confirmation tables.
3. **Fair segmentation comparison.** `scripts/run_segmentation_comparison.py`
   holds the segment count at two (constrained split), adds a raw-VAR Chow scan,
   and scans the PELT penalty and minimum length. `constrained_segments` and
   `var_change_point` live in `src/stcg/segmented.py`.
4. **Final hold-out.** Seeds 2000--2049 were used while the PELT adapter was
   corrected, so they are not clean. `scripts/run_final_confirmation.py` uses
   seeds 3000--3049 with a fixed method set and a Holm family of nine tests; the
   global VAR-Lasso is descriptive with its own interval.
5. **Metric normalisation.** `union_metrics` and `per_regime_metrics` now return
   `auprc_lift` and `auprc_normalized`; the manuscript reports both and draws no
   ordering between layers.
6. **ESS sensitivity.** `run_ess_alignment.py --all-imputations` compares
   forward fill, backward fill, linear interpolation and column means, and
   Holm-corrects the eight tests across 200 permutations.
7. **Generator stability audit.** `scripts/run_generator_stability.py` measures
   the switching path over 200 seeds instead of relying on per-state spectral
   radius alone.


### Audit corrections (2026-10-23)

- `segmented_lasso_scores` keeps adjacent time ranges separate and processes
  contiguous runs in temporal order, independently of numeric state labels.
  ESS now has 23 detected segments and 23 disjoint refits for every imputation.
- ESS uses labelled in/out-degree-preserving switch-chain nulls, including
  three-cycle reversals and rejected proposals. Each edge set uses 200 samples,
  seed 0, 100|E| burn-in proposals and 10|E| proposals between samples; methods
  share null graphs. This finite-chain null is approximate. The previous
  off-diagonal shuffle preserved only the total edge count.
- Reproduce with `python scripts/run_ess_alignment.py --all-imputations`, then
  `python tools/build_revision2_tables.py` and
  `python tools/verify_revision2_numbers.py`. Pre-correction results are retained
  in `results/revision2/ess_alignment_before_audit_2026-10-23.csv`.
- Graph F1 includes completely wrong nonempty graphs as zero; state scores use
  rows ordered by unique predicted label, state F1 is invariant to renumbering,
  and SHD at a zero edge budget selects no edges. Regression tests cover these
  cases. Historical synthetic CSVs retain their original metric versions; the
  correction does not change the manuscript's AUROC/AUPRC inputs.
- The eight-test forward-fill Holm family is unchanged. Lagged correlation on
  strong edges has raw p=0.004975 and adjusted p=0.039801; it is the only rejection
  in that family. No test survives the combined 32-test correction. These are
  exploratory degree-conditioned comparisons with 1/201 Monte Carlo resolution.


### ESS Monte Carlo expansion (2026-10-23, supersedes the 200-draw stage)

The default ESS entry now uses 5,000 draws and null seeds 0--4. The corrected
200-draw reference is archived as
`results/revision2/ess_alignment_200_draws_seed0.csv`. The current command above
writes a 32-row fixed-seed-0 report and a 160-row
`results/tables/ess_alignment_seed_sensitivity.csv`, including exceedance counts,
per-seed primary Holm p-values, and per-seed combined-32 Holm p-values. No seed
is selected by its result and p-values are not pooled.

Strong-edge forward-fill lagged correlation has reference raw p=0.003199 and
Holm p=0.025595; all five seeds pass, with Holm range 0.015997--0.036793.
All seven other primary comparisons remain above 0.05. Under combined-32
correction, linear-imputation strong-edge lagged correlation passes 4/5 seeds
(range 0.019196--0.051190), and mean imputation passes 1/5. No combined-family
rejection persists in all five seeds. This supersedes the earlier combined-family
no-rejection statement and retains the exploratory finite-chain interpretation.

All fitted AUROCs/AUPRCs and refit counts are unchanged. Null graphs are reused
across methods and imputations; the rank-sum evaluation is checked against scalar
AUROC, including ties. Independent default-command replay reproduces both CSVs
exactly. Current source/result hashes and validation are recorded in
`audit/ess_monte_carlo_5000_2026-10-23.json` and the accompanying report.


### ESS implementation and null-chain diagnostics (MLJ cleanup, 2026-10-23)

```powershell
python tools/verify_ess_refit_provenance.py
python scripts/diagnose_ess_null.py
python tools/build_revision2_tables.py
python tools/verify_revision2_numbers.py
```

The first command is read-only. It reconstructs the former global Lasso scores
(all/strong AUROC 0.576290/0.657464) and the corrected 23 disjoint refits
(0.564972/0.554761), checks equality with both the corrected 200-draw and current
5,000-draw results, and checks current synthetic source routing. This score
change is the touching-range merge repair, not a Monte Carlo effect.

The factorial, confirmatory, final-confirmation and segmentation runners already
use `src/stcg/refit.py`; only ESS and the legacy WP4 runner call the repaired
`segmented_lasso_scores`. WP4's archived CSVs are not inputs to current tables.
A seed-0 replay of all four factorial settings (24 rows) exactly matches the
stored shared AUROC/AUPRC fields. Historical F1 fields retain their old versions.

`diagnose_ess_null.py` replays the ten published chains and adds two seed-0
chains with 1,000|E| rather than 100|E| burn-in. Each still retains 5,000 draws
after 10|E| attempted transitions. Outputs are
`results/tables/ess_null_diagnostics.csv` (12 rows) and
`results/tables/ess_null_traces.csv` (60,000 rows). Counters do not change the
random stream; the ten original chain means and tail counts match the primary
CSV. The builder now produces 13 tables and eight figures, including Appendix
D's diagnostic table/figure. Primary Table 9/Figure 6 remain unchanged numerically
and are kept before the Discussion.

The target is uniform on labelled simple loop-free digraphs with fixed in/out
degrees. The supplement gives the symmetric with-replacement edge-pair and
vertex-triple proposals. Rejects are self-transitions. Stationarity and small
sample autocorrelations do not prove mixing, and the plus-one p formula does not
give ordinary sequential MCMC an exact test guarantee. Diagnostic sensitivity
chains do not replace primary chains or expand the Holm family.

The next temporal h/D-by-M comparison (raw, hard, soft, oracle; gate acceptance
separated) is recorded as planned work, not as a completed experiment here.

### Temporal-grid extension (2026-10-23, supersedes the planning statement above)

```powershell
python scripts/run_temporal_grid.py
python scripts/diagnose_temporal_gmm.py
python tools/verify_temporal_grid.py
python tools/build_revision2_tables.py
python tools/verify_revision2_numbers.py
```

Execution is complete: the 35 labelled entries in `mechanism_grid.yaml` contain
31 distinct generator configurations, giving 620 fits at seeds 0--19 and 3,500
method rows. Four duplicated configurations reuse their original observations.
The primary pipeline freezes source/config hashes before running. A changed
protocol requires a fresh `--output-dir`; historical core CSVs remain unchanged.

New `results/revision2/` inputs are `temporal_grid_detail.csv` (3,500 rows),
`temporal_grid_summary.csv` (700 rows), `temporal_grid_protocol.json`, and
`temporal_grid_gmm_numerics.csv` (360 diagnostic rows from 120 replayed fits).
The builder now produces 16 tables and nine figure designs. The temporal figure
has both a 600-dpi PNG and an embedded-font vector PDF, both included in the
package. Core Table 3 adds GMM weighted-map/argmax and continuous-emission rows;
Table 5/Figure 5 report duration and length scans. Appendix D records accepted
subgroups and numerical sensitivity; the ESS appendix is now E.

`verify_temporal_grid.py` checks all 1,400 historical hard/oracle grid values
and gate decisions, 60 core raw/hard/oracle values, and 20 historical GMM pooled
values. It verifies score pairing, every subgroup's own denominator, all missing
targets, duplicate contexts, gate fallback versus forced candidates, native
execution rates and source hashes. Dataset and historical freeze checks remain
unchanged. See the new audit record for exact effect estimates and limits.

### Matched grouping and factor diagnostics (2026-10-23)

```powershell
python scripts/reproduce_paper.py --verify
python scripts/reproduce_paper.py --rebuild
python scripts/reproduce_paper.py --rebuild --compile
```

The first command is read-only; in an extracted package it first checks every
exported payload against `PACKAGE_MANIFEST.json`. The second reads the supplied
old-fit cache, regenerates grouping analyses and all 19 tables/nine figure
designs, then checks numerical claims and provenance. The third also compiles
with Tectonic or latexmk on PATH. Export checksums refer to the delivered bytes,
before a rebuild; rebuilt PDF timestamps can differ.

`scripts/run_grouping_controls.py` generated the forty deterministic old-fit
replays at D=600/M=7 and D=300/M=19, checked all raw/candidate AUROCs and gates
against the original grid, and saved `grouping_replay_outputs.npz` with a hash
manifest. Subsequent default runs reuse it. To reconstruct rather than reuse,
run `python scripts/run_grouping_controls.py --refit --output-dir NEW_DIRECTORY`.
The core D=300/M=7 factors come directly from `state_decomposition.csv`.
No new simulation seed or independent positive-gain validation is introduced.

At D=600/M=7, `matched_random_partitions.csv` contains all 616 unlabelled
group-size-matched partitions across twenty original seeds (21 or 35 per seed).
Each partition's maps and AUROC are computed before averaging within seed.
`matched_random_seed_summary.csv` and `matched_random_contrasts.csv` use the
twenty seeds as independent units, with the existing 10,000-draw/seed-7 paired
percentile bootstrap. Hard-minus-matched-random is +0.221557
[0.192541, 0.250025]; hard-minus-raw remains +0.030690 [0.004523, 0.056608].
These selected-cell comparisons are exploratory.

`grouping_state_factors_detail.csv` (60 rows) and its three-row summary report
mean J, mean C_edge, mean per-seed product, hard and raw AUROC. Native output
is used for the two seven-window cells; all forced candidates are used at
D=300/M=19. The largest per-seed identity error is 2.22e-16. The separate
`temporal_grid_coverage.csv` reports all 35 labelled configurations' evaluable
seeds, support-change absence, observed-flip absence and duplicate links.
`tools/verify_grouping_controls.py` reanalyses the saved numeric arrays and
checks these CSVs without fitting. All ESS result/diagnostic inputs and the
original temporal-grid outputs remain unchanged. The latest provenance is in
`audit/grouping_controls_2026-10-23.md` and its JSON companion.

### Final text, information sheet and independent-environment verification (2026-10-23)

This is a prose/submission-material stage, with no new simulation or ESS fit.
Immediately after Table 6, the existing two-state identity is rewritten as
the mean hard-minus-raw gain on the same seeds. Full-precision inputs give
hard +0.149047 and raw +0.051160 from D=300 to 600, hence a descriptive gain
change of +0.097887. It is not a new theorem, interaction test or selection
threshold. Five old target/design/variance/gate/default statements are corrected.

The separate two-page information sheet and source are in
`submission_materials/mlj_contribution_information/`, including the author's
confirmation of no formal prior publication or other overlapping review.
Its four questions follow the MLJ instructions checked on 2026-10-23:
https://link.springer.com/journal/10994/submission-guidelines .

The package entries remain read-only verification and rebuilding from supplied
outputs. Choose the appropriate rebuild variant from a fresh extraction if a
prior rebuild changed manifest-covered bytes. The final stage uses a new Python
3.12.13 virtual environment, without inherited system/user site packages or
PYTHONPATH, and a fresh offline install of all 47 pinned dependencies. It checks
`--verify` and `--rebuild` in two separately extracted copies; this improves on
earlier smoke tests that used the original research environment. Exact command
logs, isolation records and checksums accompany the stage audit:
`audit/mlj_final_text_2026-10-23.md` and its JSON companion.
