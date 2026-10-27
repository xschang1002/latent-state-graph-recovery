# Experiment and analysis entry points

The study asks when sharing local graph estimates helps recover edges at the
times they are active. The [research guide](../docs/research_guide.md) maps the
questions and current supplement; the [reproduction guide](../docs/information_sciences_artifact_readme.md)
gives complete executable commands and independent destinations. This index
groups scripts by scientific task rather than revision date.

## Verify, rebuild or fit

`python -B scripts/reproduce_paper.py --verify` is the unified read-only entry.
It checks retained inputs/outputs, numerical summaries and presentation sources
without fitting. `--rebuild` regenerates diagnostics/displays from retained
evidence; it does not rerun every scorer or rewrite every frozen analysis CSV.
Add `--compile` only when a supported TeX engine is installed.

The direct-metric scripts use positional `freeze`, `run`, `verify` modes.
Other current analysis runners use flag modes such as `--verify`. For fresh
analysis or fits, the reproduction guide supplies new output directories and
explicit stage/log paths. Redirecting an output alone does not redirect an
older runner's historical execution records.

## Paired studies and generator sensitivity — Supplement S2/S4

| Entry point | Task |
|---|---|
| `run_duration_confirmation.py` | Study A: 100 independent repeats, paired at D=300/600, Lasso/gated-hard primary family and auxiliary OLS/forced output. |
| `run_schedule_robustness.py` | Study B: 50 different repeats across six duration/schedule cells, with separate secondary families. |
| `run_symmetric_sensitivity.py` | Separately specified Study C: 30 new independent repeats, two durations, five window/stride settings, 300 fits. Use both new output and new log paths. |
| `aggregation_study_common.py` | Shared A/B simulation, local scoring and evaluation helpers; original generator remains distinct from C. |

Read-only study checks are `tools/verify_duration_confirmation.py`,
`tools/verify_schedule_robustness.py` and the C runner's `--verify` mode.
The original A/B inferential families are unchanged by the later score/metric
analyses. Internal protocol freezing is distinct from public preregistration.

## Emissions and boundary metrics — Supplement S5

| Entry point | Task |
|---|---|
| `analyze_score_preserving_controls.py` | Post hoc A/B ordinary shrinkage, residual-hard and raw-fallback controls at all fixed weights; flag modes. |
| `analyze_temporal_localisation.py` | A/B positive-change AP, raw-reference F1 and stable-edge variation; positional modes. |
| `analyze_symmetric_controls.py` | Same fixed controls on retained C scores; flag modes. |
| `analyze_symmetric_localisation.py` | Same direct diagnostics on C; positional modes. |
| `analyze_boundary_sensitivity.py` | Standard and positive-change AP, five fixed raw-shared/method-own quantiles and within-/cross-group residual pair accounting, including the 1e-8 tie diagnostic; flag modes. Use both new output and new log paths for regeneration. |

Gated hard (static-max fallback) uses the candidate on acceptance and a static-max output on
rejection. Forced hard always uses the candidate. Raw fallback uses raw scores
on rejection. Each policy and the gate acceptance rate remain separate outputs.
These analysis scripts read retained scores; they do not add independent seeds
or choose a weight/threshold by test performance. Small interval-endpoint
differences in the sensitivity outputs reflect their shared cell/finite-mask
bootstrap stream, not changed old per-repeat measurements.

## Other constructions and computational reproduction — S6/S8

`run_segmentation_validation.py` applies our PELT plus non-overlapping segment
VAR-Lasso adapter to 60 retained C inputs (644 segment refits) and regenerates
and fits eight selected same-seed A/C pipeline cases. Its `--verify` mode does
not call the detector, simulator or scorer. Its explicit execution uses new
output and stage directories. All 492 selected fresh comparisons are exact in
the recorded environment; this is a bounded computational check.

`tools/audit_generator_outputs.py` reads saved A/B coefficients and observed
series for the generator-output diagnostics. It neither fits nor simulates.
Current display builders are `tools/build_information_sciences_assets.py`,
`tools/build_information_sciences_supplement.py`,
`tools/build_reviewer_controls_assets.py` and
`tools/build_evaluation_sensitivity_assets.py`. Their combined checks are
`tools/verify_information_sciences.py`, `tools/verify_reviewer_controls.py` and
`tools/verify_evaluation_sensitivity.py`.

The final publication display layer is `tools/build_narrative_assets.py`, run
last by the unified `--rebuild` route. Its `--verify` mode checks the display
copies and their source mapping without writing. It records the reuse of
original-study intervals for identical paired comparisons; original summary
outputs remain unchanged.

## Auxiliary derivations, comparisons and ESS — S3/S6/S7

The earlier controlled evidence uses `run_mechanism_grid.py`,
`run_control_taxonomy.py`, `run_aggregation_controls.py`, `run_refit_factorial.py`,
`run_known_k_check.py`, `run_segmentation_comparison.py`,
`run_generator_stability.py`, `run_theory_check.py`, `run_final_confirmation.py`,
`run_temporal_delta.py`, `run_state_decomposition.py`, `run_gate_enumeration.py`,
`run_window_contrast.py`, `run_denoising_intervention.py` and
`run_temporal_grid.py`. `run_grouping_controls.py` supplies matched partitions
and three-condition diagnostics; `diagnose_temporal_gmm.py` supplies the
responsibility/tie diagnostic. These retain their original exploratory or
separate-control scope. `freeze_original.py --check` reads the historical freeze.

ESS entry points are `build_ess_mapping.py`, `run_ess_alignment.py` and
`diagnose_ess_null.py`; `tools/verify_ess_refit_provenance.py` checks refit routing.
ESS has static expert graph truth. Older real-data runners and `fetch_*.py`
downloaders retain their acquisition requirements and original study scope.

The [historical command manifest](../docs/reproduction_manifest.md) and the
[history index](../docs/research_guide.md#reproduction-and-historical-records)
retain the older entry points and execution records. Scripts such as
`run_revision_gate_stress.py`, `run_synthetic_dynotears_baseline.py`,
`run_synthetic_neural_granger_baseline.py` and
`run_synthetic_pcmciplus_baseline.py` are archival experiments outside the current
reproduction chain. Their presence does not extend the article's comparisons.
