# Score controls and robustness evidence index — 25 October 2026

All paths are relative to the extracted archive root. This index supplements
`docs/information_sciences_evidence_index_2026-10-24.md`, which remains the
record for the unchanged A/B protocols and earlier evidence. The current title
is **State Aggregation in Dynamic Graph Recovery: Separating Structural Ranking
from Temporal Discrimination**. The revision responds to author-supplied
hypothetical review concerns; it does not claim a received external review or
reproduce the private pasted source.

## Evidence roles and independent units

- **Original A:** seeds 50000–50099, 100 independent mechanisms/histories/noise
  arrays, paired at D=300/600 with Lasso and auxiliary OLS. Its native-Lasso
  Holm4 family is unchanged. **Original B:** seeds 60000–60049, 50 different
  independent repeats paired across six duration/schedule cells. Its secondary
  Holm6 and separate Holm3 families are unchanged.
- **New A/B controls and direct metrics:** post hoc calculations from retained
  fitted outputs. They add no independent repeats and are not new confirmations.
- **Study C:** seeds 70000–70029, 30 new independent repeats paired across two
  durations and five window/stride settings, giving 300 fits. It is separate
  post-review robustness under a new symmetric generator, internally frozen
  before its execution. Its results are not pooled into the A/B test families.
- **C controls/direct metrics:** post hoc applications of the same definitions
  to the completed C outputs. Their analysis locks follow C completion.

All new contrasts are descriptive, with ordinary 95% pointwise percentile
intervals from 10,000 paired-repeat bootstrap draws, seed 20261025. There is no
new hypothesis-test family, simultaneous-interval claim or result-dependent
lambda selection. Windows, edges, cells and method rows are not independent
simulation repeats. The protocols retain explicit missing/failed denominators.

## Exact protocols and entry points

All output directories below are under `results/information_sciences/`.
Protocol SHA-256 values identify bytes, not externally attested historical dates.

| Block | Output directory / protocol | Implementation and mode syntax |
|---|---|---|
| A/B score controls | `score_preserving_controls/protocol.json` | `scripts/analyze_score_preserving_controls.py`; flags `--freeze`, `--run`, `--verify`, `--rebuild` |
| A/B direct recovery | `temporal_localisation/protocol.json` | `scripts/analyze_temporal_localisation.py`; positional `freeze`, `run`, `verify` |
| C simulation/scoring | `symmetric_sensitivity/symmetric_sensitivity_protocol.json` | `scripts/run_symmetric_sensitivity.py`; flags `--freeze`, `--run`, `--aggregate`, `--verify`; configuration `configs/symmetric_sensitivity.yaml` |
| C score controls | `symmetric_controls/protocol.json` | `scripts/analyze_symmetric_controls.py`; flags `--freeze`, `--run`, `--verify`, `--rebuild` |
| C direct recovery | `symmetric_localisation/protocol.json` | `scripts/analyze_symmetric_localisation.py`; positional `freeze`, `run`, `verify` |

Full protocol digests, in the same order:

```text
score_preserving_controls cda9cb757384c32e8c478934c6bbacf9d1c93f4f2a7a205052cb61eda56bad51
temporal_localisation    8220cffbe369d2ba3ed932de94e5be460887878f0f42ad5437d5a8604c69918f
symmetric_sensitivity  9695fba6db49b70dfee55f4aceda199d9f20769b734425302b6fed2022c66e26
symmetric_controls     544b7a5a9560bd142d1b843a9dc7c5ab665c848637585a4618f1f4e57004ea47
symmetric_localisation db922a7429f0f9639d1f087350af74954638785fda46e143b3080e8f5508de99
```

Read-only checks are `python -B tools/verify_reviewer_controls.py` for these
five blocks, or `python -B scripts/reproduce_paper.py --verify` for the complete
article artifact. The direct scripts use `verify` without leading dashes.
All scripts support an output-directory option; analysis scripts also support
`--root` for a separate input tree. An output-directory option alone does not
redirect the retained input study directories.

The unified `--rebuild` command rebuilds the new displays from the supplied new
analysis CSVs, then verifies them by retained-array recomputation. It does not
rewrite every new analysis CSV. The score-control modules expose separate
`--rebuild` flags. Direct CSV regeneration uses its script's `freeze` then `run`
in a new output directory; neither operation fits a scorer. Fresh C generation
is a separate, explicit `--freeze`/`--run` route into a new directory, as detailed
in `docs/information_sciences_artifact_readme.md`.

## Record and summary locations

### Original A/B and new score controls

Original A/B score, truth, input and numerical-diagnostic patterns are unchanged
and indexed in the 24 October evidence index. The public lazy adapter
`iter_retained_cases(root)` in `analyze_score_preserving_controls.py` reads them
one fitted case at a time. It yields failed planned cases rather than dropping
them. The pure `build_control_emissions(...)` and `evaluate_controls(...)` APIs
are shared with C and direct-metric calculations; they do not fit models.

Both `score_preserving_controls/` and `symmetric_controls/` contain:

- `detail.csv`: one case/method row, both temporal targets, lambda/family,
  duplicate endpoint, gate, valid-edge and failure metadata.
- `summary.csv`: absolute means/pointwise intervals and planned/valid/missing
  counts; `contrasts.csv`: within-repeat differences against raw and forced hard.
  C summaries also identify h, stride, M and h/D.
- `affine_checks.csv`: every nonself edge at every positive lambda, with
  time-pair order/tie counts, per-edge AUROC and missing-target reasons.
  `affine_events.csv` retains actual finite order/tie/AUROC changes, with headers
  preserved when empty. `diagnostics.json` gives the complete counts.
- `input_manifest.json`, `protocol.json`, `manifest.json`: original-input,
  configuration/source and derived-output SHA closures.

There are 700 A/B fitted cases and 9,100 score-method rows, and 300 C cases and
3,900 rows. The 672,000 nonself edge-by-lambda checks include 112,960 evaluable
changing-edge checks; all have zero new/lost ties, inversions and temporal-AUROC
changes in these actual float64 outputs. The implementation keeps finite-
precision failures observable rather than substituting idealised ranks.

Positive shrinkage uses lambda in {0.25, 0.5, 0.75, 1}; residual-hard additionally
retains lambda=0. Lambda=1 repeats raw, and residual lambda=0 repeats forced hard.
These duplicate endpoints are labelled, not extra independent methods/samples.
Raw fallback emits the candidate on acceptance and raw windows on rejection;
original native constant fallback and forced candidate remain distinct rows.

### Study C generation and fitted outputs

Within `symmetric_sensitivity/`:

- `inputs/repeat_{seed}.npz` and `.json` retain shared initial/final coefficients,
  supports, initial histories, innovations, radii/scaling and input fingerprints.
- `series/repeat_{seed}_D{300|600}.npz` and `.json` retain full burn-in/observed
  trajectories and schedules, shared across that duration's five window fits.
- `parts/repeat_{seed}_D{duration}_h{h}_s{stride}.npz` and `.json` retain scores,
  emissions, candidate/oracle maps and labels, window truth/bounds/purity,
  generator truth, fit status, warnings and denominators.
- `symmetric_sensitivity_detail.csv`, `symmetric_sensitivity_summary.csv`,
  `symmetric_sensitivity_paired.csv`, `symmetric_sensitivity_gate.csv` and
  `symmetric_sensitivity_diagnostics.csv` retain all fitted cells and summaries.
- `symmetric_sensitivity_outputs.json` seals all 790 files;
  `freeze_record.json`, `run_record_001.json` and `environment.json` are included.

`run_symmetric_sensitivity.py:iter_cases(root)` exposes a lazy array adapter.
The 300 fits use D=300/600 and (h,stride)=(200,100), (300,150), (400,200),
(300,75), (300,300), giving M=35,23,17,45,12 at T=3600. Within repeat, mechanism,
initial values and innovations are paired across every cell. Both states exclude
self support; nonself support marginals match, shared edges keep coefficients,
and common lagwise scaling preserves those shared values. This bundles generator
changes and does not isolate each nuisance factor. Known state count remains two;
the process remains linear and the local scorer remains Lasso.

### Boundary-slot and stable-edge diagnostics

`temporal_localisation/` contains A/B `detail.csv`, `summary.csv`,
`paired_vs_raw.csv`, `preservation.json`, protocol/manifest and 700 small per-case
NPZ files under `records/`. The pattern is
`records/{A|B}_{lasso|ols}_{seed}_D{duration}_{schedule}.npz` (B uses Lasso only).
`symmetric_localisation/` has the analogous outputs for 300 C cases, with pattern
`records/C_{seed}_D{duration}_h{h}_s{stride}.npz`. These per-edge caches retain
covered support-event slots and AP/F1/TP/FP/FN/variation diagnostics rather than
duplicating large emissions.

The definitions are in A/B's direct protocol; C binds its SHA and applies the
same definitions. Truth is at least one actual support change in the interval
between adjacent window centres. Multiple events collapse to one binary slot
but their counts remain available. Positive-change AP processes exact ties as
groups and treats zero strength as abstention; targets without positive or
negative covered slots remain missing. It differs from ordinary AP that includes
a final zero-score group. Boundary F1 and stable-edge variation share the same
raw-derived per-fit 95th-percentile change threshold, with strict exceedance.
This threshold is label-free and descriptive, not a calibrated false-alarm rate.
Persistent and absent edges are separately counted. The time resolution is a
centre-to-centre slot, not sample-level boundary-distance error.

The direct data retain A's auxiliary OLS as well as Lasso: 10,500 A/B and 4,500 C
case/method rows. The eighteen printed Lasso cells are A's two, B's six and C's
ten; this is not a claim that the complete underlying data contain only eighteen
condition/scorer groups.

## Exact printed and presentation locators

| Detail | Supplement locator / display input |
|---|---|
| New controls, affine invariance and fallback identity | S17 (`app:reviewer-controls`), S17.1 (`app:rank-preserving-controls`) |
| All auxiliary OLS score controls | Table S24 (`tab:reviewer-ols`); `reviewer_assets/tab_score_controls_aux.tex` |
| B gate counts and raw-fallback effects | Table S25 (`tab:reviewer-fallback`); `reviewer_assets/tab_fallback_full.tex` |
| Direct definitions and all eighteen Lasso cells | S17.2 (`app:direct-localisation`), Table S26 (`tab:reviewer-localisation`); `reviewer_assets/tab_localisation_full.tex` |
| Stable-present versus stable-absent variation | Table S27 (`tab:reviewer-stable`); `reviewer_assets/tab_stable_changes.tex` |
| Symmetric-generator/window protocol and complete grid | S18 (`app:symmetric-sensitivity`), Table S28 (`tab:reviewer-symmetric`); `reviewer_assets/tab_symmetric_full.tex` |

The display paths in this table have prefix `results/information_sciences/`.
`reviewer_assets/display_source.csv` and `manifest.json` bind displayed values
to the full numerical tables. `tools/build_reviewer_controls_assets.py` generates
the new displays. They do not replace frozen A/B or C numerical records.

## Internal clocks and transparent QA amendments

Seventeen reviewable copies are delivered in
`results/information_sciences/reviewer_provenance/`. Its `export_manifest.json`
lists each original relative source, original SHA, exported file/SHA and exact
normalization scope. JSON canonicalisation/path redaction may change bytes;
historical code snapshots preserve their bytes and have `.py.txt` suffixes.
Private original-log paths document origin and are not required archive members.
C's original freeze/run records remain in the C directory and its seal.

The following UTC values are internal wall-clock records, not file mtimes.
Their Asia/Shanghai calendar date is 25 October 2026 (UTC+08:00). Exit-observation
times in launcher records are observations of returned status, not exact kernel
process-end timestamps. There is no independent external clock attestation.

| Block / record (under reviewer_provenance unless stated) | Freeze/write bracket UTC, 24 October 2026 | Run bracket UTC, 24 October 2026 |
|---|---|---|
| A/B score controls v1: `score_controls_execution.json` | 18:57:27.381336–18:57:40.159309 | start 18:57:40.168309; exit observed 18:58:08.099606 |
| Amended A/B controls: `score_controls_amended_execution.json` | 19:02:47.606354–19:02:48.577030 | start 19:02:48.583386; exit observed 19:03:18.594552 |
| C: `symmetric_sensitivity/freeze_record.json`, `run_record_001.json` (under the main results prefix) | 19:03:07.668595–19:03:07.669605 | 19:03:31.288674–19:03:58.642525 |
| A/B direct: `temporal_localisation_freeze.json`, `temporal_localisation_run.json` | 19:05:20.237213–19:05:20.239747 | 19:05:26.714320–19:05:47.833755 |
| C controls: `symmetric_controls_execution.json` | 19:08:17.866115–19:08:19.948772 | start 19:08:56.512594; exit observed 19:09:18.573821 |
| Amended C direct: `symmetric_localisation_freeze.json`, `symmetric_localisation_run.json` | 19:17:20.159764–19:17:20.161767 | 19:17:29.027358–19:17:42.265283 |

Two amendments followed already observed initial outputs:

1. **Score-control event-list QA.** V1's `NaN != 0` filter included undefined
   per-edge AUC differences as events, although the actual tie/order/AUC changes
   were zero. The amended filter excludes undefined differences while still
   reporting genuine tie or order events independently. The previous protocol
   SHA is `9a7acfd19faa96ac80d2c415de15391e8cd75320a72f504f71c1c0126c67415f`.
   `score_controls_amendment.json` and the final protocol disclose observation
   of v1 results. Detail, summary, contrasts and complete affine checks, plus the
   scientific input manifest, are byte-identical to v1. Only the event list and
   its count changed. V1 protocol/manifest/code are exported with the
   `initial_v1__...` names; the full local v1 checkpoint was preserved.
2. **C direct-metric serialization QA.** Successful C records contain null error
   fields. V1 converted these to a literal string that the CSV reader treated
   as missing. Normalizing them to an empty field repaired the read-only replay.
   The previous protocol SHA is
   `ebfdc6b48edefdb701516942980730b452c27df2a45533faffe497f18a29d9fe`.
   `symmetric_localisation_QA_amendment.json` and
   `symmetric_localisation_QA_numeric_invariance.json` document unchanged summary/
   paired bytes, all 47 nonerror detail columns, and 300 per-edge caches. V1
   protocol/manifest/code are exported with `temporal_localisation_C_initial_v1__...`
   names. Metric definitions, parameters, seeds and underlying fits were unchanged.

These are internally source-locked post hoc analyses with an inspectable
amendment record, not evidence that the amended source had never seen earlier
results, and not public preregistration. Hash integrity establishes byte
consistency; it cannot prove historical precedence independently of the retained
program-order/clock records.

## Interpretation and closure

Shrinkage at lambda=0.25 gives A Lasso pooled AUROC 0.804/0.843 while retaining
raw temporal state AUROC 0.707/0.767. Residual-hard at that weight gives temporal
differences +0.010/+0.057; all other fixed weights remain reported. B raw fallback
at D=600 gives roughly +0.013/+0.013/+0.012, but C includes negative fallback
cells. C's default forced contrasts are approximately -0.047/+0.037; D=600 with
stride 300 gives -0.014 with an interval crossing zero. B D=600 forced output
lowers boundary-slot F1 in all schedules despite positive state-AUROC differences.
The complete outputs therefore support conditional interpretations, not an
inevitable temporal cost, a universally effective fallback or a general duration
threshold. The theory remains diagnostic and the simulated scope two-state linear.

For closure, check the outer package SHA map, then each analysis protocol's
source/input map and its output seal. Check the provenance export SHA map
against exported files; do not require private originals. Check that all indexed
supplement labels/assets exist. Finally run the read-only combined verifier to
recompute metrics/intervals from retained arrays; byte checks alone are not
scientific recomputation. Current environment, full-suite, layout and clean-
archive execution results belong to the corresponding dated audit, rather than
to historical test totals quoted in older records. Public deposition remains
pending actual author-controlled release.
