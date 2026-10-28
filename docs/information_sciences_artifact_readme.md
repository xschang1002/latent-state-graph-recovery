# State aggregation in dynamic graph recovery: reproduction guide

This archive accompanies **State Aggregation in Dynamic Graph Recovery:
Separating Structural Ranking from Temporal Discrimination**, by Xuansong Chang,
Stony Brook Institute at Anhui University, Hefei, China.
Correspondence: alec.xs.chang@gmail.com.

The scientific question is whether sharing local graph scores improves recovery
of edges at the times they are active. Pooled edge ranking, within-edge state
discrimination and adjacent-window boundary detection are evaluated separately.
`docs/research_guide.md` maps the scientific questions and results. The archive's
main article and supplement are `paper/main.pdf` and `paper/supplement.pdf`, with
TeX sources beside them. This is presentation revision
`v2026.10.28-layout.1`, with five main figures and three main tables. The
accompanying `information_sciences_reproducibility.zip` and its checksums identify
the current deliverable. The separate PDFs are `Manuscript.pdf` and
`Supplementary_Material.pdf`. The
layout revision map, `docs/ins_layout_revision_2026-10-28.md`, records moved and
combined displays in the exported archive.
Scientific inputs, fitted outputs and inferential results are unchanged.

The [versioned public release](https://github.com/xschang1002/latent-state-graph-recovery/releases/tag/v2026.10.28-layout.1)
provides the complete named reproducibility archive, both PDFs and checksums.
The preceding release, `v2026.10.28`, remains unchanged and contains the same
scientific evidence with its earlier presentation. GitHub's automatically
generated source ZIP is a different container from the named reproducibility archive.
Compilation paths remain `paper/main.pdf` and `paper/supplement.pdf`.

## What to reproduce

| Evidence | Retained outputs | Scientific role |
|---|---|---|
| Study A | `results/information_sciences/duration_confirmation/` | 100 independent repeats, seeds 50000–50099, paired at D=300/600. The original four Lasso/gated-hard tests form the primary Holm4 family; OLS and forced-hard comparisons are auxiliary. |
| Study B | `results/information_sciences/schedule_robustness/` | 50 different repeats, seeds 60000–60049, across six duration/schedule cells. Secondary Holm6 and Holm3 families are separate from A. Gate acceptance, gated output and forced candidates are distinguished. |
| Study C | `results/information_sciences/symmetric_sensitivity/` | A separately specified symmetric-generator/window study: 30 new independent repeats, seeds 70000–70029, paired over two durations and five window/stride settings, or 300 fitted cells. It neither replaces the A/B generator nor extends their original test families. |
| Emission controls | `score_preserving_controls/`, `symmetric_controls/` under the same results root | Additional fixed-weight shrinkage, residual and raw-fallback analyses of retained A/B/C scores. No weight is selected by test performance. |
| Boundary diagnostics | `temporal_localisation/`, `symmetric_localisation/`, `boundary_sensitivity/` | Both AP conventions, every fixed quantile under raw-shared/method-own references, stable-edge rates and near-zero residual pair accounting. The sensitivity block analyzes 1,000 retained cases without fitting. |
| Other partition/refit structure | `segmentation_validation/` | PELT plus our non-overlapping segment VAR-Lasso adapter on 60 C inputs, with 644 refits. Forty-five cases select 11 segments, the minimum-length cap at 23 windows; the setting is not retuned. |
| Selected fresh fitting | `segmentation_validation/fresh/`, `fresh_comparisons.csv` | Eight original-seed A/C generation/fitting reruns; all 492 recorded comparisons are exact in the recorded environment. This is neither new independent evidence nor a fresh full-grid run. |
| Auxiliary and industrial evidence | Earlier `results/revision2/` and ESS inputs/results | Exploratory controls, diagnostic derivations and static expert alignment retain their own scope. ESS does not supply time-indexed graph truth. |

The current supplement is organised by scientific topic: S1 targets and
notation; S2 generator and procedures; S3 derivations; S4 Studies A–C; S5 emission
and boundary sensitivity; S6 additional comparisons including PELT; S7 ESS; S8
implementation and reproduction. The current guide supplies detailed locations.
Dated evidence indices preserve the section numbering of their own versions.

## Output policies and inferential scope

Raw window output retains each window's scores. Forced hard always emits the
candidate hard-group scores. Gated hard (static-max fallback) accepts that candidate or emits
the edgewise raw maximum at every window on rejection (static-max fallback).
Raw fallback instead retains raw window scores on rejection. Oracle output
uses simulation truth. Gate acceptance and effects on the accepted subgroup
are separate from all-case method differences. `STCG-v1` is an implementation
identifier retained in sources and historical records.

The original A/B families are unchanged. C uses new independent repeats under a
separate stated design; the score/metric controls reuse already fitted arrays
and are post hoc. Their intervals are pointwise paired-repeat bootstrap
intervals, not simultaneous intervals or additional original-family tests.
All fixed weights, duplicate endpoints, undefined targets and failure reasons
remain available. Internal source/configuration freezing is not public
preregistration. Full inputs, per-repeat arrays, source hashes and normalized
execution records are included for traceability; they are not the scientific
contribution itself.

Repeated displays of an identical paired AUROC comparison use the original
corresponding-study interval. The publication display manifest in
`results/information_sciences/narrative_assets/manifest.json` records the
matching repeat identities, paired values and interval sources, including
sign-reversed comparisons. Raw-fallback, boundary and pair-component intervals
retain their own analysis definitions and resampling streams. All original
summary outputs remain available alongside these publication displays.

## Environment and strictly read-only verification

Python 3.12 with `requirements-revision.txt` is the tested baseline. The file
contains 47 exact dependency pins and the official CPU PyTorch wheel index;
it is not a wheel-hash lock. In a new environment:

```text
python -m pip install -r requirements-revision.txt
python -B scripts/reproduce_paper.py --verify
```

Installation may require internet access. Verification of supplied results does
not. The default verifier checks the outer archive inventory, source/input/output
seals, original A/B evidence, the control/C blocks and the
sensitivity/segmentation blocks, per-repeat metrics, float64
invariance diagnostics, direct metrics, presentation numbers and manuscript
structure. It does not generate new random inputs or fit a scorer. Some checks
deterministically replay saved innovations/trajectories; this is retained-input
verification, not a new stochastic sample.

Focused read-only routes are:

```text
python -B tools/verify_information_sciences.py
python -B tools/verify_reviewer_controls.py
python -B tools/verify_evaluation_sensitivity.py
python -B scripts/analyze_score_preserving_controls.py --verify
python -B scripts/analyze_temporal_localisation.py verify
python -B scripts/run_symmetric_sensitivity.py --verify
python -B scripts/analyze_symmetric_controls.py --verify
python -B scripts/analyze_symmetric_localisation.py verify
python -B scripts/analyze_boundary_sensitivity.py --verify
python -B scripts/run_segmentation_validation.py --verify
```

The direct-metric scripts take positional `freeze`, `run` or `verify`; they do
not take `--verify`. The score-control scripts use flag modes. For the full
regression suite, run `python -B -m pytest tests/ -q` with a fresh local
`--basetemp`. Dated test counts in older audit records describe those snapshots;
they are not a claim about the current suite.

## Rebuild displays versus regenerate analysis CSVs

```text
python -B scripts/reproduce_paper.py --rebuild
```

This route rebuilds the earlier retained-cache grouping analyses, earlier
tables/figures, saved-output generator diagnostics and the current
display assets. `tools/build_reviewer_controls_assets.py` and
`tools/build_evaluation_sensitivity_assets.py` read the already supplied analysis
CSVs. The unified route does **not** rewrite all frozen control/direct-metric CSVs and does not fit Study C again. Its verifier
still recomputes the scientific values from retained arrays and checks the
CSVs. PDF metadata such as creation time may differ after display rebuilding.
Use two separate fresh extractions for an unchanged manifest audit and a rebuild.

The two score-control CSV sets also have explicit deterministic rebuild modes:

```text
python -B scripts/analyze_score_preserving_controls.py --rebuild
python -B scripts/analyze_symmetric_controls.py --rebuild
```

They preserve the frozen protocol/input manifest and do not rewrite historical
pre-run clocks. The direct-metric CSVs can be independently regenerated into a
new output directory, for example:

```text
python -B scripts/analyze_temporal_localisation.py freeze --output-dir results/reproduction_fresh/direct_AB --stage-dir experiments/logs/reproduction_fresh/direct_AB
python -B scripts/analyze_temporal_localisation.py run --output-dir results/reproduction_fresh/direct_AB --stage-dir experiments/logs/reproduction_fresh/direct_AB
python -B scripts/analyze_temporal_localisation.py verify --output-dir results/reproduction_fresh/direct_AB --stage-dir experiments/logs/reproduction_fresh/direct_AB
```

For C use its own output and execution-record directories:

```text
python -B scripts/analyze_symmetric_localisation.py freeze --output-dir results/reproduction_fresh/direct_C --stage-dir experiments/logs/reproduction_fresh/direct_C
python -B scripts/analyze_symmetric_localisation.py run --output-dir results/reproduction_fresh/direct_C --stage-dir experiments/logs/reproduction_fresh/direct_C
python -B scripts/analyze_symmetric_localisation.py verify --output-dir results/reproduction_fresh/direct_C --stage-dir experiments/logs/reproduction_fresh/direct_C
```

These are analysis-only operations on retained inputs. Keep both directories new
for each independent reproduction. The frozen scripts retain their historical
default stage directory; explicitly redirecting it preserves the original clock
records and prevents a later provenance check from confusing a reproduction
clock with a historical original.
An analysis script's `--output-dir` changes its destination, not the location of
the retained studies; `--root` selects a separate project tree with the documented
input directory layout. Keep the delivered frozen directories intact.

The two additional blocks use flag modes. To regenerate the AP/threshold and
residual CSVs from retained A/B/C scores, use a new output **and** log directory:

```text
python -B scripts/analyze_boundary_sensitivity.py --freeze --output-dir results/reproduction_fresh/boundary_sensitivity --log-dir experiments/logs/reproduction_fresh/boundary_sensitivity
python -B scripts/analyze_boundary_sensitivity.py --run --output-dir results/reproduction_fresh/boundary_sensitivity --log-dir experiments/logs/reproduction_fresh/boundary_sensitivity
python -B scripts/analyze_boundary_sensitivity.py --verify --output-dir results/reproduction_fresh/boundary_sensitivity --log-dir experiments/logs/reproduction_fresh/boundary_sensitivity
```

This route does not fit or generate samples. Its default log directory is a
historical execution location, so redirecting only `--output-dir` is insufficient
for independent regeneration. The separate segmentation/fresh-fit route below
uses `--stage-dir`, not `--log-dir`. Both scripts' `--rebuild` modes verify and
rewrite their derived outputs from retained arrays without fitting; such a
rewrite changes manifest-covered files and is separate from read-only `--verify`.

## Fresh generation and fitting

The following are explicit, separate fitting routes into new directories.
Their existing repeat labels reproduce the frozen samples; they do not create
additional independent confirmation evidence.

Study A:

```text
python -B scripts/run_duration_confirmation.py --freeze-protocol results/reproduction_fresh/duration_A/protocol.json
python -B scripts/run_duration_confirmation.py --run-protocol results/reproduction_fresh/duration_A/protocol.json --output-dir results/reproduction_fresh/duration_A --resume
python -B tools/verify_duration_confirmation.py --output-dir results/reproduction_fresh/duration_A
```

The first `--resume` allows the protocol-only directory; it fits all planned
missing repeats. Later resumes reuse hash-valid repeats. Use the verifier,
rather than resume, when strictly read-only behaviour is required.

Study B:

```text
python -B scripts/run_schedule_robustness.py --freeze --output-dir results/reproduction_fresh/schedule_B
python -B scripts/run_schedule_robustness.py --run --output-dir results/reproduction_fresh/schedule_B
python -B tools/verify_schedule_robustness.py --output-dir results/reproduction_fresh/schedule_B
```

Study C:

```text
python -B scripts/run_symmetric_sensitivity.py --freeze --output-dir results/reproduction_fresh/symmetric_C --log-dir experiments/logs/reproduction_fresh/symmetric_C
python -B scripts/run_symmetric_sensitivity.py --run --output-dir results/reproduction_fresh/symmetric_C --log-dir experiments/logs/reproduction_fresh/symmetric_C
python -B scripts/run_symmetric_sensitivity.py --verify --output-dir results/reproduction_fresh/symmetric_C --log-dir experiments/logs/reproduction_fresh/symmetric_C
```

The C freeze records source/configuration and internal clock values without
simulating. Its run executes exactly 30 repeats × two durations × five window
settings. The same coefficient tensors, initial histories and innovations are
paired across settings; overlapping windows and fitted cells are not replicates.
`--aggregate` for B or C recalculates summaries from complete saved records
without fitting. Do not substitute these writing modes for read-only verification.

The additional PELT adapter and eight selected fresh-fit cases have one bounded
execution entry. All three commands use new output and execution-record paths:

```text
python -B scripts/run_segmentation_validation.py --freeze --output-dir results/reproduction_fresh/segmentation_validation --stage-dir experiments/logs/reproduction_fresh/segmentation_validation
python -B scripts/run_segmentation_validation.py --run --output-dir results/reproduction_fresh/segmentation_validation --stage-dir experiments/logs/reproduction_fresh/segmentation_validation
python -B scripts/run_segmentation_validation.py --verify --output-dir results/reproduction_fresh/segmentation_validation --stage-dir experiments/logs/reproduction_fresh/segmentation_validation
```

`--run` fits 60 PELT adapters/refit pipelines on retained C inputs and regenerates
and refits A seeds 50000–50001 and C seeds 70000–70001 at D=300/600, h=300,
stride=150. The latter eight cases draw their mechanism/noise inputs afresh from
the existing seed labels, rather than initialising from saved arrays. Their
comparison report contains 492 exact items; this does not establish fresh-fit
reproduction of the full experimental grid or cross-platform bit identity.
The supplied `--verify` rechecks saved comparisons without rerunning PELT,
simulators, Lasso or clustering. These two levels must not be conflated.

## Direct-metric interpretation and delivery

Temporal state AUROC ranks active/inactive windows for each changing edge.
Boundary diagnostics instead use actual support-change events in adjacent-window
centre slots. Positive-change AP treats zero change as abstention and processes
ties together; standard AP includes the final zero-strength group. Per edge,
AP_standard = AP_plus + boundary_prevalence * (1 - positive-strength_recall).
Both use the same eligible changing edges, retaining targets without both
positive and negative slots as missing. The original AP-plus and raw-q95 point
values are unchanged. New sensitivity intervals reuse one paired resampling
stream per cell/finite mask across methods, metrics and operating points, so
some interval endpoints differ slightly from the older independently seeded
tables without changing the underlying per-repeat measurements.

Boundary F1 and stable-edge variation use strict exceedance of a linear
quantile, either shared from raw changes or taken from each method's own changes.
All five fixed quantiles are reported; no test-label best-F1 choice is made.
Method-own quantiles are scale sensitivity, not independent equal-false-alarm
calibration. The older raw-q95 result is one declared amplitude operating point.
These are stride-resolution diagnostics, not sample-level boundary error or a
calibrated false-alarm guarantee. Always-present and always-absent edges are
reported separately. Missing targets are retained with denominators.

An installed Tectonic/LaTeX environment is needed only to compile the TeX PDFs;
supplied PDFs can be read without it. For example:

```text
tectonic -Z search-path=. --keep-logs paper/main.tex
tectonic -Z search-path=. --keep-logs paper/supplement.tex
```

`python -B scripts/build_information_sciences_package.py` verifies and exports
the current artifact as `information_sciences_reproducibility.zip`; it does not
publish it or overwrite the older anonymous archive. The export excludes the
working environment, Git history, unselected local execution logs and machine
paths. The selected normalized internal provenance records are explicit payloads. Its
declared author/correspondence are intentional. The outer manifest records scan
scope and binary-inspection limits. Project code is MIT; third-party data/code
retain their original notices (`LICENSE`, `docs/third_party_notices.md`).

Earlier dated audit records describe historical snapshots. The same archive
filename has been rebuilt during successive revisions; identify the copy in
hand using its actual outer manifest and file SHA, rather than an earlier audit's
ZIP hash. A local archive does not establish public deposition or journal upload.

## Historical records and release preparation

`docs/research_guide.md` links the dated evidence indices, responses and audits.
They are retained as version-specific provenance and use their original section
numbers. Normalized execution exports are under
`results/information_sciences/provenance/`, `reviewer_provenance/` and
`evaluation_provenance/`. No private original is needed to read the exported
records. Earlier audit reports describe their own snapshots, not the current
contents of a repeatedly rebuilt ZIP filename.

The public source repository is
[latent-state-graph-recovery](https://github.com/xschang1002/latent-state-graph-recovery).
Release [v2026.10.28-layout.1](https://github.com/xschang1002/latent-state-graph-recovery/releases/tag/v2026.10.28-layout.1)
provides the current presentation and complete named archive. The preceding
release, `v2026.10.28`, remains available with the same scientific inputs and
fitted results and its earlier presentation. No DOI has been assigned.
Submission metadata and the remaining journal-system steps are in
`submission_materials/information_sciences/README.md`. A GitHub release does not
constitute submission to the journal.
