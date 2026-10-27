# Temporal aggregation across the existing grid — 2026-10-23

This is the historical grid-stage record. The current manuscript, validation
counts and package are documented in `audit/grouping_controls_2026-10-23.md`;
the underlying grid results below are unchanged.

The requested experiment is complete. Existing uncommitted work was preserved
in `experiments/logs/temporal_grid_2026-10-23/before/` (57 files, with hashes in
the adjacent `before_manifest.json`). No commit or staging change was made.
The original core temporal CSVs, ESS result/diagnostic CSVs and previous PDFs
remain unchanged. This record supersedes the preceding stage's planned-work
statement; that stage's report and hashes remain historical.

## Design, execution and estimands

- Existing grid: all 35 labelled entries in `configs/mechanism_grid.yaml`,
  seeds 0--19, fixed h=300/stride=150. Four duplicate configurations reuse the
  same observations: 31 distinct configurations, 620 configuration/seed fits.
- Five recorded methods: raw windows, native gated hard aggregation, GMM
  continuous soft emission, oracle window labels without refitting, and the
  historical GMM weighted-prototype/argmax-emission variant. Primary plots
  compare the first four, with forced hard as a gate diagnostic.
- Every method uses the same local Lasso scores within a fit. Delta_temporal
  is each method's changing-edge temporal AUROC minus that seed's raw AUROC.
  Hard rejection preserves the existing static window-max fallback. Forced
  hard changes only the gate decision, retaining the same candidate partition.
- GMM/oracle have no gate and execute in all fits in the focus cells. Summary
  subsets separately record each method's actual execution and the common hard
  accepted/rejected seeds. All comparisons in a selected subset retain that
  subset's own raw baseline. Selected-subgroup means are descriptive.
- Duration: D=150,300,600,900 at T=1200/M=7. Length: T=600,1200,3000 at
  D=300/M=3,7,19. The core appears in both scans, not as a new replication.
  This is not a crossed grid; changing M also changes T and switch count.
- Pointwise 95% percentile bootstrap intervals of paired seed differences,
  10,000 draws, seed 7. No CI below five evaluable seeds, no simultaneous
  coverage or confirmatory grid significance claim, no pooling across cells.
- No changing-edge label flips in 240 of the 700 labelled configuration/seed
  contexts, including support-preserving controls and rare-state conditions.
  Their target remains undefined, with explicit denominators, never replaced
  by the constant-predictor value 0.5.

The primary source/config protocol was written before execution. The runner
rejects a changed protocol in an existing output directory. Amendment 11 in
`analysis/statistical_plan.md` records the extension and its exploratory role.

## Duration results

All duration cells accept the hard partition in 20/20 seeds. Values below are
paired changes from raw, averaged over twenty seeds.

| h/D | M | Raw AUROC | Hard delta [pointwise 95% CI] | GMM continuous delta | Oracle delta |
|---|---:|---:|---:|---:|---:|
| 1/3 | 7 | 0.792303 | +0.016787 [-0.013475, +0.043059] | -0.046309 | +0.059707 |
| 1/2 | 7 | 0.767458 | +0.030690 [+0.004523, +0.056608] | -0.060574 | +0.071059 |
| 1 | 7 | 0.716298 | -0.067197 [-0.122818, -0.012695] | -0.108608 | +0.097348 |
| 2 | 7 | 0.516846 | -0.012453 [-0.021486, -0.002896] | -0.018514 | +0.028520 |

Hard aggregation has a positive gain at h/D=1/2, with a pointwise interval
above zero, rather than a loss throughout the grid. At h/D=1/3 the positive
estimate is less precise. The small h/D=2 change occurs with near-chance raw
scores; it must not be interpreted as accurate temporal recovery.

## Length and gate results

| M | T | Gate | Raw | Native hard delta | Forced hard delta | GMM continuous delta | Oracle delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 600 | 20/20 | 0.683850 | -0.016554 | -0.016554 | -0.029946 | +0.003927 |
| 7 | 1200 | 20/20 | 0.716298 | -0.067197 | -0.067197 | -0.108608 | +0.097348 |
| 19 | 3000 | 2/20 | 0.721755 | -0.202154 | -0.042415 | -0.102021 | +0.160192 |

At M=19, native hard has CI [-0.229121,-0.163524]; forced hard has
[-0.089337,+0.005534]. The 18 rejected outputs have temporal AUROC 0.5.
Compared with forcing identical candidates, fallback lowers the all-seed mean
delta by 0.159739. That separates the mechanical fallback contribution from
grouping performance; the native -0.202 should not all be called aggregation
loss.

On the two actually aggregated fits, raw is 0.741888 and hard is 0.696016,
giving mean delta -0.045871. Individual hard gains have opposite signs; no CI
is reported at n=2. On these same two seeds GMM/oracle deltas are -0.060447 /
+0.128991, but their own native execution subset still contains all twenty
seeds. The summary preserves the 18 rejected seeds' forced candidates too.

## What the GMM control does and does not show

The previous aggregation table used soft weights to estimate state maps, then
an argmax assignment for window emission. That historical variant is retained
and explicitly named. Continuous emission additionally mixes maps using each
window's responsibilities. Both use the existing diagonal two-component GMM,
three initialisations and simulation seed. Core temporal AUROC is 0.607690 for
both (pooled AUROC 0.775642); its continuous paired gain is -0.108608
[-0.155667,-0.060317]. Core Table 3 now reports both variants.

All 620 GMM fits converge without warnings. Every maximum responsibility is
above 0.99; the largest mean normalised responsibility entropy across fits is
2.147410e-7. This particular fit is almost hard, so its negative means do not
establish a failure of diffuse soft aggregation in general.

After observing saturation and tiny emission differences, a numerical diagnostic
was recorded before its own execution: replay six distinct focus configurations
(120 fits) and round raw and both GMM predictions to 12 decimal places. It is
post hoc and changes no primary prediction. All raw and argmax AUROCs are
unchanged. At M=19 continuous mean AUROC becomes 0.619162 from 0.619735;
argmax is 0.615555. The largest per-seed continuous change is 0.014337. Mean
continuous-minus-raw gains stay negative in every focus configuration. Tiny
ranking differences receive no interpretation as a diffuse-membership benefit.

## Artifacts and reproduction

```powershell
python scripts/run_temporal_grid.py
python scripts/diagnose_temporal_gmm.py
python tools/verify_temporal_grid.py
python tools/build_revision2_tables.py
python tools/verify_revision2_numbers.py
```

- Frozen protocol: `results/revision2/temporal_grid_protocol.json`.
- Detail: `temporal_grid_detail.csv`, 3,500 rows; summary:
  `temporal_grid_summary.csv`, 700 rows (four subsets per entry/method).
- Numerical check: `temporal_grid_gmm_numerics.csv`, 360 rows.
- Main Table 3 adds both GMM emissions; Table 5/Figure 5 report duration/length
  deltas and separate gate frequency/accepted means. Appendix D has Tables
  14--15 for accepted subgroups and numerical sensitivity.
- New figure: `results/figures/rev2_temporal_grid.pdf` (embedded-font vector)
  and `.png` (600 dpi, 3840 x 4020 pixels, opaque background). Both are included
  in the anonymous artifact. Its marker/line styles distinguish methods as
  well as color. CI, sample sizes, selected subgroups and categorical levels
  are explicit in the caption.
- Latest manuscript: `paper_preprint_2026-10-23_temporal_grid.pdf`, 34 pages,
  identical to packaged `paper/main.pdf`. The builder generates 16 tables and
  nine figure designs. ESS remains in Section 10, now Table 10/Figure 7;
  its numbers and diagnostics are unchanged. Its implementation appendix is E.

## Validation and remaining scope

- Full suite: 282 passed, three existing upstream SWIG deprecation warnings,
  34.69 seconds. Nine new tests cover soft emission, temporal estimands,
  selected/native execution sets, small/missing intervals, gate/candidate
  separation, the full frozen/historical evidence chain, generated tables,
  plotted inputs and vector-figure package inclusion.
- 216 manuscript number checks pass; static source validation has no errors.
- All 1,400 historical hard/oracle temporal grid rows and gate decisions
  agree to 1e-15; 60 core raw/hard/oracle rows and 20 old GMM pooled rows also
  agree. All duplicate contexts, deltas, subgroup denominators, missing values,
  counts and frozen source hashes pass independent verification.
- The 120 numerical-diagnostic replays reproduce the primary native GMM and
  raw values before rounding. The old core/hold-out and ESS CSVs remain
  byte-identical to the starting stage.
- Dataset manifest: nine files/100.7 MB match. Historical freeze: 49/49
  summary/detail pairs match with no write. ESS refit reconstruction still passes.
- Tectonic compiles with no overfull boxes or undefined references. Changed
  main and appendix pages were rendered and reviewed. Existing underfull text,
  BibTeX volume/number and Windows BBL rerun warnings persist with resolved
  references. New content does not float into the Conclusion.
- The anonymous package is rebuilt with all 1,400 specialist records. Every
  payload/manifest hash is compared with current bytes and its manuscript PDF
  with the delivered PDF; zero identity findings are required.
  Generic Matplotlib Creator/Producer fields in the new vector figure were
  removed to satisfy the existing metadata policy; a regression checks its
  inclusion and clean scan. Its final manuscript rendering is pixel-identical
  to the reviewed version.

The duration and length results are exploratory existing-grid associations.
They do not define a universal h/D threshold or isolate M from record length.
No new held-out simulation block or diffuse-membership estimator was introduced.
All work remains uncommitted and the user's existing staged deletion is retained.

## Workflow attribution and figure destination

The local paper-writing, scientific-visualization and PDF workflows were used.
Visualization workflow attribution is retained in this record: Timothy Kassis,
Vinayak Agarwal, Yuhuan He, Darshil Patel and Aubrey M. Brueckner (2026),
*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*,
[arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
The latest metadata were verified on 2026-10-23. This is workflow attribution,
not evidence for the scientific findings.

The [Machine Learning author guidance](https://link.springer.com/journal/10994/submission-guidelines)
was checked on the same date for embedded vector fonts, final-size lettering
and redundant visual encoding. The new plot is a draft manuscript artifact at
6.4 inches wide; this is not a claim of complete submission-format compliance.
