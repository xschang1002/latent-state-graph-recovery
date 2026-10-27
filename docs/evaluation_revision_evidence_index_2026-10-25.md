# Evaluation revision: evidence and reproduction index

This is the 25 October 2026 sensitivity-stage index for *State Aggregation in
Dynamic Graph Recovery: Separating Structural Ranking from Temporal
Discrimination*. The new analyses respond to author-supplied hypothetical
comments. They are post hoc, descriptive additions, not a received editorial
decision, a public preregistration or new tests in the original A/B families.
The preceding A/B, Study C, AP-plus, raw-q95 and ESS per-repeat values are retained.

## Evidence and scope

| Block | Retained evidence | Interpretation and limit |
|---|---|---|
| Assessment before new output | `analysis/evaluation_revision_plan_2026-10-25.md` | Separates justified criticisms, existing evidence and a bounded new protocol. Internal source/clock records document local order, not independent time attestation or public preregistration. |
| Metric sensitivity | `results/information_sciences/boundary_sensitivity/{protocol.json,input_manifest.json,manifest.json,diagnostics.json}` | All 1,000 retained cases: A 400 scorer/cell cases, B 300 and C 300. No new fits or simulations. |
| AP conventions | `ap_detail.csv`, `ap_summary.csv`, `ap_paired_vs_raw.csv` in that directory | 15,000 method rows retain both positive-change AP and standard AP, with identical eligible-target rules. Per edge, `AP_standard = AP_plus + prevalence * (1 - positive_strength_recall)`. The maximum retained identity discrepancy is 1.11e-16. |
| Threshold sensitivity | `threshold_detail.csv`, `threshold_summary.csv`, `threshold_paired_vs_raw.csv` | 150,000 method/operating-point rows. Every quantile in {0.80, 0.90, 0.95, 0.975, 0.99} is reported under raw-shared and method-own references. Strict exceedance of a linear quantile is used. No test-label best-F1 selection or equal-false-alarm calibration is claimed. |
| Residual pairs | `residual_detail.csv`, `residual_summary.csv`, `residual_paired_vs_hard.csv` | 6,000 rows at lambda in {0, 1e-8, 0.25, 0.5, 0.75, 1}. Same-candidate-group and cross-group positive/negative comparison contributions explain the AUROC difference from hard emission. The maximum reconstruction discrepancy is 1.11e-16. A tiny positive weight is a tie-breaking diagnostic. |
| Different framework | `results/information_sciences/segmentation_validation/{protocol.json,manifest.json,diagnostics.json,detail.csv,summary.csv,contrasts.csv}` | PELT plus our non-overlapping segment VAR-Lasso adapter: 30 C repeats at D=300/600 on the h=300, stride=150 grid. Sixty detector fits, 644 segment refits, zero retained adapter failures. This is not a broad causal-discovery benchmark. |
| Fresh fitting | `segmentation_validation/fresh/`, `fresh_comparisons.csv` | A seeds 50000–50001 and C seeds 70000–70001, each at D=300/600: eight same-seed input-generation and fitting reruns. All 492 recorded array/scalar comparisons are exact in the recorded environment. These are not new independent seeds, a full-grid fresh run or a cross-platform identity guarantee. |
| Displays | `results/information_sciences/evaluation_assets/` | Three figures, four tables, three result snippets, `display_source.csv` and an input/source/output manifest. The complete main article has six figures and five tables. |
| Portable provenance | `results/information_sciences/evaluation_provenance/export_manifest.json` | Five normalized exports: boundary freeze, run, rebuild and preservation records, and the segmentation isolated-environment record. The manifest records both original and exported hashes; private originals are not required by a clean package. |

Within the boundary block, absent positive/negative target classes remain missing,
not zero. Per-repeat measurements precede means and paired differences; windows,
edges, fitted cells and method rows are not independent repeats. Pointwise 95%
intervals use 10,000 paired-repeat bootstrap draws with seed 20261025. One common
resampling matrix per cell/finite mask is reused across metrics, methods and
operating points. Consequently some interval endpoints differ slightly from the
earlier independently seeded tables, while their underlying AP-plus/raw-q95
measurements are unchanged. These intervals introduce no new significance family.

The external adapter uses penalized PELT (`l2`, penalty factor 1, `jump=1`,
`min_size=2`) without known segment count, recurrent k-means grouping or the
prototype gate. Forty-five of its sixty cases choose 11 segments, the
minimum-length cap at 23 windows. Its frozen setting is not retuned after this
observation. It shares the local Lasso scorer and simulated inputs with C, so it
tests a different partition/refit construction, not all scoring/model families.

## Manuscript locators and claim boundary

Supplement S19 presents AP/threshold sensitivity and residual pair accounting;
S20 presents the adapter and selected fresh-fit comparisons. The main PELT table
replaces the earlier main ESS table. The unchanged ESS numerical results remain
in Supplement Table S11, with static expert-truth and finite-chain qualifications.
The main article's mathematical contribution is task-specific diagnostic
accounting built on known ranking principles. Its empirical contribution is the
controlled change in method assessment under different targets and operating
rules. The gate remains a prototype-specific implementation case.

At D=600 the B standard-AP hard-minus-raw means are positive in all three
schedules, whereas the earlier positive-change AP intervals include zero. The
raw-shared and method-own threshold results can have different signs. Thus the
former raw-q95 F1 result cannot alone establish loss of boundary locations.
Residual AUROC can rise immediately above lambda zero through resolution of
within-group ties; that does not show improved state assignment. The PELT adapter
has positive pooled-ranking differences but no detected temporal state-AUROC
advantage in either duration; absence of a detected advantage is not equivalence.
Duration comparisons also change switch count at fixed series length and remain
configuration-specific.

## Source verification

`docs/evaluation_sensitivity_literature_2026-10-25.md` records the public-source
assessment. Its initial metadata audit covers all 50 starting bibliography
entries and all 24 starting DOI identities, not full-text reading of every work.
The two subsequently added official documentation entries (ruptures PELT and
scikit-learn AP) were separately inspected, yielding 52 current entries. Detailed
field/access limits and raw retrieval records are retained. No unpublished text
or sensor data was sent to a public search service.

Applied targeted fixes cover the NAVAR chapter DOI/pages, Tank issue year, UCI
formal citation years, the Chang byline surname/consortia and Howes's 2026 final
journal version. The ESS byline and the two documentation entries remain intact.
Unverified terminal-page fields are explicitly excluded from the audit claim.

## Reproduction routes

The package-root README is sourced from
`docs/information_sciences_artifact_readme.md`. It distinguishes:

1. `python -B scripts/reproduce_paper.py --verify`: read-only retained-output
   checks, including `tools/verify_evaluation_sensitivity.py`. Neither route fits.
2. `python -B scripts/reproduce_paper.py --rebuild`: display rebuilding from
   retained outputs; this is not fresh fitting. The new display builder is
   `tools/build_evaluation_sensitivity_assets.py`.
3. Analysis regeneration: the boundary script's `--freeze`/`--run` routes use
   both a new `--output-dir` and a new `--log-dir`; direct-metric positional
   routes use their own new `--stage-dir`. Keep historical clocks intact.
4. Selected fresh fitting: the segmentation script's `--freeze`/`--run` routes
   use a new `--output-dir` and `--stage-dir`. This repeats its fixed 60 adapter
   cases and eight selected fresh-fit comparisons. Study C's separate full-fit
   route also explicitly redirects its `--log-dir`.

All example destinations are under `results/reproduction_fresh/` and
`experiments/logs/reproduction_fresh/`. A writing/rebuild route changes
manifest-covered files; audit an unchanged extraction separately. Earlier audit
reports describe historical snapshots: the same ZIP filename has been rebuilt
more than once, so use the actual outer manifest/SHA for the artifact in hand.

The current root PDF names end in
`2026-10-25_information_sciences_sensitivity.pdf`; package counterparts remain
`paper/main.pdf` and `paper/supplement.pdf`. The archive and metadata are local
preparation artifacts. They do not establish journal-system upload or public
deposition. No repository DOI or public release URL is assigned.
