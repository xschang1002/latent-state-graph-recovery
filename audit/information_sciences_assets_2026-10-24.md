# Information Sciences main-text assets, 2026-10-24

The new presentation builder is `tools/build_information_sciences_assets.py`.
It reads completed retained outputs and never fits a scorer or changes a frozen
protocol, source file, CSV, or historical revision2 asset. The input checksums
before and after export agree. All exact source and output SHA-256 values are
recorded in `results/information_sciences/assets_manifest.json`.

## Four tables and four figures

- `tab_protocol.tex`: earlier exploratory duration background and the distinct
  frozen A/B designs, with sample length, window count, repeat count and pairing.
- `tab_main_results.tex`: all six A aggregation controls at both durations,
  using the common local Lasso scores; pooled and changing-edge temporal AUROC.
- `tab_factors.tex`: two A cells and all six B cells, using forced candidates.
  Each row averages J, C and the per-repeat JC product on the same joint finite
  repeat set. Raw reference and forced-minus-raw use that identical set. No
  product of marginal means is used; no causal attribution is inferred.
- `tab_ess_compact.tex`: all eight method/expert-edge combinations, forward-fill
  AUROC and four-imputation sensitivity range. The range is not a confidence
  interval. Existing ESS outputs are unchanged.
- `fig_target_example`: a deterministic 4-edge, 8-centre illustration with state
  counts 6/2, true support rows persistent/state0-only/state1-only/never, and
  constant occupancy scores. Exact pooled AUROC is 0.9375 and changing-edge
  temporal AUROC is 0.5. This finite example is not an experimental estimate.
- `fig_confirmation`: all four frozen A primary contrasts, including the
  negative D300 result, paired duration difference and D600 matched control;
  n=100 repeats, pointwise 95% bootstrap intervals and Holm4-adjusted p values.
- `fig_schedule`: all six B cells, forced-minus-raw, mean centre-switch counts
  and mean within-repeat window purity. Nominal mean duration is labelled
  separately from realised run lengths. Counts and purity are descriptive.
- `fig_gate`: all-repeat native and forced effects, plus accepted/fitted counts.
  Selected accepted-only subgroup effects are not substituted for full-sample
  comparisons.

Derived source CSVs accompany the finite example, each inferential figure and
the factor table. Inference never treats edges, overlapping windows or exact
random partitions as independent repeats. Unavailable estimates or intervals
are not drawn at zero. The bootstrap is the frozen pointwise percentile scheme
with 10,000 repeat resamples and seed 20261024; these are not simultaneous
intervals.

## Export and visual review

All four figures were inspected as exported PNGs. A clipped panel title and
crowded repeated score labels in the initial finite-example layout were repaired
without altering its data. Native and forced output use colour and distinct
markers, and the zero-difference baseline remains visible. The finite example
has an explicit AUROC 0.5 reference.

Post-export PDF inspection confirmed a width of 6.6 inches for each figure,
embedded DejaVu Sans TrueType/CIDFontType2 fonts, and zero embedded raster image
objects. Nominal text is at least 8.5 pt. Opaque PNG exports have 600 dpi metadata,
with dimensions 3960x1740, 3960x1920, 3960x2460 and 3960x2460. The QA report is
`experiments/logs/information_sciences_assets_2026-10-24/qa.json`.

These are general publication exports, not a journal-compliance certification.
A live request to the official Information Sciences guide for authors on
2026-10-24 returned HTTP403; the manifest preserves that verification boundary.

## Software-workflow attribution

The scientific-visualization skill materially guided figure integrity,
repeat-aware uncertainty, redundant encodings, font embedding and export review.
Its required workflow reference is recorded here and in the asset manifest;
it is not used as evidence for a scientific conclusion in the main manuscript.

Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026).
*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*.
[https://doi.org/10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065).

Author names, year and current version were checked directly against the
[arXiv record](https://arxiv.org/abs/2609.00065) on 2026-10-24. The record showed
v2, last revised 2026-09-02, with the five authors listed above and no published
journal version. The reference intentionally uses the unversioned DOI.
