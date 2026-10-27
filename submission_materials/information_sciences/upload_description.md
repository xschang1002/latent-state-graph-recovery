# Research artifact description

This artifact accompanies *State Aggregation in Dynamic Graph Recovery:
Separating Structural Ranking from Temporal Discrimination*, by Xuansong Chang.
It supports a study of when sharing local graph estimates helps recover edges
at the times they are active. The evaluation separates pooled structural ranking,
within-edge state discrimination and adjacent-window boundary detection.

The artifact includes complete simulation inputs and fitted outputs for Study A
(100 independent paired repeats), Study B (50 different repeats across six
schedule/duration cells) and the separately specified Study C (30 new independent
repeats across two durations and five window/stride settings). Their original
protocols and inferential scope are preserved. Post hoc controls reuse these
scores at every fixed mixing weight, and retain both standard/positive-change AP,
five fixed threshold quantiles under two amplitude references, stable-edge rates
and near-zero residual comparison accounting.

A PELT plus non-overlapping segment VAR-Lasso adapter provides a different
partition/refit construction on 60 C inputs, with 644 refits. Its 45 cases at the
11-segment minimum-length cap remain disclosed. Eight selected original-seed A/C
pipelines were generated and fitted afresh; all 492 recorded comparisons are
exact in the recorded environment. This is a bounded computational reproduction
check. Auxiliary exploratory results and ESS static expert alignment retain
their separate scope.

The main article, separate supplement, code, protocols, derived figures/tables,
per-repeat records and reproduction entry points are included. The README
separates read-only verification, rebuilding displays, regenerating analysis
outputs and fresh generation/fitting. The current scientific map is
`docs/research_guide.md`. Dated indices, execution records and audits remain
available as historical provenance, with their original section numbering.
`STCG` remains a software/package identifier. Simulation arrays are synthetic;
third-party data acquisition conditions and notices are retained.

Metadata version: v2026.10.27. Current PDF release assets end in
`2026-10-27_information_sciences_refined.pdf`; package PDFs are `paper/main.pdf`
and `paper/supplement.pdf`. Identify any delivered ZIP by its actual outer
manifest and file hash. Deposition metadata and remaining release steps are in
`submission_materials/information_sciences/README.md`. Source code and the full
archive are available at
https://github.com/xschang1002/latent-state-graph-recovery/releases/tag/v2026.10.27.
No DOI has been assigned. Project code is MIT licensed; third-party components
retain their own notices and conditions.
