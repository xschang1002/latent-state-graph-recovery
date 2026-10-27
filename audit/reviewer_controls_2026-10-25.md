# Reviewer-directed controls and robustness audit — 25 October 2026

This revision implements the author's hypothetical review, with no claim that
it is an external review or editorial decision. Private source wording is not
exported. The original A/B results, inferential families, ESS analysis and prior
uncommitted work remain intact. No commit, public deposit or submission is made.

## Changes supported by completed evidence

The theoretical results remain diagnostic finite-score identities. The revision
adds practical controls rather than claiming a deeper general ranking theorem:
positive edgewise-affine score transformations preserve temporal AUROC, whereas
edge-specific offsets can alter pooled comparisons. An exact fallback equation
separates acceptance effects from information discarded on rejection. The new
main title says *Temporal Discrimination*. Boundary detection is evaluated as a
distinct target rather than inferred from changing-edge AUROC.

All fixed shrinkage/residual weights .25, .5, .75 and 1 are reported, with the
hard endpoint 0 explicitly marked as a duplicate. On Study A's Lasso fits,
shrinkage at .25 raises pooled AUROC to .803695/.843119 at D=300/600 while
retaining raw state AUROC .707406/.766983. All 672,000 positive-lambda edge
checks across the retained A/B and new C fits preserve order/ties; all 112,960
evaluable changing-edge checks preserve AUROC. These are implementation checks,
not independent experimental units. Residual weight .25 gives state-AUROC
gains +.010135/+.056593 in A, versus hard's -.068869/+.038082. No best weight
is selected and no new primary correction family is introduced.

Holding B's gate fixed and using raw on rejection gives D=600 gains
+.012738 [.003307,.024163], +.013003 [.003670,.024278] and
+.011734 [.003479,.021901] for fixed, random and irregular schedules.
The original constant-fallback losses near -.21 remain reported. Raw fallback
does not guarantee beneficial accepted candidates: several C cells still lose.

Study C internally freezes a separate symmetric-support construction before
running 30 new independent repeats (70000–70029). Both states have no self
support and Bernoulli-.18 nonself marginals, conditional retention .5 and
addition probability .109756; shared coefficients are identical. Common
lag-aware scaling preserves their equality and checks final state radii. These
generator changes are bundled, not a factorial identification of each nuisance.
At T=3600, two durations and five fixed window/stride settings give 300 planned
fits, all successful, with no replacement or extension. Maximum final state
radius is .672912; no arbitrary-switching stability claim is made. Original
generator/source bytes are unchanged.

At h/stride=300/150, C's forced-minus-raw state-AUROC means are -.047 and
+.037 at D=300/600. Across the other windows, D=600 gains include -.014
at 300/300 with an interval crossing zero. Thus the new evidence extends
conditions while rejecting a universal duration/window rule. All ten cells,
gate denominators and new controls are retained, including adverse results.

Direct temporal analysis ranks true support-change slots between adjacent
window centres and reports positive-change AP with zero-strength abstention.
Boundary-slot F1 and stable-present/stable-absent score exceedances share a
label-free raw-derived 95th-percentile amplitude reference per fit. This is
stride-resolution detection, not subwindow boundary error or calibrated binary
support recovery. In A at D=600, boundary AP improves but F1's interval crosses
zero. In B at D=600, all three fixed-amplitude F1 effects are negative despite
positive state AUROC. C also shows metric-dependent effects. Its D=300/stride300
grid has no negative boundary slots, so 450 method-case AP values remain
missing with coverage, rather than receiving artificial zero/one scores.

## Traceability and transparent corrections

The new index `docs/reviewer_controls_evidence_index_2026-10-25.md` locates all
five analysis blocks, protocols, seeds, retained arrays, complete summaries and
replay routes, with Supplement S17/S18 and Tables S24–S28. The companion local
response explains how each hypothetical concern is handled and what remains
limited. ESS remains a static-target mismatch example.

Seventeen exported internal execution/QA records plus their manifest reside in
`results/information_sciences/reviewer_provenance/`. JSON normalization/path
redaction is explicit, historical code snapshots retain their bytes as text,
and original C freeze/run records remain in its 790-file seal. Internal clocks,
post hoc analyses, file metadata and external preregistration are distinguished.
There is no external clock attestation or independent human-replication claim.

Two analysis-only QA defects were corrected transparently: undefined NaN values
were wrongly included in the initial affine-event listing; a successful C
record's null error field serialized as the string `None`. Original versions,
protocol/source/output digests and correction records are retained. The first
fix leaves four numerical CSVs and input seal byte-identical; the second leaves
both summary CSVs byte-identical and every numeric detail/edge array unchanged.
Neither changes samples, parameters, metric definitions or the original studies.
The final scientific crosscheck also restricts the abstract's residual-loss
statement explicitly to Study A, retaining the adverse C window cells.

## Validation and delivery boundary

The final full-suite result, exact source/PDF hashes and computational reports
are in the companion JSON. The combined read-only gate checks the original
evidence and all five new blocks, numerical summaries/intervals, output seals,
path safety, missing-target coverage, 17 exported provenance records and new
figure/table sources. It recomputes metrics from retained arrays without fitting;
saved-input recurrence replay is distinguished from new stochastic experiments.
All 242 CSVs in the preceding final archive, 1,915 A/B files and 12 frozen
sources remain unchanged.

The compiled main article has 31 double-spaced pages, six figures, five tables
and a 188-word abstract; the separate supplement has 42 pages. Both have zero
overfull boxes or undefined citations/references and embedded fonts. All pages
were rendered and inspected, with individual checks of the new forests and
Tables S26–S28. Nonfatal legacy compiler notices and three existing SWIG
deprecation warnings are recorded rather than described as absent.

The tested v2 archive contains 5,894 payloads plus its manifest,
with actual protocols, inputs, fitted outputs, numerical analyses and source.
Two fresh extractions passed verification (176.258 seconds)
and rebuilding (215.971 seconds) in the isolated
47-pin Python environment. Rebuilding preserves all 33 generated tables,
all 68 delivered table files, every result CSV, original A/B and C scientific
guards and provenance exports. The seven current/retained main figure PDFs
keep their bytes; any changed legacy PDF retains the same drawing streams
and page boxes. No new fitting or stochastic experiment is introduced by
either route. The interrupted earlier v1 check is retained as an incomplete
attempt, not represented as a pass.

The final export changes only this completed audit and its JSON from the
tested v2 candidate. Its ZIP hash and exact current-byte/candidate-difference
comparison are kept outside its own contents to avoid self-reference. Public
repository/DOI deposition and author submission review remain outside this
local preparation.

Scientific-writing/visualization workflow attribution is retained in the software
records: Kassis et al. (2026), *Scientific Agent Skills*,
[DOI:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
The current arXiv record was checked on 25 October 2026. It is not research
evidence for the manuscript. The live journal-guide request returned HTTP 403;
the prior official-guide snapshot remains the provisional formatting basis,
without a new claim of formal submission readiness.
