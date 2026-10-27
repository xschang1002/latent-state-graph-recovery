# Response to the author's hypothetical evaluation comments

This local revision record paraphrases author-supplied hypothetical comments;
it is not a response to an actual journal decision. The initial assessment and
bounded plan preceded the new numerical outputs in
`analysis/evaluation_revision_plan_2026-10-25.md`. Internal freeze records are
not public preregistration. Exact inputs, outputs and reproduction locators are
in `docs/evaluation_revision_evidence_index_2026-10-25.md`.

## 1. Identify the increment beyond known ranking theory

**Assessment.** Correct. Optimal posterior ordering is established, and earlier
sliding-window/connectivity studies already evaluated window length, offset and
state recovery. A general optimal-ranking or first-window-sensitivity claim
would overstate the increment. Algebraic reconstruction is also not independent
evidence of a mechanism.

**Revision.** The article separates known ranking principles, graph-target
diagnostic identities and new empirical evidence. Its three contributions concern
the evaluation separation, controlled assessment of aggregation and the diagnosis
of when a particular output gains or loses under that protocol. The occupancy
ceiling is a task-specific application of classical ranking theory. Literature
coverage distinguishes post hoc local-network grouping from joint estimation.
The new AP and threshold results demonstrate that the evaluation choice can
change conclusions about the same fitted outputs.

## 2. Broaden the empirical construction and reduce the gate's centrality

**Assessment.** Correct as a limit on generality. A two-cluster research prototype
and its gate cannot support a universal claim about aggregation algorithms.

**Revision and evidence.** A distinct penalized PELT construction, followed by our
non-overlapping segment VAR-Lasso refit adapter, is evaluated on sixty existing C
inputs. It uses no recurrent k-means partition or prototype gate. It retains 644
segment refits, with no failed cases. Forty-five cases reach the 11-segment
minimum-length cap; this is disclosed rather than hidden by retuning. Pooled
ranking improves relative to raw in both durations, but neither duration has a
detected temporal state-AUROC advantage. This is limited evidence from one
additional partition/refit framework sharing the same local scorer and input
family, not validation across arbitrary causal-discovery methods. Gate geometry
and native fallback remain implementation-specific cases.

## 3. Correct the AP terminology and test operating-point sensitivity

**Assessment.** Correct. Raw-q95 exceedance is amplitude-sensitive, while ordinary
AP includes the final zero-strength tie group that the original abstention
statistic excludes. Renaming the old number would not fix the distinction.

**Revision and evidence.** Both `AP_plus` and standard AP are retained on identical
eligible targets, with the per-edge identity
`AP_standard = AP_plus + prevalence * (1 - positive_strength_recall)`.
All 1,000 saved A/B/C cases are analyzed. Five fixed quantiles are reported under
both raw-shared and method-own references; no test-set optimal F1 is selected.
The B D=600 standard-AP differences favour hard output in all three schedules,
while the corresponding positive-change AP intervals include zero. Threshold
reference changes can also reverse the F1 comparison. The earlier raw-q95 point
values are preserved and described as a declared amplitude operating point,
rather than proof of missing boundary locations. Method-own quantiles are a
scale-sensitivity check, not independent false-alarm calibration.

## 4. Separate replay, fresh fitting and actual public availability

**Assessment.** Correct. A local ZIP and cached replay cannot establish a fresh
fit or the existence of a public repository/journal attachment.

**Revision and evidence.** Eight selected same-seed A/C configurations are
generated and fitted afresh at the two durations. All 492 recorded array/scalar
comparisons are exact in the recorded environment. This is a bounded check of
those configurations, not a fresh run of the full grid, an independent-seed
confirmation or a portability guarantee. The README gives separate read-only,
display-rebuild, analysis-regeneration and fresh-fit routes. New output and
execution-log paths preserve original clocks. No public DOI, release URL or
journal-system upload is asserted; deposition remains author-controlled.

## 5. Explain the near-zero residual AUROC jump

**Assessment.** Correct. Any positive residual can resolve within-hard-group ties,
so a discontinuity at zero need not reflect better grouping or a large signal.

**Revision and evidence.** The fixed residual grid now includes 0 and 1e-8, with
same-group and cross-group comparison-pair contributions. In the original A
Lasso cells, the mean AUROC increases of approximately 0.068 (D=300) and 0.025
(D=600) immediately above zero are entirely within-group tie-resolution effects;
cross-group credit is unchanged. The decomposition is an exact diagnostic of
the retained arrays, not independent mechanism evidence or a proposal to select
1e-8 from the test results. Finite float64 tie diagnostics are kept.

## 6. Align wording, references and evidence levels

**Assessment.** The requested author corrections and terminology/scope changes
are supported. At fixed total series length, changing duration also changes the
number of switches, so it does not identify a universal duration effect.

**Revision.** Temporal state discrimination is distinguished from adjacent-window
boundary-slot detection. Boundary claims are limited to stride resolution. ESS
retains its static-truth and finite-chain limitations, and its detailed numerical
results move to the supplement while the main table presents the PELT comparison.
The ESS authors are corrected to Søren Wengel Mogensen, Karin Rathsman and Per
Nilsson. A metadata-level audit covers the fifty starting references and their
twenty-four DOI identities; the two added official implementation/AP references
are separately checked. Targeted bibliographic corrections and unresolved field
limits are recorded in the public-source audit. Detailed development history
remains in provenance records rather than being treated as scientific evidence.

The original A/B hypothesis families, C fitted outputs, AP-plus/raw-q95 per-repeat
measurements and ESS numerical results are unchanged. New sensitivity intervals
use a shared cell/finite-mask bootstrap stream, explaining minor interval-endpoint
differences from the earlier independently seeded summaries. New analyses are
descriptive and post hoc. Final scientific approval and submission remain the
author's responsibility; no publication or submission action is performed here.
