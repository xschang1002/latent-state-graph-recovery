# Evaluation sensitivity and reproduction audit — 25 October 2026

This is the current local revision audit, not a response to a received editorial decision or a public preregistration. The author's hypothetical comments were assessed before the bounded additions; see `analysis/evaluation_revision_plan_2026-10-25.md`. No commit, submission, public upload or repository identifier was created.

## Four prior-review fixes

The README's A/B and C direct-analysis routes now isolate new execution stages. Boundary analysis and full C generation also have explicit fresh log destinations. Original clocks and source locks remain intact. Regression tests exercise the actual command routes and reject reuse of historical clock paths. Study C exceptions are described as point estimates with intervals crossing zero, not proven population reversals. S18 coefficient blocks now use B, distinct from support A, and the malformed T-AUC hyphen is corrected.

## Assessment and substantive changes

The comments about increment, scope, threshold scale, zero-score AP and residual ties are supported. Existing explicit qualifications remain important: a task identity is not independent mechanism evidence, the prototype gate does not represent all aggregators, a local archive is not proof of public availability, and same-seed reruns are not independent confirmation.

The article now separates established ranking properties, task-specific graph diagnostic accounting and empirical findings. It foregrounds a reusable fixed-input evaluation protocol and examples where evaluation choices change the assessment of the same fitted output. ESS remains a static expert-graph illustration in the supplement. Its byline and targeted metadata issues are corrected; the metadata audit covers 50 original entries and 24 DOI identities, plus two separately verified official documentation entries, not a full-text audit of every cited paper.

The new retained-input analysis covers all 1,000 A/B/C scorer/cell cases, not 1,000 independent seeds: 15,000 AP rows, 150,000 threshold rows and 6,000 residual rows. It preserves standard AP and AP-plus, all five fixed quantiles under raw-shared and method-own references, and every fixed residual weight including 0 and 1e-8. Missing target classes remain missing. The AP identity and same/cross-group pair reconstruction each agree within 1.11e-16. No new fitting, simulation or test-truth threshold/weight selection occurs in this block.

These checks change the boundary interpretation: B D=600 standard AP favours hard output across all schedules, while the corresponding AP-plus intervals include zero. A common raw-q95 F1 disadvantage can become a method-own-q95 advantage. Method-own quantiles diagnose scale sensitivity; they are not independent equal-false-alarm calibration. The earlier raw-q95 observations cannot alone establish loss of boundary locations.

Residual benefits can jump immediately above zero. In A's Lasso cells, the near-zero AUROC gains of about 0.068 (D=300) and 0.025 (D=600) are entirely within-group tie resolution; cross-group pair credit does not change. The near-zero setting is a diagnostic, not a recommended weight selected from performance.

A distinct penalized PELT construction plus our non-overlapping segment VAR-Lasso adapter supplies 60 detector cases and 644 segment refits, with no failures. It has positive pooled differences in both durations but no detected temporal state-AUROC advantage; this does not establish equivalence or inferiority. Forty-five cases reach the 11-segment cap, disclosed without retuning. It still shares the local score/input family and is not a broad external causal-discovery benchmark.

Eight selected A/C same-seed pipelines were generated and fitted afresh in the pinned isolated environment: all 492 recorded array/scalar comparisons are exact. This is a bounded same-machine reproduction check, not new independent seeds, an entire-grid fresh run or a cross-platform guarantee. Duration comparisons also change switch count and remain configuration-specific.

## Preservation and validation

All 264 earlier result CSVs, 1915 A/B files, 790 C files and 1030 previous-analysis files remain unchanged. Their 12 A/B and 10 C source locks remain valid. The original staged deletion and Git HEAD are preserved. The anonymous archive and both dated controls PDFs are unchanged; new dated sensitivity PDFs are separate outputs.

The full regression suite passed **495 tests** (three existing SWIG deprecation warnings). The 80 added tests cover path isolation, both scientific additions and the fail-closed sensitivity verifier. Complete retained-input boundary reconstruction reproduced all 1,013 files byte for byte. The new cached replay passed with 4,958 checked closure files unchanged and fitting/simulation/writing blocked.

Main PDF: 34 pages. Supplement: 48 pages. All 82 current pages were rendered and inspected; new figures/tables were also checked at detail scale. All fonts are embedded, with no overfull boxes, undefined citations/references, clipped or overlapping content. Nonfatal renderer/font-cache and underfull messages are retained in local logs. The main conclusion is uninterrupted and supplement references follow S20.

Clean archive validation status: **passed**. {'all_CSV_byte_identical': 284, 'all_result_CSV_byte_identical': 278, 'all_six_fresh_direct_commands_exit0': True, 'all_tabular_tex_byte_identical': 75, 'followup_unified_verify_exit_code': 0, 'generated_tables_byte_identical': 37, 'new_fits': 0, 'new_simulations': 0, 'payload_count': 7104, 'pins_matched': 47, 'prior_result_CSV_byte_identical': 264, 'rebuild_changed_payloads': ['results/figures/rev2_temporal_grid.pdf'], 'rebuild_exit_code': 0, 'status': 'passed', 'verify_exit_code': 0}

Current protocol, results, exports, commands and limitations are indexed in `docs/evaluation_revision_evidence_index_2026-10-25.md`; the itemized assessment/revision record is `docs/evaluation_revision_response_2026-10-25.md`. Portable provenance exports are included, and private machine paths are unnecessary for package verification. Earlier same-named ZIP audits describe historical snapshots. Use the current outer manifest and archive SHA to identify the artifact in hand; this audit deliberately does not embed its enclosing ZIP's self-referential hash.

Workflow attribution: Kassis et al. (2026), DOI 10.48550/arXiv.2609.00065, documents the scientific-agent workflow; it is not evidence for the paper's research claims. Machine-readable scope and hashes are in the companion JSON. Public deposition remains author-controlled and pending.
