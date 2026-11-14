# Submission checklist navigation

The current target is Knowledge-Based Systems through Elsevier's transfer
service. Use [the KBS transfer checklist](../submission_materials/knowledge_based_systems/README.md)
and [the 29 October requirements record](knowledge_based_systems_requirements_2026-10-29.md).
The local delivery directory is `submission_package/knowledge_based_systems/`
(synchronized with public GitHub release `v2026.11.15-kbs`).

The text below is a preserved historical checklist, not a statement of the
current paper's anonymity, results, status or KBS requirements.

---

# Submission Checklist — Current Revision

Updated after the 2026-09-13 acceptance audit. Earlier PDF checks do not certify
this revision. Manual Overleaf compilation is intentionally pending.

2026-10-20 note: CIKM submission is no longer the target. The manuscript is now a
15-page preprint; a local Tectonic compile of the packaged source gives 15 pages,
zero overfull boxes, and zero undefined references/citations. PDF metadata shows
no author, institution, or local path. Remaining items below are retained as a
release checklist, not as CIKM compliance requirements.

## Source and Evidence

- [x] Anonymous author/institution placeholders are present in the source.
- [x] Method describes supplied-K acceptance; separate multi-K diagnostics are distinguished.
- [x] Real comparison discloses annotated K for all methods and label-sighted placement.
- [x] Nonsignificance is not claimed as equivalence; the historical unsupported speedup is withdrawn and the matched 5575x timing is reported separately.
- [x] cMLP pilot overlap is disclosed and post-pilot comparisons are saved.
- [x] Actual self-lag ablation and horizon-purged warning experiments are regenerated.
- [x] Historical random-window state diagnosis is labeled retrospective and optimistic.
- [x] Frozen specialist robustness/difficulty extension and separate matched timing completed (new seeds 100–119; 1,400/1,400 fits, zero failures).
- [x] GenAI Usage Disclosure is present before references.

## Reproduction and Source Delivery

- [x] Dependency entries include the added baseline dependencies.
- [x] Export rules include tools, pinned requirements and the acceptance-fix note.
- [x] Recursive validation checks included tables for unescaped math-only characters.
- [x] Current full tests and clean-environment baseline checks recorded.
- [x] Source-package and Overleaf ZIP closure, hashes and identity scan recorded.

## R10: Prior Review Resolution

Current assessment: **NOT PASSED — partially addressed.** Engineering validation
and accurate disclosure do not establish that the scientific concerns are resolved.
See [the item-by-item assessment](review_response_assessment_2026-09-13.md).

- [x] Map the meta-review and both reviews to current text, evidence and remaining gaps.
- [x] Complete the requested direct RPCMCI/MSVAR/neural-Granger comparisons.
- [x] Clarify workload assumptions, variant selection and section structure (R1 W2/W3).
- [x] Update related work and document measured, discussed and omitted comparator scope (R1 W1/R2 W3).
- [ ] Substantiate the contribution beyond narrow recovery gains (Meta/R2 W1/overall evaluations).
- [ ] Establish adequate real-data evidence for the claimed practical value (Meta/R2 W2).
- [x] Exercise actual gate branches, isolate scoring terms, and repair all-nonpositive state selection (R2 W3).
- [x] Complete fixed-origin real diagnosis and disclose negative results, class coverage and repeated-observation limits.
- [ ] Establish general reliability of the heuristic regime/count decisions (stationary and rare/noisy-state stress failures remain).
- [ ] All core prior-review concerns are adequately resolved on the evidence.

The direct comparisons are complete, but they do not establish superiority or
equivalence to Regime-PCMCI. Final PDF compilation does not close this criterion.

## Manual Final PDF Checks

- [ ] Upload the corrected Overleaf ZIP and compile using pdfLaTeX/BibTeX.
- [ ] No compile errors or unresolved references/citations.
- [ ] Page count and current submission-format requirements checked.
- [ ] Tables fit and remain readable; no material overfull boxes.
- [ ] Figures are readable at their final column widths.
- [ ] PDF opens, text is searchable, headers/page numbers are acceptable.
- [ ] PDF metadata and visible content preserve anonymity.

## Submission-System Actions

- [ ] Metadata and any required reviewer nomination completed manually.
- [ ] Current PDF and anonymous source package submitted through the chosen release channel.

The accompanying source package is the revision artifact. The manuscript no
longer represents an unverified older repository URL as the current revision.
