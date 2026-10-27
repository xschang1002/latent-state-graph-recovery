# Matched grouping controls and manuscript consolidation — 2026-10-23

This is the historical grouping-control stage. Its numerical results are
unchanged; the current text, information sheet and independent-environment
package checks are recorded in `audit/mlj_final_text_2026-10-23.md`.

This stage completes the four requested items while retaining exploratory
positive-gain wording. No new independent seed block was run. Existing
uncommitted work was snapshotted in
`experiments/logs/grouping_controls_2026-10-23/before/` (75 files with hashes).
No commit or staging operation was made; the previously staged review-file
deletion remains untouched. The previous temporal-grid PDF remains available.

## Exact matched random control

The selected cell is D=600, h=300, T=1200, M=7, seeds 0–19. For each learned
partition, enumerate every unlabelled bipartition preserving its group sizes:
21 or 35 partitions per seed, 616 in total. Recompute group maps and temporal
AUROC for each partition, then average the AUROCs within seed. Averaging
predictions before calculating AUROC is a different estimand and is covered by
a regression test. The twenty simulation seeds, not the 616 partitions, are
the independent units.

Mean temporal AUROCs are raw 0.767458, hard 0.798148 and matched random 0.576591.
The three paired comparisons use 10,000 bootstrap resamples, seed 7:

| Contrast | Mean | Pointwise 95% percentile interval |
|---|---:|---:|
| Hard − raw | +0.030690 | [+0.004523, +0.056608] |
| Hard − matched random | +0.221557 | [+0.192541, +0.250025] |
| Matched random − raw | −0.190866 | [−0.203569, −0.178838] |

These comparisons reuse the selected grid cell and original seeds. They support
added value of the learned grouping relative to this matched control at this
cell; they do not independently confirm a positive-gain duration threshold.
There is no new multiplicity-adjusted confirmatory family.

## Three-condition factor diagnostics

The original core decomposition is reused directly. The other two cells' maps
and labels were not previously saved, so forty existing fits were reconstructed
deterministically, using the original configurations and seeds. Every raw and
candidate AUROC and gate decision agrees with the original grid within 1e-15.
Numeric intermediates are retained in `grouping_replay_outputs.npz`, with
source/cache hashes; subsequent default runs read this cache without fitting.

| D | M | Hard output | E[J] | E[C_edge] | E[J C_edge] | Hard AUROC | Raw AUROC |
|---:|---:|---|---:|---:|---:|---:|---:|
| 300 | 7 | Native | 0.533333 | 0.414317 | 0.298202 | 0.649101 | 0.716298 |
| 600 | 7 | Native | 0.866667 | 0.685602 | 0.596296 | 0.798148 | 0.767458 |
| 300 | 19 | Forced, all seeds | 0.536667 | 0.581496 | 0.358680 | 0.679340 | 0.721755 |

The maximum per-seed identity error for AUC = 1/2 + J*C_edge/2 is 2.22e-16.
Mean products recover mean AUROC; multiplying marginal means does not. Both
factors are estimated jointly, so marginal differences are descriptive rather
than separate causal explanations. The M=19 forced diagnostic excludes the
native static fallback and includes all twenty candidates.

The existing-grid coverage table lists all 35 labelled configurations: 23 have
20/20 evaluable seeds, eleven have no changing support, and one has changing
support but no observed window-centre flip. Shared configuration IDs and table
annotations identify four duplicate entries. These are coverage contexts, not
additional independent replications.

## Manuscript and reproduction changes

The abstract is approximately 200 words (201 after merging PDF line-end
hyphenation). Three contributions now cover evaluation targets, conditional
aggregation gains and mechanism diagnostics. Positive comparisons are called
exploratory in the abstract, contributions, results and conclusion. Statements
about fallback versus grouping loss now refer to candidate performance; fitted
K-means is described as within 0.014 of the enumerated optimum. The conclusion
uses the mean per-seed product. The ESS repair history is shortened in the main
text, with full provenance retained in its appendix. No ESS experiment or
result/diagnostic CSV was changed.

Table 6 reports the three-condition decomposition; Appendix D adds the three
matched contrasts and complete target coverage. Table 5/Figure 5 retain the
duration/length comparison. The builder produces 19 tables and nine figure
designs. The regenerated vector temporal figure has the same plotted content;
PDF build metadata can differ. Original temporal-grid outputs, historical core
CSV inputs, source data and historical freeze inputs remain unchanged.

The new single entry is:

```powershell
python scripts/reproduce_paper.py --verify
python scripts/reproduce_paper.py --rebuild
python scripts/reproduce_paper.py --rebuild --compile
```

`--verify` is read-only and checks every exported payload hash, numerical claims,
grid provenance, grouping outputs and manuscript structure. `--rebuild` uses
retained outputs rather than refitting, then regenerates tables/figures and
rechecks results. Compilation requires Tectonic or latexmk on PATH. A fresh
directory with `run_grouping_controls.py --refit` reconstructs the same forty
old fits if desired. The root archive README distinguishes included ESS inputs
from separately acquired older Hydraulic/AReM data. Checksums apply to the
delivered bytes before any rebuild.

## Validation and deliverables

- Full regression suite: 292 passed; three existing third-party SWIG
  deprecation warnings, no failures.
- Manuscript numerical checks: 231 passed. Static source validation: no errors.
- Read-only grouping replay checks: 60 factor rows, 616 partition AUROCs,
  twenty seed summaries, 35 coverage entries, source/cache hashes and exact
  factor identities.
- Original grid checks: 1,400 historical hard/oracle rows, 60 core rows and
  twenty historical GMM pooled rows still agree. No new grid fitting.
- Historical freeze: 49/49 pairs agree; data manifest: 9/9 files agree.
- Tectonic PDF: 35 pages, zero overfull boxes or undefined references; all 21
  fonts embedded. Existing underfull reproduction lines and the prior
  volume/number BibTeX warning remain; compilation exits successfully.
- The extracted package's single-entry read-only run passed, with every payload
  unchanged afterward. Its rebuild run also passed: all 19 regenerated tables
  and seven grouping analysis/protocol outputs were byte-identical to delivery.
  The final package manifest records its
  exact payload count/checksums, complete 1,400-record specialist checkpoint
  and identity-scan findings.

Current manuscript: `paper_preprint_2026-10-23_grouping_controls.pdf`, identical
to packaged `paper/main.pdf`. Anonymous bundle:
`submission_package/stcg_anonymous_artifact.zip`. Its internal and external
manifests, and `stcg_anonymous_artifact_identity_scan.json`, record the exact
export. The accompanying JSON records source/output hashes without attempting
to include the enclosing ZIP's own hash. Final ZIP hashes and smoke-test logs
are retained in the ignored stage log directory.

Workflow attribution: scientific figure/export checks used the
scientific-visualization skill's guidance. Reference: Timothy Kassis, Vinayak
Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner (2026), *Scientific Agent
Skills: A Library of Procedural Knowledge for Research Agents*,
https://doi.org/10.48550/arXiv.2609.00065. Manuscript revision used the local
paper-writing workflow; the supplied data and repository outputs remain the
evidence for scientific claims.
