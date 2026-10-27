# Final interpretation and MLJ supporting materials — 2026-10-23

This stage adds an interpretation of existing results and prepares submission
materials. It runs no new simulation, ESS fit or null chain. The previous
uncommitted tree was snapshotted in
`experiments/logs/mlj_final_text_2026-10-23/before/` (94 files). All 34 existing
result, table and figure files covered by that snapshot remain byte-identical.
No commit or staging operation was made; the pre-existing staged deletion is
retained.

## Interpretation and local corrections

Immediately after Table 6, Eq. (12) rewrites the existing two-state hard-emission
identity on the same equally weighted seeds:

mean Delta = mean Hard − mean Raw = [mean(J*C_edge) − (2*mean Raw − 1)] / 2.

It compares the jointly attained grouping/graph-contrast ability with the
temporal ability already present in the raw scores. Full-precision inputs are:

| Condition | Mean J*C_edge | 2*mean Raw − 1 | Mean Hard − Raw |
|---|---:|---:|---:|
| D=300, M=7, native | 0.2982023654 | 0.4325957058 | −0.0671966702 |
| D=600, M=7, native | 0.5962955155 | 0.5349152069 | +0.0306901543 |
| D=300, M=19, forced | 0.3586802961 | 0.4435105209 | −0.0424151124 |

The M=19 row uses all unchanged forced candidates, as stated in the table
caption; the identity must not be substituted into the native static fallback.
From D=300 to 600 at M=7, hard increases by 0.1490465751 and raw by
0.0511597506, so hard-minus-raw improves by 0.0978868245. The paper labels this
an algebraic/descriptive rewriting, not a new theorem, interaction test or a
general rule for choosing D. It retains the exploratory positive-gain status.

Five old statements were corrected:

- AUROC/AUPRC/precision/SHD depend on the target and indexing; they are not
  inherently union-level metrics.
- Self-lags enter the regression design but not the scored edges or clustering
  features, matching the appendix configuration.
- The unsupported general variance claim for averaging/max was removed; the
  empirical similarity of union rankings is retained.
- Gate acceptance is described through fitted improvement. It requires neither
  identifiable true states nor changing support and cannot distinguish support,
  other mechanism changes and state-free fluctuations. The existing geometric
  argument remains intact.
- Table 14 describes core simulation defaults, with scan overrides and a
  separate ESS protocol.

The checker adds three mean-identity checks and three descriptive difference
checks, for 237 checks total. Independent read-only review found no additional
material inconsistency between the edited paper, information sheet and CSVs.

## Contribution information sheet

The separate two-page document and editable source are in
`submission_materials/mlj_contribution_information/`. They address main claim
and importance, precise evidence, closest work and previous publication.
Requirements were checked against the official MLJ instructions on 2026-10-23:
https://link.springer.com/journal/10994/submission-guidelines .

The sheet distinguishes the independently replicated core temporal loss from
the selected-cell exploratory gain, treats twenty seeds rather than 616
partitions as statistical units, and relates the work to Clémençon et al.
(2008), Saggioro et al.
(2020), and Killick et al.
(2012). Ranking theory is credited as the underlying classical principle;
the constrained segmentation controls do not support general method superiority.
The author explicitly confirmed in this conversation that the work has not
been formally published and has no other overlapping manuscript under review;
the final statement uses that confirmation.

## Reproduction checks and delivery

The README now consistently distinguishes packaged ESS inputs from the
excluded older Hydraulic/AReM inputs. It also explains that the alternative
rebuild commands should start from fresh extractions after a prior rebuild
changes manifest-covered bytes. The single entry verifies supplied outputs;
it does not refit all 620 synthetic configurations/seeds or rerun ESS.

A new Python 3.12.13 virtual environment was created inside the stage log
directory. All 47 pinned dependencies were installed offline from copied local
wheel caches. It has no inherited system/user site packages or PYTHONPATH;
all third-party module/distribution origins are inside the new environment.
All versions match and `pip check` succeeds. This is a clean dependency
environment on the same Windows host, rather than a new operating-system image.

The full repository suite passes 292 tests (74.92 seconds), with three existing
third-party SWIG deprecation warnings. Numerical checks pass 237/237; static
source validation reports no errors. Historical freeze checks match 49/49 pairs,
and the data manifest matches 9/9 files. The manuscript is 35 pages and the
information sheet two pages; both have zero overfull boxes or undefined
references, with all 21/8 fonts embedded. Changed manuscript pages and both
information-sheet pages were rendered and visually inspected.

In separately extracted copies, `--verify` passed in 42.708 seconds, leaving
all 2,009 payloads plus manifest unchanged and adding no files. `--rebuild`
passed in 51.078 seconds: all 19 generated tables and seven grouping
analysis/protocol outputs were byte-identical, and 218 other experiment CSVs
and the numeric fit cache were unchanged. Only the temporal figure PDF creation
time changed; page geometry and decompressed vector drawing streams were
identical. Both commands passed 237 numerical checks with empty stderr and
read retained fit outputs without refitting. These tests used the export before
adding this final audit and documentation; executable code and numerical inputs
are the same in the final export.

Exact results are recorded in the accompanying JSON and stage logs. The final
enclosing archive has a separate
payload/hash validation record, avoiding self-referential ZIP hashes inside its
own payload. Its manifest records every included file, the complete 1,400-record
specialist checkpoint and the identity-scan results.

Current manuscript: `paper_preprint_2026-10-23_mlj_final.pdf`, byte-identical to
packaged `paper/main.pdf`. The information sheet is a separate submission
attachment and is also included with its source in the anonymous artifact.
No submission or publication action was performed.
