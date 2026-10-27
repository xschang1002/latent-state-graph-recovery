# ESS and manuscript cleanup for Machine Learning — 2026-10-23

Historical stage record: the PDF, layout and hashes below describe the ESS
cleanup before the temporal-grid extension. Its original report and JSON are
preserved in the next stage's before snapshot. ESS results remain unchanged;
current manuscript/layout/package validation is in `audit/temporal_grid_2026-10-23.md`.

This round closes the ESS implementation and reporting questions. Existing
uncommitted work was preserved in
`experiments/logs/ess_mlj_cleanup_2026-10-23/before/`, with 47 file hashes in
`before_manifest.json`. No commit or staging change was made. The original
5,000-draw report/PDF and both older ESS CSVs remain as historical evidence.

## PELT observed scores: cause and scope

Increasing null draws did not change observed scores. The earlier repair to
`segmented_lasso_scores` stopped treating touching half-open refit ranges as
overlap. The former merge collapsed all detected ESS segments into one global
Lasso fit; direct global refitting reconstructs the archived scores exactly.
The corrected adapter keeps 23 disjoint chronological ranges, refits each with
the same Lasso settings, then takes the maximum across segments and lags.

| Forward-fill expert edges | Former global fit | Corrected 200 and 5,000 draws |
|---|---:|---:|
| All | 0.5762901452242825 | 0.5649717514124294 |
| Strong | 0.6574636174636175 | 0.5547609147609147 |

`python tools/verify_ess_refit_provenance.py` reconstructs this without writing
results. The legacy `run_baselines_wp4.py` also calls the repaired helper, so the
change is not exclusive to ESS code. Its archived CSVs are not used by the current
table builder. The current factorial, confirmatory, final-confirmation and
segmentation runners use `refit_group_scores`, which already keeps touching runs
separate. A replay of all 24 factorial rows at seed 0 matches all stored shared
AUROC/AUPRC fields exactly. The new regression also equates both adapters on
adequately long chronological segments. No synthetic CSV was overwritten.

## Null distribution and lightweight diagnostics

The target is uniform over simple directed graphs with fixed labelled in/out
degrees, excluding loops and parallel edges, allowing reciprocals. The code
draws edge indices or vertices independently uniformly with replacement; it does
not sample a state-dependent list of legal moves. Reverse proposals have the
same probability, so uniform measure is stationary. Two-edge switches and
three-cycle reversals connect the degree-realisation space, but this is not a
mixing bound. The manuscript cites the primary
[LaMar (2009) source](https://arxiv.org/abs/0912.3834) for this move-set context.

Diagnostics replay all ten published chains, preserving the random stream, and
add only two seed-0 longer-burn checks. All chains retain 5,000 graphs with
10|E| thinning; the primary burn-in is 100|E|, sensitivity burn-in 1,000|E|.
All ten original lagged-correlation null means and exceedance counts match the
stored primary results. The main reference and sensitivity result CSVs are
byte-for-byte unchanged in this round.

The 12-row diagnostic CSV records acceptance counts/rates separately for each
move and burn-in/sampling phase, null AUROC ACF/ESS/MCSE, and tail-indicator
ACF/ESS/MCSE plus 20-batch means MCSE. Full retained AUROC and tail traces have
60,000 rows. The supplement plots both full seed-0 traces and lags 1--40; no
autocorrelation is clipped. ESS uses a positive monotone sequence of paired
sample autocorrelations, capped at the number of retained draws. Constant
indicators receive undefined ESS/MCSE rather than falsely exact zero error.

| Primary strong-edge lagged-correlation tail | Result |
|---|---:|
| Reference exceedances | 15 |
| Estimated reference tail ESS | 5,000 |
| Estimated reference MCSE | 0.000774 |
| Reference 20-batch MCSE | 0.000703 |
| Five-seed tail ESS range | 4,657--5,000 |
| Five-seed tail MCSE range | 0.000600--0.000970 |
| Seed-0 longer-burn raw p | 0.002200 |

Sampling-phase two-edge acceptance is 62.88--62.94% for all edges and
73.25--73.34% for strong edges. Three-cycle acceptance is 0.234--0.260% and
0.042--0.053%, respectively. The latter is low and explicitly retained in the
table. These statistics reveal no marked retained-trace dependence for this
score, but are noisy for rare tails and do not prove mixing or bound chain bias.
The ordinary plus-one formula does not make sequential outputs exchangeable;
the manuscript retains approximate/exploratory language and cites
[Howes (2024, v3)](https://arxiv.org/abs/2310.04924v3).

The primary eight-test and descriptive combined-32 families are unchanged,
computed separately for each seed. Forward-fill strong-edge lagged correlation
has AUROC 0.677173, reference raw p=0.003199, Holm p=0.025595, and five-seed Holm
range 0.015997--0.036793. All seven other primary tests remain above 0.05. A
null rejection for one method and non-rejection for another does not test their
direct difference. The static expert graph cannot validate temporal localisation.

## Manuscript and artifact changes

- Abstract: mean per-sequence product E[J C_edge]=0.298, giving mean AUROC 0.649,
  replaces the misleading juxtaposition of separate factor means.
- Pooled metric: removed the residual claim that it barely responds to temporal
  localisation; the 0.17% figure describes pair composition, not a sensitivity
  bound. Cross-edge comparisons can retain temporal information.
- Static ceiling: explicitly positioned as a finite-edge specialisation of
  classical bipartite ranking, with its specific diagnostic role in dynamic
  graph recovery. The source was verified at
  [Clémençon, Lugosi and Vayatis (2008), Example 1](https://www.stat.uchicago.edu/~lekheng/meetings/mathofranking/ref/vayatis.pdf),
  DOI 10.1214/009052607000000910, Annals of Statistics 36(2):844--874.
- Preprocessing description: forward fill includes a full-hour column-mean
  fallback; standardisation also uses the full hour. The analysis is now
  explicitly offline. No estimator, imputation or observed score changed.
- Table 9 and Figure 6 are on page 21 within Section 10, before Discussion;
  the Conclusion is continuous on page 23. Appendix D explains provenance and
  proposals, and includes diagnostic Table 13/Figure 7 on page 27.
- Latest PDF: `paper_preprint_2026-10-23_mlj_cleanup.pdf` (29 pages), identical
  to the packaged `paper/main.pdf`. The builder now generates 13 tables and
  eight figures. Current hashes are in the accompanying JSON record.

## Validation

- 273 tests pass; three existing upstream SWIG deprecation warnings; 30.46 s.
  Seven new tests cover counter neutrality, rejected-proposal accounting,
  cycle moves, correlated/constant trace diagnostics, adapter agreement,
  stored trace/primary agreement, and diagnostic display inputs.
- 190 numeric checks pass; static manuscript validation reports no errors.
- Dataset manifest: nine files/100.7 MB match; historical freeze: 49/49 pairs
  match with no write.
- Primary ESS CSVs remain byte-identical to the before snapshot. All displayed
  tables and figures are generated from their CSV inputs.
- PDF compiles with no overfull boxes or undefined references. Pages 1, 8,
  20--21, 23 and 25--27 were rendered and visually reviewed. Existing underfull
  bibliography/reproduction paragraphs, a volume/number BibTeX warning and the
  Windows BBL rerun warning persist, with resolved citations and references.
- Anonymous package is rebuilt with complete 1,400 specialist records; all
  payloads and manifest hashes are compared against current source bytes,
  and its PDF is compared to the delivered PDF. No identity finding is allowed.

The proposed h/D-by-M temporal comparison (including GMM soft and separate gate
acceptance) is recorded as the next substantive study; it was not run in this
cleanup and no new cross-condition result is claimed. Work remains uncommitted.

## Workflow attribution

The local paper-writing, scientific-visualization and PDF workflows were used.
The visualization workflow attribution remains: Timothy Kassis, Vinayak Agarwal,
Yuhuan He, Darshil Patel and Aubrey M. Brueckner (2026), *Scientific Agent Skills:
A Library of Procedural Knowledge for Research Agents*,
[arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
This is workflow attribution, not scientific evidence for the manuscript.
