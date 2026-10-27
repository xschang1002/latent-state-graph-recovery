# Statistical plan (fixed before the confirmatory run)

This plan was written before seeds 2000--2049 were generated. Seeds 0--19 and
100--119 were used during development; 200--219 were used for the earlier gate
validation and are treated as development here, not as confirmation.

## Unit of replication

The independent unit is a **generated series seed**. Windows within a series
overlap and are never treated as independent. Settings (overlap, switch period,
stationarity) are conditions, not replicates: when results are pooled across
settings, the seed is the clustering unit and each seed contributes one
averaged difference.

## Primary outcomes

1. Union Edge AUROC (`union_auroc`).
2. Regime-specific Edge AUPRC (`regime_auprc_macro`), which is only defined for
   methods that return state scores.

Secondary outcomes: union AUPRC, state ARI, wall-clock seconds.

## Primary comparisons

- **C1**: STCG-v1 versus no-state window max on union AUROC.
- **C2**: STCG-v1 versus PELT + VAR-Lasso on union AUROC.
- **C3**: STCG-v1 versus no-state window max on regime AUPRC (window max has no
  state scores, so C3 is replaced by STCG-v1 versus random balanced grouping
  from the aggregation-control experiment; it is reported as secondary and is
  not part of the Holm family).

## Effect size and uncertainty

- Paired difference per seed, averaged over the four settings.
- 95% confidence interval from a seed-clustered bootstrap with 10,000 draws.
- Two-sided paired t-test on the seed-level differences.
- Holm correction over the family {C1, C2} x {union AUROC, union AUPRC} = 4 tests.
- An effect is called significant only at Holm-adjusted p < 0.05. A
  non-significant difference is reported as "not resolved", never as
  equivalence.

## Failure handling

Any fit that raises or times out is recorded in the phase-specific failure log
(`audit/failure_log_development.csv`, `audit/failure_log_confirmation.csv`, or
`audit/failure_log_final_confirmation.csv`) with the setting, seed, method and
error. The three files carry only their headers at freeze time, and the
specialist summary reports 1500/1500 successful fits; a non-empty log would be
the evidence of a failure, so an empty one must not be read as "not recorded".
Failed cells are excluded from the paired test and the number of failures is
reported next to the result; a method with more than 5% failures is flagged.

## Stopping rule

No setting is dropped, added, or re-weighted after seeing the confirmatory
results. If a comparison is significant in the opposite direction to the
manuscript's hypothesis, it is reported as such.

## Amendments (recorded after the run, before the manuscript rewrite)

1. **PELT adapter defect and re-run.** The first confirmatory run used
   `segmented_lasso_scores`, which built refit ranges from the first window's
   start to the last window's end. Adjacent ranges therefore overlapped by one
   stride (150 rows), and the refit re-used 300-point windows even when the
   partition contained one window per segment. The 2x2 factorial exposed this:
   PELT returned one segment per window in 200/200 fits, so the "segmented"
   baseline was in fact a per-window refit on overlapping data, and it appeared
   to beat the two-stage estimator. The adapter was replaced by
   `refit_group_scores`, which partitions the series at window-centre midpoints
   with no overlap. The confirmatory run was repeated on the same seeds with the
   corrected adapter; both runs are kept in `results/confirmatory` history
   through the archived CSVs, and the manuscript reports the corrected one.
   This amendment removes an artefact that favoured the baseline; it does not
   add a favourable setting.
2. **PELT segment count is reported** next to every PELT result, because the
   method returns 7/7 segments at the 7-window operating point.
3. **Prevalence and lift** were added to the aggregation experiment after the
   first draft, in response to the observation that AUPRC and AUROC cannot be
   compared across estimands without the positive rate.
4. **Lag shift was reclassified** from negative control to genuine lag-specific
   structure change. The taxonomy is fixed in
   `scripts/run_control_taxonomy.py` and was applied to the stored controls
   without changing any generator setting.

### Amendment 5: final hold-out and added outcomes

Seeds 2000--2049 were inspected while the PELT adapter was corrected, so they
are not a clean hold-out. The final confirmation uses seeds 3000--3049, fixed
before the run. The method set is STCG-v1, window max, corrected PELT+refit and
a constrained two-segment refit, with a global VAR-Lasso as a descriptive
reference. The Holm family is nine tests: three competitors on union AUROC,
union AUPRC and the temporal-graph AUROC. The global comparison is reported with
a bootstrap interval and is not part of the family. No setting or seed was
dropped after the run.

### Amendment 6: ESS Holm family and static temporal baselines

**ESS multiplicity.** The original ESS script applied Holm to all $32$ tests
(four imputations $\times$ four methods $\times$ two edge sets), while the text
described an eight-test family. Two corrections are now recorded, not one:

- The forward-fill strategy is treated as the pre-specified primary analysis,
  so its eight tests are Holm-corrected as one family ($p{=}0.010\to0.080$,
  $p{=}0.040\to0.239$). The other three imputations are exploratory sensitivity
  checks and are not part of that family.
- The combined $32$-test correction is retained in the audit for reference; no
  test survives either the $8$-test or the $32$-test correction, so the
  exploratory interpretation is unchanged.

**Static temporal baselines.** State-free methods (window mean, window maximum,
bagged mean) were given a well-defined temporal-graph score by treating their
graph as a constant prediction at every window. This is what revealed that the
fitted K-means temporal AUROC ($0.776$) sits below the static window-mean
prediction ($0.795$).

**Trend generator revision.** The `trend` control was originally a *global*
linear trend (added to every time point regardless of state) while the schedule
still produced two regimes, and `structure_counts` hard-coded $k_{\mathrm{dyn}}=1$.
These were inconsistent: ARI used a two-state label while $k_{\mathrm{dyn}}=1$.
The generator was revised so the trend is *state-specific* (state 1 drifts, state
0 does not), matching `mean_only` and `variance_only`, and `k_{\mathrm{dyn}}$ is
now $2$. This is a generator-scenario revision, not a counting fix; the `trend`
row changed (union AUROC $0.759\to0.672$, regime AUPRC $\to0.402$, ARI
$\to-0.148$), and the old and new generators must not be treated as the same
experiment.

### Amendment 7: segmentation comparison scope

The constrained and PELT segmentation results are a baseline-sensitivity study,
not a like-for-like method ranking. A fixed two-segment model under-fits the
recurring $A\to B\to A\to B$ structure, whereas PELT over-segments (one window
per segment) and re-fits very short ranges; these are two different limitations,
and neither configuration is a fully specified segmentation estimator. A
brute-force check over the $2^6=64$ contiguous partitions of seven windows
confirmed PELT returns the global optimum of its penalised cost, so the
over-segmentation is a penalty-scale issue, not an adapter bug.


### Amendment 8: repository audit corrections (2026-10-23)

The ESS adapter merged adjacent ranges because it treated a shared endpoint as
overlap. All detected segments were therefore refitted as one global range.
The corrected adapter retains adjacent ranges and merges only segments below
the existing minimum fit length; ESS has 23 detected segments and 23 refit ranges
under each of the four imputations.

The earlier ESS null shuffled nonself edge labels, preserving total edge count
but not labelled in/out degrees. It is replaced by a symmetric switch chain:
90% two-edge destination exchanges, 10% directed three-cycle reversals, with
self-loop/duplicate proposals rejected and the current state retained. For each
edge set, seed 0 produces 200 samples after 100|E| burn-in proposals with 10|E|
proposals between samples. The finite-chain approximation and p-value resolution
1/201 are reported. Samples are shared across methods.

The forward-fill eight-test family and exploratory status of the other three
imputations are retained. Recomputed forward-fill strong-edge lagged correlation
has raw p=0.004975 and Holm p=0.039801, while all-edge STCG has p=0.014925 and
Holm p=0.104478. No test passes the combined 32-test family. These replace the
ESS values and no-rejection statement recorded in Amendment 6; that amendment
remains as historical evidence. Pre-correction values are retained in
`results/revision2/ess_alignment_before_audit_2026-10-23.csv`.

The metric repairs return zero F1 for completely wrong nonempty graphs, map
arbitrary predicted labels to score rows, count unmatched states independently
of their numeric labels, and permit a zero-edge SHD budget. They do not change
the manuscript's synthetic AUROC/AUPRC estimands or the confirmatory test family.


### Amendment 9: ESS Monte Carlo resolution and seeds (2026-10-23)

The 200-draw audit repair was reproducible but its borderline primary rejection
changed across Monte Carlo seeds. The null budget is increased to 5,000 per edge
set for seeds 0--4. Seed 0 remains the fixed reference, chosen before this larger
run. The chain proposal, burn-in (100|E|), thinning (10|E|), data, imputations,
methods and fitted scores are unchanged. The seed repeats assess numerical
Monte Carlo variation on one fixed dataset; they are not additional data
replicates. The null remains a finite-chain approximation.

Each seed receives its own eight-test forward-fill Holm family, plus a
descriptive 32-test correction across imputations. No correction pools seeds,
and no minimum p-value is selected for the reference. The plus-one p-value now
has denominator 5001. Full seed records include null exceedance counts.

The primary strong-edge lagged-correlation comparison passes 5/5 seeds
(Holm 0.015997--0.036793; reference 0.025595). All seven other primary tests
remain above 0.05. The combined 32-test family rejects linear-imputation
strong-edge lagged correlation in 4/5 seeds (0.019196--0.051190), and the mean
imputation comparison in 1/5. Thus the earlier no-rejection statement is
superseded, but no combined-family rejection is stable across all five seeds.
Both favourable and threshold-crossing outcomes are reported.

The 200-draw results are preserved in
`results/revision2/ess_alignment_200_draws_seed0.csv`; the current reference and
all five seed results are `results/tables/ess_alignment.csv` and
`results/tables/ess_alignment_seed_sensitivity.csv`.


### Amendment 10: ESS diagnostics and implementation provenance (2026-10-23)

The 5,000-draw seed-0--4 primary protocol and both Holm families are unchanged.
The target is uniform over simple loop-free directed graphs with fixed labelled
in/out degrees. Two edge indices (90%) or three vertices (10%) are sampled
independently uniformly with replacement; invalid moves retain the graph.
The proposal is symmetric; sampling is not from a changing list of legal moves.
The plus-one calculation and Holm results remain approximate and exploratory,
without an exchangeable finite-sample MCMC test guarantee.

`diagnose_ess_null.py` replays all ten original chains, reproducing each stored
lagged-correlation null mean and exceedance count, and records attempted/accepted
moves, full retained AUROC traces and tail indicators. Descriptive ACF, an initial
positive monotone paired-ACF ESS capped at 5,000, and both ESS-based and 20-batch
means MCSE are reported. The reference strong-edge tail has 15 exceedances,
ESS estimate 5,000 and MCSE 0.000774. A descriptive diagnostic sensitivity here
extends only seed 0 to 1,000|E| burn-in for both edge sets, retaining 5,000 draws
and thinning 10|E|; its strong-edge raw p is 0.002200. It does not replace the
primary chain or enter either family. These statistics do not prove mixing.

The earlier PELT observed-score change is reconstructed as a separate adapter
repair: the old touching-range merge yielded global Lasso AUROCs 0.576290 /
0.657464; the corrected 23 disjoint refits yield 0.564972 / 0.554761 for all /
strong edges. These observed values were already fixed at the 200-draw stage.
The affected helper is also used by the historical WP4 runner; its CSVs are not
current manuscript inputs. Current factorial, confirmatory, final-confirmation
and segmentation panels use `refit_group_scores`, already partitioned correctly.
All 24 seed-0 factorial rows reproduce their stored AUROC/AUPRC inputs exactly.

One method's null rejection and another's non-rejection do not test their direct
difference. The ESS graph remains static coarse expert evidence, not temporal
localisation ground truth. Forward fill uses full-hour column means for leading
missing entries and whole-hour standardisation; this is now explicitly described
as offline preprocessing without changing fitted scores.

### Next planned analysis (not executed in Amendment 10)

The next substantive temporal study should compare raw windows, hard, GMM-soft
and oracle aggregation on the existing duration/window grid, report paired
T-AUC-delta differences against raw windows by h/D and M, and separate gate
acceptance from the effect conditional on performing aggregation. Table 3's
existing experiment has not been expanded in this cleanup; no new grid result
or confirmatory claim is introduced here.

### Amendment 11: temporal aggregation across the existing grid (2026-10-23)

This user-requested exploratory extension retains all 35 labelled entries of
`mechanism_grid.yaml` and seeds 0--19, with fixed h=300 and stride=150. Four
duplicate generator configurations are reused, not treated as independent data:
31 distinct configurations produce 620 configuration/seed fits. The two primary
displays are existing one-factor duration (D=150,300,600,900; M=7) and length
(T=600,1200,3000; D=300; M=3,7,19) scans. These are not a crossed h/D-by-M design.

All methods share the same window scores. Compare raw windows, native gated
hard aggregation, diagonal two-component GMM soft continuous emission, and
oracle window-label aggregation without refitting. Preserve the older GMM
weighted-prototype/argmax-emission variant as an additional diagnostic. Record
posterior entropy and fraction of diffuse responsibilities, convergence and
warnings; do not remove unfavorable/nonconverged finite fits. Neither GMM nor
oracle receives an artificial STCG gate.

For every configuration/seed/method, compute T-AUC-delta and its paired difference
against that seed's raw windows, using the existing centre-label, lag-specific,
nonself convention. Native hard rejection emits the existing static window-max
fallback (temporal AUROC 0.5 when the target is defined). Also compute the
ungated hard candidate on all seeds, without changing the actual gate. Record
gate acceptance and actual aggregation execution separately. Report both native
execution subsets and common hard-gate accepted/rejected subsets; the latter
use exactly the same seeds for every method and are selected descriptive groups,
not causal estimates of accepting the gate. Preserve undefined targets when no
changing edge flips in observed windows, with explicit evaluable denominators.

Use per-cell paired seed differences and pointwise 95% percentile bootstrap
intervals (10,000 resamples; seed 7); with fewer than five evaluable seeds report
the estimate and count but no interval. Do not pool cells or duplicate entries,
claim simultaneous coverage, or introduce a post hoc significance family.
Freeze protocol/source hashes before execution; keep previous temporal CSVs and
ESS outputs. Add GMM rows to the core temporal table only after raw/hard/oracle
core agreement and the historical argmax pooled result are checked.

Post-run diagnostic, recorded before its execution: responsibilities in the
620 fits are nearly hard (no maximum responsibility below 0.99), yet continuous
and argmax AUROC differ slightly in some cells. Replay the six distinct focus
cells and round raw and both GMM emissions to 12 decimal places to assess
ranking dependence on tiny score differences. Preserve primary predictions and
statistics; this diagnostic is post hoc and is not a tuned replacement method.

### Amendment 12: matched grouping and three-condition factor diagnostics (2026-10-23)

Retain exploratory positive-gain wording; no new seed block or confirmatory
claim is introduced. At the selected D=600, h=300, T=1200, M=7 cell, preserve
each fitted hard partition's two group sizes and enumerate every unlabelled
bipartition with those sizes (at most 35). Recompute maps and temporal AUROC
for each partition before averaging AUROCs within each seed. Compare hard--raw,
hard--matched-random and matched-random--raw using 20 existing simulation seeds
as the independent units, with the existing 10,000-resample/seed-7 pointwise
95% paired-seed percentile bootstrap. Partitions are never counted as new seeds.

Reuse the existing core factor CSV; deterministically reconstruct the forty old
fits for D=600/M=7 and D=300/M=19, whose labels/maps were not retained. Check
their raw and candidate scores against the original grid before analysis and
save numeric intermediates. Reuse that cache thereafter. Report mean J, mean
C_edge, mean per-seed J*C_edge, hard and raw temporal AUROC at the three cells,
using forced candidates for M=19, without attributing changes in marginal
means to independent causal mechanisms. Include valid-seed/no-target counts
from the existing full-grid output. No ESS experiment is expanded.
