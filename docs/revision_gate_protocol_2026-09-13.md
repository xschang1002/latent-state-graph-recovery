# Frozen revision gate and selector stress protocol

Protocol ID: `revision-gate-2026-09-13-v1`. Frozen before inspecting results from
the evaluation seeds. This is an additional diagnostic evaluation, not a change
to STCG defaults or a retrospective optimization of the existing main table.

## Inputs and fixed settings

Use independent generator seeds **100 through 119** (20 seeds per condition),
8 nodes, maximum lag 3, edge probability 0.18, innovation standard deviation
0.10, and a burn-in of 100. The production `generate_lagged_var` supplies the
ordinary stationary, switching and periodic conditions. The weak and imbalanced
extensions use exactly its coefficient sampler, RNG sequence and recurrence.
For the weak extension, the second coefficient tensor is `0.75 A + 0.25 B`;
edge truth is recomputed from the nonzero coefficients actually used. This is
a smaller coefficient change, not a claim of fewer changing edges. Convex mixing
preserves the sampler's bound of 0.85 on the total absolute incoming lag weights.
The imbalanced extension changes the observed switch position to 95% of the
series. These extensions are checked against the production generator when the
mixing factor is one and the production switch position is retained.

| Condition | T | Window / stride | Regime schedule / stress |
|---|---:|---:|---|
| stationary_many | 9000 | 300 / 150 | One graph, many windows: negative control |
| stationary_few | 1000 | 300 / 150 | One graph, original small window count: negative control |
| strong_switch_long | 9000 | 300 / 150 | Original independently sampled two-graph switch |
| weak_switch_long | 9000 | 300 / 150 | Second graph is 0.75 A + 0.25 B |
| mixed_periodic | 9000 | 300 / 150 | Period 150, intentionally mixed 300-point windows |
| short_switch | 600 | 150 / 75 | Original two-graph switch with limited samples |
| measurement_noise_switch | 9000 | 300 / 150 | Strong switch plus independent observation noise, sigma 0.20 |
| imbalanced_switch | 9000 | 300 / 150 | Second regime occupies the final 5% |

Observation noise uses seed `600000 + seed`; it is added after simulation,
because scaling all innovations in a linear standardized model is not a genuine
signal-to-noise intervention. KMeans uses seed `30000 + seed`, n_init=10 from
production, with one BLAS/OpenMP thread. All methods within a condition/seed see
the same series and local score tensors. No thresholds or cases are chosen by
examining edge truth or regime alignment. Labels are accessed for reporting only
after decisions and candidate selection have been computed.

## Gate ablations

Call the actual `stcg_v1_diagnostics` implementation, keeping the Lasso alpha
0.005, state count 2, nonself features, within-state mean, cross-state max, and
contrast mix zero. Cache only identical local Lasso tensors; the production
clustering, gate and aggregation are rerun for every variant. A regression test
compares cached results with a direct unpatched production call.

| Variant | Only change from default |
|---|---|
| default | None: primary improvement 0.12; fallback improvement 0.04, coherence 0.55, share 0.10 |
| always_accept | `auto_states=False` |
| always_reject | Primary and fallback improvement thresholds are infinity |
| no_fallback | Fallback improvement threshold is infinity |
| no_primary | Primary improvement threshold is infinity |
| no_coherence | Fallback coherence threshold is zero |
| no_min_share | Fallback minimum share threshold is zero |

Record primary acceptance, fallback-only acceptance, rejection and forced
acceptance separately. Coherence/share restrictions belong only to the fallback
path; a primary-path acceptance does not establish that these restrictions passed.
Report inferred count, acceptance, raw improvement/coherence/minimum share,
window-majority ARI and purity, Edge AUC, Pair AUC, P@K, SHD and lag accuracy.
Report false splitting separately on the two stationary negative controls.

## Independent selector ablations

For the same cached tensors, fit candidates K=1,2,3,4 with `auto_states=False`.
These are selector diagnostics on generated series, not repetitions of the
Hydraulic warning classifier experiment. The full stable score calls the actual
`state_count_selection_score`; the full warning score calls the actual
`stcg_transition_selection_score`. Their unchanged hard screens are improvement
0.04, minimum fraction 0.05, and additionally coherence 0.55 for the stable score.
Preserve the production lexicographic choice `(score, coherence, -candidate_K)`.
In particular, do not silently turn all-zero candidate scores into a K=1 fallback:
record all-zero scores and the count actually selected by that tie-break.

Independently remove (1) the balance multiplier, (2) the log2(K+1) denominator,
(3) the coherence multiplier/transition weight. The stable score additionally
tests removal of its coherence hard screen alone and removal of both coherence
screen and multiplier. Hard minimum-fraction screens remain for the multiplier
ablations. These explicit score-factor interventions are not new fitted models.
Report count selection, all-zero-score cases, false splitting, ARI and recovery
metrics; no true label appears in a scoring function or tie-break.

## Controlled fixtures, reported separately

Use nonnegative, zero-self-edge score tensors with 120 windows and 168 nonself
features. Background entries are `4 + N(0, 0.2)`; designated signal columns add
10 for their assigned group. Controlled fixtures use RNG `700000 + seed` and
are marked `evidence_kind=controlled_score_tensor`, never generator experiments.
They probe specific paths without asserting real-data prevalence:

- `coherent_weak`: 12 signal columns, balanced contiguous states.
- `shuffled_weak`: the same balanced labels in a fixed seeded random order.
- `tiny_coherent`: 12 signal columns, final 6/120 windows in state two.
- `primary_incoherent`: 36 signal columns, shuffled balanced states.
- `hierarchical_balanced`: 30 columns encode a balanced two-group split and 10
  additional columns encode four contiguous equally sized groups.
- `hierarchical_rare`: the same hierarchy with group sizes 60,30,24,6.

Actual production feature construction, KMeans, gates and aggregators process
these tensors. These fixtures have known design-group labels for alignment only;
they have no empirical causal ground truth, so edge-recovery metrics are absent.

## Reporting and stopping rules

Run every frozen condition/fixture and every listed seed once, retaining failed
cases with a visible error rather than dropping or replacing seeds. Do not rerun
with altered thresholds or fixtures to obtain a desired result. For each removal,
report paired mean differences from default/full plus a 95% paired percentile
bootstrap interval (10000 draws, seed 90913). Intervals are descriptive sensitivity
intervals, with no multiplicity-adjusted or confirmatory significance claims.
For 20-seed acceptance/false-split proportions, additionally report 95% Wilson
intervals. Record protocol/code SHA-256 and elapsed time. The resulting evidence
may demonstrate a mechanism or expose failure; neither outcome warrants changing
the production defaults without a separate prospective study.

Command: `python scripts/run_revision_gate_stress.py`

## Amendment A: deterministic rejection of nonpositive selector scores

Added after completing the frozen experiment. The original full and all
factor-removal rows explicitly retain `legacy` selection and are reproduced
unchanged. The initial protocol SHA-256 is
`62b4e67710e6f027b0ba2c16cc3a866375a5a9bbef08c1aabbdbef8b328929e3`.

The completed experiment exposed a control-flow defect: all candidate scores
could be zero after the eligibility screens, yet coherence tie-breaking still
selected a multistate candidate. The shared `reject_nonpositive` policy now
returns the actually fitted K=1 candidate when no score is positive. If this
candidate is unavailable it raises an explicit error. Positive-score ordering
is unchanged. No scoring threshold, feature, candidate set or generator was
tuned using the new outcomes.

Additional `positive_fallback` rows in each selector use this correction on the
same stored/fixed input protocol, marked `experiment_stage=postfinding_logic_fix`.
They are a post-finding correctness check, not a prospectively frozen superiority
experiment. The original gate settings in `src/stcg/self_supervised.py` are not
changed. Historical real-count runs remain reproducible with
`--selection-policy legacy`. Warning-runner adoption and any affected downstream
warning reruns are tracked separately from these selector-only diagnostics.
