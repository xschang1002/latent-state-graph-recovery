# Evaluation targets and metric glossary

This glossary describes the current manuscript's targets and their code names.
The [research guide](research_guide.md) locates the scientific evidence and
current supplement. Older API names remain available; a name containing
`temporal` does not by itself identify within-edge temporal discrimination.

## Notation and scoring entries

`X[t,i]` is the observed series, `z[t]` its simulation state, and
`A[k,l,i,j]` the binary support for `i(t-l) -> j(t)` in state `k`. Window `w`
uses half-open bounds `[start_w, stop_w)` and integer centre
`c_w = (start_w + stop_w - 1) // 2`. `S[w,l,i,j]` denotes the emitted score,
which can be a raw window score or an aggregation output. The centre target is
`A_w = A[z[c_w]]`.

An evaluated edge is the lag/source/target triple `e=(l,i,j)` with `i != j`.
Self-lag terms can enter the data-generating process and regression design;
they are excluded from scored edges and clustering features. Support is binary;
absolute coefficient scores do not evaluate sign, coefficient magnitude or a
causal effect without further assumptions.

## The distinct targets

| Target | Comparison being made | Current code/field |
|---|---|---|
| Union support | Edge scores against `A_union[e] = any_k A[k,e]` | `union_metrics`; `union_auroc` in paired-study outputs |
| Regime-specific support | Estimated state maps against true state maps after label matching | Auxiliary `per_regime_metrics` |
| Pooled time-indexed support | All `(window, edge)` scores ranked together against centre support | `pooled_temporal_auroc`; auxiliary API `temporal_graph_auroc` |
| Within-edge temporal state discrimination | Active windows ranked against inactive windows separately for each evaluable changing edge, then macro averaged | `temporal_delta_auroc`, or T-AUC_delta |
| State-label alignment | Predicted window groups against centre-state labels | ARI, matched macro F1, state-count error; auxiliary `temporal_state_metrics` |
| Score-change boundary detection | Adjacent-score changes against actual support-change events in centre-to-centre slots | `boundary_positive_change_ap`, `boundary_standard_ap`, `boundary_slot_f1` |
| Stable-edge variation | Detected score changes on edges whose binary support never changes | `stable_false_change_rate` and persistent/absent subdivisions |

### Union and regime-specific recovery

Union evaluation uses one score per nonself lagged edge. In the paired-study
helper, the union score defaults to the edgewise maximum of the emitted scores
over windows, unless an explicit union score is supplied. A good union ranking
shows which edges occur somewhere, rather than when they are active. AUROC,
non-interpolated average precision (also called AUPRC in legacy output fields),
precision at a specified count and SHD address this support target. Using the
true edge count for thresholded evaluation is an offline choice, not a
label-free operating point.

The auxiliary regime API uses maximum-overlap rectangular Hungarian matching
of predicted and true window labels. State numbering is arbitrary; unmatched
predictions and missed states are recorded separately. Regime support recovery
and label alignment are different targets. Distinct dynamical states can share
one binary support, for example when only signs or noise change.

### Pooled AUROC versus within-edge T-AUC_delta

Pooled AUROC flattens all scored window/edge entries before ranking. It mixes
comparisons of one edge across windows with comparisons between different
edges. Both components can depend on temporal predictions, so pooled AUROC
cannot alone isolate temporal state discrimination.

For within-edge AUROC, first identify nonself edges whose support differs across
true states. Retain an edge only if the evaluated centre labels contain both
active and inactive windows for that edge. Compute its AUROC over windows,
using half credit for tied positive/negative scores, then average over eligible
edges with equal weight per edge. Constant scores give 0.5 on an eligible edge;
a target with no active or no inactive window is undefined, not 0.5 or zero.
The outputs distinguish total changing edges from edges actually evaluable on
the sampled centres. This score measures ordering of active/inactive windows,
not boundary distance or thresholded graph accuracy.

`src/stcg/evaluation.py:temporal_graph_metrics` emits a state map at each
assigned window and then pools its entries. Despite its name, it computes the
pooled target. `static_temporal_graph_metrics` repeats one map on that same
pooled target. `temporal_delta_auc` computes the within-edge target for
state-map emissions. The A/B helper `evaluate_predictions` in
`scripts/aggregation_study_common.py` and the score-control evaluator accept
explicit emitted arrays and retain both targets as separate fields.

## Score-change boundary slots

Boundary diagnostics use the actual time-indexed support, not just a change in
state labels at neighbouring centres. Slot `w` is `(c_w, c_(w+1)]`. Its target
for edge `e` is positive if at least one actual support change occurs in that
interval. Multiple events are collapsed to one positive slot, with their
counts retained. A pair of identical endpoint supports can therefore still
contain a positive slot if the edge changes and changes back inside it.
Events outside the first-to-last centre interval are not covered.

The predicted strength is `abs(S[w+1,e] - S[w,e])`. It cannot locate an event
inside a slot or distinguish multiple events there. The metric resolution
changes with stride; comparisons across window grids are interpreted through
within-cell differences from raw, rather than absolute AP across strides.
Implementation: `boundary_grid`/`evaluate_localisation` in
`scripts/analyze_temporal_localisation.py` and `evaluate_scores` in
`scripts/analyze_boundary_sensitivity.py`.

### Two AP conventions

Both conventions use the same changing edges with at least one positive and
one negative covered slot. Exact score ties are processed as whole groups.
Standard non-interpolated AP includes every score group. Positive-change AP,
AP_plus, treats zero strength as abstention and omits only the final zero group.
For each eligible edge,

`AP_standard = AP_plus + pi_boundary * (1 - R_plus)`,

where `pi_boundary` is the fraction of positive slots and `R_plus` is recall
reached by strictly positive strengths. With all strengths zero, standard AP
is prevalence and AP_plus is zero. Each is macro averaged over eligible edges;
the zero-group increment is averaged per edge, rather than obtained from a
product of marginal means. Missing positive/negative classes remain missing
under both conventions. Neither is a trapezoidal precision-recall area.

### F1, amplitude references and stable edges

A detected event requires strength strictly greater than a linear-interpolated
quantile. The original diagnostic uses a common raw q95 threshold: all nonself
raw edge/slot strengths in that case/scorer define the threshold for every
method. The sensitivity analysis reports every fixed quantile in
`{0.80, 0.90, 0.95, 0.975, 0.99}` under both that raw-shared reference and each
method's own reference. No threshold is selected by test-set F1. Method-own
quantiles address uniform positive rescaling, not independent false-alarm
calibration; tie masses and score shape still matter.

Boundary F1 is computed per changing edge with at least one covered positive
slot and then macro averaged. It remains defined when every slot is positive,
although AP is then undefined. For a constant prediction on an eligible F1
target, F1 is zero. Stable-edge exceedance rates divide detected events by the
number of stable edge/slot opportunities. Always-present and always-absent
edges are also reported separately; their denominators and the fraction with
any detected change are retained.

## Output policies and averaging units

Gated hard (static-max fallback) emits the candidate on acceptance and a static-max map on
rejection: the edgewise maximum over raw windows is repeated through time.
Its retained output identifier is `native_hard`. Forced hard always emits the
candidate. Raw fallback retains the raw window
scores on rejection. Oracle emissions use known simulation states. All-case
differences, gate acceptance and effects conditional on execution are separate
summaries.

Edges are macro averaged within a repeat where specified. The statistical units
for intervals and paired contrasts are independent simulation repeats; windows,
edge pairs and method rows are not independent replicates. Undefined targets,
finite counts and failures retain their reasons and denominators. Metric
sensitivity and fixed-weight controls are post hoc and do not extend the
original A/B testing families. Study C has its own separately specified new
independent repeats.

## Auxiliary label and majority-window APIs

`temporal_state_metrics(mode="center")` compares window labels with centre
states. Its state-label boundary F1 uses a one-window tolerance and is distinct
from the score-change slot F1 above. `mode="coverage"` spreads window labels
over observed samples and majority-votes; it is a separate sensitivity target.
Window purity is the largest true-state fraction within a window and describes
mixing, not edge-support accuracy.

`windowed_edge_metrics` computes an across-edge AUROC within each window against
its majority regime, then averages valid windows. It is retained for auxiliary
and historical analyses. It is neither pooled centre-target AUROC nor the
within-edge temporal AUROC used for the main aggregation contrasts. Older output
field names are interpreted by their actual function and target rather than
silently relabelled as current primary metrics.

The main clustering comparisons supply candidate K=2; the PELT comparator uses
a penalised unknown segment count. Gate-stress controls can intentionally
supply two candidates for one true dynamical state. These designs do not
establish general unknown recurrent-state-count recovery. Real operating labels
are not graph truth, and the ESS expert graph is rolled and static rather than
lag-specific time-indexed support.
