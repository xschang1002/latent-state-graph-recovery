# Frozen additional real-data validation protocol

Frozen before running the new experiment on 2026-09-13. This protocol addresses
the label-sighted segment selection and single-condition forward-test gaps in
review R2 W2. It does not use outcome-dependent segment selection or promise a
positive result. The complete preselected grid, including single-label blocks,
state collapse, missing classes and failed fits, must be retained.

## Data and information policy

Use the existing local HydraulicSystems CSV (eight low-frequency/summary
channels; score cooler_condition and valve_condition) and AReMActivities CSV
(six channels; score activity). Coordinates depend on row count only. No labels
are used to choose intervals, K, thresholds or hyperparameters. The runner
writes a coordinate manifest before fitting and checks its hash when resuming.

AReM's repository CSV concatenates recordings in activity-directory order. Its
row order is an archival ordering, not an uninterrupted natural temporal stream.
Keep that order, report unseen-label fractions and recording boundaries, and do
not interpret failure or success as online activity monitoring. Hydraulic is a
controlled test-rig sequence, not an independently sampled field deployment.
No causal-edge ground truth is available; these are state-alignment/diagnosis
tests, not causal structure-recovery tests.

## Forward diagnosis

Use expanding training origins at 20%, 40%, 60% and 80% of each CSV. The next
20% is the test block. Window size and stride are both 300 rows. Include only
complete windows contained in their own train/test block; no window may cross a
split. Test blocks do not overlap. Past training blocks are shared across
origins, so origins and windows are not independent experimental replicates.

STCG graph features are nonself absolute VAR-Lasso coefficients (lag 3,
alpha 0.005) in each completed observation window. Local window normalization
is part of extracting that observed window's features; this is diagnosis at
window end, not prediction before seeing that window. Across-window feature
scalers, cluster centroids, classifiers and state-to-label maps fit training
windows only. Causal forward fill handles internal missing sensor values;
initial missing/nonfinite values fail explicitly.

Freeze six comparisons: training-majority prediction; supervised sensor
mean/std/change centroid; supervised graph-feature centroid; sensor logistic
regression (C=1, balanced classes, train-fitted scaling); STCG state mapping with
fixed K=2; and STCG state mapping with K chosen from 2..6 by training-only
silhouette. The latter is an explicitly separate exploratory variant, not a
claim that the existing primary scorer selects K. Both STCG variants retain
the existing gate defaults (0.12 improvement or 0.04/0.55/0.10 temporal path).
The K=2 choice is the same for both datasets and does not encode class counts.
Training labels are used only by downstream diagnosis/mapping, including the
supervised baselines. No true test K or test mapping is supplied.

Use KMeans seeds 0..4. These are algorithmic sensitivity repeats on the same
data, not five independent datasets. Report every origin, label composition,
test-class balanced accuracy, macro F1, accuracy, unseen-class window fraction,
inferred-state counts and collapse. Average seeds within each origin before
summarizing the four origins; report origin ranges and no significance test or
independence-based confidence interval. Also retain per-window predictions.

## Label-independent segment comparison

For each dataset take 1,500-row segments centered at 10%, 35%, 60% and 85% of
its row count, clamped to valid bounds. The resulting four segments per dataset
are disjoint; coordinates never depend on a transition location. All segments
are retrospective fits, with no train/test claim. Score Hydraulic's two label
columns from the same fitted assignments rather than fitting again per label.

Compare STCG graph-state clustering, standardized sensor-window KMeans, and
Regime-PCMCI, all using fixed supplied K=2 and seed 0. STCG uses its unchanged
default gate; actual inferred K is recorded. Window size/stride 300/300 gives
five windows per segment, a disclosed coarse temporal resolution. Report both
sample and window ARI/aligned accuracy, true label counts, transition count and
whether fewer than two true window labels make state recovery uninformative.
Any aggregate over multi-label windows is a post-hoc descriptive stratum only;
the full preselected grid remains primary and visible.

Regime-PCMCI keeps lag 3, 20 iterations, 10 annealings, transition cap 10,
switch threshold 0.05, pc_alpha 0.05 and alpha_level 0.01. It runs in an isolated
worker with n_jobs=1, one BLAS/OpenMP thread and a fixed 120-second wall budget
(including Python import and setup). A real timeout terminates the worker and
is recorded as a resource-limited failure with no score. No parameter fallback
or retries selected by outcome are permitted. Save successful, failed and
timed-out partials; do not discard failures from the reported grid. The timeout
is not a measurement of completed runtime or evidence of lower scientific
quality. This small-budget comparison cannot establish a general speed ratio.

## Reproduction

```powershell
.venv\Scripts\python.exe scripts\run_revision_real_validation.py --stage plan
.venv\Scripts\python.exe scripts\run_revision_real_validation.py --stage forward
.venv\Scripts\python.exe scripts\run_revision_real_validation.py --stage segments
.venv\Scripts\python.exe -m pytest -q tests/test_revision_real_validation.py
```

Tracked outputs use `results/tables/revision_real_*`. Resumable worker outputs,
source/data hashes and logs are saved under ignored
`experiments/logs/revision_real_validation/`. The frozen plan is stored in
`results/tables/revision_real_plan.csv`. Results and interpretation are appended
after execution without rewriting the protocol above.

## Forward results (completed)

All 360 planned method/seed/origin/label rows completed. These comprise eight
distinct dataset origins, with two Hydraulic label views and one AReM view;
there are 12 origin/label tasks, not 360 independent data replicates.

| Label | Multi-class test origins | Sensor centroid BA | Graph centroid BA | Fixed-K2 STCG BA | Train-silhouette STCG BA |
|---|---:|---:|---:|---:|---:|
| Hydraulic cooler | 2/4 | 0.750 | 0.724 | 0.250 | 0.298 |
| Hydraulic valve | 4/4 | 0.268 | 0.320 | 0.250 | 0.250 |
| AReM activity | 4/4 | 0.438 | 0.282 | 0.110 | 0.110 |

BA averages seeds within each origin, then averages the four origins equally.
The tracked summary contains macro F1, origin ranges and unseen-class rates;
the detail CSV contains label compositions and per-seed state diagnostics.
The same six methods and both STCG variants are reported regardless of score.

The valve task is the cleanest multi-class forward check here: all four tests
contain all four classes, with no previously unseen test class. Both STCG
state mappings remain at the majority baseline's 0.250 balanced accuracy;
the direct graph-feature centroid reaches 0.320. State compression therefore
does not establish a downstream diagnostic advantage in this check.

Cooler tests include new classes and two single-class blocks. Across origins,
25.6% of test windows on average have a label absent from the training block.
Training-only silhouette changes K for some seeds but does not close the gap
to sensor baselines; its balanced accuracy at the second origin is sensitive to
initialization (seed standard deviation about 0.427), so a favorable individual
seed would be misleading.

AReM's unseen-label fraction increases from 0.393 to 0.571, 0.714 and 0.857.
Furthermore, 70 of its 112 distinct test windows straddle archived recording
boundaries. These windows were retained by the frozen coordinate-only rule,
and their counts are reported. This limits AReM to an archival-order stress
test; it cannot supply evidence of natural online regime recovery. No label
or recording-dependent window filtering was introduced after seeing results.

The new forward checks resolve the *protocol coverage* gap: they supply
preselected chronological tasks, multi-class tests, train-only scaling and
mapping, simple successful comparators, and complete negative outcomes.
They do **not** establish stable practical superiority or close R2 W2's
substantive concern. Reducing a claim to match these results is scientifically
appropriate but is not new positive utility evidence.

Timing columns are debugging/provenance fields, not a speed comparison:
`feature_seconds` is shared feature extraction per forward origin and
`all_methods_fit_seconds` covers all six forward methods together for a label
and seed. Segment `fit_seconds` excludes shared graph-feature preparation for
STCG, while RPCMCI's worker fit covers its complete RPCMCI call. Their scopes
differ and must never be divided to claim a speedup. Only
`worker_wall_seconds` determines the fixed RPCMCI resource-budget outcome.

## Fixed-segment results (completed)

All eight sensor segments and all three methods were attempted. The two
Hydraulic label views produce 36 segment/label/method rows from 24 fits;
they must not be counted as independent fits. STCG and sensor KMeans completed
all eight segments each, and all eight Regime-PCMCI workers reached the fixed
120-second wall limit and were terminated. These are resource-limited
outcomes, not observed optimizer failures or evidence that STCG beats a
successfully fitted Regime-PCMCI model. Unknown optimizer diagnostics on
timeout are reported as missing, not as zero error-free annealings.

Every preselected cooler and AReM segment has only one true state. Both
completed methods nevertheless infer two states on all segments, yielding
ARI zero on those single-state tasks. This is evidence of a limitation of
fixed-K state partitioning, not positive recovery evidence. The three
multi-state valve window tasks have mean window ARI -0.142 for STCG and
0.074 for sensor KMeans. Including all four valve segments, sample ARI means
are 0.133 and 0.157 respectively. The difference between sample and window
metrics also underscores the coarse five-window resolution per segment.

The new segment selection is genuinely independent of labels, but its small
coverage, single-state outcomes and absent completed RPCMCI scores prevent a
positive practical-value or specialist-superiority claim. Combined with the
forward results, this strengthens the negative boundary evidence and resolves
several protocol objections while leaving the substantive R2 W2 utility
concern unresolved.

## Execution and reporting provenance

The executed plan signature is
`e9852c347280514cd20527dfe210590c7fe6f67b45662c4a6f1d7fd2b316e6f5`.
The original executed runner and manifest are preserved under the ignored log
directory as `executed_runner.py` and `executed_manifest.json`. After all fits
completed, a reporting-only correction changed unavailable timeout annealing
counts from a placeholder zero to missing; fitting, K policies, budgets and
predictions were unchanged. The runner also gained explicit reporting-only
aggregation and a selectable cache directory so old execution evidence need
not be discarded after a source update. `reporting_manifest.json` records
the updated reporting-source hash against the original execution signature.

Rebuild the retained execution's reports, without fitting or filling missing
jobs:

```powershell
.venv\Scripts\python.exe scripts\run_revision_real_validation.py --stage all --aggregate-from-manifest
```

For a new run from this checkout, preserve the old evidence with a fresh cache:

```powershell
.venv\Scripts\python.exe scripts\run_revision_real_validation.py --stage all --parts-dir experiments\logs\revision_real_validation_fresh
```

Six targeted tests passed, including real worker termination and test-label /
future-feature noninterference. Independent recomputation of all 360 forward
metric groups from the 24,480 saved window predictions agreed to a maximum
absolute difference of 1.11e-16. The reporting-only replay retained all 36
segment rows and all eight timeout outcomes. No PDF was compiled.
