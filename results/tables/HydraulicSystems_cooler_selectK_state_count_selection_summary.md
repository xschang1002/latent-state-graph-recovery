# Real State Count Selection

State counts are selected without external labels using:

`selection_score = state_improvement * temporal_coherence * balance_score / log2(K + 1)`

External labels are used only for the audit metrics shown below.

## Selected

| Dataset | Label | Selected K Distribution | Balanced Acc. | Macro F1 | Score |
|---|---|---|---:|---:|---:|
| HydraulicSystems_cooler_selectK | cooler_condition | `{"3": 5}` | 0.948 +/- 0.004 | 0.948 +/- 0.004 | 0.034 +/- 0.000 |

## Candidate Sweep

| K | Selected | Score | Improvement | Temporal Coherence | Min Fraction | Balance | Balanced Acc. | Macro F1 | Inferred Counts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 | 0.000 +/- 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.333 +/- 0.000 | 0.168 +/- 0.000 | `{"1": 5}` |
| 2 | 0 | 0.019 +/- 0.000 | 0.044 | 0.913 | 0.368 | 0.736 | 0.656 +/- 0.002 | 0.538 +/- 0.002 | `{"2": 5}` |
| 3 | 5 | 0.034 +/- 0.000 | 0.073 | 0.941 | 0.332 | 0.995 | 0.948 +/- 0.004 | 0.948 +/- 0.004 | `{"3": 5}` |
| 4 | 0 | 0.023 +/- 0.003 | 0.089 | 0.927 | 0.163 | 0.650 | 0.959 +/- 0.002 | 0.959 +/- 0.002 | `{"4": 5}` |
| 5 | 0 | 0.021 +/- 0.004 | 0.098 | 0.843 | 0.129 | 0.647 | 0.952 +/- 0.009 | 0.952 +/- 0.009 | `{"5": 5}` |
| 6 | 0 | 0.015 +/- 0.003 | 0.108 | 0.801 | 0.082 | 0.493 | 0.946 +/- 0.018 | 0.946 +/- 0.018 | `{"6": 5}` |
