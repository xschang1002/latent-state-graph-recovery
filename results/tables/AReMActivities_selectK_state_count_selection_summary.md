# Real State Count Selection

State counts are selected without external labels using:

`selection_score = state_improvement * temporal_coherence * balance_score / log2(K + 1)`

External labels are used only for the audit metrics shown below.

## Selected

| Dataset | Label | Selected K Distribution | Balanced Acc. | Macro F1 | Score |
|---|---|---|---:|---:|---:|
| AReMActivities_selectK | activity | `{"2": 5}` | 0.283 +/- 0.005 | 0.148 +/- 0.002 | 0.034 +/- 0.000 |

## Candidate Sweep

| K | Selected | Score | Improvement | Temporal Coherence | Min Fraction | Balance | Balanced Acc. | Macro F1 | Inferred Counts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 | 0.000 +/- 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.143 +/- 0.000 | 0.042 +/- 0.000 | `{"1": 5}` |
| 2 | 5 | 0.034 +/- 0.000 | 0.081 | 0.843 | 0.394 | 0.789 | 0.283 +/- 0.005 | 0.148 +/- 0.002 | `{"2": 5}` |
| 3 | 0 | 0.023 +/- 0.002 | 0.110 | 0.715 | 0.197 | 0.591 | 0.413 +/- 0.010 | 0.304 +/- 0.008 | `{"3": 5}` |
| 4 | 0 | 0.020 +/- 0.006 | 0.131 | 0.628 | 0.145 | 0.581 | 0.421 +/- 0.043 | 0.350 +/- 0.044 | `{"4": 5}` |
| 5 | 0 | 0.005 +/- 0.012 | 0.150 | 0.618 | 0.053 | 0.135 | 0.448 +/- 0.042 | 0.378 +/- 0.066 | `{"5": 5}` |
| 6 | 0 | 0.003 +/- 0.007 | 0.167 | 0.597 | 0.046 | 0.085 | 0.448 +/- 0.040 | 0.400 +/- 0.035 | `{"6": 5}` |
