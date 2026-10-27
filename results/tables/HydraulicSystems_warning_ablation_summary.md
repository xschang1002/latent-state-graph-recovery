# Hydraulic Warning STCG Selector Ablation

Configurations are selected by `validation_threshold_metric` on the chronological validation segment.
The table reports held-out test metrics for those validation-selected configurations.
The final horizon targets before each train/validation and validation/test boundary are purged; no target supervision crosses a split boundary.
STCG selection policy: `reject_nonpositive`; all-nonpositive candidate sets use their fitted K=1 candidate by default.

| Label | Horizon | Method | Objective | Lookback | Window | Validation F1 | AP | F1 | ROC AUC | State K | Active Dyn. |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| stable_flag | 3 | STCG-v1 Selected Logistic | transition | 40 | 16 | 0.275 | 0.070 | 0.068 | 0.443 | `4:2162` | 8 |
| stable_flag | 3 | Sensor+STCG-v1 Logistic | stable | 40 | 16 | 0.243 | 0.091 | 0.110 | 0.527 | `1:2162` | 0 |
| stable_flag | 5 | STCG-v1 Selected Logistic | transition | 40 | 24 | 0.296 | 0.136 | 0.225 | 0.526 | `3:2160` | 6 |
| stable_flag | 5 | Sensor+STCG-v1 Logistic | stable | 30 | 24 | 0.360 | 0.182 | 0.220 | 0.585 | `1:2170` | 0 |
| stable_flag | 10 | STCG-v1 Selected Logistic | stable | 40 | 16 | 0.402 | 0.194 | 0.390 | 0.400 | `1:2155` | 0 |
| stable_flag | 10 | Sensor+STCG-v1 Logistic | transition | 40 | 24 | 0.493 | 0.272 | 0.421 | 0.590 | `3:2155` | 6 |
| valve_condition | 3 | STCG-v1 Selected Logistic | stable | 40 | 16 | 0.361 | 0.278 | 0.364 | 0.576 | `1:2162` | 0 |
| valve_condition | 3 | Sensor+STCG-v1 Logistic | stable | 60 | 16 | 0.433 | 0.247 | 0.199 | 0.542 | `1:1418;2:724` | 12 |
| valve_condition | 5 | STCG-v1 Selected Logistic | stable | 30 | 16 | 0.547 | 0.428 | 0.496 | 0.555 | `1:2170` | 0 |
| valve_condition | 5 | Sensor+STCG-v1 Logistic | transition | 60 | 16 | 0.583 | 0.366 | 0.541 | 0.496 | `2:5;3:363;4:1772` | 18 |
| valve_condition | 10 | STCG-v1 Selected Logistic | transition | 40 | 16 | 0.848 | 0.785 | 0.758 | 0.575 | `4:2155` | 8 |
| valve_condition | 10 | Sensor+STCG-v1 Logistic | stable | 60 | 24 | 0.833 | 0.834 | 0.829 | 0.612 | `1:754;2:1381` | 12 |
