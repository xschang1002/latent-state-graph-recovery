# Hydraulic Transition Warning

Features are computed only from past cycles. The target is whether the selected label changes within the next horizon cycles.
Decision thresholds are selected on the chronological validation segment.
The final horizon targets before each train/validation and validation/test boundary are purged, so target labels never cross a split boundary. Missing sensor values use causal forward fill only.
STCG selection policy: `reject_nonpositive`. The default rejects all-nonpositive candidate sets to their fitted K=1 candidate.

| Label | Horizon | Method | Threshold | Test Pos. Rate | Balanced Acc. | Precision | Recall | F1 | AP | ROC AUC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| valve_condition | 3 | Train Majority | 0.195 | 0.222 | 0.500 | 0.222 | 1.000 | 0.363 | 0.222 | 0.500 |
| valve_condition | 3 | Sensor Logistic | 0.208 | 0.222 | 0.500 | 0.222 | 1.000 | 0.363 | 0.219 | 0.484 |
| valve_condition | 3 | Graph Ridge Logistic | 0.114 | 0.222 | 0.522 | 0.232 | 0.799 | 0.359 | 0.225 | 0.524 |
| valve_condition | 3 | Sensor+Graph Logistic | 0.045 | 0.222 | 0.496 | 0.220 | 0.944 | 0.357 | 0.230 | 0.538 |
| valve_condition | 3 | STCG-v1 Selected Logistic | 0.564 | 0.222 | 0.509 | 0.231 | 0.361 | 0.282 | 0.237 | 0.541 |
| valve_condition | 3 | Sensor+STCG-v1 Logistic | 0.734 | 0.222 | 0.525 | 0.256 | 0.292 | 0.273 | 0.239 | 0.537 |
| valve_condition | 5 | Train Majority | 0.325 | 0.370 | 0.500 | 0.370 | 1.000 | 0.541 | 0.370 | 0.500 |
| valve_condition | 5 | Sensor Logistic | 0.178 | 0.370 | 0.500 | 0.370 | 1.000 | 0.541 | 0.396 | 0.511 |
| valve_condition | 5 | Graph Ridge Logistic | 0.093 | 0.370 | 0.553 | 0.404 | 0.812 | 0.539 | 0.414 | 0.559 |
| valve_condition | 5 | Sensor+Graph Logistic | 0.010 | 0.370 | 0.524 | 0.383 | 0.908 | 0.539 | 0.426 | 0.570 |
| valve_condition | 5 | STCG-v1 Selected Logistic | 0.001 | 0.370 | 0.508 | 0.374 | 0.996 | 0.544 | 0.380 | 0.542 |
| valve_condition | 5 | Sensor+STCG-v1 Logistic | 0.005 | 0.370 | 0.504 | 0.372 | 0.988 | 0.540 | 0.387 | 0.556 |
| valve_condition | 10 | Train Majority | 0.655 | 0.742 | 0.500 | 0.742 | 1.000 | 0.852 | 0.742 | 0.500 |
| valve_condition | 10 | Sensor Logistic | 0.123 | 0.742 | 0.500 | 0.742 | 1.000 | 0.852 | 0.825 | 0.592 |
| valve_condition | 10 | Graph Ridge Logistic | 0.089 | 0.742 | 0.644 | 0.811 | 0.875 | 0.842 | 0.887 | 0.729 |
| valve_condition | 10 | Sensor+Graph Logistic | 0.000 | 0.742 | 0.500 | 0.742 | 1.000 | 0.852 | 0.901 | 0.752 |
| valve_condition | 10 | STCG-v1 Selected Logistic | 0.000 | 0.742 | 0.528 | 0.753 | 0.983 | 0.853 | 0.800 | 0.594 |
| valve_condition | 10 | Sensor+STCG-v1 Logistic | 0.057 | 0.742 | 0.583 | 0.776 | 0.975 | 0.864 | 0.792 | 0.604 |
| stable_flag | 3 | Train Majority | 0.076 | 0.082 | 0.500 | 0.082 | 1.000 | 0.151 | 0.082 | 0.500 |
| stable_flag | 3 | Sensor Logistic | 0.394 | 0.082 | 0.526 | 0.127 | 0.132 | 0.130 | 0.111 | 0.582 |
| stable_flag | 3 | Graph Ridge Logistic | 0.973 | 0.082 | 0.513 | 0.090 | 0.245 | 0.132 | 0.134 | 0.583 |
| stable_flag | 3 | Sensor+Graph Logistic | 0.964 | 0.082 | 0.517 | 0.090 | 0.321 | 0.141 | 0.116 | 0.585 |
| stable_flag | 3 | STCG-v1 Selected Logistic | 0.161 | 0.082 | 0.535 | 0.093 | 0.528 | 0.158 | 0.085 | 0.522 |
| stable_flag | 3 | Sensor+STCG-v1 Logistic | 0.014 | 0.082 | 0.568 | 0.097 | 0.774 | 0.173 | 0.099 | 0.575 |
| stable_flag | 5 | Train Majority | 0.119 | 0.128 | 0.500 | 0.128 | 1.000 | 0.227 | 0.128 | 0.500 |
| stable_flag | 5 | Sensor Logistic | 0.769 | 0.128 | 0.523 | 0.165 | 0.181 | 0.172 | 0.152 | 0.544 |
| stable_flag | 5 | Graph Ridge Logistic | 0.033 | 0.128 | 0.495 | 0.126 | 0.506 | 0.201 | 0.114 | 0.477 |
| stable_flag | 5 | Sensor+Graph Logistic | 0.079 | 0.128 | 0.544 | 0.139 | 0.964 | 0.243 | 0.110 | 0.465 |
| stable_flag | 5 | STCG-v1 Selected Logistic | 0.094 | 0.128 | 0.526 | 0.139 | 0.590 | 0.225 | 0.136 | 0.526 |
| stable_flag | 5 | Sensor+STCG-v1 Logistic | 0.027 | 0.128 | 0.524 | 0.135 | 0.843 | 0.233 | 0.160 | 0.554 |
| stable_flag | 10 | Train Majority | 0.223 | 0.244 | 0.500 | 0.244 | 1.000 | 0.393 | 0.244 | 0.500 |
| stable_flag | 10 | Sensor Logistic | 0.248 | 0.244 | 0.500 | 0.244 | 1.000 | 0.393 | 0.278 | 0.515 |
| stable_flag | 10 | Graph Ridge Logistic | 0.000 | 0.244 | 0.520 | 0.253 | 0.918 | 0.396 | 0.275 | 0.522 |
| stable_flag | 10 | Sensor+Graph Logistic | 0.000 | 0.244 | 0.530 | 0.257 | 0.930 | 0.402 | 0.278 | 0.531 |
| stable_flag | 10 | STCG-v1 Selected Logistic | 0.008 | 0.244 | 0.529 | 0.257 | 0.873 | 0.397 | 0.250 | 0.546 |
| stable_flag | 10 | Sensor+STCG-v1 Logistic | 0.008 | 0.244 | 0.560 | 0.270 | 0.962 | 0.421 | 0.272 | 0.590 |
