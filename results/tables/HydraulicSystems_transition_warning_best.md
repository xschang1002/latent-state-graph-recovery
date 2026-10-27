# Hydraulic Transition Warning Best Methods

| Label | Horizon | Best Method | Threshold | AP | AP Gain vs Majority | F1 | ROC AUC |
|---|---:|---|---:|---:|---:|---:|---:|
| stable_flag | 3 | Graph Ridge Logistic | 0.973 | 0.134 | 0.052 | 0.132 | 0.583 |
| stable_flag | 5 | Sensor+STCG-v1 Logistic | 0.027 | 0.160 | 0.032 | 0.233 | 0.554 |
| stable_flag | 10 | Sensor+Graph Logistic | 0.000 | 0.278 | 0.034 | 0.402 | 0.531 |
| valve_condition | 3 | Sensor+STCG-v1 Logistic | 0.734 | 0.239 | 0.017 | 0.273 | 0.537 |
| valve_condition | 5 | Sensor+Graph Logistic | 0.010 | 0.426 | 0.056 | 0.539 | 0.570 |
| valve_condition | 10 | Sensor+Graph Logistic | 0.000 | 0.901 | 0.159 | 0.852 | 0.752 |
