# Forward Real State Label Prediction

Graph-state centroids and state-label mappings are fit only on chronological train windows.
Test windows start after the train split and are assigned to the train-fitted centroids.

## Aggregate

| Dataset | Label | Method | N | Accuracy | Balanced Acc. | Macro F1 | Accepted Rate | State Counts |
|---|---|---|---:|---:|---:|---:|---:|---|
| HydraulicSystems_forward | cooler_condition | Forward GraphFeature Centroid | 5 | 0.966 +/- 0.000 | 0.966 +/- 0.000 | 0.491 +/- 0.000 | 1.000 | `{"3": 5}` |
| HydraulicSystems_forward | cooler_condition | Forward STCG-v1 State Mapping | 5 | 0.953 +/- 0.003 | 0.953 +/- 0.003 | 0.488 +/- 0.001 | 1.000 | `{"3": 5}` |
| HydraulicSystems_forward | cooler_condition | Forward SensorMeanStd Centroid | 5 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 | `{"3": 5}` |
| HydraulicSystems_forward | cooler_condition | Forward Train Majority | 5 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 | `{"3": 5}` |

## By Seed

| Seed | Method | Train Windows | Test Windows | Balanced Acc. | Train Counts | Test Counts | State Mapping |
|---:|---|---:|---:|---:|---|---|---|
| 0 | Forward Train Majority | 704 | 175 | 0.000 | 100:119;20:293;3:292 | 100:175 | `{}` |
| 0 | Forward SensorMeanStd Centroid | 704 | 175 | 1.000 | 100:119;20:293;3:292 | 100:175 | `{}` |
| 0 | Forward GraphFeature Centroid | 704 | 175 | 0.966 | 100:119;20:293;3:292 | 100:175 | `{}` |
| 0 | Forward STCG-v1 State Mapping | 704 | 175 | 0.954 | 100:119;20:293;3:292 | 100:175 | `{"0": "20", "1": "3", "2": "100"}` |
| 1 | Forward Train Majority | 704 | 175 | 0.000 | 100:119;20:293;3:292 | 100:175 | `{}` |
| 1 | Forward SensorMeanStd Centroid | 704 | 175 | 1.000 | 100:119;20:293;3:292 | 100:175 | `{}` |
| 1 | Forward GraphFeature Centroid | 704 | 175 | 0.966 | 100:119;20:293;3:292 | 100:175 | `{}` |
| 1 | Forward STCG-v1 State Mapping | 704 | 175 | 0.954 | 100:119;20:293;3:292 | 100:175 | `{"0": "100", "1": "20", "2": "3"}` |
| 2 | Forward Train Majority | 704 | 175 | 0.000 | 100:119;20:293;3:292 | 100:175 | `{}` |
| 2 | Forward SensorMeanStd Centroid | 704 | 175 | 1.000 | 100:119;20:293;3:292 | 100:175 | `{}` |
| 2 | Forward GraphFeature Centroid | 704 | 175 | 0.966 | 100:119;20:293;3:292 | 100:175 | `{}` |
| 2 | Forward STCG-v1 State Mapping | 704 | 175 | 0.949 | 100:119;20:293;3:292 | 100:175 | `{"0": "3", "1": "20", "2": "100"}` |
| 3 | Forward Train Majority | 704 | 175 | 0.000 | 100:119;20:293;3:292 | 100:175 | `{}` |
| 3 | Forward SensorMeanStd Centroid | 704 | 175 | 1.000 | 100:119;20:293;3:292 | 100:175 | `{}` |
| 3 | Forward GraphFeature Centroid | 704 | 175 | 0.966 | 100:119;20:293;3:292 | 100:175 | `{}` |
| 3 | Forward STCG-v1 State Mapping | 704 | 175 | 0.954 | 100:119;20:293;3:292 | 100:175 | `{"0": "100", "1": "20", "2": "3"}` |
| 4 | Forward Train Majority | 704 | 175 | 0.000 | 100:119;20:293;3:292 | 100:175 | `{}` |
| 4 | Forward SensorMeanStd Centroid | 704 | 175 | 1.000 | 100:119;20:293;3:292 | 100:175 | `{}` |
| 4 | Forward GraphFeature Centroid | 704 | 175 | 0.966 | 100:119;20:293;3:292 | 100:175 | `{}` |
| 4 | Forward STCG-v1 State Mapping | 704 | 175 | 0.954 | 100:119;20:293;3:292 | 100:175 | `{"0": "20", "1": "3", "2": "100"}` |
