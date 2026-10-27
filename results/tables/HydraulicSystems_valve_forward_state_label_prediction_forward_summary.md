# Forward Real State Label Prediction

Graph-state centroids and state-label mappings are fit only on chronological train windows.
Test windows start after the train split and are assigned to the train-fitted centroids.

## Aggregate

| Dataset | Label | Method | N | Accuracy | Balanced Acc. | Macro F1 | Accepted Rate | State Counts |
|---|---|---|---:|---:|---:|---:|---:|---|
| HydraulicSystems_valve_forward | valve_condition | Forward GraphFeature Centroid | 5 | 0.377 +/- 0.000 | 0.323 +/- 0.000 | 0.292 +/- 0.000 | 1.000 | `{"4": 5}` |
| HydraulicSystems_valve_forward | valve_condition | Forward STCG-v1 State Mapping | 5 | 0.349 +/- 0.000 | 0.250 +/- 0.000 | 0.129 +/- 0.000 | 1.000 | `{"4": 5}` |
| HydraulicSystems_valve_forward | valve_condition | Forward SensorMeanStd Centroid | 5 | 0.349 +/- 0.000 | 0.250 +/- 0.000 | 0.129 +/- 0.000 | 1.000 | `{"4": 5}` |
| HydraulicSystems_valve_forward | valve_condition | Forward Train Majority | 5 | 0.349 +/- 0.000 | 0.250 +/- 0.000 | 0.129 +/- 0.000 | 1.000 | `{"4": 5}` |

## By Seed

| Seed | Method | Train Windows | Test Windows | Balanced Acc. | Train Counts | Test Counts | State Mapping |
|---:|---|---:|---:|---:|---|---|---|
| 0 | Forward Train Majority | 704 | 175 | 0.250 | 100:395;73:106;80:104;90:99 | 100:61;73:36;80:40;90:38 | `{}` |
| 0 | Forward SensorMeanStd Centroid | 704 | 175 | 0.250 | 100:395;73:106;80:104;90:99 | 100:61;73:36;80:40;90:38 | `{}` |
| 0 | Forward GraphFeature Centroid | 704 | 175 | 0.323 | 100:395;73:106;80:104;90:99 | 100:61;73:36;80:40;90:38 | `{}` |
| 0 | Forward STCG-v1 State Mapping | 704 | 175 | 0.250 | 100:395;73:106;80:104;90:99 | 100:61;73:36;80:40;90:38 | `{"0": "100", "1": "100", "2": "100", "3": "100"}` |
| 1 | Forward Train Majority | 704 | 175 | 0.250 | 100:395;73:106;80:104;90:99 | 100:61;73:36;80:40;90:38 | `{}` |
| 1 | Forward SensorMeanStd Centroid | 704 | 175 | 0.250 | 100:395;73:106;80:104;90:99 | 100:61;73:36;80:40;90:38 | `{}` |
| 1 | Forward GraphFeature Centroid | 704 | 175 | 0.323 | 100:395;73:106;80:104;90:99 | 100:61;73:36;80:40;90:38 | `{}` |
| 1 | Forward STCG-v1 State Mapping | 704 | 175 | 0.250 | 100:395;73:106;80:104;90:99 | 100:61;73:36;80:40;90:38 | `{"0": "100", "1": "100", "2": "100", "3": "100"}` |
| 2 | Forward Train Majority | 704 | 175 | 0.250 | 100:395;73:106;80:104;90:99 | 100:61;73:36;80:40;90:38 | `{}` |
| 2 | Forward SensorMeanStd Centroid | 704 | 175 | 0.250 | 100:395;73:106;80:104;90:99 | 100:61;73:36;80:40;90:38 | `{}` |
| 2 | Forward GraphFeature Centroid | 704 | 175 | 0.323 | 100:395;73:106;80:104;90:99 | 100:61;73:36;80:40;90:38 | `{}` |
| 2 | Forward STCG-v1 State Mapping | 704 | 175 | 0.250 | 100:395;73:106;80:104;90:99 | 100:61;73:36;80:40;90:38 | `{"0": "100", "1": "100", "2": "100", "3": "100"}` |
| 3 | Forward Train Majority | 704 | 175 | 0.250 | 100:395;73:106;80:104;90:99 | 100:61;73:36;80:40;90:38 | `{}` |
| 3 | Forward SensorMeanStd Centroid | 704 | 175 | 0.250 | 100:395;73:106;80:104;90:99 | 100:61;73:36;80:40;90:38 | `{}` |
| 3 | Forward GraphFeature Centroid | 704 | 175 | 0.323 | 100:395;73:106;80:104;90:99 | 100:61;73:36;80:40;90:38 | `{}` |
| 3 | Forward STCG-v1 State Mapping | 704 | 175 | 0.250 | 100:395;73:106;80:104;90:99 | 100:61;73:36;80:40;90:38 | `{"0": "100", "1": "100", "2": "100", "3": "100"}` |
| 4 | Forward Train Majority | 704 | 175 | 0.250 | 100:395;73:106;80:104;90:99 | 100:61;73:36;80:40;90:38 | `{}` |
| 4 | Forward SensorMeanStd Centroid | 704 | 175 | 0.250 | 100:395;73:106;80:104;90:99 | 100:61;73:36;80:40;90:38 | `{}` |
| 4 | Forward GraphFeature Centroid | 704 | 175 | 0.323 | 100:395;73:106;80:104;90:99 | 100:61;73:36;80:40;90:38 | `{}` |
| 4 | Forward STCG-v1 State Mapping | 704 | 175 | 0.250 | 100:395;73:106;80:104;90:99 | 100:61;73:36;80:40;90:38 | `{"0": "100", "1": "100", "2": "100", "3": "100"}` |
