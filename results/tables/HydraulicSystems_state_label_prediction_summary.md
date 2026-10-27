# Real State Label Prediction

This downstream diagnosis uses labels only after unsupervised graph-state inference.
Window labels are split stratified by external label; train windows define label mappings and held-out windows are evaluated.

## Aggregate

| Dataset | Label | Method | N | Accuracy | Balanced Acc. | Macro F1 | Accepted Rate | State Counts |
|---|---|---|---:|---:|---:|---:|---:|---|
| HydraulicSystems | cooler_condition | GraphFeature Centroid | 5 | 0.959 +/- 0.003 | 0.959 +/- 0.003 | 0.959 +/- 0.003 | 1.000 | `{"2": 5}` |
| HydraulicSystems | cooler_condition | STCG-v1 State Mapping | 5 | 0.657 +/- 0.002 | 0.656 +/- 0.002 | 0.538 +/- 0.002 | 1.000 | `{"2": 5}` |
| HydraulicSystems | cooler_condition | SensorMeanStd Centroid | 5 | 0.994 +/- 0.004 | 0.994 +/- 0.004 | 0.994 +/- 0.004 | 1.000 | `{"2": 5}` |
| HydraulicSystems | cooler_condition | Train Majority | 5 | 0.336 +/- 0.000 | 0.333 +/- 0.000 | 0.168 +/- 0.000 | 1.000 | `{"2": 5}` |

## By Seed

| Dataset | Seed | Label | Method | Train Windows | Test Windows | Accuracy | Balanced Acc. | Macro F1 | Train Counts | Test Counts | State Mapping |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---|---|
| HydraulicSystems | 0 | cooler_condition | Train Majority | 441 | 440 | 0.336 | 0.333 | 0.168 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems | 0 | cooler_condition | SensorMeanStd Centroid | 441 | 440 | 0.995 | 0.995 | 0.995 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems | 0 | cooler_condition | GraphFeature Centroid | 441 | 440 | 0.959 | 0.959 | 0.959 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems | 0 | cooler_condition | STCG-v1 State Mapping | 441 | 440 | 0.657 | 0.655 | 0.536 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{"0": "100", "1": "3"}` |
| HydraulicSystems | 1 | cooler_condition | Train Majority | 441 | 440 | 0.336 | 0.333 | 0.168 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems | 1 | cooler_condition | SensorMeanStd Centroid | 441 | 440 | 0.991 | 0.991 | 0.991 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems | 1 | cooler_condition | GraphFeature Centroid | 441 | 440 | 0.961 | 0.961 | 0.961 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems | 1 | cooler_condition | STCG-v1 State Mapping | 441 | 440 | 0.655 | 0.653 | 0.539 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{"0": "100", "1": "3"}` |
| HydraulicSystems | 2 | cooler_condition | Train Majority | 441 | 440 | 0.336 | 0.333 | 0.168 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems | 2 | cooler_condition | SensorMeanStd Centroid | 441 | 440 | 0.989 | 0.989 | 0.989 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems | 2 | cooler_condition | GraphFeature Centroid | 441 | 440 | 0.959 | 0.959 | 0.959 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems | 2 | cooler_condition | STCG-v1 State Mapping | 441 | 440 | 0.657 | 0.655 | 0.537 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{"0": "100", "1": "3"}` |
| HydraulicSystems | 3 | cooler_condition | Train Majority | 441 | 440 | 0.336 | 0.333 | 0.168 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems | 3 | cooler_condition | SensorMeanStd Centroid | 441 | 440 | 0.995 | 0.995 | 0.995 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems | 3 | cooler_condition | GraphFeature Centroid | 441 | 440 | 0.955 | 0.954 | 0.954 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems | 3 | cooler_condition | STCG-v1 State Mapping | 441 | 440 | 0.659 | 0.658 | 0.540 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{"0": "100", "1": "3"}` |
| HydraulicSystems | 4 | cooler_condition | Train Majority | 441 | 440 | 0.336 | 0.333 | 0.168 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems | 4 | cooler_condition | SensorMeanStd Centroid | 441 | 440 | 1.000 | 1.000 | 1.000 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems | 4 | cooler_condition | GraphFeature Centroid | 441 | 440 | 0.961 | 0.961 | 0.961 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems | 4 | cooler_condition | STCG-v1 State Mapping | 441 | 440 | 0.659 | 0.658 | 0.539 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{"0": "100", "1": "3"}` |

## Interpretation

- `STCG-v1 State Mapping` is the compact unsupervised graph-state signal.
- `GraphFeature Centroid` tests whether windowed graph-score features contain label information with supervised centroids.
- `SensorMeanStd Centroid` is a simple supervised sensor-feature ceiling, not a graph-state method.
