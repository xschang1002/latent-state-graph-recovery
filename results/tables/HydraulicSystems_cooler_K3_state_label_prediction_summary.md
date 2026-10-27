# Real State Label Prediction

This downstream diagnosis uses labels only after unsupervised graph-state inference.
Window labels are split stratified by external label; train windows define label mappings and held-out windows are evaluated.

## Aggregate

| Dataset | Label | Method | N | Accuracy | Balanced Acc. | Macro F1 | Accepted Rate | State Counts |
|---|---|---|---:|---:|---:|---:|---:|---|
| HydraulicSystems_cooler_K3 | cooler_condition | GraphFeature Centroid | 5 | 0.959 +/- 0.003 | 0.959 +/- 0.003 | 0.959 +/- 0.003 | 1.000 | `{"3": 5}` |
| HydraulicSystems_cooler_K3 | cooler_condition | STCG-v1 State Mapping | 5 | 0.948 +/- 0.004 | 0.948 +/- 0.004 | 0.948 +/- 0.004 | 1.000 | `{"3": 5}` |
| HydraulicSystems_cooler_K3 | cooler_condition | SensorMeanStd Centroid | 5 | 0.994 +/- 0.004 | 0.994 +/- 0.004 | 0.994 +/- 0.004 | 1.000 | `{"3": 5}` |
| HydraulicSystems_cooler_K3 | cooler_condition | Train Majority | 5 | 0.336 +/- 0.000 | 0.333 +/- 0.000 | 0.168 +/- 0.000 | 1.000 | `{"3": 5}` |

## By Seed

| Dataset | Seed | Label | Method | Train Windows | Test Windows | Accuracy | Balanced Acc. | Macro F1 | Train Counts | Test Counts | State Mapping |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---|---|
| HydraulicSystems_cooler_K3 | 0 | cooler_condition | Train Majority | 441 | 440 | 0.336 | 0.333 | 0.168 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems_cooler_K3 | 0 | cooler_condition | SensorMeanStd Centroid | 441 | 440 | 0.995 | 0.995 | 0.995 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems_cooler_K3 | 0 | cooler_condition | GraphFeature Centroid | 441 | 440 | 0.959 | 0.959 | 0.959 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems_cooler_K3 | 0 | cooler_condition | STCG-v1 State Mapping | 441 | 440 | 0.945 | 0.945 | 0.945 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{"0": "100", "1": "20", "2": "3"}` |
| HydraulicSystems_cooler_K3 | 1 | cooler_condition | Train Majority | 441 | 440 | 0.336 | 0.333 | 0.168 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems_cooler_K3 | 1 | cooler_condition | SensorMeanStd Centroid | 441 | 440 | 0.991 | 0.991 | 0.991 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems_cooler_K3 | 1 | cooler_condition | GraphFeature Centroid | 441 | 440 | 0.961 | 0.961 | 0.961 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems_cooler_K3 | 1 | cooler_condition | STCG-v1 State Mapping | 441 | 440 | 0.948 | 0.948 | 0.948 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{"0": "20", "1": "3", "2": "100"}` |
| HydraulicSystems_cooler_K3 | 2 | cooler_condition | Train Majority | 441 | 440 | 0.336 | 0.333 | 0.168 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems_cooler_K3 | 2 | cooler_condition | SensorMeanStd Centroid | 441 | 440 | 0.989 | 0.989 | 0.989 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems_cooler_K3 | 2 | cooler_condition | GraphFeature Centroid | 441 | 440 | 0.959 | 0.959 | 0.959 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems_cooler_K3 | 2 | cooler_condition | STCG-v1 State Mapping | 441 | 440 | 0.943 | 0.943 | 0.943 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{"0": "20", "1": "100", "2": "3"}` |
| HydraulicSystems_cooler_K3 | 3 | cooler_condition | Train Majority | 441 | 440 | 0.336 | 0.333 | 0.168 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems_cooler_K3 | 3 | cooler_condition | SensorMeanStd Centroid | 441 | 440 | 0.995 | 0.995 | 0.995 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems_cooler_K3 | 3 | cooler_condition | GraphFeature Centroid | 441 | 440 | 0.955 | 0.954 | 0.954 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems_cooler_K3 | 3 | cooler_condition | STCG-v1 State Mapping | 441 | 440 | 0.948 | 0.948 | 0.948 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{"0": "20", "1": "3", "2": "100"}` |
| HydraulicSystems_cooler_K3 | 4 | cooler_condition | Train Majority | 441 | 440 | 0.336 | 0.333 | 0.168 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems_cooler_K3 | 4 | cooler_condition | SensorMeanStd Centroid | 441 | 440 | 1.000 | 1.000 | 1.000 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems_cooler_K3 | 4 | cooler_condition | GraphFeature Centroid | 441 | 440 | 0.961 | 0.961 | 0.961 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{}` |
| HydraulicSystems_cooler_K3 | 4 | cooler_condition | STCG-v1 State Mapping | 441 | 440 | 0.955 | 0.954 | 0.954 | 100:148;20:147;3:146 | 100:148;20:146;3:146 | `{"0": "20", "1": "100", "2": "3"}` |

## Interpretation

- `STCG-v1 State Mapping` is the compact unsupervised graph-state signal.
- `GraphFeature Centroid` tests whether windowed graph-score features contain label information with supervised centroids.
- `SensorMeanStd Centroid` is a simple supervised sensor-feature ceiling, not a graph-state method.
