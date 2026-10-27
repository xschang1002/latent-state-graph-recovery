# Real State Label Prediction

This downstream diagnosis uses labels only after unsupervised graph-state inference.
Window labels are split stratified by external label; train windows define label mappings and held-out windows are evaluated.

## Aggregate

| Dataset | Label | Method | N | Accuracy | Balanced Acc. | Macro F1 | Accepted Rate | State Counts |
|---|---|---|---:|---:|---:|---:|---:|---|
| AReMActivities_K4 | activity | GraphFeature Centroid | 5 | 0.662 +/- 0.034 | 0.623 +/- 0.033 | 0.617 +/- 0.035 | 1.000 | `{"4": 5}` |
| AReMActivities_K4 | activity | STCG-v1 State Mapping | 5 | 0.502 +/- 0.052 | 0.421 +/- 0.043 | 0.350 +/- 0.044 | 1.000 | `{"4": 5}` |
| AReMActivities_K4 | activity | SensorMeanStd Centroid | 5 | 0.816 +/- 0.014 | 0.810 +/- 0.012 | 0.816 +/- 0.013 | 1.000 | `{"4": 5}` |
| AReMActivities_K4 | activity | Train Majority | 5 | 0.170 +/- 0.000 | 0.143 +/- 0.000 | 0.042 +/- 0.000 | 1.000 | `{"4": 5}` |

## By Seed

| Dataset | Seed | Label | Method | Train Windows | Test Windows | Accuracy | Balanced Acc. | Macro F1 | Train Counts | Test Counts | State Mapping |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---|---|
| AReMActivities_K4 | 0 | activity | Train Majority | 140 | 141 | 0.170 | 0.143 | 0.042 | bending1:11;bending2:9;cycling:24;lying:24;sitting:24;standing:24;walking:24 | bending1:11;bending2:10;cycling:24;lying:24;sitting:24;standing:24;walking:24 | `{}` |
| AReMActivities_K4 | 0 | activity | SensorMeanStd Centroid | 140 | 141 | 0.809 | 0.799 | 0.812 | bending1:11;bending2:9;cycling:24;lying:24;sitting:24;standing:24;walking:24 | bending1:11;bending2:10;cycling:24;lying:24;sitting:24;standing:24;walking:24 | `{}` |
| AReMActivities_K4 | 0 | activity | GraphFeature Centroid | 140 | 141 | 0.674 | 0.654 | 0.648 | bending1:11;bending2:9;cycling:24;lying:24;sitting:24;standing:24;walking:24 | bending1:11;bending2:10;cycling:24;lying:24;sitting:24;standing:24;walking:24 | `{}` |
| AReMActivities_K4 | 0 | activity | STCG-v1 State Mapping | 140 | 141 | 0.411 | 0.345 | 0.275 | bending1:11;bending2:9;cycling:24;lying:24;sitting:24;standing:24;walking:24 | bending1:11;bending2:10;cycling:24;lying:24;sitting:24;standing:24;walking:24 | `{"0": "standing", "1": "cycling", "2": "walking", "3": "lying"}` |
| AReMActivities_K4 | 1 | activity | Train Majority | 140 | 141 | 0.170 | 0.143 | 0.042 | bending1:11;bending2:9;cycling:24;lying:24;sitting:24;standing:24;walking:24 | bending1:11;bending2:10;cycling:24;lying:24;sitting:24;standing:24;walking:24 | `{}` |
| AReMActivities_K4 | 1 | activity | SensorMeanStd Centroid | 140 | 141 | 0.809 | 0.816 | 0.814 | bending1:11;bending2:9;cycling:24;lying:24;sitting:24;standing:24;walking:24 | bending1:11;bending2:10;cycling:24;lying:24;sitting:24;standing:24;walking:24 | `{}` |
| AReMActivities_K4 | 1 | activity | GraphFeature Centroid | 140 | 141 | 0.716 | 0.661 | 0.661 | bending1:11;bending2:9;cycling:24;lying:24;sitting:24;standing:24;walking:24 | bending1:11;bending2:10;cycling:24;lying:24;sitting:24;standing:24;walking:24 | `{}` |
| AReMActivities_K4 | 1 | activity | STCG-v1 State Mapping | 140 | 141 | 0.511 | 0.429 | 0.354 | bending1:11;bending2:9;cycling:24;lying:24;sitting:24;standing:24;walking:24 | bending1:11;bending2:10;cycling:24;lying:24;sitting:24;standing:24;walking:24 | `{"0": "standing", "1": "walking", "2": "cycling", "3": "sitting"}` |
| AReMActivities_K4 | 2 | activity | Train Majority | 140 | 141 | 0.170 | 0.143 | 0.042 | bending1:11;bending2:9;cycling:24;lying:24;sitting:24;standing:24;walking:24 | bending1:11;bending2:10;cycling:24;lying:24;sitting:24;standing:24;walking:24 | `{}` |
| AReMActivities_K4 | 2 | activity | SensorMeanStd Centroid | 140 | 141 | 0.823 | 0.812 | 0.819 | bending1:11;bending2:9;cycling:24;lying:24;sitting:24;standing:24;walking:24 | bending1:11;bending2:10;cycling:24;lying:24;sitting:24;standing:24;walking:24 | `{}` |
| AReMActivities_K4 | 2 | activity | GraphFeature Centroid | 140 | 141 | 0.652 | 0.605 | 0.590 | bending1:11;bending2:9;cycling:24;lying:24;sitting:24;standing:24;walking:24 | bending1:11;bending2:10;cycling:24;lying:24;sitting:24;standing:24;walking:24 | `{}` |
| AReMActivities_K4 | 2 | activity | STCG-v1 State Mapping | 140 | 141 | 0.539 | 0.452 | 0.382 | bending1:11;bending2:9;cycling:24;lying:24;sitting:24;standing:24;walking:24 | bending1:11;bending2:10;cycling:24;lying:24;sitting:24;standing:24;walking:24 | `{"0": "cycling", "1": "standing", "2": "walking", "3": "sitting"}` |
| AReMActivities_K4 | 3 | activity | Train Majority | 140 | 141 | 0.170 | 0.143 | 0.042 | bending1:11;bending2:9;cycling:24;lying:24;sitting:24;standing:24;walking:24 | bending1:11;bending2:10;cycling:24;lying:24;sitting:24;standing:24;walking:24 | `{}` |
| AReMActivities_K4 | 3 | activity | SensorMeanStd Centroid | 140 | 141 | 0.801 | 0.796 | 0.799 | bending1:11;bending2:9;cycling:24;lying:24;sitting:24;standing:24;walking:24 | bending1:11;bending2:10;cycling:24;lying:24;sitting:24;standing:24;walking:24 | `{}` |
| AReMActivities_K4 | 3 | activity | GraphFeature Centroid | 140 | 141 | 0.631 | 0.590 | 0.588 | bending1:11;bending2:9;cycling:24;lying:24;sitting:24;standing:24;walking:24 | bending1:11;bending2:10;cycling:24;lying:24;sitting:24;standing:24;walking:24 | `{}` |
| AReMActivities_K4 | 3 | activity | STCG-v1 State Mapping | 140 | 141 | 0.525 | 0.440 | 0.381 | bending1:11;bending2:9;cycling:24;lying:24;sitting:24;standing:24;walking:24 | bending1:11;bending2:10;cycling:24;lying:24;sitting:24;standing:24;walking:24 | `{"0": "cycling", "1": "walking", "2": "lying", "3": "standing"}` |
| AReMActivities_K4 | 4 | activity | Train Majority | 140 | 141 | 0.170 | 0.143 | 0.042 | bending1:11;bending2:9;cycling:24;lying:24;sitting:24;standing:24;walking:24 | bending1:11;bending2:10;cycling:24;lying:24;sitting:24;standing:24;walking:24 | `{}` |
| AReMActivities_K4 | 4 | activity | SensorMeanStd Centroid | 140 | 141 | 0.837 | 0.825 | 0.835 | bending1:11;bending2:9;cycling:24;lying:24;sitting:24;standing:24;walking:24 | bending1:11;bending2:10;cycling:24;lying:24;sitting:24;standing:24;walking:24 | `{}` |
| AReMActivities_K4 | 4 | activity | GraphFeature Centroid | 140 | 141 | 0.638 | 0.603 | 0.597 | bending1:11;bending2:9;cycling:24;lying:24;sitting:24;standing:24;walking:24 | bending1:11;bending2:10;cycling:24;lying:24;sitting:24;standing:24;walking:24 | `{}` |
| AReMActivities_K4 | 4 | activity | STCG-v1 State Mapping | 140 | 141 | 0.525 | 0.440 | 0.358 | bending1:11;bending2:9;cycling:24;lying:24;sitting:24;standing:24;walking:24 | bending1:11;bending2:10;cycling:24;lying:24;sitting:24;standing:24;walking:24 | `{"0": "sitting", "1": "standing", "2": "cycling", "3": "walking"}` |

## Interpretation

- `STCG-v1 State Mapping` is the compact unsupervised graph-state signal.
- `GraphFeature Centroid` tests whether windowed graph-score features contain label information with supervised centroids.
- `SensorMeanStd Centroid` is a simple supervised sensor-feature ceiling, not a graph-state method.
