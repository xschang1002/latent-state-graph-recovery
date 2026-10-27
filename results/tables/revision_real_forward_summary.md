# Frozen forward diagnosis validation

Seeds are averaged within each origin. Ranges describe four dependent origins; no significance claim is made.

| Dataset / label | Method | Multi-state test origins | Balanced accuracy mean [range] | Macro F1 | Unseen fraction |
|---|---|---:|---|---:|---:|
| AReMActivities / activity | Graph Centroid | 4/4 | 0.282 [0.125, 0.441] | 0.126 | 0.634 |
| AReMActivities / activity | STCG Fixed K2 | 4/4 | 0.110 [0.000, 0.441] | 0.071 | 0.634 |
| AReMActivities / activity | STCG Train Silhouette K | 4/4 | 0.110 [0.000, 0.441] | 0.071 | 0.634 |
| AReMActivities / activity | Sensor Centroid | 4/4 | 0.438 [0.250, 0.500] | 0.202 | 0.634 |
| AReMActivities / activity | Sensor Logistic | 4/4 | 0.422 [0.250, 0.500] | 0.192 | 0.634 |
| AReMActivities / activity | Train Majority | 4/4 | 0.000 [0.000, 0.000] | 0.000 | 0.634 |
| HydraulicSystems / cooler_condition | Graph Centroid | 2/4 | 0.724 [0.500, 0.966] | 0.384 | 0.256 |
| HydraulicSystems / cooler_condition | STCG Fixed K2 | 2/4 | 0.250 [0.000, 0.500] | 0.141 | 0.256 |
| HydraulicSystems / cooler_condition | STCG Train Silhouette K | 2/4 | 0.298 [0.000, 0.500] | 0.165 | 0.256 |
| HydraulicSystems / cooler_condition | Sensor Centroid | 2/4 | 0.750 [0.500, 1.000] | 0.660 | 0.256 |
| HydraulicSystems / cooler_condition | Sensor Logistic | 2/4 | 0.750 [0.500, 1.000] | 0.660 | 0.256 |
| HydraulicSystems / cooler_condition | Train Majority | 2/4 | 0.125 [0.000, 0.500] | 0.099 | 0.256 |
| HydraulicSystems / valve_condition | Graph Centroid | 4/4 | 0.320 [0.250, 0.379] | 0.262 | 0.000 |
| HydraulicSystems / valve_condition | STCG Fixed K2 | 4/4 | 0.250 [0.250, 0.250] | 0.160 | 0.000 |
| HydraulicSystems / valve_condition | STCG Train Silhouette K | 4/4 | 0.250 [0.250, 0.250] | 0.160 | 0.000 |
| HydraulicSystems / valve_condition | Sensor Centroid | 4/4 | 0.268 [0.250, 0.321] | 0.194 | 0.000 |
| HydraulicSystems / valve_condition | Sensor Logistic | 4/4 | 0.265 [0.176, 0.391] | 0.202 | 0.000 |
| HydraulicSystems / valve_condition | Train Majority | 4/4 | 0.250 [0.250, 0.250] | 0.160 | 0.000 |
