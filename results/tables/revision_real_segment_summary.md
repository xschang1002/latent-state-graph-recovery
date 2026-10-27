# Label-independent fixed-segment validation

All eight selected sensor segments are retained, including no-transition segments and resource-limited failures. Hydraulic label columns reuse the same fitted states. ARI on a single true state is not informative evidence of recovery.

| Dataset / label | Segment | Method | Status | True sample/window K | Sample ARI | Window ARI | Inferred K |
|---|---:|---|---|---|---:|---:|---:|
| HydraulicSystems / cooler_condition | 0 | STCG Fixed K2 | ok | 1/1 | 0.000 | 0.000 | 2 |
| HydraulicSystems / valve_condition | 0 | STCG Fixed K2 | ok | 4/3 | 0.380 | 0.091 | 2 |
| HydraulicSystems / cooler_condition | 0 | Sensor KMeans K2 | ok | 1/1 | 0.000 | 0.000 | 2 |
| HydraulicSystems / valve_condition | 0 | Sensor KMeans K2 | ok | 4/3 | 0.207 | 0.286 | 2 |
| HydraulicSystems / cooler_condition | 0 | Regime-PCMCI K2 | timeout | 1/1 | nan | nan |  |
| HydraulicSystems / valve_condition | 0 | Regime-PCMCI K2 | timeout | 4/3 | nan | nan |  |
| HydraulicSystems / cooler_condition | 1 | STCG Fixed K2 | ok | 1/1 | 0.000 | 0.000 | 2 |
| HydraulicSystems / valve_condition | 1 | STCG Fixed K2 | ok | 1/1 | 0.000 | 0.000 | 2 |
| HydraulicSystems / cooler_condition | 1 | Sensor KMeans K2 | ok | 1/1 | 0.000 | 0.000 | 2 |
| HydraulicSystems / valve_condition | 1 | Sensor KMeans K2 | ok | 1/1 | 0.000 | 0.000 | 2 |
| HydraulicSystems / cooler_condition | 1 | Regime-PCMCI K2 | timeout | 1/1 | nan | nan |  |
| HydraulicSystems / valve_condition | 1 | Regime-PCMCI K2 | timeout | 1/1 | nan | nan |  |
| HydraulicSystems / cooler_condition | 2 | STCG Fixed K2 | ok | 1/1 | 0.000 | 0.000 | 2 |
| HydraulicSystems / valve_condition | 2 | STCG Fixed K2 | ok | 3/2 | -0.001 | -0.154 | 2 |
| HydraulicSystems / cooler_condition | 2 | Sensor KMeans K2 | ok | 1/1 | 0.000 | 0.000 | 2 |
| HydraulicSystems / valve_condition | 2 | Sensor KMeans K2 | ok | 3/2 | 0.027 | -0.154 | 2 |
| HydraulicSystems / cooler_condition | 2 | Regime-PCMCI K2 | timeout | 1/1 | nan | nan |  |
| HydraulicSystems / valve_condition | 2 | Regime-PCMCI K2 | timeout | 3/2 | nan | nan |  |
| HydraulicSystems / cooler_condition | 3 | STCG Fixed K2 | ok | 1/1 | 0.000 | 0.000 | 2 |
| HydraulicSystems / valve_condition | 3 | STCG Fixed K2 | ok | 3/3 | 0.154 | -0.364 | 2 |
| HydraulicSystems / cooler_condition | 3 | Sensor KMeans K2 | ok | 1/1 | 0.000 | 0.000 | 2 |
| HydraulicSystems / valve_condition | 3 | Sensor KMeans K2 | ok | 3/3 | 0.392 | 0.091 | 2 |
| HydraulicSystems / cooler_condition | 3 | Regime-PCMCI K2 | timeout | 1/1 | nan | nan |  |
| HydraulicSystems / valve_condition | 3 | Regime-PCMCI K2 | timeout | 3/3 | nan | nan |  |
| AReMActivities / activity | 0 | STCG Fixed K2 | ok | 1/1 | 0.000 | 0.000 | 2 |
| AReMActivities / activity | 0 | Sensor KMeans K2 | ok | 1/1 | 0.000 | 0.000 | 2 |
| AReMActivities / activity | 0 | Regime-PCMCI K2 | timeout | 1/1 | nan | nan |  |
| AReMActivities / activity | 1 | STCG Fixed K2 | ok | 1/1 | 0.000 | 0.000 | 2 |
| AReMActivities / activity | 1 | Sensor KMeans K2 | ok | 1/1 | 0.000 | 0.000 | 2 |
| AReMActivities / activity | 1 | Regime-PCMCI K2 | timeout | 1/1 | nan | nan |  |
| AReMActivities / activity | 2 | STCG Fixed K2 | ok | 1/1 | 0.000 | 0.000 | 2 |
| AReMActivities / activity | 2 | Sensor KMeans K2 | ok | 1/1 | 0.000 | 0.000 | 2 |
| AReMActivities / activity | 2 | Regime-PCMCI K2 | timeout | 1/1 | nan | nan |  |
| AReMActivities / activity | 3 | STCG Fixed K2 | ok | 1/1 | 0.000 | 0.000 | 2 |
| AReMActivities / activity | 3 | Sensor KMeans K2 | ok | 1/1 | 0.000 | 0.000 | 2 |
| AReMActivities / activity | 3 | Regime-PCMCI K2 | timeout | 1/1 | nan | nan |  |
