# Real State Label Alignment

This post-hoc diagnostic compares external window-majority labels with graph states inferred without label access.

## Aggregate

| Metric | Value |
|---|---:|
| Adjusted Rand Index | 0.735 +/- 0.006 |
| Aligned state accuracy | 0.902 +/- 0.002 |
| Mean label purity | 0.998 +/- 0.000 |
| True state count distribution | `{3: 5}` |
| Inferred state count distribution | `{2: 5}` |

## By Seed

| Dataset | Seed | Windows | True States | Inferred States | ARI | Aligned Acc. | Label Purity | True Counts | Inferred Counts | Mapping |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| HydraulicSystems | 0 | 617 | 3 | 2 | 0.742 | 0.904 | 0.998 | 100:32;20:293;3:292 | 0:316;1:301 | `{"0": "20", "1": "3"}` |
| HydraulicSystems | 1 | 617 | 3 | 2 | 0.741 | 0.904 | 0.998 | 100:32;20:293;3:292 | 0:321;1:296 | `{"0": "20", "1": "3"}` |
| HydraulicSystems | 2 | 617 | 3 | 2 | 0.730 | 0.901 | 0.998 | 100:32;20:293;3:292 | 0:317;1:300 | `{"0": "20", "1": "3"}` |
| HydraulicSystems | 3 | 617 | 3 | 2 | 0.730 | 0.901 | 0.998 | 100:32;20:293;3:292 | 0:300;1:317 | `{"0": "3", "1": "20"}` |
| HydraulicSystems | 4 | 617 | 3 | 2 | 0.730 | 0.901 | 0.998 | 100:32;20:293;3:292 | 0:300;1:317 | `{"0": "3", "1": "20"}` |

## Decision

- Window-label mismatches after alignment: 301 of 3085.
- Non-collapsed inferred states: 5 of 5 seeds; use ARI and aligned accuracy above to judge whether they match the external labels.
