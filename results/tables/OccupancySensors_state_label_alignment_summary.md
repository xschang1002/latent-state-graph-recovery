# Real State Label Alignment

This post-hoc diagnostic compares external window-majority labels with graph states inferred without label access.

## Aggregate

| Metric | Value |
|---|---:|
| Adjusted Rand Index | 0.000 +/- 0.000 |
| Aligned state accuracy | 0.737 +/- 0.000 |
| Mean label purity | 0.888 +/- 0.000 |
| True state count distribution | `{2: 5}` |
| Inferred state count distribution | `{1: 5}` |

## By Seed

| Dataset | Seed | Windows | True States | Inferred States | ARI | Aligned Acc. | Label Purity | True Counts | Inferred Counts | Mapping |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| OccupancySensors | 0 | 95 | 2 | 1 | 0.000 | 0.737 | 0.888 | 0:70;1:25 | 0:95 | `{"0": "0"}` |
| OccupancySensors | 1 | 95 | 2 | 1 | 0.000 | 0.737 | 0.888 | 0:70;1:25 | 0:95 | `{"0": "0"}` |
| OccupancySensors | 2 | 95 | 2 | 1 | 0.000 | 0.737 | 0.888 | 0:70;1:25 | 0:95 | `{"0": "0"}` |
| OccupancySensors | 3 | 95 | 2 | 1 | 0.000 | 0.737 | 0.888 | 0:70;1:25 | 0:95 | `{"0": "0"}` |
| OccupancySensors | 4 | 95 | 2 | 1 | 0.000 | 0.737 | 0.888 | 0:70;1:25 | 0:95 | `{"0": "0"}` |

## Decision

- Window-label mismatches after alignment: 125 of 475.
- A single inferred state with ARI near zero is evidence against real graph-state separation under the current settings.
