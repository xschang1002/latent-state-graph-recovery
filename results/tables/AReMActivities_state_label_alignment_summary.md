# Real State Label Alignment

This post-hoc diagnostic compares external window-majority labels with graph states inferred without label access.

## Aggregate

| Metric | Value |
|---|---:|
| Adjusted Rand Index | 0.251 +/- 0.000 |
| Aligned state accuracy | 0.487 +/- 0.000 |
| Mean label purity | 0.987 +/- 0.000 |
| True state count distribution | `{6: 5}` |
| Inferred state count distribution | `{2: 5}` |

## By Seed

| Dataset | Seed | Windows | True States | Inferred States | ARI | Aligned Acc. | Label Purity | True Counts | Inferred Counts | Mapping |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| AReMActivities | 0 | 197 | 6 | 2 | 0.251 | 0.487 | 0.987 | bending1:22;bending2:19;cycling:48;lying:48;sitting:48;standing:12 | 0:51;1:146 | `{"0": "cycling", "1": "sitting"}` |
| AReMActivities | 1 | 197 | 6 | 2 | 0.251 | 0.487 | 0.987 | bending1:22;bending2:19;cycling:48;lying:48;sitting:48;standing:12 | 0:146;1:51 | `{"0": "sitting", "1": "cycling"}` |
| AReMActivities | 2 | 197 | 6 | 2 | 0.251 | 0.487 | 0.987 | bending1:22;bending2:19;cycling:48;lying:48;sitting:48;standing:12 | 0:51;1:146 | `{"0": "cycling", "1": "sitting"}` |
| AReMActivities | 3 | 197 | 6 | 2 | 0.251 | 0.487 | 0.987 | bending1:22;bending2:19;cycling:48;lying:48;sitting:48;standing:12 | 0:51;1:146 | `{"0": "cycling", "1": "sitting"}` |
| AReMActivities | 4 | 197 | 6 | 2 | 0.251 | 0.487 | 0.987 | bending1:22;bending2:19;cycling:48;lying:48;sitting:48;standing:12 | 0:51;1:146 | `{"0": "cycling", "1": "sitting"}` |

## Decision

- Window-label mismatches after alignment: 505 of 985.
- Non-collapsed inferred states: 5 of 5 seeds; use ARI and aligned accuracy above to judge whether they match the external labels.
