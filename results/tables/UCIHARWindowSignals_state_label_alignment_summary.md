# Real State Label Alignment

This post-hoc diagnostic compares external window-majority labels with graph states inferred without label access.

## Aggregate

| Metric | Value |
|---|---:|
| Adjusted Rand Index | 0.000 +/- 0.000 |
| Aligned state accuracy | 0.396 +/- 0.000 |
| Mean label purity | 0.230 +/- 0.000 |
| True state count distribution | `{5: 5}` |
| Inferred state count distribution | `{1: 5}` |

## By Seed

| Dataset | Seed | Windows | True States | Inferred States | ARI | Aligned Acc. | Label Purity | True Counts | Inferred Counts | Mapping |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| UCIHARWindowSignals | 0 | 48 | 5 | 1 | 0.000 | 0.396 | 0.230 | LAYING:19;SITTING:4;STANDING:15;WALKING:9;WALKING_UPSTAIRS:1 | 0:48 | `{"0": "LAYING"}` |
| UCIHARWindowSignals | 1 | 48 | 5 | 1 | 0.000 | 0.396 | 0.230 | LAYING:19;SITTING:4;STANDING:15;WALKING:9;WALKING_UPSTAIRS:1 | 0:48 | `{"0": "LAYING"}` |
| UCIHARWindowSignals | 2 | 48 | 5 | 1 | 0.000 | 0.396 | 0.230 | LAYING:19;SITTING:4;STANDING:15;WALKING:9;WALKING_UPSTAIRS:1 | 0:48 | `{"0": "LAYING"}` |
| UCIHARWindowSignals | 3 | 48 | 5 | 1 | 0.000 | 0.396 | 0.230 | LAYING:19;SITTING:4;STANDING:15;WALKING:9;WALKING_UPSTAIRS:1 | 0:48 | `{"0": "LAYING"}` |
| UCIHARWindowSignals | 4 | 48 | 5 | 1 | 0.000 | 0.396 | 0.230 | LAYING:19;SITTING:4;STANDING:15;WALKING:9;WALKING_UPSTAIRS:1 | 0:48 | `{"0": "LAYING"}` |

## Decision

- Window-label mismatches after alignment: 145 of 240.
- A single inferred state with ARI near zero is evidence against real graph-state separation under the current settings.
