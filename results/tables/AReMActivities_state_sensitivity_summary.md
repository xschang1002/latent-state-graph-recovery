# Real State Sensitivity

This diagnostic tests whether real-data single-state collapse is caused by the default auto-state acceptance threshold.

## Best Setting

| Setting | ARI | Aligned Acc. | Inferred States |
|---|---:|---:|---|
| forced_k3_w300 | 0.303 +/- 0.019 | 0.525 +/- 0.038 | `{3: 5}` |

## All Settings

| Setting | Window | Requested K | Auto | Min Improvement | ARI | Aligned Acc. | Inferred States |
|---|---:|---:|---|---:|---:|---:|---|
| forced_k3_w300 | 300 | 3 | False | 0.00 | 0.303 +/- 0.019 | 0.525 +/- 0.038 | `{3: 5}` |
| loose_auto_k3_w300 | 300 | 3 | True | 0.00 | 0.303 +/- 0.019 | 0.525 +/- 0.038 | `{3: 5}` |
| forced_k4_w300 | 300 | 4 | False | 0.00 | 0.290 +/- 0.050 | 0.535 +/- 0.028 | `{4: 5}` |
| loose_auto_k4_w300 | 300 | 4 | True | 0.00 | 0.290 +/- 0.050 | 0.535 +/- 0.028 | `{4: 5}` |
| forced_k4_w120 | 120 | 4 | False | 0.00 | 0.268 +/- 0.007 | 0.485 +/- 0.014 | `{4: 5}` |
| loose_auto_k4_w120 | 120 | 4 | True | 0.00 | 0.268 +/- 0.007 | 0.485 +/- 0.014 | `{4: 5}` |
| forced_k3_w120 | 120 | 3 | False | 0.00 | 0.254 +/- 0.018 | 0.515 +/- 0.013 | `{3: 5}` |
| loose_auto_k3_w120 | 120 | 3 | True | 0.00 | 0.254 +/- 0.018 | 0.515 +/- 0.013 | `{3: 5}` |
| default_auto_k2_w300 | 300 | 2 | True | 0.12 | 0.251 +/- 0.000 | 0.487 +/- 0.000 | `{2: 5}` |
| forced_k2_w300 | 300 | 2 | False | 0.00 | 0.251 +/- 0.000 | 0.487 +/- 0.000 | `{2: 5}` |
| loose_auto_k2_w300 | 300 | 2 | True | 0.00 | 0.251 +/- 0.000 | 0.487 +/- 0.000 | `{2: 5}` |
| forced_k2_w120 | 120 | 2 | False | 0.00 | 0.209 +/- 0.001 | 0.458 +/- 0.001 | `{2: 5}` |
| loose_auto_k2_w120 | 120 | 2 | True | 0.00 | 0.209 +/- 0.001 | 0.458 +/- 0.001 | `{2: 5}` |
