# Gate Identifiability Audit (2026-10-21)

This note documents why the score-space acceptance gate is not identifiable at
the main panel's six windows, and why the change-point gate of
`docs/change_point_gate_protocol_2026-10-21.md` uses the raw series instead.

## Data

- Eight gate-stress conditions from `docs/revision_gate_protocol_2026-09-13.md`,
  twenty seeds each (100–119).
- Main panels: stationary and switching VAR at `T` in {1000, 1500, 2000, 3000},
  window 300, stride 150, twenty seeds each (0–19).

## Diagnostics

For every fit: K-means improvement, temporal coherence, restart stability
(mean pairwise ARI over twenty restarts), silhouette, BIC difference for one
versus two diagonal Gaussian components on five PCs, gap statistic for K=1 vs
K=2 on a PCA-aligned uniform reference, a label-permutation coherence test, a
contiguous change-point scan with an i.i.d. row-permutation null, the same scan
with a stationary-bootstrap block null, and the two-state versus one-state Edge
AUC.

## Finding

At six to seven windows, none of the standard cluster-count diagnostics
separates the stationary false split from the true short switch:

| Statistic | Stationary, few windows | Short switch |
|---|---:|---:|
| K-means improvement | 0.268 | 0.237 |
| Restart stability (mean ARI) | 0.282 | 0.323 |
| Silhouette | 0.075 | 0.075 |
| BIC(1) − BIC(2) | +5.6 | +17.0 |
| Gap prefers two states | 0/20 | 0/20 |
| Coherence permutation p | 0.210 | 0.069 |
| Contiguous scan, i.i.d. null p | 0.125 | 0.056 |

The i.i.d. scan null is also anti-conservative for stationary series: its mean
p falls from 0.156 at T=1000 to 0.021 at T=3000 because window scores are
autocorrelated.  A stationary-bootstrap block null restores calibration but is
too conservative to retain the true short switch at these window counts
(switching p = 0.17–0.42).

The two-state max aggregation can still raise Edge AUC on a stationary series
(the false split sharpens scores), which is why rejecting it has a small
recovery cost.  The gate decision and the recovery metric therefore answer
different questions: the score gate cannot certify that regimes exist, and the
calibrated raw-series gate trades a small stationary AUC loss for not claiming
unsupported states.

## Files

- `results/tables/gate_audit/gate_diagnostics.csv`
- `results/tables/gate_audit/scan_gate_diagnostics.csv`
- `results/tables/gate_audit/block_scan_diagnostics.csv`
- `results/tables/gate_audit/window_count_diagnostics.csv`

The change-point gate itself is implemented in
`src/stcg/change_point_gate.py` and evaluated by
`scripts/run_change_point_gate.py` under the frozen protocol.
