# Change-Point Calibrated Gate — Frozen Protocol (2026-10-21)

## Motivation

The score-space acceptance gate is not identifiable at the main panel's six
windows.  In the development audit, the current gate accepted a two-state
partition on 20/20 stationary series and on 20/20 switching series.  Standard
cluster-count diagnostics (stability, silhouette, BIC, gap, coherence
permutation) did not separate the two cases; a contiguous scan with an i.i.d.
permutation null was anti-conservative because window scores are
autocorrelated, and a block-bootstrap null was too conservative to retain the
true switch.

The structural assumption is a change in the VAR coefficients.  Testing that
assumption on the raw series has substantially more power than testing a
partition of six window scores.

## Frozen rule

- Statistic: maximum Chow-type F over contiguous splits of a global VAR(3)
  fit, with splits restricted to the central 60% of the sample.
- Calibration: parametric residual bootstrap from the global VAR fit,
  `draws = 99`, `seed = data_seed + 424242`.
- Scan step: 10 samples for `T <= 2000`, 50 samples for longer series.
- Gate: accept the two-state partition iff `p < 0.05`.
- On rejection, use the one-state fallback scores (`max` aggregation), exactly
  as the production fallback.
- The two-state scores are computed with `auto_states=False` so that the gate
  decision, not the old gate, selects between two-state and fallback scores.
- Implementation: `src/stcg/change_point_gate.py`; runner
  `scripts/run_change_point_gate.py`.
- This is an optional gate.  Production defaults and the frozen main results
  are unchanged.

## Development results (seeds that exposed the failure)

Main panel: seeds 0–19.  Gate stress: seeds 100–119.

| Group | Current accept | New accept | False split (new) | AUC current | AUC new | AUC one-state |
|---|---:|---:|---:|---:|---:|---:|
| Main stationary | 20/20 | 0/20 | 0/20 | 0.876 | 0.866 | 0.866 |
| Main switching | 20/20 | 20/20 | — | 0.812 | 0.812 | 0.798 |
| Stationary, few windows | 20/20 | 1/20 | 1/20 | 0.886 | 0.880 | 0.879 |
| Short switch | 20/20 | 20/20 | — | 0.741 | 0.741 | 0.736 |

The gate removes the false split at a cost of about 0.010 stationary Edge AUC.
That cost is real: the two-state `max` aggregation sharpens scores even when
the partition is arbitrary, so rejecting it is an interpretability gain, not a
recovery gain.  On switching data the gate preserves the two-state solution
and its recovery exactly.

## Validation plan (fresh seeds 200–219)

The rule and thresholds above are frozen before observing these seeds.

Pass criteria, fixed in advance:

1. Main stationary: new accept rate `<= 0.10`.
2. Stationary, few windows: new accept rate `<= 0.15`.
3. Main switching: new accept rate `>= 0.90` and mean Edge AUC no more than
   0.005 below the current gate.
4. Short switch: new accept rate `>= 0.90` and mean Edge AUC no more than
   0.005 below the current gate.

If all four pass, report the calibrated gate as a validated optional variant.
If any fails, report the failure and keep the current gate as default.

## Frozen hashes

- `src/stcg/change_point_gate.py`: `f7c087726172e50009999baceeed4db4f6eb90014843352c965e68b316f28222`
- `scripts/run_change_point_gate.py`: `de29e4ea1f2097a1424a40fe958bc42a9ebf24e156392dabde432dec1171036f`

## References

- Chow, G. C. (1960). Tests of equality between sets of coefficients in two
  linear regressions. *Econometrica*, 28(3), 591–605.
- Bai, J., & Perron, P. (1998). Estimating and testing linear models with
  multiple structural changes. *Econometrica*, 66(1), 47–78.
- Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the number of
  clusters in a data set via the gap statistic. *JRSS-B*, 63(2), 411–423.
- Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *JASA*,
  89(428), 1303–1313.
