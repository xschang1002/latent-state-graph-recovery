# ESS degree-null Monte Carlo expansion — 2026-10-23

Historical stage record: the hashes and layout checks below refer to the
5,000-draw expansion before the MLJ cleanup. The primary result CSVs remain
unchanged; current sampler diagnostics, prose, layout and package validation
are recorded in `audit/ess_mlj_cleanup_2026-10-23.md` and its JSON companion.

This stage increases the corrected ESS null calculation from 200 to 5,000 draws
per edge set and seed, using seeds 0--4. Seed 0 remains the fixed reference.
Existing uncommitted work was backed up under
`experiments/logs/ess_mc_5000_2026-10-23/before/`; the adjacent
`before_manifest.json` records SHA-256 hashes. No commit or staging change was
made. The historical freeze and unrelated source changes are retained.

## Protocol

Command:

```powershell
python scripts/run_ess_alignment.py --all-imputations --draws 5000 --null-seeds 0 1 2 3 4
python tools/build_revision2_tables.py
python tools/verify_revision2_numbers.py
```

Four imputations, four methods, two expert edge sets and five Monte Carlo seeds
produce 160 comparison rows. The reference CSV has the 32 rows from seed 0;
the sensitivity CSV retains all 160. Each seed has its own eight-test
forward-fill Holm family and its own descriptive combined 32-test correction.
Seeds repeat computation on the same dataset; p-values are not averaged, pooled,
or selected by significance.

The labelled in/out-degree null, 90% two-edge destination exchanges, 10% cycle
reversals, rejection self-transitions, burn-in 100|E| and thinning 10|E| are
unchanged. The finite chain remains approximate. The same null graph samples
are used across methods and imputations. The plus-one upper-tail p-value is
`(1 + exceedances) / 5001`. Exceedance counts are saved explicitly.

Fixed-score null AUROCs are computed by positive tied-rank sums, equivalent to
the scalar AUROC. Regression tests compare both calculations for tied, constant
and negative scores. This avoids sorting identical scores repeatedly. All
observed AUROCs/AUPRCs, prevalence values and refit counts match the 200-draw
stage exactly; PELT still detects and refits 23 disjoint segments.

## Results

Strong-edge lagged correlation in the forward-fill eight-test family passes
in all 5/5 seeds. Its adjusted p-values span 0.015997--0.036793, with reference
raw p=0.003199 and adjusted p=0.025595. Every other primary comparison stays
above 0.05 in every seed.

The combined 32-test result is less stable: linear-imputation strong-edge lagged
correlation passes in 4/5 seeds (range 0.019196--0.051190), while mean imputation
passes in 1/5. No comparison passes the combined family in all five seeds.
This replaces the 200-draw no-combined-rejection statement. Both passing and
threshold-crossing outcomes are retained.

| Null seed | Forward-fill raw p | Forward-fill Holm p (8) | Linear Holm p (32) | Mean Holm p (32) |
|---|---:|---:|---:|---:|
| 0 | 0.003199 | 0.025595 | 0.044791 | 0.080584 |
| 1 | 0.003999 | 0.031994 | 0.031994 | 0.099180 |
| 2 | 0.004199 | 0.033593 | 0.051190 | 0.092981 |
| 3 | 0.002000 | 0.015997 | 0.019196 | 0.043391 |
| 4 | 0.004599 | 0.036793 | 0.038392 | 0.086783 |

These are computational stability results conditional on a single hour of ESS
and the approximate degree null, not additional data replications or causal
graph validation. Five seeds do not establish stability for every possible seed
or resolve uncertainty in the data/model.

## Artifacts and validation

- Reference: `results/tables/ess_alignment.csv` (32 x 22, seed 0).
- All seeds: `results/tables/ess_alignment_seed_sensitivity.csv` (160 x 22).
- Preserved 200-draw stage: `results/revision2/ess_alignment_200_draws_seed0.csv`
  and `paper_preprint_2026-10-23.pdf`.
- Display: `results/revision2/tab_ess.tex` adds five-seed Holm ranges;
  `results/figures/rev2_ess.png` has alignment and primary seed-stability panels.
- Current PDF: `paper_preprint_2026-10-23_5000_draws.pdf`; the packaged
  `paper/main.pdf` is its identical counterpart.
- Current source/environment/result hashes: accompanying
  `audit/ess_monte_carlo_5000_2026-10-23.json`.
- Statistical-plan Amendment 9 records the protocol expansion and changed
  combined-family interpretation. The earlier audit/hash report remains a
  historical record of the 200-draw stage.

Validation actually completed:

- Full test suite: **266 passed**, zero failures/skips; three upstream SWIG
  deprecation warnings, 27.06 seconds. Twelve cases were added to the 254-case
  stage, covering the seed protocol, invalid arguments, separate Holm families,
  shared null samples, tied-rank equivalence, full result coverage and plotted data.
- **184** manuscript number checks pass; static manuscript validation has no errors.
- All nine dataset checksums match; historical summary/detail check is 49/49.
- Independent invocation using the defaults reproduces both result CSVs exactly,
  including every numeric cell, exceedance count and protocol field.
- Holm results for all five seeds were independently checked with statsmodels.
- The observed metrics and refit counts were compared to the archived 200-draw
  CSV with zero tolerance and match exactly.
- Tectonic produces 25 pages, with no overfull boxes or undefined references.
  ESS prose (page 17), Table 9 (page 20), and Figure 6 (page 21) were visually
  reviewed. Existing underfull/BibTeX and Windows BBL rerun warnings persist;
  they do not prevent compilation or leave unresolved references.

## Workflow attribution

The local `paper-writing`, `scientific-visualization` and `pdf` workflows were
used. Scientific-visualization attribution is retained from the earlier repair:
Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel and Aubrey M. Brueckner
(2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents*, [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
Metadata were verified in the earlier 2026-10-23 stage; no scientific citation
was added to the manuscript merely for this workflow attribution.
