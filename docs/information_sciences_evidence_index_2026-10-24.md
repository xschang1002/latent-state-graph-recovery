# Internal protocol and evidence index — 24 October 2026

Paths below are relative to the extracted `information_sciences_reproducibility.zip` root. This index makes the two new simulation studies and their retained internal formation records inspectable. It does **not** claim public preregistration, external timestamping or an independently witnessed freeze. The earlier exploratory results and ESS analysis remain separate evidence blocks.

## Protocols, independent units and inference

| | Study A: paired duration confirmation | Study B: schedule robustness |
|---|---|---|
| Protocol | `results/information_sciences/duration_confirmation/protocol.json`; identical copy `results/information_sciences/duration_confirmation_protocol.json` | `results/information_sciences/schedule_robustness/schedule_robustness_protocol.json` |
| New repeats | 100, seeds 50000–50099 | 50, seeds 60000–60049 |
| Conditions | D=300/600; T=1200, h=300, stride=150, M=7; Lasso and auxiliary OLS | D=300/600 × fixed phase/random initial phase/irregular; T=3600, h=300, stride=150, M=23; Lasso |
| Paired unit | An independently redrawn mechanism, initial history and innovation array, shared across both durations and scorers | The same independent unit, shared across all six cells |
| Test families | Four native-Lasso contrasts alone form primary Holm4; twelve forced/OLS auxiliary contrasts report raw p without primary adjustment | Six forced−raw tests form secondary Holm6; three directly paired duration differences Γ form a separate secondary Holm3 |
| Other estimates | OLS, forced and decomposition diagnostics are auxiliary | Native output, gate acceptance, accepted-only effects and factors are descriptive |

Both studies use two-sided paired-repeat t tests where specified, ordinary pointwise percentile-bootstrap intervals (10,000 draws; seed 20261024), and one BLAS/OpenMP thread. Edges, overlapping windows and random partitions are not independent repeats. Accepted-only B means have no subgroup interval/test. The fixed planned counts are retained without replacement or added seeds.

The human analysis plan is `analysis/information_sciences_protocol_2026-10-24.md` (SHA-256 `1d0301fcd27d195f7f5d23c805c3ddf066fb627a62ea2811c505aa10e591b8bc`). This is the plan referenced by A's retained freeze record, rather than the older `analysis/statistical_plan.md`.

- A protocol SHA-256: `b75b0de47b1e380d063c1cee44c858befc5e6651150d2bc2a7ebd1d0fa030957`.
- B protocol SHA-256: `7096be72312d8916d0d2b83a52675aae8a031c81e10593c3afe6e346c6eb5439`.
- Shared helper SHA-256: `25ea666df3ed074288d544505643b5afbf3c1850c71c3c625e8c8ff05c2bb424`.

The machine index `results/information_sciences/provenance/evidence_index.json` binds these protocols, the plan, frozen source files, retained log copies and the two output seals by full SHA-256. The protocols supply exact defaults, source hashes, RNG addresses, missingness policies and comparison definitions. A uses PCG64 via `SeedSequence([repeat_seed,domain])` with mechanism/init/innovation domains 0/1/2. B derives domain-separated SHA-256 seeds under `stcg_information_sciences_schedule_v1`, shares base inputs across cells and independently addresses schedules by duration and type. Fixed schedules start at full-time zero before burn-in removal; random phase and inclusive irregular integer durations are defined in the B protocol. The original one-pass stabilisation is unchanged; actual radii are retained.

## Where predictions, diagnostics and estimates are retained

All following record patterns are relative to the corresponding study output directory above; braces denote the documented alternatives, not literal filenames.

**Study A:** `records/repeat_{seed}_inputs.npz` stores shared coefficients/support, initial history and innovations; `records/repeat_{seed}_D{300|600}_data.npz` stores simulated observations and truth. `records/repeat_{seed}_D{300|600}_{lasso|ols}.npz` retains scores, fitted maps, labels and emitted predictions; its `.json` companion retains diagnostics, including OLS ranks/singular values and warnings. `records/repeat_{seed}.json` binds the repeat files and method rows. `detail.csv`, `summary.csv` and `contrasts.csv` hold the complete row-level and aggregate results. `environment.json` records versions; `COMPLETE.json` binds the output directory. All 100 repeats and 400 condition–scorer fits succeeded; 2,400 method rows and 11,928 exact random-partition scores are retained. These fit/partition counts do not add independent simulation units.

**Study B:** `inputs/repeat_{seed}.npz` and `.json` retain the shared base inputs and checksums. `parts/repeat_{seed}_D{300|600}_{fixed_phase|random_phase|irregular}.npz` retains full/observed schedules, saved observed series, truth, scores, candidate maps/labels, per-window purity and all emitted predictions; its `.json` companion contains method rows, fit diagnostics, sampled schedule segments/truncations, actual/centre switches and window records. Burn-in series values are not retained. Full summaries are `schedule_robustness_detail.csv`, `schedule_robustness_diagnostics.csv`, `schedule_robustness_windows.csv`, `schedule_robustness_paired.csv`, `schedule_robustness_secondary_tests.csv`, `schedule_robustness_method_summary.csv` and `schedule_robustness_gate_summary.csv`. `schedule_robustness_outputs.json` seals these files and records. All 50 repeats and 300 condition fits succeeded; 1,500 method rows and 6,900 window rows are retained.

## What the retained formation records establish

Eight original internal records are copied to `results/information_sciences/provenance/`. `export_manifest.json` gives each original relative location, original SHA-256, exported location/SHA-256 and normalization status. The selected records contain no absolute machine paths, so all copies preserve the original bytes. Log extensions are normalized to `.txt` only; original names remain in the manifest. Original ignored logs were not changed. `chronology.json` distinguishes immediate internal clock records from observations derived from file modification times and converts original UTC values to Asia/Shanghai local dates.

| Study / exported source | Retained evidence | Interpretation and limit |
|---|---|---|
| A / `A_freeze_stage.json` | Launcher clocks immediately before and after protocol freeze: 2026-10-23 20:03:25.183092 / 20:03:25.191091 UTC (2026-10-24 04:03:25 local); plan/protocol/source hashes; zero simulations in the freeze launcher | An internal program-order record around a freeze-only operation. The subsequent runner was invoked separately. No external clock attestation was retained. |
| A / `A_completion_stage.json` | First saved input mtime 20:03:39.128937 UTC; COMPLETE mtime 20:04:13.567755; post-verification record assembled at 20:05:54.608049 | The first two times were extracted **post hoc from filesystem mtimes**, not emitted as execution events. They support the internal chronology but are insufficient alone to prove precedence. The last time is record assembly, not exact verifier completion. |
| A / `A_run_stdout.txt`, `A_verify_report.txt` | Progress through the fixed 100 repeats, reported comparisons and retained-output replay counts | No per-line execution timestamps. The QA report recomputes cached results without fitting. |
| B / `B_execution_stage.json` | Prelaunch clock 2026-10-23 20:27:19.725369 UTC (2026-10-24 04:27:19 local); output directory observed to contain only the hashed frozen protocol; fixed count/threads; later exit/status observation 20:33:02.085727 UTC | No separate B freeze-event UTC is retained. This record shows the protocol already present by prelaunch and was augmented after execution. The later time is an observation of post-run status, not exact process completion. |
| B / `B_run_stdout.txt`, `B_run_stderr.txt`, `B_verify_report.json` | Progress for seeds 60000–60049; exit code 0; empty stderr; cached replay of 300 cells, 1,500 method rows and seven summaries | Together with the protocol/source and output hashes, this is inspectable local execution evidence, not public registration or independent certification. |

The runner source enforces the same ordering: A checks the source-locked protocol and writes its output copy before `run_repeat` creates saved inputs; B calls `ensure_protocol` before any `run_repeat`. Current source inspection supports interpretation of the retained records. A hash establishes byte identity, not when those bytes existed; file timestamps can be altered. These limitations apply even when every closure check passes.

## Reproduction entry points and exact manuscript locations

The default audit commands below introduce no new stochastic inputs or scorer fits. Study A's verifier deterministically replays its saved inputs to compare the recorded observations; Study B's verifier checks retained arrays without resimulating them:

```text
python -B tools/verify_duration_confirmation.py
python -B tools/verify_schedule_robustness.py
python -B scripts/reproduce_paper.py --verify
```

`python -B scripts/reproduce_paper.py --rebuild` regenerates display tables/figures from saved outputs; it writes presentation files but does not run new fits. The presentation builders are `tools/build_information_sciences_assets.py` and `tools/build_information_sciences_supplement.py`. B's `--aggregate` similarly writes summaries from saved cases and is not a read-only verifier. A/B fresh simulation entry points are `scripts/run_duration_confirmation.py --run-protocol` and `scripts/run_schedule_robustness.py --run`; they are separate, opt-in operations and should use new output directories. Freeze-only switches are `--freeze-protocol NEW_PROTOCOL.json` for A and `--freeze --output-dir NEW_DIRECTORY` for B. Neither freeze-only command simulates data. Consult each runner's `--help`; do not overwrite the retained study directories.

| Claim/detail | Exact supplementary location (current S numbering) |
|---|---|
| MSE risk; gate proof; window-mixing/denoising; Brier/order interventions | S3 (`sec:theory`); S4.1 (`sec:gate-geometry`); S4.2 (`sec:window-average`); S4.5 (`sec:value-vs-use`) |
| Earlier full controls and segmentation/refit/timing/penalty | S5, Table S1/Figure S2; S7, Tables S7–S10/Figure S6 (`sec:baselines`); S10, Table S13 (`app:penalty`) |
| Earlier accepted subsets, GMM numerics and coverage | S12, Tables S16–S18 (`app:temporal-grid`) |
| ESS imputation and reference limitations; null target/proposals/chain diagnostics | S8, Table S11/Figure S7 (`sec:real`); S13, Table S19/Figure S8 (`app:ess-null`) |
| A/B protocol, paired unit, RNG and inference | S14 (`app:frozen-studies`) |
| All sixteen A estimates, including twelve auxiliary contrasts | S14, Table S20 (`tab:supp-a-contrasts`) |
| B gate denominators, accepted-only effects and J/C/JC factors | S14, Table S21 (`tab:supp-b-gate-factors`) |
| OLS absolute scores, full fit/rank/warning coverage | S14, Table S22 (`tab:supp-fit-diagnostics`) |
| Earlier provenance; reproduction/data access | S15 (`app:provenance`); S16 (`app:reproduction`) |

The ESS table displays selected/reference values and seed ranges, rather than every seed row. Full retained sensitivity and chain rows are `results/tables/ess_alignment_seed_sensitivity.csv`, `results/tables/ess_null_diagnostics.csv` and `results/tables/ess_null_traces.csv`.

The post hoc generator/path checks are in S14.1, Table S23 (`app:generator-audit`, `tab:generator-diagnostics`). `tools/audit_generator_outputs.py` reads retained coefficients/observations and writes `results/information_sciences/generator_audit/`; its summary and per-path CSVs give final radii, finite counts, full observed maxima and first/last-300-sample RMS ratios. No cutoff, new simulation, scorer fit or sample exclusion is introduced. The `--rebuild` entry also reconstructs this one added diagnostic table.

For package closure, first check the archive's outer SHA manifest. Then check `evidence_index.json` hashes, both protocol source maps, both output seals and their repeat/case hashes; require the exported evidence paths and supplementary labels to exist. Original log paths in `export_manifest.json` document origin and are **not** required archive members. Verify cached metrics/factors separately using the read-only commands. Manuscript display numbering may be cross-checked against its source labels; byte consistency does not itself validate claims or turn internal records into prospective public registration.
