# STCG mechanism research — independent analytical probes

This package contains new toy calculations and simulations motivated by the manuscript `paper_preprint_2026-10-21(4).pdf`. It does not reproduce or modify the author's repository.

## Contents

- `RESEARCH_NOTES.md`: Chinese research notes, proofs, assumptions, results, proposed discriminating interventions, primary-source links.
- `mechanism_probes.py`: deterministic/reproducible independent probes.
- `results/summary.json`: all executed summary outputs.
- `results/gate_replicates.json`: individual gate-probe outputs, including exact and approximate objectives.
- `environment.json`: actual package versions.
- `run_output.json`: stdout captured from the executed script.

## Run

```bash
python -m pip install numpy scipy scikit-learn threadpoolctl
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python mechanism_probes.py --out results --reps 200
```

The gate study uses 200 replicates per condition. The switched-regression study uses 100 paired replicates. The selection/value illustration uses 200,000 draws. The AUROC counterexample is an exact finite construction. The segmentation study enumerates all 64 contiguous partitions of seven observations; it does not invoke or test the user's PELT adapter.

The code distinguishes facts verified in these probes from hypotheses about the original implementation. No user data, raw ESS data, or original experiment outputs are included.
