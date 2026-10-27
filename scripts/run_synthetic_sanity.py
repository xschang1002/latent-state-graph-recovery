"""Run a small synthetic generation and lagged-correlation sanity check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stcg import edge_auc, generate_lagged_var, lagged_correlation_scores, precision_at_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["var", "nonlinear_var", "switching_var"], default="var")
    parser.add_argument("--n-nodes", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--edge-prob", type=float, default=0.18)
    parser.add_argument("--noise-scale", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = generate_lagged_var(
        n_nodes=args.n_nodes,
        timesteps=args.timesteps,
        max_lag=args.max_lag,
        edge_prob=args.edge_prob,
        noise_scale=args.noise_scale,
        seed=args.seed,
        nonlinear=args.kind == "nonlinear_var",
        switching=args.kind == "switching_var",
    )
    scores = lagged_correlation_scores(dataset.x, args.max_lag)
    truth = dataset.edge_truth

    summary = {
        "kind": args.kind,
        "seed": args.seed,
        "x_shape": list(dataset.x.shape),
        "truth_shape": list(truth.shape),
        "true_lagged_edges": int(truth.sum()),
        "edge_density": float(truth.mean()),
        "lagged_correlation_edge_auc": edge_auc(scores, truth),
        "precision_at_true_k": precision_at_k(scores, truth),
        "regime_count": int(np.unique(dataset.regimes).size),
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            x=dataset.x,
            coefficients=dataset.coefficients,
            edge_truth=dataset.edge_truth,
            regimes=dataset.regimes,
            scores=scores,
        )
        summary["output"] = str(args.output)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
