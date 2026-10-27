"""Directed simple-graph nulls conditioned on each node's in/out degree."""

from __future__ import annotations

import numpy as np


def directed_degree_null(
    support: np.ndarray,
    draws: int,
    seed: int = 0,
    burn_in_per_edge: int = 100,
    thinning_per_edge: int = 10,
    *,
    diagnostics: dict | None = None,
) -> np.ndarray:
    """Sample a degree-preserving switch chain, including rejected proposals.

    Each proposal is an ordered pair of edge indices sampled independently with
    replacement (90%) or an ordered vertex triple sampled independently with
    replacement (10%). Repeated indices/vertices give rejected proposals. Pairs
    exchange destinations; isolated directed
    three-cycles reverse orientation. Self-loops and duplicate edges are rejected.
    Reverse proposals have the same probability, so the uniform distribution on
    simple graphs with the fixed labelled degree sequence is stationary. Keeping
    rejected proposals avoids bias from sampling only successful switches.

    Burn-in and thinning count proposals, not successful moves. Finite-chain
    output is an approximate conditional null, not independent exact samples.
    Three-cycle reversals allow movement between degree-equivalent orientations
    that two-edge switches alone cannot connect (e.g. a directed triangle).
    Optional diagnostics count attempted/accepted moves without consuming random
    numbers or changing the retained samples.
    """

    support = np.asarray(support)
    if support.ndim != 2 or support.shape[0] != support.shape[1]:
        raise ValueError("support must be a square adjacency matrix")
    if not np.isin(support, [0, 1]).all() or np.diag(support).any():
        raise ValueError("support must be binary with no self-loops")
    if draws < 1 or burn_in_per_edge < 0 or thinning_per_edge < 1:
        raise ValueError("Need positive draws/thinning and nonnegative burn-in")
    graph = support.astype(bool, copy=True)
    nodes = len(graph)
    edges = np.argwhere(graph)
    count = len(edges)
    samples = np.empty((draws, nodes, nodes), dtype=bool)
    counts = {phase: {move: {"attempted": 0, "accepted": 0}
                      for move in ("two_edge", "three_cycle")}
              for phase in ("burn_in", "sampling")}
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(counts)
    if count < 2:
        samples[:] = graph
        return samples
    edge_rows = np.full(graph.shape, -1, dtype=int)
    edge_rows[edges[:, 0], edges[:, 1]] = np.arange(count)
    rng = np.random.default_rng(seed)

    def advance(proposals: int, phase: str) -> None:
        triples = rng.random(proposals) < 0.1
        pairs = rng.integers(count, size=(proposals, 2))
        vertices = rng.integers(nodes, size=(proposals, 3))
        for step in range(proposals):
            counter = counts[phase]["three_cycle" if triples[step] else "two_edge"]
            counter["attempted"] += 1
            if triples[step]:
                a, b, c = vertices[step]
                if a == b or b == c or a == c:
                    continue
                old = ((a, b), (b, c), (c, a))
                new = ((b, a), (c, b), (a, c))
                if all(graph[u, v] for u, v in new):
                    old, new = new, old
                if not all(graph[u, v] for u, v in old) or any(graph[u, v] for u, v in new):
                    continue
                rows = [int(edge_rows[u, v]) for u, v in old]
            else:
                first, second = pairs[step]
                a, b = edges[first]
                c, d = edges[second]
                if a == d or c == b or graph[a, d] or graph[c, b]:
                    continue
                old, new = ((a, b), (c, d)), ((a, d), (c, b))
                rows = [first, second]
            for u, v in old:
                graph[u, v] = False
                edge_rows[u, v] = -1
            for row, (u, v) in zip(rows, new):
                graph[u, v] = True
                edge_rows[u, v] = row
                edges[row] = (u, v)
            counter["accepted"] += 1

    advance(burn_in_per_edge * count, "burn_in")
    for draw in range(draws):
        advance(thinning_per_edge * count, "sampling")
        samples[draw] = graph
    return samples
