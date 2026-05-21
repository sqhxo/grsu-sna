"""Random graph generative models: Erdős–Rényi, Barabási–Albert, Watts–Strogatz."""

from __future__ import annotations

import random

import igraph as ig


def erdos_renyi(n: int, p: float | None = None, m: int | None = None,
                seed: int | None = None) -> ig.Graph:
    if seed is not None:
        random.seed(seed)
    if m is not None:
        return ig.Graph.Erdos_Renyi(n=n, m=m, directed=False, loops=False)
    if p is None:
        raise ValueError("erdos_renyi requires either p or m")
    return ig.Graph.Erdos_Renyi(n=n, p=p, directed=False, loops=False)


def barabasi_albert(n: int, m: int = 2, seed: int | None = None) -> ig.Graph:
    if seed is not None:
        random.seed(seed)
    return ig.Graph.Barabasi(n=n, m=m, directed=False)


def watts_strogatz(n: int, k: int = 4, p: float = 0.1,
                   seed: int | None = None) -> ig.Graph:
    if seed is not None:
        random.seed(seed)
    g = ig.Graph.Watts_Strogatz(dim=1, size=n, nei=k // 2, p=p)
    g.simplify()
    return g


def matched_to(reference: ig.Graph, model: str, seed: int | None = None) -> ig.Graph:
    """Create a random graph matching reference's node and edge count."""
    n = reference.vcount()
    m = reference.ecount()
    if model == "erdos_renyi":
        return erdos_renyi(n=n, m=m, seed=seed)
    if model == "barabasi_albert":
        avg_m = max(1, round(m / n))
        return barabasi_albert(n=n, m=avg_m, seed=seed)
    if model == "watts_strogatz":
        avg_degree = (2 * m) // n
        k = max(2, avg_degree if avg_degree % 2 == 0 else avg_degree + 1)
        return watts_strogatz(n=n, k=k, p=0.1, seed=seed)
    raise ValueError(f"Unknown model: {model}")
