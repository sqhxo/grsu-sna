"""Structural graph characteristics: degree, clustering, path length, etc."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import igraph as ig
import numpy as np


@dataclass
class GraphStats:
    name: str
    n_nodes: int
    n_edges: int
    density: float
    avg_degree: float
    max_degree: int
    min_degree: int
    global_clustering: float
    avg_local_clustering: float
    avg_path_length: float
    diameter: int
    assortativity_degree: float
    n_components: int
    largest_component_fraction: float
    power_law_alpha: float | None
    power_law_xmin: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _power_law_fit(degrees: np.ndarray) -> tuple[float | None, float | None]:
    try:
        fit = ig.statistics.power_law_fit(list(degrees))
        return float(fit.alpha), float(fit.xmin)
    except Exception:
        return None, None


def compute_stats(graph: ig.Graph, name: str = "graph") -> GraphStats:
    g = graph.as_undirected() if graph.is_directed() else graph
    degrees = np.array(g.degree())

    components = g.connected_components()
    sizes = components.sizes()
    largest = max(sizes) if sizes else 0

    if largest > 1:
        giant = components.giant()
        try:
            apl = float(giant.average_path_length())
            diam = int(giant.diameter())
        except Exception:
            apl, diam = float("nan"), 0
    else:
        apl, diam = float("nan"), 0

    alpha, xmin = _power_law_fit(degrees[degrees > 0])

    return GraphStats(
        name=name,
        n_nodes=g.vcount(),
        n_edges=g.ecount(),
        density=float(g.density()),
        avg_degree=float(degrees.mean()) if len(degrees) else 0.0,
        max_degree=int(degrees.max()) if len(degrees) else 0,
        min_degree=int(degrees.min()) if len(degrees) else 0,
        global_clustering=float(g.transitivity_undirected(mode="zero")),
        avg_local_clustering=float(g.transitivity_avglocal_undirected(mode="zero")),
        avg_path_length=apl,
        diameter=diam,
        assortativity_degree=float(g.assortativity_degree()) if g.ecount() else 0.0,
        n_components=len(sizes),
        largest_component_fraction=largest / g.vcount() if g.vcount() else 0.0,
        power_law_alpha=alpha,
        power_law_xmin=xmin,
    )


def degree_distribution(graph: ig.Graph) -> tuple[np.ndarray, np.ndarray]:
    degrees = np.array(graph.degree())
    unique, counts = np.unique(degrees, return_counts=True)
    return unique, counts
