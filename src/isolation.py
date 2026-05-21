"""Detection of isolated groups in a graph.

Three heuristics:
1. Small connected components (other than the giant one).
2. Communities with low conductance (few external edges).
3. k-core periphery (low-coreness nodes).
"""

from __future__ import annotations

from dataclasses import dataclass

import igraph as ig
import numpy as np


@dataclass
class IsolatedGroup:
    members: list[int]
    size: int
    conductance: float
    internal_density: float
    kind: str

    def __repr__(self) -> str:
        return (
            f"IsolatedGroup(kind={self.kind}, size={self.size}, "
            f"conductance={self.conductance:.3f}, density={self.internal_density:.3f})"
        )


def _conductance(graph: ig.Graph, members: list[int]) -> float:
    members_set = set(members)
    cut = 0
    internal = 0
    for v in members:
        for nb in graph.neighbors(v):
            if nb in members_set:
                internal += 1
            else:
                cut += 1
    volume = 2 * internal + cut
    return cut / volume if volume else 0.0


def _internal_density(graph: ig.Graph, members: list[int]) -> float:
    if len(members) < 2:
        return 0.0
    return float(graph.subgraph(members).density())


def disconnected_components(graph: ig.Graph,
                            max_fraction: float = 0.5) -> list[IsolatedGroup]:
    """Return all components except the largest (if it dominates the graph)."""
    components = graph.connected_components()
    sizes = components.sizes()
    if not sizes:
        return []
    largest_idx = int(np.argmax(sizes))
    n = graph.vcount()
    result: list[IsolatedGroup] = []
    for i, members in enumerate(components):
        if i == largest_idx and sizes[i] / n >= max_fraction:
            continue
        result.append(IsolatedGroup(
            members=list(members),
            size=len(members),
            conductance=0.0,
            internal_density=_internal_density(graph, list(members)),
            kind="disconnected_component",
        ))
    return result


def low_conductance_communities(graph: ig.Graph, labels: np.ndarray,
                                threshold: float = 0.2,
                                min_size: int = 3) -> list[IsolatedGroup]:
    """Communities with conductance below threshold."""
    result: list[IsolatedGroup] = []
    for c in np.unique(labels):
        members = np.where(labels == c)[0].tolist()
        if len(members) < min_size:
            continue
        cond = _conductance(graph, members)
        if cond <= threshold:
            result.append(IsolatedGroup(
                members=members,
                size=len(members),
                conductance=cond,
                internal_density=_internal_density(graph, members),
                kind="low_conductance_community",
            ))
    return sorted(result, key=lambda r: r.conductance)


def periphery_by_kcore(graph: ig.Graph,
                       max_coreness: int = 1) -> list[IsolatedGroup]:
    """Nodes with coreness ≤ max_coreness, grouped as periphery."""
    coreness = np.array(graph.coreness())
    members = np.where(coreness <= max_coreness)[0].tolist()
    if not members:
        return []
    return [IsolatedGroup(
        members=members,
        size=len(members),
        conductance=_conductance(graph, members),
        internal_density=_internal_density(graph, members),
        kind=f"k_core_periphery(<= {max_coreness})",
    )]


def find_all_isolated(graph: ig.Graph,
                      labels: np.ndarray | None = None) -> list[IsolatedGroup]:
    """Apply all three heuristics and return the combined list."""
    result = disconnected_components(graph)
    if labels is not None:
        result.extend(low_conductance_communities(graph, labels))
    result.extend(periphery_by_kcore(graph))
    return result
