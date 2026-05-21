"""Community detection algorithms with a uniform interface.

Each algorithm takes an `igraph.Graph` and returns an `np.ndarray` of cluster
labels per node — compatible with the validation framework.
"""

from __future__ import annotations

import random

import igraph as ig
import leidenalg as la
import numpy as np


def louvain(graph: ig.Graph, resolution: float = 1.0,
            seed: int | None = 42) -> np.ndarray:
    partition = la.find_partition(
        graph,
        la.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )
    return np.array(partition.membership)


def louvain_classic(graph: ig.Graph) -> np.ndarray:
    return np.array(graph.community_multilevel().membership)


def leiden(graph: ig.Graph, resolution: float = 1.0,
           seed: int | None = 42) -> np.ndarray:
    partition = la.find_partition(
        graph, la.ModularityVertexPartition, seed=seed,
    )
    return np.array(partition.membership)


def label_propagation(graph: ig.Graph, seed: int | None = 42) -> np.ndarray:
    if seed is not None:
        random.seed(seed)
    return np.array(graph.community_label_propagation().membership)


def girvan_newman(graph: ig.Graph, n_clusters: int | None = None) -> np.ndarray:
    dendro = graph.community_edge_betweenness(directed=False)
    clustering = (dendro.as_clustering(n=n_clusters)
                  if n_clusters is not None else dendro.as_clustering())
    return np.array(clustering.membership)


def fast_greedy(graph: ig.Graph) -> np.ndarray:
    return np.array(graph.community_fastgreedy().as_clustering().membership)


def infomap(graph: ig.Graph, seed: int | None = 42) -> np.ndarray:
    return np.array(graph.community_infomap(trials=10).membership)


def walktrap(graph: ig.Graph, steps: int = 4) -> np.ndarray:
    return np.array(graph.community_walktrap(steps=steps).as_clustering().membership)


ALGORITHMS = {
    "louvain": louvain,
    "louvain_classic": louvain_classic,
    "leiden": leiden,
    "label_propagation": label_propagation,
    "girvan_newman": girvan_newman,
    "fast_greedy": fast_greedy,
    "infomap": infomap,
    "walktrap": walktrap,
}
