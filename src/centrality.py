"""Centrality measures for identifying opinion leaders."""

from __future__ import annotations

from dataclasses import dataclass

import igraph as ig
import numpy as np
import pandas as pd


@dataclass
class CentralityTable:
    df: pd.DataFrame

    def top(self, n: int = 10, by: str = "composite_rank") -> pd.DataFrame:
        ascending = by == "composite_rank"
        return self.df.sort_values(by, ascending=ascending).head(n)


def compute_centralities(graph: ig.Graph,
                         names: list[str] | None = None) -> CentralityTable:
    n = graph.vcount()
    names = names or (graph.vs["name"] if "name" in graph.vs.attributes()
                      else [str(i) for i in range(n)])

    degree = np.array(graph.degree())
    betweenness = np.array(graph.betweenness())
    closeness = np.array(graph.closeness())
    try:
        eigenvector = np.array(graph.eigenvector_centrality(directed=False))
    except Exception:
        eigenvector = np.full(n, np.nan)
    pagerank = np.array(graph.pagerank())

    df = pd.DataFrame({
        "node": names,
        "degree": degree,
        "betweenness": betweenness,
        "closeness": closeness,
        "eigenvector": eigenvector,
        "pagerank": pagerank,
    })

    for col in ["degree", "betweenness", "closeness", "eigenvector", "pagerank"]:
        df[f"{col}_rank"] = df[col].rank(ascending=False, method="min")

    rank_cols = [f"{c}_rank" for c in
                 ["degree", "betweenness", "closeness", "eigenvector", "pagerank"]]
    df["composite_rank"] = df[rank_cols].mean(axis=1)
    return CentralityTable(df=df)


def opinion_leaders(graph: ig.Graph, n: int = 5,
                    names: list[str] | None = None) -> pd.DataFrame:
    """Return the top-n opinion leaders (lowest composite rank)."""
    return compute_centralities(graph, names=names).top(n=n, by="composite_rank")
