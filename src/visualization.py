"""Graph and metric plots."""

from __future__ import annotations

import random
from pathlib import Path

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np


def plot_communities(graph: ig.Graph, labels: np.ndarray,
                     title: str = "",
                     out_path: str | Path | None = None,
                     node_size_by: np.ndarray | None = None,
                     names: list[str] | None = None,
                     figsize: tuple[int, int] = (10, 8),
                     seed: int = 42) -> plt.Figure:
    random.seed(seed)
    layout = graph.layout_fruchterman_reingold(niter=500)
    coords = np.array(layout.coords)

    unique = np.unique(labels)
    cmap = plt.get_cmap("tab20" if len(unique) > 10 else "tab10")
    label_to_color = {lab: cmap(i % cmap.N) for i, lab in enumerate(unique)}
    node_colors = [label_to_color[lab] for lab in labels]

    if node_size_by is not None:
        sizes = np.asarray(node_size_by, dtype=float)
        sizes = 50 + 600 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-12)
    else:
        sizes = np.full(graph.vcount(), 120.0)

    fig, ax = plt.subplots(figsize=figsize)
    for e in graph.es:
        x0, y0 = coords[e.source]
        x1, y1 = coords[e.target]
        ax.plot([x0, x1], [y0, y1], "-", color="#cccccc", linewidth=0.5, zorder=1)
    ax.scatter(coords[:, 0], coords[:, 1], s=sizes, c=node_colors,
               edgecolors="black", linewidths=0.5, zorder=2)

    if names is not None:
        for i, name in enumerate(names):
            ax.annotate(name, coords[i], fontsize=6, ha="center", va="center")

    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_degree_distribution(graph: ig.Graph,
                             title: str = "",
                             out_path: str | Path | None = None,
                             figsize: tuple[int, int] = (6, 4)) -> plt.Figure:
    degrees = np.array(graph.degree())
    unique, counts = np.unique(degrees[degrees > 0], return_counts=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.loglog(unique, counts, "o-")
    ax.set_xlabel("degree k")
    ax.set_ylabel("count P(k)")
    ax.set_title(title or "Degree distribution (log-log)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_metrics_comparison(df, out_path: str | Path | None = None,
                            metric: str = "nmi",
                            figsize: tuple[int, int] = (10, 5)) -> plt.Figure:
    pivot = df.pivot(index="algorithm", columns="dataset", values=metric)
    fig, ax = plt.subplots(figsize=figsize)
    pivot.plot.bar(ax=ax)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Algorithm comparison by {metric.upper()}")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
