"""Clustering quality metrics: NMI, AMI, ARI, purity, accuracy, modularity."""

from __future__ import annotations

import igraph as ig
import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def nmi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(normalized_mutual_info_score(y_true, y_pred))


def ami(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(adjusted_mutual_info_score(y_true, y_pred))


def ari(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(adjusted_rand_score(y_true, y_pred))


def modularity(graph: ig.Graph, labels: np.ndarray) -> float:
    return float(graph.modularity(list(labels)))


def purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = 0
    for cluster in np.unique(y_pred):
        mask = y_pred == cluster
        if not mask.any():
            continue
        _, counts = np.unique(y_true[mask], return_counts=True)
        total += counts.max()
    return total / len(y_true)


def best_permutation_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy under the optimal label permutation (Hungarian algorithm)."""
    from scipy.optimize import linear_sum_assignment

    classes_t = np.unique(y_true)
    classes_p = np.unique(y_pred)
    cm = np.zeros((len(classes_t), len(classes_p)), dtype=int)
    for i, t in enumerate(classes_t):
        for j, p in enumerate(classes_p):
            cm[i, j] = int(np.sum((y_true == t) & (y_pred == p)))
    row, col = linear_sum_assignment(-cm)
    return float(cm[row, col].sum() / len(y_true))


def all_metrics(
    graph: ig.Graph, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    return {
        "nmi": nmi(y_true, y_pred),
        "ami": ami(y_true, y_pred),
        "ari": ari(y_true, y_pred),
        "purity": purity(y_true, y_pred),
        "accuracy": best_permutation_accuracy(y_true, y_pred),
        "modularity_pred": modularity(graph, y_pred),
        "modularity_true": modularity(graph, y_true),
        "n_clusters_pred": int(len(np.unique(y_pred))),
        "n_clusters_true": int(len(np.unique(y_true))),
    }
