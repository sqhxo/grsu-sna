"""Benchmark harness: run community detection algorithms on datasets."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import igraph as ig
import numpy as np
import pandas as pd

from .datasets import LabeledGraph, load_all
from .metrics import all_metrics

Algorithm = Callable[[ig.Graph], np.ndarray]

SLOW_ALGORITHMS = {"girvan_newman", "fast_greedy"}
SLOW_NODE_LIMIT = 500


@dataclass
class AlgorithmResult:
    algorithm: str
    dataset: str
    labels_pred: np.ndarray
    labels_true: np.ndarray
    runtime_seconds: float
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def as_row(self) -> dict[str, Any]:
        row = {
            "algorithm": self.algorithm,
            "dataset": self.dataset,
            "runtime_s": self.runtime_seconds,
            "error": self.error,
        }
        row.update(self.metrics)
        return row


def run_algorithm(
    algo: Algorithm, dataset: LabeledGraph, name: str | None = None
) -> AlgorithmResult:
    """Run one algorithm on one dataset and compute metrics."""
    algo_name = name or getattr(algo, "__name__", "anonymous")
    start = time.perf_counter()
    error: str | None = None
    labels_pred = np.zeros(dataset.n_nodes, dtype=int)
    try:
        labels_pred = np.asarray(algo(dataset.graph), dtype=int)
        if labels_pred.shape != (dataset.n_nodes,):
            raise ValueError(
                f"Algorithm returned shape {labels_pred.shape}, "
                f"expected ({dataset.n_nodes},)"
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    runtime = time.perf_counter() - start

    metrics: dict[str, float] = {}
    if error is None:
        metrics = all_metrics(dataset.graph, dataset.labels, labels_pred)

    return AlgorithmResult(
        algorithm=algo_name,
        dataset=dataset.name,
        labels_pred=labels_pred,
        labels_true=dataset.labels,
        runtime_seconds=runtime,
        metrics=metrics,
        error=error,
    )


def benchmark(
    algorithms: dict[str, Algorithm],
    datasets: dict[str, LabeledGraph] | None = None,
    n_repeats: int = 1,
    skip_slow_on_large: bool = True,
) -> pd.DataFrame:
    """Run algorithms × datasets grid, return a DataFrame of metrics."""
    if datasets is None:
        datasets = load_all()

    rows: list[dict[str, Any]] = []
    for algo_name, algo in algorithms.items():
        for ds_name, ds in datasets.items():
            if (skip_slow_on_large and algo_name in SLOW_ALGORITHMS
                    and ds.n_nodes > SLOW_NODE_LIMIT):
                rows.append({
                    "algorithm": algo_name, "dataset": ds_name,
                    "error": f"skipped (n={ds.n_nodes} > {SLOW_NODE_LIMIT})",
                })
                continue
            agg: dict[str, list[float]] = {}
            last: AlgorithmResult | None = None
            for _ in range(n_repeats):
                result = run_algorithm(algo, ds, name=algo_name)
                last = result
                for k, v in result.metrics.items():
                    agg.setdefault(k, []).append(v)
                agg.setdefault("runtime_s", []).append(result.runtime_seconds)
            assert last is not None
            row: dict[str, Any] = {
                "algorithm": algo_name,
                "dataset": ds_name,
                "error": last.error,
            }
            for k, vals in agg.items():
                row[k] = float(np.mean(vals))
                if n_repeats > 1:
                    row[f"{k}_std"] = float(np.std(vals))
            rows.append(row)
    return pd.DataFrame(rows)


def quick_sanity_check(algo: Algorithm, algo_name: str = "test") -> None:
    """Run on Karate Club and print a short report (for debugging)."""
    from .datasets import load_karate

    ds = load_karate()
    result = run_algorithm(algo, ds, name=algo_name)
    print(f"{result.algorithm} on {result.dataset}")
    if result.error:
        print(f"  error: {result.error}")
        return
    print(f"  runtime: {result.runtime_seconds*1000:.2f} ms")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            print(f"  {k:>20s}: {v:.4f}")
        else:
            print(f"  {k:>20s}: {v}")
