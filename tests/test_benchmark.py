"""Tests for the benchmark harness and community detection algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from src.community import (
    fast_greedy,
    girvan_newman,
    label_propagation,
    leiden,
    louvain,
)
from src.datasets import load_karate
from src.metrics import all_metrics
from src.benchmark import benchmark, run_algorithm


def test_perfect_prediction_yields_perfect_metrics():
    ds = load_karate()
    metrics = all_metrics(ds.graph, ds.labels, ds.labels)
    assert metrics["nmi"] == pytest.approx(1.0)
    assert metrics["ari"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["purity"] == pytest.approx(1.0)


def test_random_baseline_is_poor():
    ds = load_karate()
    rng = np.random.default_rng(0)
    bad = rng.integers(0, 2, size=ds.n_nodes)
    metrics = all_metrics(ds.graph, ds.labels, bad)
    assert metrics["nmi"] < 0.3
    assert abs(metrics["ari"]) < 0.3


def test_run_algorithm_returns_result():
    ds = load_karate()
    result = run_algorithm(louvain, ds, name="louvain")
    assert result.error is None
    assert result.labels_pred.shape == (ds.n_nodes,)
    assert "nmi" in result.metrics


def test_run_algorithm_catches_errors():
    def bad_algo(graph):
        raise RuntimeError("intentional failure")

    ds = load_karate()
    result = run_algorithm(bad_algo, ds, name="bad")
    assert result.error is not None
    assert "intentional failure" in result.error


def test_run_algorithm_validates_output_shape():
    def wrong_shape(graph):
        return np.zeros(graph.vcount() + 5)

    ds = load_karate()
    result = run_algorithm(wrong_shape, ds, name="wrong")
    assert result.error is not None
    assert "shape" in result.error


@pytest.mark.parametrize("algo,name", [
    (louvain, "louvain"),
    (leiden, "leiden"),
    (label_propagation, "label_propagation"),
    (fast_greedy, "fast_greedy"),
    (girvan_newman, "girvan_newman"),
])
def test_community_algorithms_beat_random_on_karate(algo, name):
    ds = load_karate()
    result = run_algorithm(algo, ds, name=name)
    assert result.error is None, result.error
    assert result.metrics["nmi"] > 0.4, (
        f"{name} produced NMI={result.metrics['nmi']:.3f}, too low"
    )


def test_benchmark_dataframe_structure():
    algos = {"louvain": louvain, "leiden": leiden}
    df = benchmark(algos, datasets={"karate": load_karate()}, n_repeats=2)
    assert len(df) == 2
    assert {"algorithm", "dataset", "nmi", "ari"}.issubset(df.columns)
