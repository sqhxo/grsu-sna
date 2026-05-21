"""Analysis pipeline. Each step takes datasets + config, writes results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import log
from .benchmark import benchmark
from .centrality import compute_centralities, opinion_leaders
from .community import ALGORITHMS, leiden
from .datasets import LabeledGraph
from .graph_models import matched_to
from .graph_stats import compute_stats
from .isolation import find_all_isolated
from .visualization import (
    plot_communities,
    plot_degree_distribution,
    plot_metrics_comparison,
)

logger = log.get("pipeline")


def step_structural(datasets: dict[str, LabeledGraph], results_dir: Path) -> pd.DataFrame:
    logger.info("# Structural analysis vs. random models")
    rows = []
    for name, ds in datasets.items():
        rows.append({"source": "real", **compute_stats(ds.graph, name).as_dict()})
        for model in ("erdos_renyi", "barabasi_albert", "watts_strogatz"):
            try:
                g_synth = matched_to(ds.graph, model, seed=42)
                rows.append({
                    "source": model,
                    **compute_stats(g_synth, f"{name}/{model}").as_dict(),
                })
            except Exception as e:
                logger.warning("  skipped %s/%s: %s", name, model, e)
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "01_graph_stats.csv", index=False)
    cols = ["name", "source", "n_nodes", "n_edges", "density",
            "global_clustering", "avg_path_length", "assortativity_degree"]
    logger.info(df[cols].to_string(index=False))
    return df


def step_benchmark(datasets: dict[str, LabeledGraph],
                   results_dir: Path,
                   algorithms: list[str],
                   n_repeats: int) -> pd.DataFrame:
    logger.info("# Community detection benchmark")
    algos = {n: ALGORITHMS[n] for n in algorithms if n in ALGORITHMS}
    log.verbose(logger, "running %d algorithms × %d datasets × %d repeats",
                len(algos), len(datasets), n_repeats)
    results = benchmark(algos, datasets=datasets, n_repeats=n_repeats)
    results.to_csv(results_dir / "02_benchmark.csv", index=False)

    cols = ["algorithm", "dataset", "nmi", "ari", "modularity_pred",
            "n_clusters_pred", "n_clusters_true", "runtime_s"]
    show = results[cols].copy()
    for c in ("n_clusters_pred", "n_clusters_true"):
        show[c] = pd.to_numeric(show[c], errors="coerce").round(1)
    logger.info(show.to_string(index=False))

    for metric in ("nmi", "ari", "modularity_pred"):
        try:
            plot_metrics_comparison(
                results, out_path=results_dir / f"02_compare_{metric}.png",
                metric=metric,
            )
        except Exception as e:
            logger.warning("  plot %s failed: %s", metric, e)
    return results


def step_gnn(datasets: dict[str, LabeledGraph],
             results_dir: Path,
             models: list[str],
             epochs: int,
             hidden_dim: int,
             patience: int = 50,
             checkpoint_dir: str | Path = "models",
             force_retrain: bool = False) -> pd.DataFrame | None:
    logger.info("# GNN training")
    try:
        from .gnn import train_gnn
    except ImportError as e:
        logger.warning("  torch not available (%s); skipping", e)
        return None

    from .metrics import all_metrics

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds_name, ds in datasets.items():
        if ds.graph.vcount() < 20 or ds.n_communities < 2:
            log.verbose(logger, "  %s: too small, skipping", ds_name)
            continue
        for model_name in models:
            ckpt = ckpt_dir / f"{ds_name}_{model_name}.pt"
            log.verbose(logger, "  %s on %s (n=%d, k=%d)",
                        model_name, ds_name, ds.graph.vcount(), ds.n_communities)
            try:
                result = train_gnn(
                    ds.graph, ds.labels,
                    model_name=model_name,
                    epochs=epochs, hidden_dim=hidden_dim, patience=patience,
                    checkpoint_path=ckpt, force_retrain=force_retrain,
                )
            except Exception as e:
                logger.warning("    training failed: %s", e)
                continue
            metrics = all_metrics(ds.graph, ds.labels, result.labels_pred)
            origin = "cached" if result.loaded_from_cache else f"trained {result.epochs_trained}ep"
            logger.info("  %-10s %-20s [%s] train=%.3f val=%.3f test=%.3f "
                        "nmi=%.3f ari=%.3f",
                        model_name, ds_name, origin,
                        result.train_acc, result.val_acc, result.test_acc,
                        metrics["nmi"], metrics["ari"])
            row = {"model": model_name, "dataset": ds_name,
                   "loaded_from_cache": result.loaded_from_cache,
                   "epochs_trained": result.epochs_trained,
                   "train_acc": result.train_acc,
                   "val_acc": result.val_acc,
                   "test_acc": result.test_acc,
                   **{k: v for k, v in metrics.items() if isinstance(v, float)}}
            rows.append(row)

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "03_gnn_results.csv", index=False)
    return df


def step_leaders(datasets: dict[str, LabeledGraph],
                 results_dir: Path,
                 top_n: int) -> dict:
    logger.info("# Opinion leaders (top %d)", top_n)
    summary = {}
    for name, ds in datasets.items():
        names = ds.graph.vs["name"] if "name" in ds.graph.vs.attributes() else None
        try:
            leaders = opinion_leaders(ds.graph, n=top_n, names=names)
        except Exception as e:
            logger.warning("  %s: skipped (%s)", name, e)
            continue
        leaders.to_csv(results_dir / f"04_leaders_{name}.csv", index=False)
        summary[name] = leaders
        logger.info("  %s:", name)
        logger.info(leaders[["node", "degree", "betweenness", "pagerank",
                             "composite_rank"]].to_string(index=False))
    return summary


def step_isolated(datasets: dict[str, LabeledGraph], results_dir: Path) -> dict:
    logger.info("# Isolated groups")
    summary = {}
    for name, ds in datasets.items():
        if ds.graph.vcount() > 20000:
            log.verbose(logger, "  %s: too large, skipping", name)
            continue
        labels = leiden(ds.graph)
        groups = find_all_isolated(ds.graph, labels=labels)
        summary[name] = groups
        logger.info("  %s (top 5):", name)
        for g in groups[:5]:
            logger.info("    %s", g)
    return summary


def step_visualize(datasets: dict[str, LabeledGraph],
                   results_dir: Path,
                   max_nodes: int) -> None:
    logger.info("# Visualization")
    for name, ds in datasets.items():
        if ds.graph.vcount() > max_nodes:
            log.verbose(logger, "  %s: too large, skipping", name)
            continue
        try:
            labels = leiden(ds.graph)
            cent = compute_centralities(ds.graph)
            show_names = name in ("karate", "dolphins")
            plot_communities(
                ds.graph, labels,
                title=f"{name}: Leiden",
                out_path=results_dir / f"06_{name}_communities.png",
                node_size_by=cent.df["pagerank"].values,
                names=ds.graph.vs["name"] if show_names and "name" in ds.graph.vs.attributes() else None,
            )
            plot_communities(
                ds.graph, ds.labels,
                title=f"{name}: ground truth",
                out_path=results_dir / f"06_{name}_groundtruth.png",
                node_size_by=cent.df["pagerank"].values,
            )
            plot_degree_distribution(
                ds.graph,
                title=f"{name}: degree distribution",
                out_path=results_dir / f"06_{name}_degree_dist.png",
            )
            log.verbose(logger, "  %s: 3 plots saved", name)
        except Exception as e:
            logger.warning("  %s: plot failed: %s", name, e)
