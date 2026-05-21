"""Entry point: load config, load datasets, run enabled steps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from src import config, datasets, log, pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()

    cfg = config.load(args.config)
    logger = log.setup(cfg.log_level)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(exist_ok=True)

    logger.info("# Datasets")
    loaded = {}
    for name in cfg.datasets:
        try:
            ds = datasets.load(name)
            loaded[name] = ds
            logger.info("  %s", ds)
        except Exception as e:
            logger.warning("  failed to load %s: %s: %s",
                           name, type(e).__name__, e)

    if not loaded:
        logger.error("No datasets loaded, aborting.")
        return

    if cfg.step_enabled("structural"):
        pipeline.step_structural(loaded, results_dir)
    if cfg.step_enabled("benchmark"):
        pipeline.step_benchmark(
            loaded, results_dir,
            algorithms=cfg.benchmark.get("algorithms", []),
            n_repeats=cfg.benchmark.get("n_repeats", 1),
        )
    if cfg.step_enabled("gnn"):
        pipeline.step_gnn(
            loaded, results_dir,
            models=cfg.gnn.get("models", ["gcn"]),
            epochs=cfg.gnn.get("epochs", 500),
            hidden_dim=cfg.gnn.get("hidden_dim", 64),
            patience=cfg.gnn.get("patience", 50),
            checkpoint_dir=cfg.gnn.get("checkpoint_dir", "models"),
            force_retrain=cfg.gnn.get("force_retrain", False),
        )
    if cfg.step_enabled("leaders"):
        pipeline.step_leaders(loaded, results_dir,
                              top_n=cfg.leaders.get("top_n", 5))
    if cfg.step_enabled("isolated"):
        pipeline.step_isolated(loaded, results_dir)
    if cfg.step_enabled("visualize"):
        pipeline.step_visualize(loaded, results_dir,
                                max_nodes=cfg.visualize.get("max_nodes", 5000))

    logger.info("Done. Results in %s", results_dir)


if __name__ == "__main__":
    main()
