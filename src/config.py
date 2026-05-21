"""Load config.toml into a typed object."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    log_level: str = "normal"
    results_dir: Path = Path("results")
    datasets: list[str] = field(default_factory=list)
    steps: dict[str, bool] = field(default_factory=dict)
    benchmark: dict = field(default_factory=dict)
    gnn: dict = field(default_factory=dict)
    leaders: dict = field(default_factory=dict)
    visualize: dict = field(default_factory=dict)

    def step_enabled(self, name: str) -> bool:
        return self.steps.get(name, True)


def load(path: str | Path = "config.toml") -> Config:
    path = Path(path)
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return Config(
        log_level=data.get("log_level", "normal"),
        results_dir=Path(data.get("results_dir", "results")),
        datasets=data.get("datasets", {}).get("include", []),
        steps=data.get("steps", {}),
        benchmark=data.get("benchmark", {}),
        gnn=data.get("gnn", {}),
        leaders=data.get("leaders", {}),
        visualize=data.get("visualize", {}),
    )
