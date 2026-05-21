"""Project-wide logging.

Levels:
    quiet   — only final results and errors
    normal  — section headers and per-dataset summaries (default)
    verbose — per-algorithm progress, dataset descriptions
    debug   — internal details, stack traces
"""

from __future__ import annotations

import logging
import sys

LEVELS = {
    "quiet": logging.WARNING,
    "normal": logging.INFO,
    "verbose": 15,
    "debug": logging.DEBUG,
}

logging.addLevelName(15, "VERBOSE")


def setup(level: str = "normal") -> logging.Logger:
    lvl = LEVELS.get(level, logging.INFO)
    root = logging.getLogger("grsu")
    root.handlers.clear()
    root.setLevel(lvl)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.propagate = False
    return root


def get(name: str = "grsu") -> logging.Logger:
    return logging.getLogger(name if name.startswith("grsu") else f"grsu.{name}")


def verbose(logger: logging.Logger, msg: str, *args) -> None:
    logger.log(15, msg, *args)
