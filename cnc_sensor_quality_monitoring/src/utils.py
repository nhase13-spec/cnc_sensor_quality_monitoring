"""Shared utilities for configuration, paths, and reproducibility."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = project_root() / "configs" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(relative: str, config: dict[str, Any]) -> Path:
    return project_root() / relative


def ensure_dirs(config: dict[str, Any]) -> None:
    for key in ("raw_dir", "interim_dir", "processed_dir", "figures_dir", "tables_dir", "metrics_dir", "models_dir", "experiments_dir"):
        path = project_root() / config["paths"][key]
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")
