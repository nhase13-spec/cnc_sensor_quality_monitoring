"""Load experiment metadata and multivariate CNC time-series files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from src.utils import load_config, project_root


LEAKAGE_COLUMNS = {
    "No",
    "tool_condition",
    "machining_finalized",
    "passed_visual_inspection",
    "material",
}


def load_train_metadata(config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    raw_dir = project_root() / config["paths"]["raw_dir"]
    train_path = raw_dir / config["data"]["train_file"]
    meta = pd.read_csv(train_path)
    meta = meta.rename(columns={config["data"]["experiment_id_col"]: "experiment_id"})
    return meta


def experiment_file_path(experiment_id: int, config: dict | None = None) -> Path:
    config = config or load_config()
    raw_dir = project_root() / config["paths"]["raw_dir"]
    pattern = config["data"]["experiment_pattern"]
    return raw_dir / pattern.format(experiment_id)


def load_experiment_timeseries(experiment_id: int, config: dict | None = None) -> pd.DataFrame:
    path = experiment_file_path(experiment_id, config)
    df = pd.read_csv(path)
    df.insert(0, "experiment_id", experiment_id)
    df.insert(1, "sample_index", range(len(df)))
    return df


def iter_experiments(config: dict | None = None) -> Iterator[tuple[int, pd.DataFrame]]:
    meta = load_train_metadata(config)
    for experiment_id in sorted(meta["experiment_id"].unique()):
        yield int(experiment_id), load_experiment_timeseries(int(experiment_id), config)


def load_all_timeseries(config: dict | None = None) -> pd.DataFrame:
    frames = [ts for _, ts in iter_experiments(config)]
    return pd.concat(frames, ignore_index=True)


def numeric_sensor_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"experiment_id", "sample_index", "Machining_Process"}
    cols: list[str] = []
    for col in df.columns:
        if col in exclude or col.startswith("invalid_") or col == "is_invalid_measurement":
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def merge_labels(timeseries: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    label_cols = [c for c in meta.columns if c != "experiment_id"]
    return timeseries.merge(meta, on="experiment_id", how="left")
