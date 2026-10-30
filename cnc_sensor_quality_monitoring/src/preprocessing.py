"""Time-series cleaning, validity flags, and machining-phase filtering."""

from __future__ import annotations

import pandas as pd

from src.data_loader import numeric_sensor_columns
from src.utils import load_config


def add_validity_flags(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    rules = config["data"]["invalid_rules"]
    out = df.copy()
    out["invalid_feedrate_flag"] = out[rules["feedrate_flag_col"]] == rules["feedrate_flag_value"]
    out["invalid_x_position_flag"] = out[rules["x_position_flag_col"]] == rules["x_position_flag_value"]
    # README notes program-number anomalies, but during valid runs the active program is 1.
    # Flagging all non-zero program numbers would delete the entire time series.
    out["invalid_program_flag"] = False
    out["is_invalid_measurement"] = out["invalid_feedrate_flag"] | out["invalid_x_position_flag"]
    return out


def filter_active_machining(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    labels = set(config["data"]["active_machining_labels"])
    if "Machining_Process" not in df.columns:
        return df.copy()
    return df[df["Machining_Process"].isin(labels)].copy()


def preprocess_experiment(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    cleaned = add_validity_flags(df, config)
    cleaned = cleaned[~cleaned["is_invalid_measurement"]].copy()
    if config["features"]["use_active_machining_only"]:
        cleaned = filter_active_machining(cleaned, config)
    return cleaned.reset_index(drop=True)


def channel_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = numeric_sensor_columns(df)
    rows = []
    for col in numeric_cols:
        series = df[col]
        rows.append(
            {
                "channel": col,
                "missing_count": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean() * 100),
                "constant": bool(series.nunique(dropna=True) <= 1),
                "min": float(series.min()) if series.notna().any() else None,
                "max": float(series.max()) if series.notna().any() else None,
                "std": float(series.std()) if series.notna().any() else None,
            }
        )
    return pd.DataFrame(rows)
