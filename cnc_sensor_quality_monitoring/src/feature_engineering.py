"""Experiment-level feature extraction from multivariate CNC telemetry."""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from src.data_loader import load_train_metadata, numeric_sensor_columns
from src.preprocessing import preprocess_experiment
from src.signal_processing import (
    autocorr_lag1,
    crest_factor,
    rms,
    spectral_features,
    trend_slope,
    wavelet_energy,
)
from src.utils import load_config, project_root


STAT_SUFFIXES = ("mean", "std", "rms", "min", "max", "ptp", "median", "iqr", "skew", "kurtosis", "crest")

# Many CNC channels are flat or nearly flat within an experiment; skew/kurtosis is not meaningful then.
_FLAT_SIGNAL_PTP = 1e-10


def _skew_kurtosis(values: np.ndarray) -> tuple[float, float]:
    if values.size < 3 or float(np.ptp(values)) < _FLAT_SIGNAL_PTP:
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(stats.skew(values)), float(stats.kurtosis(values))


def _basic_stats(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {suffix: np.nan for suffix in STAT_SUFFIXES}
    q1, q3 = np.percentile(values, [25, 75])
    skew, kurtosis = _skew_kurtosis(values)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": rms(values),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "ptp": float(np.max(values) - np.min(values)),
        "median": float(np.median(values)),
        "iqr": float(q3 - q1),
        "skew": skew,
        "kurtosis": kurtosis,
        "crest": crest_factor(values),
    }


def extract_channel_features(values: np.ndarray, channel: str, fs: float, min_samples: int) -> dict[str, float]:
    feats: dict[str, float] = {}
    for suffix, value in _basic_stats(values).items():
        feats[f"{channel}__{suffix}"] = value
    diff = np.diff(values[np.isfinite(values)])
    feats[f"{channel}__diff_mean"] = float(np.mean(diff)) if diff.size else np.nan
    feats[f"{channel}__diff_std"] = float(np.std(diff)) if diff.size else np.nan
    feats[f"{channel}__energy"] = float(np.sum(values[np.isfinite(values)] ** 2))
    feats[f"{channel}__trend_slope"] = trend_slope(values)
    feats[f"{channel}__autocorr_lag1"] = autocorr_lag1(values)
    for key, value in spectral_features(values, fs=fs, min_samples=min_samples).items():
        feats[f"{channel}__{key}"] = value
    feats[f"{channel}__wavelet_energy"] = wavelet_energy(values)
    return feats


def extract_experiment_features(df: pd.DataFrame, config: dict | None = None) -> dict[str, float]:
    config = config or load_config()
    fs = 1.0 / config["data"]["sampling_interval_s"]
    min_samples = config["features"]["frequency_min_samples"]
    channels = numeric_sensor_columns(df)
    features: dict[str, float] = {"n_samples": float(len(df))}
    for channel in channels:
        features.update(extract_channel_features(df[channel].to_numpy(), channel, fs, min_samples))
    return features


def build_experiment_feature_matrix(config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    meta = load_train_metadata(config)
    rows: list[dict] = []
    for experiment_id in sorted(meta["experiment_id"].unique()):
        from src.data_loader import load_experiment_timeseries

        raw = load_experiment_timeseries(int(experiment_id), config)
        processed = preprocess_experiment(raw, config)
        feats = extract_experiment_features(processed, config)
        feats["experiment_id"] = int(experiment_id)
        rows.append(feats)
    feature_df = pd.DataFrame(rows)
    feature_df = feature_df.merge(meta, on="experiment_id", how="left")
    feat_cols = feature_columns(feature_df, config)
    # Drop columns that are entirely missing (can occur for saturated/empty channels).
    keep = []
    for c in feat_cols:
        series = feature_df[c]
        if not series.notna().any():
            continue
        if series.notna().sum() >= 2 and series.std(skipna=True) == 0:
            continue
        keep.append(c)
    # Feed rate is fixed within each experiment → these derived features are all-NaN or uninformative.
    drop = sorted(set(feat_cols) - set(keep))
    drop.extend(
        c
        for c in (
            "M1_CURRENT_FEEDRATE__skew",
            "M1_CURRENT_FEEDRATE__kurtosis",
            "M1_CURRENT_FEEDRATE__autocorr_lag1",
        )
        if c in feature_df.columns
    )
    drop = sorted(set(drop))
    if drop:
        feature_df = feature_df.drop(columns=drop)
    return feature_df


def save_feature_matrix(config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    feature_df = build_experiment_feature_matrix(config)
    out_path = project_root() / config["paths"]["processed_dir"] / "experiment_features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(out_path, index=False)
    return feature_df


def meta_columns(config: dict | None = None) -> list[str]:
    config = config or load_config()
    return [
        "experiment_id",
        "material",
        "feedrate",
        "clamp_pressure",
        config["data"]["tool_condition_col"],
        config["data"]["machining_completed_col"],
        config["data"]["visual_inspection_col"],
    ]


def feature_columns(feature_df: pd.DataFrame, config: dict | None = None) -> list[str]:
    meta_set = set(meta_columns(config))
    return [c for c in feature_df.columns if c not in meta_set]


def select_columns_by_group(all_feature_cols: list[str], group: str, config: dict | None = None) -> list[str]:
    config = config or load_config()
    if group == "combined_sensors":
        merged = select_columns_by_group(all_feature_cols, "spindle", config) + select_columns_by_group(
            all_feature_cols, "axis_current", config
        ) + select_columns_by_group(all_feature_cols, "axis_motion", config)
        return list(dict.fromkeys(merged))
    if group == "all_features":
        return list(all_feature_cols)
    group_cfg = config["ablation_groups"][group]
    if group == "process_only":
        return [c for c in group_cfg if c in {"feedrate", "clamp_pressure"}]
    if group == "spindle":
        prefix = group_cfg["prefix"]
        return [c for c in all_feature_cols if c.startswith(prefix)]
    if group == "axis_current":
        suffixes = tuple(group_cfg["suffixes"])
        return [c for c in all_feature_cols if any(part in c for part in suffixes)]
    if group == "axis_motion":
        tokens = group_cfg["contains"]
        return [c for c in all_feature_cols if any(token in c for token in tokens)]
    raise KeyError(f"Unknown ablation group: {group}")
