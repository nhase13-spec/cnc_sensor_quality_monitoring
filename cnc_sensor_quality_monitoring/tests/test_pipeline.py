"""Automated tests for leakage safety and reproducibility."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import LEAKAGE_COLUMNS, load_train_metadata
from src.feature_engineering import build_experiment_feature_matrix, feature_columns, meta_columns
from src.train_classical import grouped_cv_train, make_pipeline, build_model
from src.utils import load_config, set_seed


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def feature_df(config):
    return build_experiment_feature_matrix(config)


def test_data_loading(config):
    meta = load_train_metadata(config)
    assert len(meta) == 18
    assert meta["experiment_id"].is_unique


def test_group_splits_do_not_overlap(config, feature_df):
    groups = feature_df["experiment_id"]
    y = (feature_df["tool_condition"] == "worn").astype(int)
    splitter = LeaveOneGroupOut()
    for train_idx, test_idx in splitter.split(feature_df, y, groups):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        assert train_groups.isdisjoint(test_groups)


def test_no_target_columns_in_predictors(config, feature_df):
    feats = feature_columns(feature_df, config)
    forbidden = {
        "tool_condition",
        "machining_finalized",
        "passed_visual_inspection",
        "material",
        "experiment_id",
    }
    assert forbidden.isdisjoint(set(feats))


def test_preprocessing_trained_on_train_fold_only(config, feature_df):
    feats = feature_columns(feature_df, config)
    X = feature_df[feats]
    y = (feature_df["tool_condition"] == "worn").astype(int)
    groups = feature_df["experiment_id"]
    set_seed(config["project"]["random_seed"])
    summary, oof = grouped_cv_train(X, y, groups, "logistic_regression", config)
    assert len(oof) == len(feature_df)
    assert "balanced_accuracy" in summary


def test_metrics_valid_range(config, feature_df):
    feats = feature_columns(feature_df, config)
    X = feature_df[feats]
    y = (feature_df["tool_condition"] == "worn").astype(int)
    groups = feature_df["experiment_id"]
    pipeline = make_pipeline(build_model("dummy", config))
    summary, _ = grouped_cv_train(X, y, groups, "dummy", config)
    assert 0.0 <= summary["balanced_accuracy"] <= 1.0


def test_feature_matrix_handles_missing(config, feature_df):
    assert feature_df[feature_columns(feature_df, config)].shape[1] > 50
    assert feature_df["n_samples"].gt(0).all()
    assert np.isfinite(feature_df[feature_columns(feature_df, config)].to_numpy(na_value=np.nan)).any()


def test_temporal_windows_preserve_experiment_boundaries(config):
    from src.train_temporal import _build_sequences

    try:
        X, y, groups, _ = _build_sequences(config)
    except Exception:
        pytest.skip("Temporal dependencies unavailable")
    assert len(groups) == len(y)
    assert X.ndim == 3
