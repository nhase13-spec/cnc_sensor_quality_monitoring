"""Unsupervised anomaly detection with grouped evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.evaluation import get_group_splitter
from src.feature_engineering import feature_columns
from src.utils import load_config, project_root, save_json


def _anomaly_metrics(y_true_binary: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true_binary == 1)).sum())
    fp = int(((pred == 1) & (y_true_binary == 0)).sum())
    fn = int(((pred == 0) & (y_true_binary == 1)).sum())
    tn = int(((pred == 0) & (y_true_binary == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "detection_rate": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def run_anomaly_detection(feature_df: pd.DataFrame, config: dict | None = None) -> dict[str, Any]:
    config = config or load_config()
    feature_cols = feature_columns(feature_df, config)
    X = feature_df[feature_cols]
    groups = feature_df["experiment_id"].astype(int)
    tool_worn = (feature_df["tool_condition"].astype(str) == "worn").astype(int).to_numpy()
    visual_fail = np.where(
        feature_df["passed_visual_inspection"].isna(),
        np.nan,
        (feature_df["passed_visual_inspection"].astype(str) == "no").astype(float),
    )

    splitter = get_group_splitter(config["validation"]["primary"], groups.nunique())
    methods = {
        "isolation_forest": IsolationForest(contamination=0.15, random_state=config["project"]["random_seed"]),
        "one_class_svm": OneClassSVM(nu=0.15, kernel="rbf", gamma="scale"),
    }
    results: dict[str, Any] = {}
    for method_name, estimator in methods.items():
        oof_scores = np.full(len(feature_df), np.nan)
        for train_idx, test_idx in splitter.split(X, tool_worn, groups):
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", estimator),
                ]
            )
            X_train = X.iloc[train_idx]
            # Train on unworn experiments only when available in fold.
            unworn_mask = feature_df.iloc[train_idx]["tool_condition"].astype(str) == "unworn"
            if unworn_mask.sum() >= 2:
                pipeline.fit(X_train[unworn_mask])
            else:
                pipeline.fit(X_train)
            if hasattr(pipeline.named_steps["model"], "decision_function"):
                raw = pipeline.decision_function(X.iloc[test_idx])
                scores = -raw
            else:
                pred = pipeline.predict(X.iloc[test_idx])
                scores = np.where(pred == -1, 1.0, 0.0)
            oof_scores[test_idx] = scores
        threshold = float(np.nanpercentile(oof_scores, 85))
        tool_metrics = _anomaly_metrics(tool_worn, oof_scores, threshold)
        visual_mask = ~np.isnan(visual_fail)
        visual_metrics = None
        if visual_mask.sum() >= 4:
            visual_metrics = _anomaly_metrics(visual_fail[visual_mask].astype(int), oof_scores[visual_mask], threshold)
        results[method_name] = {
            "threshold": threshold,
            "tool_worn_reference": tool_metrics,
            "visual_fail_reference": visual_metrics,
        }
        pd.DataFrame(
            {
                "experiment_id": groups,
                "anomaly_score": oof_scores,
                "tool_worn": tool_worn,
                "visual_fail": visual_fail,
            }
        ).to_csv(project_root() / config["paths"]["metrics_dir"] / f"anomaly_{method_name}.csv", index=False)

    # PCA reconstruction error baseline.
    recon_errors = np.full(len(feature_df), np.nan)
    for train_idx, test_idx in splitter.split(X, tool_worn, groups):
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_train = imputer.fit_transform(X.iloc[train_idx])
        X_train = scaler.fit_transform(X_train)
        n_components = min(10, X_train.shape[1], X_train.shape[0] - 1)
        pca = PCA(n_components=max(2, n_components), random_state=config["project"]["random_seed"])
        pca.fit(X_train)
        X_test = scaler.transform(imputer.transform(X.iloc[test_idx]))
        recon = pca.inverse_transform(pca.transform(X_test))
        recon_errors[test_idx] = np.mean((X_test - recon) ** 2, axis=1)
    threshold = float(np.nanpercentile(recon_errors, 85))
    results["pca_reconstruction"] = {
        "threshold": threshold,
        "tool_worn_reference": _anomaly_metrics(tool_worn, recon_errors, threshold),
    }
    save_json(results, project_root() / config["paths"]["metrics_dir"] / "anomaly_results.json")
    return results
