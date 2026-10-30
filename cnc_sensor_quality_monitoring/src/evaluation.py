"""Metrics and grouped validation helpers."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold

# Fixed label space for binary tasks (worn/unworn, yes/no, etc.)
BINARY_LABELS = [0, 1]


def get_group_splitter(strategy: str, n_groups: int):
    if strategy == "leave_one_experiment_out":
        return LeaveOneGroupOut()
    if strategy == "group_kfold":
        n_splits = min(5, n_groups)
        return GroupKFold(n_splits=n_splits)
    raise ValueError(strategy)


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_prob))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, Any]:
    # LOEO folds often have one test sample / one class; metrics are still valid when aggregated OOF.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, labels=BINARY_LABELS, average="binary", pos_label=1, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, labels=BINARY_LABELS, average="binary", pos_label=1, zero_division=0)
            ),
            "f1": float(f1_score(y_true, y_pred, labels=BINARY_LABELS, average="binary", pos_label=1, zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=BINARY_LABELS, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=BINARY_LABELS).tolist(),
        }
        if y_prob is not None:
            metrics["roc_auc"] = safe_roc_auc(y_true, y_prob)
            metrics["pr_auc"] = safe_pr_auc(y_true, y_prob)
            metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
    return metrics


def aggregate_oof_predictions(fold_frames: list[pd.DataFrame]) -> pd.DataFrame:
    oof = pd.concat(fold_frames, ignore_index=True)
    return oof.sort_values("experiment_id").reset_index(drop=True)


def random_split_demo_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model,
    random_seed: int,
) -> dict[str, Any]:
    """Demonstration only: observation-level split at experiment feature granularity."""
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    preds: list[pd.DataFrame] = []
    for train_idx, test_idx in splitter.split(X, y):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        pred = model.predict(X.iloc[test_idx])
        frame = pd.DataFrame(
            {
                "experiment_id": groups.iloc[test_idx].to_numpy(),
                "y_true": y.iloc[test_idx].to_numpy(),
                "y_pred": pred,
                "y_prob": prob,
            }
        )
        preds.append(frame)
    oof = aggregate_oof_predictions(preds)
    return classification_metrics(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy(), oof["y_prob"].to_numpy())
