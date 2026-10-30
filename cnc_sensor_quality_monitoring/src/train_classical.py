"""Classical ML with leakage-safe grouped cross-validation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from src.evaluation import aggregate_oof_predictions, classification_metrics, get_group_splitter, random_split_demo_metrics
from src.feature_engineering import feature_columns, meta_columns, select_columns_by_group
from src.utils import load_config, save_json, project_root


def _encode_binary(series: pd.Series, positive_class: str) -> pd.Series:
    return (series.astype(str) == positive_class).astype(int)


def prepare_task_frame(feature_df: pd.DataFrame, task_name: str, config: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    task_cfg = config["tasks"][task_name]
    target_col = task_cfg["target_col"]
    frame = feature_df.copy()
    if task_cfg.get("require_machining_completed"):
        completed_col = config["data"]["machining_completed_col"]
        frame = frame[frame[completed_col].astype(str) == "yes"].copy()
        frame = frame[frame[target_col].notna()].copy()
    y = _encode_binary(frame[target_col], task_cfg["positive_class"])
    groups = frame["experiment_id"].astype(int)
    all_feats = feature_columns(feature_df, config)
    return frame, y, groups, all_feats


def apply_fast_profile(config: dict) -> dict:
    """Smaller ensembles for quicker iteration (same protocol, slightly noisier metrics)."""
    import copy

    cfg = copy.deepcopy(config)
    cfg["models"]["classical"]["random_forest"]["n_estimators"] = 100
    cfg["models"]["classical"]["xgboost"]["n_estimators"] = 50
    return cfg


def build_model(model_name: str, config: dict, class_weight: str | None = "balanced") -> Any:
    if model_name == "dummy":
        return DummyClassifier(strategy="stratified", random_state=config["project"]["random_seed"])
    if model_name == "logistic_regression":
        params = config["models"]["classical"]["logistic_regression"]
        return LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            class_weight=class_weight,
            random_state=config["project"]["random_seed"],
        )
    if model_name == "random_forest":
        params = config["models"]["classical"]["random_forest"]
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight=class_weight,
            random_state=config["project"]["random_seed"],
            n_jobs=-1,
        )
    if model_name == "xgboost":
        params = config["models"]["classical"]["xgboost"]
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            eval_metric="logloss",
            random_state=config["project"]["random_seed"],
        )
    if model_name == "linear_svc":
        base = LinearSVC(class_weight=class_weight, random_state=config["project"]["random_seed"])
        return CalibratedClassifierCV(base, cv=3)
    raise ValueError(model_name)


def _safe_f_classif(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ANOVA F-score that returns 0 (not NaN) for constant columns inside a training fold."""
    with np.errstate(all="ignore"):
        scores, _ = f_classif(X, y)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def make_pipeline(model, select_k: int | None = None) -> Pipeline:
    """Impute -> scale -> (optional in-fold SelectKBest) -> model.

    ``select_k`` keeps the selection *inside* the pipeline so that it is re-fitted on the training
    experiments of every fold only (selecting on all 18 experiments first would leak labels).
    """
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if select_k is not None:
        steps.append(("select", SelectKBest(score_func=_safe_f_classif, k=select_k)))
    steps.append(("model", model))
    return Pipeline(steps=steps)


def grouped_cv_train(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_name: str,
    config: dict,
    progress_label: str | None = None,
    select_k: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    splitter = get_group_splitter(config["validation"]["primary"], groups.nunique())
    fold_frames: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    n_folds = groups.nunique()
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        if progress_label and (fold_idx == 0 or fold_idx == n_folds - 1 or (fold_idx + 1) % 6 == 0):
            print(f"    {progress_label}: fold {fold_idx + 1}/{n_folds}", flush=True)
        pipeline = make_pipeline(build_model(model_name, config), select_k=select_k)
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
        pred = pipeline.predict(X.iloc[test_idx])
        y_test = y.iloc[test_idx].to_numpy()
        fold_metric = classification_metrics(y_test, pred, prob)
        fold_metric["fold"] = fold_idx
        fold_metric["n_test"] = int(len(test_idx))
        fold_metric["test_experiments"] = groups.iloc[test_idx].tolist()
        fold_metrics.append(fold_metric)
        fold_frames.append(
            pd.DataFrame(
                {
                    "experiment_id": groups.iloc[test_idx].to_numpy(),
                    "y_true": y_test,
                    "y_pred": pred,
                    "y_prob": prob,
                    "fold": fold_idx,
                    "model": model_name,
                }
            )
        )
    oof = aggregate_oof_predictions(fold_frames)
    summary = classification_metrics(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy(), oof["y_prob"].to_numpy())
    summary["fold_metrics"] = fold_metrics
    summary["model"] = model_name
    return summary, oof


def run_classical_experiments(
    feature_df: pd.DataFrame,
    config: dict | None = None,
    *,
    skip_ablation: bool = False,
    skip_random_demo: bool = False,
) -> dict[str, Any]:
    config = config or load_config()
    results: dict[str, Any] = {}
    model_names = ["dummy", "logistic_regression", "random_forest", "xgboost"]
    ablation_groups = ["process_only", "spindle", "axis_current", "axis_motion", "combined_sensors", "all_features"]

    for task_name, task_cfg in config["tasks"].items():
        if not task_cfg.get("enabled"):
            continue
        print(f"Task: {task_name} ({task_cfg['target_col']})", flush=True)
        frame, y, groups, all_feats = prepare_task_frame(feature_df, task_name, config)
        task_result: dict[str, Any] = {"n_experiments": int(len(frame)), "class_balance": y.value_counts().to_dict()}

        # Process-parameter baseline (metadata only).
        X_process = frame[["feedrate", "clamp_pressure"]].copy()
        print("  process baseline (RF, LOEO)...", flush=True)
        process_summary, process_oof = grouped_cv_train(
            X_process, y, groups, "random_forest", config, progress_label="process baseline"
        )
        task_result["process_baseline"] = process_summary

        model_summaries = {}
        oof_predictions = {}
        for model_name in model_names:
            print(f"  model: {model_name} ({len(all_feats)} features, LOEO)...", flush=True)
            X_all = frame[all_feats]
            summary, oof = grouped_cv_train(X_all, y, groups, model_name, config, progress_label=model_name)
            model_summaries[model_name] = summary
            oof_predictions[model_name] = oof
        task_result["models"] = model_summaries

        # Random split demo on experiment-level rows (still only 18 samples; optimistic).
        if not skip_random_demo:
            print("  random-split demo (RF)...", flush=True)
            rf = make_pipeline(build_model("random_forest", config))
            task_result["random_split_demo"] = random_split_demo_metrics(
                frame[all_feats], y, groups, rf, config["project"]["random_seed"]
            )

        ablation = {}
        if not skip_ablation:
            for group in ablation_groups:
                if group == "process_only":
                    cols = ["feedrate", "clamp_pressure"]
                else:
                    cols = select_columns_by_group(all_feats, group, config)
                if not cols:
                    continue
                print(f"  ablation: {group} ({len(cols)} features)...", flush=True)
                summary, _ = grouped_cv_train(frame[cols], y, groups, "random_forest", config, progress_label=group)
                ablation[group] = {"n_features": len(cols), "metrics": summary}
        task_result["ablation"] = ablation
        results[task_name] = task_result

        metrics_dir = project_root() / config["paths"]["metrics_dir"]
        for model_name, oof in oof_predictions.items():
            oof.to_csv(metrics_dir / f"oof_{task_name}_{model_name}.csv", index=False)
        process_oof.to_csv(metrics_dir / f"oof_{task_name}_process_baseline.csv", index=False)

    save_json(results, project_root() / config["paths"]["metrics_dir"] / "classical_results.json")
    return results
