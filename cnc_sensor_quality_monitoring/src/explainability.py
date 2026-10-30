"""Model explainability for classical pipelines."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import feature_columns, meta_columns
from src.train_classical import prepare_task_frame
from src.utils import load_config, project_root


def _aggregate_importance(names: list[str], values: np.ndarray, top_n: int = 25) -> pd.DataFrame:
    channel_importance: dict[str, float] = {}
    for name, value in zip(names, values):
        channel = name.split("__")[0] if "__" in name else name
        channel_importance[channel] = channel_importance.get(channel, 0.0) + float(abs(value))
    frame = pd.DataFrame({"channel": list(channel_importance.keys()), "importance": list(channel_importance.values())})
    return frame.sort_values("importance", ascending=False).head(top_n)


def run_explainability(feature_df: pd.DataFrame, config: dict | None = None) -> None:
    config = config or load_config()
    figures_dir = project_root() / config["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    frame, y, groups, all_feats = prepare_task_frame(feature_df, "tool_condition", config)
    X = frame[all_feats]
    train_mask = groups != groups.iloc[-1]
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=5,
                    class_weight="balanced",
                    random_state=config["project"]["random_seed"],
                ),
            ),
        ]
    )
    pipeline.fit(X[train_mask], y[train_mask])
    X_train = pipeline.named_steps["scaler"].transform(pipeline.named_steps["imputer"].transform(X[train_mask]))
    model = pipeline.named_steps["model"]
    perm = permutation_importance(model, X_train, y[train_mask], n_repeats=10, random_state=config["project"]["random_seed"])
    perm_df = _aggregate_importance(all_feats, perm.importances_mean)
    perm_df.to_csv(project_root() / config["paths"]["tables_dir"] / "permutation_importance_tool_condition.csv", index=False)
    plt.figure(figsize=(10, 6))
    plt.barh(perm_df["channel"], perm_df["importance"])
    plt.gca().invert_yaxis()
    plt.title("Permutation importance (tool condition, held-out last experiment excluded from fit)")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance_tool_condition.png", dpi=200)
    plt.close()

    # SHAP on a small background set due to n=18.
    background = shap.sample(X_train, min(12, X_train.shape[0]), random_state=config["project"]["random_seed"])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(background)
    values = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap.summary_plot(values, background, feature_names=all_feats, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary_tool_condition.png", dpi=200)
    plt.close()
