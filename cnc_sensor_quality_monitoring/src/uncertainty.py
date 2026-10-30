"""Probability calibration and reliability analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from src.utils import load_config, project_root, save_json


def run_uncertainty_analysis(config: dict | None = None) -> dict:
    config = config or load_config()
    metrics_dir = project_root() / config["paths"]["metrics_dir"]
    figures_dir = project_root() / config["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    oof_path = metrics_dir / "oof_tool_condition_random_forest.csv"
    if not oof_path.exists():
        return {"status": "skipped", "reason": "OOF predictions not found"}
    oof = pd.read_csv(oof_path)
    prob_true, prob_pred = calibration_curve(oof["y_true"], oof["y_prob"], n_bins=4, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.plot(prob_pred, prob_true, marker="o", label="Random Forest (LOEO aggregate)")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Reliability diagram — tool condition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "reliability_tool_condition.png", dpi=200)
    plt.close()
    summary = {
        "n_predictions": int(len(oof)),
        "brier_score": float(np.mean((oof["y_prob"] - oof["y_true"]) ** 2)),
        "calibration_bins": {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()},
        "note": "Only 18 independent experiments; calibration is indicative, not production-grade.",
    }
    save_json(summary, metrics_dir / "uncertainty_results.json")
    return summary
