"""Bootstrap confidence intervals for out-of-fold classification metrics.

Unit of resampling = one experiment (one out-of-fold prediction per experiment under
leave-one-experiment-out). With 14-18 experiments these intervals are WIDE; that is the point:
they show how little a single point estimate such as "balanced accuracy 0.55" can support.

Note: the bootstrap captures sampling variability of the evaluation set only. It does not capture
variability of the training set (each fold's model) and cannot create information that 18
experiments do not contain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.utils import load_config, project_root, save_json

METRIC_NAMES = ("balanced_accuracy", "macro_f1", "recall", "roc_auc")


def _fast_metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None) -> dict[str, float]:
    """Binary metrics in numpy. Returns NaN for undefined metrics (single-class resample)."""
    pos = y == 1
    neg = ~pos
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    out = {name: np.nan for name in METRIC_NAMES}
    if n_pos == 0 or n_neg == 0:
        return out
    tp = float(np.sum(pred[pos] == 1))
    fn = n_pos - tp
    tn = float(np.sum(pred[neg] == 0))
    fp = n_neg - tn
    tpr = tp / n_pos
    tnr = tn / n_neg
    out["recall"] = tpr
    out["balanced_accuracy"] = 0.5 * (tpr + tnr)

    def _f1(tp_, fp_, fn_) -> float:
        denom = 2 * tp_ + fp_ + fn_
        return 0.0 if denom == 0 else 2 * tp_ / denom

    out["macro_f1"] = 0.5 * (_f1(tp, fp, fn) + _f1(tn, fn, fp))
    if prob is not None:
        ranks = rankdata(prob)  # average ranks handle ties (Mann-Whitney U form of AUC)
        out["roc_auc"] = float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return out


def bootstrap_classification_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile bootstrap CIs over experiments.

    Resamples that contain a single class are discarded for the metrics that are undefined there
    (the share of discarded resamples is reported as ``n_valid``).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    prob = None if y_prob is None else np.asarray(y_prob, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    point = _fast_metrics(y_true, y_pred, prob)
    draws = {name: [] for name in METRIC_NAMES}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = _fast_metrics(y_true[idx], y_pred[idx], None if prob is None else prob[idx])
        for name in METRIC_NAMES:
            if np.isfinite(m[name]):
                draws[name].append(m[name])
    result: dict[str, Any] = {"n": int(n), "n_boot": int(n_boot)}
    for name in METRIC_NAMES:
        values = np.asarray(draws[name], dtype=float)
        if values.size == 0:
            result[name] = {"point": point[name], "lo": np.nan, "hi": np.nan, "n_valid": 0}
            continue
        lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        entry = {"point": point[name], "lo": float(lo), "hi": float(hi), "n_valid": int(values.size)}
        if name == "balanced_accuracy":
            # Share of resamples at or below chance level (0.5): a simple, honest "is it better than a coin?" gauge.
            entry["share_le_chance"] = float(np.mean(values <= 0.5))
        result[name] = entry
    return result


def ci_rows_from_oof(oof: pd.DataFrame, *, n_boot: int, seed: int, alpha: float, **labels: Any) -> dict[str, Any]:
    """Flatten bootstrap output for one OOF frame into a single table row."""
    ci = bootstrap_classification_ci(
        oof["y_true"].to_numpy(),
        oof["y_pred"].to_numpy(),
        oof["y_prob"].to_numpy() if "y_prob" in oof else None,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
    )
    row: dict[str, Any] = dict(labels)
    row["n"] = ci["n"]
    for name in METRIC_NAMES:
        row[name] = ci[name]["point"]
        row[f"{name}_lo"] = ci[name]["lo"]
        row[f"{name}_hi"] = ci[name]["hi"]
    row["ba_share_le_chance"] = ci["balanced_accuracy"].get("share_le_chance", np.nan)
    return row


def run_bootstrap(config: dict | None = None) -> pd.DataFrame:
    """Bootstrap CIs for every stored OOF file from the original classical run."""
    config = config or load_config()
    b = config["bootstrap"]
    metrics_dir = project_root() / config["paths"]["metrics_dir"]
    tables_dir = project_root() / config["paths"]["tables_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for task in config["tasks"]:
        for path in sorted(metrics_dir.glob(f"oof_{task}_*.csv")):
            model = path.stem[len(f"oof_{task}_") :]
            oof = pd.read_csv(path)
            rows.append(
                ci_rows_from_oof(oof, n_boot=b["n_boot"], seed=b["seed"], alpha=b["alpha"], task=task, model=model)
            )
    table = pd.DataFrame(rows)
    table.to_csv(tables_dir / "bootstrap_ci_classical.csv", index=False)
    save_json({"bootstrap": b, "rows": rows}, metrics_dir / "bootstrap_ci_classical.json")
    print(f"Bootstrap CIs written for {len(table)} OOF prediction sets.", flush=True)
    return table
