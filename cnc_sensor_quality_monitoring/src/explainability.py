"""Model explainability for the tool-condition classifier (Random Forest).

What was wrong in the first version (kept here as a record, because the same mistakes are easy to repeat):

1. Permutation importance was computed on the TRAINING data with the default accuracy score. A forest
   fitted on 17 experiments classifies all 17 perfectly, so shuffling one feature never changes the
   accuracy and every importance is exactly 0 -> the chart was meaningless.
2. ``TreeExplainer.shap_values`` returns an array of shape (n_samples, n_features, n_classes) in current
   SHAP versions, not a list. The old code passed the whole 3-D array to ``summary_plot``.
3. Importances were summed over the 864 individual features of a channel, which inflates channels that
   simply have more features.

What this module does instead
-----------------------------
* Scores with the Brier score (continuous in the predicted probability, so it does not saturate).
* Permutes whole CHANNELS at once (all of a channel's features shuffled with the same row permutation).
* Reports two different quantities and labels them as such:
    - ``in_sample_reliance``: how much the final model (fitted on all 18 experiments) relies on a channel
      for the data it was trained on. Describes the model, says nothing about generalisation.
    - ``out_of_fold``: for experiment i, the model that never saw experiment i is applied to experiment i
      with a channel's values replaced by those of another experiment. Positive = the channel genuinely
      helps predict unseen experiments. With a model that does not generalise this is ~0, and that is
      the honest answer.
* Raises if the output is degenerate (all zeros / non-finite) instead of silently writing an empty chart.
* Handles every SHAP output layout and asserts the shape before plotting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402

from src.robustness import build_variants  # noqa: E402
from src.train_classical import build_model, make_pipeline, prepare_task_frame  # noqa: E402
from src.utils import load_config, project_root  # noqa: E402


def channel_groups(names: list[str]) -> dict[str, list[str]]:
    """Group feature names by sensor channel (text before the first double underscore)."""
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(name.split("__")[0] if "__" in name else name, []).append(name)
    return groups


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def in_sample_reliance(pipeline, X: pd.DataFrame, y: np.ndarray, groups: dict[str, list[str]], n_repeats: int, seed: int) -> pd.DataFrame:
    """Increase in Brier score when a channel is shuffled across experiments (fitted data)."""
    rng = np.random.default_rng(seed)
    base = brier(y, pipeline.predict_proba(X)[:, 1])
    rows = []
    n = len(X)
    for channel, cols in groups.items():
        deltas = []
        for _ in range(n_repeats):
            perm = rng.permutation(n)
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[perm]
            deltas.append(brier(y, pipeline.predict_proba(Xp)[:, 1]) - base)
        rows.append({"channel": channel, "n_features": len(cols), "importance_mean": np.mean(deltas), "importance_std": np.std(deltas)})
    return pd.DataFrame(rows)


def out_of_fold_importance(
    X: pd.DataFrame, y: np.ndarray, groups: dict[str, list[str]], config: dict, n_repeats: int, seed: int
) -> pd.DataFrame:
    """Leave-one-experiment-out permutation importance (valid for unseen experiments)."""
    n = len(X)
    rng = np.random.default_rng(seed)
    channels = list(groups.keys())
    col_index = {ch: [X.columns.get_loc(c) for c in cols] for ch, cols in groups.items()}
    # perms[g, r, i] = which experiment's channel values replace experiment i's values.
    perms = np.stack([[rng.permutation(n) for _ in range(n_repeats)] for _ in channels])
    values = X.to_numpy(dtype=float)

    base_prob = np.zeros(n)
    perm_prob = np.zeros((len(channels), n_repeats, n))
    for i in range(n):
        train_idx = np.array([j for j in range(n) if j != i])
        pipe = make_pipeline(build_model("random_forest", config))
        pipe.fit(X.iloc[train_idx], y[train_idx])
        base_prob[i] = pipe.predict_proba(X.iloc[[i]])[0, 1]
        batch = np.tile(values[i], (len(channels) * n_repeats, 1))
        k = 0
        for g, ch in enumerate(channels):
            idx = col_index[ch]
            for r in range(n_repeats):
                batch[k, idx] = values[perms[g, r, i]][idx]
                k += 1
        probs = pipe.predict_proba(pd.DataFrame(batch, columns=X.columns))[:, 1]
        perm_prob[:, :, i] = probs.reshape(len(channels), n_repeats)

    base = brier(y, base_prob)
    rows = []
    for g, ch in enumerate(channels):
        deltas = [brier(y, perm_prob[g, r]) - base for r in range(n_repeats)]
        rows.append({"channel": ch, "n_features": len(groups[ch]), "importance_mean": np.mean(deltas), "importance_std": np.std(deltas)})
    out = pd.DataFrame(rows)
    out.attrs["oof_brier"] = base
    return out


def shap_positive_class_values(raw, n_samples: int, n_features: int) -> np.ndarray:
    """Return SHAP values of the positive class as (n_samples, n_features) for any SHAP output layout."""
    if isinstance(raw, list):  # older SHAP: list [class0, class1]
        values = np.asarray(raw[1])
    else:
        values = np.asarray(raw)
        if values.ndim == 3:  # newer SHAP: (n_samples, n_features, n_classes)
            values = values[:, :, 1]
    if values.shape != (n_samples, n_features):
        raise RuntimeError(f"Unexpected SHAP shape {values.shape}; expected {(n_samples, n_features)}.")
    return values


def _assert_not_degenerate(table: pd.DataFrame, label: str) -> None:
    imp = table["importance_mean"].to_numpy(dtype=float)
    if not np.isfinite(imp).all() or np.allclose(imp, 0.0):
        raise RuntimeError(f"Degenerate importance output for '{label}': all values are zero or non-finite.")


def _plot_importance(ax, table: pd.DataFrame, title: str, top_n: int = 15) -> None:
    top = table.reindex(table["importance_mean"].abs().sort_values(ascending=False).index).head(top_n).iloc[::-1]
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in top["importance_mean"]]
    ax.barh(top["channel"], top["importance_mean"], xerr=top["importance_std"], color=colors, ecolor="grey", capsize=2)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Increase in Brier score when channel is shuffled\n(mean +/- SD over repeats)")
    ax.set_title(title, fontsize=10)


def _explain_one_set(task_frame, y, cols, suffix, config, n_repeats, seed, figures_dir, tables_dir) -> list[dict]:
    X = task_frame[cols].reset_index(drop=True)
    yv = y.to_numpy()
    groups = channel_groups(cols)

    pipeline = make_pipeline(build_model("random_forest", config))
    pipeline.fit(X, yv)

    reliance = in_sample_reliance(pipeline, X, yv, groups, n_repeats, seed)
    _assert_not_degenerate(reliance, f"in_sample_reliance{suffix}")
    oof = out_of_fold_importance(X, yv, groups, config, n_repeats, seed)
    _assert_not_degenerate(oof, f"out_of_fold{suffix}")

    table = pd.concat(
        [reliance.assign(kind="in_sample_reliance"), oof.assign(kind="out_of_fold")], ignore_index=True
    ).sort_values(["kind", "importance_mean"], ascending=[True, False])
    table.insert(0, "feature_set", suffix.strip("_") or "all_minus_n_samples")
    table.to_csv(tables_dir / f"permutation_importance_tool_condition{suffix}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _plot_importance(axes[0], reliance, "Model reliance (fitted on all 18 experiments)\nDescribes the model, NOT generalisation")
    _plot_importance(axes[1], oof, "Out-of-fold importance (leave-one-experiment-out)\nValid for unseen experiments; ~0 = no generalising signal")
    fig.suptitle(f"Tool condition - channel permutation importance ({len(cols)} features, {len(groups)} channels)", fontsize=11)
    fig.tight_layout()
    fig.savefig(figures_dir / f"feature_importance_tool_condition{suffix}.png", dpi=200)
    plt.close(fig)

    # SHAP: tree model on the transformed (imputed + scaled) features of all 18 experiments.
    transformed = pipeline[:-1].transform(X)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    sv = shap_positive_class_values(explainer.shap_values(transformed), transformed.shape[0], transformed.shape[1])
    if np.allclose(sv, 0.0):
        raise RuntimeError(f"All SHAP values are zero for '{suffix}'.")
    shap.summary_plot(sv, transformed, feature_names=cols, show=False, max_display=15)
    plt.title("SHAP (positive class = worn) - fitted model, all 18 experiments\nExplains the model's behaviour; not evidence of generalisation", fontsize=9)
    plt.tight_layout()
    plt.savefig(figures_dir / f"shap_summary_tool_condition{suffix}.png", dpi=200, bbox_inches="tight")
    plt.close()

    return [
        {
            "feature_set": suffix.strip("_") or "all_minus_n_samples",
            "n_features": len(cols),
            "oof_brier": float(oof.attrs["oof_brier"]),
            "top_reliance_channel": reliance.sort_values("importance_mean", ascending=False).iloc[0]["channel"],
            "max_oof_importance": float(oof["importance_mean"].max()),
        }
    ]


def run_explainability(feature_df: pd.DataFrame, config: dict | None = None, n_repeats: int = 20) -> list[dict]:
    config = config or load_config()
    seed = config["project"]["random_seed"]
    figures_dir = project_root() / config["paths"]["figures_dir"]
    tables_dir = project_root() / config["paths"]["tables_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    frame, y, _, all_feats = prepare_task_frame(feature_df, "tool_condition", config)
    variants = build_variants(all_feats, config)
    summaries: list[dict] = []
    # Primary figure names keep their original file names (the broken originals are replaced).
    for variant, suffix in (("no_n_samples", ""), ("apriori_reduced", "_reduced")):
        print(f"  explainability: {variant} ({len(variants[variant]['cols'])} features)", flush=True)
        summaries += _explain_one_set(frame, y, variants[variant]["cols"], suffix, config, n_repeats, seed, figures_dir, tables_dir)
    return summaries
