"""Reviewer-driven controlled extensions for Project 2 (CNC sensor monitoring). Hypothesis-generating only.

Principles
----------
* Variants are PREDEFINED (thresholds, PCA k, physical sensor groups); no variant is chosen from outer-fold
  results and none is promoted as "best".
* One feature row per experiment; leave-one-experiment-out (LOEO) folds are produced by the SAME splitter as the
  original pipeline, so every variant is evaluated on identical outer folds.
* Imputer, scaler, correlation filter and PCA are fitted inside each training fold only.
* The base for the new variants is the 712-feature set without experiment length, duration-scaling energies and
  program-progress channels (`no_extensive_or_progress`); the original 864-feature set is kept only as the
  reproduction reference. `n_samples` is never used.
* The original results are reproduced exactly (reference variant) before the new numbers are reported.
* 14-18 independent experiments: all intervals are wide and describe evaluation sampling only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.cluster.hierarchy import leaves_list, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.bootstrap_ci import _fast_metrics, ci_rows_from_oof  # noqa: E402
from src.dimred import CorrelationFilter, fold_signature, loeo_collect, max_pca_components  # noqa: E402
from src.evaluation import classification_metrics, get_group_splitter  # noqa: E402
from src.explainability import brier, out_of_fold_importance, shap_positive_class_values  # noqa: E402
from src.feature_engineering import select_columns_by_group  # noqa: E402
from src.feature_structure import groupings  # noqa: E402
from src.robustness import build_variants as build_robustness_variants  # noqa: E402
from src.train_classical import build_model, make_pipeline, prepare_task_frame  # noqa: E402
from src.utils import load_config, project_root, save_json  # noqa: E402

MODELS = ("logistic_regression", "random_forest", "xgboost")
CORR_THRESHOLDS = (0.95, 0.80)
PCA_KS = (1, 2, 3, 5, 8)
REPRODUCTION_TOLERANCE = 1e-9
N_PERM_REPEATS = 20
PRIMARY_TASK = "tool_condition"


# -------------------------------------------------------------------------------------------------------
def make_variants(all_feats: list[str], config: dict, n_train_units: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rb = build_robustness_variants(all_feats, config)
    base = rb["no_extensive_or_progress"]["cols"]
    base_set = set(base)
    apriori = rb["apriori_reduced"]["cols"]
    assert set(apriori) <= base_set, "a-priori features must lie inside the length-free base set"
    assert "n_samples" not in base_set
    cap = max_pca_components(n_train_units)
    variants: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    def add(name, family, cols, **kw):
        variants.append({"name": name, "family": family, "cols": cols, **kw})

    add("orig_864", "reference", list(all_feats))
    add("base_712", "base", list(base))
    for t in CORR_THRESHOLDS:
        add(f"base_712_corr{t}", "correlation_filter", list(base), corr_threshold=t)
    for k in PCA_KS:
        if k > cap:
            skipped.append({"variant": f"base_712_pca{k}", "reason": f"k={k} > cap {cap} (= n_train_experiments//2, n_train={n_train_units})"})
            continue
        add(f"base_712_pca{k}", "pca", list(base), pca_k=k)
    for g in ("spindle", "axis_current", "axis_motion", "combined_sensors"):
        cols = [c for c in select_columns_by_group(all_feats, g, config) if c in base_set]
        add(f"phys_{g}", "physical_group", cols)
    add("phys_apriori18", "physical_group", list(apriori))
    return variants, skipped


# -------------------------------------------------------------------------------------------------------
def paired_delta_ba(oof_a: pd.DataFrame, oof_b: pd.DataFrame, n_boot: int, seed: int, alpha: float) -> dict[str, float]:
    """Paired bootstrap over experiments of BA(a) - BA(b) (same resampled experiments for both)."""
    assert np.array_equal(oof_a["experiment_id"].to_numpy(), oof_b["experiment_id"].to_numpy())
    y = oof_a["y_true"].to_numpy().astype(int)
    pa, pb = oof_a["y_pred"].to_numpy().astype(int), oof_b["y_pred"].to_numpy().astype(int)
    rng = np.random.default_rng(seed)
    n = len(y)
    point = _fast_metrics(y, pa, None)["balanced_accuracy"] - _fast_metrics(y, pb, None)["balanced_accuracy"]
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ma = _fast_metrics(y[idx], pa[idx], None)["balanced_accuracy"]
        mb = _fast_metrics(y[idx], pb[idx], None)["balanced_accuracy"]
        if np.isfinite(ma) and np.isfinite(mb):
            d.append(ma - mb)
    d = np.asarray(d)
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"dBA": float(point), "dBA_lo": float(lo), "dBA_hi": float(hi), "share_a_better": float(np.mean(d > 0)), "share_tied": float(np.mean(d == 0))}


# -------------------------------------------------------------------------------------------------------
# A. Descriptive correlation structure (label-free, all experiments of the task: description only)
# -------------------------------------------------------------------------------------------------------
def _impute(values: np.ndarray) -> np.ndarray:
    return SimpleImputer(strategy="median").fit_transform(values)


def correlation_structure(X: pd.DataFrame) -> dict[str, Any]:
    vals = _impute(X.to_numpy(dtype=float))
    names = np.array(X.columns[~np.isnan(X.to_numpy(dtype=float)).all(axis=0)])
    keep_var = vals.std(axis=0) > 1e-12
    vals, names = vals[:, keep_var], names[keep_var]
    corr = np.corrcoef(vals, rowvar=False)
    iu = np.triu_indices_from(corr, k=1)
    absr = np.abs(corr[iu])
    Z = StandardScaler().fit_transform(vals)
    evr = PCA(svd_solver="full").fit(Z).explained_variance_ratio_
    cum = np.cumsum(evr)
    curve = []
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        curve.append((t, int(CorrelationFilter(t).fit(Z).keep_.sum())))
    return {
        "n_experiments": int(vals.shape[0]), "n_features": int(vals.shape[1]), "corr": corr, "names": names, "abs_pairs": absr,
        "share_abs_r_ge_0.8": float(np.mean(absr >= 0.8)), "share_abs_r_ge_0.9": float(np.mean(absr >= 0.9)), "share_abs_r_ge_0.95": float(np.mean(absr >= 0.95)),
        "median_abs_r": float(np.median(absr)), "cum_evr": cum, "pcs_for_90": int(np.searchsorted(cum, 0.90) + 1), "pcs_for_95": int(np.searchsorted(cum, 0.95) + 1),
        "rank_limit": int(vals.shape[0] - 1), "keep_curve": curve,
    }


def _cluster_order(corr: np.ndarray) -> np.ndarray:
    d = 1 - np.abs(corr)
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0)
    return leaves_list(linkage(squareform(np.clip(d, 0, None), checks=False), method="average"))


def plot_correlation(cs: dict[str, Any], path: Path, channel_path: Path) -> None:
    corr, names = cs["corr"], cs["names"]
    order = _cluster_order(corr)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    im = axes[0].imshow(corr[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    axes[0].set_title(f"Pearson r, {cs['n_features']} features (hierarchically clustered)\n{cs['n_experiments']} experiments; descriptive, label-free", fontsize=9)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im, ax=axes[0], fraction=0.045)
    axes[1].hist(cs["abs_pairs"], bins=50, color="#4c72b0")
    axes[1].axvline(0.95, color="red", ls="--", lw=1)
    axes[1].set_xlabel("|r| between feature pairs")
    axes[1].set_ylabel("number of pairs")
    axes[1].set_title(f"{100 * cs['share_abs_r_ge_0.9']:.0f}% of pairs have |r| >= 0.9 (median |r| = {cs['median_abs_r']:.2f})", fontsize=9)
    t, k = zip(*cs["keep_curve"])
    axes[2].plot(t, k, "o-", color="#c44e52")
    axes[2].set_xlabel("correlation-filter threshold |r|")
    axes[2].set_ylabel("features kept")
    axes[2].set_title(f"Features surviving the filter (of {cs['n_features']}); rank limit = {cs['rank_limit']}", fontsize=9)
    fig.suptitle("Project 2 - feature redundancy (descriptive; with only 18 experiments, sizeable |r| also arises by chance alone)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    # channel-level view (mean statistic of each channel)
    idx = [i for i, n in enumerate(names) if n.endswith("__mean")]
    sub = corr[np.ix_(idx, idx)]
    o = _cluster_order(sub)
    labels = [names[idx[i]].split("__")[0] for i in o]
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(sub[np.ix_(o, o)], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_title("Pearson r between channel means (18 experiments, clustered)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(channel_path, dpi=160)
    plt.close(fig)


def pca_fold_spectrum(X: pd.DataFrame, config: dict, groups: pd.Series) -> dict[str, Any]:
    splitter = get_group_splitter(config["validation"]["primary"], groups.nunique())
    vals = X.to_numpy(dtype=float)
    ratios = []
    for tr, _ in splitter.split(vals, groups=groups):
        Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(vals[tr]))
        r = PCA(svd_solver="full").fit(Z).explained_variance_ratio_
        out = np.zeros(len(tr))
        out[: len(r)] = r
        ratios.append(out)
    fr = np.vstack(ratios)
    return {"fold_ratios": fr, "n_train": int(fr.shape[1]), "effective_rank_min": int((fr > 1e-8).sum(axis=1).min()), "effective_rank_max": int((fr > 1e-8).sum(axis=1).max())}


def plot_pca_variance(sp: dict[str, Any], cap: int, path: Path) -> None:
    fr = sp["fold_ratios"]
    k = np.arange(1, fr.shape[1] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    axes[0].bar(k, fr.mean(0), yerr=[fr.mean(0) - fr.min(0), fr.max(0) - fr.mean(0)], capsize=2, color="#4c72b0")
    axes[0].set_xlabel("principal component")
    axes[0].set_ylabel("explained variance ratio")
    axes[0].set_title("Per-fold PCA, standardised TRAINING experiments only\n(mean, min-max over the folds)", fontsize=9)
    cum = np.cumsum(fr, axis=1)
    axes[1].plot(k, cum.mean(0), "o-", color="#4c72b0")
    axes[1].fill_between(k, cum.min(0), cum.max(0), alpha=0.25, color="#4c72b0")
    for kk in PCA_KS:
        axes[1].axvline(kk, color="grey", ls=":", lw=0.8)
    axes[1].axvline(cap, color="red", ls="--", lw=1.2, label=f"sample-size cap k <= {cap} (= n_train // 2)")
    axes[1].set_xlabel("number of components")
    axes[1].set_ylabel("cumulative explained variance")
    axes[1].legend(fontsize=8)
    axes[1].set_title(f"Predefined k = {list(PCA_KS)} (dotted); rank limit = {sp['n_train'] - 1} components", fontsize=9)
    fig.suptitle(f"Project 2 - PCA spectrum, {PRIMARY_TASK} (712 features, {sp['n_train']} training experiments per fold)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# -------------------------------------------------------------------------------------------------------
# D. Grouped importance
# -------------------------------------------------------------------------------------------------------
def oof_grouped_shap_rf(X: pd.DataFrame, y: pd.Series, groups_map: dict[str, list[str]], config: dict, split_groups: pd.Series):
    """Out-of-fold SHAP of the random forest. Group value = SIGNED sum of SHAP values of the group's columns
    (SHAP is additive, so group values + base value reproduce the predicted probability exactly); importance =
    mean |group value|. Correlated columns split credit arbitrarily; group sizes differ, so larger groups get
    larger values partly just by containing more columns. Explains the model, not the machine."""
    import shap

    col2g = {c: g for g, cols in groups_map.items() for c in cols}
    G = list(groups_map)
    n = len(y)
    out = np.full((n, len(G)), np.nan)
    gap = 0.0
    splitter = get_group_splitter(config["validation"]["primary"], split_groups.nunique())
    for tr, te in splitter.split(X, y, split_groups):
        pipe = make_pipeline(build_model("random_forest", config)).fit(X.iloc[tr], y.iloc[tr])
        names = pipe[:-1].get_feature_names_out(np.asarray(list(X.columns), dtype=object))
        Xt = pipe[:-1].transform(X.iloc[te])
        explainer = shap.TreeExplainer(pipe.named_steps["model"])
        sv = shap_positive_class_values(explainer.shap_values(Xt), Xt.shape[0], Xt.shape[1])
        gi = np.array([G.index(col2g[nm]) for nm in names])
        for j in range(len(G)):
            out[te, j] = sv[:, gi == j].sum(axis=1)
        base = float(np.ravel(explainer.expected_value)[-1])
        gap = max(gap, float(np.max(np.abs(base + sv.sum(axis=1) - pipe.predict_proba(X.iloc[te])[:, 1]))))
    return pd.DataFrame({"group": G, "n_features": [len(groups_map[g]) for g in G], "mean_abs_grouped_shap": np.abs(out).mean(0), "mean_signed_grouped_shap": out.mean(0)}), gap


def plot_grouped(perm: pd.DataFrame, shap_df: pd.DataFrame, path: Path) -> None:
    schemes = list(perm.scheme.unique())
    fig, axes = plt.subplots(2, len(schemes), figsize=(5 * len(schemes), 8))
    for j, s in enumerate(schemes):
        p = perm[perm.scheme == s].sort_values("importance_mean")
        ax = axes[0, j]
        ax.barh([f"{g} ({n})" for g, n in zip(p.group, p.n_features)], p.importance_mean, xerr=p.importance_std, color=["#1f77b4" if v >= 0 else "#d62728" for v in p.importance_mean], capsize=2)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{s}: out-of-fold permutation (Brier increase)\nerror bars = permutation repeats only", fontsize=9)
        q = shap_df[shap_df.scheme == s].sort_values("mean_abs_grouped_shap")
        ax = axes[1, j]
        ax.barh([f"{g} ({n})" for g, n in zip(q.group, q.n_features)], q.mean_abs_grouped_shap, color="#55a868")
        ax.set_title(f"{s}: out-of-fold grouped SHAP, mean |signed group sum|", fontsize=9)
    fig.suptitle("Project 2 - tool condition: what the random forest relies on, by physical/statistical group (n features in brackets; "
                 "model reliance, not causal effect; larger groups tend to score higher)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# -------------------------------------------------------------------------------------------------------
# Plots for comparison tables
# -------------------------------------------------------------------------------------------------------
def plot_variant_forest(summary: pd.DataFrame, path: Path) -> None:
    tasks = list(summary.task.unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.2 * len(tasks), 8), sharey=True)
    axes = np.atleast_1d(axes)
    variants = [v for v in summary.variant.unique() if v != "dummy_ref"]
    colors = {"logistic_regression": "#4c72b0", "random_forest": "#dd8452", "xgboost": "#55a868"}
    for ax, task in zip(axes, tasks):
        sub = summary[summary.task == task]
        for mi, m in enumerate(MODELS):
            s = sub[sub.model == m].set_index("variant").reindex(variants)
            y = np.arange(len(variants)) + (mi - 1) * 0.22
            ax.errorbar(s.balanced_accuracy, y, xerr=[s.balanced_accuracy - s.balanced_accuracy_lo, s.balanced_accuracy_hi - s.balanced_accuracy], fmt="o", ms=4, color=colors[m], label=m, lw=1, capsize=1.5)
        ax.axvline(0.5, color="black", ls="--", lw=0.9)
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(variants, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("balanced accuracy (95% bootstrap CI over experiments)")
        ax.set_title(f"{task} (LOEO, n={int(sub.n.iloc[0])})", fontsize=10)
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Project 2 - all predefined variants on identical LOEO folds (dashed = chance). No variant is selected as 'best'.", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pca_performance(summary: pd.DataFrame, path: Path) -> None:
    tasks = list(summary.task.unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.6 * len(tasks), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    colors = {"logistic_regression": "#4c72b0", "random_forest": "#dd8452", "xgboost": "#55a868"}
    for ax, task in zip(axes, tasks):
        sub = summary[summary.task == task]
        for mi, m in enumerate(MODELS):
            s = sub[(sub.model == m) & sub.variant.str.startswith("base_712_pca")].copy()
            s["k"] = s.pca_k.astype(int)
            s = s.sort_values("k")
            ax.errorbar(s.k + (mi - 1) * 0.08, s.balanced_accuracy, yerr=[s.balanced_accuracy - s.balanced_accuracy_lo, s.balanced_accuracy_hi - s.balanced_accuracy], marker="o", ms=4, capsize=2, lw=1, color=colors[m], label=m)
            ref = sub[(sub.model == m) & (sub.variant == "base_712")].balanced_accuracy
            if len(ref):
                ax.axhline(ref.iloc[0], color=colors[m], ls=":", lw=1)
        ax.axhline(0.5, color="black", ls="--", lw=0.9)
        ax.set_xlabel("number of principal components k")
        ax.set_title(task, fontsize=10)
        ax.set_xticks(sorted(sub.pca_k.dropna().astype(int).unique()))
    axes[0].set_ylabel("balanced accuracy (95% CI)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Project 2 - balanced accuracy vs number of PCs (PCA inside each training fold; dotted = same model on all 712 features; k limited to n_train//2)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_stability(summary: pd.DataFrame, path: Path, task: str) -> None:
    sub = summary[(summary.task == task) & (summary.model != "dummy")]
    variants = [v for v in sub.variant.unique() if v != "dummy_ref"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    w = 0.8 / len(MODELS)
    for mi, m in enumerate(MODELS):
        s = sub[sub.model == m].set_index("variant").reindex(variants)
        axes[0].bar(np.arange(len(variants)) + mi * w, s.stability_prob_sd, w, label=m)
        axes[1].bar(np.arange(len(variants)) + mi * w, s.top20_jaccard, w, label=m)
    for ax, t in zip(axes, ["jackknife SD of predicted probability (lower = more stable)", "top-20 feature overlap between fold models, Jaccard (higher = more stable)\n(not defined for PCA variants)"]):
        ax.set_xticks(np.arange(len(variants)) + 0.4 - w / 2)
        ax.set_xticklabels(variants, rotation=70, ha="right", fontsize=7)
        ax.set_title(t, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"Project 2 - model stability across leave-one-experiment-out fold models ({task})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusions(summary: pd.DataFrame, task: str, labels: tuple[str, str], path: Path, variants: list[str]) -> None:
    fig, axes = plt.subplots(len(MODELS), len(variants), figsize=(2.6 * len(variants), 2.5 * len(MODELS)))
    for i, m in enumerate(MODELS):
        for j, v in enumerate(variants):
            ax = axes[i, j]
            r = summary[(summary.task == task) & (summary.model == m) & (summary.variant == v)]
            if r.empty:
                ax.axis("off")
                continue
            r = r.iloc[0]
            cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.sum() / 2))
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, int(cm[a, b]), ha="center", va="center", fontsize=11, color="white" if cm[a, b] > cm.sum() / 4 else "black")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["pred " + labels[0][:6], "pred " + labels[1][:6]], fontsize=6)
            ax.set_yticklabels([labels[0][:8], labels[1][:8]], fontsize=6)
            ax.set_title(f"{m[:6]} | {v}\nBA {r.balanced_accuracy:.2f}, F1m {r.macro_f1:.2f}", fontsize=7)
    fig.suptitle(f"{task}: pooled out-of-fold confusion matrices (rows = truth)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# -------------------------------------------------------------------------------------------------------
def run_spindle_position_diagnostic(feature_df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Leakage diagnostic for the 'phys_spindle' group.

    Spindle position channels (S1_ActualPosition / S1_CommandPosition) may accumulate with machining time, i.e.
    act as a hidden proxy for experiment length even after `n_samples` is removed. This compares the spindle group
    with and without those channels and with them alone, and reports how strongly each position feature tracks the
    experiment length (Spearman, descriptive, all experiments of the task)."""
    from scipy.stats import spearmanr

    config = config or load_config()
    root = project_root()
    tables_dir = root / config["paths"]["tables_dir"]
    b = config["bootstrap"]
    rows, corr_rows = [], []
    for task, task_cfg in config["tasks"].items():
        if not task_cfg.get("enabled"):
            continue
        frame, y, groups, all_feats = prepare_task_frame(feature_df, task, config)
        frame, y, groups = frame.reset_index(drop=True), y.reset_index(drop=True), groups.reset_index(drop=True)
        variants, _ = make_variants(all_feats, config, int(groups.nunique()) - 1)
        spindle = next(v for v in variants if v["name"] == "phys_spindle")["cols"]
        pos = [c for c in spindle if c.split("__")[0] in ("S1_ActualPosition", "S1_CommandPosition")]
        sets = {"phys_spindle": spindle, "phys_spindle_without_position": [c for c in spindle if c not in pos], "spindle_position_only": pos}
        for name, cols in sets.items():
            for model_name in ("logistic_regression", "random_forest"):
                res = loeo_collect(frame[cols], y, groups, model_name, config)
                row = ci_rows_from_oof(res["oof"], n_boot=b["n_boot"], seed=b["seed"], alpha=b["alpha"], task=task, variant=name, model=model_name)
                row["n_features"] = len(cols)
                rows.append(row)
        for c in pos:
            r = spearmanr(frame[c], frame["n_samples"])
            corr_rows.append({"task": task, "feature": c, "spearman_with_n_samples": float(r.statistic), "p": float(r.pvalue)})
    out = pd.DataFrame(rows)
    out.to_csv(tables_dir / "extended_spindle_position_diagnostic.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(tables_dir / "extended_spindle_position_vs_length.csv", index=False)
    return out


def run_extended(feature_df: pd.DataFrame, config: dict | None = None) -> dict[str, Any]:
    config = config or load_config()
    root = project_root()
    metrics_dir, tables_dir, figures_dir = (root / config["paths"][k] for k in ("metrics_dir", "tables_dir", "figures_dir"))
    reports_dir = root / "reports"
    for d in (metrics_dir, tables_dir, figures_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    b = config["bootstrap"]
    stored = json.loads((metrics_dir / "classical_results.json").read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    oof_store: dict[tuple[str, str, str], pd.DataFrame] = {}
    skipped_all: list[dict[str, Any]] = []
    reproduction: list[dict[str, Any]] = []
    task_meta: dict[str, Any] = {}
    sig_check: dict[str, bool] = {}

    for task, task_cfg in config["tasks"].items():
        if not task_cfg.get("enabled"):
            continue
        frame, y, groups, all_feats = prepare_task_frame(feature_df, task, config)
        frame, y, groups = frame.reset_index(drop=True), y.reset_index(drop=True), groups.reset_index(drop=True)
        n_train = int(groups.nunique()) - 1
        variants, skipped = make_variants(all_feats, config, n_train)
        skipped_all += [{"task": task, **s} for s in skipped]
        sigs = fold_signature(groups, config)
        task_meta[task] = {"n_experiments": int(len(frame)), "n_train_experiments": n_train, "pca_cap": max_pca_components(n_train),
                           "class_balance": {int(k): int(v) for k, v in y.value_counts().items()}, "n_folds": len(sigs)}
        for v in variants:
            X = frame[v["cols"]]
            for model_name in MODELS:
                print(f"  {task} | {v['name']} | {model_name} ({len(v['cols'])} features)", flush=True)
                res = loeo_collect(X, y, groups, model_name, config, corr_threshold=v.get("corr_threshold"), pca_k=v.get("pca_k"))
                oof = res["oof"].assign(task=task, variant=v["name"])
                oof_store[(task, v["name"], model_name)] = oof
                row = ci_rows_from_oof(oof, n_boot=b["n_boot"], seed=b["seed"], alpha=b["alpha"], task=task, variant=v["name"], model=model_name)
                met = classification_metrics(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy(), oof["y_prob"].to_numpy())
                (tn, fp), (fn, tp) = met["confusion_matrix"]
                row.update(
                    family=v["family"], n_features_input=len(v["cols"]), n_features_used_mean=res["n_features_used_mean"],
                    n_features_used_min=res["n_features_used_min"], n_features_used_max=res["n_features_used_max"],
                    pca_k=v.get("pca_k"), corr_threshold=v.get("corr_threshold"), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                    stability_prob_sd=res["stability_prob_sd"], stability_label_flip=res["stability_label_flip"], top20_jaccard=res["top_k_jaccard"],
                    brier=met["brier_score"], pr_auc=met["pr_auc"],
                )
                rows.append(row)
                if v["name"] == "orig_864":
                    old = stored[task]["models"][model_name]["balanced_accuracy"]
                    reproduction.append({"task": task, "model": model_name, "stored": old, "rerun": met["balanced_accuracy"], "abs_diff": abs(old - met["balanced_accuracy"])})
        # identical-folds evidence: every variant uses the same splitter on the same rows
        sig_check[task] = all(
            sorted(tuple(sorted(o["experiment_id"].tolist())) for _, o in oof.groupby("fold")) == sorted(tuple(sorted(groups.iloc[list(s)].tolist())) for s in sigs)
            for (t, _, _), oof in oof_store.items() if t == task
        )
        # dummy reference (stratified-random classifier; uses no features)
        res = loeo_collect(frame[all_feats], y, groups, "dummy", config)
        oof = res["oof"].assign(task=task, variant="dummy_ref")
        row = ci_rows_from_oof(oof, n_boot=b["n_boot"], seed=b["seed"], alpha=b["alpha"], task=task, variant="dummy_ref", model="dummy")
        met = classification_metrics(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy(), oof["y_prob"].to_numpy())
        (tn, fp), (fn, tp) = met["confusion_matrix"]
        row.update(family="reference", n_features_input=0, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), brier=met["brier_score"], pr_auc=met["pr_auc"])
        rows.append(row)
        oof_store[(task, "dummy_ref", "dummy")] = oof

    summary = pd.DataFrame(rows)
    max_diff = max(r["abs_diff"] for r in reproduction)
    repro = {"rows": reproduction, "max_abs_diff": max_diff, "reproduces_original": bool(max_diff <= REPRODUCTION_TOLERANCE)}

    # paired differences vs the 712-feature base, same model, same experiments
    paired = []
    for (task, variant, model), oof in oof_store.items():
        if model == "dummy" or variant == "base_712":
            continue
        ref = oof_store[(task, "base_712", model)]
        paired.append({"task": task, "variant": variant, "model": model, **paired_delta_ba(oof, ref, b["n_boot"], b["seed"], b["alpha"])})
    paired_df = pd.DataFrame(paired)
    summary = summary.merge(paired_df.rename(columns={"dBA": "dBA_vs_base712", "dBA_lo": "dBA_vs_base712_lo", "dBA_hi": "dBA_vs_base712_hi", "share_a_better": "share_better_than_base712", "share_tied": "share_tied_with_base712"}), on=["task", "variant", "model"], how="left")
    summary.to_csv(tables_dir / "extended_comparison.csv", index=False)
    summary[["task", "variant", "model", "tn", "fp", "fn", "tp"]].to_csv(tables_dir / "extended_confusion_matrices.csv", index=False)
    pd.DataFrame(skipped_all).to_csv(tables_dir / "extended_skipped_variants.csv", index=False)
    pd.concat(oof_store.values(), ignore_index=True).to_csv(metrics_dir / "extended_oof.csv", index=False)

    # ---- A/B descriptive structure on the primary task (all experiments; label-free description only) -----
    frame, y, groups, all_feats = prepare_task_frame(feature_df, PRIMARY_TASK, config)
    frame, y, groups = frame.reset_index(drop=True), y.reset_index(drop=True), groups.reset_index(drop=True)
    variants, _ = make_variants(all_feats, config, int(groups.nunique()) - 1)
    base_cols = next(v["cols"] for v in variants if v["name"] == "base_712")
    cs = correlation_structure(frame[base_cols])
    plot_correlation(cs, figures_dir / "extended_correlation_heatmap_features.png", figures_dir / "extended_correlation_heatmap_channels.png")
    sp = pca_fold_spectrum(frame[base_cols], config, groups)
    plot_pca_variance(sp, max_pca_components(int(groups.nunique()) - 1), figures_dir / "extended_pca_explained_variance.png")
    plot_variant_forest(summary[summary.variant != "dummy_ref"], figures_dir / "extended_variant_comparison_balanced_accuracy.png")
    plot_pca_performance(summary, figures_dir / "extended_performance_vs_pca_components.png")
    plot_stability(summary, figures_dir / "extended_model_stability.png", PRIMARY_TASK)
    label_pairs = {"tool_condition": ("unworn", "worn"), "visual_inspection": ("failed", "passed"), "machining_completed": ("aborted", "finalised")}
    for task in summary.task.unique():
        plot_confusions(summary, task, label_pairs.get(task, ("neg", "pos")), figures_dir / f"extended_confusion_matrices_{task}.png",
                        ["orig_864", "base_712", "base_712_corr0.95", "base_712_pca3", "phys_axis_current", "phys_apriori18"])

    # ---- D. grouped importance (tool condition, base_712) ------------------------------------------------
    scheme_groups = groupings(base_cols)
    X_base = frame[base_cols]
    perm_tables, shap_tables, shap_gap = [], [], 0.0
    for scheme, gmap in scheme_groups.items():
        print(f"  grouped permutation/SHAP | {scheme} ({len(gmap)} groups)", flush=True)
        p = out_of_fold_importance(X_base, y.to_numpy(), gmap, config, N_PERM_REPEATS, b["seed"])
        p["scheme"] = scheme
        p = p.rename(columns={"channel": "group"})
        p["oof_brier"] = p.attrs.get("oof_brier", np.nan)
        perm_tables.append(p)
        s, gap = oof_grouped_shap_rf(X_base, y, gmap, config, groups)
        s["scheme"] = scheme
        shap_tables.append(s)
        shap_gap = max(shap_gap, gap)
    perm = pd.concat(perm_tables, ignore_index=True)
    shap_df = pd.concat(shap_tables, ignore_index=True)
    perm.to_csv(tables_dir / "extended_grouped_permutation_importance.csv", index=False)
    shap_df.to_csv(tables_dir / "extended_grouped_shap.csv", index=False)
    plot_grouped(perm, shap_df, figures_dir / "extended_grouped_importance_tool_condition.png")

    results = {
        "bootstrap": b, "models": list(MODELS), "corr_thresholds": list(CORR_THRESHOLDS), "pca_ks": list(PCA_KS),
        "tasks": task_meta, "skipped_variants": skipped_all, "reproduction_check": repro,
        "identical_outer_folds_all_variants": sig_check, "grouped_shap_max_additivity_gap": shap_gap,
        "correlation_structure_tool_condition_712": {k: v for k, v in cs.items() if k not in ("corr", "names", "abs_pairs", "cum_evr")},
        "pca_fold_spectrum_tool_condition": {"effective_rank_min": sp["effective_rank_min"], "effective_rank_max": sp["effective_rank_max"], "n_train": sp["n_train"]},
    }
    save_json(results, metrics_dir / "extended_results.json")
    run_spindle_position_diagnostic(feature_df, config)
    print("Reproduction:", {k: repro[k] for k in ("max_abs_diff", "reproduces_original")}, "| folds identical:", sig_check, "| SHAP gap:", shap_gap, flush=True)
    return results
