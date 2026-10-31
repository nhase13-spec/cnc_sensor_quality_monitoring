"""Leakage-safe dimensionality reduction for leave-one-experiment-out (LOEO) validation.

Every data-dependent step is a step of the sklearn Pipeline, so it is re-fitted on the TRAINING experiments
of each fold only:

    median imputer -> StandardScaler -> [CorrelationFilter] -> [PCA(k)] -> classifier

* CorrelationFilter is label-free: it keeps the first column of every group of columns whose |Pearson r| with an
  already-kept column reaches the threshold; constant columns are dropped.
* PCA is applied AFTER the training-fold standardisation. k is predefined and never tuned on test results.
* The repeated windows/profiles of one experiment are already aggregated to one row per experiment; the
  experiment id is used only to define folds (it is never a feature).

Sample-size rule (`max_pca_components`): the number of INDEPENDENT training units is the number of training
experiments, not the number of sensor samples. At most half of them may be used as components.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import aggregate_oof_predictions, get_group_splitter
from src.train_classical import build_model


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Greedy label-free filter on training data: drops constants and near-duplicate columns."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):  # y is deliberately ignored
        X = np.asarray(X, dtype=float)
        p = X.shape[1]
        keep = X.std(axis=0) > 1e-12
        with np.errstate(all="ignore"):
            corr = np.nan_to_num(np.corrcoef(X, rowvar=False))
        for j in range(p):
            if keep[j]:
                keep[j + 1 :] &= ~(np.abs(corr[j, j + 1 :]) >= self.threshold)
        if not keep.any():
            keep[0] = True
        self.keep_ = keep
        self.n_features_in_ = p
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features, dtype=object)
        return names[self.keep_]


def max_pca_components(n_independent_train_units: int) -> int:
    """At most half the number of independent training units, and never above the rank limit n - 1."""
    return max(1, min(n_independent_train_units - 1, n_independent_train_units // 2))


def build_pipeline(model: Any, *, corr_threshold: float | None = None, pca_k: int | None = None) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if corr_threshold is not None:
        steps.append(("corr_filter", CorrelationFilter(corr_threshold)))
    if pca_k is not None:
        steps.append(("pca", PCA(n_components=pca_k, svd_solver="full")))
    steps.append(("model", model))
    return Pipeline(steps)


def fold_signature(groups: pd.Series, config: dict) -> list[tuple[int, ...]]:
    """Held-out row positions of each outer fold (used to prove identical folds across variants)."""
    splitter = get_group_splitter(config["validation"]["primary"], groups.nunique())
    return [tuple(te.tolist()) for _, te in splitter.split(np.zeros(len(groups)), groups=groups)]


def _top_names(pipe: Pipeline, columns: list[str], model_name: str, k: int) -> frozenset[str] | None:
    model = pipe.named_steps["model"]
    if model_name == "logistic_regression":
        imp = np.abs(model.coef_).ravel()
    elif hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    else:
        return None
    names = pipe[:-1].get_feature_names_out(np.asarray(columns, dtype=object))
    order = np.argsort(-imp, kind="stable")[:k]
    return frozenset(str(names[i]) for i in order if imp[i] > 0)


def loeo_collect(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_name: str,
    config: dict,
    *,
    corr_threshold: float | None = None,
    pca_k: int | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    """LOEO out-of-fold predictions plus model-stability descriptors.

    stability_prob_sd    mean over experiments of the SD (over fold models trained WITH that experiment, i.e. all
                         folds except its own) of the predicted positive-class probability.
    stability_label_flip mean over experiments of the share of those fold models that disagree with their majority label.
    top_k_jaccard        mean pairwise Jaccard overlap of each fold model's top-k important features
                         (only for variants where feature names survive, i.e. no PCA).
    """
    splitter = get_group_splitter(config["validation"]["primary"], groups.nunique())
    columns = list(X.columns)
    n = len(y)
    frames, probs_all, labels_all, fold_of_row = [], [], [], np.full(n, -1)
    n_used, tops, n_components = [], [], []
    for fold, (tr, te) in enumerate(splitter.split(X, y, groups)):
        model = build_model(model_name, config)
        if "n_jobs" in model.get_params():
            # Parallel tree averaging changes the floating-point summation order between runs (~1e-16). A predicted
            # probability of exactly 0.5 (common with 300 trees on 17 experiments) can then flip its label, so a
            # multithreaded forest is not reproducible at ties. Single-threaded fits are deterministic.
            model.set_params(n_jobs=1)
        pipe = build_pipeline(model, corr_threshold=corr_threshold, pca_k=pca_k)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        prob_te = pipe.predict_proba(X.iloc[te])[:, 1]
        pred_te = pipe.predict(X.iloc[te])
        frames.append(
            pd.DataFrame(
                {
                    "experiment_id": groups.iloc[te].to_numpy(),
                    "y_true": y.iloc[te].to_numpy(),
                    "y_pred": pred_te,
                    "y_prob": prob_te,
                    "fold": fold,
                    "model": model_name,
                }
            )
        )
        fold_of_row[te] = fold
        probs_all.append(pipe.predict_proba(X)[:, 1])
        labels_all.append(pipe.predict(X))
        n_used.append(int(pipe[:-1].transform(X.iloc[te[:1]]).shape[1]))
        if pca_k is None and model_name != "dummy":
            tops.append(_top_names(pipe, columns, model_name, top_k))
        if pca_k is not None:
            n_components.append(int(pipe.named_steps["pca"].n_components_))
    oof = aggregate_oof_predictions(frames)
    P, L = np.vstack(probs_all), np.vstack(labels_all)
    sd = np.empty(n)
    flip = np.empty(n)
    for i in range(n):
        keep = np.arange(P.shape[0]) != fold_of_row[i]
        sd[i] = P[keep, i].std(ddof=1)
        lab = L[keep, i]
        majority = int(lab.mean() >= 0.5)
        flip[i] = float(np.mean(lab != majority))
    jacc = np.nan
    valid = [t for t in tops if t]
    if len(valid) >= 2:
        jacc = float(np.mean([len(a & b) / len(a | b) for a, b in combinations(valid, 2)]))
    return {
        "oof": oof,
        "stability_prob_sd": float(sd.mean()),
        "stability_label_flip": float(flip.mean()),
        "top_k_jaccard": jacc,
        "n_features_used_mean": float(np.mean(n_used)),
        "n_features_used_min": int(np.min(n_used)),
        "n_features_used_max": int(np.max(n_used)),
    }
