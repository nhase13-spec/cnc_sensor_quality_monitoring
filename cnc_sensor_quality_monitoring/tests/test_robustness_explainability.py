"""Tests for bootstrap CIs, robustness variants and the repaired explainability helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score

from src.bootstrap_ci import _fast_metrics, bootstrap_classification_ci
from src.explainability import brier, channel_groups, in_sample_reliance, shap_positive_class_values
from src.robustness import build_variants
from src.train_classical import build_model, make_pipeline
from src.utils import load_config


def test_fast_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 30)
    prob = np.clip(y * 0.3 + rng.random(30) * 0.7, 0, 1)
    pred = (prob > 0.5).astype(int)
    m = _fast_metrics(y, pred, prob)
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, pred))
    assert m["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro"))
    assert m["recall"] == pytest.approx(recall_score(y, pred))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, prob))


def test_bootstrap_interval_is_ordered_and_reproducible():
    rng = np.random.default_rng(1)
    y = np.array([0, 1] * 9)
    prob = np.clip(y * 0.2 + rng.random(18) * 0.8, 0, 1)
    pred = (prob > 0.5).astype(int)
    ci = bootstrap_classification_ci(y, pred, prob, n_boot=500, seed=3)
    ba = ci["balanced_accuracy"]
    assert 0.0 <= ba["lo"] <= ba["hi"] <= 1.0
    assert ba["point"] == pytest.approx(balanced_accuracy_score(y, pred))
    # Same seed -> identical interval (reproducibility); the interval is wide for n=18.
    again = bootstrap_classification_ci(y, pred, prob, n_boot=500, seed=3)
    assert again["balanced_accuracy"] == ba
    assert ba["hi"] - ba["lo"] > 0.1


def test_bootstrap_perfect_predictions_give_degenerate_interval():
    y = np.array([0, 1] * 9)
    ci = bootstrap_classification_ci(y, y.copy(), y.astype(float), n_boot=200, seed=0)
    assert ci["balanced_accuracy"]["lo"] == ci["balanced_accuracy"]["hi"] == 1.0


def test_variants_exclude_n_samples_and_keep_fixed_apriori_set():
    cfg = load_config()
    channels = ["S1_CurrentFeedback", "S1_OutputPower", "X1_CurrentFeedback", "Y1_CurrentFeedback", "X1_OutputPower", "Y1_OutputPower"]
    stats = ["mean", "std", "iqr"]
    cols = ["n_samples"] + [f"{c}__{s}" for c in channels for s in stats]
    cols += ["S1_OutputPower__energy", "M1_sequence_number__max", "X1_ActualPosition__wavelet_energy"]
    variants = build_variants(cols, cfg)
    assert "n_samples" not in variants["no_n_samples"]["cols"]
    strict = variants["no_extensive_or_progress"]["cols"]
    assert "S1_OutputPower__energy" not in strict
    assert "M1_sequence_number__max" not in strict
    assert "X1_ActualPosition__wavelet_energy" not in strict
    assert len(variants["apriori_reduced"]["cols"]) == len(channels) * len(stats)
    assert variants["kbest10"]["select_k"] == 10
    assert variants["n_samples_only"]["cols"] == ["n_samples"]


def test_variants_fail_loudly_when_apriori_feature_missing():
    cfg = load_config()
    with pytest.raises(KeyError):
        build_variants(["n_samples", "S1_CurrentFeedback__mean"], cfg)


def test_in_sample_reliance_is_not_degenerate_and_ranks_informative_channel_first():
    cfg = load_config()
    rng = np.random.default_rng(5)
    n = 24
    y = np.array([0, 1] * (n // 2))
    X = pd.DataFrame(
        {
            "signal__mean": y + rng.normal(0, 0.3, n),
            "signal__std": y * 0.5 + rng.normal(0, 0.3, n),
            "noise__mean": rng.normal(size=n),
            "noise__std": rng.normal(size=n),
        }
    )
    pipe = make_pipeline(build_model("random_forest", cfg))
    pipe.fit(X, y)
    table = in_sample_reliance(pipe, X, y, channel_groups(list(X.columns)), n_repeats=10, seed=0)
    assert not np.allclose(table["importance_mean"], 0.0)
    top = table.sort_values("importance_mean", ascending=False).iloc[0]["channel"]
    assert top == "signal"


@pytest.mark.parametrize("layout", ["list", "3d", "2d"])
def test_shap_positive_class_values_handles_all_layouts(layout):
    n, f = 5, 7
    base = np.random.default_rng(0).normal(size=(n, f))
    raw = {"list": [base * -1, base], "3d": np.stack([base * -1, base], axis=2), "2d": base}[layout]
    out = shap_positive_class_values(raw, n, f)
    assert out.shape == (n, f)
    np.testing.assert_allclose(out, base)


def test_shap_shape_mismatch_raises():
    with pytest.raises(RuntimeError):
        shap_positive_class_values(np.zeros((5, 3)), 5, 7)


def test_brier_known_value():
    assert brier(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(0.25)
