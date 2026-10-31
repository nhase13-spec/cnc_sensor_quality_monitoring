"""Leakage-prevention and reproducibility tests for the reviewer-driven extensions (Project 2)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut

from src.dimred import CorrelationFilter, build_pipeline, fold_signature, loeo_collect, max_pca_components
from src.evaluation import classification_metrics
from src.extended_analysis import MODELS, PCA_KS, make_variants, oof_grouped_shap_rf, paired_delta_ba
from src.feature_structure import groupings, parse_feature
from src.train_classical import build_model, prepare_task_frame
from src.utils import load_config, project_root

CACHE = project_root() / "data" / "processed" / "experiment_features.csv"
needs_data = pytest.mark.skipif(not CACHE.is_file(), reason="processed feature matrix not available")


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def feature_df():
    return pd.read_csv(CACHE)


def _noise(n=18, p=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(rng.permutation(np.r_[np.ones(n // 2), np.zeros(n - n // 2)]).astype(int))
    return X, y, pd.Series(np.arange(1, n + 1))


# ------------------------------------------------------------------ structure / rules ---------------------
def test_pca_cap_rule():
    assert max_pca_components(17) == 8
    assert max_pca_components(13) == 6
    assert max_pca_components(2) == 1
    assert max_pca_components(1) == 1


@needs_data
def test_variants_use_no_length_or_id_features_and_respect_pca_cap(feature_df, config):
    for task, n_train_expected in (("tool_condition", 17), ("visual_inspection", 13), ("machining_completed", 17)):
        frame, y, groups, all_feats = prepare_task_frame(feature_df, task, config)
        assert groups.nunique() - 1 == n_train_expected
        variants, skipped = make_variants(all_feats, config, int(groups.nunique()) - 1)
        base = next(v for v in variants if v["name"] == "base_712")["cols"]
        for v in variants:
            if v["name"] != "orig_864":
                for forbidden in ("n_samples", "experiment_id", "tool_condition", "machining_finalized", "passed_visual_inspection", "feedrate", "clamp_pressure"):
                    assert forbidden not in v["cols"]
                assert set(v["cols"]) <= set(base)
            if v.get("pca_k"):
                assert v["pca_k"] <= max_pca_components(n_train_expected)
        for k in PCA_KS:
            if k > max_pca_components(n_train_expected):
                assert any(s["variant"] == f"base_712_pca{k}" for s in skipped)
    # visual inspection has 14 experiments -> k=8 must be skipped
    _, _, g_vis, f_vis = prepare_task_frame(feature_df, "visual_inspection", config)
    _, skipped = make_variants(f_vis, config, int(g_vis.nunique()) - 1)
    assert any(s["variant"] == "base_712_pca8" for s in skipped)


@needs_data
def test_feature_groupings_partition_base_set(feature_df, config):
    _, _, _, all_feats = prepare_task_frame(feature_df, "tool_condition", config)
    base = next(v for v in make_variants(all_feats, config, 17)[0] if v["name"] == "base_712")["cols"]
    g = groupings(base)
    for scheme, groups in g.items():
        flat = [c for cols in groups.values() for c in cols]
        assert sorted(flat) == sorted(base), scheme
    assert parse_feature("S1_OutputPower__mean") == ("S1_OutputPower", "mean")
    with pytest.raises(ValueError):
        parse_feature("n_samples")


# ------------------------------------------------------------------ leakage prevention --------------------
def test_correlation_filter_is_label_free_and_deterministic():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(30, 3))
    X = np.column_stack([base, base[:, 0] * -3 + 1e-3 * rng.normal(size=30), np.ones(30)])
    a = CorrelationFilter(0.95).fit(X, rng.integers(0, 2, 30))
    b = CorrelationFilter(0.95).fit(X, rng.integers(0, 2, 30))
    assert np.array_equal(a.keep_, b.keep_)
    assert list(a.keep_) == [True, True, True, False, False]
    assert a.transform(rng.normal(size=(4, 5))).shape == (4, 3)
    names = a.get_feature_names_out(["a", "b", "c", "d", "e"])
    assert list(names) == ["a", "b", "c"]


def test_filter_and_pca_are_fitted_on_training_experiments_only():
    X, y, groups = _noise(n=18, p=60, seed=2)
    held = 5
    tr = np.array([i for i in range(18) if i != held])
    pipe = build_pipeline(build_model("logistic_regression", load_config()), corr_threshold=0.9, pca_k=3).fit(X.iloc[tr], y.iloc[tr])
    scaler, pca, filt = pipe.named_steps["scaler"], pipe.named_steps["pca"], pipe.named_steps["corr_filter"]
    assert scaler.n_samples_seen_ == 17
    assert pca.n_samples_ == 17
    assert np.allclose(pca.mean_, 0, atol=1e-8)  # mean of standardised TRAIN rows
    assert filt.n_features_in_ == 60
    # modifying the held-out experiment cannot change any fitted object
    X2 = X.copy()
    X2.iloc[held] = 1e6
    pipe2 = build_pipeline(build_model("logistic_regression", load_config()), corr_threshold=0.9, pca_k=3).fit(X2.iloc[tr], y.iloc[tr])
    assert np.array_equal(pipe2.named_steps["corr_filter"].keep_, filt.keep_)
    assert np.allclose(pipe2.named_steps["pca"].components_, pca.components_)


def test_noise_features_show_no_skill_in_fold_but_leaky_selection_does(config):
    X, y, groups = _noise(n=18, p=300, seed=3)
    honest = loeo_collect(X, y, groups, "logistic_regression", config, corr_threshold=0.9, pca_k=3)["oof"]
    ba_honest = classification_metrics(honest.y_true.to_numpy(), honest.y_pred.to_numpy(), honest.y_prob.to_numpy())["balanced_accuracy"]
    cols = list(X.columns[SelectKBest(f_classif, k=10).fit(X, y).get_support()])  # selection on ALL experiments incl. held-out: leaky
    pred = np.zeros(len(y), dtype=int)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        pred[te] = LogisticRegression(max_iter=2000).fit(X.iloc[tr][cols], y.iloc[tr]).predict(X.iloc[te][cols])
    ba_leaky = classification_metrics(y.to_numpy(), pred, pred.astype(float))["balanced_accuracy"]
    assert ba_honest < 0.75  # no real signal on pure noise
    assert ba_leaky > ba_honest + 0.1  # the harness would expose a leaky selector


@needs_data
def test_each_fold_holds_out_exactly_one_whole_experiment(feature_df, config):
    for task in config["tasks"]:
        frame, y, groups, _ = prepare_task_frame(feature_df, task, config)
        groups = groups.reset_index(drop=True)
        sig = fold_signature(groups, config)
        assert len(sig) == groups.nunique()
        assert sum(len(s) for s in sig) == len(groups)
        for s in sig:
            assert len(set(groups.iloc[list(s)])) == 1
        assert len(frame) == groups.nunique()  # one row per experiment: windows are already aggregated


@needs_data
def test_experiment_id_and_targets_are_not_in_any_feature_list(feature_df, config):
    for task in config["tasks"]:
        _, _, _, all_feats = prepare_task_frame(feature_df, task, config)
        for forbidden in ("experiment_id", "tool_condition", "machining_finalized", "passed_visual_inspection", "feedrate", "clamp_pressure", "material"):
            assert forbidden not in all_feats


# ------------------------------------------------------------------ reproducibility -----------------------
@needs_data
def test_loeo_collect_is_deterministic_and_reproduces_original_results(feature_df, config):
    stored = json.loads((project_root() / config["paths"]["metrics_dir"] / "classical_results.json").read_text(encoding="utf-8"))
    frame, y, groups, all_feats = prepare_task_frame(feature_df, "tool_condition", config)
    frame, y, groups = frame.reset_index(drop=True), y.reset_index(drop=True), groups.reset_index(drop=True)
    for m in ("logistic_regression", "random_forest"):
        a = loeo_collect(frame[all_feats], y, groups, m, config)
        b = loeo_collect(frame[all_feats], y, groups, m, config)
        # bit-identical probabilities AND labels (a multithreaded forest is not: a 0.5 tie can flip at the 1e-16 level)
        assert np.array_equal(a["oof"].y_prob.to_numpy(), b["oof"].y_prob.to_numpy())
        assert np.array_equal(a["oof"].y_pred.to_numpy(), b["oof"].y_pred.to_numpy())
        met = classification_metrics(a["oof"].y_true.to_numpy(), a["oof"].y_pred.to_numpy(), a["oof"].y_prob.to_numpy())
        assert abs(met["balanced_accuracy"] - stored["tool_condition"]["models"][m]["balanced_accuracy"]) < 1e-9
        assert 0 <= a["stability_prob_sd"] <= 0.5
        assert 0 <= a["stability_label_flip"] <= 0.5
        assert np.isnan(a["top_k_jaccard"]) or 0 <= a["top_k_jaccard"] <= 1


# ------------------------------------------------------------------ importance / paired CI ----------------
@needs_data
def test_grouped_shap_is_additive(feature_df, config):
    frame, y, groups, all_feats = prepare_task_frame(feature_df, "tool_condition", config)
    frame, y, groups = frame.reset_index(drop=True), y.reset_index(drop=True), groups.reset_index(drop=True)
    base = next(v for v in make_variants(all_feats, config, 17)[0] if v["name"] == "phys_apriori18")["cols"]
    table, gap = oof_grouped_shap_rf(frame[base], y, groupings(base)["signal_type"], config, groups)
    assert gap < 1e-6
    assert (table.mean_abs_grouped_shap >= 0).all()


def test_paired_delta_identical_predictions_is_zero_and_sign_is_correct():
    y = np.array([0, 1] * 9)
    good = pd.DataFrame({"experiment_id": np.arange(18), "y_true": y, "y_pred": y})
    bad = good.assign(y_pred=1 - y)
    same = paired_delta_ba(good, good, 500, 1, 0.05)
    assert same["dBA"] == 0 and same["dBA_lo"] == 0 and same["dBA_hi"] == 0
    better = paired_delta_ba(good, bad, 500, 1, 0.05)
    assert better["dBA"] == pytest.approx(1.0) and better["dBA_lo"] > 0.9
    with pytest.raises(AssertionError):
        paired_delta_ba(good, bad.assign(experiment_id=np.arange(18)[::-1]), 10, 1, 0.05)


def test_models_are_the_predefined_set():
    assert MODELS == ("logistic_regression", "random_forest", "xgboost")
