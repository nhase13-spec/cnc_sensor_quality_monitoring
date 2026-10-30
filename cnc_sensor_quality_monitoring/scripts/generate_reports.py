"""Generate research reports from saved pipeline metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "results" / "metrics" / name).read_text(encoding="utf-8"))


def model_comparison_table(classical: dict) -> pd.DataFrame:
    rows = []
    for task, payload in classical.items():
        for model_name, metrics in payload["models"].items():
            rows.append(
                {
                    "task": task,
                    "model": model_name,
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "roc_auc": metrics.get("roc_auc"),
                    "validation": "leave_one_experiment_out",
                }
            )
        pb = payload["process_baseline"]
        rows.append(
            {
                "task": task,
                "model": "process_baseline_rf",
                "balanced_accuracy": pb["balanced_accuracy"],
                "macro_f1": pb["macro_f1"],
                "roc_auc": pb.get("roc_auc"),
                "validation": "leave_one_experiment_out",
            }
        )
    frame = pd.DataFrame(rows)
    out = ROOT / "results" / "tables" / "model_comparison.csv"
    frame.to_csv(out, index=False)
    return frame


def write_technical_report(classical: dict, anomaly: dict, temporal: dict, uncertainty: dict, audit: dict) -> None:
    tc = classical["tool_condition"]
    best_tc = max(tc["models"].items(), key=lambda kv: kv[1]["balanced_accuracy"])
    vi = classical["visual_inspection"]
    best_vi = max(vi["models"].items(), key=lambda kv: kv[1]["balanced_accuracy"])
    mc = classical["machining_completed"]
    best_mc = max(mc["models"].items(), key=lambda kv: kv[1]["balanced_accuracy"])

    text = f"""# Multivariate CNC Sensor Analytics for Tool Condition and Machining Quality Monitoring: An Explainable Machine Learning Study

## Abstract

This study evaluates whether multivariate in-process CNC telemetry from the University of Michigan SMART CNC milling dataset can predict tool condition and machining quality indicators under **leave-one-experiment-out (LOEO)** validation. With only **18 independent machining experiments**, predictive performance is modest and unstable across folds. Sensor-derived features improved tool-wear classification over feed-rate/clamp-pressure baselines (best LOEO balanced accuracy **{best_tc[1]['balanced_accuracy']:.3f}** with {best_tc[0]} vs **{tc['process_baseline']['balanced_accuracy']:.3f}** for process parameters). Visual inspection outcomes were largely explained by process settings already recorded in metadata. Machining completion was predicted most reliably by random forest on sensor features (**{best_mc[1]['balanced_accuracy']:.3f}** LOEO balanced accuracy).

## 1. Introduction

Project 2 complements a separate surface-roughness modelling study (machining parameters → measured roughness) by asking whether **in-process sensor streams** carry information about **tool condition** and **proxy quality labels** (visual inspection, run completion).

## 2. Industrial research motivation

Production monitoring must generalise across setups. This dataset provides controlled variation in tool wear, feed rate, and clamp pressure with synchronized axis/spindle telemetry at **10 Hz**.

## 3. Dataset description

- **Source:** Kaggle — CNC Mill Tool Wear (Sharon Sun / University of Michigan SMART lab, CC0)
- **Experiments:** {audit['n_experiments']} wax-block runs
- **Sensor rows:** {audit['n_sensor_rows_total']:,} (100 ms sampling)
- **Labels:** categorical `tool_condition` (worn/unworn), `machining_finalized`, `passed_visual_inspection` (14 labelled)
- **Not provided:** measured surface roughness, residual stress, subsurface damage, or fatigue life

Class distributions are documented in `results/tables/data_audit.json`.

## 4. Preprocessing

Invalid telemetry rows were removed when `M1_CURRENT_FEEDRATE == 50` or `X1_ActualPosition == 198`. Active machining layers (`Layer * Up/Down`) were retained. The README program-number heuristic was **not** applied because the active program number is 1 during valid cuts (applying it removed all samples).

## 5. Feature engineering

Per-experiment aggregates were computed for all numeric CNC channels: statistical, temporal, low-rate spectral (Welch, 10 Hz), and wavelet-energy proxies. This yields **1071** sensor features before dropping all-NaN columns.

## 6. Experimental design

Primary validation: **LOEO** (18 folds, one experiment held out each time). A **random stratified 5-fold** split on experiment-level rows is reported only to demonstrate split sensitivity (E2).

### Hypothesis outcomes

| ID | Result |
|----|--------|
| H1 | **Partially supported** for tool condition (sensors > process baseline). **Not supported** for visual inspection (process baseline matches sensor models). |
| H2 | Random-split vs LOEO differences are task-dependent; LOEO scores are not uniformly lower in this small sample. |
| H3 | **Not tested** — PyTorch unavailable in execution environment; temporal CNN skipped (see `temporal_results.json`). |
| H4 | **Partially supported** — reduced sensor groups can match full feature sets on some tasks (e.g. combined sensors LOEO **{tc['ablation']['combined_sensors']['metrics']['balanced_accuracy']:.3f}** vs all features **{tc['ablation']['all_features']['metrics']['balanced_accuracy']:.3f}** for tool condition). |

## 7. Results — classification (LOEO)

### Task A: Tool condition (n=18)

| Model | Balanced accuracy | ROC-AUC |
|-------|-------------------|---------|
| Process baseline (RF) | {tc['process_baseline']['balanced_accuracy']:.3f} | {tc['process_baseline'].get('roc_auc')} |
| Dummy | {tc['models']['dummy']['balanced_accuracy']:.3f} | {tc['models']['dummy'].get('roc_auc')} |
| Logistic regression | {tc['models']['logistic_regression']['balanced_accuracy']:.3f} | {tc['models']['logistic_regression'].get('roc_auc')} |
| Random forest | {tc['models']['random_forest']['balanced_accuracy']:.3f} | {tc['models']['random_forest'].get('roc_auc')} |
| XGBoost | {tc['models']['xgboost']['balanced_accuracy']:.3f} | {tc['models']['xgboost'].get('roc_auc')} |

**Best:** {best_tc[0]} (balanced accuracy {best_tc[1]['balanced_accuracy']:.3f}).

### Task B: Visual inspection (n=14 completed runs)

Best LOEO model: **{best_vi[0]}** with balanced accuracy **{best_vi[1]['balanced_accuracy']:.3f}** — equal to the feed/clamp process baseline (**{vi['process_baseline']['balanced_accuracy']:.3f}**), indicating strong confounding between experimental design factors and inspection outcome.

### Task C: Machining finalized (n=18)

Best LOEO model: **{best_mc[0]}** with balanced accuracy **{best_mc[1]['balanced_accuracy']:.3f}** (process baseline **{mc['process_baseline']['balanced_accuracy']:.3f}**).

## 8. Sensor ablation (tool condition, RF, LOEO)

See `results/metrics/classical_results.json` → `tool_condition.ablation`.

## 9. Anomaly detection (exploratory)

Unsupervised detectors trained on normal (unworn) folds show weak alignment with worn-tool labels at experiment level (e.g. Isolation Forest recall **{anomaly['isolation_forest']['tool_worn_reference']['recall']:.2f}**, precision **{anomaly['isolation_forest']['tool_worn_reference']['precision']:.2f}**). High scores do not imply physical defects.

## 10. Explainability

Permutation importance and SHAP plots for tool condition are saved under `results/figures/`. These indicate associative sensor contributions, not causal mechanisms.

## 11. Uncertainty

LOEO aggregated Brier score for tool condition (random forest): **{uncertainty['brier_score']:.3f}**. With 18 groups, calibration is indicative only.

## 12. Limitations

- Only **18 independent experiments** → high variance in LOEO folds; many folds are single-class.
- Wax machining on a research testbed — **not Rolls-Royce data** and not aerospace metal surface integrity.
- Visual inspection labels are categorical QA proxies, not metrology.
- Temporal deep learning experiment skipped pending PyTorch installation.

## 13. Relevance to aerospace manufacturing

Demonstrates methodology for leakage-safe sensor analytics and production-oriented sensor subsetting; results cannot be transferred quantitatively to Trent machining without new labelled production data.

## 14. Future work

- Collect more independent experiments with linked surface-integrity metrology.
- Run 1D CNN / TCN comparisons after installing PyTorch.
- Joint framework with Project 1 (parameters → roughness) without assuming dataset mergeability.

## 15. Conclusion

Multivariate CNC telemetry contains **limited but non-zero** information for tool-wear discrimination under strict experiment-level validation, while quality-proxy labels in this dataset are **strongly confounded** with process settings. Honest reporting of weak/generalisation-limited performance is essential for EngD-quality research.

## References

- Sun S. CNC Mill Tool Wear dataset, University of Michigan SMART lab (via Kaggle, CC0).
- Dataset README reproduced in `data/raw/README.txt`.
"""
    (ROOT / "reports" / "technical_research_report.md").write_text(text, encoding="utf-8")


def write_portfolio_summary(classical: dict) -> None:
    tc = classical["tool_condition"]["models"]["logistic_regression"]
    text = f"""# Sensor-Based CNC Machining Quality Monitoring Using Explainable Machine Learning

## EngD portfolio summary (concise)

**Motivation.** Aerospace finish-machining requires linking in-process signals to tool health and quality outcomes. This project investigates sensor-based monitoring as a complement to my separate study predicting **surface roughness from machining parameters** ([Project 1 repository](https://github.com/nhase13-spec/surface-roughness-machine-learning-model)).

**Dataset.** University of Michigan SMART CNC wax milling dataset (18 experiments, 10 Hz axis/spindle telemetry, CC0 via Kaggle).

**Method.** Leakage-safe LOEO validation, experiment-level feature extraction from active cutting segments, classical ML benchmarks, sensor ablation, SHAP/permutation explainability, and exploratory anomaly detection.

**Key results (actual LOEO metrics).**

- Tool condition: logistic regression balanced accuracy **{tc['balanced_accuracy']:.3f}** vs process-parameter baseline **{classical['tool_condition']['process_baseline']['balanced_accuracy']:.3f}**.
- Visual inspection (n=14): sensor models did **not** outperform feed/clamp baselines (both ≈ **{classical['visual_inspection']['process_baseline']['balanced_accuracy']:.3f}** balanced accuracy).
- Machining completion: random forest **{classical['machining_completed']['models']['random_forest']['balanced_accuracy']:.3f}** balanced accuracy with sensor features.

**Failure analysis.** Small sample size drives fold instability; several LOEO folds contain a single class (ROC-AUC undefined). Temporal CNN comparison was not executed because PyTorch could not be installed in the build environment.

**Limitations.** Not Rolls-Royce data; labels are not surface-integrity measurements; datasets are not physically linked for end-to-end modelling.

**Future research.** Acquire production multi-experiment campaigns with metrology; deploy grouped validation in digital thread prototypes; compare reduced sensor sets against research-grade arrays.

**Complementarity with Project 1.** Project 1 maps **parameters → roughness**; Project 2 maps **sensors → tool/inspection proxies**. Together they outline a future **sensor → process → surface integrity** research agenda without claiming a validated merged model.
"""
    (ROOT / "reports" / "portfolio_summary.md").write_text(text, encoding="utf-8")


def write_limitations(audit: dict) -> None:
    text = f"""# Known limitations and skipped work

1. **Sample size:** {audit['n_experiments']} independent experiments only.
2. **Label semantics:** Tool condition is categorical wear state, not measured tool wear depth.
3. **Visual inspection:** Missing for 4 incomplete runs; not equivalent to surface roughness or subsurface damage.
4. **Preprocessing:** Program-number invalid rule documented in README was deemed incompatible with observed data (program 1 during cuts).
5. **Deep learning:** PyTorch installation failed (network/SSL restrictions); E5 temporal comparison skipped.
6. **PyWavelets:** Not installed; wavelet energy uses a deterministic fallback implementation.
7. **Anomaly detection:** Unsupervised scores show poor recall for worn tools at experiment level.
8. **Raw data redistribution:** Verify Kaggle licence before publishing raw files in Git; use download instructions in README.
"""
    (ROOT / "reports" / "limitations.md").write_text(text, encoding="utf-8")


def main() -> None:
    classical = _load("classical_results.json")
    anomaly = _load("anomaly_results.json")
    temporal = _load("temporal_results.json")
    uncertainty = _load("uncertainty_results.json")
    audit = json.loads((ROOT / "results" / "tables" / "data_audit.json").read_text(encoding="utf-8"))
    model_comparison_table(classical)
    write_technical_report(classical, anomaly, temporal, uncertainty, audit)
    write_portfolio_summary(classical)
    write_limitations(audit)
    print("Reports written to reports/")


if __name__ == "__main__":
    main()
