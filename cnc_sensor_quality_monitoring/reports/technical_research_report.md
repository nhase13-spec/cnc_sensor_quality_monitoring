# Multivariate CNC Sensor Analytics for Tool Condition and Machining Quality Monitoring: An Explainable Machine Learning Study

## Abstract

This study evaluates whether multivariate in-process CNC telemetry from the University of Michigan SMART CNC milling dataset can predict tool condition and machining quality indicators under **leave-one-experiment-out (LOEO)** validation. With only **18 independent machining experiments**, predictive performance is modest and unstable across folds. Sensor-derived features improved tool-wear classification over feed-rate/clamp-pressure baselines (best LOEO balanced accuracy **0.550** with logistic_regression vs **0.388** for process parameters). Visual inspection outcomes were largely explained by process settings already recorded in metadata. Machining completion was predicted most reliably by random forest on sensor features (**1.000** LOEO balanced accuracy).

## 1. Introduction

Project 2 complements a separate surface-roughness modelling study (machining parameters → measured roughness) by asking whether **in-process sensor streams** carry information about **tool condition** and **proxy quality labels** (visual inspection, run completion).

## 2. Industrial research motivation

Production monitoring must generalise across setups. This dataset provides controlled variation in tool wear, feed rate, and clamp pressure with synchronized axis/spindle telemetry at **10 Hz**.

## 3. Dataset description

- **Source:** Kaggle — CNC Mill Tool Wear (Sharon Sun / University of Michigan SMART lab, CC0)
- **Experiments:** 18 wax-block runs
- **Sensor rows:** 25,286 (100 ms sampling)
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
| H4 | **Partially supported** — reduced sensor groups can match full feature sets on some tasks (e.g. combined sensors LOEO **0.588** vs all features **0.475** for tool condition). |

## 7. Results — classification (LOEO)

### Task A: Tool condition (n=18)

| Model | Balanced accuracy | ROC-AUC |
|-------|-------------------|---------|
| Process baseline (RF) | 0.388 | 0.275 |
| Dummy | 0.500 | 0.5 |
| Logistic regression | 0.550 | 0.675 |
| Random forest | 0.475 | 0.5125 |
| XGBoost | 0.300 | 0.1875 |

**Best:** logistic_regression (balanced accuracy 0.550).

### Task B: Visual inspection (n=14 completed runs)

Best LOEO model: **random_forest** with balanced accuracy **0.775** — equal to the feed/clamp process baseline (**0.775**), indicating strong confounding between experimental design factors and inspection outcome.

### Task C: Machining finalized (n=18)

Best LOEO model: **random_forest** with balanced accuracy **1.000** (process baseline **0.643**).

## 8. Sensor ablation (tool condition, RF, LOEO)

See `results/metrics/classical_results.json` → `tool_condition.ablation`.

## 9. Anomaly detection (exploratory)

Unsupervised detectors trained on normal (unworn) folds show weak alignment with worn-tool labels at experiment level (e.g. Isolation Forest recall **0.20**, precision **0.67**). High scores do not imply physical defects.

## 10. Explainability

Permutation importance and SHAP plots for tool condition are saved under `results/figures/`. These indicate associative sensor contributions, not causal mechanisms.

## 11. Uncertainty

LOEO aggregated Brier score for tool condition (random forest): **0.259**. With 18 groups, calibration is indicative only.

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
