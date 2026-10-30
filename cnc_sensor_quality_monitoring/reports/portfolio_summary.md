# Sensor-Based CNC Machining Quality Monitoring Using Explainable Machine Learning

## EngD portfolio summary (concise)

**Motivation.** Aerospace finish-machining requires linking in-process signals to tool health and quality outcomes. This project investigates sensor-based monitoring as a complement to my separate study predicting **surface roughness from machining parameters** ([Project 1 repository](https://github.com/nhase13-spec/surface-roughness-machine-learning-model)).

**Dataset.** University of Michigan SMART CNC wax milling dataset (18 experiments, 10 Hz axis/spindle telemetry, CC0 via Kaggle).

**Method.** Leakage-safe LOEO validation, experiment-level feature extraction from active cutting segments, classical ML benchmarks, sensor ablation, SHAP/permutation explainability, and exploratory anomaly detection.

**Key results (actual LOEO metrics).**

- Tool condition: logistic regression balanced accuracy **0.550** vs process-parameter baseline **0.388**.
- Visual inspection (n=14): sensor models did **not** outperform feed/clamp baselines (both ≈ **0.775** balanced accuracy).
- Machining completion: random forest **1.000** balanced accuracy with sensor features.

**Failure analysis.** Small sample size drives fold instability; several LOEO folds contain a single class (ROC-AUC undefined). Temporal CNN comparison was not executed because PyTorch could not be installed in the build environment.

**Limitations.** Not Rolls-Royce data; labels are not surface-integrity measurements; datasets are not physically linked for end-to-end modelling.

**Future research.** Acquire production multi-experiment campaigns with metrology; deploy grouped validation in digital thread prototypes; compare reduced sensor sets against research-grade arrays.

**Complementarity with Project 1.** Project 1 maps **parameters → roughness**; Project 2 maps **sensors → tool/inspection proxies**. Together they outline a future **sensor → process → surface integrity** research agenda without claiming a validated merged model.
