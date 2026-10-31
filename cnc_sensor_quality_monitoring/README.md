# Sensor-Based CNC Tool Wear and Machining Quality Monitoring Using Explainable Machine Learning

Research-grade Python pipeline for the University of Michigan SMART **CNC Mill Tool Wear** dataset ([Kaggle: `shasun/tool-wear-detection-in-cnc-mill`](https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill), CC0).

Complements the separate surface-roughness study: [surface-roughness-machine-learning-model](https://github.com/nhase13-spec/surface-roughness-machine-learning-model).

## Research questions

1. Can multivariate in-process CNC sensors predict tool condition and quality proxies on **unseen experiments**?
2. Which sensors matter most?
3. How do classical models compare with temporal models (when PyTorch is available)?
4. How much does **experiment-level** validation change conclusions vs random splits?
5. Can unsupervised anomaly scores align with labels?
6. Can reduced sensor subsets retain performance?

## Dataset (verified)

| Item | Value |
|------|-------|
| Independent experiments | 18 |
| Sensor samples (total) | 25,286 |
| Sampling interval | 100 ms (10 Hz) |
| Tool condition | worn (10) / unworn (8) |
| Machining finalized | yes (14) / no (4) |
| Visual inspection | yes (10) / no (4) / missing (4 incomplete runs) |

Full audit: `results/tables/data_audit.json` and `results/tables/data_dictionary.csv`.

**Important:** Labels are categorical QA indicators — not measured surface roughness, residual stress, or fatigue life. This is **not** Rolls-Royce production data.

## Methodology

```text
raw experiment CSVs + train.csv
        │
        ▼
 validity filter + active layer segmentation
        │
        ▼
 experiment-level sensor features
        │
        ├── LOEO classical ML (primary)
        ├── random-split demo (leakage illustration)
        ├── sensor ablation
        ├── SHAP / permutation importance
        ├── anomaly detection (exploratory)
        └── 1D CNN (optional; requires PyTorch)
```

Primary validation: **Leave-One-Experiment-Out (LOEO)**.

## Installation

Python 3.11+ recommended (tested on 3.12).

```bash
cd cnc_sensor_quality_monitoring
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Dataset location (included in repo)

Raw files are at the **repository root** (not inside this subfolder):

```text
Model/
  data/raw/          ← train.csv, experiment_01…18.csv, README.txt
  cnc_sensor_quality_monitoring/   ← code + results (you run commands here)
```

No Kaggle download is required if you cloned the full repository. See [`../data/README.md`](../data/README.md).

## Reproduction

```bash
python run_pipeline.py --stage all
python scripts/generate_reports.py
pytest tests/ -q
```

Stages: `audit`, `eda`, `features`, `classical`, `temporal`, `anomaly`, `explain`, `uncertainty`, `bootstrap`, `robustness`.

## Model comparison (LOEO, actual run)

See `results/tables/model_comparison.csv`. Highlights:

| Task | Best model | Balanced accuracy |
|------|------------|-------------------|
| Tool condition | Logistic regression | 0.550 |
| Visual inspection | Random forest / process baseline | 0.775 |
| Machining finalized | Random forest | 1.000 (explained by experiment length alone, see below) |

Process-parameter-only baseline (tool condition): 0.388 balanced accuracy. Both this and the best sensor model (0.550) have bootstrap intervals that include 0.5 (chance), so **the data do not show that sensor features help for tool condition**.

Temporal CNN: **skipped** — PyTorch not installed in the reference run (`results/metrics/temporal_results.json`).

## Uncertainty and robustness (added after the first run)

Run with `python run_pipeline.py --stage bootstrap` and `--stage robustness` (original results are not overwritten).

* `results/tables/bootstrap_ci_classical.csv` - 95% bootstrap CIs (resampling experiments) for every stored OOF prediction set. With 14-18 experiments they are wide; a perfect model gets a degenerate [1, 1] interval, which is a bootstrap limitation, not certainty.
* `results/tables/robustness_summary.csv`, `reports/robustness_report.md` - the same LOEO protocol with: `n_samples` removed; duration-scaling and program-counter features also removed; a fixed a-priori 18-feature motor-load set; in-fold SelectKBest (k = 10, 20, selection refitted per fold); and a diagnostic `n_samples`-only model. The re-run of the original 864-feature set reproduces the stored results exactly.
* Findings: (1) for **machining finalised**, `n_samples` alone gives balanced accuracy 1.000 with Random Forest, identical to the full 864-feature model, so that result reflects run length, not sensor information; without it, 0.75-0.875 (RF without `n_samples`: 0.875 [0.62, 1.00]; the other variants have intervals reaching 0.5), and 0.75 once duration-scaling and program-counter features are also removed. (2) **Tool condition**: no variant is above chance - the best point estimate is 0.600 (interval includes 0.5) and several XGBoost/length-only variants fall below 0.5. (3) **Visual inspection** peaks at 0.825 with selected features but the interval is [0.50, 1.00] and the process-only baseline is 0.775. With 18 experiments none of these differences is established.
* Explainability (`feature_importance_tool_condition*.png`, `shap_summary_tool_condition*.png`, `permutation_importance_tool_condition*.csv`): Brier-scored, channel-level permutation importance, reported both as in-sample model reliance (describes the model) and as out-of-fold importance (valid for unseen experiments), plus SHAP for the fitted model. The original version scored accuracy on training data (always 0) and mis-indexed the SHAP array; both are fixed and covered by tests. SHAP/reliance describe what the model uses, not evidence that it generalises.

## Controlled extensions (added after reviewer feedback)

Run with `python run_pipeline.py --stage extended --reuse-features` (explicit stage; not part of `--stage all`; ~25 min). Original results are untouched; new files are named `extended_*` and are generated locally (not committed to the repository). Full discussion: `reports/research_addendum.md`.

* In-fold correlation filter (|r| 0.95 / 0.80), PCA after training-fold standardisation (k = 1, 2, 3, 5, 8, capped at n_train // 2), physically selected sensor groups and the a-priori 18-feature set, all on the length-free 712-feature base, evaluated with LOEO on identical folds; balanced accuracy, macro F1, pooled confusion matrices, bootstrap CIs, paired differences versus the base, jackknife stability, and out-of-fold grouped permutation importance / grouped SHAP.
* Headline: none of the filtering, PCA or sensor-group variants reliably improves on the 712-feature base (3 of 114 paired intervals exclude 0, all in the unfavourable direction); PCA made fold models less stable. Treat every comparison as hypothesis-generating (14-18 experiments).
* Caveat: spindle position features track experiment length even after the earlier filters, so `machining_completed` remains length-contaminated (`extended_spindle_position_diagnostic.csv`).
* Tests (`tests/test_extended.py`) cover leakage prevention (in-fold fitting, label-free filter, noise canary with a deliberately leaky selector, identical folds, no ids/targets/length in features, PCA cap) and reproducibility (exact reproduction of stored results, deterministic reruns, SHAP additivity).

## Key figures

- `results/figures/label_distributions.png`
- `results/figures/sensor_timeseries_experiment01.png`
- `results/figures/sensor_correlation_heatmap.png`
- `results/figures/feature_importance_tool_condition.png`
- `results/figures/shap_summary_tool_condition.png`
- `results/figures/reliability_tool_condition.png`

## Reports

- `reports/technical_research_report.md`
- `reports/portfolio_summary.md`
- `reports/limitations.md`

## Repository structure

See project brief in the EngD prompt — implemented under `src/`, `configs/config.yaml`, `notebooks/` (exploration notebooks align with pipeline stages; primary execution is `run_pipeline.py`).

## Licence and data redistribution

Dataset is **CC0**. Raw files are stored under `data/raw/` at repo root; cite the University of Michigan SMART lab / original Kaggle dataset in publications.

## References

- Sharon Sun, CNC Mill Tool Wear, University of Michigan SMART lab (Kaggle, CC0).
- Dataset README: `data/raw/README.txt`.
