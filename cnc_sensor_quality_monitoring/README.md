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

Stages: `audit`, `eda`, `features`, `classical`, `temporal`, `anomaly`, `explain`, `uncertainty`.

## Model comparison (LOEO, actual run)

See `results/tables/model_comparison.csv`. Highlights:

| Task | Best model | Balanced accuracy |
|------|------------|-------------------|
| Tool condition | Logistic regression | 0.550 |
| Visual inspection | Random forest / process baseline | 0.775 |
| Machining finalized | Random forest | 0.875 |

Process-parameter-only baseline (tool condition): **0.388** balanced accuracy → sensor features help modestly under LOEO.

Temporal CNN: **skipped** — PyTorch not installed in the reference run (`results/metrics/temporal_results.json`).

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
