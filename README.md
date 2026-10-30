# Sensor-Based Tool Wear Prediction and Machining Quality Monitoring

EngD research codebase (Project 2): multivariate CNC sensor analytics with leakage-safe **leave-one-experiment-out** validation.

## Repository layout

```text
Model/
├── data/
│   └── raw/                    CNC dataset (bundled locally)
├── cnc_sensor_quality_monitoring/
│   ├── run_pipeline.py         Main entry point
│   ├── configs/config.yaml     Points to ../data/raw
│   ├── src/                    Pipeline modules
│   ├── results/                Metrics and figures
│   └── reports/                EngD markdown reports
└── README.md                   This file
```

## Quick start

```cmd
cd cnc_sensor_quality_monitoring
pip install -r requirements.txt
python run_pipeline.py --stage classical --reuse-features
python scripts\generate_reports.py
```

Dataset: see [data/README.md](data/README.md).
