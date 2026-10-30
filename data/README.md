# CNC Mill Tool Wear — local dataset

This folder contains the **University of Michigan SMART lab** CNC milling dataset (Kaggle mirror: *shasun/tool-wear-detection-in-cnc-mill*, **CC0**).

Files are **bundled in the repository** so you do not need to download from Kaggle.

## Layout

```text
data/raw/
  train.csv              Experiment metadata and labels (18 runs)
  experiment_01.csv … experiment_18.csv   Multivariate time series (~10 Hz)
  README.txt             Variable definitions (from the dataset authors)
  test_artifact.jpg      Example machined wax part (optional)
```

## Provenance

- Source: University of Michigan System-level Manufacturing and Automation Research Testbed (SMART)
- Acknowledgement: Sharon Sun / Kaggle dataset card
- **Not** Rolls-Royce or production aerospace data

The pipeline reads this path via `configs/config.yaml` → `paths.raw_dir: ../data/raw`.
