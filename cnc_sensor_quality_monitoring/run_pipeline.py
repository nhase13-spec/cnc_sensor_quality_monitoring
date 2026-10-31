"""End-to-end research pipeline for CNC sensor analytics."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anomaly_detection import run_anomaly_detection
from src.data_loader import iter_experiments, load_all_timeseries, load_train_metadata
from src.explainability import run_explainability
from src.feature_engineering import build_experiment_feature_matrix, meta_columns
from src.preprocessing import channel_quality_report, preprocess_experiment
from src.train_classical import run_classical_experiments
from src.train_temporal import run_temporal_experiment
from src.uncertainty import run_uncertainty_analysis
from src.utils import ensure_dirs, load_config, project_root, save_json, set_seed


def run_data_audit(config: dict) -> dict:
    meta = load_train_metadata(config)
    all_ts = load_all_timeseries(config)
    lengths = all_ts.groupby("experiment_id").size()
    audit = {
        "n_experiments": int(meta["experiment_id"].nunique()),
        "n_sensor_rows_total": int(len(all_ts)),
        "n_sensor_columns": int(len(all_ts.columns)),
        "experiment_lengths": lengths.to_dict(),
        "sampling_interval_s_documented": config["data"]["sampling_interval_s"],
        "tool_condition_distribution": meta["tool_condition"].value_counts().to_dict(),
        "machining_completed_distribution": meta[config["data"]["machining_completed_col"]].value_counts().to_dict(),
        "visual_inspection_distribution": meta[config["data"]["visual_inspection_col"]].value_counts(dropna=False).to_dict(),
        "feedrate_levels": meta["feedrate"].value_counts().to_dict(),
        "clamp_pressure_levels": meta["clamp_pressure"].value_counts().to_dict(),
        "missing_values_train": meta.isna().sum().to_dict(),
        "duplicate_experiment_ids": int(meta["experiment_id"].duplicated().sum()),
        "leakage_risks": [
            "Random row-level splits would leak temporal structure within experiments.",
            "Process parameters (feedrate, clamp pressure) correlate with experimental design and must not substitute for grouped validation.",
            "Visual inspection labels are missing for incomplete runs and must be excluded from that task.",
        ],
    }
    tables_dir = project_root() / config["paths"]["tables_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    meta.to_csv(tables_dir / "experiment_metadata.csv", index=False)
    pd.DataFrame([audit]).T.rename(columns={0: "value"}).to_csv(tables_dir / "data_audit_summary.csv")
    save_json(audit, tables_dir / "data_audit.json")

    # Data dictionary from README + observed dtypes.
    dict_rows = []
    for col in meta.columns:
        dict_rows.append({"table": "train", "column": col, "dtype": str(meta[col].dtype), "role": "metadata/label"})
    sample = all_ts.head(1)
    for col in all_ts.columns:
        role = "sensor" if col not in meta_columns(config) and col not in {"sample_index", "Machining_Process"} else "metadata/process"
        dict_rows.append({"table": "experiment_xx", "column": col, "dtype": str(sample[col].dtype), "role": role})
    pd.DataFrame(dict_rows).to_csv(tables_dir / "data_dictionary.csv", index=False)
    return audit


def run_eda(config: dict) -> None:
    meta = load_train_metadata(config)
    figures_dir = project_root() / config["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Class distributions
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    meta["tool_condition"].value_counts().plot(kind="bar", ax=axes[0], title="Tool condition")
    meta[config["data"]["machining_completed_col"]].value_counts().plot(kind="bar", ax=axes[1], title="Machining finalized")
    meta[config["data"]["visual_inspection_col"]].value_counts(dropna=False).plot(kind="bar", ax=axes[2], title="Visual inspection")
    plt.tight_layout()
    plt.savefig(figures_dir / "label_distributions.png", dpi=200)
    plt.close()

    processed_rows = []
    raw_lengths = []
    for experiment_id, ts in iter_experiments(config):
        raw_lengths.append(len(ts))
        proc = preprocess_experiment(ts, config)
        processed_rows.append(proc)
    processed = pd.concat(processed_rows, ignore_index=True)
    numeric_cols = [c for c in processed.columns if c.startswith("X1_") or c.startswith("S1_Current")]
    sample_cols = numeric_cols[:6]
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in sample_cols:
        ax.plot(processed.loc[processed["experiment_id"] == 1, col].values[:400], label=col, alpha=0.8)
    ax.set_title("Experiment 1 — sample sensor channels (first 400 active/valid samples)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "sensor_timeseries_experiment01.png", dpi=200)
    plt.close()

    corr = processed[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr.iloc[:15, :15], cmap="coolwarm", center=0)
    plt.title("Sensor correlation (subset of X/S channels, processed data)")
    plt.tight_layout()
    plt.savefig(figures_dir / "sensor_correlation_heatmap.png", dpi=200)
    plt.close()

    quality = channel_quality_report(processed)
    quality.to_csv(project_root() / config["paths"]["tables_dir"] / "channel_quality.csv", index=False)
    plt.figure(figsize=(10, 5))
    plt.bar(quality["channel"].head(20), quality["missing_pct"].head(20))
    plt.xticks(rotation=90)
    plt.ylabel("Missing %")
    plt.title("Missing data by channel (processed experiments)")
    plt.tight_layout()
    plt.savefig(figures_dir / "missing_data_channels.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(raw_lengths, bins=12, edgecolor="black")
    plt.xlabel("Time-series length (raw samples per experiment)")
    plt.ylabel("Count")
    plt.title("Machining duration proxy — sample counts")
    plt.tight_layout()
    plt.savefig(figures_dir / "experiment_length_distribution.png", dpi=200)
    plt.close()


def configure_warnings() -> None:
    """LOEO on 18 experiments produces many benign sklearn metric/imputation warnings."""
    if os.environ.get("CNC_PIPELINE_VERBOSE", "").lower() in ("1", "true", "yes"):
        return
    warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\.metrics\._classification")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\.impute\._base")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\.model_selection\._split")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"scipy\.stats")


def main() -> None:
    configure_warnings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["all", "audit", "eda", "features", "classical", "temporal", "anomaly", "explain", "uncertainty", "bootstrap", "robustness", "extended"])
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Faster classical ML: smaller RF/XGB, skip ablation and random-split demo",
    )
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Load data/processed/experiment_features.csv instead of rebuilding features",
    )
    args = parser.parse_args()
    config = load_config()
    if args.fast:
        from src.train_classical import apply_fast_profile

        config = apply_fast_profile(config)
        print("Fast mode: reduced tree ensembles; ablation/random demo skipped for classical stage.", flush=True)
    ensure_dirs(config)
    set_seed(config["project"]["random_seed"])

    if args.stage in {"all", "audit"}:
        audit = run_data_audit(config)
        print("Data audit complete:", audit["n_experiments"], "experiments,", audit["n_sensor_rows_total"], "sensor rows")
    if args.stage in {"all", "eda"}:
        run_eda(config)
        print("EDA figures saved.")
    feature_df = None
    feature_cache = project_root() / config["paths"]["processed_dir"] / "experiment_features.csv"
    need_features = args.stage in {"all", "features", "classical", "temporal", "anomaly", "explain", "uncertainty", "robustness", "extended"}
    if need_features:
        if args.reuse_features and feature_cache.is_file():
            feature_df = pd.read_csv(feature_cache)
            print("Loaded cached feature matrix:", feature_df.shape, flush=True)
        else:
            feature_df = build_experiment_feature_matrix(config)
            feature_df.to_csv(feature_cache, index=False)
            print("Feature matrix:", feature_df.shape, flush=True)
    if args.stage in {"all", "classical"} and feature_df is not None:
        print("Classical ML started (many LOEO fits — typically several minutes on CPU)...", flush=True)
        results = run_classical_experiments(
            feature_df,
            config,
            skip_ablation=args.fast,
            skip_random_demo=args.fast,
        )
        print("Classical experiments complete for tasks:", list(results.keys()))
        for task, payload in results.items():
            best = max(payload["models"].items(), key=lambda kv: kv[1]["balanced_accuracy"])
            print(
                f"  {task}: best={best[0]} "
                f"LOEO balanced_accuracy={best[1]['balanced_accuracy']:.3f}"
            )
        print("Metrics saved to results\\metrics\\classical_results.json")
    if args.stage in {"all", "temporal"} and feature_df is not None:
        temporal = run_temporal_experiment(config)
        print("Temporal experiment:", temporal.get("status", "completed"))
    if args.stage in {"all", "anomaly"} and feature_df is not None:
        anomaly = run_anomaly_detection(feature_df, config)
        print("Anomaly detection methods:", list(anomaly.keys()))
    if args.stage in {"all", "explain"} and feature_df is not None:
        run_explainability(feature_df, config)
        print("Explainability artifacts saved.")
    if args.stage in {"all", "uncertainty"}:
        uncertainty = run_uncertainty_analysis(config)
        print("Uncertainty analysis:", uncertainty.get("status", "completed"))
    if args.stage in {"all", "robustness"} and feature_df is not None:
        from src.robustness import run_robustness

        print("Robustness experiments (no n_samples, reduced feature sets, in-fold selection)...", flush=True)
        run_robustness(feature_df, config)
    if args.stage == "extended" and feature_df is not None:  # explicit only: not part of "all"
        from src.extended_analysis import run_extended

        print("Extended controlled experiments (correlation filter, PCA, physical groups, grouped importance)...", flush=True)
        run_extended(feature_df, config)
    if args.stage in {"all", "bootstrap"}:
        from src.bootstrap_ci import run_bootstrap

        run_bootstrap(config)


if __name__ == "__main__":
    main()
