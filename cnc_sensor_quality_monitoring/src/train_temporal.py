"""Lightweight temporal model (1D CNN) with grouped validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data_loader import load_train_metadata, numeric_sensor_columns
from src.evaluation import classification_metrics, get_group_splitter
from src.preprocessing import preprocess_experiment
from src.utils import load_config, project_root, save_json


def _build_sequences(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    meta = load_train_metadata(config)
    seq_len = config["models"]["temporal"]["sequence_length"]
    xs: list[np.ndarray] = []
    ys: list[int] = []
    groups: list[int] = []
    for experiment_id in sorted(meta["experiment_id"].unique()):
        from src.data_loader import load_experiment_timeseries

        raw = load_experiment_timeseries(int(experiment_id), config)
        processed = preprocess_experiment(raw, config)
        channels = numeric_sensor_columns(processed)
        matrix = processed[channels].to_numpy(dtype=float)
        matrix = np.nan_to_num(matrix, nan=0.0)
        label = int(meta.loc[meta["experiment_id"] == experiment_id, "tool_condition"].iloc[0] == "worn")
        if matrix.shape[0] < seq_len:
            pad = np.zeros((seq_len - matrix.shape[0], matrix.shape[1]))
            matrix = np.vstack([matrix, pad])
        for start in range(0, max(1, matrix.shape[0] - seq_len + 1), seq_len):
            xs.append(matrix[start : start + seq_len])
            ys.append(label)
            groups.append(int(experiment_id))
    return np.asarray(xs), np.asarray(ys), np.asarray(groups), channels


def run_temporal_experiment(config: dict | None = None) -> dict[str, Any]:
    config = config or load_config()
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        result = {
            "status": "skipped",
            "reason": "PyTorch is not installed in this environment; temporal comparison was not executed.",
            "recommendation": "Install torch and rerun `python run_pipeline.py --stage temporal`.",
        }
        save_json(result, project_root() / config["paths"]["metrics_dir"] / "temporal_results.json")
        return result

    X, y, groups, channels = _build_sequences(config)
    if len(np.unique(groups)) < 8:
        result = {"status": "skipped", "reason": "Insufficient independent experiments for stable deep learning."}
        save_json(result, project_root() / config["paths"]["metrics_dir"] / "temporal_results.json")
        return result

    device = torch.device("cpu")

    class CNN1D(nn.Module):
        def __init__(self, n_channels: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            x = x.transpose(1, 2)
            return self.net(x).squeeze(-1)

    splitter = get_group_splitter(config["validation"]["primary"], len(np.unique(groups)))
    fold_preds: list[pd.DataFrame] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        model = CNN1D(n_channels=X.shape[2]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["models"]["temporal"]["learning_rate"])
        loss_fn = nn.BCEWithLogitsLoss()
        X_train = torch.tensor(X[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config["models"]["temporal"]["batch_size"], shuffle=True)
        model.train()
        for _ in range(config["models"]["temporal"]["max_epochs"]):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X[test_idx], dtype=torch.float32).to(device)).cpu().numpy()
            prob = 1 / (1 + np.exp(-logits))
        pred = (prob >= 0.5).astype(int)
        test_groups = groups[test_idx]
        for exp_id in np.unique(test_groups):
            mask = test_groups == exp_id
            exp_prob = float(np.mean(prob[mask]))
            exp_pred = int(exp_prob >= 0.5)
            exp_true = int(y[test_idx][mask][0])
            fold_preds.append(
                pd.DataFrame(
                    [{"experiment_id": int(exp_id), "y_true": exp_true, "y_pred": exp_pred, "y_prob": exp_prob, "fold": fold_idx}]
                )
            )
    oof = pd.concat(fold_preds, ignore_index=True).drop_duplicates(subset=["experiment_id"], keep="last")
    metrics = classification_metrics(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy(), oof["y_prob"].to_numpy())
    metrics["status"] = "completed"
    metrics["n_channels"] = len(channels)
    oof.to_csv(project_root() / config["paths"]["metrics_dir"] / "oof_tool_condition_cnn1d.csv", index=False)
    save_json(metrics, project_root() / config["paths"]["metrics_dir"] / "temporal_results.json")
    return metrics
