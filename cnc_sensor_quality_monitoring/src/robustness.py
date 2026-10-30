"""Robustness experiments for Project 2: do the results survive removal of likely shortcuts?

Variants (all evaluated with the same leave-one-experiment-out protocol as the main run):

* ``all_features_reference``  - the original 864 features (must reproduce the stored results).
* ``no_n_samples``            - drops ``n_samples`` (experiment length). Length is a plausible shortcut for
                                "machining finalised" because an aborted run is shorter.
* ``no_extensive_or_progress``- additionally drops duration-scaling features (energy / spectral energy /
                                wavelet energy) and program-counter channels, which also encode run length
                                or how far the program got.
* ``apriori_reduced``         - small fixed feature set (a handful of motor-load channels x mean/std/IQR),
                                defined in config.yaml before looking at these results.
* ``kbest10`` / ``kbest20``   - in-fold SelectKBest (F-test) on ``no_n_samples``; selection is refitted in
                                every training fold, so it cannot leak the held-out experiment.
* ``n_samples_only``          - DIAGNOSTIC: only experiment length. If this alone matches the full model on a
                                task, the full model's skill on that task is explained by length.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.bootstrap_ci import ci_rows_from_oof
from src.train_classical import grouped_cv_train, prepare_task_frame
from src.utils import load_config, project_root, save_json

REFERENCE_TOLERANCE = 1e-9


def build_variants(all_feats: list[str], config: dict) -> dict[str, dict[str, Any]]:
    """Return {variant: {"cols": [...], "select_k": int | None, "note": str}}."""
    rc = config["robustness"]
    no_n = [c for c in all_feats if c != "n_samples"]

    extensive = tuple(rc["extensive_suffixes"])
    progress = tuple(rc["progress_prefixes"])
    strict = [c for c in no_n if not c.endswith(extensive) and not c.startswith(progress)]

    apriori = [f"{ch}__{st}" for ch in rc["apriori_channels"] for st in rc["apriori_stats"]]
    missing = [c for c in apriori if c not in all_feats]
    if missing:
        # Fail loudly rather than silently evaluating a different set than the one documented.
        raise KeyError(f"A-priori features missing from the feature matrix: {missing}")

    variants: dict[str, dict[str, Any]] = {
        "all_features_reference": {"cols": list(all_feats), "select_k": None, "note": "original feature set"},
        "no_n_samples": {"cols": no_n, "select_k": None, "note": "drops experiment length"},
        "no_extensive_or_progress": {
            "cols": strict,
            "select_k": None,
            "note": "drops length, duration-scaling energies, program-counter channels",
        },
        "apriori_reduced": {"cols": apriori, "select_k": None, "note": "fixed a-priori motor-load features"},
    }
    for k in rc["kbest"]:
        variants[f"kbest{k}"] = {"cols": no_n, "select_k": int(k), "note": f"in-fold SelectKBest k={k}"}
    variants["n_samples_only"] = {"cols": ["n_samples"], "select_k": None, "note": "diagnostic: length only"}
    return variants


def run_robustness(feature_df: pd.DataFrame, config: dict | None = None) -> dict[str, Any]:
    config = config or load_config()
    rc = config["robustness"]
    b = config["bootstrap"]
    root = project_root()
    metrics_dir = root / config["paths"]["metrics_dir"]
    tables_dir = root / config["paths"]["tables_dir"]
    reports_dir = root / "reports"
    for d in (metrics_dir, tables_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    stored_path = metrics_dir / "classical_results.json"
    stored = None
    if stored_path.exists():
        import json

        stored = json.loads(stored_path.read_text(encoding="utf-8"))

    results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []
    reproduction: list[dict[str, Any]] = []

    for task_name, task_cfg in config["tasks"].items():
        if not task_cfg.get("enabled"):
            continue
        frame, y, groups, all_feats = prepare_task_frame(feature_df, task_name, config)
        variants = build_variants(all_feats, config)
        results[task_name] = {"n_experiments": int(len(frame)), "class_balance": y.value_counts().to_dict()}
        for variant_name, spec in variants.items():
            results[task_name][variant_name] = {"n_features": len(spec["cols"]), "note": spec["note"], "models": {}}
            for model_name in rc["models"]:
                if model_name == "dummy" and variant_name != "all_features_reference":
                    continue  # dummy ignores features; one run is enough
                label = f"{task_name}/{variant_name}/{model_name}"
                print(f"  {label} ({len(spec['cols'])} features)", flush=True)
                # Single-feature variant: wrap into DataFrame so that the pipeline keeps column names.
                summary, oof = grouped_cv_train(
                    frame[spec["cols"]], y, groups, model_name, config, select_k=spec["select_k"]
                )
                fold_free = {k: v for k, v in summary.items() if k != "fold_metrics"}
                results[task_name][variant_name]["models"][model_name] = fold_free

                oof = oof.assign(task=task_name, variant=variant_name)
                oof_frames.append(oof)
                row = ci_rows_from_oof(
                    oof,
                    n_boot=b["n_boot"],
                    seed=b["seed"],
                    alpha=b["alpha"],
                    task=task_name,
                    variant=variant_name,
                    model=model_name,
                    n_features=len(spec["cols"]) if spec["select_k"] is None else spec["select_k"],
                )
                summary_rows.append(row)

                if variant_name == "all_features_reference" and stored is not None:
                    old = stored[task_name]["models"][model_name]["balanced_accuracy"]
                    new = fold_free["balanced_accuracy"]
                    reproduction.append(
                        {
                            "task": task_name,
                            "model": model_name,
                            "stored_balanced_accuracy": old,
                            "rerun_balanced_accuracy": new,
                            "abs_diff": abs(old - new),
                        }
                    )

    max_diff = max((r["abs_diff"] for r in reproduction), default=None)
    results["_reproduction_check"] = {
        "rows": reproduction,
        "max_abs_diff": max_diff,
        "reproduces_original": bool(max_diff is not None and max_diff <= REFERENCE_TOLERANCE),
    }
    results["_config"] = {"bootstrap": b, "robustness": rc}

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables_dir / "robustness_summary.csv", index=False)
    pd.concat(oof_frames, ignore_index=True).to_csv(metrics_dir / "robustness_oof.csv", index=False)
    save_json(results, metrics_dir / "robustness_results.json")
    _write_report(summary, results, reports_dir / "robustness_report.md")
    print(
        f"Robustness done. Reproduction check max |diff| = {max_diff} "
        f"(reproduces original: {results['_reproduction_check']['reproduces_original']}).",
        flush=True,
    )
    return results


def _fmt(row: pd.Series, metric: str = "balanced_accuracy") -> str:
    return f"{row[metric]:.3f} [{row[metric + '_lo']:.2f}, {row[metric + '_hi']:.2f}]"


def _write_report(summary: pd.DataFrame, results: dict[str, Any], path) -> None:
    """Data-driven tables only; interpretation is kept conservative and stated in the README/chat."""
    lines = [
        "# Project 2 robustness and uncertainty report",
        "",
        "Generated by `python run_pipeline.py --stage robustness`. Protocol: leave-one-experiment-out; "
        "values are balanced accuracy with 95% percentile-bootstrap interval over experiments "
        f"(B={results['_config']['bootstrap']['n_boot']}).",
        "",
        "**Reading guide.** With 14-18 experiments the intervals are wide. A point estimate whose interval "
        "contains 0.5 is not distinguishable from chance. Perfect predictions give a degenerate interval "
        "[1.00, 1.00] because every bootstrap resample is also perfect; that does not mean certainty.",
        "",
        f"Reproduction check (original 864-feature results re-computed): "
        f"max |difference| = {results['_reproduction_check']['max_abs_diff']}.",
        "",
    ]
    for task, g in summary.groupby("task", sort=False):
        lines += [f"## {task}", "", "| Variant | #features | " + " | ".join(sorted(g['model'].unique())) + " |"]
        models = sorted(g["model"].unique())
        lines.append("|---|---|" + "|".join(["---"] * len(models)) + "|")
        for variant, gv in g.groupby("variant", sort=False):
            cells = []
            for m in models:
                r = gv[gv["model"] == m]
                cells.append(_fmt(r.iloc[0]) if len(r) else "-")
            lines.append(f"| {variant} | {int(gv['n_features'].iloc[0])} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
