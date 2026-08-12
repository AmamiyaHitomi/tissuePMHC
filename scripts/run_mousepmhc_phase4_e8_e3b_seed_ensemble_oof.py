#!/usr/bin/env python3
"""Build Phase 4 E8 train-only OOF ensembles from frozen Phase 3 E3b predictions.

E8 performs no training and never opens the fixed test file.  It validates that
every requested E3b seed contains exactly one OOF prediction per training row,
then produces two predeclared, equal-weight candidates:

1. probability mean across independently trained seeds (the primary candidate);
2. task-wise percentile-rank mean across independently trained seeds (a fixed
   robustness ablation, not selected using the test set).

The original E3b files are read-only.  This script writes a self-contained
Phase 4 result directory with aligned ensemble predictions, per-task metrics,
summary metrics, seed-stability diagnostics, and a comparison to the mean
single-seed E3b performance.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mousePMHC_phase4_e8_e3b_seed_ensemble_oof"
SOURCE_CANDIDATE = "mousePMHC_phase3_e3b_task_balanced_mmoe_min200"
PRIMARY_CANDIDATE = "mousePMHC_phase4_e8_e3b_3seed_probability_mean"
RANK_CANDIDATE = "mousePMHC_phase4_e8_e3b_3seed_task_rank_mean"
KEYS = ["sample_id", "target_tissue", "mhc_restriction", "label"]
METRICS = ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def validate_predictions(frame: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    required = {"split", "candidate", "seed", *KEYS, "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"E8 source predictions are missing columns: {sorted(missing)}")
    if set(frame["split"].dropna().unique()) != {"oof"}:
        raise ValueError("E8 accepts OOF predictions only.")
    if set(frame["candidate"].dropna().unique()) != {SOURCE_CANDIDATE}:
        raise ValueError("E8 source candidate does not match the frozen Phase 3 E3b candidate.")
    available = sorted(int(seed) for seed in frame.seed.drop_duplicates())
    requested = sorted(set(seeds))
    unavailable = sorted(set(requested) - set(available))
    if unavailable:
        raise ValueError(f"Requested seeds are absent from E3b OOF predictions: {unavailable}; available={available}")
    selected = frame[frame.seed.isin(requested)].copy()
    if selected.score.isna().any() or not np.isfinite(selected.score.to_numpy(dtype=float)).all():
        raise ValueError("E8 source predictions contain missing or non-finite scores.")
    if ((selected.score < 0.0) | (selected.score > 1.0)).any():
        raise ValueError("E8 expects probability scores in [0, 1].")
    expected_keys: set[tuple[object, ...]] | None = None
    for seed, subset in selected.groupby("seed", sort=True):
        seed_started = time.perf_counter()
        if subset.duplicated("sample_id").any():
            raise AssertionError(f"E8 source seed {seed} has duplicate sample_id values.")
        if subset[["sample_id", "target_tissue", "mhc_restriction", "label"]].isna().any().any():
            raise ValueError(f"E8 source seed {seed} has missing alignment keys.")
        keys = set(map(tuple, subset[KEYS].itertuples(index=False, name=None)))
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise AssertionError(f"E8 source seed {seed} does not cover the same labeled samples as the other seeds.")
        print(
            f"E8 seed={int(seed)} OOF validation rows={len(subset)} "
            f"elapsed={time.perf_counter() - seed_started:.2f}s",
            flush=True,
        )
    if expected_keys is None or not expected_keys:
        raise ValueError("E8 source predictions are empty after seed selection.")
    return selected


def task_percentile_rank(frame: pd.DataFrame) -> pd.Series:
    """Return within-task percentile ranks; ties receive their average rank."""
    return frame.groupby(["seed", "target_tissue", "mhc_restriction"], sort=False)["score"].rank(
        method="average", pct=True
    )


def build_ensembles(source: pd.DataFrame, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = source.sort_values(["seed", "sample_id"], kind="stable").copy()
    ordered["task_percentile_rank"] = task_percentile_rank(ordered)
    grouped = ordered.groupby(KEYS, sort=True, as_index=False).agg(
        score_probability_mean=("score", "mean"),
        score_task_rank_mean=("task_percentile_rank", "mean"),
        probability_std_across_seeds=("score", lambda values: float(np.std(values, ddof=0))),
        n_members=("seed", "nunique"),
    )
    if not (grouped.n_members == len(seeds)).all():
        raise AssertionError("E8 ensemble rows do not contain every requested seed.")
    shared = grouped[KEYS + ["probability_std_across_seeds", "n_members"]].copy()
    probability = shared.copy()
    probability.insert(0, "split", "oof")
    probability.insert(1, "candidate", PRIMARY_CANDIDATE)
    probability["score"] = grouped.score_probability_mean
    ranks = shared.copy()
    ranks.insert(0, "split", "oof")
    ranks.insert(1, "candidate", RANK_CANDIDATE)
    ranks["score"] = grouped.score_task_rank_mean
    ensemble = pd.concat([probability, ranks], ignore_index=True)
    ensemble = ensemble[["split", "candidate", *KEYS, "score", "probability_std_across_seeds", "n_members"]]
    return ensemble, ordered


def per_task_metrics(frame: pd.DataFrame, candidate_column: str = "candidate") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate, tissue, h2), task in frame.groupby([candidate_column, "target_tissue", "mhc_restriction"], sort=True):
        rows.append({
            "experiment_name": EXPERIMENT,
            "candidate": candidate,
            "target_tissue": tissue,
            "mhc_restriction": h2,
            "oof_rows": len(task),
            **base.evaluate(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float)),
        })
    return pd.DataFrame(rows)


def summary_metrics(per_task: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate, records in per_task.groupby("candidate", sort=True):
        row: dict[str, object] = {"experiment_name": EXPERIMENT, "candidate": candidate}
        for metric in METRICS:
            row[f"mean_task_{metric}"] = float(records[metric].mean())
        row["worst_task_auroc"] = float(records.auroc.min())
        row["worst6_task_auroc"] = float(records.nsmallest(6, "auroc").auroc.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def single_seed_metrics(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for seed, predictions in source.groupby("seed", sort=True):
        task_metrics = per_task_metrics(predictions.assign(candidate=f"e3b_seed_{int(seed)}"))
        row: dict[str, object] = {"experiment_name": EXPERIMENT, "candidate": SOURCE_CANDIDATE, "seed": int(seed)}
        for metric in METRICS:
            row[f"mean_task_{metric}"] = float(task_metrics[metric].mean())
        row["worst_task_auroc"] = float(task_metrics.auroc.min())
        row["worst6_task_auroc"] = float(task_metrics.nsmallest(6, "auroc").auroc.mean())
        rows.append(row)
    seed_summary = pd.DataFrame(rows)
    stability_rows: list[dict[str, object]] = []
    for metric in [column for column in seed_summary.columns if column.startswith("mean_task_") or column.startswith("worst")]:
        stability_rows.append({
            "experiment_name": EXPERIMENT,
            "candidate": SOURCE_CANDIDATE,
            "metric": metric,
            "n_independent_seeds": len(seed_summary),
            "seed_mean": float(seed_summary[metric].mean()),
            "seed_sd": float(seed_summary[metric].std(ddof=1)),
            "seed_min": float(seed_summary[metric].min()),
            "seed_max": float(seed_summary[metric].max()),
        })
    return seed_summary, pd.DataFrame(stability_rows)


def comparison_table(summary: pd.DataFrame, seed_summary: pd.DataFrame) -> pd.DataFrame:
    means = {column: float(seed_summary[column].mean()) for column in seed_summary if column.startswith("mean_task_") or column.startswith("worst")}
    rows: list[dict[str, object]] = []
    for _, record in summary.iterrows():
        row = {"experiment_name": EXPERIMENT, "candidate": record.candidate, "baseline": "E3b single-seed metric mean"}
        for metric, value in means.items():
            row[f"baseline_{metric}"] = value
            row[metric] = float(record[metric])
            row[f"delta_{metric}"] = float(record[metric]) - value
        rows.append(row)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    total_started = time.perf_counter()
    print("E8 has no training epochs: it only validates and ensembles frozen E3b OOF predictions.", flush=True)
    source = pd.read_csv(args.source_predictions)
    seeds = sorted(set(args.seeds))
    if len(seeds) < 2:
        raise ValueError("E8 requires at least two independently trained seeds.")
    source = validate_predictions(source, seeds)
    ensemble, _ = build_ensembles(source, seeds)
    ensemble_per_task = per_task_metrics(ensemble)
    ensemble_summary = summary_metrics(ensemble_per_task)
    seed_summary, stability = single_seed_metrics(source)
    comparison = comparison_table(ensemble_summary, seed_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensemble.to_csv(args.output_dir / "mousePMHC_phase4_e8_oof_predictions.csv", index=False)
    ensemble_per_task.to_csv(args.output_dir / "mousePMHC_phase4_e8_oof_per_task_metrics.csv", index=False)
    ensemble_summary.to_csv(args.output_dir / "mousePMHC_phase4_e8_oof_summary_metrics.csv", index=False)
    seed_summary.to_csv(args.output_dir / "mousePMHC_phase4_e8_single_seed_summary_metrics.csv", index=False)
    stability.to_csv(args.output_dir / "mousePMHC_phase4_e8_oof_stability_metrics.csv", index=False)
    comparison.to_csv(args.output_dir / "mousePMHC_phase4_e8_oof_comparison.csv", index=False)
    metadata = {
        "experiment_name": EXPERIMENT,
        "source_experiment": "mousePMHC_phase3_e3b_task_balanced_mmoe_min200_oof",
        "source_candidate": SOURCE_CANDIDATE,
        "source_predictions": str(args.source_predictions),
        "test_data_read": False,
        "method": "equal-weight averaging of frozen, independently trained E3b OOF predictions",
        "selected_seeds": seeds,
        "n_members": len(seeds),
        "primary_candidate": PRIMARY_CANDIDATE,
        "primary_fusion": "probability_mean",
        "robustness_candidate": RANK_CANDIDATE,
        "robustness_fusion": "within-task percentile-rank mean",
        "alignment_keys": KEYS,
        "n_unique_oof_rows": int(ensemble.sample_id.nunique()),
        "n_tasks": int(ensemble[["target_tissue", "mhc_restriction"]].drop_duplicates().shape[0]),
        "selection_policy": "no retraining, seed selection, weight tuning, or fixed-test access",
        "outputs": {
            "predictions": "mousePMHC_phase4_e8_oof_predictions.csv",
            "per_task": "mousePMHC_phase4_e8_oof_per_task_metrics.csv",
            "summary": "mousePMHC_phase4_e8_oof_summary_metrics.csv",
            "single_seed_summary": "mousePMHC_phase4_e8_single_seed_summary_metrics.csv",
            "stability": "mousePMHC_phase4_e8_oof_stability_metrics.csv",
            "comparison": "mousePMHC_phase4_e8_oof_comparison.csv",
        },
    }
    (args.output_dir / "mousePMHC_phase4_e8_oof_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("E8 ensemble summary", flush=True)
    print(ensemble_summary.to_string(index=False), flush=True)
    print("E8 comparison with single-seed E3b mean", flush=True)
    print(comparison.to_string(index=False), flush=True)
    print(f"E8 total elapsed={time.perf_counter() - total_started:.2f}s", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-predictions",
        type=Path,
        default=project_path("results/mousePMHC_phase3_e3b_task_balanced_mmoe_min200_oof/mousePMHC_phase3_e3b_oof_predictions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_path("results/mousePMHC_phase4_e8_e3b_seed_ensemble_oof"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260704, 20260705, 20260706])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
