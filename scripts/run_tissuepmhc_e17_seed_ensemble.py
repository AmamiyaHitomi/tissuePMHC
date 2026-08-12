#!/usr/bin/env python3
"""Run E17 prediction averaging across independently trained E14a seeds.

The input is E14's sample-level prediction file.  For each requested ensemble
size, E17 averages probabilities within ``global_aux`` and within
``hla_plain`` across the selected seeds, then applies E15's winning task-wise
rank-average fusion.  It therefore adds no new training and preserves exact
sample alignment by sample_id/tissue/HLA before averaging.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_tissuepmhc_auxiliary_soft_ensemble as e14
import run_tissuepmhc_e15_fusion_ablation as e15
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PER_TASK_COLUMNS = [
    "experiment_name", "seed", "model", "n_member_seeds", "member_seeds",
    "target_tissue", "mhc_restriction", "test_rows", "test_positive", "test_negative",
    *e14.METRICS, "global_branch", "hla_branch", "fusion_formula",
]
PREDICTION_COLUMNS = [
    "experiment_name", "n_member_seeds", "member_seeds", "branch", "sample_id",
    "target_tissue", "mhc_restriction", "label", "mean_probability", "mean_logit",
    "probability_std_across_seeds",
]


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def require_seed_aligned_predictions(path: Path, seeds: list[int]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"E14 branch prediction file not found: {path}")
    # ``NA`` is a literal tissue identifier in the Human benchmark, not a
    # missing value.  Preserving it here prevents pandas' default NA parsing
    # from silently dropping the NA--HLA-A*02:01 task during groupby.
    df = pd.read_csv(path, keep_default_na=False)
    required = set(e14.BRANCH_PREDICTION_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"E14 prediction file is missing columns: {sorted(missing)}")
    df = df[df["branch"].isin(["global_aux", "hla_plain"]) & df["seed"].isin(seeds)].copy()
    available = sorted(int(seed) for seed in df["seed"].unique())
    absent = sorted(set(seeds) - set(available))
    if absent:
        raise ValueError(f"Requested seeds are absent from E14 predictions: {absent}")
    key = ["sample_id", "target_tissue", "mhc_restriction", "branch", "seed"]
    if df.duplicated(key, keep=False).any():
        raise ValueError("E14 prediction file has duplicate sample/branch/seed alignment keys.")
    expected = len(seeds) * 2
    counts = df.groupby(["sample_id", "target_tissue", "mhc_restriction"], sort=False).size()
    if not (counts == expected).all():
        raise ValueError("Some samples are missing a selected seed or one E14a branch.")
    label_counts = df.groupby(["sample_id", "target_tissue", "mhc_restriction"])["label"].nunique()
    if not (label_counts == 1).all():
        raise ValueError("Labels disagree across E14 seed predictions.")
    return df


def aggregate_predictions(df: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    keys = ["sample_id", "target_tissue", "mhc_restriction", "branch"]
    aggregated = df.groupby(keys, as_index=False, sort=True).agg(
        label=("label", "first"),
        mean_probability=("probability", "mean"),
        mean_logit=("logit", "mean"),
        probability_std_across_seeds=("probability", lambda values: float(np.std(values, ddof=0))),
    )
    wide = aggregated.pivot(
        index=["sample_id", "target_tissue", "mhc_restriction", "label"],
        columns="branch",
        values=["mean_probability", "mean_logit"],
    )
    wide.columns = [f"{value}_{branch}" for value, branch in wide.columns]
    result = wide.reset_index()
    if result.isna().any().any():
        raise ValueError("Averaged E17 predictions are missing a branch value.")
    return result


def build_rows(aggregated: pd.DataFrame, seeds: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    member_seeds = ",".join(str(seed) for seed in seeds)
    model = f"e17_{len(seeds)}seed_rank_average"
    for (tissue, hla), task in aggregated.groupby(["target_tissue", "mhc_restriction"], sort=True):
        score = e15.fusion_scores(task.rename(columns={
            "mean_probability_global_aux": "probability_global_aux",
            "mean_probability_hla_plain": "probability_hla_plain",
            "mean_logit_global_aux": "logit_global_aux",
            "mean_logit_hla_plain": "logit_hla_plain",
        }))["e15_task_rank_average"]
        y_true = task["label"].to_numpy(dtype=int)
        rows.append({
            "experiment_name": "E17_seed_prediction_ensemble", "seed": 0, "model": model,
            "n_member_seeds": len(seeds), "member_seeds": member_seeds,
            "target_tissue": tissue, "mhc_restriction": hla,
            "test_rows": len(task), "test_positive": int(y_true.sum()), "test_negative": int(len(task) - y_true.sum()),
            **base.evaluate(y_true, score), "global_branch": "global_aux", "hla_branch": "hla_plain",
            "fusion_formula": "task_rank_average(mean_seed_probability_global, mean_seed_probability_hla)",
        })
    return rows


def build_prediction_rows(raw: pd.DataFrame, seeds: list[int]) -> list[dict[str, object]]:
    member_seeds = ",".join(str(seed) for seed in seeds)
    output = raw.groupby(["sample_id", "target_tissue", "mhc_restriction", "branch"], as_index=False, sort=True).agg(
        label=("label", "first"), mean_probability=("probability", "mean"), mean_logit=("logit", "mean"),
        probability_std_across_seeds=("probability", lambda values: float(np.std(values, ddof=0))),
    )
    output.insert(0, "experiment_name", "E17_seed_prediction_ensemble")
    output.insert(1, "n_member_seeds", len(seeds))
    output.insert(2, "member_seeds", member_seeds)
    return output.to_dict("records")


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    all_available = pd.read_csv(args.branch_predictions, usecols=["seed"])["seed"].unique().tolist()
    selected = sorted(args.seeds) if args.seeds else sorted(int(seed) for seed in all_available)
    if not args.ensemble_sizes:
        args.ensemble_sizes = [len(selected)]
    raw = require_seed_aligned_predictions(args.branch_predictions, selected)
    rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    print(f"input: {args.branch_predictions}")
    print(f"available/selected seeds: {sorted(int(seed) for seed in all_available)} / {selected}")
    for size in args.ensemble_sizes:
        if size > len(selected):
            raise ValueError(f"Cannot form a {size}-seed ensemble from {len(selected)} selected seeds.")
        member_seeds = selected[:size]
        ensemble_started = time.perf_counter()
        subset = raw[raw["seed"].isin(member_seeds)].copy()
        aggregated = aggregate_predictions(subset, member_seeds)
        ensemble_rows = build_rows(aggregated, member_seeds)
        rows.extend(ensemble_rows)
        prediction_rows.extend(build_prediction_rows(subset, member_seeds))
        print(f"  {size}-seed mean_auroc={np.mean([row['auroc'] for row in ensemble_rows]):.4f}")
        print(f"time ensemble_size={size} duration={e14.format_duration(time.perf_counter() - ensemble_started)}")
    summary = base.summarize_results(rows)
    stability = base.summarize_seed_stability(summary)
    base.write_csv(args.per_task_output, PER_TASK_COLUMNS, rows)
    base.write_csv(args.summary_output, base.SUMMARY_COLUMNS, summary)
    base.write_csv(args.stability_output, base.STABILITY_COLUMNS, stability)
    base.write_csv(args.predictions_output, PREDICTION_COLUMNS, prediction_rows)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps({
        "experiment_name": "E17_seed_prediction_ensemble", "branch_predictions": str(args.branch_predictions),
        "selected_seeds": selected, "ensemble_sizes": args.ensemble_sizes,
        "fusion": "task_rank_average", "per_task_output": str(args.per_task_output),
        "predictions_output": str(args.predictions_output),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    for path in [args.per_task_output, args.summary_output, args.stability_output, args.predictions_output, args.metadata_output]:
        print(f"wrote: {path}")
    print(f"run total time: {e14.format_duration(time.perf_counter() - started)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-predictions", type=Path, default=project_path("results/tissuePMHC_auxiliary_soft_ensemble/branch_predictions.csv"))
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seed order used to form nested ensembles; default uses all available seeds.")
    parser.add_argument("--ensemble-sizes", nargs="+", type=int, default=None, help="For example: 3, or 3 5 after five E14 seeds are available.")
    parser.add_argument("--per-task-output", type=Path, default=project_path("results/tissuePMHC_e17_seed_ensemble/per_task_metrics.csv"))
    parser.add_argument("--summary-output", type=Path, default=project_path("results/tissuePMHC_e17_seed_ensemble/summary_metrics.csv"))
    parser.add_argument("--stability-output", type=Path, default=project_path("results/tissuePMHC_e17_seed_ensemble/stability_metrics.csv"))
    parser.add_argument("--predictions-output", type=Path, default=project_path("results/tissuePMHC_e17_seed_ensemble/branch_predictions.csv"))
    parser.add_argument("--metadata-output", type=Path, default=project_path("results/tissuePMHC_e17_seed_ensemble/metadata.json"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
