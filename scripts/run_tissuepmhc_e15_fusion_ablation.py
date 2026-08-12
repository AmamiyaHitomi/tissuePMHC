#!/usr/bin/env python3
"""Run E15 fixed fusion-rule ablation on aligned E14a branch predictions.

E15 evaluates only E14a's two core branches:

* ``global_aux`` (auxiliary global branch)
* ``hla_plain`` (plain HLA-specific branch)

For every seed and tissue/HLA task, it evaluates probability averaging, logit
averaging, and within-task percentile-rank averaging.  E14 writes the input
CSV with one row per sample and branch; E15 verifies the required alignment by
``sample_id``, ``target_tissue``, and ``mhc_restriction`` before fusion.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_tissuepmhc_auxiliary_soft_ensemble as e14
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ["e15_probability_average", "e15_logit_average", "e15_task_rank_average"]
PER_TASK_COLUMNS = [
    "experiment_name", "seed", "model", "target_tissue", "mhc_restriction",
    "test_rows", "test_positive", "test_negative", *e14.METRICS,
    "global_branch", "hla_branch", "fusion_formula",
]


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def format_duration(seconds: float) -> str:
    return e14.format_duration(seconds)


def require_aligned_e14a_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"E14 branch prediction file not found: {path}. Run "
            "run_tissuepmhc_auxiliary_soft_ensemble.py first."
        )
    df = pd.read_csv(path)
    required = set(e14.BRANCH_PREDICTION_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"E14 prediction file is missing columns: {sorted(missing)}")
    df = df[df["branch"].isin(["global_aux", "hla_plain"])].copy()
    if df.empty:
        raise ValueError("E14 prediction file contains no global_aux/hla_plain rows for E15.")
    key = ["seed", "sample_id", "target_tissue", "mhc_restriction"]
    duplicate = df.duplicated([*key, "branch"], keep=False)
    if duplicate.any():
        raise ValueError("E14 prediction file has duplicate rows for an alignment key and branch.")
    wide = df.pivot(index=key, columns="branch", values=["label", "probability", "logit"])
    if wide[[('label', 'global_aux'), ('label', 'hla_plain')]].isna().any().any():
        raise ValueError("Some E14 samples are missing one of the two E15 branches.")
    if not np.array_equal(
        wide[("label", "global_aux")].to_numpy(), wide[("label", "hla_plain")].to_numpy()
    ):
        raise ValueError("Labels disagree between E14 global_aux and hla_plain predictions.")
    wide.columns = [f"{value}_{branch}" for value, branch in wide.columns]
    return wide.reset_index().sort_values(key).reset_index(drop=True)


def fusion_scores(task: pd.DataFrame) -> dict[str, np.ndarray]:
    global_probability = task["probability_global_aux"].to_numpy(dtype=float)
    hla_probability = task["probability_hla_plain"].to_numpy(dtype=float)
    global_logit = task["logit_global_aux"].to_numpy(dtype=float)
    hla_logit = task["logit_hla_plain"].to_numpy(dtype=float)
    # pct=True provides a [0, 1] score and method='average' makes tied scores
    # deterministic and symmetric between the two branches.
    global_rank = pd.Series(global_probability).rank(method="average", pct=True).to_numpy()
    hla_rank = pd.Series(hla_probability).rank(method="average", pct=True).to_numpy()
    logit_mean = 0.5 * (global_logit + hla_logit)
    return {
        "e15_probability_average": 0.5 * (global_probability + hla_probability),
        "e15_logit_average": 1.0 / (1.0 + np.exp(-logit_mean)),
        "e15_task_rank_average": 0.5 * (global_rank + hla_rank),
    }


def make_rows(predictions: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    group_columns = ["seed", "target_tissue", "mhc_restriction"]
    formulas = {
        "e15_probability_average": "0.5 * (global_probability + hla_probability)",
        "e15_logit_average": "sigmoid(0.5 * (global_logit + hla_logit))",
        "e15_task_rank_average": "0.5 * (within_task_percentile_rank(global_probability) + within_task_percentile_rank(hla_probability))",
    }
    for (seed, tissue, hla), task in predictions.groupby(group_columns, sort=True):
        y_true = task["label_global_aux"].to_numpy(dtype=int)
        for model, score in fusion_scores(task).items():
            rows.append({
                "experiment_name": "E15_fixed_fusion_ablation",
                "seed": int(seed), "model": model,
                "target_tissue": tissue, "mhc_restriction": hla,
                "test_rows": len(task),
                "test_positive": int(y_true.sum()), "test_negative": int(len(task) - y_true.sum()),
                **base.evaluate(y_true, score),
                "global_branch": "global_aux", "hla_branch": "hla_plain",
                "fusion_formula": formulas[model],
            })
    return rows


def run(args: argparse.Namespace) -> None:
    run_start = time.perf_counter()
    predictions = require_aligned_e14a_predictions(args.branch_predictions)
    rows: list[dict[str, object]] = []
    print(f"input: {args.branch_predictions}")
    print(f"aligned_samples: {len(predictions)}")
    for seed in sorted(predictions["seed"].unique()):
        seed_start = time.perf_counter()
        seed_rows = make_rows(predictions[predictions["seed"] == seed])
        rows.extend(seed_rows)
        for model in MODEL_NAMES:
            values = [float(row["auroc"]) for row in seed_rows if row["model"] == model]
            print(f"  {model} seed={seed} mean_auroc={np.mean(values):.4f}")
        print(f"time seed_total seed={seed} duration={format_duration(time.perf_counter() - seed_start)}")

    summary_rows = base.summarize_results(rows)
    stability_rows = base.summarize_seed_stability(summary_rows)
    base.write_csv(args.per_task_output, PER_TASK_COLUMNS, rows)
    base.write_csv(args.summary_output, base.SUMMARY_COLUMNS, summary_rows)
    base.write_csv(args.stability_output, base.STABILITY_COLUMNS, stability_rows)
    metadata = {
        "experiment_name": "E15_fixed_fusion_ablation",
        "branch_predictions": str(args.branch_predictions),
        "per_task_output": str(args.per_task_output),
        "summary_output": str(args.summary_output),
        "stability_output": str(args.stability_output),
        "models": MODEL_NAMES,
        "branches": ["global_aux", "hla_plain"],
        "n_aligned_samples": len(predictions),
        "n_tasks": int(predictions[["target_tissue", "mhc_restriction"]].drop_duplicates().shape[0]),
        "seeds": sorted(int(seed) for seed in predictions["seed"].unique()),
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote: {args.per_task_output}")
    print(f"wrote: {args.summary_output}")
    print(f"wrote: {args.stability_output}")
    print(f"wrote: {args.metadata_output}")
    print(f"run total time: {format_duration(time.perf_counter() - run_start)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-predictions", type=Path, default=project_path("results/tissuePMHC_auxiliary_soft_ensemble/branch_predictions.csv"))
    parser.add_argument("--per-task-output", type=Path, default=project_path("results/tissuePMHC_e15_fusion_ablation/per_task_metrics.csv"))
    parser.add_argument("--summary-output", type=Path, default=project_path("results/tissuePMHC_e15_fusion_ablation/summary_metrics.csv"))
    parser.add_argument("--stability-output", type=Path, default=project_path("results/tissuePMHC_e15_fusion_ablation/stability_metrics.csv"))
    parser.add_argument("--metadata-output", type=Path, default=project_path("results/tissuePMHC_e15_fusion_ablation/metadata.json"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
