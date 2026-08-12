#!/usr/bin/env python3
"""Train only new E14a seeds needed to extend E17 from three to five members.

The script trains the unchanged E14a core branches (global_aux and hla_plain)
for new seeds only.  It leaves historical E14 results untouched and writes a
separate combined branch-prediction CSV that E17 can use for 3- and 5-seed
prediction averaging.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

import run_tissuepmhc_auxiliary_soft_ensemble as e14
import run_tissuepmhc_e17_seed_ensemble as e17
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEW_SEEDS = [20260707, 20260708]


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def train_e14a_seed(
    args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any,
    train_df: pd.DataFrame, test_df: pd.DataFrame, mappings: dict[str, Any], peptide_length: int,
    device: str, seed: int,
) -> tuple[dict, dict]:
    print("  train global_aux branch")
    started = time.perf_counter()
    global_model, _ = e14.train_aux_branch(
        args, torch, nn, DataLoader, TensorDataset, train_df, mappings["task_to_id"],
        len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]), peptide_length, device, seed,
        "global_aux", "all_tasks", True,
    )
    print(f"    time global_aux duration={e14.format_duration(time.perf_counter() - started)}")
    global_predictions, _ = e14.predict_branch(
        args, torch, DataLoader, TensorDataset, global_model, train_df, test_df,
        mappings["task_to_id"], peptide_length, device, seed, "global_aux", "all_tasks", True,
    )

    hla_predictions: dict = {}
    hlas = sorted(set(train_df["mhc_restriction"]) & set(test_df["mhc_restriction"]))
    for index, hla in enumerate(hlas, start=1):
        hla_train = train_df[train_df["mhc_restriction"] == hla].copy()
        hla_test = test_df[test_df["mhc_restriction"] == hla].copy()
        tasks = sorted(set(hla_train["task_name"]) & set(hla_test["task_name"]))
        if not tasks:
            continue
        task_to_id = {task: task_index for task_index, task in enumerate(tasks)}
        print(f"  train hla_plain branch {index:02d}/{len(hlas)} {hla} n_tasks={len(tasks)}")
        started = time.perf_counter()
        hla_model, _ = e14.train_aux_branch(
            args, torch, nn, DataLoader, TensorDataset, hla_train, task_to_id,
            len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]), peptide_length, device, seed,
            "hla_plain", hla, False,
        )
        print(f"    time hla_plain {hla} duration={e14.format_duration(time.perf_counter() - started)}")
        predictions, _ = e14.predict_branch(
            args, torch, DataLoader, TensorDataset, hla_model, hla_train, hla_test,
            task_to_id, peptide_length, device, seed, "hla_plain", hla, False,
        )
        hla_predictions.update(predictions)
    return global_predictions, hla_predictions


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    if not args.existing_branch_predictions.is_file():
        raise FileNotFoundError(f"Existing E14 predictions not found: {args.existing_branch_predictions}")
    existing = pd.read_csv(args.existing_branch_predictions)
    required = set(e14.BRANCH_PREDICTION_COLUMNS)
    missing = required - set(existing.columns)
    if missing:
        raise ValueError(f"Existing E14 prediction file is missing columns: {sorted(missing)}")
    existing = existing[existing["branch"].isin(["global_aux", "hla_plain"])].copy()
    existing_seeds = sorted(int(seed) for seed in existing["seed"].unique())
    overlap = sorted(set(args.new_seeds) & set(existing_seeds))
    if overlap:
        raise ValueError(f"New seeds already exist in the input file: {overlap}")

    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    train_df, test_df = base.read_dataset(args.train), base.read_dataset(args.test)
    train_df, test_df, mappings = base.add_task_columns(train_df, test_df)
    if args.max_tasks:
        keep = set(mappings["tasks"][:args.max_tasks])
        train_df, test_df = train_df[train_df.task_name.isin(keep)].copy(), test_df[test_df.task_name.isin(keep)].copy()
        train_df, test_df, mappings = base.add_task_columns(train_df, test_df)
    peptide_length = int(max(train_df.peptide_sequence.str.len().max(), test_df.peptide_sequence.str.len().max()))
    new_rows: list[dict[str, object]] = []
    print(f"device: {device}; existing_seeds: {existing_seeds}; new_seeds: {args.new_seeds}")
    for seed in args.new_seeds:
        seed_started = time.perf_counter()
        e14.set_seed(seed, torch)
        print(f"experiment: E17_incremental_E14a seed={seed}")
        global_predictions, hla_predictions = train_e14a_seed(
            args, torch, nn, DataLoader, TensorDataset, train_df, test_df, mappings, peptide_length, device, seed
        )
        new_rows.extend(e14.build_branch_prediction_rows(seed, global_predictions))
        new_rows.extend(e14.build_branch_prediction_rows(seed, hla_predictions))
        print(f"time seed_total seed={seed} duration={e14.format_duration(time.perf_counter() - seed_started)}")
    new_predictions = pd.DataFrame(new_rows, columns=e14.BRANCH_PREDICTION_COLUMNS)
    combined = pd.concat([existing, new_predictions], ignore_index=True)
    alignment_key = ["seed", "branch", "sample_id", "target_tissue", "mhc_restriction"]
    if combined.duplicated(alignment_key, keep=False).any():
        raise ValueError("Combined predictions contain duplicate seed/branch/sample alignment keys.")
    combined = combined.sort_values(alignment_key).reset_index(drop=True)
    args.new_predictions_output.parent.mkdir(parents=True, exist_ok=True)
    new_predictions.to_csv(args.new_predictions_output, index=False, encoding="utf-8")
    combined.to_csv(args.combined_predictions_output, index=False, encoding="utf-8")
    args.metadata_output.write_text(json.dumps({
        "experiment_name": "E17_incremental_E14a_seeds", "device": device,
        "existing_seeds": existing_seeds, "new_seeds": args.new_seeds,
        "combined_seeds": sorted(int(seed) for seed in combined["seed"].unique()),
        "branches": ["global_aux", "hla_plain"], "new_predictions_output": str(args.new_predictions_output),
        "combined_predictions_output": str(args.combined_predictions_output),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote: {args.new_predictions_output}")
    print(f"wrote: {args.combined_predictions_output}")
    print(f"wrote: {args.metadata_output}")
    if not args.skip_e17:
        print("run E17 3-seed and 5-seed prediction ensembles")
        e17.run(argparse.Namespace(
            branch_predictions=args.combined_predictions_output,
            seeds=None,
            ensemble_sizes=[3, 5],
            per_task_output=args.e17_per_task_output,
            summary_output=args.e17_summary_output,
            stability_output=args.e17_stability_output,
            predictions_output=args.e17_predictions_output,
            metadata_output=args.e17_metadata_output,
        ))
    print(f"run total time: {e14.format_duration(time.perf_counter() - started)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz")); parser.add_argument("--test", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_test.csv.gz"))
    parser.add_argument("--existing-branch-predictions", type=Path, default=project_path("results/tissuePMHC_auxiliary_soft_ensemble/branch_predictions.csv")); parser.add_argument("--new-seeds", nargs="+", type=int, default=DEFAULT_NEW_SEEDS); parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--batch-size", type=int, default=512); parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--embedding-dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=128); parser.add_argument("--dropout", type=float, default=0.2); parser.add_argument("--tissue-loss-weight", type=float, default=0.1); parser.add_argument("--hla-loss-weight", type=float, default=0.1); parser.add_argument("--max-grad-norm", type=float, default=1.0); parser.add_argument("--max-tasks", type=int, default=0, help="Optional smoke-test task limit.")
    parser.add_argument("--new-predictions-output", type=Path, default=project_path("results/tissuePMHC_e17_seed_ensemble/new_seed_branch_predictions.csv")); parser.add_argument("--combined-predictions-output", type=Path, default=project_path("results/tissuePMHC_e17_seed_ensemble/combined_5seed_branch_predictions.csv")); parser.add_argument("--metadata-output", type=Path, default=project_path("results/tissuePMHC_e17_seed_ensemble/incremental_seed_metadata.json"))
    parser.add_argument("--skip-e17", action="store_true", help="Only train new seeds; do not automatically run the final 3-vs-5 seed E17 comparison.")
    parser.add_argument("--e17-per-task-output", type=Path, default=project_path("results/tissuePMHC_e17_seed_ensemble/per_task_metrics.csv")); parser.add_argument("--e17-summary-output", type=Path, default=project_path("results/tissuePMHC_e17_seed_ensemble/summary_metrics.csv")); parser.add_argument("--e17-stability-output", type=Path, default=project_path("results/tissuePMHC_e17_seed_ensemble/stability_metrics.csv")); parser.add_argument("--e17-predictions-output", type=Path, default=project_path("results/tissuePMHC_e17_seed_ensemble/branch_predictions.csv")); parser.add_argument("--e17-metadata-output", type=Path, default=project_path("results/tissuePMHC_e17_seed_ensemble/metadata.json"))
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())
