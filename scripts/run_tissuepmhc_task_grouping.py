#!/usr/bin/env python3
"""Run E6 task grouping experiments for tissuePMHC.

Roadmap role: E6-on-E2 performance line.
This analyzes whether the E2 shared-head model should share across all tasks,
within HLA groups, or within tissue groups.

E6 tests whether the strongest current baseline,

    shared peptide encoder + task-specific heads

benefits from selective sharing. Instead of sharing one peptide encoder across
all tasks, this script trains separate shared-head models inside either:

1. each HLA group
2. each tissue group

The grouped models are compared against the original all-task E2 results.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPEAT_SEEDS = [20260704, 20260705, 20260706]
MODEL_CHOICES = ["hla_grouped", "tissue_grouped"]

GROUP_CONTEXT_COLUMNS = ["group_type", "group_name"]
GROUP_METRIC_COLUMNS = (
    base.METRIC_COLUMNS[:3] + GROUP_CONTEXT_COLUMNS + base.METRIC_COLUMNS[3:]
)

GROUP_SUMMARY_COLUMNS = [
    "experiment_name",
    "seed",
    "model",
    "group_type",
    "group_name",
    "n_tasks",
    "train_rows",
    "test_rows",
    "mean_accuracy",
    "mean_balanced_accuracy",
    "mean_auroc",
    "mean_auprc",
    "mean_f1",
    "mean_mcc",
    "weighted_mean_accuracy",
    "weighted_mean_auroc",
    "weighted_mean_auprc",
    "worst_5_mean_auroc",
    "worst_5_mean_auprc",
]

COMPARISON_COLUMNS = [
    "seed",
    "target_tissue",
    "mhc_restriction",
    "group_type",
    "group_name",
    "baseline_model",
    "candidate_model",
    "delta_accuracy",
    "delta_balanced_accuracy",
    "delta_auroc",
    "delta_auprc",
    "delta_f1",
    "delta_mcc",
]


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def grouped_experiment_name(model_name: str) -> str:
    if model_name == "hla_grouped":
        return "E6_HLA_grouped_shared_heads"
    if model_name == "tissue_grouped":
        return "E6_tissue_grouped_shared_heads"
    raise ValueError(f"Unknown model: {model_name}")


def grouping_column(model_name: str) -> tuple[str, str]:
    if model_name == "hla_grouped":
        return "hla", "mhc_restriction"
    if model_name == "tissue_grouped":
        return "tissue", "target_tissue"
    raise ValueError(f"Unknown model: {model_name}")


def add_group_context(
    rows: list[dict[str, object]],
    experiment_name: str,
    seed: int,
    group_type: str,
    group_name: str,
) -> list[dict[str, object]]:
    for row in rows:
        row["experiment_name"] = experiment_name
        row["seed"] = seed
        row["group_type"] = group_type
        row["group_name"] = group_name
    return rows


def train_and_evaluate_group(
    args: argparse.Namespace,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    peptide_length: int,
    device: str,
    model_name: str,
    group_type: str,
    group_name: str,
) -> list[dict[str, object]]:
    group_train, group_test, group_mappings = base.add_task_columns(train_df, test_df)
    if len(group_mappings["tasks"]) < args.min_tasks_per_group:
        return []

    _, SharedTaskHeadsModel, _ = base.define_models(nn)
    x_train = base.encode_peptides(group_train["peptide_sequence"], peptide_length)
    task_train = group_train["task_id"].to_numpy(dtype=np.int64)
    y_train = group_train["label"].to_numpy(dtype=np.int64)
    loader = base.build_loader(
        torch,
        DataLoader,
        TensorDataset,
        [x_train, task_train, y_train],
        args.batch_size,
        True,
    )

    model = SharedTaskHeadsModel(
        peptide_length,
        len(group_mappings["tasks"]),
        args.embedding_dim,
        args.hidden_dim,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    base.train_binary_model(torch, model, loader, optimizer, loss_fn, device, "task_heads", args.epochs)

    rows = []
    for task_name in group_mappings["tasks"]:
        train_task = group_train[group_train["task_name"] == task_name]
        test_task = group_test[group_test["task_name"] == task_name]
        x_test = base.encode_peptides(test_task["peptide_sequence"], peptide_length)
        task_test = test_task["task_id"].to_numpy(dtype=np.int64)
        y_test = test_task["label"].to_numpy(dtype=np.int64)
        test_loader = base.build_loader(
            torch,
            DataLoader,
            TensorDataset,
            [x_test, task_test, y_test],
            args.batch_size,
            False,
        )
        y_true, y_score = base.predict_scores(torch, model, test_loader, device, "task_heads")
        metrics = base.evaluate(y_true, y_score)
        rows.append(base.make_metric_row(model_name, train_task, test_task, metrics))

    return add_group_context(
        rows,
        grouped_experiment_name(model_name),
        args.current_seed,
        group_type,
        group_name,
    )


def run_grouped_model(
    args: argparse.Namespace,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    peptide_length: int,
    device: str,
    seed: int,
    model_name: str,
) -> list[dict[str, object]]:
    base.set_seed(seed, torch)
    args.current_seed = seed
    group_type, column = grouping_column(model_name)
    group_names = sorted(set(train_df[column]) & set(test_df[column]))
    rows: list[dict[str, object]] = []
    print(f"experiment: {grouped_experiment_name(model_name)} seed={seed} groups={len(group_names)}")

    for group_index, group_name in enumerate(group_names, start=1):
        group_train = train_df[train_df[column] == group_name].copy()
        group_test = test_df[test_df[column] == group_name].copy()
        group_tasks = sorted(set(group_train["task_name"]) & set(group_test["task_name"]))
        if len(group_tasks) < args.min_tasks_per_group:
            print(f"  skip {group_type}={group_name} n_tasks={len(group_tasks)}")
            continue
        print(
            f"  {group_index:02d}/{len(group_names)} {group_type}={group_name} "
            f"n_tasks={len(group_tasks)} train_rows={len(group_train)}"
        )
        rows.extend(
            train_and_evaluate_group(
                args,
                torch,
                nn,
                DataLoader,
                TensorDataset,
                group_train,
                group_test,
                peptide_length,
                device,
                model_name,
                group_type,
                group_name,
            )
        )
    return rows


def summarize_groups(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["experiment_name"]),
            int(row["seed"]),
            str(row["model"]),
            str(row["group_type"]),
            str(row["group_name"]),
        )
        grouped[key].append(row)

    summaries = []
    for (experiment_name, seed, model, group_type, group_name), group_rows in grouped.items():
        weights = np.asarray([float(row["test_rows"]) for row in group_rows], dtype=np.float64)
        summary = {
            "experiment_name": experiment_name,
            "seed": seed,
            "model": model,
            "group_type": group_type,
            "group_name": group_name,
            "n_tasks": len(group_rows),
            "train_rows": int(sum(int(row["train_rows"]) for row in group_rows)),
            "test_rows": int(sum(int(row["test_rows"]) for row in group_rows)),
        }
        for metric in ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]:
            values = np.asarray([float(row[metric]) for row in group_rows], dtype=np.float64)
            summary[f"mean_{metric}"] = float(np.mean(values))
        for metric in ["accuracy", "auroc", "auprc"]:
            values = np.asarray([float(row[metric]) for row in group_rows], dtype=np.float64)
            summary[f"weighted_mean_{metric}"] = float(np.average(values, weights=weights))
        aurocs = np.sort(np.asarray([float(row["auroc"]) for row in group_rows], dtype=np.float64))
        auprcs = np.sort(np.asarray([float(row["auprc"]) for row in group_rows], dtype=np.float64))
        summary["worst_5_mean_auroc"] = float(np.mean(aurocs[: min(5, len(aurocs))]))
        summary["worst_5_mean_auprc"] = float(np.mean(auprcs[: min(5, len(auprcs))]))
        summaries.append(summary)

    summaries.sort(
        key=lambda row: (
            str(row["experiment_name"]),
            int(row["seed"]),
            str(row["group_type"]),
            str(row["group_name"]),
        )
    )
    return summaries


def read_e2_baseline(path: Path) -> dict[tuple[int, str, str], dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find E2 baseline metrics: {path}")
    baseline_df = pd.read_csv(path)
    if "model" in baseline_df.columns:
        baseline_df = baseline_df[baseline_df["model"] == "shared_peptide_encoder_task_heads"].copy()

    baseline_rows: dict[tuple[int, str, str], dict[str, object]] = {}
    for row in baseline_df.to_dict("records"):
        key = (int(row["seed"]), str(row["target_tissue"]), str(row["mhc_restriction"]))
        baseline_rows[key] = row
    return baseline_rows


def compare_against_e2(
    rows: list[dict[str, object]],
    baseline_rows: dict[tuple[int, str, str], dict[str, object]],
) -> list[dict[str, object]]:
    comparisons = []
    for row in rows:
        key = (int(row["seed"]), str(row["target_tissue"]), str(row["mhc_restriction"]))
        baseline_row = baseline_rows.get(key)
        if baseline_row is None:
            continue
        comparison = {
            "seed": int(row["seed"]),
            "target_tissue": row["target_tissue"],
            "mhc_restriction": row["mhc_restriction"],
            "group_type": row["group_type"],
            "group_name": row["group_name"],
            "baseline_model": "E2_all_tasks_shared_heads",
            "candidate_model": row["model"],
        }
        for metric in ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]:
            comparison[f"delta_{metric}"] = float(row[metric]) - float(baseline_row[metric])
        comparisons.append(comparison)

    comparisons.sort(
        key=lambda row: (
            int(row["seed"]),
            str(row["group_type"]),
            str(row["group_name"]),
            -float(row["delta_auroc"]),
        )
    )
    return comparisons


def run(args: argparse.Namespace) -> None:
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = base.read_dataset(args.train)
    test_df = base.read_dataset(args.test)
    train_df, test_df, mappings = base.add_task_columns(train_df, test_df)
    if args.max_tasks:
        keep_tasks = set(mappings["tasks"][: args.max_tasks])
        train_df = train_df[train_df["task_name"].isin(keep_tasks)].copy()
        test_df = test_df[test_df["task_name"].isin(keep_tasks)].copy()
        train_df, test_df, mappings = base.add_task_columns(train_df, test_df)

    peptide_length = int(max(train_df["peptide_sequence"].str.len().max(), test_df["peptide_sequence"].str.len().max()))
    result_rows: list[dict[str, object]] = []
    print(f"device: {device}")
    print(f"n_tasks: {len(mappings['tasks'])}")
    print(f"models: {args.models}")

    for seed in args.seeds:
        for model_name in args.models:
            result_rows.extend(
                run_grouped_model(
                    args,
                    torch,
                    nn,
                    DataLoader,
                    TensorDataset,
                    train_df,
                    test_df,
                    peptide_length,
                    device,
                    seed,
                    model_name,
                )
            )

    summary_rows = base.summarize_results(result_rows)
    stability_rows = base.summarize_seed_stability(summary_rows)
    group_summary_rows = summarize_groups(result_rows)
    baseline_rows = read_e2_baseline(args.baseline_per_task)
    comparison_rows = compare_against_e2(result_rows, baseline_rows)

    base.write_csv(args.per_task_output, GROUP_METRIC_COLUMNS, result_rows)
    base.write_csv(args.summary_output, base.SUMMARY_COLUMNS, summary_rows)
    base.write_csv(args.stability_output, base.STABILITY_COLUMNS, stability_rows)
    base.write_csv(args.group_summary_output, GROUP_SUMMARY_COLUMNS, group_summary_rows)
    base.write_csv(args.comparison_output, COMPARISON_COLUMNS, comparison_rows)

    metadata = {
        "train": str(args.train),
        "test": str(args.test),
        "baseline_per_task": str(args.baseline_per_task),
        "per_task_output": str(args.per_task_output),
        "summary_output": str(args.summary_output),
        "stability_output": str(args.stability_output),
        "group_summary_output": str(args.group_summary_output),
        "comparison_output": str(args.comparison_output),
        "n_tasks": len(mappings["tasks"]),
        "models": args.models,
        "seeds": args.seeds,
        "device": device,
        "peptide_length": peptide_length,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "min_tasks_per_group": args.min_tasks_per_group,
        "task_mapping": mappings["task_to_id"],
        "tissue_mapping": mappings["tissue_to_id"],
        "hla_mapping": mappings["hla_to_id"],
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote: {args.per_task_output}")
    print(f"wrote: {args.summary_output}")
    print(f"wrote: {args.stability_output}")
    print(f"wrote: {args.group_summary_output}")
    print(f"wrote: {args.comparison_output}")
    print(f"wrote: {args.metadata_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz"))
    parser.add_argument("--test", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_test.csv.gz"))
    parser.add_argument(
        "--baseline-per-task",
        type=Path,
        default=project_path("results/tissuePMHC_neural_baselines_v2/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--per-task-output",
        type=Path,
        default=project_path("results/tissuePMHC_task_grouping/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=project_path("results/tissuePMHC_task_grouping/summary_metrics.csv"),
    )
    parser.add_argument(
        "--stability-output",
        type=Path,
        default=project_path("results/tissuePMHC_task_grouping/stability_metrics.csv"),
    )
    parser.add_argument(
        "--group-summary-output",
        type=Path,
        default=project_path("results/tissuePMHC_task_grouping/group_summary_metrics.csv"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=project_path("results/tissuePMHC_task_grouping/comparison_metrics.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=project_path("results/tissuePMHC_task_grouping/metadata.json"),
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=MODEL_CHOICES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_REPEAT_SEEDS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--min-tasks-per-group",
        type=int,
        default=1,
        help="Groups with fewer tasks are skipped; keep the default 1 to cover every task.",
    )
    parser.add_argument("--max-tasks", type=int, default=0, help="Optional smoke-test limit on the number of tasks.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
