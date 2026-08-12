#!/usr/bin/env python3
"""Run E7 validation-based selective sharing for tissuePMHC.

Roadmap role: E7-on-E2 performance line.
This extends E2/E6 by choosing either the global shared-head branch or the
HLA-grouped branch per task using validation performance.

E7 keeps two branches:

1. global E2 branch: one shared peptide encoder across all tasks
2. HLA-grouped branch: one shared peptide encoder within each HLA group

For each task, branch selection is based only on a validation split carved from
the training set. The selected branch is then evaluated on the held-out test
set. This avoids choosing branches directly from test performance.
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
DEFAULT_REPEAT_SEEDS = [20260704, 20260705, 20260706]
METRICS = ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]

CANDIDATE_COLUMNS = [
    "experiment_name",
    "seed",
    "split",
    "branch",
    "group_name",
    "target_tissue",
    "mhc_restriction",
    "train_rows",
    "eval_rows",
    "train_positive",
    "train_negative",
    "eval_positive",
    "eval_negative",
    *METRICS,
]

SELECTION_COLUMNS = [
    "experiment_name",
    "seed",
    "target_tissue",
    "mhc_restriction",
    "selected_branch",
    "selected_group_name",
    "selection_metric",
    "global_validation_metric",
    "hla_validation_metric",
    "validation_delta_hla_minus_global",
]

SELECTED_COLUMNS = [
    *base.METRIC_COLUMNS,
    "selected_branch",
    "selected_group_name",
    "selection_metric",
    "global_validation_metric",
    "hla_validation_metric",
    "validation_delta_hla_minus_global",
]

COMPARISON_COLUMNS = [
    "seed",
    "target_tissue",
    "mhc_restriction",
    "baseline_model",
    "candidate_model",
    "selected_branch",
    "selected_group_name",
    "delta_accuracy",
    "delta_balanced_accuracy",
    "delta_auroc",
    "delta_auprc",
    "delta_f1",
    "delta_mcc",
]


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def split_train_validation(
    train_df: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []

    for (_, _), task_rows in train_df.groupby(["task_name", "label"], sort=True):
        indices = task_rows.index.to_numpy().copy()
        rng.shuffle(indices)
        n_validation = int(round(len(indices) * validation_fraction))
        n_validation = max(1, min(n_validation, len(indices) - 1))
        validation_indices.extend(indices[:n_validation])
        train_indices.extend(indices[n_validation:])

    train_core = train_df.loc[train_indices].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    validation_df = train_df.loc[validation_indices].sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    return train_core, validation_df


def prepare_with_mapping(df: pd.DataFrame, task_to_id: dict[str, int]) -> pd.DataFrame:
    prepared = df[df["task_name"].isin(task_to_id)].copy()
    prepared["task_id"] = prepared["task_name"].map(task_to_id).astype(np.int64)
    return prepared


def train_shared_heads_model(
    args: argparse.Namespace,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    train_df: pd.DataFrame,
    task_to_id: dict[str, int],
    peptide_length: int,
    device: str,
) -> Any:
    _, SharedTaskHeadsModel, _ = base.define_models(nn)
    train_mapped = prepare_with_mapping(train_df, task_to_id)
    x_train = base.encode_peptides(train_mapped["peptide_sequence"], peptide_length)
    task_train = train_mapped["task_id"].to_numpy(dtype=np.int64).copy()
    y_train = train_mapped["label"].to_numpy(dtype=np.int64).copy()
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
        len(task_to_id),
        args.embedding_dim,
        args.hidden_dim,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    base.train_binary_model(torch, model, loader, optimizer, loss_fn, device, "task_heads", args.epochs)
    return model


def count_labels(rows: pd.DataFrame) -> tuple[int, int]:
    positive = int((rows["label"] == 1).sum())
    return positive, int(len(rows) - positive)


def make_candidate_row(
    experiment_name: str,
    seed: int,
    split: str,
    branch: str,
    group_name: str,
    train_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    metrics: dict[str, float],
) -> dict[str, object]:
    train_positive, train_negative = count_labels(train_rows)
    eval_positive, eval_negative = count_labels(eval_rows)
    first = eval_rows.iloc[0]
    return {
        "experiment_name": experiment_name,
        "seed": seed,
        "split": split,
        "branch": branch,
        "group_name": group_name,
        "target_tissue": first["target_tissue"],
        "mhc_restriction": first["mhc_restriction"],
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_positive": train_positive,
        "train_negative": train_negative,
        "eval_positive": eval_positive,
        "eval_negative": eval_negative,
        **metrics,
    }


def evaluate_branch(
    args: argparse.Namespace,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    model: Any,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    task_to_id: dict[str, int],
    peptide_length: int,
    device: str,
    split: str,
    branch: str,
    group_name: str,
    seed: int,
) -> list[dict[str, object]]:
    eval_mapped = prepare_with_mapping(eval_df, task_to_id)
    train_mapped = prepare_with_mapping(train_df, task_to_id)
    rows = []
    for task_name in sorted(set(eval_mapped["task_name"]) & set(train_mapped["task_name"])):
        train_task = train_mapped[train_mapped["task_name"] == task_name]
        eval_task = eval_mapped[eval_mapped["task_name"] == task_name]
        x_eval = base.encode_peptides(eval_task["peptide_sequence"], peptide_length)
        task_eval = eval_task["task_id"].to_numpy(dtype=np.int64)
        y_eval = eval_task["label"].to_numpy(dtype=np.int64)
        loader = base.build_loader(
            torch,
            DataLoader,
            TensorDataset,
            [x_eval, task_eval, y_eval],
            args.batch_size,
            False,
        )
        y_true, y_score = base.predict_scores(torch, model, loader, device, "task_heads")
        metrics = base.evaluate(y_true, y_score)
        rows.append(
            make_candidate_row(
                "E7_selective_hla_or_global",
                seed,
                split,
                branch,
                group_name,
                train_task,
                eval_task,
                metrics,
            )
        )
    return rows


def task_key(row: dict[str, object]) -> tuple[str, str]:
    return str(row["target_tissue"]), str(row["mhc_restriction"])


def train_and_evaluate_candidates(
    args: argparse.Namespace,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    full_train_df: pd.DataFrame,
    train_core: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task_to_id: dict[str, int],
    peptide_length: int,
    device: str,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidate_rows: list[dict[str, object]] = []
    print("  train validation global branch")
    step_start = time.perf_counter()
    validation_global_model = train_shared_heads_model(
        args, torch, nn, DataLoader, TensorDataset, train_core, task_to_id, peptide_length, device
    )
    print(
        f"    time train_validation_global_branch all_tasks "
        f"n_tasks={len(task_to_id)} train_rows={len(train_core)} "
        f"duration={format_duration(time.perf_counter() - step_start)}"
    )
    candidate_rows.extend(
        evaluate_branch(
            args,
            torch,
            DataLoader,
            TensorDataset,
            validation_global_model,
            train_core,
            validation_df,
            task_to_id,
            peptide_length,
            device,
            "validation",
            "global",
            "all_tasks",
            seed,
        )
    )

    hla_groups = sorted(set(train_core["mhc_restriction"]) & set(validation_df["mhc_restriction"]))
    for group_index, hla in enumerate(hla_groups, start=1):
        hla_train = train_core[train_core["mhc_restriction"] == hla].copy()
        hla_validation = validation_df[validation_df["mhc_restriction"] == hla].copy()
        local_tasks = sorted(set(hla_train["task_name"]) & set(hla_validation["task_name"]))
        if not local_tasks:
            continue
        local_task_to_id = {task: index for index, task in enumerate(local_tasks)}
        print(f"  train validation HLA branch {group_index:02d}/{len(hla_groups)} {hla} n_tasks={len(local_tasks)}")
        step_start = time.perf_counter()
        validation_hla_model = train_shared_heads_model(
            args,
            torch,
            nn,
            DataLoader,
            TensorDataset,
            hla_train,
            local_task_to_id,
            peptide_length,
            device,
        )
        print(
            f"    time train_validation_hla_branch {hla} "
            f"n_tasks={len(local_task_to_id)} train_rows={len(hla_train)} "
            f"duration={format_duration(time.perf_counter() - step_start)}"
        )
        candidate_rows.extend(
            evaluate_branch(
                args,
                torch,
                DataLoader,
                TensorDataset,
                validation_hla_model,
                hla_train,
                hla_validation,
                local_task_to_id,
                peptide_length,
                device,
                "validation",
                "hla_grouped",
                hla,
                seed,
            )
        )

    print("  train final global branch")
    step_start = time.perf_counter()
    final_global_model = train_shared_heads_model(
        args, torch, nn, DataLoader, TensorDataset, full_train_df, task_to_id, peptide_length, device
    )
    print(
        f"    time train_final_global_branch all_tasks "
        f"n_tasks={len(task_to_id)} train_rows={len(full_train_df)} "
        f"duration={format_duration(time.perf_counter() - step_start)}"
    )
    candidate_rows.extend(
        evaluate_branch(
            args,
            torch,
            DataLoader,
            TensorDataset,
            final_global_model,
            full_train_df,
            test_df,
            task_to_id,
            peptide_length,
            device,
            "test",
            "global",
            "all_tasks",
            seed,
        )
    )

    final_hla_groups = sorted(set(full_train_df["mhc_restriction"]) & set(test_df["mhc_restriction"]))
    for group_index, hla in enumerate(final_hla_groups, start=1):
        hla_train = full_train_df[full_train_df["mhc_restriction"] == hla].copy()
        hla_test = test_df[test_df["mhc_restriction"] == hla].copy()
        local_tasks = sorted(set(hla_train["task_name"]) & set(hla_test["task_name"]))
        if not local_tasks:
            continue
        local_task_to_id = {task: index for index, task in enumerate(local_tasks)}
        print(f"  train final HLA branch {group_index:02d}/{len(final_hla_groups)} {hla} n_tasks={len(local_tasks)}")
        step_start = time.perf_counter()
        final_hla_model = train_shared_heads_model(
            args,
            torch,
            nn,
            DataLoader,
            TensorDataset,
            hla_train,
            local_task_to_id,
            peptide_length,
            device,
        )
        print(
            f"    time train_final_hla_branch {hla} "
            f"n_tasks={len(local_task_to_id)} train_rows={len(hla_train)} "
            f"duration={format_duration(time.perf_counter() - step_start)}"
        )
        candidate_rows.extend(
            evaluate_branch(
                args,
                torch,
                DataLoader,
                TensorDataset,
                final_hla_model,
                hla_train,
                hla_test,
                local_task_to_id,
                peptide_length,
                device,
                "test",
                "hla_grouped",
                hla,
                seed,
            )
        )

    return candidate_rows, select_test_rows(candidate_rows, args.selection_metric)


def format_duration(seconds: float) -> str:
    minutes, remaining_seconds = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h {minutes:02d}m {remaining_seconds:05.2f}s"
    if minutes:
        return f"{minutes}m {remaining_seconds:05.2f}s"
    return f"{remaining_seconds:.2f}s"


def select_test_rows(
    candidate_rows: list[dict[str, object]],
    selection_metric: str,
) -> list[dict[str, object]]:
    validation_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    test_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in candidate_rows:
        key = (*task_key(row), str(row["branch"]))
        if row["split"] == "validation":
            validation_by_key[key] = row
        elif row["split"] == "test":
            test_by_key[key] = row

    selected_rows = []
    task_keys = sorted({key[:2] for key in validation_by_key})
    for tissue, hla in task_keys:
        global_validation = validation_by_key.get((tissue, hla, "global"))
        hla_validation = validation_by_key.get((tissue, hla, "hla_grouped"))
        if global_validation is None:
            continue
        if hla_validation is not None and float(hla_validation[selection_metric]) > float(global_validation[selection_metric]):
            selected_branch = "hla_grouped"
            selected_validation = hla_validation
        else:
            selected_branch = "global"
            selected_validation = global_validation

        selected_test = test_by_key.get((tissue, hla, selected_branch))
        if selected_test is None:
            continue
        global_metric = float(global_validation[selection_metric])
        hla_metric = float(hla_validation[selection_metric]) if hla_validation is not None else np.nan
        validation_delta = hla_metric - global_metric if hla_validation is not None else np.nan

        metric_row = {
            "experiment_name": "E7_selective_hla_or_global",
            "seed": int(selected_test["seed"]),
            "model": "e7_selective_hla_or_global",
            "target_tissue": selected_test["target_tissue"],
            "mhc_restriction": selected_test["mhc_restriction"],
            "train_rows": selected_test["train_rows"],
            "test_rows": selected_test["eval_rows"],
            "train_positive": selected_test["train_positive"],
            "train_negative": selected_test["train_negative"],
            "test_positive": selected_test["eval_positive"],
            "test_negative": selected_test["eval_negative"],
            "selected_branch": selected_branch,
            "selected_group_name": selected_test["group_name"],
            "selection_metric": selection_metric,
            "global_validation_metric": global_metric,
            "hla_validation_metric": hla_metric,
            "validation_delta_hla_minus_global": validation_delta,
        }
        for metric in METRICS:
            metric_row[metric] = selected_test[metric]
        selected_rows.append(metric_row)

    selected_rows.sort(key=lambda row: (int(row["seed"]), str(row["target_tissue"]), str(row["mhc_restriction"])))
    return selected_rows


def make_selection_rows(selected_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in selected_rows:
        rows.append(
            {
                "experiment_name": row["experiment_name"],
                "seed": row["seed"],
                "target_tissue": row["target_tissue"],
                "mhc_restriction": row["mhc_restriction"],
                "selected_branch": row["selected_branch"],
                "selected_group_name": row["selected_group_name"],
                "selection_metric": row["selection_metric"],
                "global_validation_metric": row["global_validation_metric"],
                "hla_validation_metric": row["hla_validation_metric"],
                "validation_delta_hla_minus_global": row["validation_delta_hla_minus_global"],
            }
        )
    return rows


def read_e2_baseline(path: Path) -> dict[tuple[int, str, str], dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find E2 baseline metrics: {path}")
    baseline_df = pd.read_csv(path)
    if "model" in baseline_df.columns:
        baseline_df = baseline_df[baseline_df["model"] == "shared_peptide_encoder_task_heads"].copy()
    return {
        (int(row["seed"]), str(row["target_tissue"]), str(row["mhc_restriction"])): row
        for row in baseline_df.to_dict("records")
    }


def compare_against_e2(
    selected_rows: list[dict[str, object]],
    baseline_rows: dict[tuple[int, str, str], dict[str, object]],
) -> list[dict[str, object]]:
    comparisons = []
    for row in selected_rows:
        key = (int(row["seed"]), str(row["target_tissue"]), str(row["mhc_restriction"]))
        baseline_row = baseline_rows.get(key)
        if baseline_row is None:
            continue
        comparison = {
            "seed": int(row["seed"]),
            "target_tissue": row["target_tissue"],
            "mhc_restriction": row["mhc_restriction"],
            "baseline_model": "E2_all_tasks_shared_heads",
            "candidate_model": "e7_selective_hla_or_global",
            "selected_branch": row["selected_branch"],
            "selected_group_name": row["selected_group_name"],
        }
        for metric in METRICS:
            comparison[f"delta_{metric}"] = float(row[metric]) - float(baseline_row[metric])
        comparisons.append(comparison)
    comparisons.sort(key=lambda row: (int(row["seed"]), -float(row["delta_auroc"])))
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
    all_candidate_rows: list[dict[str, object]] = []
    all_selected_rows: list[dict[str, object]] = []
    run_start = time.perf_counter()

    print(f"device: {device}")
    print(f"n_tasks: {len(mappings['tasks'])}")
    print(f"selection_metric: {args.selection_metric}")
    for seed in args.seeds:
        seed_start = time.perf_counter()
        base.set_seed(seed, torch)
        print(f"experiment: E7_selective_hla_or_global seed={seed}")
        train_core, validation_df = split_train_validation(train_df, args.validation_fraction, seed)
        print(f"  train_core_rows={len(train_core)} validation_rows={len(validation_df)}")
        candidate_rows, selected_rows = train_and_evaluate_candidates(
            args,
            torch,
            nn,
            DataLoader,
            TensorDataset,
            train_df,
            train_core,
            validation_df,
            test_df,
            mappings["task_to_id"],
            peptide_length,
            device,
            seed,
        )
        all_candidate_rows.extend(candidate_rows)
        all_selected_rows.extend(selected_rows)
        n_hla = sum(1 for row in selected_rows if row["selected_branch"] == "hla_grouped")
        print(f"  selected hla_grouped tasks={n_hla}/{len(selected_rows)}")
        print(f"  time seed_total seed={seed} duration={format_duration(time.perf_counter() - seed_start)}")

    summary_rows = base.summarize_results(all_selected_rows)
    stability_rows = base.summarize_seed_stability(summary_rows)
    selection_rows = make_selection_rows(all_selected_rows)
    baseline_rows = read_e2_baseline(args.baseline_per_task)
    comparison_rows = compare_against_e2(all_selected_rows, baseline_rows)

    base.write_csv(args.per_task_output, SELECTED_COLUMNS, all_selected_rows)
    base.write_csv(args.summary_output, base.SUMMARY_COLUMNS, summary_rows)
    base.write_csv(args.stability_output, base.STABILITY_COLUMNS, stability_rows)
    base.write_csv(args.candidate_output, CANDIDATE_COLUMNS, all_candidate_rows)
    base.write_csv(args.selection_output, SELECTION_COLUMNS, selection_rows)
    base.write_csv(args.comparison_output, COMPARISON_COLUMNS, comparison_rows)
    print(f"time run_total duration={format_duration(time.perf_counter() - run_start)}")

    metadata = {
        "train": str(args.train),
        "test": str(args.test),
        "baseline_per_task": str(args.baseline_per_task),
        "per_task_output": str(args.per_task_output),
        "summary_output": str(args.summary_output),
        "stability_output": str(args.stability_output),
        "candidate_output": str(args.candidate_output),
        "selection_output": str(args.selection_output),
        "comparison_output": str(args.comparison_output),
        "n_tasks": len(mappings["tasks"]),
        "seeds": args.seeds,
        "device": device,
        "peptide_length": peptide_length,
        "selection_metric": args.selection_metric,
        "validation_fraction": args.validation_fraction,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "task_mapping": mappings["task_to_id"],
        "tissue_mapping": mappings["tissue_to_id"],
        "hla_mapping": mappings["hla_to_id"],
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote: {args.per_task_output}")
    print(f"wrote: {args.summary_output}")
    print(f"wrote: {args.stability_output}")
    print(f"wrote: {args.candidate_output}")
    print(f"wrote: {args.selection_output}")
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
        default=project_path("results/tissuePMHC_selective_grouping/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=project_path("results/tissuePMHC_selective_grouping/summary_metrics.csv"),
    )
    parser.add_argument(
        "--stability-output",
        type=Path,
        default=project_path("results/tissuePMHC_selective_grouping/stability_metrics.csv"),
    )
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=project_path("results/tissuePMHC_selective_grouping/candidate_metrics.csv"),
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=project_path("results/tissuePMHC_selective_grouping/selection_metrics.csv"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=project_path("results/tissuePMHC_selective_grouping/comparison_metrics.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=project_path("results/tissuePMHC_selective_grouping/metadata.json"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_REPEAT_SEEDS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--selection-metric", choices=["auroc", "auprc", "mcc"], default="auroc")
    parser.add_argument("--max-tasks", type=int, default=0, help="Optional smoke-test limit on the number of tasks.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
