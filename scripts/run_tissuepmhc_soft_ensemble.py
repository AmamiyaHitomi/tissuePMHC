#!/usr/bin/env python3
"""Run E8 soft ensemble of global and HLA-grouped branches for tissuePMHC.

Roadmap role: E8-on-E2/E6 performance main line.
This is the current strongest standard-split model: it blends the E2 global
branch with the HLA-grouped branch instead of hard-selecting one branch.

E8 keeps the leakage-safe two-stage design from E7:

1. Train validation branches on train-core.
2. Use validation metrics to compute per-task ensemble weights.
3. Retrain final branches on the full training set.
4. Evaluate weighted test scores.

Unlike E7, E8 does not hard-select one branch. It blends scores:

    final_score = w_hla * hla_score + (1 - w_hla) * global_score
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
import run_tissuepmhc_selective_grouping as e7


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPEAT_SEEDS = [20260704, 20260705, 20260706]
METRICS = ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]

MODEL_NAMES = [
    "e8a_fixed_average",
    "e8b_validation_delta_clipped",
    "e8c_validation_softmax",
]

ENSEMBLE_COLUMNS = [
    *base.METRIC_COLUMNS,
    "ensemble_strategy",
    "selection_metric",
    "hla_weight",
    "global_weight",
    "global_validation_metric",
    "hla_validation_metric",
    "validation_delta_hla_minus_global",
]

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

WEIGHT_COLUMNS = [
    "experiment_name",
    "seed",
    "target_tissue",
    "mhc_restriction",
    "ensemble_strategy",
    "selection_metric",
    "hla_weight",
    "global_weight",
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
    "ensemble_strategy",
    "hla_weight",
    "delta_accuracy",
    "delta_balanced_accuracy",
    "delta_auroc",
    "delta_auprc",
    "delta_f1",
    "delta_mcc",
]


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def format_duration(seconds: float) -> str:
    minutes, remaining_seconds = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h {minutes:02d}m {remaining_seconds:05.2f}s"
    if minutes:
        return f"{minutes}m {remaining_seconds:05.2f}s"
    return f"{remaining_seconds:.2f}s"


def count_labels(rows: pd.DataFrame) -> tuple[int, int]:
    positive = int((rows["label"] == 1).sum())
    return positive, int(len(rows) - positive)


def candidate_row(
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
        "experiment_name": "E8_soft_ensemble",
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
    train_mapped = e7.prepare_with_mapping(train_df, task_to_id)
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


def predict_task_scores(
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
) -> tuple[dict[tuple[str, str], dict[str, object]], list[dict[str, object]]]:
    train_mapped = e7.prepare_with_mapping(train_df, task_to_id)
    eval_mapped = e7.prepare_with_mapping(eval_df, task_to_id)
    predictions: dict[tuple[str, str], dict[str, object]] = {}
    candidate_rows = []

    for task_name in sorted(set(train_mapped["task_name"]) & set(eval_mapped["task_name"])):
        train_task = train_mapped[train_mapped["task_name"] == task_name]
        eval_task = eval_mapped[eval_mapped["task_name"] == task_name]
        x_eval = base.encode_peptides(eval_task["peptide_sequence"], peptide_length)
        task_eval = eval_task["task_id"].to_numpy(dtype=np.int64).copy()
        y_eval = eval_task["label"].to_numpy(dtype=np.int64).copy()
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
        row = candidate_row(seed, split, branch, group_name, train_task, eval_task, metrics)
        candidate_rows.append(row)
        first = eval_task.iloc[0]
        key = (str(first["target_tissue"]), str(first["mhc_restriction"]))
        predictions[key] = {
            "train_task": train_task,
            "eval_task": eval_task,
            "y_true": y_true,
            "y_score": y_score,
            "metrics": metrics,
            "candidate_row": row,
        }

    return predictions, candidate_rows


def clipped_delta_weight(global_metric: float, hla_metric: float, args: argparse.Namespace) -> float:
    delta = hla_metric - global_metric
    return float(np.clip(args.clipped_base_weight + args.clipped_scale * delta, args.min_hla_weight, args.max_hla_weight))


def softmax_weight(global_metric: float, hla_metric: float, args: argparse.Namespace) -> float:
    delta = np.clip((hla_metric - global_metric) / args.softmax_temperature, -50.0, 50.0)
    return float(1.0 / (1.0 + np.exp(-delta)))


def compute_hla_weight(strategy: str, global_metric: float, hla_metric: float, args: argparse.Namespace) -> float:
    if not np.isfinite(hla_metric):
        return 0.0
    if strategy == "e8a_fixed_average":
        return 0.5
    if strategy == "e8b_validation_delta_clipped":
        return clipped_delta_weight(global_metric, hla_metric, args)
    if strategy == "e8c_validation_softmax":
        return softmax_weight(global_metric, hla_metric, args)
    raise ValueError(f"Unknown ensemble strategy: {strategy}")


def make_metric_row(
    seed: int,
    strategy: str,
    train_task: pd.DataFrame,
    eval_task: pd.DataFrame,
    metrics: dict[str, float],
    hla_weight: float,
    global_validation_metric: float,
    hla_validation_metric: float,
    selection_metric: str,
) -> dict[str, object]:
    train_positive, train_negative = count_labels(train_task)
    test_positive, test_negative = count_labels(eval_task)
    first = eval_task.iloc[0]
    hla_metric = hla_validation_metric
    validation_delta = hla_metric - global_validation_metric if np.isfinite(hla_metric) else np.nan
    return {
        "experiment_name": "E8_soft_ensemble",
        "seed": seed,
        "model": strategy,
        "target_tissue": first["target_tissue"],
        "mhc_restriction": first["mhc_restriction"],
        "train_rows": len(train_task),
        "test_rows": len(eval_task),
        "train_positive": train_positive,
        "train_negative": train_negative,
        "test_positive": test_positive,
        "test_negative": test_negative,
        **metrics,
        "ensemble_strategy": strategy,
        "selection_metric": selection_metric,
        "hla_weight": hla_weight,
        "global_weight": 1.0 - hla_weight,
        "global_validation_metric": global_validation_metric,
        "hla_validation_metric": hla_metric,
        "validation_delta_hla_minus_global": validation_delta,
    }


def make_weight_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "experiment_name": row["experiment_name"],
        "seed": row["seed"],
        "target_tissue": row["target_tissue"],
        "mhc_restriction": row["mhc_restriction"],
        "ensemble_strategy": row["ensemble_strategy"],
        "selection_metric": row["selection_metric"],
        "hla_weight": row["hla_weight"],
        "global_weight": row["global_weight"],
        "global_validation_metric": row["global_validation_metric"],
        "hla_validation_metric": row["hla_validation_metric"],
        "validation_delta_hla_minus_global": row["validation_delta_hla_minus_global"],
    }


def build_ensembles(
    args: argparse.Namespace,
    seed: int,
    validation_global: dict[tuple[str, str], dict[str, object]],
    validation_hla: dict[tuple[str, str], dict[str, object]],
    test_global: dict[tuple[str, str], dict[str, object]],
    test_hla: dict[tuple[str, str], dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    result_rows = []
    weight_rows = []
    task_keys = sorted(set(test_global) & set(validation_global))

    for key in task_keys:
        global_validation_metric = float(validation_global[key]["metrics"][args.selection_metric])
        hla_validation_metric = (
            float(validation_hla[key]["metrics"][args.selection_metric]) if key in validation_hla else np.nan
        )
        global_test = test_global[key]
        hla_test = test_hla.get(key)
        if hla_test is None:
            continue
        if not np.array_equal(global_test["y_true"], hla_test["y_true"]):
            raise ValueError(f"Mismatched labels for task {key}")

        y_true = global_test["y_true"]
        global_score = global_test["y_score"]
        hla_score = hla_test["y_score"]
        for strategy in MODEL_NAMES:
            hla_weight = compute_hla_weight(strategy, global_validation_metric, hla_validation_metric, args)
            ensemble_score = hla_weight * hla_score + (1.0 - hla_weight) * global_score
            metrics = base.evaluate(y_true, ensemble_score)
            row = make_metric_row(
                seed,
                strategy,
                global_test["train_task"],
                global_test["eval_task"],
                metrics,
                hla_weight,
                global_validation_metric,
                hla_validation_metric,
                args.selection_metric,
            )
            result_rows.append(row)
            weight_rows.append(make_weight_row(row))

    result_rows.sort(key=lambda row: (int(row["seed"]), str(row["model"]), str(row["target_tissue"]), str(row["mhc_restriction"])))
    weight_rows.sort(key=lambda row: (int(row["seed"]), str(row["ensemble_strategy"]), str(row["target_tissue"]), str(row["mhc_restriction"])))
    return result_rows, weight_rows


def train_and_predict_branches(
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
) -> tuple[
    dict[tuple[str, str], dict[str, object]],
    dict[tuple[str, str], dict[str, object]],
    dict[tuple[str, str], dict[str, object]],
    dict[tuple[str, str], dict[str, object]],
    list[dict[str, object]],
]:
    candidate_rows: list[dict[str, object]] = []

    print("  train validation global branch")
    step_start = time.perf_counter()
    validation_global_model = train_shared_heads_model(
        args, torch, nn, DataLoader, TensorDataset, train_core, task_to_id, peptide_length, device
    )
    print(
        f"    time train_validation_global_branch all_tasks n_tasks={len(task_to_id)} "
        f"train_rows={len(train_core)} duration={format_duration(time.perf_counter() - step_start)}"
    )
    validation_global, rows = predict_task_scores(
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
    candidate_rows.extend(rows)

    validation_hla: dict[tuple[str, str], dict[str, object]] = {}
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
            f"    time train_validation_hla_branch {hla} n_tasks={len(local_task_to_id)} "
            f"train_rows={len(hla_train)} duration={format_duration(time.perf_counter() - step_start)}"
        )
        predictions, rows = predict_task_scores(
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
        validation_hla.update(predictions)
        candidate_rows.extend(rows)

    print("  train final global branch")
    step_start = time.perf_counter()
    final_global_model = train_shared_heads_model(
        args, torch, nn, DataLoader, TensorDataset, full_train_df, task_to_id, peptide_length, device
    )
    print(
        f"    time train_final_global_branch all_tasks n_tasks={len(task_to_id)} "
        f"train_rows={len(full_train_df)} duration={format_duration(time.perf_counter() - step_start)}"
    )
    test_global, rows = predict_task_scores(
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
    candidate_rows.extend(rows)

    test_hla: dict[tuple[str, str], dict[str, object]] = {}
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
            f"    time train_final_hla_branch {hla} n_tasks={len(local_task_to_id)} "
            f"train_rows={len(hla_train)} duration={format_duration(time.perf_counter() - step_start)}"
        )
        predictions, rows = predict_task_scores(
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
        test_hla.update(predictions)
        candidate_rows.extend(rows)

    return validation_global, validation_hla, test_global, test_hla, candidate_rows


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
            "baseline_model": "E2_all_tasks_shared_heads",
            "candidate_model": row["model"],
            "ensemble_strategy": row["ensemble_strategy"],
            "hla_weight": row["hla_weight"],
        }
        for metric in METRICS:
            comparison[f"delta_{metric}"] = float(row[metric]) - float(baseline_row[metric])
        comparisons.append(comparison)
    comparisons.sort(key=lambda row: (str(row["candidate_model"]), int(row["seed"]), -float(row["delta_auroc"])))
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
    candidate_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    run_start = time.perf_counter()

    print(f"device: {device}")
    print(f"n_tasks: {len(mappings['tasks'])}")
    print(f"selection_metric: {args.selection_metric}")
    print(f"ensemble_strategies: {MODEL_NAMES}")
    for seed in args.seeds:
        seed_start = time.perf_counter()
        base.set_seed(seed, torch)
        print(f"experiment: E8_soft_ensemble seed={seed}")
        train_core, validation_df = e7.split_train_validation(train_df, args.validation_fraction, seed)
        print(f"  train_core_rows={len(train_core)} validation_rows={len(validation_df)}")
        validation_global, validation_hla, test_global, test_hla, seed_candidate_rows = train_and_predict_branches(
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
        seed_result_rows, seed_weight_rows = build_ensembles(
            args,
            seed,
            validation_global,
            validation_hla,
            test_global,
            test_hla,
        )
        result_rows.extend(seed_result_rows)
        candidate_rows.extend(seed_candidate_rows)
        weight_rows.extend(seed_weight_rows)
        for strategy in MODEL_NAMES:
            weights = [float(row["hla_weight"]) for row in seed_weight_rows if row["ensemble_strategy"] == strategy]
            print(
                f"  {strategy} hla_weight_mean={float(np.mean(weights)):.4f} "
                f"min={float(np.min(weights)):.4f} max={float(np.max(weights)):.4f}"
            )
        print(f"  time seed_total seed={seed} duration={format_duration(time.perf_counter() - seed_start)}")

    summary_rows = base.summarize_results(result_rows)
    stability_rows = base.summarize_seed_stability(summary_rows)
    baseline_rows = read_e2_baseline(args.baseline_per_task)
    comparison_rows = compare_against_e2(result_rows, baseline_rows)

    base.write_csv(args.per_task_output, ENSEMBLE_COLUMNS, result_rows)
    base.write_csv(args.summary_output, base.SUMMARY_COLUMNS, summary_rows)
    base.write_csv(args.stability_output, base.STABILITY_COLUMNS, stability_rows)
    base.write_csv(args.candidate_output, CANDIDATE_COLUMNS, candidate_rows)
    base.write_csv(args.weight_output, WEIGHT_COLUMNS, weight_rows)
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
        "weight_output": str(args.weight_output),
        "comparison_output": str(args.comparison_output),
        "n_tasks": len(mappings["tasks"]),
        "seeds": args.seeds,
        "device": device,
        "peptide_length": peptide_length,
        "selection_metric": args.selection_metric,
        "validation_fraction": args.validation_fraction,
        "ensemble_strategies": MODEL_NAMES,
        "min_hla_weight": args.min_hla_weight,
        "max_hla_weight": args.max_hla_weight,
        "clipped_base_weight": args.clipped_base_weight,
        "clipped_scale": args.clipped_scale,
        "softmax_temperature": args.softmax_temperature,
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
    print(f"wrote: {args.weight_output}")
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
        default=project_path("results/tissuePMHC_soft_ensemble/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=project_path("results/tissuePMHC_soft_ensemble/summary_metrics.csv"),
    )
    parser.add_argument(
        "--stability-output",
        type=Path,
        default=project_path("results/tissuePMHC_soft_ensemble/stability_metrics.csv"),
    )
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=project_path("results/tissuePMHC_soft_ensemble/candidate_metrics.csv"),
    )
    parser.add_argument(
        "--weight-output",
        type=Path,
        default=project_path("results/tissuePMHC_soft_ensemble/weight_metrics.csv"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=project_path("results/tissuePMHC_soft_ensemble/comparison_metrics.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=project_path("results/tissuePMHC_soft_ensemble/metadata.json"),
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
    parser.add_argument("--min-hla-weight", type=float, default=0.15)
    parser.add_argument("--max-hla-weight", type=float, default=0.85)
    parser.add_argument("--clipped-base-weight", type=float, default=0.5)
    parser.add_argument("--clipped-scale", type=float, default=5.0)
    parser.add_argument("--softmax-temperature", type=float, default=0.02)
    parser.add_argument("--max-tasks", type=int, default=0, help="Optional smoke-test limit on the number of tasks.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
