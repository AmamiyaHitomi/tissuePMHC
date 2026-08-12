#!/usr/bin/env python3
"""Run E11 DB-MTL on the E2 shared-head performance line for tissuePMHC.

Roadmap role: E11-on-E2 performance line.

DB-MTL is used here as dynamic-balanced multi-task learning. It keeps the E2
architecture:

    shared peptide encoder + task-specific heads

and dynamically reweights task losses by two signals:

1. loss scale: tasks with larger current loss receive more weight.
2. training rate: tasks whose loss decreases more slowly receive more weight.

This is a loss-balancing experiment, not an HLA pseudo-sequence experiment.
The E4 pseudo-sequence line remains a separate biological-representation branch.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPEAT_SEEDS = [20260704, 20260705, 20260706]
MODEL_CHOICES = ["e2_task_balanced", "e11_dbmtl"]
METRICS = ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]

WEIGHT_COLUMNS = [
    "experiment_name",
    "seed",
    "model",
    "epoch",
    "task_name",
    "target_tissue",
    "mhc_restriction",
    "dbmtl_weight",
    "epoch_mean_loss",
    "initial_loss",
    "relative_loss",
]

DIAGNOSTIC_COLUMNS = [
    "experiment_name",
    "seed",
    "model",
    "epoch",
    "mean_loss",
    "min_task_loss",
    "max_task_loss",
    "min_dbmtl_weight",
    "max_dbmtl_weight",
    "weight_entropy",
]

COMPARISON_COLUMNS = [
    "seed",
    "target_tissue",
    "mhc_restriction",
    "baseline_model",
    "candidate_model",
    "baseline_source",
    "delta_accuracy",
    "delta_balanced_accuracy",
    "delta_auroc",
    "delta_auprc",
    "delta_f1",
    "delta_mcc",
]


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def set_seed(seed: int, torch: Any) -> None:
    base.set_seed(seed, torch)
    random.seed(seed)
    np.random.seed(seed)


def entropy(weights: np.ndarray) -> float:
    normalized = weights / (weights.sum() + 1e-12)
    clipped = np.clip(normalized, 1e-12, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def make_task_arrays(
    train_df: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
) -> list[dict[str, object]]:
    task_arrays: list[dict[str, object]] = []
    for task_name in mappings["tasks"]:
        rows = train_df[train_df["task_name"] == task_name]
        tissue, hla = task_name.split("||", 1)
        task_arrays.append(
            {
                "task_name": task_name,
                "target_tissue": tissue,
                "mhc_restriction": hla,
                "task_id": int(mappings["task_to_id"][task_name]),
                "peptides": base.encode_peptides(rows["peptide_sequence"], peptide_length),
                "labels": rows["label"].to_numpy(dtype=np.int64),
            }
        )
    return task_arrays


def sample_task_batches(
    rng: np.random.Generator,
    task_arrays: list[dict[str, object]],
    task_batch_size: int,
) -> list[dict[str, np.ndarray]]:
    batches = []
    for task in task_arrays:
        labels = task["labels"]
        indices = rng.integers(0, len(labels), size=task_batch_size)
        batches.append(
            {
                "peptides": task["peptides"][indices],
                "task_ids": np.full(task_batch_size, int(task["task_id"]), dtype=np.int64),
                "labels": labels[indices],
            }
        )
    return batches


def task_loss(torch: Any, model: Any, batch: dict[str, np.ndarray], device: str) -> Any:
    peptides = torch.as_tensor(batch["peptides"], device=device)
    task_ids = torch.as_tensor(batch["task_ids"], device=device)
    labels = torch.as_tensor(batch["labels"], device=device)
    logits = model(peptides, task_ids)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())


def define_shared_heads_model(args: argparse.Namespace, nn: Any, n_tasks: int, peptide_length: int) -> Any:
    _, SharedTaskHeadsModel, _ = base.define_models(nn)
    return SharedTaskHeadsModel(
        peptide_length,
        n_tasks,
        args.embedding_dim,
        args.hidden_dim,
        args.dropout,
    )


def update_dbmtl_weights(
    args: argparse.Namespace,
    mean_losses: np.ndarray,
    initial_losses: np.ndarray,
) -> np.ndarray:
    eps = 1e-8
    loss_scale = mean_losses / (np.mean(mean_losses) + eps)
    relative_loss = mean_losses / (initial_losses + eps)
    relative_loss = relative_loss / (np.mean(relative_loss) + eps)
    raw = np.power(loss_scale, args.loss_scale_power) * np.power(relative_loss, args.training_rate_power)
    raw = np.clip(raw, args.min_task_weight, args.max_task_weight)
    return raw / (np.mean(raw) + eps)


def train_one_model(
    args: argparse.Namespace,
    torch: Any,
    nn: Any,
    train_df: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    device: str,
    seed: int,
    model_name: str,
) -> tuple[Any, list[dict[str, object]], list[dict[str, object]]]:
    model = define_shared_heads_model(args, nn, len(mappings["tasks"]), peptide_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    task_arrays = make_task_arrays(train_df, mappings, peptide_length)
    rng = np.random.default_rng(seed)
    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch <= 0:
        max_rows = max(len(task["labels"]) for task in task_arrays)
        steps_per_epoch = int(np.ceil(max_rows / args.task_batch_size))

    use_dbmtl = model_name == "e11_dbmtl"
    experiment_name = "E11_DB_MTL" if use_dbmtl else "E2_task_balanced"
    task_weights = np.ones(len(task_arrays), dtype=np.float64)
    initial_losses: np.ndarray | None = None
    weight_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []

    print(f"  train setup: epochs={args.epochs} steps_per_epoch={steps_per_epoch}")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        epoch_losses = []
        weight_tensor = torch.as_tensor(task_weights, dtype=torch.float32, device=device)
        for _ in range(steps_per_epoch):
            batches = sample_task_batches(rng, task_arrays, args.task_batch_size)
            optimizer.zero_grad(set_to_none=True)
            losses = torch.stack([task_loss(torch, model, batch, device) for batch in batches])
            if use_dbmtl:
                train_loss = torch.mean(weight_tensor * losses)
            else:
                train_loss = torch.mean(losses)
            train_loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            epoch_losses.append([float(loss.detach().cpu()) for loss in losses])

        mean_losses = np.asarray(epoch_losses, dtype=np.float64).mean(axis=0)
        if initial_losses is None:
            initial_losses = mean_losses.copy()
        if use_dbmtl:
            task_weights = update_dbmtl_weights(args, mean_losses, initial_losses)
        else:
            task_weights = np.ones(len(task_arrays), dtype=np.float64)
        relative_losses = mean_losses / (initial_losses + 1e-8)

        for task_index, task in enumerate(task_arrays):
            weight_rows.append(
                {
                    "experiment_name": experiment_name,
                    "seed": seed,
                    "model": model_name,
                    "epoch": epoch,
                    "task_name": task["task_name"],
                    "target_tissue": task["target_tissue"],
                    "mhc_restriction": task["mhc_restriction"],
                    "dbmtl_weight": float(task_weights[task_index]),
                    "epoch_mean_loss": float(mean_losses[task_index]),
                    "initial_loss": float(initial_losses[task_index]),
                    "relative_loss": float(relative_losses[task_index]),
                }
            )

        diagnostic_rows.append(
            {
                "experiment_name": experiment_name,
                "seed": seed,
                "model": model_name,
                "epoch": epoch,
                "mean_loss": float(np.mean(mean_losses)),
                "min_task_loss": float(np.min(mean_losses)),
                "max_task_loss": float(np.max(mean_losses)),
                "min_dbmtl_weight": float(np.min(task_weights)),
                "max_dbmtl_weight": float(np.max(task_weights)),
                "weight_entropy": entropy(task_weights),
            }
        )
        print(
            f"  epoch {epoch:02d}/{args.epochs} "
            f"time={time.perf_counter() - epoch_start:.2f}s "
            f"mean_loss={float(np.mean(mean_losses)):.4f} "
            f"min_weight={float(np.min(task_weights)):.4f} max_weight={float(np.max(task_weights)):.4f}"
        )

    return model, weight_rows, diagnostic_rows


def evaluate_shared_heads(
    args: argparse.Namespace,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    model: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    device: str,
    model_name: str,
) -> list[dict[str, object]]:
    rows = []
    for task_name in mappings["tasks"]:
        train_task = train_df[train_df["task_name"] == task_name]
        test_task = test_df[test_df["task_name"] == task_name]
        x_test = base.encode_peptides(test_task["peptide_sequence"], peptide_length)
        task_test = test_task["task_id"].to_numpy(dtype=np.int64).copy()
        y_test = test_task["label"].to_numpy(dtype=np.int64).copy()
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
    return rows


def add_experiment_context(rows: list[dict[str, object]], experiment_name: str, seed: int) -> list[dict[str, object]]:
    for row in rows:
        row["experiment_name"] = experiment_name
        row["seed"] = seed
    return rows


def compare_internal(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows_by_key: dict[tuple[int, str, str, str], dict[str, object]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["target_tissue"]), str(row["mhc_restriction"]), str(row["model"]))
        rows_by_key[key] = row

    comparisons = []
    for (seed, tissue, hla, model), candidate in rows_by_key.items():
        if model != "e11_dbmtl":
            continue
        baseline = rows_by_key.get((seed, tissue, hla, "e2_task_balanced"))
        if baseline is None:
            continue
        comparison = {
            "seed": seed,
            "target_tissue": tissue,
            "mhc_restriction": hla,
            "baseline_model": "e2_task_balanced",
            "candidate_model": "e11_dbmtl",
            "baseline_source": "internal_task_balanced_e2",
        }
        for metric in METRICS:
            comparison[f"delta_{metric}"] = float(candidate[metric]) - float(baseline[metric])
        comparisons.append(comparison)
    comparisons.sort(key=lambda row: (int(row["seed"]), -float(row["delta_auroc"])))
    return comparisons


def read_external_rows(path: Path, model: str) -> dict[tuple[int, str, str], dict[str, object]]:
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    if "model" in table.columns:
        table = table[table["model"] == model].copy()
    return {
        (int(row["seed"]), str(row["target_tissue"]), str(row["mhc_restriction"])): row
        for row in table.to_dict("records")
    }


def compare_external(
    rows: list[dict[str, object]],
    baseline_rows: dict[tuple[int, str, str], dict[str, object]],
    baseline_model: str,
    baseline_source: str,
) -> list[dict[str, object]]:
    comparisons = []
    for row in rows:
        if row["model"] != "e11_dbmtl":
            continue
        key = (int(row["seed"]), str(row["target_tissue"]), str(row["mhc_restriction"]))
        baseline = baseline_rows.get(key)
        if baseline is None:
            continue
        comparison = {
            "seed": key[0],
            "target_tissue": key[1],
            "mhc_restriction": key[2],
            "baseline_model": baseline_model,
            "candidate_model": "e11_dbmtl",
            "baseline_source": baseline_source,
        }
        for metric in METRICS:
            comparison[f"delta_{metric}"] = float(row[metric]) - float(baseline[metric])
        comparisons.append(comparison)
    return comparisons


def run_one_experiment(
    args: argparse.Namespace,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    device: str,
    seed: int,
    model_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    set_seed(seed, torch)
    experiment_name = "E11_DB_MTL" if model_name == "e11_dbmtl" else "E2_task_balanced"
    print(f"experiment: {experiment_name} seed={seed}")
    experiment_start = time.perf_counter()
    model, weight_rows, diagnostic_rows = train_one_model(
        args,
        torch,
        nn,
        train_df,
        mappings,
        peptide_length,
        device,
        seed,
        model_name,
    )
    metric_rows = evaluate_shared_heads(
        args,
        torch,
        DataLoader,
        TensorDataset,
        model,
        train_df,
        test_df,
        mappings,
        peptide_length,
        device,
        model_name,
    )
    print(f"seed/model time: seed={seed} model={model_name} time={time.perf_counter() - experiment_start:.2f}s")
    return add_experiment_context(metric_rows, experiment_name, seed), weight_rows, diagnostic_rows


def run(args: argparse.Namespace) -> None:
    run_start = time.perf_counter()
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
    weight_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []

    print(f"device: {device}")
    print(f"n_tasks: {len(mappings['tasks'])}")
    print(f"task_batch_size: {args.task_batch_size}")
    print(f"models: {args.models}")

    for seed in args.seeds:
        for model_name in args.models:
            metrics, weights, diagnostics = run_one_experiment(
                args,
                torch,
                nn,
                DataLoader,
                TensorDataset,
                train_df,
                test_df,
                mappings,
                peptide_length,
                device,
                seed,
                model_name,
            )
            result_rows.extend(metrics)
            weight_rows.extend(weights)
            diagnostic_rows.extend(diagnostics)

    summary_rows = base.summarize_results(result_rows)
    stability_rows = base.summarize_seed_stability(summary_rows)
    comparison_rows = compare_internal(result_rows)

    e2_external = read_external_rows(args.e2_baseline_per_task, "shared_peptide_encoder_task_heads")
    e8_external = read_external_rows(args.e8_per_task, "e8a_fixed_average")
    e9_external = read_external_rows(args.e9_per_task, "e9_e2_cagrad")
    e10_external = read_external_rows(args.e10_per_task, "e10_mmoe")
    e10b_external = read_external_rows(args.e10b_per_task, "e10b_4experts_256")
    external_comparisons = []
    external_comparisons.extend(
        compare_external(result_rows, e2_external, "shared_peptide_encoder_task_heads", str(args.e2_baseline_per_task))
    )
    external_comparisons.extend(compare_external(result_rows, e8_external, "e8a_fixed_average", str(args.e8_per_task)))
    external_comparisons.extend(compare_external(result_rows, e9_external, "e9_e2_cagrad", str(args.e9_per_task)))
    external_comparisons.extend(compare_external(result_rows, e10_external, "e10_mmoe", str(args.e10_per_task)))
    external_comparisons.extend(
        compare_external(result_rows, e10b_external, "e10b_4experts_256", str(args.e10b_per_task))
    )
    external_comparisons.sort(
        key=lambda row: (str(row["baseline_model"]), int(row["seed"]), -float(row["delta_auroc"]))
    )

    base.write_csv(args.per_task_output, base.METRIC_COLUMNS, result_rows)
    base.write_csv(args.summary_output, base.SUMMARY_COLUMNS, summary_rows)
    base.write_csv(args.stability_output, base.STABILITY_COLUMNS, stability_rows)
    base.write_csv(args.comparison_output, COMPARISON_COLUMNS, comparison_rows)
    base.write_csv(args.external_comparison_output, COMPARISON_COLUMNS, external_comparisons)
    base.write_csv(args.weight_output, WEIGHT_COLUMNS, weight_rows)
    base.write_csv(args.diagnostic_output, DIAGNOSTIC_COLUMNS, diagnostic_rows)

    metadata = {
        "train": str(args.train),
        "test": str(args.test),
        "e2_baseline_per_task": str(args.e2_baseline_per_task),
        "e8_per_task": str(args.e8_per_task),
        "e9_per_task": str(args.e9_per_task),
        "e10_per_task": str(args.e10_per_task),
        "e10b_per_task": str(args.e10b_per_task),
        "per_task_output": str(args.per_task_output),
        "summary_output": str(args.summary_output),
        "stability_output": str(args.stability_output),
        "comparison_output": str(args.comparison_output),
        "external_comparison_output": str(args.external_comparison_output),
        "weight_output": str(args.weight_output),
        "diagnostic_output": str(args.diagnostic_output),
        "n_tasks": len(mappings["tasks"]),
        "models": args.models,
        "seeds": args.seeds,
        "device": device,
        "peptide_length": peptide_length,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "task_batch_size": args.task_batch_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss_scale_power": args.loss_scale_power,
        "training_rate_power": args.training_rate_power,
        "min_task_weight": args.min_task_weight,
        "max_task_weight": args.max_task_weight,
        "max_grad_norm": args.max_grad_norm,
        "task_mapping": mappings["task_to_id"],
        "tissue_mapping": mappings["tissue_to_id"],
        "hla_mapping": mappings["hla_to_id"],
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote: {args.per_task_output}")
    print(f"wrote: {args.summary_output}")
    print(f"wrote: {args.stability_output}")
    print(f"wrote: {args.comparison_output}")
    print(f"wrote: {args.external_comparison_output}")
    print(f"wrote: {args.weight_output}")
    print(f"wrote: {args.diagnostic_output}")
    print(f"wrote: {args.metadata_output}")
    print(f"run total time: {time.perf_counter() - run_start:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz"))
    parser.add_argument("--test", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_test.csv.gz"))
    parser.add_argument(
        "--e2-baseline-per-task",
        type=Path,
        default=project_path("results/tissuePMHC_neural_baselines_v2/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--e8-per-task",
        type=Path,
        default=project_path("results/tissuePMHC_soft_ensemble/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--e9-per-task",
        type=Path,
        default=project_path("results/tissuePMHC_cagrad/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--e10-per-task",
        type=Path,
        default=project_path("results/tissuePMHC_mmoe/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--e10b-per-task",
        type=Path,
        default=project_path("results/tissuePMHC_mmoe_tuning/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--per-task-output",
        type=Path,
        default=project_path("results/tissuePMHC_dbmtl/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=project_path("results/tissuePMHC_dbmtl/summary_metrics.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=project_path("results/tissuePMHC_dbmtl/metadata.json"),
    )
    parser.add_argument(
        "--stability-output",
        type=Path,
        default=project_path("results/tissuePMHC_dbmtl/stability_metrics.csv"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=project_path("results/tissuePMHC_dbmtl/comparison_metrics.csv"),
    )
    parser.add_argument(
        "--external-comparison-output",
        type=Path,
        default=project_path("results/tissuePMHC_dbmtl/external_comparison_metrics.csv"),
    )
    parser.add_argument(
        "--weight-output",
        type=Path,
        default=project_path("results/tissuePMHC_dbmtl/dbmtl_weight_history.csv"),
    )
    parser.add_argument(
        "--diagnostic-output",
        type=Path,
        default=project_path("results/tissuePMHC_dbmtl/dbmtl_diagnostics.csv"),
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=MODEL_CHOICES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_REPEAT_SEEDS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=0,
        help="If 0, use ceil(max task train rows / task_batch_size).",
    )
    parser.add_argument("--task-batch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512, help="Evaluation batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--loss-scale-power",
        type=float,
        default=0.5,
        help="How strongly DB-MTL upweights tasks with larger current loss.",
    )
    parser.add_argument(
        "--training-rate-power",
        type=float,
        default=0.5,
        help="How strongly DB-MTL upweights tasks whose loss decreases more slowly.",
    )
    parser.add_argument("--min-task-weight", type=float, default=0.25)
    parser.add_argument("--max-task-weight", type=float, default=4.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-tasks", type=int, default=0, help="Optional smoke-test limit on the number of tasks.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
