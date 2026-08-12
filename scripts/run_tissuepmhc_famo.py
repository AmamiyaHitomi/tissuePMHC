#!/usr/bin/env python3
"""Run E5 adaptive task weighting for tissuePMHC.

Roadmap role: E5-on-E2 performance line.
Although roadmap v1 imagined FAMO after E4, experiments showed E4 was weaker
than E2, so this script correctly applies FAMO to the E2 shared-head baseline.

E5 tests whether FAMO-style adaptive task loss weighting improves the strongest
current architecture:

    shared peptide encoder + task-specific heads

For a fair comparison, this script runs both:

1. E2_task_balanced: equal mean loss across tasks
2. E5_E2_FAMO: adaptive FAMO task loss weighting

FAMO means Fast Adaptive Multitask Optimization. It dynamically updates task
loss weights so task losses decrease in a more balanced way.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPEAT_SEEDS = [20260704, 20260705, 20260706]
MODEL_CHOICES = ["e2_task_balanced", "e5_famo"]

WEIGHT_COLUMNS = [
    "experiment_name",
    "seed",
    "epoch",
    "task_name",
    "target_tissue",
    "mhc_restriction",
    "famo_weight",
    "epoch_mean_loss",
]

COMPARISON_COLUMNS = [
    "seed",
    "target_tissue",
    "mhc_restriction",
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


class FamoTaskWeighting:
    def __init__(
        self,
        torch: Any,
        n_tasks: int,
        device: str,
        gamma: float,
        weight_lr: float,
        max_grad_norm: float,
    ):
        self.torch = torch
        self.n_tasks = n_tasks
        self.device = device
        self.min_losses = torch.zeros(n_tasks, device=device)
        self.logits = torch.zeros(n_tasks, device=device, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.logits], lr=weight_lr, weight_decay=gamma)
        self.max_grad_norm = max_grad_norm
        self.previous_losses = None

    def weights(self):
        return self.torch.softmax(self.logits, dim=-1)

    def weighted_loss(self, losses):
        self.previous_losses = losses.detach()
        weights = self.weights()
        shifted_losses = losses - self.min_losses + 1e-8
        normalizer = (weights / shifted_losses).sum().detach()
        return (shifted_losses.log() * weights / normalizer).sum()

    def backward(self, losses, shared_parameters) -> Any:
        loss = self.weighted_loss(losses)
        loss.backward()
        if self.max_grad_norm > 0:
            self.torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_grad_norm)
        return loss

    def update(self, current_losses) -> None:
        if self.previous_losses is None:
            return
        previous = self.previous_losses - self.min_losses + 1e-8
        current = current_losses.detach() - self.min_losses + 1e-8
        loss_drop = previous.log() - current.log()
        with self.torch.enable_grad():
            grad = self.torch.autograd.grad(
                self.weights(),
                self.logits,
                grad_outputs=loss_drop.detach(),
            )[0]
        self.optimizer.zero_grad()
        self.logits.grad = grad
        self.optimizer.step()


def set_seed(seed: int, torch: Any) -> None:
    base.set_seed(seed, torch)
    random.seed(seed)
    np.random.seed(seed)


def make_task_training_arrays(
    train_df: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
) -> list[dict[str, np.ndarray | str | int]]:
    task_arrays = []
    for task_name in mappings["tasks"]:
        rows = train_df[train_df["task_name"] == task_name]
        task_arrays.append(
            {
                "task_name": task_name,
                "task_id": int(mappings["task_to_id"][task_name]),
                "peptides": base.encode_peptides(rows["peptide_sequence"], peptide_length),
                "labels": rows["label"].to_numpy(dtype=np.int64),
            }
        )
    return task_arrays


def sample_balanced_task_batch(
    rng: np.random.Generator,
    task_arrays: list[dict[str, np.ndarray | str | int]],
    task_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    peptide_batches = []
    task_id_batches = []
    label_batches = []
    for task in task_arrays:
        labels = task["labels"]
        indices = rng.integers(0, len(labels), size=task_batch_size)
        peptide_batches.append(task["peptides"][indices])
        task_id_batches.append(np.full(task_batch_size, int(task["task_id"]), dtype=np.int64))
        label_batches.append(labels[indices])
    return (
        np.concatenate(peptide_batches, axis=0),
        np.concatenate(task_id_batches, axis=0),
        np.concatenate(label_batches, axis=0),
    )


def task_loss_vector(torch: Any, model: Any, peptides: Any, task_ids: Any, labels: Any, n_tasks: int, task_batch_size: int):
    logits = model(peptides, task_ids)
    per_sample_losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels.float(),
        reduction="none",
    )
    return per_sample_losses.reshape(n_tasks, task_batch_size).mean(dim=1)


def train_task_balanced_shared_heads(
    args: argparse.Namespace,
    torch: Any,
    nn: Any,
    train_df: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    device: str,
    seed: int,
    use_famo: bool,
) -> tuple[Any, list[dict[str, object]]]:
    _, SharedTaskHeadsModel, _ = base.define_models(nn)
    model = SharedTaskHeadsModel(
        peptide_length,
        len(mappings["tasks"]),
        args.embedding_dim,
        args.hidden_dim,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    task_arrays = make_task_training_arrays(train_df, mappings, peptide_length)
    rng = np.random.default_rng(seed)
    n_tasks = len(task_arrays)
    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch <= 0:
        max_rows = max(len(task["labels"]) for task in task_arrays)
        steps_per_epoch = int(np.ceil(max_rows / args.task_batch_size))

    famo = None
    if use_famo:
        famo = FamoTaskWeighting(
            torch,
            n_tasks=n_tasks,
            device=device,
            gamma=args.famo_gamma,
            weight_lr=args.famo_weight_lr,
            max_grad_norm=args.famo_max_grad_norm,
        )

    history_rows: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for _ in range(steps_per_epoch):
            x_batch, task_batch, y_batch = sample_balanced_task_batch(rng, task_arrays, args.task_batch_size)
            peptides = torch.as_tensor(x_batch, device=device)
            task_ids = torch.as_tensor(task_batch, device=device)
            labels = torch.as_tensor(y_batch, device=device)

            optimizer.zero_grad()
            losses = task_loss_vector(torch, model, peptides, task_ids, labels, n_tasks, args.task_batch_size)
            if use_famo:
                assert famo is not None
                train_loss = famo.backward(losses, model.encoder.parameters())
            else:
                train_loss = losses.mean()
                train_loss.backward()
            optimizer.step()

            if use_famo:
                with torch.no_grad():
                    current_losses = task_loss_vector(
                        torch,
                        model,
                        peptides,
                        task_ids,
                        labels,
                        n_tasks,
                        args.task_batch_size,
                    )
                assert famo is not None
                famo.update(current_losses)
                epoch_losses.append(current_losses.detach().cpu().numpy())
            else:
                epoch_losses.append(losses.detach().cpu().numpy())

        if use_famo:
            assert famo is not None
            weights = famo.weights().detach().cpu().numpy()
        else:
            weights = np.full(n_tasks, 1.0 / n_tasks, dtype=np.float64)
        mean_losses = np.stack(epoch_losses, axis=0).mean(axis=0)
        for task_index, task_name in enumerate(mappings["tasks"]):
            tissue, hla = task_name.split("||", 1)
            history_rows.append(
                {
                    "epoch": epoch,
                    "task_name": task_name,
                    "target_tissue": tissue,
                    "mhc_restriction": hla,
                    "famo_weight": float(weights[task_index]),
                    "epoch_mean_loss": float(mean_losses[task_index]),
                }
            )
        print(
            f"  epoch {epoch:02d}/{args.epochs} "
            f"mean_loss={float(np.mean(mean_losses)):.4f} "
            f"min_weight={float(np.min(weights)):.4f} max_weight={float(np.max(weights)):.4f}"
        )

    return model, history_rows


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
        task_test = test_task["task_id"].to_numpy(dtype=np.int64)
        y_test = test_task["label"].to_numpy(dtype=np.int64)
        test_loader = base.build_loader(torch, DataLoader, TensorDataset, [x_test, task_test, y_test], args.batch_size, False)
        y_true, y_score = base.predict_scores(torch, model, test_loader, device, "task_heads")
        metrics = base.evaluate(y_true, y_score)
        rows.append(base.make_metric_row(model_name, train_task, test_task, metrics))
    return rows


def add_experiment_context(rows: list[dict[str, object]], experiment_name: str, seed: int) -> list[dict[str, object]]:
    for row in rows:
        row["experiment_name"] = experiment_name
        row["seed"] = seed
    return rows


def compare_famo_against_balanced(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows_by_key: dict[tuple[int, str, str, str], dict[str, object]] = {}
    for row in rows:
        key = (
            int(row["seed"]),
            str(row["target_tissue"]),
            str(row["mhc_restriction"]),
            str(row["model"]),
        )
        rows_by_key[key] = row

    comparison_rows = []
    for (seed, target_tissue, mhc_restriction, model), candidate in rows_by_key.items():
        if model != "e5_famo":
            continue
        baseline_row = rows_by_key.get((seed, target_tissue, mhc_restriction, "e2_task_balanced"))
        if baseline_row is None:
            continue
        comparison = {
            "seed": seed,
            "target_tissue": target_tissue,
            "mhc_restriction": mhc_restriction,
            "baseline_model": "e2_task_balanced",
            "candidate_model": "e5_famo",
        }
        for metric in ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]:
            comparison[f"delta_{metric}"] = float(candidate[metric]) - float(baseline_row[metric])
        comparison_rows.append(comparison)

    comparison_rows.sort(key=lambda row: (int(row["seed"]), -float(row["delta_auroc"])))
    return comparison_rows


def with_weight_context(rows: list[dict[str, object]], experiment_name: str, seed: int) -> list[dict[str, object]]:
    for row in rows:
        row["experiment_name"] = experiment_name
        row["seed"] = seed
    return rows


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
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    set_seed(seed, torch)
    use_famo = model_name == "e5_famo"
    experiment_name = "E5_E2_FAMO" if use_famo else "E2_task_balanced"
    print(f"experiment: {experiment_name} seed={seed}")
    model, weight_rows = train_task_balanced_shared_heads(
        args,
        torch,
        nn,
        train_df,
        mappings,
        peptide_length,
        device,
        seed,
        use_famo,
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
    return (
        add_experiment_context(metric_rows, experiment_name, seed),
        with_weight_context(weight_rows, experiment_name, seed),
    )


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
    weight_rows: list[dict[str, object]] = []

    print(f"device: {device}")
    print(f"n_tasks: {len(mappings['tasks'])}")
    print(f"task_batch_size: {args.task_batch_size}")
    for seed in args.seeds:
        for model_name in args.models:
            metrics, weights = run_one_experiment(
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

    summary_rows = base.summarize_results(result_rows)
    stability_rows = base.summarize_seed_stability(summary_rows)
    comparison_rows = compare_famo_against_balanced(result_rows)
    base.write_csv(args.per_task_output, base.METRIC_COLUMNS, result_rows)
    base.write_csv(args.summary_output, base.SUMMARY_COLUMNS, summary_rows)
    base.write_csv(args.stability_output, base.STABILITY_COLUMNS, stability_rows)
    base.write_csv(args.comparison_output, COMPARISON_COLUMNS, comparison_rows)
    base.write_csv(args.weight_output, WEIGHT_COLUMNS, weight_rows)

    metadata = {
        "train": str(args.train),
        "test": str(args.test),
        "per_task_output": str(args.per_task_output),
        "summary_output": str(args.summary_output),
        "stability_output": str(args.stability_output),
        "comparison_output": str(args.comparison_output),
        "weight_output": str(args.weight_output),
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
        "famo_gamma": args.famo_gamma,
        "famo_weight_lr": args.famo_weight_lr,
        "famo_max_grad_norm": args.famo_max_grad_norm,
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
    print(f"wrote: {args.weight_output}")
    print(f"wrote: {args.metadata_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz"))
    parser.add_argument("--test", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_test.csv.gz"))
    parser.add_argument(
        "--per-task-output",
        type=Path,
        default=project_path("results/tissuePMHC_famo/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=project_path("results/tissuePMHC_famo/summary_metrics.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=project_path("results/tissuePMHC_famo/metadata.json"),
    )
    parser.add_argument(
        "--stability-output",
        type=Path,
        default=project_path("results/tissuePMHC_famo/stability_metrics.csv"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=project_path("results/tissuePMHC_famo/comparison_metrics.csv"),
    )
    parser.add_argument(
        "--weight-output",
        type=Path,
        default=project_path("results/tissuePMHC_famo/task_weight_history.csv"),
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
    parser.add_argument("--task-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512, help="Evaluation batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--famo-gamma", type=float, default=0.01)
    parser.add_argument("--famo-weight-lr", type=float, default=0.025)
    parser.add_argument("--famo-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-tasks", type=int, default=0, help="Optional smoke-test limit on the number of tasks.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
