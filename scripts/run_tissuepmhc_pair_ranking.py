#!/usr/bin/env python3
"""Run E12 paired ranking loss on the E2 shared-head line for tissuePMHC.

Roadmap role: E12-on-E2/E8 performance line.

The tissuePMHC dataset was built as positive/negative peptide pairs. E12 uses
that structure directly. For each task, the model sees matched pairs and is
trained with:

    total_loss = BCE(pos/neg labels) + rank_weight * softplus(margin - (pos_logit - neg_logit))

`paired ranking loss` means the model is penalized when the positive peptide in
a pair does not score higher than the matched negative peptide. This keeps the
E2 architecture:

    shared peptide encoder + task-specific heads

and changes only the training objective.
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
MODEL_CHOICES = ["e2_pair_bce", "e12_pair_ranking"]
METRICS = ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]

DIAGNOSTIC_COLUMNS = [
    "experiment_name",
    "seed",
    "model",
    "epoch",
    "mean_total_loss",
    "mean_bce_loss",
    "mean_ranking_loss",
    "mean_pair_accuracy",
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


def define_shared_heads_model(args: argparse.Namespace, nn: Any, n_tasks: int, peptide_length: int) -> Any:
    _, SharedTaskHeadsModel, _ = base.define_models(nn)
    return SharedTaskHeadsModel(
        peptide_length,
        n_tasks,
        args.embedding_dim,
        args.hidden_dim,
        args.dropout,
    )


def make_pair_task_arrays(
    train_df: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
) -> list[dict[str, object]]:
    task_arrays: list[dict[str, object]] = []
    for task_name in mappings["tasks"]:
        rows = train_df[train_df["task_name"] == task_name]
        tissue, hla = task_name.split("||", 1)
        pos_rows = []
        neg_rows = []
        for _, pair_rows in rows.groupby("pair_id", sort=False):
            if len(pair_rows) != 2 or set(pair_rows["label"]) != {0, 1}:
                continue
            pos_rows.append(pair_rows[pair_rows["label"] == 1].iloc[0])
            neg_rows.append(pair_rows[pair_rows["label"] == 0].iloc[0])
        if not pos_rows:
            raise ValueError(f"No valid positive/negative pairs found for task: {task_name}")
        pos_df = pd.DataFrame(pos_rows)
        neg_df = pd.DataFrame(neg_rows)
        task_arrays.append(
            {
                "task_name": task_name,
                "target_tissue": tissue,
                "mhc_restriction": hla,
                "task_id": int(mappings["task_to_id"][task_name]),
                "positive_peptides": base.encode_peptides(pos_df["peptide_sequence"], peptide_length),
                "negative_peptides": base.encode_peptides(neg_df["peptide_sequence"], peptide_length),
                "n_pairs": len(pos_df),
            }
        )
    return task_arrays


def sample_pair_batches(
    rng: np.random.Generator,
    task_arrays: list[dict[str, object]],
    pair_batch_size: int,
) -> list[dict[str, np.ndarray]]:
    batches = []
    for task in task_arrays:
        n_pairs = int(task["n_pairs"])
        indices = rng.integers(0, n_pairs, size=pair_batch_size)
        batches.append(
            {
                "positive_peptides": task["positive_peptides"][indices],
                "negative_peptides": task["negative_peptides"][indices],
                "task_ids": np.full(pair_batch_size, int(task["task_id"]), dtype=np.int64),
            }
        )
    return batches


def pair_losses(
    args: argparse.Namespace,
    torch: Any,
    model: Any,
    batch: dict[str, np.ndarray],
    device: str,
) -> tuple[Any, Any, Any, Any]:
    positive = torch.as_tensor(batch["positive_peptides"], device=device)
    negative = torch.as_tensor(batch["negative_peptides"], device=device)
    task_ids = torch.as_tensor(batch["task_ids"], device=device)
    pos_logits = model(positive, task_ids)
    neg_logits = model(negative, task_ids)
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    ranking_loss = torch.nn.functional.softplus(args.ranking_margin - (pos_logits - neg_logits)).mean()
    pair_accuracy = (pos_logits > neg_logits).float().mean()
    return bce_loss, ranking_loss, pair_accuracy, pos_logits - neg_logits


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
) -> tuple[Any, list[dict[str, object]]]:
    model = define_shared_heads_model(args, nn, len(mappings["tasks"]), peptide_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    task_arrays = make_pair_task_arrays(train_df, mappings, peptide_length)
    rng = np.random.default_rng(seed)
    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch <= 0:
        max_pairs = max(int(task["n_pairs"]) for task in task_arrays)
        steps_per_epoch = int(np.ceil(max_pairs / args.pair_batch_size))

    use_ranking = model_name == "e12_pair_ranking"
    experiment_name = "E12_pair_ranking" if use_ranking else "E2_pair_BCE"
    diagnostic_rows: list[dict[str, object]] = []

    print(f"  train setup: epochs={args.epochs} steps_per_epoch={steps_per_epoch}")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_losses = []
        bce_losses = []
        ranking_losses = []
        pair_accuracies = []
        for _ in range(steps_per_epoch):
            batches = sample_pair_batches(rng, task_arrays, args.pair_batch_size)
            optimizer.zero_grad(set_to_none=True)
            batch_total_losses = []
            batch_bce_losses = []
            batch_ranking_losses = []
            batch_pair_accuracies = []
            for batch in batches:
                bce_loss, ranking_loss, pair_accuracy, _ = pair_losses(args, torch, model, batch, device)
                total_loss = bce_loss + args.ranking_weight * ranking_loss if use_ranking else bce_loss
                batch_total_losses.append(total_loss)
                batch_bce_losses.append(bce_loss.detach())
                batch_ranking_losses.append(ranking_loss.detach())
                batch_pair_accuracies.append(pair_accuracy.detach())
            train_loss = torch.stack(batch_total_losses).mean()
            train_loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_losses.append(float(train_loss.detach().cpu()))
            bce_losses.append(float(torch.stack(batch_bce_losses).mean().cpu()))
            ranking_losses.append(float(torch.stack(batch_ranking_losses).mean().cpu()))
            pair_accuracies.append(float(torch.stack(batch_pair_accuracies).mean().cpu()))

        diagnostic_rows.append(
            {
                "experiment_name": experiment_name,
                "seed": seed,
                "model": model_name,
                "epoch": epoch,
                "mean_total_loss": float(np.mean(total_losses)),
                "mean_bce_loss": float(np.mean(bce_losses)),
                "mean_ranking_loss": float(np.mean(ranking_losses)),
                "mean_pair_accuracy": float(np.mean(pair_accuracies)),
            }
        )
        print(
            f"  epoch {epoch:02d}/{args.epochs} "
            f"time={time.perf_counter() - epoch_start:.2f}s "
            f"total_loss={float(np.mean(total_losses)):.4f} "
            f"bce={float(np.mean(bce_losses)):.4f} "
            f"rank={float(np.mean(ranking_losses)):.4f} "
            f"pair_acc={float(np.mean(pair_accuracies)):.4f}"
        )

    return model, diagnostic_rows


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
        if model != "e12_pair_ranking":
            continue
        baseline = rows_by_key.get((seed, tissue, hla, "e2_pair_bce"))
        if baseline is None:
            continue
        comparison = {
            "seed": seed,
            "target_tissue": tissue,
            "mhc_restriction": hla,
            "baseline_model": "e2_pair_bce",
            "candidate_model": "e12_pair_ranking",
            "baseline_source": "internal_pair_bce",
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
        if row["model"] != "e12_pair_ranking":
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
            "candidate_model": "e12_pair_ranking",
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
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    set_seed(seed, torch)
    experiment_name = "E12_pair_ranking" if model_name == "e12_pair_ranking" else "E2_pair_BCE"
    print(f"experiment: {experiment_name} seed={seed}")
    experiment_start = time.perf_counter()
    model, diagnostic_rows = train_one_model(
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
    return add_experiment_context(metric_rows, experiment_name, seed), diagnostic_rows


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
    diagnostic_rows: list[dict[str, object]] = []

    print(f"device: {device}")
    print(f"n_tasks: {len(mappings['tasks'])}")
    print(f"pair_batch_size: {args.pair_batch_size}")
    print(f"models: {args.models}")

    for seed in args.seeds:
        for model_name in args.models:
            metrics, diagnostics = run_one_experiment(
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
            diagnostic_rows.extend(diagnostics)

    summary_rows = base.summarize_results(result_rows)
    stability_rows = base.summarize_seed_stability(summary_rows)
    comparison_rows = compare_internal(result_rows)

    e2_external = read_external_rows(args.e2_baseline_per_task, "shared_peptide_encoder_task_heads")
    e8_external = read_external_rows(args.e8_per_task, "e8a_fixed_average")
    e10_external = read_external_rows(args.e10_per_task, "e10_mmoe")
    e11_external = read_external_rows(args.e11_per_task, "e11_dbmtl")
    external_comparisons = []
    external_comparisons.extend(
        compare_external(result_rows, e2_external, "shared_peptide_encoder_task_heads", str(args.e2_baseline_per_task))
    )
    external_comparisons.extend(compare_external(result_rows, e8_external, "e8a_fixed_average", str(args.e8_per_task)))
    external_comparisons.extend(compare_external(result_rows, e10_external, "e10_mmoe", str(args.e10_per_task)))
    external_comparisons.extend(compare_external(result_rows, e11_external, "e11_dbmtl", str(args.e11_per_task)))
    external_comparisons.sort(
        key=lambda row: (str(row["baseline_model"]), int(row["seed"]), -float(row["delta_auroc"]))
    )

    base.write_csv(args.per_task_output, base.METRIC_COLUMNS, result_rows)
    base.write_csv(args.summary_output, base.SUMMARY_COLUMNS, summary_rows)
    base.write_csv(args.stability_output, base.STABILITY_COLUMNS, stability_rows)
    base.write_csv(args.comparison_output, COMPARISON_COLUMNS, comparison_rows)
    base.write_csv(args.external_comparison_output, COMPARISON_COLUMNS, external_comparisons)
    base.write_csv(args.diagnostic_output, DIAGNOSTIC_COLUMNS, diagnostic_rows)

    metadata = {
        "train": str(args.train),
        "test": str(args.test),
        "e2_baseline_per_task": str(args.e2_baseline_per_task),
        "e8_per_task": str(args.e8_per_task),
        "e10_per_task": str(args.e10_per_task),
        "e11_per_task": str(args.e11_per_task),
        "per_task_output": str(args.per_task_output),
        "summary_output": str(args.summary_output),
        "stability_output": str(args.stability_output),
        "comparison_output": str(args.comparison_output),
        "external_comparison_output": str(args.external_comparison_output),
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
        "pair_batch_size": args.pair_batch_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "ranking_weight": args.ranking_weight,
        "ranking_margin": args.ranking_margin,
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
        "--e10-per-task",
        type=Path,
        default=project_path("results/tissuePMHC_mmoe/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--e11-per-task",
        type=Path,
        default=project_path("results/tissuePMHC_dbmtl/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--per-task-output",
        type=Path,
        default=project_path("results/tissuePMHC_pair_ranking/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=project_path("results/tissuePMHC_pair_ranking/summary_metrics.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=project_path("results/tissuePMHC_pair_ranking/metadata.json"),
    )
    parser.add_argument(
        "--stability-output",
        type=Path,
        default=project_path("results/tissuePMHC_pair_ranking/stability_metrics.csv"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=project_path("results/tissuePMHC_pair_ranking/comparison_metrics.csv"),
    )
    parser.add_argument(
        "--external-comparison-output",
        type=Path,
        default=project_path("results/tissuePMHC_pair_ranking/external_comparison_metrics.csv"),
    )
    parser.add_argument(
        "--diagnostic-output",
        type=Path,
        default=project_path("results/tissuePMHC_pair_ranking/ranking_diagnostics.csv"),
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=MODEL_CHOICES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_REPEAT_SEEDS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=0,
        help="If 0, use ceil(max task train pairs / pair_batch_size).",
    )
    parser.add_argument("--pair-batch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512, help="Evaluation batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    parser.add_argument("--ranking-margin", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-tasks", type=int, default=0, help="Optional smoke-test limit on the number of tasks.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
