#!/usr/bin/env python3
"""Run E13 auxiliary tissue/HLA prediction for tissuePMHC.

Roadmap role: E13 auxiliary-task experiment on the E2/E8 performance line.

E13 keeps the main task unchanged:

    tissue-HLA peptide binary classification

and adds two auxiliary tasks on the shared peptide encoder:

1. predict target tissue
2. predict HLA restriction

`auxiliary task` means an extra training objective that is not the final
evaluation target, but may help the shared encoder learn useful structure.
Here, the goal is to test whether tissue/HLA supervision can improve the
shared representation more than loss weighting or pair ranking did.
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
MODEL_CHOICES = ["e2_sample_bce", "e13_aux_tissue_hla"]
METRICS = ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]

DIAGNOSTIC_COLUMNS = [
    "experiment_name",
    "seed",
    "model",
    "epoch",
    "mean_total_loss",
    "mean_bce_loss",
    "mean_tissue_loss",
    "mean_hla_loss",
    "mean_tissue_accuracy",
    "mean_hla_accuracy",
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


def define_aux_model(
    args: argparse.Namespace,
    torch: Any,
    nn: Any,
    peptide_length: int,
    n_tasks: int,
    n_tissues: int,
    n_hlas: int,
) -> Any:
    class AuxiliaryTaskModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(len(base.AA_TO_INDEX) + 1, args.embedding_dim, padding_idx=base.PAD_INDEX)
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(peptide_length * args.embedding_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.ReLU(),
            )
            self.heads = nn.ModuleList([nn.Linear(args.hidden_dim, 1) for _ in range(n_tasks)])
            self.tissue_classifier = nn.Linear(args.hidden_dim, n_tissues)
            self.hla_classifier = nn.Linear(args.hidden_dim, n_hlas)

        def encode(self, peptide_ids: Any) -> Any:
            return self.encoder(self.embedding(peptide_ids))

        def forward(self, peptide_ids: Any, task_ids: Any) -> Any:
            encoded = self.encode(peptide_ids)
            logits = encoded.new_empty(encoded.shape[0])
            for task_id in torch.unique(task_ids):
                mask = task_ids == task_id
                logits[mask] = self.heads[int(task_id.item())](encoded[mask]).squeeze(-1)
            return logits

        def auxiliary_logits(self, peptide_ids: Any) -> tuple[Any, Any]:
            encoded = self.encode(peptide_ids)
            return self.tissue_classifier(encoded), self.hla_classifier(encoded)

    return AuxiliaryTaskModel()


def build_loader(
    args: argparse.Namespace,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    df: pd.DataFrame,
    peptide_length: int,
    shuffle: bool,
) -> Any:
    x = base.encode_peptides(df["peptide_sequence"], peptide_length)
    task_ids = df["task_id"].to_numpy(dtype=np.int64).copy()
    tissue_ids = df["tissue_id"].to_numpy(dtype=np.int64).copy()
    hla_ids = df["hla_id"].to_numpy(dtype=np.int64).copy()
    labels = df["label"].to_numpy(dtype=np.int64).copy()
    return base.build_loader(
        torch,
        DataLoader,
        TensorDataset,
        [x, task_ids, tissue_ids, hla_ids, labels],
        args.batch_size,
        shuffle,
    )


def train_one_model(
    args: argparse.Namespace,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    train_df: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    device: str,
    seed: int,
    model_name: str,
) -> tuple[Any, list[dict[str, object]]]:
    model = define_aux_model(
        args,
        torch,
        nn,
        peptide_length,
        len(mappings["tasks"]),
        len(mappings["tissue_to_id"]),
        len(mappings["hla_to_id"]),
    ).to(device)
    loader = build_loader(args, torch, DataLoader, TensorDataset, train_df, peptide_length, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    use_aux = model_name == "e13_aux_tissue_hla"
    experiment_name = "E13_auxiliary_tissue_hla" if use_aux else "E2_sample_BCE"
    diagnostics: list[dict[str, object]] = []

    print(f"  train setup: epochs={args.epochs} batches_per_epoch={len(loader)}")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_losses = []
        bce_losses = []
        tissue_losses = []
        hla_losses = []
        tissue_accuracies = []
        hla_accuracies = []
        for batch in loader:
            peptide_ids, task_ids, tissue_ids, hla_ids, labels = [item.to(device) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            logits = model(peptide_ids, task_ids)
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
            tissue_logits, hla_logits = model.auxiliary_logits(peptide_ids)
            tissue_loss = torch.nn.functional.cross_entropy(tissue_logits, tissue_ids)
            hla_loss = torch.nn.functional.cross_entropy(hla_logits, hla_ids)
            if use_aux:
                loss = bce_loss + args.tissue_loss_weight * tissue_loss + args.hla_loss_weight * hla_loss
            else:
                loss = bce_loss
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_losses.append(float(loss.detach().cpu()))
            bce_losses.append(float(bce_loss.detach().cpu()))
            tissue_losses.append(float(tissue_loss.detach().cpu()))
            hla_losses.append(float(hla_loss.detach().cpu()))
            tissue_accuracies.append(float((tissue_logits.argmax(dim=1) == tissue_ids).float().mean().detach().cpu()))
            hla_accuracies.append(float((hla_logits.argmax(dim=1) == hla_ids).float().mean().detach().cpu()))

        diagnostics.append(
            {
                "experiment_name": experiment_name,
                "seed": seed,
                "model": model_name,
                "epoch": epoch,
                "mean_total_loss": float(np.mean(total_losses)),
                "mean_bce_loss": float(np.mean(bce_losses)),
                "mean_tissue_loss": float(np.mean(tissue_losses)),
                "mean_hla_loss": float(np.mean(hla_losses)),
                "mean_tissue_accuracy": float(np.mean(tissue_accuracies)),
                "mean_hla_accuracy": float(np.mean(hla_accuracies)),
            }
        )
        print(
            f"  epoch {epoch:02d}/{args.epochs} "
            f"time={time.perf_counter() - epoch_start:.2f}s "
            f"total_loss={float(np.mean(total_losses)):.4f} "
            f"bce={float(np.mean(bce_losses)):.4f} "
            f"tissue_acc={float(np.mean(tissue_accuracies)):.4f} "
            f"hla_acc={float(np.mean(hla_accuracies)):.4f}"
        )

    return model, diagnostics


def evaluate_model(
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
    model.eval()
    for task_name in mappings["tasks"]:
        train_task = train_df[train_df["task_name"] == task_name]
        test_task = test_df[test_df["task_name"] == task_name]
        loader = build_loader(args, torch, DataLoader, TensorDataset, test_task, peptide_length, False)
        scores = []
        labels = []
        with torch.no_grad():
            for batch in loader:
                peptide_ids, task_ids, _, _, y = [item.to(device) for item in batch]
                logits = model(peptide_ids, task_ids)
                scores.append(torch.sigmoid(logits).cpu().numpy())
                labels.append(y.cpu().numpy())
        y_true = np.concatenate(labels)
        y_score = np.concatenate(scores)
        rows.append(base.make_metric_row(model_name, train_task, test_task, base.evaluate(y_true, y_score)))
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
        if model != "e13_aux_tissue_hla":
            continue
        baseline = rows_by_key.get((seed, tissue, hla, "e2_sample_bce"))
        if baseline is None:
            continue
        comparison = {
            "seed": seed,
            "target_tissue": tissue,
            "mhc_restriction": hla,
            "baseline_model": "e2_sample_bce",
            "candidate_model": "e13_aux_tissue_hla",
            "baseline_source": "internal_sample_bce",
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
        if row["model"] != "e13_aux_tissue_hla":
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
            "candidate_model": "e13_aux_tissue_hla",
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
    experiment_name = "E13_auxiliary_tissue_hla" if model_name == "e13_aux_tissue_hla" else "E2_sample_BCE"
    print(f"experiment: {experiment_name} seed={seed}")
    experiment_start = time.perf_counter()
    model, diagnostics = train_one_model(
        args,
        torch,
        nn,
        DataLoader,
        TensorDataset,
        train_df,
        mappings,
        peptide_length,
        device,
        seed,
        model_name,
    )
    metric_rows = evaluate_model(
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
    return add_experiment_context(metric_rows, experiment_name, seed), diagnostics


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
    e12_external = read_external_rows(args.e12_per_task, "e12_pair_ranking")
    external_comparisons = []
    external_comparisons.extend(
        compare_external(result_rows, e2_external, "shared_peptide_encoder_task_heads", str(args.e2_baseline_per_task))
    )
    external_comparisons.extend(compare_external(result_rows, e8_external, "e8a_fixed_average", str(args.e8_per_task)))
    external_comparisons.extend(compare_external(result_rows, e10_external, "e10_mmoe", str(args.e10_per_task)))
    external_comparisons.extend(compare_external(result_rows, e12_external, "e12_pair_ranking", str(args.e12_per_task)))
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
        "e12_per_task": str(args.e12_per_task),
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
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "tissue_loss_weight": args.tissue_loss_weight,
        "hla_loss_weight": args.hla_loss_weight,
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
        "--e12-per-task",
        type=Path,
        default=project_path("results/tissuePMHC_pair_ranking/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--per-task-output",
        type=Path,
        default=project_path("results/tissuePMHC_auxiliary_tasks/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=project_path("results/tissuePMHC_auxiliary_tasks/summary_metrics.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=project_path("results/tissuePMHC_auxiliary_tasks/metadata.json"),
    )
    parser.add_argument(
        "--stability-output",
        type=Path,
        default=project_path("results/tissuePMHC_auxiliary_tasks/stability_metrics.csv"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=project_path("results/tissuePMHC_auxiliary_tasks/comparison_metrics.csv"),
    )
    parser.add_argument(
        "--external-comparison-output",
        type=Path,
        default=project_path("results/tissuePMHC_auxiliary_tasks/external_comparison_metrics.csv"),
    )
    parser.add_argument(
        "--diagnostic-output",
        type=Path,
        default=project_path("results/tissuePMHC_auxiliary_tasks/auxiliary_diagnostics.csv"),
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
    parser.add_argument("--tissue-loss-weight", type=float, default=0.1)
    parser.add_argument("--hla-loss-weight", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-tasks", type=int, default=0, help="Optional smoke-test limit on the number of tasks.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
