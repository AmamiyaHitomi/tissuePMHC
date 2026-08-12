#!/usr/bin/env python3
"""Run repeated E2 baseline and conditioned-model tuning for tissuePMHC.

Roadmap role: E1/E2/E3 baseline runner.
This file establishes E2 shared heads as the early performance main line and
also evaluates E3 tissue/HLA ID conditioning.

This is the second experiment runner. The first-version script is kept in
run_tissuepmhc_neural_baselines.py for reproducibility.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: index + 1 for index, aa in enumerate(AMINO_ACIDS)}
PAD_INDEX = 0
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPEAT_SEEDS = [20260704, 20260705, 20260706]

MODEL_CHOICES = [
    "neural_single_task",
    "shared_peptide_encoder_task_heads",
    "conditioned_tissue_hla",
]

METRIC_COLUMNS = [
    "experiment_name",
    "seed",
    "model",
    "target_tissue",
    "mhc_restriction",
    "train_rows",
    "test_rows",
    "train_positive",
    "train_negative",
    "test_positive",
    "test_negative",
    "accuracy",
    "balanced_accuracy",
    "auroc",
    "auprc",
    "f1",
    "mcc",
]

SUMMARY_COLUMNS = [
    "experiment_name",
    "seed",
    "model",
    "n_tasks",
    "mean_accuracy",
    "median_accuracy",
    "mean_balanced_accuracy",
    "median_balanced_accuracy",
    "mean_auroc",
    "median_auroc",
    "mean_auprc",
    "median_auprc",
    "mean_f1",
    "median_f1",
    "mean_mcc",
    "median_mcc",
    "weighted_mean_accuracy",
    "weighted_mean_auroc",
    "weighted_mean_auprc",
    "worst_10_mean_auroc",
    "worst_10_mean_auprc",
]

STABILITY_COLUMNS = [
    "experiment_name",
    "model",
    "n_seeds",
    "mean_auroc_mean",
    "mean_auroc_std",
    "mean_auroc_min",
    "mean_auroc_max",
    "mean_auprc_mean",
    "mean_auprc_std",
    "mean_auprc_min",
    "mean_auprc_max",
    "mean_accuracy_mean",
    "mean_accuracy_std",
    "mean_mcc_mean",
    "mean_mcc_std",
    "worst_10_mean_auroc_mean",
    "worst_10_mean_auroc_std",
]

CONDITIONED_TUNING_CONFIGS = [
    {
        "experiment_name": "E3_conditioned_default",
        "overrides": {},
    },
    {
        "experiment_name": "E3_conditioned_wider_condition",
        "overrides": {"condition_dim": 32, "hidden_dim": 128, "dropout": 0.2, "learning_rate": 1e-3},
    },
    {
        "experiment_name": "E3_conditioned_wider_hidden",
        "overrides": {"condition_dim": 32, "hidden_dim": 256, "dropout": 0.2, "learning_rate": 5e-4},
    },
    {
        "experiment_name": "E3_conditioned_low_dropout",
        "overrides": {"condition_dim": 32, "hidden_dim": 128, "dropout": 0.1, "learning_rate": 1e-3},
    },
]


def require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for neural baselines, but it is not installed in this Python environment. "
            "Install torch first, then rerun this script."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def open_text_input(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find input file: {path}. "
            f"The project root inferred from this script is: {PROJECT_ROOT}"
        )
    with open_text_input(path) as f:
        df = pd.read_csv(f)
    df["label"] = df["label"].astype(np.int64)
    return df


def encode_peptides(peptides: pd.Series, peptide_length: int | None = None) -> np.ndarray:
    if peptide_length is None:
        peptide_length = int(peptides.str.len().max())
    encoded = np.full((len(peptides), peptide_length), PAD_INDEX, dtype=np.int64)
    for row_index, peptide in enumerate(peptides):
        if len(peptide) > peptide_length:
            raise ValueError(f"Peptide is longer than peptide_length={peptide_length}: {peptide}")
        for position, aa in enumerate(peptide):
            encoded[row_index, position] = AA_TO_INDEX[aa]
    return encoded


def add_task_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["task_name"] = train_df["target_tissue"] + "||" + train_df["mhc_restriction"]
    test_df["task_name"] = test_df["target_tissue"] + "||" + test_df["mhc_restriction"]
    tasks = sorted(set(train_df["task_name"]) & set(test_df["task_name"]))
    tissues = sorted(set(train_df["target_tissue"]) | set(test_df["target_tissue"]))
    hlas = sorted(set(train_df["mhc_restriction"]) | set(test_df["mhc_restriction"]))

    task_to_id = {task: index for index, task in enumerate(tasks)}
    tissue_to_id = {tissue: index for index, tissue in enumerate(tissues)}
    hla_to_id = {hla: index for index, hla in enumerate(hlas)}

    train_df = train_df[train_df["task_name"].isin(task_to_id)].copy()
    test_df = test_df[test_df["task_name"].isin(task_to_id)].copy()
    for df in [train_df, test_df]:
        df["task_id"] = df["task_name"].map(task_to_id).astype(np.int64)
        df["tissue_id"] = df["target_tissue"].map(tissue_to_id).astype(np.int64)
        df["hla_id"] = df["mhc_restriction"].map(hla_to_id).astype(np.int64)

    mappings = {
        "tasks": tasks,
        "task_to_id": task_to_id,
        "tissue_to_id": tissue_to_id,
        "hla_to_id": hla_to_id,
    }
    return train_df, test_df, mappings


def set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(torch: Any, DataLoader: Any, TensorDataset: Any, arrays: list[np.ndarray], batch_size: int, shuffle: bool):
    tensors = [torch.as_tensor(array) for array in arrays]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def define_models(nn: Any):
    class PeptideEncoder(nn.Module):
        def __init__(self, peptide_length: int, embedding_dim: int, hidden_dim: int, dropout: float):
            super().__init__()
            self.embedding = nn.Embedding(len(AA_TO_INDEX) + 1, embedding_dim, padding_idx=PAD_INDEX)
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(peptide_length * embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        def forward(self, peptide_ids):
            return self.network(self.embedding(peptide_ids))

    class NeuralSingleTaskModel(nn.Module):
        def __init__(self, peptide_length: int, embedding_dim: int, hidden_dim: int, dropout: float):
            super().__init__()
            self.encoder = PeptideEncoder(peptide_length, embedding_dim, hidden_dim, dropout)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, peptide_ids):
            return self.head(self.encoder(peptide_ids)).squeeze(-1)

    class SharedTaskHeadsModel(nn.Module):
        def __init__(
            self,
            peptide_length: int,
            n_tasks: int,
            embedding_dim: int,
            hidden_dim: int,
            dropout: float,
        ):
            super().__init__()
            self.encoder = PeptideEncoder(peptide_length, embedding_dim, hidden_dim, dropout)
            self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_tasks)])

        def forward(self, peptide_ids, task_ids):
            encoded = self.encoder(peptide_ids)
            logits = encoded.new_empty(encoded.shape[0])
            for task_id in torch_unique(task_ids):
                mask = task_ids == task_id
                logits[mask] = self.heads[int(task_id.item())](encoded[mask]).squeeze(-1)
            return logits

    class ConditionedTissueHlaModel(nn.Module):
        def __init__(
            self,
            peptide_length: int,
            n_tissues: int,
            n_hlas: int,
            embedding_dim: int,
            hidden_dim: int,
            condition_dim: int,
            dropout: float,
        ):
            super().__init__()
            self.encoder = PeptideEncoder(peptide_length, embedding_dim, hidden_dim, dropout)
            self.tissue_embedding = nn.Embedding(n_tissues, condition_dim)
            self.hla_embedding = nn.Embedding(n_hlas, condition_dim)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim + 2 * condition_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, peptide_ids, tissue_ids, hla_ids):
            features = [
                self.encoder(peptide_ids),
                self.tissue_embedding(tissue_ids),
                self.hla_embedding(hla_ids),
            ]
            return self.classifier(torch_cat(features, dim=1)).squeeze(-1)

    return NeuralSingleTaskModel, SharedTaskHeadsModel, ConditionedTissueHlaModel


def torch_unique(values):
    import torch

    return torch.unique(values)


def torch_cat(values, dim: int):
    import torch

    return torch.cat(values, dim=dim)


def train_binary_model(
    torch: Any,
    model: Any,
    loader: Any,
    optimizer: Any,
    loss_fn: Any,
    device: str,
    model_kind: str,
    epochs: int,
    run_label: str = "unspecified",
) -> None:
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        losses: list[float] = []
        for batch in loader:
            batch = [item.to(device) for item in batch]
            optimizer.zero_grad()
            if model_kind == "single":
                logits = model(batch[0])
                labels = batch[1].float()
            elif model_kind == "task_heads":
                logits = model(batch[0], batch[1])
                labels = batch[2].float()
            elif model_kind == "conditioned":
                logits = model(batch[0], batch[1], batch[2])
                labels = batch[3].float()
            else:
                raise ValueError(f"Unknown model_kind: {model_kind}")
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(
            f"  epoch {epoch:02d}/{epochs} run={run_label} "
            f"time={time.perf_counter() - epoch_start:.2f}s "
            f"mean_loss={float(np.mean(losses)):.4f}",
            flush=True,
        )


def predict_scores(torch: Any, model: Any, loader: Any, device: str, model_kind: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = [item.to(device) for item in batch]
            if model_kind == "single":
                logits = model(batch[0])
                y = batch[1]
            elif model_kind == "task_heads":
                logits = model(batch[0], batch[1])
                y = batch[2]
            elif model_kind == "conditioned":
                logits = model(batch[0], batch[1], batch[2])
                y = batch[3]
            else:
                raise ValueError(f"Unknown model_kind: {model_kind}")
            scores.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.cpu().numpy())
    return np.concatenate(labels), np.concatenate(scores)


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_score),
        "auprc": average_precision_score(y_true, y_score),
        "f1": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def count_labels(rows: pd.DataFrame) -> tuple[int, int]:
    positive = int((rows["label"] == 1).sum())
    return positive, int(len(rows) - positive)


def make_metric_row(model_name: str, train_rows: pd.DataFrame, test_rows: pd.DataFrame, metrics: dict[str, float]):
    train_positive, train_negative = count_labels(train_rows)
    test_positive, test_negative = count_labels(test_rows)
    first = test_rows.iloc[0]
    return {
        "model": model_name,
        "target_tissue": first["target_tissue"],
        "mhc_restriction": first["mhc_restriction"],
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_positive": train_positive,
        "train_negative": train_negative,
        "test_positive": test_positive,
        "test_negative": test_negative,
        **metrics,
    }


def add_experiment_context(
    rows: list[dict[str, object]], experiment_name: str, seed: int
) -> list[dict[str, object]]:
    for row in rows:
        row["experiment_name"] = experiment_name
        row["seed"] = seed
    return rows


def summarize_results(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows_by_model: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_model[(str(row["experiment_name"]), int(row["seed"]), str(row["model"]))].append(row)

    summary_rows = []
    for (experiment_name, seed, model), model_rows in rows_by_model.items():
        weights = np.asarray([float(row["test_rows"]) for row in model_rows], dtype=np.float64)
        summary = {
            "experiment_name": experiment_name,
            "seed": seed,
            "model": model,
            "n_tasks": len(model_rows),
        }
        for metric in ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]:
            values = np.asarray([float(row[metric]) for row in model_rows], dtype=np.float64)
            summary[f"mean_{metric}"] = float(np.mean(values))
            summary[f"median_{metric}"] = float(np.median(values))
        for metric in ["accuracy", "auroc", "auprc"]:
            values = np.asarray([float(row[metric]) for row in model_rows], dtype=np.float64)
            summary[f"weighted_mean_{metric}"] = float(np.average(values, weights=weights))
        aurocs = np.sort(np.asarray([float(row["auroc"]) for row in model_rows], dtype=np.float64))
        auprcs = np.sort(np.asarray([float(row["auprc"]) for row in model_rows], dtype=np.float64))
        summary["worst_10_mean_auroc"] = float(np.mean(aurocs[: min(10, len(aurocs))]))
        summary["worst_10_mean_auprc"] = float(np.mean(auprcs[: min(10, len(auprcs))]))
        summary_rows.append(summary)

    summary_rows.sort(
        key=lambda row: (str(row["experiment_name"]), int(row["seed"]), -float(row["mean_auroc"]))
    )
    return summary_rows


def summarize_seed_stability(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows_by_experiment: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        rows_by_experiment[(str(row["experiment_name"]), str(row["model"]))].append(row)

    stability_rows = []
    for (experiment_name, model), rows in rows_by_experiment.items():
        stability = {
            "experiment_name": experiment_name,
            "model": model,
            "n_seeds": len(rows),
        }
        for metric in ["mean_auroc", "mean_auprc"]:
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            stability[f"{metric}_mean"] = float(np.mean(values))
            stability[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            stability[f"{metric}_min"] = float(np.min(values))
            stability[f"{metric}_max"] = float(np.max(values))
        for metric in ["mean_accuracy", "mean_mcc", "worst_10_mean_auroc"]:
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            stability[f"{metric}_mean"] = float(np.mean(values))
            stability[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        stability_rows.append(stability)

    stability_rows.sort(key=lambda row: float(row["mean_auroc_mean"]), reverse=True)
    return stability_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_single_task_models(
    args: argparse.Namespace,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    peptide_length: int,
    device: str,
) -> list[dict[str, object]]:
    NeuralSingleTaskModel, _, _ = define_models(nn)
    rows = []
    tasks = sorted(set(train_df["task_name"]) & set(test_df["task_name"]))
    for task_index, task_name in enumerate(tasks, start=1):
        train_task = train_df[train_df["task_name"] == task_name]
        test_task = test_df[test_df["task_name"] == task_name]
        x_train = encode_peptides(train_task["peptide_sequence"], peptide_length)
        y_train = train_task["label"].to_numpy(dtype=np.int64)
        x_test = encode_peptides(test_task["peptide_sequence"], peptide_length)
        y_test = test_task["label"].to_numpy(dtype=np.int64)

        loader = build_loader(torch, DataLoader, TensorDataset, [x_train, y_train], args.batch_size, True)
        test_loader = build_loader(torch, DataLoader, TensorDataset, [x_test, y_test], args.batch_size, False)
        model = NeuralSingleTaskModel(peptide_length, args.embedding_dim, args.hidden_dim, args.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        train_binary_model(
            torch, model, loader, optimizer, loss_fn, device, "single",
            args.single_task_epochs, f"seed={args.seed} model=neural_single_task task={task_name}",
        )
        y_true, y_score = predict_scores(torch, model, test_loader, device, "single")
        metrics = evaluate(y_true, y_score)
        rows.append(make_metric_row("neural_single_task", train_task, test_task, metrics))
        print(f"  {task_index:02d}/{len(tasks)} {task_name.replace('||', ' ')} auroc={metrics['auroc']:.4f}")
    return rows


def run_shared_task_heads(
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
) -> list[dict[str, object]]:
    _, SharedTaskHeadsModel, _ = define_models(nn)
    x_train = encode_peptides(train_df["peptide_sequence"], peptide_length)
    task_train = train_df["task_id"].to_numpy(dtype=np.int64)
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    loader = build_loader(torch, DataLoader, TensorDataset, [x_train, task_train, y_train], args.batch_size, True)

    model = SharedTaskHeadsModel(
        peptide_length,
        len(mappings["tasks"]),
        args.embedding_dim,
        args.hidden_dim,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    train_binary_model(
        torch, model, loader, optimizer, loss_fn, device, "task_heads",
        args.epochs, f"seed={args.seed} model=shared_peptide_encoder_task_heads",
    )

    rows = []
    for task_name in mappings["tasks"]:
        train_task = train_df[train_df["task_name"] == task_name]
        test_task = test_df[test_df["task_name"] == task_name]
        x_test = encode_peptides(test_task["peptide_sequence"], peptide_length)
        task_test = test_task["task_id"].to_numpy(dtype=np.int64)
        y_test = test_task["label"].to_numpy(dtype=np.int64)
        test_loader = build_loader(torch, DataLoader, TensorDataset, [x_test, task_test, y_test], args.batch_size, False)
        y_true, y_score = predict_scores(torch, model, test_loader, device, "task_heads")
        metrics = evaluate(y_true, y_score)
        rows.append(make_metric_row("shared_peptide_encoder_task_heads", train_task, test_task, metrics))
    return rows


def run_conditioned_model(
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
) -> list[dict[str, object]]:
    _, _, ConditionedTissueHlaModel = define_models(nn)
    x_train = encode_peptides(train_df["peptide_sequence"], peptide_length)
    tissue_train = train_df["tissue_id"].to_numpy(dtype=np.int64)
    hla_train = train_df["hla_id"].to_numpy(dtype=np.int64)
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    loader = build_loader(
        torch,
        DataLoader,
        TensorDataset,
        [x_train, tissue_train, hla_train, y_train],
        args.batch_size,
        True,
    )

    model = ConditionedTissueHlaModel(
        peptide_length,
        len(mappings["tissue_to_id"]),
        len(mappings["hla_to_id"]),
        args.embedding_dim,
        args.hidden_dim,
        args.condition_dim,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    train_binary_model(
        torch, model, loader, optimizer, loss_fn, device, "conditioned",
        args.epochs, f"seed={args.seed} model=conditioned_tissue_hla",
    )

    rows = []
    for task_name in mappings["tasks"]:
        train_task = train_df[train_df["task_name"] == task_name]
        test_task = test_df[test_df["task_name"] == task_name]
        x_test = encode_peptides(test_task["peptide_sequence"], peptide_length)
        tissue_test = test_task["tissue_id"].to_numpy(dtype=np.int64)
        hla_test = test_task["hla_id"].to_numpy(dtype=np.int64)
        y_test = test_task["label"].to_numpy(dtype=np.int64)
        test_loader = build_loader(
            torch,
            DataLoader,
            TensorDataset,
            [x_test, tissue_test, hla_test, y_test],
            args.batch_size,
            False,
        )
        y_true, y_score = predict_scores(torch, model, test_loader, device, "conditioned")
        metrics = evaluate(y_true, y_score)
        rows.append(make_metric_row("conditioned_tissue_hla", train_task, test_task, metrics))
    return rows


def clone_args_with_overrides(args: argparse.Namespace, seed: int, overrides: dict[str, object]) -> argparse.Namespace:
    experiment_args = copy.copy(args)
    experiment_args.seed = seed
    for key, value in overrides.items():
        setattr(experiment_args, key, value)
    return experiment_args


def build_experiment_plan(args: argparse.Namespace) -> list[dict[str, object]]:
    experiments: list[dict[str, object]] = []
    if args.experiment_plan == "e2_conditioned_tuning":
        for seed in args.seeds:
            experiments.append(
                {
                    "experiment_name": "E2_shared_peptide_encoder_task_heads",
                    "model": "shared_peptide_encoder_task_heads",
                    "seed": seed,
                    "overrides": {},
                }
            )
            for config in CONDITIONED_TUNING_CONFIGS:
                experiments.append(
                    {
                        "experiment_name": config["experiment_name"],
                        "model": "conditioned_tissue_hla",
                        "seed": seed,
                        "overrides": config["overrides"],
                    }
                )
        return experiments

    for seed in args.seeds:
        for model_name in args.models:
            experiments.append(
                {
                    "experiment_name": model_name,
                    "model": model_name,
                    "seed": seed,
                    "overrides": {},
                }
            )
    return experiments


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
    experiment: dict[str, object],
) -> list[dict[str, object]]:
    experiment_name = str(experiment["experiment_name"])
    model_name = str(experiment["model"])
    seed = int(experiment["seed"])
    overrides = dict(experiment["overrides"])
    experiment_args = clone_args_with_overrides(args, seed, overrides)
    set_seed(seed, torch)

    override_text = ", ".join(f"{key}={value}" for key, value in sorted(overrides.items())) or "default"
    print(f"experiment: {experiment_name} seed={seed} model={model_name} config={override_text}")

    if model_name == "neural_single_task":
        rows = run_single_task_models(
            experiment_args, torch, nn, DataLoader, TensorDataset, train_df, test_df, peptide_length, device
        )
    elif model_name == "shared_peptide_encoder_task_heads":
        rows = run_shared_task_heads(
            experiment_args,
            torch,
            nn,
            DataLoader,
            TensorDataset,
            train_df,
            test_df,
            mappings,
            peptide_length,
            device,
        )
    elif model_name == "conditioned_tissue_hla":
        rows = run_conditioned_model(
            experiment_args,
            torch,
            nn,
            DataLoader,
            TensorDataset,
            train_df,
            test_df,
            mappings,
            peptide_length,
            device,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return add_experiment_context(rows, experiment_name, seed)


def run(args: argparse.Namespace) -> None:
    run_start = time.perf_counter()
    torch, nn, DataLoader, TensorDataset = require_torch()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = read_dataset(args.train)
    test_df = read_dataset(args.test)
    train_df, test_df, mappings = add_task_columns(train_df, test_df)
    if args.max_tasks:
        keep_tasks = set(mappings["tasks"][: args.max_tasks])
        train_df = train_df[train_df["task_name"].isin(keep_tasks)].copy()
        test_df = test_df[test_df["task_name"].isin(keep_tasks)].copy()
        train_df, test_df, mappings = add_task_columns(train_df, test_df)

    peptide_length = int(max(train_df["peptide_sequence"].str.len().max(), test_df["peptide_sequence"].str.len().max()))
    result_rows: list[dict[str, object]] = []
    experiments = build_experiment_plan(args)
    print(f"device: {device}")
    print(f"n_experiments: {len(experiments)}")
    for experiment_index, experiment in enumerate(experiments, start=1):
        print(f"[{experiment_index}/{len(experiments)}]")
        result_rows.extend(
            run_one_experiment(
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
                experiment,
            )
        )

    summary_rows = summarize_results(result_rows)
    stability_rows = summarize_seed_stability(summary_rows)
    write_csv(args.per_task_output, METRIC_COLUMNS, result_rows)
    write_csv(args.summary_output, SUMMARY_COLUMNS, summary_rows)
    write_csv(args.stability_output, STABILITY_COLUMNS, stability_rows)

    metadata = {
        "train": str(args.train),
        "test": str(args.test),
        "per_task_output": str(args.per_task_output),
        "summary_output": str(args.summary_output),
        "n_tasks": len(mappings["tasks"]),
        "models": args.models,
        "experiment_plan": args.experiment_plan,
        "seeds": args.seeds,
        "device": device,
        "peptide_length": peptide_length,
        "amino_acid_encoding": "integer amino-acid IDs passed through a trainable embedding layer",
        "embedding_dim": args.embedding_dim,
        "condition_dim": args.condition_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "single_task_epochs": args.single_task_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "conditioned_tuning_configs": CONDITIONED_TUNING_CONFIGS,
        "task_mapping": mappings["task_to_id"],
        "tissue_mapping": mappings["tissue_to_id"],
        "hla_mapping": mappings["hla_to_id"],
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote: {args.per_task_output}")
    print(f"wrote: {args.summary_output}")
    print(f"wrote: {args.stability_output}")
    print(f"wrote: {args.metadata_output}")
    print(f"run total time: {time.perf_counter() - run_start:.2f}s", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz"))
    parser.add_argument("--test", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_test.csv.gz"))
    parser.add_argument(
        "--per-task-output",
        type=Path,
        default=project_path("results/tissuePMHC_neural_baselines_v2/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=project_path("results/tissuePMHC_neural_baselines_v2/summary_metrics.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=project_path("results/tissuePMHC_neural_baselines_v2/metadata.json"),
    )
    parser.add_argument(
        "--stability-output",
        type=Path,
        default=project_path("results/tissuePMHC_neural_baselines_v2/stability_metrics.csv"),
    )
    parser.add_argument(
        "--experiment-plan",
        choices=["e2_conditioned_tuning", "custom"],
        default="e2_conditioned_tuning",
        help="Default plan runs E2 plus conditioned-model tuning over repeated seeds.",
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=MODEL_CHOICES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_REPEAT_SEEDS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--single-task-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--condition-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-tasks", type=int, default=0, help="Optional smoke-test limit on the number of tasks.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
