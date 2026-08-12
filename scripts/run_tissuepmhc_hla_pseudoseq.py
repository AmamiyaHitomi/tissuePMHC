#!/usr/bin/env python3
"""Run E4 HLA pseudo-sequence conditioning experiments for tissuePMHC.

Roadmap role: E4 biological-representation line.
This tests whether explicit HLA pseudo-sequence information improves over HLA
ID embeddings; it is not the current performance main line because E4 did not
outperform E2/E8 on the standard split.

E4 compares:

1. conditioned model with trainable HLA ID embedding
2. conditioned model with an HLA pseudo-sequence encoder

The pseudo-sequence table is expected as a CSV with columns:

hla,pseudo_sequence
"""

from __future__ import annotations

import argparse
import copy
import csv
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

MODEL_CHOICES = [
    "conditioned_hla_embedding",
    "conditioned_hla_pseudoseq",
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


def require_pseudo_sequences(path: Path, hlas: list[str]) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find HLA pseudo-sequence table: {path}. "
            "Run this script with --write-pseudo-template to create a fill-in template."
        )

    table = pd.read_csv(path)
    required_columns = {"hla", "pseudo_sequence"}
    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")

    pseudo_by_hla: dict[str, str] = {}
    for row in table.itertuples(index=False):
        hla = str(getattr(row, "hla")).strip()
        pseudo_sequence = str(getattr(row, "pseudo_sequence")).strip().upper()
        if hla and pseudo_sequence and pseudo_sequence.lower() != "nan":
            unknown = sorted(set(pseudo_sequence) - set(base.AMINO_ACIDS))
            if unknown:
                raise ValueError(f"{hla} has unsupported amino-acid letters in pseudo_sequence: {unknown}")
            pseudo_by_hla[hla] = pseudo_sequence

    missing_hlas = [hla for hla in hlas if hla not in pseudo_by_hla]
    if missing_hlas:
        raise ValueError(
            "Pseudo-sequence table does not cover all HLA alleles in this dataset: "
            + ", ".join(missing_hlas)
        )

    lengths = {len(pseudo_by_hla[hla]) for hla in hlas}
    if len(lengths) != 1:
        raise ValueError(
            "All pseudo-sequences must have the same length for this encoder. "
            f"Observed lengths: {sorted(lengths)}"
        )
    return pseudo_by_hla


def write_pseudo_template(path: Path, hlas: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["hla", "pseudo_sequence", "source"])
        writer.writeheader()
        for hla in hlas:
            writer.writerow({"hla": hla, "pseudo_sequence": "", "source": ""})


def encode_hla_pseudosequences(
    hla_ids: np.ndarray,
    id_to_hla: dict[int, str],
    pseudo_by_hla: dict[str, str],
    pseudo_length: int,
) -> np.ndarray:
    encoded = np.full((len(hla_ids), pseudo_length), base.PAD_INDEX, dtype=np.int64)
    for row_index, hla_id in enumerate(hla_ids):
        pseudo_sequence = pseudo_by_hla[id_to_hla[int(hla_id)]]
        for position, aa in enumerate(pseudo_sequence):
            encoded[row_index, position] = base.AA_TO_INDEX[aa]
    return encoded


def define_pseudoseq_model(nn: Any):
    _, _, _ = base.define_models(nn)

    class SequenceEncoder(nn.Module):
        def __init__(self, sequence_length: int, embedding_dim: int, hidden_dim: int, dropout: float):
            super().__init__()
            self.embedding = nn.Embedding(len(base.AA_TO_INDEX) + 1, embedding_dim, padding_idx=base.PAD_INDEX)
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(sequence_length * embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        def forward(self, sequence_ids):
            return self.network(self.embedding(sequence_ids))

    class ConditionedHlaPseudoSeqModel(nn.Module):
        def __init__(
            self,
            peptide_length: int,
            pseudo_length: int,
            n_tissues: int,
            peptide_embedding_dim: int,
            hla_embedding_dim: int,
            peptide_hidden_dim: int,
            hla_hidden_dim: int,
            tissue_dim: int,
            classifier_hidden_dim: int,
            dropout: float,
        ):
            super().__init__()
            self.peptide_encoder = SequenceEncoder(peptide_length, peptide_embedding_dim, peptide_hidden_dim, dropout)
            self.hla_encoder = SequenceEncoder(pseudo_length, hla_embedding_dim, hla_hidden_dim, dropout)
            self.tissue_embedding = nn.Embedding(n_tissues, tissue_dim)
            self.classifier = nn.Sequential(
                nn.Linear(peptide_hidden_dim + hla_hidden_dim + tissue_dim, classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, 1),
            )

        def forward(self, peptide_ids, tissue_ids, hla_pseudo_ids):
            features = [
                self.peptide_encoder(peptide_ids),
                self.tissue_embedding(tissue_ids),
                self.hla_encoder(hla_pseudo_ids),
            ]
            return self.classifier(base.torch_cat(features, dim=1)).squeeze(-1)

    return ConditionedHlaPseudoSeqModel


def train_pseudoseq_model(
    torch: Any,
    model: Any,
    loader: Any,
    optimizer: Any,
    loss_fn: Any,
    device: str,
    epochs: int,
    run_label: str = "unspecified",
) -> None:
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        losses: list[float] = []
        for batch in loader:
            peptide_ids, tissue_ids, hla_pseudo_ids, labels = [item.to(device) for item in batch]
            optimizer.zero_grad()
            logits = model(peptide_ids, tissue_ids, hla_pseudo_ids)
            loss = loss_fn(logits, labels.float())
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(
            f"  epoch {epoch:02d}/{epochs} run={run_label} "
            f"time={time.perf_counter() - epoch_start:.2f}s "
            f"mean_loss={float(np.mean(losses)):.4f}",
            flush=True,
        )


def predict_pseudoseq_scores(torch: Any, model: Any, loader: Any, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            peptide_ids, tissue_ids, hla_pseudo_ids, y = [item.to(device) for item in batch]
            logits = model(peptide_ids, tissue_ids, hla_pseudo_ids)
            scores.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.cpu().numpy())
    return np.concatenate(labels), np.concatenate(scores)


def run_conditioned_hla_pseudoseq(
    args: argparse.Namespace,
    torch: Any,
    nn: Any,
    DataLoader: Any,
    TensorDataset: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    pseudo_by_hla: dict[str, str],
    device: str,
) -> list[dict[str, object]]:
    ConditionedHlaPseudoSeqModel = define_pseudoseq_model(nn)
    id_to_hla = {hla_id: hla for hla, hla_id in mappings["hla_to_id"].items()}
    pseudo_length = len(next(iter(pseudo_by_hla.values())))

    x_train = base.encode_peptides(train_df["peptide_sequence"], peptide_length)
    tissue_train = train_df["tissue_id"].to_numpy(dtype=np.int64)
    hla_train = encode_hla_pseudosequences(
        train_df["hla_id"].to_numpy(dtype=np.int64),
        id_to_hla,
        pseudo_by_hla,
        pseudo_length,
    )
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    loader = base.build_loader(
        torch,
        DataLoader,
        TensorDataset,
        [x_train, tissue_train, hla_train, y_train],
        args.batch_size,
        True,
    )

    model = ConditionedHlaPseudoSeqModel(
        peptide_length,
        pseudo_length,
        len(mappings["tissue_to_id"]),
        args.embedding_dim,
        args.hla_embedding_dim,
        args.hidden_dim,
        args.hla_hidden_dim,
        args.condition_dim,
        args.classifier_hidden_dim,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    train_pseudoseq_model(
        torch, model, loader, optimizer, loss_fn, device, args.epochs,
        f"seed={args.seed} model=conditioned_hla_pseudoseq",
    )

    rows = []
    for task_name in mappings["tasks"]:
        train_task = train_df[train_df["task_name"] == task_name]
        test_task = test_df[test_df["task_name"] == task_name]
        x_test = base.encode_peptides(test_task["peptide_sequence"], peptide_length)
        tissue_test = test_task["tissue_id"].to_numpy(dtype=np.int64)
        hla_test = encode_hla_pseudosequences(
            test_task["hla_id"].to_numpy(dtype=np.int64),
            id_to_hla,
            pseudo_by_hla,
            pseudo_length,
        )
        y_test = test_task["label"].to_numpy(dtype=np.int64)
        test_loader = base.build_loader(
            torch,
            DataLoader,
            TensorDataset,
            [x_test, tissue_test, hla_test, y_test],
            args.batch_size,
            False,
        )
        y_true, y_score = predict_pseudoseq_scores(torch, model, test_loader, device)
        metrics = base.evaluate(y_true, y_score)
        rows.append(base.make_metric_row("conditioned_hla_pseudoseq", train_task, test_task, metrics))
    return rows


def clone_args_with_best_hla_embedding_config(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    experiment_args = copy.copy(args)
    experiment_args.seed = seed
    experiment_args.condition_dim = args.hla_embedding_condition_dim
    experiment_args.hidden_dim = args.hla_embedding_hidden_dim
    experiment_args.dropout = args.hla_embedding_dropout
    experiment_args.learning_rate = args.hla_embedding_learning_rate
    return experiment_args


def add_experiment_context(rows: list[dict[str, object]], experiment_name: str, seed: int) -> list[dict[str, object]]:
    for row in rows:
        row["experiment_name"] = experiment_name
        row["seed"] = seed
    return rows


def compare_pseudoseq_against_embedding(rows: list[dict[str, object]]) -> list[dict[str, object]]:
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
        if model != "conditioned_hla_pseudoseq":
            continue
        baseline = rows_by_key.get((seed, target_tissue, mhc_restriction, "conditioned_hla_embedding"))
        if baseline is None:
            continue
        comparison = {
            "seed": seed,
            "target_tissue": target_tissue,
            "mhc_restriction": mhc_restriction,
            "baseline_model": "conditioned_hla_embedding",
            "candidate_model": "conditioned_hla_pseudoseq",
        }
        for metric in ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]:
            comparison[f"delta_{metric}"] = float(candidate[metric]) - float(baseline[metric])
        comparison_rows.append(comparison)

    comparison_rows.sort(key=lambda row: (int(row["seed"]), -float(row["delta_auroc"])))
    return comparison_rows


def run(args: argparse.Namespace) -> None:
    run_start = time.perf_counter()
    train_df = base.read_dataset(args.train)
    test_df = base.read_dataset(args.test)
    train_df, test_df, mappings = base.add_task_columns(train_df, test_df)
    if args.max_tasks:
        keep_tasks = set(mappings["tasks"][: args.max_tasks])
        train_df = train_df[train_df["task_name"].isin(keep_tasks)].copy()
        test_df = test_df[test_df["task_name"].isin(keep_tasks)].copy()
        train_df, test_df, mappings = base.add_task_columns(train_df, test_df)

    hlas = sorted(mappings["hla_to_id"])
    if args.write_pseudo_template:
        write_pseudo_template(args.pseudo_sequences, hlas)
        print(f"wrote pseudo-sequence template: {args.pseudo_sequences}")
        return

    pseudo_by_hla = require_pseudo_sequences(args.pseudo_sequences, hlas)
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    peptide_length = int(max(train_df["peptide_sequence"].str.len().max(), test_df["peptide_sequence"].str.len().max()))

    result_rows: list[dict[str, object]] = []
    print(f"device: {device}")
    print(f"n_tasks: {len(mappings['tasks'])}")
    print(f"n_hlas: {len(hlas)}")
    print(f"pseudo_length: {len(next(iter(pseudo_by_hla.values())))}")

    for seed in args.seeds:
        args.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        base.set_seed(seed, torch)

        if "conditioned_hla_embedding" in args.models:
            print(f"experiment: E3_conditioned_hla_embedding_best seed={seed}")
            embedding_args = clone_args_with_best_hla_embedding_config(args, seed)
            rows = base.run_conditioned_model(
                embedding_args,
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
            for row in rows:
                row["model"] = "conditioned_hla_embedding"
            result_rows.extend(add_experiment_context(rows, "E3_conditioned_hla_embedding_best", seed))

        if "conditioned_hla_pseudoseq" in args.models:
            print(f"experiment: E4_conditioned_hla_pseudoseq seed={seed}")
            base.set_seed(seed, torch)
            rows = run_conditioned_hla_pseudoseq(
                args,
                torch,
                nn,
                DataLoader,
                TensorDataset,
                train_df,
                test_df,
                mappings,
                peptide_length,
                pseudo_by_hla,
                device,
            )
            result_rows.extend(add_experiment_context(rows, "E4_conditioned_hla_pseudoseq", seed))

    summary_rows = base.summarize_results(result_rows)
    stability_rows = base.summarize_seed_stability(summary_rows)
    comparison_rows = compare_pseudoseq_against_embedding(result_rows)
    base.write_csv(args.per_task_output, base.METRIC_COLUMNS, result_rows)
    base.write_csv(args.summary_output, base.SUMMARY_COLUMNS, summary_rows)
    base.write_csv(args.stability_output, base.STABILITY_COLUMNS, stability_rows)
    base.write_csv(args.comparison_output, COMPARISON_COLUMNS, comparison_rows)

    metadata = {
        "train": str(args.train),
        "test": str(args.test),
        "pseudo_sequences": str(args.pseudo_sequences),
        "per_task_output": str(args.per_task_output),
        "summary_output": str(args.summary_output),
        "stability_output": str(args.stability_output),
        "comparison_output": str(args.comparison_output),
        "n_tasks": len(mappings["tasks"]),
        "n_hlas": len(hlas),
        "models": args.models,
        "seeds": args.seeds,
        "device": device,
        "peptide_length": peptide_length,
        "pseudo_length": len(next(iter(pseudo_by_hla.values()))),
        "amino_acid_encoding": "integer amino-acid IDs passed through trainable embedding layers",
        "embedding_dim": args.embedding_dim,
        "hla_embedding_dim": args.hla_embedding_dim,
        "condition_dim": args.condition_dim,
        "hidden_dim": args.hidden_dim,
        "hla_hidden_dim": args.hla_hidden_dim,
        "classifier_hidden_dim": args.classifier_hidden_dim,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hla_embedding_baseline_config": {
            "condition_dim": args.hla_embedding_condition_dim,
            "hidden_dim": args.hla_embedding_hidden_dim,
            "dropout": args.hla_embedding_dropout,
            "learning_rate": args.hla_embedding_learning_rate,
        },
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
    print(f"wrote: {args.metadata_output}")
    print(f"run total time: {time.perf_counter() - run_start:.2f}s", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz"))
    parser.add_argument("--test", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_test.csv.gz"))
    parser.add_argument(
        "--pseudo-sequences",
        type=Path,
        default=project_path("data/processed/hla_pseudo_sequences.csv"),
    )
    parser.add_argument(
        "--per-task-output",
        type=Path,
        default=project_path("results/tissuePMHC_hla_pseudoseq/per_task_metrics.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=project_path("results/tissuePMHC_hla_pseudoseq/summary_metrics.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=project_path("results/tissuePMHC_hla_pseudoseq/metadata.json"),
    )
    parser.add_argument(
        "--stability-output",
        type=Path,
        default=project_path("results/tissuePMHC_hla_pseudoseq/stability_metrics.csv"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=project_path("results/tissuePMHC_hla_pseudoseq/comparison_metrics.csv"),
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=MODEL_CHOICES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_REPEAT_SEEDS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hla-embedding-dim", type=int, default=16)
    parser.add_argument("--condition-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hla-hidden-dim", type=int, default=64)
    parser.add_argument("--classifier-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hla-embedding-condition-dim", type=int, default=32)
    parser.add_argument("--hla-embedding-hidden-dim", type=int, default=128)
    parser.add_argument("--hla-embedding-dropout", type=float, default=0.2)
    parser.add_argument("--hla-embedding-learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-tasks", type=int, default=0, help="Optional smoke-test limit on the number of tasks.")
    parser.add_argument(
        "--write-pseudo-template",
        action="store_true",
        help="Write a CSV template listing all HLA alleles needed by the current train/test split, then exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
