#!/usr/bin/env python3
"""Run mousePMHC Phase 3 E2: H2-grouped hard-sharing train-only OOF.

E2 is the minimal selective-sharing control for the 24-task mousePMHC
benchmark.  Tasks share a peptide encoder only with tasks having the same H2
restriction; each tissue-H2 task retains a separate linear prediction head.
It never reads the fixed test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_tissuepmhc_e26_all_in_one as folds
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mousePMHC_phase3_e2_h2_grouped_oof"
CANDIDATE = "mousePMHC_phase3_e2_h2_grouped_shared_encoder_task_heads"
KEYS = ["sample_id", "target_tissue", "mhc_restriction"]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def validate_input(frame: pd.DataFrame) -> None:
    if set(frame.dataset) != {"mousePMHC"} or set(frame.split) != {"train"}:
        raise ValueError("E2 accepts mousePMHC training rows only.")
    if not frame.mhc_restriction.str.startswith("H2-").all():
        raise ValueError("E2 found a non-H2 restriction.")


def arrays(frame: pd.DataFrame, peptide_length: int) -> list[np.ndarray]:
    return [
        base.encode_peptides(frame.peptide_sequence, peptide_length).copy(),
        frame.task_id.to_numpy(dtype=np.int64, copy=True),
        frame.hla_id.to_numpy(dtype=np.int64, copy=True),
        frame.label.to_numpy(dtype=np.int64, copy=True),
    ]


def define_model(nn: Any, peptide_length: int, n_tasks: int, n_h2: int,
                 embedding_dim: int, hidden_dim: int, dropout: float) -> Any:
    class PeptideEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(len(base.AA_TO_INDEX) + 1, embedding_dim,
                                          padding_idx=base.PAD_INDEX)
            self.network = nn.Sequential(
                nn.Flatten(), nn.Linear(peptide_length * embedding_dim, hidden_dim),
                nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )

        def forward(self, peptide_ids: Any) -> Any:
            return self.network(self.embedding(peptide_ids))

    class H2GroupedSharedTaskHeadsModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.h2_encoders = nn.ModuleList([PeptideEncoder() for _ in range(n_h2)])
            self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_tasks)])

        def forward(self, peptide_ids: Any, task_ids: Any, h2_ids: Any) -> Any:
            logits = peptide_ids.new_empty(peptide_ids.shape[0], dtype=self.heads[0].weight.dtype)
            for h2_id in base.torch_unique(h2_ids):
                h2_mask = h2_ids == h2_id
                features = self.h2_encoders[int(h2_id.item())](peptide_ids[h2_mask])
                h2_task_ids = task_ids[h2_mask]
                h2_logits = features.new_empty(features.shape[0])
                for task_id in base.torch_unique(h2_task_ids):
                    task_mask = h2_task_ids == task_id
                    h2_logits[task_mask] = self.heads[int(task_id.item())](features[task_mask]).squeeze(-1)
                logits[h2_mask] = h2_logits
            return logits

    return H2GroupedSharedTaskHeadsModel()


def train_model(torch: Any, model: Any, loader: Any, optimizer: Any, loss_fn: Any,
                device: str, epochs: int) -> None:
    model.train()
    for _ in range(epochs):
        for peptide_ids, task_ids, h2_ids, labels in loader:
            peptide_ids, task_ids, h2_ids, labels = (item.to(device) for item in (peptide_ids, task_ids, h2_ids, labels))
            optimizer.zero_grad()
            loss = loss_fn(model(peptide_ids, task_ids, h2_ids), labels.float())
            loss.backward()
            optimizer.step()


def predict_scores(torch: Any, model: Any, loader: Any, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    with torch.no_grad():
        for peptide_ids, task_ids, h2_ids, labels in loader:
            peptide_ids, task_ids, h2_ids = peptide_ids.to(device), task_ids.to(device), h2_ids.to(device)
            all_labels.append(labels.numpy())
            all_scores.append(torch.sigmoid(model(peptide_ids, task_ids, h2_ids)).cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_scores)


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (tissue, mhc), task in predictions.groupby(["target_tissue", "mhc_restriction"], sort=True):
        rows.append({"experiment_name": EXPERIMENT, "candidate": CANDIDATE,
                     "target_tissue": tissue, "mhc_restriction": mhc, "oof_rows": len(task),
                     **base.evaluate(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float))})
    per_task = pd.DataFrame(rows)
    summary = {"experiment_name": EXPERIMENT, "candidate": CANDIDATE}
    for metric in ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]:
        summary[f"mean_task_{metric}"] = float(per_task[metric].mean())
    return per_task, pd.DataFrame([summary])


def run(args: argparse.Namespace) -> None:
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    raw = base.read_dataset(args.train)
    validate_input(raw)
    train, _, mappings = base.add_task_columns(raw, raw.copy())
    if len(mappings["tasks"]) < 2:
        raise ValueError("E2 requires at least two tissue-H2 tasks.")
    task_h2_counts = train.groupby("task_id").hla_id.nunique()
    if not (task_h2_counts == 1).all():
        raise AssertionError("Every task must map to exactly one H2 restriction.")
    peptide_length = int(train.peptide_sequence.str.len().max())
    assignments = folds.make_pair_grouped_folds(train, args.oof_folds, args.oof_split_seed)
    parts: list[pd.DataFrame] = []
    for fold in range(args.oof_folds):
        fitting, held_out = train[assignments != fold].copy(), train[assignments == fold].copy()
        base.set_seed(args.seed, torch)
        model = define_model(nn, peptide_length, len(mappings["tasks"]), len(mappings["hla_to_id"]),
                             args.embedding_dim, args.hidden_dim, args.dropout).to(device)
        train_loader = base.build_loader(torch, DataLoader, TensorDataset, arrays(fitting, peptide_length), args.batch_size, True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        print(f"mousePMHC E2 H2-grouped OOF fold={fold + 1}/{args.oof_folds} fit={len(fitting)} holdout={len(held_out)}", flush=True)
        train_model(torch, model, train_loader, optimizer, nn.BCEWithLogitsLoss(), device, args.epochs)
        held_loader = base.build_loader(torch, DataLoader, TensorDataset, arrays(held_out, peptide_length), args.batch_size, False)
        labels, scores = predict_scores(torch, model, held_loader, device)
        if not np.array_equal(labels, held_out.label.to_numpy(dtype=np.int64)):
            raise AssertionError("E2 OOF labels are not aligned with held-out rows.")
        output = held_out[KEYS + ["label"]].copy()
        output.insert(0, "split", "oof"); output.insert(1, "candidate", CANDIDATE); output.insert(2, "seed", args.seed)
        output["score"] = scores
        parts.append(output[["split", "candidate", "seed", *KEYS, "label", "score"]])
    predictions = pd.concat(parts, ignore_index=True)
    if len(predictions) != len(train) or predictions.sample_id.duplicated().any() or set(predictions.sample_id) != set(train.sample_id):
        raise AssertionError("E2 OOF predictions do not cover every training sample exactly once.")
    per_task, summary = metric_tables(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "mousePMHC_phase3_e2_oof_predictions.csv", index=False)
    per_task.to_csv(args.output_dir / "mousePMHC_phase3_e2_oof_per_task_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "mousePMHC_phase3_e2_oof_summary_metrics.csv", index=False)
    metadata = {
        "experiment_name": EXPERIMENT, "candidate": CANDIDATE, "species": "Mus musculus", "mhc_system": "H2-I",
        "mouse_experiment_id": "E2", "method": "one shared peptide encoder per H2 restriction; task-specific linear heads",
        "test_data_read": False, "train": str(args.train), "n_rows": len(train), "n_pairs": int(train.pair_id.nunique()),
        "n_tasks": len(mappings["tasks"]), "tasks": mappings["tasks"], "n_h2_groups": len(mappings["hla_to_id"]),
        "h2_groups": mappings["hla_to_id"], "seed": args.seed, "oof_folds": args.oof_folds,
        "oof_split_seed": args.oof_split_seed, "device": device, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim, "dropout": args.dropout,
        "comparison_baseline": "mousePMHC_phase3_e1_shared_peptide_encoder_task_heads",
    }
    (args.output_dir / "mousePMHC_phase3_e2_oof_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=project_path("results/mousePMHC_phase3_e2_h2_grouped_oof"))
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--oof-folds", type=int, default=3); parser.add_argument("--oof-split-seed", type=int, default=20260711)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=128); parser.add_argument("--dropout", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
