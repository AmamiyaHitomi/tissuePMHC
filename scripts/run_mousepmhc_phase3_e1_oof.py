#!/usr/bin/env python3
"""Run mousePMHC E1 using the human Phase 1 E2 shared-task-head method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_tissuepmhc_e26_all_in_one as folds
import run_tissuepmhc_neural_baselines_v2 as human_e2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mousePMHC_phase3_e1_oof"
CANDIDATE = "mousePMHC_phase3_e1_shared_peptide_encoder_task_heads"
KEYS = ["sample_id", "target_tissue", "mhc_restriction"]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def validate_input(frame: pd.DataFrame) -> None:
    if set(frame.dataset) != {"mousePMHC"} or set(frame.split) != {"train"}:
        raise ValueError("mousePMHC E1 accepts mousePMHC train rows only.")
    if not frame.mhc_restriction.str.startswith("H2-").all():
        raise ValueError("mousePMHC E1 found a non-H2 MHC restriction.")


def arrays(frame: pd.DataFrame, peptide_length: int) -> list[np.ndarray]:
    return [
        human_e2.encode_peptides(frame.peptide_sequence, peptide_length).copy(),
        frame.task_id.to_numpy(dtype=np.int64, copy=True),
        frame.label.to_numpy(dtype=np.int64, copy=True),
    ]


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (tissue, mhc), task in predictions.groupby(["target_tissue", "mhc_restriction"], sort=True):
        rows.append({"experiment_name": EXPERIMENT, "candidate": CANDIDATE,
            "target_tissue": tissue, "mhc_restriction": mhc, "oof_rows": len(task),
            **human_e2.evaluate(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float))})
    per_task = pd.DataFrame(rows)
    summary = {"experiment_name": EXPERIMENT, "candidate": CANDIDATE}
    for metric in ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]:
        summary[f"mean_task_{metric}"] = float(per_task[metric].mean())
    return per_task, pd.DataFrame([summary])


def run(args: argparse.Namespace) -> None:
    torch, nn, DataLoader, TensorDataset = human_e2.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    raw = human_e2.read_dataset(args.train)
    validate_input(raw)
    train, _, mappings = human_e2.add_task_columns(raw, raw.copy())
    if args.max_tasks:
        keep = set(mappings["tasks"][:args.max_tasks])
        train = train[train.task_name.isin(keep)].copy()
        train, _, mappings = human_e2.add_task_columns(train, train.copy())
    peptide_length = int(train.peptide_sequence.str.len().max())
    assignments = folds.make_pair_grouped_folds(train, args.oof_folds, args.oof_split_seed)
    _, SharedTaskHeadsModel, _ = human_e2.define_models(nn)
    parts: list[pd.DataFrame] = []
    for fold in range(args.oof_folds):
        fitting, held_out = train[assignments != fold].copy(), train[assignments == fold].copy()
        human_e2.set_seed(args.seed, torch)
        model = SharedTaskHeadsModel(peptide_length, len(mappings["tasks"]), args.embedding_dim, args.hidden_dim, args.dropout).to(device)
        train_loader = human_e2.build_loader(torch, DataLoader, TensorDataset, arrays(fitting, peptide_length), args.batch_size, True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        loss = nn.BCEWithLogitsLoss()
        print(f"mousePMHC E1 OOF fold={fold + 1}/{args.oof_folds} fit={len(fitting)} holdout={len(held_out)}", flush=True)
        human_e2.train_binary_model(torch, model, train_loader, optimizer, loss, device, "task_heads", args.epochs)
        held_loader = human_e2.build_loader(torch, DataLoader, TensorDataset, arrays(held_out, peptide_length), args.batch_size, False)
        labels, scores = human_e2.predict_scores(torch, model, held_loader, device, "task_heads")
        if not np.array_equal(labels, held_out.label.to_numpy(dtype=np.int64)):
            raise AssertionError("mousePMHC E1 OOF labels are not aligned with held-out rows.")
        output = held_out[KEYS + ["label"]].copy()
        output.insert(0, "split", "oof"); output.insert(1, "candidate", CANDIDATE); output.insert(2, "seed", args.seed)
        output["score"] = scores
        parts.append(output[["split", "candidate", "seed", *KEYS, "label", "score"]])
    predictions = pd.concat(parts, ignore_index=True)
    if len(predictions) != len(train) or set(predictions.sample_id) != set(train.sample_id) or predictions.sample_id.duplicated().any():
        raise AssertionError("mousePMHC E1 OOF predictions do not cover every training sample exactly once.")
    per_task, summary = metric_tables(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "mousePMHC_phase3_e1_oof_predictions.csv", index=False)
    per_task.to_csv(args.output_dir / "mousePMHC_phase3_e1_oof_per_task_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "mousePMHC_phase3_e1_oof_summary_metrics.csv", index=False)
    metadata = {
        "experiment_name": EXPERIMENT, "species": "Mus musculus", "mhc_system": "H2-I",
        "mouse_experiment_id": "E1",
        "human_method_source": "tissuePMHC Phase 1 E2 shared peptide encoder + task-specific heads",
        "test_data_read": False, "train": str(args.train), "n_rows": len(train), "n_pairs": int(train.pair_id.nunique()),
        "n_tasks": len(mappings["tasks"]), "seed": args.seed, "oof_folds": args.oof_folds,
        "oof_split_seed": args.oof_split_seed, "device": device, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "embedding_dim": args.embedding_dim, "hidden_dim": args.hidden_dim, "dropout": args.dropout,
        "naming_policy": "mousePMHC_phase3_<experiment>; human tissuePMHC outputs are never overwritten",
    }
    (args.output_dir / "mousePMHC_phase3_e1_oof_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=project_path("results/mousePMHC_phase3_e1_oof"))
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--oof-folds", type=int, default=3); parser.add_argument("--oof-split-seed", type=int, default=20260711)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=128); parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-tasks", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
