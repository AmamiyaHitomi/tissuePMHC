#!/usr/bin/env python3
"""Add matched E1 train-only OOF seeds without overwriting the 20260704 result.

The existing E1 result is the 20260704 screen. This entry point runs only the
predeclared additional seeds 20260705 and 20260706 by default, writing a
separate directory so E1 can be compared fairly with E3b's three seeds.
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
EXPERIMENT = "mousePMHC_phase3_e1_additional_seeds_oof"
CANDIDATE = "mousePMHC_phase3_e1_shared_peptide_encoder_task_heads"
KEYS = ["sample_id", "target_tissue", "mhc_restriction"]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def validate_input(frame: pd.DataFrame) -> None:
    if set(frame.dataset) != {"mousePMHC"} or set(frame.split) != {"train"}:
        raise ValueError("E1 additional-seed run accepts mousePMHC training rows only.")
    if not frame.mhc_restriction.str.startswith("H2-").all():
        raise ValueError("E1 found a non-H2 restriction.")


def arrays(frame: pd.DataFrame, peptide_length: int) -> list[np.ndarray]:
    return [
        base.encode_peptides(frame.peptide_sequence, peptide_length).copy(),
        frame.task_id.to_numpy(dtype=np.int64, copy=True),
        frame.label.to_numpy(dtype=np.int64, copy=True),
    ]


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (seed, tissue, h2), task in predictions.groupby(["seed", "target_tissue", "mhc_restriction"], sort=True):
        rows.append({"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "seed": int(seed),
                     "target_tissue": tissue, "mhc_restriction": h2, "oof_rows": len(task),
                     **base.evaluate(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float))})
    per_task = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for seed, records in per_task.groupby("seed", sort=True):
        row: dict[str, object] = {"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "seed": int(seed)}
        for metric in ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]:
            row[f"mean_task_{metric}"] = float(records[metric].mean())
        row["worst_task_auroc"] = float(records.auroc.min())
        row["worst6_task_auroc"] = float(records.nsmallest(6, "auroc").auroc.mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    stability: list[dict[str, object]] = []
    for metric in [column for column in summary.columns if column.startswith("mean_task_") or column.startswith("worst")]:
        stability.append({"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "metric": metric,
                          "seed_mean": float(summary[metric].mean()), "seed_sd": float(summary[metric].std(ddof=1)),
                          "seed_min": float(summary[metric].min()), "seed_max": float(summary[metric].max())})
    return per_task, summary, pd.DataFrame(stability)


def run(args: argparse.Namespace) -> None:
    if 20260704 in args.seeds:
        raise ValueError("This script is for additional seeds only; 20260704 is already stored in the primary E1 result directory.")
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    raw = base.read_dataset(args.train); validate_input(raw)
    train, _, mappings = base.add_task_columns(raw, raw.copy())
    peptide_length = int(train.peptide_sequence.str.len().max())
    assignments = folds.make_pair_grouped_folds(train, args.oof_folds, args.oof_split_seed)
    _, SharedTaskHeadsModel, _ = base.define_models(nn)
    parts: list[pd.DataFrame] = []
    for seed in args.seeds:
        for fold in range(args.oof_folds):
            fitting, held_out = train[assignments != fold].copy(), train[assignments == fold].copy()
            base.set_seed(seed, torch)
            model = SharedTaskHeadsModel(peptide_length, len(mappings["tasks"]), args.embedding_dim, args.hidden_dim, args.dropout).to(device)
            loader = base.build_loader(torch, DataLoader, TensorDataset, arrays(fitting, peptide_length), args.batch_size, True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            print(f"mousePMHC E1 additional seed={seed} fold={fold + 1}/{args.oof_folds} fit={len(fitting)} holdout={len(held_out)}", flush=True)
            base.train_binary_model(torch, model, loader, optimizer, nn.BCEWithLogitsLoss(), device, "task_heads", args.epochs)
            held_loader = base.build_loader(torch, DataLoader, TensorDataset, arrays(held_out, peptide_length), args.batch_size, False)
            labels, scores = base.predict_scores(torch, model, held_loader, device, "task_heads")
            if not np.array_equal(labels, held_out.label.to_numpy(dtype=np.int64)):
                raise AssertionError("E1 additional-seed OOF labels are not aligned with held-out rows.")
            output = held_out[KEYS + ["label"]].copy()
            output.insert(0, "split", "oof"); output.insert(1, "candidate", CANDIDATE); output.insert(2, "seed", int(seed))
            output["score"] = scores
            parts.append(output[["split", "candidate", "seed", *KEYS, "label", "score"]])
    predictions = pd.concat(parts, ignore_index=True)
    expected = len(train) * len(args.seeds)
    if len(predictions) != expected or predictions.duplicated(["seed", "sample_id"]).any():
        raise AssertionError("E1 additional-seed predictions must cover every train row exactly once per seed.")
    per_task, summary, stability = metric_tables(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "mousePMHC_phase3_e1_additional_oof_predictions.csv", index=False)
    per_task.to_csv(args.output_dir / "mousePMHC_phase3_e1_additional_oof_per_task_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "mousePMHC_phase3_e1_additional_oof_summary_metrics.csv", index=False)
    stability.to_csv(args.output_dir / "mousePMHC_phase3_e1_additional_oof_stability_metrics.csv", index=False)
    metadata = {
        "experiment_name": EXPERIMENT, "candidate": CANDIDATE, "purpose": "matched E1 seeds for E3b comparison; primary 20260704 E1 outputs are preserved",
        "test_data_read": False, "train": str(args.train), "n_rows": len(train), "n_pairs": int(train.pair_id.nunique()),
        "n_tasks": len(mappings["tasks"]), "seeds": args.seeds, "oof_folds": args.oof_folds,
        "oof_split_seed": args.oof_split_seed, "device": device, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim, "dropout": args.dropout,
        "primary_e1_seed_and_directory": {"seed": 20260704, "directory": "results/mousePMHC_phase3_e1_oof"},
    }
    (args.output_dir / "mousePMHC_phase3_e1_additional_oof_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(stability.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=project_path("results/mousePMHC_phase3_e1_oof_additional_seeds"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260705, 20260706])
    parser.add_argument("--oof-folds", type=int, default=3); parser.add_argument("--oof-split-seed", type=int, default=20260711)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=128); parser.add_argument("--dropout", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
