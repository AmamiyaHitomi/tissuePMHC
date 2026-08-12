#!/usr/bin/env python3
"""Confirm E3b: three-seed task-balanced Factorized MMoE train-only OOF.

E3b is the equal-weight task-balanced MMoE control that outperformed E1 and
the FAMO variant in the initial 20260704 screen.  It confirms stability across
three predeclared training seeds without reading the fixed test split.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_mousepmhc_phase3_e3_factorized_mmoe_oof as e3
import run_mousepmhc_phase3_e5_famo_mmoe_oof as e5
import run_tissuepmhc_e26_all_in_one as folds
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mousePMHC_phase3_e3b_task_balanced_mmoe_min200_oof"
CANDIDATE = "mousePMHC_phase3_e3b_task_balanced_mmoe_min200"
KEYS = ["sample_id", "target_tissue", "mhc_restriction"]
GATE_COLUMNS = [
    "experiment_name", "candidate", "seed", "fold", "task_name", "target_tissue",
    "mhc_restriction", "expert_id", "gate_weight",
]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def collect_final_gates(model: Any, frame: pd.DataFrame, torch: Any, DataLoader: Any,
                        TensorDataset: Any, peptide_length: int, batch_size: int, device: str,
                        seed: int, fold: int) -> list[dict[str, object]]:
    totals: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    loader = e3.build_loader(torch, DataLoader, TensorDataset, frame, peptide_length, batch_size, False)
    model.eval()
    with torch.no_grad():
        for peptide, task, tissue, h2, _ in loader:
            peptide, task, tissue, h2 = [value.to(device) for value in (peptide, task, tissue, h2)]
            _, gates = model(peptide, task, tissue, h2, return_gates=True)
            for task_id in torch.unique(task):
                mask = task == task_id
                numeric_id = int(task_id.item())
                totals[numeric_id] = totals.get(numeric_id, np.zeros(gates.shape[1])) + gates[mask].sum(dim=0).cpu().numpy()
                counts[numeric_id] = counts.get(numeric_id, 0) + int(mask.sum().item())
    lookup = frame[["task_id", "task_name", "target_tissue", "mhc_restriction"]].drop_duplicates().set_index("task_id")
    rows: list[dict[str, object]] = []
    for task_id, total in sorted(totals.items()):
        info = lookup.loc[task_id]
        for expert_id, weight in enumerate(total / counts[task_id]):
            rows.append({"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "seed": seed, "fold": fold,
                         "task_name": info.task_name, "target_tissue": info.target_tissue,
                         "mhc_restriction": info.mhc_restriction, "expert_id": expert_id,
                         "gate_weight": float(weight)})
    return rows


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
    stability_rows: list[dict[str, object]] = []
    for metric in [column for column in summary.columns if column.startswith("mean_task_") or column.startswith("worst")]:
        stability_rows.append({"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "metric": metric,
                               "seed_mean": float(summary[metric].mean()), "seed_sd": float(summary[metric].std(ddof=1)),
                               "seed_min": float(summary[metric].min()), "seed_max": float(summary[metric].max())})
    return per_task, summary, pd.DataFrame(stability_rows)


def run(args: argparse.Namespace) -> None:
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    raw = base.read_dataset(args.train); e3.validate_input(raw)
    train, _, mappings = base.add_task_columns(raw, raw.copy())
    peptide_length = int(train.peptide_sequence.str.len().max())
    assignments = folds.make_pair_grouped_folds(train, args.oof_folds, args.oof_split_seed)
    prediction_parts: list[pd.DataFrame] = []; gate_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        seed_args = copy.copy(args); seed_args.seed = int(seed)
        for fold in range(args.oof_folds):
            fitting, held_out = train[assignments != fold].copy(), train[assignments == fold].copy()
            base.set_seed(seed_args.seed, torch)
            print(f"E3b seed={seed_args.seed} fold={fold + 1}/{args.oof_folds} fit={len(fitting)} holdout={len(held_out)} device={device}", flush=True)
            model, _ = e5.train_fold(seed_args, torch, nn, fitting, mappings, peptide_length, device, fold,
                                     "e3_task_balanced_control")
            scores = e5.predict(torch, DataLoader, TensorDataset, model, held_out, peptide_length, args.batch_size, device)
            output = held_out[KEYS + ["label"]].copy()
            output.insert(0, "split", "oof"); output.insert(1, "candidate", CANDIDATE); output.insert(2, "seed", seed_args.seed)
            output["score"] = scores
            prediction_parts.append(output[["split", "candidate", "seed", *KEYS, "label", "score"]])
            gate_rows.extend(collect_final_gates(model, fitting, torch, DataLoader, TensorDataset, peptide_length,
                                                 args.batch_size, device, seed_args.seed, fold))
    predictions = pd.concat(prediction_parts, ignore_index=True)
    expected = len(train) * len(args.seeds)
    if len(predictions) != expected or predictions.duplicated(["seed", "sample_id"]).any():
        raise AssertionError("E3b OOF predictions must cover every train row exactly once per seed.")
    per_task, summary, stability = metric_tables(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "mousePMHC_phase3_e3b_oof_predictions.csv", index=False)
    per_task.to_csv(args.output_dir / "mousePMHC_phase3_e3b_oof_per_task_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "mousePMHC_phase3_e3b_oof_summary_metrics.csv", index=False)
    stability.to_csv(args.output_dir / "mousePMHC_phase3_e3b_oof_stability_metrics.csv", index=False)
    pd.DataFrame(gate_rows, columns=GATE_COLUMNS).to_csv(args.output_dir / "mousePMHC_phase3_e3b_gate_weights.csv", index=False)
    metadata = {
        "experiment_name": EXPERIMENT, "candidate": CANDIDATE, "backbone": "E3 Factorized MMoE",
        "training": "equal-weight task-balanced batches; no FAMO", "test_data_read": False,
        "train": str(args.train), "n_rows": len(train), "n_pairs": int(train.pair_id.nunique()),
        "n_tasks": len(mappings["tasks"]), "n_tissues": len(mappings["tissue_to_id"]), "n_h2": len(mappings["hla_to_id"]),
        "seeds": args.seeds, "oof_folds": args.oof_folds, "oof_split_seed": args.oof_split_seed,
        "device": device, "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch,
        "task_batch_size": args.task_batch_size, "batch_size": args.batch_size, "n_experts": args.n_experts,
        "expert_dim": args.expert_dim, "condition_dim": args.condition_dim, "gate_entropy_weight": args.gate_entropy_weight,
        "screen_source": "E5 equal-weight task-balanced control, seed 20260704",
    }
    (args.output_dir / "mousePMHC_phase3_e3b_oof_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(stability.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=project_path("results/mousePMHC_phase3_e3b_task_balanced_mmoe_min200_oof"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260704, 20260705, 20260706])
    parser.add_argument("--oof-folds", type=int, default=3); parser.add_argument("--oof-split-seed", type=int, default=20260711)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--task-batch-size", type=int, default=16); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--expert-dim", type=int, default=64); parser.add_argument("--condition-dim", type=int, default=16)
    parser.add_argument("--gate-hidden-dim", type=int, default=64); parser.add_argument("--n-experts", type=int, default=3); parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.01); parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=5); parser.add_argument("--famo-gamma", type=float, default=0.01); parser.add_argument("--famo-weight-lr", type=float, default=0.025)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
