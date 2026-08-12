#!/usr/bin/env python3
"""Run mousePMHC Phase 3 E5: FAMO on the E3 Factorized MMoE backbone.

Both a task-balanced equal-weight control and FAMO are run with the identical
MMoE architecture and balanced task batches.  This isolates dynamic loss
weighting from changes in sampling.  The fixed test split is never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_mousepmhc_phase3_e3_factorized_mmoe_oof as e3
import run_tissuepmhc_e26_all_in_one as folds
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mousePMHC_phase3_e5_famo_mmoe_min200_oof"
MODELS = ["e3_task_balanced_control", "e5_famo_mmoe"]
KEYS = ["sample_id", "target_tissue", "mhc_restriction"]
WEIGHT_COLUMNS = [
    "experiment_name", "candidate", "seed", "fold", "epoch", "phase", "task_name",
    "target_tissue", "mhc_restriction", "famo_weight", "epoch_mean_loss", "mean_gate_entropy",
]
COMPARISON_COLUMNS = [
    "target_tissue", "mhc_restriction", "baseline_candidate", "candidate",
    "delta_accuracy", "delta_balanced_accuracy", "delta_auroc", "delta_auprc", "delta_f1", "delta_mcc",
]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


class FamoTaskWeighting:
    """FAMO weights with the loss-ratio update from the project FAMO implementation."""

    def __init__(self, torch: Any, n_tasks: int, device: str, gamma: float, weight_lr: float) -> None:
        self.torch = torch
        self.min_losses = torch.zeros(n_tasks, device=device)
        self.logits = torch.zeros(n_tasks, device=device, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.logits], lr=weight_lr, weight_decay=gamma)
        self.previous_losses: Any | None = None

    def weights(self) -> Any:
        return self.torch.softmax(self.logits, dim=-1)

    def weighted_loss(self, losses: Any) -> Any:
        self.previous_losses = losses.detach()
        weights = self.weights()
        shifted = losses - self.min_losses + 1e-8
        normalizer = (weights / shifted).sum().detach()
        return (shifted.log() * weights / normalizer).sum()

    def update(self, current_losses: Any) -> None:
        if self.previous_losses is None:
            return
        previous = self.previous_losses - self.min_losses + 1e-8
        current = current_losses.detach() - self.min_losses + 1e-8
        loss_drop = previous.log() - current.log()
        with self.torch.enable_grad():
            gradient = self.torch.autograd.grad(self.weights(), self.logits, grad_outputs=loss_drop.detach())[0]
        self.optimizer.zero_grad()
        self.logits.grad = gradient
        self.optimizer.step()


def task_arrays(frame: pd.DataFrame, mappings: dict[str, Any], peptide_length: int) -> list[dict[str, Any]]:
    arrays: list[dict[str, Any]] = []
    for task_name in mappings["tasks"]:
        task = frame[frame.task_name == task_name]
        arrays.append({
            "task_name": task_name, "task_id": int(mappings["task_to_id"][task_name]),
            "peptide": base.encode_peptides(task.peptide_sequence, peptide_length),
            "tissue": task.tissue_id.to_numpy(dtype=np.int64), "h2": task.hla_id.to_numpy(dtype=np.int64),
            "label": task.label.to_numpy(dtype=np.int64),
        })
    return arrays


def sample_balanced_batch(rng: np.random.Generator, arrays: list[dict[str, Any]], task_batch_size: int) -> tuple[np.ndarray, ...]:
    samples: list[list[np.ndarray]] = [[], [], [], [], []]
    for task in arrays:
        indices = rng.integers(0, len(task["label"]), size=task_batch_size)
        samples[0].append(task["peptide"][indices])
        samples[1].append(np.full(task_batch_size, task["task_id"], dtype=np.int64))
        samples[2].append(task["tissue"][indices])
        samples[3].append(task["h2"][indices])
        samples[4].append(task["label"][indices])
    return tuple(np.concatenate(values, axis=0) for values in samples)


def task_loss_vector(torch: Any, model: Any, peptide: Any, task: Any, tissue: Any, h2: Any,
                     label: Any, n_tasks: int, task_batch_size: int) -> tuple[Any, Any]:
    logits, gates = model(peptide, task, tissue, h2, return_gates=True)
    per_row = torch.nn.functional.binary_cross_entropy_with_logits(logits, label.float(), reduction="none")
    return per_row.reshape(n_tasks, task_batch_size).mean(dim=1), gates


def train_fold(args: argparse.Namespace, torch: Any, nn: Any, fitting: pd.DataFrame,
               mappings: dict[str, Any], peptide_length: int, device: str, fold: int,
               candidate: str) -> tuple[Any, list[dict[str, object]]]:
    model = e3.define_model(torch, nn, peptide_length, len(mappings["tasks"]), len(mappings["tissue_to_id"]),
                            len(mappings["hla_to_id"]), args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    arrays = task_arrays(fitting, mappings, peptide_length)
    n_tasks = len(arrays)
    steps = args.steps_per_epoch or int(np.ceil(max(len(item["label"]) for item in arrays) / args.task_batch_size))
    rng = np.random.default_rng(args.seed)
    famo = FamoTaskWeighting(torch, n_tasks, device, args.famo_gamma, args.famo_weight_lr) if candidate == "e5_famo_mmoe" else None
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        model.train(); epoch_losses: list[np.ndarray] = []; epoch_entropies: list[float] = []
        use_famo = famo is not None and epoch > args.warmup_epochs
        for _ in range(steps):
            batch = sample_balanced_batch(rng, arrays, args.task_batch_size)
            peptide, task, tissue, h2, label = [torch.as_tensor(value, device=device) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            losses, gates = task_loss_vector(torch, model, peptide, task, tissue, h2, label, n_tasks, args.task_batch_size)
            entropy = -(gates * torch.log(gates.clamp_min(1e-12))).sum(dim=1).mean()
            objective = famo.weighted_loss(losses) if use_famo else losses.mean()
            train_loss = objective - args.gate_entropy_weight * entropy
            train_loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if use_famo:
                with torch.no_grad():
                    updated_losses, _ = task_loss_vector(torch, model, peptide, task, tissue, h2, label, n_tasks, args.task_batch_size)
                famo.update(updated_losses)
                epoch_losses.append(updated_losses.detach().cpu().numpy())
            else:
                epoch_losses.append(losses.detach().cpu().numpy())
            epoch_entropies.append(float(entropy.detach().cpu()))
        weights = famo.weights().detach().cpu().numpy() if use_famo else np.full(n_tasks, 1.0 / n_tasks)
        mean_losses = np.stack(epoch_losses).mean(axis=0)
        phase = "famo" if use_famo else "equal_weight_warmup"
        for index, task_name in enumerate(mappings["tasks"]):
            tissue_name, h2_name = task_name.split("||", 1)
            history.append({"experiment_name": EXPERIMENT, "candidate": candidate, "seed": args.seed, "fold": fold,
                            "epoch": epoch, "phase": phase, "task_name": task_name, "target_tissue": tissue_name,
                            "mhc_restriction": h2_name, "famo_weight": float(weights[index]),
                            "epoch_mean_loss": float(mean_losses[index]), "mean_gate_entropy": float(np.mean(epoch_entropies))})
        print(f"E5 candidate={candidate} fold={fold + 1}/{args.oof_folds} epoch={epoch}/{args.epochs} phase={phase} mean_loss={float(mean_losses.mean()):.4f} min_weight={float(weights.min()):.4f} max_weight={float(weights.max()):.4f}", flush=True)
    return model, history


def predict(torch: Any, DataLoader: Any, TensorDataset: Any, model: Any, frame: pd.DataFrame,
            peptide_length: int, batch_size: int, device: str) -> np.ndarray:
    loader = e3.build_loader(torch, DataLoader, TensorDataset, frame, peptide_length, batch_size, False)
    scores: list[np.ndarray] = []; model.eval()
    with torch.no_grad():
        for peptide, task, tissue, h2, _ in loader:
            peptide, task, tissue, h2 = [value.to(device) for value in (peptide, task, tissue, h2)]
            scores.append(torch.sigmoid(model(peptide, task, tissue, h2)).cpu().numpy())
    return np.concatenate(scores)


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (candidate, tissue, h2), task in predictions.groupby(["candidate", "target_tissue", "mhc_restriction"], sort=True):
        rows.append({"experiment_name": EXPERIMENT, "candidate": candidate, "target_tissue": tissue,
                     "mhc_restriction": h2, "oof_rows": len(task),
                     **base.evaluate(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float))})
    per_task = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    for candidate, rows_for_candidate in per_task.groupby("candidate", sort=True):
        summary: dict[str, object] = {"experiment_name": EXPERIMENT, "candidate": candidate}
        for metric in ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]:
            summary[f"mean_task_{metric}"] = float(rows_for_candidate[metric].mean())
        summary["worst_task_auroc"] = float(rows_for_candidate.auroc.min())
        summaries.append(summary)
    return per_task, pd.DataFrame(summaries)


def comparisons(per_task: pd.DataFrame) -> pd.DataFrame:
    keys = ["target_tissue", "mhc_restriction"]
    baseline = per_task[per_task.candidate == "e3_task_balanced_control"].set_index(keys)
    candidate = per_task[per_task.candidate == "e5_famo_mmoe"].set_index(keys)
    rows: list[dict[str, object]] = []
    for key in baseline.index:
        item: dict[str, object] = {"target_tissue": key[0], "mhc_restriction": key[1],
                                   "baseline_candidate": "e3_task_balanced_control", "candidate": "e5_famo_mmoe"}
        for metric in ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]:
            item[f"delta_{metric}"] = float(candidate.loc[key, metric] - baseline.loc[key, metric])
        rows.append(item)
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def run(args: argparse.Namespace) -> None:
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    raw = base.read_dataset(args.train); e3.validate_input(raw)
    train, _, mappings = base.add_task_columns(raw, raw.copy())
    peptide_length = int(train.peptide_sequence.str.len().max())
    assignments = folds.make_pair_grouped_folds(train, args.oof_folds, args.oof_split_seed)
    parts: list[pd.DataFrame] = []; history: list[dict[str, object]] = []
    for candidate in args.models:
        for fold in range(args.oof_folds):
            fitting, held_out = train[assignments != fold].copy(), train[assignments == fold].copy()
            base.set_seed(args.seed, torch)
            print(f"mousePMHC E5 candidate={candidate} OOF fold={fold + 1}/{args.oof_folds} fit={len(fitting)} holdout={len(held_out)} device={device}", flush=True)
            model, fold_history = train_fold(args, torch, nn, fitting, mappings, peptide_length, device, fold, candidate)
            scores = predict(torch, DataLoader, TensorDataset, model, held_out, peptide_length, args.batch_size, device)
            output = held_out[KEYS + ["label"]].copy()
            output.insert(0, "split", "oof"); output.insert(1, "candidate", candidate); output.insert(2, "seed", args.seed)
            output["score"] = scores; parts.append(output[["split", "candidate", "seed", *KEYS, "label", "score"]]); history.extend(fold_history)
    predictions = pd.concat(parts, ignore_index=True)
    expected_rows = len(train) * len(args.models)
    if len(predictions) != expected_rows or predictions.duplicated(["candidate", "sample_id"]).any():
        raise AssertionError("E5 OOF predictions must cover every train row exactly once per candidate.")
    per_task, summary = metric_tables(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "mousePMHC_phase3_e5_oof_predictions.csv", index=False)
    per_task.to_csv(args.output_dir / "mousePMHC_phase3_e5_oof_per_task_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "mousePMHC_phase3_e5_oof_summary_metrics.csv", index=False)
    comparisons(per_task).to_csv(args.output_dir / "mousePMHC_phase3_e5_famo_comparison.csv", index=False)
    pd.DataFrame(history, columns=WEIGHT_COLUMNS).to_csv(args.output_dir / "mousePMHC_phase3_e5_famo_weight_history.csv", index=False)
    metadata = {
        "experiment_name": EXPERIMENT, "models": args.models, "backbone": "E3 Factorized MMoE", "test_data_read": False,
        "train": str(args.train), "n_rows": len(train), "n_pairs": int(train.pair_id.nunique()), "n_tasks": len(mappings["tasks"]),
        "n_tissues": len(mappings["tissue_to_id"]), "n_h2": len(mappings["hla_to_id"]), "seed": args.seed,
        "oof_folds": args.oof_folds, "oof_split_seed": args.oof_split_seed, "device": device, "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch, "task_batch_size": args.task_batch_size, "batch_size": args.batch_size,
        "warmup_epochs": args.warmup_epochs, "famo_gamma": args.famo_gamma, "famo_weight_lr": args.famo_weight_lr,
        "n_experts": args.n_experts, "expert_dim": args.expert_dim, "condition_dim": args.condition_dim,
        "gate_entropy_weight": args.gate_entropy_weight, "comparison": "FAMO versus equal-weight task-balanced control using identical E3 backbone and batches",
    }
    (args.output_dir / "mousePMHC_phase3_e5_oof_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=project_path("results/mousePMHC_phase3_e5_famo_mmoe_min200_oof"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--seed", type=int, default=20260704); parser.add_argument("--oof-folds", type=int, default=3); parser.add_argument("--oof-split-seed", type=int, default=20260711)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=0); parser.add_argument("--task-batch-size", type=int, default=16); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=128); parser.add_argument("--expert-dim", type=int, default=64)
    parser.add_argument("--condition-dim", type=int, default=16); parser.add_argument("--gate-hidden-dim", type=int, default=64); parser.add_argument("--n-experts", type=int, default=3); parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.01); parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--famo-gamma", type=float, default=0.01); parser.add_argument("--famo-weight-lr", type=float, default=0.025)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
