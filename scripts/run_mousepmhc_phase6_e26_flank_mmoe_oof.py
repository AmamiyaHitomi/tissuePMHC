#!/usr/bin/env python3
"""Run Phase 6 E26: flank/position-augmented Factorized MMoE train-only OOF.

This candidate preserves the E3b task-balanced MMoE backbone and adds a small
tissue-conditioned processing branch from source-protein flanks and positions.
The branch enters through a learnable scalar initialized to zero.  No fixed
test file is accepted or opened by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_mousepmhc_phase3_e5_famo_mmoe_oof as e5
import run_tissuepmhc_e26_all_in_one as folds
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mousePMHC_phase6_e26_flank_mmoe_oof"
CANDIDATE = "mousePMHC_phase6_e26_e3b_zero_init_flank_position_branch"
BASELINE = "mousePMHC_phase3_e3b_task_balanced_mmoe_min200"
KEYS = ["sample_id", "target_tissue", "mhc_restriction", "label"]
AA = "ACDEFGHIKLMNPQRSTVWY"
FLANK_TO_ID = {aa: index + 1 for index, aa in enumerate(AA)}
PAD_ID = 0


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def validate(train: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    required_train = {"dataset", "split", "sample_id", "pair_id", "label", "target_tissue", "mhc_restriction", "peptide_sequence"}
    required_features = {"sample_id", "flank_left", "flank_right", "position_relative", "distance_to_nterm", "distance_to_cterm", "flank_available"}
    if missing := required_train - set(train.columns):
        raise ValueError(f"E26 train is missing columns: {sorted(missing)}")
    if missing := required_features - set(features.columns):
        raise ValueError(f"E26 flank features are missing columns: {sorted(missing)}")
    if set(train.dataset) != {"mousePMHC"} or set(train.split) != {"train"}:
        raise ValueError("E26 accepts mousePMHC train rows only.")
    if not train.peptide_sequence.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{9}").all():
        raise ValueError("E26 requires standard 9-mer peptides.")
    if features.sample_id.duplicated().any() or set(features.sample_id) != set(train.sample_id):
        raise ValueError("E26 flank features must align one-to-one with every train sample.")
    merged = train.merge(features, on="sample_id", how="left", validate="one_to_one")
    if merged.flank_left.isna().any() or merged.flank_right.isna().any():
        raise ValueError("E26 found missing flank feature rows after merge.")
    lengths = merged.flank_left.str.len().unique().tolist() + merged.flank_right.str.len().unique().tolist()
    if set(lengths) != {10}:
        raise ValueError("E26 requires fixed 10-aa left and right flanks.")
    if not merged.flank_left.str.fullmatch(r"[-ACDEFGHIKLMNPQRSTVWY]{10}").all() or not merged.flank_right.str.fullmatch(r"[-ACDEFGHIKLMNPQRSTVWY]{10}").all():
        raise ValueError("E26 found invalid flank tokens.")
    return merged


def encode_flanks(left: pd.Series, right: pd.Series) -> np.ndarray:
    values = left.str.cat(right).tolist()
    return np.asarray([[FLANK_TO_ID.get(character, PAD_ID) for character in sequence] for sequence in values], dtype=np.int64)


def numeric_features(frame: pd.DataFrame) -> np.ndarray:
    available = frame.flank_available.to_numpy(dtype=np.float32)
    rel = np.where(available > 0, frame.position_relative.to_numpy(dtype=np.float32), 0.0)
    distance_n = np.maximum(frame.distance_to_nterm.to_numpy(dtype=np.float32), 0.0)
    distance_c = np.maximum(frame.distance_to_cterm.to_numpy(dtype=np.float32), 0.0)
    nterm = np.where(available > 0, np.log1p(distance_n) / math.log(10001.0), 0.0)
    cterm = np.where(available > 0, np.log1p(distance_c) / math.log(10001.0), 0.0)
    return np.stack([rel, nterm, cterm, available], axis=1).astype(np.float32)


def define_model(torch: Any, nn: Any, args: argparse.Namespace, n_tasks: int, n_tissues: int, n_mhcs: int) -> Any:
    class FlankMMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.amino_embedding = nn.Embedding(21, args.embedding_dim, padding_idx=0)
            self.peptide_encoder = nn.Sequential(nn.Flatten(), nn.Linear(9 * args.embedding_dim, args.hidden_dim), nn.ReLU(), nn.Dropout(args.dropout))
            self.tissue_embedding = nn.Embedding(n_tissues, args.condition_dim)
            self.mhc_embedding = nn.Embedding(n_mhcs, args.condition_dim)
            self.experts = nn.ModuleList([nn.Sequential(nn.Linear(args.hidden_dim, args.expert_dim), nn.ReLU(), nn.Dropout(args.dropout), nn.Linear(args.expert_dim, args.expert_dim), nn.ReLU()) for _ in range(args.n_experts)])
            self.gate = nn.Sequential(nn.Linear(args.hidden_dim + 2 * args.condition_dim, args.gate_hidden_dim), nn.ReLU(), nn.Linear(args.gate_hidden_dim, args.n_experts))
            self.heads = nn.ModuleList([nn.Linear(args.expert_dim, 1) for _ in range(n_tasks)])
            self.flank_embedding = nn.Embedding(len(FLANK_TO_ID) + 1, args.flank_embedding_dim, padding_idx=PAD_ID)
            self.processing_branch = nn.Sequential(nn.Linear(20 * args.flank_embedding_dim + 4 + args.condition_dim, args.processing_hidden_dim), nn.ReLU(), nn.Dropout(args.dropout), nn.Linear(args.processing_hidden_dim, args.expert_dim), nn.ReLU())
            self.processing_scale = nn.Parameter(torch.zeros(()))

        def forward(self, peptide: Any, task: Any, tissue: Any, h2: Any, flank: Any, numeric: Any, return_gates: bool = False):
            encoded = self.peptide_encoder(self.amino_embedding(peptide))
            tissue_repr = self.tissue_embedding(tissue)
            expert_outputs = torch.stack([expert(encoded) for expert in self.experts], dim=1)
            gates = torch.softmax(self.gate(torch.cat([encoded, tissue_repr, self.mhc_embedding(h2)], dim=1)), dim=1)
            mixed = (expert_outputs * gates.unsqueeze(-1)).sum(dim=1)
            processing = self.processing_branch(torch.cat([self.flank_embedding(flank).flatten(1), numeric, tissue_repr], dim=1))
            mixed = mixed + self.processing_scale * processing
            logits = torch.empty(len(task), device=task.device)
            for task_id in task.unique():
                mask = task == task_id
                logits[mask] = self.heads[int(task_id.item())](mixed[mask]).squeeze(-1)
            return (logits, gates) if return_gates else logits
    return FlankMMoE()


def task_arrays(frame: pd.DataFrame, mappings: dict[str, Any]) -> list[dict[str, Any]]:
    arrays = []
    for task_name in mappings["tasks"]:
        part = frame[frame.task_name == task_name]
        arrays.append({"task_id": mappings["task_to_id"][task_name], "peptide": base.encode_peptides(part.peptide_sequence, 9), "tissue": part.tissue_id.to_numpy(np.int64), "h2": part.hla_id.to_numpy(np.int64), "flank": encode_flanks(part.flank_left, part.flank_right), "numeric": numeric_features(part), "label": part.label.to_numpy(np.int64)})
    return arrays


def sample_batch(rng: np.random.Generator, arrays: list[dict[str, Any]], size: int) -> tuple[np.ndarray, ...]:
    values: list[list[np.ndarray]] = [[] for _ in range(7)]
    for item in arrays:
        index = rng.integers(0, len(item["label"]), size=size)
        values[0].append(item["peptide"][index]); values[1].append(np.full(size, item["task_id"], dtype=np.int64)); values[2].append(item["tissue"][index]); values[3].append(item["h2"][index]); values[4].append(item["flank"][index]); values[5].append(item["numeric"][index]); values[6].append(item["label"][index])
    return tuple(np.concatenate(value, axis=0) for value in values)


def fit_predict(args: argparse.Namespace, torch: Any, nn: Any, fitting: pd.DataFrame, held_out: pd.DataFrame, mappings: dict[str, Any], seed: int, device: str) -> tuple[np.ndarray, float]:
    model = define_model(torch, nn, args, len(mappings["tasks"]), len(mappings["tissue_to_id"]), len(mappings["hla_to_id"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    arrays = task_arrays(fitting, mappings); steps = args.steps_per_epoch or int(np.ceil(max(len(item["label"]) for item in arrays) / args.task_batch_size)); rng = np.random.default_rng(seed)
    for epoch in range(1, args.epochs + 1):
        losses = []; scales = []; model.train()
        for _ in range(steps):
            peptide, task, tissue, h2, flank, numeric, label = [torch.as_tensor(value, device=device) for value in sample_batch(rng, arrays, args.task_batch_size)]
            optimizer.zero_grad(set_to_none=True); logits, gates = model(peptide, task, tissue, h2, flank, numeric, return_gates=True)
            row_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label.float(), reduction="none")
            loss = row_loss.reshape(len(arrays), args.task_batch_size).mean(1).mean() - args.gate_entropy_weight * (-(gates * torch.log(gates.clamp_min(1e-12))).sum(1).mean())
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm); optimizer.step(); losses.append(float(loss.detach().cpu())); scales.append(float(model.processing_scale.detach().cpu()))
        print(f"E26 seed={seed} epoch={epoch}/{args.epochs} loss={np.mean(losses):.5f} processing_scale={np.mean(scales):.5f}", flush=True)
    model.eval(); output = []; batch = args.batch_size
    with torch.no_grad():
        for start in range(0, len(held_out), batch):
            part = held_out.iloc[start:start + batch]; peptide = torch.as_tensor(base.encode_peptides(part.peptide_sequence, 9).copy(), device=device); task = torch.as_tensor(part.task_id.to_numpy(np.int64, copy=True), device=device); tissue = torch.as_tensor(part.tissue_id.to_numpy(np.int64, copy=True), device=device); h2 = torch.as_tensor(part.hla_id.to_numpy(np.int64, copy=True), device=device); flank = torch.as_tensor(encode_flanks(part.flank_left, part.flank_right), device=device); numeric = torch.as_tensor(numeric_features(part), device=device)
            output.append(torch.sigmoid(model(peptide, task, tissue, h2, flank, numeric)).cpu().numpy())
    return np.concatenate(output), float(model.processing_scale.detach().cpu())


def run(args: argparse.Namespace) -> None:
    torch, nn, _, _ = base.require_torch(); device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    train = validate(base.read_dataset(args.train), pd.read_csv(args.flank_features, keep_default_na=False)); train, _, mappings = base.add_task_columns(train, train.copy())
    if args.max_tasks:
        keep = mappings["tasks"][:args.max_tasks]; train = train[train.task_name.isin(keep)].copy(); train, _, mappings = base.add_task_columns(train, train.copy())
    assignment = folds.make_pair_grouped_folds(train, args.oof_folds, args.oof_split_seed); parts = []; scales = []
    for seed in args.seeds:
        for fold in range(args.oof_folds):
            base.set_seed(seed, torch); fitting, held_out = train[assignment != fold].copy(), train[assignment == fold].copy(); print(f"E26 seed={seed} fold={fold + 1}/{args.oof_folds} fit={len(fitting)} holdout={len(held_out)} device={device}", flush=True)
            scores, scale = fit_predict(args, torch, nn, fitting, held_out, mappings, seed, device); out = held_out[KEYS].copy(); out.insert(0, "split", "oof"); out.insert(1, "candidate", CANDIDATE); out.insert(2, "seed", seed); out["fold"] = fold; out["score"] = scores; parts.append(out); scales.append({"seed": seed, "fold": fold, "final_processing_scale": scale})
    predictions = pd.concat(parts, ignore_index=True)
    if predictions.duplicated(["seed", "sample_id"]).any() or len(predictions) != len(train) * len(args.seeds): raise AssertionError("E26 OOF coverage failed.")
    rows = []
    for (seed, tissue, h2), part in predictions.groupby(["seed", "target_tissue", "mhc_restriction"], sort=True): rows.append({"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "seed": seed, "target_tissue": tissue, "mhc_restriction": h2, "oof_rows": len(part), **base.evaluate(part.label.to_numpy(int), part.score.to_numpy(float))})
    per_task = pd.DataFrame(rows); summary = per_task.groupby("seed")[["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]].mean().add_prefix("mean_task_").reset_index(); summary.insert(0, "candidate", CANDIDATE); summary.insert(0, "experiment_name", EXPERIMENT); summary["worst6_task_auroc"] = [per_task[per_task.seed == seed].nsmallest(6, "auroc").auroc.mean() for seed in summary.seed]
    h2 = per_task.groupby(["seed", "mhc_restriction"])[["auroc", "auprc"]].mean().reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True); predictions.to_csv(args.output_dir / "mousePMHC_phase6_e26_oof_predictions.csv", index=False); per_task.to_csv(args.output_dir / "mousePMHC_phase6_e26_oof_per_task_metrics.csv", index=False); summary.to_csv(args.output_dir / "mousePMHC_phase6_e26_oof_summary_metrics.csv", index=False); h2.to_csv(args.output_dir / "mousePMHC_phase6_e26_oof_h2_metrics.csv", index=False); pd.DataFrame(scales).to_csv(args.output_dir / "mousePMHC_phase6_e26_processing_scales.csv", index=False)
    metadata = {"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "test_data_read": False, "train": str(args.train), "flank_features": str(args.flank_features), "n_rows": len(train), "n_tasks": len(mappings["tasks"]), "seeds": args.seeds, "oof_folds": args.oof_folds, "oof_split_seed": args.oof_split_seed, "device": device, "epochs": args.epochs, "training_note": "E3b OOF checkpoints do not exist; this is a same-protocol end-to-end retrain with a zero-initialized branch, not checkpoint fine-tuning."}
    (args.output_dir / "mousePMHC_phase6_e26_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"); print(summary.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/mousePMHC/mousePMHC_train.csv.gz")); parser.add_argument("--flank-features", type=Path, default=project_path("data/mousePMHC/mousePMHC_train_flank_features.csv.gz")); parser.add_argument("--output-dir", type=Path, default=project_path("results/mousePMHC_phase6_e26_flank_mmoe_oof")); parser.add_argument("--seeds", nargs="+", type=int, default=[20260704, 20260705, 20260706]); parser.add_argument("--oof-folds", type=int, default=3); parser.add_argument("--oof-split-seed", type=int, default=20260711); parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto"); parser.add_argument("--epochs", type=int, default=16); parser.add_argument("--steps-per-epoch", type=int, default=0); parser.add_argument("--task-batch-size", type=int, default=16); parser.add_argument("--batch-size", type=int, default=512); parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--embedding-dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=128); parser.add_argument("--expert-dim", type=int, default=64); parser.add_argument("--condition-dim", type=int, default=16); parser.add_argument("--gate-hidden-dim", type=int, default=64); parser.add_argument("--n-experts", type=int, default=3); parser.add_argument("--flank-embedding-dim", type=int, default=8); parser.add_argument("--processing-hidden-dim", type=int, default=32); parser.add_argument("--dropout", type=float, default=0.2); parser.add_argument("--gate-entropy-weight", type=float, default=0.01); parser.add_argument("--max-grad-norm", type=float, default=1.0); parser.add_argument("--max-tasks", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
