#!/usr/bin/env python3
"""Run mousePMHC Phase 3 E7: Recon-style H2-selective adapters, train-only OOF.

E7 keeps the successful E3b Factorized MMoE global-sharing backbone.  In each
seed and OOF fold, a short task-balanced probe measures pairwise cosine
similarity between H2-group gradients at the shared peptide encoder.  By
default, the predeclared H2-Kk residual adapter targets the stable Kk loss
observed in the completed E3b comparison; the audit is saved to test that
mechanistic premise.  ``--adapter-h2 auto`` instead selects the most
conflicting fitting-fold H2.  All other H2 groups use the unmodified shared
representation, and the fixed test split is never opened.
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
EXPERIMENT = "mousePMHC_phase3_e7_recon_h2_adapters_min200_oof"
CANDIDATE = "mousePMHC_phase3_e7_recon_h2_selective_adapter"
KEYS = ["sample_id", "target_tissue", "mhc_restriction"]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def define_model(torch: Any, nn: Any, peptide_length: int, n_tasks: int, n_tissues: int, n_h2: int,
                 adapted_h2_ids: set[int], args: argparse.Namespace) -> Any:
    """E3b MMoE plus residual adapters only for H2 groups selected by the audit."""
    class ReconH2AdapterMMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.amino_embedding = nn.Embedding(len(base.AA_TO_INDEX) + 1, args.embedding_dim, padding_idx=base.PAD_INDEX)
            self.peptide_encoder = nn.Sequential(
                nn.Flatten(), nn.Linear(peptide_length * args.embedding_dim, args.hidden_dim), nn.ReLU(), nn.Dropout(args.dropout),
            )
            self.h2_adapters = nn.ModuleDict({
                str(h2_id): nn.Sequential(nn.Linear(args.hidden_dim, args.adapter_dim), nn.ReLU(), nn.Linear(args.adapter_dim, args.hidden_dim))
                for h2_id in sorted(adapted_h2_ids)
            })
            self.tissue_embedding = nn.Embedding(n_tissues, args.condition_dim)
            self.mhc_embedding = nn.Embedding(n_h2, args.condition_dim)
            self.experts = nn.ModuleList([
                nn.Sequential(nn.Linear(args.hidden_dim, args.expert_dim), nn.ReLU(), nn.Dropout(args.dropout),
                              nn.Linear(args.expert_dim, args.expert_dim), nn.ReLU())
                for _ in range(args.n_experts)
            ])
            self.gate = nn.Sequential(nn.Linear(args.hidden_dim + 2 * args.condition_dim, args.gate_hidden_dim), nn.ReLU(),
                                      nn.Linear(args.gate_hidden_dim, args.n_experts))
            self.heads = nn.ModuleList([nn.Linear(args.expert_dim, 1) for _ in range(n_tasks)])

        def shared_representation(self, peptide_ids: Any, h2_ids: Any) -> Any:
            representation = self.peptide_encoder(self.amino_embedding(peptide_ids))
            for h2_key, adapter in self.h2_adapters.items():
                mask = h2_ids == int(h2_key)
                if mask.any():
                    representation = representation.clone()
                    representation[mask] = representation[mask] + adapter(representation[mask])
            return representation

        def forward(self, peptide_ids: Any, task_ids: Any, tissue_ids: Any, h2_ids: Any,
                    return_gates: bool = False) -> Any:
            peptide = self.shared_representation(peptide_ids, h2_ids)
            expert_outputs = torch.stack([expert(peptide) for expert in self.experts], dim=1)
            gate_input = torch.cat([peptide, self.tissue_embedding(tissue_ids), self.mhc_embedding(h2_ids)], dim=1)
            gates = torch.softmax(self.gate(gate_input), dim=1)
            mixed = (expert_outputs * gates.unsqueeze(-1)).sum(dim=1)
            logits = mixed.new_empty(mixed.shape[0])
            for task_id in torch.unique(task_ids):
                mask = task_ids == task_id
                logits[mask] = self.heads[int(task_id.item())](mixed[mask]).squeeze(-1)
            return (logits, gates) if return_gates else logits
    return ReconH2AdapterMMoE()


def h2_arrays(frame: pd.DataFrame, mappings: dict[str, Any], peptide_length: int) -> list[dict[str, Any]]:
    arrays = e5.task_arrays(frame, mappings, peptide_length)
    for item in arrays:
        task = frame[frame.task_name == item["task_name"]]
        item["h2_id"] = int(task.hla_id.iloc[0])
    return arrays


def h2_balanced_batch(rng: np.random.Generator, arrays: list[dict[str, Any]], h2_id: int, task_batch_size: int) -> tuple[np.ndarray, ...]:
    members = [item for item in arrays if item["h2_id"] == h2_id]
    return e5.sample_balanced_batch(rng, members, task_batch_size)


def train_probe(args: argparse.Namespace, torch: Any, nn: Any, arrays: list[dict[str, Any]], mappings: dict[str, Any],
                peptide_length: int, device: str, seed: int) -> Any:
    """Train a small E3b-compatible probe before its shared-layer gradient audit."""
    model = define_model(torch, nn, peptide_length, len(arrays), len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]), set(), args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed + 3001)
    for _ in range(args.audit_warmup_epochs):
        for _ in range(args.audit_warmup_steps):
            batch = e5.sample_balanced_batch(rng, arrays, args.audit_task_batch_size)
            peptide, task, tissue, h2, label = [torch.as_tensor(value, device=device) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            losses, _ = e5.task_loss_vector(torch, model, peptide, task, tissue, h2, label, len(arrays), args.audit_task_batch_size)
            losses.mean().backward(); optimizer.step()
    return model


def audit_h2_conflicts(args: argparse.Namespace, torch: Any, nn: Any, arrays: list[dict[str, Any]], mappings: dict[str, Any],
                       peptide_length: int, device: str, seed: int) -> tuple[pd.DataFrame, set[int]]:
    """Return an H2-gradient cosine matrix and the train-only selected adapter H2."""
    model = train_probe(args, torch, nn, arrays, mappings, peptide_length, device, seed)
    model.train(); rng = np.random.default_rng(seed + 6001)
    h2_ids = sorted({item["h2_id"] for item in arrays})
    gradients: dict[int, Any] = {}
    for h2_id in h2_ids:
        batch = h2_balanced_batch(rng, arrays, h2_id, args.audit_task_batch_size)
        peptide, task, tissue, h2, label = [torch.as_tensor(value, device=device) for value in batch]
        model.zero_grad(set_to_none=True)
        logits, _ = model(peptide, task, tissue, h2, return_gates=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label.float())
        loss.backward()
        gradients[h2_id] = torch.cat([parameter.grad.detach().flatten() for parameter in model.peptide_encoder.parameters() if parameter.grad is not None])
    h2_names = {int(value): key for key, value in mappings["hla_to_id"].items()}
    rows: list[dict[str, object]] = []
    mean_other: dict[int, float] = {}
    for source in h2_ids:
        values = []
        for target in h2_ids:
            cosine = float(torch.nn.functional.cosine_similarity(gradients[source], gradients[target], dim=0).cpu())
            rows.append({"source_h2": h2_names[source], "target_h2": h2_names[target], "source_h2_id": source,
                         "target_h2_id": target, "gradient_cosine": cosine})
            if source != target:
                values.append(cosine)
        mean_other[source] = float(np.mean(values))
    if args.adapter_h2 == "auto":
        worst_h2 = min(h2_ids, key=lambda h2_id: mean_other[h2_id])
        selected = {worst_h2} if mean_other[worst_h2] < args.audit_conflict_threshold else set()
    else:
        requested = {item.strip() for item in args.adapter_h2.split(",") if item.strip()}
        unknown = requested - set(mappings["hla_to_id"])
        if unknown:
            raise ValueError(f"Unknown --adapter-h2 values: {sorted(unknown)}")
        selected = {int(mappings["hla_to_id"][name]) for name in requested}
    table = pd.DataFrame(rows)
    table["source_mean_other_cosine"] = table.source_h2_id.map(mean_other)
    table["selected_for_adapter"] = table.source_h2_id.isin(selected)
    return table, selected


def train_fold(args: argparse.Namespace, torch: Any, nn: Any, fitting: pd.DataFrame, mappings: dict[str, Any],
               peptide_length: int, device: str, seed: int, fold: int, adapted_h2_ids: set[int]) -> Any:
    model = define_model(torch, nn, peptide_length, len(mappings["tasks"]), len(mappings["tissue_to_id"]),
                         len(mappings["hla_to_id"]), adapted_h2_ids, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    arrays = e5.task_arrays(fitting, mappings, peptide_length)
    steps = args.steps_per_epoch or int(np.ceil(max(len(item["label"]) for item in arrays) / args.task_batch_size))
    rng = np.random.default_rng(seed + 9001 + fold)
    for epoch in range(1, args.epochs + 1):
        model.train(); losses: list[float] = []; entropies: list[float] = []
        for _ in range(steps):
            batch = e5.sample_balanced_batch(rng, arrays, args.task_batch_size)
            peptide, task, tissue, h2, label = [torch.as_tensor(value, device=device) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            task_losses, gates = e5.task_loss_vector(torch, model, peptide, task, tissue, h2, label, len(arrays), args.task_batch_size)
            entropy = -(gates * torch.log(gates.clamp_min(1e-12))).sum(dim=1).mean()
            objective = task_losses.mean() - args.gate_entropy_weight * entropy
            objective.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step(); losses.append(float(task_losses.mean().detach().cpu())); entropies.append(float(entropy.detach().cpu()))
        print(f"E7 seed={seed} fold={fold + 1}/{args.oof_folds} epoch={epoch}/{args.epochs} task_balanced_bce={np.mean(losses):.4f} gate_entropy={np.mean(entropies):.4f} adapters={sorted(adapted_h2_ids)}", flush=True)
    return model


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for (seed, tissue, h2), task in predictions.groupby(["seed", "target_tissue", "mhc_restriction"], sort=True):
        records.append({"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "seed": int(seed), "target_tissue": tissue,
                        "mhc_restriction": h2, "oof_rows": len(task), **base.evaluate(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float))})
    per_task = pd.DataFrame(records); summaries: list[dict[str, object]] = []
    for seed, subset in per_task.groupby("seed", sort=True):
        row: dict[str, object] = {"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "seed": int(seed)}
        for metric in ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]:
            row[f"mean_task_{metric}"] = float(subset[metric].mean())
        row["worst_task_auroc"] = float(subset.auroc.min()); row["worst6_task_auroc"] = float(subset.nsmallest(6, "auroc").auroc.mean())
        summaries.append(row)
    summary = pd.DataFrame(summaries); stability = []
    for metric in [column for column in summary if column.startswith("mean_task_") or column.startswith("worst")]:
        stability.append({"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "metric": metric,
                          "seed_mean": float(summary[metric].mean()), "seed_sd": float(summary[metric].std(ddof=1)),
                          "seed_min": float(summary[metric].min()), "seed_max": float(summary[metric].max())})
    return per_task, summary, pd.DataFrame(stability)


def run(args: argparse.Namespace) -> None:
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    raw = base.read_dataset(args.train); e3.validate_input(raw)
    train, _, mappings = base.add_task_columns(raw, raw.copy())
    peptide_length = int(train.peptide_sequence.str.len().max())
    assignments = folds.make_pair_grouped_folds(train, args.oof_folds, args.oof_split_seed)
    prediction_parts: list[pd.DataFrame] = []; audit_parts: list[pd.DataFrame] = []; selection_rows: list[dict[str, object]] = []
    h2_names = {int(value): key for key, value in mappings["hla_to_id"].items()}
    for seed in args.seeds:
        for fold in range(args.oof_folds):
            fitting, held_out = train[assignments != fold].copy(), train[assignments == fold].copy()
            base.set_seed(int(seed), torch)
            print(f"E7 seed={seed} fold={fold + 1}/{args.oof_folds} fitting={len(fitting)} holdout={len(held_out)}: auditing H2 gradients", flush=True)
            audit, adapted_h2_ids = audit_h2_conflicts(args, torch, nn, h2_arrays(fitting, mappings, peptide_length), mappings,
                                                       peptide_length, device, int(seed) + fold * 101)
            audit.insert(0, "fold", fold); audit.insert(0, "seed", int(seed)); audit.insert(0, "candidate", CANDIDATE); audit.insert(0, "experiment_name", EXPERIMENT)
            audit_parts.append(audit)
            selection_rows.append({"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "seed": int(seed), "fold": fold,
                                   "adapter_h2": ",".join(h2_names[item] for item in sorted(adapted_h2_ids)) or "none",
                                   "n_adapted_h2": len(adapted_h2_ids)})
            print(f"E7 seed={seed} fold={fold + 1}: selected adapters={selection_rows[-1]['adapter_h2']}", flush=True)
            base.set_seed(int(seed), torch)
            model = train_fold(args, torch, nn, fitting, mappings, peptide_length, device, int(seed), fold, adapted_h2_ids)
            scores = e5.predict(torch, DataLoader, TensorDataset, model, held_out, peptide_length, args.batch_size, device)
            output = held_out[KEYS + ["label"]].copy()
            output.insert(0, "split", "oof"); output.insert(1, "candidate", CANDIDATE); output.insert(2, "seed", int(seed)); output["score"] = scores
            prediction_parts.append(output[["split", "candidate", "seed", *KEYS, "label", "score"]])
    predictions = pd.concat(prediction_parts, ignore_index=True)
    if len(predictions) != len(train) * len(args.seeds) or predictions.duplicated(["seed", "sample_id"]).any():
        raise AssertionError("E7 OOF predictions must cover every training row exactly once per seed.")
    per_task, summary, stability = metric_tables(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "mousePMHC_phase3_e7_oof_predictions.csv", index=False)
    per_task.to_csv(args.output_dir / "mousePMHC_phase3_e7_oof_per_task_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "mousePMHC_phase3_e7_oof_summary_metrics.csv", index=False)
    stability.to_csv(args.output_dir / "mousePMHC_phase3_e7_oof_stability_metrics.csv", index=False)
    pd.concat(audit_parts, ignore_index=True).to_csv(args.output_dir / "mousePMHC_phase3_e7_h2_gradient_audit.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(args.output_dir / "mousePMHC_phase3_e7_adapter_selection.csv", index=False)
    metadata = {"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "test_data_read": False,
                "backbone": "E3b task-balanced Factorized MMoE", "method": "Recon-style H2 gradient audit plus selective residual adapter",
                "train": str(args.train), "n_rows": len(train), "n_pairs": int(train.pair_id.nunique()), "n_tasks": len(mappings["tasks"]),
                "seeds": args.seeds, "oof_folds": args.oof_folds, "oof_split_seed": args.oof_split_seed, "device": device,
                "adapter_h2": args.adapter_h2, "audit_conflict_threshold": args.audit_conflict_threshold,
                "audit_warmup_epochs": args.audit_warmup_epochs, "audit_warmup_steps": args.audit_warmup_steps,
                "audit_task_batch_size": args.audit_task_batch_size, "adapter_dim": args.adapter_dim,
                "selection_protocol": "gradient audit re-estimated in every seed-fold fitting partition; default H2-Kk adapter was predeclared from completed E3b OOF diagnosis; OOF and fixed-test rows are excluded"}
    (args.output_dir / "mousePMHC_phase3_e7_oof_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False), flush=True); print(stability.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=project_path("results/mousePMHC_phase3_e7_recon_h2_adapters_min200_oof"))
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
    parser.add_argument("--adapter-dim", type=int, default=16)
    parser.add_argument("--adapter-h2", default="H2-Kk", help="comma-separated H2 names (default: H2-Kk), or auto for fold-specific audit selection")
    parser.add_argument("--audit-conflict-threshold", type=float, default=0.0)
    parser.add_argument("--audit-warmup-epochs", type=int, default=3); parser.add_argument("--audit-warmup-steps", type=int, default=12)
    parser.add_argument("--audit-task-batch-size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
