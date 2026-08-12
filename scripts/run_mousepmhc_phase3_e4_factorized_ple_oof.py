#!/usr/bin/env python3
"""Run mousePMHC Phase 3 E4: min200 train-only OOF Factorized PLE-lite.

E4 keeps one global peptide expert and adds compact tissue- and H2-specific
experts. A peptide/tissue/H2-conditioned gate mixes exactly these routes for
each sample before a task-specific linear head. There are deliberately no
task-private experts: several current tasks have only 124 training pairs.
The fixed test split is never opened.
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
EXPERIMENT = "mousePMHC_phase3_e4_factorized_ple_min200_oof"
CANDIDATE = "mousePMHC_phase3_e4_factorized_ple_min200"
KEYS = ["sample_id", "target_tissue", "mhc_restriction"]
GATE_COLUMNS = [
    "experiment_name", "candidate", "seed", "fold", "epoch", "task_name",
    "target_tissue", "mhc_restriction", "route", "gate_weight",
]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def validate_input(frame: pd.DataFrame) -> None:
    if set(frame.dataset) != {"mousePMHC"} or set(frame.split) != {"train"}:
        raise ValueError("E4 accepts mousePMHC training rows only.")
    if not frame.mhc_restriction.str.startswith("H2-").all():
        raise ValueError("E4 found a non-H2 restriction.")
    if not frame.peptide_sequence.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{9}").all():
        raise ValueError("E4 requires standard unmodified 9-mer peptides.")


def arrays(frame: pd.DataFrame, peptide_length: int) -> list[np.ndarray]:
    return [
        base.encode_peptides(frame.peptide_sequence, peptide_length).copy(),
        frame.task_id.to_numpy(dtype=np.int64, copy=True),
        frame.tissue_id.to_numpy(dtype=np.int64, copy=True),
        frame.hla_id.to_numpy(dtype=np.int64, copy=True),
        frame.label.to_numpy(dtype=np.int64, copy=True),
    ]


def define_model(torch: Any, nn: Any, peptide_length: int, n_tasks: int, n_tissues: int,
                 n_h2: int, args: argparse.Namespace) -> Any:
    class FactorizedPLELite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.amino_embedding = nn.Embedding(len(base.AA_TO_INDEX) + 1, args.embedding_dim,
                                                padding_idx=base.PAD_INDEX)
            self.peptide_encoder = nn.Sequential(
                nn.Flatten(), nn.Linear(peptide_length * args.embedding_dim, args.hidden_dim),
                nn.ReLU(), nn.Dropout(args.dropout),
            )
            self.tissue_embedding = nn.Embedding(n_tissues, args.condition_dim)
            self.h2_embedding = nn.Embedding(n_h2, args.condition_dim)

            def expert() -> Any:
                return nn.Sequential(
                    nn.Linear(args.hidden_dim, args.expert_dim), nn.ReLU(), nn.Dropout(args.dropout),
                    nn.Linear(args.expert_dim, args.expert_dim), nn.ReLU(),
                )

            self.global_expert = expert()
            self.tissue_experts = nn.ModuleList([expert() for _ in range(n_tissues)])
            self.h2_experts = nn.ModuleList([expert() for _ in range(n_h2)])
            self.gate = nn.Sequential(
                nn.Linear(args.hidden_dim + 2 * args.condition_dim, args.gate_hidden_dim), nn.ReLU(),
                nn.Linear(args.gate_hidden_dim, 3),
            )
            self.heads = nn.ModuleList([nn.Linear(args.expert_dim, 1) for _ in range(n_tasks)])

        @staticmethod
        def routed_expert(peptide: Any, identifiers: Any, experts: Any) -> Any:
            output = peptide.new_empty((peptide.shape[0], experts[0][-2].out_features))
            for identifier in torch.unique(identifiers):
                mask = identifiers == identifier
                output[mask] = experts[int(identifier.item())](peptide[mask])
            return output

        def forward(self, peptide_ids: Any, task_ids: Any, tissue_ids: Any, h2_ids: Any,
                    return_gates: bool = False) -> Any:
            peptide = self.peptide_encoder(self.amino_embedding(peptide_ids))
            global_route = self.global_expert(peptide)
            tissue_route = self.routed_expert(peptide, tissue_ids, self.tissue_experts)
            h2_route = self.routed_expert(peptide, h2_ids, self.h2_experts)
            routes = torch.stack([global_route, tissue_route, h2_route], dim=1)
            gate_input = torch.cat([peptide, self.tissue_embedding(tissue_ids), self.h2_embedding(h2_ids)], dim=1)
            gates = torch.softmax(self.gate(gate_input), dim=1)
            mixed = (routes * gates.unsqueeze(-1)).sum(dim=1)
            logits = mixed.new_empty(mixed.shape[0])
            for task_id in torch.unique(task_ids):
                mask = task_ids == task_id
                logits[mask] = self.heads[int(task_id.item())](mixed[mask]).squeeze(-1)
            return (logits, gates) if return_gates else logits

    return FactorizedPLELite()


def build_loader(torch: Any, DataLoader: Any, TensorDataset: Any, frame: pd.DataFrame,
                 peptide_length: int, batch_size: int, shuffle: bool) -> Any:
    return base.build_loader(torch, DataLoader, TensorDataset, arrays(frame, peptide_length), batch_size, shuffle)


def task_balanced_bce(torch: Any, logits: Any, labels: Any, task_ids: Any) -> Any:
    per_example = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    return torch.stack([per_example[task_ids == task_id].mean() for task_id in torch.unique(task_ids)]).mean()


def collect_gate_rows(model: Any, frame: pd.DataFrame, torch: Any, DataLoader: Any,
                      TensorDataset: Any, peptide_length: int, batch_size: int, device: str,
                      seed: int, fold: int, epoch: int) -> list[dict[str, object]]:
    totals: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    model.eval()
    loader = build_loader(torch, DataLoader, TensorDataset, frame, peptide_length, batch_size, False)
    with torch.no_grad():
        for peptide, task, tissue, h2, _ in loader:
            peptide, task, tissue, h2 = [value.to(device) for value in (peptide, task, tissue, h2)]
            _, gates = model(peptide, task, tissue, h2, return_gates=True)
            for task_id in torch.unique(task):
                mask = task == task_id
                numeric_id = int(task_id.item())
                totals[numeric_id] = totals.get(numeric_id, np.zeros(3)) + gates[mask].sum(dim=0).cpu().numpy()
                counts[numeric_id] = counts.get(numeric_id, 0) + int(mask.sum().item())
    lookup = frame[["task_id", "task_name", "target_tissue", "mhc_restriction"]].drop_duplicates().set_index("task_id")
    rows: list[dict[str, object]] = []
    for task_id, total in sorted(totals.items()):
        info = lookup.loc[task_id]
        for route, weight in zip(["global", "tissue", "h2"], total / counts[task_id], strict=True):
            rows.append({"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "seed": seed, "fold": fold,
                         "epoch": epoch, "task_name": info.task_name, "target_tissue": info.target_tissue,
                         "mhc_restriction": info.mhc_restriction, "route": route, "gate_weight": float(weight)})
    return rows


def train_fold(args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any,
               fitting: pd.DataFrame, mappings: dict[str, Any], peptide_length: int, device: str, fold: int) -> tuple[Any, list[dict[str, object]]]:
    model = define_model(torch, nn, peptide_length, len(mappings["tasks"]), len(mappings["tissue_to_id"]),
                         len(mappings["hla_to_id"]), args).to(device)
    loader = build_loader(torch, DataLoader, TensorDataset, fitting, peptide_length, args.batch_size, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses: list[float] = []; entropies: list[float] = []
        for peptide, task, tissue, h2, label in loader:
            peptide, task, tissue, h2, label = [value.to(device) for value in (peptide, task, tissue, h2, label)]
            optimizer.zero_grad(set_to_none=True)
            logits, gates = model(peptide, task, tissue, h2, return_gates=True)
            bce = task_balanced_bce(torch, logits, label, task)
            entropy = -(gates * torch.log(gates.clamp_min(1e-12))).sum(dim=1).mean()
            loss = bce - args.gate_entropy_weight * entropy
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(bce.detach().cpu())); entropies.append(float(entropy.detach().cpu()))
        if epoch == 1 or epoch == args.epochs or epoch % args.gate_diagnostic_interval == 0:
            history.extend(collect_gate_rows(model, fitting, torch, DataLoader, TensorDataset, peptide_length,
                                             args.batch_size, device, args.seed, fold, epoch))
        print(f"E4 fold={fold + 1}/{args.oof_folds} epoch={epoch}/{args.epochs} task_balanced_bce={np.mean(losses):.4f} gate_entropy={np.mean(entropies):.4f}", flush=True)
    return model, history


def predict_fold(torch: Any, DataLoader: Any, TensorDataset: Any, model: Any, held_out: pd.DataFrame,
                 peptide_length: int, batch_size: int, device: str) -> np.ndarray:
    loader = build_loader(torch, DataLoader, TensorDataset, held_out, peptide_length, batch_size, False)
    scores: list[np.ndarray] = []; model.eval()
    with torch.no_grad():
        for peptide, task, tissue, h2, _ in loader:
            peptide, task, tissue, h2 = [value.to(device) for value in (peptide, task, tissue, h2)]
            scores.append(torch.sigmoid(model(peptide, task, tissue, h2)).cpu().numpy())
    return np.concatenate(scores)


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (tissue, h2), task in predictions.groupby(["target_tissue", "mhc_restriction"], sort=True):
        rows.append({"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "target_tissue": tissue,
                     "mhc_restriction": h2, "oof_rows": len(task),
                     **base.evaluate(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float))})
    per_task = pd.DataFrame(rows)
    summary = {"experiment_name": EXPERIMENT, "candidate": CANDIDATE}
    for metric in ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]:
        summary[f"mean_task_{metric}"] = float(per_task[metric].mean())
    summary["worst_task_auroc"] = float(per_task.auroc.min())
    return per_task, pd.DataFrame([summary])


def run(args: argparse.Namespace) -> None:
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    raw = base.read_dataset(args.train); validate_input(raw)
    train, _, mappings = base.add_task_columns(raw, raw.copy())
    if len(mappings["tasks"]) < 2:
        raise ValueError("E4 requires at least two tissue-H2 tasks.")
    peptide_length = int(train.peptide_sequence.str.len().max())
    assignments = folds.make_pair_grouped_folds(train, args.oof_folds, args.oof_split_seed)
    prediction_parts: list[pd.DataFrame] = []; gate_rows: list[dict[str, object]] = []
    for fold in range(args.oof_folds):
        fitting, held_out = train[assignments != fold].copy(), train[assignments == fold].copy()
        base.set_seed(args.seed, torch)
        print(f"mousePMHC E4 OOF fold={fold + 1}/{args.oof_folds} fit={len(fitting)} holdout={len(held_out)} device={device}", flush=True)
        model, history = train_fold(args, torch, nn, DataLoader, TensorDataset, fitting, mappings, peptide_length, device, fold)
        scores = predict_fold(torch, DataLoader, TensorDataset, model, held_out, peptide_length, args.batch_size, device)
        output = held_out[KEYS + ["label"]].copy()
        output.insert(0, "split", "oof"); output.insert(1, "candidate", CANDIDATE); output.insert(2, "seed", args.seed)
        output["score"] = scores
        prediction_parts.append(output[["split", "candidate", "seed", *KEYS, "label", "score"]]); gate_rows.extend(history)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    if len(predictions) != len(train) or predictions.sample_id.duplicated().any() or set(predictions.sample_id) != set(train.sample_id):
        raise AssertionError("E4 OOF predictions must cover every training sample exactly once.")
    per_task, summary = metric_tables(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "mousePMHC_phase3_e4_oof_predictions.csv", index=False)
    per_task.to_csv(args.output_dir / "mousePMHC_phase3_e4_oof_per_task_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "mousePMHC_phase3_e4_oof_summary_metrics.csv", index=False)
    pd.DataFrame(gate_rows, columns=GATE_COLUMNS).to_csv(args.output_dir / "mousePMHC_phase3_e4_gate_weight_history.csv", index=False)
    metadata = {
        "experiment_name": EXPERIMENT, "candidate": CANDIDATE, "mouse_experiment_id": "E4",
        "species": "Mus musculus", "mhc_system": "H2-I", "test_data_read": False, "train": str(args.train),
        "n_rows": len(train), "n_pairs": int(train.pair_id.nunique()), "n_tasks": len(mappings["tasks"]),
        "n_tissues": len(mappings["tissue_to_id"]), "n_h2": len(mappings["hla_to_id"]), "seed": args.seed,
        "oof_folds": args.oof_folds, "oof_split_seed": args.oof_split_seed, "device": device, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "embedding_dim": args.embedding_dim, "hidden_dim": args.hidden_dim, "expert_dim": args.expert_dim,
        "condition_dim": args.condition_dim, "gate_hidden_dim": args.gate_hidden_dim, "dropout": args.dropout,
        "gate_entropy_weight": args.gate_entropy_weight,
        "routes": ["one global expert", "one matched tissue expert", "one matched H2 expert"],
        "private_task_experts": False, "loss": "mean of per-task BCE values present in each batch",
        "comparison_baseline": "mousePMHC_phase3_e3_factorized_mmoe_min200",
    }
    (args.output_dir / "mousePMHC_phase3_e4_oof_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=project_path("results/mousePMHC_phase3_e4_factorized_ple_min200_oof"))
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--oof-folds", type=int, default=3); parser.add_argument("--oof-split-seed", type=int, default=20260711)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--expert-dim", type=int, default=32); parser.add_argument("--condition-dim", type=int, default=16)
    parser.add_argument("--gate-hidden-dim", type=int, default=48); parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.005)
    parser.add_argument("--max-grad-norm", type=float, default=1.0); parser.add_argument("--gate-diagnostic-interval", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
