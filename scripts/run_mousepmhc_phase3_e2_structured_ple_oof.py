#!/usr/bin/env python3
"""Run mousePMHC E2: train-only OOF Mouse-Structured PLE.

E2 is intentionally small because the fixed min500pairs cohort has only five
tasks and roughly 413-470 training pairs per task.  It combines two global
experts with one tissue, one H2, and one task-private expert.  Gates are
conditioned on peptide, tissue, H2, and task representations.  The script
never opens the fixed test split.
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
EXPERIMENT = "mousePMHC_phase3_e2_structured_ple_oof"
CANDIDATE = "mousePMHC_phase3_e2_mouse_structured_ple"
KEYS = ["sample_id", "target_tissue", "mhc_restriction"]
GATE_COLUMNS = [
    "experiment_name", "candidate", "seed", "fold", "epoch", "task_name",
    "target_tissue", "mhc_restriction", "route", "gate_weight",
]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def validate_input(frame: pd.DataFrame) -> None:
    required = {*KEYS, "dataset", "split", "pair_id", "label", "peptide_sequence"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if set(frame.dataset) != {"mousePMHC"} or set(frame.split) != {"train"}:
        raise ValueError("E2 accepts only mousePMHC training rows.")
    if not frame.mhc_restriction.str.startswith("H2-").all():
        raise ValueError("E2 found a non-H2 MHC restriction.")
    if not frame.peptide_sequence.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{9}").all():
        raise ValueError("E2 requires standard unmodified 9-mer peptides.")


def make_arrays(frame: pd.DataFrame, peptide_length: int) -> list[np.ndarray]:
    return [
        base.encode_peptides(frame.peptide_sequence, peptide_length).copy(),
        frame.task_id.to_numpy(dtype=np.int64, copy=True),
        frame.tissue_id.to_numpy(dtype=np.int64, copy=True),
        frame.hla_id.to_numpy(dtype=np.int64, copy=True),
        frame.label.to_numpy(dtype=np.int64, copy=True),
    ]


def define_model(
    torch: Any,
    nn: Any,
    peptide_length: int,
    n_tasks: int,
    n_tissues: int,
    n_mhcs: int,
    args: argparse.Namespace,
) -> Any:
    class MouseStructuredPLE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.amino_embedding = nn.Embedding(len(base.AA_TO_INDEX) + 1, args.embedding_dim, padding_idx=base.PAD_INDEX)
            self.peptide_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(peptide_length * args.embedding_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
            )
            self.tissue_embedding = nn.Embedding(n_tissues, args.condition_dim)
            self.mhc_embedding = nn.Embedding(n_mhcs, args.condition_dim)
            self.task_embedding = nn.Embedding(n_tasks, args.condition_dim)

            def expert() -> Any:
                return nn.Sequential(
                    nn.Linear(args.hidden_dim, args.expert_dim), nn.ReLU(),
                    nn.Dropout(args.dropout), nn.Linear(args.expert_dim, args.expert_dim), nn.ReLU(),
                )

            self.global_experts = nn.ModuleList([expert() for _ in range(args.n_global_experts)])
            self.tissue_experts = nn.ModuleList([expert() for _ in range(n_tissues)])
            self.mhc_experts = nn.ModuleList([expert() for _ in range(n_mhcs)])
            self.task_experts = nn.ModuleList([expert() for _ in range(n_tasks)])
            route_count = args.n_global_experts + 3
            self.gate = nn.Sequential(
                nn.Linear(args.hidden_dim + 3 * args.condition_dim, args.gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.gate_hidden_dim, route_count),
            )
            self.heads = nn.ModuleList([nn.Linear(args.expert_dim, 1) for _ in range(n_tasks)])

        def _apply_experts(self, experts: Any, shared: Any, ids: Any) -> Any:
            result = shared.new_empty((shared.shape[0], args.expert_dim))
            for expert_id in torch.unique(ids):
                mask = ids == expert_id
                result[mask] = experts[int(expert_id.item())](shared[mask])
            return result

        def forward(self, peptide_ids: Any, task_ids: Any, tissue_ids: Any, mhc_ids: Any, return_gates: bool = False):
            shared = self.peptide_encoder(self.amino_embedding(peptide_ids))
            global_outputs = [expert(shared) for expert in self.global_experts]
            tissue_output = self._apply_experts(self.tissue_experts, shared, tissue_ids)
            mhc_output = self._apply_experts(self.mhc_experts, shared, mhc_ids)
            task_output = self._apply_experts(self.task_experts, shared, task_ids)
            routes = torch.stack([*global_outputs, tissue_output, mhc_output, task_output], dim=1)
            gate_input = torch.cat([
                shared, self.tissue_embedding(tissue_ids), self.mhc_embedding(mhc_ids), self.task_embedding(task_ids),
            ], dim=1)
            gates = torch.softmax(self.gate(gate_input), dim=1)
            mixed = (routes * gates.unsqueeze(-1)).sum(dim=1)
            logits = mixed.new_empty(mixed.shape[0])
            for task_id in torch.unique(task_ids):
                mask = task_ids == task_id
                logits[mask] = self.heads[int(task_id.item())](mixed[mask]).squeeze(-1)
            return (logits, gates) if return_gates else logits

    return MouseStructuredPLE()


def build_loader(torch: Any, DataLoader: Any, TensorDataset: Any, frame: pd.DataFrame, peptide_length: int, batch_size: int, shuffle: bool):
    return base.build_loader(torch, DataLoader, TensorDataset, make_arrays(frame, peptide_length), batch_size, shuffle)


def task_balanced_bce(torch: Any, logits: Any, labels: Any, task_ids: Any) -> Any:
    per_example = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    task_losses = [per_example[task_ids == task_id].mean() for task_id in torch.unique(task_ids)]
    return torch.stack(task_losses).mean()


def gate_diagnostics(model: Any, frame: pd.DataFrame, torch: Any, device: str, peptide_length: int, DataLoader: Any, TensorDataset: Any, batch_size: int, seed: int, fold: int, epoch: int) -> list[dict[str, object]]:
    loader = build_loader(torch, DataLoader, TensorDataset, frame, peptide_length, batch_size, False)
    route_sum: dict[int, np.ndarray] = {}
    route_count: dict[int, int] = {}
    model.eval()
    with torch.no_grad():
        for peptide, task, tissue, mhc, _ in loader:
            peptide, task, tissue, mhc = [x.to(device) for x in (peptide, task, tissue, mhc)]
            _, gates = model(peptide, task, tissue, mhc, return_gates=True)
            for task_id in torch.unique(task):
                mask = task == task_id
                numeric_id = int(task_id.item())
                values = gates[mask].sum(dim=0).cpu().numpy()
                route_sum[numeric_id] = route_sum.get(numeric_id, np.zeros_like(values)) + values
                route_count[numeric_id] = route_count.get(numeric_id, 0) + int(mask.sum().item())
    routes = [*(f"global_{index}" for index in range(model.n_global_experts if hasattr(model, "n_global_experts") else 2)), "tissue", "mhc", "task"]
    # The names are deterministic from the gate width; do not infer a biological ordering from weights.
    route_count_total = next(iter(route_sum.values())).shape[0] if route_sum else 0
    routes = [*(f"global_{index}" for index in range(route_count_total - 3)), "tissue", "mhc", "task"]
    rows: list[dict[str, object]] = []
    task_lookup = frame[["task_id", "task_name", "target_tissue", "mhc_restriction"]].drop_duplicates().set_index("task_id")
    for task_id, totals in sorted(route_sum.items()):
        info = task_lookup.loc[task_id]
        for route, weight in zip(routes, totals / route_count[task_id]):
            rows.append({
                "experiment_name": EXPERIMENT, "candidate": CANDIDATE, "seed": seed, "fold": fold, "epoch": epoch,
                "task_name": info.task_name, "target_tissue": info.target_tissue,
                "mhc_restriction": info.mhc_restriction, "route": route, "gate_weight": float(weight),
            })
    return rows


def train_fold(args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any, fitting: pd.DataFrame, mappings: dict[str, Any], peptide_length: int, device: str, fold: int):
    model = define_model(torch, nn, peptide_length, len(mappings["tasks"]), len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]), args).to(device)
    # Stored for route diagnostic labels, not used as a model feature.
    model.n_global_experts = args.n_global_experts
    loader = build_loader(torch, DataLoader, TensorDataset, fitting, peptide_length, args.batch_size, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for peptide, task, tissue, mhc, label in loader:
            peptide, task, tissue, mhc, label = [x.to(device) for x in (peptide, task, tissue, mhc, label)]
            optimizer.zero_grad(set_to_none=True)
            logits = model(peptide, task, tissue, mhc)
            loss = task_balanced_bce(torch, logits, label, task)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == args.epochs or epoch % args.gate_diagnostic_interval == 0:
            history.extend(gate_diagnostics(model, fitting, torch, device, peptide_length, DataLoader, TensorDataset, args.batch_size, args.seed, fold, epoch))
        print(f"E2 fold={fold + 1}/{args.oof_folds} epoch={epoch}/{args.epochs} task_balanced_bce={np.mean(losses):.4f}", flush=True)
    return model, history


def predict_fold(torch: Any, DataLoader: Any, TensorDataset: Any, model: Any, held_out: pd.DataFrame, peptide_length: int, batch_size: int, device: str) -> np.ndarray:
    loader = build_loader(torch, DataLoader, TensorDataset, held_out, peptide_length, batch_size, False)
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for peptide, task, tissue, mhc, _ in loader:
            peptide, task, tissue, mhc = [x.to(device) for x in (peptide, task, tissue, mhc)]
            scores.append(torch.sigmoid(model(peptide, task, tissue, mhc)).cpu().numpy())
    return np.concatenate(scores)


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (tissue, mhc), task in predictions.groupby(["target_tissue", "mhc_restriction"], sort=True):
        rows.append({
            "experiment_name": EXPERIMENT, "candidate": CANDIDATE, "target_tissue": tissue, "mhc_restriction": mhc,
            "oof_rows": len(task), **base.evaluate(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float)),
        })
    per_task = pd.DataFrame(rows)
    summary = {"experiment_name": EXPERIMENT, "candidate": CANDIDATE}
    for metric in ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]:
        summary[f"mean_task_{metric}"] = float(per_task[metric].mean())
    summary["worst_task_auroc"] = float(per_task.auroc.min())
    return per_task, pd.DataFrame([summary])


def run(args: argparse.Namespace) -> None:
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    raw = base.read_dataset(args.train)
    validate_input(raw)
    train, _, mappings = base.add_task_columns(raw, raw.copy())
    if len(mappings["tasks"]) != 5 and not args.allow_nonfive_task_smoke:
        raise ValueError(f"E2 is registered for the fixed five-task cohort, found {len(mappings['tasks'])} tasks.")
    peptide_length = int(train.peptide_sequence.str.len().max())
    assignments = folds.make_pair_grouped_folds(train, args.oof_folds, args.oof_split_seed)
    parts: list[pd.DataFrame] = []
    gate_rows: list[dict[str, object]] = []
    for fold in range(args.oof_folds):
        fitting = train[assignments != fold].copy()
        held_out = train[assignments == fold].copy()
        base.set_seed(args.seed, torch)
        print(f"mousePMHC E2 OOF fold={fold + 1}/{args.oof_folds} fit={len(fitting)} holdout={len(held_out)} device={device}", flush=True)
        model, history = train_fold(args, torch, nn, DataLoader, TensorDataset, fitting, mappings, peptide_length, device, fold)
        scores = predict_fold(torch, DataLoader, TensorDataset, model, held_out, peptide_length, args.batch_size, device)
        if len(scores) != len(held_out):
            raise AssertionError("E2 prediction count does not match OOF holdout rows.")
        output = held_out[KEYS + ["label"]].copy()
        output.insert(0, "split", "oof")
        output.insert(1, "candidate", CANDIDATE)
        output.insert(2, "seed", args.seed)
        output["score"] = scores
        parts.append(output[["split", "candidate", "seed", *KEYS, "label", "score"]])
        gate_rows.extend(history)

    predictions = pd.concat(parts, ignore_index=True)
    if len(predictions) != len(train) or set(predictions.sample_id) != set(train.sample_id) or predictions.sample_id.duplicated().any():
        raise AssertionError("E2 OOF predictions must cover every training sample exactly once.")
    per_task, summary = metric_tables(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "mousePMHC_phase3_e2_oof_predictions.csv", index=False)
    per_task.to_csv(args.output_dir / "mousePMHC_phase3_e2_oof_per_task_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "mousePMHC_phase3_e2_oof_summary_metrics.csv", index=False)
    pd.DataFrame(gate_rows, columns=GATE_COLUMNS).to_csv(args.output_dir / "mousePMHC_phase3_e2_gate_weight_history.csv", index=False)
    metadata = {
        "experiment_name": EXPERIMENT, "candidate": CANDIDATE, "mouse_experiment_id": "E2",
        "species": "Mus musculus", "mhc_system": "H2-I", "test_data_read": False,
        "train": str(args.train), "n_rows": len(train), "n_pairs": int(train.pair_id.nunique()),
        "n_tasks": len(mappings["tasks"]), "tasks": mappings["tasks"], "n_tissues": len(mappings["tissue_to_id"]),
        "n_mhcs": len(mappings["hla_to_id"]), "seed": args.seed, "oof_folds": args.oof_folds,
        "oof_split_seed": args.oof_split_seed, "device": device, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim, "expert_dim": args.expert_dim, "condition_dim": args.condition_dim,
        "gate_hidden_dim": args.gate_hidden_dim, "dropout": args.dropout, "n_global_experts": args.n_global_experts,
        "routing": "two global experts + matching tissue expert + matching H2 expert + matching task-private expert",
        "gate_input": "peptide representation + tissue embedding + H2 embedding + task embedding",
        "loss": "mean of per-task BCE values present in each batch", "method_reference": "Tang et al. (2020), PLE; project-specific structured adaptation",
        "naming_policy": "mousePMHC_phase3_<experiment>; human tissuePMHC outputs are never overwritten",
    }
    (args.output_dir / "mousePMHC_phase3_e2_oof_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=project_path("results/mousePMHC_phase3_e2_oof"))
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--oof-folds", type=int, default=3)
    parser.add_argument("--oof-split-seed", type=int, default=20260711)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--expert-dim", type=int, default=64)
    parser.add_argument("--condition-dim", type=int, default=16)
    parser.add_argument("--gate-hidden-dim", type=int, default=64)
    parser.add_argument("--n-global-experts", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gate-diagnostic-interval", type=int, default=5)
    parser.add_argument("--allow-nonfive-task-smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
