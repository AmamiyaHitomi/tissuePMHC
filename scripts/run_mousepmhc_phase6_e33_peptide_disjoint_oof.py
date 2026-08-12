#!/usr/bin/env python3
"""Run frozen E15 under three-fold peptide-disjoint train-only OOF.

Pairs are atomic.  Because each pair contains two peptides and a peptide can
occur in multiple pairs/tasks, folds are assigned to connected components of
the bipartite pair--peptide graph.  This guarantees both pair integrity and
zero peptide overlap between fitting and held-out partitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_mousepmhc_phase4_e15_five_seed_confirmation as e15
import run_tissuepmhc_neural_baselines_v2 as base


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mousePMHC_phase6_e33_peptide_disjoint_oof"
CANDIDATE = "mousePMHC_phase6_e33_frozen_e15_5seed_probability_mean"
MEMBER = "mousePMHC_phase6_e33_frozen_e3b_member"
SEEDS = [20260704, 20260705, 20260706, 20260707, 20260708]
KEYS = ["sample_id", "pair_id", "target_tissue", "mhc_restriction", "peptide_sequence", "label"]
METRICS = ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]


def path(relative: str) -> Path:
    return ROOT / relative


def sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DSU:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def component_table(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = train[["pair_id", "task_name"]].drop_duplicates()
    if pairs.pair_id.duplicated().any():
        raise AssertionError("A pair_id occurs in more than one task.")
    pair_peptides = train[["pair_id", "peptide_sequence"]].drop_duplicates()
    dsu = DSU(pair_peptides.pair_id.astype(str).unique().tolist())
    for _, group in pair_peptides.groupby("peptide_sequence", sort=False):
        ids = group.pair_id.astype(str).tolist()
        for pair_id in ids[1:]:
            dsu.union(ids[0], pair_id)
    pairs = pairs.copy()
    pairs["pair_id"] = pairs.pair_id.astype(str)
    pairs["component_id"] = pairs.pair_id.map(dsu.find)
    component_sizes = pairs.groupby("component_id").size().rename("n_pairs")
    component_tasks = pairs.groupby(["component_id", "task_name"]).size().rename("n_pairs_task").reset_index()
    components = component_sizes.reset_index()
    return pairs, components.merge(
        component_tasks.groupby("component_id").size().rename("n_tasks").reset_index(), on="component_id"
    )


def assign_components(pairs: pd.DataFrame, components: pd.DataFrame, n_folds: int, seed: int) -> dict[str, int]:
    tasks = sorted(pairs.task_name.unique())
    task_to_id = {task: index for index, task in enumerate(tasks)}
    target = pairs.groupby("task_name").size().reindex(tasks).to_numpy(dtype=float) / n_folds
    vectors: dict[str, np.ndarray] = {}
    for component_id, group in pairs.groupby("component_id", sort=False):
        vector = np.zeros(len(tasks), dtype=float)
        counts = group.groupby("task_name").size()
        for task, count in counts.items():
            vector[task_to_id[task]] = count
        vectors[component_id] = vector
    rng = np.random.default_rng(seed)
    order = components.copy()
    order["tie"] = rng.random(len(order))
    order = order.sort_values(["n_pairs", "n_tasks", "tie"], ascending=[False, False, True])
    fold_counts = np.zeros((n_folds, len(tasks)), dtype=float)
    fold_totals = np.zeros(n_folds, dtype=float)
    assignments: dict[str, int] = {}
    for component_id in order.component_id:
        vector = vectors[component_id]
        scores = []
        for fold in range(n_folds):
            candidate = fold_counts.copy()
            candidate[fold] += vector
            task_imbalance = np.mean(((candidate - target) / np.maximum(target, 1.0)) ** 2)
            totals = fold_totals.copy()
            totals[fold] += vector.sum()
            total_imbalance = np.var(totals / max(pairs.shape[0] / n_folds, 1.0))
            scores.append(task_imbalance + 0.05 * total_imbalance)
        best_score = min(scores)
        eligible = [fold for fold, score in enumerate(scores) if abs(score - best_score) < 1e-12]
        fold = int(rng.choice(eligible))
        assignments[component_id] = fold
        fold_counts[fold] += vector
        fold_totals[fold] += vector.sum()
    return assignments


def make_folds(train: pd.DataFrame, n_folds: int, seed: int) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    pairs, components = component_table(train)
    component_fold = assign_components(pairs, components, n_folds, seed)
    pairs["fold"] = pairs.component_id.map(component_fold).astype(int)
    row_folds = train.pair_id.astype(str).map(pairs.set_index("pair_id").fold)
    if row_folds.isna().any():
        raise AssertionError("Some rows did not receive a peptide-component fold.")
    audits: list[dict[str, Any]] = []
    all_peptides = set(train.peptide_sequence)
    for fold in range(n_folds):
        held = train[row_folds == fold]
        fitting = train[row_folds != fold]
        overlap = set(held.peptide_sequence) & set(fitting.peptide_sequence)
        pair_overlap = set(held.pair_id.astype(str)) & set(fitting.pair_id.astype(str))
        audits.append({
            "fold": fold,
            "fit_rows": len(fitting),
            "held_out_rows": len(held),
            "fit_pairs": fitting.pair_id.nunique(),
            "held_out_pairs": held.pair_id.nunique(),
            "fit_peptides": fitting.peptide_sequence.nunique(),
            "held_out_peptides": held.peptide_sequence.nunique(),
            "peptide_overlap": len(overlap),
            "pair_overlap": len(pair_overlap),
            "held_out_tasks": held.task_name.nunique(),
        })
        if overlap or pair_overlap:
            raise AssertionError(f"Fold {fold} violates peptide/pair disjointness.")
        if held.task_name.nunique() != train.task_name.nunique():
            raise AssertionError(f"Fold {fold} does not contain every task.")
    pair_assignments = pairs[["pair_id", "task_name", "component_id", "fold"]].sort_values(["fold", "task_name", "pair_id"])
    summary = {
        "n_rows": len(train),
        "n_pairs": train.pair_id.nunique(),
        "n_unique_peptides": len(all_peptides),
        "n_components": len(components),
        "largest_component_pairs": int(components.n_pairs.max()),
        "components_with_multiple_pairs": int((components.n_pairs > 1).sum()),
    }
    return row_folds.astype(int), pair_assignments, {"summary": summary, "folds": audits}


def metric_tables(ensemble: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (tissue, h2), task in ensemble.groupby(["target_tissue", "mhc_restriction"], sort=True):
        rows.append({
            "experiment_name": EXPERIMENT,
            "candidate": CANDIDATE,
            "target_tissue": tissue,
            "mhc_restriction": h2,
            "oof_rows": len(task),
            "oof_pairs": task.pair_id.nunique(),
            **base.evaluate(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float)),
        })
    per_task = pd.DataFrame(rows)
    summary: dict[str, Any] = {"experiment_name": EXPERIMENT, "candidate": CANDIDATE}
    for metric in METRICS:
        summary[f"mean_task_{metric}"] = float(per_task[metric].mean())
    summary["worst_task_auroc"] = float(per_task.auroc.min())
    summary["worst6_task_auroc"] = float(per_task.nsmallest(6, "auroc").auroc.mean())
    by_h2 = per_task.groupby("mhc_restriction", as_index=False).agg(
        n_tasks=("auroc", "size"), mean_task_auroc=("auroc", "mean"), mean_task_auprc=("auprc", "mean")
    )
    by_h2.insert(0, "experiment_name", EXPERIMENT)
    by_h2.insert(1, "candidate", CANDIDATE)
    return per_task, pd.DataFrame([summary]), by_h2


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    if args.seeds != SEEDS:
        raise ValueError(f"E33 is frozen to seeds {SEEDS}.")
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    raw = base.read_dataset(args.train)
    e15.validate_train(raw)
    train, _, mappings = base.add_task_columns(raw, raw.copy())
    peptide_length = int(train.peptide_sequence.str.len().iloc[0])
    assignments, pair_assignments, split_audit = make_folds(train, args.oof_folds, args.split_seed)
    if args.audit_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pair_assignments.to_csv(args.output_dir / "mousePMHC_phase6_e33_pair_fold_assignments.csv", index=False)
        (args.output_dir / "mousePMHC_phase6_e33_split_audit.json").write_text(json.dumps(split_audit, indent=2), encoding="utf-8")
        print(json.dumps(split_audit, indent=2), flush=True)
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        allowed = {"mousePMHC_phase6_e33_pair_fold_assignments.csv", "mousePMHC_phase6_e33_split_audit.json"}
        unexpected = {item.name for item in args.output_dir.iterdir()} - allowed
        if unexpected:
            raise FileExistsError(f"Refusing to overwrite E33 output: {sorted(unexpected)}")
    parts: list[pd.DataFrame] = []
    parameter_count: int | None = None
    for seed in args.seeds:
        for fold in range(args.oof_folds):
            fitting = train[assignments != fold].copy()
            held_out = train[assignments == fold].copy()
            base.set_seed(int(seed), torch)
            print(f"E33 seed={seed} fold={fold + 1}/{args.oof_folds} fit_rows={len(fitting)} held_out_rows={len(held_out)}", flush=True)
            model = e15.train_e3b(args, torch, nn, fitting, mappings, peptide_length, device, int(seed), "peptide-disjoint-oof", fold)
            if parameter_count is None:
                parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
            scores = e15.e5.predict(torch, DataLoader, TensorDataset, model, held_out, peptide_length, args.batch_size, device)
            output = held_out[KEYS].copy()
            output.insert(0, "split", "peptide_disjoint_oof")
            output.insert(1, "candidate", MEMBER)
            output.insert(2, "seed", int(seed))
            output.insert(3, "fold", int(fold))
            output["score"] = scores
            parts.append(output)
    members = pd.concat(parts, ignore_index=True)
    if len(members) != len(train) * len(args.seeds) or members.duplicated(["seed", "sample_id"]).any():
        raise AssertionError("E33 member predictions do not cover every training row once per seed.")
    ensemble = members.groupby(KEYS, as_index=False).agg(
        fold=("fold", "first"), score=("score", "mean"),
        prediction_std_across_seeds=("score", lambda values: float(np.std(values, ddof=0))),
        n_members=("seed", "nunique"),
    )
    if not (ensemble.n_members == len(args.seeds)).all():
        raise AssertionError("E33 ensemble lacks one or more seed members.")
    ensemble.insert(0, "split", "peptide_disjoint_oof")
    ensemble.insert(1, "candidate", CANDIDATE)
    per_task, summary, by_h2 = metric_tables(ensemble)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_assignments.to_csv(args.output_dir / "mousePMHC_phase6_e33_pair_fold_assignments.csv", index=False)
    (args.output_dir / "mousePMHC_phase6_e33_split_audit.json").write_text(json.dumps(split_audit, indent=2), encoding="utf-8")
    members.to_csv(args.output_dir / "mousePMHC_phase6_e33_member_oof_predictions.csv", index=False)
    ensemble.to_csv(args.output_dir / "mousePMHC_phase6_e33_ensemble_oof_predictions.csv", index=False)
    per_task.to_csv(args.output_dir / "mousePMHC_phase6_e33_per_task_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "mousePMHC_phase6_e33_summary_metrics.csv", index=False)
    by_h2.to_csv(args.output_dir / "mousePMHC_phase6_e33_by_h2_metrics.csv", index=False)
    metadata = {
        "experiment_name": EXPERIMENT,
        "candidate": CANDIDATE,
        "status": "completed",
        "test_data_read": False,
        "train": str(args.train),
        "train_sha256": sha256(args.train),
        "split_definition": "Connected components of the pair--peptide graph; components assigned to three balanced folds.",
        "pair_disjoint": True,
        "peptide_disjoint": True,
        "protein_disjoint": False,
        "seeds": args.seeds,
        "oof_folds": args.oof_folds,
        "split_seed": args.split_seed,
        "frozen_structure": "E3b task-balanced Factorized MMoE",
        "fusion": "equal-weight five-seed probability mean",
        "device": device,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "task_batch_size": args.task_batch_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "expert_dim": args.expert_dim,
        "condition_dim": args.condition_dim,
        "gate_hidden_dim": args.gate_hidden_dim,
        "n_experts": args.n_experts,
        "dropout": args.dropout,
        "gate_entropy_weight": args.gate_entropy_weight,
        "max_grad_norm": args.max_grad_norm,
        "parameter_count": parameter_count,
        "split_audit": split_audit,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation_rule": "Robustness evaluation only; no model or hyperparameter selection from E33.",
    }
    (args.output_dir / "mousePMHC_phase6_e33_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=path("results/mousePMHC_phase6_e33_peptide_disjoint_oof"))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--oof-folds", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=20260711)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--task-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--expert-dim", type=int, default=64)
    parser.add_argument("--condition-dim", type=int, default=16)
    parser.add_argument("--gate-hidden-dim", type=int, default=64)
    parser.add_argument("--n-experts", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
