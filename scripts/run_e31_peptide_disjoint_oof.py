#!/usr/bin/env python3
"""Run frozen Phase 7 E29 under peptide-disjoint train-only OOF.

Pairs are kept atomic. Shared peptides connect pairs into components, and
complete components are assigned to balanced folds. Consequently neither a
pair nor a peptide can occur in both fitting and held-out data.

The program never reads the Phase 7 standard test set. It prints the duration
of every training epoch (via the frozen E29 trainer), every fold, every seed,
and the complete run. Console output is also persisted in ``run.log``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd

from _runner import DATA_DIR, RESULTS_DIR, enable_original_modules


EXPERIMENT = "Phase7_min200_E31_peptide_disjoint_oof"
CANDIDATE_PREFIX = "phase7_min200_e31_frozen_e29"
SEEDS = [20260704, 20260705, 20260706]
KEYS = ["sample_id", "target_tissue", "mhc_restriction"]
METRICS = ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Tee(TextIO):
    """Mirror text to the terminal and a persistent UTF-8 log."""

    def __init__(self, terminal: TextIO, log: TextIO) -> None:
        self.terminal = terminal
        self.log = log

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


class DSU:
    def __init__(self, values: list[str]) -> None:
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
    pairs = train[["pair_id", "task_name"]].drop_duplicates().copy()
    if pairs["pair_id"].duplicated().any():
        raise AssertionError("A pair_id occurs in more than one task.")
    pairs["pair_id"] = pairs["pair_id"].astype(str)
    pair_peptides = train[["pair_id", "peptide_sequence"]].drop_duplicates().copy()
    pair_peptides["pair_id"] = pair_peptides["pair_id"].astype(str)
    dsu = DSU(pair_peptides["pair_id"].unique().tolist())
    for _, group in pair_peptides.groupby("peptide_sequence", sort=False):
        pair_ids = group["pair_id"].tolist()
        for pair_id in pair_ids[1:]:
            dsu.union(pair_ids[0], pair_id)
    pairs["component_id"] = pairs["pair_id"].map(dsu.find)
    components = pairs.groupby("component_id").agg(
        n_pairs=("pair_id", "size"), n_tasks=("task_name", "nunique")
    ).reset_index()
    return pairs, components


def assign_components(
    pairs: pd.DataFrame, components: pd.DataFrame, n_folds: int, seed: int,
) -> dict[str, int]:
    tasks = sorted(pairs["task_name"].unique())
    task_to_id = {task: index for index, task in enumerate(tasks)}
    target = pairs.groupby("task_name").size().reindex(tasks).to_numpy(dtype=float) / n_folds
    vectors: dict[str, np.ndarray] = {}
    for component_id, group in pairs.groupby("component_id", sort=False):
        vector = np.zeros(len(tasks), dtype=float)
        for task, count in group.groupby("task_name").size().items():
            vector[task_to_id[task]] = count
        vectors[str(component_id)] = vector
    rng = np.random.default_rng(seed)
    order = components.copy()
    order["tie"] = rng.random(len(order))
    order = order.sort_values(["n_pairs", "n_tasks", "tie"], ascending=[False, False, True])
    fold_counts = np.zeros((n_folds, len(tasks)), dtype=float)
    fold_totals = np.zeros(n_folds, dtype=float)
    assignments: dict[str, int] = {}
    for component_id_value in order["component_id"]:
        component_id = str(component_id_value)
        vector = vectors[component_id]
        scores: list[float] = []
        for fold in range(n_folds):
            candidate = fold_counts.copy()
            candidate[fold] += vector
            task_imbalance = np.mean(((candidate - target) / np.maximum(target, 1.0)) ** 2)
            totals = fold_totals.copy()
            totals[fold] += vector.sum()
            total_imbalance = np.var(totals / max(len(pairs) / n_folds, 1.0))
            scores.append(float(task_imbalance + 0.05 * total_imbalance))
        best = min(scores)
        eligible = [fold for fold, score in enumerate(scores) if abs(score - best) < 1e-12]
        fold = int(rng.choice(eligible))
        assignments[component_id] = fold
        fold_counts[fold] += vector
        fold_totals[fold] += vector.sum()
    return assignments


def make_folds(
    train: pd.DataFrame, n_folds: int, seed: int,
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    pairs, components = component_table(train)
    component_fold = assign_components(pairs, components, n_folds, seed)
    pairs["fold"] = pairs["component_id"].astype(str).map(component_fold).astype(int)
    fold_by_pair = pairs.set_index("pair_id")["fold"]
    row_folds = train["pair_id"].astype(str).map(fold_by_pair)
    if row_folds.isna().any():
        raise AssertionError("Some rows did not receive a component fold.")
    audits: list[dict[str, Any]] = []
    for fold in range(n_folds):
        fitting = train[row_folds != fold]
        held_out = train[row_folds == fold]
        peptide_overlap = set(fitting["peptide_sequence"]) & set(held_out["peptide_sequence"])
        pair_overlap = set(fitting["pair_id"].astype(str)) & set(held_out["pair_id"].astype(str))
        audit = {
            "fold": fold,
            "fit_rows": len(fitting),
            "held_out_rows": len(held_out),
            "fit_pairs": fitting["pair_id"].nunique(),
            "held_out_pairs": held_out["pair_id"].nunique(),
            "fit_peptides": fitting["peptide_sequence"].nunique(),
            "held_out_peptides": held_out["peptide_sequence"].nunique(),
            "peptide_overlap": len(peptide_overlap),
            "pair_overlap": len(pair_overlap),
            "held_out_tasks": held_out["task_name"].nunique(),
        }
        audits.append(audit)
        if peptide_overlap or pair_overlap:
            raise AssertionError(f"Fold {fold} violates peptide/pair disjointness.")
        if held_out["task_name"].nunique() != train["task_name"].nunique():
            raise AssertionError(f"Fold {fold} does not contain every task.")
    assignment_table = pairs[["pair_id", "task_name", "component_id", "fold"]].sort_values(
        ["fold", "task_name", "pair_id"]
    )
    audit = {
        "summary": {
            "n_rows": len(train),
            "n_pairs": train["pair_id"].nunique(),
            "n_unique_peptides": train["peptide_sequence"].nunique(),
            "n_components": len(components),
            "largest_component_pairs": int(components["n_pairs"].max()),
            "components_with_multiple_pairs": int((components["n_pairs"] > 1).sum()),
        },
        "folds": audits,
    }
    return row_folds.astype(int), assignment_table, audit


def metric_tables(e29: Any, base: Any, ensemble: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (tissue, hla), task in ensemble.groupby(["target_tissue", "mhc_restriction"], sort=True):
        rows.append({
            "experiment_name": EXPERIMENT,
            "candidate": f"{CANDIDATE_PREFIX}_{len(SEEDS)}seed_mean",
            "target_tissue": tissue,
            "mhc_restriction": hla,
            "oof_rows": len(task),
            **base.evaluate(task["label"].to_numpy(dtype=int), task["score"].to_numpy(dtype=float)),
        })
    per_task = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "experiment_name": EXPERIMENT,
        "candidate": f"{CANDIDATE_PREFIX}_{len(SEEDS)}seed_mean",
    }
    for metric in METRICS:
        summary[f"mean_task_{metric}"] = float(per_task[metric].mean())
    summary["worst_task_auroc"] = float(per_task["auroc"].min())
    summary["worst10_task_auroc"] = float(per_task.nsmallest(10, "auroc")["auroc"].mean())
    return per_task, pd.DataFrame([summary])


def execute(args: argparse.Namespace, e29: Any, base: Any) -> None:
    total_started = time.perf_counter()
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    raw = base.read_dataset(args.train)
    train, _, mappings = base.add_task_columns(raw, raw.copy())
    peptide_length = int(train["peptide_sequence"].str.len().max())
    assignments, pair_assignments, split_audit = make_folds(train, args.oof_folds, args.split_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_assignments.to_csv(args.output_dir / "pair_fold_assignments.csv", index=False)
    (args.output_dir / "split_audit.json").write_text(
        json.dumps(split_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(split_audit, indent=2), flush=True)
    if args.audit_only:
        print(f"total time: {format_duration(time.perf_counter() - total_started)}", flush=True)
        return
    existing = {path.name for path in args.output_dir.iterdir()} - {
        "pair_fold_assignments.csv", "split_audit.json", "run.log", "checkpoints"
    }
    if existing:
        raise FileExistsError(f"Refusing to overwrite E31 output: {sorted(existing)}")

    parts: list[pd.DataFrame] = []
    timing_rows: list[dict[str, Any]] = []
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        seed_started = time.perf_counter()
        print(f"SEED START seed={seed}", flush=True)
        for fold in range(args.oof_folds):
            fold_started = time.perf_counter()
            fitting = train[assignments != fold].copy()
            held_out = train[assignments == fold].copy()
            checkpoint = checkpoint_dir / f"member_seed_{seed}_fold_{fold}.csv.gz"
            print(
                f"E31 seed={seed} fold={fold + 1}/{args.oof_folds} "
                f"fit_rows={len(fitting)} held_out_rows={len(held_out)}",
                flush=True,
            )
            if checkpoint.is_file():
                member = pd.read_csv(checkpoint)
                expected_ids = set(held_out["sample_id"].astype(str))
                checkpoint_ids = set(member["sample_id"].astype(str))
                if (
                    len(member) != len(held_out)
                    or checkpoint_ids != expected_ids
                    or set(member["seed"].astype(int)) != {int(seed)}
                    or set(member["fold"].astype(int)) != {int(fold)}
                ):
                    raise ValueError(f"Invalid E31 checkpoint: {checkpoint}")
                print(f"CHECKPOINT REUSED seed={seed} fold={fold + 1}/{args.oof_folds}: {checkpoint}", flush=True)
            else:
                prediction = e29.predict_seed(
                    args, torch, nn, DataLoader, TensorDataset, fitting, held_out, mappings,
                    peptide_length, device, int(seed), f"peptide_disjoint_fold_{fold}",
                )
                member = e29.candidate_rows(
                    "peptide_disjoint_oof", f"{CANDIDATE_PREFIX}_seed_{seed}", prediction
                )
                # e29.candidate_rows uses seed=0 for aggregate candidates.
                # E31 stores independent members, so preserve the real seed.
                member["seed"] = int(seed)
                member.insert(3, "fold", int(fold))
                member.to_csv(checkpoint, index=False)
                print(f"CHECKPOINT WRITTEN seed={seed} fold={fold + 1}/{args.oof_folds}: {checkpoint}", flush=True)
            parts.append(member)
            fold_seconds = time.perf_counter() - fold_started
            timing_rows.append({
                "level": "fold", "seed": seed, "fold": fold,
                "elapsed_seconds": fold_seconds, "elapsed": format_duration(fold_seconds),
            })
            print(
                f"FOLD DONE seed={seed} fold={fold + 1}/{args.oof_folds} "
                f"time={format_duration(fold_seconds)}",
                flush=True,
            )
        seed_seconds = time.perf_counter() - seed_started
        timing_rows.append({
            "level": "seed", "seed": seed, "fold": -1,
            "elapsed_seconds": seed_seconds, "elapsed": format_duration(seed_seconds),
        })
        print(f"SEED DONE seed={seed} time={format_duration(seed_seconds)}", flush=True)

    members = pd.concat(parts, ignore_index=True)
    expected = len(train) * len(args.seeds)
    if len(members) != expected or members.duplicated(["seed", "sample_id"]).any():
        raise AssertionError("Member predictions do not cover every row exactly once per seed.")
    ensemble = members.groupby(KEYS + ["label"], as_index=False).agg(
        fold=("fold", "first"),
        score=("score", "mean"),
        prediction_std_across_seeds=("score", lambda values: float(np.std(values, ddof=0))),
        n_members=("seed", "nunique"),
    )
    if not (ensemble["n_members"] == len(args.seeds)).all():
        raise AssertionError("The seed ensemble is incomplete.")
    ensemble.insert(0, "split", "peptide_disjoint_oof")
    ensemble.insert(1, "candidate", f"{CANDIDATE_PREFIX}_{len(args.seeds)}seed_mean")
    per_task, summary = metric_tables(e29, base, ensemble)

    members.to_csv(args.output_dir / "member_oof_predictions.csv", index=False)
    ensemble.to_csv(args.output_dir / "ensemble_oof_predictions.csv", index=False)
    per_task.to_csv(args.output_dir / "per_task_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    total_seconds = time.perf_counter() - total_started
    timing_rows.append({
        "level": "total", "seed": 0, "fold": -1,
        "elapsed_seconds": total_seconds, "elapsed": format_duration(total_seconds),
    })
    pd.DataFrame(timing_rows).to_csv(args.output_dir / "timing.csv", index=False)
    metadata = {
        "experiment_name": EXPERIMENT,
        "candidate": f"{CANDIDATE_PREFIX}_{len(args.seeds)}seed_mean",
        "status": "completed",
        "test_data_read": False,
        "train": str(args.train.resolve()),
        "train_sha256": sha256(args.train),
        "split_definition": "Connected components of the pair--peptide graph assigned to balanced folds.",
        "pair_disjoint": True,
        "peptide_disjoint": True,
        "protein_disjoint": False,
        "seeds": args.seeds,
        "oof_folds": args.oof_folds,
        "split_seed": args.split_seed,
        "frozen_structure": "Phase 7 E29 multi-kernel CNN with global-aux and HLA-plain rank fusion",
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "embedding_dim": args.embedding_dim,
        "kernel_sizes": args.kernel_sizes,
        "conv_channels": args.conv_channels,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "tissue_loss_weight": args.tissue_loss_weight,
        "hla_loss_weight": args.hla_loss_weight,
        "split_audit": split_audit,
        "elapsed_seconds": total_seconds,
        "interpretation_rule": "Robustness evaluation only; no selection or tuning from E31.",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(f"TOTAL DONE time={format_duration(total_seconds)} ({total_seconds:.3f}s)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", type=Path,
        default=DATA_DIR / "tissuePMHC_phase7_min200_train.csv.gz",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=RESULTS_DIR / "tissuePMHC_phase7_min200_e31_peptide_disjoint_oof",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--oof-folds", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=20260711)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--kernel-sizes", nargs="+", type=int, default=[2, 3, 5])
    parser.add_argument("--conv-channels", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--tissue-loss-weight", type=float, default=0.1)
    parser.add_argument("--hla-loss-weight", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds != SEEDS:
        raise ValueError(f"E31 is frozen to seeds {SEEDS}.")
    if args.oof_folds != 3:
        raise ValueError("E31 is frozen to three OOF folds.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    enable_original_modules()
    import run_tissuepmhc_e29_multikernel_cnn_oof as e29
    import run_tissuepmhc_neural_baselines_v2 as base

    log_path = args.output_dir / "run.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        tee = Tee(sys.stdout, log)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            execute(args, e29, base)


if __name__ == "__main__":
    main()
