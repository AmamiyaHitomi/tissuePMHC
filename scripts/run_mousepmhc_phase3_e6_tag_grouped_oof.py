#!/usr/bin/env python3
"""Run mousePMHC Phase 3 E6: train-only TAG-guided grouped sharing OOF.

For every seed and OOF fold, a small shared-encoder probe is first trained on
the fitting rows only.  TAG-style directed transfer affinity is then measured:
one source-task update is applied and the relative loss change for every target
task is recorded.  The symmetrised affinities define a greedy, size-capped task
partition.  A separate peptide encoder is trained within each discovered group,
with an output head retained for every tissue-H2 task.  The held-out OOF fold
and the fixed test split never participate in affinity estimation or grouping.
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
import run_mousepmhc_phase3_e3b_task_balanced_mmoe_oof as e3b
import run_tissuepmhc_e26_all_in_one as folds
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mousePMHC_phase3_e6_tag_grouped_min200_oof"
CANDIDATE = "mousePMHC_phase3_e6_tag_grouped_sharing"
KEYS = ["sample_id", "target_tissue", "mhc_restriction"]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def task_arrays(frame: pd.DataFrame, mappings: dict[str, Any], peptide_length: int) -> list[dict[str, Any]]:
    """Materialise one deterministic array bundle per task from fitting rows."""
    result: list[dict[str, Any]] = []
    for task_name in mappings["tasks"]:
        task = frame[frame.task_name == task_name]
        result.append({
            "task_id": int(mappings["task_to_id"][task_name]), "task_name": task_name,
            "peptide": base.encode_peptides(task.peptide_sequence, peptide_length),
            "label": task.label.to_numpy(dtype=np.int64),
        })
    return result


def sample_task_batch(rng: np.random.Generator, item: dict[str, Any], batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = rng.integers(0, len(item["label"]), size=batch_size)
    return (item["peptide"][indices], np.full(batch_size, item["task_id"], dtype=np.int64), item["label"][indices])


def sample_balanced_batch(rng: np.random.Generator, arrays: list[dict[str, Any]], task_batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pieces = [sample_task_batch(rng, item, task_batch_size) for item in arrays]
    return tuple(np.concatenate([piece[index] for piece in pieces]) for index in range(3))


def define_shared_model(torch: Any, nn: Any, peptide_length: int, n_tasks: int, args: argparse.Namespace) -> Any:
    """A compact shared probe used solely for train-only affinity measurement."""
    class SharedProbe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.amino_embedding = nn.Embedding(len(base.AA_TO_INDEX) + 1, args.embedding_dim, padding_idx=base.PAD_INDEX)
            self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(peptide_length * args.embedding_dim, args.hidden_dim), nn.ReLU(), nn.Dropout(args.dropout))
            self.heads = nn.ModuleList([nn.Linear(args.hidden_dim, 1) for _ in range(n_tasks)])

        def forward(self, peptide_ids: Any, task_ids: Any) -> Any:
            representation = self.encoder(self.amino_embedding(peptide_ids))
            logits = representation.new_empty(representation.shape[0])
            for task_id in torch.unique(task_ids):
                mask = task_ids == task_id
                logits[mask] = self.heads[int(task_id.item())](representation[mask]).squeeze(-1)
            return logits
    return SharedProbe()


def tag_affinity(args: argparse.Namespace, torch: Any, nn: Any, arrays: list[dict[str, Any]],
                 peptide_length: int, device: str, seed: int) -> np.ndarray:
    """Estimate directed task transfer affinity after a one-step source update.

    ``A[i, j]`` is the relative target-loss reduction on task j after an update
    on task i.  It is evaluated from a frozen probe state, so order of source
    tasks cannot affect the matrix.
    """
    n_tasks = len(arrays)
    rng = np.random.default_rng(seed + 8101)
    model = define_shared_model(torch, nn, peptide_length, n_tasks, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model.train()
    for _ in range(args.affinity_warmup_epochs):
        for _ in range(args.affinity_warmup_steps):
            peptide, task, label = sample_balanced_batch(rng, arrays, args.affinity_batch_size)
            peptide, task, label = [torch.as_tensor(value, device=device) for value in (peptide, task, label)]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(peptide, task), label.float())
            loss.backward(); optimizer.step()
    model.eval()
    evaluation_batches = [sample_task_batch(rng, item, args.affinity_batch_size) for item in arrays]
    with torch.no_grad():
        before = []
        for peptide, task, label in evaluation_batches:
            peptide, task, label = [torch.as_tensor(value, device=device) for value in (peptide, task, label)]
            before.append(float(torch.nn.functional.binary_cross_entropy_with_logits(model(peptide, task), label.float()).cpu()))
    reference_state = copy.deepcopy(model.state_dict())
    affinity = np.zeros((n_tasks, n_tasks), dtype=float)
    for source, source_batch in enumerate(evaluation_batches):
        model.load_state_dict(reference_state); model.train()
        probe_optimizer = torch.optim.SGD(model.parameters(), lr=args.affinity_update_lr)
        peptide, task, label = [torch.as_tensor(value, device=device) for value in source_batch]
        probe_optimizer.zero_grad(set_to_none=True)
        source_loss = torch.nn.functional.binary_cross_entropy_with_logits(model(peptide, task), label.float())
        source_loss.backward(); probe_optimizer.step(); model.eval()
        with torch.no_grad():
            for target, target_batch in enumerate(evaluation_batches):
                peptide, task, label = [torch.as_tensor(value, device=device) for value in target_batch]
                after = float(torch.nn.functional.binary_cross_entropy_with_logits(model(peptide, task), label.float()).cpu())
                affinity[source, target] = (before[target] - after) / max(before[target], 1e-8)
    return affinity


def make_groups(sym_affinity: np.ndarray, max_group_size: int, threshold: float) -> list[list[int]]:
    """Greedily merge the most mutually helpful groups, without forcing merges."""
    groups = [[index] for index in range(len(sym_affinity))]
    while True:
        best_score, best_pair = -np.inf, None
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                if len(groups[left]) + len(groups[right]) > max_group_size:
                    continue
                score = float(sym_affinity[np.ix_(groups[left], groups[right])].mean())
                if score > best_score:
                    best_score, best_pair = score, (left, right)
        if best_pair is None or best_score <= threshold:
            break
        left, right = best_pair
        groups[left] = sorted(groups[left] + groups[right])
        groups.pop(right)
    return sorted(groups, key=lambda group: group[0])


def define_grouped_model(torch: Any, nn: Any, peptide_length: int, n_tasks: int,
                         groups: list[list[int]], args: argparse.Namespace) -> Any:
    task_to_group = np.empty(n_tasks, dtype=np.int64)
    for group_id, group in enumerate(groups):
        task_to_group[group] = group_id

    class GroupedTaskHeads(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.amino_embedding = nn.Embedding(len(base.AA_TO_INDEX) + 1, args.embedding_dim, padding_idx=base.PAD_INDEX)
            self.encoders = nn.ModuleList([
                nn.Sequential(nn.Flatten(), nn.Linear(peptide_length * args.embedding_dim, args.hidden_dim), nn.ReLU(), nn.Dropout(args.dropout))
                for _ in groups
            ])
            self.heads = nn.ModuleList([nn.Linear(args.hidden_dim, 1) for _ in range(n_tasks)])

        def forward(self, peptide_ids: Any, task_ids: Any) -> Any:
            embedded = self.amino_embedding(peptide_ids)
            logits = embedded.new_empty(embedded.shape[0])
            for task_id in torch.unique(task_ids):
                numeric_task = int(task_id.item()); mask = task_ids == task_id
                representation = self.encoders[int(task_to_group[numeric_task])](embedded[mask])
                logits[mask] = self.heads[numeric_task](representation).squeeze(-1)
            return logits
    return GroupedTaskHeads()


def train_grouped_model(args: argparse.Namespace, torch: Any, nn: Any, arrays: list[dict[str, Any]],
                        peptide_length: int, groups: list[list[int]], device: str, seed: int, fold: int) -> Any:
    model = define_grouped_model(torch, nn, peptide_length, len(arrays), groups, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed + 16001 + fold)
    steps = args.steps_per_epoch or int(np.ceil(max(len(item["label"]) for item in arrays) / args.task_batch_size))
    for epoch in range(1, args.epochs + 1):
        model.train(); losses: list[float] = []
        for _ in range(steps):
            peptide, task, label = sample_balanced_batch(rng, arrays, args.task_batch_size)
            peptide, task, label = [torch.as_tensor(value, device=device) for value in (peptide, task, label)]
            optimizer.zero_grad(set_to_none=True)
            per_row = torch.nn.functional.binary_cross_entropy_with_logits(model(peptide, task), label.float(), reduction="none")
            loss = per_row.reshape(len(arrays), args.task_batch_size).mean(dim=1).mean()
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        print(f"E6 seed={seed} fold={fold + 1}/{args.oof_folds} epoch={epoch}/{args.epochs} task_balanced_bce={np.mean(losses):.4f} groups={list(map(len, groups))}", flush=True)
    return model


def predict(torch: Any, DataLoader: Any, TensorDataset: Any, model: Any, frame: pd.DataFrame,
            peptide_length: int, batch_size: int, device: str) -> np.ndarray:
    arrays = [
        base.encode_peptides(frame.peptide_sequence, peptide_length).copy(),
        frame.task_id.to_numpy(dtype=np.int64, copy=True),
        frame.label.to_numpy(dtype=np.int64, copy=True),
    ]
    loader = base.build_loader(torch, DataLoader, TensorDataset, arrays, batch_size, False)
    scores: list[np.ndarray] = []; model.eval()
    with torch.no_grad():
        for peptide, task, _ in loader:
            scores.append(torch.sigmoid(model(peptide.to(device), task.to(device))).cpu().numpy())
    return np.concatenate(scores)


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (seed, tissue, h2), task in predictions.groupby(["seed", "target_tissue", "mhc_restriction"], sort=True):
        rows.append({"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "seed": int(seed), "target_tissue": tissue, "mhc_restriction": h2,
                     "oof_rows": len(task), **base.evaluate(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float))})
    per_task = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for seed, task_metrics in per_task.groupby("seed", sort=True):
        row: dict[str, object] = {"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "seed": int(seed)}
        for metric in ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]:
            row[f"mean_task_{metric}"] = float(task_metrics[metric].mean())
        row["worst_task_auroc"] = float(task_metrics.auroc.min())
        row["worst6_task_auroc"] = float(task_metrics.nsmallest(6, "auroc").auroc.mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    stability_rows = []
    for metric in [column for column in summary if column.startswith("mean_task_") or column.startswith("worst")]:
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
    prediction_parts: list[pd.DataFrame] = []; affinity_rows: list[dict[str, object]] = []; group_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for fold in range(args.oof_folds):
            fitting, held_out = train[assignments != fold].copy(), train[assignments == fold].copy()
            base.set_seed(int(seed), torch)
            arrays = task_arrays(fitting, mappings, peptide_length)
            print(f"E6 seed={seed} fold={fold + 1}/{args.oof_folds} fitting={len(fitting)} holdout={len(held_out)}: estimating TAG affinity", flush=True)
            directed = tag_affinity(args, torch, nn, arrays, peptide_length, device, int(seed) + fold * 101)
            symmetric = (directed + directed.T) / 2.0
            groups = make_groups(symmetric, args.max_group_size, args.affinity_threshold)
            for source in range(len(arrays)):
                for target in range(len(arrays)):
                    affinity_rows.append({"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "seed": seed, "fold": fold,
                                          "source_task": arrays[source]["task_name"], "target_task": arrays[target]["task_name"],
                                          "directed_affinity": float(directed[source, target]), "symmetric_affinity": float(symmetric[source, target])})
            for group_id, group in enumerate(groups):
                for task_id in group:
                    tissue, h2 = arrays[task_id]["task_name"].split("||", 1)
                    group_rows.append({"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "seed": seed, "fold": fold,
                                       "group_id": group_id, "group_size": len(group), "task_id": task_id,
                                       "task_name": arrays[task_id]["task_name"], "target_tissue": tissue, "mhc_restriction": h2})
            print(f"E6 seed={seed} fold={fold + 1}: discovered {len(groups)} groups with sizes {list(map(len, groups))}", flush=True)
            base.set_seed(int(seed), torch)
            model = train_grouped_model(args, torch, nn, arrays, peptide_length, groups, device, int(seed), fold)
            scores = predict(torch, DataLoader, TensorDataset, model, held_out, peptide_length, args.batch_size, device)
            output = held_out[KEYS + ["label"]].copy()
            output.insert(0, "split", "oof"); output.insert(1, "candidate", CANDIDATE); output.insert(2, "seed", int(seed)); output["score"] = scores
            prediction_parts.append(output[["split", "candidate", "seed", *KEYS, "label", "score"]])
    predictions = pd.concat(prediction_parts, ignore_index=True)
    if len(predictions) != len(train) * len(args.seeds) or predictions.duplicated(["seed", "sample_id"]).any():
        raise AssertionError("E6 OOF predictions must cover each train row exactly once per seed.")
    per_task, summary, stability = metric_tables(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "mousePMHC_phase3_e6_oof_predictions.csv", index=False)
    per_task.to_csv(args.output_dir / "mousePMHC_phase3_e6_oof_per_task_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "mousePMHC_phase3_e6_oof_summary_metrics.csv", index=False)
    stability.to_csv(args.output_dir / "mousePMHC_phase3_e6_oof_stability_metrics.csv", index=False)
    pd.DataFrame(affinity_rows).to_csv(args.output_dir / "mousePMHC_phase3_e6_tag_affinities.csv", index=False)
    pd.DataFrame(group_rows).to_csv(args.output_dir / "mousePMHC_phase3_e6_discovered_groups.csv", index=False)
    metadata = {"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "test_data_read": False,
                "method": "TAG-style one-step train-only transfer affinity followed by greedy grouped hard sharing",
                "train": str(args.train), "n_rows": len(train), "n_pairs": int(train.pair_id.nunique()), "n_tasks": len(mappings["tasks"]),
                "seeds": args.seeds, "oof_folds": args.oof_folds, "oof_split_seed": args.oof_split_seed, "device": device,
                "epochs": args.epochs, "task_batch_size": args.task_batch_size, "steps_per_epoch": args.steps_per_epoch,
                "affinity_warmup_epochs": args.affinity_warmup_epochs, "affinity_warmup_steps": args.affinity_warmup_steps,
                "affinity_batch_size": args.affinity_batch_size, "affinity_update_lr": args.affinity_update_lr,
                "affinity_threshold": args.affinity_threshold, "max_group_size": args.max_group_size,
                "selection_protocol": "groups re-estimated independently from each seed-fold fitting partition; no OOF/test rows used"}
    (args.output_dir / "mousePMHC_phase3_e6_oof_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False), flush=True); print(stability.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=project_path("results/mousePMHC_phase3_e6_tag_grouped_min200_oof"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260704, 20260705, 20260706])
    parser.add_argument("--oof-folds", type=int, default=3); parser.add_argument("--oof-split-seed", type=int, default=20260711)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--task-batch-size", type=int, default=16); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=128); parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--affinity-warmup-epochs", type=int, default=3); parser.add_argument("--affinity-warmup-steps", type=int, default=12)
    parser.add_argument("--affinity-batch-size", type=int, default=32); parser.add_argument("--affinity-update-lr", type=float, default=0.05)
    parser.add_argument("--affinity-threshold", type=float, default=0.0); parser.add_argument("--max-group-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
