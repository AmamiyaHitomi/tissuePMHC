#!/usr/bin/env python3
"""Run Phase 3 E30 CNN + tiny self-attention interaction OOF experiments.

This runner is deliberately development/OOF-only: it has no standard-test
argument and refuses to use the frozen Phase 2 test CSV.  It supports both
the E30 interaction encoder and a matched E29 CNN baseline on identical
pair- or peptide-grouped folds.  Run the baseline first, then give its OOF
prediction file to E30 for a leakage-safe comparison.

E30 encoder (the pre-registered primary configuration):

* E29 position-aware multi-kernel CNN, producing ``z_cnn`` (128 dimensions);
* one tiny self-attention block over the nine peptide positions
  (d_model=16, heads=2, FFN=32), producing ``z_interaction`` (128 dimensions);
* ``z = z_cnn + alpha * z_interaction`` with learnable ``alpha`` initialized
  to 0.1.

The global auxiliary branch, HLA plain branch, task-rank fusion, optimizer and
all E29 training defaults remain unchanged.  The main Phase 3 protocol is
peptide-grouped OOF; pair-grouped OOF is available as a continuity diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_tissuepmhc_auxiliary_soft_ensemble as e14
import run_tissuepmhc_e26_greedy_ensemble_selection as e26
import run_tissuepmhc_e29_multikernel_cnn_oof as e29
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEYS = e29.KEYS
PREDICTION_COLUMNS = e29.PREDICTION_COLUMNS
TASK_COLUMNS = e26.TASK_COLUMNS


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def format_duration(seconds: float) -> str:
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes)}m {remainder:04.1f}s" if minutes >= 1 else f"{remainder:.1f}s"


def candidate_name(args: argparse.Namespace, seed: int) -> str:
    prefix = "e30_interaction" if args.encoder == "e30_interaction" else "e29_cnn"
    return f"{prefix}_seed_{seed}"


def mean_candidate_name(args: argparse.Namespace) -> str:
    prefix = "e30_interaction" if args.encoder == "e30_interaction" else "e29_cnn"
    return f"{prefix}_{len(args.seeds)}seed_mean"


def stable_uint64(value: str, seed: int) -> int:
    payload = f"{seed}:{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="little", signed=False)


def make_pair_grouped_folds(train_df: pd.DataFrame, folds: int, split_seed: int) -> pd.Series:
    """Assign pair_id groups within each task, matching the Phase 2 protocol."""
    rng = np.random.default_rng(split_seed)
    assignment = pd.Series(index=train_df.index, dtype="int64")
    for task_name, task in train_df.groupby("task_name", sort=True):
        groups = np.asarray(sorted(task["pair_id"].astype(str).unique()))
        if len(groups) < folds:
            raise ValueError(f"Task {task_name} has fewer than {folds} pair groups.")
        shuffled = rng.permutation(groups)
        group_to_fold = {group: position % folds for position, group in enumerate(shuffled)}
        assignment.loc[task.index] = task["pair_id"].astype(str).map(group_to_fold).astype(int)
    return assignment.astype(int)


def make_global_peptide_grouped_folds(train_df: pd.DataFrame, folds: int, split_seed: int) -> pd.Series:
    """Balance task/label counts while keeping peptide and pair components intact.

    A peptide may occur in several tissue-HLA tasks, while each ``pair_id`` joins
    one positive and one pseudo-negative peptide.  We therefore group connected
    components of the peptide--pair bipartite graph: no peptide *or pair* can leak
    between fitting and held-out folds.  The deterministic greedy objective then
    minimizes squared deviation from each task/label cell's ideal fold count.
    """
    if train_df["peptide_sequence"].isna().any():
        raise ValueError("Peptide-grouped OOF requires non-null peptide_sequence values.")
    task_labels = sorted({(task, int(label)) for task, label in zip(train_df["task_name"], train_df["label"])})
    cell_to_index = {cell: index for index, cell in enumerate(task_labels)}
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    peptide_nodes = train_df["peptide_sequence"].astype(str).map(lambda value: f"peptide:{value}")
    pair_nodes = train_df["pair_id"].astype(str).map(lambda value: f"pair:{value}")
    for peptide_node, pair_node in zip(peptide_nodes, pair_nodes):
        union(peptide_node, pair_node)
    components = peptide_nodes.map(find)

    group_counts: dict[str, np.ndarray] = {}
    working = train_df.assign(_component=components.to_numpy())
    for component, group in working.groupby("_component", sort=True):
        counts = np.zeros(len(task_labels), dtype=np.float64)
        for (task_name, label), count in group.groupby(["task_name", "label"], sort=True).size().items():
            counts[cell_to_index[(task_name, int(label))]] = float(count)
        group_counts[str(component)] = counts
    total = np.sum(list(group_counts.values()), axis=0)
    if (total < folds).any():
        unsupported = [task_labels[index] for index, count in enumerate(total) if count < folds]
        raise ValueError(f"Peptide-grouped OOF has task/label cells with fewer than {folds} rows: {unsupported[:5]}")
    target = total / float(folds)
    # Large, multi-task peptide groups are placed first; the hash makes exact ties deterministic.
    ordered = sorted(
        group_counts,
        key=lambda component: (
            -float(group_counts[component].sum()), -int((group_counts[component] > 0).sum()),
            stable_uint64(component, split_seed),
        ),
    )
    fold_counts = np.zeros((folds, len(task_labels)), dtype=np.float64)
    assignment_by_component: dict[str, int] = {}
    denominator = np.maximum(target, 1.0)
    for component in ordered:
        contribution = group_counts[component]
        costs = []
        for fold in range(folds):
            proposed = fold_counts.copy()
            proposed[fold] += contribution
            balance_cost = float(np.square((proposed - target) / denominator).sum())
            load_cost = float((proposed[fold].sum() / max(total.sum() / folds, 1.0)) ** 2) * 1e-4
            costs.append(balance_cost + load_cost)
        best = min(range(folds), key=lambda fold: (costs[fold], fold_counts[fold].sum(), fold))
        assignment_by_component[component] = best
        fold_counts[best] += contribution
    assignment = components.map(assignment_by_component)
    if assignment.isna().any():
        raise AssertionError("Some peptide/pair components were not assigned an OOF fold.")
    assignment = assignment.astype(int)
    for fold in range(folds):
        held_out = train_df.loc[assignment == fold]
        for task_name, task in held_out.groupby("task_name", sort=True):
            if task["label"].nunique() != 2:
                raise ValueError(
                    f"Peptide-grouped fold {fold} leaves task {task_name} without both labels; "
                    "use a different frozen split seed or improve the assignment algorithm."
                )
    return assignment


def make_oof_folds(train_df: pd.DataFrame, grouping: str, folds: int, split_seed: int) -> pd.Series:
    if folds < 2:
        raise ValueError("--oof-folds must be at least two.")
    if grouping == "pair":
        assignment = make_pair_grouped_folds(train_df, folds, split_seed)
        group_column = "pair_id"
    elif grouping == "peptide":
        assignment = make_global_peptide_grouped_folds(train_df, folds, split_seed)
        group_column = "peptide_sequence"
    else:
        raise ValueError(f"Unsupported grouping: {grouping}")
    for fold in range(folds):
        fitting = train_df.loc[assignment != fold]
        held_out = train_df.loc[assignment == fold]
        overlap = set(fitting[group_column].astype(str)) & set(held_out[group_column].astype(str))
        if overlap:
            raise AssertionError(f"{group_column} leakage in fold {fold}: {next(iter(overlap))}")
        if grouping == "peptide":
            pair_overlap = set(fitting["pair_id"].astype(str)) & set(held_out["pair_id"].astype(str))
            if pair_overlap:
                raise AssertionError(f"pair_id leakage in peptide-grouped fold {fold}: {next(iter(pair_overlap))}")
    return assignment


def define_shared_heads_model(
    args: argparse.Namespace, nn: Any, peptide_length: int, n_tasks: int, n_tissues: int, n_hlas: int,
    use_aux: bool,
) -> Any:
    """Build either E29 CNN or the E30 interaction residual encoder."""
    import torch

    class E30InteractionSharedHeads(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.peptide_length = peptide_length
            self.embedding = nn.Embedding(len(base.AA_TO_INDEX) + 1, args.embedding_dim, padding_idx=base.PAD_INDEX)
            self.position_embedding = nn.Parameter(torch.zeros((1, peptide_length, args.embedding_dim), dtype=torch.float32))
            self.convolutions = nn.ModuleList([
                nn.Conv1d(args.embedding_dim, args.conv_channels, kernel_size=kernel, padding=kernel // 2)
                for kernel in args.kernel_sizes
            ])
            cnn_dim = peptide_length * args.conv_channels * len(args.kernel_sizes)
            self.cnn_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(cnn_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.ReLU(),
            )
            self.is_interaction_encoder = args.encoder == "e30_interaction"
            if self.is_interaction_encoder:
                if args.embedding_dim % args.attention_heads:
                    raise ValueError("--embedding-dim must be divisible by --attention-heads.")
                self.attention = nn.MultiheadAttention(
                    args.embedding_dim, args.attention_heads, dropout=args.attention_dropout, batch_first=True,
                )
                self.attention_norm_1 = nn.LayerNorm(args.embedding_dim)
                self.attention_ffn = nn.Sequential(
                    nn.Linear(args.embedding_dim, args.attention_ffn_dim),
                    nn.ReLU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(args.attention_ffn_dim, args.embedding_dim),
                )
                self.attention_norm_2 = nn.LayerNorm(args.embedding_dim)
                self.interaction_projection = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(peptide_length * args.embedding_dim, args.hidden_dim),
                    nn.ReLU(),
                )
                self.alpha = nn.Parameter(torch.tensor(float(args.interaction_alpha_init), dtype=torch.float32))
            self.heads = nn.ModuleList([nn.Linear(args.hidden_dim, 1) for _ in range(n_tasks)])
            self.use_aux = use_aux
            if use_aux:
                self.tissue_classifier = nn.Linear(args.hidden_dim, n_tissues)
                self.hla_classifier = nn.Linear(args.hidden_dim, n_hlas)

        def encode(self, peptide_ids: Any) -> Any:
            embedded = self.embedding(peptide_ids) + self.position_embedding
            channels_first = embedded.transpose(1, 2)
            features = []
            for convolution in self.convolutions:
                # Even kernels create one extra slot under symmetric padding.
                convolved = convolution(channels_first)[..., :self.peptide_length]
                features.append(torch.relu(convolved).transpose(1, 2))
            z_cnn = self.cnn_encoder(torch.cat(features, dim=2))
            if not self.is_interaction_encoder:
                return z_cnn
            attended, _ = self.attention(embedded, embedded, embedded, need_weights=False)
            attended = self.attention_norm_1(embedded + attended)
            attended = self.attention_norm_2(attended + self.attention_ffn(attended))
            z_interaction = self.interaction_projection(attended)
            return z_cnn + self.alpha * z_interaction

        def forward(self, peptide_ids: Any, task_ids: Any) -> Any:
            encoded = self.encode(peptide_ids)
            logits = encoded.new_empty(encoded.shape[0])
            for task_id in torch.unique(task_ids):
                mask = task_ids == task_id
                logits[mask] = self.heads[int(task_id.item())](encoded[mask]).squeeze(-1)
            return logits

        def auxiliary_logits(self, peptide_ids: Any) -> tuple[Any, Any]:
            if not self.use_aux:
                raise RuntimeError("Plain HLA branch has no auxiliary classifiers.")
            encoded = self.encode(peptide_ids)
            return self.tissue_classifier(encoded), self.hla_classifier(encoded)

    return E30InteractionSharedHeads()


def train_branch(
    args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any,
    train_df: pd.DataFrame, task_to_id: dict[str, int], peptide_length: int, device: str,
    branch: str, group_name: str, use_aux: bool, n_tissues: int, n_hlas: int,
) -> Any:
    model = define_shared_heads_model(args, nn, peptide_length, len(task_to_id), n_tissues, n_hlas, use_aux).to(device)
    arrays = e29.mapped_arrays(train_df, task_to_id, peptide_length)
    loader = e29.build_loader(torch, DataLoader, TensorDataset, arrays, args.batch_size, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    cross_entropy = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        for peptides, task_ids, tissue_ids, hla_ids, labels in loader:
            peptides, task_ids = peptides.to(device), task_ids.to(device)
            tissue_ids, hla_ids, labels = tissue_ids.to(device), hla_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(peptides, task_ids)
            loss = bce(logits, labels.float())
            if use_aux:
                tissue_logits, hla_logits = model.auxiliary_logits(peptides)
                loss = loss + args.tissue_loss_weight * cross_entropy(tissue_logits, tissue_ids)
                loss = loss + args.hla_loss_weight * cross_entropy(hla_logits, hla_ids)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        alpha = float(model.alpha.detach().cpu()) if getattr(model, "is_interaction_encoder", False) else None
        args.alpha_history.append({
            "seed": args._active_seed, "context": args._active_context, "branch": branch,
            "group": group_name, "epoch": epoch, "alpha": alpha,
            "epoch_seconds": time.perf_counter() - started,
        })
        alpha_text = "" if alpha is None else f" alpha={alpha:.6f}"
        print(
            f"    epoch branch={branch} group={group_name} {epoch}/{args.epochs}{alpha_text} "
            f"duration={format_duration(time.perf_counter() - started)}",
            flush=True,
        )
    return model


def predict_seed(
    args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any,
    fitting: pd.DataFrame, prediction: pd.DataFrame, mappings: dict[str, Any], peptide_length: int,
    device: str, seed: int, context: str,
) -> pd.DataFrame:
    # E29's branch orchestration is retained; only its model factory and trainer
    # are replaced during this call.
    args._active_seed, args._active_context = seed, context
    original_factory, original_trainer = e29.define_cnn_shared_heads_model, e29.train_branch
    try:
        e29.define_cnn_shared_heads_model = define_shared_heads_model
        e29.train_branch = train_branch
        return e29.predict_seed(
            args, torch, nn, DataLoader, TensorDataset, fitting, prediction, mappings, peptide_length,
            device, seed, context,
        )
    finally:
        e29.define_cnn_shared_heads_model, e29.train_branch = original_factory, original_trainer


def append_seed_mean(args: argparse.Namespace, predictions: pd.DataFrame) -> pd.DataFrame:
    if len(args.seeds) < 2:
        return predictions
    members = [candidate_name(args, seed) for seed in args.seeds]
    subset = predictions[predictions["candidate"].isin(members)]
    pivot = subset.pivot(index=["split", "seed", *KEYS, "label"], columns="candidate", values="score")
    if pivot.isna().any().any():
        raise AssertionError("E30 seed predictions are not aligned for the mean.")
    mean = pivot.mean(axis=1).rename("score").reset_index()
    mean.insert(1, "candidate", mean_candidate_name(args))
    return pd.concat([predictions, mean[PREDICTION_COLUMNS]], ignore_index=True)


def candidate_scores(frame: pd.DataFrame, candidate: str) -> tuple[pd.DataFrame, np.ndarray]:
    subset = frame[frame["candidate"] == candidate].copy()
    if subset.empty:
        raise ValueError(f"Candidate {candidate!r} is absent.")
    labels, candidates, matrix = e26.aligned_matrix(subset)
    if candidates != [candidate]:
        raise AssertionError("Expected a single prediction candidate.")
    return labels, matrix[:, 0]


def screen_against_e29(args: argparse.Namespace, oof: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if args.baseline_oof_predictions is None:
        raise ValueError("E30 screening requires --baseline-oof-predictions from a matched E29 OOF run.")
    primary = candidate_name(args, args.seeds[0]) if len(args.seeds) == 1 else mean_candidate_name(args)
    labels, scores = candidate_scores(oof, primary)
    baseline = e26.read_predictions(args.baseline_oof_predictions, "oof")
    baseline_labels, baseline_scores = candidate_scores(baseline, args.baseline_candidate)
    merged = labels.merge(
        baseline_labels, on=e26.KEY_COLUMNS, how="inner", validate="one_to_one", suffixes=("_candidate", "_baseline"),
    )
    if len(merged) != len(labels) or len(merged) != len(baseline_labels):
        raise ValueError("E30 and matched E29 OOF keys do not align exactly.")
    if not np.array_equal(merged["label_candidate"].to_numpy(), merged["label_baseline"].to_numpy()):
        raise ValueError("E30 and matched E29 OOF labels disagree.")
    baseline_index = pd.MultiIndex.from_frame(baseline_labels[e26.KEY_COLUMNS])
    candidate_index = pd.MultiIndex.from_frame(labels[e26.KEY_COLUMNS])
    order = baseline_index.get_indexer(candidate_index)
    if (order < 0).any():
        raise AssertionError("Could not align E29 baseline scores to E30 keys.")
    baseline_scores = baseline_scores[order]
    candidate_metric = e26.metric_summary(labels, scores)
    baseline_metric = e26.metric_summary(labels, baseline_scores)
    task_rows: list[dict[str, Any]] = []
    task_wins = 0
    for (_, tissue, hla), task in labels.assign(candidate_score=scores, baseline_score=baseline_scores).groupby(TASK_COLUMNS, sort=True):
        y_true = task["label"].to_numpy(dtype=int)
        candidate_auroc = base.evaluate(y_true, task["candidate_score"].to_numpy(dtype=float))["auroc"]
        baseline_auroc = base.evaluate(y_true, task["baseline_score"].to_numpy(dtype=float))["auroc"]
        delta = candidate_auroc - baseline_auroc
        task_wins += int(delta > 0)
        task_rows.append({
            "target_tissue": tissue, "mhc_restriction": hla, "rows": len(task),
            "e29_auroc": baseline_auroc, "e30_auroc": candidate_auroc, "delta_auroc": delta,
        })
    deltas = {
        "mean_auroc": candidate_metric["mean_auroc"] - baseline_metric["mean_auroc"],
        "mean_auprc": candidate_metric["mean_auprc"] - baseline_metric["mean_auprc"],
        "worst_10_mean_auroc": candidate_metric["worst_10_mean_auroc"] - baseline_metric["worst_10_mean_auroc"],
        "task_wins": task_wins,
    }
    checks = {
        "mean_auroc": deltas["mean_auroc"] >= args.minimum_mean_auroc_gain,
        "mean_auprc": deltas["mean_auprc"] >= -args.maximum_mean_auprc_drop,
        "worst_10_mean_auroc": deltas["worst_10_mean_auroc"] >= -args.maximum_worst10_auroc_drop,
        "task_wins": task_wins >= args.minimum_task_wins,
    }
    return {
        "primary_candidate": primary,
        "baseline_candidate": args.baseline_candidate,
        "grouping": args.grouping,
        "candidate_oof": candidate_metric,
        "baseline_oof": baseline_metric,
        "deltas": deltas,
        "thresholds": {
            "minimum_mean_auroc_gain": args.minimum_mean_auroc_gain,
            "maximum_mean_auprc_drop": args.maximum_mean_auprc_drop,
            "maximum_worst10_auroc_drop": args.maximum_worst10_auroc_drop,
            "minimum_task_wins": args.minimum_task_wins,
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
    }, task_rows


def generate_oof(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    train = base.read_dataset(args.train)
    train, _, mappings = base.add_task_columns(train, train.copy())
    if args.max_tasks:
        keep = set(mappings["tasks"][:args.max_tasks])
        train = train[train["task_name"].isin(keep)].copy()
        train, _, mappings = base.add_task_columns(train, train.copy())
    peptide_length = int(train["peptide_sequence"].str.len().max())
    if peptide_length != 9 and not args.allow_non_9mer:
        raise ValueError(f"E30 is pre-registered for 9-mer peptides, found maximum length {peptide_length}.")
    folds = make_oof_folds(train, args.grouping, args.oof_folds, args.oof_split_seed)
    parts = []
    for fold in range(args.oof_folds):
        fitting, held_out = train.loc[folds != fold].copy(), train.loc[folds == fold].copy()
        print(
            f"OOF grouping={args.grouping} fold={fold + 1}/{args.oof_folds} "
            f"fit_rows={len(fitting)} holdout_rows={len(held_out)}",
            flush=True,
        )
        for seed in args.seeds:
            prediction = predict_seed(
                args, torch, nn, DataLoader, TensorDataset, fitting, held_out, mappings, peptide_length,
                device, seed, f"{args.grouping}_oof_fold_{fold}",
            )
            row = prediction.copy()
            row.insert(0, "split", "oof")
            row.insert(1, "candidate", candidate_name(args, seed))
            row.insert(2, "seed", seed)
            parts.append(row[PREDICTION_COLUMNS])
    oof = append_seed_mean(args, pd.concat(parts, ignore_index=True))
    fold_rows = train[["sample_id", "target_tissue", "mhc_restriction", "label", "pair_id", "peptide_sequence"]].copy()
    fold_rows.insert(0, "fold", folds.to_numpy())
    return oof, {
        "device": device,
        "n_tasks": len(mappings["tasks"]),
        "peptide_length": peptide_length,
        "fold_rows": fold_rows,
    }


def validate_args(args: argparse.Namespace) -> None:
    if not args.train.is_file():
        raise FileNotFoundError(f"Development train file does not exist: {args.train}")
    frozen_test = (PROJECT_ROOT / "data/tissuePMHC/tissuePMHC_test.csv.gz").resolve()
    if args.train.resolve() == frozen_test:
        raise ValueError("E30 refuses the frozen Phase 2 standard test CSV as --train.")
    if not args.seeds:
        raise ValueError("--seeds must not be empty.")
    if args.oof_folds < 2:
        raise ValueError("--oof-folds must be at least two.")
    if not args.kernel_sizes or any(kernel < 1 for kernel in args.kernel_sizes):
        raise ValueError("--kernel-sizes must contain positive integers.")
    if args.encoder == "e30_interaction":
        if args.attention_heads < 1 or args.attention_ffn_dim < 1:
            raise ValueError("Attention heads and FFN dimension must be positive.")
        if args.embedding_dim % args.attention_heads:
            raise ValueError("--embedding-dim must be divisible by --attention-heads.")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    started = time.perf_counter()
    args.alpha_history = []
    oof, details = generate_oof(args)
    for output in (
        args.oof_predictions_output, args.fold_assignments_output, args.diagnostics_output,
        args.task_comparison_output, args.run_manifest_output,
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(args.oof_predictions_output, index=False)
    details["fold_rows"].to_csv(args.fold_assignments_output, index=False)
    screen: dict[str, Any] | None = None
    task_rows: list[dict[str, Any]] = []
    if args.encoder == "e30_interaction" and not args.skip_screen:
        screen, task_rows = screen_against_e29(args, oof)
        pd.DataFrame(task_rows).to_csv(args.task_comparison_output, index=False)
        print(f"E30 OOF screen passed={screen['passed']}", flush=True)
    args.diagnostics_output.write_text(json.dumps(args.alpha_history, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "experiment_name": "E30_interaction_oof" if args.encoder == "e30_interaction" else "E29_phase3_matched_baseline_oof",
        "encoder": args.encoder,
        "test_policy": "This script has no test path and never reads the frozen Phase 2 standard test CSV.",
        "train": str(args.train),
        "grouping": args.grouping,
        "oof_folds": args.oof_folds,
        "oof_split_seed": args.oof_split_seed,
        "seeds": args.seeds,
        "architecture": {
            "embedding_dim": args.embedding_dim,
            "kernel_sizes": args.kernel_sizes,
            "conv_channels": args.conv_channels,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "attention_heads": args.attention_heads if args.encoder == "e30_interaction" else None,
            "attention_ffn_dim": args.attention_ffn_dim if args.encoder == "e30_interaction" else None,
            "attention_dropout": args.attention_dropout if args.encoder == "e30_interaction" else None,
            "interaction_alpha_init": args.interaction_alpha_init if args.encoder == "e30_interaction" else None,
        },
        "training": {
            "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "tissue_loss_weight": args.tissue_loss_weight,
            "hla_loss_weight": args.hla_loss_weight,
        },
        "details": {key: value for key, value in details.items() if key != "fold_rows"},
        "screen": screen,
    }
    args.run_manifest_output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote: {args.oof_predictions_output}", flush=True)
    print(f"wrote: {args.fold_assignments_output}", flush=True)
    print(f"wrote: {args.diagnostics_output}", flush=True)
    print(f"wrote: {args.run_manifest_output}", flush=True)
    print(f"run total time: {format_duration(time.perf_counter() - started)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = project_path("results/tissuePMHC_e30_interaction_oof")
    parser.add_argument("--train", type=Path, default=project_path("phase3/splits/phase3_development.csv.gz"))
    parser.add_argument("--encoder", choices=["e29_cnn", "e30_interaction"], default="e30_interaction")
    parser.add_argument("--grouping", choices=["pair", "peptide"], default="peptide")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260704])
    parser.add_argument("--oof-folds", type=int, default=3)
    parser.add_argument("--oof-split-seed", type=int, default=20260712)
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
    parser.add_argument("--attention-heads", type=int, default=2)
    parser.add_argument("--attention-ffn-dim", type=int, default=32)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--interaction-alpha-init", type=float, default=0.1)
    parser.add_argument("--allow-non-9mer", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=0, help="Optional model-path smoke-test task limit.")
    parser.add_argument("--baseline-oof-predictions", type=Path)
    parser.add_argument("--baseline-candidate", default="e29_cnn_seed_20260704")
    parser.add_argument("--skip-screen", action="store_true", help="Write OOF output without an E29 comparison (for matched baseline runs).")
    parser.add_argument("--minimum-mean-auroc-gain", type=float, default=0.002)
    parser.add_argument("--maximum-mean-auprc-drop", type=float, default=0.0005)
    parser.add_argument("--maximum-worst10-auroc-drop", type=float, default=0.001)
    parser.add_argument("--minimum-task-wins", type=int, default=28)
    parser.add_argument("--oof-predictions-output", type=Path, default=root / "oof_predictions.csv")
    parser.add_argument("--fold-assignments-output", type=Path, default=root / "fold_assignments.csv")
    parser.add_argument("--diagnostics-output", type=Path, default=root / "training_diagnostics.json")
    parser.add_argument("--task-comparison-output", type=Path, default=root / "oof_task_comparison_vs_e29.csv")
    parser.add_argument("--run-manifest-output", type=Path, default=root / "run_manifest.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
