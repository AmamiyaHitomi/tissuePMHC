#!/usr/bin/env python3
"""Run Phase 5 E16: layered primary-task gradient audit of the E3b MMoE.

The script trains the unchanged task-balanced E3b model on pair-grouped OOF
fitting partitions.  At early, middle, and late epochs it reuses deterministic
fitting-only audit batches and measures gradients from each of the 24 primary
task BCE losses.  No auxiliary loss or fixed-test file is read.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_mousepmhc_phase3_e3_factorized_mmoe_oof as e3
import run_mousepmhc_phase3_e5_famo_mmoe_oof as e5
import run_tissuepmhc_e26_all_in_one as folds
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mousePMHC_phase5_e16_gradient_audit"
CANDIDATE = "e3b_layered_primary_task_gradient_audit"
LAYERS = ["peptide_embedding", "shared_encoder", "expert_0", "expert_1", "expert_2", "gate"]
KEYS = ["sample_id", "target_tissue", "mhc_restriction"]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def layer_parameters(model: Any) -> dict[str, list[Any]]:
    """Map roadmap layers to disjoint parameter groups.

    Gate includes the tissue/H2 condition embeddings because they are learned
    exclusively through the routing path.  Task heads are intentionally absent.
    """
    groups = {
        "peptide_embedding": list(model.amino_embedding.parameters()),
        "shared_encoder": list(model.peptide_encoder.parameters()),
        "gate": [*model.tissue_embedding.parameters(), *model.mhc_embedding.parameters(), *model.gate.parameters()],
    }
    for index, expert in enumerate(model.experts):
        groups[f"expert_{index}"] = list(expert.parameters())
    expected = ["peptide_embedding", "shared_encoder", *[f"expert_{i}" for i in range(len(model.experts))], "gate"]
    if list(groups) != ["peptide_embedding", "shared_encoder", "gate", *[f"expert_{i}" for i in range(len(model.experts))]]:
        raise AssertionError("Unexpected gradient-layer construction.")
    return {name: groups[name] for name in expected}


def make_fixed_audit_batches(
    arrays: list[dict[str, Any]], task_batch_size: int, batches_per_task: int, seed: int,
) -> dict[int, list[tuple[np.ndarray, ...]]]:
    """Sample fitting-only batches once; the exact arrays are reused at all stages."""
    rng = np.random.default_rng(seed)
    result: dict[int, list[tuple[np.ndarray, ...]]] = {}
    for item in arrays:
        task_id = int(item["task_id"])
        batches: list[tuple[np.ndarray, ...]] = []
        for _ in range(batches_per_task):
            indices = rng.integers(0, len(item["label"]), size=task_batch_size)
            batches.append((
                item["peptide"][indices].copy(),
                np.full(task_batch_size, task_id, dtype=np.int64),
                item["tissue"][indices].copy(), item["h2"][indices].copy(), item["label"][indices].copy(),
            ))
        result[task_id] = batches
    return result


def audit_task_gradients(
    torch: Any, model: Any, fixed_batches: dict[int, list[tuple[np.ndarray, ...]]], device: str,
) -> dict[str, Any]:
    """Average each task/layer gradient over fixed batches before computing cosines."""
    groups = layer_parameters(model)
    task_ids = sorted(fixed_batches)
    accumulated = {
        layer: {task_id: torch.zeros(sum(p.numel() for p in params), device=device) for task_id in task_ids}
        for layer, params in groups.items()
    }
    # eval() prevents dropout masks from adding noise to cross-stage comparisons.
    was_training = model.training
    model.eval()
    for task_id in task_ids:
        for batch in fixed_batches[task_id]:
            peptide, task, tissue, h2, label = [torch.as_tensor(value, device=device) for value in batch]
            logits = model(peptide, task, tissue, h2)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label.float())
            all_parameters = [parameter for params in groups.values() for parameter in params]
            all_gradients = torch.autograd.grad(loss, all_parameters, allow_unused=True)
            offset = 0
            for layer, params in groups.items():
                pieces = []
                for parameter, gradient in zip(params, all_gradients[offset: offset + len(params)]):
                    pieces.append((gradient.detach() if gradient is not None else torch.zeros_like(parameter)).reshape(-1))
                accumulated[layer][task_id] += torch.cat(pieces)
                offset += len(params)
    divisor = float(len(next(iter(fixed_batches.values()))))
    for layer in accumulated:
        for task_id in accumulated[layer]:
            accumulated[layer][task_id] /= divisor
    model.train(was_training)
    return accumulated


def cosine_value(torch: Any, left: Any, right: Any) -> float:
    left_norm = float(left.norm().cpu())
    right_norm = float(right.norm().cpu())
    if left_norm == 0.0 or right_norm == 0.0:
        return float("nan")
    return float(torch.nn.functional.cosine_similarity(left, right, dim=0).cpu())


def matrix_rows(
    torch: Any, gradients: dict[int, Any], labels: dict[int, str], common: dict[str, Any], aggregation: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids = sorted(gradients)
    norms = {key: float(gradients[key].norm().cpu()) for key in ids}
    positive_norms = [value for value in norms.values() if value > 0]
    min_norm = min(positive_norms) if positive_norms else 0.0
    max_norm = max(norms.values()) if norms else 0.0
    norm_ratio = max_norm / min_norm if min_norm > 0 else float("inf") if max_norm > 0 else float("nan")
    for source in ids:
        for target in ids:
            smaller_norm = min(norms[source], norms[target])
            pair_norm_ratio = max(norms[source], norms[target]) / smaller_norm if smaller_norm > 0 else (
                float("inf") if max(norms[source], norms[target]) > 0 else float("nan")
            )
            rows.append({
                **common, "aggregation": aggregation,
                "source_id": source, "target_id": target,
                "source_name": labels[source], "target_name": labels[target],
                "gradient_cosine": cosine_value(torch, gradients[source], gradients[target]),
                "source_gradient_norm": norms[source], "target_gradient_norm": norms[target],
                "source_target_norm_ratio": pair_norm_ratio,
                "matrix_max_min_norm_ratio": norm_ratio,
            })
    return rows


def aggregate_gradients(torch: Any, task_gradients: dict[int, Any], task_to_group: dict[int, int]) -> dict[int, Any]:
    result: dict[int, Any] = {}
    for group_id in sorted(set(task_to_group.values())):
        members = [task_gradients[task_id] for task_id in sorted(task_gradients) if task_to_group[task_id] == group_id]
        result[group_id] = torch.stack(members, dim=0).mean(dim=0)
    return result


def off_diagonal(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame.source_id < frame.target_id].sort_values(["source_id", "target_id"])


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 2 or np.std(left[mask]) == 0 or np.std(right[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def build_summaries(matrices: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    keys = ["seed", "fold", "epoch", "stage", "layer", "aggregation"]
    for key, group in matrices.groupby(keys, sort=True):
        pairs = off_diagonal(group)
        cosines = pairs.gradient_cosine.to_numpy(dtype=float)
        finite = cosines[np.isfinite(cosines)]
        negatives = finite[finite < 0]
        records.append({
            **dict(zip(keys, key)), "n_pairs": len(finite),
            "mean_pair_cosine": float(np.mean(finite)) if len(finite) else float("nan"),
            "min_pair_cosine": float(np.min(finite)) if len(finite) else float("nan"),
            "negative_cosine_ratio": float(np.mean(finite < 0)) if len(finite) else float("nan"),
            "negative_cosine_strength": float(np.mean(np.maximum(-finite, 0.0))) if len(finite) else float("nan"),
            "mean_gradient_norm": float(group.groupby("source_id").source_gradient_norm.first().mean()),
            "max_min_norm_ratio": float(group.matrix_max_min_norm_ratio.iloc[0]),
        })
    summary = pd.DataFrame(records)
    summary["conflict_top_quartile"] = False
    task_rows = summary.aggregation == "task"
    for _, indices in summary[task_rows].groupby(["seed", "fold", "epoch"], sort=True).groups.items():
        strengths = summary.loc[indices, "negative_cosine_strength"]
        cutoff = float(strengths.quantile(0.75))
        summary.loc[indices, "conflict_top_quartile"] = (strengths >= cutoff) & (strengths > 0)
    return summary


def build_correlations(matrices: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_keys = ["layer", "aggregation"]
    run_keys = ["seed", "fold", "epoch", "stage"]
    for (layer, aggregation), subset in matrices.groupby(group_keys, sort=True):
        vectors: dict[tuple[int, int, int, str], np.ndarray] = {}
        for key, matrix in subset.groupby(run_keys, sort=True):
            vectors[key] = off_diagonal(matrix).gradient_cosine.to_numpy(dtype=float)
        ordered = sorted(vectors)
        for index, left_key in enumerate(ordered):
            for right_key in ordered[index + 1:]:
                same_seed_fold = left_key[:2] == right_key[:2]
                rows.append({
                    "layer": layer, "aggregation": aggregation,
                    "seed_a": left_key[0], "fold_a": left_key[1], "epoch_a": left_key[2], "stage_a": left_key[3],
                    "seed_b": right_key[0], "fold_b": right_key[1], "epoch_b": right_key[2], "stage_b": right_key[3],
                    "same_seed_fold": same_seed_fold,
                    "adjacent_stage": same_seed_fold and abs(left_key[2] - right_key[2]) > 0 and
                        abs(["early", "middle", "late"].index(left_key[3]) - ["early", "middle", "late"].index(right_key[3])) == 1,
                    "matrix_correlation": safe_correlation(vectors[left_key], vectors[right_key]),
                })
    return pd.DataFrame(rows)


def stable_layer_table(summary: pd.DataFrame, correlations: pd.DataFrame, n_folds: int, n_seeds: int) -> pd.DataFrame:
    """Apply the preregistered recurrence and adjacent-stage-correlation rule."""
    task_summary = summary[summary.aggregation == "task"].copy()
    adjacent = correlations[(correlations.aggregation == "task") & correlations.adjacent_stage].copy()
    required_folds = math.ceil(2 * n_folds / 3)
    required_seeds = math.ceil(2 * n_seeds / 3)
    rows: list[dict[str, Any]] = []
    for layer in sorted(task_summary.layer.unique()):
        layer_summary = task_summary[task_summary.layer == layer]
        support: list[tuple[int, int, str, str, float]] = []
        for item in adjacent[adjacent.layer == layer].itertuples(index=False):
            left = layer_summary[(layer_summary.seed == item.seed_a) & (layer_summary.fold == item.fold_a) &
                                 (layer_summary.epoch == item.epoch_a)]
            right = layer_summary[(layer_summary.seed == item.seed_b) & (layer_summary.fold == item.fold_b) &
                                  (layer_summary.epoch == item.epoch_b)]
            if len(left) == 1 and len(right) == 1 and bool(left.conflict_top_quartile.iloc[0]) and \
                    bool(right.conflict_top_quartile.iloc[0]) and item.matrix_correlation >= 0.5:
                support.append((int(item.seed_a), int(item.fold_a), item.stage_a, item.stage_b, float(item.matrix_correlation)))
        support_seeds = sorted({item[0] for item in support})
        support_folds = sorted({item[1] for item in support})
        support_stages = sorted({stage for item in support for stage in item[2:4]})
        rows.append({
            "layer": layer, "stable_conflict_layer": len(support_seeds) >= required_seeds and
                len(support_folds) >= required_folds and len(support_stages) >= 2,
            "supporting_adjacent_run_pairs": len(support),
            "supporting_seeds": ",".join(map(str, support_seeds)),
            "supporting_folds": ",".join(map(str, support_folds)),
            "supporting_stages": ",".join(support_stages),
            "required_seed_count": required_seeds, "required_fold_count": required_folds,
            "mean_support_correlation": float(np.mean([item[4] for item in support])) if support else float("nan"),
        })
    return pd.DataFrame(rows)


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (seed, tissue, h2), task in predictions.groupby(["seed", "target_tissue", "mhc_restriction"], sort=True):
        rows.append({"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "seed": int(seed),
                     "target_tissue": tissue, "mhc_restriction": h2, "oof_rows": len(task),
                     **base.evaluate(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float))})
    per_task = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for seed, records in per_task.groupby("seed", sort=True):
        row: dict[str, Any] = {"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "seed": int(seed)}
        for metric in ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]:
            row[f"mean_task_{metric}"] = float(records[metric].mean())
        row["worst_task_auroc"] = float(records.auroc.min())
        row["worst6_task_auroc"] = float(records.nsmallest(6, "auroc").auroc.mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    stability_rows: list[dict[str, Any]] = []
    for metric in [column for column in summary if column.startswith("mean_task_") or column.startswith("worst")]:
        stability_rows.append({"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "metric": metric,
                               "seed_mean": float(summary[metric].mean()), "seed_sd": float(summary[metric].std(ddof=1)),
                               "seed_min": float(summary[metric].min()), "seed_max": float(summary[metric].max())})
    return per_task, summary, pd.DataFrame(stability_rows)


def train_and_audit_fold(
    args: argparse.Namespace, torch: Any, nn: Any, fitting: pd.DataFrame, mappings: dict[str, Any],
    peptide_length: int, device: str, seed: int, fold: int,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model = e3.define_model(torch, nn, peptide_length, len(mappings["tasks"]), len(mappings["tissue_to_id"]),
                            len(mappings["hla_to_id"]), args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    arrays = e5.task_arrays(fitting, mappings, peptide_length)
    steps = args.steps_per_epoch or int(np.ceil(max(len(item["label"]) for item in arrays) / args.task_batch_size))
    train_rng = np.random.default_rng(seed)
    fixed_batches = make_fixed_audit_batches(
        arrays, args.audit_task_batch_size, args.audit_batches_per_task,
        seed + 100_003 * (fold + 1) + args.audit_batch_seed_offset,
    )
    task_lookup = fitting[["task_id", "task_name", "tissue_id", "target_tissue", "hla_id", "mhc_restriction"]].drop_duplicates()
    task_names = dict(zip(task_lookup.task_id.astype(int), task_lookup.task_name))
    tissue_names = dict(zip(task_lookup.tissue_id.astype(int), task_lookup.target_tissue))
    h2_names = dict(zip(task_lookup.hla_id.astype(int), task_lookup.mhc_restriction))
    task_to_tissue = dict(zip(task_lookup.task_id.astype(int), task_lookup.tissue_id.astype(int)))
    task_to_h2 = dict(zip(task_lookup.task_id.astype(int), task_lookup.hla_id.astype(int)))
    audit_rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    stage_by_epoch = dict(zip(args.audit_epochs, ["early", "middle", "late"]))
    started = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        model.train(); losses: list[float] = []; entropies: list[float] = []
        for _ in range(steps):
            batch = e5.sample_balanced_batch(train_rng, arrays, args.task_batch_size)
            peptide, task, tissue, h2, label = [torch.as_tensor(value, device=device) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            task_losses, gates = e5.task_loss_vector(torch, model, peptide, task, tissue, h2, label, len(arrays), args.task_batch_size)
            entropy = -(gates * torch.log(gates.clamp_min(1e-12))).sum(dim=1).mean()
            objective = task_losses.mean() - args.gate_entropy_weight * entropy
            objective.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(task_losses.mean().detach().cpu())); entropies.append(float(entropy.detach().cpu()))
        history.append({"seed": seed, "fold": fold, "epoch": epoch, "mean_task_bce": float(np.mean(losses)),
                        "mean_gate_entropy": float(np.mean(entropies))})
        if epoch in stage_by_epoch:
            print(f"E16 seed={seed} fold={fold + 1}/{args.oof_folds} epoch={epoch}: auditing fixed fitting batches", flush=True)
            gradients = audit_task_gradients(torch, model, fixed_batches, device)
            for layer, task_gradients in gradients.items():
                common = {"experiment_name": EXPERIMENT, "candidate": CANDIDATE, "seed": seed, "fold": fold,
                          "epoch": epoch, "stage": stage_by_epoch[epoch], "layer": layer}
                audit_rows.extend(matrix_rows(torch, task_gradients, task_names, common, "task"))
                audit_rows.extend(matrix_rows(torch, aggregate_gradients(torch, task_gradients, task_to_h2), h2_names, common, "h2"))
                audit_rows.extend(matrix_rows(torch, aggregate_gradients(torch, task_gradients, task_to_tissue), tissue_names, common, "tissue"))
        print(f"E16 seed={seed} fold={fold + 1}/{args.oof_folds} epoch={epoch}/{args.epochs} "
              f"task_balanced_bce={np.mean(losses):.4f} gate_entropy={np.mean(entropies):.4f}", flush=True)
    resource = {
        "seed": seed, "fold": fold, "wall_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0,
    }
    return model, audit_rows, history, resource


def resolve_audit_epochs(args: argparse.Namespace) -> list[int]:
    if args.audit_epochs:
        epochs = sorted(set(args.audit_epochs))
        if len(epochs) != 3 or epochs[0] < 1 or epochs[-1] > args.epochs:
            raise ValueError("--audit-epochs must contain exactly three distinct epochs within training.")
        return epochs
    if args.epochs < 3:
        raise ValueError("At least 3 epochs are required for distinct early/middle/late audits.")
    return [1, (args.epochs + 1) // 2, args.epochs]


def validate_protocol(args: argparse.Namespace) -> None:
    if args.oof_folds < 1 or len(args.seeds) < 1:
        raise ValueError("At least one fold and seed are required.")
    if args.audit_batches_per_task < 2:
        raise ValueError("E16 requires multiple fixed fitting batches per task (>=2).")
    if args.n_experts != 3:
        raise ValueError("E16 is frozen to the E3b architecture with exactly 3 experts.")
    args.audit_epochs = resolve_audit_epochs(args)


def run(args: argparse.Namespace) -> None:
    validate_protocol(args)
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    raw = base.read_dataset(args.train); e3.validate_input(raw)
    train, _, mappings = base.add_task_columns(raw, raw.copy())
    if len(mappings["tasks"]) != 24:
        raise ValueError(f"E16 expects the frozen 24-task benchmark, found {len(mappings['tasks'])} tasks.")
    peptide_length = int(train.peptide_sequence.str.len().max())
    assignments = folds.make_pair_grouped_folds(train, args.oof_folds, args.oof_split_seed)
    prediction_parts: list[pd.DataFrame] = []
    matrix_rows_all: list[dict[str, Any]] = []
    history_all: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seed in args.seeds:
        for fold in range(args.oof_folds):
            fitting, held_out = train[assignments != fold].copy(), train[assignments == fold].copy()
            base.set_seed(int(seed), torch)
            print(f"E16 seed={seed} fold={fold + 1}/{args.oof_folds} fit={len(fitting)} holdout={len(held_out)} device={device}", flush=True)
            model, audit_rows, history, resource = train_and_audit_fold(
                args, torch, nn, fitting, mappings, peptide_length, device, int(seed), fold,
            )
            scores = e5.predict(torch, DataLoader, TensorDataset, model, held_out, peptide_length, args.batch_size, device)
            output = held_out[KEYS + ["label"]].copy()
            output.insert(0, "split", "oof"); output.insert(1, "candidate", CANDIDATE); output.insert(2, "seed", int(seed))
            output["score"] = scores
            prediction_parts.append(output[["split", "candidate", "seed", *KEYS, "label", "score"]])
            matrix_rows_all.extend(audit_rows); history_all.extend(history); resources.append(resource)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    if len(predictions) != len(train) * len(args.seeds) or predictions.duplicated(["seed", "sample_id"]).any():
        raise AssertionError("E16 E3b OOF predictions must cover every training row exactly once per seed.")
    matrices = pd.DataFrame(matrix_rows_all)
    summary = build_summaries(matrices)
    correlations = build_correlations(matrices)
    stable = stable_layer_table(summary, correlations, args.oof_folds, len(args.seeds))
    per_task, oof_summary, oof_stability = metric_tables(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "mousePMHC_phase5_e16"
    predictions.to_csv(args.output_dir / f"{prefix}_oof_predictions.csv", index=False)
    per_task.to_csv(args.output_dir / f"{prefix}_oof_per_task_metrics.csv", index=False)
    oof_summary.to_csv(args.output_dir / f"{prefix}_oof_summary_metrics.csv", index=False)
    oof_stability.to_csv(args.output_dir / f"{prefix}_oof_stability_metrics.csv", index=False)
    matrices.to_csv(args.output_dir / f"{prefix}_gradient_matrices.csv.gz", index=False, compression="gzip")
    summary.to_csv(args.output_dir / f"{prefix}_gradient_summary.csv", index=False)
    correlations.to_csv(args.output_dir / f"{prefix}_matrix_correlations.csv.gz", index=False, compression="gzip")
    stable.to_csv(args.output_dir / f"{prefix}_stable_conflict_layers.csv", index=False)
    pd.DataFrame(history_all).to_csv(args.output_dir / f"{prefix}_training_history.csv", index=False)
    pd.DataFrame(resources).to_csv(args.output_dir / f"{prefix}_resource_usage.csv", index=False)
    # E16's model is exactly the matched E3b baseline; retain an explicit artifact for downstream provenance.
    oof_summary.assign(matched_baseline="E3b", delta_mean_task_auroc=0.0).to_csv(
        args.output_dir / f"{prefix}_matched_e3b_baseline.csv", index=False,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "experiment_name": EXPERIMENT, "candidate": CANDIDATE, "test_data_read": False,
        "purpose": "layered primary-task gradient audit; not used for model ranking",
        "backbone": "unchanged E3b task-balanced Factorized MMoE", "train": str(args.train),
        "n_rows": len(train), "n_pairs": int(train.pair_id.nunique()), "n_tasks": len(mappings["tasks"]),
        "seeds": args.seeds, "oof_folds": args.oof_folds, "oof_split_seed": args.oof_split_seed,
        "epochs": args.epochs, "audit_epochs": args.audit_epochs, "audit_stages": ["early", "middle", "late"],
        "audit_batches_per_task": args.audit_batches_per_task, "audit_task_batch_size": args.audit_task_batch_size,
        "audit_batch_policy": "deterministically sampled from each fitting task once per seed-fold and reused at all stages",
        "gradient_policy": "primary-task BCE only; gradients averaged across fixed batches before cosine; dropout disabled during audit",
        "layer_policy": {"peptide_embedding": "amino_embedding", "shared_encoder": "peptide_encoder",
                         "experts": "each expert separately", "gate": "tissue_embedding + mhc_embedding + gate network",
                         "excluded": "24 task heads and all auxiliary losses"},
        "aggregation_policy": "H2/tissue gradients are equal-weight means of their member task gradients",
        "negative_conflict_strength": "mean(max(-pairwise_cosine, 0)) over unordered task pairs",
        "stable_conflict_rule": "support requires a layer to be in the within-run top quartile of negative task-cosine strength at both adjacent stages and matrix correlation >=0.5; stable requires supporting runs to cover ceil(2/3 folds), ceil(2/3 seeds), and >=2 stages",
        "model_parameter_count": int(parameter_count), "device": device,
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": max((item["peak_gpu_memory_bytes"] for item in resources), default=0),
        "git_commit": git_commit(), "cli": [sys.executable, *sys.argv],
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "cuda_available": bool(torch.cuda.is_available()), "cuda_version": torch.version.cuda,
                        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV")},
        "decision": "E18/E20/E21 are eligible only for layers marked stable_conflict_layer=true; E16 itself does not rank models",
    }
    (args.output_dir / f"{prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(oof_summary.to_string(index=False), flush=True)
    print(stable.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=project_path("results/mousePMHC_phase5_e16_gradient_audit"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260704, 20260705, 20260706])
    parser.add_argument("--oof-folds", type=int, default=3); parser.add_argument("--oof-split-seed", type=int, default=20260711)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--task-batch-size", type=int, default=16); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--expert-dim", type=int, default=64); parser.add_argument("--condition-dim", type=int, default=16)
    parser.add_argument("--gate-hidden-dim", type=int, default=64); parser.add_argument("--n-experts", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2); parser.add_argument("--gate-entropy-weight", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--audit-epochs", nargs=3, type=int, default=None,
                        help="Three distinct early/middle/late epochs; default is 1, ceil(epochs/2), epochs.")
    parser.add_argument("--audit-batches-per-task", type=int, default=4)
    parser.add_argument("--audit-task-batch-size", type=int, default=16)
    parser.add_argument("--audit-batch-seed-offset", type=int, default=160016)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
