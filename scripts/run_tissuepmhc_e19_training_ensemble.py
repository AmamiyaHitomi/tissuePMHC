#!/usr/bin/env python3
"""Run E19 checkpoint and snapshot ensembles for the two E14a core branches.

For each seed, the script compares:
* final_checkpoint: ordinary E14a training, final epoch only;
* checkpoint_ensemble: average selected late ordinary-training checkpoints;
* snapshot_ensemble: cosine-restart training, average cycle-end snapshots.

Each variant averages predictions within global_aux and hla_plain, then applies
the E15 task-wise rank fusion rule between the two branches.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_tissuepmhc_auxiliary_soft_ensemble as e14
import run_tissuepmhc_e15_fusion_ablation as e15
import run_tissuepmhc_neural_baselines_v2 as base

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ["final_checkpoint", "checkpoint_ensemble", "snapshot_ensemble"]
PER_TASK_COLUMNS = [
    "experiment_name", "seed", "model", "training_variant", "n_member_checkpoints", "target_tissue", "mhc_restriction",
    "test_rows", "test_positive", "test_negative", *e14.METRICS, "fusion_formula",
]


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def capture_rng_state(torch: Any) -> dict[str, Any]:
    """Capture all RNG streams so paired training differs only by schedule."""
    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else None,
    }


def restore_rng_state(torch: Any, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def train_with_snapshots(
    args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any, train_df: pd.DataFrame,
    task_to_id: dict[str, int], n_tissues: int, n_hlas: int, peptide_length: int, device: str,
    branch: str, group_name: str, use_aux: bool, mode: str,
) -> tuple[Any, list[dict[str, Any]]]:
    mapped = e14.e7.prepare_with_mapping(train_df, task_to_id)
    model = e14.define_aux_shared_heads_model(args, torch, nn, peptide_length, len(task_to_id), n_tissues, n_hlas).to(device)
    loader = e14.build_aux_loader(args, torch, DataLoader, TensorDataset, mapped, peptide_length, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None
    if mode == "snapshot":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.snapshot_cycle_epochs, T_mult=1)
    snapshots: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        for batch in loader:
            peptide_ids, task_ids, tissue_ids, hla_ids, labels = [item.to(device) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            logits = model(peptide_ids, task_ids)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
            tissue_logits, hla_logits = model.auxiliary_logits(peptide_ids)
            loss = bce
            if use_aux:
                loss = loss + args.tissue_loss_weight * torch.nn.functional.cross_entropy(tissue_logits, tissue_ids)
                loss = loss + args.hla_loss_weight * torch.nn.functional.cross_entropy(hla_logits, hla_ids)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        if scheduler is not None:
            scheduler.step(epoch)
        if epoch in args.member_epochs:
            snapshots.append(copy.deepcopy(model.state_dict()))
        print(f"    time epoch mode={mode} branch={branch} group={group_name} epoch={epoch}/{args.epochs} duration={e14.format_duration(time.perf_counter() - epoch_started)}")
    if not snapshots:
        raise ValueError("No ensemble member checkpoints were captured; check --member-epochs.")
    return model, snapshots


def predict_members(
    args: argparse.Namespace, torch: Any, DataLoader: Any, TensorDataset: Any, model: Any, states: list[dict[str, Any]],
    train_df: pd.DataFrame, test_df: pd.DataFrame, task_to_id: dict[str, int], peptide_length: int, device: str,
) -> dict[tuple[str, str], dict[str, object]]:
    train_mapped, test_mapped = e14.e7.prepare_with_mapping(train_df, task_to_id), e14.e7.prepare_with_mapping(test_df, task_to_id)
    result: dict[tuple[str, str], dict[str, object]] = {}
    for task_name in sorted(set(train_mapped.task_name) & set(test_mapped.task_name)):
        train_task, test_task = train_mapped[train_mapped.task_name == task_name], test_mapped[test_mapped.task_name == task_name]
        loader = e14.build_aux_loader(args, torch, DataLoader, TensorDataset, test_task, peptide_length, False)
        member_scores = []
        labels = None
        for state in states:
            model.load_state_dict(state)
            model.eval()
            scores, member_labels = [], []
            with torch.no_grad():
                for batch in loader:
                    peptide_ids, task_ids, _, _, y = [item.to(device) for item in batch]
                    scores.append(torch.sigmoid(model(peptide_ids, task_ids)).cpu().numpy())
                    member_labels.append(y.cpu().numpy())
            member_scores.append(np.concatenate(scores))
            current_labels = np.concatenate(member_labels)
            if labels is None: labels = current_labels
            elif not np.array_equal(labels, current_labels): raise AssertionError("Inconsistent member labels")
        first = test_task.iloc[0]
        result[(str(first.target_tissue), str(first.mhc_restriction))] = {"train_task": train_task, "test_task": test_task, "y_true": labels, "y_score": np.mean(member_scores, axis=0)}
    return result


def run_branch_variants(
    args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any, train_df: pd.DataFrame,
    test_df: pd.DataFrame, task_to_id: dict[str, int], n_tissues: int, n_hlas: int, peptide_length: int, device: str,
    branch: str, group_name: str, use_aux: bool,
) -> dict[str, dict]:
    # Start ordinary and cosine-restart training from identical initial RNG
    # states, so their difference is the learning-rate schedule rather than a
    # different random initialization or first DataLoader shuffle.
    rng_state = capture_rng_state(torch)
    ordinary_model, ordinary_states = train_with_snapshots(args, torch, nn, DataLoader, TensorDataset, train_df, task_to_id, n_tissues, n_hlas, peptide_length, device, branch, group_name, use_aux, "ordinary")
    # This is the RNG position an ordinary E14a run would have reached before
    # starting the next HLA group.  Restore it after the extra snapshot pass.
    ordinary_end_rng_state = capture_rng_state(torch)
    restore_rng_state(torch, rng_state)
    snapshot_model, snapshot_states = train_with_snapshots(args, torch, nn, DataLoader, TensorDataset, train_df, task_to_id, n_tissues, n_hlas, peptide_length, device, branch, group_name, use_aux, "snapshot")
    restore_rng_state(torch, ordinary_end_rng_state)
    return {
        "final_checkpoint": predict_members(args, torch, DataLoader, TensorDataset, ordinary_model, [ordinary_states[-1]], train_df, test_df, task_to_id, peptide_length, device),
        "checkpoint_ensemble": predict_members(args, torch, DataLoader, TensorDataset, ordinary_model, ordinary_states, train_df, test_df, task_to_id, peptide_length, device),
        "snapshot_ensemble": predict_members(args, torch, DataLoader, TensorDataset, snapshot_model, snapshot_states, train_df, test_df, task_to_id, peptide_length, device),
    }


def train_seed(args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any, train_df: pd.DataFrame, test_df: pd.DataFrame, mappings: dict[str, Any], peptide_length: int, device: str) -> tuple[dict[str, dict], dict[str, dict]]:
    print("  train global_aux variants")
    global_variants = run_branch_variants(args, torch, nn, DataLoader, TensorDataset, train_df, test_df, mappings["task_to_id"], len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]), peptide_length, device, "global_aux", "all_tasks", True)
    hla_variants = {variant: {} for variant in VARIANTS}
    hlas = sorted(set(train_df.mhc_restriction) & set(test_df.mhc_restriction))
    for index, hla in enumerate(hlas, 1):
        hla_train, hla_test = train_df[train_df.mhc_restriction == hla].copy(), test_df[test_df.mhc_restriction == hla].copy()
        tasks = sorted(set(hla_train.task_name) & set(hla_test.task_name))
        if not tasks: continue
        print(f"  train hla_plain variants {index:02d}/{len(hlas)} {hla}")
        local_mapping = {task: i for i, task in enumerate(tasks)}
        variants = run_branch_variants(args, torch, nn, DataLoader, TensorDataset, hla_train, hla_test, local_mapping, len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]), peptide_length, device, "hla_plain", hla, False)
        for variant in VARIANTS: hla_variants[variant].update(variants[variant])
    return global_variants, hla_variants


def make_rows(seed: int, global_variants: dict[str, dict], hla_variants: dict[str, dict], n_members: int) -> list[dict[str, object]]:
    rows = []
    for variant in VARIANTS:
        for key in sorted(set(global_variants[variant]) & set(hla_variants[variant])):
            g, h = global_variants[variant][key], hla_variants[variant][key]
            if not np.array_equal(g["y_true"], h["y_true"]): raise ValueError(f"Label mismatch {variant} {key}")
            task = g["test_task"]
            scores = e15.fusion_scores(pd.DataFrame({"probability_global_aux": g["y_score"], "probability_hla_plain": h["y_score"], "logit_global_aux": np.zeros(len(g["y_score"])), "logit_hla_plain": np.zeros(len(h["y_score"]))}))["e15_task_rank_average"]
            first = task.iloc[0]
            rows.append({"experiment_name": "E19_training_ensemble", "seed": seed, "model": f"e19_{variant}_rank_average", "training_variant": variant, "n_member_checkpoints": 1 if variant == "final_checkpoint" else n_members, "target_tissue": first.target_tissue, "mhc_restriction": first.mhc_restriction, "test_rows": len(task), "test_positive": int(g["y_true"].sum()), "test_negative": int(len(task) - g["y_true"].sum()), **base.evaluate(g["y_true"], scores), "fusion_formula": "task_rank_average(global_branch, hla_branch)"})
    return rows


def run(args: argparse.Namespace) -> None:
    if args.epochs not in args.member_epochs: raise ValueError("member_epochs must include the final epoch for final_checkpoint.")
    if any(epoch < 1 or epoch > args.epochs for epoch in args.member_epochs): raise ValueError("member_epochs must fall within training epochs.")
    started = time.perf_counter(); torch, nn, DataLoader, TensorDataset = base.require_torch(); device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    train_df, test_df = base.read_dataset(args.train), base.read_dataset(args.test); train_df, test_df, mappings = base.add_task_columns(train_df, test_df)
    if args.max_tasks:
        keep = set(mappings["tasks"][:args.max_tasks]); train_df, test_df = train_df[train_df.task_name.isin(keep)].copy(), test_df[test_df.task_name.isin(keep)].copy(); train_df, test_df, mappings = base.add_task_columns(train_df, test_df)
    peptide_length = int(max(train_df.peptide_sequence.str.len().max(), test_df.peptide_sequence.str.len().max()))
    print(f"device: {device}; member_epochs: {args.member_epochs}; snapshot_cycle_epochs: {args.snapshot_cycle_epochs}")
    rows = []
    for seed in args.seeds:
        seed_started = time.perf_counter(); e14.set_seed(seed, torch); print(f"experiment: E19_training_ensemble seed={seed}")
        global_variants, hla_variants = train_seed(args, torch, nn, DataLoader, TensorDataset, train_df, test_df, mappings, peptide_length, device)
        seed_rows = make_rows(seed, global_variants, hla_variants, len(args.member_epochs)); rows.extend(seed_rows)
        for variant in VARIANTS: print(f"  {variant} mean_auroc={np.mean([r['auroc'] for r in seed_rows if r['training_variant']==variant]):.4f}")
        print(f"time seed_total seed={seed} duration={e14.format_duration(time.perf_counter()-seed_started)}")
    summary = base.summarize_results(rows); stability = base.summarize_seed_stability(summary)
    base.write_csv(args.per_task_output, PER_TASK_COLUMNS, rows); base.write_csv(args.summary_output, base.SUMMARY_COLUMNS, summary); base.write_csv(args.stability_output, base.STABILITY_COLUMNS, stability)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True); args.metadata_output.write_text(json.dumps({"experiment_name":"E19_training_ensemble","seeds":args.seeds,"device":device,"member_epochs":args.member_epochs,"snapshot_cycle_epochs":args.snapshot_cycle_epochs,"fusion":"task_rank_average"},indent=2,ensure_ascii=False),encoding="utf-8")
    for path in [args.per_task_output,args.summary_output,args.stability_output,args.metadata_output]: print(f"wrote: {path}")
    print(f"run total time: {e14.format_duration(time.perf_counter()-started)}")


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--train",type=Path,default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz")); p.add_argument("--test",type=Path,default=project_path("data/tissuePMHC/tissuePMHC_test.csv.gz")); p.add_argument("--seeds",nargs="+",type=int,default=e14.DEFAULT_REPEAT_SEEDS); p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    p.add_argument("--epochs",type=int,default=25); p.add_argument("--member-epochs",nargs="+",type=int,default=[15,20,25]); p.add_argument("--snapshot-cycle-epochs",type=int,default=5); p.add_argument("--batch-size",type=int,default=512); p.add_argument("--learning-rate",type=float,default=1e-3); p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--embedding-dim",type=int,default=16); p.add_argument("--hidden-dim",type=int,default=128); p.add_argument("--dropout",type=float,default=0.2); p.add_argument("--tissue-loss-weight",type=float,default=0.1); p.add_argument("--hla-loss-weight",type=float,default=0.1); p.add_argument("--max-grad-norm",type=float,default=1.0); p.add_argument("--max-tasks",type=int,default=0)
    p.add_argument("--per-task-output",type=Path,default=project_path("results/tissuePMHC_e19_training_ensemble/per_task_metrics.csv")); p.add_argument("--summary-output",type=Path,default=project_path("results/tissuePMHC_e19_training_ensemble/summary_metrics.csv")); p.add_argument("--stability-output",type=Path,default=project_path("results/tissuePMHC_e19_training_ensemble/stability_metrics.csv")); p.add_argument("--metadata-output",type=Path,default=project_path("results/tissuePMHC_e19_training_ensemble/metadata.json")); return p.parse_args()

if __name__ == "__main__": run(parse_args())
