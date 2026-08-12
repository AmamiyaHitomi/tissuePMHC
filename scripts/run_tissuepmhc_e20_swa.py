#!/usr/bin/env python3
"""Run E20: standard late-training SWA ablation for the two E14a branches.

For every seed and branch, training is shared through ``swa_start_epoch`` and
then split into two reproducible continuations:

* final_checkpoint: the original E14a AdamW trajectory;
* swa: AdamW + SWALR, with AveragedModel updated after every late epoch.

The script reports branch-level metrics and four task-rank fusion combinations:
final/final, SWA/final, final/SWA, and SWA/SWA.  It does not select a winner on
the test set; the outputs are an ablation table for downstream validation/OOF
selection.
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
import run_tissuepmhc_e19_training_ensemble as e19
import run_tissuepmhc_neural_baselines_v2 as base

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRANCH_VARIANTS = ["final_checkpoint", "swa"]
FUSION_VARIANTS = [
    ("final_checkpoint", "final_checkpoint"),
    ("swa", "final_checkpoint"),
    ("final_checkpoint", "swa"),
    ("swa", "swa"),
]
PER_TASK_COLUMNS = [
    "experiment_name", "seed", "model", "evaluation_level", "global_variant", "hla_variant",
    "n_swa_models", "target_tissue", "mhc_restriction", "test_rows", "test_positive", "test_negative",
    *e14.METRICS, "fusion_formula",
]


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def capture_rng_state(torch: Any) -> dict[str, Any]:
    """Capture every RNG used by this training pipeline."""
    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None,
    }


def restore_rng_state(torch: Any, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def rng_states_equal(torch: Any, left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["python"] != right["python"]:
        return False
    left_np, right_np = left["numpy"], right["numpy"]
    if left_np[0] != right_np[0] or not np.array_equal(left_np[1], right_np[1]):
        return False
    if left_np[2:] != right_np[2:]:
        return False
    if not torch.equal(left["torch_cpu"], right["torch_cpu"]):
        return False
    left_cuda, right_cuda = left["torch_cuda"], right["torch_cuda"]
    if (left_cuda is None) != (right_cuda is None):
        return False
    if left_cuda is not None:
        if len(left_cuda) != len(right_cuda):
            return False
        if any(not torch.equal(a, b) for a, b in zip(left_cuda, right_cuda)):
            return False
    return True


def train_epoch(
    args: argparse.Namespace, torch: Any, model: Any, loader: Any, optimizer: Any,
    device: str, use_aux: bool,
) -> None:
    model.train()
    for batch in loader:
        peptide_ids, task_ids, tissue_ids, hla_ids, labels = [item.to(device) for item in batch]
        optimizer.zero_grad(set_to_none=True)
        logits = model(peptide_ids, task_ids)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        tissue_logits, hla_logits = model.auxiliary_logits(peptide_ids)
        if use_aux:
            loss = loss + args.tissue_loss_weight * torch.nn.functional.cross_entropy(tissue_logits, tissue_ids)
            loss = loss + args.hla_loss_weight * torch.nn.functional.cross_entropy(hla_logits, hla_ids)
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()


def train_branch_variants(
    args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any,
    train_df: pd.DataFrame, test_df: pd.DataFrame, task_to_id: dict[str, int], n_tissues: int,
    n_hlas: int, peptide_length: int, device: str, branch: str, group_name: str, use_aux: bool,
) -> dict[str, dict]:
    """Train a shared prefix, then matched ordinary and SWA continuations."""
    mapped = e14.e7.prepare_with_mapping(train_df, task_to_id)
    loader = e14.build_aux_loader(args, torch, DataLoader, TensorDataset, mapped, peptide_length, True)
    model = e14.define_aux_shared_heads_model(
        args, torch, nn, peptide_length, len(task_to_id), n_tissues, n_hlas,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    for epoch in range(1, args.swa_start_epoch + 1):
        started = time.perf_counter()
        train_epoch(args, torch, model, loader, optimizer, device, use_aux)
        print(
            f"    time phase=shared branch={branch} group={group_name} "
            f"epoch={epoch}/{args.epochs} duration={e14.format_duration(time.perf_counter() - started)}",
            flush=True,
        )

    prefix_model_state = copy.deepcopy(model.state_dict())
    prefix_optimizer_state = copy.deepcopy(optimizer.state_dict())
    prefix_rng_state = capture_rng_state(torch)

    # Original E14a continuation: constant-learning-rate AdamW through the final epoch.
    final_epoch_rng_states = []
    for epoch in range(args.swa_start_epoch + 1, args.epochs + 1):
        final_epoch_rng_states.append(capture_rng_state(torch))
        started = time.perf_counter()
        train_epoch(args, torch, model, loader, optimizer, device, use_aux)
        print(
            f"    time phase=final branch={branch} group={group_name} "
            f"epoch={epoch}/{args.epochs} duration={e14.format_duration(time.perf_counter() - started)}",
            flush=True,
        )
    final_state = copy.deepcopy(model.state_dict())
    final_end_rng_state = capture_rng_state(torch)

    # Restore the exact prefix, optimizer, and shuffle RNG before the SWA continuation.
    model.load_state_dict(prefix_model_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.load_state_dict(prefix_optimizer_state)
    restore_rng_state(torch, prefix_rng_state)

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(
        optimizer,
        swa_lr=args.swa_lr,
        anneal_epochs=args.swa_anneal_epochs,
        anneal_strategy=args.swa_anneal_strategy,
    )
    n_swa_models = 0
    for epoch in range(args.swa_start_epoch + 1, args.epochs + 1):
        if args.verify_rng_pairing:
            expected_rng = final_epoch_rng_states[epoch - args.swa_start_epoch - 1]
            actual_rng = capture_rng_state(torch)
            if not rng_states_equal(torch, expected_rng, actual_rng):
                raise RuntimeError(
                    f"RNG mismatch before paired continuations: branch={branch} "
                    f"group={group_name} epoch={epoch}"
                )
        started = time.perf_counter()
        train_epoch(args, torch, model, loader, optimizer, device, use_aux)
        swa_model.update_parameters(model)
        n_swa_models += 1
        swa_scheduler.step()
        print(
            f"    time phase=swa branch={branch} group={group_name} "
            f"epoch={epoch}/{args.epochs} n_averaged={n_swa_models} "
            f"lr={optimizer.param_groups[0]['lr']:.6g} "
            f"duration={e14.format_duration(time.perf_counter() - started)}",
            flush=True,
        )

    if args.verify_rng_pairing and not rng_states_equal(
        torch, final_end_rng_state, capture_rng_state(torch),
    ):
        raise RuntimeError(
            f"RNG mismatch after paired continuations: branch={branch} group={group_name}"
        )

    # AuxSharedHeadsModel currently has no BatchNorm buffers, so update_bn is unnecessary.
    swa_state = copy.deepcopy(swa_model.module.state_dict())
    return {
        "final_checkpoint": e19.predict_members(
            args, torch, DataLoader, TensorDataset, model, [final_state], train_df, test_df,
            task_to_id, peptide_length, device,
        ),
        "swa": e19.predict_members(
            args, torch, DataLoader, TensorDataset, model, [swa_state], train_df, test_df,
            task_to_id, peptide_length, device,
        ),
    }


def train_seed(
    args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any,
    train_df: pd.DataFrame, test_df: pd.DataFrame, mappings: dict[str, Any], peptide_length: int,
    device: str,
) -> tuple[dict[str, dict], dict[str, dict]]:
    print("  train global_aux final/SWA", flush=True)
    global_variants = train_branch_variants(
        args, torch, nn, DataLoader, TensorDataset, train_df, test_df, mappings["task_to_id"],
        len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]), peptide_length, device,
        "global_aux", "all_tasks", True,
    )
    hla_variants = {variant: {} for variant in BRANCH_VARIANTS}
    hlas = sorted(set(train_df.mhc_restriction) & set(test_df.mhc_restriction))
    for index, hla in enumerate(hlas, 1):
        hla_train = train_df[train_df.mhc_restriction == hla].copy()
        hla_test = test_df[test_df.mhc_restriction == hla].copy()
        tasks = sorted(set(hla_train.task_name) & set(hla_test.task_name))
        if not tasks:
            continue
        print(f"  train hla_plain final/SWA {index:02d}/{len(hlas)} {hla}", flush=True)
        variants = train_branch_variants(
            args, torch, nn, DataLoader, TensorDataset, hla_train, hla_test,
            {task: i for i, task in enumerate(tasks)}, len(mappings["tissue_to_id"]),
            len(mappings["hla_to_id"]), peptide_length, device, "hla_plain", hla, False,
        )
        for variant in BRANCH_VARIANTS:
            hla_variants[variant].update(variants[variant])
    return global_variants, hla_variants


def row_base(seed: int, model: str, evaluation_level: str, global_variant: str, hla_variant: str, task: dict[str, Any], n_swa_models: int, fusion_formula: str) -> dict[str, object]:
    first = task["test_task"].iloc[0]
    labels = task["y_true"]
    return {
        "experiment_name": "E20_swa",
        "seed": seed,
        "model": model,
        "evaluation_level": evaluation_level,
        "global_variant": global_variant,
        "hla_variant": hla_variant,
        "n_swa_models": n_swa_models,
        "target_tissue": first.target_tissue,
        "mhc_restriction": first.mhc_restriction,
        "test_rows": len(labels),
        "test_positive": int(labels.sum()),
        "test_negative": int(len(labels) - labels.sum()),
        "fusion_formula": fusion_formula,
    }


def make_rows(
    seed: int, global_variants: dict[str, dict], hla_variants: dict[str, dict], n_swa_models: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    branch_rows: list[dict[str, object]] = []
    fused_rows: list[dict[str, object]] = []
    common_keys = sorted(set(global_variants["final_checkpoint"]) & set(hla_variants["final_checkpoint"]))

    for key in common_keys:
        for variant in BRANCH_VARIANTS:
            global_task, hla_task = global_variants[variant][key], hla_variants[variant][key]
            if not np.array_equal(global_task["y_true"], hla_task["y_true"]):
                raise ValueError(f"Label mismatch for branch metrics {variant} {key}")
            global_row = row_base(
                seed, f"e20_global_{variant}", "global_branch", variant, "not_applicable",
                global_task, n_swa_models if variant == "swa" else 1, "global_branch_only",
            )
            global_row.update(base.evaluate(global_task["y_true"], global_task["y_score"]))
            branch_rows.append(global_row)
            hla_row = row_base(
                seed, f"e20_hla_{variant}", "hla_branch", "not_applicable", variant,
                hla_task, n_swa_models if variant == "swa" else 1, "hla_branch_only",
            )
            hla_row.update(base.evaluate(hla_task["y_true"], hla_task["y_score"]))
            branch_rows.append(hla_row)

        for global_variant, hla_variant in FUSION_VARIANTS:
            global_task = global_variants[global_variant][key]
            hla_task = hla_variants[hla_variant][key]
            if not np.array_equal(global_task["y_true"], hla_task["y_true"]):
                raise ValueError(f"Label mismatch for fusion {global_variant}/{hla_variant} {key}")
            scores = np.asarray(e15.fusion_scores(pd.DataFrame({
                "probability_global_aux": global_task["y_score"],
                "probability_hla_plain": hla_task["y_score"],
                "logit_global_aux": np.zeros(len(global_task["y_score"])),
                "logit_hla_plain": np.zeros(len(hla_task["y_score"])),
            }))["e15_task_rank_average"])
            short_global = "final" if global_variant == "final_checkpoint" else "swa"
            short_hla = "final" if hla_variant == "final_checkpoint" else "swa"
            row = row_base(
                seed, f"e20_global_{short_global}_hla_{short_hla}_rank_average", "fused",
                global_variant, hla_variant, global_task,
                n_swa_models if "swa" in (global_variant, hla_variant) else 1,
                "task_rank_average(global_branch, hla_branch)",
            )
            row.update(base.evaluate(global_task["y_true"], scores))
            fused_rows.append(row)
    return branch_rows, fused_rows


def write_metric_set(per_task_path: Path, summary_path: Path, stability_path: Path, rows: list[dict[str, object]]) -> None:
    summary = base.summarize_results(rows)
    stability = base.summarize_seed_stability(summary)
    base.write_csv(per_task_path, PER_TASK_COLUMNS, rows)
    base.write_csv(summary_path, base.SUMMARY_COLUMNS, summary)
    base.write_csv(stability_path, base.STABILITY_COLUMNS, stability)


def run(args: argparse.Namespace) -> None:
    if not 1 <= args.swa_start_epoch < args.epochs:
        raise ValueError("swa_start_epoch must be in [1, epochs - 1].")
    if args.swa_lr <= 0:
        raise ValueError("swa_lr must be positive.")
    if not 1 <= args.swa_anneal_epochs <= args.epochs - args.swa_start_epoch:
        raise ValueError("swa_anneal_epochs must not exceed the number of SWA epochs.")

    started = time.perf_counter()
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    train_df, test_df = base.read_dataset(args.train), base.read_dataset(args.test)
    train_df, test_df, mappings = base.add_task_columns(train_df, test_df)
    if args.max_tasks:
        keep = set(mappings["tasks"][:args.max_tasks])
        train_df = train_df[train_df.task_name.isin(keep)].copy()
        test_df = test_df[test_df.task_name.isin(keep)].copy()
        train_df, test_df, mappings = base.add_task_columns(train_df, test_df)
    peptide_length = int(max(train_df.peptide_sequence.str.len().max(), test_df.peptide_sequence.str.len().max()))
    n_swa_models = args.epochs - args.swa_start_epoch
    branch_rows: list[dict[str, object]] = []
    fused_rows: list[dict[str, object]] = []
    print(
        f"device: {device}; epochs: {args.epochs}; swa_start_epoch: {args.swa_start_epoch}; "
        f"swa_lr: {args.swa_lr}; n_swa_models: {n_swa_models}",
        flush=True,
    )
    for seed in args.seeds:
        seed_started = time.perf_counter()
        e14.set_seed(seed, torch)
        print(f"experiment: E20_swa seed={seed}", flush=True)
        global_variants, hla_variants = train_seed(
            args, torch, nn, DataLoader, TensorDataset, train_df, test_df, mappings,
            peptide_length, device,
        )
        seed_branch_rows, seed_fused_rows = make_rows(seed, global_variants, hla_variants, n_swa_models)
        branch_rows.extend(seed_branch_rows)
        fused_rows.extend(seed_fused_rows)
        for model in sorted({row["model"] for row in seed_fused_rows}):
            values = [row["auroc"] for row in seed_fused_rows if row["model"] == model]
            print(f"  {model} mean_auroc={np.mean(values):.4f}", flush=True)
        print(
            f"time seed_total seed={seed} duration={e14.format_duration(time.perf_counter() - seed_started)}",
            flush=True,
        )

    write_metric_set(args.per_task_output, args.summary_output, args.stability_output, fused_rows)
    write_metric_set(
        args.branch_per_task_output, args.branch_summary_output, args.branch_stability_output, branch_rows,
    )
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps({
        "experiment_name": "E20_swa",
        "seeds": args.seeds,
        "device": device,
        "epochs": args.epochs,
        "swa_start_epoch": args.swa_start_epoch,
        "swa_lr": args.swa_lr,
        "swa_anneal_epochs": args.swa_anneal_epochs,
        "swa_anneal_strategy": args.swa_anneal_strategy,
        "n_swa_models": n_swa_models,
        "optimizer": "AdamW",
        "fusion": "task_rank_average",
        "fusion_variants": [
            {"global": global_variant, "hla": hla_variant}
            for global_variant, hla_variant in FUSION_VARIANTS
        ],
        "selection_policy": "report all ablations; do not choose using test metrics",
        "batch_norm_update": "not required: current E14a model contains no BatchNorm layers",
        "rng_pairing_verified": args.verify_rng_pairing,
        "rng_states": ["python", "numpy", "torch_cpu", "all_torch_cuda_devices"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    for path in [
        args.per_task_output, args.summary_output, args.stability_output,
        args.branch_per_task_output, args.branch_summary_output, args.branch_stability_output,
        args.metadata_output,
    ]:
        print(f"wrote: {path}", flush=True)
    print(f"run total time: {e14.format_duration(time.perf_counter() - started)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz"))
    parser.add_argument("--test", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_test.csv.gz"))
    parser.add_argument("--seeds", nargs="+", type=int, default=e14.DEFAULT_REPEAT_SEEDS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--swa-start-epoch", type=int, default=15)
    parser.add_argument("--swa-lr", type=float, default=5e-4)
    parser.add_argument("--swa-anneal-epochs", type=int, default=5)
    parser.add_argument("--swa-anneal-strategy", choices=["cos", "linear"], default="cos")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--tissue-loss-weight", type=float, default=0.1)
    parser.add_argument("--hla-loss-weight", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument(
        "--no-verify-rng-pairing",
        action="store_false",
        dest="verify_rng_pairing",
        help="Disable paired-trajectory RNG assertions (not recommended).",
    )
    parser.set_defaults(verify_rng_pairing=True)
    output_dir = project_path("results/tissuePMHC_e20_swa")
    parser.add_argument("--per-task-output", type=Path, default=output_dir / "per_task_metrics.csv")
    parser.add_argument("--summary-output", type=Path, default=output_dir / "summary_metrics.csv")
    parser.add_argument("--stability-output", type=Path, default=output_dir / "stability_metrics.csv")
    parser.add_argument("--branch-per-task-output", type=Path, default=output_dir / "branch_per_task_metrics.csv")
    parser.add_argument("--branch-summary-output", type=Path, default=output_dir / "branch_summary_metrics.csv")
    parser.add_argument("--branch-stability-output", type=Path, default=output_dir / "branch_stability_metrics.csv")
    parser.add_argument("--metadata-output", type=Path, default=output_dir / "metadata.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
