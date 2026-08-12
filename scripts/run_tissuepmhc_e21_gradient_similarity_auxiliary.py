#!/usr/bin/env python3
"""Run E21: gradient-similarity gating for E14a auxiliary losses.

Only E14a's global auxiliary branch is retrained.  The already-saved E14 HLA
plain predictions are reused with the same seed, so this experiment isolates
whether dynamically gating the tissue/HLA auxiliary losses improves the global
representation.  Test labels are never used to set gates or hyperparameters.

At every ``--gating-interval`` training batches, the script measures cosine
similarity on the shared encoder between the primary BCE gradient and each
auxiliary-task gradient.  The respective auxiliary loss receives
``base_weight * max(0, cosine)`` until the next measurement.
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
PER_TASK_COLUMNS = [
    "experiment_name", "seed", "model", "target_tissue", "mhc_restriction",
    "test_rows", "test_positive", "test_negative", *e14.METRICS,
    "global_branch", "hla_branch", "fusion_formula",
]
GLOBAL_PER_TASK_COLUMNS = [
    "experiment_name", "seed", "model", "target_tissue", "mhc_restriction",
    "test_rows", "test_positive", "test_negative", *e14.METRICS,
]
DIAGNOSTIC_COLUMNS = [
    "experiment_name", "seed", "epoch", "mean_total_loss", "mean_bce_loss",
    "mean_tissue_loss", "mean_hla_loss", "mean_tissue_gate", "mean_hla_gate",
    "mean_tissue_cosine", "mean_hla_cosine", "tissue_conflict_fraction",
    "hla_conflict_fraction", "gating_measurements",
]
PREDICTION_COLUMNS = [
    "experiment_name", "seed", "branch", "branch_model", "sample_id", "target_tissue",
    "mhc_restriction", "label", "probability", "logit",
]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def capture_rng_state(torch: Any) -> dict[str, Any]:
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


def rng_states_equal(torch: Any, left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["python"] != right["python"]:
        return False
    left_np, right_np = left["numpy"], right["numpy"]
    if left_np[0] != right_np[0] or not np.array_equal(left_np[1], right_np[1]) or left_np[2:] != right_np[2:]:
        return False
    if not torch.equal(left["torch_cpu"], right["torch_cpu"]):
        return False
    left_cuda, right_cuda = left["torch_cuda"], right["torch_cuda"]
    return (left_cuda is None and right_cuda is None) or (
        left_cuda is not None and right_cuda is not None
        and len(left_cuda) == len(right_cuda)
        and all(torch.equal(a, b) for a, b in zip(left_cuda, right_cuda))
    )


def preflight_rng_roundtrip(torch: Any) -> None:
    """Verify save/restore before a run without changing later randomness."""
    before = capture_rng_state(torch)
    try:
        random.random()
        np.random.random()
        torch.rand(3)
        if torch.cuda.is_available():
            torch.rand(3, device="cuda")
    finally:
        restore_rng_state(torch, before)
    if not rng_states_equal(torch, before, capture_rng_state(torch)):
        raise RuntimeError("RNG preflight failed: Python/NumPy/Torch state did not round-trip exactly.")


def task_logits_from_encoded(torch: Any, model: Any, encoded: Any, task_ids: Any) -> Any:
    logits = encoded.new_empty(encoded.shape[0])
    for task_id in torch.unique(task_ids):
        mask = task_ids == task_id
        logits[mask] = model.heads[int(task_id.item())](encoded[mask]).squeeze(-1)
    return logits


def cosine_on_shared_encoder(torch: Any, primary: Any, auxiliary: Any, shared_parameters: list[Any]) -> float:
    """Return a detached cosine; autograd.grad leaves parameter .grad untouched."""
    primary_grads = torch.autograd.grad(primary, shared_parameters, retain_graph=True, allow_unused=True)
    auxiliary_grads = torch.autograd.grad(auxiliary, shared_parameters, retain_graph=True, allow_unused=True)
    dot = primary.new_zeros(())
    primary_norm = primary.new_zeros(())
    auxiliary_norm = primary.new_zeros(())
    for primary_grad, auxiliary_grad in zip(primary_grads, auxiliary_grads):
        if primary_grad is None or auxiliary_grad is None:
            continue
        dot = dot + (primary_grad.detach() * auxiliary_grad.detach()).sum()
        primary_norm = primary_norm + primary_grad.detach().square().sum()
        auxiliary_norm = auxiliary_norm + auxiliary_grad.detach().square().sum()
    denominator = torch.sqrt(primary_norm) * torch.sqrt(auxiliary_norm)
    if float(denominator.detach().cpu()) == 0.0:
        return 0.0
    return float((dot / denominator).clamp(-1.0, 1.0).detach().cpu())


def train_gradient_gated_global(
    args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any,
    train_df: pd.DataFrame, mappings: dict[str, Any], peptide_length: int, device: str, seed: int,
) -> tuple[Any, list[dict[str, object]]]:
    mapped = e14.e7.prepare_with_mapping(train_df, mappings["task_to_id"])
    loader = e14.build_aux_loader(args, torch, DataLoader, TensorDataset, mapped, peptide_length, True)
    model = e14.define_aux_shared_heads_model(
        args, torch, nn, peptide_length, len(mappings["task_to_id"]),
        len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    shared_parameters = [parameter for parameter in model.embedding.parameters()] + list(model.encoder.parameters())
    diagnostics: list[dict[str, object]] = []
    tissue_gate, hla_gate = args.tissue_loss_weight, args.hla_loss_weight

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        values: dict[str, list[float]] = {key: [] for key in [
            "total", "bce", "tissue_loss", "hla_loss", "tissue_gate", "hla_gate", "tissue_cosine", "hla_cosine",
        ]}
        tissue_conflicts = hla_conflicts = measurements = 0
        for batch_index, batch in enumerate(loader):
            peptide_ids, task_ids, tissue_ids, hla_ids, labels = [item.to(device) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            # One shared encoded representation makes the gradient comparison meaningful and avoids a
            # second dropout draw for the auxiliary losses in this E21-specific training path.
            encoded = model.encode(peptide_ids)
            logits = task_logits_from_encoded(torch, model, encoded, task_ids)
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
            tissue_loss = torch.nn.functional.cross_entropy(model.tissue_classifier(encoded), tissue_ids)
            hla_loss = torch.nn.functional.cross_entropy(model.hla_classifier(encoded), hla_ids)
            if batch_index % args.gating_interval == 0:
                tissue_cosine = cosine_on_shared_encoder(torch, bce_loss, tissue_loss, shared_parameters)
                hla_cosine = cosine_on_shared_encoder(torch, bce_loss, hla_loss, shared_parameters)
                tissue_gate = args.tissue_loss_weight * max(0.0, tissue_cosine)
                hla_gate = args.hla_loss_weight * max(0.0, hla_cosine)
                tissue_conflicts += int(tissue_cosine <= 0.0)
                hla_conflicts += int(hla_cosine <= 0.0)
                measurements += 1
                values["tissue_cosine"].append(tissue_cosine)
                values["hla_cosine"].append(hla_cosine)
            total_loss = bce_loss + tissue_gate * tissue_loss + hla_gate * hla_loss
            total_loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            values["total"].append(float(total_loss.detach().cpu()))
            values["bce"].append(float(bce_loss.detach().cpu()))
            values["tissue_loss"].append(float(tissue_loss.detach().cpu()))
            values["hla_loss"].append(float(hla_loss.detach().cpu()))
            values["tissue_gate"].append(tissue_gate)
            values["hla_gate"].append(hla_gate)
        diagnostics.append({
            "experiment_name": "E21_gradient_similarity_auxiliary", "seed": seed, "epoch": epoch,
            "mean_total_loss": float(np.mean(values["total"])), "mean_bce_loss": float(np.mean(values["bce"])),
            "mean_tissue_loss": float(np.mean(values["tissue_loss"])), "mean_hla_loss": float(np.mean(values["hla_loss"])),
            "mean_tissue_gate": float(np.mean(values["tissue_gate"])), "mean_hla_gate": float(np.mean(values["hla_gate"])),
            "mean_tissue_cosine": float(np.mean(values["tissue_cosine"])) if values["tissue_cosine"] else float("nan"),
            "mean_hla_cosine": float(np.mean(values["hla_cosine"])) if values["hla_cosine"] else float("nan"),
            "tissue_conflict_fraction": tissue_conflicts / measurements if measurements else float("nan"),
            "hla_conflict_fraction": hla_conflicts / measurements if measurements else float("nan"),
            "gating_measurements": measurements,
        })
        print(
            f"  E21 seed={seed} epoch={epoch}/{args.epochs} "
            f"tissue_gate={diagnostics[-1]['mean_tissue_gate']:.5f} hla_gate={diagnostics[-1]['mean_hla_gate']:.5f} "
            f"duration={e14.format_duration(time.perf_counter() - started)}", flush=True,
        )
    return model, diagnostics


def prediction_frame(
    args: argparse.Namespace, torch: Any, DataLoader: Any, TensorDataset: Any, model: Any,
    train_df: pd.DataFrame, test_df: pd.DataFrame, mappings: dict[str, Any], peptide_length: int,
    device: str, seed: int,
) -> pd.DataFrame:
    predictions, _ = e14.predict_branch(
        args, torch, DataLoader, TensorDataset, model, train_df, test_df, mappings["task_to_id"],
        peptide_length, device, seed, "global_aux_gradient_similarity", "all_tasks", True,
    )
    rows = []
    for _, task in sorted(predictions.items()):
        task_df = task["test_task"]
        rows.append(pd.DataFrame({
            "experiment_name": "E21_gradient_similarity_auxiliary", "seed": seed,
            "branch": "global_aux_gradient_similarity", "branch_model": "aux_shared_heads_gradient_similarity",
            "sample_id": task_df["sample_id"].to_numpy(), "target_tissue": task_df["target_tissue"].to_numpy(),
            "mhc_restriction": task_df["mhc_restriction"].to_numpy(), "label": task["y_true"],
            "probability": task["y_score"], "logit": task["y_logit"],
        }))
    return pd.concat(rows, ignore_index=True)


def make_rows(reference: pd.DataFrame, candidate_global: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    keys = ["seed", "sample_id", "target_tissue", "mhc_restriction"]
    candidate = candidate_global.rename(columns={
        "label": "label_candidate", "probability": "probability_candidate", "logit": "logit_candidate",
    })[keys + ["label_candidate", "probability_candidate", "logit_candidate"]]
    merged = reference.merge(candidate, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(reference):
        raise ValueError("E21 global predictions do not cover every saved E14 HLA prediction.")
    if not np.array_equal(merged["label_global_aux"].to_numpy(), merged["label_candidate"].to_numpy()):
        raise ValueError("E21 labels disagree with saved E14 branch predictions.")
    rows: list[dict[str, object]] = []
    global_rows: list[dict[str, object]] = []
    for (seed, tissue, hla), task in merged.groupby(["seed", "target_tissue", "mhc_restriction"], sort=True):
        y_true = task["label_candidate"].to_numpy(dtype=int)
        baseline = e15.fusion_scores(task)["e15_task_rank_average"]
        candidate_rank = pd.Series(task["probability_candidate"].to_numpy()).rank(method="average", pct=True).to_numpy()
        hla_rank = pd.Series(task["probability_hla_plain"].to_numpy()).rank(method="average", pct=True).to_numpy()
        candidate_score = 0.5 * (candidate_rank + hla_rank)
        common = {
            "experiment_name": "E21_gradient_similarity_auxiliary", "seed": int(seed),
            "target_tissue": tissue, "mhc_restriction": hla, "test_rows": len(task),
            "test_positive": int(y_true.sum()), "test_negative": int(len(task) - y_true.sum()),
            "global_branch": "global_aux_gradient_similarity", "hla_branch": "hla_plain",
            "fusion_formula": "0.5 * (within_task_rank(global_probability) + within_task_rank(hla_probability))",
        }
        rows.append({**common, "model": "e14_reference_task_rank_average", **base.evaluate(y_true, baseline)})
        rows.append({**common, "model": "e21_gradient_similarity_hla_plain_rank_average", **base.evaluate(y_true, candidate_score)})
        global_rows.append({
            "experiment_name": "E21_gradient_similarity_auxiliary", "seed": int(seed),
            "model": "e21_global_aux_gradient_similarity", "target_tissue": tissue, "mhc_restriction": hla,
            "test_rows": len(task), "test_positive": int(y_true.sum()), "test_negative": int(len(task) - y_true.sum()),
            **base.evaluate(y_true, task["probability_candidate"].to_numpy(dtype=float)),
        })
    return rows, global_rows


def run(args: argparse.Namespace) -> None:
    if args.gating_interval < 1:
        raise ValueError("gating_interval must be positive.")
    if args.tissue_loss_weight < 0 or args.hla_loss_weight < 0:
        raise ValueError("Auxiliary base weights must be non-negative.")
    started = time.perf_counter()
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    e14.set_seed(args.rng_preflight_seed, torch)
    preflight_rng_roundtrip(torch)
    print("RNG preflight passed: Python/NumPy/CPU-Torch/CUDA states round-trip exactly.", flush=True)
    train_df, test_df = base.read_dataset(args.train), base.read_dataset(args.test)
    train_df, test_df, mappings = base.add_task_columns(train_df, test_df)
    if args.max_tasks:
        keep = set(mappings["tasks"][:args.max_tasks])
        train_df, test_df = train_df[train_df.task_name.isin(keep)].copy(), test_df[test_df.task_name.isin(keep)].copy()
        train_df, test_df, mappings = base.add_task_columns(train_df, test_df)
    reference = e15.require_aligned_e14a_predictions(args.e14_branch_predictions)
    reference = reference[reference.seed.isin(args.seeds)].copy()
    if sorted(reference.seed.astype(int).unique()) != sorted(args.seeds):
        raise ValueError("Saved E14 branch predictions do not contain all requested E21 seeds.")
    allowed = test_df[["sample_id", "target_tissue", "mhc_restriction"]]
    reference = reference.merge(allowed, on=["sample_id", "target_tissue", "mhc_restriction"], how="inner")
    peptide_length = int(max(train_df.peptide_sequence.str.len().max(), test_df.peptide_sequence.str.len().max()))
    print(f"device={device}; tasks={len(mappings['tasks'])}; seeds={args.seeds}; gating_interval={args.gating_interval}", flush=True)
    diagnostic_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for seed in args.seeds:
        e14.set_seed(seed, torch)
        expected_state = capture_rng_state(torch)
        if not rng_states_equal(torch, expected_state, capture_rng_state(torch)):
            raise RuntimeError(f"RNG changed before E21 seed={seed} training.")
        seed_started = time.perf_counter()
        model, seed_diagnostics = train_gradient_gated_global(
            args, torch, nn, DataLoader, TensorDataset, train_df, mappings, peptide_length, device, seed,
        )
        diagnostic_rows.extend(seed_diagnostics)
        prediction_rows.append(prediction_frame(
            args, torch, DataLoader, TensorDataset, model, train_df, test_df, mappings, peptide_length, device, seed,
        ))
        print(f"time seed_total seed={seed} duration={e14.format_duration(time.perf_counter() - seed_started)}", flush=True)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    rows, global_rows = make_rows(reference, predictions)
    summary = base.summarize_results(rows)
    stability = base.summarize_seed_stability(summary)
    global_summary = base.summarize_results(global_rows)
    base.write_csv(args.per_task_output, PER_TASK_COLUMNS, rows)
    base.write_csv(args.summary_output, base.SUMMARY_COLUMNS, summary)
    base.write_csv(args.stability_output, base.STABILITY_COLUMNS, stability)
    base.write_csv(args.global_per_task_output, GLOBAL_PER_TASK_COLUMNS, global_rows)
    base.write_csv(args.global_summary_output, base.SUMMARY_COLUMNS, global_summary)
    base.write_csv(args.diagnostics_output, DIAGNOSTIC_COLUMNS, diagnostic_rows)
    base.write_csv(args.predictions_output, PREDICTION_COLUMNS, predictions.to_dict("records"))
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps({
        "experiment_name": "E21_gradient_similarity_auxiliary", "device": device, "seeds": args.seeds,
        "epochs": args.epochs, "gating_interval": args.gating_interval,
        "base_tissue_loss_weight": args.tissue_loss_weight, "base_hla_loss_weight": args.hla_loss_weight,
        "gating": "base_weight * max(0, cosine(primary_BCE_gradient, auxiliary_gradient))",
        "shared_parameters": ["embedding", "encoder"],
        "fixed_hla_source": str(args.e14_branch_predictions),
        "fusion": "task-rank average(new E21 global, fixed E14 HLA plain)",
        "selection_policy": "No validation/test metric is used to set gates; gates use current training-batch gradients only.",
        "rng_preflight": "passed round-trip check for Python, NumPy, CPU Torch, and all CUDA devices before loading data",
        "rng_seed_reset_before_each_training_seed": True,
        "outputs": {"per_task": str(args.per_task_output), "diagnostics": str(args.diagnostics_output)},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    for path in [args.per_task_output, args.summary_output, args.stability_output, args.global_per_task_output, args.global_summary_output, args.diagnostics_output, args.predictions_output, args.metadata_output]:
        print(f"wrote: {path}", flush=True)
    print(f"run total time: {e14.format_duration(time.perf_counter() - started)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz"))
    parser.add_argument("--test", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_test.csv.gz"))
    parser.add_argument("--e14-branch-predictions", type=Path, default=project_path("results/tissuePMHC_auxiliary_soft_ensemble/branch_predictions.csv"))
    parser.add_argument("--seeds", nargs="+", type=int, default=e14.DEFAULT_REPEAT_SEEDS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=128); parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--tissue-loss-weight", type=float, default=0.1); parser.add_argument("--hla-loss-weight", type=float, default=0.1)
    parser.add_argument("--gating-interval", type=int, default=10); parser.add_argument("--max-grad-norm", type=float, default=1.0); parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--rng-preflight-seed", type=int, default=20260711)
    root = project_path("results/tissuePMHC_e21_gradient_similarity_auxiliary")
    parser.add_argument("--per-task-output", type=Path, default=root / "per_task_metrics.csv"); parser.add_argument("--summary-output", type=Path, default=root / "summary_metrics.csv"); parser.add_argument("--stability-output", type=Path, default=root / "stability_metrics.csv")
    parser.add_argument("--global-per-task-output", type=Path, default=root / "global_per_task_metrics.csv"); parser.add_argument("--global-summary-output", type=Path, default=root / "global_summary_metrics.csv")
    parser.add_argument("--diagnostics-output", type=Path, default=root / "gradient_gating_diagnostics.csv"); parser.add_argument("--predictions-output", type=Path, default=root / "global_branch_predictions.csv"); parser.add_argument("--metadata-output", type=Path, default=root / "metadata.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
