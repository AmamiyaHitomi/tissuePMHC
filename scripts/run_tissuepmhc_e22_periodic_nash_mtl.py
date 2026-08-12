#!/usr/bin/env python3
"""Run E22: periodic Nash-MTL over three macro objectives.

The objectives are the primary 44-head BCE, tissue classification, and HLA
classification. Nash weights are recomputed periodically from gradients on the
shared embedding/encoder and reused between updates. Only the global branch is
retrained; the saved E14 HLA-plain predictions are reused for paired evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_tissuepmhc_auxiliary_soft_ensemble as e14
import run_tissuepmhc_e15_fusion_ablation as e15
import run_tissuepmhc_e21_gradient_similarity_auxiliary as e21
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "E22_periodic_nash_mtl"
MODEL = "e22_periodic_nash_hla_plain_rank_average"
GLOBAL_MODEL = "e22_global_periodic_nash_mtl"
OBJECTIVES = ("primary", "tissue", "hla")

DIAGNOSTIC_COLUMNS = [
    "experiment_name", "seed", "epoch", "mean_total_loss", "mean_bce_loss",
    "mean_tissue_loss", "mean_hla_loss", "mean_primary_weight",
    "mean_tissue_weight", "mean_hla_weight", "mean_primary_grad_norm",
    "mean_tissue_grad_norm", "mean_hla_grad_norm",
    "mean_primary_effective_fraction", "mean_tissue_effective_fraction",
    "mean_hla_effective_fraction", "mean_gram_condition", "nash_updates",
    "nash_failures", "min_observed_weight", "max_observed_weight",
]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def shared_gradient_vectors(torch: Any, losses: list[Any], parameters: list[Any]) -> Any:
    """Return one flattened detached shared-gradient vector per objective."""
    vectors = []
    for loss in losses:
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        pieces = [
            (gradient.detach() if gradient is not None else torch.zeros_like(parameter)).reshape(-1)
            for gradient, parameter in zip(gradients, parameters)
        ]
        vectors.append(torch.cat(pieces))
    return torch.stack(vectors)


def bounded_simplex(weights: np.ndarray, lower: float, upper: float, total: float) -> np.ndarray:
    """Project positive weights to box bounds while preserving their total."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 1e-12)
    weights *= total / weights.sum()
    for _ in range(20):
        clipped = np.clip(weights, lower, upper)
        free = (clipped > lower + 1e-12) & (clipped < upper - 1e-12)
        error = total - clipped.sum()
        weights = clipped
        if abs(error) < 1e-10:
            break
        if not free.any():
            break
        weights[free] += error * weights[free] / weights[free].sum()
    if abs(weights.sum() - total) > 1e-7:
        raise ValueError("Infeasible Nash weight bounds for the requested total.")
    return weights


def solve_nash_weights(
    gram: np.ndarray, previous: np.ndarray, lower: float, upper: float,
    max_evaluations: int, tolerance: float, total: float | None = None,
) -> tuple[np.ndarray, bool]:
    """Solve G alpha = 1/alpha in log space, then set sum(alpha)=3."""
    from scipy.optimize import least_squares

    gram = np.asarray(gram, dtype=np.float64)
    scale = max(float(np.linalg.norm(gram)), 1e-12)
    normalized = gram / scale + np.eye(len(gram)) * 1e-8

    def residual(log_weights: np.ndarray) -> np.ndarray:
        weights = np.exp(np.clip(log_weights, -20.0, 20.0))
        product = normalized @ weights
        if np.any(product <= 0) or not np.all(np.isfinite(product)):
            return np.full_like(weights, 1e6)
        return np.log(weights) + np.log(product)

    result = least_squares(
        residual, np.log(np.maximum(previous, 1e-8)), max_nfev=max_evaluations,
        ftol=tolerance, xtol=tolerance, gtol=tolerance,
    )
    raw = np.exp(np.clip(result.x, -20.0, 20.0))
    weights = bounded_simplex(raw, lower, upper, float(len(gram)) if total is None else total)
    valid = bool(result.success and np.all(np.isfinite(weights)))
    return weights, valid


def train_periodic_nash(
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
    shared = list(model.embedding.parameters()) + list(model.encoder.parameters())
    weights = np.ones(3, dtype=np.float64)
    diagnostics: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        values = {key: [] for key in [
            "total", "bce", "tissue_loss", "hla_loss", "primary_weight", "tissue_weight",
            "hla_weight", "primary_grad_norm", "tissue_grad_norm", "hla_grad_norm",
            "primary_effective_fraction", "tissue_effective_fraction", "hla_effective_fraction",
            "gram_condition",
        ]}
        updates = failures = 0
        observed_weights: list[float] = []
        for batch_index, batch in enumerate(loader):
            peptide_ids, task_ids, tissue_ids, hla_ids, labels = [item.to(device) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            encoded = model.encode(peptide_ids)
            logits = e21.task_logits_from_encoded(torch, model, encoded, task_ids)
            losses = [
                torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float()),
                torch.nn.functional.cross_entropy(model.tissue_classifier(encoded), tissue_ids),
                torch.nn.functional.cross_entropy(model.hla_classifier(encoded), hla_ids),
            ]
            if batch_index % args.nash_interval == 0:
                gradient_matrix = shared_gradient_vectors(torch, losses, shared)
                gram = (gradient_matrix @ gradient_matrix.T).double().cpu().numpy()
                norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
                try:
                    active = norms > max(float(norms.max()) * 1e-8, 1e-12)
                    candidate = np.full(3, args.min_weight, dtype=np.float64)
                    active_total = 3.0 - args.min_weight * int((~active).sum())
                    solved, success = solve_nash_weights(
                        gram[np.ix_(active, active)], weights[active], args.min_weight,
                        args.max_weight, args.nash_max_evaluations, args.nash_tolerance,
                        total=active_total,
                    )
                    candidate[active] = solved
                    if success:
                        weights = candidate
                    else:
                        failures += 1
                except (ValueError, FloatingPointError):
                    failures += 1
                effective = weights * norms
                effective = effective / max(float(effective.sum()), 1e-12)
                condition = float(np.linalg.cond(gram + np.eye(3) * 1e-12))
                for name, value in zip(OBJECTIVES, norms):
                    values[f"{name}_grad_norm"].append(float(value))
                for name, value in zip(OBJECTIVES, effective):
                    values[f"{name}_effective_fraction"].append(float(value))
                values["gram_condition"].append(condition)
                updates += 1
            total_loss = sum(float(weight) * loss for weight, loss in zip(weights, losses))
            total_loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            values["total"].append(float(total_loss.detach().cpu()))
            values["bce"].append(float(losses[0].detach().cpu()))
            values["tissue_loss"].append(float(losses[1].detach().cpu()))
            values["hla_loss"].append(float(losses[2].detach().cpu()))
            for name, weight in zip(OBJECTIVES, weights):
                values[f"{name}_weight"].append(float(weight))
                observed_weights.append(float(weight))

        mean = lambda key: float(np.mean(values[key])) if values[key] else float("nan")
        row = {
            "experiment_name": EXPERIMENT, "seed": seed, "epoch": epoch,
            "mean_total_loss": mean("total"), "mean_bce_loss": mean("bce"),
            "mean_tissue_loss": mean("tissue_loss"), "mean_hla_loss": mean("hla_loss"),
            "mean_primary_weight": mean("primary_weight"), "mean_tissue_weight": mean("tissue_weight"),
            "mean_hla_weight": mean("hla_weight"), "mean_primary_grad_norm": mean("primary_grad_norm"),
            "mean_tissue_grad_norm": mean("tissue_grad_norm"), "mean_hla_grad_norm": mean("hla_grad_norm"),
            "mean_primary_effective_fraction": mean("primary_effective_fraction"),
            "mean_tissue_effective_fraction": mean("tissue_effective_fraction"),
            "mean_hla_effective_fraction": mean("hla_effective_fraction"),
            "mean_gram_condition": mean("gram_condition"), "nash_updates": updates,
            "nash_failures": failures, "min_observed_weight": min(observed_weights),
            "max_observed_weight": max(observed_weights),
        }
        diagnostics.append(row)
        print(
            f"  E22 seed={seed} epoch={epoch}/{args.epochs} "
            f"weights=({row['mean_primary_weight']:.3f},{row['mean_tissue_weight']:.3f},"
            f"{row['mean_hla_weight']:.3f}) failures={failures}/{updates} "
            f"duration={e14.format_duration(time.perf_counter() - started)}", flush=True,
        )
    return model, diagnostics


def prediction_frame(args: argparse.Namespace, torch: Any, DataLoader: Any, TensorDataset: Any, model: Any,
                     train_df: pd.DataFrame, test_df: pd.DataFrame, mappings: dict[str, Any],
                     peptide_length: int, device: str, seed: int) -> pd.DataFrame:
    predictions, _ = e14.predict_branch(
        args, torch, DataLoader, TensorDataset, model, train_df, test_df, mappings["task_to_id"],
        peptide_length, device, seed, "global_periodic_nash_mtl", "all_tasks", True,
    )
    frames = []
    for _, task in sorted(predictions.items()):
        task_df = task["test_task"]
        frames.append(pd.DataFrame({
            "experiment_name": EXPERIMENT, "seed": seed, "branch": "global_periodic_nash_mtl",
            "branch_model": "aux_shared_heads_periodic_nash_mtl", "sample_id": task_df["sample_id"].to_numpy(),
            "target_tissue": task_df["target_tissue"].to_numpy(),
            "mhc_restriction": task_df["mhc_restriction"].to_numpy(), "label": task["y_true"],
            "probability": task["y_score"], "logit": task["y_logit"],
        }))
    return pd.concat(frames, ignore_index=True)


def make_rows(reference: pd.DataFrame, candidate_global: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    keys = ["seed", "sample_id", "target_tissue", "mhc_restriction"]
    candidate = candidate_global.rename(columns={"label": "label_candidate", "probability": "probability_candidate"})
    merged = reference.merge(candidate[keys + ["label_candidate", "probability_candidate"]], on=keys,
                             how="inner", validate="one_to_one")
    if len(merged) != len(reference) or not np.array_equal(merged["label_global_aux"], merged["label_candidate"]):
        raise ValueError("E22 predictions do not align exactly with saved E14 predictions.")
    rows, global_rows = [], []
    for (seed, tissue, hla), task in merged.groupby(["seed", "target_tissue", "mhc_restriction"], sort=True):
        y_true = task["label_candidate"].to_numpy(dtype=int)
        baseline = e15.fusion_scores(task)["e15_task_rank_average"]
        global_probability = task["probability_candidate"].to_numpy(dtype=float)
        global_rank = pd.Series(global_probability).rank(method="average", pct=True).to_numpy()
        hla_rank = pd.Series(task["probability_hla_plain"]).rank(method="average", pct=True).to_numpy()
        common = {
            "experiment_name": EXPERIMENT, "seed": int(seed), "target_tissue": tissue,
            "mhc_restriction": hla, "test_rows": len(task), "test_positive": int(y_true.sum()),
            "test_negative": int(len(task) - y_true.sum()), "global_branch": "global_periodic_nash_mtl",
            "hla_branch": "hla_plain", "fusion_formula": "0.5 * (within_task_rank(global) + within_task_rank(hla))",
        }
        rows.append({**common, "model": "e14_reference_task_rank_average", **base.evaluate(y_true, baseline)})
        rows.append({**common, "model": MODEL, **base.evaluate(y_true, 0.5 * (global_rank + hla_rank))})
        global_rows.append({
            "experiment_name": EXPERIMENT, "seed": int(seed), "model": GLOBAL_MODEL,
            "target_tissue": tissue, "mhc_restriction": hla, "test_rows": len(task),
            "test_positive": int(y_true.sum()), "test_negative": int(len(task) - y_true.sum()),
            **base.evaluate(y_true, global_probability),
        })
    return rows, global_rows


def run(args: argparse.Namespace) -> None:
    if args.nash_interval < 1 or args.epochs < 1:
        raise ValueError("epochs and nash_interval must be positive.")
    if not (0 < args.min_weight < 1 < args.max_weight < 3):
        raise ValueError("Require 0 < min_weight < 1 < max_weight < 3.")
    if 3 * args.min_weight > 3 or 3 * args.max_weight < 3:
        raise ValueError("Nash weight bounds are infeasible for sum(weights)=3.")
    started = time.perf_counter()
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    e14.set_seed(args.rng_preflight_seed, torch)
    e21.preflight_rng_roundtrip(torch)
    train_df, test_df = base.read_dataset(args.train), base.read_dataset(args.test)
    train_df, test_df, mappings = base.add_task_columns(train_df, test_df)
    if args.max_tasks:
        keep = set(mappings["tasks"][:args.max_tasks])
        train_df, test_df = train_df[train_df.task_name.isin(keep)].copy(), test_df[test_df.task_name.isin(keep)].copy()
        train_df, test_df, mappings = base.add_task_columns(train_df, test_df)
    reference = e15.require_aligned_e14a_predictions(args.e14_branch_predictions)
    reference = reference[reference.seed.isin(args.seeds)].copy()
    if sorted(reference.seed.astype(int).unique()) != sorted(args.seeds):
        raise ValueError("Saved E14 predictions do not contain every requested E22 seed.")
    reference = reference.merge(test_df[["sample_id", "target_tissue", "mhc_restriction"]],
                                on=["sample_id", "target_tissue", "mhc_restriction"], how="inner")
    peptide_length = int(max(train_df.peptide_sequence.str.len().max(), test_df.peptide_sequence.str.len().max()))
    print(f"device={device}; tasks={len(mappings['tasks'])}; seeds={args.seeds}; nash_interval={args.nash_interval}", flush=True)
    diagnostics, prediction_frames = [], []
    for seed in args.seeds:
        e14.set_seed(seed, torch)
        model, seed_diagnostics = train_periodic_nash(
            args, torch, nn, DataLoader, TensorDataset, train_df, mappings, peptide_length, device, seed,
        )
        diagnostics.extend(seed_diagnostics)
        prediction_frames.append(prediction_frame(
            args, torch, DataLoader, TensorDataset, model, train_df, test_df, mappings, peptide_length, device, seed,
        ))
    predictions = pd.concat(prediction_frames, ignore_index=True)
    rows, global_rows = make_rows(reference, predictions)
    summary, global_summary = base.summarize_results(rows), base.summarize_results(global_rows)
    base.write_csv(args.per_task_output, e21.PER_TASK_COLUMNS, rows)
    base.write_csv(args.summary_output, base.SUMMARY_COLUMNS, summary)
    base.write_csv(args.stability_output, base.STABILITY_COLUMNS, base.summarize_seed_stability(summary))
    base.write_csv(args.global_per_task_output, e21.GLOBAL_PER_TASK_COLUMNS, global_rows)
    base.write_csv(args.global_summary_output, base.SUMMARY_COLUMNS, global_summary)
    base.write_csv(args.diagnostics_output, DIAGNOSTIC_COLUMNS, diagnostics)
    base.write_csv(args.predictions_output, e21.PREDICTION_COLUMNS, predictions.to_dict("records"))
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps({
        "experiment_name": EXPERIMENT, "device": device, "seeds": args.seeds, "epochs": args.epochs,
        "objectives": list(OBJECTIVES), "nash_interval": args.nash_interval,
        "weight_constraint": {"sum": 3.0, "min": args.min_weight, "max": args.max_weight},
        "solver": "positive log-space least_squares for G alpha = 1/alpha with warm start",
        "shared_parameters": ["embedding", "encoder"], "fixed_hla_source": str(args.e14_branch_predictions),
        "selection_policy": "Nash weights use training-batch shared gradients only; test metrics never set weights.",
        "outputs": {"per_task": str(args.per_task_output), "diagnostics": str(args.diagnostics_output)},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    for path in [args.per_task_output, args.summary_output, args.stability_output, args.global_per_task_output,
                 args.global_summary_output, args.diagnostics_output, args.predictions_output, args.metadata_output]:
        print(f"wrote: {path}", flush=True)
    print(f"run total time: {e14.format_duration(time.perf_counter() - started)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz"))
    parser.add_argument("--test", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_test.csv.gz"))
    parser.add_argument("--e14-branch-predictions", type=Path, default=project_path("results/tissuePMHC_auxiliary_soft_ensemble/branch_predictions.csv"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260704])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=128); parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--nash-interval", type=int, default=10); parser.add_argument("--min-weight", type=float, default=0.05); parser.add_argument("--max-weight", type=float, default=2.90)
    parser.add_argument("--nash-max-evaluations", type=int, default=100); parser.add_argument("--nash-tolerance", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0); parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--rng-preflight-seed", type=int, default=20260711)
    root = project_path("results/tissuePMHC_e22_periodic_nash_mtl")
    parser.add_argument("--per-task-output", type=Path, default=root / "per_task_metrics.csv"); parser.add_argument("--summary-output", type=Path, default=root / "summary_metrics.csv"); parser.add_argument("--stability-output", type=Path, default=root / "stability_metrics.csv")
    parser.add_argument("--global-per-task-output", type=Path, default=root / "global_per_task_metrics.csv"); parser.add_argument("--global-summary-output", type=Path, default=root / "global_summary_metrics.csv")
    parser.add_argument("--diagnostics-output", type=Path, default=root / "nash_weight_diagnostics.csv"); parser.add_argument("--predictions-output", type=Path, default=root / "global_branch_predictions.csv"); parser.add_argument("--metadata-output", type=Path, default=root / "metadata.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
