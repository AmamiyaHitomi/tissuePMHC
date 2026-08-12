#!/usr/bin/env python3
"""Validation-only selection and paired diagnosis of Human auxiliary losses.

This program deliberately reads only the original Human training file.  It
creates one reproducible, pair-grouped internal validation split per seed,
evaluates a preregistered tied tissue/HLA auxiliary-weight grid, and records
shared-encoder gradient cosine similarities once per epoch.  It never opens
the previously inspected fixed test split.

Every epoch, seed, and total duration is printed and written to a timing CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_occurrence_equal_ablation_mhc_only as ablation  # noqa: E402
import run_tissuepmhc_neural_baselines_v2 as base  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "human_auxiliary_validation"
DEFAULT_SEEDS = (20260721, 20260722, 20260723, 20260724, 20260725)
DEFAULT_GRID = (0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30)


def split_pair_grouped_within_task(
    frame: pd.DataFrame, fraction: float, split_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hold out complete positive/negative pairs within every task."""
    validation_indices: list[int] = []
    audit_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(split_seed)
    for task_name, task in frame.groupby("task_name", sort=True):
        pairs = task["pair_id"].drop_duplicates().to_numpy()
        if len(pairs) < 2:
            raise ValueError(f"{task_name}: cannot form train/validation pair split with {len(pairs)} pairs")
        n_validation = max(1, min(len(pairs) - 1, int(round(len(pairs) * fraction))))
        selected = set(rng.permutation(pairs)[:n_validation])
        chosen = task.index[task["pair_id"].isin(selected)]
        validation_indices.extend(chosen.tolist())
        audit_rows.append({
            "task_name": task_name,
            "total_pairs": len(pairs),
            "fit_pairs": len(pairs) - n_validation,
            "validation_pairs": n_validation,
            "fit_rows": len(task) - len(chosen),
            "validation_rows": len(chosen),
        })
    validation = frame.loc[validation_indices].copy()
    fitting = frame.drop(index=validation_indices).copy()
    if set(fitting["pair_id"].astype(str)) & set(validation["pair_id"].astype(str)):
        raise AssertionError("pair_id leakage between fitting and validation partitions")
    for name, subset in (("fitting", fitting), ("validation", validation)):
        labels = subset.groupby("pair_id", sort=False)["label"].agg(lambda item: set(map(int, item)))
        if not labels.map(lambda item: item == {0, 1}).all():
            raise AssertionError(f"{name}: every retained pair must contain one positive and one negative")
    return fitting.reset_index(drop=True), validation.reset_index(drop=True), pd.DataFrame(audit_rows)


def score_weight(
    prediction: pd.DataFrame,
    train: pd.DataFrame,
    seed: int,
    weight: float,
    worst_k: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    per_task, summary = ablation.evaluate_prediction(
        prediction, train, "human", f"aux_weight_{weight:g}", seed, worst_k
    )
    per_task.insert(1, "auxiliary_weight", weight)
    summary["auxiliary_weight"] = weight
    return per_task, summary


def summarize_weights(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = ["mean_task_auroc", "mean_task_auprc", "mean_task_pair_accuracy", "worst_k_mean_auroc"]
    rows: list[dict[str, object]] = []
    for weight, group in per_seed.groupby("auxiliary_weight", sort=True):
        row: dict[str, object] = {"auxiliary_weight": weight, "n_seeds": int(group["seed"].nunique())}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auxiliary_weight").reset_index(drop=True)


def paired_seed_deltas(per_seed: pd.DataFrame, reference_weight: float, candidate_weight: float) -> pd.DataFrame:
    reference = per_seed[per_seed["auxiliary_weight"] == reference_weight].set_index("seed")
    candidate = per_seed[per_seed["auxiliary_weight"] == candidate_weight].set_index("seed")
    if set(reference.index) != set(candidate.index):
        raise AssertionError("paired seed comparison has incomplete seed coverage")
    output = pd.DataFrame({"seed": sorted(reference.index)})
    for metric in ["mean_task_auroc", "mean_task_auprc", "mean_task_pair_accuracy", "worst_k_mean_auroc"]:
        output[f"{metric}_delta_weight_{candidate_weight:g}_minus_{reference_weight:g}"] = [
            float(candidate.loc[seed, metric] - reference.loc[seed, metric]) for seed in output["seed"]
        ]
    return output


def summarize_gradients(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    data = pd.read_csv(path)
    cosine_columns = ["primary_tissue_cosine", "primary_mhc_cosine", "tissue_mhc_cosine"]
    rows: list[dict[str, object]] = []
    for (weight, branch), group in data.groupby(["config", "branch"], sort=True):
        numeric_weight = float(str(weight).removeprefix("aux_weight_"))
        row: dict[str, object] = {
            "auxiliary_weight": numeric_weight,
            "branch": branch,
            "n_epoch_diagnostics": len(group),
        }
        for column in cosine_columns:
            values = group[column].to_numpy(dtype=float)
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_negative_fraction"] = float((values < 0).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["auxiliary_weight", "branch"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--weight-grid", nargs="+", type=float, default=list(DEFAULT_GRID))
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--validation-split-seed", type=int, default=20260808)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.seeds) < 5:
        raise ValueError("Use at least five seeds for the paired confirmation.")
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be in (0, 0.5).")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive.")
    weights = sorted(set(args.weight_grid))
    if weights != list(DEFAULT_GRID):
        raise ValueError(f"The preregistered grid is fixed at {list(DEFAULT_GRID)}.")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    torch_parts = base.require_torch()
    torch = torch_parts[0]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_path = PROJECT_ROOT / ablation.SPECIES_CONFIGS["human"].train
    source = base.read_dataset(train_path)
    source["target_tissue"] = source["target_tissue"].fillna("NA")
    # Task labels are needed to stratify the internal pair split.  The second
    # argument is a disposable copy solely to obtain the shared mapping.
    source, _, _ = base.add_task_columns(source, source.copy())
    split_started = time.perf_counter()
    fitting, validation, split_audit = split_pair_grouped_within_task(
        source, args.validation_fraction, args.validation_split_seed
    )
    fitting, validation, mappings = base.add_task_columns(fitting, validation)
    peptide_length = int(fitting["peptide_sequence"].str.len().iloc[0])
    split_audit.to_csv(args.output / "validation_split_audit.csv", index=False)
    contract = {
        "status": "running",
        "experiment": "human_auxiliary_weight_validation_only",
        "test_access": "forbidden: this program reads only the original training file",
        "source_train": str(train_path.relative_to(PROJECT_ROOT)),
        "source_train_sha256": ablation.sha256(train_path),
        "validation_split": "pair-grouped within task",
        "validation_fraction": args.validation_fraction,
        "validation_split_seed": args.validation_split_seed,
        "seeds": args.seeds,
        "preregistered_tied_tissue_and_mhc_weight_grid": weights,
        "selection_rule": "maximize five-seed mean validation task-macro AUROC; ties select the smaller weight",
        "fixed_test_policy": "Fit the selected weight on all training data and evaluate once only on a newly untouched confirmation split.",
        "epochs": args.epochs,
        "device": device,
        "human_config_template": asdict(ablation.SPECIES_CONFIGS["human"]),
    }
    (args.output / "run_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(
        f"[VALIDATION START] fit_rows={len(fitting)} validation_rows={len(validation)} "
        f"tasks={len(mappings['tasks'])} seeds={args.seeds} grid={weights} device={device}", flush=True
    )

    timing = ablation.TimingLogger(args.output / "timing_results.csv")
    gradient_logger = ablation.GradientLogger(args.output / "gradient_epoch_diagnostics.csv")
    per_task_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    all_started = time.perf_counter()
    base_config = ablation.SPECIES_CONFIGS["human"]
    try:
        for seed in args.seeds:
            seed_started = time.perf_counter()
            zero_config = replace(base_config, tissue_loss_weight=0.0, mhc_loss_weight=0.0)
            zero_predictions, _ = ablation.fit_two_branch(
                train=fitting, test=validation, mappings=mappings, peptide_length=peptide_length,
                config=zero_config, config_name="aux_weight_0", encoder_kind="cnn",
                use_auxiliary=False, seed=seed, epochs=args.epochs, device=device,
                torch_parts=torch_parts, timing=timing,
            )
            per_task, summary = score_weight(zero_predictions["rank"], fitting, seed, 0.0, base_config.worst_k)
            per_task_frames.append(per_task); summary_rows.append(summary)
            mhc_prediction = zero_predictions["mhc"]
            for weight in weights[1:]:
                weighted_config = replace(base_config, tissue_loss_weight=weight, mhc_loss_weight=weight)
                predictions, _ = ablation.fit_two_branch(
                    train=fitting, test=validation, mappings=mappings, peptide_length=peptide_length,
                    config=weighted_config, config_name=f"aux_weight_{weight:g}", encoder_kind="cnn",
                    use_auxiliary=True, seed=seed, epochs=args.epochs, device=device,
                    torch_parts=torch_parts, timing=timing, gradient_logger=gradient_logger,
                    precomputed_mhc_prediction=mhc_prediction,
                )
                per_task, summary = score_weight(predictions["rank"], fitting, seed, weight, base_config.worst_k)
                per_task_frames.append(per_task); summary_rows.append(summary)
            elapsed = time.perf_counter() - seed_started
            timing.write(scope="validation_seed", species="human", config="auxiliary_weight_grid", seed=seed,
                         epochs=args.epochs, elapsed_seconds=f"{elapsed:.6f}", status="completed")
            print(f"[VALIDATION SEED TIME] seed={seed} elapsed_seconds={elapsed:.3f}", flush=True)

        per_task_data = pd.concat(per_task_frames, ignore_index=True)
        per_seed = pd.DataFrame(summary_rows).sort_values(["auxiliary_weight", "seed"])
        weight_summary = summarize_weights(per_seed)
        chosen = weight_summary.sort_values(
            ["mean_task_auroc_mean", "auxiliary_weight"], ascending=[False, True]
        ).iloc[0]
        selected_weight = float(chosen["auxiliary_weight"])
        paired = paired_seed_deltas(per_seed, 0.0, 0.30)
        gradient_summary = summarize_gradients(args.output / "gradient_epoch_diagnostics.csv")
        per_task_data.to_csv(args.output / "validation_per_task_metrics.csv", index=False)
        per_seed.to_csv(args.output / "validation_per_seed_metrics.csv", index=False)
        weight_summary.to_csv(args.output / "validation_weight_summary.csv", index=False)
        paired.to_csv(args.output / "paired_seed_deltas_weight_0p30_minus_0.csv", index=False)
        gradient_summary.to_csv(args.output / "gradient_conflict_summary.csv", index=False)
        (args.output / "selected_weight.json").write_text(json.dumps({
            "selected_tissue_weight": selected_weight,
            "selected_mhc_weight": selected_weight,
            "selection_metric": "five-seed mean validation task-macro AUROC",
            "selection_value": float(chosen["mean_task_auroc_mean"]),
            "test_policy": contract["fixed_test_policy"],
        }, indent=2), encoding="utf-8")
        contract["status"] = "completed"
    finally:
        elapsed = time.perf_counter() - all_started
        contract["elapsed_seconds"] = elapsed
        (args.output / "run_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
        timing.write(scope="validation_total", species="human", config="auxiliary_weight_grid",
                     epochs=args.epochs, elapsed_seconds=f"{elapsed:.6f}", status=contract["status"])
        print(f"[VALIDATION TOTAL TIME] elapsed_seconds={elapsed:.3f} status={contract['status']}", flush=True)


if __name__ == "__main__":
    main()
