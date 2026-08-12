#!/usr/bin/env python3
"""One-command preregistered E29 extension from three to five seeds.

Only seeds 20260707 and 20260708 are trained.  Existing E29 seeds 20260704-06
are reused.  New OOF predictions are generated first; test data is read and
full-train test models are created only when all preregistered OOF gates pass.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

import run_tissuepmhc_e26_greedy_ensemble_selection as e26
import run_tissuepmhc_e29_multikernel_cnn_oof as e29


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_SEEDS = [20260704, 20260705, 20260706]
NEW_SEEDS = [20260707, 20260708]
ALL_SEEDS = OLD_SEEDS + NEW_SEEDS
MIN_MEAN_AUROC_GAIN = 0.0010
MIN_WORST10_AUROC_GAIN = -0.0010
MIN_MEAN_AUPRC_GAIN = -0.0005


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def fixed_e29_args(args: argparse.Namespace, seeds: list[int]) -> SimpleNamespace:
    """Build the exact frozen E29 training configuration."""
    return SimpleNamespace(
        train=args.train, test=args.test, seeds=seeds, oof_folds=3, oof_split_seed=20260711,
        device=args.device, epochs=25, batch_size=512, learning_rate=1e-3, weight_decay=1e-4,
        embedding_dim=16, kernel_sizes=[2, 3, 5], conv_channels=32, hidden_dim=128,
        dropout=0.2, tissue_loss_weight=0.1, hla_loss_weight=0.1, max_grad_norm=1.0,
        max_tasks=0,
    )


def require_single_seed_candidates(frame: pd.DataFrame, seeds: list[int], source: Path) -> pd.DataFrame:
    names = [f"e29_cnn_seed_{seed}" for seed in seeds]
    available = set(frame["candidate"].unique())
    missing = sorted(set(names) - available)
    if missing:
        raise ValueError(f"{source} is missing required E29 seed candidate(s): {missing}")
    result = frame[frame["candidate"].isin(names)].copy()
    counts = result.groupby("candidate").size()
    if counts.nunique() != 1:
        raise ValueError(f"Seed candidates in {source} do not cover equal row counts.")
    return result


def metric_for_candidate(frame: pd.DataFrame, candidate: str) -> dict[str, float]:
    labels, scores = e29.candidate_scores_from_long(frame, candidate)
    return e26.metric_summary(labels, scores)


def aligned_concat(old: pd.DataFrame, new: pd.DataFrame, split: str) -> pd.DataFrame:
    combined = pd.concat([old, new], ignore_index=True)
    duplicated = combined.duplicated(["candidate", "seed", *e29.KEYS], keep=False)
    if duplicated.any():
        raise ValueError(f"Duplicate {split} candidate/sample keys after combining old and new seeds.")
    expected = len(ALL_SEEDS)
    coverage = combined.groupby(["sample_id", "target_tissue", "mhc_restriction"]).size()
    if not (coverage == expected).all():
        raise ValueError(f"Combined {split} predictions do not contain all five seed candidates per sample.")
    return e29.append_seed_mean(combined, split, ALL_SEEDS)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def preregistration_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment_name": "E29_incremental_5seed",
        "preregistered_before_new_seed_training": True,
        "old_seeds": OLD_SEEDS,
        "new_seeds": NEW_SEEDS,
        "all_seeds": ALL_SEEDS,
        "training_policy": "Train only seeds 20260707 and 20260708; reuse fixed seed 20260704-06 predictions.",
        "oof_policy": "3-fold pair-grouped OOF; test is not read unless all OOF gates pass.",
        "gates": {
            "minimum_mean_auroc_gain_5seed_minus_3seed": MIN_MEAN_AUROC_GAIN,
            "minimum_worst10_auroc_gain_5seed_minus_3seed": MIN_WORST10_AUROC_GAIN,
            "minimum_mean_auprc_gain_5seed_minus_3seed": MIN_MEAN_AUPRC_GAIN,
        },
        "test_policy": "One fixed 5-seed evaluation; no member removal, weighting, or further standard-split tuning.",
        "human_readable_preregistration": str(args.preregistration_document),
    }


def evaluate_gate(old_oof: pd.DataFrame, combined_oof: pd.DataFrame) -> dict[str, Any]:
    metrics_3 = metric_for_candidate(old_oof, "e29_cnn_3seed_mean")
    metrics_5 = metric_for_candidate(combined_oof, "e29_cnn_5seed_mean")
    deltas = {
        "mean_auroc": metrics_5["mean_auroc"] - metrics_3["mean_auroc"],
        "worst_10_mean_auroc": metrics_5["worst_10_mean_auroc"] - metrics_3["worst_10_mean_auroc"],
        "mean_auprc": metrics_5["mean_auprc"] - metrics_3["mean_auprc"],
    }
    checks = {
        "mean_auroc_gain": deltas["mean_auroc"] >= MIN_MEAN_AUROC_GAIN,
        "worst10_auroc_noninferiority": deltas["worst_10_mean_auroc"] >= MIN_WORST10_AUROC_GAIN,
        "mean_auprc_noninferiority": deltas["mean_auprc"] >= MIN_MEAN_AUPRC_GAIN,
    }
    return {
        "e29_3seed_oof": metrics_3,
        "e29_5seed_oof": metrics_5,
        "deltas_5seed_minus_3seed": deltas,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def formal_evaluation_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        test_predictions_output=args.combined_test_output,
        per_task_output=args.per_task_output,
        summary_output=args.summary_output,
        stability_output=args.stability_output,
        e17_per_task=args.e17_per_task,
        e17_comparison_output=args.e17_comparison_output,
        seeds=ALL_SEEDS,
    )


def test_comparison(args: argparse.Namespace) -> dict[str, Any]:
    summary_5 = pd.read_csv(args.summary_output)
    row_5 = summary_5[summary_5["model"] == "e29_cnn_5seed_mean"]
    if len(row_5) != 1:
        raise ValueError("Formal E29 5-seed summary does not contain exactly one ensemble row.")
    summary_3 = pd.read_csv(args.existing_3seed_summary)
    row_3 = summary_3[summary_3["model"] == "e29_cnn_3seed_mean"]
    if len(row_3) != 1:
        raise ValueError("Existing E29 3-seed summary does not contain exactly one ensemble row.")
    metrics = ["mean_auroc", "mean_auprc", "mean_accuracy", "mean_mcc", "worst_10_mean_auroc"]
    values_3 = {metric: float(row_3.iloc[0][metric]) for metric in metrics}
    values_5 = {metric: float(row_5.iloc[0][metric]) for metric in metrics}
    return {
        "e29_3seed_test": values_3,
        "e29_5seed_test": values_5,
        "deltas_5seed_minus_3seed": {metric: values_5[metric] - values_3[metric] for metric in metrics},
        "selection_note": "Test comparison is reporting only and did not affect OOF gating or membership.",
    }


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    if not args.preregistration_document.is_file():
        raise FileNotFoundError(f"Missing preregistration document: {args.preregistration_document}")
    preregistration = preregistration_payload(args)
    write_json(args.preregistration_output, preregistration)
    print(f"wrote preregistration: {args.preregistration_output}", flush=True)

    # OOF stage: no test file is opened before the gate is finalized.
    old_oof = e26.read_predictions(args.existing_3seed_oof, "oof")
    old_oof_singles = require_single_seed_candidates(old_oof, OLD_SEEDS, args.existing_3seed_oof)
    new_args = fixed_e29_args(args, NEW_SEEDS)
    new_oof, details = e29.generate_oof(new_args)
    new_oof_singles = require_single_seed_candidates(new_oof, NEW_SEEDS, args.new_seed_oof_output)
    args.new_seed_oof_output.parent.mkdir(parents=True, exist_ok=True)
    new_oof.to_csv(args.new_seed_oof_output, index=False)
    combined_oof = aligned_concat(old_oof_singles, new_oof_singles, "oof")
    combined_oof.to_csv(args.combined_oof_output, index=False)
    gate = evaluate_gate(old_oof, combined_oof)
    write_json(args.gate_output, {
        **preregistration, "status": "oof_complete", "gate": gate, "runtime_details": details,
    })
    print(json.dumps(gate, indent=2, ensure_ascii=False), flush=True)
    if not gate["passed"]:
        print("E29 5-seed OOF gate failed; stopping before reading or generating test predictions.", flush=True)
        return

    # Test stage begins only after the gate passed.
    if not args.existing_3seed_test.is_file():
        raise FileNotFoundError(f"Missing fixed E29 3-seed test predictions: {args.existing_3seed_test}")
    old_test = e26.read_predictions(args.existing_3seed_test, "test")
    old_test_singles = require_single_seed_candidates(old_test, OLD_SEEDS, args.existing_3seed_test)
    new_test = e29.generate_test(new_args)
    new_test_singles = require_single_seed_candidates(new_test, NEW_SEEDS, args.new_seed_test_output)
    args.new_seed_test_output.parent.mkdir(parents=True, exist_ok=True)
    new_test.to_csv(args.new_seed_test_output, index=False)
    combined_test = aligned_concat(old_test_singles, new_test_singles, "test")
    combined_test.to_csv(args.combined_test_output, index=False)
    e29.evaluate_existing_test_predictions(formal_evaluation_args(args))
    comparison = test_comparison(args)
    write_json(args.metadata_output, {
        **preregistration, "status": "complete", "gate": gate, "test_comparison": comparison,
        "outputs": {
            "combined_oof": str(args.combined_oof_output), "combined_test": str(args.combined_test_output),
            "summary": str(args.summary_output), "per_task": str(args.per_task_output),
        },
    })
    print(json.dumps(comparison, indent=2, ensure_ascii=False), flush=True)
    print(f"run total time: {e29.format_duration(time.perf_counter() - started)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = project_path("results/tissuePMHC_e29_multikernel_cnn_5seed")
    old_root = project_path("results/tissuePMHC_e29_multikernel_cnn_3seed")
    parser.add_argument("--train", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz"))
    parser.add_argument("--test", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_test.csv.gz"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--preregistration-document", type=Path, default=project_path("E29_5SEED_PREREGISTRATION_zh.md"))
    parser.add_argument("--preregistration-output", type=Path, default=root / "preregistration.json")
    parser.add_argument("--existing-3seed-oof", type=Path, default=old_root / "oof_predictions.csv")
    parser.add_argument("--existing-3seed-test", type=Path, default=old_root / "test_predictions.csv")
    parser.add_argument("--existing-3seed-summary", type=Path, default=old_root / "summary_metrics.csv")
    parser.add_argument("--new-seed-oof-output", type=Path, default=root / "new_seed_oof_predictions.csv")
    parser.add_argument("--combined-oof-output", type=Path, default=root / "combined_5seed_oof_predictions.csv")
    parser.add_argument("--gate-output", type=Path, default=root / "oof_gate.json")
    parser.add_argument("--new-seed-test-output", type=Path, default=root / "new_seed_test_predictions.csv")
    parser.add_argument("--combined-test-output", type=Path, default=root / "combined_5seed_test_predictions.csv")
    parser.add_argument("--per-task-output", type=Path, default=root / "per_task_metrics.csv")
    parser.add_argument("--summary-output", type=Path, default=root / "summary_metrics.csv")
    parser.add_argument("--stability-output", type=Path, default=root / "stability_metrics.csv")
    parser.add_argument("--e17-per-task", type=Path, default=project_path("results/tissuePMHC_e17_seed_ensemble/per_task_metrics.csv"))
    parser.add_argument("--e17-comparison-output", type=Path, default=root / "e17_5seed_comparison_metrics.csv")
    parser.add_argument("--metadata-output", type=Path, default=root / "metadata.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
