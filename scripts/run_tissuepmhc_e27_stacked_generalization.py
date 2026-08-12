#!/usr/bin/env python3
"""Run E27: leakage-safe OOF stacked generalization for tissuePMHC.

The script consumes the long-format, pair-grouped OOF and separately generated
test prediction files from E26.  It first reads and fits only the OOF split,
using each candidate's within-task percentile rank as a feature.  Only after
the stacker and its coefficients are fixed does it read test predictions.

The default meta-model is intentionally simple: one globally shared logistic
regression with a pre-specified L2 strength (``--l2-c 0.1``).  The test split
must never be used to select candidates, regularization, or model form.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import run_tissuepmhc_auxiliary_soft_ensemble as e14
import run_tissuepmhc_e26_greedy_ensemble_selection as e26
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PER_TASK_COLUMNS = [
    "experiment_name", "seed", "model", "selection_source", "n_candidates", "candidates",
    "target_tissue", "mhc_restriction", "test_rows", "test_positive", "test_negative",
    *e14.METRICS, "fusion_formula",
]
COEFFICIENT_COLUMNS = ["candidate", "coefficient", "intercept", "l2_c"]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def allowed_candidates(candidates: list[str], requested: list[str] | None) -> list[str]:
    """Choose a deterministic, non-SWA candidate list without using labels."""
    allowed = sorted(requested) if requested else [name for name in candidates if "swa" not in name.lower()]
    if len(set(allowed)) != len(allowed):
        raise ValueError("--candidates must not contain duplicates.")
    if not allowed:
        raise ValueError("No non-SWA candidates remain; pass an explicit --candidates allow-list.")
    absent = sorted(set(allowed) - set(candidates))
    if absent:
        raise ValueError(f"Requested candidate(s) absent from OOF predictions: {absent}")
    return allowed


def feature_matrix(candidates: list[str], all_candidates: list[str], matrix: np.ndarray) -> np.ndarray:
    """Take selected task-rank columns in the exact metadata/CSV order."""
    positions = [all_candidates.index(candidate) for candidate in candidates]
    return matrix[:, positions]


def make_rows(labels: pd.DataFrame, probabilities: np.ndarray, candidates: list[str]) -> list[dict[str, object]]:
    output = labels.copy()
    output["probability"] = probabilities
    rows: list[dict[str, object]] = []
    for (seed, tissue, hla), task in output.groupby(e26.TASK_COLUMNS, sort=True):
        y_true = task["label"].to_numpy(dtype=int)
        rows.append({
            "experiment_name": "E27_stacked_generalization",
            "seed": int(seed),
            "model": "e27_oof_rank_logistic_stacker",
            "selection_source": "OOF_only",
            "n_candidates": len(candidates),
            "candidates": ",".join(candidates),
            "target_tissue": tissue,
            "mhc_restriction": hla,
            "test_rows": len(task),
            "test_positive": int(y_true.sum()),
            "test_negative": int(len(task) - y_true.sum()),
            **base.evaluate(y_true, task["probability"].to_numpy(dtype=float)),
            "fusion_formula": "sigmoid(intercept + sum(coefficient * task_percentile_rank(candidate_score)))",
        })
    return rows


def summarize_oof(labels: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    """Diagnostic training-fit metrics; not used to choose any configuration."""
    return e26.metric_summary(labels, probabilities)


def run(args: argparse.Namespace) -> None:
    if args.l2_c <= 0:
        raise ValueError("--l2-c must be positive.")
    if args.max_iter < 1:
        raise ValueError("--max-iter must be positive.")

    # All candidate decisions and meta-model fitting happen before test is read.
    oof = e26.read_predictions(args.oof_predictions, "oof")
    oof_labels, oof_all_candidates, oof_matrix = e26.aligned_matrix(oof)
    candidates = allowed_candidates(oof_all_candidates, args.candidates)
    x_oof = feature_matrix(candidates, oof_all_candidates, oof_matrix)
    y_oof = oof_labels["label"].to_numpy(dtype=int)
    stacker = LogisticRegression(
        C=args.l2_c, solver="lbfgs", max_iter=args.max_iter,
    )
    stacker.fit(x_oof, y_oof)
    oof_probability = stacker.predict_proba(x_oof)[:, 1]
    oof_metrics = summarize_oof(oof_labels, oof_probability)
    print(f"OOF stacker mean task AUROC (diagnostic only): {oof_metrics['mean_auroc']:.6f}", flush=True)
    print("OOF stacker is fixed; loading test predictions for one evaluation.", flush=True)

    test = e26.read_predictions(args.test_predictions, "test")
    test_labels, test_all_candidates, test_matrix = e26.aligned_matrix(test)
    absent_test = sorted(set(candidates) - set(test_all_candidates))
    if absent_test:
        raise ValueError(f"Test predictions are missing OOF-fitted candidate(s): {absent_test}")
    x_test = feature_matrix(candidates, test_all_candidates, test_matrix)
    test_probability = stacker.predict_proba(x_test)[:, 1]
    rows = make_rows(test_labels, test_probability, candidates)
    summary = base.summarize_results(rows)
    stability = base.summarize_seed_stability(summary)

    base.write_csv(args.per_task_output, PER_TASK_COLUMNS, rows)
    base.write_csv(args.summary_output, base.SUMMARY_COLUMNS, summary)
    base.write_csv(args.stability_output, base.STABILITY_COLUMNS, stability)
    base.write_csv(args.coefficients_output, COEFFICIENT_COLUMNS, [
        {
            "candidate": candidate,
            "coefficient": float(coefficient),
            "intercept": float(stacker.intercept_[0]),
            "l2_c": args.l2_c,
        }
        for candidate, coefficient in zip(candidates, stacker.coef_[0], strict=True)
    ])
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps({
        "experiment_name": "E27_stacked_generalization",
        "oof_predictions": str(args.oof_predictions),
        "test_predictions": str(args.test_predictions),
        "candidate_pool": candidates,
        "meta_model": "sklearn.linear_model.LogisticRegression(solver=lbfgs, penalty=l2)",
        "features": "within-task percentile ranks of candidate scores",
        "l2_c": args.l2_c,
        "max_iter": args.max_iter,
        "intercept": float(stacker.intercept_[0]),
        "coefficients": {candidate: float(value) for candidate, value in zip(candidates, stacker.coef_[0], strict=True)},
        "oof_training_fit_metrics_diagnostic_only": oof_metrics,
        "selection_policy": "Candidate allow-list and L2 C are fixed before test predictions are read; test is evaluated once.",
        "rng_policy": "Deterministic lbfgs solver; no random state is supplied or advanced.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    for path in [args.per_task_output, args.summary_output, args.stability_output, args.coefficients_output, args.metadata_output]:
        print(f"wrote: {path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = project_path("results/tissuePMHC_e27_stacked_generalization")
    e26_root = project_path("results/tissuePMHC_e26_greedy_ensemble_selection")
    parser.add_argument("--oof-predictions", type=Path, default=e26_root / "oof_predictions.csv")
    parser.add_argument("--test-predictions", type=Path, default=e26_root / "test_predictions.csv")
    parser.add_argument("--candidates", nargs="+", default=None, help="Optional pre-specified candidate allow-list; SWA is excluded by default.")
    parser.add_argument("--l2-c", type=float, default=0.1, help="Pre-specified inverse L2 strength; do not tune after test evaluation.")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--per-task-output", type=Path, default=root / "per_task_metrics.csv")
    parser.add_argument("--summary-output", type=Path, default=root / "summary_metrics.csv")
    parser.add_argument("--stability-output", type=Path, default=root / "stability_metrics.csv")
    parser.add_argument("--coefficients-output", type=Path, default=root / "coefficients.csv")
    parser.add_argument("--metadata-output", type=Path, default=root / "metadata.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
