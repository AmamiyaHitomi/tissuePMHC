from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .common import (
        DEFAULT_RESULTS,
        SPECS,
        atomic_json,
        bh_adjust,
        load_strict_folds,
        make_standard_folds,
        paired_comparison,
        per_task_metrics,
        read_benchmark,
        sha256,
    )
except ImportError:
    from common import (
        DEFAULT_RESULTS,
        SPECS,
        atomic_json,
        bh_adjust,
        load_strict_folds,
        make_standard_folds,
        paired_comparison,
        per_task_metrics,
        read_benchmark,
        sha256,
    )


def fit_models(frame: pd.DataFrame) -> tuple[Any, Any]:
    usable = frame[["external_score", "main_score", "label"]].dropna()
    if usable["label"].nunique() != 2:
        raise ValueError("Stacker fitting data must contain both labels.")
    external = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=20260724),
    )
    combined = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=20260724),
    )
    external.fit(usable[["external_score"]], usable["label"].astype(int))
    combined.fit(
        usable[["external_score", "main_score"]], usable["label"].astype(int)
    )
    return external, combined


def predict_models(
    frame: pd.DataFrame, external: Any, combined: Any
) -> pd.DataFrame:
    result = frame.copy()
    result["external_only_score"] = np.nan
    result["external_plus_tissue_score"] = np.nan
    usable = result[["external_score", "main_score"]].notna().all(axis=1)
    result.loc[usable, "external_only_score"] = external.predict_proba(
        result.loc[usable, ["external_score"]]
    )[:, 1]
    result.loc[usable, "external_plus_tissue_score"] = combined.predict_proba(
        result.loc[usable, ["external_score", "main_score"]]
    )[:, 1]
    return result


def cross_fit(
    frame: pd.DataFrame, fold_by_sample: pd.Series
) -> pd.DataFrame:
    work = frame.copy()
    work["stack_fold"] = work["sample_id"].map(fold_by_sample)
    if work["stack_fold"].isna().any():
        raise ValueError("Stack fold mapping does not cover all rows.")
    predictions = []
    for fold in sorted(work["stack_fold"].astype(int).unique()):
        fitting = work[work["stack_fold"] != fold]
        held = work[work["stack_fold"] == fold]
        external, combined = fit_models(fitting)
        predictions.append(predict_models(held, external, combined))
    result = pd.concat(predictions, ignore_index=True)
    if len(result) != len(frame) or result["sample_id"].duplicated().any():
        raise AssertionError("Cross-fitted stack predictions are incomplete.")
    return result


def stack(
    row_predictions_path: Path,
    output_dir: Path,
    bootstrap_iterations: int,
) -> None:
    rows = pd.read_csv(row_predictions_path)
    required = {
        "species",
        "protocol",
        "model",
        "sample_id",
        "pair_id",
        "target_tissue",
        "mhc_restriction",
        "peptide_sequence",
        "label",
        "task_name",
        "external_score",
        "main_score",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"External row predictions miss columns: {missing}")

    prediction_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    paired_rows: list[dict[str, Any]] = []
    delta_frames: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, Any]] = []
    for (species, model), model_rows in rows.groupby(["species", "model"], sort=True):
        spec = SPECS[species]
        train = read_benchmark(spec.train, species, "train")
        standard_folds = make_standard_folds(train)
        strict_folds = load_strict_folds(train, spec.strict_manifest)
        standard_map = pd.Series(
            standard_folds.to_numpy(int), index=train["sample_id"].astype(str)
        )
        strict_map = pd.Series(
            strict_folds.to_numpy(int), index=train["sample_id"].astype(str)
        )

        standard = model_rows[
            model_rows["protocol"] == "matched_standard_oof"
        ].copy()
        strict = model_rows[
            model_rows["protocol"] == "peptide_disjoint_oof"
        ].copy()
        fixed = model_rows[model_rows["protocol"] == "standard_fixed_test"].copy()
        if not len(standard) or not len(strict) or not len(fixed):
            raise ValueError(f"{species}/{model} lacks one of the three protocols.")

        stacked_standard = cross_fit(standard, standard_map)
        stacked_strict = cross_fit(strict, strict_map)
        external_fit, combined_fit = fit_models(standard)
        stacked_fixed = predict_models(fixed, external_fit, combined_fit)
        stacked_fixed["stack_fold"] = -1

        # Coefficients are reported for the fixed-test stacker trained on standard OOF.
        external_lr = external_fit.named_steps["logisticregression"]
        combined_lr = combined_fit.named_steps["logisticregression"]
        coefficient_rows.extend(
            [
                {
                    "species": species,
                    "external_model": model,
                    "stacker": "external_only",
                    "feature": "external_score",
                    "standardized_coefficient": float(external_lr.coef_[0, 0]),
                    "intercept": float(external_lr.intercept_[0]),
                },
                {
                    "species": species,
                    "external_model": model,
                    "stacker": "external_plus_tissue",
                    "feature": "external_score",
                    "standardized_coefficient": float(combined_lr.coef_[0, 0]),
                    "intercept": float(combined_lr.intercept_[0]),
                },
                {
                    "species": species,
                    "external_model": model,
                    "stacker": "external_plus_tissue",
                    "feature": "main_score",
                    "standardized_coefficient": float(combined_lr.coef_[0, 1]),
                    "intercept": float(combined_lr.intercept_[0]),
                },
            ]
        )

        for protocol_frame in (stacked_standard, stacked_strict, stacked_fixed):
            protocol = str(protocol_frame["protocol"].iloc[0])
            protocol_frame["external_model"] = model
            prediction_frames.append(protocol_frame)
            external_metrics, _ = per_task_metrics(
                protocol_frame, "external_only_score", "external_only_stacker"
            )
            combined_metrics, _ = per_task_metrics(
                protocol_frame,
                "external_plus_tissue_score",
                "external_plus_tissue_stacker",
            )
            for metric_frame, stacker in (
                (external_metrics, "external_only_stacker"),
                (combined_metrics, "external_plus_tissue_stacker"),
            ):
                metric_frame.insert(0, "species", species)
                metric_frame.insert(1, "protocol", protocol)
                metric_frame.insert(2, "external_model", model)
                metric_frame["stacker"] = stacker
                metric_frames.append(metric_frame)
            for metric in ("auroc", "auprc", "pair_acc"):
                summary, deltas = paired_comparison(
                    external_metrics,
                    combined_metrics,
                    metric,
                    bootstrap_iterations=bootstrap_iterations,
                )
                summary.update(
                    {
                        "species": species,
                        "protocol": protocol,
                        "external_model": model,
                        "comparison": "external_plus_tissue_minus_external_only",
                    }
                )
                paired_rows.append(summary)
                deltas.insert(0, "species", species)
                deltas.insert(1, "protocol", protocol)
                deltas.insert(2, "external_model", model)
                deltas.insert(3, "metric", metric)
                delta_frames.append(deltas)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    paired = pd.DataFrame(paired_rows)
    deltas = pd.concat(delta_frames, ignore_index=True)
    coefficients = pd.DataFrame(coefficient_rows)
    paired["bh_family"] = (
        paired["species"] + "||" + paired["protocol"] + "||" + paired["metric"]
    )
    paired["wilcoxon_qvalue"] = paired.groupby("bh_family", group_keys=False)[
        "wilcoxon_pvalue"
    ].apply(bh_adjust)
    summary = (
        metrics.groupby(
            ["species", "protocol", "external_model", "stacker"], as_index=False
        )
        .agg(
            n_tasks=("task_name", "nunique"),
            mean_task_auroc=("auroc", "mean"),
            mean_task_auprc=("auprc", "mean"),
            mean_task_pair_acc=("pair_acc", "mean"),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "row_predictions.csv.gz", index=False)
    metrics.to_csv(output_dir / "per_task_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    paired.to_csv(output_dir / "paired_statistics.csv", index=False)
    deltas.to_csv(output_dir / "per_task_differences.csv", index=False)
    coefficients.to_csv(output_dir / "fixed_test_stacker_coefficients.csv", index=False)
    atomic_json(
        output_dir / "metadata.json",
        {
            "purpose": "Cross-fitted incremental signal after controlling frozen general pMHC score",
            "input": str(row_predictions_path.resolve()),
            "input_sha256": sha256(row_predictions_path),
            "stacker": (
                "StandardScaler + L2 logistic regression (C=1); no tissue/task feature "
                "or task intercept"
            ),
            "oof_policy": "Fit on two outer folds and predict the held outer fold.",
            "fixed_test_policy": (
                "Fit stackers on matched-standard OOF rows only, then predict fixed test."
            ),
            "bootstrap_iterations": bootstrap_iterations,
            "difference_direction": "external_plus_tissue_minus_external_only",
            "analysis_role": "Secondary incremental analysis; not a replacement for raw-score baselines.",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-predictions", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_RESULTS / "stack_increment"
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stack(args.row_predictions, args.output_dir, args.bootstrap_iterations)
