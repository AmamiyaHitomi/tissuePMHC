#!/usr/bin/env python3
"""Run the human Phase 1 E0 traditional baselines on mousePMHC train-only OOF.

The original E0 runner reports a fixed-test evaluation.  This mouse-specific
runner instead uses the same feature encodings and estimators inside the same
pair-grouped OOF protocol as mouse E14a, enabling a fair Phase 3 comparison
without opening the fixed mouse test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_tissuepmhc_baselines as human_e0
import run_tissuepmhc_e26_all_in_one as folds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mousePMHC_phase3_e0_oof"
KEYS = ["sample_id", "target_tissue", "mhc_restriction"]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def validate_input(frame: pd.DataFrame) -> None:
    required = {*KEYS, "dataset", "split", "pair_id", "label", "peptide_sequence"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if set(frame.dataset) != {"mousePMHC"} or set(frame.split) != {"train"}:
        raise ValueError("E0 Phase 3 accepts mousePMHC train rows only.")
    if not frame.mhc_restriction.str.startswith("H2-").all():
        raise ValueError("E0 Phase 3 found a non-H2 MHC restriction.")
    if not frame.peptide_sequence.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{9}").all():
        raise ValueError("E0 Phase 3 requires standard 9-mer peptides.")


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for candidate, candidate_rows in predictions.groupby("candidate", sort=True):
        for (tissue, mhc), task in candidate_rows.groupby(["target_tissue", "mhc_restriction"], sort=True):
            metrics = human_e0.evaluate(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float))
            rows.append({"experiment_name": EXPERIMENT, "candidate": candidate,
                "target_tissue": tissue, "mhc_restriction": mhc, "oof_rows": len(task), **metrics})
    per_task = pd.DataFrame(rows)
    summary = per_task.groupby("candidate", sort=True)[["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]].mean().reset_index()
    summary.insert(0, "experiment_name", EXPERIMENT)
    return per_task, summary.rename(columns={column: f"mean_task_{column}" for column in summary if column not in {"experiment_name", "candidate"}})


def run(args: argparse.Namespace) -> None:
    train = pd.read_csv(args.train, keep_default_na=False)
    validate_input(train)
    train = train.copy()
    train["task_name"] = train.target_tissue + "||" + train.mhc_restriction
    available_tasks = sorted(train.task_name.unique())
    if args.max_tasks:
        available_tasks = available_tasks[:args.max_tasks]
        train = train[train.task_name.isin(available_tasks)].copy()
    assignments = folds.make_pair_grouped_folds(train, args.oof_folds, args.oof_split_seed)
    model_names = list(human_e0.get_models(args.seed))
    parts: list[pd.DataFrame] = []
    for fold in range(args.oof_folds):
        fitting, held_out = train[assignments != fold], train[assignments == fold]
        print(f"mousePMHC E0 OOF fold={fold + 1}/{args.oof_folds} fit={len(fitting)} holdout={len(held_out)}", flush=True)
        for model_name in model_names:
            encoder, estimator = human_e0.get_models(args.seed)[model_name]
            scores = np.empty(len(held_out), dtype=float)
            for _, task in held_out.groupby("task_name", sort=True):
                task_name = task.task_name.iloc[0]
                fit_task = fitting[fitting.task_name == task_name]
                x_fit = encoder(fit_task.peptide_sequence.tolist())
                y_fit = fit_task.label.to_numpy(dtype=int)
                x_holdout = encoder(task.peptide_sequence.tolist())
                estimator.fit(x_fit, y_fit)
                scores[held_out.index.get_indexer(task.index)] = human_e0.predict_scores(estimator, x_holdout)
            output = held_out[KEYS + ["label"]].copy()
            output.insert(0, "split", "oof")
            output.insert(1, "candidate", f"mousePMHC_phase3_e0_{model_name}")
            output.insert(2, "seed", args.seed)
            output["score"] = scores
            parts.append(output[["split", "candidate", "seed", *KEYS, "label", "score"]])
    predictions = pd.concat(parts, ignore_index=True)
    expected_ids = set(train.sample_id)
    for candidate, rows in predictions.groupby("candidate", sort=True):
        if len(rows) != len(train) or set(rows.sample_id) != expected_ids or rows.sample_id.duplicated().any():
            raise AssertionError(f"Incomplete OOF coverage for {candidate}")
    per_task, summary = metric_tables(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "mousePMHC_phase3_e0_oof_predictions.csv", index=False)
    per_task.to_csv(args.output_dir / "mousePMHC_phase3_e0_oof_per_task_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "mousePMHC_phase3_e0_oof_summary_metrics.csv", index=False)
    metadata = {
        "experiment_name": EXPERIMENT, "species": "Mus musculus", "mhc_system": "H2-I",
        "human_method_source": "tissuePMHC Phase 1 E0 traditional baselines",
        "test_data_read": False, "train": str(args.train), "n_rows": len(train),
        "n_pairs": int(train.pair_id.nunique()), "n_tasks": len(available_tasks),
        "seed": args.seed, "oof_folds": args.oof_folds, "oof_split_seed": args.oof_split_seed,
        "models": model_names,
        "naming_policy": "mousePMHC_phase3_<experiment>; human tissuePMHC outputs are never overwritten",
    }
    (args.output_dir / "mousePMHC_phase3_e0_oof_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=project_path("results/mousePMHC_phase3_e0_oof"))
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--oof-folds", type=int, default=3)
    parser.add_argument("--oof-split-seed", type=int, default=20260711)
    parser.add_argument("--max-tasks", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
