#!/usr/bin/env python3
"""Run simple per tissue-HLA binary classifiers for the tissuePMHC dataset.

Roadmap role: E0 traditional single-task baseline.
Each tissue-HLA task is trained independently; this is the reference point
before neural multi-task learning.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: index for index, aa in enumerate(AMINO_ACIDS)}

BLOSUM62 = {
    "A": [4, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -2, -1, -1, -1, 1, 0, 0, -3, -2],
    "C": [0, 9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],
    "D": [-2, -3, 6, 2, -3, -1, -1, -3, -1, -4, -3, 1, -1, 0, -2, 0, -1, -3, -4, -3],
    "E": [-1, -4, 2, 5, -3, -2, 0, -3, 1, -3, -2, 0, -1, 2, 0, 0, -1, -2, -3, -2],
    "F": [-2, -2, -3, -3, 6, -3, -1, 0, -3, 0, 0, -3, -4, -3, -3, -2, -2, -1, 1, 3],
    "G": [0, -3, -1, -2, -3, 6, -2, -4, -2, -4, -3, -1, -2, -2, -2, 0, -2, -3, -2, -3],
    "H": [-2, -3, -1, 0, -1, -2, 8, -3, -1, -3, -2, 1, -2, 0, 0, -1, -2, -3, -2, 2],
    "I": [-1, -1, -3, -3, 0, -4, -3, 4, -3, 2, 1, -3, -3, -3, -3, -2, -1, 3, -3, -1],
    "K": [-1, -3, -1, 1, -3, -2, -1, -3, 5, -2, -1, 0, -1, 1, 2, 0, -1, -2, -3, -2],
    "L": [-1, -1, -4, -3, 0, -4, -3, 2, -2, 4, 2, -3, -3, -2, -2, -2, -1, 1, -2, -1],
    "M": [-1, -1, -3, -2, 0, -3, -2, 1, -1, 2, 5, -2, -2, 0, -1, -1, -1, 1, -1, -1],
    "N": [-2, -3, 1, 0, -3, -1, 1, -3, 0, -3, -2, 6, -2, 0, 0, 1, 0, -3, -4, -2],
    "P": [-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2, 7, -1, -2, -1, -1, -2, -4, -3],
    "Q": [-1, -3, 0, 2, -3, -2, 0, -3, 1, -2, 0, 0, -1, 5, 1, 0, -1, -2, -2, -1],
    "R": [-1, -3, -2, 0, -3, -2, 0, -3, 2, -2, -1, 0, -2, 1, 5, -1, -1, -3, -3, -2],
    "S": [1, -1, 0, 0, -2, 0, -1, -2, 0, -2, -1, 1, -1, 0, -1, 4, 1, -2, -3, -2],
    "T": [0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1, 0, -1, -1, -1, 1, 5, 0, -2, -2],
    "V": [0, -1, -3, -2, -1, -3, -3, 3, -2, 1, 1, -3, -2, -2, -3, -2, 0, 4, -3, -1],
    "W": [-3, -2, -4, -3, 1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11, 2],
    "Y": [-2, -2, -3, -2, 3, -3, 2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1, 2, 7],
}

METRIC_COLUMNS = [
    "model",
    "target_tissue",
    "mhc_restriction",
    "train_rows",
    "test_rows",
    "train_positive",
    "train_negative",
    "test_positive",
    "test_negative",
    "accuracy",
    "balanced_accuracy",
    "auroc",
    "auprc",
    "f1",
    "mcc",
]

SUMMARY_COLUMNS = [
    "model",
    "n_tasks",
    "mean_accuracy",
    "median_accuracy",
    "mean_balanced_accuracy",
    "median_balanced_accuracy",
    "mean_auroc",
    "median_auroc",
    "mean_auprc",
    "median_auprc",
    "mean_f1",
    "median_f1",
    "mean_mcc",
    "median_mcc",
    "weighted_mean_accuracy",
    "weighted_mean_auroc",
    "weighted_mean_auprc",
]


def open_text_input(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def read_dataset(path: Path) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with open_text_input(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped[(row["target_tissue"], row["mhc_restriction"])].append(row)
    return grouped


def encode_blosum62(peptides: list[str]) -> np.ndarray:
    return np.asarray([[score for aa in peptide for score in BLOSUM62[aa]] for peptide in peptides], dtype=np.float32)


def encode_onehot(peptides: list[str]) -> np.ndarray:
    x = np.zeros((len(peptides), len(peptides[0]) * len(AMINO_ACIDS)), dtype=np.float32)
    for row_index, peptide in enumerate(peptides):
        for position, aa in enumerate(peptide):
            x[row_index, position * len(AMINO_ACIDS) + AA_TO_INDEX[aa]] = 1.0
    return x


def get_models(seed: int):
    return {
        "blosum62_logistic_regression": (
            encode_blosum62,
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed),
            ),
        ),
        "onehot_logistic_regression": (
            encode_onehot,
            LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed),
        ),
        "blosum62_random_forest": (
            encode_blosum62,
            RandomForestClassifier(
                n_estimators=200,
                max_features="sqrt",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        "blosum62_extra_trees": (
            encode_blosum62,
            ExtraTreesClassifier(
                n_estimators=200,
                max_features="sqrt",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        "blosum62_hist_gradient_boosting": (
            encode_blosum62,
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                l2_regularization=0.01,
                random_state=seed,
            ),
        ),
    }


def predict_scores(model, x_test: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x_test)
    return model.predict(x_test)


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_score),
        "auprc": average_precision_score(y_true, y_score),
        "f1": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def summarize_results(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows_by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_model[str(row["model"])].append(row)

    summary_rows = []
    for model, model_rows in rows_by_model.items():
        weights = np.asarray([float(row["test_rows"]) for row in model_rows], dtype=np.float64)
        summary = {"model": model, "n_tasks": len(model_rows)}
        for metric in ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]:
            values = np.asarray([float(row[metric]) for row in model_rows], dtype=np.float64)
            summary[f"mean_{metric}"] = float(np.mean(values))
            summary[f"median_{metric}"] = float(np.median(values))
        for metric in ["accuracy", "auroc", "auprc"]:
            values = np.asarray([float(row[metric]) for row in model_rows], dtype=np.float64)
            summary[f"weighted_mean_{metric}"] = float(np.average(values, weights=weights))
        summary_rows.append(summary)

    summary_rows.sort(key=lambda row: float(row["mean_auroc"]), reverse=True)
    return summary_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    train_by_group = read_dataset(args.train)
    test_by_group = read_dataset(args.test)
    groups = sorted(set(train_by_group) & set(test_by_group))
    models = get_models(args.seed)

    result_rows: list[dict[str, object]] = []
    for model_name, (encoder, estimator) in models.items():
        print(f"model: {model_name}")
        for group_index, group in enumerate(groups, start=1):
            train_rows = train_by_group[group]
            test_rows = test_by_group[group]
            x_train = encoder([row["peptide_sequence"] for row in train_rows])
            y_train = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int8)
            x_test = encoder([row["peptide_sequence"] for row in test_rows])
            y_test = np.asarray([int(row["label"]) for row in test_rows], dtype=np.int8)

            estimator.fit(x_train, y_train)
            y_score = predict_scores(estimator, x_test)
            metrics = evaluate(y_test, y_score)
            train_positive = int(np.sum(y_train == 1))
            test_positive = int(np.sum(y_test == 1))
            result_rows.append(
                {
                    "model": model_name,
                    "target_tissue": group[0],
                    "mhc_restriction": group[1],
                    "train_rows": len(train_rows),
                    "test_rows": len(test_rows),
                    "train_positive": train_positive,
                    "train_negative": len(train_rows) - train_positive,
                    "test_positive": test_positive,
                    "test_negative": len(test_rows) - test_positive,
                    **metrics,
                }
            )
            print(f"  {group_index:02d}/{len(groups)} {group[0]} {group[1]} auroc={metrics['auroc']:.4f}")

    summary_rows = summarize_results(result_rows)
    write_csv(args.per_task_output, METRIC_COLUMNS, result_rows)
    write_csv(args.summary_output, SUMMARY_COLUMNS, summary_rows)

    metadata = {
        "train": str(args.train),
        "test": str(args.test),
        "per_task_output": str(args.per_task_output),
        "summary_output": str(args.summary_output),
        "n_tasks": len(groups),
        "models": list(models),
        "seed": args.seed,
        "feature_sets": {
            "blosum62": "9 positions x 20 BLOSUM62 substitution scores",
            "onehot": "9 positions x 20 amino-acid one-hot features",
        },
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote: {args.per_task_output}")
    print(f"wrote: {args.summary_output}")
    print(f"wrote: {args.metadata_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/tissuePMHC/tissuePMHC_train.csv.gz"))
    parser.add_argument("--test", type=Path, default=Path("data/tissuePMHC/tissuePMHC_test.csv.gz"))
    parser.add_argument("--per-task-output", type=Path, default=Path("results/tissuePMHC_baselines/per_task_metrics.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/tissuePMHC_baselines/summary_metrics.csv"))
    parser.add_argument("--metadata-output", type=Path, default=Path("results/tissuePMHC_baselines/metadata.json"))
    parser.add_argument("--seed", type=int, default=20260704)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
