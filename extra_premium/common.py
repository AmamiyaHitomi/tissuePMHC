"""Shared utilities for the isolated humanPMHC premium quick tests.

The original model implementations remain frozen under ``scripts/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
ORIGINAL_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data" / "humanPMHC_premium"
TRAIN_PATH = DATA_DIR / "humanPMHC_train.csv.gz"
TEST_PATH = DATA_DIR / "humanPMHC_test.csv.gz"
RESULTS_ROOT = EXPERIMENT_ROOT / "results"
EXTERNAL_ROOT = EXPERIMENT_ROOT / "external"

# Default seed used by the original single-seed premium runners.
SEED = 20260704

# Frozen defaults used by the original E2/E14/E29 human experiments.
DEFAULT_EPOCHS = 25
DEFAULT_BATCH_SIZE = 512
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EMBEDDING_DIM = 16
DEFAULT_HIDDEN_DIM = 128
DEFAULT_DROPOUT = 0.2
DEFAULT_TISSUE_LOSS_WEIGHT = 0.1
DEFAULT_HLA_LOSS_WEIGHT = 0.1
DEFAULT_MAX_GRAD_NORM = 1.0

REQUIRED_COLUMNS = {
    "dataset",
    "split",
    "sample_id",
    "pair_id",
    "label",
    "target_tissue",
    "mhc_restriction",
    "peptide_sequence",
}

PREDICTION_COLUMNS = [
    "model",
    "seed",
    "sample_id",
    "pair_id",
    "target_tissue",
    "mhc_restriction",
    "label",
    "score",
]

PER_TASK_COLUMNS = [
    "model",
    "seed",
    "target_tissue",
    "mhc_restriction",
    "train_rows",
    "test_rows",
    "accuracy",
    "balanced_accuracy",
    "auroc",
    "auprc",
    "f1",
    "mcc",
    "pair_accuracy",
]


def enable_original_modules() -> None:
    """Make the frozen model implementations importable without copying them."""
    # Importing the frozen source must not create/update __pycache__ files in
    # the original scripts directory.
    sys.dont_write_bytecode = True
    scripts_dir = str(ORIGINAL_SCRIPTS_DIR)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


def enable_project_package_imports() -> None:
    """Allow imports from existing project packages such as extra1.issue5."""
    sys.dont_write_bytecode = True
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def basic_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="auto uses CUDA when available; otherwise CPU.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Default 25 reproduces the original training length. Use a smaller value only for a smoke run.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser


def resolve_device(requested: str, torch: Any) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return requested


def model_args(cli_args: argparse.Namespace, **overrides: Any) -> SimpleNamespace:
    values = {
        "epochs": int(cli_args.epochs),
        "batch_size": int(cli_args.batch_size),
        "learning_rate": DEFAULT_LEARNING_RATE,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "embedding_dim": DEFAULT_EMBEDDING_DIM,
        "hidden_dim": DEFAULT_HIDDEN_DIM,
        "dropout": DEFAULT_DROPOUT,
        "tissue_loss_weight": DEFAULT_TISSUE_LOSS_WEIGHT,
        "hla_loss_weight": DEFAULT_HLA_LOSS_WEIGHT,
        "max_grad_norm": DEFAULT_MAX_GRAD_NORM,
        "kernel_sizes": [2, 3, 5],
        "conv_channels": 32,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def load_premium_data(base: Any) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], int]:
    """Load and strictly audit the isolated premium train/test files."""
    train = base.read_dataset(TRAIN_PATH)
    test = base.read_dataset(TEST_PATH)

    for split_name, frame in (("train", train), ("test", test)):
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{split_name} is missing required columns: {sorted(missing)}")
        if set(frame["dataset"].astype(str)) != {"humanPMHC"}:
            raise ValueError(f"{split_name} contains a non-humanPMHC dataset value.")
        if set(frame["split"].astype(str)) != {split_name}:
            raise ValueError(f"{split_name} contains an unexpected split value.")
        if set(frame["label"].astype(int)) != {0, 1}:
            raise ValueError(f"{split_name} labels must contain both 0 and 1.")
        if not frame["peptide_sequence"].astype(str).str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{9}").all():
            raise ValueError(f"{split_name} must contain only canonical 9-mer peptides.")

        pair_labels = frame.groupby("pair_id", sort=False)["label"].agg(list)
        if not pair_labels.map(lambda labels: sorted(int(value) for value in labels) == [0, 1]).all():
            raise ValueError(f"Every {split_name} pair_id must contain exactly one positive and one negative row.")

    overlap = set(train["pair_id"].astype(str)) & set(test["pair_id"].astype(str))
    if overlap:
        raise ValueError(f"Train/test pair_id leakage detected: {len(overlap)} overlapping pairs.")

    original_train_rows = len(train)
    original_test_rows = len(test)
    train, test, mappings = base.add_task_columns(train, test)
    if len(train) != original_train_rows or len(test) != original_test_rows:
        raise ValueError("Some rows were dropped because their tissue-HLA task was absent from one split.")

    peptide_length = int(
        max(
            train["peptide_sequence"].str.len().max(),
            test["peptide_sequence"].str.len().max(),
        )
    )
    return train, test, mappings, peptide_length


def pair_accuracy(task: pd.DataFrame) -> float:
    """Fraction of pairs whose positive row receives a higher score."""
    wide = task.pivot(index="pair_id", columns="label", values="score")
    if list(wide.columns) != [0, 1] or wide.isna().any().any():
        raise ValueError("Pair accuracy requires exactly one label-0 and one label-1 score per pair.")
    return float((wide[1] > wide[0]).mean())


def save_basic_test_results(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    scores: np.ndarray | pd.Series,
    base: Any,
    settings: dict[str, Any],
    *,
    seed: int = SEED,
) -> dict[str, Any]:
    """Write only predictions, per-task metrics, and one compact summary."""
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.shape != (len(test),):
        raise ValueError(f"Expected {len(test)} test scores, received shape={score_array.shape}.")
    if not np.isfinite(score_array).all():
        raise ValueError("Test scores contain NaN or infinite values.")

    predictions = test[
        ["sample_id", "pair_id", "target_tissue", "mhc_restriction", "label"]
    ].copy()
    predictions.insert(0, "seed", seed)
    predictions.insert(0, "model", model_name)
    predictions["score"] = score_array

    train_counts = (
        train.groupby(["target_tissue", "mhc_restriction"], sort=True)
        .size()
        .rename("train_rows")
    )
    metric_rows: list[dict[str, Any]] = []
    for (tissue, hla), task in predictions.groupby(
        ["target_tissue", "mhc_restriction"], sort=True
    ):
        metrics = base.evaluate(
            task["label"].to_numpy(dtype=np.int64),
            task["score"].to_numpy(dtype=np.float64),
        )
        metric_rows.append(
            {
                "model": model_name,
                "seed": seed,
                "target_tissue": tissue,
                "mhc_restriction": hla,
                "train_rows": int(train_counts.loc[(tissue, hla)]),
                "test_rows": int(len(task)),
                **metrics,
                "pair_accuracy": pair_accuracy(task),
            }
        )
    per_task = pd.DataFrame(metric_rows, columns=PER_TASK_COLUMNS)

    global_metrics = base.evaluate(
        predictions["label"].to_numpy(dtype=np.int64),
        predictions["score"].to_numpy(dtype=np.float64),
    )
    worst_n = min(10, len(per_task))
    summary = {
        "model": model_name,
        "seed": seed,
        "n_tasks": int(len(per_task)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "mean_task_auroc": float(per_task["auroc"].mean()),
        "mean_task_auprc": float(per_task["auprc"].mean()),
        "mean_task_accuracy": float(per_task["accuracy"].mean()),
        "mean_task_mcc": float(per_task["mcc"].mean()),
        "mean_task_pair_accuracy": float(per_task["pair_accuracy"].mean()),
        "worst_10_mean_auroc": float(per_task.nsmallest(worst_n, "auroc")["auroc"].mean()),
        "global_auroc": float(global_metrics["auroc"]),
        "global_auprc": float(global_metrics["auprc"]),
        "global_accuracy": float(global_metrics["accuracy"]),
        "global_mcc": float(global_metrics["mcc"]),
    }

    output_dir = RESULTS_ROOT / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    per_task.to_csv(output_dir / "per_task_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "summary_metrics.csv", index=False)
    (output_dir / "run_settings.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "seed": seed,
                "train": str(TRAIN_PATH),
                "test": str(TEST_PATH),
                **settings,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"wrote: {output_dir / 'test_predictions.csv'}", flush=True)
    print(f"wrote: {output_dir / 'per_task_metrics.csv'}", flush=True)
    print(f"wrote: {output_dir / 'summary_metrics.csv'}", flush=True)
    print(
        f"{model_name}: mean_task_auroc={summary['mean_task_auroc']:.5f}, "
        f"mean_task_auprc={summary['mean_task_auprc']:.5f}, "
        f"pair_accuracy={summary['mean_task_pair_accuracy']:.5f}",
        flush=True,
    )
    return summary
