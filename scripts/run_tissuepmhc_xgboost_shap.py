#!/usr/bin/env python3
"""Train a global XGBoost tissue-pMHC classifier and explain it with SHAP.

The script consumes the project's standard train/test CSV files. Features are
deliberately simple and interpretable: peptide positional one-hot encoding,
amino-acid composition, a few physicochemical summaries, tissue, and MHC.

Example (quick end-to-end check):
    python scripts/run_tissuepmhc_xgboost_shap.py --smoke

Example (full default dataset):
    python scripts/run_tissuepmhc_xgboost_shap.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBIC = set("AILMFWVY")
AROMATIC = set("FWY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
REQUIRED_COLUMNS = {
    "label",
    "target_tissue",
    "mhc_restriction",
    "peptide_sequence",
}


def import_xgboost_and_shap():
    """Give an actionable error instead of a long import traceback."""
    try:
        import shap
        import xgboost as xgb
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install it with:\n"
            "  python -m pip install xgboost shap\n"
            f"Original error: {exc}"
        ) from exc
    return xgb, shap


def load_split(path: Path, max_rows: int | None, seed: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(np.int8)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{path}: label must contain only 0/1")
    for column in ("target_tissue", "mhc_restriction", "peptide_sequence"):
        frame[column] = frame[column].fillna("UNKNOWN").astype(str)
    frame["peptide_sequence"] = frame["peptide_sequence"].str.upper().str.strip()

    if max_rows is not None and len(frame) > max_rows:
        # Keep complete positive/negative pairs so pair-accuracy remains meaningful.
        if "pair_id" in frame.columns and frame["pair_id"].notna().all():
            pair_sizes = frame.groupby("pair_id", sort=False).size()
            shuffled_pairs = pair_sizes.sample(frac=1.0, random_state=seed)
            selected_pairs: list[object] = []
            selected_rows = 0
            for pair_id, pair_size in shuffled_pairs.items():
                if selected_rows + int(pair_size) > max_rows:
                    continue
                selected_pairs.append(pair_id)
                selected_rows += int(pair_size)
                if selected_rows == max_rows:
                    break
            frame = frame.loc[frame["pair_id"].isin(selected_pairs)].copy()
        else:
            frame = frame.sample(n=max_rows, random_state=seed)
        frame = frame.sort_index().reset_index(drop=True)
    return frame


def learn_categories(train: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "target_tissue": sorted(train["target_tissue"].unique().tolist()),
        "mhc_restriction": sorted(train["mhc_restriction"].unique().tolist()),
    }


def make_features(
    frame: pd.DataFrame,
    categories: dict[str, list[str]],
    max_length: int,
) -> pd.DataFrame:
    """Create a dense, human-readable feature matrix for TreeSHAP."""
    peptides = frame["peptide_sequence"].tolist()
    values: dict[str, np.ndarray] = {}

    for position in range(max_length):
        residues = np.asarray(
            [peptide[position] if position < len(peptide) else "PAD" for peptide in peptides],
            dtype=object,
        )
        for aa in AMINO_ACIDS:
            values[f"peptide_pos_{position + 1}_{aa}"] = (residues == aa).astype(np.float32)

    lengths = np.asarray([max(len(peptide), 1) for peptide in peptides], dtype=np.float32)
    values["peptide_length"] = lengths
    for aa in AMINO_ACIDS:
        values[f"peptide_fraction_{aa}"] = np.asarray(
            [peptide.count(aa) / max(len(peptide), 1) for peptide in peptides], dtype=np.float32
        )
    for name, residue_set in (
        ("hydrophobic_fraction", HYDROPHOBIC),
        ("aromatic_fraction", AROMATIC),
        ("positive_fraction", POSITIVE),
        ("negative_fraction", NEGATIVE),
    ):
        values[name] = np.asarray(
            [sum(aa in residue_set for aa in peptide) / max(len(peptide), 1) for peptide in peptides],
            dtype=np.float32,
        )
    values["charge_proxy"] = values["positive_fraction"] - values["negative_fraction"]

    for column, known_values in categories.items():
        raw = frame[column].to_numpy(dtype=object)
        for category in known_values:
            values[f"{column}={category}"] = (raw == category).astype(np.float32)

    return pd.DataFrame(values, index=frame.index, dtype=np.float32)


def safe_binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(np.int8)
    result = {
        "n_rows": int(len(y_true)),
        "positive_rows": int(np.sum(y_true == 1)),
        "negative_rows": int(np.sum(y_true == 0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc": float("nan"),
        "auprc": float("nan"),
    }
    if np.unique(y_true).size == 2:
        result["auroc"] = float(roc_auc_score(y_true, y_score))
        result["auprc"] = float(average_precision_score(y_true, y_score))
    return result


def paired_accuracy(predictions: pd.DataFrame) -> float:
    if "pair_id" not in predictions.columns:
        return float("nan")
    correct: list[float] = []
    for _, group in predictions.groupby("pair_id", sort=False):
        positive = group.loc[group["label"] == 1, "prediction"].to_numpy()
        negative = group.loc[group["label"] == 0, "prediction"].to_numpy()
        if len(positive) == 1 and len(negative) == 1:
            correct.append(float(positive[0] > negative[0]) + 0.5 * float(positive[0] == negative[0]))
    return float(np.mean(correct)) if correct else float("nan")


def build_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, object]] = []
    for (tissue, mhc), group in predictions.groupby(["target_tissue", "mhc_restriction"], sort=True):
        metrics = safe_binary_metrics(group["label"].to_numpy(), group["prediction"].to_numpy())
        rows.append({"target_tissue": tissue, "mhc_restriction": mhc, **metrics, "pair_accuracy": paired_accuracy(group)})
    per_task = pd.DataFrame(rows)
    overall = safe_binary_metrics(predictions["label"].to_numpy(), predictions["prediction"].to_numpy())
    overall["pair_accuracy"] = paired_accuracy(predictions)
    overall["n_tasks"] = int(len(per_task))
    if not per_task.empty:
        overall["macro_auroc"] = float(per_task["auroc"].mean())
        overall["macro_auprc"] = float(per_task["auprc"].mean())
        overall["macro_pair_accuracy"] = float(per_task["pair_accuracy"].mean())
    return per_task, overall


def normalize_shap_values(raw_values: object) -> np.ndarray:
    if isinstance(raw_values, list):
        raw_values = raw_values[-1]
    array = np.asarray(raw_values)
    if array.ndim == 3:
        array = array[:, :, -1]
    if array.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {array.shape}")
    return array


def run(args: argparse.Namespace) -> None:
    xgb, shap = import_xgboost_and_shap()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_train_rows = args.max_train_rows
    max_test_rows = args.max_test_rows
    n_estimators = args.n_estimators
    shap_sample_size = args.shap_sample_size
    if args.smoke:
        max_train_rows = min(max_train_rows or 4_000, 4_000)
        max_test_rows = min(max_test_rows or 1_000, 1_000)
        n_estimators = min(n_estimators, 30)
        shap_sample_size = min(shap_sample_size, 200)

    total_start = time.perf_counter()
    train = load_split(args.train, max_train_rows, args.seed)
    test = load_split(args.test, max_test_rows, args.seed + 1)
    max_length = max(train["peptide_sequence"].str.len().max(), test["peptide_sequence"].str.len().max())
    categories = learn_categories(train)
    x_train_frame = make_features(train, categories, int(max_length))
    x_test_frame = make_features(test, categories, int(max_length))
    y_train = train["label"].to_numpy(dtype=np.int8)
    y_test = test["label"].to_numpy(dtype=np.int8)

    print(f"seed={args.seed}")
    print(f"train_rows={len(train)} test_rows={len(test)} features={x_train_frame.shape[1]}")
    print("Training log (one line per boosting round):")
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        n_estimators=n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        tree_method="hist",
    )
    train_start = time.perf_counter()
    model.fit(
        x_train_frame,
        y_train,
        eval_set=[(x_test_frame, y_test)],
        verbose=True,
    )
    train_seconds = time.perf_counter() - train_start

    scores = model.predict_proba(x_test_frame)[:, 1]
    keep_columns = [
        column
        for column in ("sample_id", "pair_id", "label", "target_tissue", "mhc_restriction", "peptide_sequence")
        if column in test.columns
    ]
    predictions = test[keep_columns].copy()
    predictions["prediction"] = scores
    per_task, overall = build_metrics(predictions)

    sample_n = min(shap_sample_size, len(test))
    sample_indices = np.random.default_rng(args.seed).choice(len(test), size=sample_n, replace=False)
    shap_x = x_test_frame.iloc[sample_indices].copy()
    shap_meta = predictions.iloc[sample_indices].reset_index(drop=True)
    shap_start = time.perf_counter()
    explainer = shap.TreeExplainer(model)
    shap_values = normalize_shap_values(explainer.shap_values(shap_x))
    shap_seconds = time.perf_counter() - shap_start

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame(
        {"feature": x_train_frame.columns, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    shap_table = pd.concat(
        [
            shap_meta,
            pd.DataFrame(shap_values, columns=[f"shap::{name}" for name in x_train_frame.columns]),
        ],
        axis=1,
    )

    predictions.to_csv(args.output_dir / "test_predictions.csv.gz", index=False)
    per_task.to_csv(args.output_dir / "per_task_metrics.csv", index=False)
    pd.DataFrame([overall]).to_csv(args.output_dir / "summary_metrics.csv", index=False)
    importance.to_csv(args.output_dir / "shap_feature_importance.csv", index=False)
    shap_table.to_csv(args.output_dir / "shap_values.csv.gz", index=False)
    model.save_model(args.output_dir / "xgboost_model.json")

    shap.summary_plot(shap_values, shap_x, max_display=args.shap_max_display, show=False)
    plt.tight_layout()
    plt.savefig(args.output_dir / "shap_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close()
    shap.summary_plot(shap_values, shap_x, plot_type="bar", max_display=args.shap_max_display, show=False)
    plt.tight_layout()
    plt.savefig(args.output_dir / "shap_bar.png", dpi=180, bbox_inches="tight")
    plt.close()

    total_seconds = time.perf_counter() - total_start
    timing = pd.DataFrame(
        [
            {"seed": args.seed, "stage": "training", "seconds": train_seconds},
            {"seed": args.seed, "stage": "shap", "seconds": shap_seconds},
            {"seed": args.seed, "stage": "total", "seconds": total_seconds},
        ]
    )
    timing.to_csv(args.output_dir / "timing_results.csv", index=False)

    metadata = {
        "train": str(args.train),
        "test": str(args.test),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "smoke": args.smoke,
        "train_rows": len(train),
        "test_rows": len(test),
        "n_features": x_train_frame.shape[1],
        "max_peptide_length": int(max_length),
        "shap_rows": sample_n,
        "categories": categories,
        "model_parameters": model.get_params(),
        "versions": {
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "shap": shap.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"overall_auroc={overall['auroc']:.4f} overall_auprc={overall['auprc']:.4f}")
    print(f"seed={args.seed} training_seconds={train_seconds:.3f} total_seconds={total_seconds:.3f}")
    print(f"wrote={args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/tissuePMHC/tissuePMHC_train.csv.gz"))
    parser.add_argument("--test", type=Path, default=Path("data/tissuePMHC/tissuePMHC_test.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tissuePMHC_xgboost_shap"))
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-child-weight", type=float, default=2.0)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--reg-alpha", type=float, default=0.05)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--shap-sample-size", type=int, default=1_000)
    parser.add_argument("--shap-max-display", type=int, default=25)
    parser.add_argument("--smoke", action="store_true", help="Use capped data, 30 trees, and 200 SHAP rows")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
