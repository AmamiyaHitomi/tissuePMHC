from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

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


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts"
ROW_KEYS = [
    "sample_id",
    "pair_id",
    "target_tissue",
    "mhc_restriction",
    "peptide_sequence",
    "label",
]
TASK_KEYS = ["target_tissue", "mhc_restriction"]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 25
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    embedding_dim: int = 16
    hidden_dim: int = 128
    dropout: float = 0.2
    tissue_loss_weight: float = 0.1
    mhc_loss_weight: float = 0.1
    max_grad_norm: float = 1.0
    expert_dim: int = 64
    condition_dim: int = 16
    gate_hidden_dim: int = 64
    n_experts: int = 3
    gate_entropy_weight: float = 0.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_training_data(path: Path, species: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(ROW_KEYS) | {"dataset", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} misses required columns: {missing}")
    if set(frame["split"]) != {"train"}:
        raise ValueError("Issue 9 runners accept training rows only; fixed test data must not be read.")
    if frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique.")
    pair_sizes = frame.groupby("pair_id").size()
    if not (pair_sizes == 2).all():
        raise ValueError("Every pair_id must contain exactly two rows.")
    pair_labels = frame.groupby("pair_id")["label"].agg(["sum", "nunique"])
    if not ((pair_labels["sum"] == 1) & (pair_labels["nunique"] == 2)).all():
        raise ValueError("Every pair must contain one positive and one negative row.")
    frame = frame.copy()
    frame["task_name"] = (
        frame["target_tissue"].astype(str) + "||" + frame["mhc_restriction"].astype(str)
    )
    expected_prefix = "HLA-" if species == "human" else "H2-"
    if not frame["mhc_restriction"].str.startswith(expected_prefix).all():
        raise ValueError(f"{species} data contain an unexpected MHC restriction.")
    return frame


def load_frozen_folds(
    train: pd.DataFrame,
    manifest_path: Path,
    expected_folds: int = 3,
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    manifest = pd.read_csv(manifest_path, dtype={"pair_id": str, "component_id": str})
    required = {"pair_id", "task_name", "component_id", "fold"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Fold manifest misses columns: {missing}")
    if manifest["pair_id"].duplicated().any():
        raise ValueError("Fold manifest has duplicate pair_id values.")
    if set(manifest["fold"].astype(int)) != set(range(expected_folds)):
        raise ValueError(f"Fold manifest must contain folds 0..{expected_folds - 1}.")

    data_pairs = train[["pair_id", "task_name"]].drop_duplicates().copy()
    data_pairs["pair_id"] = data_pairs["pair_id"].astype(str)
    joined = data_pairs.merge(
        manifest[["pair_id", "task_name", "component_id", "fold"]],
        on="pair_id",
        how="outer",
        validate="one_to_one",
        suffixes=("_data", "_manifest"),
        indicator=True,
    )
    if not (joined["_merge"] == "both").all():
        counts = joined["_merge"].value_counts().to_dict()
        raise ValueError(f"Training data and frozen fold manifest differ: {counts}")
    if not (joined["task_name_data"] == joined["task_name_manifest"]).all():
        raise ValueError("Task names disagree between training data and manifest.")

    fold_by_pair = manifest.set_index("pair_id")["fold"].astype(int)
    row_folds = train["pair_id"].astype(str).map(fold_by_pair)
    if row_folds.isna().any():
        raise AssertionError("Some training rows were not assigned a fold.")

    audits: list[dict[str, Any]] = []
    for fold in range(expected_folds):
        fitting = train[row_folds != fold]
        held = train[row_folds == fold]
        peptide_overlap = set(fitting["peptide_sequence"]) & set(held["peptide_sequence"])
        pair_overlap = set(fitting["pair_id"].astype(str)) & set(held["pair_id"].astype(str))
        missing_tasks = sorted(set(train["task_name"]) - set(held["task_name"]))
        if peptide_overlap or pair_overlap or missing_tasks:
            raise ValueError(
                f"Frozen fold {fold} failed audit: peptide_overlap={len(peptide_overlap)}, "
                f"pair_overlap={len(pair_overlap)}, missing_tasks={len(missing_tasks)}"
            )
        audits.append(
            {
                "fold": fold,
                "fit_rows": len(fitting),
                "held_rows": len(held),
                "fit_pairs": int(fitting["pair_id"].nunique()),
                "held_pairs": int(held["pair_id"].nunique()),
                "peptide_overlap": 0,
                "pair_overlap": 0,
                "held_tasks": int(held["task_name"].nunique()),
            }
        )
    audit = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "n_pairs": int(train["pair_id"].nunique()),
        "n_components": int(manifest["component_id"].nunique()),
        "n_folds": expected_folds,
        "folds": audits,
    }
    return row_folds.astype(int), manifest, audit


def mappings(frame: pd.DataFrame) -> dict[str, Any]:
    tasks = sorted(frame["task_name"].unique())
    tissues = sorted(frame["target_tissue"].unique())
    mhcs = sorted(frame["mhc_restriction"].unique())
    return {
        "tasks": tasks,
        "task_to_id": {value: index for index, value in enumerate(tasks)},
        "tissue_to_id": {value: index for index, value in enumerate(tissues)},
        "mhc_to_id": {value: index for index, value in enumerate(mhcs)},
    }


def add_ids(frame: pd.DataFrame, maps: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    result["task_id"] = result["task_name"].map(maps["task_to_id"])
    result["tissue_id"] = result["target_tissue"].map(maps["tissue_to_id"])
    result["mhc_id"] = result["mhc_restriction"].map(maps["mhc_to_id"])
    if result[["task_id", "tissue_id", "mhc_id"]].isna().any().any():
        raise ValueError("Categorical mapping failed.")
    return result


def seed_everything(seed: int, torch: Any | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def device_name(requested: str, torch: Any) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return requested


def prediction_rows(
    held: pd.DataFrame,
    scores: np.ndarray,
    species: str,
    model: str,
    seed: int,
    fold: int,
) -> pd.DataFrame:
    if len(scores) != len(held):
        raise ValueError("Prediction count does not match held-out rows.")
    result = held[ROW_KEYS].copy()
    result.insert(0, "species", species)
    result.insert(1, "model", model)
    result.insert(2, "seed", int(seed))
    result.insert(3, "fold", int(fold))
    result["score"] = np.asarray(scores, dtype=float)
    if not np.isfinite(result["score"]).all():
        raise ValueError(f"{model} produced non-finite scores.")
    return result


def validate_member_predictions(
    members: pd.DataFrame,
    train: pd.DataFrame,
    models: Iterable[str],
    seeds_by_model: dict[str, list[int]],
) -> None:
    expected_ids = set(train["sample_id"].astype(str))
    for model in models:
        for seed in seeds_by_model[model]:
            subset = members[(members["model"] == model) & (members["seed"] == seed)]
            ids = set(subset["sample_id"].astype(str))
            if len(subset) != len(train) or ids != expected_ids or subset["sample_id"].duplicated().any():
                raise AssertionError(f"Incomplete OOF coverage for model={model}, seed={seed}.")


def ensemble_predictions(members: pd.DataFrame) -> pd.DataFrame:
    keys = ["species", "model", *ROW_KEYS]
    expected = members.groupby("model")["seed"].nunique().to_dict()
    ensemble = members.groupby(keys, as_index=False).agg(
        fold=("fold", "first"),
        score=("score", "mean"),
        score_std=("score", lambda x: float(np.std(x, ddof=0))),
        n_members=("seed", "nunique"),
    )
    for model, count in expected.items():
        subset = ensemble[ensemble["model"] == model]
        if not (subset["n_members"] == count).all():
            raise AssertionError(f"Incomplete ensemble for {model}.")
    return ensemble


def _safe_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if len(np.unique(labels)) != 2:
        return {
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "auroc": np.nan,
            "auprc": np.nan,
            "f1": np.nan,
            "mcc": np.nan,
        }
    predictions = (scores >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }


def pair_accuracy(frame: pd.DataFrame) -> dict[str, float | int]:
    pivot_label = frame.pivot(index="pair_id", columns="label", values="score")
    if set(pivot_label.columns) != {0, 1} or pivot_label.isna().any().any():
        raise ValueError("PairAcc requires one scored positive and negative row per pair.")
    margins = pivot_label[1] - pivot_label[0]
    wins = int((margins > 0).sum())
    ties = int((margins == 0).sum())
    losses = int((margins < 0).sum())
    return {
        "n_pairs": int(len(margins)),
        "pair_acc": float(wins / len(margins)),
        "pair_acc_half_ties": float((wins + 0.5 * ties) / len(margins)),
        "pair_wins": wins,
        "pair_ties": ties,
        "pair_losses": losses,
        "median_pair_margin": float(np.median(margins)),
    }


def per_task_metrics(predictions: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    group_keys = ["species", "model"]
    if aggregation == "member":
        group_keys.append("seed")
    rows: list[dict[str, Any]] = []
    for keys, task in predictions.groupby(group_keys + TASK_KEYS, sort=True):
        values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_keys + TASK_KEYS, values))
        row["aggregation"] = aggregation
        row["n_rows"] = len(task)
        row["n_positive"] = int((task["label"] == 1).sum())
        row["n_negative"] = int((task["label"] == 0).sum())
        row.update(_safe_metrics(task["label"].to_numpy(int), task["score"].to_numpy(float)))
        row.update(pair_accuracy(task))
        rows.append(row)
    return pd.DataFrame(rows)


def summary_metrics(per_task: pd.DataFrame) -> pd.DataFrame:
    keys = ["species", "model", "aggregation"]
    if "seed" in per_task.columns:
        keys.append("seed")
    rows: list[dict[str, Any]] = []
    metrics = ["auroc", "auprc", "pair_acc", "pair_acc_half_ties"]
    for values, group in per_task.groupby(keys, dropna=False, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        row = dict(zip(keys, values))
        row["n_tasks"] = len(group)
        for metric in metrics:
            array = group[metric].dropna().to_numpy(float)
            row[f"mean_task_{metric}"] = float(np.mean(array))
            row[f"median_task_{metric}"] = float(np.median(array))
        row["worst_task_auroc"] = float(group["auroc"].min())
        count = min(10, len(group))
        row["worst10_task_auroc"] = float(group.nsmallest(count, "auroc")["auroc"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def save_outputs(
    output_dir: Path,
    members: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ensemble = ensemble_predictions(members)
    member_metrics = per_task_metrics(members, "member")
    ensemble_metrics = per_task_metrics(ensemble, "ensemble")
    members.to_csv(output_dir / "member_oof_predictions.csv.gz", index=False)
    ensemble.to_csv(output_dir / "ensemble_oof_predictions.csv.gz", index=False)
    member_metrics.to_csv(output_dir / "member_per_task_metrics.csv", index=False)
    ensemble_metrics.to_csv(output_dir / "ensemble_per_task_metrics.csv", index=False)
    pd.concat(
        [summary_metrics(member_metrics), summary_metrics(ensemble_metrics)],
        ignore_index=True,
    ).to_csv(output_dir / "summary_metrics.csv", index=False)
    atomic_json(output_dir / "metadata.json", metadata)


class Timer:
    def __init__(self) -> None:
        self.started = time.perf_counter()

    def elapsed(self) -> float:
        return float(time.perf_counter() - self.started)


def format_duration(seconds: float) -> str:
    minutes, remainder = divmod(float(seconds), 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {remainder:04.1f}s"
    if minutes:
        return f"{minutes:d}m {remainder:04.1f}s"
    return f"{remainder:.1f}s"


def config_dict(config: TrainConfig) -> dict[str, Any]:
    return asdict(config)
