"""Shared read-only helpers for final-phase analyses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "results" / "final_phase"
SPLIT_SEED = 20260711
N_FOLDS = 3


@dataclass(frozen=True)
class Experiment:
    species: str
    train: Path
    standard_predictions: Path
    standard_candidate: str
    strict_predictions: Path
    strict_assignments: Path
    task_kind: str


EXPERIMENTS = {
    "human": Experiment(
        species="human",
        train=ROOT / "data/tissuePMHC_phase7_min200/tissuePMHC_phase7_min200_train.csv.gz",
        standard_predictions=ROOT / "results/tissuePMHC_phase7_min200_e29_multikernel_cnn/oof_predictions.csv",
        standard_candidate="phase7_min200_e29_cnn_3seed_mean",
        strict_predictions=ROOT / "results/tissuePMHC_phase7_min200_e31_peptide_disjoint_oof/ensemble_oof_predictions.csv",
        strict_assignments=ROOT / "results/tissuePMHC_phase7_min200_e31_peptide_disjoint_oof/pair_fold_assignments.csv",
        task_kind="tissue-HLA",
    ),
    "mouse": Experiment(
        species="mouse",
        train=ROOT / "data/mousePMHC/mousePMHC_train.csv.gz",
        standard_predictions=ROOT / "results/mousePMHC_phase4_e15_five_seed_confirmation/mousePMHC_phase4_e15_oof_predictions.csv",
        standard_candidate="mousePMHC_phase4_e15_e3b_5seed_probability_mean",
        strict_predictions=ROOT / "results/mousePMHC_phase6_e33_peptide_disjoint_oof/mousePMHC_phase6_e33_ensemble_oof_predictions.csv",
        strict_assignments=ROOT / "results/mousePMHC_phase6_e33_peptide_disjoint_oof/mousePMHC_phase6_e33_pair_fold_assignments.csv",
        task_kind="tissue-H2",
    ),
}


def ensure_output(name: str) -> Path:
    path = OUTPUT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_train(experiment: Experiment) -> pd.DataFrame:
    frame = pd.read_csv(require(experiment.train))
    required = {
        "sample_id", "pair_id", "label", "target_tissue", "mhc_restriction",
        "peptide_sequence", "molecule_parent_uniprot_id",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{experiment.train} misses columns: {sorted(missing)}")
    frame["task_name"] = frame["target_tissue"] + "||" + frame["mhc_restriction"]
    return frame


def read_predictions(experiment: Experiment, protocol: str) -> pd.DataFrame:
    if protocol not in {"standard", "strict"}:
        raise ValueError(protocol)
    path = experiment.standard_predictions if protocol == "standard" else experiment.strict_predictions
    frame = pd.read_csv(require(path))
    if protocol == "standard" and "candidate" in frame:
        frame = frame[frame["candidate"] == experiment.standard_candidate].copy()
    if frame.empty:
        raise ValueError(f"No {protocol} predictions found for {experiment.species}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample predictions in {path}")
    return frame


def attach_data(experiment: Experiment, predictions: pd.DataFrame) -> pd.DataFrame:
    train = read_train(experiment)
    columns = [
        "sample_id", "pair_id", "label", "target_tissue", "mhc_restriction",
        "task_name", "peptide_sequence", "molecule_parent_uniprot_id",
    ]
    extra = [column for column in train.columns if provenance_like(column)]
    columns.extend(column for column in extra if column not in columns)
    prediction_columns = [column for column in predictions.columns if column not in columns or column == "sample_id"]
    merged = train[columns].merge(
        predictions[prediction_columns], on="sample_id", how="inner", validate="one_to_one",
        suffixes=("", "_prediction"),
    )
    if len(merged) != len(train) or len(merged) != len(predictions):
        raise ValueError(
            f"Prediction/data coverage mismatch for {experiment.species}: "
            f"train={len(train)} predictions={len(predictions)} merged={len(merged)}"
        )
    if "label_prediction" in merged and not np.array_equal(
        merged["label"].to_numpy(), merged["label_prediction"].to_numpy()
    ):
        raise ValueError("Prediction labels disagree with dataset labels")
    return merged


def make_standard_folds(train: pd.DataFrame, folds: int = N_FOLDS, seed: int = SPLIT_SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    assignment = pd.Series(index=train.index, dtype="int64")
    for task_name, task in train.groupby("task_name", sort=True):
        pairs = np.asarray(sorted(task["pair_id"].unique()))
        shuffled = rng.permutation(pairs)
        pair_to_fold = {pair: index % folds for index, pair in enumerate(shuffled)}
        assignment.loc[task.index] = task["pair_id"].map(pair_to_fold).astype(int)
    if assignment.isna().any():
        raise AssertionError("Some standard rows lack fold assignments")
    return assignment.astype(int)


def read_strict_pair_folds(experiment: Experiment) -> pd.DataFrame:
    frame = pd.read_csv(require(experiment.strict_assignments))
    required = {"pair_id", "fold"}
    if required - set(frame.columns):
        raise ValueError(f"Strict assignment misses {sorted(required - set(frame.columns))}")
    result = frame[["pair_id", "fold"]].drop_duplicates()
    if result["pair_id"].duplicated().any():
        raise ValueError("A strict pair has multiple folds")
    return result


def evaluate(y_true: Iterable[int], y_score: Iterable[float]) -> dict[str, float]:
    truth = np.asarray(list(y_true), dtype=int)
    score = np.asarray(list(y_score), dtype=float)
    prediction = (score >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "auroc": float(roc_auc_score(truth, score)),
        "auprc": float(average_precision_score(truth, score)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(truth, prediction)),
    }


def task_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tissue, mhc), task in frame.groupby(["target_tissue", "mhc_restriction"], sort=True):
        rows.append({
            "target_tissue": tissue,
            "mhc_restriction": mhc,
            "rows": len(task),
            "pairs": task["pair_id"].nunique() if "pair_id" in task else len(task) // 2,
            **evaluate(task["label"], task["score"]),
        })
    return pd.DataFrame(rows)


def provenance_like(column: str) -> bool:
    lowered = column.lower()
    tokens = ("pmid", "study", "assay", "publication", "submitted", "date", "source_id", "dataset_id")
    return any(token in lowered for token in tokens)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with require(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

