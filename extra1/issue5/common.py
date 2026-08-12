from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
ISSUE_DIR = ROOT / "extra" / "issue5"
DEFAULT_RESULTS = ROOT / "results" / "issue5_general_pmhc"
ROW_KEYS = [
    "sample_id",
    "pair_id",
    "target_tissue",
    "mhc_restriction",
    "peptide_sequence",
    "label",
]
TASK_KEYS = ["target_tissue", "mhc_restriction"]
AA_PATTERN = r"[ACDEFGHIKLMNPQRSTVWY]{9}"


@dataclass(frozen=True)
class SpeciesSpec:
    species: str
    train: Path
    test: Path
    standard_predictions: Path
    standard_candidate: str
    fixed_predictions: Path
    fixed_candidate: str
    strict_predictions: Path
    strict_candidate: str
    strict_manifest: Path
    seeds: tuple[int, ...]


SPECS = {
    "human": SpeciesSpec(
        species="human",
        train=ROOT / "data/tissuePMHC_phase7_min200/tissuePMHC_phase7_min200_train.csv.gz",
        test=ROOT / "data/tissuePMHC_phase7_min200/tissuePMHC_phase7_min200_test.csv.gz",
        standard_predictions=ROOT
        / "results/tissuePMHC_phase7_min200_e29_multikernel_cnn/oof_predictions.csv",
        standard_candidate="phase7_min200_e29_cnn_3seed_mean",
        fixed_predictions=ROOT
        / "results/tissuePMHC_phase7_min200_e29_multikernel_cnn/test_predictions.csv",
        fixed_candidate="phase7_min200_e29_cnn_3seed_mean",
        strict_predictions=ROOT
        / "results/tissuePMHC_phase7_min200_e31_peptide_disjoint_oof/ensemble_oof_predictions.csv",
        strict_candidate="phase7_min200_e31_frozen_e29_3seed_mean",
        strict_manifest=ROOT
        / "results/tissuePMHC_phase7_min200_e31_peptide_disjoint_oof/pair_fold_assignments.csv",
        seeds=(20260704, 20260705, 20260706),
    ),
    "mouse": SpeciesSpec(
        species="mouse",
        train=ROOT / "data/mousePMHC/mousePMHC_train.csv.gz",
        test=ROOT / "data/mousePMHC/mousePMHC_test.csv.gz",
        standard_predictions=ROOT
        / "results/mousePMHC_phase4_e15_five_seed_confirmation/mousePMHC_phase4_e15_oof_predictions.csv",
        standard_candidate="mousePMHC_phase4_e15_e3b_5seed_probability_mean",
        fixed_predictions=ROOT
        / "results/mousePMHC_phase6_e32_e15_fixed_test/mousePMHC_phase6_e32_fixed_test_predictions.csv",
        fixed_candidate="mousePMHC_phase6_e32_frozen_e15_5seed_probability_mean",
        strict_predictions=ROOT
        / "results/mousePMHC_phase6_e33_peptide_disjoint_oof/mousePMHC_phase6_e33_ensemble_oof_predictions.csv",
        strict_candidate="mousePMHC_phase6_e33_frozen_e15_5seed_probability_mean",
        strict_manifest=ROOT
        / "results/mousePMHC_phase6_e33_peptide_disjoint_oof/mousePMHC_phase6_e33_pair_fold_assignments.csv",
        seeds=(20260704, 20260705, 20260706, 20260707, 20260708),
    ),
}


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


def query_id(species: str, peptide: str, mhc: str) -> str:
    payload = f"{species}\t{peptide}\t{mhc}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def tool_allele(mhc: str) -> str:
    if mhc.startswith("HLA-"):
        return mhc
    if mhc.startswith("H2-"):
        return "H-2-" + mhc[3:]
    raise ValueError(f"Unsupported MHC name: {mhc}")


def safe_name(mhc: str) -> str:
    return (
        tool_allele(mhc)
        .replace("*", "_")
        .replace(":", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def read_benchmark(path: Path, species: str, expected_split: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(ROW_KEYS + ["dataset", "split"]) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} misses required columns: {missing}")
    if expected_split is not None and set(frame["split"]) != {expected_split}:
        raise ValueError(f"{path} is not exclusively split={expected_split}.")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate sample_id values.")
    if not frame["peptide_sequence"].astype(str).str.fullmatch(AA_PATTERN).all():
        raise ValueError(f"{path} contains non-standard or non-9mer peptides.")
    prefix = "HLA-" if species == "human" else "H2-"
    if not frame["mhc_restriction"].astype(str).str.startswith(prefix).all():
        raise ValueError(f"{path} contains unexpected MHC restrictions for {species}.")
    pair_check = frame.groupby("pair_id")["label"].agg(["size", "sum", "nunique"])
    if not (
        (pair_check["size"] == 2)
        & (pair_check["sum"] == 1)
        & (pair_check["nunique"] == 2)
    ).all():
        raise ValueError(f"{path} does not contain complete one-positive/one-negative pairs.")
    result = frame.copy()
    result["task_name"] = (
        result["target_tissue"].astype(str) + "||" + result["mhc_restriction"].astype(str)
    )
    result["query_id"] = [
        query_id(species, peptide, mhc)
        for peptide, mhc in zip(result["peptide_sequence"], result["mhc_restriction"])
    ]
    return result


def build_query_frame(species: str) -> pd.DataFrame:
    spec = SPECS[species]
    union = pd.concat(
        [
            read_benchmark(spec.train, species, "train"),
            read_benchmark(spec.test, species, "test"),
        ],
        ignore_index=True,
    )
    queries = union[
        ["query_id", "peptide_sequence", "mhc_restriction"]
    ].drop_duplicates()
    if queries["query_id"].duplicated().any():
        raise AssertionError("query_id collision detected.")
    queries.insert(0, "species", species)
    queries["tool_allele"] = queries["mhc_restriction"].map(tool_allele)
    return queries.sort_values(
        ["mhc_restriction", "peptide_sequence"], kind="stable"
    ).reset_index(drop=True)


def load_score_caches(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise ValueError("At least one score cache is required.")
    frame = pd.concat(frames, ignore_index=True)
    required = {
        "query_id",
        "species",
        "peptide_sequence",
        "mhc_restriction",
        "predictor",
        "scoring_mode",
        "score",
        "is_supported",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Score caches miss columns: {missing}")
    keys = ["query_id", "predictor", "scoring_mode"]
    if frame.duplicated(keys).any():
        duplicates = frame.loc[frame.duplicated(keys, keep=False), keys].head()
        raise ValueError(f"Duplicate cached scores:\n{duplicates}")
    valid = frame["is_supported"].astype(bool)
    if not np.isfinite(frame.loc[valid, "score"].astype(float)).all():
        raise ValueError("Supported cache rows must have finite scores.")
    if frame.loc[~valid, "score"].notna().any():
        raise ValueError("Unsupported cache rows must have missing scores.")
    return frame


def select_main_predictions(
    path: Path, candidate: str, data: pd.DataFrame, protocol: str
) -> pd.DataFrame:
    predictions = pd.read_csv(path)
    required = {"sample_id", "candidate", "score", "label"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"{path} misses main-prediction columns: {missing}")
    selected = predictions[predictions["candidate"].astype(str) == candidate].copy()
    if len(selected) != len(data):
        raise ValueError(
            f"{protocol}: candidate {candidate!r} has {len(selected)} rows; expected {len(data)}."
        )
    if selected["sample_id"].duplicated().any():
        raise ValueError(f"{protocol}: duplicate main-model sample_id.")
    selected = data[ROW_KEYS + ["task_name", "query_id"]].merge(
        selected[["sample_id", "label", "score"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_prediction"),
    )
    if selected["score"].isna().any():
        raise ValueError(f"{protocol}: main predictions do not cover the benchmark.")
    if not (
        selected["label"].astype(int) == selected["label_prediction"].astype(int)
    ).all():
        raise ValueError(f"{protocol}: main-prediction labels disagree with benchmark.")
    return selected.drop(columns="label_prediction").rename(columns={"score": "main_score"})


def pair_status(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_id, pair in frame.groupby("pair_id", sort=False):
        supported = pair[score_column].notna()
        rows.append(
            {
                "pair_id": pair_id,
                "task_name": pair["task_name"].iloc[0],
                "target_tissue": pair["target_tissue"].iloc[0],
                "mhc_restriction": pair["mhc_restriction"].iloc[0],
                "n_scored_rows": int(supported.sum()),
                "complete_pair": bool(supported.all() and len(pair) == 2),
            }
        )
    return pd.DataFrame(rows)


def per_task_metrics(
    frame: pd.DataFrame, score_column: str, model: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pair_status(frame, score_column)
    metric_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for task_name, task in frame.groupby("task_name", sort=True):
        task_pairs = pairs[pairs["task_name"] == task_name]
        complete_ids = set(task_pairs.loc[task_pairs["complete_pair"], "pair_id"])
        scored = task[task["pair_id"].isin(complete_ids)].copy()
        base = {
            "model": model,
            "task_name": task_name,
            "target_tissue": task["target_tissue"].iloc[0],
            "mhc_restriction": task["mhc_restriction"].iloc[0],
            "n_rows": int(len(task)),
            "n_scored_rows": int(task[score_column].notna().sum()),
            "n_pairs": int(len(task_pairs)),
            "n_complete_pairs": int(len(complete_ids)),
            "row_coverage": float(task[score_column].notna().mean()),
            "pair_coverage": float(task_pairs["complete_pair"].mean()),
            "full_coverage": bool(task[score_column].notna().all()),
        }
        coverage_rows.append(base)
        if not complete_ids:
            continue
        labels = scored["label"].to_numpy(int)
        scores = scored[score_column].to_numpy(float)
        if len(np.unique(labels)) != 2:
            continue
        pivot = scored.pivot(index="pair_id", columns="label", values=score_column)
        margins = pivot[1] - pivot[0]
        metric_rows.append(
            {
                **base,
                "auroc": float(roc_auc_score(labels, scores)),
                "auprc": float(average_precision_score(labels, scores)),
                "pair_acc": float((margins > 0).mean()),
                "pair_acc_half_ties": float(
                    ((margins > 0).sum() + 0.5 * (margins == 0).sum()) / len(margins)
                ),
                "pair_wins": int((margins > 0).sum()),
                "pair_ties": int((margins == 0).sum()),
                "pair_losses": int((margins < 0).sum()),
                "median_pair_margin": float(np.median(margins)),
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(coverage_rows)


def hodges_lehmann_onesample(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    walsh = (values[:, None] + values[None, :]) / 2.0
    return float(np.median(walsh[np.triu_indices(len(values))]))


def bootstrap_mean_ci(
    values: np.ndarray, seed: int = 20260724, iterations: int = 10000
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    chunk = 1000
    for start in range(0, iterations, chunk):
        stop = min(start + chunk, iterations)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paired_comparison(
    external: pd.DataFrame,
    main: pd.DataFrame,
    metric: str,
    seed: int = 20260724,
    bootstrap_iterations: int = 10000,
) -> tuple[dict[str, Any], pd.DataFrame]:
    left = external[["task_name", metric]].rename(columns={metric: "external_value"})
    right = main[["task_name", metric]].rename(columns={metric: "main_value"})
    joined = left.merge(right, on="task_name", how="inner", validate="one_to_one")
    joined["difference_main_minus_external"] = (
        joined["main_value"] - joined["external_value"]
    )
    delta = joined["difference_main_minus_external"].to_numpy(float)
    if not len(delta):
        raise ValueError(f"No paired tasks for {metric}.")
    low, high = bootstrap_mean_ci(delta, seed, bootstrap_iterations)
    nonzero = delta[delta != 0]
    if len(nonzero):
        test = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
        statistic, pvalue = float(test.statistic), float(test.pvalue)
    else:
        statistic, pvalue = 0.0, 1.0
    summary = {
        "metric": metric,
        "n_tasks": int(len(delta)),
        "external_mean": float(joined["external_value"].mean()),
        "main_mean": float(joined["main_value"].mean()),
        "mean_difference": float(np.mean(delta)),
        "median_difference": float(np.median(delta)),
        "hodges_lehmann_difference": hodges_lehmann_onesample(delta),
        "bootstrap_mean_ci_low": low,
        "bootstrap_mean_ci_high": high,
        "wins": int((delta > 0).sum()),
        "ties": int((delta == 0).sum()),
        "losses": int((delta < 0).sum()),
        "wilcoxon_statistic": statistic,
        "wilcoxon_pvalue": pvalue,
    }
    return summary, joined


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    values = pvalues.astype(float).to_numpy()
    count = len(values)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * count / np.arange(1, count + 1))[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty(count, dtype=float)
    result[order] = adjusted
    return pd.Series(result, index=pvalues.index)


def make_standard_folds(train: pd.DataFrame, folds: int = 3, seed: int = 20260711) -> pd.Series:
    rng = np.random.default_rng(seed)
    assignment = pd.Series(index=train.index, dtype="int64")
    for task_name, task in train.groupby("task_name", sort=True):
        pairs = np.asarray(sorted(task["pair_id"].astype(str).unique()))
        shuffled = rng.permutation(pairs)
        pair_to_fold = {pair: index % folds for index, pair in enumerate(shuffled)}
        assignment.loc[task.index] = task["pair_id"].astype(str).map(pair_to_fold)
    if assignment.isna().any():
        raise AssertionError("Some rows lack a standard-fold assignment.")
    return assignment.astype(int)


def load_strict_folds(train: pd.DataFrame, manifest_path: Path) -> pd.Series:
    manifest = pd.read_csv(manifest_path, dtype={"pair_id": str})
    required = {"pair_id", "fold"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Strict manifest misses columns: {missing}")
    if manifest["pair_id"].duplicated().any():
        raise ValueError("Strict manifest contains duplicate pair IDs.")
    mapped = train["pair_id"].astype(str).map(manifest.set_index("pair_id")["fold"])
    if mapped.isna().any() or set(mapped.astype(int)) != {0, 1, 2}:
        raise ValueError("Strict manifest does not exactly cover the benchmark.")
    for fold in range(3):
        fit = train[mapped != fold]
        held = train[mapped == fold]
        if set(fit["peptide_sequence"]) & set(held["peptide_sequence"]):
            raise ValueError(f"Strict fold {fold} has peptide leakage.")
    return mapped.astype(int)


def seed_everything(seed: int, torch: Any | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def finite_or_none(value: Any) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None
