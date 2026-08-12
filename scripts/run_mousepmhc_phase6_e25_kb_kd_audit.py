#!/usr/bin/env python3
"""Run Phase 6 E25: train-only Kb/Kd data and E15 residual audit.

This is a diagnostic-only experiment.  It validates the paired-label
construction, summarizes task and H2 data coverage, and audits frozen E15
five-seed OOF errors and member disagreement.  It never opens fixed test data
and does not fit or select a predictive model.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mousePMHC_phase6_e25_kb_kd_audit"
E15_CANDIDATE = "mousePMHC_phase4_e15_e3b_5seed_probability_mean"
E3B_CANDIDATE = "mousePMHC_phase3_e3b_task_balanced_mmoe_min200"
ORIGINAL_SEEDS = [20260704, 20260705, 20260706]
NEW_SEEDS = [20260707, 20260708]
ALL_SEEDS = [*ORIGINAL_SEEDS, *NEW_SEEDS]
KEYS = ["sample_id", "target_tissue", "mhc_restriction", "label"]
TASK_KEYS = ["target_tissue", "mhc_restriction"]
AA = "ACDEFGHIKLMNPQRSTVWY"


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def validate_train(frame: pd.DataFrame) -> None:
    required = {
        "dataset", "split", "sample_id", "pair_id", "label", "target_tissue", "mhc_restriction",
        "peptide_sequence", "molecule_parent_uniprot_id",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"E25 train file is missing columns: {sorted(missing)}")
    if set(frame.dataset) != {"mousePMHC"} or set(frame.split) != {"train"}:
        raise ValueError("E25 accepts mousePMHC train rows only.")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("E25 found a non-binary label.")
    if not frame.mhc_restriction.str.startswith("H2-").all():
        raise ValueError("E25 found a non-H2 task.")
    if not frame.peptide_sequence.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{9}").all():
        raise ValueError("E25 requires standard unmodified 9-mer peptides.")
    if frame.sample_id.duplicated().any():
        raise ValueError("E25 train sample_id values must be unique.")


def validate_member_predictions(frame: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "candidate", "seed", *KEYS, "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"E25 seed predictions are missing columns: {sorted(missing)}")
    frame = frame[(frame.split == "oof") & (frame.candidate == E3B_CANDIDATE) & frame.seed.isin(ALL_SEEDS)].copy()
    if set(frame.seed.unique()) != set(ALL_SEEDS):
        raise ValueError("E25 requires all five frozen E3b/E15 seed predictions.")
    expected = set(train.sample_id)
    for seed, part in frame.groupby("seed", sort=True):
        if len(part) != len(train) or part.sample_id.duplicated().any() or set(part.sample_id) != expected:
            raise ValueError(f"E25 seed {seed} does not provide exactly one aligned OOF score per train row.")
    labels = train.set_index("sample_id").label
    if not frame.apply(lambda row: int(labels.loc[row.sample_id]) == int(row.label), axis=1).all():
        raise ValueError("E25 seed prediction labels do not align with train labels.")
    return frame


def validate_e15_predictions(frame: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "candidate", *KEYS, "score", "prediction_std_across_seeds", "n_members"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"E25 E15 predictions are missing columns: {sorted(missing)}")
    frame = frame[(frame.split == "oof") & (frame.candidate == E15_CANDIDATE)].copy()
    if len(frame) != len(train) or frame.sample_id.duplicated().any() or set(frame.sample_id) != set(train.sample_id):
        raise ValueError("E25 E15 predictions do not align one-to-one with train rows.")
    if not (frame.n_members == 5).all():
        raise ValueError("E25 requires the frozen five-member E15 ensemble.")
    return frame


def pair_integrity(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for pair_id, part in train.groupby("pair_id", sort=True):
        labels = sorted(part.label.tolist())
        records.append({
            "pair_id": pair_id,
            "n_rows": len(part),
            "labels": ";".join(map(str, labels)),
            "single_task": int(part[TASK_KEYS].drop_duplicates().shape[0] == 1),
            "single_uniprot": int(part.molecule_parent_uniprot_id.nunique(dropna=False) == 1),
            "distinct_peptides": int(part.peptide_sequence.nunique() == 2),
            "valid_pair": int(len(part) == 2 and labels == [0, 1] and part[TASK_KEYS].drop_duplicates().shape[0] == 1
                              and part.molecule_parent_uniprot_id.nunique(dropna=False) == 1 and part.peptide_sequence.nunique() == 2),
        })
    pairs = pd.DataFrame(records)
    summary = pd.DataFrame([{
        "n_pairs": len(pairs), "valid_pairs": int(pairs.valid_pair.sum()), "invalid_pairs": int((1 - pairs.valid_pair).sum()),
        "non_binary_pairs": int((pairs.labels != "0;1").sum()), "multi_task_pairs": int((pairs.single_task == 0).sum()),
        "multi_uniprot_pairs": int((pairs.single_uniprot == 0).sum()), "same_peptide_pairs": int((pairs.distinct_peptides == 0).sum()),
    }])
    return pairs, summary


def task_data_audit(train: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    pair_tasks = train[["pair_id", *TASK_KEYS]].drop_duplicates().merge(pairs[["pair_id", "valid_pair"]], on="pair_id", how="left")
    rows: list[dict[str, Any]] = []
    for task, part in train.groupby(TASK_KEYS, sort=True):
        tissue, h2 = task
        labels_by_peptide = part.groupby("peptide_sequence").label.nunique()
        task_pairs = pair_tasks[(pair_tasks.target_tissue == tissue) & (pair_tasks.mhc_restriction == h2)]
        rows.append({
            "target_tissue": tissue, "mhc_restriction": h2, "train_rows": len(part), "train_pairs": int(part.pair_id.nunique()),
            "positive_rows": int((part.label == 1).sum()), "negative_rows": int((part.label == 0).sum()),
            "unique_peptides": int(part.peptide_sequence.nunique()), "unique_uniprots": int(part.molecule_parent_uniprot_id.nunique()),
            "rows_per_uniprot": float(len(part) / part.molecule_parent_uniprot_id.nunique()),
            "peptides_with_both_labels_in_task": int((labels_by_peptide > 1).sum()),
            "invalid_pairs": int((task_pairs.valid_pair == 0).sum()),
        })
    return pd.DataFrame(rows)


def per_task_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (tissue, h2), part in predictions.groupby(TASK_KEYS, sort=True):
        metrics = base.evaluate(part.label.to_numpy(dtype=int), part.score.to_numpy(dtype=float))
        rows.append({"target_tissue": tissue, "mhc_restriction": h2, "oof_rows": len(part), **metrics})
    return pd.DataFrame(rows)


def pair_margin_audit(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = predictions.set_index("sample_id")
    rows: list[dict[str, Any]] = []
    for pair_id, part in predictions.groupby("pair_id", sort=True):
        if len(part) != 2 or set(part.label) != {0, 1}:
            continue
        positive = part[part.label == 1].iloc[0]
        negative = part[part.label == 0].iloc[0]
        margin = float(positive.score - negative.score)
        rows.append({
            "pair_id": pair_id, "target_tissue": positive.target_tissue, "mhc_restriction": positive.mhc_restriction,
            "molecule_parent_uniprot_id": positive.molecule_parent_uniprot_id,
            "positive_sample_id": positive.sample_id, "negative_sample_id": negative.sample_id,
            "positive_peptide": positive.peptide_sequence, "negative_peptide": negative.peptide_sequence,
            "positive_score": float(positive.score), "negative_score": float(negative.score), "score_margin": margin,
            "pair_rank_correct": int(margin > 0), "pair_rank_tie": int(margin == 0),
            "positive_seed_score_std": float(positive.seed_score_std), "negative_seed_score_std": float(negative.seed_score_std),
            "mean_seed_score_std": float((positive.seed_score_std + negative.seed_score_std) / 2),
        })
    detail = pd.DataFrame(rows)
    summary = detail.groupby(TASK_KEYS, sort=True).agg(
        paired_rows=("pair_id", "size"), paired_rank_accuracy=("pair_rank_correct", "mean"),
        tied_pair_fraction=("pair_rank_tie", "mean"), mean_pair_margin=("score_margin", "mean"),
        median_pair_margin=("score_margin", "median"), wrong_pair_fraction=("pair_rank_correct", lambda values: float((values == 0).mean())),
        mean_seed_score_std=("mean_seed_score_std", "mean"), wrong_pair_seed_score_std=("mean_seed_score_std", lambda values: float("nan")),
    ).reset_index()
    # The conditional statistic needs the original groups rather than agg's scalar input.
    wrong = detail[detail.pair_rank_correct == 0].groupby(TASK_KEYS, sort=True).mean_seed_score_std.mean().rename("wrong_pair_seed_score_std").reset_index()
    summary = summary.drop(columns=["wrong_pair_seed_score_std"]).merge(wrong, on=TASK_KEYS, how="left")
    return detail, summary


def h2_summary(task_metrics: pd.DataFrame, task_data: pd.DataFrame, pair_summary: pd.DataFrame) -> pd.DataFrame:
    merged = task_metrics.merge(task_data, on=TASK_KEYS, validate="one_to_one").merge(pair_summary, on=TASK_KEYS, validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for h2, part in merged.groupby("mhc_restriction", sort=True):
        rows.append({
            "mhc_restriction": h2, "n_tasks": len(part), "mean_task_auroc": float(part.auroc.mean()),
            "mean_task_auprc": float(part.auprc.mean()), "worst_task_auroc": float(part.auroc.min()),
            "mean_task_paired_rank_accuracy": float(part.paired_rank_accuracy.mean()),
            "mean_task_seed_score_std": float(part.mean_seed_score_std.mean()), "total_train_pairs": int(part.train_pairs.sum()),
            "mean_train_pairs": float(part.train_pairs.mean()), "mean_unique_uniprots": float(part.unique_uniprots.mean()),
        })
    return pd.DataFrame(rows)


def position_enrichment(train: pd.DataFrame) -> pd.DataFrame:
    """Smoothed positive-vs-negative amino-acid log odds, stratified by H2 and position."""
    rows: list[dict[str, Any]] = []
    for h2, group in train.groupby("mhc_restriction", sort=True):
        positive = group[group.label == 1].peptide_sequence.tolist()
        negative = group[group.label == 0].peptide_sequence.tolist()
        for position in range(9):
            pos_counts = {aa: sum(peptide[position] == aa for peptide in positive) for aa in AA}
            neg_counts = {aa: sum(peptide[position] == aa for peptide in negative) for aa in AA}
            for aa in AA:
                # Laplace smoothing keeps finite effects for small Kb/Kd groups.
                pos_rate = (pos_counts[aa] + 1.0) / (len(positive) + len(AA))
                neg_rate = (neg_counts[aa] + 1.0) / (len(negative) + len(AA))
                rows.append({"mhc_restriction": h2, "position_1based": position + 1, "amino_acid": aa,
                             "positive_count": pos_counts[aa], "negative_count": neg_counts[aa],
                             "smoothed_log2_odds_positive_vs_negative": float(np.log2(pos_rate / neg_rate))})
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    train = pd.read_csv(args.train, keep_default_na=False)
    validate_train(train)
    train = train.copy()
    if args.max_tasks:
        selected_tasks = sorted((train.target_tissue + "||" + train.mhc_restriction).unique())[:args.max_tasks]
        train = train[(train.target_tissue + "||" + train.mhc_restriction).isin(selected_tasks)].copy()
    selected_sample_ids = set(train.sample_id)
    member_predictions = pd.concat([pd.read_csv(args.original_seed_predictions), pd.read_csv(args.new_seed_predictions)], ignore_index=True)
    member_predictions = member_predictions[member_predictions.sample_id.isin(selected_sample_ids)].copy()
    member_predictions = validate_member_predictions(member_predictions, train)
    e15_predictions = pd.read_csv(args.e15_predictions)
    e15_predictions = e15_predictions[e15_predictions.sample_id.isin(selected_sample_ids)].copy()
    e15_predictions = validate_e15_predictions(e15_predictions, train)
    member_scores = member_predictions.pivot(index="sample_id", columns="seed", values="score").reindex(columns=ALL_SEEDS)
    if member_scores.isna().any().any():
        raise ValueError("E25 could not construct complete five-seed score matrix.")
    ensemble = e15_predictions.merge(train.drop(columns=["label"]), on=["sample_id", "target_tissue", "mhc_restriction"], how="left", validate="one_to_one")
    ensemble["seed_score_mean"] = member_scores.loc[ensemble.sample_id].mean(axis=1).to_numpy()
    ensemble["seed_score_std"] = member_scores.loc[ensemble.sample_id].std(axis=1, ddof=0).to_numpy()
    if not np.allclose(ensemble.score.to_numpy(), ensemble.seed_score_mean.to_numpy(), atol=1e-6, rtol=1e-6):
        raise AssertionError("Frozen E15 scores do not equal the mean of the five member scores.")
    if not np.allclose(ensemble.prediction_std_across_seeds.to_numpy(), ensemble.seed_score_std.to_numpy(), atol=1e-6, rtol=1e-6):
        raise AssertionError("Frozen E15 score dispersion does not equal the member-score dispersion.")
    pairs, pair_integrity_summary = pair_integrity(train)
    task_data = task_data_audit(train, pairs)
    task_metrics = per_task_metrics(ensemble)
    pair_detail, pair_summary = pair_margin_audit(ensemble)
    task_audit = task_data.merge(task_metrics, on=TASK_KEYS, validate="one_to_one").merge(pair_summary, on=TASK_KEYS, validate="one_to_one")
    task_audit["is_kb_or_kd"] = task_audit.mhc_restriction.isin(["H2-Kb", "H2-Kd"])
    h2_audit = h2_summary(task_metrics, task_data, pair_summary)
    enrichment = position_enrichment(train)
    difficult = task_audit[task_audit.mhc_restriction.isin(["H2-Kb", "H2-Kd"])].sort_values(
        ["auroc", "mean_pair_margin"], ascending=[True, True]
    ).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_dir / "mousePMHC_phase6_e25_pair_integrity.csv", index=False)
    pair_integrity_summary.to_csv(args.output_dir / "mousePMHC_phase6_e25_pair_integrity_summary.csv", index=False)
    task_audit.to_csv(args.output_dir / "mousePMHC_phase6_e25_task_audit.csv", index=False)
    h2_audit.to_csv(args.output_dir / "mousePMHC_phase6_e25_h2_audit.csv", index=False)
    pair_detail.to_csv(args.output_dir / "mousePMHC_phase6_e25_pair_margin_detail.csv", index=False)
    difficult.to_csv(args.output_dir / "mousePMHC_phase6_e25_kb_kd_difficult_tasks.csv", index=False)
    enrichment.to_csv(args.output_dir / "mousePMHC_phase6_e25_h2_position_enrichment.csv", index=False)
    metadata = {
        "experiment_name": EXPERIMENT, "purpose": "diagnostic only; no predictive model is trained or selected",
        "test_data_read": False, "fixed_test_requested": False, "train": str(args.train),
        "e15_predictions": str(args.e15_predictions), "original_seed_predictions": str(args.original_seed_predictions),
        "new_seed_predictions": str(args.new_seed_predictions), "n_rows": int(len(train)), "n_pairs": int(train.pair_id.nunique()),
        "n_tasks": int(train[TASK_KEYS].drop_duplicates().shape[0]), "oof_folds": 3, "oof_split_seed": 20260711,
        "seeds": ALL_SEEDS, "h2_focus": ["H2-Kb", "H2-Kd"], "git_commit": git_commit(),
        "python": sys.version, "platform": platform.platform(),
        "outputs": {
            "pair_integrity": "mousePMHC_phase6_e25_pair_integrity.csv",
            "pair_integrity_summary": "mousePMHC_phase6_e25_pair_integrity_summary.csv",
            "task_audit": "mousePMHC_phase6_e25_task_audit.csv", "h2_audit": "mousePMHC_phase6_e25_h2_audit.csv",
            "pair_margin_detail": "mousePMHC_phase6_e25_pair_margin_detail.csv",
            "kb_kd_difficult_tasks": "mousePMHC_phase6_e25_kb_kd_difficult_tasks.csv",
            "h2_position_enrichment": "mousePMHC_phase6_e25_h2_position_enrichment.csv",
        },
    }
    (args.output_dir / "mousePMHC_phase6_e25_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pair_integrity_summary.to_string(index=False), flush=True)
    print(h2_audit.to_string(index=False), flush=True)
    print("\nKb/Kd difficult tasks:", flush=True)
    print(difficult[[*TASK_KEYS, "train_pairs", "unique_uniprots", "auroc", "auprc", "paired_rank_accuracy", "mean_pair_margin", "mean_seed_score_std"]].to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument("--e15-predictions", type=Path, default=project_path(
        "results/mousePMHC_phase4_e15_five_seed_confirmation/mousePMHC_phase4_e15_oof_predictions.csv"))
    parser.add_argument("--original-seed-predictions", type=Path, default=project_path(
        "results/mousePMHC_phase3_e3b_task_balanced_mmoe_min200_oof/mousePMHC_phase3_e3b_oof_predictions.csv"))
    parser.add_argument("--new-seed-predictions", type=Path, default=project_path(
        "results/mousePMHC_phase4_e15_five_seed_confirmation/mousePMHC_phase4_e15_new_seed_oof_predictions.csv"))
    parser.add_argument("--output-dir", type=Path, default=project_path("results/mousePMHC_phase6_e25_kb_kd_audit"))
    parser.add_argument("--max-tasks", type=int, default=0, help="Diagnostic smoke-run subset; 0 means all 24 tasks.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
