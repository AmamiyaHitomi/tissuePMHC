#!/usr/bin/env python3
"""Run E26: validation/OOF greedy ensemble selection.

This script deliberately separates *selection* from *evaluation*.  It reads
aligned candidate predictions for an ``oof`` split, greedily selects an
ensemble using only those predictions, and then evaluates that fixed ensemble
once against separately supplied ``test`` predictions.

The required long-format input columns are::

    split,candidate,seed,sample_id,target_tissue,mhc_restriction,label,score

``seed`` is optional; omitted seeds are treated as zero.  Candidate prediction
generation is intentionally outside this script: each OOF prediction must be
made by a model that did not train on that sample.  The selector has no random
operations: tied candidates are resolved by candidate name, making its output
independent of Python, NumPy, PyTorch, CUDA, and DataLoader RNG states.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_tissuepmhc_auxiliary_soft_ensemble as e14
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEY_COLUMNS = ["seed", "sample_id", "target_tissue", "mhc_restriction"]
TASK_COLUMNS = ["seed", "target_tissue", "mhc_restriction"]
REQUIRED_COLUMNS = {"split", "candidate", "sample_id", "target_tissue", "mhc_restriction", "label", "score"}
PER_TASK_COLUMNS = [
    "experiment_name", "seed", "model", "selection_source", "n_members", "members",
    "target_tissue", "mhc_restriction", "test_rows", "test_positive", "test_negative",
    *e14.METRICS, "fusion_formula",
]
TRAJECTORY_COLUMNS = [
    "step", "added_candidate", "members", "n_members", "oof_mean_auroc",
    "oof_worst_10_mean_auroc", "oof_mean_auprc", "improvement_over_previous", "selected",
]
MEMBER_COLUMNS = ["step", "candidate", "selection_source"]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def read_predictions(path: Path, expected_split: str) -> pd.DataFrame:
    """Read and strictly validate one split without looking at the other."""
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if "seed" not in frame.columns:
        frame.insert(2, "seed", 0)
    frame = frame.copy()
    frame["split"] = frame["split"].astype(str).str.lower()
    unexpected = sorted(set(frame["split"]) - {expected_split})
    if unexpected:
        raise ValueError(
            f"{path} must contain only split={expected_split!r}; found {unexpected}. "
            "Keep OOF and test predictions in separate files to prevent leakage."
        )
    frame["candidate"] = frame["candidate"].astype(str)
    if (frame["candidate"].str.len() == 0).any():
        raise ValueError(f"{path} contains an empty candidate name.")
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    frame["score"] = pd.to_numeric(frame["score"], errors="raise").astype(float)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{path} labels must be binary.")
    if not np.isfinite(frame["score"].to_numpy()).all():
        raise ValueError(f"{path} contains a non-finite score.")
    duplicated = frame.duplicated(["candidate", *KEY_COLUMNS], keep=False)
    if duplicated.any():
        raise ValueError(f"{path} contains duplicate candidate/sample alignment keys.")
    label_counts = frame.groupby(KEY_COLUMNS, sort=False)["label"].nunique()
    if not (label_counts == 1).all():
        raise ValueError(f"{path} has inconsistent labels between candidates.")
    candidate_sets = frame.groupby("candidate", sort=True).apply(
        lambda group: set(map(tuple, group[KEY_COLUMNS].itertuples(index=False, name=None))),
        include_groups=False,
    )
    reference = next(iter(candidate_sets))
    missing_alignment = [name for name, keys in candidate_sets.items() if keys != reference]
    if missing_alignment:
        raise ValueError(
            f"{path} candidates do not cover identical samples: {missing_alignment}. "
            "Do not silently intersect predictions, because that changes task metrics."
        )
    class_counts = frame.groupby(TASK_COLUMNS, sort=False)["label"].nunique()
    if not (class_counts == 2).all():
        bad = [str(key) for key, value in class_counts.items() if value != 2]
        raise ValueError(f"{path} has single-class task(s), e.g. {bad[:3]}")
    return frame.sort_values(["candidate", *KEY_COLUMNS]).reset_index(drop=True)


def ranks_by_task(frame: pd.DataFrame) -> pd.DataFrame:
    """Make candidates comparable while preserving each task's ordering."""
    ranked = frame.copy()
    ranked["task_rank"] = ranked.groupby(["candidate", *TASK_COLUMNS], sort=False)["score"].rank(
        method="average", pct=True
    )
    return ranked


def aligned_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    ranked = ranks_by_task(frame)
    labels = ranked.drop_duplicates(KEY_COLUMNS)[KEY_COLUMNS + ["label"]].sort_values(KEY_COLUMNS).reset_index(drop=True)
    pivot = ranked.pivot(index=KEY_COLUMNS, columns="candidate", values="task_rank").reindex(labels.set_index(KEY_COLUMNS).index)
    if pivot.isna().any().any():
        raise AssertionError("Prediction alignment failed after validation.")
    return labels, list(pivot.columns), pivot.to_numpy(dtype=np.float64)


def metric_summary(labels: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    rows = []
    working = labels.copy()
    working["score"] = scores
    for _, task in working.groupby(TASK_COLUMNS, sort=True):
        rows.append(base.evaluate(task["label"].to_numpy(dtype=int), task["score"].to_numpy(dtype=float)))
    aurocs = np.asarray([row["auroc"] for row in rows], dtype=float)
    auprcs = np.asarray([row["auprc"] for row in rows], dtype=float)
    n_worst = min(10, len(aurocs))
    return {
        "mean_auroc": float(aurocs.mean()),
        "worst_10_mean_auroc": float(np.sort(aurocs)[:n_worst].mean()),
        "mean_auprc": float(auprcs.mean()),
    }


def better(left: dict[str, float], right: dict[str, float], tolerance: float) -> bool:
    """Lexicographic OOF objective: mean AUROC, tail AUROC, then AUPRC."""
    for field in ("mean_auroc", "worst_10_mean_auroc", "mean_auprc"):
        delta = left[field] - right[field]
        if delta > tolerance:
            return True
        if delta < -tolerance:
            return False
    return False


def greedy_select(
    labels: pd.DataFrame, candidates: list[str], matrix: np.ndarray, max_members: int, min_improvement: float,
) -> tuple[list[str], list[dict[str, object]]]:
    """Choose candidates with replacement, so repetition represents a weight."""
    selected: list[int] = []
    running = np.zeros(matrix.shape[0], dtype=np.float64)
    previous: dict[str, float] | None = None
    trajectory: list[dict[str, object]] = []
    for step in range(1, max_members + 1):
        proposals = []
        denominator = len(selected) + 1
        for index, candidate in enumerate(candidates):
            summary = metric_summary(labels, (running + matrix[:, index]) / denominator)
            proposals.append((candidate, index, summary))
        # The name makes equivalent metric ties deterministic, without consulting random state.
        proposals.sort(key=lambda item: item[0])
        best_candidate, best_index, best_summary = proposals[0]
        for candidate, index, summary in proposals[1:]:
            if better(summary, best_summary, tolerance=1e-12):
                best_candidate, best_index, best_summary = candidate, index, summary
        improvement = best_summary["mean_auroc"] if previous is None else best_summary["mean_auroc"] - previous["mean_auroc"]
        if previous is not None and improvement < min_improvement:
            break
        selected.append(best_index)
        running += matrix[:, best_index]
        previous = best_summary
        members = [candidates[index] for index in selected]
        trajectory.append({
            "step": step, "added_candidate": best_candidate, "members": ",".join(members),
            "n_members": len(members), "oof_mean_auroc": best_summary["mean_auroc"],
            "oof_worst_10_mean_auroc": best_summary["worst_10_mean_auroc"],
            "oof_mean_auprc": best_summary["mean_auprc"], "improvement_over_previous": improvement,
            "selected": True,
        })
    if not selected:
        raise RuntimeError("No ensemble member met min_improvement; use a non-positive threshold for the first member.")
    return [candidates[index] for index in selected], trajectory


def test_rows(labels: pd.DataFrame, candidates: list[str], matrix: np.ndarray, members: list[str]) -> list[dict[str, object]]:
    index = {candidate: position for position, candidate in enumerate(candidates)}
    scores = np.mean(np.column_stack([matrix[:, index[member]] for member in members]), axis=1)
    output = labels.copy()
    output["score"] = scores
    rows: list[dict[str, object]] = []
    for (seed, tissue, hla), task in output.groupby(TASK_COLUMNS, sort=True):
        y_true = task["label"].to_numpy(dtype=int)
        rows.append({
            "experiment_name": "E26_greedy_ensemble_selection", "seed": int(seed),
            "model": "e26_oof_greedy_task_rank_average", "selection_source": "OOF_only",
            "n_members": len(members), "members": ",".join(members),
            "target_tissue": tissue, "mhc_restriction": hla, "test_rows": len(task),
            "test_positive": int(y_true.sum()), "test_negative": int(len(task) - y_true.sum()),
            **base.evaluate(y_true, task["score"].to_numpy(dtype=float)),
            "fusion_formula": "mean(task_percentile_rank(candidate_score) for selected members)",
        })
    return rows


def run(args: argparse.Namespace) -> None:
    if args.max_members < 1:
        raise ValueError("max_members must be positive.")
    if args.min_improvement < 0:
        raise ValueError("min_improvement must be non-negative.")
    # Read OOF first and perform all selection before test labels are ever loaded.
    oof = read_predictions(args.oof_predictions, "oof")
    oof_labels, oof_candidates, oof_matrix = aligned_matrix(oof)
    # E20 showed that SWA degraded the paired branches.  It can still be
    # deliberately ablated by passing --candidates ..., but it is never added
    # accidentally when the complete prediction file is used.
    allowed = (
        sorted(args.candidates)
        if args.candidates
        else [candidate for candidate in oof_candidates if "swa" not in candidate.lower()]
    )
    if not allowed:
        raise ValueError("No non-SWA candidates remain; pass an explicit --candidates allow-list.")
    absent = sorted(set(allowed) - set(oof_candidates))
    if absent:
        raise ValueError(f"Requested candidate(s) absent from OOF predictions: {absent}")
    oof_positions = [oof_candidates.index(candidate) for candidate in allowed]
    members, trajectory = greedy_select(
        oof_labels, allowed, oof_matrix[:, oof_positions], args.max_members, args.min_improvement,
    )
    print(f"OOF-selected members: {members}", flush=True)
    print(f"OOF final mean task AUROC: {trajectory[-1]['oof_mean_auroc']:.6f}", flush=True)

    test = read_predictions(args.test_predictions, "test")
    test_labels, test_candidates, test_matrix = aligned_matrix(test)
    if set(test_candidates) != set(allowed):
        raise ValueError("Test candidates must exactly match the OOF candidate set used for selection.")
    test_positions = [test_candidates.index(candidate) for candidate in allowed]
    rows = test_rows(test_labels, allowed, test_matrix[:, test_positions], members)
    summary = base.summarize_results(rows)
    stability = base.summarize_seed_stability(summary)

    base.write_csv(args.per_task_output, PER_TASK_COLUMNS, rows)
    base.write_csv(args.summary_output, base.SUMMARY_COLUMNS, summary)
    base.write_csv(args.stability_output, base.STABILITY_COLUMNS, stability)
    base.write_csv(args.trajectory_output, TRAJECTORY_COLUMNS, trajectory)
    base.write_csv(args.members_output, MEMBER_COLUMNS, [
        {"step": step, "candidate": candidate, "selection_source": "OOF_only"}
        for step, candidate in enumerate(members, start=1)
    ])
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps({
        "experiment_name": "E26_greedy_ensemble_selection",
        "oof_predictions": str(args.oof_predictions), "test_predictions": str(args.test_predictions),
        "candidate_pool": allowed, "selected_members": members,
        "max_members": args.max_members, "min_improvement": args.min_improvement,
        "selection_metric": "lexicographic(mean_task_auroc, worst_10_mean_task_auroc, mean_task_auprc)",
        "fusion": "mean task percentile ranks, with replacement",
        "selection_policy": "OOF only; test labels are loaded only after selection is finalized",
        "rng_policy": "No random operations; candidate-name tie break; no Python/NumPy/PyTorch/CUDA RNG state is read or advanced",
        "swa_policy": "SWA candidates are excluded by default; include only by explicit --candidates request",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    for path in [args.per_task_output, args.summary_output, args.stability_output, args.trajectory_output, args.members_output, args.metadata_output]:
        print(f"wrote: {path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", default=None, help="Optional allow-list. Do not include SWA unless intentionally ablated.")
    parser.add_argument("--max-members", type=int, default=20)
    parser.add_argument("--min-improvement", type=float, default=1e-4)
    parser.add_argument("--per-task-output", type=Path, default=project_path("results/tissuePMHC_e26_greedy_ensemble_selection/per_task_metrics.csv"))
    parser.add_argument("--summary-output", type=Path, default=project_path("results/tissuePMHC_e26_greedy_ensemble_selection/summary_metrics.csv"))
    parser.add_argument("--stability-output", type=Path, default=project_path("results/tissuePMHC_e26_greedy_ensemble_selection/stability_metrics.csv"))
    parser.add_argument("--trajectory-output", type=Path, default=project_path("results/tissuePMHC_e26_greedy_ensemble_selection/oof_selection_trajectory.csv"))
    parser.add_argument("--members-output", type=Path, default=project_path("results/tissuePMHC_e26_greedy_ensemble_selection/selected_members.csv"))
    parser.add_argument("--metadata-output", type=Path, default=project_path("results/tissuePMHC_e26_greedy_ensemble_selection/metadata.json"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
