#!/usr/bin/env python3
"""Create the Phase 3 development and locked protein-disjoint confirmation split.

The input is the original Phase 2 training CSV.  ``molecule_parent_uniprot_id``
is the indivisible group: the script assigns every parent protein to exactly one
partition, while greedily balancing the 44 tissue-HLA task/label cells near the
requested confirmation fraction.  It writes:

* development CSV (labels retained for Phase 3 OOF training);
* confirmation features CSV (``label`` removed);
* confirmation labels CSV (evaluation-only key/label table);
* group assignment, task-count audit, and manifest with SHA-256 checksums.

The split seed and all selected group assignments are deterministic.  Do not
regenerate the files with another seed after model development has started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEYS = ["sample_id", "target_tissue", "mhc_restriction"]
REQUIRED_COLUMNS = [*KEYS, "label", "pair_id", "molecule_parent_uniprot_id"]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_name(frame: pd.DataFrame) -> pd.Series:
    return frame["target_tissue"].astype(str) + "__" + frame["mhc_restriction"].astype(str)


def build_group_matrix(frame: pd.DataFrame) -> tuple[list[str], list[tuple[str, int]], np.ndarray, np.ndarray]:
    """Return parent IDs, task/label cells, and one count vector per parent."""
    cells = sorted({(task, int(label)) for task, label in zip(task_name(frame), frame["label"])})
    cell_to_index = {cell: index for index, cell in enumerate(cells)}
    groups = sorted(frame["molecule_parent_uniprot_id"].astype(str).unique())
    group_to_index = {group: index for index, group in enumerate(groups)}
    matrix = np.zeros((len(groups), len(cells)), dtype=np.float64)
    working = frame.assign(_task=task_name(frame), _parent=frame["molecule_parent_uniprot_id"].astype(str))
    for (parent, task, label), count in working.groupby(["_parent", "_task", "label"], sort=True).size().items():
        matrix[group_to_index[parent], cell_to_index[(task, int(label))]] = float(count)
    return groups, cells, matrix, matrix.sum(axis=0)


def split_objective(
    confirmation_counts: np.ndarray, target_counts: np.ndarray, total_counts: np.ndarray,
    confirmation_groups: int, n_groups: int, confirmation_fraction: float,
) -> float:
    # Cell-level balance dominates.  The weak group-count term keeps the result
    # close to the requested proportion when parent proteins have uneven sizes.
    scale = np.maximum(target_counts, 1.0)
    cell_cost = float(np.square((confirmation_counts - target_counts) / scale).sum())
    row_cost = float(((confirmation_counts.sum() - target_counts.sum()) / max(target_counts.sum(), 1.0)) ** 2)
    group_target = confirmation_fraction * n_groups
    group_cost = float(((confirmation_groups - group_target) / max(n_groups, 1)) ** 2)
    del total_counts  # Explicitly document that total counts are represented by target_counts.
    return cell_cost + 0.05 * row_cost + 0.01 * group_cost


def propose_assignment(
    matrix: np.ndarray, totals: np.ndarray, confirmation_fraction: float, seed: int,
) -> tuple[np.ndarray, float]:
    """Deterministically greedily assign parents, then make local flip improvements."""
    n_groups = len(matrix)
    rng = np.random.default_rng(seed)
    tie_breaker = rng.random(n_groups)
    order = sorted(
        range(n_groups),
        key=lambda index: (-float(matrix[index].sum()), -int((matrix[index] > 0).sum()), float(tie_breaker[index])),
    )
    target = totals * confirmation_fraction
    assignment = np.zeros(n_groups, dtype=bool)  # True means confirmation.
    confirmation_counts = np.zeros_like(totals, dtype=np.float64)
    confirmation_groups = 0
    for index in order:
        stay_cost = split_objective(
            confirmation_counts, target, totals, confirmation_groups, n_groups, confirmation_fraction,
        )
        move_cost = split_objective(
            confirmation_counts + matrix[index], target, totals, confirmation_groups + 1, n_groups, confirmation_fraction,
        )
        if move_cost < stay_cost:
            assignment[index] = True
            confirmation_counts += matrix[index]
            confirmation_groups += 1
    # A small deterministic local search repairs greedy decisions without
    # introducing an uncontrolled hyperparameter search.
    for _ in range(3):
        changed = False
        for index in rng.permutation(n_groups):
            current_cost = split_objective(
                confirmation_counts, target, totals, confirmation_groups, n_groups, confirmation_fraction,
            )
            if assignment[index]:
                candidate_counts, candidate_groups = confirmation_counts - matrix[index], confirmation_groups - 1
            else:
                candidate_counts, candidate_groups = confirmation_counts + matrix[index], confirmation_groups + 1
            candidate_cost = split_objective(
                candidate_counts, target, totals, candidate_groups, n_groups, confirmation_fraction,
            )
            if candidate_cost + 1e-12 < current_cost:
                assignment[index] = not assignment[index]
                confirmation_counts, confirmation_groups = candidate_counts, candidate_groups
                changed = True
        if not changed:
            break
    objective = split_objective(confirmation_counts, target, totals, confirmation_groups, n_groups, confirmation_fraction)
    return assignment, objective


def task_counts(frame: pd.DataFrame, partition: str) -> pd.DataFrame:
    working = frame.assign(partition=partition)
    summary = (
        working.groupby(["partition", "target_tissue", "mhc_restriction"], sort=True)
        .agg(
            rows=("label", "size"),
            positives=("label", "sum"),
            parent_uniprots=("molecule_parent_uniprot_id", "nunique"),
            pairs=("pair_id", "nunique"),
            peptides=("peptide_sequence", "nunique"),
        )
        .reset_index()
    )
    summary["negatives"] = summary["rows"] - summary["positives"]
    return summary[[
        "partition", "target_tissue", "mhc_restriction", "rows", "positives", "negatives",
        "parent_uniprots", "pairs", "peptides",
    ]]


def validate_split(development: pd.DataFrame, confirmation: pd.DataFrame, expected_tasks: set[tuple[str, str]]) -> None:
    parent_overlap = set(development["molecule_parent_uniprot_id"].astype(str)) & set(confirmation["molecule_parent_uniprot_id"].astype(str))
    if parent_overlap:
        raise AssertionError(f"Parent UniProt leakage: {next(iter(parent_overlap))}")
    pair_overlap = set(development["pair_id"].astype(str)) & set(confirmation["pair_id"].astype(str))
    if pair_overlap:
        raise AssertionError(f"pair_id leakage: {next(iter(pair_overlap))}")
    for name, frame in (("development", development), ("confirmation", confirmation)):
        observed_tasks = set(zip(frame["target_tissue"], frame["mhc_restriction"]))
        missing = expected_tasks - observed_tasks
        if missing:
            raise ValueError(f"{name} is missing task(s): {sorted(missing)[:5]}")
        label_counts = frame.groupby(["target_tissue", "mhc_restriction"], sort=True)["label"].nunique()
        invalid = label_counts[label_counts != 2]
        if not invalid.empty:
            raise ValueError(f"{name} task(s) without both labels: {invalid.index.tolist()[:5]}")


def choose_assignment(
    frame: pd.DataFrame, fraction: float, split_seed: int, attempts: int,
) -> tuple[list[str], np.ndarray, float, int, list[tuple[str, int]], np.ndarray]:
    groups, cells, matrix, totals = build_group_matrix(frame)
    if len(groups) < 2:
        raise ValueError("At least two parent UniProt groups are required.")
    if (totals < 2).any():
        unsupported = [cells[index] for index, total in enumerate(totals) if total < 2]
        raise ValueError(f"Cannot split task/label cells represented by fewer than two rows: {unsupported[:5]}")
    expected_tasks = set(zip(frame["target_tissue"], frame["mhc_restriction"]))
    best: tuple[np.ndarray, float, int] | None = None
    for attempt in range(attempts):
        assignment, objective = propose_assignment(matrix, totals, fraction, split_seed + attempt)
        confirmation_groups = {groups[index] for index, selected in enumerate(assignment) if selected}
        confirmation = frame[frame["molecule_parent_uniprot_id"].astype(str).isin(confirmation_groups)]
        development = frame[~frame["molecule_parent_uniprot_id"].astype(str).isin(confirmation_groups)]
        try:
            validate_split(development, confirmation, expected_tasks)
        except (AssertionError, ValueError):
            continue
        if best is None or objective < best[1]:
            best = assignment, objective, attempt
    if best is None:
        raise RuntimeError(
            f"No valid protein-disjoint split was found in {attempts} deterministic attempts. "
            "Increase --attempts or inspect task/parent support; do not fall back to random row splitting."
        )
    assignment, objective, attempt = best
    return groups, assignment, objective, attempt, cells, totals


def run(args: argparse.Namespace) -> None:
    if not 0.0 < args.confirmation_fraction < 1.0:
        raise ValueError("--confirmation-fraction must lie strictly between 0 and 1.")
    if args.attempts < 1:
        raise ValueError("--attempts must be positive.")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    source = pd.read_csv(args.input)
    missing = [column for column in REQUIRED_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError(f"Input is missing required column(s): {missing}")
    if source["molecule_parent_uniprot_id"].isna().any() or source["pair_id"].isna().any():
        raise ValueError("Parent UniProt and pair_id must be non-null for a protein-disjoint split.")
    if source[KEYS].duplicated().any():
        raise ValueError("Input prediction keys are not unique.")
    if set(source["label"].unique()) - {0, 1}:
        raise ValueError("Input labels must be binary 0/1.")

    groups, assignment, objective, chosen_attempt, cells, totals = choose_assignment(
        source, args.confirmation_fraction, args.split_seed, args.attempts,
    )
    confirmation_parents = {groups[index] for index, selected in enumerate(assignment) if selected}
    confirmation = source[source["molecule_parent_uniprot_id"].astype(str).isin(confirmation_parents)].copy()
    development = source[~source["molecule_parent_uniprot_id"].astype(str).isin(confirmation_parents)].copy()
    expected_tasks = set(zip(source["target_tissue"], source["mhc_restriction"]))
    validate_split(development, confirmation, expected_tasks)

    assignments = pd.DataFrame({
        "molecule_parent_uniprot_id": groups,
        "partition": np.where(assignment, "confirmation", "development"),
    })
    parent_sizes = source.groupby(source["molecule_parent_uniprot_id"].astype(str), sort=True).agg(
        rows=("label", "size"), positives=("label", "sum"), pairs=("pair_id", "nunique"),
    ).reset_index(names="molecule_parent_uniprot_id")
    assignments = assignments.merge(parent_sizes, on="molecule_parent_uniprot_id", how="left", validate="one_to_one")
    counts = pd.concat([task_counts(development, "development"), task_counts(confirmation, "confirmation")], ignore_index=True)
    confirmation_features = confirmation.drop(columns=["label"])
    confirmation_labels = confirmation[KEYS + ["label"]].copy()

    outputs = [
        args.development_output, args.confirmation_features_output, args.confirmation_labels_output,
        args.assignments_output, args.task_counts_output, args.manifest_output,
    ]
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps({
            "dry_run": True, "chosen_attempt": chosen_attempt, "objective": objective,
            "development_rows": len(development), "confirmation_rows": len(confirmation),
            "development_parents": int((~assignment).sum()), "confirmation_parents": int(assignment.sum()),
            "confirmation_row_fraction": len(confirmation) / len(source),
            "confirmation_parent_fraction": float(assignment.mean()),
        }, indent=2), flush=True)
        return

    development.to_csv(args.development_output, index=False, compression="gzip")
    confirmation_features.to_csv(args.confirmation_features_output, index=False, compression="gzip")
    confirmation_labels.to_csv(args.confirmation_labels_output, index=False, compression="gzip")
    assignments.to_csv(args.assignments_output, index=False)
    counts.to_csv(args.task_counts_output, index=False)
    manifest: dict[str, Any] = {
        "experiment": "Phase3_protein_disjoint_split",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "split_seed": args.split_seed,
        "confirmation_fraction_requested": args.confirmation_fraction,
        "attempts_considered": args.attempts,
        "chosen_attempt_offset": chosen_attempt,
        "objective": objective,
        "group_column": "molecule_parent_uniprot_id",
        "algorithm": "deterministic_greedy_parent_group_assignment_with_local_flips",
        "counts": {
            "source_rows": len(source), "development_rows": len(development), "confirmation_rows": len(confirmation),
            "source_parents": len(groups), "development_parents": int((~assignment).sum()),
            "confirmation_parents": int(assignment.sum()),
            "confirmation_row_fraction": len(confirmation) / len(source),
            "confirmation_parent_fraction": float(assignment.mean()),
            "tasks": len(expected_tasks), "task_label_cells": len(cells),
        },
        "outputs": {
            "development": {"path": str(args.development_output), "sha256": sha256_file(args.development_output)},
            "confirmation_features": {"path": str(args.confirmation_features_output), "sha256": sha256_file(args.confirmation_features_output)},
            "confirmation_labels": {"path": str(args.confirmation_labels_output), "sha256": sha256_file(args.confirmation_labels_output)},
            "group_assignments": {"path": str(args.assignments_output), "sha256": sha256_file(args.assignments_output)},
            "task_counts": {"path": str(args.task_counts_output), "sha256": sha256_file(args.task_counts_output)},
        },
    }
    args.manifest_output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote: {args.development_output}", flush=True)
    print(f"wrote: {args.confirmation_features_output}", flush=True)
    print(f"wrote: {args.confirmation_labels_output}", flush=True)
    print(f"wrote: {args.assignments_output}", flush=True)
    print(f"wrote: {args.task_counts_output}", flush=True)
    print(f"wrote: {args.manifest_output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = project_path("phase3/splits")
    parser.add_argument("--input", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz"))
    parser.add_argument("--split-seed", type=int, default=20260713)
    parser.add_argument("--confirmation-fraction", type=float, default=0.20)
    parser.add_argument("--attempts", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--development-output", type=Path, default=root / "phase3_development.csv.gz")
    parser.add_argument("--confirmation-features-output", type=Path, default=root / "phase3_confirmation_features.csv.gz")
    parser.add_argument("--confirmation-labels-output", type=Path, default=root / "phase3_confirmation_labels.csv.gz")
    parser.add_argument("--assignments-output", type=Path, default=root / "parent_group_assignments.csv")
    parser.add_argument("--task-counts-output", type=Path, default=root / "task_counts.csv")
    parser.add_argument("--manifest-output", type=Path, default=root / "split_manifest.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
