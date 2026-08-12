from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


PAIR_INVARIANTS = [
    "target_tissue",
    "mhc_restriction",
    "molecule_parent_uniprot_id",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def occurrence_distribution(frame: pd.DataFrame, label: int) -> dict[str, int]:
    counts = (
        frame.loc[frame["label"] == label, "presentation_tissue_count"]
        .astype(int)
        .value_counts()
        .sort_index()
    )
    return {str(int(key)): int(value) for key, value in counts.items()}


def auroc(labels: pd.Series, scores: pd.Series) -> float:
    labels = labels.astype(int).reset_index(drop=True)
    scores = scores.astype(float).reset_index(drop=True)
    positive = int(labels.sum())
    negative = int(len(labels) - positive)
    ranks = scores.rank(method="average")
    rank_sum = float(ranks.loc[labels == 1].sum())
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def audit_partition(frame: pd.DataFrame) -> dict[str, object]:
    grouped = frame.groupby("pair_id", sort=False, dropna=False)
    pair_sizes = grouped.size()
    label_sets = grouped["label"].agg(lambda values: tuple(sorted(map(int, values))))
    invariant_counts = grouped[PAIR_INVARIANTS].nunique(dropna=False)
    occurrence_counts = grouped["presentation_tissue_count"].nunique(dropna=False)

    label_counts = frame["label"].astype(int).value_counts().sort_index()
    positive = frame.loc[frame["label"] == 1]
    negative = frame.loc[frame["label"] == 0]
    positive_distribution = occurrence_distribution(frame, 1)
    negative_distribution = occurrence_distribution(frame, 0)
    task_columns = ["target_tissue", "mhc_restriction"]
    task_occurrence_aurocs: list[float] = []
    unequal_task_distributions = 0
    for _, task_frame in frame.groupby(task_columns, dropna=False, sort=False):
        task_positive = occurrence_distribution(task_frame, 1)
        task_negative = occurrence_distribution(task_frame, 0)
        unequal_task_distributions += int(task_positive != task_negative)
        task_occurrence_aurocs.append(
            auroc(task_frame["label"], task_frame["presentation_tissue_count"])
        )

    mismatches = {
        "pairs_not_two_rows": int((pair_sizes != 2).sum()),
        "pairs_without_one_row_per_label": int((label_sets != (0, 1)).sum()),
        "pairs_with_task_or_parent_mismatch": int((invariant_counts != 1).any(axis=1).sum()),
        "pairs_with_unequal_positive_occurrence_count": int((occurrence_counts != 1).sum()),
        "unequal_label_row_counts": int(label_counts.get(0, 0) != label_counts.get(1, 0)),
        "unequal_occurrence_distributions": int(positive_distribution != negative_distribution),
        "unequal_occurrence_sums": int(
            positive["presentation_tissue_count"].astype(int).sum()
            != negative["presentation_tissue_count"].astype(int).sum()
        ),
        "tasks_with_unequal_occurrence_distributions": unequal_task_distributions,
    }
    return {
        "rows": int(len(frame)),
        "pairs": int(frame["pair_id"].nunique()),
        "label_0_rows": int(label_counts.get(0, 0)),
        "label_1_rows": int(label_counts.get(1, 0)),
        "label_0_positive_occurrence_sum": int(
            negative["presentation_tissue_count"].astype(int).sum()
        ),
        "label_1_positive_occurrence_sum": int(
            positive["presentation_tissue_count"].astype(int).sum()
        ),
        "label_0_positive_occurrence_distribution": negative_distribution,
        "label_1_positive_occurrence_distribution": positive_distribution,
        "occurrence_count_only_task_auroc_min": min(task_occurrence_aurocs),
        "occurrence_count_only_task_auroc_mean": sum(task_occurrence_aurocs)
        / len(task_occurrence_aurocs),
        "occurrence_count_only_task_auroc_max": max(task_occurrence_aurocs),
        "mismatches": mismatches,
        "passed": not any(mismatches.values()),
    }


def audit_species(root: Path, species: str) -> dict[str, object]:
    dataset_dir = root / "data" / f"{species}PMHC_occurence_equal_dataset"
    partitions: dict[str, pd.DataFrame] = {}
    files: dict[str, object] = {}
    for split in ("train", "test"):
        path = dataset_dir / f"{species}PMHC_{split}.csv.gz"
        frame = pd.read_csv(path, keep_default_na=False)
        partitions[split] = frame
        files[split] = {"path": str(path.resolve()), "sha256": sha256(path)}

    combined = pd.concat([partitions["train"], partitions["test"]], ignore_index=True)
    task_columns = ["target_tissue", "mhc_restriction"]
    test_pairs_by_task = (
        partitions["test"].groupby(task_columns, dropna=False)["pair_id"].nunique()
    )
    result = {
        "files": files,
        "n_tasks": int(combined.groupby(task_columns, dropna=False).ngroups),
        "n_tissues": int(combined["target_tissue"].nunique(dropna=False)),
        "n_mhc_restrictions": int(combined["mhc_restriction"].nunique(dropna=False)),
        "test_pairs_per_task_min": int(test_pairs_by_task.min()),
        "test_pairs_per_task_max": int(test_pairs_by_task.max()),
        "train": audit_partition(partitions["train"]),
        "test": audit_partition(partitions["test"]),
        "complete": audit_partition(combined),
    }
    result["passed"] = bool(
        result["train"]["passed"]
        and result["test"]["passed"]
        and result["complete"]["passed"]
        and result["test_pairs_per_task_min"] == 50
        and result["test_pairs_per_task_max"] == 50
    )
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "extra3" / "occurrence_balancing_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        "audit": "occurrence balancing for matched positive--pseudo-negative pairs",
        "definition": (
            "Each pair has one row per label, identical tissue/MHC/parent protein, "
            "and identical presentation_tissue_count for its two peptides."
        ),
        "species": {
            species: audit_species(args.root, species) for species in ("human", "mouse")
        },
    }
    report["passed"] = all(item["passed"] for item in report["species"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit("Occurrence-balancing audit failed.")


if __name__ == "__main__":
    main()
