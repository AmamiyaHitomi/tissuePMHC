#!/usr/bin/env python3
"""Audit the fixed mousePMHC Phase 3 benchmark before model training.

This script never trains a model and never uses labels to tune a decision. It
checks construction invariants, quantifies train/test overlap, and records
which parts of the human tissuePMHC protocol are exactly aligned.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "dataset", "split", "sample_id", "pair_id", "label", "target_tissue",
    "mhc_restriction", "peptide_sequence", "molecule_parent_uniprot_id",
    "source_molecule", "source_molecule_uniprot_id", "molecule_parent",
    "reported_tissues_same_hla_uniprot",
]
STANDARD_9MER = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{9}$")


def read_frame(path: Path, expected_split: str) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if set(frame["split"]) != {expected_split}:
        raise ValueError(f"{path} has unexpected split values: {sorted(set(frame['split']))}")
    frame["label"] = frame["label"].astype(int)
    return frame


def pair_errors(frame: pd.DataFrame) -> dict[str, int]:
    invalid_size = invalid_labels = task_mismatch = protein_mismatch = 0
    for _, pair in frame.groupby("pair_id", sort=False):
        if len(pair) != 2:
            invalid_size += 1
        if sorted(pair["label"].tolist()) != [0, 1]:
            invalid_labels += 1
        if pair[["target_tissue", "mhc_restriction"]].drop_duplicates().shape[0] != 1:
            task_mismatch += 1
        if pair["molecule_parent_uniprot_id"].nunique() != 1:
            protein_mismatch += 1
    return {
        "invalid_pair_size": invalid_size,
        "invalid_pair_labels": invalid_labels,
        "pair_task_mismatch": task_mismatch,
        "pair_protein_mismatch": protein_mismatch,
    }


def tissue_annotation_errors(frame: pd.DataFrame) -> dict[str, int]:
    positive_missing = negative_present = 0
    for row in frame.itertuples(index=False):
        reported = {value.strip() for value in row.reported_tissues_same_hla_uniprot.split(";") if value.strip()}
        if row.label == 1 and row.target_tissue not in reported:
            positive_missing += 1
        if row.label == 0 and row.target_tissue in reported:
            negative_present += 1
    return {
        "positive_missing_target_tissue_annotation": positive_missing,
        "negative_reported_in_target_tissue": negative_present,
    }


def task_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tasks = sorted(set(zip(train.target_tissue, train.mhc_restriction)) | set(zip(test.target_tissue, test.mhc_restriction)))
    for tissue, mhc in tasks:
        tr = train[(train.target_tissue == tissue) & (train.mhc_restriction == mhc)]
        te = test[(test.target_tissue == tissue) & (test.mhc_restriction == mhc)]
        rows.append({
            "target_tissue": tissue, "mhc_restriction": mhc,
            "train_rows": len(tr), "train_pairs": tr.pair_id.nunique(),
            "train_positive": int((tr.label == 1).sum()), "train_negative": int((tr.label == 0).sum()),
            "test_rows": len(te), "test_pairs": te.pair_id.nunique(),
            "test_positive": int((te.label == 1).sum()), "test_negative": int((te.label == 0).sum()),
            "unique_train_peptides": tr.peptide_sequence.nunique(), "unique_test_peptides": te.peptide_sequence.nunique(),
            "unique_train_proteins": tr.molecule_parent_uniprot_id.nunique(), "unique_test_proteins": te.molecule_parent_uniprot_id.nunique(),
        })
    return pd.DataFrame(rows)


def overlap_report(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    def overlap(column: str) -> int:
        return len(set(train[column]) & set(test[column]))

    train_tasks_by_peptide = (
        train.groupby("peptide_sequence", sort=False)
        .apply(lambda frame: set(zip(frame.target_tissue, frame.mhc_restriction)), include_groups=False)
        .to_dict()
    )
    train_tasks_by_protein = (
        train.groupby("molecule_parent_uniprot_id", sort=False)
        .apply(lambda frame: set(zip(frame.target_tissue, frame.mhc_restriction)), include_groups=False)
        .to_dict()
    )

    def exclusive_test_row_categories(
        column: str,
        fitting_tasks: dict[str, set[tuple[str, str]]],
    ) -> dict[str, int]:
        counts = {"current_task": 0, "other_task_only": 0, "absent_from_all_training": 0}
        for row in test.itertuples(index=False):
            entity = getattr(row, column)
            current_task = (row.target_tissue, row.mhc_restriction)
            observed_tasks = fitting_tasks.get(entity, set())
            if current_task in observed_tasks:
                counts["current_task"] += 1
            elif observed_tasks:
                counts["other_task_only"] += 1
            else:
                counts["absent_from_all_training"] += 1
        return counts

    train_task_peptides = set(zip(train.target_tissue, train.mhc_restriction, train.peptide_sequence))
    test_task_peptides = set(zip(test.target_tissue, test.mhc_restriction, test.peptide_sequence))
    train_task_proteins = set(zip(train.target_tissue, train.mhc_restriction, train.molecule_parent_uniprot_id))
    test_task_proteins = set(zip(test.target_tissue, test.mhc_restriction, test.molecule_parent_uniprot_id))
    return {
        "sample_id_overlap": overlap("sample_id"),
        "pair_id_overlap": overlap("pair_id"),
        "peptide_overlap_global": overlap("peptide_sequence"),
        "protein_overlap_global": overlap("molecule_parent_uniprot_id"),
        "task_peptide_overlap": len(train_task_peptides & test_task_peptides),
        "task_protein_overlap": len(train_task_proteins & test_task_proteins),
        "peptide_test_rows_by_scope": exclusive_test_row_categories(
            "peptide_sequence", train_tasks_by_peptide
        ),
        "protein_test_rows_by_scope": exclusive_test_row_categories(
            "molecule_parent_uniprot_id", train_tasks_by_protein
        ),
    }


def protocol_alignment(args: argparse.Namespace, mouse_train: pd.DataFrame) -> dict[str, Any]:
    human_train = pd.read_csv(args.human_train, nrows=1, keep_default_na=False)
    human_meta = json.loads(args.human_metadata.read_text(encoding="utf-8"))
    mouse_meta = json.loads(args.mouse_metadata.read_text(encoding="utf-8"))
    return {
        "same_output_columns": list(human_train.columns) == list(mouse_train.columns),
        "same_peptide_length_filter": "peptide length = 9" in human_meta["filters"] and "peptide length = 9" in mouse_meta["filters"],
        "same_unmodified_filter": "unmodified standard amino-acid peptide" in human_meta["filters"] and "unmodified standard amino-acid peptide" in mouse_meta["filters"],
        "same_positive_measurement_filter": "positive IEDB qualitative measurement" in human_meta["filters"] and "positive IEDB qualitative measurement" in mouse_meta["filters"],
        "mouse_min_pairs_matches_config": mouse_meta["min_pairs_filter"] == f"> {args.expected_min_pairs} pairs per tissue-H2 group",
        "human_min_pairs_filter": human_meta["min_pairs_filter"],
        "mouse_min_pairs_filter": mouse_meta["min_pairs_filter"],
        "same_test_pairs_per_task": human_meta.get("test_pairs_per_tissue_hla") == mouse_meta.get("test_pairs_per_tissue_mhc"),
        "same_random_seed": human_meta["random_seed"] == mouse_meta["random_seed"],
        "intended_species_difference": {"human": "Homo sapiens / HLA-I", "mouse": "Mus musculus / H2-I"},
    }


def run(args: argparse.Namespace) -> None:
    train = read_frame(args.train, "train")
    test = read_frame(args.test, "test")
    combined = pd.concat([train, test], ignore_index=True)
    tasks = task_table(train, test)
    overlaps = overlap_report(train, test)
    errors = {
        **{f"train_{key}": value for key, value in pair_errors(train).items()},
        **{f"test_{key}": value for key, value in pair_errors(test).items()},
        **tissue_annotation_errors(combined),
        "duplicate_sample_ids": int(combined.sample_id.duplicated().sum()),
        "non_mouse_dataset_rows": int((combined.dataset != "mousePMHC").sum()),
        "non_h2_rows": int((~combined.mhc_restriction.str.startswith("H2-")).sum()),
        "invalid_9mer_rows": int((~combined.peptide_sequence.map(lambda value: bool(STANDARD_9MER.fullmatch(value)))).sum()),
        "unbalanced_tasks": int(((tasks.train_positive != tasks.train_negative) | (tasks.test_positive != tasks.test_negative)).sum()),
        "tasks_without_100_test_pairs": int((tasks.test_pairs != 100).sum()),
    }
    errors["pair_id_overlap"] = overlaps["pair_id_overlap"]
    errors["task_peptide_overlap"] = overlaps["task_peptide_overlap"]
    alignment = protocol_alignment(args, train)
    passed = not any(errors.values()) and all(value for value in alignment.values() if isinstance(value, bool))
    report = {
        "audit_name": "mousePMHC_phase3_data_audit",
        "passed": passed,
        "policy": "Random pair-disjoint split matching the human standard-split protocol; protein overlap is measured, not prohibited.",
        "train": {"rows": len(train), "pairs": train.pair_id.nunique()},
        "test": {"rows": len(test), "pairs": test.pair_id.nunique()},
        "n_tasks": len(tasks), "errors": errors, "overlap": overlaps,
        "human_protocol_alignment": alignment,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(args.output_dir / "mousePMHC_phase3_task_inventory.csv", index=False)
    (args.output_dir / "mousePMHC_phase3_data_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_error and not passed:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument("--test", type=Path, default=Path("data/mousePMHC/mousePMHC_test.csv.gz"))
    parser.add_argument("--mouse-metadata", type=Path, default=Path("data/mousePMHC/mousePMHC_metadata.json"))
    parser.add_argument("--human-train", type=Path, default=Path("data/tissuePMHC/tissuePMHC_train.csv.gz"))
    parser.add_argument("--human-metadata", type=Path, default=Path("data/tissuePMHC/tissuePMHC_metadata.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/mousePMHC_phase3_data_audit"))
    parser.add_argument("--expected-min-pairs", type=int, default=200,
                        help="Required mouse task-inclusion threshold; independent from the human benchmark threshold.")
    parser.add_argument("--fail-on-error", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
