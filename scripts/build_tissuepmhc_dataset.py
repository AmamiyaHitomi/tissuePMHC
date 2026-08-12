#!/usr/bin/env python3
"""Build the tissuePMHC benchmark dataset from paired tissue-specificity samples.

Roadmap role: data preparation line.
This produces the standard train/test split consumed by all downstream model
experiments.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


OUTPUT_COLUMNS = [
    "dataset",
    "split",
    "sample_id",
    "pair_id",
    "label",
    "target_tissue",
    "mhc_restriction",
    "peptide_sequence",
    "molecule_parent_uniprot_id",
    "source_molecule",
    "source_molecule_uniprot_id",
    "molecule_parent",
    "reported_tissues_same_hla_uniprot",
]

SUMMARY_COLUMNS = [
    "target_tissue",
    "mhc_restriction",
    "total_pairs_before_filter",
    "train_pairs",
    "test_pairs",
    "train_rows",
    "test_rows",
    "train_positive_rows",
    "train_negative_rows",
    "test_positive_rows",
    "test_negative_rows",
    "n_uniprot_ids",
]


def open_text_input(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def open_text_output(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", newline="", encoding="utf-8")
    return path.open("w", newline="", encoding="utf-8")


def read_pairs(path: Path) -> dict[str, list[dict[str, str]]]:
    pairs: dict[str, list[dict[str, str]]] = defaultdict(list)
    with open_text_input(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs[row["pair_id"]].append(row)
    return pairs


def validate_input_pairs(pairs: dict[str, list[dict[str, str]]]) -> None:
    invalid_pairs = 0
    mismatched_group = 0
    mismatched_uniprot = 0
    for rows in pairs.values():
        if len(rows) != 2 or sorted(row["label"] for row in rows) != ["0", "1"]:
            invalid_pairs += 1
            continue
        tissues = {row["target_tissue"] for row in rows}
        hlas = {row["mhc_restriction"] for row in rows}
        uniprots = {row["molecule_parent_uniprot_id"] for row in rows}
        if len(tissues) != 1 or len(hlas) != 1:
            mismatched_group += 1
        if len(uniprots) != 1:
            mismatched_uniprot += 1

    if invalid_pairs or mismatched_group or mismatched_uniprot:
        raise ValueError(
            "Invalid input pairs: "
            f"invalid_pairs={invalid_pairs}, "
            f"mismatched_group={mismatched_group}, "
            f"mismatched_uniprot={mismatched_uniprot}"
        )


def add_output_fields(row: dict[str, str], split: str, sample_index: int, dataset_name: str) -> dict[str, str]:
    output_row = {column: row.get(column, "") for column in OUTPUT_COLUMNS}
    output_row["dataset"] = dataset_name
    output_row["split"] = split
    output_row["sample_id"] = f"{dataset_name}_{split}_{sample_index:012d}"
    return output_row


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with open_text_output(path) as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def count_labels(rows: list[dict[str, str]]) -> Counter:
    return Counter(row["label"] for row in rows)


def validate_output(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], summary_rows: list[dict[str, object]]) -> None:
    train_pair_ids = {row["pair_id"] for row in train_rows}
    test_pair_ids = {row["pair_id"] for row in test_rows}
    overlap = train_pair_ids & test_pair_ids
    if overlap:
        raise ValueError(f"Train/test pair_id overlap: {len(overlap)}")

    for split_name, rows in [("train", train_rows), ("test", test_rows)]:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[row["pair_id"]].append(row)
        invalid = sum(1 for pair_rows in grouped.values() if len(pair_rows) != 2 or sorted(r["label"] for r in pair_rows) != ["0", "1"])
        if invalid:
            raise ValueError(f"Invalid {split_name} pairs: {invalid}")

    test_counts_by_group = Counter((row["target_tissue"], row["mhc_restriction"], row["label"]) for row in test_rows)
    train_counts_by_group = Counter((row["target_tissue"], row["mhc_restriction"], row["label"]) for row in train_rows)
    for row in summary_rows:
        group = (str(row["target_tissue"]), str(row["mhc_restriction"]))
        if row["test_pairs"] != 100:
            raise ValueError(f"Unexpected test_pairs for {group}: {row['test_pairs']}")
        if test_counts_by_group[(group[0], group[1], "0")] != 100 or test_counts_by_group[(group[0], group[1], "1")] != 100:
            raise ValueError(f"Unbalanced test labels for {group}")
        if train_counts_by_group[(group[0], group[1], "0")] != row["train_pairs"] or train_counts_by_group[(group[0], group[1], "1")] != row["train_pairs"]:
            raise ValueError(f"Unbalanced train labels for {group}")


def build_dataset(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    pairs = read_pairs(args.input)
    validate_input_pairs(pairs)

    pair_ids_by_group: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pair_id, rows in pairs.items():
        group = (rows[0]["target_tissue"], rows[0]["mhc_restriction"])
        pair_ids_by_group[group].append(pair_id)

    train_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, object]] = []
    train_sample_index = 0
    test_sample_index = 0

    for group in sorted(pair_ids_by_group):
        pair_ids = sorted(pair_ids_by_group[group])
        total_pairs = len(pair_ids)
        if total_pairs <= args.min_pairs:
            continue

        test_pair_ids = set(rng.sample(pair_ids, args.test_pairs))
        train_pair_ids = [pair_id for pair_id in pair_ids if pair_id not in test_pair_ids]

        group_train_rows: list[dict[str, str]] = []
        group_test_rows: list[dict[str, str]] = []
        for pair_id in pair_ids:
            split = "test" if pair_id in test_pair_ids else "train"
            for row in sorted(pairs[pair_id], key=lambda item: item["label"], reverse=True):
                if split == "test":
                    test_sample_index += 1
                    output_row = add_output_fields(row, split, test_sample_index, args.dataset_name)
                    test_rows.append(output_row)
                    group_test_rows.append(output_row)
                else:
                    train_sample_index += 1
                    output_row = add_output_fields(row, split, train_sample_index, args.dataset_name)
                    train_rows.append(output_row)
                    group_train_rows.append(output_row)

        train_labels = count_labels(group_train_rows)
        test_labels = count_labels(group_test_rows)
        uniprot_ids = {row["molecule_parent_uniprot_id"] for row in group_train_rows + group_test_rows}
        summary_rows.append(
            {
                "target_tissue": group[0],
                "mhc_restriction": group[1],
                "total_pairs_before_filter": total_pairs,
                "train_pairs": len(train_pair_ids),
                "test_pairs": len(test_pair_ids),
                "train_rows": len(group_train_rows),
                "test_rows": len(group_test_rows),
                "train_positive_rows": train_labels["1"],
                "train_negative_rows": train_labels["0"],
                "test_positive_rows": test_labels["1"],
                "test_negative_rows": test_labels["0"],
                "n_uniprot_ids": len(uniprot_ids),
            }
        )

    summary_rows.sort(key=lambda row: (-int(row["total_pairs_before_filter"]), row["target_tissue"], row["mhc_restriction"]))
    validate_output(train_rows, test_rows, summary_rows)

    write_rows(args.train_output, train_rows)
    write_rows(args.test_output, test_rows)
    write_summary(args.summary_output, summary_rows)

    metadata = {
        "dataset": args.dataset_name,
        "description": args.description,
        "input": str(args.input),
        "train_output": str(args.train_output),
        "test_output": str(args.test_output),
        "summary_output": str(args.summary_output),
        "min_pairs_filter": f"> {args.min_pairs} pairs per tissue-{args.mhc_name} group",
        "test_pairs_per_tissue_mhc": args.test_pairs,
        "random_seed": args.seed,
        "n_tissue_hla_groups": len(summary_rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_pairs": len({row["pair_id"] for row in train_rows}),
        "test_pairs": len({row["pair_id"] for row in test_rows}),
        "labels": {"1": f"reported in target tissue-{args.mhc_name}", "0": f"same UniProt and {args.mhc_name}, reported in other tissue but not target tissue-{args.mhc_name}"},
        "filters": args.filters or [
            "human peptide source",
            "human host",
            "MHC-I",
            "four-digit HLA allele",
            "positive IEDB qualitative measurement",
            "molecule_parent_uniprot_id available",
            "unmodified standard amino-acid peptide",
            "peptide length = 9",
        ],
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"input_pairs: {len(pairs)}")
    print(f"selected_tissue_hla_groups: {len(summary_rows)}")
    print(f"train_pairs: {metadata['train_pairs']}")
    print(f"test_pairs: {metadata['test_pairs']}")
    print(f"train_rows: {len(train_rows)}")
    print(f"test_rows: {len(test_rows)}")
    print(f"wrote: {args.train_output}")
    print(f"wrote: {args.test_output}")
    print(f"wrote: {args.summary_output}")
    print(f"wrote: {args.metadata_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/iedb_tissue_specificity_pairs.csv.gz"))
    parser.add_argument("--train-output", type=Path, default=Path("data/tissuePMHC/tissuePMHC_train.csv.gz"))
    parser.add_argument("--test-output", type=Path, default=Path("data/tissuePMHC/tissuePMHC_test.csv.gz"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/tissuePMHC/tissuePMHC_summary.csv"))
    parser.add_argument("--metadata-output", type=Path, default=Path("data/tissuePMHC/tissuePMHC_metadata.json"))
    parser.add_argument("--min-pairs", type=int, default=500)
    parser.add_argument("--test-pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--dataset-name", default="tissuePMHC")
    parser.add_argument("--mhc-name", default="HLA")
    parser.add_argument("--description", default="Paired binary benchmark dataset for tissue-HLA-specific presentation preference of unmodified human 9-mer HLA-I peptides.")
    parser.add_argument("--filter", dest="filters", action="append", default=[], help="Repeatable provenance filter recorded in metadata")
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
