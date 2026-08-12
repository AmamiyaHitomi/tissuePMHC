#!/usr/bin/env python3
"""Build paired positive/negative peptide samples for tissue-specific presentation.

Roadmap role: data preparation line.
This creates the positive/negative peptide pairs used by all later E2/E8
performance-line and E4 biological-representation-line experiments.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import random
from collections import defaultdict
from pathlib import Path


PAIR_COLUMNS = [
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
    "n_pairs",
    "n_positive_rows",
    "n_negative_rows",
    "n_positive_peptides",
    "n_negative_peptides",
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


def read_records(path: Path) -> list[dict[str, str]]:
    with open_text_input(path) as f:
        return list(csv.DictReader(f))


def build_pairs(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    records = read_records(args.input)

    peptides_by_hla_uniprot: dict[tuple[str, str], set[str]] = defaultdict(set)
    peptides_by_tissue_hla: dict[tuple[str, str], set[str]] = defaultdict(set)
    peptides_by_tissue_hla_uniprot: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    tissues_by_hla_uniprot_peptide: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    annotation_by_hla_uniprot_peptide: dict[tuple[str, str, str], dict[str, str]] = {}

    for row in records:
        tissue = row["source_tissue"]
        hla = row["mhc_restriction"]
        peptide = row["peptide_sequence"]
        uniprot = row["molecule_parent_uniprot_id"]
        if not peptide or not hla or not tissue or not uniprot or uniprot == "NA":
            continue

        hla_uniprot = (hla, uniprot)
        hla_uniprot_peptide = (hla, uniprot, peptide)
        peptides_by_hla_uniprot[hla_uniprot].add(peptide)
        peptides_by_tissue_hla[(tissue, hla)].add(peptide)
        peptides_by_tissue_hla_uniprot[(tissue, hla, uniprot)].add(peptide)
        tissues_by_hla_uniprot_peptide[hla_uniprot_peptide].add(tissue)
        annotation_by_hla_uniprot_peptide.setdefault(hla_uniprot_peptide, row)

    pair_rows: list[dict[str, str]] = []
    summary: dict[tuple[str, str], dict[str, set[str] | int]] = {}
    used_positive_peptides_by_tissue_hla: dict[tuple[str, str], set[str]] = defaultdict(set)
    used_negative_peptides_by_tissue_hla: dict[tuple[str, str], set[str]] = defaultdict(set)
    pair_id = 0

    for target_tissue, hla, uniprot in sorted(peptides_by_tissue_hla_uniprot):
        tissue_hla = (target_tissue, hla)
        positive_peptides = sorted(
            peptides_by_tissue_hla_uniprot[(target_tissue, hla, uniprot)]
            - used_positive_peptides_by_tissue_hla[tissue_hla]
        )
        peptides_reported_in_target_tissue_hla = peptides_by_tissue_hla[(target_tissue, hla)]
        negative_peptides = []
        for candidate_peptide in peptides_by_hla_uniprot[(hla, uniprot)]:
            if candidate_peptide in peptides_reported_in_target_tissue_hla:
                continue
            if candidate_peptide in used_negative_peptides_by_tissue_hla[tissue_hla]:
                continue
            candidate_tissues = tissues_by_hla_uniprot_peptide[(hla, uniprot, candidate_peptide)]
            if target_tissue in candidate_tissues:
                continue
            negative_peptides.append(candidate_peptide)

        if not positive_peptides or not negative_peptides:
            continue

        n_pairs_for_group = min(len(positive_peptides), len(negative_peptides))
        selected_positive_peptides = rng.sample(positive_peptides, n_pairs_for_group)
        selected_negative_peptides = rng.sample(sorted(negative_peptides), n_pairs_for_group)
        rng.shuffle(selected_negative_peptides)
        used_positive_peptides_by_tissue_hla[tissue_hla].update(selected_positive_peptides)
        used_negative_peptides_by_tissue_hla[tissue_hla].update(selected_negative_peptides)

        for positive_peptide, negative_peptide in zip(selected_positive_peptides, selected_negative_peptides):
            positive_key = (hla, uniprot, positive_peptide)
            negative_key = (hla, uniprot, negative_peptide)
            positive_annotation = annotation_by_hla_uniprot_peptide[positive_key]
            negative_annotation = annotation_by_hla_uniprot_peptide[negative_key]
            pair_id += 1
            pair_id_text = f"pair_{pair_id:012d}"

            positive_tissues = tissues_by_hla_uniprot_peptide[positive_key]
            negative_tissues = tissues_by_hla_uniprot_peptide[negative_key]

            pair_rows.append(
                {
                    "pair_id": pair_id_text,
                    "label": "1",
                    "target_tissue": target_tissue,
                    "mhc_restriction": hla,
                    "peptide_sequence": positive_peptide,
                    "molecule_parent_uniprot_id": uniprot,
                    "source_molecule": positive_annotation["source_molecule"],
                    "source_molecule_uniprot_id": positive_annotation["source_molecule_uniprot_id"],
                    "molecule_parent": positive_annotation["molecule_parent"],
                    "reported_tissues_same_hla_uniprot": ";".join(sorted(positive_tissues)),
                }
            )
            pair_rows.append(
                {
                    "pair_id": pair_id_text,
                    "label": "0",
                    "target_tissue": target_tissue,
                    "mhc_restriction": hla,
                    "peptide_sequence": negative_peptide,
                    "molecule_parent_uniprot_id": uniprot,
                    "source_molecule": negative_annotation["source_molecule"],
                    "source_molecule_uniprot_id": negative_annotation["source_molecule_uniprot_id"],
                    "molecule_parent": negative_annotation["molecule_parent"],
                    "reported_tissues_same_hla_uniprot": ";".join(sorted(negative_tissues)),
                }
            )

            summary_key = (target_tissue, hla)
            if summary_key not in summary:
                summary[summary_key] = {
                    "n_pairs": 0,
                    "n_positive_rows": 0,
                    "n_negative_rows": 0,
                    "positive_peptides": set(),
                    "negative_peptides": set(),
                    "uniprot_ids": set(),
                }
            summary_item = summary[summary_key]
            summary_item["n_pairs"] = int(summary_item["n_pairs"]) + 1
            summary_item["n_positive_rows"] = int(summary_item["n_positive_rows"]) + 1
            summary_item["n_negative_rows"] = int(summary_item["n_negative_rows"]) + 1
            summary_item["positive_peptides"].add(positive_peptide)  # type: ignore[union-attr]
            summary_item["negative_peptides"].add(negative_peptide)  # type: ignore[union-attr]
            summary_item["uniprot_ids"].add(uniprot)  # type: ignore[union-attr]

    with open_text_output(args.output) as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_COLUMNS)
        writer.writeheader()
        writer.writerows(pair_rows)

    summary_rows = []
    for (target_tissue, hla), values in summary.items():
        summary_rows.append(
            {
                "target_tissue": target_tissue,
                "mhc_restriction": hla,
                "n_pairs": values["n_pairs"],
                "n_positive_rows": values["n_positive_rows"],
                "n_negative_rows": values["n_negative_rows"],
                "n_positive_peptides": len(values["positive_peptides"]),  # type: ignore[arg-type]
                "n_negative_peptides": len(values["negative_peptides"]),  # type: ignore[arg-type]
                "n_uniprot_ids": len(values["uniprot_ids"]),  # type: ignore[arg-type]
            }
        )
    summary_rows.sort(key=lambda row: (-int(row["n_pairs"]), row["target_tissue"], row["mhc_restriction"]))

    with args.summary_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"input_rows: {len(records)}")
    print(f"pairs: {pair_id}")
    print(f"paired_rows: {len(pair_rows)}")
    print(f"tissue_hla_pairs_with_pairs: {len(summary_rows)}")
    validate_pairs(pair_rows, peptides_by_tissue_hla)
    print(f"wrote: {args.output}")
    print(f"wrote: {args.summary_output}")


def validate_pairs(pair_rows: list[dict[str, str]], peptides_by_tissue_hla: dict[tuple[str, str], set[str]]) -> None:
    rows_by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
    label_counts_by_tissue_hla: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"0": 0, "1": 0})
    peptides_by_tissue_hla_label: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    for row in pair_rows:
        rows_by_pair[row["pair_id"]].append(row)
        label_counts_by_tissue_hla[(row["target_tissue"], row["mhc_restriction"])][row["label"]] += 1
        peptides_by_tissue_hla_label[
            (row["target_tissue"], row["mhc_restriction"], row["label"])
        ].add(row["peptide_sequence"])

    invalid_pair_ids = 0
    mismatched_uniprot_pair_ids = 0
    negative_reported_in_target = 0
    for rows in rows_by_pair.values():
        if len(rows) != 2 or sorted(row["label"] for row in rows) != ["0", "1"]:
            invalid_pair_ids += 1
            continue
        positive = next(row for row in rows if row["label"] == "1")
        negative = next(row for row in rows if row["label"] == "0")
        if positive["molecule_parent_uniprot_id"] != negative["molecule_parent_uniprot_id"]:
            mismatched_uniprot_pair_ids += 1
        target_key = (negative["target_tissue"], negative["mhc_restriction"])
        if negative["peptide_sequence"] in peptides_by_tissue_hla[target_key]:
            negative_reported_in_target += 1

    tissue_hla_label_count_mismatches = sum(
        1 for counts in label_counts_by_tissue_hla.values() if counts["0"] != counts["1"]
    )
    tissue_hla_unique_peptide_mismatches = 0
    tissue_hla_duplicate_label_peptides = 0
    for tissue_hla, counts in label_counts_by_tissue_hla.items():
        positive_peptides = peptides_by_tissue_hla_label[(tissue_hla[0], tissue_hla[1], "1")]
        negative_peptides = peptides_by_tissue_hla_label[(tissue_hla[0], tissue_hla[1], "0")]
        if counts["1"] != len(positive_peptides) or counts["0"] != len(negative_peptides):
            tissue_hla_duplicate_label_peptides += 1
        if len(positive_peptides) != len(negative_peptides):
            tissue_hla_unique_peptide_mismatches += 1

    print(f"validation_invalid_pair_ids: {invalid_pair_ids}")
    print(f"validation_mismatched_uniprot_pair_ids: {mismatched_uniprot_pair_ids}")
    print(f"validation_negative_reported_in_target_tissue_hla: {negative_reported_in_target}")
    print(f"validation_tissue_hla_label_count_mismatches: {tissue_hla_label_count_mismatches}")
    print(f"validation_tissue_hla_duplicate_label_peptides: {tissue_hla_duplicate_label_peptides}")
    print(f"validation_tissue_hla_unique_peptide_mismatches: {tissue_hla_unique_peptide_mismatches}")

    if any(
        [
            invalid_pair_ids,
            mismatched_uniprot_pair_ids,
            negative_reported_in_target,
            tissue_hla_label_count_mismatches,
            tissue_hla_duplicate_label_peptides,
            tissue_hla_unique_peptide_mismatches,
        ]
    ):
        raise ValueError("Pair validation failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/iedb_human_mhci_ligands_unique_peptide_mhc_tissue_protein.csv.gz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/iedb_tissue_specificity_pairs.csv.gz"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/processed/iedb_tissue_specificity_pairs_summary.csv"),
    )
    parser.add_argument("--seed", type=int, default=20260704)
    return parser.parse_args()


if __name__ == "__main__":
    build_pairs(parse_args())
