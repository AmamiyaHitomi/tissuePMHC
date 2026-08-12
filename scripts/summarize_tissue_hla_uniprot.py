#!/usr/bin/env python3
"""Summarize tissue-level HLA allele and UniProt coverage from processed IEDB data.

Roadmap role: data preparation line.
This script checks whether the processed ligand data has enough tissue-HLA and
protein coverage before building the tissuePMHC benchmark.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path


def open_text_input(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(args: argparse.Namespace) -> None:
    tissue_hlas: dict[str, set[str]] = defaultdict(set)
    tissue_peptides: dict[str, set[str]] = defaultdict(set)
    tissue_uniprots: dict[str, set[str]] = defaultdict(set)

    tissue_hla_uniprots: dict[tuple[str, str], set[str]] = defaultdict(set)
    tissue_hla_peptides: dict[tuple[str, str], set[str]] = defaultdict(set)
    tissue_hla_rows: dict[tuple[str, str], int] = defaultdict(int)

    with open_text_input(args.input) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tissue = row["source_tissue"]
            hla = row["mhc_restriction"]
            peptide = row["peptide_sequence"]
            uniprot = row["molecule_parent_uniprot_id"]

            tissue_hlas[tissue].add(hla)
            tissue_peptides[tissue].add(peptide)
            tissue_uniprots[tissue].add(uniprot)

            key = (tissue, hla)
            tissue_hla_uniprots[key].add(uniprot)
            tissue_hla_peptides[key].add(peptide)
            tissue_hla_rows[key] += 1

    tissue_rows = [
        {
            "source_tissue": tissue,
            "n_hla_alleles": len(tissue_hlas[tissue]),
            "n_molecule_parent_uniprot_ids": len(tissue_uniprots[tissue]),
            "n_peptides": len(tissue_peptides[tissue]),
        }
        for tissue in tissue_hlas
    ]
    tissue_rows.sort(key=lambda row: (-int(row["n_hla_alleles"]), row["source_tissue"]))

    tissue_hla_rows_out = [
        {
            "source_tissue": tissue,
            "mhc_restriction": hla,
            "n_molecule_parent_uniprot_ids": len(tissue_hla_uniprots[(tissue, hla)]),
            "n_peptides": len(tissue_hla_peptides[(tissue, hla)]),
            "n_peptide_mhc_tissue_protein_rows": tissue_hla_rows[(tissue, hla)],
        }
        for tissue, hla in tissue_hla_uniprots
    ]
    tissue_hla_rows_out.sort(
        key=lambda row: (
            row["source_tissue"],
            -int(row["n_molecule_parent_uniprot_ids"]),
            row["mhc_restriction"],
        )
    )

    write_csv(
        args.tissue_output,
        ["source_tissue", "n_hla_alleles", "n_molecule_parent_uniprot_ids", "n_peptides"],
        tissue_rows,
    )
    write_csv(
        args.tissue_hla_output,
        [
            "source_tissue",
            "mhc_restriction",
            "n_molecule_parent_uniprot_ids",
            "n_peptides",
            "n_peptide_mhc_tissue_protein_rows",
        ],
        tissue_hla_rows_out,
    )

    print(f"tissues: {len(tissue_rows)}")
    print(f"tissue_hla_pairs: {len(tissue_hla_rows_out)}")
    print(f"wrote: {args.tissue_output}")
    print(f"wrote: {args.tissue_hla_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/iedb_human_mhci_ligands_unique_peptide_mhc_tissue_protein.csv.gz"),
    )
    parser.add_argument(
        "--tissue-output",
        type=Path,
        default=Path("data/processed/iedb_tissue_summary.csv"),
    )
    parser.add_argument(
        "--tissue-hla-output",
        type=Path,
        default=Path("data/processed/iedb_tissue_hla_uniprot_summary.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    summarize(parse_args())
