#!/usr/bin/env python3
"""Extract human MHC-I presented human peptides from the IEDB MHC ligand export.

Roadmap role: data preparation line.
This is the first extraction step before any E2/E8 performance-line or E4
biological-representation-line model is trained.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import zipfile
from collections import Counter
from pathlib import Path


REQUIRED_COLUMNS = {
    "assay_id": ("Assay ID", "IEDB IRI"),
    "reference_pmid": ("Reference", "PMID"),
    "reference_title": ("Reference", "Title"),
    "epitope_iri": ("Epitope", "Epitope IRI"),
    "peptide_object_type": ("Epitope", "Object Type"),
    "peptide_sequence": ("Epitope", "Name"),
    "modified_residues": ("Epitope", "Modified residues"),
    "modifications": ("Epitope", "Modifications"),
    "source_molecule": ("Epitope", "Source Molecule"),
    "source_molecule_iri": ("Epitope", "Source Molecule IRI"),
    "molecule_parent": ("Epitope", "Molecule Parent"),
    "molecule_parent_iri": ("Epitope", "Molecule Parent IRI"),
    "peptide_source_organism": ("Epitope", "Source Organism"),
    "peptide_species": ("Epitope", "Species"),
    "host": ("Host", "Name"),
    "assay_method": ("Assay", "Method"),
    "qualitative_measurement": ("Assay", "Qualitative Measurement"),
    "source_tissue": ("Antigen Presenting Cell", "Source Tissue"),
    "source_tissue_iri": ("Antigen Presenting Cell", "Source Tissue IRI"),
    "antigen_presenting_cell": ("Antigen Presenting Cell", "Name"),
    "mhc_restriction": ("MHC Restriction", "Name"),
    "mhc_restriction_iri": ("MHC Restriction", "IRI"),
    "mhc_class": ("MHC Restriction", "Class"),
}

EVIDENCE_COLUMNS = [
    "assay_id",
    "reference_pmid",
    "reference_title",
    "epitope_iri",
    "peptide_sequence",
    "peptide_object_type",
    "modified_residues",
    "modifications",
    "source_molecule",
    "source_molecule_iri",
    "source_molecule_uniprot_id",
    "molecule_parent",
    "molecule_parent_iri",
    "molecule_parent_uniprot_id",
    "peptide_source_organism",
    "peptide_species",
    "host",
    "source_tissue",
    "source_tissue_iri",
    "antigen_presenting_cell",
    "mhc_restriction",
    "mhc_restriction_iri",
    "mhc_class",
    "assay_method",
    "qualitative_measurement",
]

UNIQUE_COLUMNS = [
    "peptide_sequence",
    "mhc_restriction",
    "source_tissue",
    "peptide_source_organism",
    "peptide_species",
    "mhc_class",
]

PROTEIN_UNIQUE_COLUMNS = [
    "peptide_sequence",
    "mhc_restriction",
    "source_tissue",
    "source_molecule",
    "source_molecule_iri",
    "source_molecule_uniprot_id",
    "molecule_parent",
    "molecule_parent_iri",
    "molecule_parent_uniprot_id",
    "peptide_source_organism",
    "peptide_species",
    "mhc_class",
]

FOUR_DIGIT_HLA_RE = re.compile(r"^HLA-[A-Za-z0-9]+\*\d{2}:?\d{2}$")
UNIPROT_RE = re.compile(r"uniprot(?:\.org)?/(?:uniprot/)?([A-Za-z0-9]+)(?:\.\d+)?(?:$|[/?#])")
STANDARD_PEPTIDE_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


def normalize_unknown(value: str) -> str:
    value = (value or "").strip()
    return value if value else "NA"


def is_human(value: str) -> bool:
    value = (value or "").strip().lower()
    return value in {"human", "homo sapiens"} or "homo sapiens" in value


def is_positive(value: str) -> bool:
    return (value or "").strip().lower().startswith("positive")


def is_blank(value: str) -> bool:
    return not (value or "").strip()


def is_unmodified_standard_peptide(peptide: str, modified_residues: str, modifications: str) -> bool:
    return (
        is_blank(modified_residues)
        and is_blank(modifications)
        and bool(STANDARD_PEPTIDE_RE.fullmatch((peptide or "").strip().upper()))
    )


def has_four_digit_hla_type(value: str) -> bool:
    return bool(FOUR_DIGIT_HLA_RE.fullmatch((value or "").strip()))


def extract_uniprot_id(value: str) -> str:
    match = UNIPROT_RE.search((value or "").strip())
    return match.group(1) if match else "NA"


def open_text_output(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", newline="", encoding="utf-8")
    return path.open("w", newline="", encoding="utf-8")


def build_index(header_group: list[str], header_name: list[str]) -> dict[str, int]:
    lookup = {(group, name): idx for idx, (group, name) in enumerate(zip(header_group, header_name))}
    missing = [key for key in REQUIRED_COLUMNS.values() if key not in lookup]
    if missing:
        formatted = ", ".join(f"{group}/{name}" for group, name in missing)
        raise ValueError(f"Missing required IEDB columns: {formatted}")
    return {name: lookup[column_key] for name, column_key in REQUIRED_COLUMNS.items()}


def extract(args: argparse.Namespace) -> Counter:
    summary = Counter()
    unique_keys: set[tuple[str, str, str, str, str, str]] = set()
    protein_unique_keys: set[tuple[str, ...]] = set()

    with zipfile.ZipFile(args.input_zip) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV file found in {args.input_zip}")
        csv_name = csv_names[0]

        with zf.open(csv_name) as raw_file, open_text_output(args.evidence_output) as evidence_file:
            rows = (line.decode("utf-8-sig", errors="replace") for line in raw_file)
            reader = csv.reader(rows)
            header_group = next(reader)
            header_name = next(reader)
            idx = build_index(header_group, header_name)

            evidence_writer = csv.DictWriter(evidence_file, fieldnames=EVIDENCE_COLUMNS)
            evidence_writer.writeheader()

            for row in reader:
                summary["rows_total"] += 1
                if len(row) < len(header_group):
                    summary["rows_short"] += 1
                    continue

                mhc_class = row[idx["mhc_class"]].strip()
                if mhc_class != "I":
                    continue
                summary["rows_mhci"] += 1

                peptide_object_type = row[idx["peptide_object_type"]].strip()
                if args.linear_peptide_only and peptide_object_type.lower() != "linear peptide":
                    continue

                peptide_sequence = row[idx["peptide_sequence"]].strip()
                if not peptide_sequence:
                    continue

                modified_residues = row[idx["modified_residues"]]
                modifications = row[idx["modifications"]]
                if args.require_unmodified and not is_unmodified_standard_peptide(
                    peptide_sequence,
                    modified_residues,
                    modifications,
                ):
                    continue
                summary["rows_unmodified_standard_peptide"] += 1

                if args.peptide_length and len(peptide_sequence) != args.peptide_length:
                    continue
                summary[f"rows_peptide_length_{args.peptide_length}"] += 1

                peptide_source_organism = row[idx["peptide_source_organism"]].strip()
                peptide_species = row[idx["peptide_species"]].strip()
                if not (is_human(peptide_source_organism) or is_human(peptide_species)):
                    continue
                summary["rows_human_peptide_source"] += 1

                host = row[idx["host"]].strip()
                if args.require_human_host and not is_human(host):
                    continue
                summary["rows_human_host"] += 1

                qualitative_measurement = row[idx["qualitative_measurement"]].strip()
                if args.positive_only and not is_positive(qualitative_measurement):
                    continue
                summary["rows_positive"] += 1

                mhc_restriction = row[idx["mhc_restriction"]].strip()
                if args.require_four_digit_hla and not has_four_digit_hla_type(mhc_restriction):
                    continue
                summary["rows_four_digit_hla"] += 1

                molecule_parent_iri = row[idx["molecule_parent_iri"]]
                molecule_parent_uniprot_id = extract_uniprot_id(molecule_parent_iri)
                if args.require_molecule_parent_uniprot and molecule_parent_uniprot_id == "NA":
                    continue
                summary["rows_molecule_parent_uniprot"] += 1

                source_tissue = normalize_unknown(row[idx["source_tissue"]])
                record = {
                    "assay_id": row[idx["assay_id"]].strip(),
                    "reference_pmid": row[idx["reference_pmid"]].strip(),
                    "reference_title": row[idx["reference_title"]].strip(),
                    "epitope_iri": row[idx["epitope_iri"]].strip(),
                    "peptide_sequence": peptide_sequence,
                    "peptide_object_type": peptide_object_type,
                    "modified_residues": normalize_unknown(modified_residues),
                    "modifications": normalize_unknown(modifications),
                    "source_molecule": normalize_unknown(row[idx["source_molecule"]]),
                    "source_molecule_iri": normalize_unknown(row[idx["source_molecule_iri"]]),
                    "source_molecule_uniprot_id": extract_uniprot_id(row[idx["source_molecule_iri"]]),
                    "molecule_parent": normalize_unknown(row[idx["molecule_parent"]]),
                    "molecule_parent_iri": normalize_unknown(molecule_parent_iri),
                    "molecule_parent_uniprot_id": molecule_parent_uniprot_id,
                    "peptide_source_organism": peptide_source_organism,
                    "peptide_species": peptide_species,
                    "host": host,
                    "source_tissue": source_tissue,
                    "source_tissue_iri": normalize_unknown(row[idx["source_tissue_iri"]]),
                    "antigen_presenting_cell": normalize_unknown(row[idx["antigen_presenting_cell"]]),
                    "mhc_restriction": mhc_restriction,
                    "mhc_restriction_iri": normalize_unknown(row[idx["mhc_restriction_iri"]]),
                    "mhc_class": mhc_class,
                    "assay_method": normalize_unknown(row[idx["assay_method"]]),
                    "qualitative_measurement": qualitative_measurement,
                }
                evidence_writer.writerow(record)
                summary["rows_written_evidence"] += 1

                unique_keys.add(tuple(record[column] for column in UNIQUE_COLUMNS))
                protein_unique_keys.add(tuple(record[column] for column in PROTEIN_UNIQUE_COLUMNS))

    with open_text_output(args.unique_output) as unique_file:
        unique_writer = csv.writer(unique_file)
        unique_writer.writerow(UNIQUE_COLUMNS)
        for key in sorted(unique_keys):
            unique_writer.writerow(key)

    with open_text_output(args.protein_unique_output) as protein_unique_file:
        protein_unique_writer = csv.writer(protein_unique_file)
        protein_unique_writer.writerow(PROTEIN_UNIQUE_COLUMNS)
        for key in sorted(protein_unique_keys):
            protein_unique_writer.writerow(key)

    summary["rows_written_unique_peptide_mhc_tissue"] = len(unique_keys)
    summary["rows_written_unique_peptide_mhc_tissue_protein"] = len(protein_unique_keys)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zip", type=Path, default=Path("data/raw/mhc_ligand_full_single_file.zip"))
    parser.add_argument("--evidence-output", type=Path, default=Path("data/processed/iedb_human_mhci_ligands.csv.gz"))
    parser.add_argument("--unique-output", type=Path, default=Path("data/processed/iedb_human_mhci_ligands_unique_peptide_mhc_tissue.csv.gz"))
    parser.add_argument("--protein-unique-output", type=Path, default=Path("data/processed/iedb_human_mhci_ligands_unique_peptide_mhc_tissue_protein.csv.gz"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/processed/iedb_human_mhci_ligands_summary.json"))
    parser.add_argument("--positive-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-human-host", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--linear-peptide-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-four-digit-hla", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-molecule-parent-uniprot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-unmodified", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--peptide-length", type=int, default=9)
    return parser.parse_args()


if __name__ == "__main__":
    summary = extract(parse_args())
    for key, value in summary.items():
        print(f"{key}: {value}")
