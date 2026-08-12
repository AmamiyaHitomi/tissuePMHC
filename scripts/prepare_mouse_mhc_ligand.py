#!/usr/bin/env python3
"""Extract a reproducible mouse MHC-I ligand evidence table from IEDB CSV export.

The IEDB single-file export has two header rows.  This script intentionally
keeps row-level provenance before deduplication, then emits the unique records
consumed by the existing paired tissue-specificity construction script.
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


REQUIRED = {
    "assay_id": ("Assay ID", "IEDB IRI"), "reference": ("Reference", "PMID"),
    "peptide": ("Epitope", "Name"), "object_type": ("Epitope", "Object Type"),
    "modified": ("Epitope", "Modified residues"), "modifications": ("Epitope", "Modifications"),
    "source_molecule": ("Epitope", "Source Molecule"), "source_molecule_iri": ("Epitope", "Source Molecule IRI"),
    "molecule_parent": ("Epitope", "Molecule Parent"), "molecule_parent_iri": ("Epitope", "Molecule Parent IRI"),
    "source_organism": ("Epitope", "Source Organism"), "species": ("Epitope", "Species"),
    "host": ("Host", "Name"), "qualitative": ("Assay", "Qualitative Measurement"),
    "tissue": ("Antigen Presenting Cell", "Source Tissue"), "tissue_iri": ("Antigen Presenting Cell", "Source Tissue IRI"),
    "apc": ("Antigen Presenting Cell", "Name"), "mhc": ("MHC Restriction", "Name"),
    "mhc_iri": ("MHC Restriction", "IRI"), "mhc_class": ("MHC Restriction", "Class"),
}
OUT = ["assay_id", "reference_pmid", "peptide_sequence", "source_molecule", "source_molecule_iri", "source_molecule_uniprot_id", "molecule_parent", "molecule_parent_iri", "molecule_parent_uniprot_id", "peptide_source_organism", "peptide_species", "host", "source_tissue", "source_tissue_iri", "antigen_presenting_cell", "mhc_restriction", "mhc_restriction_iri", "mhc_class", "qualitative_measurement"]
UNIQUE = ["peptide_sequence", "mhc_restriction", "source_tissue", "source_molecule", "source_molecule_iri", "source_molecule_uniprot_id", "molecule_parent", "molecule_parent_iri", "molecule_parent_uniprot_id", "peptide_source_organism", "peptide_species", "mhc_class"]
AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
UNIPROT = re.compile(r"uniprot(?:\.org)?/(?:uniprot/)?([A-Za-z0-9]+)(?:\.\d+)?(?:$|[/?#])")


def mouse(value: str) -> bool:
    return "mus musculus" in (value or "").strip().lower() or (value or "").strip().lower() == "mouse"


def uniprot(value: str) -> str:
    match = UNIPROT.search(value or "")
    return match.group(1) if match else "NA"


def open_out(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return gzip.open(path, "wt", newline="", encoding="utf-8") if path.suffix == ".gz" else path.open("w", newline="", encoding="utf-8")


def clean(value: str) -> str:
    return (value or "").strip() or "NA"


def main(args: argparse.Namespace) -> None:
    counts: Counter[str] = Counter()
    unique: dict[tuple[str, ...], dict[str, str]] = {}
    if args.input.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(args.input)
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected one CSV in {args.input}, got {names}")
        raw_context = archive.open(names[0])
    else:
        archive = None
        raw_context = args.input.open("rb")
    try:
        with raw_context as raw, open_out(args.evidence_output) as output:
            reader = csv.reader(line.decode("utf-8-sig", errors="replace") for line in raw)
            group, name = next(reader), next(reader)
            lookup = {(a, b): i for i, (a, b) in enumerate(zip(group, name))}
            missing = [value for value in REQUIRED.values() if value not in lookup]
            if missing:
                raise ValueError(f"Missing IEDB columns: {missing}")
            idx = {key: lookup[value] for key, value in REQUIRED.items()}
            writer = csv.DictWriter(output, fieldnames=OUT); writer.writeheader()
            for row in reader:
                counts["rows_total"] += 1
                if len(row) < len(group): continue
                if row[idx["mhc_class"]].strip() != "I": continue
                counts["mhc_i"] += 1
                peptide = row[idx["peptide"]].strip().upper()
                if (row[idx["object_type"]].strip().lower() != "linear peptide" or len(peptide) != args.peptide_length or
                    not AA.fullmatch(peptide) or row[idx["modified"]].strip() or row[idx["modifications"]].strip()): continue
                counts["unmodified_linear_length"] += 1
                if not (mouse(row[idx["source_organism"]]) or mouse(row[idx["species"]])): continue
                if args.require_mouse_host and not mouse(row[idx["host"]]): continue
                if not row[idx["qualitative"]].strip().lower().startswith("positive"): continue
                mhc = row[idx["mhc"]].strip()
                if not mhc.startswith("H2-"): continue
                parent_iri = row[idx["molecule_parent_iri"]]
                parent_id = uniprot(parent_iri)
                if parent_id == "NA": continue
                record = {"assay_id": clean(row[idx["assay_id"]]), "reference_pmid": clean(row[idx["reference"]]), "peptide_sequence": peptide,
                    "source_molecule": clean(row[idx["source_molecule"]]), "source_molecule_iri": clean(row[idx["source_molecule_iri"]]), "source_molecule_uniprot_id": uniprot(row[idx["source_molecule_iri"]]),
                    "molecule_parent": clean(row[idx["molecule_parent"]]), "molecule_parent_iri": clean(parent_iri), "molecule_parent_uniprot_id": parent_id,
                    "peptide_source_organism": clean(row[idx["source_organism"]]), "peptide_species": clean(row[idx["species"]]), "host": clean(row[idx["host"]]),
                    "source_tissue": clean(row[idx["tissue"]]), "source_tissue_iri": clean(row[idx["tissue_iri"]]), "antigen_presenting_cell": clean(row[idx["apc"]]),
                    "mhc_restriction": mhc, "mhc_restriction_iri": clean(row[idx["mhc_iri"]]), "mhc_class": "I", "qualitative_measurement": clean(row[idx["qualitative"]])}
                writer.writerow(record); counts["evidence_rows"] += 1
                unique.setdefault(tuple(record[col] for col in UNIQUE), record)
    finally:
        if archive is not None:
            archive.close()
    with open_out(args.unique_output) as output:
        writer = csv.DictWriter(output, fieldnames=UNIQUE); writer.writeheader()
        for key in sorted(unique): writer.writerow({column: unique[key][column] for column in UNIQUE})
    counts["unique_rows"] = len(unique)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/mhc_ligand_full.csv"), help="IEDB single-file .csv or .zip export")
    parser.add_argument("--evidence-output", type=Path, default=Path("data/processed/iedb_mouse_mhci_ligands.csv.gz"))
    parser.add_argument("--unique-output", type=Path, default=Path("data/processed/iedb_mouse_mhci_ligands_unique_protein.csv.gz"))
    parser.add_argument("--summary-output", type=Path, default=Path("data/processed/iedb_mouse_mhci_ligands_summary.json"))
    parser.add_argument("--peptide-length", type=int, default=9)
    parser.add_argument("--require-mouse-host", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__": main(parse_args())
