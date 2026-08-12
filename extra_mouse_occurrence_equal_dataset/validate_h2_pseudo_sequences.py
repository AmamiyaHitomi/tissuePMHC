#!/usr/bin/env python3
"""Validate the provenance-tracked H2 pseudo-sequence input table."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TABLE = ROOT / "data" / "processed" / "h2_pseudo_sequences.csv"
PROVENANCE = HERE / "h2_pseudo_sequence_provenance.json"
TRAIN = ROOT / "data" / "mousePMHC_occurence_equal_dataset" / "mousePMHC_train.csv.gz"
TEST = ROOT / "data" / "mousePMHC_occurence_equal_dataset" / "mousePMHC_test.csv.gz"
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    table = pd.read_csv(TABLE)
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    expected = sorted(
        set(pd.read_csv(TRAIN)["mhc_restriction"].astype(str))
        | set(pd.read_csv(TEST)["mhc_restriction"].astype(str))
    )
    observed = sorted(table["hla"].astype(str).tolist())
    if observed != expected:
        raise ValueError(f"H2 coverage mismatch: expected={expected}, observed={observed}")
    if table["hla"].duplicated().any():
        raise ValueError("Duplicate H2 names")
    lengths = sorted(table["pseudo_sequence"].astype(str).str.len().unique().tolist())
    if lengths != [34]:
        raise ValueError(f"Expected 34-residue pseudo-sequences, observed={lengths}")
    for row in table.itertuples(index=False):
        unknown = sorted(set(str(row.pseudo_sequence)) - AMINO_ACIDS)
        if unknown:
            raise ValueError(f"{row.hla}: unsupported residues {unknown}")
        if row.source_sha256 != provenance["source_sha256"]:
            raise ValueError(f"{row.hla}: source hash disagrees with provenance")
        if provenance["dataset_to_source_alias"].get(row.hla) != row.source_alias:
            raise ValueError(f"{row.hla}: source alias disagrees with provenance")
    audit = {
        "status": "passed",
        "table": str(TABLE.resolve()),
        "table_sha256": sha256(TABLE),
        "source_file": provenance["source_file"],
        "source_sha256": provenance["source_sha256"],
        "alleles": observed,
        "pseudo_length": 34,
        "n_sequences": len(table),
    }
    output = HERE / "results" / "v7_full_rerun" / "h2_pseudo_sequence_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
