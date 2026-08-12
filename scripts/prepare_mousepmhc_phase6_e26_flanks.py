#!/usr/bin/env python3
"""Build leakage-free local flank/position features for Phase 6 E26.

The builder uses only the supplied reference proteome and mousePMHC *train*
rows.  A peptide contributes flank features only if it maps exactly once to its
declared parent UniProt sequence; ambiguous or absent mappings are retained
with explicit missing values rather than dropped.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mousePMHC_phase6_e26_flank_features"
AA = set("ACDEFGHIKLMNPQRSTVWY")


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")


def parse_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    accession: str | None = None
    pieces: list[str] = []
    with open_text(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if accession is not None:
                    sequences[accession] = "".join(pieces)
                fields = line[1:].split("|", 2)
                accession = fields[1] if len(fields) >= 2 else line[1:].split()[0]
                pieces = []
            else:
                pieces.append(line)
    if accession is not None:
        sequences[accession] = "".join(pieces)
    if not sequences:
        raise ValueError(f"No sequences parsed from {path}")
    return sequences


def occurrences(sequence: str, peptide: str) -> list[int]:
    starts: list[int] = []
    start = sequence.find(peptide)
    while start >= 0:
        starts.append(start)
        start = sequence.find(peptide, start + 1)
    return starts


def run(args: argparse.Namespace) -> None:
    train = pd.read_csv(args.train, keep_default_na=False)
    required = {"dataset", "split", "sample_id", "peptide_sequence", "molecule_parent_uniprot_id"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"Missing train columns: {sorted(missing)}")
    if set(train.dataset) != {"mousePMHC"} or set(train.split) != {"train"}:
        raise ValueError("E26 feature builder accepts mousePMHC train rows only.")
    if not train.peptide_sequence.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{9}").all():
        raise ValueError("E26 feature builder requires standard 9-mer peptides.")
    sequences = parse_fasta(args.reference_fasta)
    rows: list[dict[str, object]] = []
    flank = int(args.flank_length)
    for row in train.itertuples(index=False):
        accession = row.molecule_parent_uniprot_id
        sequence = sequences.get(accession)
        starts = occurrences(sequence, row.peptide_sequence) if sequence else []
        mapped = len(starts) == 1
        if mapped:
            start = starts[0]
            left = sequence[max(0, start - flank):start]
            right = sequence[start + len(row.peptide_sequence):start + len(row.peptide_sequence) + flank]
            left = "-" * (flank - len(left)) + left
            right = right + "-" * (flank - len(right))
            if not set(left.replace("-", "")).issubset(AA) or not set(right.replace("-", "")).issubset(AA):
                mapped = False
        if not mapped:
            start = -1
            left = "-" * flank
            right = "-" * flank
        rows.append({
            "sample_id": row.sample_id, "molecule_parent_uniprot_id": accession,
            "flank_left": left, "flank_right": right, "protein_length": len(sequence) if sequence else 0,
            "peptide_start_0based": start, "position_relative": (start / max(len(sequence) - 1, 1)) if mapped else -1.0,
            "distance_to_nterm": start if mapped else -1, "distance_to_cterm": (len(sequence) - (start + 9)) if mapped else -1,
            "mapping_status": "unique" if mapped else ("missing_accession" if sequence is None else f"ambiguous_{len(starts)}"),
            "flank_available": int(mapped),
        })
    features = pd.DataFrame(rows)
    if features.sample_id.duplicated().any() or len(features) != len(train):
        raise AssertionError("Feature output must contain exactly one row per train sample.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False, compression="gzip")
    status = features.mapping_status.value_counts().rename_axis("mapping_status").reset_index(name="n_rows")
    status.to_csv(args.output.with_name(args.output.stem.replace(".csv", "") + "_mapping_summary.csv"), index=False)
    metadata = {
        "experiment_name": EXPERIMENT, "test_data_read": False, "train": str(args.train),
        "reference_fasta": str(args.reference_fasta), "output": str(args.output), "flank_length": flank,
        "n_train_rows": len(train), "n_reference_sequences": len(sequences),
        "unique_mapping_fraction": float(features.flank_available.mean()),
        "mapping_status_counts": {str(key): int(value) for key, value in features.mapping_status.value_counts().items()},
        "missing_policy": "retain row with '-' flank tokens and -1 numeric values",
    }
    args.output.with_name(args.output.stem.replace(".csv", "") + "_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(status.to_string(index=False), flush=True)
    print(f"unique_mapping_fraction={features.flank_available.mean():.6f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument("--reference-fasta", type=Path, default=project_path("data/mousePMHC/uniprot_mouse_reference.fasta.gz"))
    parser.add_argument("--output", type=Path, default=project_path("data/mousePMHC/mousePMHC_train_flank_features.csv.gz"))
    parser.add_argument("--flank-length", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
