#!/usr/bin/env python3
"""Build an HLA class I pseudo-sequence table from IPD-IMGT/HLA protein FASTA files.

Roadmap role: E4 preparation for the biological-representation line.
The generated pseudo-sequences are used by E4/E4b models, not by the current
E2/E8 performance main line.

The output table is intended for scripts/run_tissuepmhc_hla_pseudoseq.py.
It uses the common NetMHCpan-style 34 residue positions for MHC-I binding-site
pseudo-sequences and extracts those positions from the mature heavy-chain
sequence. IPD-IMGT/HLA protein FASTA entries include a signal peptide at the
N-terminus, so this script removes the first 24 amino acids before indexing.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HLAS = [
    "HLA-A*02:01",
    "HLA-A*24:02",
    "HLA-B*07:02",
    "HLA-B*15:01",
    "HLA-B*15:02",
    "HLA-B*27:05",
    "HLA-B*40:01",
    "HLA-B*40:02",
    "HLA-B*51:01",
    "HLA-C*03:04",
    "HLA-C*05:01",
    "HLA-C*12:02",
]

PSEUDO_POSITIONS = [
    7,
    9,
    24,
    45,
    59,
    62,
    63,
    66,
    67,
    69,
    70,
    73,
    74,
    76,
    77,
    80,
    81,
    84,
    95,
    97,
    99,
    114,
    116,
    118,
    143,
    147,
    150,
    152,
    156,
    158,
    159,
    163,
    167,
    171,
]
SIGNAL_PEPTIDE_LENGTH = 24


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def parse_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    current_header: str | None = None
    current_lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    records[current_header] = "".join(current_lines)
                current_header = line[1:]
                current_lines = []
            else:
                current_lines.append(line)
    if current_header is not None:
        records[current_header] = "".join(current_lines)
    return records


def allele_two_field(header: str) -> str | None:
    match = re.search(r"\b([ABC]\*\d+:\d+)", header)
    if not match:
        return None
    return f"HLA-{match.group(1)}"


def choose_sequences(fasta_paths: list[Path], hlas: list[str]) -> dict[str, tuple[str, str]]:
    wanted = set(hlas)
    chosen: dict[str, tuple[str, str]] = {}
    for fasta_path in fasta_paths:
        for header, sequence in parse_fasta(fasta_path).items():
            hla = allele_two_field(header)
            if hla not in wanted or hla in chosen:
                continue
            if len(sequence) < SIGNAL_PEPTIDE_LENGTH + max(PSEUDO_POSITIONS):
                continue
            chosen[hla] = (header, sequence)

    missing = sorted(wanted - set(chosen))
    if missing:
        raise ValueError("Could not find full-length protein sequences for: " + ", ".join(missing))
    return chosen


def make_pseudo_sequence(sequence: str) -> str:
    mature_sequence = sequence[SIGNAL_PEPTIDE_LENGTH:]
    return "".join(mature_sequence[position - 1] for position in PSEUDO_POSITIONS)


def read_hlas_from_dataset(train_path: Path) -> list[str]:
    df = pd.read_csv(train_path)
    return sorted(df["mhc_restriction"].unique())


def run(args: argparse.Namespace) -> None:
    hlas = read_hlas_from_dataset(args.train) if args.from_train else args.hlas
    chosen = choose_sequences(args.fasta, hlas)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "hla",
                "pseudo_sequence",
                "source",
                "source_header",
                "positions",
                "signal_peptide_trim",
            ],
        )
        writer.writeheader()
        for hla in sorted(hlas):
            header, sequence = chosen[hla]
            writer.writerow(
                {
                    "hla": hla,
                    "pseudo_sequence": make_pseudo_sequence(sequence),
                    "source": "IPD-IMGT/HLA protein FASTA; NetMHCpan-style 34 MHC-I pseudo-sequence positions",
                    "source_header": header,
                    "positions": " ".join(str(position) for position in PSEUDO_POSITIONS),
                    "signal_peptide_trim": SIGNAL_PEPTIDE_LENGTH,
                }
            )

    print(f"wrote: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz"))
    parser.add_argument(
        "--fasta",
        nargs="+",
        type=Path,
        default=[
            project_path("data/processed/A_prot.fasta"),
            project_path("data/processed/B_prot.fasta"),
            project_path("data/processed/C_prot.fasta"),
        ],
    )
    parser.add_argument("--output", type=Path, default=project_path("data/processed/hla_pseudo_sequences.csv"))
    parser.add_argument("--hlas", nargs="+", default=DEFAULT_HLAS)
    parser.add_argument("--from-train", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
