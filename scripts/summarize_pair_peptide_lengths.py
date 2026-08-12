#!/usr/bin/env python3
"""Summarize peptide length distributions in paired tissue-specificity samples.

Roadmap role: data preparation line.
This validates peptide length distributions before the final tissuePMHC split.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
from collections import Counter
from pathlib import Path


AA_RE = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]")


def open_text_input(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def peptide_length(peptide: str) -> int:
    base_sequence = peptide.split(" + ", 1)[0].strip().upper()
    return len(AA_RE.findall(base_sequence))


def summarize(args: argparse.Namespace) -> None:
    counts = {"0": Counter(), "1": Counter()}
    totals = Counter()

    with open_text_input(args.input) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"]
            length = peptide_length(row["peptide_sequence"])
            counts[label][length] += 1
            totals[label] += 1

    all_lengths = sorted(set(counts["0"]) | set(counts["1"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "peptide_length",
                "positive_count",
                "negative_count",
                "positive_fraction",
                "negative_fraction",
            ],
        )
        writer.writeheader()
        for length in all_lengths:
            positive_count = counts["1"][length]
            negative_count = counts["0"][length]
            writer.writerow(
                {
                    "peptide_length": length,
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "positive_fraction": positive_count / totals["1"] if totals["1"] else 0,
                    "negative_fraction": negative_count / totals["0"] if totals["0"] else 0,
                }
            )

    print(f"positive_total: {totals['1']}")
    print(f"negative_total: {totals['0']}")
    print(f"wrote: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/iedb_tissue_specificity_pairs.csv.gz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/iedb_tissue_specificity_pair_length_distribution.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    summarize(parse_args())
