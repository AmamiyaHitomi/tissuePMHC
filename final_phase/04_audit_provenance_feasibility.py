#!/usr/bin/env python3
"""Inspect available study/PMID/assay/date provenance without modifying data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import EXPERIMENTS, ROOT, ensure_output, provenance_like, write_json


DEFAULT_EXTRA = [
    ROOT / "data/processed/iedb_tissue_specificity_pairs.csv.gz",
    ROOT / "data/processed/iedb_mouse_tissue_specificity_pairs.csv.gz",
]


def inspect(path: Path) -> tuple[list[dict], dict]:
    frame = pd.read_csv(path)
    candidates = [column for column in frame.columns if provenance_like(column)]
    rows = []
    for column in candidates:
        non_null = int(frame[column].notna().sum())
        non_empty = int(frame[column].fillna("").astype(str).str.strip().ne("").sum())
        rows.append({
            "file": str(path), "column": column, "rows": len(frame),
            "non_null": non_null, "non_empty": non_empty,
            "coverage_pct": 100.0 * non_empty / max(len(frame), 1),
            "unique_non_empty": int(frame.loc[frame[column].fillna("").astype(str).str.strip().ne(""), column].nunique()),
        })
    usable = [row for row in rows if row["coverage_pct"] >= 80 and row["unique_non_empty"] >= 2]
    status = {
        "file": str(path), "rows": len(frame), "all_columns": list(frame.columns),
        "candidate_columns": candidates, "usable_columns_80pct": [row["column"] for row in usable],
        "study_disjoint_feasible_from_this_file": bool(usable),
    }
    return rows, status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra", nargs="*", type=Path, default=DEFAULT_EXTRA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = ensure_output("04_provenance_feasibility")
    paths = [experiment.train for experiment in EXPERIMENTS.values()]
    paths.extend(path for path in args.extra if path.is_file())
    rows, statuses = [], []
    for path in dict.fromkeys(path.resolve() for path in paths):
        part, status = inspect(path)
        rows.extend(part)
        statuses.append(status)
    pd.DataFrame(rows, columns=[
        "file", "column", "rows", "non_null", "non_empty", "coverage_pct", "unique_non_empty"
    ]).to_csv(output / "provenance_column_audit.csv", index=False)
    write_json(output / "provenance_feasibility.json", {
        "files": statuses,
        "interpretation": "A false result documents that study/PMID-disjoint evaluation cannot be reconstructed from the inspected files alone.",
    })
    for status in statuses:
        print(status["file"], status["usable_columns_80pct"])


if __name__ == "__main__":
    main()
