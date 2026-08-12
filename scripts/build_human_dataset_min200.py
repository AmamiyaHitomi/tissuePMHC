#!/usr/bin/env python3
"""Build the isolated human Phase 7 benchmark with ``min_pairs = 200``."""

from __future__ import annotations

import sys
import json
from typing import Any

from _runner import DATASET_NAME, DATA_DIR, MIN_PAIRS, enable_original_modules, path


def is_present(value: object) -> bool:
    normalized = "" if value is None else str(value).strip().casefold()
    return normalized not in {"", "na", "n/a", "nan", "none", "null"}


def filter_unassigned_task_pairs(pairs: dict[str, list[dict[str, str]]]) -> tuple[dict[str, list[dict[str, str]]], dict[str, int]]:
    """Remove pairs that cannot define a tissue-HLA prediction task.

    The raw IEDB-derived pair table contains a small number of records with an
    empty target tissue.  They previously became blank CSV fields, then pandas
    read them as NaN and E2 could not sort task names.  A pair is retained only
    when every row has a non-empty, pair-consistent tissue and HLA value.
    """
    kept: dict[str, list[dict[str, str]]] = {}
    audit = {"input_pairs": len(pairs), "dropped_missing_task_fields": 0, "dropped_inconsistent_task_fields": 0}
    for pair_id, rows in pairs.items():
        tissues = {str(row.get("target_tissue", "")).strip() for row in rows}
        hlas = {str(row.get("mhc_restriction", "")).strip() for row in rows}
        if not all(is_present(row.get("target_tissue")) and is_present(row.get("mhc_restriction")) for row in rows):
            audit["dropped_missing_task_fields"] += 1
        elif len(tissues) != 1 or len(hlas) != 1:
            audit["dropped_inconsistent_task_fields"] += 1
        else:
            kept[pair_id] = rows
    audit["retained_pairs"] = len(kept)
    return kept, audit


def main() -> None:
    enable_original_modules()
    import build_tissuepmhc_dataset as builder

    # The fixed arguments are appended last, so callers cannot accidentally
    # overwrite the Phase 7 threshold, identity, or output locations.
    fixed_args = [
        "--input", str(path("data", "processed", "iedb_tissue_specificity_pairs.csv.gz")),
        "--train-output", str(DATA_DIR / "tissuePMHC_phase7_min200_train.csv.gz"),
        "--test-output", str(DATA_DIR / "tissuePMHC_phase7_min200_test.csv.gz"),
        "--summary-output", str(DATA_DIR / "tissuePMHC_phase7_min200_summary.csv"),
        "--metadata-output", str(DATA_DIR / "tissuePMHC_phase7_min200_metadata.json"),
        "--min-pairs", str(MIN_PAIRS),
        "--dataset-name", DATASET_NAME,
        "--description", "Phase 7 isolated human tissue-HLA benchmark with total_pairs > 200.",
    ]
    previous_argv = sys.argv
    original_read_pairs = builder.read_pairs
    audit: dict[str, Any] = {}

    def read_phase7_pairs(input_path):
        nonlocal audit
        pairs = original_read_pairs(input_path)
        filtered, audit = filter_unassigned_task_pairs(pairs)
        return filtered

    try:
        sys.argv = [previous_argv[0], *previous_argv[1:], *fixed_args]
        builder.read_pairs = read_phase7_pairs
        builder.build_dataset(builder.parse_args())
    finally:
        builder.read_pairs = original_read_pairs
        sys.argv = previous_argv

    audit_path = DATA_DIR / "tissuePMHC_phase7_min200_input_filter_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote: {audit_path}")


if __name__ == "__main__":
    main()
