#!/usr/bin/env python3
"""Create Phase 7's independent protein-disjoint development split."""

from __future__ import annotations

import sys

from _runner import DATA_DIR, enable_original_modules


def main() -> None:
    enable_original_modules()
    import build_phase3_protein_disjoint_split as splitter

    root = DATA_DIR / "protein_disjoint_split"
    fixed_args = [
        "--input", str(DATA_DIR / "tissuePMHC_phase7_min200_train.csv.gz"),
        "--development-output", str(root / "phase7_development.csv.gz"),
        "--confirmation-features-output", str(root / "phase7_confirmation_features.csv.gz"),
        "--confirmation-labels-output", str(root / "phase7_confirmation_labels.csv.gz"),
        "--assignments-output", str(root / "parent_group_assignments.csv"),
        "--task-counts-output", str(root / "task_counts.csv"),
        "--manifest-output", str(root / "split_manifest.json"),
    ]
    previous_argv = sys.argv
    try:
        sys.argv = [previous_argv[0], *previous_argv[1:], *fixed_args]
        splitter.run(splitter.parse_args())
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()

