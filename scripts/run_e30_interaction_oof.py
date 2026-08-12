#!/usr/bin/env python3
"""Run E29/E30 interaction encoders on Phase 7's min200 development split."""

from __future__ import annotations

import sys

from _runner import DATA_DIR, RESULTS_DIR, enable_original_modules


def main() -> None:
    enable_original_modules()
    import run_tissuepmhc_e30_interaction_oof as e30

    root = RESULTS_DIR / "tissuePMHC_phase7_min200_e30_interaction_oof"
    fixed_args = [
        "--train", str(DATA_DIR / "protein_disjoint_split" / "phase7_development.csv.gz"),
        "--oof-predictions-output", str(root / "oof_predictions.csv"),
        "--fold-assignments-output", str(root / "fold_assignments.csv"),
        "--diagnostics-output", str(root / "training_diagnostics.json"),
        "--task-comparison-output", str(root / "oof_task_comparison_vs_e29.csv"),
        "--run-manifest-output", str(root / "run_manifest.json"),
    ]
    previous_argv = sys.argv
    try:
        sys.argv = [previous_argv[0], *previous_argv[1:], *fixed_args]
        e30.run(e30.parse_args())
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
