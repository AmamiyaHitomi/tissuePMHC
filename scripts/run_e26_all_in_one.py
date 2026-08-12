#!/usr/bin/env python3
"""Create the min200 E26 OOF baseline required by the Phase 7 E29 screen."""

from __future__ import annotations

import sys

from _runner import DATA_DIR, RESULTS_DIR, enable_original_modules


def main() -> None:
    enable_original_modules()
    import run_tissuepmhc_e26_all_in_one as e26

    root = RESULTS_DIR / "tissuePMHC_phase7_min200_e26_greedy_ensemble_selection"
    fixed_args = [
        "--train", str(DATA_DIR / "tissuePMHC_phase7_min200_train.csv.gz"),
        "--test", str(DATA_DIR / "tissuePMHC_phase7_min200_test.csv.gz"),
        "--oof-predictions-output", str(root / "oof_predictions.csv"),
        "--test-predictions-output", str(root / "test_predictions.csv"),
        "--per-task-output", str(root / "per_task_metrics.csv"),
        "--summary-output", str(root / "summary_metrics.csv"),
        "--stability-output", str(root / "stability_metrics.csv"),
        "--trajectory-output", str(root / "oof_selection_trajectory.csv"),
        "--members-output", str(root / "selected_members.csv"),
        "--selection-metadata-output", str(root / "selection_metadata.json"),
        "--metadata-output", str(root / "metadata.json"),
    ]
    previous_argv = sys.argv
    try:
        sys.argv = [previous_argv[0], *previous_argv[1:], *fixed_args]
        e26.run(e26.parse_args())
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
