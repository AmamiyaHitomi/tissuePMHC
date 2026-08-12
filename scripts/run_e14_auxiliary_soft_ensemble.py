#!/usr/bin/env python3
"""Run human E14a with Phase 7 min200 inputs and isolated outputs."""

from __future__ import annotations

import sys

from _runner import DATA_DIR, RESULTS_DIR, enable_original_modules


def main() -> None:
    enable_original_modules()
    import run_tissuepmhc_auxiliary_soft_ensemble as e14

    root = RESULTS_DIR / "tissuePMHC_phase7_min200_e14_auxiliary_soft_ensemble"
    e2_root = RESULTS_DIR / "tissuePMHC_phase7_min200_e2_shared_heads"
    fixed_args = [
        "--train", str(DATA_DIR / "tissuePMHC_phase7_min200_train.csv.gz"),
        "--test", str(DATA_DIR / "tissuePMHC_phase7_min200_test.csv.gz"),
        "--e2-per-task", str(e2_root / "per_task_metrics.csv"),
        # E8/E13 are optional historic comparators.  Dedicated Phase 7 paths
        # ensure that an old min500 result is never read by mistake.
        "--e8-per-task", str(RESULTS_DIR / "tissuePMHC_phase7_min200_e8_soft_ensemble" / "per_task_metrics.csv"),
        "--e13-per-task", str(RESULTS_DIR / "tissuePMHC_phase7_min200_e13_auxiliary_tasks" / "per_task_metrics.csv"),
        "--per-task-output", str(root / "per_task_metrics.csv"),
        "--summary-output", str(root / "summary_metrics.csv"),
        "--stability-output", str(root / "stability_metrics.csv"),
        "--candidate-output", str(root / "candidate_metrics.csv"),
        "--diagnostic-output", str(root / "auxiliary_diagnostics.csv"),
        "--comparison-output", str(root / "external_comparison_metrics.csv"),
        "--branch-predictions-output", str(root / "branch_predictions.csv"),
        "--metadata-output", str(root / "metadata.json"),
    ]
    previous_argv = sys.argv
    try:
        sys.argv = [previous_argv[0], *previous_argv[1:], *fixed_args]
        e14.run(e14.parse_args())
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()

