#!/usr/bin/env python3
"""Build the Phase 7 E17 prediction ensemble from Phase 7 E14 outputs."""

from __future__ import annotations

import sys

from _runner import RESULTS_DIR, enable_original_modules


def main() -> None:
    enable_original_modules()
    import run_tissuepmhc_e17_seed_ensemble as e17

    root = RESULTS_DIR / "tissuePMHC_phase7_min200_e17_seed_ensemble"
    e14_root = RESULTS_DIR / "tissuePMHC_phase7_min200_e14_auxiliary_soft_ensemble"
    fixed_args = [
        "--branch-predictions", str(e14_root / "branch_predictions.csv"),
        "--per-task-output", str(root / "per_task_metrics.csv"),
        "--summary-output", str(root / "summary_metrics.csv"),
        "--stability-output", str(root / "stability_metrics.csv"),
        "--predictions-output", str(root / "branch_predictions.csv"),
        "--metadata-output", str(root / "metadata.json"),
    ]
    previous_argv = sys.argv
    try:
        sys.argv = [previous_argv[0], *previous_argv[1:], *fixed_args]
        e17.run(e17.parse_args())
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
