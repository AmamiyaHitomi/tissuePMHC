#!/usr/bin/env python3
"""Run the human E29 CNN on only the isolated Phase 7 min200 dataset."""

from __future__ import annotations

import sys

from _runner import DATA_DIR, RESULTS_DIR, enable_original_modules


def main() -> None:
    enable_original_modules()
    import run_tissuepmhc_e29_multikernel_cnn_oof as e29

    root = RESULTS_DIR / "tissuePMHC_phase7_min200_e29_multikernel_cnn"
    e26_root = RESULTS_DIR / "tissuePMHC_phase7_min200_e26_greedy_ensemble_selection"
    e17_root = RESULTS_DIR / "tissuePMHC_phase7_min200_e17_seed_ensemble"
    fixed_args = [
        "--train", str(DATA_DIR / "tissuePMHC_phase7_min200_train.csv.gz"),
        "--test", str(DATA_DIR / "tissuePMHC_phase7_min200_test.csv.gz"),
        "--experiment-name", "Phase7_min200_E29_multikernel_cnn_oof",
        "--candidate-prefix", "phase7_min200_e29_cnn",
        "--baseline-oof-predictions", str(e26_root / "oof_predictions.csv"),
        "--matching-baseline-candidate", "e14_final_3seed_mean",
        "--fusion-baseline-candidate", "e14_final_3seed_mean",
        "--oof-predictions-output", str(root / "oof_predictions.csv"),
        "--test-predictions-output", str(root / "test_predictions.csv"),
        "--task-comparison-output", str(root / "oof_task_comparison.csv"),
        "--screen-summary-output", str(root / "oof_screen_summary.json"),
        "--per-task-output", str(root / "per_task_metrics.csv"),
        "--summary-output", str(root / "summary_metrics.csv"),
        "--stability-output", str(root / "stability_metrics.csv"),
        "--e17-per-task", str(e17_root / "per_task_metrics.csv"),
        "--e17-comparison-output", str(root / "e17_3seed_comparison_metrics.csv"),
    ]
    previous_argv = sys.argv
    try:
        # OOF-only is the safe default; a caller may explicitly add
        # ``--run-test`` after it, once the matched Phase 7 baseline exists.
        sys.argv = [previous_argv[0], "--no-run-test", *previous_argv[1:], *fixed_args]
        args = e29.parse_args()
        e29.run(args)
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
