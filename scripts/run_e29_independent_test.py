#!/usr/bin/env python3
"""Run only the locked Phase 7 E29 full-train independent-test stage."""

from __future__ import annotations

import json
import sys
import time

from _runner import DATA_DIR, RESULTS_DIR, enable_original_modules


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def validate_locked_oof(args, screen_path) -> dict:
    if not screen_path.is_file():
        raise FileNotFoundError(
            f"Missing passing OOF screen: {screen_path}. Run Phase 7 E29 OOF first."
        )
    payload = json.loads(screen_path.read_text(encoding="utf-8"))
    if not payload.get("screen", {}).get("passed", False):
        raise RuntimeError("The locked E29 OOF screen did not pass; test evaluation is refused.")
    if payload.get("candidate_prefix") != args.candidate_prefix:
        raise ValueError("Candidate prefix differs from the locked OOF experiment.")
    if payload.get("seeds") != args.seeds:
        raise ValueError("Seed list differs from the locked OOF experiment.")

    architecture = payload.get("architecture", {})
    expected_architecture = {
        "embedding_dim": args.embedding_dim,
        "kernel_sizes": args.kernel_sizes,
        "conv_channels": args.conv_channels,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }
    for key, value in expected_architecture.items():
        if architecture.get(key) != value:
            raise ValueError(f"Architecture setting {key} differs from the locked OOF experiment.")

    training = payload.get("training", {})
    expected_training = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "tissue_loss_weight": args.tissue_loss_weight,
        "hla_loss_weight": args.hla_loss_weight,
    }
    for key, value in expected_training.items():
        if training.get(key) != value:
            raise ValueError(f"Training setting {key} differs from the locked OOF experiment.")
    return payload


def main() -> None:
    enable_original_modules()
    import run_tissuepmhc_e29_multikernel_cnn_oof as e29

    root = RESULTS_DIR / "tissuePMHC_phase7_min200_e29_multikernel_cnn"
    e26_root = RESULTS_DIR / "tissuePMHC_phase7_min200_e26_greedy_ensemble_selection"
    e17_root = RESULTS_DIR / "tissuePMHC_phase7_min200_e17_seed_ensemble"
    screen_path = root / "oof_screen_summary.json"
    fixed_args = [
        "--train", str(DATA_DIR / "tissuePMHC_phase7_min200_train.csv.gz"),
        "--test", str(DATA_DIR / "tissuePMHC_phase7_min200_test.csv.gz"),
        "--experiment-name", "Phase7_min200_E29_multikernel_cnn_independent_test",
        "--candidate-prefix", "phase7_min200_e29_cnn",
        "--baseline-oof-predictions", str(e26_root / "oof_predictions.csv"),
        "--matching-baseline-candidate", "e14_final_3seed_mean",
        "--fusion-baseline-candidate", "e14_final_3seed_mean",
        "--oof-predictions-output", str(root / "oof_predictions.csv"),
        "--test-predictions-output", str(root / "test_predictions.csv"),
        "--task-comparison-output", str(root / "oof_task_comparison.csv"),
        "--screen-summary-output", str(screen_path),
        "--per-task-output", str(root / "per_task_metrics.csv"),
        "--summary-output", str(root / "summary_metrics.csv"),
        "--stability-output", str(root / "stability_metrics.csv"),
        "--e17-per-task", str(e17_root / "per_task_metrics.csv"),
        "--e17-comparison-output", str(root / "e17_3seed_comparison_metrics.csv"),
    ]
    previous_argv = sys.argv
    try:
        sys.argv = [previous_argv[0], *previous_argv[1:], *fixed_args]
        args = e29.parse_args()
    finally:
        sys.argv = previous_argv

    locked = validate_locked_oof(args, screen_path)
    print(
        "Locked OOF accepted: "
        f"AUROC={locked['screen']['cnn_oof']['mean_auroc']:.6f}, "
        f"passed={locked['screen']['passed']}",
        flush=True,
    )
    print("Starting full-train independent-test prediction (OOF will not be rerun).", flush=True)
    started = time.perf_counter()
    test_predictions = e29.generate_test(args)
    args.test_predictions_output.parent.mkdir(parents=True, exist_ok=True)
    test_predictions.to_csv(args.test_predictions_output, index=False)
    print(f"wrote: {args.test_predictions_output}", flush=True)
    e29.evaluate_existing_test_predictions(args)
    print(f"independent-test total time: {format_duration(time.perf_counter() - started)}", flush=True)


if __name__ == "__main__":
    main()
