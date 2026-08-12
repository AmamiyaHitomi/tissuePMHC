#!/usr/bin/env python3
"""Run the human E2 shared-head baseline on the isolated Phase 7 min200 data."""

from __future__ import annotations

import sys
import time
from typing import Any

from _runner import DATA_DIR, RESULTS_DIR, enable_original_modules


def format_duration(seconds: float) -> str:
    minutes, remainder = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {remainder:04.1f}s"
    if minutes:
        return f"{minutes:d}m {remainder:04.1f}s"
    return f"{remainder:.1f}s"


def synchronize_cuda(torch: Any, device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def install_terminal_timers(e2: Any) -> tuple[Any, Any]:
    """Add Phase 7-only timing logs without changing saved result schemas."""
    original_train = e2.train_binary_model
    original_run_one = e2.run_one_experiment

    def timed_train_binary_model(
        torch: Any,
        model: Any,
        loader: Any,
        optimizer: Any,
        loss_fn: Any,
        device: str,
        model_kind: str,
        epochs: int,
    ) -> None:
        model.train()
        training_started = time.perf_counter()
        for epoch_index in range(1, epochs + 1):
            synchronize_cuda(torch, device)
            epoch_started = time.perf_counter()
            print(
                f"    epoch {epoch_index:02d}/{epochs:02d} started "
                f"(batches={len(loader)})",
                flush=True,
            )
            for batch in loader:
                batch = [item.to(device) for item in batch]
                optimizer.zero_grad()
                if model_kind == "single":
                    logits = model(batch[0])
                    labels = batch[1].float()
                elif model_kind == "task_heads":
                    logits = model(batch[0], batch[1])
                    labels = batch[2].float()
                elif model_kind == "conditioned":
                    logits = model(batch[0], batch[1], batch[2])
                    labels = batch[3].float()
                else:
                    raise ValueError(f"Unknown model_kind: {model_kind}")
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
            synchronize_cuda(torch, device)
            epoch_seconds = time.perf_counter() - epoch_started
            total_seconds = time.perf_counter() - training_started
            print(
                f"    epoch {epoch_index:02d}/{epochs:02d} finished: "
                f"epoch_time={format_duration(epoch_seconds)}, "
                f"training_elapsed={format_duration(total_seconds)}",
                flush=True,
            )

    def timed_run_one_experiment(*args: Any, **kwargs: Any):
        torch = args[1] if len(args) > 1 else kwargs["torch"]
        device = args[9] if len(args) > 9 else kwargs["device"]
        experiment = args[10] if len(args) > 10 else kwargs["experiment"]
        seed = int(experiment["seed"])
        synchronize_cuda(torch, device)
        seed_started = time.perf_counter()
        print(f"  seed {seed} started", flush=True)
        try:
            return original_run_one(*args, **kwargs)
        finally:
            synchronize_cuda(torch, device)
            print(
                f"  seed {seed} finished: seed_time={format_duration(time.perf_counter() - seed_started)}",
                flush=True,
            )

    e2.train_binary_model = timed_train_binary_model
    e2.run_one_experiment = timed_run_one_experiment
    return original_train, original_run_one


def main() -> None:
    enable_original_modules()
    import run_tissuepmhc_neural_baselines_v2 as e2

    root = RESULTS_DIR / "tissuePMHC_phase7_min200_e2_shared_heads"
    fixed_args = [
        "--train", str(DATA_DIR / "tissuePMHC_phase7_min200_train.csv.gz"),
        "--test", str(DATA_DIR / "tissuePMHC_phase7_min200_test.csv.gz"),
        "--experiment-plan", "custom",
        "--models", "shared_peptide_encoder_task_heads",
        "--per-task-output", str(root / "per_task_metrics.csv"),
        "--summary-output", str(root / "summary_metrics.csv"),
        "--stability-output", str(root / "stability_metrics.csv"),
        "--metadata-output", str(root / "metadata.json"),
    ]
    previous_argv = sys.argv
    original_train, original_run_one = install_terminal_timers(e2)
    total_started = time.perf_counter()
    try:
        sys.argv = [previous_argv[0], *previous_argv[1:], *fixed_args]
        e2.run(e2.parse_args())
    finally:
        e2.train_binary_model = original_train
        e2.run_one_experiment = original_run_one
        sys.argv = previous_argv
        print(
            f"Phase 7 E2 total_time={format_duration(time.perf_counter() - total_started)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
