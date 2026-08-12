"""Run E0/E2/E14a/E29 for the three standard seeds, with wall-clock timing."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


SEEDS = (20260704, 20260705, 20260706)
ENTRIES = (
    "run_e0_traditional.py",
    "run_e2_shared_heads.py",
    "run_e14a_auxiliary_dual_branch.py",
    "run_e29_multikernel_cnn.py",
)
ROOT = Path(__file__).resolve().parent
TIMING_PATH = ROOT / "results" / "orchestration_timing.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    overall_started = time.perf_counter()
    rows = []
    for seed in SEEDS:
        seed_started = time.perf_counter()
        print(f"seed {seed} started", flush=True)
        for entry in ENTRIES:
            entry_started_at = datetime.now(timezone.utc).astimezone().isoformat()
            entry_started = time.perf_counter()
            command = [
                sys.executable,
                str(ROOT / entry),
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
            ]
            completed = subprocess.run(command, cwd=ROOT.parent, check=False)
            elapsed = time.perf_counter() - entry_started
            status = "completed" if completed.returncode == 0 else "failed"
            rows.append(
                {
                    "scope": "entry",
                    "entry": entry,
                    "seed": seed,
                    "started_at": entry_started_at,
                    "elapsed_seconds": f"{elapsed:.6f}",
                    "status": status,
                }
            )
            if completed.returncode != 0:
                write_timing(rows)
                raise SystemExit(completed.returncode)
        seed_elapsed = time.perf_counter() - seed_started
        rows.append(
            {
                "scope": "seed",
                "entry": "all_basic_models",
                "seed": seed,
                "started_at": "",
                "elapsed_seconds": f"{seed_elapsed:.6f}",
                "status": "completed",
            }
        )
        print(f"seed {seed} total elapsed_seconds={seed_elapsed:.3f}", flush=True)

    overall_elapsed = time.perf_counter() - overall_started
    rows.append(
        {
            "scope": "overall",
            "entry": "all_models_all_seeds",
            "seed": "",
            "started_at": "",
            "elapsed_seconds": f"{overall_elapsed:.6f}",
            "status": "completed",
        }
    )
    write_timing(rows)
    print(f"all seeds total elapsed_seconds={overall_elapsed:.3f}", flush=True)


def write_timing(rows: list[dict[str, object]]) -> None:
    TIMING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TIMING_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scope", "entry", "seed", "started_at", "elapsed_seconds", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
