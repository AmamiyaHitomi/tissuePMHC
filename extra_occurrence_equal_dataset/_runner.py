"""Run one unchanged premium entry point with dataset-local configuration."""

from __future__ import annotations

import csv
import runpy
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import common


def _timed(entry_name: str, action) -> None:
    started_at = datetime.now(timezone.utc).astimezone().isoformat()
    started = time.perf_counter()
    status = "failed"
    print(f"seed={common.SEED} entry={entry_name} started_at={started_at}", flush=True)
    try:
        action()
        status = "completed"
    finally:
        elapsed = time.perf_counter() - started
        timing_path = getattr(
            common,
            "TIMING_RESULTS_PATH",
            common.RESULTS_ROOT / "timing_results.csv",
        )
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not timing_path.exists()
        with timing_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["entry", "seed", "started_at", "elapsed_seconds", "status"],
            )
            if new_file:
                writer.writeheader()
            writer.writerow(
                {
                    "entry": entry_name,
                    "seed": common.SEED,
                    "started_at": started_at,
                    "elapsed_seconds": f"{elapsed:.6f}",
                    "status": status,
                }
            )
        print(
            f"total entry={entry_name} seed={common.SEED} "
            f"elapsed_seconds={elapsed:.3f} status={status}",
            flush=True,
        )


def run(source_name: str) -> None:
    source = common.PROJECT_ROOT / "extra_premium" / source_name

    def action() -> None:
        # The unchanged source runner imports ``common`` by name.
        sys.modules["common"] = common
        runpy.run_path(str(source), run_name="__main__")

    _timed(source_name, action)


def run_callable(entry_name: str, action) -> None:
    _timed(entry_name, action)
