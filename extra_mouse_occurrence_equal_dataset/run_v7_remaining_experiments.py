#!/usr/bin/env python3
"""Run only missing Mouse v7 experiments on the occurrence-equal dataset.

All generated files are isolated under ``results/v7_full_rerun``. Completed
occurrence-equal runs are documented as reused; old-dataset results are never
accepted as inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
TRANSFER_DIR = PROJECT_ROOT / "extra2" / "mouse_2human_experiments"
TRAIN = PROJECT_ROOT / "data" / "mousePMHC_occurence_equal_dataset" / "mousePMHC_train.csv.gz"
TEST = PROJECT_ROOT / "data" / "mousePMHC_occurence_equal_dataset" / "mousePMHC_test.csv.gz"
H2_PSEUDO = PROJECT_ROOT / "data" / "processed" / "h2_pseudo_sequences.csv"
OUTPUT_ROOT = HERE / "results" / "v7_full_rerun"
SEEDS = (20260704, 20260705, 20260706)

# Dependency order. These entries are missing from the already completed
# occurrence-equal E0/E2/E14a/E29 runs.
RUN_ORDER = (
    ("neural_single_conditioned", "run_neural_single_conditioned.py"),
    ("h2_pseudoseq", "run_h2_pseudoseq.py"),
    ("h2_hybrid", "run_h2_hybrid.py"),
    ("tissue_grouping", "run_tissue_grouping.py"),
    ("selective_grouping", "run_selective_grouping.py"),
    ("adaptive_soft_ensemble", "run_adaptive_soft_ensemble.py"),
    ("cagrad", "run_cagrad.py"),
    ("mmoe_tuning", "run_mmoe_tuning.py"),
    ("dbmtl", "run_dbmtl.py"),
    ("auxiliary_soft", "run_auxiliary_soft.py"),
    ("mlp_dual_seed_ensemble", "run_mlp_dual_seed_ensemble.py"),
)

REUSED_NEW_DATA = {
    "e0_traditional": HERE / "results" / "e0_traditional",
    "e2_shared_heads": HERE / "results" / "e2_shared_heads",
    "e14a_auxiliary_dual_branch": HERE / "results" / "e14a_auxiliary_dual_branch",
    "e29_multikernel_cnn": HERE / "results" / "e29_multikernel_cnn",
    "external_predictors": HERE / "results" / "external_predictors",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed(name: str) -> bool:
    target = OUTPUT_ROOT / name
    return (target / "transfer_contract.json").is_file() and (target / "summary_metrics.csv").is_file()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def build_command(wrapper: Path, args: argparse.Namespace, experiment: str | None = None) -> list[str]:
    command = [
        sys.executable, str(wrapper),
        "--train", str(TRAIN.resolve()),
        "--test", str(TEST.resolve()),
        "--output-root", str(OUTPUT_ROOT.resolve()),
        "--device", args.device,
        "--seeds", *(str(seed) for seed in SEEDS),
        "--max-tasks", "0",
        "--h2-pseudo-sequences", str(H2_PSEUDO.resolve()),
    ]
    if args.epochs is not None:
        command.extend(("--epochs", str(args.epochs)))
    target = OUTPUT_ROOT / experiment if experiment else None
    if args.overwrite or (target is not None and target.is_dir() and not completed(experiment)):
        command.append("--overwrite")
    return command


def stream(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (TRAIN, TEST, H2_PSEUDO):
        if not path.is_file():
            raise FileNotFoundError(path)

    reuse = []
    for name, path in REUSED_NEW_DATA.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Expected occurrence-equal result is missing: {path}")
        reuse.append({"experiment": name, "path": str(path.resolve()), "status": "reused_occurrence_equal"})

    manifest = {
        "suite": "mouse_v7_occurrence_equal_only",
        "started_utc": utc_now(),
        "finished_utc": None,
        "train": str(TRAIN.resolve()),
        "test": str(TEST.resolve()),
        "train_sha256": sha256(TRAIN),
        "test_sha256": sha256(TEST),
        "h2_pseudo_sequences": str(H2_PSEUDO.resolve()),
        "h2_pseudo_sequences_sha256": sha256(H2_PSEUDO),
        "seeds": list(SEEDS),
        "reused": reuse,
        "not_run": [
            {"experiment": "tissuepmhc_full", "reason": "replaced by completed occurrence-equal e29_multikernel_cnn"},
        ],
        "entries": [],
    }
    if args.dry_run:
        for name, filename in RUN_ORDER:
            print(" ".join(build_command(TRANSFER_DIR / filename, args, name)))
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = OUTPUT_ROOT / "run_manifest.json"
    save_json(status_path, manifest)
    timings = []
    suite_start = time.perf_counter()
    failed = False
    for name, filename in RUN_ORDER:
        entry = {"experiment": name, "started_utc": utc_now(), "status": "pending"}
        manifest["entries"].append(entry)
        if completed(name) and not args.overwrite:
            entry.update(status="skipped_completed", finished_utc=utc_now())
            print(f"[SKIP completed occurrence-equal] {name}", flush=True)
            save_json(status_path, manifest)
            continue
        command = build_command(TRANSFER_DIR / filename, args, name)
        entry.update(status="running", command=command)
        save_json(status_path, manifest)
        started = time.perf_counter()
        print(f"[RUN] {name} seeds={list(SEEDS)}", flush=True)
        return_code = stream(command, OUTPUT_ROOT / "run_logs" / f"{name}.log")
        elapsed = time.perf_counter() - started
        timings.append({"experiment": name, "seconds": f"{elapsed:.6f}", "return_code": return_code})
        entry.update(
            status="completed" if return_code == 0 else "failed",
            return_code=return_code,
            seconds=elapsed,
            finished_utc=utc_now(),
        )
        save_json(status_path, manifest)
        if return_code != 0:
            failed = True
            if not args.continue_on_error:
                break

    total = time.perf_counter() - suite_start
    timings.append({"experiment": "TOTAL", "seconds": f"{total:.6f}", "return_code": int(failed)})
    with (OUTPUT_ROOT / "orchestration_timing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("experiment", "seconds", "return_code"))
        writer.writeheader()
        writer.writerows(timings)
    manifest.update(finished_utc=utc_now(), total_seconds=total, status="failed" if failed else "completed")
    save_json(status_path, manifest)
    print(f"suite total time: {total:.2f}s", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
