#!/usr/bin/env python3
"""Run every missing human-to-mouse method transfer in dependency order."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from mouse_transfer_common import (
    DEFAULT_H2_PSEUDO,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEEDS,
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    EXPERIMENTS,
    PROJECT_ROOT,
    resolved,
)


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
    ("tissuepmhc_full", "run_tissuepmhc_full.py"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all 12 mouse transfer entries sequentially. The entries cover "
            "all 16 human methods not already present in historical mouse work."
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--h2-pseudo-sequences", type=Path, default=DEFAULT_H2_PSEUDO)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip entries that already have a contract and summary_metrics.csv.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed entry and continue with later independent entries.",
    )
    parser.add_argument(
        "--require-h2-pseudo",
        action="store_true",
        help="Fail before starting if the sourced H2 pseudo-sequence file is absent.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow individual entries to reuse nonempty output folders.",
    )
    return parser.parse_args()


def is_complete(output_root: Path, name: str) -> bool:
    target = output_root / name
    return (
        (target / "transfer_contract.json").is_file()
        and (target / "summary_metrics.csv").is_file()
    )


def build_command(
    wrapper: Path,
    args: argparse.Namespace,
    output_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(wrapper),
        "--train",
        str(resolved(args.train)),
        "--test",
        str(resolved(args.test)),
        "--output-root",
        str(output_root),
        "--device",
        args.device,
        "--seeds",
        *(str(seed) for seed in args.seeds),
        "--max-tasks",
        str(args.max_tasks),
        "--h2-pseudo-sequences",
        str(resolved(args.h2_pseudo_sequences)),
    ]
    if args.epochs is not None:
        command.extend(("--epochs", str(args.epochs)))
    if args.dry_run:
        command.append("--dry-run")
    if args.overwrite:
        command.append("--overwrite")
    return command


def save_status(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    output_root = resolved(args.output_root)
    h2_pseudo = resolved(args.h2_pseudo_sequences)
    suite_dir = Path(__file__).resolve().parent

    missing_inputs = [
        path for path in (resolved(args.train), resolved(args.test)) if not path.is_file()
    ]
    if not args.dry_run and missing_inputs:
        raise FileNotFoundError(
            "Missing mouse input file(s):\n"
            + "\n".join(f"  {path}" for path in missing_inputs)
        )
    if not args.dry_run and args.require_h2_pseudo and not h2_pseudo.is_file():
        raise FileNotFoundError(
            f"Missing experimentally sourced H2 pseudo-sequences: {h2_pseudo}"
        )

    status = {
        "suite": "mouse_human_method_transfer_all_in_one",
        "project_root": str(PROJECT_ROOT),
        "output_root": str(output_root),
        "started_utc": utc_now(),
        "finished_utc": None,
        "dry_run": args.dry_run,
        "run_order": [name for name, _ in RUN_ORDER],
        "entries": [],
    }
    status_path = output_root / "all_in_one_status.json"
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        save_status(status_path, status)

    failed = False
    for name, filename in RUN_ORDER:
        experiment = EXPERIMENTS[name]
        wrapper = suite_dir / filename
        entry = {
            "experiment": name,
            "human_methods": list(experiment.human_methods),
            "wrapper": str(wrapper),
            "started_utc": utc_now(),
            "finished_utc": None,
            "status": "pending",
            "return_code": None,
        }
        status["entries"].append(entry)

        if not args.dry_run and args.resume and is_complete(output_root, name):
            entry["status"] = "skipped_completed"
            entry["finished_utc"] = utc_now()
            save_status(status_path, status)
            print(f"[SKIP completed] {name}", flush=True)
            continue

        if (
            not args.dry_run
            and experiment.requires_h2_pseudo
            and not h2_pseudo.is_file()
        ):
            entry["status"] = "skipped_missing_h2_pseudo"
            entry["finished_utc"] = utc_now()
            entry["message"] = (
                "Provide a sourced H2 pseudo-sequence file and rerun; "
                "fabricated sequences are not permitted."
            )
            save_status(status_path, status)
            print(f"[SKIP missing H2 pseudo-sequences] {name}", flush=True)
            continue

        command = build_command(wrapper, args, output_root)
        entry["command"] = command
        entry["status"] = "running"
        if not args.dry_run:
            save_status(status_path, status)
        print(f"[RUN] {name}", flush=True)

        try:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        except KeyboardInterrupt:
            entry["status"] = "interrupted"
            entry["finished_utc"] = utc_now()
            if not args.dry_run:
                save_status(status_path, status)
            raise

        entry["return_code"] = completed.returncode
        entry["finished_utc"] = utc_now()
        if completed.returncode == 0:
            entry["status"] = "dry_run_ok" if args.dry_run else "completed"
            if not args.dry_run:
                save_status(status_path, status)
            print(f"[OK] {name}", flush=True)
            continue

        entry["status"] = "failed"
        failed = True
        if not args.dry_run:
            save_status(status_path, status)
        print(
            f"[FAILED rc={completed.returncode}] {name}",
            file=sys.stderr,
            flush=True,
        )
        if not args.continue_on_error:
            break

    status["finished_utc"] = utc_now()
    skipped_h2 = any(
        entry["status"] == "skipped_missing_h2_pseudo"
        for entry in status["entries"]
    )
    status["result"] = (
        "failed"
        if failed
        else "completed_with_missing_h2_skips"
        if skipped_h2
        else "completed"
    )
    if not args.dry_run:
        save_status(status_path, status)
        print(f"Status manifest: {status_path}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
