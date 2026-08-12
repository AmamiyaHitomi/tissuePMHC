#!/usr/bin/env python3
"""Run the human 157-task Table-5 suite plus Selected Factorized MMoE."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from migrate_44tasks_to_157tasks import (
    DEFAULT_OUTPUT_ROOT,
DEFAULT_SEEDS,
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    EXPERIMENTS,
    PROJECT_ROOT,
    TABLE5_EXPERIMENTS,
    child_command,
    dependency_closure,
    resolved,
)

SELECTED_FACTORIZED = "e15_selected_factorized_mmoe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the migration set from "
            "migrate_44tasks_to_157tasks.py --preset table5, followed by the "
            "human Selected Factorized MMoE transfer."
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional common epoch override; default uses each runner's setting.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help=(
            "Smoke-test task limit for legacy migrations where supported; "
            "the Selected Factorized MMoE entry always requires all 157 tasks."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow nonempty target experiment directories after manual review.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip completed entries and retry entries without summary_metrics.csv.",
    )
    parser.add_argument(
        "--skip-experiments",
        nargs="+",
        choices=[*TABLE5_EXPERIMENTS, SELECTED_FACTORIZED],
        default=[],
        help="Skip additional optional Table-5 experiments.",
    )
    parser.add_argument(
        "--include-cagrad",
        action="store_true",
        help="Include E9 CAGrad; it is skipped by default because of its cost.",
    )
    args = parser.parse_args()

    # child_command() consumes the same namespace as the general migration CLI.
    # These fixed values prevent this dedicated entry from selecting the 25-item
    # "all" preset or an arbitrary subset.
    args.preset = "table5"
    args.experiments = None
    args.list = False
    args._run_one = None
    return args


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        missing = [
            path
            for path in (resolved(args.train), resolved(args.test))
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing Phase-7 input file(s):\n"
                + "\n".join(f"  {path}" for path in missing)
            )

    skipped = set(args.skip_experiments)
    if not args.include_cagrad:
        skipped.add("e9_cagrad")
    requested = [name for name in TABLE5_EXPERIMENTS if name not in skipped]
    selected = dependency_closure(requested)
    if set(selected) != set(requested):
        unexpected = sorted(set(selected) - set(requested))
        raise RuntimeError(
            "Cannot skip experiments required by the remaining run: "
            + ", ".join(unexpected)
        )

    print("Human 157-task Table-5 all-in-one order:")
    if skipped:
        print("  skipped: " + ", ".join(sorted(skipped)))
    for index, name in enumerate(selected, start=1):
        print(f"  {index:02d}. {name}: {EXPERIMENTS[name].description}")
    if SELECTED_FACTORIZED not in skipped:
        print(
            f"  {len(selected) + 1:02d}. {SELECTED_FACTORIZED}: "
            "mouse-selected task-balanced Factorized MMoE."
        )
    print(f"Isolated output root: {resolved(args.output_root)}")

    for name in selected:
        target_dir = resolved(args.output_root) / name
        complete = (
            (target_dir / "migration_contract.json").is_file()
            and (target_dir / "summary_metrics.csv").is_file()
        )
        if not args.dry_run and args.resume and complete:
            print(f"\n=== {name} ===")
            print("[SKIP completed]", flush=True)
            continue

        print(f"\n=== {name} ===", flush=True)
        command = child_command(args, name)
        incomplete_nonempty = (
            not args.dry_run
            and args.resume
            and target_dir.is_dir()
            and any(target_dir.iterdir())
            and not complete
        )
        if incomplete_nonempty and "--overwrite" not in command:
            command.append("--overwrite")
            print("[RETRY incomplete output directory]", flush=True)
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
        )
        if completed.returncode != 0:
            print(
                f"Stopped after {name} failed with return code "
                f"{completed.returncode}.",
                file=sys.stderr,
            )
            return completed.returncode

    if SELECTED_FACTORIZED not in skipped:
        name = SELECTED_FACTORIZED
        target_dir = resolved(args.output_root) / name
        complete = (
            (target_dir / "migration_contract.json").is_file()
            and (target_dir / "summary_metrics.csv").is_file()
        )
        print(f"\n=== {name} ===", flush=True)
        if not args.dry_run and args.resume and complete:
            print("[SKIP completed]", flush=True)
            return 0
        command = [
            sys.executable,
            str(
                Path(__file__).resolve().with_name(
                    "run_selected_factorized_mmoe_157tasks.py"
                )
            ),
            "--train",
            str(resolved(args.train)),
            "--test",
            str(resolved(args.test)),
            "--output-dir",
            str(target_dir),
            "--device",
            args.device,
            "--seeds",
            *[str(seed) for seed in args.seeds],
        ]
        if args.epochs is not None:
            command.extend(["--epochs", str(args.epochs)])
        if args.dry_run:
            command.append("--dry-run")
        if not args.resume:
            command.append("--no-resume")
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            print(
                f"Stopped after {name} failed with return code "
                f"{completed.returncode}.",
                file=sys.stderr,
            )
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
