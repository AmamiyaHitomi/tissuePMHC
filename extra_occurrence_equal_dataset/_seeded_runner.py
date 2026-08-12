"""Seed-aware adapter for the unchanged single-seed premium runners."""

from __future__ import annotations

import argparse
import sys

import common
from _runner import run


def run_seeded(source_name: str) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, default=common.SEED)
    seed_args, remaining = parser.parse_known_args()
    if seed_args.seed < 0:
        raise ValueError("--seed must be non-negative")

    seed = int(seed_args.seed)
    seed_root = common.EXPERIMENT_ROOT / "results" / "seed_runs" / str(seed)
    common.SEED = seed
    common.RESULTS_ROOT = seed_root
    # Functions re-exported from the premium module retain that module's
    # globals, so update it as well.
    common._MODULE.SEED = seed
    common._MODULE.RESULTS_ROOT = seed_root
    sys.argv = [sys.argv[0], *remaining]
    run(source_name)

