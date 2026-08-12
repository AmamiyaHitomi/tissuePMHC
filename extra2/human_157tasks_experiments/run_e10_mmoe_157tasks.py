#!/usr/bin/env python3
"""Run migrated E10 MMoE from the extra2 experiment suite."""

from migrate_44tasks_to_157tasks import run_single_experiment_entry


if __name__ == "__main__":
    run_single_experiment_entry("e10_mmoe")
