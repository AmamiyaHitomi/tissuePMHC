#!/usr/bin/env python3
"""Run migrated E1/E2/E3 neural baselines from the extra2 experiment suite."""

from migrate_44tasks_to_157tasks import run_single_experiment_entry


if __name__ == "__main__":
    run_single_experiment_entry("e1_e2_e3_neural")
