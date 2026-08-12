#!/usr/bin/env python3
"""Run migrated E4 HLA pseudo-sequence model from the extra2 suite."""

from migrate_44tasks_to_157tasks import run_single_experiment_entry


if __name__ == "__main__":
    run_single_experiment_entry("e4_hla_pseudoseq")
