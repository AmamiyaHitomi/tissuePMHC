#!/usr/bin/env python3
"""Run one isolated Human occurrence-equal experiment."""

from human_experiment_common import entry_parser, run_entry


if __name__ == "__main__":
    arguments = entry_parser().parse_args()
    run_entry(arguments.experiment, arguments)

