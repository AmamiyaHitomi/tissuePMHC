#!/usr/bin/env python3
"""A2: E14a with query-tissue auxiliary supervision on positive rows only."""

from tissue_aux_ablation import run_experiment


if __name__ == "__main__":
    run_experiment("positive_only_tissue")

