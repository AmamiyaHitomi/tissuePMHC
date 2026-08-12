#!/usr/bin/env python3
"""A0: reproduce current E14a HLA + query-target-tissue auxiliary supervision."""

from tissue_aux_ablation import run_experiment


if __name__ == "__main__":
    run_experiment("current")

