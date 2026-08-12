#!/usr/bin/env python3
"""A1: E14a with HLA auxiliary supervision only; query-tissue auxiliary is removed."""

from tissue_aux_ablation import run_experiment


if __name__ == "__main__":
    run_experiment("hla_only")

