#!/usr/bin/env python3
"""A3: E14a with train-only observed-tissue multi-label auxiliary supervision."""

from tissue_aux_ablation import run_experiment


if __name__ == "__main__":
    run_experiment("observed_tissue_multilabel")

