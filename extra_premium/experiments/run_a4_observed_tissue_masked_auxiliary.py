#!/usr/bin/env python3
"""A4: E14a train-only observed-tissue multi-label loss with unknown masking."""

from tissue_aux_ablation import run_experiment


if __name__ == "__main__":
    run_experiment("observed_tissue_masked")

