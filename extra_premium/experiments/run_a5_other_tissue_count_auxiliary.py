#!/usr/bin/env python3
"""A5: E14a with train-only other-tissue-count auxiliary classification."""

from tissue_aux_ablation import run_experiment


if __name__ == "__main__":
    run_experiment("other_tissue_count")

