#!/usr/bin/env python3
"""Validate both frozen issue-9 datasets and fold manifests without training."""

from __future__ import annotations

import json

try:
    from .common import ROOT, load_frozen_folds, read_training_data
except ImportError:
    from common import ROOT, load_frozen_folds, read_training_data


def main() -> None:
    specifications = {
        "human": (
            ROOT
            / "data"
            / "tissuePMHC_phase7_min200"
            / "tissuePMHC_phase7_min200_train.csv.gz",
            ROOT
            / "results"
            / "tissuePMHC_phase7_min200_e31_peptide_disjoint_oof"
            / "pair_fold_assignments.csv",
        ),
        "mouse": (
            ROOT / "data" / "mousePMHC" / "mousePMHC_train.csv.gz",
            ROOT
            / "results"
            / "mousePMHC_phase6_e33_peptide_disjoint_oof"
            / "mousePMHC_phase6_e33_pair_fold_assignments.csv",
        ),
    }
    audits = {}
    for species, (data_path, manifest_path) in specifications.items():
        data = read_training_data(data_path, species)
        _, _, audit = load_frozen_folds(data, manifest_path, 3)
        audits[species] = audit
    print(json.dumps(audits, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
