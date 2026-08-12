from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ISSUE9 = Path(__file__).resolve().parents[1]
if str(ISSUE9) not in sys.path:
    sys.path.insert(0, str(ISSUE9))

from analyze import bh_adjust, hodges_lehmann_one_sample  # noqa: E402
from common import ensemble_predictions, pair_accuracy  # noqa: E402


def test_pair_accuracy_strict_and_half_ties() -> None:
    frame = pd.DataFrame(
        {
            "pair_id": ["a", "a", "b", "b", "c", "c"],
            "label": [1, 0, 1, 0, 1, 0],
            "score": [0.8, 0.2, 0.2, 0.7, 0.5, 0.5],
        }
    )
    result = pair_accuracy(frame)
    assert result["n_pairs"] == 3
    assert result["pair_wins"] == 1
    assert result["pair_ties"] == 1
    assert result["pair_losses"] == 1
    assert result["pair_acc"] == 1 / 3
    assert result["pair_acc_half_ties"] == 0.5


def test_ensemble_is_probability_mean() -> None:
    rows = []
    for seed, score in [(1, 0.2), (2, 0.8)]:
        rows.append(
            {
                "species": "mouse",
                "model": "m",
                "seed": seed,
                "fold": 0,
                "sample_id": "s",
                "pair_id": "p",
                "target_tissue": "t",
                "mhc_restriction": "H2-X",
                "peptide_sequence": "AAAAAAAAA",
                "label": 1,
                "score": score,
            }
        )
    result = ensemble_predictions(pd.DataFrame(rows))
    assert len(result) == 1
    assert result.loc[0, "score"] == 0.5
    assert result.loc[0, "n_members"] == 2


def test_hodges_lehmann_uses_walsh_averages() -> None:
    assert hodges_lehmann_one_sample(np.array([1.0, 2.0, 3.0])) == 2.0


def test_bh_adjustment_is_monotone_in_rank() -> None:
    p = pd.Series([0.01, 0.04, 0.03, 0.20])
    adjusted = bh_adjust(p)
    assert np.all((adjusted >= p.to_numpy()) & (adjusted <= 1.0))
    order = np.argsort(p.to_numpy())
    assert np.all(np.diff(adjusted[order]) >= -1e-15)
