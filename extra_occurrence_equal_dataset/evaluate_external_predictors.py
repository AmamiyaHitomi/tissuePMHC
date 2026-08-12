from pathlib import Path

import pandas as pd

import common
from _runner import run


_read_csv = pd.read_csv


def _read_csv_with_tissue_na(path, *args, **kwargs):
    frame = _read_csv(path, *args, **kwargs)
    try:
        is_test = Path(path).resolve() == common.TEST_PATH.resolve()
    except TypeError:
        is_test = False
    if is_test and "target_tissue" in frame:
        frame["target_tissue"] = frame["target_tissue"].fillna("NA")
    return frame


pd.read_csv = _read_csv_with_tissue_na
try:
    run("evaluate_external_predictors.py")
finally:
    pd.read_csv = _read_csv
