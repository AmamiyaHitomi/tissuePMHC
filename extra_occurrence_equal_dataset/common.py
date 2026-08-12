"""Dataset-local configuration layered on the premium basic runners."""

from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = PROJECT_ROOT / "extra_premium" / "common.py"
_SPEC = importlib.util.spec_from_file_location("_premium_basic_common", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load premium common module: {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

# Override all path globals in the source module. Functions copied below retain
# this module as their globals, so they also see these isolated paths.
_MODULE.EXPERIMENT_ROOT = Path(__file__).resolve().parent
_MODULE.DATA_DIR = PROJECT_ROOT / "data" / "humanPMHC_occurence_equal_dataset"
_MODULE.TRAIN_PATH = _MODULE.DATA_DIR / "humanPMHC_train.csv.gz"
_MODULE.TEST_PATH = _MODULE.DATA_DIR / "humanPMHC_test.csv.gz"
_MODULE.RESULTS_ROOT = _MODULE.EXPERIMENT_ROOT / "results"
_MODULE.EXTERNAL_ROOT = _MODULE.EXPERIMENT_ROOT / "external"
TIMING_RESULTS_PATH = _MODULE.RESULTS_ROOT / "timing_results.csv"

for _name in dir(_MODULE):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_MODULE, _name)


def load_premium_data(base):
    """Load this dataset while preserving its literal ``NA`` tissue category.

    The CSV uses ``NA`` as a real target-tissue label. Pandas' default NA
    vocabulary converts it to NaN, which makes task-name sorting fail. Only
    this task-defining column is restored; missing values in metadata columns
    retain the source runner's normal parsing semantics.
    """
    original_read_dataset = base.read_dataset

    def read_dataset_with_tissue_na(path):
        frame = original_read_dataset(path)
        frame["target_tissue"] = frame["target_tissue"].fillna("NA")
        return frame

    base.read_dataset = read_dataset_with_tissue_na
    try:
        return _MODULE.load_premium_data(base)
    finally:
        base.read_dataset = original_read_dataset


def save_basic_test_results(
    model_name,
    train,
    test,
    scores,
    base,
    settings,
    *,
    seed=None,
):
    """Forward to the premium saver with the runtime seed, not its bound default."""
    runtime_seed = int(_MODULE.SEED if seed is None else seed)
    return _MODULE.save_basic_test_results(
        model_name,
        train,
        test,
        scores,
        base,
        settings,
        seed=runtime_seed,
    )
