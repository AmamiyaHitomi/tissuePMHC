"""Mouse occurrence-equal configuration for the frozen premium basic runners."""

from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = PROJECT_ROOT / "extra_premium" / "common.py"
_SPEC = importlib.util.spec_from_file_location("_mouse_occurrence_basic_common", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load premium common module: {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_MODULE.EXPERIMENT_ROOT = Path(__file__).resolve().parent
_MODULE.DATA_DIR = PROJECT_ROOT / "data" / "mousePMHC_occurence_equal_dataset"
_MODULE.TRAIN_PATH = _MODULE.DATA_DIR / "mousePMHC_train.csv.gz"
_MODULE.TEST_PATH = _MODULE.DATA_DIR / "mousePMHC_test.csv.gz"
_MODULE.RESULTS_ROOT = _MODULE.EXPERIMENT_ROOT / "results"
_MODULE.EXTERNAL_ROOT = _MODULE.EXPERIMENT_ROOT / "external"
TIMING_RESULTS_PATH = _MODULE.RESULTS_ROOT / "timing_results.csv"

for _name in dir(_MODULE):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_MODULE, _name)


def load_premium_data(base):
    """Load and strictly audit the mouse occurrence-equal fixed split."""
    train = base.read_dataset(TRAIN_PATH)
    test = base.read_dataset(TEST_PATH)

    for split_name, frame in (("train", train), ("test", test)):
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{split_name} is missing required columns: {sorted(missing)}")
        if set(frame["dataset"].astype(str)) != {"mousePMHC"}:
            raise ValueError(f"{split_name} contains a non-mousePMHC dataset value.")
        if set(frame["split"].astype(str)) != {split_name}:
            raise ValueError(f"{split_name} contains an unexpected split value.")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"{split_name} sample_id values must be unique.")
        if set(frame["label"].astype(int)) != {0, 1}:
            raise ValueError(f"{split_name} labels must contain both 0 and 1.")
        if not frame["mhc_restriction"].astype(str).str.startswith("H2-").all():
            raise ValueError(f"{split_name} contains a non-H2 restriction.")
        if not frame["peptide_sequence"].astype(str).str.fullmatch(
            r"[ACDEFGHIKLMNPQRSTVWY]{9}"
        ).all():
            raise ValueError(f"{split_name} must contain only canonical 9-mer peptides.")

        pair_check = frame.groupby("pair_id", sort=False).agg(
            pair_size=("label", "size"),
            label_sum=("label", "sum"),
            label_count=("label", "nunique"),
            tissue_count=("target_tissue", "nunique"),
            mhc_count=("mhc_restriction", "nunique"),
        )
        valid_pairs = (
            pair_check["pair_size"].eq(2)
            & pair_check["label_sum"].eq(1)
            & pair_check["label_count"].eq(2)
            & pair_check["tissue_count"].eq(1)
            & pair_check["mhc_count"].eq(1)
        )
        if not valid_pairs.all():
            raise ValueError(
                f"Every {split_name} pair must contain one positive and one negative "
                "for one tissue-H2 task."
            )

    pair_overlap = set(train["pair_id"].astype(str)) & set(test["pair_id"].astype(str))
    if pair_overlap:
        raise ValueError(f"Train/test pair_id leakage: {len(pair_overlap)} pairs.")
    task_peptide_columns = ["target_tissue", "mhc_restriction", "peptide_sequence"]
    train_examples = set(map(tuple, train[task_peptide_columns].astype(str).to_numpy()))
    test_examples = set(map(tuple, test[task_peptide_columns].astype(str).to_numpy()))
    if train_examples & test_examples:
        raise ValueError("Train/test exact tissue-H2-peptide leakage detected.")

    original_train_rows = len(train)
    original_test_rows = len(test)
    train, test, mappings = base.add_task_columns(train, test)
    if len(train) != original_train_rows or len(test) != original_test_rows:
        raise ValueError("Some rows were dropped because their task was absent from one split.")
    if len(mappings["tasks"]) != 11:
        raise ValueError(f"Expected 11 mouse tasks, found {len(mappings['tasks'])}.")

    peptide_length = int(
        max(train["peptide_sequence"].str.len().max(), test["peptide_sequence"].str.len().max())
    )
    return train, test, mappings, peptide_length


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
    """Forward to the shared saver with the runtime seed."""
    runtime_seed = int(_MODULE.SEED if seed is None else seed)
    return _MODULE.save_basic_test_results(
        model_name, train, test, scores, base, settings, seed=runtime_seed
    )
