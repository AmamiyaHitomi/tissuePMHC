"""Shared helpers for Phase 7 isolated human benchmark entry points."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data" / "tissuePMHC_phase7_min200"
RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_NAME = "tissuePMHC_phase7_min200"
MIN_PAIRS = 200


def enable_original_modules() -> None:
    """Allow Phase 7 wrappers to reuse the frozen model implementations."""
    scripts_dir = str(SCRIPTS_DIR)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


def path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)

