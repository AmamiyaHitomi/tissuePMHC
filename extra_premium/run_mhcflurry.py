#!/usr/bin/env python3
"""Run frozen MHCflurry 2.2.1 on the premium test-only query file."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import common


DEFAULT_EXECUTABLE = Path(
    os.environ.get("MHCFLURRY_EXECUTABLE")
    or shutil.which("mhcflurry-predict")
    or "mhcflurry-predict"
)
DEFAULT_MODEL_DIR = Path(
    os.environ.get("MHCFLURRY_MODELS_DIR", "~/.local/share/mhcflurry/models")
).expanduser()
QUERY_PATH = common.EXTERNAL_ROOT / "queries" / "premium_test_mhcflurry_input.csv"
RAW_DIR = common.EXTERNAL_ROOT / "raw_outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    if not QUERY_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {QUERY_PATH}. Run build_external_queries.py first."
        )
    if not args.executable.is_file():
        raise FileNotFoundError(f"MHCflurry executable not found: {args.executable}")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"MHCflurry model directory not found: {args.model_dir}")

    common.enable_project_package_imports()
    from extra1.issue5.run_predictors import run_mhcflurry

    # The installed Python 3.13 environment otherwise opens mhcgnomes YAML
    # resources with the Windows GBK locale and fails on UTF-8 content.
    os.environ["PYTHONUTF8"] = "1"
    run_mhcflurry(
        str(args.executable),
        QUERY_PATH,
        RAW_DIR / "premium_test_mhcflurry.csv",
        RAW_DIR / "premium_test_mhcflurry.metadata.json",
        True,
        args.model_dir,
        "none",
    )


if __name__ == "__main__":
    main()
