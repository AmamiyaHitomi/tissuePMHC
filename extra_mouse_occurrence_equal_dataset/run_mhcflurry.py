"""Run MHCflurry through its Python entry point on mouse test queries."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import common
from _runner import run_callable


MODEL_DIR = (
    Path(os.environ["LOCALAPPDATA"])
    / "mhcflurry"
    / "mhcflurry"
    / "4"
    / "2.2.0"
    / "models_class1_presentation"
    / "models"
)
QUERY_PATH = common.EXTERNAL_ROOT / "queries" / "mouse_test_mhcflurry_input.csv"
RAW_DIR = common.EXTERNAL_ROOT / "raw_outputs"
OUTPUT_PATH = RAW_DIR / "mouse_test_mhcflurry.csv"
METADATA_PATH = RAW_DIR / "mouse_test_mhcflurry.metadata.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not QUERY_PATH.is_file():
        raise FileNotFoundError(f"Missing query input: {QUERY_PATH}")
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Missing MHCflurry models: {MODEL_DIR}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONUTF8"] = "1"
    from mhcflurry.predict_command import run

    original_argv = sys.argv
    sys.argv = [
        "mhcflurry-predict",
        str(QUERY_PATH.resolve()),
        "--out",
        str(OUTPUT_PATH.resolve()),
        "--no-throw",
        "--no-flanking",
        "--models",
        str(MODEL_DIR.resolve()),
    ]
    try:
        run(sys.argv[1:])
    finally:
        sys.argv = original_argv
    if not OUTPUT_PATH.is_file():
        raise RuntimeError("MHCflurry returned without writing predictions.")
    METADATA_PATH.write_text(
        json.dumps(
            {
                "tool": "MHCflurry",
                "version": "2.2.1",
                "run_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "input": str(QUERY_PATH.resolve()),
                "input_sha256": sha256(QUERY_PATH),
                "output": str(OUTPUT_PATH.resolve()),
                "output_sha256": sha256(OUTPUT_PATH),
                "model_dir": str(MODEL_DIR.resolve()),
                "flanking_policy": "disabled",
                "entry_point": "mhcflurry.predict_command.run",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote: {OUTPUT_PATH}", flush=True)


run_callable("run_mhcflurry.py", main)
