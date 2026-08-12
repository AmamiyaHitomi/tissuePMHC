#!/usr/bin/env python3
"""Run five-seed SHAP, ISM, and cross-tissue ISM after model lock.

Both the locked auxiliary model and its paired no-auxiliary comparator are
analysed.  Existing completed stages are resumable and skipped.  Explanation
outputs are never used to revise the selected auxiliary weight.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "results" / "human_postselection_formal_v1"
DEFAULT_OUTPUT = ROOT / "results" / "human_locked_interpretability_v1"
SEEDS = [20260721, 20260722, 20260723, 20260724, 20260725]


def append_timing(path: Path, condition: str, stage: str, seconds: float, status: str) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["condition", "stage", "seconds", "status"]
        )
        if new:
            writer.writeheader()
        writer.writerow({"condition": condition, "stage": stage,
                         "seconds": f"{seconds:.6f}", "status": status})


def run_stage(
    *, condition: str, stage: str, command: list[str], completion: Path, timing: Path
) -> None:
    if completion.is_file():
        print(f"[INTERPRETABILITY SKIP] condition={condition} stage={stage}", flush=True)
        return
    started = time.perf_counter()
    print(f"[INTERPRETABILITY START] condition={condition} stage={stage}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - started
    if not completion.is_file():
        raise RuntimeError(f"Stage did not produce completion metadata: {completion}")
    append_timing(timing, condition, stage, elapsed, "completed")
    print(
        f"[INTERPRETABILITY TIME] condition={condition} stage={stage} "
        f"elapsed_seconds={elapsed:.3f}", flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, default=FORMAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formal_contract_path = args.formal_root / "run_contract.json"
    if not formal_contract_path.is_file():
        raise RuntimeError("Formal post-selection run has not started")
    formal_contract = json.loads(formal_contract_path.read_text(encoding="utf-8"))
    if formal_contract.get("status") != "completed":
        raise RuntimeError("Formal post-selection run is incomplete")
    selected_weight = float(formal_contract["selected_weight"])
    args.output.mkdir(parents=True, exist_ok=True)
    timing = args.output / "timing_results.csv"
    started = time.perf_counter()
    conditions = {"locked_auxiliary": selected_weight, "no_auxiliary": 0.0}
    for condition, weight in conditions.items():
        source = args.formal_root / "interpretability" / condition
        checkpoints = source / "checkpoints"
        branches = source / "branch_predictions.csv.gz"
        target = args.output / condition
        shap_output = target / "shap"
        ism_output = target / "ism"
        cross_output = target / "cross_tissue_ism"
        common_seeds = ["--seeds", *map(str, SEEDS)]
        run_stage(
            condition=condition,
            stage="shap_expected_ig",
            command=[
                sys.executable, "scripts/run_occurrence_equal_e29_shap.py",
                "--reuse-checkpoints",
                "--checkpoint-root", str(checkpoints),
                "--branch-predictions-reference", str(branches),
                "--reference-predictions", str(target / "no_external_reference.csv"),
                "--output-dir", str(shap_output),
                "--aux-weight", str(weight),
                "--device", args.device,
                *common_seeds,
            ],
            completion=shap_output / "metadata.json",
            timing=timing,
        )
        run_stage(
            condition=condition,
            stage="exhaustive_ism",
            command=[
                sys.executable, "scripts/run_occurrence_equal_e29_ism.py",
                "--checkpoint-root", str(checkpoints),
                "--branch-predictions", str(branches),
                "--shap-observed", str(shap_output / "sample_observed_shap.csv.gz"),
                "--output-dir", str(ism_output),
                "--aux-weight", str(weight),
                "--device", args.device,
                *common_seeds,
            ],
            completion=ism_output / "metadata.json",
            timing=timing,
        )
        run_stage(
            condition=condition,
            stage="cross_tissue_ism",
            command=[
                sys.executable, "scripts/run_occurrence_equal_e29_cross_tissue_ism.py",
                "--checkpoint-root", str(checkpoints),
                "--branch-predictions", str(branches),
                "--output-dir", str(cross_output),
                "--device", args.device,
                *common_seeds,
            ],
            completion=cross_output / "metadata.json",
            timing=timing,
        )
    elapsed = time.perf_counter() - started
    append_timing(timing, "all", "total", elapsed, "completed")
    (args.output / "run_contract.json").write_text(json.dumps({
        "status": "completed",
        "selected_weight": selected_weight,
        "conditions": conditions,
        "seeds": SEEDS,
        "selection_use": False,
        "elapsed_seconds": elapsed,
    }, indent=2), encoding="utf-8")
    print(f"[INTERPRETABILITY TOTAL TIME] elapsed_seconds={elapsed:.3f}", flush=True)


if __name__ == "__main__":
    main()
