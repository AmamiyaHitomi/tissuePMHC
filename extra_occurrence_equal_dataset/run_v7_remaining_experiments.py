#!/usr/bin/env python3
"""Run/resume all missing Human v7 occurrence-equal experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from human_experiment_common import DEFAULT_HLA_PSEUDO, DEFAULT_SEEDS, DEFAULT_TEST, DEFAULT_TRAIN, EXPERIMENTS


HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "results" / "v7_full_rerun"
RUNNER = HERE / "run_human_experiment.py"
RUN_ORDER = tuple(EXPERIMENTS)
REUSED_NEW_DATA = {
    "e0_traditional": HERE / "results" / "e0_traditional",
    "e2_shared_heads": HERE / "results" / "e2_shared_heads",
    "e14a_auxiliary_dual_branch": HERE / "results" / "e14a_auxiliary_dual_branch",
    "e29_multikernel_cnn": HERE / "results" / "e29_multikernel_cnn",
    "external_predictors": HERE / "results" / "external_predictors",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def completed(name: str) -> bool:
    target = OUTPUT_ROOT / name
    return (target / "human_run_contract.json").is_file() and (target / "summary_metrics.csv").is_file()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def seed_completed(name: str, seed: int) -> bool:
    target = OUTPUT_ROOT / "seed_runs" / str(seed) / name
    return (target / "human_run_contract.json").is_file() and (target / "summary_metrics.csv").is_file()


def command(
    name: str,
    args: argparse.Namespace,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    output_root: Path = OUTPUT_ROOT,
    dependency_root: Path | None = None,
) -> list[str]:
    value = [
        sys.executable, str(RUNNER), name,
        "--train", str(DEFAULT_TRAIN.resolve()), "--test", str(DEFAULT_TEST.resolve()),
        "--output-root", str(output_root.resolve()),
        "--hla-pseudo-sequences", str(DEFAULT_HLA_PSEUDO.resolve()),
        "--device", args.device, "--seeds", *(str(seed) for seed in seeds),
    ]
    if dependency_root is not None:
        value.extend(("--dependency-root", str(dependency_root.resolve())))
    if args.epochs is not None:
        value.extend(("--epochs", str(args.epochs)))
    if args.overwrite or ((output_root / name).exists() and not completed(name)):
        value.append("--overwrite")
    return value


def rebuild_stability(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment_name, model), values in summary.groupby(["experiment_name", "model"], sort=False):
        def stat(column: str, operation: str) -> float:
            series = values[column].astype(float)
            if operation == "std":
                return float(series.std(ddof=1)) if len(series) > 1 else float("nan")
            return float(getattr(series, operation)())
        rows.append({
            "experiment_name": experiment_name, "model": model, "n_seeds": len(values),
            "mean_auroc_mean": stat("mean_auroc", "mean"), "mean_auroc_std": stat("mean_auroc", "std"),
            "mean_auroc_min": stat("mean_auroc", "min"), "mean_auroc_max": stat("mean_auroc", "max"),
            "mean_auprc_mean": stat("mean_auprc", "mean"), "mean_auprc_std": stat("mean_auprc", "std"),
            "mean_auprc_min": stat("mean_auprc", "min"), "mean_auprc_max": stat("mean_auprc", "max"),
            "mean_accuracy_mean": stat("mean_accuracy", "mean"), "mean_accuracy_std": stat("mean_accuracy", "std"),
            "mean_mcc_mean": stat("mean_mcc", "mean"), "mean_mcc_std": stat("mean_mcc", "std"),
            "worst_10_mean_auroc_mean": stat("worst_10_mean_auroc", "mean"),
            "worst_10_mean_auroc_std": stat("worst_10_mean_auroc", "std"),
        })
    return pd.DataFrame(rows)


def merge_seed_outputs(name: str) -> None:
    seed_targets = [OUTPUT_ROOT / "seed_runs" / str(seed) / name for seed in DEFAULT_SEEDS]
    for target in seed_targets:
        if not (target / "summary_metrics.csv").is_file():
            raise FileNotFoundError(f"Cannot merge incomplete seed result: {target}")
    target = OUTPUT_ROOT / name
    target.mkdir(parents=True, exist_ok=True)
    csv_names = sorted({path.name for directory in seed_targets for path in directory.glob("*.csv")})
    for filename in csv_names:
        if filename == "stability_metrics.csv":
            continue
        parts = [pd.read_csv(directory / filename) for directory in seed_targets if (directory / filename).is_file()]
        pd.concat(parts, ignore_index=True, sort=False).to_csv(target / filename, index=False)
    summary = pd.read_csv(target / "summary_metrics.csv")
    rebuild_stability(summary).to_csv(target / "stability_metrics.csv", index=False)
    contracts = [json.loads((directory / "human_run_contract.json").read_text(encoding="utf-8")) for directory in seed_targets]
    contract = contracts[0]
    contract["created_utc"] = now()
    contract["target_dir"] = str(target.resolve())
    contract["arguments"]["seeds"] = list(DEFAULT_SEEDS)
    contract["seed_contracts"] = [str((directory / "human_run_contract.json").resolve()) for directory in seed_targets]
    (target / "human_run_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    (target / "metadata.json").write_text(json.dumps({
        "suite": "human_v7_occurrence_equal_seed_merge", "experiment": name,
        "seeds": list(DEFAULT_SEEDS), "seed_directories": [str(path.resolve()) for path in seed_targets],
        "merged_utc": now(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def stream(value: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            value, cwd=HERE.parent, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (DEFAULT_TRAIN, DEFAULT_TEST, DEFAULT_HLA_PSEUDO):
        if not path.is_file():
            raise FileNotFoundError(path)
    reused = []
    for name, path in REUSED_NEW_DATA.items():
        if not (path / "summary_metrics.csv").is_file():
            raise FileNotFoundError(f"Missing completed Human new-data result: {path}")
        reused.append({"experiment": name, "path": str(path.resolve()), "status": "reused_occurrence_equal"})
    manifest: dict[str, object] = {
        "suite": "human_v7_occurrence_equal_only", "started_utc": now(), "finished_utc": None,
        "train": str(DEFAULT_TRAIN.resolve()), "test": str(DEFAULT_TEST.resolve()),
        "train_sha256": sha256(DEFAULT_TRAIN), "test_sha256": sha256(DEFAULT_TEST),
        "hla_pseudo_sequences": str(DEFAULT_HLA_PSEUDO.resolve()),
        "hla_pseudo_sequences_sha256": sha256(DEFAULT_HLA_PSEUDO),
        "seeds": list(DEFAULT_SEEDS), "reused": reused, "entries": [],
        "note": "E14a is reused as a reported row; auxiliary_soft recomputes shared branches only because E14b and E17 require unsaved branch predictions.",
    }
    if args.dry_run:
        for name in RUN_ORDER:
            if name == "mlp_dual_seed_ensemble":
                print(subprocess.list2cmdline(command(name, args)))
            else:
                for seed in DEFAULT_SEEDS:
                    seed_root = OUTPUT_ROOT / "seed_runs" / str(seed)
                    print(subprocess.list2cmdline(command(name, args, seeds=(seed,), output_root=seed_root, dependency_root=OUTPUT_ROOT)))
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_ROOT / "run_manifest.json"
    save_json(manifest_path, manifest)
    timing_rows = []
    suite_start = time.perf_counter()
    failed = False
    for name in RUN_ORDER:
        entry = {"experiment": name, "started_utc": now(), "status": "pending"}
        manifest["entries"].append(entry)
        if completed(name) and not args.overwrite:
            entry.update(status="skipped_completed", finished_utc=now())
            print(f"[SKIP completed Human new-data] {name}", flush=True)
            save_json(manifest_path, manifest)
            continue
        started = time.perf_counter()
        return_code = 0
        if name == "mlp_dual_seed_ensemble":
            value = command(name, args)
            entry.update(status="running", command=value)
            save_json(manifest_path, manifest)
            print(f"[RUN Human aggregate] {name} seeds={list(DEFAULT_SEEDS)}", flush=True)
            return_code = stream(value, OUTPUT_ROOT / "run_logs" / f"{name}.log")
        else:
            entry.update(status="running", seed_status=[])
            save_json(manifest_path, manifest)
            for seed in DEFAULT_SEEDS:
                seed_entry = {"seed": seed, "status": "pending", "started_utc": now()}
                entry["seed_status"].append(seed_entry)
                if seed_completed(name, seed) and not args.overwrite:
                    seed_entry.update(status="skipped_completed", finished_utc=now())
                    print(f"[SKIP completed seed] {name} seed={seed}", flush=True)
                    save_json(manifest_path, manifest)
                    continue
                seed_root = OUTPUT_ROOT / "seed_runs" / str(seed)
                value = command(name, args, seeds=(seed,), output_root=seed_root, dependency_root=OUTPUT_ROOT)
                seed_entry.update(status="running", command=value)
                save_json(manifest_path, manifest)
                seed_start = time.perf_counter()
                print(f"[RUN Human seed] {name} seed={seed}", flush=True)
                return_code = stream(value, OUTPUT_ROOT / "run_logs" / f"{name}_seed_{seed}.log")
                seed_seconds = time.perf_counter() - seed_start
                timing_rows.append({"experiment": f"{name}:seed={seed}", "seconds": f"{seed_seconds:.6f}", "return_code": return_code})
                seed_entry.update(status="completed" if return_code == 0 else "failed", seconds=seed_seconds, finished_utc=now())
                save_json(manifest_path, manifest)
                if return_code:
                    break
            if return_code == 0:
                merge_seed_outputs(name)
        seconds = time.perf_counter() - started
        timing_rows.append({"experiment": name, "seconds": f"{seconds:.6f}", "return_code": return_code})
        entry.update(status="completed" if return_code == 0 else "failed", return_code=return_code, seconds=seconds, finished_utc=now())
        save_json(manifest_path, manifest)
        if return_code:
            failed = True
            if not args.continue_on_error:
                break
    total = time.perf_counter() - suite_start
    timing_rows.append({"experiment": "TOTAL", "seconds": f"{total:.6f}", "return_code": int(failed)})
    with (OUTPUT_ROOT / "orchestration_timing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("experiment", "seconds", "return_code"))
        writer.writeheader()
        writer.writerows(timing_rows)
    manifest.update(finished_utc=now(), total_seconds=total, status="failed" if failed else "completed")
    save_json(manifest_path, manifest)
    print(f"Human suite total time: {total:.2f}s", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
