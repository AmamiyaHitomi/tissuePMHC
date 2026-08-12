#!/usr/bin/env python3
"""Stage-1 hyperparameter screen for Human occurrence-equal TissuePMHC E29.

The fixed test split is intentionally never opened. Every configuration is
evaluated with deterministic, task-stratified, pair-grouped OOF folds made
from the training split. Each config/seed/fold unit is resumable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TRAIN_PATH = PROJECT_ROOT / "data" / "humanPMHC_occurence_equal_dataset" / "humanPMHC_train.csv.gz"
DEFAULT_OUTPUT = HERE / "results" / "e29_stage1_oof"
DEFAULT_SEEDS = (20260704,)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_tissuepmhc_e29_multikernel_cnn_oof as e29  # noqa: E402
import run_tissuepmhc_neural_baselines_v2 as base  # noqa: E402


BASE_CONFIG: dict[str, Any] = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "embedding_dim": 16,
    "kernel_sizes": [2, 3, 5],
    "conv_channels": 32,
    "hidden_dim": 128,
    "dropout": 0.2,
    "tissue_loss_weight": 0.1,
    "hla_loss_weight": 0.1,
    "max_grad_norm": 1.0,
}

CONFIG_OVERRIDES: dict[str, dict[str, Any]] = {
    "baseline": {},
    "lr_3e-4": {"learning_rate": 3e-4},
    "lr_6e-4": {"learning_rate": 6e-4},
    "lr_2e-3": {"learning_rate": 2e-3},
    "wd_1e-5": {"weight_decay": 1e-5},
    "wd_1e-3": {"weight_decay": 1e-3},
    "dropout_0.10": {"dropout": 0.1},
    "dropout_0.35": {"dropout": 0.35},
    "aux_0.03": {"tissue_loss_weight": 0.03, "hla_loss_weight": 0.03},
    "aux_0.30": {"tissue_loss_weight": 0.3, "hla_loss_weight": 0.3},
    "lr_2e-3_dropout_0.35_aux_0.30": {
        "learning_rate": 2e-3,
        "dropout": 0.35,
        "tissue_loss_weight": 0.3,
        "hla_loss_weight": 0.3,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=20260803)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--configs", nargs="+", choices=tuple(CONFIG_OVERRIDES), default=list(CONFIG_OVERRIDES))
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run baseline only with two folds and one epoch in a separate output directory.",
    )
    return parser.parse_args()


def validate_cli(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if not args.seeds or any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must contain non-negative integers")


def load_training_data() -> tuple[pd.DataFrame, dict[str, Any], int]:
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(TRAIN_PATH)
    raw = base.read_dataset(TRAIN_PATH)
    # "NA" is a real Human tissue category, not a missing value.
    raw["target_tissue"] = raw["target_tissue"].fillna("NA")
    required = {"sample_id", "pair_id", "label", "target_tissue", "mhc_restriction", "peptide_sequence"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")
    pair_labels = raw.groupby("pair_id", sort=False)["label"].agg(list)
    valid_pairs = pair_labels.map(lambda labels: sorted(map(int, labels)) == [0, 1])
    if not valid_pairs.all():
        raise ValueError(f"Invalid matched pairs: {int((~valid_pairs).sum())}")
    prepared, _, mappings = base.add_task_columns(raw, raw)
    if len(prepared) != len(raw):
        raise ValueError("Task mapping unexpectedly dropped training rows")
    peptide_lengths = prepared["peptide_sequence"].astype(str).str.len()
    if peptide_lengths.nunique() != 1:
        raise ValueError(f"Expected one peptide length, observed {sorted(peptide_lengths.unique())}")
    return prepared.reset_index(drop=True), mappings, int(peptide_lengths.iloc[0])


def resolve_device(requested: str, torch: Any) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def full_config(name: str, cli: argparse.Namespace) -> dict[str, Any]:
    values = dict(BASE_CONFIG)
    values.update(CONFIG_OVERRIDES[name])
    values.update({"epochs": int(cli.epochs), "batch_size": int(cli.batch_size)})
    return values


def pair_accuracy(task: pd.DataFrame) -> float:
    wide = task.pivot(index="pair_id", columns="label", values="score")
    if set(wide.columns) != {0, 1} or wide.isna().any().any():
        raise ValueError("Pair accuracy requires exactly one positive and one negative per pair")
    return float((wide[1] > wide[0]).mean())


def evaluate_predictions(
    predictions: pd.DataFrame,
    train_counts: pd.Series,
    config_name: str,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (tissue, hla), task in predictions.groupby(["target_tissue", "mhc_restriction"], sort=True):
        metrics = base.evaluate(
            task["label"].to_numpy(dtype=np.int64),
            task["score"].to_numpy(dtype=np.float64),
        )
        rows.append(
            {
                "config": config_name,
                "seed": seed,
                "target_tissue": tissue,
                "mhc_restriction": hla,
                "train_rows": int(train_counts.loc[(tissue, hla)]),
                "validation_rows": int(len(task)),
                **metrics,
                "pair_accuracy": pair_accuracy(task),
            }
        )
    per_task = pd.DataFrame(rows)
    if len(per_task) != 77:
        raise ValueError(f"Expected 77 tasks, got {len(per_task)}")
    summary = {
        "config": config_name,
        "seed": seed,
        "n_tasks": int(len(per_task)),
        "validation_rows": int(len(predictions)),
        "mean_auroc": float(per_task["auroc"].mean()),
        "mean_auprc": float(per_task["auprc"].mean()),
        "mean_accuracy": float(per_task["accuracy"].mean()),
        "mean_mcc": float(per_task["mcc"].mean()),
        "mean_pair_accuracy": float(per_task["pair_accuracy"].mean()),
        "worst_10_mean_auroc": float(per_task.nsmallest(10, "auroc")["auroc"].mean()),
    }
    return per_task, summary


def append_timing(path: Path, row: dict[str, Any]) -> None:
    fields = ["config", "seed", "fold", "epochs", "started_utc", "elapsed_seconds", "status"]
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow({key: row[key] for key in fields})


def unit_paths(output: Path, config_name: str, seed: int, fold: int) -> tuple[Path, Path]:
    directory = output / "fold_runs" / config_name / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"fold_{fold}_predictions.csv.gz", directory / f"fold_{fold}_metadata.json"


def run_unit(
    cli: argparse.Namespace,
    torch_parts: tuple[Any, Any, Any, Any],
    device: str,
    all_data: pd.DataFrame,
    fold_assignment: pd.Series,
    mappings: dict[str, Any],
    peptide_length: int,
    config_name: str,
    seed: int,
    fold: int,
) -> Path:
    torch, nn, DataLoader, TensorDataset = torch_parts
    prediction_path, metadata_path = unit_paths(cli.output, config_name, seed, fold)
    params = full_config(config_name, cli)
    if prediction_path.is_file() and metadata_path.is_file() and not cli.overwrite:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        contract_matches = (
            metadata.get("status") == "completed"
            and metadata.get("parameters") == params
            and metadata.get("train") == str(TRAIN_PATH.resolve())
            and int(metadata.get("split_seed", cli.split_seed)) == cli.split_seed
            and int(metadata.get("folds", cli.folds)) == cli.folds
        )
        if contract_matches:
            print(f"[SKIP] config={config_name} seed={seed} fold={fold}", flush=True)
            return prediction_path

    fitting = all_data[fold_assignment != fold].copy()
    validation = all_data[fold_assignment == fold].copy()
    if set(fitting["pair_id"]) & set(validation["pair_id"]):
        raise AssertionError("Pair leakage detected between fitting and validation")
    model_args = SimpleNamespace(**params)
    started_utc = utc_now()
    started = time.perf_counter()
    status = "failed"
    print(
        f"[RUN] config={config_name} seed={seed} fold={fold}/{cli.folds - 1} "
        f"fit_rows={len(fitting)} val_rows={len(validation)} params={json.dumps(params, sort_keys=True)}",
        flush=True,
    )
    try:
        prediction = e29.predict_seed(
            model_args,
            torch,
            nn,
            DataLoader,
            TensorDataset,
            fitting,
            validation,
            mappings,
            peptide_length,
            device,
            seed,
            f"tuning_{config_name}_fold_{fold}",
        )
        pair_lookup = validation[["sample_id", "pair_id"]]
        prediction = prediction.merge(pair_lookup, on="sample_id", how="left", validate="one_to_one")
        if prediction["pair_id"].isna().any() or len(prediction) != len(validation):
            raise ValueError("Validation predictions do not align one-to-one with validation rows")
        prediction.insert(0, "fold", fold)
        prediction.insert(0, "config", config_name)
        prediction.insert(1, "seed", seed)
        prediction.to_csv(prediction_path, index=False, compression="gzip")
        status = "completed"
        return prediction_path
    finally:
        elapsed = time.perf_counter() - started
        metadata_path.write_text(
            json.dumps(
                {
                    "status": status,
                    "config": config_name,
                    "seed": seed,
                    "fold": fold,
                    "folds": cli.folds,
                    "split_seed": cli.split_seed,
                    "train": str(TRAIN_PATH.resolve()),
                    "fixed_test_opened": False,
                    "parameters": params,
                    "started_utc": started_utc,
                    "elapsed_seconds": elapsed,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        append_timing(
            cli.output / "timing_results.csv",
            {
                "config": config_name,
                "seed": seed,
                "fold": fold,
                "epochs": cli.epochs,
                "started_utc": started_utc,
                "elapsed_seconds": f"{elapsed:.6f}",
                "status": status,
            },
        )
        print(
            f"[TIME] config={config_name} seed={seed} fold={fold} "
            f"elapsed_seconds={elapsed:.3f} status={status}",
            flush=True,
        )


def aggregate_seed(
    output: Path,
    all_data: pd.DataFrame,
    config_name: str,
    seed: int,
    prediction_paths: list[Path],
) -> dict[str, Any]:
    combined = pd.concat([pd.read_csv(path, keep_default_na=False) for path in prediction_paths], ignore_index=True)
    if len(combined) != len(all_data) or combined["sample_id"].nunique() != len(all_data):
        raise ValueError(
            f"OOF coverage mismatch for {config_name} seed={seed}: rows={len(combined)}, "
            f"unique_samples={combined['sample_id'].nunique()}, expected={len(all_data)}"
        )
    expected_ids = set(all_data["sample_id"].astype(str))
    if set(combined["sample_id"].astype(str)) != expected_ids:
        raise ValueError(f"OOF sample IDs mismatch for {config_name} seed={seed}")
    train_counts = all_data.groupby(["target_tissue", "mhc_restriction"], sort=True).size()
    per_task, summary = evaluate_predictions(combined, train_counts, config_name, seed)
    target = output / "seed_summaries" / config_name / f"seed_{seed}"
    target.mkdir(parents=True, exist_ok=True)
    combined.to_csv(target / "oof_predictions.csv.gz", index=False, compression="gzip")
    per_task.to_csv(target / "per_task_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(target / "summary_metrics.csv", index=False)
    print(
        f"[OOF] config={config_name} seed={seed} mean_auroc={summary['mean_auroc']:.6f} "
        f"mean_auprc={summary['mean_auprc']:.6f} worst10={summary['worst_10_mean_auroc']:.6f}",
        flush=True,
    )
    return summary


def write_leaderboard(output: Path, summaries: list[dict[str, Any]]) -> pd.DataFrame:
    per_seed = pd.DataFrame(summaries)
    per_seed.to_csv(output / "per_seed_summary.csv", index=False)
    metric_columns = [
        "mean_auroc",
        "mean_auprc",
        "mean_accuracy",
        "mean_mcc",
        "mean_pair_accuracy",
        "worst_10_mean_auroc",
    ]
    leaderboard = per_seed.groupby("config", as_index=False)[metric_columns].agg(["mean", "std"])
    leaderboard.columns = ["config"] + [f"{metric}_{stat}" for metric, stat in leaderboard.columns.tolist()[1:]]
    leaderboard = leaderboard.sort_values(
        ["mean_auroc_mean", "worst_10_mean_auroc_mean"], ascending=False
    ).reset_index(drop=True)
    leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    return leaderboard


def main() -> None:
    cli = parse_args()
    if cli.smoke:
        cli.output = cli.output.parent / f"{cli.output.name}_smoke"
        cli.configs = ["baseline"]
        cli.seeds = [cli.seeds[0]]
        cli.folds = 2
        cli.epochs = 1
    validate_cli(cli)
    selected_configs = list(dict.fromkeys(cli.configs))
    if cli.max_configs > 0:
        selected_configs = selected_configs[: cli.max_configs]
    cli.output = cli.output.resolve()
    cli.output.mkdir(parents=True, exist_ok=True)
    (cli.output / "search_space.json").write_text(
        json.dumps(
            {
                "base": BASE_CONFIG,
                "selected_configs": {name: full_config(name, cli) for name in selected_configs},
                "seeds": cli.seeds,
                "folds": cli.folds,
                "split_seed": cli.split_seed,
                "train": str(TRAIN_PATH.resolve()),
                "fixed_test_policy": "The fixed test file is never opened during tuning.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    torch_parts = base.require_torch()
    device = resolve_device(cli.device, torch_parts[0])
    all_data, mappings, peptide_length = load_training_data()
    fold_assignment = e29.make_pair_grouped_folds(all_data, cli.folds, cli.split_seed)
    fold_counts = pd.DataFrame({"fold": fold_assignment, "task_name": all_data["task_name"]}).groupby(
        ["fold", "task_name"]
    ).size()
    if fold_counts.index.get_level_values("task_name").nunique() != 77:
        raise ValueError("Fold assignment does not cover all 77 tasks")
    print(
        f"[START] device={device} rows={len(all_data)} tasks={len(mappings['tasks'])} "
        f"configs={selected_configs} seeds={cli.seeds} folds={cli.folds} epochs={cli.epochs}",
        flush=True,
    )
    total_started_utc = utc_now()
    total_started = time.perf_counter()
    summaries: list[dict[str, Any]] = []
    for config_name in selected_configs:
        for seed in cli.seeds:
            seed_started = time.perf_counter()
            paths = [
                run_unit(
                    cli,
                    torch_parts,
                    device,
                    all_data,
                    fold_assignment,
                    mappings,
                    peptide_length,
                    config_name,
                    seed,
                    fold,
                )
                for fold in range(cli.folds)
            ]
            summaries.append(aggregate_seed(cli.output, all_data, config_name, seed, paths))
            print(
                f"[SEED TIME] config={config_name} seed={seed} "
                f"elapsed_seconds={time.perf_counter() - seed_started:.3f}",
                flush=True,
            )
    leaderboard = write_leaderboard(cli.output, summaries)
    total_elapsed = time.perf_counter() - total_started
    (cli.output / "total_timing.json").write_text(
        json.dumps(
            {
                "started_utc": total_started_utc,
                "finished_utc": utc_now(),
                "elapsed_seconds": total_elapsed,
                "configs": selected_configs,
                "seeds": cli.seeds,
                "folds": cli.folds,
                "epochs": cli.epochs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(leaderboard.to_string(index=False), flush=True)
    print(f"[TOTAL TIME] elapsed_seconds={total_elapsed:.3f} output={cli.output}", flush=True)


if __name__ == "__main__":
    main()
