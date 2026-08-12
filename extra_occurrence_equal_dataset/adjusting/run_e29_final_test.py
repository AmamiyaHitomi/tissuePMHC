#!/usr/bin/env python3
"""Train the frozen tuned E29 configuration and evaluate the fixed test once."""

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
TEST_PATH = PROJECT_ROOT / "data" / "humanPMHC_occurence_equal_dataset" / "humanPMHC_test.csv.gz"
BEST_CONFIG_PATH = HERE / "results" / "e29_stage1_oof" / "best_config.json"
DEFAULT_OUTPUT = HERE / "results" / "e29_final_test"
DEFAULT_SEEDS = (20260704, 20260705, 20260706)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_tissuepmhc_e29_multikernel_cnn_oof as e29  # noqa: E402
import run_tissuepmhc_neural_baselines_v2 as base  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-config", type=Path, default=BEST_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_frozen_config(path: Path) -> tuple[str, dict[str, Any]]:
    contract = json.loads(path.resolve().read_text(encoding="utf-8"))
    if contract.get("config") != "lr_2e-3_dropout_0.35_aux_0.30":
        raise ValueError(f"Unexpected selected config: {contract.get('config')}")
    params = dict(contract["parameters"])
    params["epochs"] = int(contract["epochs"])
    params["batch_size"] = int(contract["batch_size"])
    expected = {
        "learning_rate": 0.002,
        "weight_decay": 0.0001,
        "embedding_dim": 16,
        "kernel_sizes": [2, 3, 5],
        "conv_channels": 32,
        "hidden_dim": 128,
        "dropout": 0.35,
        "tissue_loss_weight": 0.3,
        "hla_loss_weight": 0.3,
        "max_grad_norm": 1.0,
        "epochs": 20,
        "batch_size": 512,
    }
    if params != expected:
        raise ValueError(f"Frozen parameter contract changed:\nexpected={expected}\nobserved={params}")
    return str(contract["config"]), params


def read_split(path: Path) -> pd.DataFrame:
    frame = base.read_dataset(path)
    frame["target_tissue"] = frame["target_tissue"].fillna("NA")
    return frame


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], int]:
    train = read_split(TRAIN_PATH)
    test = read_split(TEST_PATH)
    for split_name, frame in (("train", train), ("test", test)):
        pair_labels = frame.groupby("pair_id", sort=False)["label"].agg(list)
        valid = pair_labels.map(lambda labels: sorted(map(int, labels)) == [0, 1])
        if not valid.all():
            raise ValueError(f"{split_name} contains {int((~valid).sum())} invalid pairs")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"{split_name} contains duplicate sample_id values")
    overlap = set(train["pair_id"].astype(str)) & set(test["pair_id"].astype(str))
    if overlap:
        raise ValueError(f"Train/test pair leakage detected: {len(overlap)} pairs")
    prepared_train, prepared_test, mappings = base.add_task_columns(train, test)
    if len(prepared_train) != len(train) or len(prepared_test) != len(test):
        raise ValueError("Task mapping dropped train or test rows")
    if len(mappings["tasks"]) != 77:
        raise ValueError(f"Expected 77 tasks, got {len(mappings['tasks'])}")
    lengths = pd.concat(
        [prepared_train["peptide_sequence"].str.len(), prepared_test["peptide_sequence"].str.len()]
    )
    if lengths.nunique() != 1:
        raise ValueError(f"Unexpected peptide lengths: {sorted(lengths.unique())}")
    return prepared_train, prepared_test, mappings, int(lengths.iloc[0])


def resolve_device(requested: str, torch: Any) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def pair_accuracy(task: pd.DataFrame) -> float:
    wide = task.pivot(index="pair_id", columns="label", values="score")
    if set(wide.columns) != {0, 1} or wide.isna().any().any():
        raise ValueError("Pair accuracy requires one positive and one negative score per pair")
    return float((wide[1] > wide[0]).mean())


def evaluate_predictions(
    predictions: pd.DataFrame,
    train_counts: pd.Series,
    model: str,
    seed_label: str | int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (tissue, hla), task in predictions.groupby(["target_tissue", "mhc_restriction"], sort=True):
        metrics = base.evaluate(
            task["label"].to_numpy(dtype=np.int64),
            task["score"].to_numpy(dtype=np.float64),
        )
        rows.append(
            {
                "model": model,
                "seed": seed_label,
                "target_tissue": tissue,
                "mhc_restriction": hla,
                "train_rows": int(train_counts.loc[(tissue, hla)]),
                "test_rows": int(len(task)),
                **metrics,
                "pair_accuracy": pair_accuracy(task),
            }
        )
    per_task = pd.DataFrame(rows)
    if len(per_task) != 77:
        raise ValueError(f"Expected metrics for 77 tasks, got {len(per_task)}")
    summary = {
        "model": model,
        "seed": seed_label,
        "n_tasks": int(len(per_task)),
        "train_rows": int(train_counts.sum()),
        "test_rows": int(len(predictions)),
        "mean_task_auroc": float(per_task["auroc"].mean()),
        "mean_task_auprc": float(per_task["auprc"].mean()),
        "mean_task_accuracy": float(per_task["accuracy"].mean()),
        "mean_task_mcc": float(per_task["mcc"].mean()),
        "mean_task_pair_accuracy": float(per_task["pair_accuracy"].mean()),
        "worst_10_mean_auroc": float(per_task.nsmallest(10, "auroc")["auroc"].mean()),
    }
    return per_task, summary


def append_timing(path: Path, row: dict[str, Any]) -> None:
    fields = ["seed", "epochs", "started_utc", "elapsed_seconds", "status"]
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow({key: row[key] for key in fields})


def run_seed(
    cli: argparse.Namespace,
    torch_parts: tuple[Any, Any, Any, Any],
    device: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    mappings: dict[str, Any],
    peptide_length: int,
    config_name: str,
    params: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    target = cli.output / "seed_runs" / f"seed_{seed}"
    prediction_path = target / "test_predictions.csv.gz"
    per_task_path = target / "per_task_metrics.csv"
    summary_path = target / "summary_metrics.csv"
    metadata_path = target / "metadata.json"
    if all(path.is_file() for path in (prediction_path, per_task_path, summary_path, metadata_path)) and not cli.overwrite:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("status") == "completed"
            and metadata.get("parameters") == params
            and metadata.get("train") == str(TRAIN_PATH.resolve())
            and metadata.get("test") == str(TEST_PATH.resolve())
        ):
            print(f"[SKIP] completed seed={seed}", flush=True)
            return (
                pd.read_csv(prediction_path, keep_default_na=False),
                pd.read_csv(per_task_path, keep_default_na=False),
                pd.read_csv(summary_path).iloc[0].to_dict(),
            )
    target.mkdir(parents=True, exist_ok=True)
    torch, nn, DataLoader, TensorDataset = torch_parts
    started_utc = utc_now()
    started = time.perf_counter()
    status = "failed"
    print(f"[RUN FINAL] config={config_name} seed={seed} device={device} params={params}", flush=True)
    try:
        prediction = e29.predict_seed(
            SimpleNamespace(**params),
            torch,
            nn,
            DataLoader,
            TensorDataset,
            train,
            test,
            mappings,
            peptide_length,
            device,
            seed,
            "final_full_train_fixed_test",
        )
        prediction = prediction.merge(
            test[["sample_id", "pair_id"]], on="sample_id", how="left", validate="one_to_one"
        )
        if len(prediction) != len(test) or prediction["pair_id"].isna().any():
            raise ValueError("Final predictions do not align with the fixed test rows")
        prediction.insert(0, "seed", seed)
        prediction.insert(0, "model", "e29_tuned")
        train_counts = train.groupby(["target_tissue", "mhc_restriction"], sort=True).size()
        per_task, summary = evaluate_predictions(prediction, train_counts, "e29_tuned", seed)
        prediction.to_csv(prediction_path, index=False, compression="gzip")
        per_task.to_csv(per_task_path, index=False)
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        status = "completed"
        print(
            f"[FINAL SEED] seed={seed} AUROC={summary['mean_task_auroc']:.6f} "
            f"AUPRC={summary['mean_task_auprc']:.6f} worst10={summary['worst_10_mean_auroc']:.6f}",
            flush=True,
        )
        return prediction, per_task, summary
    finally:
        elapsed = time.perf_counter() - started
        metadata_path.write_text(
            json.dumps(
                {
                    "status": status,
                    "config": config_name,
                    "seed": seed,
                    "train": str(TRAIN_PATH.resolve()),
                    "test": str(TEST_PATH.resolve()),
                    "parameters": params,
                    "started_utc": started_utc,
                    "elapsed_seconds": elapsed,
                    "fixed_test_final_evaluation": True,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        append_timing(
            cli.output / "timing_results.csv",
            {
                "seed": seed,
                "epochs": params["epochs"],
                "started_utc": started_utc,
                "elapsed_seconds": f"{elapsed:.6f}",
                "status": status,
            },
        )
        print(f"[SEED TIME] seed={seed} elapsed_seconds={elapsed:.3f} status={status}", flush=True)


def main() -> None:
    cli = parse_args()
    if list(cli.seeds) != list(DEFAULT_SEEDS):
        raise ValueError(f"Final seed contract is frozen to {list(DEFAULT_SEEDS)}")
    cli.output = cli.output.resolve()
    cli.output.mkdir(parents=True, exist_ok=True)
    config_name, params = load_frozen_config(cli.best_config)
    torch_parts = base.require_torch()
    device = resolve_device(cli.device, torch_parts[0])
    # The fixed test is opened exactly here, once, after the configuration is frozen.
    train, test, mappings, peptide_length = load_data()
    run_started_utc = utc_now()
    run_started = time.perf_counter()
    predictions: list[pd.DataFrame] = []
    per_tasks: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    print(
        f"[FINAL START] train_rows={len(train)} test_rows={len(test)} tasks={len(mappings['tasks'])} "
        f"seeds={cli.seeds} device={device}",
        flush=True,
    )
    for seed in cli.seeds:
        prediction, per_task, summary = run_seed(
            cli,
            torch_parts,
            device,
            train,
            test,
            mappings,
            peptide_length,
            config_name,
            params,
            seed,
        )
        predictions.append(prediction)
        per_tasks.append(per_task)
        summaries.append(summary)

    all_predictions = pd.concat(predictions, ignore_index=True)
    all_per_task = pd.concat(per_tasks, ignore_index=True)
    per_seed_summary = pd.DataFrame(summaries)
    all_predictions.to_csv(cli.output / "all_seed_test_predictions.csv.gz", index=False, compression="gzip")
    all_per_task.to_csv(cli.output / "all_seed_per_task_metrics.csv", index=False)
    per_seed_summary.to_csv(cli.output / "per_seed_summary.csv", index=False)

    key_columns = ["sample_id", "pair_id", "target_tissue", "mhc_restriction", "label"]
    ensemble = all_predictions.groupby(key_columns, as_index=False, dropna=False)["score"].mean()
    ensemble.insert(0, "seed", "three_seed_mean")
    ensemble.insert(0, "model", "e29_tuned_three_seed_mean")
    train_counts = train.groupby(["target_tissue", "mhc_restriction"], sort=True).size()
    ensemble_per_task, ensemble_summary = evaluate_predictions(
        ensemble, train_counts, "e29_tuned_three_seed_mean", "three_seed_mean"
    )
    ensemble.to_csv(cli.output / "ensemble_test_predictions.csv.gz", index=False, compression="gzip")
    ensemble_per_task.to_csv(cli.output / "ensemble_per_task_metrics.csv", index=False)
    pd.DataFrame([ensemble_summary]).to_csv(cli.output / "ensemble_summary.csv", index=False)

    metrics = [
        "mean_task_auroc",
        "mean_task_auprc",
        "mean_task_accuracy",
        "mean_task_mcc",
        "mean_task_pair_accuracy",
        "worst_10_mean_auroc",
    ]
    seed_metric_summary: dict[str, Any] = {"aggregation": "mean_and_sample_sd_of_seed_level_metrics"}
    for metric in metrics:
        values = pd.to_numeric(per_seed_summary[metric])
        seed_metric_summary[f"{metric}_mean"] = float(values.mean())
        seed_metric_summary[f"{metric}_sd"] = float(values.std(ddof=1))
    pd.DataFrame([seed_metric_summary]).to_csv(cli.output / "seed_metric_summary.csv", index=False)

    elapsed = time.perf_counter() - run_started
    final_contract = {
        "status": "completed",
        "config": config_name,
        "best_config_source": str(cli.best_config.resolve()),
        "train": str(TRAIN_PATH.resolve()),
        "test": str(TEST_PATH.resolve()),
        "fixed_test_evaluation_count": 1,
        "seeds": cli.seeds,
        "parameters": params,
        "started_utc": run_started_utc,
        "finished_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "seed_metric_summary": seed_metric_summary,
        "ensemble_summary": ensemble_summary,
    }
    (cli.output / "final_contract.json").write_text(
        json.dumps(final_contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print(f"[ENSEMBLE] {ensemble_summary}", flush=True)
    print(f"[FINAL TOTAL TIME] elapsed_seconds={elapsed:.3f} output={cli.output}", flush=True)


if __name__ == "__main__":
    main()
