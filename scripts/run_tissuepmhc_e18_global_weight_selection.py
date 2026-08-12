#!/usr/bin/env python3
"""Run E18 validation-selected global weights for E14a rank fusion.

Weights are selected only from an inner validation split of the training data.
The independent test predictions are read from E14's saved branch predictions,
so the test labels never participate in choosing the global/HLA fusion weight.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_tissuepmhc_auxiliary_soft_ensemble as e14
import run_tissuepmhc_e15_fusion_ablation as e15
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PER_TASK_COLUMNS = [
    "experiment_name", "seed", "model", "target_tissue", "mhc_restriction", "test_rows", "test_positive", "test_negative",
    *e14.METRICS, "global_weight", "hla_weight", "weight_source", "fusion_formula",
]
WEIGHT_SEARCH_COLUMNS = ["global_weight", "hla_weight", "mean_task_auroc", "mean_task_auprc", "n_seed_task_rows", "selected"]


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def split_train_validation(train_df: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out complete pair_id groups within every task, preserving pair integrity."""
    if not 0 < fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5.")
    rng = np.random.default_rng(seed)
    validation_indices: list[int] = []
    for _, task in train_df.groupby("task_name", sort=True):
        pairs = np.asarray(sorted(task["pair_id"].unique()))
        n_validation = max(1, int(round(len(pairs) * fraction)))
        chosen = set(rng.permutation(pairs)[:n_validation])
        validation_indices.extend(task.index[task["pair_id"].isin(chosen)].tolist())
    validation = train_df.loc[validation_indices].copy()
    fitting = train_df.drop(index=validation_indices).copy()
    if set(fitting["pair_id"]) & set(validation["pair_id"]):
        raise AssertionError("pair_id leakage between E18 fitting and validation partitions")
    return fitting, validation


def predict_e14a_branches(
    args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any,
    fitting_df: pd.DataFrame, prediction_df: pd.DataFrame, mappings: dict[str, Any], peptide_length: int,
    device: str, seed: int,
) -> tuple[dict, dict]:
    print("  train validation global_aux branch")
    global_model, _ = e14.train_aux_branch(
        args, torch, nn, DataLoader, TensorDataset, fitting_df, mappings["task_to_id"],
        len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]), peptide_length, device, seed,
        "global_aux", "validation_all_tasks", True,
    )
    global_predictions, _ = e14.predict_branch(
        args, torch, DataLoader, TensorDataset, global_model, fitting_df, prediction_df,
        mappings["task_to_id"], peptide_length, device, seed, "global_aux", "validation_all_tasks", True,
    )
    hla_predictions: dict = {}
    hlas = sorted(set(fitting_df["mhc_restriction"]) & set(prediction_df["mhc_restriction"]))
    for index, hla in enumerate(hlas, start=1):
        hla_fit = fitting_df[fitting_df["mhc_restriction"] == hla].copy()
        hla_validation = prediction_df[prediction_df["mhc_restriction"] == hla].copy()
        tasks = sorted(set(hla_fit["task_name"]) & set(hla_validation["task_name"]))
        if not tasks:
            continue
        task_to_id = {task: task_index for task_index, task in enumerate(tasks)}
        print(f"  train validation hla_plain {index:02d}/{len(hlas)} {hla}")
        model, _ = e14.train_aux_branch(
            args, torch, nn, DataLoader, TensorDataset, hla_fit, task_to_id,
            len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]), peptide_length, device, seed,
            "hla_plain", f"validation_{hla}", False,
        )
        predictions, _ = e14.predict_branch(
            args, torch, DataLoader, TensorDataset, model, hla_fit, hla_validation,
            task_to_id, peptide_length, device, seed, "hla_plain", hla, False,
        )
        hla_predictions.update(predictions)
    return global_predictions, hla_predictions


def predictions_to_frame(global_predictions: dict, hla_predictions: dict) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for key in sorted(set(global_predictions) & set(hla_predictions)):
        global_prediction, hla_prediction = global_predictions[key], hla_predictions[key]
        global_task, hla_task = global_prediction["test_task"], hla_prediction["test_task"]
        if not np.array_equal(global_task["sample_id"].to_numpy(), hla_task["sample_id"].to_numpy()):
            raise ValueError(f"E18 branch sample IDs are misaligned for {key}")
        if not np.array_equal(global_prediction["y_true"], hla_prediction["y_true"]):
            raise ValueError(f"E18 branch labels disagree for {key}")
        rows.append(pd.DataFrame({
            "sample_id": global_task["sample_id"].to_numpy(), "target_tissue": global_task["target_tissue"].to_numpy(),
            "mhc_restriction": global_task["mhc_restriction"].to_numpy(), "label": global_prediction["y_true"],
            "probability_global_aux": global_prediction["y_score"], "probability_hla_plain": hla_prediction["y_score"],
        }))
    return pd.concat(rows, ignore_index=True)


def rank_weight_score(task: pd.DataFrame, global_weight: float) -> np.ndarray:
    global_rank = task["probability_global_aux"].rank(method="average", pct=True).to_numpy()
    hla_rank = task["probability_hla_plain"].rank(method="average", pct=True).to_numpy()
    return global_weight * global_rank + (1.0 - global_weight) * hla_rank


def evaluate_weight(predictions: pd.DataFrame, global_weight: float) -> list[dict[str, float]]:
    rows = []
    for _, task in predictions.groupby(["seed", "target_tissue", "mhc_restriction"], sort=True):
        rows.append(base.evaluate(task["label"].to_numpy(dtype=int), rank_weight_score(task, global_weight)))
    return rows


def select_weight(validation_predictions: pd.DataFrame, weights: list[float]) -> tuple[float, list[dict[str, object]]]:
    search_rows = []
    for weight in weights:
        values = evaluate_weight(validation_predictions, weight)
        search_rows.append({
            "global_weight": weight, "hla_weight": 1.0 - weight,
            "mean_task_auroc": float(np.mean([row["auroc"] for row in values])),
            "mean_task_auprc": float(np.mean([row["auprc"] for row in values])),
            "n_seed_task_rows": len(values), "selected": False,
        })
    best_score = max(float(row["mean_task_auroc"]) for row in search_rows)
    tied = [row for row in search_rows if np.isclose(float(row["mean_task_auroc"]), best_score)]
    selected = min(tied, key=lambda row: abs(float(row["global_weight"]) - 0.5))
    selected["selected"] = True
    return float(selected["global_weight"]), search_rows


def make_test_rows(test_predictions: pd.DataFrame, weights: list[tuple[str, float, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (seed, tissue, hla), task in test_predictions.groupby(["seed", "target_tissue", "mhc_restriction"], sort=True):
        y_true = task["label"].to_numpy(dtype=int)
        for model, global_weight, source in weights:
            rows.append({
                "experiment_name": "E18_global_weight_selection", "seed": int(seed), "model": model,
                "target_tissue": tissue, "mhc_restriction": hla,
                "test_rows": len(task), "test_positive": int(y_true.sum()), "test_negative": int(len(task) - y_true.sum()),
                **base.evaluate(y_true, rank_weight_score(task, global_weight)),
                "global_weight": global_weight, "hla_weight": 1.0 - global_weight, "weight_source": source,
                "fusion_formula": "global_weight * global_task_rank + hla_weight * hla_task_rank",
            })
    return rows


def test_predictions_from_e14(path: Path, seeds: list[int]) -> pd.DataFrame:
    wide = e15.require_aligned_e14a_predictions(path)
    wide = wide[wide["seed"].isin(seeds)].copy()
    if sorted(int(seed) for seed in wide["seed"].unique()) != seeds:
        raise ValueError("E14 test prediction file does not contain every requested E18 seed.")
    return wide.rename(columns={"label_global_aux": "label"})[
        ["seed", "sample_id", "target_tissue", "mhc_restriction", "label", "probability_global_aux", "probability_hla_plain"]
    ]


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    train_df = base.read_dataset(args.train)
    test_df = base.read_dataset(args.test)
    train_df, test_df, mappings = base.add_task_columns(train_df, test_df)
    if args.max_tasks:
        keep = set(mappings["tasks"][:args.max_tasks])
        train_df, test_df = train_df[train_df.task_name.isin(keep)].copy(), test_df[test_df.task_name.isin(keep)].copy()
        train_df, test_df, mappings = base.add_task_columns(train_df, test_df)
    peptide_length = int(max(train_df.peptide_sequence.str.len().max(), test_df.peptide_sequence.str.len().max()))
    test_predictions = test_predictions_from_e14(args.e14_branch_predictions, args.seeds)
    allowed_tasks = test_df[["target_tissue", "mhc_restriction"]].drop_duplicates()
    test_predictions = test_predictions.merge(allowed_tasks, on=["target_tissue", "mhc_restriction"], how="inner")
    validation_frames = []
    print(f"device: {device}; validation_fraction: {args.validation_fraction}; weight_grid: {args.weight_grid}")
    for seed in args.seeds:
        seed_started = time.perf_counter()
        e14.set_seed(seed, torch)
        fitting, validation = split_train_validation(train_df, args.validation_fraction, seed)
        print(f"experiment: E18 validation seed={seed} fit_rows={len(fitting)} validation_rows={len(validation)}")
        global_predictions, hla_predictions = predict_e14a_branches(args, torch, nn, DataLoader, TensorDataset, fitting, validation, mappings, peptide_length, device, seed)
        frame = predictions_to_frame(global_predictions, hla_predictions)
        frame.insert(0, "seed", seed)
        validation_frames.append(frame)
        print(f"time seed_total seed={seed} duration={e14.format_duration(time.perf_counter() - seed_started)}")
    validation_predictions = pd.concat(validation_frames, ignore_index=True)
    selected_weight, search_rows = select_weight(validation_predictions, args.weight_grid)
    print(f"selected_global_weight={selected_weight:.2f}; selected_hla_weight={1.0 - selected_weight:.2f}")
    results = make_test_rows(test_predictions, [
        ("e18_fixed_0.50_rank_average", 0.5, "fixed"),
        ("e18_validation_selected_rank_average", selected_weight, "inner_validation_mean_task_auroc"),
    ])
    summary, stability = base.summarize_results(results), base.summarize_seed_stability(base.summarize_results(results))
    base.write_csv(args.per_task_output, PER_TASK_COLUMNS, results)
    base.write_csv(args.summary_output, base.SUMMARY_COLUMNS, summary)
    base.write_csv(args.stability_output, base.STABILITY_COLUMNS, stability)
    base.write_csv(args.weight_search_output, WEIGHT_SEARCH_COLUMNS, search_rows)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps({
        "experiment_name": "E18_global_weight_selection", "seeds": args.seeds, "device": device,
        "validation_fraction": args.validation_fraction, "validation_split": "pair_id_grouped_within_task",
        "weight_grid": args.weight_grid, "selected_global_weight": selected_weight,
        "selected_hla_weight": 1.0 - selected_weight, "fusion": "weighted_task_rank_average",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    for path in [args.per_task_output, args.summary_output, args.stability_output, args.weight_search_output, args.metadata_output]: print(f"wrote: {path}")
    print(f"run total time: {e14.format_duration(time.perf_counter() - started)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz")); parser.add_argument("--test", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_test.csv.gz"))
    parser.add_argument("--e14-branch-predictions", type=Path, default=project_path("results/tissuePMHC_auxiliary_soft_ensemble/branch_predictions.csv"))
    parser.add_argument("--seeds", nargs="+", type=int, default=e14.DEFAULT_REPEAT_SEEDS); parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--validation-fraction", type=float, default=0.2); parser.add_argument("--weight-grid", nargs="+", type=float, default=[round(value * 0.05, 2) for value in range(21)])
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--batch-size", type=int, default=512); parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=128); parser.add_argument("--dropout", type=float, default=0.2); parser.add_argument("--tissue-loss-weight", type=float, default=0.1); parser.add_argument("--hla-loss-weight", type=float, default=0.1); parser.add_argument("--max-grad-norm", type=float, default=1.0); parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--per-task-output", type=Path, default=project_path("results/tissuePMHC_e18_global_weight_selection/per_task_metrics.csv")); parser.add_argument("--summary-output", type=Path, default=project_path("results/tissuePMHC_e18_global_weight_selection/summary_metrics.csv")); parser.add_argument("--stability-output", type=Path, default=project_path("results/tissuePMHC_e18_global_weight_selection/stability_metrics.csv")); parser.add_argument("--weight-search-output", type=Path, default=project_path("results/tissuePMHC_e18_global_weight_selection/validation_weight_search.csv")); parser.add_argument("--metadata-output", type=Path, default=project_path("results/tissuePMHC_e18_global_weight_selection/metadata.json"))
    return parser.parse_args()


if __name__ == "__main__": run(parse_args())
