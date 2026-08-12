#!/usr/bin/env python3
"""Run E16 MC-Dropout inference ensembles for the E14a branches.

For each seed, E16 trains the unchanged E14a global auxiliary and HLA plain
branches once.  At inference it keeps dropout active, obtains 20 stochastic
predictions per sample, and uses the nested first 5, 10, and 20 passes as the
MC averages.  The two branch averages are then fused by E15's best rule:
within-task percentile-rank averaging.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_tissuepmhc_auxiliary_soft_ensemble as e14
import run_tissuepmhc_e15_fusion_ablation as e15
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MC_PASSES = [5, 10, 20]
PER_TASK_COLUMNS = [
    "experiment_name", "seed", "model", "mc_passes", "target_tissue", "mhc_restriction",
    "train_rows", "test_rows", "train_positive", "train_negative", "test_positive", "test_negative",
    *e14.METRICS, "global_branch", "hla_branch", "fusion_formula",
]
BRANCH_PREDICTION_COLUMNS = [
    "experiment_name", "seed", "mc_passes", "branch", "sample_id", "target_tissue", "mhc_restriction",
    "label", "mean_probability", "mean_logit", "probability_std",
]


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def capture_rng_state(torch: Any) -> dict[str, Any]:
    """Capture every RNG that later branch training could observe."""
    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else None,
    }


def restore_rng_state(torch: Any, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def mc_predict_task(
    args: argparse.Namespace,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    model: Any,
    test_task: pd.DataFrame,
    peptide_length: int,
    device: str,
) -> dict[int, dict[str, np.ndarray]]:
    """Return nested MC means for all requested pass counts for one task."""
    loader = e14.build_aux_loader(args, torch, DataLoader, TensorDataset, test_task, peptide_length, False)
    probabilities = {passes: [] for passes in MC_PASSES}
    logits_means = {passes: [] for passes in MC_PASSES}
    probability_stds = {passes: [] for passes in MC_PASSES}
    labels: list[np.ndarray] = []
    model.train()  # Deliberately activate only the model's dropout layers at inference.
    with torch.no_grad():
        for batch in loader:
            peptide_ids, task_ids, _, _, y = [item.to(device) for item in batch]
            draws = []
            for _ in range(max(MC_PASSES)):
                draws.append(model(peptide_ids, task_ids))
            stacked_logits = torch.stack(draws, dim=0)
            labels.append(y.cpu().numpy())
            for passes in MC_PASSES:
                selected_logits = stacked_logits[:passes]
                selected_probabilities = torch.sigmoid(selected_logits)
                probabilities[passes].append(selected_probabilities.mean(dim=0).cpu().numpy())
                logits_means[passes].append(selected_logits.mean(dim=0).cpu().numpy())
                probability_stds[passes].append(selected_probabilities.std(dim=0, unbiased=False).cpu().numpy())
    y_true = np.concatenate(labels)
    return {
        passes: {
            "y_true": y_true,
            "probability": np.concatenate(probabilities[passes]),
            "logit": np.concatenate(logits_means[passes]),
            "probability_std": np.concatenate(probability_stds[passes]),
        }
        for passes in MC_PASSES
    }


def predict_mc_branch(
    args: argparse.Namespace,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    model: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task_to_id: dict[str, int],
    peptide_length: int,
    device: str,
) -> dict[tuple[str, str], dict[str, object]]:
    train_mapped = e14.e7.prepare_with_mapping(train_df, task_to_id)
    test_mapped = e14.e7.prepare_with_mapping(test_df, task_to_id)
    predictions: dict[tuple[str, str], dict[str, object]] = {}
    for task_name in sorted(set(train_mapped["task_name"]) & set(test_mapped["task_name"])):
        train_task = train_mapped[train_mapped["task_name"] == task_name]
        test_task = test_mapped[test_mapped["task_name"] == task_name]
        first = test_task.iloc[0]
        predictions[(str(first["target_tissue"]), str(first["mhc_restriction"]))] = {
            "train_task": train_task,
            "test_task": test_task,
            "mc": mc_predict_task(args, torch, DataLoader, TensorDataset, model, test_task, peptide_length, device),
        }
    return predictions


def predict_mc_without_advancing_training_rng(torch: Any, *args: Any, **kwargs: Any) -> dict[tuple[str, str], dict[str, object]]:
    """Run stochastic inference but leave later model initialization unchanged.

    E14 trains the global branch and each HLA branch sequentially from one
    seeded RNG stream.  MC dropout also draws random masks; restoring the RNG
    state after inference ensures those masks cannot change the initialization
    or DataLoader shuffle order of the next HLA branch.
    """
    rng_state = capture_rng_state(torch)
    try:
        return predict_mc_branch(*args, **kwargs)
    finally:
        restore_rng_state(torch, rng_state)


def train_e14a_models(
    args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any,
    train_df: pd.DataFrame, test_df: pd.DataFrame, mappings: dict[str, Any], peptide_length: int,
    device: str, seed: int,
) -> tuple[dict[tuple[str, str], dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    print("  train global_aux branch")
    started = time.perf_counter()
    global_model, _ = e14.train_aux_branch(
        args, torch, nn, DataLoader, TensorDataset, train_df, mappings["task_to_id"],
        len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]), peptide_length, device, seed,
        "global_aux", "all_tasks", True,
    )
    print(f"    time global_aux duration={e14.format_duration(time.perf_counter() - started)}")
    global_predictions = predict_mc_without_advancing_training_rng(
        torch,
        args, torch, DataLoader, TensorDataset, global_model, train_df, test_df,
        mappings["task_to_id"], peptide_length, device,
    )

    hla_predictions: dict[tuple[str, str], dict[str, object]] = {}
    hlas = sorted(set(train_df["mhc_restriction"]) & set(test_df["mhc_restriction"]))
    for index, hla in enumerate(hlas, start=1):
        hla_train = train_df[train_df["mhc_restriction"] == hla].copy()
        hla_test = test_df[test_df["mhc_restriction"] == hla].copy()
        tasks = sorted(set(hla_train["task_name"]) & set(hla_test["task_name"]))
        if not tasks:
            continue
        task_to_id = {task: task_index for task_index, task in enumerate(tasks)}
        print(f"  train hla_plain branch {index:02d}/{len(hlas)} {hla} n_tasks={len(tasks)}")
        started = time.perf_counter()
        hla_model, _ = e14.train_aux_branch(
            args, torch, nn, DataLoader, TensorDataset, hla_train, task_to_id,
            len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]), peptide_length, device, seed,
            "hla_plain", hla, False,
        )
        print(f"    time hla_plain {hla} duration={e14.format_duration(time.perf_counter() - started)}")
        hla_predictions.update(predict_mc_without_advancing_training_rng(
            torch,
            args, torch, DataLoader, TensorDataset, hla_model, hla_train, hla_test,
            task_to_id, peptide_length, device,
        ))
    return global_predictions, hla_predictions


def build_result_rows(seed: int, global_predictions: dict, hla_predictions: dict) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    results: list[dict[str, object]] = []
    branch_rows: list[dict[str, object]] = []
    for key in sorted(set(global_predictions) & set(hla_predictions)):
        global_prediction, hla_prediction = global_predictions[key], hla_predictions[key]
        global_task, hla_task = global_prediction["test_task"], hla_prediction["test_task"]
        if not np.array_equal(global_task["sample_id"].to_numpy(), hla_task["sample_id"].to_numpy()):
            raise ValueError(f"Misaligned E16 sample IDs for task {key}")
        train_task = global_prediction["train_task"]
        for passes in MC_PASSES:
            global_mc, hla_mc = global_prediction["mc"][passes], hla_prediction["mc"][passes]
            if not np.array_equal(global_mc["y_true"], hla_mc["y_true"]):
                raise ValueError(f"Mismatched E16 labels for task {key}")
            fusion_input = pd.DataFrame({
                "probability_global_aux": global_mc["probability"],
                "probability_hla_plain": hla_mc["probability"],
                "logit_global_aux": global_mc["logit"], "logit_hla_plain": hla_mc["logit"],
            })
            score = e15.fusion_scores(fusion_input)["e15_task_rank_average"]
            first = global_task.iloc[0]
            train_positive, train_negative = e14.count_labels(train_task)
            test_positive, test_negative = e14.count_labels(global_task)
            results.append({
                "experiment_name": "E16_mc_dropout_ensemble", "seed": seed,
                "model": f"e16_mc_dropout_{passes}_rank_average", "mc_passes": passes,
                "target_tissue": first["target_tissue"], "mhc_restriction": first["mhc_restriction"],
                "train_rows": len(train_task), "test_rows": len(global_task),
                "train_positive": train_positive, "train_negative": train_negative,
                "test_positive": test_positive, "test_negative": test_negative,
                **base.evaluate(global_mc["y_true"], score),
                "global_branch": "global_aux", "hla_branch": "hla_plain",
                "fusion_formula": "task_rank_average(mean_mc_probability_global, mean_mc_probability_hla)",
            })
            for branch, task, mc in [("global_aux", global_task, global_mc), ("hla_plain", hla_task, hla_mc)]:
                for sample_id, label, probability, logit, std in zip(task["sample_id"], mc["y_true"], mc["probability"], mc["logit"], mc["probability_std"], strict=True):
                    branch_rows.append({
                        "experiment_name": "E16_mc_dropout_ensemble", "seed": seed, "mc_passes": passes,
                        "branch": branch, "sample_id": sample_id, "target_tissue": first["target_tissue"],
                        "mhc_restriction": first["mhc_restriction"], "label": int(label),
                        "mean_probability": float(probability), "mean_logit": float(logit), "probability_std": float(std),
                    })
    return results, branch_rows


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    train_df, test_df = base.read_dataset(args.train), base.read_dataset(args.test)
    train_df, test_df, mappings = base.add_task_columns(train_df, test_df)
    if args.max_tasks:
        keep = set(mappings["tasks"][:args.max_tasks])
        train_df, test_df = train_df[train_df.task_name.isin(keep)].copy(), test_df[test_df.task_name.isin(keep)].copy()
        train_df, test_df, mappings = base.add_task_columns(train_df, test_df)
    peptide_length = int(max(train_df.peptide_sequence.str.len().max(), test_df.peptide_sequence.str.len().max()))
    print(f"device: {device}")
    print(f"n_tasks: {len(mappings['tasks'])}; mc_passes: {MC_PASSES}; fusion: task_rank_average")
    rows: list[dict[str, object]] = []
    branch_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        seed_started = time.perf_counter()
        e14.set_seed(seed, torch)
        print(f"experiment: E16_mc_dropout_ensemble seed={seed}")
        global_predictions, hla_predictions = train_e14a_models(args, torch, nn, DataLoader, TensorDataset, train_df, test_df, mappings, peptide_length, device, seed)
        seed_rows, seed_branch_rows = build_result_rows(seed, global_predictions, hla_predictions)
        rows.extend(seed_rows); branch_rows.extend(seed_branch_rows)
        for passes in MC_PASSES:
            values = [row["auroc"] for row in seed_rows if row["mc_passes"] == passes]
            print(f"  mc_passes={passes} mean_auroc={np.mean(values):.4f}")
        print(f"time seed_total seed={seed} duration={e14.format_duration(time.perf_counter() - seed_started)}")
    summary, stability = base.summarize_results(rows), base.summarize_seed_stability(base.summarize_results(rows))
    base.write_csv(args.per_task_output, PER_TASK_COLUMNS, rows)
    base.write_csv(args.summary_output, base.SUMMARY_COLUMNS, summary)
    base.write_csv(args.stability_output, base.STABILITY_COLUMNS, stability)
    base.write_csv(args.branch_predictions_output, BRANCH_PREDICTION_COLUMNS, branch_rows)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps({"experiment_name":"E16_mc_dropout_ensemble","seeds":args.seeds,"mc_passes":MC_PASSES,"device":device,"n_tasks":len(mappings["tasks"]),"fusion":"task_rank_average","per_task_output":str(args.per_task_output),"branch_predictions_output":str(args.branch_predictions_output)}, indent=2, ensure_ascii=False), encoding="utf-8")
    for path in [args.per_task_output, args.summary_output, args.stability_output, args.branch_predictions_output, args.metadata_output]: print(f"wrote: {path}")
    print(f"run total time: {e14.format_duration(time.perf_counter() - started)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz"))
    parser.add_argument("--test", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_test.csv.gz"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=e14.DEFAULT_REPEAT_SEEDS)
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=128); parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--tissue-loss-weight", type=float, default=0.1); parser.add_argument("--hla-loss-weight", type=float, default=0.1); parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--per-task-output", type=Path, default=project_path("results/tissuePMHC_e16_mc_dropout_ensemble/per_task_metrics.csv"))
    parser.add_argument("--summary-output", type=Path, default=project_path("results/tissuePMHC_e16_mc_dropout_ensemble/summary_metrics.csv"))
    parser.add_argument("--stability-output", type=Path, default=project_path("results/tissuePMHC_e16_mc_dropout_ensemble/stability_metrics.csv"))
    parser.add_argument("--branch-predictions-output", type=Path, default=project_path("results/tissuePMHC_e16_mc_dropout_ensemble/branch_predictions.csv"))
    parser.add_argument("--metadata-output", type=Path, default=project_path("results/tissuePMHC_e16_mc_dropout_ensemble/metadata.json"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
