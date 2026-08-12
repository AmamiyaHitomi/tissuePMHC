#!/usr/bin/env python3
"""Run the complete E26 workflow with one command.

It generates pair-grouped OOF predictions and independent test predictions for
three ordinary E14a seeds plus their MC-dropout counterparts, builds two
seed-mean candidates, then calls the leakage-safe E26 greedy selector.  No
intermediate command or manually prepared CSV is needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

import run_tissuepmhc_auxiliary_soft_ensemble as e14
import run_tissuepmhc_e15_fusion_ablation as e15
import run_tissuepmhc_e16_mc_dropout_ensemble as e16
import run_tissuepmhc_e26_greedy_ensemble_selection as selector
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEYS = ["sample_id", "target_tissue", "mhc_restriction"]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def make_pair_grouped_folds(train_df: pd.DataFrame, folds: int, split_seed: int) -> pd.Series:
    """Assign each pair_id to one OOF fold inside its task, deterministically."""
    if folds < 2:
        raise ValueError("oof_folds must be at least two.")
    rng = np.random.default_rng(split_seed)  # local Generator: does not advance global NumPy RNG
    assignment = pd.Series(index=train_df.index, dtype="int64")
    for task_name, task in train_df.groupby("task_name", sort=True):
        pairs = np.asarray(sorted(task.pair_id.unique()))
        if len(pairs) < folds:
            raise ValueError(f"Task {task_name} has {len(pairs)} pairs, fewer than oof_folds={folds}.")
        shuffled = rng.permutation(pairs)
        pair_to_fold = {pair: index % folds for index, pair in enumerate(shuffled)}
        assignment.loc[task.index] = task.pair_id.map(pair_to_fold).astype(int)
    if assignment.isna().any():
        raise AssertionError("Some training rows were not assigned an OOF fold.")
    for fold in range(folds):
        fitting, held_out = train_df[assignment != fold], train_df[assignment == fold]
        if set(fitting.pair_id) & set(held_out.pair_id):
            raise AssertionError(f"pair_id leakage in OOF fold {fold}.")
    return assignment.astype(int)


def fuse_standard(global_predictions: dict, hla_predictions: dict) -> pd.DataFrame:
    """Convert aligned E14 branch predictions into one fused candidate score."""
    rows = []
    for key in sorted(set(global_predictions) & set(hla_predictions)):
        global_task, hla_task = global_predictions[key], hla_predictions[key]
        global_df, hla_df = global_task["test_task"], hla_task["test_task"]
        if not np.array_equal(global_df.sample_id.to_numpy(), hla_df.sample_id.to_numpy()):
            raise ValueError(f"Branch sample IDs differ for {key}.")
        if not np.array_equal(global_task["y_true"], hla_task["y_true"]):
            raise ValueError(f"Branch labels differ for {key}.")
        scores = e15.fusion_scores(pd.DataFrame({
            "probability_global_aux": global_task["y_score"], "probability_hla_plain": hla_task["y_score"],
            "logit_global_aux": np.zeros(len(global_df)), "logit_hla_plain": np.zeros(len(global_df)),
        }))["e15_task_rank_average"]
        rows.append(pd.DataFrame({
            "sample_id": global_df.sample_id.to_numpy(), "target_tissue": global_df.target_tissue.to_numpy(),
            "mhc_restriction": global_df.mhc_restriction.to_numpy(), "label": global_task["y_true"], "score": scores,
        }))
    return pd.concat(rows, ignore_index=True)


def fuse_mc20(global_predictions: dict, hla_predictions: dict) -> pd.DataFrame:
    """Fuse MC-20 means; MC inference restores all RNG streams before return."""
    rows = []
    for key in sorted(set(global_predictions) & set(hla_predictions)):
        global_task, hla_task = global_predictions[key], hla_predictions[key]
        global_df, hla_df = global_task["test_task"], hla_task["test_task"]
        if not np.array_equal(global_df.sample_id.to_numpy(), hla_df.sample_id.to_numpy()):
            raise ValueError(f"MC branch sample IDs differ for {key}.")
        global_mc, hla_mc = global_task["mc"][20], hla_task["mc"][20]
        if not np.array_equal(global_mc["y_true"], hla_mc["y_true"]):
            raise ValueError(f"MC branch labels differ for {key}.")
        scores = e15.fusion_scores(pd.DataFrame({
            "probability_global_aux": global_mc["probability"], "probability_hla_plain": hla_mc["probability"],
            "logit_global_aux": global_mc["logit"], "logit_hla_plain": hla_mc["logit"],
        }))["e15_task_rank_average"]
        rows.append(pd.DataFrame({
            "sample_id": global_df.sample_id.to_numpy(), "target_tissue": global_df.target_tissue.to_numpy(),
            "mhc_restriction": global_df.mhc_restriction.to_numpy(), "label": global_mc["y_true"], "score": scores,
        }))
    return pd.concat(rows, ignore_index=True)


def predict_seed(
    args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any,
    fitting: pd.DataFrame, prediction: pd.DataFrame, mappings: dict[str, Any], peptide_length: int,
    device: str, seed: int, context: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train each E14a branch once and emit ordinary plus MC-20 predictions."""
    e14.set_seed(seed, torch)
    print(f"  train seed={seed} context={context} global_aux", flush=True)
    global_model, _ = e14.train_aux_branch(
        args, torch, nn, DataLoader, TensorDataset, fitting, mappings["task_to_id"],
        len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]), peptide_length, device, seed,
        "global_aux", context, True,
    )
    global_standard, _ = e14.predict_branch(
        args, torch, DataLoader, TensorDataset, global_model, fitting, prediction,
        mappings["task_to_id"], peptide_length, device, seed, "global_aux", context, True,
    )
    global_mc = e16.predict_mc_without_advancing_training_rng(
        torch, args, torch, DataLoader, TensorDataset, global_model, fitting, prediction,
        mappings["task_to_id"], peptide_length, device,
    )
    hla_standard, hla_mc = {}, {}
    hlas = sorted(set(fitting.mhc_restriction) & set(prediction.mhc_restriction))
    for index, hla in enumerate(hlas, start=1):
        hla_fit = fitting[fitting.mhc_restriction == hla].copy()
        hla_prediction = prediction[prediction.mhc_restriction == hla].copy()
        tasks = sorted(set(hla_fit.task_name) & set(hla_prediction.task_name))
        if not tasks:
            continue
        task_to_id = {task: task_index for task_index, task in enumerate(tasks)}
        print(f"  train seed={seed} context={context} hla={index:02d}/{len(hlas)} {hla}", flush=True)
        model, _ = e14.train_aux_branch(
            args, torch, nn, DataLoader, TensorDataset, hla_fit, task_to_id,
            len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]), peptide_length, device, seed,
            "hla_plain", f"{context}_{hla}", False,
        )
        standard, _ = e14.predict_branch(
            args, torch, DataLoader, TensorDataset, model, hla_fit, hla_prediction,
            task_to_id, peptide_length, device, seed, "hla_plain", hla, False,
        )
        hla_standard.update(standard)
        hla_mc.update(e16.predict_mc_without_advancing_training_rng(
            torch, args, torch, DataLoader, TensorDataset, model, hla_fit, hla_prediction,
            task_to_id, peptide_length, device,
        ))
    return fuse_standard(global_standard, hla_standard), fuse_mc20(global_mc, hla_mc)


def candidate_rows(split: str, candidate: str, frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "split", split)
    output.insert(1, "candidate", candidate)
    output.insert(2, "seed", 0)  # model seed belongs in candidate name; all candidates cover this evaluation repeat.
    return output[["split", "candidate", "seed", *KEYS, "label", "score"]]


def append_seed_means(frame: pd.DataFrame, split: str, seeds: list[int]) -> pd.DataFrame:
    output = [frame]
    for prefix in ("e14_final", "e16_mc20"):
        members = [f"{prefix}_seed_{seed}" for seed in seeds]
        subset = frame[frame.candidate.isin(members)]
        pivot = subset.pivot(index=["split", "seed", *KEYS, "label"], columns="candidate", values="score")
        if pivot.isna().any().any():
            raise AssertionError(f"Seed predictions are not aligned for {prefix} {split}.")
        mean = pivot.mean(axis=1).rename("score").reset_index()
        mean.insert(1, "candidate", f"{prefix}_{len(seeds)}seed_mean")
        output.append(mean[["split", "candidate", "seed", *KEYS, "label", "score"]])
    return pd.concat(output, ignore_index=True)


def generate_predictions(args: argparse.Namespace) -> tuple[Path, Path, dict[str, object]]:
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    train, test = base.read_dataset(args.train), base.read_dataset(args.test)
    train, test, mappings = base.add_task_columns(train, test)
    if args.max_tasks:
        keep = set(mappings["tasks"][:args.max_tasks])
        train, test = train[train.task_name.isin(keep)].copy(), test[test.task_name.isin(keep)].copy()
        train, test, mappings = base.add_task_columns(train, test)
    peptide_length = int(max(train.peptide_sequence.str.len().max(), test.peptide_sequence.str.len().max()))
    folds = make_pair_grouped_folds(train, args.oof_folds, args.oof_split_seed)
    oof_parts, test_parts = [], []
    for fold in range(args.oof_folds):
        fitting, held_out = train[folds != fold].copy(), train[folds == fold].copy()
        print(f"OOF fold={fold + 1}/{args.oof_folds} fit_rows={len(fitting)} holdout_rows={len(held_out)}", flush=True)
        for seed in args.seeds:
            final, mc20 = predict_seed(args, torch, nn, DataLoader, TensorDataset, fitting, held_out, mappings, peptide_length, device, seed, f"oof_fold_{fold}")
            oof_parts.extend([candidate_rows("oof", f"e14_final_seed_{seed}", final), candidate_rows("oof", f"e16_mc20_seed_{seed}", mc20)])
    for seed in args.seeds:
        print(f"Full-train test candidate seed={seed}", flush=True)
        final, mc20 = predict_seed(args, torch, nn, DataLoader, TensorDataset, train, test, mappings, peptide_length, device, seed, "full_train_test")
        test_parts.extend([candidate_rows("test", f"e14_final_seed_{seed}", final), candidate_rows("test", f"e16_mc20_seed_{seed}", mc20)])
    oof = append_seed_means(pd.concat(oof_parts, ignore_index=True), "oof", args.seeds)
    test_predictions = append_seed_means(pd.concat(test_parts, ignore_index=True), "test", args.seeds)
    args.oof_predictions_output.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(args.oof_predictions_output, index=False)
    test_predictions.to_csv(args.test_predictions_output, index=False)
    return args.oof_predictions_output, args.test_predictions_output, {"device": device, "n_tasks": len(mappings["tasks"]), "peptide_length": peptide_length}


def run(args: argparse.Namespace) -> None:
    oof_path, test_path, details = generate_predictions(args)
    selection_args = SimpleNamespace(
        oof_predictions=oof_path, test_predictions=test_path, candidates=None,
        max_members=args.max_members, min_improvement=args.min_improvement,
        per_task_output=args.per_task_output, summary_output=args.summary_output,
        stability_output=args.stability_output, trajectory_output=args.trajectory_output,
        members_output=args.members_output, metadata_output=args.selection_metadata_output,
    )
    selector.run(selection_args)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps({
        "experiment_name": "E26_all_in_one", "seeds": args.seeds, "oof_folds": args.oof_folds,
        "oof_split_seed": args.oof_split_seed, "candidate_pool": "E14 final per-seed + E16 MC-20 per-seed + seed means",
        "swa_excluded": True, "rng_controls": [
            "OOF assignment uses local numpy.default_rng", "E14 training resets all RNGs per model seed",
            "MC inference restores Python/NumPy/CPU-Torch/CUDA RNG state", "E26 selection is deterministic",
        ], "outputs": {"oof_predictions": str(oof_path), "test_predictions": str(test_path)}, **details,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote: {args.metadata_output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_train.csv.gz"))
    parser.add_argument("--test", type=Path, default=project_path("data/tissuePMHC/tissuePMHC_test.csv.gz"))
    parser.add_argument("--seeds", nargs="+", type=int, default=e14.DEFAULT_REPEAT_SEEDS)
    parser.add_argument("--oof-folds", type=int, default=3); parser.add_argument("--oof-split-seed", type=int, default=20260711)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=128); parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--tissue-loss-weight", type=float, default=0.1); parser.add_argument("--hla-loss-weight", type=float, default=0.1); parser.add_argument("--max-grad-norm", type=float, default=1.0); parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--max-members", type=int, default=20); parser.add_argument("--min-improvement", type=float, default=1e-4)
    root = project_path("results/tissuePMHC_e26_greedy_ensemble_selection")
    parser.add_argument("--oof-predictions-output", type=Path, default=root / "oof_predictions.csv"); parser.add_argument("--test-predictions-output", type=Path, default=root / "test_predictions.csv")
    parser.add_argument("--per-task-output", type=Path, default=root / "per_task_metrics.csv"); parser.add_argument("--summary-output", type=Path, default=root / "summary_metrics.csv"); parser.add_argument("--stability-output", type=Path, default=root / "stability_metrics.csv")
    parser.add_argument("--trajectory-output", type=Path, default=root / "oof_selection_trajectory.csv"); parser.add_argument("--members-output", type=Path, default=root / "selected_members.csv"); parser.add_argument("--selection-metadata-output", type=Path, default=root / "selection_metadata.json"); parser.add_argument("--metadata-output", type=Path, default=root / "metadata.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
