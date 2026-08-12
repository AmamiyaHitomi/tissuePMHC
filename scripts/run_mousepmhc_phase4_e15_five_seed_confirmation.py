#!/usr/bin/env python3
"""Run Phase 4 E15: preregistered five-seed confirmation of frozen E8/E3b.

The only eligible structure is the E3b task-balanced Factorized MMoE.  The
existing OOF members 20260704/05/06 are read as frozen predictions; this
script trains exactly the preregistered new members 20260707/08 on identical
three-fold, pair-grouped OOF splits.  It then compares equal-weight 5-seed
probability averaging with the frozen 3-seed probability average.

The fixed test data are not opened by default.  They can be read only when
--run-fixed-test is explicitly supplied *and* all preregistered OOF gates pass.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_mousepmhc_phase3_e3_factorized_mmoe_oof as e3
import run_mousepmhc_phase3_e5_famo_mmoe_oof as e5
import run_tissuepmhc_e26_all_in_one as folds
import run_tissuepmhc_neural_baselines_v2 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "mousePMHC_phase4_e15_five_seed_confirmation"
SOURCE_CANDIDATE = "mousePMHC_phase3_e3b_task_balanced_mmoe_min200"
THREE_SEED_CANDIDATE = "mousePMHC_phase4_e15_e3b_3seed_probability_mean"
FIVE_SEED_CANDIDATE = "mousePMHC_phase4_e15_e3b_5seed_probability_mean"
KEYS = ["sample_id", "target_tissue", "mhc_restriction", "label"]
METRICS = ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"]
DEFAULT_ORIGINAL_SEEDS = [20260704, 20260705, 20260706]
DEFAULT_NEW_SEEDS = [20260707, 20260708]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def format_duration(seconds: float) -> str:
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes)}m {remainder:04.1f}s" if minutes >= 1 else f"{remainder:.1f}s"


def validate_train(frame: pd.DataFrame) -> None:
    e3.validate_input(frame)


def validate_test(frame: pd.DataFrame) -> None:
    required = {"dataset", "split", "sample_id", "target_tissue", "mhc_restriction", "pair_id", "label", "peptide_sequence"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"E15 fixed test is missing columns: {sorted(missing)}")
    if set(frame.dataset) != {"mousePMHC"} or set(frame.split) != {"test"}:
        raise ValueError("E15 fixed test must contain mousePMHC test rows only.")
    if not frame.mhc_restriction.str.startswith("H2-").all() or not frame.peptide_sequence.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{9}").all():
        raise ValueError("E15 fixed test has non-H2 restrictions or invalid peptides.")


def train_e3b(args: argparse.Namespace, torch: Any, nn: Any, fitting: pd.DataFrame, mappings: dict[str, Any],
              peptide_length: int, device: str, seed: int, stage: str, fold: int | None) -> Any:
    """Exact E3b equal-weight task-balanced training, with Phase 4 timing logs."""
    seed_args = copy.copy(args)
    seed_args.seed = int(seed)
    model = e3.define_model(torch, nn, peptide_length, len(mappings["tasks"]), len(mappings["tissue_to_id"]), len(mappings["hla_to_id"]), seed_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    arrays = e5.task_arrays(fitting, mappings, peptide_length)
    steps = args.steps_per_epoch or int(np.ceil(max(len(item["label"]) for item in arrays) / args.task_batch_size))
    rng = np.random.default_rng(seed)
    location = f"fold={fold + 1}/{args.oof_folds}" if fold is not None else "full-train"
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter(); losses: list[float] = []; entropies: list[float] = []
        model.train()
        for _ in range(steps):
            batch = e5.sample_balanced_batch(rng, arrays, args.task_batch_size)
            peptide, task, tissue, h2, label = [torch.as_tensor(value, device=device) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            task_losses, gates = e5.task_loss_vector(torch, model, peptide, task, tissue, h2, label, len(arrays), args.task_batch_size)
            entropy = -(gates * torch.log(gates.clamp_min(1e-12))).sum(dim=1).mean()
            objective = task_losses.mean() - args.gate_entropy_weight * entropy
            objective.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(task_losses.mean().detach().cpu())); entropies.append(float(entropy.detach().cpu()))
        print(
            f"E15 stage={stage} seed={seed} {location} epoch={epoch}/{args.epochs} "
            f"task_balanced_bce={np.mean(losses):.5f} gate_entropy={np.mean(entropies):.5f} "
            f"elapsed={format_duration(time.perf_counter() - epoch_started)}",
            flush=True,
        )
    return model


def source_predictions(path: Path, original_seeds: list[int], train: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"split", "candidate", "seed", *KEYS, "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"E15 frozen E3b source is missing columns: {sorted(missing)}")
    frame = frame[(frame.split == "oof") & (frame.candidate == SOURCE_CANDIDATE) & frame.seed.isin(original_seeds)].copy()
    if set(frame.seed.unique()) != set(original_seeds):
        raise AssertionError("E15 frozen E3b source lacks one or more preregistered original seeds.")
    if frame.duplicated(["seed", "sample_id"]).any() or len(frame) != len(train) * len(original_seeds):
        raise AssertionError("E15 frozen E3b OOF source is incomplete or duplicated.")
    expected = set(map(tuple, train[KEYS].itertuples(index=False, name=None)))
    for seed, subset in frame.groupby("seed", sort=True):
        if set(map(tuple, subset[KEYS].itertuples(index=False, name=None))) != expected:
            raise AssertionError(f"E15 frozen source seed {seed} does not align with current train data.")
    return frame


def ensemble_predictions(members: pd.DataFrame, seeds: list[int], candidate: str) -> pd.DataFrame:
    grouped = members[members.seed.isin(seeds)].groupby(KEYS, as_index=False).agg(
        score=("score", "mean"), prediction_std_across_seeds=("score", lambda values: float(np.std(values, ddof=0))),
        n_members=("seed", "nunique"),
    )
    if not (grouped.n_members == len(seeds)).all():
        raise AssertionError(f"E15 {candidate} ensemble does not contain every expected seed per row.")
    grouped.insert(0, "split", "oof"); grouped.insert(1, "candidate", candidate)
    return grouped[["split", "candidate", *KEYS, "score", "prediction_std_across_seeds", "n_members"]]


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for (candidate, tissue, h2), task in predictions.groupby(["candidate", "target_tissue", "mhc_restriction"], sort=True):
        records.append({"experiment_name": EXPERIMENT, "candidate": candidate, "target_tissue": tissue,
                        "mhc_restriction": h2, "oof_rows": len(task),
                        **base.evaluate(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float))})
    per_task = pd.DataFrame(records); rows: list[dict[str, object]] = []
    for candidate, tasks in per_task.groupby("candidate", sort=True):
        row: dict[str, object] = {"experiment_name": EXPERIMENT, "candidate": candidate}
        for metric in METRICS:
            row[f"mean_task_{metric}"] = float(tasks[metric].mean())
        row["worst_task_auroc"] = float(tasks.auroc.min()); row["worst6_task_auroc"] = float(tasks.nsmallest(6, "auroc").auroc.mean())
        rows.append(row)
    return per_task, pd.DataFrame(rows)


def oof_gate(summary: pd.DataFrame, args: argparse.Namespace) -> tuple[dict[str, object], bool]:
    three = summary[summary.candidate == THREE_SEED_CANDIDATE]
    five = summary[summary.candidate == FIVE_SEED_CANDIDATE]
    if len(three) != 1 or len(five) != 1:
        raise AssertionError("E15 requires exactly one 3-seed and one 5-seed OOF summary.")
    three_row, five_row = three.iloc[0], five.iloc[0]
    deltas = {
        "mean_task_auroc": float(five_row.mean_task_auroc - three_row.mean_task_auroc),
        "mean_task_auprc": float(five_row.mean_task_auprc - three_row.mean_task_auprc),
        "worst6_task_auroc": float(five_row.worst6_task_auroc - three_row.worst6_task_auroc),
    }
    checks = {
        "mean_auroc_gain_at_least_minimum": deltas["mean_task_auroc"] >= args.min_mean_auroc_gain,
        "mean_auprc_not_too_low": deltas["mean_task_auprc"] >= -args.max_mean_auprc_drop,
        "worst6_auroc_not_too_low": deltas["worst6_task_auroc"] >= -args.max_worst6_auroc_drop,
    }
    payload: dict[str, object] = {
        "three_seed_candidate": THREE_SEED_CANDIDATE, "five_seed_candidate": FIVE_SEED_CANDIDATE,
        "three_seed_metrics": {metric: float(three_row[metric]) for metric in ["mean_task_auroc", "mean_task_auprc", "worst6_task_auroc"]},
        "five_seed_metrics": {metric: float(five_row[metric]) for metric in ["mean_task_auroc", "mean_task_auprc", "worst6_task_auroc"]},
        "deltas": deltas,
        "thresholds": {"min_mean_auroc_gain": args.min_mean_auroc_gain, "max_mean_auprc_drop": args.max_mean_auprc_drop,
                       "max_worst6_auroc_drop": args.max_worst6_auroc_drop},
        "checks": checks,
    }
    return payload, bool(all(checks.values()))


def full_train_test_predictions(args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any,
                                raw_train: pd.DataFrame, device: str, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Read fixed test only after the OOF gate and generate one 5-seed test ensemble."""
    raw_test = base.read_dataset(args.test)
    validate_test(raw_test)
    train, test, mappings = base.add_task_columns(raw_train, raw_test)
    peptide_length = int(train.peptide_sequence.str.len().iloc[0])
    if set(test.task_name) - set(mappings["tasks"]):
        raise AssertionError("E15 fixed test contains a task absent from train.")
    parts: list[pd.DataFrame] = []; parameter_count = 0
    for seed in seeds:
        seed_started = time.perf_counter(); base.set_seed(int(seed), torch)
        print(f"E15 fixed-test seed={seed} full-train start train_rows={len(train)} test_rows={len(test)} device={device}", flush=True)
        model = train_e3b(args, torch, nn, train, mappings, peptide_length, device, int(seed), "fixed-test", None)
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        scores = e5.predict(torch, DataLoader, TensorDataset, model, test, peptide_length, args.batch_size, device)
        output = test[KEYS].copy(); output.insert(0, "split", "test"); output.insert(1, "candidate", FIVE_SEED_CANDIDATE); output.insert(2, "seed", int(seed)); output["score"] = scores
        parts.append(output[["split", "candidate", "seed", *KEYS, "score"]])
        print(f"E15 fixed-test seed={seed} complete elapsed={format_duration(time.perf_counter() - seed_started)}", flush=True)
    member_predictions = pd.concat(parts, ignore_index=True)
    ensemble = member_predictions.groupby(KEYS, as_index=False).agg(score=("score", "mean"), n_members=("seed", "nunique"))
    if not (ensemble.n_members == len(seeds)).all():
        raise AssertionError("E15 fixed-test ensemble lacks a member for one or more rows.")
    ensemble.insert(0, "split", "test"); ensemble.insert(1, "candidate", FIVE_SEED_CANDIDATE)
    metrics_rows: list[dict[str, object]] = []
    for (tissue, h2), task in ensemble.groupby(["target_tissue", "mhc_restriction"], sort=True):
        metrics_rows.append({"experiment_name": EXPERIMENT, "candidate": FIVE_SEED_CANDIDATE, "target_tissue": tissue,
                             "mhc_restriction": h2, "test_rows": len(task), **base.evaluate(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float))})
    per_task = pd.DataFrame(metrics_rows)
    summary: dict[str, object] = {"experiment_name": EXPERIMENT, "candidate": FIVE_SEED_CANDIDATE}
    for metric in METRICS:
        summary[f"mean_task_{metric}"] = float(per_task[metric].mean())
    summary["worst_task_auroc"] = float(per_task.auroc.min()); summary["worst6_task_auroc"] = float(per_task.nsmallest(6, "auroc").auroc.mean())
    return ensemble, pd.concat([per_task, pd.DataFrame([summary])], ignore_index=True, sort=False), parameter_count


def run(args: argparse.Namespace) -> None:
    total_started = time.perf_counter()
    if args.original_seeds != DEFAULT_ORIGINAL_SEEDS or args.new_seeds != DEFAULT_NEW_SEEDS:
        raise ValueError(f"E15 is preregistered for original seeds {DEFAULT_ORIGINAL_SEEDS} and new seeds {DEFAULT_NEW_SEEDS} exactly.")
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    raw_train = base.read_dataset(args.train); validate_train(raw_train)
    train, _, mappings = base.add_task_columns(raw_train, raw_train.copy())
    peptide_length = int(train.peptide_sequence.str.len().iloc[0])
    original = source_predictions(args.original_oof_predictions, args.original_seeds, train)
    assignments = folds.make_pair_grouped_folds(train, args.oof_folds, args.oof_split_seed)
    new_parts: list[pd.DataFrame] = []; parameter_count: int | None = None
    for seed in args.new_seeds:
        seed_started = time.perf_counter()
        for fold in range(args.oof_folds):
            fold_started = time.perf_counter(); fitting, held_out = train[assignments != fold].copy(), train[assignments == fold].copy()
            base.set_seed(int(seed), torch)
            print(f"E15 OOF new-seed={seed} fold={fold + 1}/{args.oof_folds} start fit_rows={len(fitting)} holdout_rows={len(held_out)} device={device}", flush=True)
            model = train_e3b(args, torch, nn, fitting, mappings, peptide_length, device, int(seed), "oof", fold)
            if parameter_count is None:
                parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
            scores = e5.predict(torch, DataLoader, TensorDataset, model, held_out, peptide_length, args.batch_size, device)
            output = held_out[KEYS].copy(); output.insert(0, "split", "oof"); output.insert(1, "candidate", SOURCE_CANDIDATE); output.insert(2, "seed", int(seed)); output["score"] = scores
            new_parts.append(output[["split", "candidate", "seed", *KEYS, "score"]])
            print(f"E15 OOF new-seed={seed} fold={fold + 1}/{args.oof_folds} complete elapsed={format_duration(time.perf_counter() - fold_started)}", flush=True)
        print(f"E15 OOF new-seed={seed} complete elapsed={format_duration(time.perf_counter() - seed_started)}", flush=True)
    new_predictions = pd.concat(new_parts, ignore_index=True)
    if len(new_predictions) != len(train) * len(args.new_seeds) or new_predictions.duplicated(["seed", "sample_id"]).any():
        raise AssertionError("E15 new-seed OOF predictions must cover every train row exactly once per seed.")
    members = pd.concat([original, new_predictions], ignore_index=True)
    three = ensemble_predictions(members, args.original_seeds, THREE_SEED_CANDIDATE)
    five = ensemble_predictions(members, [*args.original_seeds, *args.new_seeds], FIVE_SEED_CANDIDATE)
    oof_predictions = pd.concat([three, five], ignore_index=True)
    per_task, summary = metric_tables(oof_predictions)
    gate, passed = oof_gate(summary, args); gate["passed"] = passed
    args.output_dir.mkdir(parents=True, exist_ok=True)
    new_predictions.to_csv(args.output_dir / "mousePMHC_phase4_e15_new_seed_oof_predictions.csv", index=False)
    oof_predictions.to_csv(args.output_dir / "mousePMHC_phase4_e15_oof_predictions.csv", index=False)
    per_task.to_csv(args.output_dir / "mousePMHC_phase4_e15_oof_per_task_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "mousePMHC_phase4_e15_oof_summary_metrics.csv", index=False)
    (args.output_dir / "mousePMHC_phase4_e15_oof_gate.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")
    test_data_read = False; test_outputs: dict[str, str] = {}; fixed_test_parameter_count: int | None = None
    if args.run_fixed_test:
        if not passed:
            print("E15 OOF gate failed: fixed test was not opened.", flush=True)
        else:
            test_data_read = True
            test_ensemble, test_metrics, fixed_test_parameter_count = full_train_test_predictions(args, torch, nn, DataLoader, TensorDataset, raw_train, device, [*args.original_seeds, *args.new_seeds])
            test_ensemble.to_csv(args.output_dir / "mousePMHC_phase4_e15_fixed_test_predictions.csv", index=False)
            test_metrics.to_csv(args.output_dir / "mousePMHC_phase4_e15_fixed_test_metrics.csv", index=False)
            test_outputs = {"predictions": "mousePMHC_phase4_e15_fixed_test_predictions.csv", "metrics": "mousePMHC_phase4_e15_fixed_test_metrics.csv"}
    metadata = {
        "experiment_name": EXPERIMENT, "frozen_structure": "E3b task-balanced Factorized MMoE", "fusion": "equal-weight probability mean",
        "test_data_read": test_data_read, "fixed_test_requested": args.run_fixed_test, "oof_gate_passed": passed,
        "train": str(args.train), "test": str(args.test) if test_data_read else None, "n_rows": len(train), "n_pairs": int(train.pair_id.nunique()),
        "n_tasks": len(mappings["tasks"]), "original_seeds": args.original_seeds, "new_seeds": args.new_seeds,
        "oof_folds": args.oof_folds, "oof_split_seed": args.oof_split_seed, "device": device, "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch, "task_batch_size": args.task_batch_size, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "embedding_dim": args.embedding_dim, "hidden_dim": args.hidden_dim,
        "expert_dim": args.expert_dim, "condition_dim": args.condition_dim, "gate_hidden_dim": args.gate_hidden_dim, "n_experts": args.n_experts,
        "dropout": args.dropout, "gate_entropy_weight": args.gate_entropy_weight, "max_grad_norm": args.max_grad_norm,
        "parameter_count": parameter_count, "fixed_test_parameter_count": fixed_test_parameter_count,
        "original_oof_predictions": str(args.original_oof_predictions), "oof_gate": gate, "fixed_test_outputs": test_outputs,
        "preregistration": "new seeds, equal weights, and OOF gate are fixed; no seed subset selection or weight tuning is permitted",
    }
    (args.output_dir / "mousePMHC_phase4_e15_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("E15 OOF summary", flush=True); print(summary.to_string(index=False), flush=True)
    print("E15 OOF gate", flush=True); print(json.dumps(gate, indent=2, ensure_ascii=False), flush=True)
    print(f"E15 total complete elapsed={format_duration(time.perf_counter() - total_started)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument("--test", type=Path, default=project_path("data/mousePMHC/mousePMHC_test.csv.gz"))
    parser.add_argument("--original-oof-predictions", type=Path, default=project_path("results/mousePMHC_phase3_e3b_task_balanced_mmoe_min200_oof/mousePMHC_phase3_e3b_oof_predictions.csv"))
    parser.add_argument("--output-dir", type=Path, default=project_path("results/mousePMHC_phase4_e15_five_seed_confirmation"))
    parser.add_argument("--original-seeds", nargs="+", type=int, default=DEFAULT_ORIGINAL_SEEDS)
    parser.add_argument("--new-seeds", nargs="+", type=int, default=DEFAULT_NEW_SEEDS)
    parser.add_argument("--oof-folds", type=int, default=3); parser.add_argument("--oof-split-seed", type=int, default=20260711)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--task-batch-size", type=int, default=16); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=128); parser.add_argument("--expert-dim", type=int, default=64)
    parser.add_argument("--condition-dim", type=int, default=16); parser.add_argument("--gate-hidden-dim", type=int, default=64); parser.add_argument("--n-experts", type=int, default=3); parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.01); parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--min-mean-auroc-gain", type=float, default=0.0010); parser.add_argument("--max-mean-auprc-drop", type=float, default=0.0005); parser.add_argument("--max-worst6-auroc-drop", type=float, default=0.0010)
    parser.add_argument("--run-fixed-test", action="store_true", help="Read fixed test only after all preregistered OOF gates pass.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
