#!/usr/bin/env python3
"""Run Phase 5 E17 task-wise ranking losses on the frozen E3b MMoE.

E17a ranks the positive and negative rows from the same original pair_id.
E17b ranks every positive against every negative inside each task sub-batch.
Both variants and their matched BCE-only E3b control use identical pair-aware,
task-balanced fitting batches.  The fixed test split is never read.

Timing is printed for every epoch, every seed, and the complete run.  Measured
times are deliberately not written to any result file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
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
EXPERIMENT = "mousePMHC_phase5_e17_taskwise_ranking_mmoe_oof"
BASELINE = "e3b_matched_pair_batch_control"
E17A = "e17a_matched_pair_ranking"
E17B = "e17b_taskwise_all_pair_ranking"
KEYS = ["sample_id", "target_tissue", "mhc_restriction"]
METRICS = ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc", "brier"]


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def validate_pairs(frame: pd.DataFrame) -> None:
    grouped = frame.groupby(["task_name", "pair_id"], sort=False).label.agg(["size", "nunique", "sum"])
    bad = grouped[(grouped["size"] != 2) | (grouped["nunique"] != 2) | (grouped["sum"] != 1)]
    if len(bad):
        raise ValueError(f"E17 requires exactly one positive and one negative per task/pair_id; found {len(bad)} invalid pairs.")


def pair_task_arrays(frame: pd.DataFrame, mappings: dict[str, Any], peptide_length: int) -> list[dict[str, Any]]:
    arrays: list[dict[str, Any]] = []
    for task_name in mappings["tasks"]:
        task = frame[frame.task_name == task_name].copy()
        positive = task[task.label == 1].set_index("pair_id").sort_index()
        negative = task[task.label == 0].set_index("pair_id").sort_index()
        if not positive.index.equals(negative.index):
            raise AssertionError(f"Positive/negative pair_id mismatch in {task_name}.")
        arrays.append({
            "task_name": task_name,
            "task_id": int(mappings["task_to_id"][task_name]),
            "positive_peptide": base.encode_peptides(positive.peptide_sequence, peptide_length),
            "negative_peptide": base.encode_peptides(negative.peptide_sequence, peptide_length),
            "tissue_id": int(task.tissue_id.iloc[0]), "h2_id": int(task.hla_id.iloc[0]),
            "n_pairs": len(positive),
        })
    return arrays


def sample_pair_balanced_batch(
    rng: np.random.Generator, arrays: list[dict[str, Any]], task_batch_size: int,
) -> tuple[np.ndarray, ...]:
    """Return equal-size task blocks, each made from complete positive/negative pairs."""
    pairs_per_task = task_batch_size // 2
    samples: list[list[np.ndarray]] = [[], [], [], [], [], []]
    for item in arrays:
        chosen = rng.integers(0, item["n_pairs"], size=pairs_per_task)
        peptide = np.empty((task_batch_size, item["positive_peptide"].shape[1]), dtype=np.int64)
        peptide[0::2] = item["positive_peptide"][chosen]
        peptide[1::2] = item["negative_peptide"][chosen]
        samples[0].append(peptide)
        samples[1].append(np.full(task_batch_size, item["task_id"], dtype=np.int64))
        samples[2].append(np.full(task_batch_size, item["tissue_id"], dtype=np.int64))
        samples[3].append(np.full(task_batch_size, item["h2_id"], dtype=np.int64))
        samples[4].append(np.tile(np.asarray([1, 0], dtype=np.int64), pairs_per_task))
        samples[5].append(np.repeat(np.arange(pairs_per_task, dtype=np.int64), 2))
    return tuple(np.concatenate(values, axis=0) for values in samples)


def task_losses(
    torch: Any, model: Any, peptide: Any, task: Any, tissue: Any, h2: Any, label: Any,
    n_tasks: int, task_batch_size: int, candidate: str, ranking_lambda: float, tau: float,
) -> tuple[Any, Any, Any, int]:
    logits, gates = model(peptide, task, tissue, h2, return_gates=True)
    per_row = torch.nn.functional.binary_cross_entropy_with_logits(logits, label.float(), reduction="none")
    bce = per_row.reshape(n_tasks, task_batch_size).mean(dim=1)
    if candidate == BASELINE:
        rank = logits.new_zeros(n_tasks)
        return bce, rank, gates, 0
    task_logits = logits.reshape(n_tasks, task_batch_size)
    # The pair-aware sampler guarantees [positive, negative] for each adjacent row pair.
    positive = task_logits[:, 0::2]
    negative = task_logits[:, 1::2]
    if candidate == E17A:
        differences = positive - negative
        comparisons_per_task = positive.shape[1]
    elif candidate == E17B:
        differences = positive.unsqueeze(2) - negative.unsqueeze(1)
        comparisons_per_task = positive.shape[1] * negative.shape[1]
    else:
        raise ValueError(f"Unknown candidate: {candidate}")
    rank = torch.nn.functional.softplus(-differences / tau).reshape(n_tasks, -1).mean(dim=1)
    return bce, rank, gates, int(comparisons_per_task * n_tasks)


def train_fold(
    args: argparse.Namespace, torch: Any, nn: Any, fitting: pd.DataFrame, mappings: dict[str, Any],
    peptide_length: int, device: str, seed: int, fold: int, candidate: str,
) -> tuple[Any, list[dict[str, Any]], int]:
    model = e3.define_model(
        torch, nn, peptide_length, len(mappings["tasks"]), len(mappings["tissue_to_id"]),
        len(mappings["hla_to_id"]), args,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    arrays = pair_task_arrays(fitting, mappings, peptide_length)
    n_tasks = len(arrays)
    steps = args.steps_per_epoch or int(math.ceil(max(item["n_pairs"] * 2 for item in arrays) / args.task_batch_size))
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    peak_memory = 0
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train(); bce_values: list[float] = []; rank_values: list[float] = []; entropies: list[float] = []
        comparison_count = 0
        for _ in range(steps):
            batch = sample_pair_balanced_batch(rng, arrays, args.task_batch_size)
            peptide, task, tissue, h2, label, _ = [torch.as_tensor(value, device=device) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            bce, rank, gates, comparisons = task_losses(
                torch, model, peptide, task, tissue, h2, label, n_tasks, args.task_batch_size,
                candidate, args.ranking_lambda, args.ranking_tau,
            )
            entropy = -(gates * torch.log(gates.clamp_min(1e-12))).sum(dim=1).mean()
            objective = bce.mean() + args.ranking_lambda * rank.mean() - args.gate_entropy_weight * entropy
            objective.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            bce_values.append(float(bce.mean().detach().cpu()))
            rank_values.append(float(rank.mean().detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))
            comparison_count += comparisons
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated()))
        epoch_seconds = time.perf_counter() - epoch_started
        history.append({
            "experiment_name": EXPERIMENT, "candidate": candidate, "seed": seed, "fold": fold,
            "epoch": epoch, "mean_task_bce": float(np.mean(bce_values)),
            "mean_task_ranking_loss": float(np.mean(rank_values)),
            "mean_gate_entropy": float(np.mean(entropies)), "ranking_comparisons": comparison_count,
        })
        print(
            f"E17 candidate={candidate} seed={seed} fold={fold + 1}/{args.oof_folds} "
            f"epoch={epoch}/{args.epochs} bce={np.mean(bce_values):.4f} rank={np.mean(rank_values):.4f} "
            f"epoch_seconds={epoch_seconds:.3f}", flush=True,
        )
    return model, history, peak_memory


def predict(
    torch: Any, DataLoader: Any, TensorDataset: Any, model: Any, frame: pd.DataFrame,
    peptide_length: int, batch_size: int, device: str,
) -> np.ndarray:
    return e5.predict(torch, DataLoader, TensorDataset, model, frame, peptide_length, batch_size, device)


def evaluate_with_brier(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    result = base.evaluate(labels, scores)
    result["brier"] = float(np.mean((scores - labels) ** 2))
    return result


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (candidate, seed, tissue, h2), task in predictions.groupby(
        ["candidate", "seed", "target_tissue", "mhc_restriction"], sort=True,
    ):
        rows.append({
            "experiment_name": EXPERIMENT, "candidate": candidate, "seed": int(seed),
            "target_tissue": tissue, "mhc_restriction": h2, "oof_rows": len(task),
            **evaluate_with_brier(task.label.to_numpy(dtype=int), task.score.to_numpy(dtype=float)),
        })
    per_task = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for (candidate, seed), subset in per_task.groupby(["candidate", "seed"], sort=True):
        row: dict[str, Any] = {"experiment_name": EXPERIMENT, "candidate": candidate, "seed": int(seed)}
        for metric in METRICS:
            row[f"mean_task_{metric}"] = float(subset[metric].mean())
        row["worst_task_auroc"] = float(subset.auroc.min())
        row["worst6_task_auroc"] = float(subset.nsmallest(6, "auroc").auroc.mean())
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    stability_rows: list[dict[str, Any]] = []
    numeric = [column for column in summary if column.startswith("mean_task_") or column.startswith("worst")]
    for candidate, subset in summary.groupby("candidate", sort=True):
        for metric in numeric:
            stability_rows.append({
                "experiment_name": EXPERIMENT, "candidate": candidate, "metric": metric,
                "seed_count": len(subset), "seed_mean": float(subset[metric].mean()),
                "seed_sd": float(subset[metric].std(ddof=1)) if len(subset) > 1 else float("nan"),
                "seed_min": float(subset[metric].min()), "seed_max": float(subset[metric].max()),
            })
    return per_task, summary, pd.DataFrame(stability_rows)


def task_deltas(per_task: pd.DataFrame, candidate: str) -> pd.DataFrame:
    keys = ["seed", "target_tissue", "mhc_restriction"]
    baseline = per_task[per_task.candidate == BASELINE].set_index(keys)
    selected = per_task[per_task.candidate == candidate].set_index(keys)
    common = baseline.index.intersection(selected.index)
    rows: list[dict[str, Any]] = []
    for key in common:
        row: dict[str, Any] = {
            "candidate": candidate, "baseline_candidate": BASELINE, "seed": int(key[0]),
            "target_tissue": key[1], "mhc_restriction": key[2],
        }
        for metric in METRICS:
            # All deltas use candidate - baseline; lower Brier is therefore better.
            row[f"delta_{metric}"] = float(selected.loc[key, metric] - baseline.loc[key, metric])
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, iterations: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    means = np.empty(iterations, dtype=float)
    for start in range(0, iterations, 1000):
        stop = min(start + 1000, iterations)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def comparison_tables(
    per_task: pd.DataFrame, summary: pd.DataFrame, candidate: str, args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    deltas = task_deltas(per_task, candidate)
    averaged = deltas.groupby(["target_tissue", "mhc_restriction"], sort=True)[
        [f"delta_{metric}" for metric in METRICS]
    ].mean().reset_index()
    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        lower, upper = bootstrap_ci(averaged[f"delta_{metric}"].to_numpy(dtype=float), rng, args.bootstrap_iterations)
        bootstrap_rows.append({
            "candidate": candidate, "baseline_candidate": BASELINE, "metric": metric,
            "task_mean_delta": float(averaged[f"delta_{metric}"].mean()),
            "ci95_lower": lower, "ci95_upper": upper, "bootstrap_iterations": args.bootstrap_iterations,
        })
    bootstrap = pd.DataFrame(bootstrap_rows)
    h2 = averaged.groupby("mhc_restriction", sort=True)[[f"delta_{metric}" for metric in METRICS]].mean().reset_index()
    candidate_summary = summary[summary.candidate == candidate].set_index("seed")
    baseline_summary = summary[summary.candidate == BASELINE].set_index("seed")
    common_seeds = candidate_summary.index.intersection(baseline_summary.index)
    mean_auroc_delta = float((candidate_summary.loc[common_seeds, "mean_task_auroc"] - baseline_summary.loc[common_seeds, "mean_task_auroc"]).mean())
    mean_auprc_delta = float((candidate_summary.loc[common_seeds, "mean_task_auprc"] - baseline_summary.loc[common_seeds, "mean_task_auprc"]).mean())
    worst6_delta = float((candidate_summary.loc[common_seeds, "worst6_task_auroc"] - baseline_summary.loc[common_seeds, "worst6_task_auroc"]).mean())
    auroc_ci = bootstrap.set_index("metric").loc["auroc"]
    mcc_ci = bootstrap.set_index("metric").loc["mcc"]
    decision = {
        "candidate": candidate, "matched_seeds": [int(value) for value in common_seeds],
        "mean_task_auroc_delta": mean_auroc_delta, "mean_task_auprc_delta": mean_auprc_delta,
        "worst6_task_auroc_delta": worst6_delta,
        "worst_task_auroc_delta": float((candidate_summary.loc[common_seeds, "worst_task_auroc"] - baseline_summary.loc[common_seeds, "worst_task_auroc"]).mean()),
        "mean_task_mcc_delta": float(averaged.delta_mcc.mean()),
        "mean_task_brier_delta": float(averaged.delta_brier.mean()),
        "task_auroc_wins": int((averaged.delta_auroc > 0).sum()),
        "task_auroc_losses": int((averaged.delta_auroc < 0).sum()),
        "minimum_h2_auroc_delta": float(h2.delta_auroc.min()),
        "auroc_bootstrap_ci95": [float(auroc_ci.ci95_lower), float(auroc_ci.ci95_upper)],
        "mcc_bootstrap_ci95": [float(mcc_ci.ci95_lower), float(mcc_ci.ci95_upper)],
    }
    return deltas, h2, bootstrap, decision


def screening_pass(decision: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    checks = {
        "auroc_noninferiority": decision["mean_task_auroc_delta"] >= -0.001,
        "auprc_protection": decision["mean_task_auprc_delta"] >= -0.001,
        "worst6_protection": decision["worst6_task_auroc_delta"] >= -0.003,
        "mcc_not_significantly_worse": decision["mcc_bootstrap_ci95"][1] >= 0.0,
    }
    return all(checks.values()), checks


def formal_pass(decision: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    checks = {
        "auroc_gain": decision["mean_task_auroc_delta"] >= 0.003,
        "auprc_protection": decision["mean_task_auprc_delta"] >= -0.001,
        "worst6_protection": decision["worst6_task_auroc_delta"] >= -0.003,
        "h2_protection": decision["minimum_h2_auroc_delta"] >= -0.010,
        "task_wins": decision["task_auroc_wins"] >= 14,
        "auroc_ci_lower": decision["auroc_bootstrap_ci95"][0] > -0.001,
        "mcc_not_significantly_worse": decision["mcc_bootstrap_ci95"][1] >= 0.0,
    }
    return all(checks.values()), checks


def run_candidate_seed(
    args: argparse.Namespace, torch: Any, nn: Any, DataLoader: Any, TensorDataset: Any,
    train: pd.DataFrame, assignments: pd.Series, mappings: dict[str, Any], peptide_length: int,
    device: str, seed: int, candidate: str,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], int]:
    parts: list[pd.DataFrame] = []; history: list[dict[str, Any]] = []; peak_memory = 0
    for fold in range(args.oof_folds):
        fitting, held_out = train[assignments != fold].copy(), train[assignments == fold].copy()
        base.set_seed(seed, torch)
        print(f"E17 candidate={candidate} seed={seed} fold={fold + 1}/{args.oof_folds} fit={len(fitting)} holdout={len(held_out)}", flush=True)
        model, fold_history, fold_peak = train_fold(
            args, torch, nn, fitting, mappings, peptide_length, device, seed, fold, candidate,
        )
        scores = predict(torch, DataLoader, TensorDataset, model, held_out, peptide_length, args.batch_size, device)
        output = held_out[KEYS + ["label"]].copy()
        output.insert(0, "split", "oof"); output.insert(1, "candidate", candidate); output.insert(2, "seed", seed)
        output["score"] = scores
        parts.append(output[["split", "candidate", "seed", *KEYS, "label", "score"]])
        history.extend(fold_history); peak_memory = max(peak_memory, fold_peak)
    return parts, history, peak_memory


def validate_protocol(args: argparse.Namespace) -> None:
    if not args.seeds:
        raise ValueError("At least one seed is required.")
    if args.task_batch_size < 2 or args.task_batch_size % 2:
        raise ValueError("--task-batch-size must be an even integer >=2 for complete pairs.")
    if args.ranking_lambda != 0.25 or args.ranking_tau != 1.0:
        raise ValueError("E17 is preregistered with lambda=0.25 and tau=1.0; these values may not be changed.")
    if args.n_experts != 3:
        raise ValueError("E17 is frozen to the three-expert E3b architecture.")


def run(args: argparse.Namespace) -> None:
    validate_protocol(args)
    total_started = time.perf_counter()
    torch, nn, DataLoader, TensorDataset = base.require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    raw = base.read_dataset(args.train); e3.validate_input(raw)
    train, _, mappings = base.add_task_columns(raw, raw.copy())
    validate_pairs(train)
    if len(mappings["tasks"]) != 24:
        raise ValueError(f"E17 expects 24 tasks, found {len(mappings['tasks'])}.")
    peptide_length = int(train.peptide_sequence.str.len().max())
    assignments = folds.make_pair_grouped_folds(train, args.oof_folds, args.oof_split_seed)
    prediction_parts: list[pd.DataFrame] = []; history: list[dict[str, Any]] = []
    peak_memory_by_candidate: dict[str, int] = {}
    seed_started = time.perf_counter()
    screen_seed = int(args.seeds[0])
    for candidate in [BASELINE, E17A, E17B]:
        parts, records, peak = run_candidate_seed(
            args, torch, nn, DataLoader, TensorDataset, train, assignments, mappings,
            peptide_length, device, screen_seed, candidate,
        )
        prediction_parts.extend(parts); history.extend(records)
        peak_memory_by_candidate[candidate] = max(peak_memory_by_candidate.get(candidate, 0), peak)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    print(f"E17 seed={screen_seed} seed_seconds={time.perf_counter() - seed_started:.3f}", flush=True)

    screen_predictions = pd.concat(prediction_parts, ignore_index=True)
    screen_per_task, screen_summary, _ = metric_tables(screen_predictions)
    scores = screen_summary[screen_summary.seed == screen_seed].set_index("candidate").mean_task_auroc
    difference = float(scores[E17B] - scores[E17A])
    selected = E17A if abs(difference) < 0.001 else (E17B if difference > 0 else E17A)
    _, _, _, screen_decision = comparison_tables(screen_per_task, screen_summary, selected, args)
    passed_screen, screen_checks = screening_pass(screen_decision)
    print(
        f"E17 screening seed={screen_seed} e17a_auroc={scores[E17A]:.6f} e17b_auroc={scores[E17B]:.6f} "
        f"selected={selected} pass={passed_screen}", flush=True,
    )

    if passed_screen:
        for seed_value in args.seeds[1:]:
            seed = int(seed_value); seed_started = time.perf_counter()
            for candidate in [BASELINE, selected]:
                parts, records, peak = run_candidate_seed(
                    args, torch, nn, DataLoader, TensorDataset, train, assignments, mappings,
                    peptide_length, device, seed, candidate,
                )
                prediction_parts.extend(parts); history.extend(records)
                peak_memory_by_candidate[candidate] = max(peak_memory_by_candidate.get(candidate, 0), peak)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            print(f"E17 seed={seed} seed_seconds={time.perf_counter() - seed_started:.3f}", flush=True)
    else:
        print("E17 screening failed; additional seeds are not run by the preregistered stopping rule.", flush=True)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    if predictions.duplicated(["candidate", "seed", "sample_id"]).any():
        raise AssertionError("Duplicate E17 candidate/seed/sample OOF prediction.")
    expected_per_member = len(train)
    member_counts = predictions.groupby(["candidate", "seed"]).size()
    if not (member_counts == expected_per_member).all():
        raise AssertionError("Every executed E17 candidate/seed must cover all training rows exactly once.")
    per_task, summary, stability = metric_tables(predictions)
    deltas, h2, bootstrap, decision = comparison_tables(per_task, summary, selected, args)
    decision["screen_seed"] = screen_seed
    decision["e17a_screen_auroc"] = float(scores[E17A]); decision["e17b_screen_auroc"] = float(scores[E17B])
    decision["e17b_minus_e17a_screen_auroc"] = difference
    decision["selection_rule"] = "choose higher screen-seed mean task AUROC; if absolute difference <0.001 choose E17a"
    decision["screen_pass"] = passed_screen; decision["screen_checks"] = screen_checks
    if passed_screen and len(args.seeds) >= 3:
        passed_formal, formal_checks = formal_pass(decision)
        decision["formal_three_seed_pass"] = passed_formal; decision["formal_checks"] = formal_checks
        decision["final_status"] = "advance_to_e24_candidate_pool" if passed_formal else "stop_after_three_seed_confirmation"
    elif passed_screen:
        decision["formal_three_seed_pass"] = None; decision["formal_checks"] = {}
        decision["final_status"] = "screen_passed_but_three_seed_confirmation_not_requested"
    else:
        decision["formal_three_seed_pass"] = False; decision["formal_checks"] = {}
        decision["final_status"] = "stopped_after_single_seed_screen"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "mousePMHC_phase5_e17"
    predictions.to_csv(args.output_dir / f"{prefix}_oof_predictions.csv", index=False)
    per_task.to_csv(args.output_dir / f"{prefix}_oof_per_task_metrics.csv", index=False)
    summary.to_csv(args.output_dir / f"{prefix}_oof_summary_metrics.csv", index=False)
    stability.to_csv(args.output_dir / f"{prefix}_oof_stability_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / f"{prefix}_mechanism_diagnostics.csv", index=False)
    deltas.to_csv(args.output_dir / f"{prefix}_paired_task_deltas.csv", index=False)
    h2.to_csv(args.output_dir / f"{prefix}_h2_macro_deltas.csv", index=False)
    bootstrap.to_csv(args.output_dir / f"{prefix}_task_paired_bootstrap.csv", index=False)
    summary[summary.candidate == BASELINE].to_csv(args.output_dir / f"{prefix}_matched_e3b_baseline.csv", index=False)
    (args.output_dir / f"{prefix}_decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    parameter_probe = e3.define_model(
        torch, nn, peptide_length, len(mappings["tasks"]), len(mappings["tissue_to_id"]),
        len(mappings["hla_to_id"]), args,
    )
    parameter_count = sum(parameter.numel() for parameter in parameter_probe.parameters())
    metadata = {
        "experiment_name": EXPERIMENT, "test_data_read": False,
        "train": str(args.train), "n_rows": len(train), "n_pairs": int(train.pair_id.nunique()),
        "n_tasks": len(mappings["tasks"]), "seeds_requested": args.seeds,
        "seeds_executed_by_candidate": {
            name: sorted(int(value) for value in values.seed.unique())
            for name, values in predictions.groupby("candidate", sort=True)
        },
        "oof_folds": args.oof_folds, "oof_split_seed": args.oof_split_seed,
        "backbone": "E3b Factorized MMoE", "epochs": args.epochs,
        "task_batch_size_rows": args.task_batch_size, "pairs_per_task_batch": args.task_batch_size // 2,
        "batch_policy": "each task block samples complete fitting-only pair_ids; identical batches for baseline/E17a/E17b",
        "ranking_lambda": args.ranking_lambda, "ranking_tau": args.ranking_tau,
        "e17a": "softplus(-(positive_logit-negative_logit)/tau) for matched pair_id only",
        "e17b": "mean softplus(-(positive_logit-negative_logit)/tau) over all positive-negative combinations within task block",
        "selected_variant": selected, "model_parameter_count": int(parameter_count),
        "peak_gpu_memory_bytes_by_candidate": peak_memory_by_candidate,
        "timing_policy": "epoch, seed, and total times are printed to the terminal only; no measured time is saved in result files",
        "bootstrap_iterations": args.bootstrap_iterations, "bootstrap_seed": args.bootstrap_seed,
        "device": device, "git_commit": git_commit(), "cli": [sys.executable, *sys.argv],
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__, "cuda_available": bool(torch.cuda.is_available()),
                        "cuda_version": torch.version.cuda, "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV")},
        "final_status": decision["final_status"],
    }
    (args.output_dir / f"{prefix}_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2, ensure_ascii=False), flush=True)
    print(f"E17 total_seconds={time.perf_counter() - total_started:.3f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=project_path("data/mousePMHC/mousePMHC_train.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=project_path("results/mousePMHC_phase5_e17_taskwise_ranking_mmoe_oof"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260704, 20260705, 20260706])
    parser.add_argument("--oof-folds", type=int, default=3); parser.add_argument("--oof-split-seed", type=int, default=20260711)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--task-batch-size", type=int, default=16); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--expert-dim", type=int, default=64); parser.add_argument("--condition-dim", type=int, default=16)
    parser.add_argument("--gate-hidden-dim", type=int, default=64); parser.add_argument("--n-experts", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2); parser.add_argument("--gate-entropy-weight", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ranking-lambda", type=float, default=0.25); parser.add_argument("--ranking-tau", type=float, default=1.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000); parser.add_argument("--bootstrap-seed", type=int, default=20260717)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
