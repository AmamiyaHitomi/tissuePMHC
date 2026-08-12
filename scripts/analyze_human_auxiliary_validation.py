#!/usr/bin/env python3
"""Audit and statistically summarize the Human auxiliary validation run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METRICS = ("auroc", "auprc", "pair_accuracy")


def stable_seed(*parts: object) -> int:
    payload = "||".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def bootstrap_ci(values: np.ndarray, seed: int, replicates: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(frame: pd.DataFrame, family_columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    output["holm_p"] = np.nan
    for _, family in output.groupby(family_columns, sort=False):
        ordered = family.sort_values("wilcoxon_p")
        adjusted: list[float] = []
        running = 0.0
        count = len(ordered)
        for rank, value in enumerate(ordered["wilcoxon_p"].to_numpy(dtype=float)):
            running = max(running, min(1.0, (count - rank) * value))
            adjusted.append(running)
        output.loc[ordered.index, "holm_p"] = adjusted
    return output


def task_comparisons(per_task: pd.DataFrame) -> pd.DataFrame:
    averaged = (
        per_task.groupby(["auxiliary_weight", "target_tissue", "mhc_restriction"], as_index=False)
        [list(METRICS)].mean()
    )
    reference = averaged[averaged["auxiliary_weight"] == 0].drop(columns="auxiliary_weight")
    rows: list[dict[str, object]] = []
    for weight in sorted(set(averaged["auxiliary_weight"]) - {0.0}):
        candidate = averaged[averaged["auxiliary_weight"] == weight].drop(columns="auxiliary_weight")
        merged = reference.merge(
            candidate, on=["target_tissue", "mhc_restriction"], validate="one_to_one",
            suffixes=("_zero", "_candidate"),
        )
        for metric in METRICS:
            delta = (merged[f"{metric}_candidate"] - merged[f"{metric}_zero"]).to_numpy(dtype=float)
            nonzero = delta[delta != 0]
            p_value = float(wilcoxon(nonzero, alternative="two-sided").pvalue) if len(nonzero) else 1.0
            low, high = bootstrap_ci(delta, stable_seed("task", weight, metric))
            rows.append({
                "candidate_weight": weight,
                "reference_weight": 0.0,
                "metric": metric,
                "n_tasks": len(delta),
                "mean_delta": float(delta.mean()),
                "median_delta": float(np.median(delta)),
                "ci95_low": low,
                "ci95_high": high,
                "wins": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
                "losses": int((delta < 0).sum()),
                "wilcoxon_p": p_value,
            })
    return holm_adjust(pd.DataFrame(rows), ["metric"])


def seed_comparisons(per_seed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = (
        "mean_task_auroc", "mean_task_auprc", "mean_task_pair_accuracy", "worst_k_mean_auroc"
    )
    reference = per_seed[per_seed["auxiliary_weight"] == 0].set_index("seed")
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for weight in sorted(set(per_seed["auxiliary_weight"]) - {0.0}):
        candidate = per_seed[per_seed["auxiliary_weight"] == weight].set_index("seed")
        if set(candidate.index) != set(reference.index):
            raise AssertionError(f"weight={weight}: incomplete paired seed panel")
        for seed in sorted(reference.index):
            row: dict[str, object] = {"candidate_weight": weight, "reference_weight": 0.0, "seed": seed}
            for metric in metrics:
                row[f"{metric}_delta"] = float(candidate.loc[seed, metric] - reference.loc[seed, metric])
            detail_rows.append(row)
        for metric in metrics:
            delta = np.array([
                candidate.loc[seed, metric] - reference.loc[seed, metric]
                for seed in sorted(reference.index)
            ], dtype=float)
            low, high = bootstrap_ci(delta, stable_seed("seed", weight, metric), 100000)
            p_value = float(wilcoxon(delta, alternative="two-sided", method="exact").pvalue)
            summary_rows.append({
                "candidate_weight": weight,
                "reference_weight": 0.0,
                "metric": metric,
                "n_seeds": len(delta),
                "mean_delta": float(delta.mean()),
                "sd_delta": float(delta.std(ddof=1)),
                "ci95_bootstrap_low": low,
                "ci95_bootstrap_high": high,
                "positive_seeds": int((delta > 0).sum()),
                "negative_seeds": int((delta < 0).sum()),
                "wilcoxon_exact_p": p_value,
            })
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def gradient_temporal_summary(gradients: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for weight, group in gradients.groupby("config", sort=True):
        numeric_weight = float(str(weight).removeprefix("aux_weight_"))
        for window_name, window in (
            ("all", group), ("early_epochs_1_5", group[group["epoch"] <= 5]),
            ("late_epochs_16_20", group[group["epoch"] >= 16]),
        ):
            row: dict[str, object] = {
                "auxiliary_weight": numeric_weight,
                "window": window_name,
                "n": len(window),
            }
            for auxiliary in ("tissue", "mhc"):
                cosine = window[f"primary_{auxiliary}_cosine"].to_numpy(dtype=float)
                primary_norm = window["primary_gradient_norm"].to_numpy(dtype=float)
                auxiliary_norm = window[f"{auxiliary}_gradient_norm"].to_numpy(dtype=float)
                row[f"primary_{auxiliary}_cosine_mean"] = float(cosine.mean())
                row[f"primary_{auxiliary}_negative_fraction"] = float((cosine < 0).mean())
                row[f"weighted_{auxiliary}_to_primary_norm_ratio_mean"] = float(
                    np.mean(numeric_weight * auxiliary_norm / primary_norm)
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["auxiliary_weight", "window"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = args.input.resolve()
    contract = json.loads((root / "run_contract.json").read_text(encoding="utf-8"))
    per_seed = pd.read_csv(root / "validation_per_seed_metrics.csv", keep_default_na=False)
    per_task = pd.read_csv(root / "validation_per_task_metrics.csv", keep_default_na=False)
    gradients = pd.read_csv(root / "gradient_epoch_diagnostics.csv")
    weights = [float(value) for value in sorted(per_seed["auxiliary_weight"].unique())]
    seeds = [int(value) for value in sorted(per_seed["seed"].unique())]
    expected_task_rows = len(weights) * len(seeds) * 77
    audit = {
        "contract_status": contract["status"],
        "weights": weights,
        "seeds": seeds,
        "per_seed_rows_observed": len(per_seed),
        "per_seed_rows_expected": len(weights) * len(seeds),
        "per_task_rows_observed": len(per_task),
        "per_task_rows_expected": expected_task_rows,
        "gradient_rows_observed": len(gradients),
        "gradient_rows_expected": (len(weights) - 1) * len(seeds) * contract["epochs"],
        "stderr_bytes": (root / "console_error.log").stat().st_size,
        "elapsed_seconds": contract["elapsed_seconds"],
    }
    if audit["contract_status"] != "completed" or any([
        audit["per_seed_rows_observed"] != audit["per_seed_rows_expected"],
        audit["per_task_rows_observed"] != audit["per_task_rows_expected"],
        audit["gradient_rows_observed"] != audit["gradient_rows_expected"],
        audit["stderr_bytes"] != 0,
    ]):
        raise AssertionError(f"Run completeness audit failed: {audit}")
    (root / "analysis_completeness_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    task_stats = task_comparisons(per_task)
    seed_detail, seed_stats = seed_comparisons(per_seed)
    gradient_stats = gradient_temporal_summary(gradients)
    task_stats.to_csv(root / "analysis_task_paired_comparisons.csv", index=False)
    seed_detail.to_csv(root / "analysis_seed_paired_deltas.csv", index=False)
    seed_stats.to_csv(root / "analysis_seed_paired_summary.csv", index=False)
    gradient_stats.to_csv(root / "analysis_gradient_temporal_summary.csv", index=False)

    selected = json.loads((root / "selected_weight.json").read_text(encoding="utf-8"))
    selected_weight = float(selected["selected_tissue_weight"])
    weight_summary = pd.read_csv(root / "validation_weight_summary.csv")
    selected_row = weight_summary[weight_summary["auxiliary_weight"] == selected_weight].iloc[0]
    zero_row = weight_summary[weight_summary["auxiliary_weight"] == 0].iloc[0]
    original_row = weight_summary[weight_summary["auxiliary_weight"] == 0.3].iloc[0]
    selected_seed = seed_stats[
        (seed_stats["candidate_weight"] == selected_weight) &
        (seed_stats["metric"] == "mean_task_auroc")
    ].iloc[0]
    selected_task = task_stats[
        (task_stats["candidate_weight"] == selected_weight) & (task_stats["metric"] == "auroc")
    ].iloc[0]
    original_seed = seed_stats[
        (seed_stats["candidate_weight"] == 0.3) &
        (seed_stats["metric"] == "mean_task_auroc")
    ].iloc[0]
    original_task = task_stats[
        (task_stats["candidate_weight"] == 0.3) & (task_stats["metric"] == "auroc")
    ].iloc[0]
    report = f"""# Human auxiliary-supervision validation analysis

## Run audit

- Status: completed; stderr is empty.
- Seeds: {', '.join(map(str, seeds))}.
- Grid: {', '.join(map(str, weights))}.
- Complete records: {len(per_seed)} seed/weight summaries, {len(per_task)} task records, {len(gradients)} epoch-gradient diagnostics.
- Total training time: {contract['elapsed_seconds']:.3f} seconds.
- The previously inspected fixed test split was not read.

## Primary result

The preregistered selection rule chose tied tissue/HLA weight **{selected_weight:g}**.
Its five-seed validation task-macro AUROC was {selected_row['mean_task_auroc_mean']:.6f} ± {selected_row['mean_task_auroc_sd']:.6f}, versus {zero_row['mean_task_auroc_mean']:.6f} ± {zero_row['mean_task_auroc_sd']:.6f} without auxiliary supervision.
The paired seed mean delta was {selected_seed['mean_delta']:+.6f}; all {int(selected_seed['positive_seeds'])}/{int(selected_seed['n_seeds'])} seeds were positive. The five-seed bootstrap interval was [{selected_seed['ci95_bootstrap_low']:+.6f}, {selected_seed['ci95_bootstrap_high']:+.6f}], while the exact two-sided seed-level Wilcoxon p-value was {selected_seed['wilcoxon_exact_p']:.4f} (minimum attainable with five consistently signed nonzero pairs is 0.0625).

Across 77 tasks after averaging seeds, the {selected_weight:g}-minus-zero AUROC delta was {selected_task['mean_delta']:+.6f} [{selected_task['ci95_low']:+.6f}, {selected_task['ci95_high']:+.6f}], with W/T/L={int(selected_task['wins'])}/{int(selected_task['ties'])}/{int(selected_task['losses'])}, raw Wilcoxon p={selected_task['wilcoxon_p']:.6g}, Holm p={selected_task['holm_p']:.6g}.

## Original 0.30 weight versus no auxiliary

The original 0.30 setting reached AUROC {original_row['mean_task_auroc_mean']:.6f} ± {original_row['mean_task_auroc_sd']:.6f}; its paired seed delta versus zero was only {original_seed['mean_delta']:+.6f}, with {int(original_seed['positive_seeds'])}/{int(original_seed['n_seeds'])} positive seeds and exact p={original_seed['wilcoxon_exact_p']:.4f}. Task-level delta was {original_task['mean_delta']:+.6f} [{original_task['ci95_low']:+.6f}, {original_task['ci95_high']:+.6f}], W/T/L={int(original_task['wins'])}/{int(original_task['ties'])}/{int(original_task['losses'])}, Holm p={original_task['holm_p']:.6g}.

Thus, the earlier apparent no-auxiliary advantage is not reproduced under paired stochastic training on the internal validation split. Auxiliary supervision is useful at moderate weight, but 0.30 is over-weighted and yields an unstable, small gain.

## Gradient diagnosis

Mean primary–auxiliary cosine similarities are slightly positive, not persistently negative. However, negative cosine epochs remain common, especially at larger weights. This supports intermittent gradient conflict rather than universal negative transfer. See `analysis_gradient_temporal_summary.csv` for early/late windows and weighted gradient-norm ratios.

## Decision

Use 0.20/0.20 as the validation-selected Human setting under the preregistered rule. Do not claim a confirmed test improvement: the old fixed test was already inspected. A publication-level confirmatory statement requires a genuinely new untouched split or a separately preregistered nested-CV run. Do not use the old test to choose between 0, 0.20, and 0.30.
"""
    (root / "ANALYSIS_REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
