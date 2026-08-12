#!/usr/bin/env python3
"""Create issue-9 architecture tables and nominal task-paired statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

try:
    from .common import ROOT, TASK_KEYS, atomic_json
except ImportError:
    from common import ROOT, TASK_KEYS, atomic_json


METRICS = ["auroc", "auprc", "pair_acc"]
MAIN_MODELS = {
    "human": "human_tissuepmhc_net",
    "mouse": "mouse_factorized_mmoe",
}


def hodges_lehmann_one_sample(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    walsh = (values[:, None] + values[None, :]) / 2.0
    return float(np.median(walsh[np.triu_indices(len(values))]))


def bootstrap_mean_interval(
    differences: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, 1000):
        stop = min(start + 1000, repetitions)
        draws = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paired_rows(
    per_task: pd.DataFrame,
    species: str,
    main_model: str,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    species_rows = per_task[per_task["species"] == species].copy()
    models = sorted(set(species_rows["model"]) - {main_model})
    stats: list[dict[str, Any]] = []
    deltas: list[pd.DataFrame] = []
    for baseline in models:
        for metric in METRICS:
            left = species_rows[species_rows["model"] == main_model][TASK_KEYS + [metric]]
            right = species_rows[species_rows["model"] == baseline][TASK_KEYS + [metric]]
            merged = left.merge(
                right,
                on=TASK_KEYS,
                how="inner",
                validate="one_to_one",
                suffixes=("_main", "_baseline"),
            ).dropna()
            if len(merged) != len(left) or len(merged) != len(right):
                raise ValueError(
                    f"Task coverage mismatch: {species} {main_model} vs {baseline}, {metric}"
                )
            difference = (
                merged[f"{metric}_main"] - merged[f"{metric}_baseline"]
            ).to_numpy(float)
            tolerance = 1e-12
            wins = int((difference > tolerance).sum())
            ties = int((np.abs(difference) <= tolerance).sum())
            losses = int((difference < -tolerance).sum())
            if np.allclose(difference, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                test = wilcoxon(
                    difference,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto",
                )
                statistic, p_value = float(test.statistic), float(test.pvalue)
            ci_low, ci_high = bootstrap_mean_interval(
                difference,
                repetitions,
                seed + len(stats),
            )
            stats.append(
                {
                    "species": species,
                    "main_model": main_model,
                    "baseline_model": baseline,
                    "metric": metric,
                    "n_tasks": len(merged),
                    "main_mean": float(merged[f"{metric}_main"].mean()),
                    "baseline_mean": float(merged[f"{metric}_baseline"].mean()),
                    "mean_difference": float(np.mean(difference)),
                    "median_difference": float(np.median(difference)),
                    "hodges_lehmann_difference": hodges_lehmann_one_sample(difference),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "task_bootstrap_mean_ci_low": ci_low,
                    "task_bootstrap_mean_ci_high": ci_high,
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_nominal": p_value,
                }
            )
            detail = merged[TASK_KEYS].copy()
            detail.insert(0, "species", species)
            detail.insert(1, "main_model", main_model)
            detail.insert(2, "baseline_model", baseline)
            detail.insert(3, "metric", metric)
            detail["main_value"] = merged[f"{metric}_main"]
            detail["baseline_value"] = merged[f"{metric}_baseline"]
            detail["difference"] = difference
            deltas.append(detail)
    return pd.DataFrame(stats), pd.concat(deltas, ignore_index=True)


def bh_adjust(p_values: pd.Series) -> np.ndarray:
    values = p_values.to_numpy(float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def add_fdr(stats: pd.DataFrame) -> pd.DataFrame:
    result = stats.copy()
    result["wilcoxon_p_bh_fdr"] = np.nan
    # Each species-metric block is one pre-specified model-comparison family.
    for _, indices in result.groupby(["species", "metric"]).groups.items():
        result.loc[indices, "wilcoxon_p_bh_fdr"] = bh_adjust(
            result.loc[indices, "wilcoxon_p_nominal"]
        )
    return result


def architecture_table(per_task: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (species, model), group in per_task.groupby(["species", "model"], sort=True):
        row: dict[str, Any] = {
            "species": species,
            "model": model,
            "n_tasks": len(group),
        }
        for metric in METRICS:
            row[f"mean_task_{metric}"] = float(group[metric].mean())
            row[f"median_task_{metric}"] = float(group[metric].median())
        row["worst_task_auroc"] = float(group["auroc"].min())
        row["worst10_task_auroc"] = float(
            group.nsmallest(min(10, len(group)), "auroc")["auroc"].mean()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def seed_stability(member_metrics: pd.DataFrame) -> pd.DataFrame:
    seed_summary = (
        member_metrics.groupby(["species", "model", "seed"], as_index=False)
        .agg(
            mean_task_auroc=("auroc", "mean"),
            mean_task_auprc=("auprc", "mean"),
            mean_task_pair_acc=("pair_acc", "mean"),
        )
    )
    return (
        seed_summary.groupby(["species", "model"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            single_seed_20260704_auroc=(
                "mean_task_auroc",
                lambda values: float(
                    seed_summary.loc[values.index]
                    .set_index("seed")
                    .loc[20260704, "mean_task_auroc"]
                )
                if 20260704 in set(seed_summary.loc[values.index, "seed"])
                else np.nan,
            ),
            seed_mean_auroc=("mean_task_auroc", "mean"),
            seed_sd_auroc=("mean_task_auroc", "std"),
            seed_mean_auprc=("mean_task_auprc", "mean"),
            seed_sd_auprc=("mean_task_auprc", "std"),
            seed_mean_pair_acc=("mean_task_pair_acc", "mean"),
            seed_sd_pair_acc=("mean_task_pair_acc", "std"),
        )
    )


def worst_groups(per_task: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for grouping in ["target_tissue", "mhc_restriction"]:
        table = (
            per_task.groupby(["species", "model", grouping], as_index=False)
            .agg(
                n_tasks=("auroc", "size"),
                mean_task_auroc=("auroc", "mean"),
                mean_task_auprc=("auprc", "mean"),
                mean_task_pair_acc=("pair_acc", "mean"),
            )
            .rename(columns={grouping: "group"})
        )
        table.insert(2, "group_type", grouping)
        rows.append(table)
    result = pd.concat(rows, ignore_index=True)
    result["is_worst_auroc_group"] = (
        result.groupby(["species", "model", "group_type"])["mean_task_auroc"]
        .transform("min")
        .eq(result["mean_task_auroc"])
    )
    return result


def run(args: argparse.Namespace) -> None:
    inputs = {
        "human_ensemble": args.human_dir / "ensemble_per_task_metrics.csv",
        "human_member": args.human_dir / "member_per_task_metrics.csv",
        "mouse_ensemble": args.mouse_dir / "ensemble_per_task_metrics.csv",
        "mouse_member": args.mouse_dir / "member_per_task_metrics.csv",
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Run human and mouse issue9 experiments first. Missing: {missing}")
    ensemble = pd.concat(
        [pd.read_csv(inputs["human_ensemble"]), pd.read_csv(inputs["mouse_ensemble"])],
        ignore_index=True,
    )
    members = pd.concat(
        [pd.read_csv(inputs["human_member"]), pd.read_csv(inputs["mouse_member"])],
        ignore_index=True,
    )
    stats_parts: list[pd.DataFrame] = []
    delta_parts: list[pd.DataFrame] = []
    for index, (species, main_model) in enumerate(MAIN_MODELS.items()):
        stats, deltas = paired_rows(
            ensemble,
            species,
            main_model,
            args.bootstrap_repetitions,
            args.bootstrap_seed + index * 10000,
        )
        stats_parts.append(stats)
        delta_parts.append(deltas)
    stats = add_fdr(pd.concat(stats_parts, ignore_index=True))
    deltas = pd.concat(delta_parts, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    architecture_table(ensemble).to_csv(
        args.output_dir / "strict_architecture_comparison.csv", index=False
    )
    stats.to_csv(args.output_dir / "paired_statistics.csv", index=False)
    deltas.to_csv(args.output_dir / "per_task_differences.csv", index=False)
    seed_stability(members).to_csv(args.output_dir / "seed_stability.csv", index=False)
    worst_groups(ensemble).to_csv(args.output_dir / "worst_group_metrics.csv", index=False)
    decision = {}
    for species, main in MAIN_MODELS.items():
        primary = stats[
            (stats["species"] == species)
            & (stats["metric"] == "auroc")
        ]
        decision[species] = {
            "main_model": main,
            "all_mean_differences_positive": bool((primary["mean_difference"] > 0).all()),
            "all_bh_fdr_below_0_05": bool((primary["wilcoxon_p_bh_fdr"] < 0.05).all()),
            "interpretation": (
                "Architecture superiority is supported only if effects are positive, "
                "uncertainty intervals and task-level patterns are coherent, and the "
                "pre-specified comparisons survive multiplicity control."
            ),
        }
    atomic_json(
        args.output_dir / "analysis_metadata.json",
        {
            "inputs": {key: str(value.resolve()) for key, value in inputs.items()},
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_unit": "tissue-MHC task",
            "wilcoxon_unit": "tissue-MHC task",
            "inference_scope": "nominal task-level inference; tasks are not independent external cohorts",
            "bh_fdr_family": "within each species and metric across pre-specified baseline-vs-main comparisons",
            "decision_summary": decision,
        },
    )
    print(f"issue9 analysis complete: {args.output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--human-dir", type=Path, default=ROOT / "results" / "issue9_human_strict"
    )
    parser.add_argument(
        "--mouse-dir", type=Path, default=ROOT / "results" / "issue9_mouse_strict"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "issue9_analysis"
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
