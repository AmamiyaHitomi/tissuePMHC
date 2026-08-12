#!/usr/bin/env python3
"""Paired task-level statistics for standard versus peptide-disjoint OOF."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import EXPERIMENTS, attach_data, ensure_output, read_predictions, task_metrics, write_json


METRICS = ["auroc", "auprc", "accuracy", "mcc"]


def bh_fdr(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for rank_index in range(len(values) - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, values[original_index] * len(values) / rank)
        adjusted[original_index] = running
    return adjusted.clip(0, 1).tolist()


def hodges_lehmann_one_sample(values: np.ndarray) -> float:
    walsh = []
    for left in range(len(values)):
        walsh.extend((values[left] + values[left:]) / 2.0)
    return float(np.median(np.asarray(walsh)))


def paired_bootstrap(strict: np.ndarray, standard: np.ndarray, repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(strict), size=(repeats, len(strict)))
    strict_boot = strict[indices].mean(axis=1)
    delta_boot = (strict - standard)[indices].mean(axis=1)
    return {
        "strict_ci_low": float(np.quantile(strict_boot, 0.025)),
        "strict_ci_high": float(np.quantile(strict_boot, 0.975)),
        "delta_ci_low": float(np.quantile(delta_boot, 0.025)),
        "delta_ci_high": float(np.quantile(delta_boot, 0.975)),
    }


def component_cluster_bootstrap(frame: pd.DataFrame, repeats: int, seed: int) -> dict:
    """Resample peptide components within task and recompute macro AUROC/AUPRC."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    if repeats <= 0:
        return {"status": "not_run", "hint": "Pass --component-bootstrap N to run."}
    rng = np.random.default_rng(seed)
    task_groups = []
    for task_name, task in frame.groupby("task_name", sort=True):
        components = {key: value for key, value in task.groupby("component_id", sort=False)}
        task_groups.append((task_name, components))
    macro_auroc, macro_auprc = [], []
    for _ in range(repeats):
        task_auroc, task_auprc = [], []
        for _, components in task_groups:
            keys = list(components)
            selected = rng.choice(keys, size=len(keys), replace=True)
            sampled = pd.concat([components[key] for key in selected], ignore_index=True)
            if sampled["label"].nunique() < 2:
                continue
            task_auroc.append(roc_auc_score(sampled["label"], sampled["score"]))
            task_auprc.append(average_precision_score(sampled["label"], sampled["score"]))
        macro_auroc.append(np.mean(task_auroc))
        macro_auprc.append(np.mean(task_auprc))
    return {
        "status": "completed", "repeats": repeats,
        "auroc_ci95": [float(np.quantile(macro_auroc, 0.025)), float(np.quantile(macro_auroc, 0.975))],
        "auprc_ci95": [float(np.quantile(macro_auprc, 0.025)), float(np.quantile(macro_auprc, 0.975))],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--component-bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = ensure_output("05_statistics")
    comparisons, result_rows, component_results = [], [], {}
    for species, experiment in EXPERIMENTS.items():
        standard_frame = attach_data(experiment, read_predictions(experiment, "standard"))
        strict_frame = attach_data(experiment, read_predictions(experiment, "strict"))
        standard = task_metrics(standard_frame)
        strict = task_metrics(strict_frame)
        keys = ["target_tissue", "mhc_restriction"]
        comparison = standard.merge(strict, on=keys, suffixes=("_standard", "_strict"), validate="one_to_one")
        comparison.insert(0, "species", species)
        for metric in METRICS:
            comparison[f"delta_{metric}"] = comparison[f"{metric}_strict"] - comparison[f"{metric}_standard"]
            strict_values = comparison[f"{metric}_strict"].to_numpy(dtype=float)
            standard_values = comparison[f"{metric}_standard"].to_numpy(dtype=float)
            delta = strict_values - standard_values
            test = wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
            row = {
                "species": species, "metric": metric, "n_tasks": len(delta),
                "standard_mean": float(standard_values.mean()),
                "strict_mean": float(strict_values.mean()),
                "mean_delta": float(delta.mean()),
                "median_delta": float(np.median(delta)),
                "hodges_lehmann_delta": hodges_lehmann_one_sample(delta),
                "wins": int((delta > 0).sum()), "ties": int((delta == 0).sum()),
                "losses": int((delta < 0).sum()),
                "wilcoxon_statistic": float(test.statistic), "p_value": float(test.pvalue),
                **paired_bootstrap(strict_values, standard_values, args.bootstrap, args.seed),
            }
            result_rows.append(row)
        comparisons.append(comparison)
        strict_assignments = pd.read_csv(experiment.strict_assignments)[["pair_id", "component_id"]].drop_duplicates()
        cluster_frame = strict_frame.merge(strict_assignments, on="pair_id", how="left", validate="many_to_one")
        component_results[species] = component_cluster_bootstrap(
            cluster_frame, args.component_bootstrap, args.seed
        )
    results = pd.DataFrame(result_rows)
    results["fdr_bh"] = bh_fdr(results["p_value"].tolist())
    comparison_frame = pd.concat(comparisons, ignore_index=True)
    results.to_csv(output / "paired_statistical_tests.csv", index=False)
    comparison_frame.to_csv(output / "standard_vs_strict_task_metrics.csv", index=False)
    write_json(output / "component_cluster_bootstrap.json", component_results)
    write_json(output / "metadata.json", {
        "task_bootstrap_repeats": args.bootstrap,
        "component_bootstrap_repeats": args.component_bootstrap,
        "seed": args.seed,
        "fdr": "Benjamini-Hochberg across all species/metric tests in paired_statistical_tests.csv",
        "warning": "Task bootstrap does not include model retraining variance.",
    })
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

