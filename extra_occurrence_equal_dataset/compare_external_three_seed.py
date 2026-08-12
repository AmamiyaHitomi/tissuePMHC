"""Paired task-level comparison of three-seed models and external predictors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parent / "results"
TRAINED_EXPERIMENTS = (
    "e0_traditional",
    "e2_shared_heads",
    "e14a_auxiliary_dual_branch",
    "e29_multikernel_cnn",
)
EXTERNAL_MODELS = (
    "mhcflurry_2.2.1::presentation_score",
    "netmhcpan_4.1b::el_rank",
    "netmhcpan_4.1b::ba_rank",
)
METRICS = ("auroc", "auprc", "pair_accuracy")
TASK_KEYS = ["target_tissue", "mhc_restriction"]
BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 20260704


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPEATS, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    trained_parts = []
    for experiment in TRAINED_EXPERIMENTS:
        frame = pd.read_csv(ROOT / experiment / "per_task_metrics.csv")
        frame["target_tissue"] = frame["target_tissue"].fillna("NA")
        if set(frame["seed"].astype(int)) != {20260704, 20260705, 20260706}:
            raise ValueError(f"{experiment} does not contain the required three seeds")
        trained_parts.append(frame)
    trained = pd.concat(trained_parts, ignore_index=True)
    trained_mean = (
        trained.groupby(["model", *TASK_KEYS], sort=False)[list(METRICS)]
        .mean()
        .reset_index()
    )

    external = pd.read_csv(ROOT / "external_predictors" / "per_task_metrics.csv")
    external["target_tissue"] = external["target_tissue"].fillna("NA")
    external = external[external["model"].isin(EXTERNAL_MODELS)].copy()

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary_rows = []
    detail_parts = []
    for trained_model, local in trained_mean.groupby("model", sort=False):
        for external_model in EXTERNAL_MODELS:
            baseline = external[external["model"].eq(external_model)]
            paired = local.merge(
                baseline,
                on=TASK_KEYS,
                how="inner",
                suffixes=("_trained", "_external"),
                validate="one_to_one",
            )
            if len(paired) != 77:
                raise ValueError(
                    f"Expected 77 paired tasks for {trained_model} vs {external_model}; got {len(paired)}"
                )
            for metric in METRICS:
                difference = (
                    paired[f"{metric}_trained"] - paired[f"{metric}_external"]
                ).to_numpy(dtype=np.float64)
                ci_low, ci_high = bootstrap_ci(difference, rng)
                nonzero = difference[~np.isclose(difference, 0.0, atol=1e-12)]
                p_value = (
                    float(wilcoxon(nonzero, alternative="two-sided").pvalue)
                    if len(nonzero)
                    else 1.0
                )
                summary_rows.append(
                    {
                        "trained_model": trained_model,
                        "external_model": external_model,
                        "metric": metric,
                        "n_tasks": len(paired),
                        "trained_three_seed_task_mean": float(
                            paired[f"{metric}_trained"].mean()
                        ),
                        "external_task_mean": float(
                            paired[f"{metric}_external"].mean()
                        ),
                        "paired_mean_difference": float(difference.mean()),
                        "bootstrap_95_ci_low": ci_low,
                        "bootstrap_95_ci_high": ci_high,
                        "task_wins": int((difference > 1e-12).sum()),
                        "task_ties": int(np.isclose(difference, 0.0, atol=1e-12).sum()),
                        "task_losses": int((difference < -1e-12).sum()),
                        "wilcoxon_two_sided_p": p_value,
                        "bootstrap_repeats": BOOTSTRAP_REPEATS,
                        "bootstrap_seed": BOOTSTRAP_SEED,
                    }
                )
                detail = paired[TASK_KEYS].copy()
                detail.insert(0, "metric", metric)
                detail.insert(0, "external_model", external_model)
                detail.insert(0, "trained_model", trained_model)
                detail["trained_three_seed_task_mean"] = paired[f"{metric}_trained"]
                detail["external_task_value"] = paired[f"{metric}_external"]
                detail["difference"] = difference
                detail_parts.append(detail)

    summary = pd.DataFrame(summary_rows)
    p_values = summary["wilcoxon_two_sided_p"].to_numpy(dtype=np.float64)
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate(
        (len(p_values) - np.arange(len(p_values))) * p_values[order]
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    summary["wilcoxon_holm_p"] = adjusted
    summary = summary.sort_values(
        ["metric", "paired_mean_difference"], ascending=[True, False]
    )
    summary.to_csv(ROOT / "external_comparison_three_seed.csv", index=False)
    pd.concat(detail_parts, ignore_index=True).to_csv(
        ROOT / "external_comparison_three_seed_per_task.csv", index=False
    )
    print(
        summary[
            [
                "trained_model",
                "external_model",
                "metric",
                "paired_mean_difference",
                "bootstrap_95_ci_low",
                "bootstrap_95_ci_high",
                "task_wins",
                "task_ties",
                "task_losses",
                "wilcoxon_two_sided_p",
                "wilcoxon_holm_p",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
