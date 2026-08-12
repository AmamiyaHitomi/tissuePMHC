"""Paired task comparison of three-seed mouse models and frozen predictors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import common


ROOT = common.RESULTS_ROOT
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
    low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_adjusted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * values[order]
    )
    adjusted = np.empty_like(sorted_adjusted)
    adjusted[order] = np.minimum(sorted_adjusted, 1.0)
    return adjusted


def main() -> None:
    trained_parts = []
    for experiment in TRAINED_EXPERIMENTS:
        frame = pd.read_csv(ROOT / experiment / "per_task_metrics.csv")
        if set(frame["seed"].astype(int)) != {20260704, 20260705, 20260706}:
            raise ValueError(f"{experiment} does not contain the required seeds")
        trained_parts.append(frame)
    trained = pd.concat(trained_parts, ignore_index=True)
    trained_mean = (
        trained.groupby(["model", *TASK_KEYS], sort=False)[list(METRICS)]
        .mean().reset_index()
    )
    external = pd.read_csv(ROOT / "external_predictors" / "per_task_metrics.csv")
    external = external[external["model"].isin(EXTERNAL_MODELS)].copy()

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary_rows = []
    detail_parts = []
    for trained_model, local in trained_mean.groupby("model", sort=False):
        for external_model in EXTERNAL_MODELS:
            paired = local.merge(
                external[external["model"].eq(external_model)],
                on=TASK_KEYS,
                how="inner",
                suffixes=("_trained", "_external"),
                validate="one_to_one",
            )
            if len(paired) != 11:
                raise ValueError(
                    f"Expected 11 tasks for {trained_model} vs {external_model}; got {len(paired)}"
                )
            for metric in METRICS:
                difference = (
                    paired[f"{metric}_trained"] - paired[f"{metric}_external"]
                ).to_numpy(dtype=np.float64)
                ci_low, ci_high = bootstrap_ci(difference, rng)
                nonzero = difference[~np.isclose(difference, 0.0, atol=1e-12)]
                p_value = (
                    float(wilcoxon(nonzero, alternative="two-sided").pvalue)
                    if len(nonzero) else 1.0
                )
                summary_rows.append(
                    {
                        "trained_model": trained_model,
                        "external_model": external_model,
                        "metric": metric,
                        "n_tasks": 11,
                        "trained_three_seed_task_mean": float(
                            paired[f"{metric}_trained"].mean()
                        ),
                        "external_task_mean": float(paired[f"{metric}_external"].mean()),
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
    summary["wilcoxon_holm_p"] = holm_adjust(
        summary["wilcoxon_two_sided_p"].to_numpy(dtype=np.float64)
    )
    summary.sort_values(
        ["metric", "paired_mean_difference"], ascending=[True, False]
    ).to_csv(ROOT / "external_comparison_three_seed.csv", index=False)
    pd.concat(detail_parts, ignore_index=True).to_csv(
        ROOT / "external_comparison_three_seed_per_task.csv", index=False
    )
    print(summary[[
        "trained_model", "external_model", "metric", "paired_mean_difference",
        "bootstrap_95_ci_low", "bootstrap_95_ci_high", "task_wins", "task_ties",
        "task_losses", "wilcoxon_two_sided_p", "wilcoxon_holm_p"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
