"""Aggregate fixed-test results for the standard three human seeds."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import common


SEEDS = (20260704, 20260705, 20260706)
EXPERIMENTS = (
    "e0_traditional",
    "e2_shared_heads",
    "e14a_auxiliary_dual_branch",
    "e29_multikernel_cnn",
)
METRICS = (
    "mean_task_auroc",
    "mean_task_auprc",
    "mean_task_accuracy",
    "mean_task_mcc",
    "mean_task_pair_accuracy",
    "worst_10_mean_auroc",
    "global_auroc",
    "global_auprc",
    "global_accuracy",
    "global_mcc",
)


def main() -> None:
    results = common.EXPERIMENT_ROOT / "results"
    combined_summaries: list[pd.DataFrame] = []
    stability_parts: list[pd.DataFrame] = []
    for experiment in EXPERIMENTS:
        predictions: list[pd.DataFrame] = []
        per_task: list[pd.DataFrame] = []
        summaries: list[pd.DataFrame] = []
        for seed in SEEDS:
            source = results / "seed_runs" / str(seed) / experiment
            required = [
                source / "test_predictions.csv",
                source / "per_task_metrics.csv",
                source / "summary_metrics.csv",
                source / "run_settings.json",
            ]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError("Missing seed result files:\n" + "\n".join(missing))
            pred = pd.read_csv(required[0])
            task = pd.read_csv(required[1])
            summary = pd.read_csv(required[2])
            if set(pred["seed"].astype(int)) != {seed}:
                raise ValueError(f"Prediction seed mismatch in {source}")
            if set(task["seed"].astype(int)) != {seed}:
                raise ValueError(f"Metric seed mismatch in {source}")
            if set(summary["seed"].astype(int)) != {seed}:
                raise ValueError(f"Summary seed mismatch in {source}")
            predictions.append(pred)
            per_task.append(task)
            summaries.append(summary)

        output = results / experiment
        output.mkdir(parents=True, exist_ok=True)
        prediction_frame = pd.concat(predictions, ignore_index=True)
        task_frame = pd.concat(per_task, ignore_index=True)
        summary_frame = pd.concat(summaries, ignore_index=True)
        prediction_frame.to_csv(output / "test_predictions.csv", index=False)
        task_frame.to_csv(output / "per_task_metrics.csv", index=False)
        summary_frame.to_csv(output / "summary_metrics.csv", index=False)

        available_metrics = [column for column in METRICS if column in summary_frame]
        grouped = summary_frame.groupby("model", sort=False)[available_metrics]
        mean = grouped.mean().add_suffix("_mean")
        std = grouped.std(ddof=1).add_suffix("_std")
        minimum = grouped.min().add_suffix("_min")
        maximum = grouped.max().add_suffix("_max")
        stability = pd.concat([mean, std, minimum, maximum], axis=1).reset_index()
        stability.insert(1, "experiment", experiment)
        stability.insert(2, "n_seeds", len(SEEDS))
        stability.to_csv(output / "stability_metrics.csv", index=False)
        stability_parts.append(stability)
        combined_summaries.append(summary_frame.assign(experiment=experiment))

        (output / "multi_seed_settings.json").write_text(
            json.dumps(
                {
                    "experiment": experiment,
                    "seeds": list(SEEDS),
                    "seed_results": [
                        str(results / "seed_runs" / str(seed) / experiment)
                        for seed in SEEDS
                    ],
                    "aggregation": "concatenate predictions/per-task/summary; stability uses sample std (ddof=1)",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"aggregated {experiment}: predictions={len(prediction_frame)}, summaries={len(summary_frame)}")

    pd.concat(combined_summaries, ignore_index=True).to_csv(
        results / "three_seed_summary_all.csv", index=False
    )
    stability_all = pd.concat(stability_parts, ignore_index=True)
    stability_all.to_csv(results / "three_seed_stability_all.csv", index=False)

    comparison = stability_all[
        [
            "model",
            "n_seeds",
            "mean_task_auroc_mean",
            "mean_task_auroc_std",
            "mean_task_auprc_mean",
            "mean_task_auprc_std",
            "mean_task_pair_accuracy_mean",
            "mean_task_pair_accuracy_std",
        ]
    ].copy()
    comparison.insert(0, "source", "trained_three_seed")
    external_path = results / "external_predictors" / "summary_metrics.csv"
    if external_path.is_file():
        external = pd.read_csv(external_path)
        external = external[external["is_primary_mode"].astype(bool)].copy()
        external_rows = pd.DataFrame(
            {
                "source": "frozen_external_predictor",
                "model": external["model"],
                "n_seeds": 0,
                "mean_task_auroc_mean": external["mean_task_auroc"],
                "mean_task_auroc_std": float("nan"),
                "mean_task_auprc_mean": external["mean_task_auprc"],
                "mean_task_auprc_std": float("nan"),
                "mean_task_pair_accuracy_mean": external["mean_task_pair_accuracy"],
                "mean_task_pair_accuracy_std": float("nan"),
            }
        )
        comparison = pd.concat([comparison, external_rows], ignore_index=True)
    comparison.sort_values("mean_task_auroc_mean", ascending=False).to_csv(
        results / "three_seed_model_comparison.csv", index=False
    )


if __name__ == "__main__":
    main()
