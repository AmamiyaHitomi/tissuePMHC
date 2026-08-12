"""Aggregate the three fixed-test mouse runs and append frozen external models."""

from __future__ import annotations

import json

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
    all_summaries = []
    all_stability = []
    for experiment in EXPERIMENTS:
        predictions = []
        per_task = []
        summaries = []
        for seed in SEEDS:
            source = results / "seed_runs" / str(seed) / experiment
            paths = {
                "pred": source / "test_predictions.csv",
                "task": source / "per_task_metrics.csv",
                "summary": source / "summary_metrics.csv",
                "settings": source / "run_settings.json",
            }
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError("Missing seed results:\n" + "\n".join(missing))
            pred = pd.read_csv(paths["pred"])
            task = pd.read_csv(paths["task"])
            summary = pd.read_csv(paths["summary"])
            for name, frame in (("predictions", pred), ("per-task", task), ("summary", summary)):
                if set(frame["seed"].astype(int)) != {seed}:
                    raise ValueError(f"{experiment} {name} seed mismatch for {seed}")
            task_counts = task.groupby("model", sort=False).size()
            if not task_counts.eq(11).all() or not summary["n_tasks"].astype(int).eq(11).all():
                raise ValueError(
                    f"{experiment} seed {seed} does not contain 11 tasks per model"
                )
            predictions.append(pred)
            per_task.append(task)
            summaries.append(summary)

        output = results / experiment
        output.mkdir(parents=True, exist_ok=True)
        pred_frame = pd.concat(predictions, ignore_index=True)
        task_frame = pd.concat(per_task, ignore_index=True)
        summary_frame = pd.concat(summaries, ignore_index=True)
        pred_frame.to_csv(output / "test_predictions.csv", index=False)
        task_frame.to_csv(output / "per_task_metrics.csv", index=False)
        summary_frame.to_csv(output / "summary_metrics.csv", index=False)

        available = [metric for metric in METRICS if metric in summary_frame]
        grouped = summary_frame.groupby("model", sort=False)[available]
        stability = pd.concat(
            [
                grouped.mean().add_suffix("_mean"),
                grouped.std(ddof=1).add_suffix("_std"),
                grouped.min().add_suffix("_min"),
                grouped.max().add_suffix("_max"),
            ],
            axis=1,
        ).reset_index()
        stability.insert(1, "experiment", experiment)
        stability.insert(2, "n_seeds", len(SEEDS))
        stability.to_csv(output / "stability_metrics.csv", index=False)
        all_stability.append(stability)
        all_summaries.append(summary_frame.assign(experiment=experiment))
        (output / "multi_seed_settings.json").write_text(
            json.dumps(
                {
                    "experiment": experiment,
                    "seeds": list(SEEDS),
                    "split": "supplied standard train/test; no same-label or peptide-disjoint split",
                    "n_tasks": 11,
                    "aggregation": "fixed-test seed mean and sample standard deviation",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"aggregated {experiment}: rows={len(pred_frame)}, summaries={len(summary_frame)}")

    pd.concat(all_summaries, ignore_index=True).to_csv(
        results / "three_seed_summary_all.csv", index=False
    )
    stability_all = pd.concat(all_stability, ignore_index=True)
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
