#!/usr/bin/env python3
"""One-seed premium test of the five original human E0 traditional baselines."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import common


EXPERIMENT_NAME = "e0_traditional"


def main() -> None:
    # Parse before importing the frozen model modules or loading datasets so
    # ``--help`` remains a fast, side-effect-free smoke test.
    argparse.ArgumentParser(description=__doc__).parse_args()
    common.enable_original_modules()
    import run_tissuepmhc_baselines as e0
    import run_tissuepmhc_neural_baselines_v2 as base

    train, test, mappings, _ = common.load_premium_data(base)
    models = e0.get_models(common.SEED)
    groups = sorted(
        set(zip(train["target_tissue"], train["mhc_restriction"], strict=True))
        & set(zip(test["target_tissue"], test["mhc_restriction"], strict=True))
    )

    print(
        f"{EXPERIMENT_NAME}: seed={common.SEED}, train_rows={len(train)}, "
        f"test_rows={len(test)}, tasks={len(groups)}, models={len(models)}",
        flush=True,
    )
    started = time.perf_counter()
    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []

    for model_index, (model_name, (encoder, estimator)) in enumerate(models.items(), start=1):
        print(f"[{model_index}/{len(models)}] model={model_name}", flush=True)
        for group_index, (tissue, hla) in enumerate(groups, start=1):
            train_task = train[
                train["target_tissue"].eq(tissue) & train["mhc_restriction"].eq(hla)
            ]
            test_task = test[
                test["target_tissue"].eq(tissue) & test["mhc_restriction"].eq(hla)
            ]
            x_train = encoder(train_task["peptide_sequence"].astype(str).tolist())
            y_train = train_task["label"].to_numpy(dtype=np.int8)
            x_test = encoder(test_task["peptide_sequence"].astype(str).tolist())
            y_test = test_task["label"].to_numpy(dtype=np.int8)

            estimator.fit(x_train, y_train)
            scores = np.asarray(e0.predict_scores(estimator, x_test), dtype=np.float64)
            task_predictions = test_task[
                ["sample_id", "pair_id", "target_tissue", "mhc_restriction", "label"]
            ].copy()
            task_predictions.insert(0, "seed", common.SEED)
            task_predictions.insert(0, "model", model_name)
            task_predictions["score"] = scores
            prediction_parts.append(task_predictions)

            metrics = e0.evaluate(y_test, scores)
            metric_rows.append(
                {
                    "model": model_name,
                    "seed": common.SEED,
                    "target_tissue": tissue,
                    "mhc_restriction": hla,
                    "train_rows": int(len(train_task)),
                    "test_rows": int(len(test_task)),
                    **metrics,
                    "pair_accuracy": common.pair_accuracy(task_predictions),
                }
            )
            print(
                f"  task {group_index:02d}/{len(groups)} {tissue} {hla} "
                f"auroc={metrics['auroc']:.4f}",
                flush=True,
            )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    per_task = pd.DataFrame(metric_rows, columns=common.PER_TASK_COLUMNS)
    summary_rows: list[dict[str, object]] = []
    for model_name, model_tasks in per_task.groupby("model", sort=False):
        model_predictions = predictions[predictions["model"].eq(model_name)]
        global_metrics = e0.evaluate(
            model_predictions["label"].to_numpy(dtype=np.int8),
            model_predictions["score"].to_numpy(dtype=np.float64),
        )
        worst_n = min(10, len(model_tasks))
        summary_rows.append(
            {
                "model": model_name,
                "seed": common.SEED,
                "n_tasks": int(len(model_tasks)),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "mean_task_auroc": float(model_tasks["auroc"].mean()),
                "mean_task_auprc": float(model_tasks["auprc"].mean()),
                "mean_task_accuracy": float(model_tasks["accuracy"].mean()),
                "mean_task_mcc": float(model_tasks["mcc"].mean()),
                "mean_task_pair_accuracy": float(model_tasks["pair_accuracy"].mean()),
                "worst_10_mean_auroc": float(
                    model_tasks.nsmallest(worst_n, "auroc")["auroc"].mean()
                ),
                "global_auroc": float(global_metrics["auroc"]),
                "global_auprc": float(global_metrics["auprc"]),
                "global_accuracy": float(global_metrics["accuracy"]),
                "global_mcc": float(global_metrics["mcc"]),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        "mean_task_auroc", ascending=False
    )

    output_dir = common.RESULTS_ROOT / EXPERIMENT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    per_task.to_csv(output_dir / "per_task_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    (output_dir / "run_settings.json").write_text(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "seed": common.SEED,
                "train": str(common.TRAIN_PATH),
                "test": str(common.TEST_PATH),
                "source_model": "scripts/run_tissuepmhc_baselines.py",
                "n_tasks": len(mappings["tasks"]),
                "models": list(models),
                "training": "one independent estimator per tissue-HLA task",
                "elapsed_seconds": time.perf_counter() - started,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"wrote: {output_dir / 'test_predictions.csv'}", flush=True)
    print(f"wrote: {output_dir / 'per_task_metrics.csv'}", flush=True)
    print(f"wrote: {output_dir / 'summary_metrics.csv'}", flush=True)
    print(
        summary[
            ["model", "mean_task_auroc", "mean_task_auprc", "mean_task_pair_accuracy"]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
