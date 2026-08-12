from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .common import (
        DEFAULT_RESULTS,
        SPECS,
        atomic_json,
        bh_adjust,
        paired_comparison,
        per_task_metrics,
        read_benchmark,
        select_main_predictions,
        sha256,
    )
except ImportError:
    from common import (
        DEFAULT_RESULTS,
        SPECS,
        atomic_json,
        bh_adjust,
        paired_comparison,
        per_task_metrics,
        read_benchmark,
        select_main_predictions,
        sha256,
    )


def protocol_info(species: str) -> dict[str, tuple[pd.DataFrame, Path, str]]:
    spec = SPECS[species]
    train = read_benchmark(spec.train, species, "train")
    test = read_benchmark(spec.test, species, "test")
    return {
        "standard_oof": (train, spec.standard_predictions, spec.standard_candidate),
        "strict_oof": (train, spec.strict_predictions, spec.strict_candidate),
        "fixed_test": (test, spec.fixed_predictions, spec.fixed_candidate),
    }


def analyze(
    predictions_paths: list[Path], output_dir: Path, bootstrap_iterations: int
) -> None:
    predictions = pd.concat(
        [pd.read_csv(path) for path in predictions_paths], ignore_index=True
    )
    required = {"species", "protocol", "sample_id", "score"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"MHC-only predictions miss columns: {missing}")
    metric_frames: list[pd.DataFrame] = []
    paired_rows: list[dict[str, Any]] = []
    delta_frames: list[pd.DataFrame] = []
    for (species, protocol), candidate in predictions.groupby(
        ["species", "protocol"], sort=True
    ):
        data, main_path, main_candidate = protocol_info(species)[protocol]
        if len(candidate) != len(data):
            raise ValueError(f"{species}/{protocol} prediction coverage is incomplete.")
        joined = data.merge(
            candidate[["sample_id", "score"]],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        main = select_main_predictions(
            main_path, main_candidate, data, f"{species}/{protocol}"
        )
        mhc_metrics, _ = per_task_metrics(joined, "score", f"{species}_mhc_only")
        main_metrics, _ = per_task_metrics(main, "main_score", main_candidate)
        for frame, model in ((mhc_metrics, f"{species}_mhc_only"), (main_metrics, main_candidate)):
            frame.insert(0, "species", species)
            frame.insert(1, "protocol", protocol)
            frame["evaluated_model"] = model
            metric_frames.append(frame)
        for metric in ("auroc", "auprc", "pair_acc"):
            summary, deltas = paired_comparison(
                mhc_metrics,
                main_metrics,
                metric,
                bootstrap_iterations=bootstrap_iterations,
            )
            summary.update(
                {
                    "species": species,
                    "protocol": protocol,
                    "baseline_model": f"{species}_mhc_only",
                    "main_model": main_candidate,
                }
            )
            paired_rows.append(summary)
            deltas.insert(0, "species", species)
            deltas.insert(1, "protocol", protocol)
            deltas.insert(2, "metric", metric)
            delta_frames.append(deltas)
    metrics = pd.concat(metric_frames, ignore_index=True)
    paired = pd.DataFrame(paired_rows)
    paired["bh_family"] = (
        paired["species"] + "||" + paired["protocol"] + "||" + paired["metric"]
    )
    paired["wilcoxon_qvalue"] = paired.groupby("bh_family", group_keys=False)[
        "wilcoxon_pvalue"
    ].apply(bh_adjust)
    deltas = pd.concat(delta_frames, ignore_index=True)
    summary = (
        metrics.groupby(
            ["species", "protocol", "evaluated_model"], as_index=False
        )
        .agg(
            n_tasks=("task_name", "nunique"),
            mean_task_auroc=("auroc", "mean"),
            mean_task_auprc=("auprc", "mean"),
            mean_task_pair_acc=("pair_acc", "mean"),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "per_task_metrics.csv", index=False)
    paired.to_csv(output_dir / "paired_statistics.csv", index=False)
    deltas.to_csv(output_dir / "per_task_differences.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    atomic_json(
        output_dir / "metadata.json",
        {
            "purpose": "Capacity-matched MHC-only versus frozen full-model comparison",
            "prediction_files": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in predictions_paths
            ],
            "bootstrap_iterations": bootstrap_iterations,
            "difference_direction": "main_model_minus_mhc_only",
            "inference_scope": "Nominal task-level inference.",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_RESULTS / "mhc_only_analysis"
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(args.predictions, args.output_dir, args.bootstrap_iterations)
