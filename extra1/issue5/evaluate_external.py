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
        load_score_caches,
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
        load_score_caches,
        paired_comparison,
        per_task_metrics,
        read_benchmark,
        select_main_predictions,
        sha256,
    )


def protocols(species: str) -> list[tuple[str, pd.DataFrame, Path, str]]:
    spec = SPECS[species]
    train = read_benchmark(spec.train, species, "train")
    test = read_benchmark(spec.test, species, "test")
    return [
        (
            "standard_fixed_test",
            test,
            spec.fixed_predictions,
            spec.fixed_candidate,
        ),
        (
            "matched_standard_oof",
            train,
            spec.standard_predictions,
            spec.standard_candidate,
        ),
        (
            "peptide_disjoint_oof",
            train,
            spec.strict_predictions,
            spec.strict_candidate,
        ),
    ]


def evaluate(
    cache_paths: list[Path],
    output_dir: Path,
    bootstrap_iterations: int,
) -> None:
    cache = load_score_caches(cache_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    row_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    paired_rows: list[dict[str, Any]] = []
    delta_frames: list[pd.DataFrame] = []

    for species in SPECS:
        species_cache = cache[cache["species"] == species]
        if species_cache.empty:
            continue
        for protocol, data, main_path, main_candidate in protocols(species):
            main = select_main_predictions(
                main_path, main_candidate, data, f"{species}/{protocol}"
            )
            for (predictor, scoring_mode), scores in species_cache.groupby(
                ["predictor", "scoring_mode"], sort=True
            ):
                model = f"{predictor}::{scoring_mode}"
                joined = main.merge(
                    scores[["query_id", "score", "is_supported", "missing_reason"]],
                    on="query_id",
                    how="left",
                    validate="many_to_one",
                )
                joined["is_supported"] = joined["is_supported"].fillna(False)
                joined["missing_reason"] = joined["missing_reason"].fillna(
                    "query_absent_from_cache"
                )
                joined["species"] = species
                joined["protocol"] = protocol
                joined["model"] = model
                joined = joined.rename(columns={"score": "external_score"})
                row_frames.append(
                    joined[
                        [
                            "species",
                            "protocol",
                            "model",
                            *[
                                column
                                for column in joined.columns
                                if column
                                in {
                                    "sample_id",
                                    "pair_id",
                                    "target_tissue",
                                    "mhc_restriction",
                                    "peptide_sequence",
                                    "label",
                                    "task_name",
                                    "query_id",
                                    "main_score",
                                    "external_score",
                                    "is_supported",
                                    "missing_reason",
                                }
                            ],
                        ]
                    ]
                )
                external_metrics, coverage = per_task_metrics(
                    joined, "external_score", model
                )
                main_covered = joined[joined["external_score"].notna()].copy()
                # per_task_metrics itself restricts to complete pairs.
                main_metrics, _ = per_task_metrics(
                    main_covered, "main_score", "TissuePMHC_main"
                )
                for frame, evaluated_model in (
                    (external_metrics, model),
                    (main_metrics, "TissuePMHC_main_on_covered_pairs"),
                ):
                    frame.insert(0, "species", species)
                    frame.insert(1, "protocol", protocol)
                    frame["evaluated_model"] = evaluated_model
                    metric_frames.append(frame)
                coverage.insert(0, "species", species)
                coverage.insert(1, "protocol", protocol)
                coverage_frames.append(coverage)
                for metric in ("auroc", "auprc", "pair_acc"):
                    if external_metrics.empty or main_metrics.empty:
                        continue
                    summary, deltas = paired_comparison(
                        external_metrics,
                        main_metrics,
                        metric,
                        bootstrap_iterations=bootstrap_iterations,
                    )
                    summary.update(
                        {
                            "species": species,
                            "protocol": protocol,
                            "external_model": model,
                            "main_model": main_candidate,
                        }
                    )
                    paired_rows.append(summary)
                    deltas.insert(0, "species", species)
                    deltas.insert(1, "protocol", protocol)
                    deltas.insert(2, "external_model", model)
                    deltas.insert(3, "metric", metric)
                    delta_frames.append(deltas)

    if not row_frames:
        raise ValueError("No score cache matched human or mouse.")
    rows = pd.concat(row_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    coverage = pd.concat(coverage_frames, ignore_index=True)
    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired["bh_family"] = (
            paired["species"].astype(str)
            + "||"
            + paired["protocol"].astype(str)
            + "||"
            + paired["metric"].astype(str)
        )
        paired["wilcoxon_qvalue"] = paired.groupby("bh_family", group_keys=False)[
            "wilcoxon_pvalue"
        ].apply(bh_adjust)
    deltas = pd.concat(delta_frames, ignore_index=True) if delta_frames else pd.DataFrame()

    rows.to_csv(output_dir / "row_predictions.csv.gz", index=False)
    metrics.to_csv(output_dir / "per_task_metrics.csv", index=False)
    coverage.to_csv(output_dir / "coverage_audit.csv", index=False)
    paired.to_csv(output_dir / "paired_statistics.csv", index=False)
    deltas.to_csv(output_dir / "per_task_differences.csv", index=False)

    summary = (
        metrics.groupby(
            ["species", "protocol", "evaluated_model"], as_index=False, sort=True
        )
        .agg(
            n_tasks=("task_name", "nunique"),
            mean_task_auroc=("auroc", "mean"),
            median_task_auroc=("auroc", "median"),
            mean_task_auprc=("auprc", "mean"),
            median_task_auprc=("auprc", "median"),
            mean_task_pair_acc=("pair_acc", "mean"),
            median_task_pair_acc=("pair_acc", "median"),
        )
    )
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    atomic_json(
        output_dir / "metadata.json",
        {
            "purpose": "Frozen general pMHC signal controls for Issue 5",
            "cache_files": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in cache_paths
            ],
            "bootstrap_iterations": bootstrap_iterations,
            "score_policy": "higher cached score is always stronger",
            "coverage_policy": (
                "Metrics use complete scored pairs only; main-model metrics are recomputed "
                "on the identical covered pair set."
            ),
            "inference_scope": "Nominal task-level inference; tasks are not independent cohorts.",
            "protocol_note": (
                "Matched-standard and peptide-disjoint OOF use the same train row pool, "
                "so a frozen external predictor has identical standalone metrics when "
                "coverage is identical; paired main-model comparisons differ."
            ),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-cache", type=Path, action="append", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_RESULTS / "external_evaluation"
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.score_cache, args.output_dir, args.bootstrap_iterations)
