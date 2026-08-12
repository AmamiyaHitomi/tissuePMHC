#!/usr/bin/env python3
"""Build paper-ready CSV tables entirely from frozen outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import EXPERIMENTS, OUTPUT_ROOT, attach_data, ensure_output, read_predictions, read_train, task_metrics


def main() -> None:
    output = ensure_output("06_tables")
    benchmark_rows, metric_rows, task_parts = [], [], []
    for species, experiment in EXPERIMENTS.items():
        train = read_train(experiment)
        benchmark_rows.append({
            "species": species, "tasks": train["task_name"].nunique(),
            "tissues": train["target_tissue"].nunique(), "mhc_restrictions": train["mhc_restriction"].nunique(),
            "train_rows": len(train), "train_pairs": train["pair_id"].nunique(),
            "unique_peptides": train["peptide_sequence"].nunique(),
            "unique_parent_uniprot": train["molecule_parent_uniprot_id"].nunique(),
        })
        for protocol in ("standard", "strict"):
            frame = attach_data(experiment, read_predictions(experiment, protocol))
            tasks = task_metrics(frame)
            tasks.insert(0, "protocol", protocol)
            tasks.insert(0, "species", species)
            task_parts.append(tasks)
            metric_rows.append({
                "species": species, "protocol": protocol, "n_tasks": len(tasks),
                **{f"mean_task_{metric}": float(tasks[metric].mean()) for metric in [
                    "accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "mcc"
                ]},
                "worst_group_auroc": float(tasks.nsmallest(10 if species == "human" else 6, "auroc")["auroc"].mean()),
            })
    pd.DataFrame(benchmark_rows).to_csv(output / "table_1_benchmark_statistics.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output / "table_4_standard_strict_summary.csv", index=False)
    pd.concat(task_parts, ignore_index=True).to_csv(output / "supplement_all_task_metrics.csv", index=False)

    optional = {
        "table_pairacc.csv": OUTPUT_ROOT / "01_pairacc/pairacc_summary.csv",
        "table_fold_matching.csv": OUTPUT_ROOT / "02_matched_fold_audit/matching_summary.csv",
        "table_overlap_audit.csv": OUTPUT_ROOT / "02_matched_fold_audit/protocol_overlap_audit.csv",
        "table_parent_protein_overlap.csv": OUTPUT_ROOT / "03_parent_protein_overlap/protein_overlap_summary.csv",
        "table_statistical_tests.csv": OUTPUT_ROOT / "05_statistics/paired_statistical_tests.csv",
    }
    for destination, source in optional.items():
        if source.is_file():
            pd.read_csv(source).to_csv(output / destination, index=False)

    timing_parts = []
    timing_sources = {
        "human": Path("results/tissuePMHC_phase7_min200_e31_peptide_disjoint_oof/timing.csv"),
        "mouse": Path("results/mousePMHC_phase6_e33_peptide_disjoint_oof/mousePMHC_phase6_e33_metadata.json"),
    }
    human_timing = timing_sources["human"]
    if human_timing.is_file():
        frame = pd.read_csv(human_timing)
        frame.insert(0, "species", "human")
        timing_parts.append(frame)
    if timing_parts:
        pd.concat(timing_parts, ignore_index=True).to_csv(output / "supplement_training_times.csv", index=False)
    print(f"wrote tables to {output}")


if __name__ == "__main__":
    main()

