#!/usr/bin/env python3
"""Audit task-wise size matching between standard and peptide-disjoint folds."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import EXPERIMENTS, ensure_output, make_standard_folds, read_strict_pair_folds, read_train, write_json


def protocol_rows(train: pd.DataFrame, assignments: pd.Series, species: str, protocol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = train.copy()
    working["fold"] = assignments.to_numpy(dtype=int)
    task_rows = []
    overlap_rows = []
    for fold in sorted(working["fold"].unique()):
        held = working[working["fold"] == fold]
        fitting = working[working["fold"] != fold]
        overlap_rows.append({
            "species": species,
            "protocol": protocol,
            "fold": int(fold),
            "held_rows": len(held),
            "held_pairs": held["pair_id"].nunique(),
            "held_peptides": held["peptide_sequence"].nunique(),
            "pair_overlap": len(set(held["pair_id"]) & set(fitting["pair_id"])),
            "peptide_overlap": len(set(held["peptide_sequence"]) & set(fitting["peptide_sequence"])),
            "held_peptides_seen_in_fit_pct": 100.0 * len(
                set(held["peptide_sequence"]) & set(fitting["peptide_sequence"])
            ) / held["peptide_sequence"].nunique(),
        })
        held_counts = held.groupby("task_name")["pair_id"].nunique()
        fit_counts = fitting.groupby("task_name")["pair_id"].nunique()
        for task_name in sorted(train["task_name"].unique()):
            tissue, mhc = task_name.split("||", 1)
            task_rows.append({
                "species": species,
                "protocol": protocol,
                "fold": int(fold),
                "target_tissue": tissue,
                "mhc_restriction": mhc,
                "held_pairs": int(held_counts.get(task_name, 0)),
                "fit_pairs": int(fit_counts.get(task_name, 0)),
            })
    return pd.DataFrame(task_rows), pd.DataFrame(overlap_rows)


def main() -> None:
    output = ensure_output("02_matched_fold_audit")
    task_parts, overlap_parts = [], []
    for species, experiment in EXPERIMENTS.items():
        train = read_train(experiment)
        standard = make_standard_folds(train)
        strict_pairs = read_strict_pair_folds(experiment)
        strict = train["pair_id"].map(strict_pairs.set_index("pair_id")["fold"])
        if strict.isna().any():
            raise ValueError(f"Strict fold assignment misses rows for {species}")
        for protocol, assignments in (("standard", standard), ("strict", strict.astype(int))):
            tasks, overlap = protocol_rows(train, assignments, species, protocol)
            task_parts.append(tasks)
            overlap_parts.append(overlap)
    tasks = pd.concat(task_parts, ignore_index=True)
    overlap = pd.concat(overlap_parts, ignore_index=True)
    standard = tasks[tasks["protocol"] == "standard"].drop(columns="protocol")
    strict = tasks[tasks["protocol"] == "strict"].drop(columns="protocol")
    keys = ["species", "fold", "target_tissue", "mhc_restriction"]
    comparison = standard.merge(strict, on=keys, suffixes=("_standard", "_strict"), validate="one_to_one")
    comparison["held_pair_difference"] = comparison["held_pairs_strict"] - comparison["held_pairs_standard"]
    comparison["absolute_held_pair_difference"] = comparison["held_pair_difference"].abs()
    comparison["relative_held_pair_difference_pct"] = 100.0 * comparison["absolute_held_pair_difference"] / comparison[
        "held_pairs_standard"
    ].clip(lower=1)
    summary = comparison.groupby("species", as_index=False).agg(
        n_task_folds=("held_pair_difference", "size"),
        exact_matches=("held_pair_difference", lambda values: int((values == 0).sum())),
        mean_absolute_pair_difference=("absolute_held_pair_difference", "mean"),
        max_absolute_pair_difference=("absolute_held_pair_difference", "max"),
        mean_relative_difference_pct=("relative_held_pair_difference_pct", "mean"),
        max_relative_difference_pct=("relative_held_pair_difference_pct", "max"),
    )
    summary["exact_match_pct"] = 100.0 * summary["exact_matches"] / summary["n_task_folds"]
    tasks.to_csv(output / "protocol_task_fold_sizes.csv", index=False)
    comparison.to_csv(output / "standard_vs_strict_task_fold_comparison.csv", index=False)
    overlap.to_csv(output / "protocol_overlap_audit.csv", index=False)
    summary.to_csv(output / "matching_summary.csv", index=False)
    write_json(output / "interpretation.json", {
        "causal_warning": "A large size mismatch means the observed standard-to-strict gap cannot be attributed only to peptide disjointness.",
        "standard_fold_seed": 20260711,
        "folds": 3,
    })
    print(summary.to_string(index=False))
    print(overlap.to_string(index=False))


if __name__ == "__main__":
    main()

