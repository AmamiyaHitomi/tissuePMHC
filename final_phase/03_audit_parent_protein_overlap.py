#!/usr/bin/env python3
"""Audit parent-UniProt overlap in standard and peptide-disjoint folds."""

from __future__ import annotations

import pandas as pd

from common import EXPERIMENTS, ensure_output, make_standard_folds, read_strict_pair_folds, read_train


def audit(train: pd.DataFrame, assignments: pd.Series, species: str, protocol: str) -> tuple[list[dict], list[dict]]:
    working = train.copy()
    working["fold"] = assignments.to_numpy(dtype=int)
    global_rows, task_rows = [], []
    for fold in sorted(working["fold"].unique()):
        held, fitting = working[working.fold == fold], working[working.fold != fold]
        held_proteins = set(held["molecule_parent_uniprot_id"].dropna().astype(str))
        fit_proteins = set(fitting["molecule_parent_uniprot_id"].dropna().astype(str))
        seen = held_proteins & fit_proteins
        global_rows.append({
            "species": species, "protocol": protocol, "fold": int(fold),
            "held_unique_proteins": len(held_proteins),
            "fit_unique_proteins": len(fit_proteins),
            "overlap_unique_proteins": len(seen),
            "held_unique_seen_pct": 100.0 * len(seen) / max(len(held_proteins), 1),
            "held_rows_with_seen_protein": int(held["molecule_parent_uniprot_id"].astype(str).isin(seen).sum()),
            "held_rows_seen_pct": 100.0 * held["molecule_parent_uniprot_id"].astype(str).isin(seen).mean(),
        })
        for task_name, task_held in held.groupby("task_name", sort=True):
            task_fit = fitting[fitting.task_name == task_name]
            hp = set(task_held["molecule_parent_uniprot_id"].dropna().astype(str))
            fp = set(task_fit["molecule_parent_uniprot_id"].dropna().astype(str))
            task_rows.append({
                "species": species, "protocol": protocol, "fold": int(fold), "task_name": task_name,
                "held_unique_proteins": len(hp), "overlap_unique_proteins": len(hp & fp),
                "held_unique_seen_pct": 100.0 * len(hp & fp) / max(len(hp), 1),
            })
    return global_rows, task_rows


def main() -> None:
    output = ensure_output("03_parent_protein_overlap")
    global_rows, task_rows = [], []
    for species, experiment in EXPERIMENTS.items():
        train = read_train(experiment)
        standard = make_standard_folds(train)
        strict_map = read_strict_pair_folds(experiment).set_index("pair_id")["fold"]
        strict = train["pair_id"].map(strict_map)
        for protocol, assignments in (("standard", standard), ("strict", strict.astype(int))):
            global_part, task_part = audit(train, assignments, species, protocol)
            global_rows.extend(global_part)
            task_rows.extend(task_part)
    global_frame = pd.DataFrame(global_rows)
    task_frame = pd.DataFrame(task_rows)
    summary = global_frame.groupby(["species", "protocol"], as_index=False).agg(
        mean_held_unique_seen_pct=("held_unique_seen_pct", "mean"),
        min_held_unique_seen_pct=("held_unique_seen_pct", "min"),
        max_held_unique_seen_pct=("held_unique_seen_pct", "max"),
        mean_held_rows_seen_pct=("held_rows_seen_pct", "mean"),
    )
    global_frame.to_csv(output / "global_fold_protein_overlap.csv", index=False)
    task_frame.to_csv(output / "task_fold_protein_overlap.csv", index=False)
    summary.to_csv(output / "protein_overlap_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

