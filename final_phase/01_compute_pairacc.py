#!/usr/bin/env python3
"""Compute pair-ranking accuracy for human/mouse standard and strict OOF."""

from __future__ import annotations

import pandas as pd

from common import EXPERIMENTS, attach_data, ensure_output, read_predictions, write_json


def pair_rows(frame: pd.DataFrame, species: str, protocol: str) -> pd.DataFrame:
    rows = []
    for (tissue, mhc, pair_id), pair in frame.groupby(
        ["target_tissue", "mhc_restriction", "pair_id"], sort=True
    ):
        positive = pair.loc[pair["label"] == 1, "score"]
        negative = pair.loc[pair["label"] == 0, "score"]
        if len(positive) != 1 or len(negative) != 1:
            raise ValueError(f"Pair {pair_id} is not one-positive/one-negative")
        delta = float(positive.iloc[0] - negative.iloc[0])
        rows.append({
            "species": species,
            "protocol": protocol,
            "target_tissue": tissue,
            "mhc_restriction": mhc,
            "pair_id": pair_id,
            "positive_score": float(positive.iloc[0]),
            "negative_score": float(negative.iloc[0]),
            "score_delta": delta,
            "pair_correct": float(delta > 0) + 0.5 * float(delta == 0),
        })
    return pd.DataFrame(rows)


def main() -> None:
    output = ensure_output("01_pairacc")
    all_pairs = []
    for species, experiment in EXPERIMENTS.items():
        for protocol in ("standard", "strict"):
            predictions = read_predictions(experiment, protocol)
            all_pairs.append(pair_rows(attach_data(experiment, predictions), species, protocol))
    pairs = pd.concat(all_pairs, ignore_index=True)
    per_task = pairs.groupby(
        ["species", "protocol", "target_tissue", "mhc_restriction"], as_index=False
    ).agg(n_pairs=("pair_id", "size"), pairacc=("pair_correct", "mean"), median_margin=("score_delta", "median"))
    summary = per_task.groupby(["species", "protocol"], as_index=False).agg(
        n_tasks=("pairacc", "size"),
        n_pairs=("n_pairs", "sum"),
        macro_pairacc=("pairacc", "mean"),
        worst_task_pairacc=("pairacc", "min"),
        median_task_pairacc=("pairacc", "median"),
    )
    wide = summary.pivot(index="species", columns="protocol", values="macro_pairacc")
    delta = (wide["strict"] - wide["standard"]).rename("strict_minus_standard_pairacc").reset_index()
    summary = summary.merge(delta, on="species", how="left")
    pairs.to_csv(output / "pair_level_results.csv.gz", index=False)
    per_task.to_csv(output / "per_task_pairacc.csv", index=False)
    summary.to_csv(output / "pairacc_summary.csv", index=False)
    write_json(output / "metadata.json", {
        "tie_policy": "A tied pair contributes 0.5.",
        "formula": "mean(positive_score > negative_score) with half credit for ties",
        "outputs": ["pair_level_results.csv.gz", "per_task_pairacc.csv", "pairacc_summary.csv"],
    })
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

