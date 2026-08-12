"""Audit label ambiguity after tissue identity is removed.

The tissue-blind control receives only peptide sequence and MHC restriction.
This script counts peptide--MHC queries that retain both labels across tissue
rows and builds the complete mouse saved-test tissue summary used in the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BENCHMARKS = {
    ("Human", "Training pool"): Path(
        "data/tissuePMHC_phase7_min200/tissuePMHC_phase7_min200_train.csv.gz"
    ),
    ("Human", "Saved internal test"): Path(
        "data/tissuePMHC_phase7_min200/tissuePMHC_phase7_min200_test.csv.gz"
    ),
    ("Mouse", "Training pool"): Path("data/mousePMHC/mousePMHC_train.csv.gz"),
    ("Mouse", "Saved internal test"): Path("data/mousePMHC/mousePMHC_test.csv.gz"),
}

MOUSE_METRICS = Path(
    "results/mousePMHC_phase6_e32_e15_fixed_test/"
    "mousePMHC_phase6_e32_fixed_test_metrics.csv"
)
MOUSE_PREDICTIONS = Path(
    "results/mousePMHC_phase6_e32_e15_fixed_test/"
    "mousePMHC_phase6_e32_fixed_test_predictions.csv"
)


def read_benchmark(root: Path, path: Path) -> pd.DataFrame:
    return pd.read_csv(
        root / path,
        usecols=[
            "sample_id",
            "pair_id",
            "label",
            "target_tissue",
            "mhc_restriction",
            "peptide_sequence",
        ],
    )


def conflict_summary(
    species: str, partition: str, frame: pd.DataFrame
) -> dict[str, object]:
    query_columns = ["peptide_sequence", "mhc_restriction"]
    within_task_conflicts = (
        frame.groupby(
            ["target_tissue", "mhc_restriction", "peptide_sequence"], sort=False
        )["label"].nunique()
        > 1
    ).sum()
    if within_task_conflicts:
        raise ValueError(
            f"{species} {partition} contains {within_task_conflicts} "
            "within-task conflicting queries."
        )
    label_counts = frame.groupby(query_columns, sort=False)["label"].nunique()
    conflict_keys = label_counts[label_counts > 1].index
    row_keys = pd.MultiIndex.from_frame(frame[query_columns])
    conflict_rows = row_keys.isin(conflict_keys)
    return {
        "species": species,
        "partition": partition,
        "rows": len(frame),
        "unique_peptide_mhc_queries": len(label_counts),
        "conflicting_queries": len(conflict_keys),
        "conflicting_query_percent": 100.0 * len(conflict_keys) / len(label_counts),
        "rows_in_conflicting_queries": int(conflict_rows.sum()),
        "conflicting_row_percent": 100.0 * conflict_rows.mean(),
    }


def build_conflict_audit(root: Path) -> pd.DataFrame:
    loaded: dict[tuple[str, str], pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for (species, partition), path in BENCHMARKS.items():
        frame = read_benchmark(root, path)
        loaded[(species, partition)] = frame
        rows.append(conflict_summary(species, partition, frame))

    for species in ("Human", "Mouse"):
        complete = pd.concat(
            [
                loaded[(species, "Training pool")],
                loaded[(species, "Saved internal test")],
            ],
            ignore_index=True,
        )
        rows.append(conflict_summary(species, "Complete benchmark", complete))

    order = {
        ("Human", "Training pool"): 0,
        ("Human", "Saved internal test"): 1,
        ("Human", "Complete benchmark"): 2,
        ("Mouse", "Training pool"): 3,
        ("Mouse", "Saved internal test"): 4,
        ("Mouse", "Complete benchmark"): 5,
    }
    result = pd.DataFrame(rows)
    result["_order"] = [
        order[(species, partition)]
        for species, partition in zip(result["species"], result["partition"])
    ]
    return result.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def build_mouse_tissue_summary(root: Path) -> pd.DataFrame:
    metrics = pd.read_csv(root / MOUSE_METRICS)
    metrics = metrics[metrics["target_tissue"].notna()].copy()
    predictions = pd.read_csv(root / MOUSE_PREDICTIONS)
    test = read_benchmark(root, BENCHMARKS[("Mouse", "Saved internal test")])

    joined = predictions.merge(
        test[
            [
                "sample_id",
                "pair_id",
                "label",
                "target_tissue",
                "mhc_restriction",
            ]
        ],
        on=["sample_id", "label", "target_tissue", "mhc_restriction"],
        validate="one_to_one",
    )
    paired = joined.pivot(
        index=["target_tissue", "mhc_restriction", "pair_id"],
        columns="label",
        values="score",
    ).reset_index()
    paired["within_pair_correct"] = (paired[1] > paired[0]).astype(float)
    pair_metrics = (
        paired.groupby(["target_tissue", "mhc_restriction"], as_index=False)
        .agg(
            test_pairs=("pair_id", "size"),
            within_pair_accuracy=("within_pair_correct", "mean"),
        )
    )
    task_metrics = metrics.merge(
        pair_metrics,
        on=["target_tissue", "mhc_restriction"],
        validate="one_to_one",
    )
    result = (
        task_metrics.groupby("target_tissue", as_index=False)
        .agg(
            h2_restrictions=("mhc_restriction", "size"),
            test_pairs=("test_pairs", "sum"),
            mean_auroc=("auroc", "mean"),
            mean_auprc=("auprc", "mean"),
            within_pair_accuracy=("within_pair_accuracy", "mean"),
            minimum_task_auroc=("auroc", "min"),
            maximum_task_auroc=("auroc", "max"),
        )
        .sort_values("mean_auroc")
        .reset_index(drop=True)
    )
    if len(result) != 13:
        raise ValueError(f"Expected 13 mouse tissues, found {len(result)}.")
    if result["h2_restrictions"].sum() != 24 or result["test_pairs"].sum() != 2400:
        raise ValueError("Mouse tissue summary does not recover 24 tasks and 2,400 pairs.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/final_phase/10_tissue_blind_conflict_audit"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    conflicts = build_conflict_audit(root)
    mouse_tissues = build_mouse_tissue_summary(root)
    conflicts.to_csv(output_dir / "tissue_blind_label_conflicts.csv", index=False)
    mouse_tissues.to_csv(output_dir / "mouse_saved_test_tissue_summary.csv", index=False)
    print(conflicts.to_string(index=False))
    print()
    print(mouse_tissues.to_string(index=False))


if __name__ == "__main__":
    main()
