#!/usr/bin/env python3
"""Collect completed 157-task migrations into Table-5 candidate rows.

The collector only reads the isolated migration root created by
``migrate_44tasks_to_157tasks.py``.  It never mixes archived 44-task outputs
with Phase-7 outputs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "results" / "tissuePMHC_phase7_min200_migrated_44tasks"
)
FIELDS = [
    "migration_experiment",
    "experiment_name",
    "model",
    "n_tasks",
    "n_seeds",
    "mean_auroc",
    "mean_auprc",
    "mean_accuracy",
    "mean_mcc",
    "worst_10_mean_auroc",
    "source_file",
]
DEFAULT_MIN_METHODS = 10


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def first(row: dict[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = row.get(name, "")
        if value != "":
            return value
    return default


def normalize_stability(
    experiment: str, path: Path, rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "migration_experiment": experiment,
                "experiment_name": first(row, "experiment_name", "model"),
                "model": first(row, "model", "experiment_name"),
                "n_tasks": first(row, "n_tasks", default="157"),
                "n_seeds": first(row, "n_seeds"),
                "mean_auroc": first(row, "mean_auroc_mean"),
                "mean_auprc": first(row, "mean_auprc_mean"),
                "mean_accuracy": first(row, "mean_accuracy_mean"),
                "mean_mcc": first(row, "mean_mcc_mean"),
                "worst_10_mean_auroc": first(
                    row, "worst_10_mean_auroc_mean"
                ),
                "source_file": str(path),
            }
        )
    return normalized


def normalize_summary(
    experiment: str, path: Path, rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "migration_experiment": experiment,
                "experiment_name": first(row, "experiment_name", "model"),
                "model": first(row, "model", "experiment_name"),
                "n_tasks": first(row, "n_tasks", default="157"),
                "n_seeds": "1" if first(row, "seed") else "",
                "mean_auroc": first(row, "mean_auroc"),
                "mean_auprc": first(row, "mean_auprc"),
                "mean_accuracy": first(row, "mean_accuracy"),
                "mean_mcc": first(row, "mean_mcc"),
                "worst_10_mean_auroc": first(row, "worst_10_mean_auroc"),
                "source_file": str(path),
            }
        )
    return normalized


def collect(input_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for experiment_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        contract = experiment_dir / "migration_contract.json"
        if not contract.is_file():
            continue
        stability = experiment_dir / "stability_metrics.csv"
        summary = experiment_dir / "summary_metrics.csv"
        if stability.is_file():
            rows.extend(
                normalize_stability(
                    experiment_dir.name, stability, read_csv(stability)
                )
            )
        elif summary.is_file():
            rows.extend(
                normalize_summary(experiment_dir.name, summary, read_csv(summary))
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "| Migration | Model | Tasks | Seeds | Mean AUROC | Mean AUPRC | Accuracy | MCC | Worst-10 AUROC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {migration_experiment} | {model} | {n_tasks} | {n_seeds} | "
            "{mean_auroc} | {mean_auprc} | {mean_accuracy} | {mean_mcc} | "
            "{worst_10_mean_auroc} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument(
        "--min-methods",
        type=int,
        default=DEFAULT_MIN_METHODS,
        help=(
            "Required number of distinct migrated methods. The collector never "
            "truncates rows when this threshold is exceeded."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    csv_output = args.csv_output or input_root / "table5_migrated_rows.csv"
    markdown_output = (
        args.markdown_output or input_root / "table5_migrated_rows.md"
    )
    rows = collect(input_root)
    if not rows:
        raise RuntimeError(f"No completed migration summaries found in {input_root}")
    distinct_methods = {
        (row["migration_experiment"], row["experiment_name"], row["model"])
        for row in rows
    }
    if len(distinct_methods) < args.min_methods:
        raise RuntimeError(
            f"Only {len(distinct_methods)} distinct migrated methods were found; "
            f"Table 5 requires at least {args.min_methods}. Run additional "
            "experiments from the table5 or all migration preset."
        )
    write_csv(csv_output, rows)
    write_markdown(markdown_output, rows)
    print(
        f"Wrote {len(rows)} rows for {len(distinct_methods)} distinct methods "
        f"to {csv_output}; no rows were truncated."
    )
    print(f"Wrote {markdown_output}")


if __name__ == "__main__":
    main()
