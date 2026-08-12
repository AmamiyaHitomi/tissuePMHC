#!/usr/bin/env python3
"""Create the publication figure for the mousePMHC Phase 3 pair learning curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_STYLE = {
    "e0_blosum62_extra_trees": ("E0 BLOSUM62 ExtraTrees", "#2673B8", "o"),
    "e0_onehot_logistic": ("E0 one-hot logistic", "#E18A20", "s"),
    "e1_shared_task_heads": ("mousePMHC E1 shared task heads", "#2D9A68", "^"),
}


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def run(args: argparse.Namespace) -> None:
    summary = pd.read_csv(args.summary)
    coverage = pd.read_csv(args.coverage)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })
    fig, (ax, ax_coverage) = plt.subplots(
        2, 1, figsize=(8.2, 7.1), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.15], "hspace": 0.12},
        layout="constrained",
    )

    for model, (label, color, marker) in MODEL_STYLE.items():
        data = summary[summary.model == model].sort_values("pairs_per_task")
        ax.errorbar(
            data.pairs_per_task, data.mean_auroc, yerr=data.repeat_sd_auroc,
            color=color, marker=marker, markersize=5.5, linewidth=2,
            elinewidth=1, capsize=3, label=label,
        )
        last = data.iloc[-1]
        ax.annotate(
            f"{last.mean_auroc:.3f}", (last.pairs_per_task, last.mean_auroc),
            xytext=(7, 0), textcoords="offset points", va="center", color=color,
        )

    ax.axvspan(450, 500, color="#9AA0A6", alpha=0.13, linewidth=0)
    ax.text(475, 0.736, "450-500 near-plateau", ha="center", va="top", color="#666666", fontsize=9)
    ax.set_ylim(0.60, 0.74)
    ax.set_ylabel("Mean task AUROC")
    ax.set_title("mousePMHC Phase 3 learning curves: all eligible tasks per threshold")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.75)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.995, 0.015, "Error bars: +/- 1 SD across 5 repeated subsamples",
        transform=ax.transAxes, ha="right", va="bottom", color="#666666", fontsize=8.5,
    )

    ax_coverage.plot(
        coverage.min_pairs, coverage.n_tasks, color="#7655B5",
        marker="D", markersize=5, linewidth=2,
    )
    for row in coverage.itertuples(index=False):
        ax_coverage.annotate(
            str(row.n_tasks), (row.min_pairs, row.n_tasks), xytext=(0, 7),
            textcoords="offset points", ha="center", color="#7655B5", fontsize=9,
        )
    ax_coverage.set_ylabel("Eligible tasks")
    ax_coverage.set_xlabel("Minimum pairs per tissue-H2 task")
    ax_coverage.set_ylim(0, max(coverage.n_tasks) + 8)
    ax_coverage.set_xticks(coverage.min_pairs)
    ax_coverage.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.75)

    png = args.output_dir / "01_mousePMHC_phase3_pair_learning_curve.png"
    pdf = args.output_dir / "01_mousePMHC_phase3_pair_learning_curve.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "figure": "mousePMHC Phase 3 pair learning curve",
        "summary_source": str(args.summary),
        "coverage_source": str(args.coverage),
        "cohort_policy": "Each point includes every tissue-H2 task eligible at that threshold; every included task is subsampled to the plotted pair count.",
        "outputs": [str(png), str(pdf)],
    }
    (args.output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(png)
    print(pdf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = project_path("results/mousePMHC_phase3_pair_learning_curve")
    parser.add_argument("--summary", type=Path, default=root / "mousePMHC_phase3_pair_learning_curve_summary.csv")
    parser.add_argument("--coverage", type=Path, default=root / "mousePMHC_phase3_pair_threshold_coverage.csv")
    parser.add_argument("--output-dir", type=Path, default=project_path("results/figures_phase3"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
