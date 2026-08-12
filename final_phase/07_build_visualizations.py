#!/usr/bin/env python3
"""Generate final-phase PNG/PDF figures from existing CSV predictions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from common import EXPERIMENTS, OUTPUT_ROOT, attach_data, ensure_output, read_predictions, task_metrics


COLORS = {"human": "#2563EB", "mouse": "#EA580C"}


def task_comparison() -> pd.DataFrame:
    parts = []
    for species, experiment in EXPERIMENTS.items():
        standard = task_metrics(attach_data(experiment, read_predictions(experiment, "standard")))
        strict = task_metrics(attach_data(experiment, read_predictions(experiment, "strict")))
        merged = standard.merge(
            strict, on=["target_tissue", "mhc_restriction"], suffixes=("_standard", "_strict"), validate="one_to_one"
        )
        merged.insert(0, "species", species)
        merged["delta_auroc"] = merged["auroc_strict"] - merged["auroc_standard"]
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)


def scatter(comparison: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for axis, species in zip(axes, ("human", "mouse")):
        frame = comparison[comparison.species == species]
        axis.scatter(frame.auroc_standard, frame.auroc_strict, s=24, alpha=0.75, color=COLORS[species])
        axis.plot([0.5, 1], [0.5, 1], "--", color="#64748B", linewidth=1)
        axis.set(xlim=(0.5, 1), ylim=(0.5, 1), xlabel="Standard OOF AUROC", ylabel="Peptide-disjoint OOF AUROC", title=species.capitalize())
        axis.grid(alpha=0.2)
    fig.suptitle("Task-wise standard versus peptide-disjoint performance")
    return fig


def deltas(comparison: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for axis, species in zip(axes, ("human", "mouse")):
        values = comparison.loc[comparison.species == species, "delta_auroc"]
        axis.hist(values, bins=20, color=COLORS[species], alpha=0.85)
        axis.axvline(0, color="black", linewidth=1)
        axis.axvline(values.mean(), color="#7C3AED", linestyle="--", linewidth=1.5, label=f"mean={values.mean():.3f}")
        axis.set(xlabel="Strict − standard AUROC", ylabel="Tasks", title=species.capitalize())
        axis.legend(frameon=False)
    fig.suptitle("Generalization-gap distribution")
    return fig


def locus_plot(comparison: pd.DataFrame) -> plt.Figure:
    frame = comparison.copy()
    frame["group"] = np.where(
        frame.species == "human",
        frame.mhc_restriction.str.extract(r"HLA-([ABC])", expand=False).fillna("other"),
        frame.mhc_restriction,
    )
    summary = frame.groupby(["species", "group"], as_index=False).delta_auroc.mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for axis, species in zip(axes, ("human", "mouse")):
        item = summary[summary.species == species]
        axis.bar(item.group, item.delta_auroc, color=COLORS[species])
        axis.axhline(0, color="black", linewidth=1)
        axis.set(xlabel="MHC group", ylabel="Mean strict − standard AUROC", title=species.capitalize())
    fig.suptitle("Generalization gap by MHC group")
    return fig


def overlap_plot() -> plt.Figure:
    source = OUTPUT_ROOT / "03_parent_protein_overlap/protein_overlap_summary.csv"
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if not source.is_file():
        axis.text(0.5, 0.5, "Run 03_audit_parent_protein_overlap.py first", ha="center", va="center")
        axis.axis("off")
        return fig
    frame = pd.read_csv(source)
    labels = frame.species + "\n" + frame.protocol
    axis.bar(labels, frame.mean_held_unique_seen_pct, color=[COLORS[item] for item in frame.species])
    axis.set(ylabel="Held-out parent proteins seen in fitting (%)", title="Parent-protein overlap remains after peptide-disjoint splitting")
    axis.set_ylim(0, 105)
    return fig


def main() -> None:
    output = ensure_output("07_figures")
    comparison = task_comparison()
    comparison.to_csv(output / "figure_source_task_comparison.csv", index=False)
    figures = [
        ("01_standard_vs_strict_scatter", scatter(comparison)),
        ("02_task_delta_distribution", deltas(comparison)),
        ("03_mhc_group_gap", locus_plot(comparison)),
        ("04_parent_protein_overlap", overlap_plot()),
    ]
    with PdfPages(output / "final_phase_figures.pdf") as pdf:
        for name, figure in figures:
            figure.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    print(f"wrote figures to {output}")


if __name__ == "__main__":
    main()
