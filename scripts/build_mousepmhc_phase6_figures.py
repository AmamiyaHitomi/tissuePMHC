#!/usr/bin/env python3
"""Build report-ready mousePMHC Phase 6 E32/E33 figures and source tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "oof": "#6f7c80",
    "test": "#147d92",
    "positive": "#147d92",
    "negative": "#b24a46",
    "peptide": "#d17a22",
    "protein": "#7b61a8",
    "dark": "#30373b",
    "grid": "#c7ced1",
}
H2_COLORS = {"H2-Db": "#147d92", "H2-Kb": "#d17a22", "H2-Kd": "#7b61a8", "H2-Kk": "#b24a46"}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.alpha": 0.22, "grid.linewidth": 0.7,
        "legend.frameon": False, "savefig.dpi": 300,
    })


def phase6_milestones(results: Path) -> pd.DataFrame:
    """Combine frozen historical milestones, Phase 6 screens, and E32 confirmation."""
    e8 = pd.read_csv(results / "mousePMHC_phase4_e8_e3b_seed_ensemble_oof" / "mousePMHC_phase4_e8_oof_summary_metrics.csv")
    e15 = pd.read_csv(results / "mousePMHC_phase4_e15_five_seed_confirmation" / "mousePMHC_phase4_e15_oof_summary_metrics.csv")
    e8 = e8.loc[e8.candidate.eq("mousePMHC_phase4_e8_e3b_3seed_probability_mean")].iloc[0]
    e15 = e15.loc[e15.candidate.eq("mousePMHC_phase4_e15_e3b_5seed_probability_mean")].iloc[0]
    phase6 = {}
    paths = {
        "E26 flank/position branch": results / "mousePMHC_phase6_e26_flank_mmoe_oof" / "mousePMHC_phase6_e26_oof_summary_metrics.csv",
        "E27 H2-Kd adapter": results / "mousePMHC_phase6_e27_kd_adapter_oof" / "mousePMHC_phase6_e27_oof_summary_metrics.csv",
        "E28 H2-Kb/Kd adapters": results / "mousePMHC_phase6_e28_kb_kd_adapters_oof" / "mousePMHC_phase6_e28_oof_summary_metrics.csv",
    }
    for label, file in paths.items():
        table = pd.read_csv(file)
        phase6[label] = table[["mean_task_auroc", "mean_task_auprc", "worst6_task_auroc"]].mean()
    fixed = pd.read_csv(results / "mousePMHC_phase6_e32_e15_fixed_test" / "mousePMHC_phase6_e32_fixed_test_metrics.csv")
    fixed = fixed.loc[fixed.target_tissue.isna()].iloc[0]
    rows = [
        ("E32 frozen E15 (fixed test)", "confirmation", fixed.mean_task_auroc, fixed.mean_task_auprc, fixed.worst6_task_auroc),
        ("E15 5-seed ensemble (OOF)", "frozen", e15.mean_task_auroc, e15.mean_task_auprc, e15.worst6_task_auroc),
        ("E8 3-seed ensemble (OOF)", "ensemble", e8.mean_task_auroc, e8.mean_task_auprc, e8.worst6_task_auroc),
        ("E7 H2-Kk adapter (OOF)", "earlier", 0.8180, 0.8065, 0.6816),
        ("E3b MMoE baseline (OOF)", "earlier", 0.8148, 0.8043, 0.6771),
        ("E1 shared encoder (OOF)", "earlier", 0.8073, 0.7929, 0.6617),
        ("E0 BLOSUM62 RF (OOF)", "reference", 0.7530, 0.7292, np.nan),
    ]
    for label, values in phase6.items():
        rows.append((f"{label} (OOF)", "stopped", values.mean_task_auroc, values.mean_task_auprc, values.worst6_task_auroc))
    return pd.DataFrame(rows, columns=["label", "kind", "mean_auroc", "mean_auprc", "worst6_auroc"]).sort_values("mean_auroc", ascending=False).reset_index(drop=True)


def milestone_color(kind: str) -> str:
    return {
        "confirmation": COLORS["test"],
        "frozen": COLORS["peptide"],
        "ensemble": "#4f98a8",
        "stopped": COLORS["negative"],
    }.get(kind, COLORS["oof"])


def plot_model_milestones(summary: pd.DataFrame) -> plt.Figure:
    summary = summary.sort_values("mean_auroc", ascending=False).reset_index(drop=True)
    metrics = [("mean_auroc", "Mean task AUROC"), ("mean_auprc", "Mean task AUPRC"), ("worst6_auroc", "Worst-6 mean AUROC")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.9), sharey=True)
    y = np.arange(len(summary)); colors = [milestone_color(kind) for kind in summary.kind]
    baseline = summary.loc[summary.label.str.startswith("E3b MMoE baseline")].iloc[0]
    for ax, (metric, title) in zip(axes, metrics):
        values = summary[metric].to_numpy(float); finite = values[np.isfinite(values)]
        lower = finite.min() - 0.012
        for index, value in enumerate(values):
            if np.isfinite(value):
                ax.barh(index, value - lower, left=lower, color=colors[index], alpha=0.93)
                ax.text(value + 0.001, index, f"{value:.4f}", va="center", fontsize=8)
            else:
                ax.text(lower + 0.002, index, "not reported", va="center", fontsize=8, color=COLORS["oof"])
        ax.axvline(float(baseline[metric]), color=COLORS["dark"], linestyle="--", linewidth=1.2, zorder=3)
        ax.set_title(title); ax.set_xlabel("Task-macro score")
        ax.set_xlim(lower, finite.max() + 0.028); ax.set_yticks(y, summary.label)
    axes[0].set_ylabel("Research milestone"); axes[0].invert_yaxis()
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=color, markeredgecolor="none", markersize=8, label=label)
        for color, label in [
            (COLORS["test"], "Frozen fixed-test confirmation"),
            (COLORS["peptide"], "Frozen OOF endpoint"),
            ("#4f98a8", "Earlier seed ensemble"),
            (COLORS["oof"], "Earlier OOF milestone"),
            (COLORS["negative"], "Stopped Phase 6 candidate"),
        ]
    ]
    handles.append(plt.Line2D([0], [0], color=COLORS["dark"], linestyle="--", linewidth=1.2, label="E3b OOF baseline"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.060), ncol=6, fontsize=8)
    fig.suptitle("mousePMHC milestones: Phase 6 preserves E15 and adds frozen fixed-test confirmation", y=1.0)
    return fig


def load_confirmation(results: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oof_summary = pd.read_csv(results / "mousePMHC_phase4_e15_five_seed_confirmation" / "mousePMHC_phase4_e15_oof_summary_metrics.csv")
    oof_summary = oof_summary.loc[oof_summary.candidate.eq("mousePMHC_phase4_e15_e3b_5seed_probability_mean")].iloc[0]
    fixed = pd.read_csv(results / "mousePMHC_phase6_e32_e15_fixed_test" / "mousePMHC_phase6_e32_fixed_test_metrics.csv")
    fixed_summary = fixed.loc[fixed.target_tissue.isna()].iloc[0]
    fixed_tasks = fixed.loc[fixed.target_tissue.notna()].copy()
    comparison = pd.DataFrame([
        {"evaluation": "Train-only OOF", "mean_auroc": oof_summary.mean_task_auroc, "mean_auprc": oof_summary.mean_task_auprc, "worst6_auroc": oof_summary.worst6_task_auroc},
        {"evaluation": "Frozen fixed test", "mean_auroc": fixed_summary.mean_task_auroc, "mean_auprc": fixed_summary.mean_task_auprc, "worst6_auroc": fixed_summary.worst6_task_auroc},
    ])
    oof_tasks = pd.read_csv(results / "mousePMHC_phase4_e15_five_seed_confirmation" / "mousePMHC_phase4_e15_oof_per_task_metrics.csv")
    oof_tasks = oof_tasks.loc[oof_tasks.candidate.eq("mousePMHC_phase4_e15_e3b_5seed_probability_mean")].copy()
    tasks = fixed_tasks.merge(oof_tasks, on=["target_tissue", "mhc_restriction"], suffixes=("_test", "_oof"), validate="one_to_one")
    tasks["task"] = tasks.target_tissue + " | " + tasks.mhc_restriction
    tasks["delta_auroc"] = tasks.auroc_test - tasks.auroc_oof
    tasks["delta_auprc"] = tasks.auprc_test - tasks.auprc_oof
    h2 = tasks.groupby("mhc_restriction", as_index=False).agg(
        n_tasks=("task", "size"),
        oof_auroc=("auroc_oof", "mean"), test_auroc=("auroc_test", "mean"),
        oof_auprc=("auprc_oof", "mean"), test_auprc=("auprc_test", "mean"),
    )
    h2["delta_auroc"] = h2.test_auroc - h2.oof_auroc
    h2["delta_auprc"] = h2.test_auprc - h2.oof_auprc
    return comparison, tasks, h2


def plot_confirmation(comparison: pd.DataFrame) -> plt.Figure:
    metrics = [("mean_auroc", "Mean task AUROC"), ("mean_auprc", "Mean task AUPRC"), ("worst6_auroc", "Worst-6 AUROC")]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4))
    x = np.arange(2)
    for ax, (metric, title) in zip(axes, metrics):
        values = comparison[metric].to_numpy(float)
        bars = ax.bar(x, values, color=[COLORS["oof"], COLORS["test"]], width=0.62)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
        ax.annotate(f"{values[1] - values[0]:+.4f}", xy=(1, values[1]), xytext=(0, 22), textcoords="offset points", ha="center", color=COLORS["test"], fontsize=9)
        ax.set_xticks(x, ["OOF", "Fixed test"])
        ax.set_ylim(min(values) - 0.025, max(values) + 0.035)
        ax.set_ylabel("Task-macro score")
        ax.set_title(title)
    fig.suptitle("Frozen E15 confirms on the internal pair-disjoint mouse fixed test", y=1.01)
    return fig


def plot_task_deltas(tasks: pd.DataFrame) -> plt.Figure:
    ordered = tasks.sort_values("delta_auroc").reset_index(drop=True)
    y = np.arange(len(ordered))
    colors = ordered.mhc_restriction.map(H2_COLORS)
    fig, ax = plt.subplots(figsize=(9.3, 8.8))
    ax.barh(y, ordered.delta_auroc, color=colors, alpha=0.92)
    ax.axvline(0, color=COLORS["dark"], linewidth=1)
    ax.set_yticks(y, ordered.task, fontsize=7.5)
    ax.set_xlabel("Fixed-test AUROC minus OOF AUROC")
    ax.set_title(f"Task-level confirmation heterogeneity: {(ordered.delta_auroc > 0).sum()} improved, {(ordered.delta_auroc < 0).sum()} declined")
    for index, value in enumerate(ordered.delta_auroc):
        ax.text(value + (0.002 if value >= 0 else -0.002), index, f"{value:+.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=6.8)
    handles = [plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=color, markeredgecolor="none", markersize=8, label=h2) for h2, color in H2_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", ncol=2, fontsize=8)
    return fig


def plot_h2_confirmation(h2: pd.DataFrame) -> plt.Figure:
    h2 = h2.set_index("mhc_restriction").loc[list(H2_COLORS)].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    x = np.arange(len(h2)); width = 0.34
    for ax, metric, title in [(axes[0], "auroc", "Mean task AUROC"), (axes[1], "auprc", "Mean task AUPRC")]:
        oof = h2[f"oof_{metric}"].to_numpy(float); test = h2[f"test_{metric}"].to_numpy(float)
        ax.bar(x - width / 2, oof, width, label="OOF", color=COLORS["oof"])
        ax.bar(x + width / 2, test, width, label="Fixed test", color=COLORS["test"])
        for index, delta in enumerate(test - oof):
            ax.text(index, max(oof[index], test[index]) + 0.012, f"Δ {delta:+.3f}", ha="center", fontsize=8)
        ax.set_xticks(x, h2.mhc_restriction)
        ax.set_ylim(0.67, 0.96)
        ax.set_ylabel("Task-macro score")
        ax.set_title(title)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Fixed-test effects differ by H2 group", y=1.01)
    return fig


def generalization_tables(train_path: Path, test_path: Path, e33_audit: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path, usecols=["peptide_sequence", "molecule_parent_uniprot_id"])
    test = pd.read_csv(test_path, usecols=["peptide_sequence", "molecule_parent_uniprot_id"])
    rows = []
    for label, column in [("Peptide", "peptide_sequence"), ("Parent UniProt", "molecule_parent_uniprot_id")]:
        seen = set(train[column]); overlap = set(test[column]) & seen
        rows.append({"entity": label, "unique_seen_fraction": len(overlap) / test[column].nunique(), "row_seen_fraction": test[column].isin(seen).mean(), "overlap_unique": len(overlap), "test_unique": test[column].nunique()})
    overlap = pd.DataFrame(rows)
    audit = json.loads(e33_audit.read_text(encoding="utf-8"))
    folds = pd.DataFrame(audit["folds"])
    return overlap, folds


def plot_generalization_boundary(overlap: pd.DataFrame, folds: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    y = np.arange(len(overlap)); height = 0.32
    axes[0].barh(y - height / 2, 100 * overlap.unique_seen_fraction, height, color=COLORS["peptide"], label="Unique entities")
    axes[0].barh(y + height / 2, 100 * overlap.row_seen_fraction, height, color=COLORS["protein"], label="Test rows")
    axes[0].set_yticks(y, overlap.entity)
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Seen in training (%)")
    axes[0].set_title("E32 is pair-disjoint, not entity-disjoint")
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2)
    for row_index, row in overlap.iterrows():
        axes[0].text(100 * row.unique_seen_fraction + 1, row_index - height / 2, f"{100 * row.unique_seen_fraction:.1f}%", va="center", fontsize=8)
        axes[0].text(100 * row.row_seen_fraction + 1, row_index + height / 2, f"{100 * row.row_seen_fraction:.1f}%", va="center", fontsize=8)
    x = np.arange(len(folds))
    bars = axes[1].bar(x, folds.held_out_pairs, color=COLORS["test"], width=0.58)
    for bar, (_, row) in zip(bars, folds.iterrows()):
        axes[1].text(bar.get_x() + bar.get_width() / 2, row.held_out_pairs + 22, f"{int(row.held_out_pairs):,} pairs", ha="center", fontsize=8)
        axes[1].text(bar.get_x() + bar.get_width() / 2, row.held_out_pairs / 2, "24 tasks\n0 peptide overlap", ha="center", va="center", color="white", fontsize=8)
    axes[1].set_xticks(x, [f"Fold {int(fold) + 1}" for fold in folds.fold])
    axes[1].set_ylim(0, folds.held_out_pairs.max() + 180)
    axes[1].set_ylabel("Held-out pairs")
    axes[1].set_title("E33 component split is feasible without deletion")
    fig.suptitle("Generalization boundary and peptide-disjoint split feasibility", y=1.01)
    return fig


def save(fig: plt.Figure, output: Path, stem: str, pdf: PdfPages) -> None:
    fig.tight_layout(); fig.savefig(output / f"{stem}.png", bbox_inches="tight"); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "mousePMHC" / "mousePMHC_train.csv.gz")
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "mousePMHC" / "mousePMHC_test.csv.gz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "figures_phase6")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True); apply_style()
    milestones = phase6_milestones(args.results_dir)
    comparison, tasks, h2 = load_confirmation(args.results_dir)
    overlap, folds = generalization_tables(args.train, args.test, args.results_dir / "mousePMHC_phase6_e33_peptide_disjoint_oof" / "mousePMHC_phase6_e33_split_audit.json")
    milestones.to_csv(args.output_dir / "model_milestones.csv", index=False)
    comparison.to_csv(args.output_dir / "e32_oof_fixed_comparison.csv", index=False)
    tasks.to_csv(args.output_dir / "e32_task_deltas.csv", index=False)
    h2.to_csv(args.output_dir / "e32_h2_comparison.csv", index=False)
    overlap.to_csv(args.output_dir / "e32_entity_overlap.csv", index=False)
    folds.to_csv(args.output_dir / "e33_split_feasibility.csv", index=False)
    with PdfPages(args.output_dir / "mousepmhc_phase6_e32_e33_figures.pdf") as pdf:
        save(plot_model_milestones(milestones), args.output_dir, "00_phase6_model_milestones", pdf)
        save(plot_confirmation(comparison), args.output_dir, "01_e32_oof_fixed_confirmation", pdf)
        save(plot_task_deltas(tasks), args.output_dir, "02_e32_task_auroc_deltas", pdf)
        save(plot_h2_confirmation(h2), args.output_dir, "03_e32_h2_confirmation", pdf)
        save(plot_generalization_boundary(overlap, folds), args.output_dir, "04_generalization_boundary", pdf)
    metadata = {
        "status": "phase6_e32_fixed_test_and_e33_split_audit",
        "n_figures": 5,
        "fixed_test_result": {"mean_task_auroc": 0.8562208333333333, "mean_task_auprc": 0.8506252251615378, "worst6_auroc": 0.7244666666666667},
        "e33_model_training_completed": False,
        "interpretation": "E32 is an internal pair-disjoint fixed test with entity overlap; E33 shows split feasibility only and has no model performance result.",
    }
    (args.output_dir / "figure_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote five Phase 6 figures, source tables, and one PDF to: {args.output_dir}")


if __name__ == "__main__":
    main()
