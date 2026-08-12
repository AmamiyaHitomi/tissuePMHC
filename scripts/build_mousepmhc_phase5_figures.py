#!/usr/bin/env python3
"""Create report-ready figures for mousePMHC Phases 3-5.

The script reads completed train-only OOF artifacts and the mouse training CSV.
It never opens the fixed test split and does not train any model.
"""

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
from matplotlib.colors import LogNorm


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "baseline": "#6f7c80",
    "ensemble": "#d17a22",
    "final": "#147d92",
    "negative": "#b24a46",
    "neutral": "#c7ced1",
    "dark": "#30373b",
}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.alpha": 0.22, "grid.linewidth": 0.7,
        "legend.frameon": False, "savefig.dpi": 300,
    })


def read_summary(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def phase_milestones(results: Path) -> pd.DataFrame:
    """Use frozen report values for older milestones and result files for Phase 4/5."""
    e8 = read_summary(results / "mousePMHC_phase4_e8_e3b_seed_ensemble_oof" / "mousePMHC_phase4_e8_oof_summary_metrics.csv")
    e15 = read_summary(results / "mousePMHC_phase4_e15_five_seed_confirmation" / "mousePMHC_phase4_e15_oof_summary_metrics.csv")
    e17 = read_summary(results / "mousePMHC_phase5_e17_taskwise_ranking_mmoe_oof" / "mousePMHC_phase5_e17_oof_summary_metrics.csv")
    e19 = read_summary(results / "mousePMHC_phase5_e19_hierarchical_heads_oof" / "mousePMHC_phase5_e19_oof_summary_metrics.csv")
    e8_prob = e8.loc[e8.candidate.eq("mousePMHC_phase4_e8_e3b_3seed_probability_mean")].iloc[0]
    e15_row = e15.loc[e15.candidate.eq("mousePMHC_phase4_e15_e3b_5seed_probability_mean")].iloc[0]
    e17a = e17.loc[e17.candidate.eq("e17a_matched_pair_ranking")].iloc[0]
    e19_row = e19.loc[e19.candidate.eq("e19_hierarchical_tissue_h2_head")].iloc[0]
    rows = [
        ("E15 5-seed ensemble (P4 frozen)", "final", e15_row.mean_task_auroc, e15_row.mean_task_auprc, e15_row.worst6_task_auroc),
        ("E8 3-seed ensemble (P4)", "ensemble", e8_prob.mean_task_auroc, e8_prob.mean_task_auprc, e8_prob.worst6_task_auroc),
        ("E7 H2-Kk adapter (P3)", "milestone", 0.8180, 0.8065, 0.6816),
        ("E3b MMoE baseline (P3)", "milestone", 0.8148, 0.8043, 0.6771),
        ("E1 shared encoder (P3)", "earlier", 0.8073, 0.7929, 0.6617),
        ("E17a ranking screen (P5)", "negative", e17a.mean_task_auroc, e17a.mean_task_auprc, e17a.worst6_task_auroc),
        ("E19 hierarchical screen (P5)", "negative", e19_row.mean_task_auroc, e19_row.mean_task_auprc, e19_row.worst6_task_auroc),
        ("E9 CNN (P4)", "negative", 0.7888, 0.7740, 0.6447),
        ("E12 auxiliary (P4)", "negative", 0.7755, 0.7568, 0.6390),
        ("E0 BLOSUM62 RF (P3 reference)", "earlier", 0.7530, 0.7292, np.nan),
    ]
    return pd.DataFrame(rows, columns=["label", "kind", "mean_auroc", "mean_auprc", "worst6_auroc"])


def milestone_color(kind: str) -> str:
    return {"final": COLORS["final"], "ensemble": COLORS["ensemble"], "negative": COLORS["negative"]}.get(kind, COLORS["baseline"])


def plot_model_milestones(summary: pd.DataFrame) -> plt.Figure:
    summary = summary.sort_values("mean_auroc", ascending=False).reset_index(drop=True)
    metrics = [("mean_auroc", "Mean task AUROC"), ("mean_auprc", "Mean task AUPRC"), ("worst6_auroc", "Worst-6 mean AUROC")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.8), sharey=True)
    y = np.arange(len(summary)); colors = [milestone_color(kind) for kind in summary.kind]
    for ax, (metric, title) in zip(axes, metrics):
        values = summary[metric].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        lower = finite.min() - 0.012
        ref = float(summary.loc[summary.label.str.startswith("E0 "), metric].iloc[0]) if np.isfinite(summary.loc[summary.label.str.startswith("E0 "), metric].iloc[0]) else None
        for index, value in enumerate(values):
            if np.isfinite(value):
                ax.barh(index, value - lower, left=lower, color=colors[index], alpha=0.92)
                ax.text(value + 0.001, index, f"{value:.4f}", va="center", fontsize=8)
            else:
                ax.text(lower + 0.002, index, "not reported", va="center", fontsize=8, color=COLORS["baseline"])
        if ref is not None:
            ax.axvline(ref, color=COLORS["dark"], linestyle="--", linewidth=1.1, zorder=3)
        ax.set_title(title); ax.set_xlabel("Task-macro score")
        ax.set_xlim(lower, finite.max() + 0.025); ax.set_yticks(y, summary.label)
    axes[0].set_ylabel("Research milestone"); axes[0].invert_yaxis()
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS[key], markeredgecolor="none", markersize=8, label=label)
        for key, label in [("baseline", "Earlier single-model milestone"), ("ensemble", "Seed ensemble"), ("final", "Frozen OOF endpoint"), ("negative", "Stopped candidate")]
    ]
    handles.append(plt.Line2D([0], [0], color=COLORS["dark"], linestyle="--", linewidth=1.1, label="E0 reference"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.045), ncol=5, fontsize=8)
    fig.suptitle("mousePMHC Phase 3-5 OOF progression: E15 remains the frozen development endpoint", y=1.0)
    return fig


def phase5_task_deltas(results: Path) -> pd.DataFrame:
    e17 = pd.read_csv(results / "mousePMHC_phase5_e17_taskwise_ranking_mmoe_oof" / "mousePMHC_phase5_e17_paired_task_deltas.csv")
    e19 = pd.read_csv(results / "mousePMHC_phase5_e19_hierarchical_heads_oof" / "mousePMHC_phase5_e19_paired_task_deltas.csv")
    e17["comparison"] = "E17a ranking − matched E3b"
    e19["comparison"] = "E19 hierarchy − matched E3b"
    table = pd.concat([e17, e19], ignore_index=True)
    table["task"] = table.target_tissue + " | " + table.mhc_restriction
    return table


def plot_phase5_task_deltas(deltas: pd.DataFrame) -> plt.Figure:
    names = ["E17a ranking − matched E3b", "E19 hierarchy − matched E3b"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 9), sharey=False)
    for ax, name in zip(axes, names):
        ordered = deltas.loc[deltas.comparison.eq(name)].sort_values("delta_auroc").reset_index(drop=True)
        colors = np.where(ordered.delta_auroc >= 0, COLORS["final"], COLORS["negative"])
        y = np.arange(len(ordered))
        ax.barh(y, ordered.delta_auroc, color=colors)
        ax.axvline(0, color=COLORS["dark"], linewidth=0.9)
        ax.set_yticks(y, ordered.task, fontsize=7)
        ax.set_xlabel("Task AUROC delta")
        ax.set_title(f"{name}\n{(ordered.delta_auroc > 0).sum()} wins / {(ordered.delta_auroc < 0).sum()} losses; mean {ordered.delta_auroc.mean():+.4f}")
    fig.suptitle("Phase 5 candidate effects by tissue-H2 task", y=1.0)
    return fig


def mouse_coverage(train: Path) -> pd.DataFrame:
    data = pd.read_csv(train, usecols=["target_tissue", "mhc_restriction", "pair_id"])
    return data.groupby(["target_tissue", "mhc_restriction"], as_index=False).pair_id.nunique().rename(columns={"pair_id": "train_pairs"})


def plot_mouse_coverage(coverage: pd.DataFrame) -> plt.Figure:
    pivot = coverage.pivot(index="target_tissue", columns="mhc_restriction", values="train_pairs").fillna(0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values().index]
    matrix = pivot.to_numpy(); positive = matrix[matrix > 0]
    masked = np.ma.masked_where(matrix <= 0, matrix)
    fig, ax = plt.subplots(figsize=(8.8, max(5.2, 0.46 * len(pivot))))
    image = ax.imshow(masked, aspect="auto", cmap="Blues", norm=LogNorm(vmin=positive.min(), vmax=positive.max()))
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xlabel("H2 restriction"); ax.set_ylabel("Target tissue")
    ax.set_title("mousePMHC Phase 3-5 benchmark coverage (train pairs; logarithmic scale)")
    threshold = np.sqrt(positive.min() * positive.max())
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = matrix[row, col]
            if value > 0:
                ax.text(col, row, str(int(value)), ha="center", va="center", fontsize=7,
                        color="white" if value > threshold else COLORS["dark"])
    colorbar = fig.colorbar(image, ax=ax, pad=0.02); colorbar.set_label("Train pairs")
    return fig


def seed_scaling(results: Path) -> pd.DataFrame:
    e3b = pd.read_csv(results / "mousePMHC_phase3_e3b_task_balanced_mmoe_min200_oof" / "mousePMHC_phase3_e3b_oof_stability_metrics.csv")
    e8 = pd.read_csv(results / "mousePMHC_phase4_e8_e3b_seed_ensemble_oof" / "mousePMHC_phase4_e8_oof_summary_metrics.csv")
    e15 = pd.read_csv(results / "mousePMHC_phase4_e15_five_seed_confirmation" / "mousePMHC_phase4_e15_oof_summary_metrics.csv")
    def stability(metric: str) -> float:
        return float(e3b.loc[e3b.metric.eq(metric), "seed_mean"].iloc[0])
    e8_row = e8.loc[e8.candidate.eq("mousePMHC_phase4_e8_e3b_3seed_probability_mean")].iloc[0]
    e15_row = e15.loc[e15.candidate.eq("mousePMHC_phase4_e15_e3b_5seed_probability_mean")].iloc[0]
    return pd.DataFrame([
        {"label": "E3b single-seed mean", "seeds": 1, "mean_auroc": stability("mean_task_auroc"), "mean_auprc": stability("mean_task_auprc"), "worst6_auroc": stability("worst6_task_auroc")},
        {"label": "E8 probability ensemble", "seeds": 3, "mean_auroc": e8_row.mean_task_auroc, "mean_auprc": e8_row.mean_task_auprc, "worst6_auroc": e8_row.worst6_task_auroc},
        {"label": "E15 probability ensemble", "seeds": 5, "mean_auroc": e15_row.mean_task_auroc, "mean_auprc": e15_row.mean_task_auprc, "worst6_auroc": e15_row.worst6_task_auroc},
    ])


def plot_seed_scaling(scaling: pd.DataFrame) -> plt.Figure:
    metrics = [("mean_auroc", "Mean task AUROC"), ("mean_auprc", "Mean task AUPRC"), ("worst6_auroc", "Worst-6 task AUROC")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    for ax, (metric, title) in zip(axes, metrics):
        ax.plot(scaling.seeds, scaling[metric], color=COLORS["final"], marker="s", linewidth=2)
        for _, row in scaling.iterrows():
            ax.annotate(f"{row[metric]:.4f}", (row.seeds, row[metric]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(scaling.seeds, ["1 model", "3 seeds", "5 seeds"])
        ax.set_ylabel("Task-macro score"); ax.set_title(title)
        values = scaling[metric]; ax.set_ylim(values.min() - 0.007, values.max() + 0.009)
    fig.suptitle("Independent seed averaging is the reproducible Phase 4 gain", y=1.0)
    return fig


def e19_diagnostics(results: Path) -> pd.DataFrame:
    return pd.read_csv(results / "mousePMHC_phase5_e19_hierarchical_heads_oof" / "mousePMHC_phase5_e19_mechanism_diagnostics.csv")


def plot_e19_diagnostics(diagnostics: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = np.where(diagnostics.small_sample_task, COLORS["negative"], COLORS["final"])
    axes[0].scatter(diagnostics.fitting_pairs, diagnostics.interaction_to_global_logit_ratio, c=colors, alpha=0.8, s=32)
    axes[0].axhline(1.0, color=COLORS["dark"], linestyle="--", linewidth=1)
    liver = diagnostics.loc[diagnostics.task_name.eq("liver||H2-Kk")]
    axes[0].scatter(liver.fitting_pairs, liver.interaction_to_global_logit_ratio, facecolors="none", edgecolors=COLORS["dark"], s=95, linewidths=1.2)
    axes[0].annotate("liver | H2-Kk", (liver.fitting_pairs.mean(), liver.interaction_to_global_logit_ratio.mean()), xytext=(8, 8), textcoords="offset points", fontsize=8)
    axes[0].set_xlabel("Fitting pairs per task"); axes[0].set_ylabel("Interaction / global logit RMS")
    axes[0].set_title("Local interaction dominance in a small Kk task")
    component_cols = ["global_logit_rms", "tissue_logit_rms", "h2_logit_rms", "interaction_logit_rms"]
    component_names = ["Global", "Tissue", "H2", "Interaction"]
    by_fold = diagnostics.groupby("fold")[component_cols].mean()
    x = np.arange(len(component_cols)); width = 0.22
    for idx, (fold, row) in enumerate(by_fold.iterrows()):
        axes[1].bar(x + (idx - 1) * width, row.to_numpy(), width, label=f"Fold {fold + 1}", color=[COLORS["baseline"], COLORS["ensemble"], COLORS["final"]][idx])
    axes[1].set_xticks(x, component_names); axes[1].set_ylabel("Mean task logit RMS")
    axes[1].set_title("Hierarchical logit components by OOF fold")
    axes[1].legend(fontsize=8)
    fig.suptitle("E19 mechanism diagnostics: aggregate restraint does not prevent local dominance", y=1.0)
    return fig


def phase5_h2_deltas(results: Path) -> pd.DataFrame:
    e17 = pd.read_csv(results / "mousePMHC_phase5_e17_taskwise_ranking_mmoe_oof" / "mousePMHC_phase5_e17_h2_macro_deltas.csv")
    e19 = pd.read_csv(results / "mousePMHC_phase5_e19_hierarchical_heads_oof" / "mousePMHC_phase5_e19_h2_macro_deltas.csv")
    e17["candidate"] = "E17a matched-pair ranking"
    e19["candidate"] = "E19 hierarchical head"
    return pd.concat([e17, e19], ignore_index=True)


def plot_phase5_h2_deltas(deltas: pd.DataFrame) -> plt.Figure:
    metrics = [("delta_auroc", "AUROC delta"), ("delta_auprc", "AUPRC delta"), ("delta_mcc", "MCC delta")]
    candidates = list(deltas.candidate.unique()); h2s = list(deltas.mhc_restriction.unique())
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7), sharex=True)
    x = np.arange(len(h2s)); width = 0.34
    for ax, (metric, title) in zip(axes, metrics):
        for index, candidate in enumerate(candidates):
            subset = deltas.loc[deltas.candidate.eq(candidate)].set_index("mhc_restriction").loc[h2s]
            ax.bar(x + (index - 0.5) * width, subset[metric], width, label=candidate,
                   color=[COLORS["ensemble"], COLORS["negative"]][index])
        ax.axhline(0, color=COLORS["dark"], linewidth=0.9)
        ax.set_xticks(x, h2s); ax.set_title(title); ax.set_ylabel("Candidate − matched E3b")
    for ax in axes:
        ax.set_ylabel("Candidate minus matched E3b")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8)
    fig.subplots_adjust(bottom=0.23, wspace=0.35)
    fig.suptitle("Phase 5 H2-level effects: no candidate protects all H2 groups", y=1.0)
    return fig


def save(fig: plt.Figure, output: Path, stem: str, pdf: PdfPages) -> None:
    fig.tight_layout(); fig.savefig(output / f"{stem}.png", bbox_inches="tight"); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "mousePMHC" / "mousePMHC_train.csv.gz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "figures_phase5")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True); apply_style()
    milestones = phase_milestones(args.results_dir)
    task_deltas = phase5_task_deltas(args.results_dir)
    coverage = mouse_coverage(args.train)
    scaling = seed_scaling(args.results_dir)
    diagnostics = e19_diagnostics(args.results_dir)
    h2 = phase5_h2_deltas(args.results_dir)
    milestones.to_csv(args.output_dir / "model_milestones.csv", index=False)
    task_deltas.to_csv(args.output_dir / "phase5_task_deltas.csv", index=False)
    coverage.to_csv(args.output_dir / "mouse_task_coverage.csv", index=False)
    scaling.to_csv(args.output_dir / "seed_scaling.csv", index=False)
    diagnostics.to_csv(args.output_dir / "e19_mechanism_diagnostics.csv", index=False)
    h2.to_csv(args.output_dir / "phase5_h2_deltas.csv", index=False)
    with PdfPages(args.output_dir / "mousepmhc_phase3_phase5_figures.pdf") as pdf:
        save(plot_model_milestones(milestones), args.output_dir, "01_model_milestones", pdf)
        save(plot_phase5_task_deltas(task_deltas), args.output_dir, "02_phase5_task_auroc_deltas", pdf)
        save(plot_mouse_coverage(coverage), args.output_dir, "03_mouse_task_coverage", pdf)
        save(plot_seed_scaling(scaling), args.output_dir, "04_seed_ensemble_scaling", pdf)
        save(plot_e19_diagnostics(diagnostics), args.output_dir, "05_e19_hierarchical_diagnostics", pdf)
        save(plot_phase5_h2_deltas(h2), args.output_dir, "06_phase5_h2_effects", pdf)
    metadata = {
        "status": "phase3_to_phase5_train_only_oof_summary",
        "main_result": "E15 5-seed task-balanced Factorized MMoE probability ensemble",
        "n_figures": 6,
        "test_data_read": False,
        "benchmark_warning": "All figures summarize pair-grouped train-only OOF; they are not fixed-test, peptide-disjoint, protein-disjoint, unseen-H2, or external validation results.",
    }
    (args.output_dir / "figure_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote six figures, source tables, and one PDF to: {args.output_dir}")


if __name__ == "__main__":
    main()
